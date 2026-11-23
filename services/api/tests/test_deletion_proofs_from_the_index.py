"""A deletion is proven from the reference index, and checked again by the database.

Proving that one artifact is unreferenced used to read every stored document
while the deletion held the database writer, so its cost grew with the store.
The proof now asks the index, which answers in a few indexed reads; the
database then asks the same question again at the delete itself, so a proof
minted wrongly fails there instead of removing something still held. These
cases hold the cost, the check inside the database, and what a deletion
declines while the index is behind.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from test_chat_deletion import _image_exchange
from test_chat_recovery import _chat, _command, _history, _impact
from test_chat_recovery_media_atomic import _trash
from test_recovery_expiry import DEADLINE
from test_recovery_expiry import _trash as _trash_to_expire
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm import artifact_reference_index, main, recovery_maintenance
from local_lm.artifact_deletion_authority import (
    _PROOF_SEAL,
    ArtifactDeletionProofError,
    ArtifactReferenceDataError,
    ArtifactReferenceIndexStale,
    _ArtifactDeletionProof,
    activate_artifact_deletion_proof,
    artifact_deletion_proof_references,
    mint_artifact_deletion_proof,
    open_reference_index_writer,
)
from local_lm.artifact_library import (
    begin_artifact_write_fence,
    ensure_library_entry,
    referenced_artifact_ids,
)
from local_lm.artifact_reference_extractors import SOURCE_ARTIFACTS, SOURCE_COLUMNS
from local_lm.artifact_reference_index import (
    IndexedReferenceSnapshot,
    drain_changes,
    form_reference_snapshot,
)
from local_lm.artifact_reference_index_schema import (
    CHANGE_TRIGGER_SQL,
    DELETE_GUARD_SQL,
    RETAINING_COLUMNS,
    trigger_name,
)
from local_lm.artifacts import ArtifactStore, RetentionCleanupSummary
from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import Base, SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.models import (
    Artifact,
    Chat,
    Job,
    MessagePart,
    MessageReference,
    RecoveryItem,
    Run,
    WorkStep,
)

RAW_JOB = (
    "INSERT INTO jobs (id, kind, status, progress, phase, progress_json, queue_priority, "
    "payload_json, result_json, attempt, cancellable, created_at, updated_at) VALUES "
    "(?, 'chat', 'queued', 0, 'queued', '{}', 0, ?, '{}', 0, 1, "
    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
)


@pytest.fixture
def library(tmp_path: Path) -> Iterator[tuple[ArtifactStore, Session]]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{tmp_path / 'library.sqlite3'}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    try:
        yield ArtifactStore(settings), session
    finally:
        session.close()
        engine.dispose()


def _stored(store: ArtifactStore, session: Session, content: bytes, **metadata: Any) -> Artifact:
    artifact = store.ingest_bytes(
        session,
        content,
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        metadata=metadata or None,
    )
    session.commit()
    return artifact


def _settle(session: Session) -> None:
    begin_artifact_write_fence(session)
    drain_changes(session.connection(), limit=None)
    session.commit()


def _snapshot(session: Session) -> IndexedReferenceSnapshot:
    begin_artifact_write_fence(session)
    snapshot = form_reference_snapshot(session)
    assert snapshot is not None
    return snapshot


def _steps_to_delete_one(store: ArtifactStore, session: Session, jobs: int) -> int:
    """SQLite's virtual machine steps to prove and delete one artifact among `jobs` jobs."""

    kept = _stored(store, session, b"kept by every job")
    pad = "0123456789abcdef" * 128
    for _ in range(jobs):
        session.add(Job(payload_json={"pad": pad}, result_json={"artifact_id": kept.id}))
    session.commit()
    target = _stored(store, session, b"named by nothing")
    _settle(session)
    driver = session.connection().connection.driver_connection
    assert isinstance(driver, sqlite3.Connection)
    steps = [0]

    def count() -> int:
        steps[0] += 1
        return 0

    begin_artifact_write_fence(session)
    previous = artifact_reference_index.CHECK_AGAINST_WALK
    artifact_reference_index.CHECK_AGAINST_WALK = False
    driver.set_progress_handler(count, 100)
    try:
        store._delete_artifact(session, target)
    finally:
        driver.set_progress_handler(None, 100)
        artifact_reference_index.CHECK_AGAINST_WALK = previous
    session.commit()
    assert session.get(Artifact, target.id) is None
    return steps[0]


def test_proving_one_deletion_costs_the_same_in_a_four_times_larger_store(
    tmp_path: Path,
) -> None:
    """The walk read every job's document under the writer; the proof reads a few rows."""

    measured = []
    for jobs in (300, 1200):
        settings = Settings(data_dir=tmp_path / f"data-{jobs}")
        settings.prepare()
        engine = create_engine(f"sqlite:///{tmp_path / f'store-{jobs}.sqlite3'}")
        Base.metadata.create_all(engine)
        try:
            with Session(engine, expire_on_commit=False) as session:
                measured.append(_steps_to_delete_one(ArtifactStore(settings), session, jobs))
        finally:
            engine.dispose()
    small, large = measured
    assert large < small * 1.25, measured


def _forged_proof(session: Session, artifact_id: str) -> object:
    """A proof naming an artifact the mint would refuse, as a wrong mint would make one."""

    snapshot = _snapshot(session)
    return _ArtifactDeletionProof(
        session=session,
        transaction=session.get_transaction(),
        candidate_ids=frozenset({artifact_id}),
        references=snapshot,
        seal=_PROOF_SEAL,
    )


@pytest.mark.parametrize("holder", ["job", "message part"])
def test_the_database_refuses_a_proven_deletion_of_a_held_artifact(
    library: tuple[ArtifactStore, Session], holder: str
) -> None:
    store, session = library
    held = _stored(store, session, b"still held")
    if holder == "job":
        session.add(Job(payload_json={}, result_json={"artifact_id": held.id}))
    else:
        session.add(
            MessagePart(message_id="message", position=0, type="image", artifact_id=held.id)
        )
    session.commit()
    _settle(session)
    with pytest.raises(ArtifactDeletionProofError, match="retained"):
        mint_artifact_deletion_proof(session, {held.id}, _snapshot(session))
    session.rollback()

    # The flush guard refuses a forged proof before anything reaches the database.
    proof = _forged_proof(session, held.id)
    session.delete(held)
    with (
        pytest.raises(ArtifactReferenceDataError),
        activate_artifact_deletion_proof(session, proof),
    ):
        session.flush()
    session.rollback()
    # A statement that skips the flush meets the database's own check.
    proof = _forged_proof(session, held.id)
    with (
        pytest.raises(IntegrityError, match="retained by reference"),
        activate_artifact_deletion_proof(session, proof),
    ):
        session.execute(text("DELETE FROM artifacts WHERE id = :id"), {"id": held.id})
    session.rollback()
    assert session.get(Artifact, held.id) is not None


def test_the_database_refuses_a_proven_deletion_over_an_undrained_or_unreadable_index(
    library: tuple[ArtifactStore, Session],
) -> None:
    store, session = library
    free = _stored(store, session, b"held by nothing")
    _settle(session)
    for planted, message in (
        ("INSERT INTO artifact_reference_changes VALUES (1, 'unread', 0)", "not indexed"),
        ("INSERT INTO artifact_reference_edges VALUES ('', 1, 'unreadable')", "invalid"),
    ):
        proof = mint_artifact_deletion_proof(session, {free.id}, _snapshot(session))
        with open_reference_index_writer(session.connection()):
            session.execute(text(planted))
        # Bring the snapshot's change counter forward past the planted row, as
        # nothing legitimate could, so only the database's own check is left.
        snapshot = artifact_deletion_proof_references(proof)
        assert isinstance(snapshot, IndexedReferenceSnapshot)
        snapshot._changes = snapshot._driver.total_changes
        session.delete(free)
        with (
            pytest.raises(IntegrityError, match=message),
            activate_artifact_deletion_proof(session, proof),
        ):
            session.flush()
        session.rollback()
        assert session.get(Artifact, free.id) is not None


def test_a_repeated_member_name_cannot_hide_a_reference(
    library: tuple[ArtifactStore, Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk keeps the last copy of a repeated name, so it alone would allow this."""

    monkeypatch.setattr(artifact_reference_index, "CHECK_AGAINST_WALK", False)
    store, session = library
    first = _stored(store, session, b"named first")
    second = _stored(store, session, b"named second")
    session.connection().exec_driver_sql(
        RAW_JOB, ("twice", f'{{"artifact_id": "{first.id}", "artifact_id": "{second.id}"}}')
    )
    session.commit()
    begin_artifact_write_fence(session)
    with pytest.raises(ArtifactReferenceDataError):
        store._delete_artifact(session, first)
    session.rollback()
    assert session.get(Artifact, first.id) is not None


def test_membership_follows_namers_through_chains_and_cycles(
    library: tuple[ArtifactStore, Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, session = library
    leaf = _stored(store, session, b"leaf")
    middle = _stored(store, session, b"middle", poster_artifact_id=leaf.id)
    top = _stored(store, session, b"top", poster_artifact_id=middle.id)
    session.add(Job(payload_json={"artifact_id": top.id}, result_json={}))
    loop_one = _stored(store, session, b"loop one")
    loop_two = _stored(store, session, b"loop two", poster_artifact_id=loop_one.id)
    loop_one.metadata_json = {"browser_proxy_artifact_id": loop_two.id}
    free = _stored(store, session, b"free")
    session.commit()

    snapshot = _snapshot(session)
    answers = {name.id: name.id in snapshot for name in (leaf, middle, top, loop_one, free)}
    assert answers == {
        leaf.id: True,
        middle.id: True,
        top.id: True,
        loop_one.id: False,
        free.id: False,
    }
    assert snapshot.namers(loop_one.id) == {loop_two.id}
    assert frozenset(snapshot) == {leaf.id, middle.id, top.id}
    session.rollback()

    monkeypatch.setattr(artifact_reference_index, "MAX_REVERSE_SEARCH", 0)
    snapshot = _snapshot(session)
    assert leaf.id in snapshot and free.id not in snapshot, "past the bound it reads everything"
    session.rollback()


def test_a_large_store_found_stale_declines_until_it_is_rebuilt(
    library: tuple[ArtifactStore, Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, session = library
    preview = _stored(store, session, b"aged preview", temporary_preview=True)
    session.execute(text("CREATE TABLE throwaway (id INTEGER)"))
    session.commit()
    monkeypatch.setattr(artifact_reference_index, "MAX_INLINE_REBUILD_ROWS", 0)

    begin_artifact_write_fence(session)
    with pytest.raises(ArtifactReferenceIndexStale):
        referenced_artifact_ids(session, for_deletion=True)
    session.rollback()
    assert store.delete_temporary_preview(session, preview.id) is False
    session.rollback()
    assert session.get(Artifact, preview.id) is not None

    monkeypatch.undo()
    assert store.delete_temporary_preview(session, preview.id) is True
    session.commit()


async def test_a_deletion_while_the_index_is_rebuilt_is_a_conflict_that_says_so(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import db

    with db.SessionLocal() as session:
        store = ArtifactStore(settings)
        artifact = _stored(store, session, b"deleted during a rebuild")
        session.execute(text("CREATE TABLE throwaway (id INTEGER)"))
        session.commit()
    monkeypatch.setattr(artifact_reference_index, "MAX_INLINE_REBUILD_ROWS", 0)
    # Hold the background rebuild off, so the store stays as a large one is
    # while it is rebuilt.
    monkeypatch.setattr(
        artifact_reference_index,
        "maintenance_step",
        lambda _session: artifact_reference_index.MaintenanceStep(0, 0, False),
    )

    response = await client.delete(f"/api/artifacts/{artifact.id}")

    assert response.status_code == 409
    assert response.json()["code"] == "artifact-references-indexing"


def test_the_sweep_builds_objects_only_for_what_it_removes_or_marks(
    library: tuple[ArtifactStore, Session],
) -> None:
    store, session = library
    kept = [_stored(store, session, f"kept {serial}".encode()) for serial in range(40)]
    for artifact in kept:
        session.add(Job(payload_json={"artifact_id": artifact.id}, result_json={}))
    previews = [
        _stored(store, session, f"preview {serial}".encode(), temporary_preview=True)
        for serial in range(3)
    ]
    session.commit()
    session.expunge_all()
    loaded: list[str] = []

    def count(target: Artifact, _context: object) -> None:
        loaded.append(target.id)

    event.listen(Artifact, "load", count)
    try:
        summary = store.cleanup_retention(
            session, retention_days=30, temporary_hours=0, dry_run=False
        )
        session.commit()
    finally:
        event.remove(Artifact, "load", count)
    assert summary.removed_row_count == len(previews)
    assert set(loaded) <= {preview.id for preview in previews}


async def test_the_sweep_waits_until_the_index_is_ready(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def batch(*_args: object, **_kwargs: object) -> Any:
        calls.append("batch")
        raise AssertionError("a batch ran before the index was ready")

    store = ArtifactStore(settings)
    monkeypatch.setattr(store, "cleanup_retention", batch)
    ready = asyncio.Event()
    sweep = asyncio.create_task(main.sweep_artifact_retention(store, settings, index_ready=ready))
    await asyncio.sleep(0.1)
    assert calls == [] and not sweep.done()
    sweep.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sweep


def test_the_guard_migration_adds_and_removes_only_the_guard(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", dev=True)
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "7c3e9a1f5b20")
    database = settings.state_dir / "local-lm.sqlite3"

    def triggers() -> dict[str, str]:
        with sqlite3.connect(database) as connection:
            return {
                name: sql
                for name, sql in connection.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
                )
            }

    before = triggers()
    command.upgrade(config, "b8d24f6a9c13")
    after = triggers()
    assert set(after) - set(before) == {"artifact_reference_index_delete_guard"}
    assert after["artifact_reference_index_delete_guard"].strip() == DELETE_GUARD_SQL.strip()
    assert {name: after[name] for name in before} == before
    command.downgrade(config, "7c3e9a1f5b20")
    assert triggers() == before


def test_a_proof_evaluates_no_stored_document(library: tuple[ArtifactStore, Session]) -> None:
    """The stored-document check moved to the rebuild; a deletion runs none of it."""

    store, session = library
    for serial in range(5):
        kept = _stored(store, session, f"kept {serial}".encode())
        session.add(Job(payload_json={"artifact_id": kept.id}, result_json={}))
    target = _stored(store, session, b"proven")
    session.commit()
    _settle(session)
    begin_artifact_write_fence(session)
    references = referenced_artifact_ids(session, for_deletion=True)
    driver = session.connection().connection.driver_connection
    assert isinstance(driver, sqlite3.Connection)
    calls: list[object] = []

    def observed(value: object) -> int:
        calls.append(value)
        return 1

    driver.create_function("json_valid", 1, observed)
    previous = artifact_reference_index.CHECK_AGAINST_WALK
    artifact_reference_index.CHECK_AGAINST_WALK = False
    try:
        store._delete_artifact(session, target, references=references)
        again = referenced_artifact_ids(session, for_deletion=True)
        assert target.id not in again
    finally:
        artifact_reference_index.CHECK_AGAINST_WALK = previous
    assert calls == []
    session.commit()
    assert session.scalar(select(Artifact.id).where(Artifact.id == target.id)) is None


#: Where a planted row puts the held artifact's id.
HELD = object()
STAMP = "2026-01-01 00:00:00"


def _row(table: str, **values: object) -> Callable[[Session, str], None]:
    """Plant one row of `table` naming the held artifact wherever a value is `HELD`.

    A callable value is given the held artifact's id and stored as it answers.
    """

    def plant(session: Session, held: str) -> None:
        row = {
            column: held if value is HELD else value(held) if callable(value) else value
            for column, value in values.items()
        }
        names = ", ".join(row)
        slots = ", ".join(f":{column}" for column in row)
        session.execute(text(f"INSERT INTO {table} ({names}) VALUES ({slots})"), row)

    return plant


def _revision_part(session: Session, held: str) -> None:
    _row(
        "response_revisions",
        id="revision",
        message_id="message",
        sequence=1,
        status="complete",
        created_at=STAMP,
        updated_at=STAMP,
    )(session, held)
    _row(
        "response_revision_parts",
        id="revision-part",
        response_revision_id="revision",
        position=0,
        type="image",
        metadata_json="{}",
        artifact_id=HELD,
        created_at=STAMP,
        updated_at=STAMP,
    )(session, held)


#: One way for each kind of stored row to hold an artifact, by its table.
HOLDERS: dict[str, Callable[[Session, str], None]] = {
    "jobs": lambda session, held: session.add(
        Job(payload_json={}, result_json={"artifact_id": held})
    ),
    "runs": lambda session, held: session.add(
        Run(
            chat_id="chat",
            user_message_id="user",
            assistant_message_id="assistant",
            provenance_json={"input_artifact_ids": [held]},
        )
    ),
    "work_steps": lambda session, held: session.add(
        WorkStep(
            plan_id="plan",
            ordinal=0,
            operation="image_edit",
            input_bindings_json=[{"artifact_id": held}],
        )
    ),
    "message_references": lambda session, held: session.add(
        MessageReference(
            message_id="message",
            position=0,
            reference_subject_id="subject",
            mention_slug="subject",
            subject_name="Subject",
            subject_kind="person",
            artifact_ids_json=[held],
        )
    ),
    "chats": lambda session, held: session.add(
        Chat(title="Studio", scope="studio", origin_json={"source_artifact_id": held})
    ),
    "message_parts": lambda session, held: session.add(
        MessagePart(message_id="message", position=0, type="image", artifact_id=held)
    ),
    "response_revision_parts": _revision_part,
    "reference_assets": _row(
        "reference_assets",
        id="asset",
        reference_subject_id="subject",
        artifact_id=HELD,
        purpose="identity",
        sort_order=0,
        validation_state="unchecked",
        validation_reasons_json="[]",
        review_version=1,
        created_at=STAMP,
        updated_at=STAMP,
    ),
    "run_context_artifacts": _row("run_context_artifacts", run_id="run", artifact_id=HELD),
    "setup_verifications": _row(
        "setup_verifications",
        id="verification",
        role="image",
        evidence_key="evidence",
        state="passed",
        model_install_id="install",
        profile_id="profile",
        input_artifact_id=HELD,
        created_at=STAMP,
        updated_at=STAMP,
    ),
    "artifact_library_entries": _row(
        "artifact_library_entries",
        id=lambda held: f"libentry:{held}",
        artifact_id=HELD,
        display_name="Kept",
        favorite=0,
        state="visible",
        version=1,
        created_at=STAMP,
        updated_at=STAMP,
    ),
    "comfy_registry_source_artifact_reviews": _row(
        "comfy_registry_source_artifact_reviews",
        id="review",
        source_declaration="declaration",
        source_declaration_sha256="0" * 64,
        repository="repository",
        source_commit="0" * 40,
        artifact_id=HELD,
        artifact_sha256="0" * 64,
        artifact_size_bytes=1,
        wheel_filename="package-1.0-py3-none-any.whl",
        wheel_distribution="package",
        wheel_version="1.0",
        evidence_json="{}",
        reviewer_kind="local-human",
        review_sha256="0" * 64,
        reviewed_at=STAMP,
        created_at=STAMP,
        updated_at=STAMP,
    ),
    "chat_composer_draft_attachments": _row(
        "chat_composer_draft_attachments",
        chat_id="chat",
        position=0,
        artifact_id=HELD,
        kind="image",
        origin="upload",
    ),
}


def test_every_kind_of_row_that_holds_an_artifact_is_tried() -> None:
    documents = {
        table for source, (table, _) in SOURCE_COLUMNS.items() if source != SOURCE_ARTIFACTS
    }
    columns = {table for table, _column in RETAINING_COLUMNS}
    assert set(HOLDERS) == documents | columns | {"response_revision_parts"}


@pytest.mark.parametrize("holder", sorted(HOLDERS))
def test_each_kind_of_holder_retains_through_the_search_the_mint_and_the_guard(
    library: tuple[ArtifactStore, Session], holder: str
) -> None:
    store, session = library
    held = _stored(store, session, f"held by {holder}".encode())
    free = _stored(store, session, f"beside {holder}".encode())
    HOLDERS[holder](session, held.id)
    session.commit()
    _settle(session)

    snapshot = _snapshot(session)
    assert held.id in snapshot and free.id not in snapshot
    assert snapshot._retained is None, "answered by the search, not by the whole set"
    with pytest.raises(ArtifactDeletionProofError, match="retained"):
        mint_artifact_deletion_proof(session, {held.id}, snapshot)
    session.rollback()

    proof = _forged_proof(session, held.id)
    with (
        pytest.raises(IntegrityError, match="retained by reference"),
        activate_artifact_deletion_proof(session, proof),
    ):
        session.execute(text("DELETE FROM artifacts WHERE id = :id"), {"id": held.id})
    session.rollback()
    assert session.get(Artifact, held.id) is not None


@pytest.mark.parametrize("answer", ["search", "namers"])
def test_the_walk_check_asks_what_a_deletion_asks(
    library: tuple[ArtifactStore, Session], monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    """A wrong answer from the search or the namers a mint reads is caught, and nothing kept."""

    store, session = library
    poster = _stored(store, session, b"a poster")
    video = _stored(store, session, b"a video", poster_artifact_id=poster.id)
    if answer == "search":
        session.add(Job(payload_json={"artifact_id": video.id}, result_json={}))
        monkeypatch.setattr(IndexedReferenceSnapshot, "_search", lambda _self, _name: False)
    else:
        # Nothing is retained, so only the namers can disagree.
        monkeypatch.setattr(IndexedReferenceSnapshot, "namers", lambda _self, _name: frozenset())
    session.commit()
    _settle(session)
    monkeypatch.setattr(artifact_reference_index, "shadow_disagreements", 0)

    begin_artifact_write_fence(session)
    snapshot = referenced_artifact_ids(session, for_deletion=True)

    assert artifact_reference_index.shadow_disagreements == 1
    assert isinstance(snapshot, IndexedReferenceSnapshot)
    assert snapshot._retained is None and snapshot._answers == {}
    session.rollback()


def _uncertifiable(session: Session) -> None:
    # Dropping a trigger moves the schema cookie, so the index is stale, and
    # no rebuild may certify it while the trigger is gone.
    session.execute(text(f"DROP TRIGGER {trigger_name(CHANGE_TRIGGER_SQL[0])}"))
    session.commit()


def test_a_store_that_cannot_be_certified_declines_rather_than_fails(
    library: tuple[ArtifactStore, Session],
) -> None:
    store, session = library
    preview = _stored(store, session, b"an aged preview", temporary_preview=True)
    _uncertifiable(session)

    begin_artifact_write_fence(session)
    with pytest.raises(ArtifactReferenceDataError):
        referenced_artifact_ids(session, for_deletion=True)
    session.rollback()
    assert store.delete_temporary_preview(session, preview.id) is False
    session.rollback()
    assert session.get(Artifact, preview.id) is not None


async def test_a_deletion_the_index_cannot_prove_is_a_conflict(
    client: AsyncClient, settings: Settings
) -> None:
    with SessionLocal() as session:
        artifact = _stored(ArtifactStore(settings), session, b"kept while unproven")
        _uncertifiable(session)

    response = await client.delete(f"/api/artifacts/{artifact.id}")

    assert response.status_code == 409
    assert response.json()["code"] == "artifact-references-unavailable"
    with SessionLocal() as session:
        assert session.get(Artifact, artifact.id) is not None


async def test_a_purge_drains_everything_its_own_deletion_recorded(
    client: AsyncClient, app: FastAPI
) -> None:
    """Only the purging transaction can drain what deleting the conversation recorded."""

    chat_id = await _chat(client)
    await _image_exchange(client, chat_id, "Create a picture of garden beds")
    message_id = _history(chat_id)["messages"][0][0]
    with SessionLocal() as session:
        session.add_all(
            MessageReference(
                message_id=message_id,
                position=position,
                reference_subject_id="subject",
                mention_slug="subject",
                subject_name="Subject",
                subject_kind="person",
            )
            for position in range(2_001)
        )
        session.commit()
    deletion_id = await _trash(client, chat_id)
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")

    purged = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={
            **_command(preview, "purge-many-references"),
            "acknowledgement": "permanently-delete",
        },
    )

    assert purged.status_code == 200, purged.text
    with SessionLocal() as session:
        assert session.get(Chat, chat_id) is None
        assert session.scalar(select(MessageReference.id).limit(1)) is None


def _hold_a_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the store stale as a large one is while upkeep rebuilds it, upkeep held off."""

    monkeypatch.setattr(artifact_reference_index, "MAX_INLINE_REBUILD_ROWS", 0)
    monkeypatch.setattr(
        artifact_reference_index,
        "maintenance_step",
        lambda _session: artifact_reference_index.MaintenanceStep(0, 0, False),
    )
    with SessionLocal() as session:
        session.execute(text("CREATE TABLE throwaway (id INTEGER)"))
        session.commit()


async def test_recovery_says_the_index_is_being_rebuilt_rather_than_invalid(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id = await _chat(client)
    await _image_exchange(client, chat_id, "Create a picture of garden beds")
    deletion_id = await _trash(client, chat_id)
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    with SessionLocal() as session:
        picture = _stored(app.state.services.artifacts, session, b"a library picture")
        entry = ensure_library_entry(session, picture)
        assert entry is not None
        entry_id = entry.id
        session.commit()
    _hold_a_rebuild(monkeypatch)

    media = await client.get(f"/api/artifact-library/{entry_id}/deletion-impact")
    purge = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={**_command(preview, "purge-during-rebuild"), "acknowledgement": "permanently-delete"},
    )

    for response in (media, purge):
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "artifact-references-indexing"
    with SessionLocal() as session:
        item = session.get(RecoveryItem, deletion_id)
        assert item is not None and item.state == "recoverable"


@pytest.mark.usefixtures("client")
async def test_expiry_defers_an_item_while_the_index_is_rebuilt(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    deletion_id, _entry_id, _artifact_id = await _trash_to_expire(app, "media_library_entry")
    _hold_a_rebuild(monkeypatch)

    result = await recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE)

    assert result.examined == result.deferred == 1 and result.purged == 0
    with SessionLocal() as session:
        item = session.get(RecoveryItem, deletion_id)
        assert item is not None and item.state == "recoverable"


async def test_the_sweep_waits_out_a_rebuild_longer_than_a_busy_writer(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    waits = main.RETENTION_LOCK_RETRIES + 2

    def batch(*_args: object, **_kwargs: object) -> RetentionCleanupSummary:
        nonlocal calls
        calls += 1
        if calls <= waits:
            raise ArtifactReferenceIndexStale()
        return RetentionCleanupSummary(
            marked_count=0, pending_count=0, removed_count=0, reclaimed_bytes=0
        )

    store = ArtifactStore(settings)
    monkeypatch.setattr(store, "cleanup_retention", batch)
    monkeypatch.setattr(main, "RETENTION_INDEX_PAUSE_SECONDS", 0.0, raising=False)

    await main.sweep_artifact_retention(store, settings, pause_seconds=0.0)

    assert calls == waits + 1, "the sweep ran its batch once the index was current"


def test_a_record_due_its_check_holds_a_large_store_whatever_marks_it(
    library: tuple[ArtifactStore, Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upkeep runs the check; a proof would run it under the writer for the whole store."""

    store, session = library
    artifact = _stored(store, session, b"recorded as due its check")
    _settle(session)
    with open_reference_index_writer(session.connection()):
        session.execute(
            text("INSERT INTO artifact_reference_changes VALUES (:source, :key, 2)"),
            {"source": SOURCE_ARTIFACTS, "key": artifact.id},
        )
    session.commit()
    monkeypatch.setattr(artifact_reference_index, "MAX_INLINE_REBUILD_ROWS", 0)

    begin_artifact_write_fence(session)
    with pytest.raises(ArtifactReferenceIndexStale):
        referenced_artifact_ids(session, for_deletion=True)
    session.rollback()
