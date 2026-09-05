from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from local_lm.artifact_deletion_authority import (
    ARTIFACT_DELETION_AUTHORIZED_SQL,
    ArtifactDeletionProofError,
    activate_artifact_deletion_proof,
    mint_artifact_deletion_proof,
)
from local_lm.artifact_library import (
    ArtifactReferenceDataError,
    begin_artifact_write_fence,
    referenced_artifact_ids,
)
from local_lm.artifact_library_schema import STORED_JSON_INVALID_SQL
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact, Chat


@pytest.fixture
def proof_session(tmp_path: Path) -> Iterator[tuple[ArtifactStore, Session]]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{tmp_path / 'proof.sqlite3'}")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        yield ArtifactStore(settings), session
    engine.dispose()


def _artifact(store: ArtifactStore, session: Session, value: bytes) -> Artifact:
    artifact = store.ingest_bytes(
        session,
        value,
        kind=ArtifactKind.OTHER,
        media_type="application/octet-stream",
    )
    session.commit()
    return artifact


def _proof(session: Session, *artifact_ids: str) -> object:
    begin_artifact_write_fence(session)
    references = referenced_artifact_ids(session, for_deletion=True)
    return mint_artifact_deletion_proof(session, artifact_ids, references)


def test_connection_authority_is_closed_except_for_the_exact_active_ids(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    first = _artifact(store, session, b"first")
    second = _artifact(store, session, b"second")
    proof = _proof(session, first.id)
    statement = text(f"SELECT {ARTIFACT_DELETION_AUTHORIZED_SQL}(:artifact_id)")

    assert session.scalar(statement, {"artifact_id": first.id}) == 0
    with activate_artifact_deletion_proof(session, proof):
        assert session.scalar(statement, {"artifact_id": first.id}) == 1
        assert session.scalar(statement, {"artifact_id": second.id}) == 0
    assert session.scalar(statement, {"artifact_id": first.id}) == 0


def test_proof_requires_the_complete_snapshot_from_the_same_session(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    artifact = _artifact(store, session, b"same session")
    begin_artifact_write_fence(session)

    with pytest.raises(ArtifactDeletionProofError, match="complete reference graph"):
        mint_artifact_deletion_proof(session, {artifact.id}, frozenset())

    references = referenced_artifact_ids(session, for_deletion=True)
    other = Session(session.get_bind())
    try:
        other.connection()
        with pytest.raises(ArtifactDeletionProofError, match="complete reference graph"):
            mint_artifact_deletion_proof(other, {artifact.id}, references)
    finally:
        other.rollback()
        other.close()


def test_empty_and_retained_deletion_sets_are_refused(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    artifact = _artifact(store, session, b"retained")
    session.add(
        Chat(title="Reference", scope="studio", origin_json={"source_artifact_id": artifact.id})
    )
    session.commit()
    begin_artifact_write_fence(session)
    references = referenced_artifact_ids(session, for_deletion=True)

    with pytest.raises(ArtifactDeletionProofError, match="cannot be empty"):
        mint_artifact_deletion_proof(session, (), references)

    with pytest.raises(ArtifactDeletionProofError, match="retained artifact"):
        mint_artifact_deletion_proof(session, {artifact.id}, references)


def test_proof_refuses_an_extra_artifact(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    allowed = _artifact(store, session, b"allowed")
    extra = _artifact(store, session, b"extra")
    proof = _proof(session, allowed.id)

    with pytest.raises(ArtifactDeletionProofError, match="does not cover"):
        store._delete_artifact(session, extra, proof=proof)

    assert session.get(Artifact, allowed.id) is not None
    assert session.get(Artifact, extra.id) is not None


def test_flush_listener_refuses_a_deletion_outside_the_active_proof(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    allowed = _artifact(store, session, b"flush allowed")
    extra = _artifact(store, session, b"flush extra")
    proof = _proof(session, allowed.id)

    session.delete(extra)
    with (
        pytest.raises(ArtifactDeletionProofError, match="does not match"),
        activate_artifact_deletion_proof(session, proof),
    ):
        session.flush()
    session.rollback()

    assert session.get(Artifact, allowed.id) is not None
    assert session.get(Artifact, extra.id) is not None


def test_proof_expires_with_its_transaction(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    artifact = _artifact(store, session, b"stale")
    proof = _proof(session, artifact.id)
    session.rollback()
    current = session.get(Artifact, artifact.id)
    assert current is not None

    with pytest.raises(ArtifactDeletionProofError, match="another transaction"):
        store._delete_artifact(session, current, proof=proof)


def test_selective_removal_preview_cannot_authorize_deletion(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    artifact = _artifact(store, session, b"preview")
    begin_artifact_write_fence(session)
    preview = referenced_artifact_ids(session, exclude_message_payload_for="one-message")
    with pytest.raises(ArtifactDeletionProofError, match="complete reference graph"):
        mint_artifact_deletion_proof(session, {artifact.id}, preview)


def test_same_transaction_sql_write_invalidates_the_proof(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    target = _artifact(store, session, b"sql target")
    naming = _artifact(store, session, b"sql naming")
    proof = _proof(session, target.id)
    session.execute(
        text("UPDATE artifacts SET metadata_json = :metadata WHERE id = :id"),
        {"metadata": '{"poster_artifact_id": "' + target.id + '"}', "id": naming.id},
    )
    with pytest.raises(ArtifactDeletionProofError, match="changed"):
        store._delete_artifact(session, target, proof=proof)
    assert session.get(Artifact, target.id) is not None


def test_unretained_metadata_namer_still_prevents_low_level_deletion(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    target = _artifact(store, session, b"poster")
    naming = _artifact(store, session, b"video")
    naming.metadata_json = {"poster_artifact_id": target.id}
    session.commit()
    with pytest.raises(ValueError, match="retained"):
        store._delete_artifact(session, target)
    assert session.get(Artifact, target.id) is not None


def test_retention_observes_stop_between_actual_deletions(
    proof_session: tuple[ArtifactStore, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, session = proof_session
    now = datetime.now(UTC)
    for value in (b"first preview", b"second preview", b"third preview"):
        artifact = _artifact(store, session, value)
        artifact.metadata_json = {"temporary_preview": True}
        artifact.created_at = now - timedelta(days=2)
    session.commit()
    deleted: list[str] = []
    original = store._delete_artifact

    def remove(session: Session, artifact: Artifact, *, proof: object | None = None) -> None:
        original(session, artifact, proof=proof)
        deleted.append(artifact.id)

    monkeypatch.setattr(store, "_delete_artifact", remove)
    result = store.cleanup_retention(
        session,
        retention_days=30,
        temporary_hours=24,
        dry_run=False,
        now=now,
        should_stop=lambda: bool(deleted),
    )
    assert result.truncated
    assert result.removed_count == 1
    assert len(deleted) == 1
    assert len(session.scalars(select(Artifact)).all()) == 2


def test_pending_reference_write_refuses_and_restores_staged_bytes(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    target = _artifact(store, session, b"pending target")
    naming = _artifact(store, session, b"pending naming")
    proof = _proof(session, target.id)
    path = store.resolve(target)
    naming.metadata_json = {"poster_artifact_id": target.id}
    with pytest.raises(ArtifactDeletionProofError, match="changed"):
        store._delete_artifact(session, target, proof=proof)
    assert path.read_bytes() == b"pending target"
    session.rollback()
    assert session.get(Artifact, target.id) is not None


def test_proof_expires_after_its_savepoint_ends(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    target = _artifact(store, session, b"savepoint target")
    begin_artifact_write_fence(session)
    with session.begin_nested():
        proof = _proof(session, target.id)
    with pytest.raises(ArtifactDeletionProofError, match="transaction"):
        store._delete_artifact(session, target, proof=proof)
    assert session.get(Artifact, target.id) is not None


def test_authorized_delete_skips_json_scans_but_unproven_delete_evaluates_them(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    proven = _artifact(store, session, b"proven delete")
    unproven = _artifact(store, session, b"unproven delete")
    _artifact(store, session, b"remaining metadata")
    proof = _proof(session, proven.id)
    driver = session.connection().connection.driver_connection
    assert isinstance(driver, sqlite3.Connection)
    calls: list[object] = []

    def observed_json_valid(value: object) -> int:
        calls.append(value)
        if not isinstance(value, str):
            return 0
        try:
            json.loads(value)
        except ValueError:
            return 0
        return 1

    driver.create_function("json_valid", 1, observed_json_valid)
    store._delete_artifact(session, proven, proof=proof)
    assert calls == []
    session.execute(text("DELETE FROM artifacts WHERE id = :id"), {"id": unproven.id})
    assert calls
    assert session.scalar(select(Artifact.id).where(Artifact.id == unproven.id)) is None


def test_ordinary_reference_publication_does_not_scan_all_stored_json(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    target = _artifact(store, session, b"ordinary reference target")
    driver = session.connection().connection.driver_connection
    assert isinstance(driver, sqlite3.Connection)
    statements: list[str] = []
    driver.set_trace_callback(statements.append)
    try:
        session.add(
            Chat(
                title="Published reference",
                scope="studio",
                origin_json={"source_artifact_id": target.id},
            )
        )
        session.commit()
    finally:
        driver.set_trace_callback(None)

    assert any(statement.startswith("INSERT INTO chats ") for statement in statements)

    def normalize(value: str) -> str:
        return " ".join(value.split())

    assert normalize(STORED_JSON_INVALID_SQL) not in {normalize(value) for value in statements}


def test_unrelated_invalid_metadata_blocks_deletion_but_not_reference_publication(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    target = _artifact(store, session, b"valid publication target")
    corrupt = _artifact(store, session, b"legacy invalid metadata")
    trigger = session.scalar(
        text(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'artifacts_artifact_reference_update_guard'"
        )
    )
    assert isinstance(trigger, str)
    session.execute(text("DROP TRIGGER artifacts_artifact_reference_update_guard"))
    session.execute(
        text("UPDATE artifacts SET metadata_json = '[]' WHERE id = :id"),
        {"id": corrupt.id},
    )
    session.execute(text(trigger))
    session.commit()
    session.expire_all()

    published = Chat(
        title="Allowed reference", scope="studio", origin_json={"source_artifact_id": target.id}
    )
    session.add(published)
    session.commit()
    assert session.get(Chat, published.id) is published
    with pytest.raises(ArtifactReferenceDataError, match="invalid"):
        store._delete_artifact(session, corrupt)
    assert session.get(Artifact, corrupt.id) is not None


def test_ordinary_reference_graph_cannot_authorize_deletion(
    proof_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = proof_session
    target = _artifact(store, session, b"ordinary graph has no delete authority")
    begin_artifact_write_fence(session)
    references = referenced_artifact_ids(session)
    with pytest.raises(ArtifactDeletionProofError, match="complete reference graph"):
        mint_artifact_deletion_proof(session, {target.id}, references)


@pytest.mark.parametrize("stop", ["count", "shutdown"])
@pytest.mark.parametrize("pin", ["favorite", "reference"])
def test_truncated_retention_keeps_marks_and_clears_stale_retained_timestamps(
    proof_session: tuple[ArtifactStore, Session],
    stop: str,
    pin: str,
) -> None:
    store, session = proof_session
    now = datetime.now(UTC)
    retained = _artifact(store, session, b"temporarily retained")
    newly_unreferenced = _artifact(store, session, b"newly unreferenced")
    expired = _artifact(store, session, b"expired preview")
    retained.created_at = now - timedelta(days=4)
    retained.metadata_json = {"unreferenced_at": (now - timedelta(days=60)).isoformat()}
    newly_unreferenced.created_at = now - timedelta(days=3)
    expired.created_at = now - timedelta(days=2)
    expired.metadata_json = {"temporary_preview": True}
    chat = Chat(
        title="Temporary retention", scope="studio", origin_json={"source_artifact_id": retained.id}
    )
    if pin == "favorite":
        retained.favorite = True
    else:
        session.add(chat)
    session.commit()

    result = store.cleanup_retention(
        session,
        retention_days=30,
        temporary_hours=24,
        dry_run=False,
        now=now,
        max_deletions=0 if stop == "count" else None,
        should_stop=(lambda: True) if stop == "shutdown" else None,
    )
    session.commit()
    session.expire_all()
    assert result.truncated
    assert result.removed_count == 0
    assert result.marked_count == 1
    assert "unreferenced_at" not in retained.metadata_json
    assert newly_unreferenced.metadata_json["unreferenced_at"] == now.isoformat()

    if pin == "favorite":
        retained.favorite = False
    else:
        session.delete(chat)
    session.commit()
    later = now + timedelta(days=1)
    store.cleanup_retention(
        session, retention_days=30, temporary_hours=24, dry_run=False, now=later
    )
    session.commit()
    session.expire_all()
    assert session.get(Artifact, retained.id) is retained
    assert retained.metadata_json["unreferenced_at"] == later.isoformat()
