from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Event

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm import artifact_library
from local_lm.artifact_library import (
    REFERENCE_CORRUPT,
    ArtifactLibraryConflict,
    ArtifactReferenceDataError,
    ensure_library_entry,
    library_entry_id,
    referenced_artifact_ids,
    set_library_favorite,
)
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base, create_database_engine
from local_lm.domain import ArtifactKind
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Job,
    Message,
    MessagePart,
    MessageReference,
    Run,
    WorkPlan,
    WorkStep,
)


@pytest.fixture
def library_session(tmp_path: Path) -> Iterator[tuple[ArtifactStore, Session]]:
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


def test_generic_ingest_is_not_library_publication(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact = store.ingest_bytes(
        session, b"pixels", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    session.commit()

    assert session.scalars(select(ArtifactLibraryEntry)).all() == []
    assert ensure_library_entry(session, artifact) is not None
    session.commit()
    entry = session.scalar(select(ArtifactLibraryEntry))
    assert entry is not None
    assert entry.id == f"libentry:sha256:{artifact.sha256}"
    assert entry.display_name == artifact.sha256
    assert entry.state == "visible"
    assert entry.version == 1


def test_publication_is_idempotent_and_non_media_never_gets_membership(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    image = store.ingest_bytes(
        session,
        b"same image",
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="portrait.png",
    )
    export = store.ingest_bytes(
        session, b"archive", kind=ArtifactKind.EXPORT, media_type="application/zip"
    )
    first = ensure_library_entry(session, image)
    second = ensure_library_entry(session, image)

    assert first is second
    assert ensure_library_entry(session, export) is None
    assert (
        session.scalar(
            select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == image.id)
        )
        is first
    )
    assert len(session.scalars(select(ArtifactLibraryEntry)).all()) == 1


def test_conflict_loser_publication_reuses_canonical_entry(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact = store.ingest_bytes(
        session, b"conflict", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    session.commit()
    first = ensure_library_entry(session, artifact)
    session.commit()
    assert first is not None

    with Session(session.get_bind(), expire_on_commit=False) as other:
        same_artifact = other.get(Artifact, artifact.id)
        assert same_artifact is not None
        loser = ensure_library_entry(other, same_artifact)
        other.commit()
        assert loser is not None
        assert loser.id == first.id
    assert len(session.scalars(select(ArtifactLibraryEntry)).all()) == 1


def test_concurrent_publications_converge(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact = store.ingest_bytes(
        session, b"concurrent", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    session.commit()
    barrier = Barrier(2)

    def publish() -> str:
        with Session(session.get_bind(), expire_on_commit=False) as worker:
            owned = worker.get(Artifact, artifact.id)
            assert owned is not None
            barrier.wait()
            entry = ensure_library_entry(worker, owned)
            assert entry is not None
            worker.commit()
            return entry.id

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _index: publish(), range(2)))
    assert ids == [library_entry_id(artifact), library_entry_id(artifact)]
    session.expire_all()
    assert len(session.scalars(select(ArtifactLibraryEntry)).all()) == 1


def test_favorite_is_entry_canonical_and_dual_written(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact = store.ingest_bytes(
        session, b"favorite", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    entry = ensure_library_entry(session, artifact)
    assert entry is not None

    changed = set_library_favorite(session, artifact, True)
    session.commit()
    assert changed.favorite is True
    assert changed.version == 2
    assert artifact.favorite is True


def test_stale_favorite_writer_is_idempotent_or_fixed_conflict(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact = store.ingest_bytes(
        session, b"favorite-race", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    ensure_library_entry(session, artifact)
    session.commit()

    with (
        Session(session.get_bind(), expire_on_commit=False) as first,
        Session(session.get_bind(), expire_on_commit=False) as stale,
    ):
        first_artifact = first.get(Artifact, artifact.id)
        stale_artifact = stale.get(Artifact, artifact.id)
        assert first_artifact is not None and stale_artifact is not None
        first_entry = first.scalar(
            select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == artifact.id)
        )
        stale_entry = stale.scalar(
            select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == artifact.id)
        )
        assert first_entry is not None and stale_entry is not None

        set_library_favorite(first, first_artifact, True)
        first.commit()
        with pytest.raises(ArtifactLibraryConflict, match="changed"):
            set_library_favorite(stale, stale_artifact, False)
        stale.rollback()
        retry = set_library_favorite(stale, stale_artifact, True)
        stale.commit()
        assert retry.favorite is True
        assert retry.version == 2

    session.expire_all()
    stored_artifact = session.get(Artifact, artifact.id)
    stored_entry = session.scalar(
        select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == artifact.id)
    )
    assert stored_artifact is not None and stored_artifact.favorite is True
    assert stored_entry is not None and stored_entry.favorite is True


def test_visible_membership_pins_bytes_without_legacy_favorite(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact = store.ingest_bytes(
        session, b"retained", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    ensure_library_entry(session, artifact)
    session.commit()

    summary = store.cleanup_retention(
        session,
        retention_days=1,
        temporary_hours=1,
        dry_run=False,
        now=datetime.now(UTC) + timedelta(days=365),
    )
    assert summary.removed_count == 0
    assert store.resolve(artifact).exists()
    assert "unreferenced_at" not in artifact.metadata_json


def test_trashed_membership_pins_bounded_poster_closure(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    poster = store.ingest_bytes(
        session, b"poster", kind=ArtifactKind.THUMBNAIL, media_type="image/png"
    )
    video = store.ingest_bytes(session, b"video", kind=ArtifactKind.VIDEO, media_type="video/mp4")
    video.metadata_json = {"poster_artifact_id": poster.id}
    entry = ensure_library_entry(session, video)
    assert entry is not None
    entry.state = "trashed"
    entry.deleted_at = datetime.now(UTC)
    entry.recovery_id = "recovery-1"
    entry.version += 1
    session.commit()

    assert {video.id, poster.id} <= referenced_artifact_ids(session)


def test_corrupt_reference_json_aborts_retention_with_fixed_text(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    _store, session = library_session
    session.add(Job(payload_json={"artifact_ids": "not-a-list"}, result_json={}))
    with pytest.raises(ArtifactReferenceDataError, match=f"^{REFERENCE_CORRUPT}$"):
        session.flush()


def test_reference_row_cap_fails_before_parsing(
    library_session: tuple[ArtifactStore, Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    _store, session = library_session
    session.add(Job(payload_json={}, result_json={}))
    session.flush()
    monkeypatch.setattr(artifact_library, "MAX_REFERENCE_ROWS", 0)
    monkeypatch.setattr(
        artifact_library,
        "_job_ids",
        lambda _value: pytest.fail("parser reached after cap failure"),
    )

    with pytest.raises(ArtifactReferenceDataError, match=f"^{REFERENCE_CORRUPT}$"):
        referenced_artifact_ids(session)


def test_job_reference_parser_is_bounded_and_keeps_exact_fields(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    first = store.ingest_bytes(
        session, b"job-input", kind=ArtifactKind.OTHER, media_type="application/octet-stream"
    )
    second = store.ingest_bytes(
        session, b"job-source", kind=ArtifactKind.OTHER, media_type="application/octet-stream"
    )
    session.add(
        Job(
            payload_json={"input_artifact_ids": [first.id]},
            result_json={"nested": {"source_artifact_id": second.id}},
        )
    )
    session.flush()
    assert {first.id, second.id} <= referenced_artifact_ids(session)


def test_run_and_ordered_step_masks_are_strong_references(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    run_mask = store.ingest_bytes(
        session, b"run-mask", kind=ArtifactKind.INPUT, media_type="image/png"
    ).id
    step_mask = store.ingest_bytes(
        session, b"step-mask", kind=ArtifactKind.INPUT, media_type="image/png"
    ).id
    session.add(
        Run(
            id="run_mask_reference",
            chat_id="missing_chat",
            user_message_id="missing_user",
            assistant_message_id="missing_assistant",
            settings_json={"mask": {"artifact_id": run_mask}},
        )
    )
    session.add(
        WorkStep(
            id="step_mask_reference",
            plan_id="missing_plan",
            ordinal=0,
            operation="image_edit",
            settings_json={"mask": {"artifact_id": step_mask}},
        )
    )
    session.flush()

    assert {run_mask, step_mask} <= referenced_artifact_ids(session)


def test_corrupt_mask_reference_fails_closed(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    _store, session = library_session
    session.add(
        Run(
            id="run_corrupt_mask",
            chat_id="missing_chat",
            user_message_id="missing_user",
            assistant_message_id="missing_assistant",
            settings_json={"mask": "not-an-object"},
        )
    )
    with pytest.raises(ArtifactReferenceDataError, match=f"^{REFERENCE_CORRUPT}$"):
        session.flush()


def test_low_level_delete_rechecks_membership_authority(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact = store.ingest_bytes(
        session, b"cannot delete", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    ensure_library_entry(session, artifact)
    session.commit()

    with pytest.raises(ValueError, match="still retained"):
        store._delete_artifact(session, artifact)
    assert store.resolve(artifact).exists()
    assert session.get(type(artifact), artifact.id) is artifact


def test_entry_deletion_is_refused_and_never_releases_artifact_bytes(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact = store.ingest_bytes(
        session, b"system cleanup", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    entry = ensure_library_entry(session, artifact)
    assert entry is not None
    session.commit()

    with pytest.raises(ValueError, match="Media Library membership"):
        store.delete_library_artifact(session, artifact)

    session.delete(entry)
    with pytest.raises(IntegrityError, match="deletion is not authorized"):
        session.commit()
    session.rollback()
    assert session.scalars(select(ArtifactLibraryEntry)).all() == [entry]
    assert session.get(type(artifact), artifact.id) is artifact
    assert store.resolve(artifact).exists()

    with pytest.raises(IntegrityError, match="deletion is not authorized"):
        session.connection().exec_driver_sql(
            "DELETE FROM artifact_library_entries WHERE id = ?", (entry.id,)
        )
    session.rollback()
    assert session.get(ArtifactLibraryEntry, entry.id) is not None
    assert store.resolve(artifact).exists()


def test_recovery_ids_are_unique_for_trashed_entries(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    first = store.ingest_bytes(session, b"first", kind=ArtifactKind.IMAGE, media_type="image/png")
    second = store.ingest_bytes(session, b"second", kind=ArtifactKind.IMAGE, media_type="image/png")
    first_entry = ensure_library_entry(session, first)
    second_entry = ensure_library_entry(session, second)
    assert first_entry is not None and second_entry is not None
    session.commit()

    first_entry.state = "trashed"
    first_entry.deleted_at = datetime.now(UTC)
    first_entry.recovery_id = "same-recovery"
    first_entry.version += 1
    session.commit()
    second_entry.state = "trashed"
    second_entry.deleted_at = datetime.now(UTC)
    second_entry.recovery_id = "same-recovery"
    second_entry.version += 1
    with pytest.raises(IntegrityError):
        session.commit()


def test_delete_write_fence_prevents_a_concurrent_reference_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'fence.sqlite3'}",
    )
    settings.prepare()
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    store = ArtifactStore(settings)
    with Session(engine, expire_on_commit=False) as seed:
        artifact = store.ingest_bytes(
            seed, b"fenced", kind=ArtifactKind.OTHER, media_type="application/octet-stream"
        )
        chat = Chat(title="Fence")
        seed.add(chat)
        seed.flush()
        message = Message(chat_id=chat.id, role="assistant")
        seed.add(message)
        seed.commit()
        artifact_id = artifact.id
        message_id = message.id

    scanned = Event()
    writer_started = Event()
    release_delete = Event()
    original = store.referenced_artifact_ids

    def paused_references(session: Session) -> set[str]:
        found = original(session)
        scanned.set()
        assert release_delete.wait(5)
        return found

    monkeypatch.setattr(store, "referenced_artifact_ids", paused_references)

    def delete() -> None:
        with Session(engine) as session:
            owned = session.get(Artifact, artifact_id)
            assert owned is not None
            store._delete_artifact(session, owned)
            session.commit()

    def reference() -> str:
        assert scanned.wait(5)
        with Session(engine) as session:
            session.add(
                MessagePart(
                    message_id=message_id,
                    position=0,
                    type="image",
                    artifact_id=artifact_id,
                )
            )
            writer_started.set()
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return "refused"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(delete)
        referencing = pool.submit(reference)
        assert writer_started.wait(5)
        release_delete.set()
        deleting.result(timeout=10)
        assert referencing.result(timeout=10) == "refused"

    with Session(engine) as session:
        assert session.get(Artifact, artifact_id) is None
        assert session.scalars(select(MessagePart)).all() == []
    engine.dispose()


@pytest.mark.parametrize(
    "reference_kind",
    ["job", "run", "work_step", "message_reference", "chat_origin"],
)
def test_delete_write_fence_rejects_late_json_reference_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_kind: str,
) -> None:
    settings = Settings(
        data_dir=tmp_path / reference_kind,
        database_url=f"sqlite:///{tmp_path / reference_kind}.sqlite3",
    )
    settings.prepare()
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    store = ArtifactStore(settings)
    with Session(engine, expire_on_commit=False) as seed:
        artifact = store.ingest_bytes(
            seed, b"json-fenced", kind=ArtifactKind.OTHER, media_type="application/octet-stream"
        )
        chat = Chat(title="Fence")
        seed.add(chat)
        seed.flush()
        user = Message(chat_id=chat.id, role="user")
        assistant = Message(chat_id=chat.id, role="assistant")
        seed.add_all((user, assistant))
        seed.flush()
        plan = WorkPlan(chat_id=chat.id, transcript_sequence=1)
        seed.add(plan)
        seed.commit()
        artifact_id = artifact.id
        chat_id = chat.id
        user_id = user.id
        assistant_id = assistant.id
        plan_id = plan.id

    scanned = Event()
    writer_started = Event()
    release_delete = Event()
    original = store.referenced_artifact_ids

    def paused_references(session: Session) -> set[str]:
        found = original(session)
        scanned.set()
        assert release_delete.wait(5)
        return found

    monkeypatch.setattr(store, "referenced_artifact_ids", paused_references)

    def delete() -> None:
        with Session(engine) as session:
            owned = session.get(Artifact, artifact_id)
            assert owned is not None
            store._delete_artifact(session, owned)
            session.commit()

    def reference() -> str:
        assert scanned.wait(5)
        with Session(engine) as session:
            if reference_kind == "job":
                session.add(Job(payload_json={"artifact_id": artifact_id}))
            elif reference_kind == "run":
                session.add(
                    Run(
                        chat_id=chat_id,
                        user_message_id=user_id,
                        assistant_message_id=assistant_id,
                        settings_json={"mask": {"artifact_id": artifact_id}},
                    )
                )
            elif reference_kind == "work_step":
                session.add(
                    WorkStep(
                        plan_id=plan_id,
                        ordinal=0,
                        operation="image_edit",
                        input_bindings_json=[{"artifact_id": artifact_id}],
                    )
                )
            elif reference_kind == "message_reference":
                session.add(
                    MessageReference(
                        message_id=user_id,
                        position=0,
                        reference_subject_id="historical-subject",
                        mention_slug="historical-subject",
                        subject_name="Historical Subject",
                        subject_kind="person",
                        artifact_ids_json=[artifact_id],
                    )
                )
            else:
                chat = session.get(Chat, chat_id)
                assert chat is not None
                chat.scope = "studio"
                chat.origin_json = {"source_artifact_id": artifact_id}
            writer_started.set()
            try:
                session.commit()
            except ArtifactReferenceDataError:
                session.rollback()
                return "refused"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(delete)
        referencing = pool.submit(reference)
        assert writer_started.wait(5)
        release_delete.set()
        deleting.result(timeout=10)
        assert referencing.result(timeout=10) == "refused"

    with Session(engine) as session:
        assert session.get(Artifact, artifact_id) is None
        assert session.scalars(select(Job)).all() == []
        assert session.scalars(select(Run)).all() == []
        assert session.scalars(select(WorkStep)).all() == []
        assert session.scalars(select(MessageReference)).all() == []
        stored_chat = session.get(Chat, chat_id)
        assert stored_chat is not None
        assert stored_chat.scope == "standard"
        assert stored_chat.origin_json == {}
    engine.dispose()
