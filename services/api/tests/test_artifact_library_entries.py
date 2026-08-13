from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from local_lm import artifact_library
from local_lm.artifact_library import (
    REFERENCE_CORRUPT,
    ArtifactReferenceDataError,
    ensure_library_entry,
    referenced_artifact_ids,
    set_library_favorite,
)
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind
from local_lm.models import ArtifactLibraryEntry, Job


@pytest.fixture
def library_session(tmp_path: Path) -> tuple[ArtifactStore, Session]:
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
    video = store.ingest_bytes(
        session, b"video", kind=ArtifactKind.VIDEO, media_type="video/mp4"
    )
    video.metadata_json = {"poster_artifact_id": poster.id}
    entry = ensure_library_entry(session, video)
    assert entry is not None
    entry.state = "trashed"
    entry.deleted_at = datetime.now(UTC)
    entry.recovery_id = "recovery-1"
    session.commit()

    assert {video.id, poster.id} <= referenced_artifact_ids(session)


def test_corrupt_reference_json_aborts_retention_with_fixed_text(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    _store, session = library_session
    session.add(Job(payload_json={"artifact_ids": "not-a-list"}, result_json={}))
    session.flush()

    with pytest.raises(ArtifactReferenceDataError, match=f"^{REFERENCE_CORRUPT}$"):
        referenced_artifact_ids(session)


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
    _store, session = library_session
    session.add(
        Job(
            payload_json={"input_artifact_ids": ["sha256:" + "1" * 64]},
            result_json={"nested": {"source_artifact_id": "sha256:" + "2" * 64}},
        )
    )
    session.flush()
    assert {"sha256:" + "1" * 64, "sha256:" + "2" * 64} <= referenced_artifact_ids(session)


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


def test_authorized_release_allows_system_cleanup(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact = store.ingest_bytes(
        session, b"system cleanup", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    ensure_library_entry(session, artifact)
    session.commit()

    with pytest.raises(ValueError, match="Media Library membership"):
        store.delete_library_artifact(session, artifact)

    refs, removed, reclaimed = store.delete_library_artifact(
        session, artifact, release_membership=True
    )
    session.commit()
    assert refs == 0
    assert removed == 1
    assert reclaimed == artifact.size_bytes or reclaimed >= 0
    assert session.scalars(select(ArtifactLibraryEntry)).all() == []
    assert session.get(type(artifact), artifact.id) is None
