from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact


@pytest.fixture
def artifact_session(tmp_path: Path) -> tuple[ArtifactStore, Session]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{tmp_path / 'artifacts.sqlite3'}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    try:
        yield ArtifactStore(settings), session
    finally:
        session.close()
        engine.dispose()


def test_deduplicated_ingest_repairs_missing_and_corrupt_cas_files(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    content = b"durable content"
    artifact = store.ingest_bytes(
        session,
        content,
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="first.png",
    )
    session.commit()
    path = store.resolve(artifact)

    path.unlink()
    repaired_missing = store.ingest_bytes(
        session,
        content,
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
    )
    session.commit()
    assert repaired_missing.id == artifact.id
    assert path.read_bytes() == content

    path.write_bytes(b"x" * len(content))
    repaired_corrupt = store.ingest_bytes(
        session,
        content,
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
    )
    session.commit()
    assert repaired_corrupt.id == artifact.id
    assert store.verified_path(repaired_corrupt).read_bytes() == content


def test_artifact_delete_restores_file_when_database_transaction_rolls_back(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    content = b"\x89PNG\r\n\x1a\nrollback"
    artifact = store.ingest_bytes(
        session,
        content,
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
    )
    session.commit()
    path = store.resolve(artifact)

    store.delete_library_artifact(session, artifact)
    assert not path.exists()
    session.rollback()

    restored = session.get(Artifact, artifact.id)
    assert restored is not None
    assert store.verified_path(restored).read_bytes() == content


def test_artifact_delete_removes_staged_file_only_after_commit(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    artifact = store.ingest_bytes(
        session,
        b"\x89PNG\r\n\x1a\ncommit",
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
    )
    session.commit()
    path = store.resolve(artifact)

    store.delete_library_artifact(session, artifact)
    trash = store.root / ".delete-pending"
    assert not path.exists()
    assert any(trash.iterdir())

    session.commit()

    assert not path.exists()
    assert not trash.exists()


def test_retention_reconciles_crash_interrupted_artifact_deletions(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    keep = store.ingest_bytes(
        session,
        b"keep after rollback",
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
    )
    remove = store.ingest_bytes(
        session,
        b"remove after commit",
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
    )
    session.commit()
    trash = store.root / ".delete-pending"
    trash.mkdir()
    keep_path = store.resolve(keep)
    remove_path = store.resolve(remove)
    keep_staged = trash / f"{keep.sha256}.{'a' * 32}"
    remove_staged = trash / f"{remove.sha256}.{'b' * 32}"
    os.replace(keep_path, keep_staged)
    os.replace(remove_path, remove_staged)
    session.delete(remove)
    session.commit()

    store.cleanup_retention(
        session,
        retention_days=30,
        temporary_hours=24,
        dry_run=False,
    )
    session.commit()

    assert keep_path.read_bytes() == b"keep after rollback"
    assert not remove_staged.exists()
    assert not trash.exists()


def test_retention_removes_only_aged_unindexed_canonical_files(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    content = b"orphaned after a failed database commit"
    digest = hashlib.sha256(content).hexdigest()
    orphan = store.root / digest[:2] / digest[2:4] / digest
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(content)
    old = datetime.now(UTC) - timedelta(hours=25)
    os.utime(orphan, (old.timestamp(), old.timestamp()))
    crashed_ingest = store.root / "ingest-crashed"
    crashed_ingest.write_bytes(b"ingest")
    crashed_proxy = store.root / "video-proxy-crashed.mp4"
    crashed_proxy.write_bytes(b"proxy")
    restore_partial = orphan.with_name(f"{digest}.restore-partial")
    restore_partial.write_bytes(b"restore")
    randomized_restore_partial = orphan.with_name(f".{digest}.random.restore-partial")
    randomized_restore_partial.write_bytes(b"random")
    for temporary in (
        crashed_ingest,
        crashed_proxy,
        restore_partial,
        randomized_restore_partial,
    ):
        os.utime(temporary, (old.timestamp(), old.timestamp()))
    fresh_ingest = store.root / "ingest-fresh"
    fresh_ingest.write_bytes(b"preserve")
    unrelated = store.root / "do-not-delete.txt"
    unrelated.write_text("preserve", encoding="utf-8")

    cleanup = store.cleanup_retention(
        session,
        retention_days=30,
        temporary_hours=24,
        dry_run=False,
    )
    session.commit()

    assert cleanup.removed_count == 5
    assert cleanup.reclaimed_bytes == len(content) + len(b"ingestproxyrestorerandom")
    assert not orphan.exists()
    assert not crashed_ingest.exists()
    assert not crashed_proxy.exists()
    assert not restore_partial.exists()
    assert not randomized_restore_partial.exists()
    assert fresh_ingest.read_bytes() == b"preserve"
    assert unrelated.read_text(encoding="utf-8") == "preserve"


def test_artifact_metadata_strips_client_paths_and_spoofed_media_is_downloaded(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    spoofed = store.ingest_bytes(
        session,
        b"<html><script>alert('not an image')</script></html>",
        kind=ArtifactKind.INPUT,
        media_type="image/png",
        original_name=r"C:\Users\private\secret.png",
    )
    session.commit()

    path, media_type, disposition = store.delivery_metadata(spoofed)

    assert spoofed.original_name == "secret.png"
    assert path.read_bytes().startswith(b"<html>")
    assert media_type == "application/octet-stream"
    assert disposition == "attachment"


def test_artifact_resolve_rejects_noncanonical_database_paths(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, _session = artifact_session
    digest = hashlib.sha256(b"content").hexdigest()
    artifact = Artifact(
        id=f"sha256:{digest}",
        sha256=digest,
        kind=ArtifactKind.IMAGE.value,
        media_type="image/png",
        size_bytes=7,
        relative_path=f"{digest[:2]}/{digest[2:4]}/../redirected",
        metadata_json={},
    )

    with pytest.raises(ValueError, match="not canonical"):
        store.resolve(artifact)
