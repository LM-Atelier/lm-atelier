from __future__ import annotations

import shutil
from collections.abc import Iterator
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
def artifact_session(tmp_path: Path) -> Iterator[tuple[ArtifactStore, Session]]:
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


def _ingest(store: ArtifactStore, session: Session) -> Artifact:
    artifact = store.ingest_bytes(
        session,
        b"guard-bytes",
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="guard.png",
    )
    session.commit()
    return artifact


def test_verified_path_refuses_when_the_stored_file_is_a_directory(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    artifact = _ingest(store, session)
    path = store.resolve(artifact)
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="artifact file size does not match its record"):
        store.verified_path(artifact)


def test_verified_path_refuses_when_the_stored_size_disagrees(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    artifact = _ingest(store, session)
    path = store.resolve(artifact)
    path.write_bytes(b"x" * (artifact.size_bytes + 1))
    with pytest.raises(ValueError, match="artifact file size does not match its record"):
        store.verified_path(artifact)


def test_resolve_refuses_when_the_store_root_is_replaced_by_a_link(
    artifact_session: tuple[ArtifactStore, Session],
    tmp_path: Path,
) -> None:
    store, session = artifact_session
    artifact = _ingest(store, session)
    original_root = store.root
    outside = tmp_path / "outside-store"
    shutil.copytree(original_root, outside)
    relocated = original_root.parent / (original_root.name + ".relocated")
    original_root.rename(relocated)
    original_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="artifact path escapes store"):
        store.resolve(artifact)
