from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.artifacts import ArtifactStore, _path_follows_a_link
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


def test_a_real_directory_root_is_not_reported_as_a_link(tmp_path: Path) -> None:
    target = tmp_path / "artifacts"
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    store = ArtifactStore(settings, root=target)
    assert store.root == target.resolve()
    assert store.root_followed_a_link is False


def test_a_relative_real_directory_is_not_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    relative = Path("artifacts")
    relative.mkdir()
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    store = ArtifactStore(settings, root=relative)
    assert store.root == (tmp_path / "artifacts").resolve()
    assert store.root_followed_a_link is False
    assert str(store.requested_root) != str(store.root)


def test_a_linked_root_reports_the_named_path(tmp_path: Path) -> None:
    real = tmp_path / "real-artifacts"
    real.mkdir()
    linked = tmp_path / "linked-artifacts"
    linked.symlink_to(real, target_is_directory=True)
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    store = ArtifactStore(settings, root=linked)
    assert store.root == real.resolve()
    assert store.root_followed_a_link is True
    assert store.requested_root == linked


def test_a_linked_parent_of_the_root_is_reported(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    (real_parent / "artifacts").mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    requested = linked_parent / "artifacts"
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    store = ArtifactStore(settings, root=requested)
    assert store.root == (real_parent / "artifacts").resolve()
    assert store.root_followed_a_link is True
    assert store.requested_root == requested
    assert requested.is_symlink() is False


def test_the_link_predicate_does_not_fire_when_inspection_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real-artifacts"
    real.mkdir()
    linked = tmp_path / "linked-artifacts"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        "local_lm.artifacts.is_link_or_reparse",
        lambda *_args, **_kwargs: False,
    )
    assert _path_follows_a_link(linked) is False
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    store = ArtifactStore(settings, root=linked)
    assert store.root_followed_a_link is False
