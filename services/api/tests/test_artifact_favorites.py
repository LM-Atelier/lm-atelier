"""Favorites: pinned against the sweep, never against an explicit delete."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx2 import AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind


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


def test_a_favorite_is_never_marked_or_swept(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    artifact = store.ingest_bytes(
        session,
        b"kept pixels",
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="kept.png",
    )
    artifact.favorite = True
    session.commit()

    late = datetime.now(UTC) + timedelta(days=365)
    summary = store.cleanup_retention(
        session,
        retention_days=1,
        temporary_hours=1,
        dry_run=False,
        now=late,
    )

    assert summary.removed_count == 0
    session.refresh(artifact)
    assert "unreferenced_at" not in artifact.metadata_json
    assert store.resolve(artifact).exists()


def test_favoriting_clears_an_existing_retention_mark(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    artifact = store.ingest_bytes(
        session,
        b"reprieved pixels",
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
    )
    session.commit()
    # First sweep marks the unreferenced artifact and starts its clock.
    store.cleanup_retention(session, retention_days=30, temporary_hours=1, dry_run=False)
    session.refresh(artifact)
    assert "unreferenced_at" in artifact.metadata_json

    artifact.favorite = True
    session.commit()
    late = datetime.now(UTC) + timedelta(days=365)
    summary = store.cleanup_retention(
        session,
        retention_days=30,
        temporary_hours=1,
        dry_run=False,
        now=late,
    )

    assert summary.removed_count == 0
    session.refresh(artifact)
    # The mark is gone: unfavoriting later restarts the clock from zero.
    assert "unreferenced_at" not in artifact.metadata_json


async def test_the_flag_toggles_filters_and_library_membership_blocks_legacy_deletion(
    client: AsyncClient,
) -> None:
    from local_lm.db import SessionLocal
    from local_lm.domain import ArtifactKind as Kind
    from local_lm.models import Artifact, ArtifactLibraryEntry

    with SessionLocal() as session:
        session.add(
            Artifact(
                id="art_favorite_flow",
                sha256="f" * 64,
                kind=Kind.IMAGE.value,
                media_type="image/png",
                size_bytes=4,
                relative_path="cas/f/favorite.png",
                original_name="favorite.png",
            )
        )
        session.commit()

    flagged = await client.patch("/api/artifacts/art_favorite_flow", json={"favorite": True})
    assert flagged.status_code == 200
    assert flagged.json()["favorite"] is True
    with SessionLocal() as session:
        entry = session.get(ArtifactLibraryEntry, "libentry:sha256:" + "f" * 64)
        assert entry is not None
        assert entry.favorite is True
        assert entry.version == 2

    only_favorites = await client.get("/api/artifacts", params={"favorites": "true"})
    assert [item["id"] for item in only_favorites.json()] == ["art_favorite_flow"]
    everything = await client.get("/api/artifacts")
    assert any(item["id"] == "art_favorite_flow" for item in everything.json())

    cleared = await client.patch("/api/artifacts/art_favorite_flow", json={"favorite": False})
    assert cleared.json()["favorite"] is False
    assert (await client.get("/api/artifacts", params={"favorites": "true"})).json() == []

    # Without Trash support, legacy hard deletion cannot bypass membership.
    await client.patch("/api/artifacts/art_favorite_flow", json={"favorite": True})
    deleted = await client.delete("/api/artifacts/art_favorite_flow")
    assert deleted.status_code == 409
    assert deleted.json() == {
        "detail": "This media item is retained by its Media Library membership.",
        "code": "artifact-in-use",
    }
    remaining = (await client.get("/api/artifacts")).json()
    assert [item["id"] for item in remaining] == ["art_favorite_flow"]
    assert remaining[0]["favorite"] is True


async def test_favorite_conflict_is_fixed_and_private(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import api
    from local_lm.artifact_library import ArtifactLibraryConflict

    created = await client.post(
        "/api/artifacts?kind=image",
        files={"file": ("race.png", b"race", "image/png")},
    )
    marker = "private stale writer marker"

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise ArtifactLibraryConflict(marker)

    monkeypatch.setattr(api, "set_library_favorite", refuse)
    response = await client.patch(f"/api/artifacts/{created.json()['id']}", json={"favorite": True})
    assert response.status_code == 409
    assert response.json() == {
        "detail": "The Media Library item changed. Refresh and try again.",
        "code": "artifact-library-conflict",
    }
    assert marker not in response.text


async def test_a_missing_artifact_refuses_with_a_stable_code(client: AsyncClient) -> None:
    response = await client.patch("/api/artifacts/absent", json={"favorite": True})
    assert response.status_code == 404
    assert response.json()["code"] == "artifact-not-found"


async def test_only_explicit_media_upload_is_published_to_library(
    client: AsyncClient,
) -> None:
    from local_lm.db import SessionLocal
    from local_lm.models import ArtifactLibraryEntry

    image = await client.post(
        "/api/artifacts?kind=image",
        files={"file": ("published.png", b"published", "image/png")},
    )
    raw_input = await client.post(
        "/api/artifacts",
        files={"file": ("input.png", b"input", "image/png")},
    )
    assert image.status_code == raw_input.status_code == 201
    with SessionLocal() as session:
        entries = session.scalars(select(ArtifactLibraryEntry)).all()
        assert [entry.artifact_id for entry in entries] == [image.json()["id"]]
