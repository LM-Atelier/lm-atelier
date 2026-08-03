"""Favorites: pinned against the sweep, never against an explicit delete."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx2 import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind


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


async def test_the_flag_toggles_filters_and_never_blocks_deletion(client: AsyncClient) -> None:
    from local_lm.db import SessionLocal
    from local_lm.domain import ArtifactKind as Kind
    from local_lm.models import Artifact

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

    only_favorites = await client.get("/api/artifacts", params={"favorites": "true"})
    assert [item["id"] for item in only_favorites.json()] == ["art_favorite_flow"]
    everything = await client.get("/api/artifacts")
    assert any(item["id"] == "art_favorite_flow" for item in everything.json())

    cleared = await client.patch("/api/artifacts/art_favorite_flow", json={"favorite": False})
    assert cleared.json()["favorite"] is False
    assert (await client.get("/api/artifacts", params={"favorites": "true"})).json() == []

    # Explicit deletion wins over the flag; a favorite is not undeletable.
    await client.patch("/api/artifacts/art_favorite_flow", json={"favorite": True})
    deleted = await client.delete("/api/artifacts/art_favorite_flow")
    assert deleted.status_code == 200
    assert (await client.get("/api/artifacts")).json() == []


async def test_a_missing_artifact_refuses_with_a_stable_code(client: AsyncClient) -> None:
    response = await client.patch("/api/artifacts/absent", json={"favorite": True})
    assert response.status_code == 404
    assert response.json()["code"] == "artifact-not-found"
