"""Recovery identity, deadlines and command results survive real application restarts."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import func, select
from test_chat_recovery import _command, _impact
from test_media_recovery import CONTENT, _seed

from local_lm import recovery_api
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.main import create_app
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Job,
    MediaCollectionMembership,
    MediaTagAssignment,
    RecoveryItem,
)


@asynccontextmanager
async def _open(settings: Settings) -> AsyncIterator[tuple[FastAPI, AsyncClient]]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            session = await client.post("/api/session")
            assert session.status_code == 200
            client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]
            yield app, client


@pytest.mark.parametrize("action", ["restore", "expired-purge"])
async def test_restart_keeps_the_original_membership_deadline_and_exact_operation_results(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    async with _open(settings) as (app, client):
        with SessionLocal() as session:
            artifact, entry, collection, tag = _seed(app.state.services.artifacts, session)
            artifact_id, entry_id, collection_id, tag_id = (
                artifact.id,
                entry.id,
                collection.id,
                tag.id,
            )
        preview = await _impact(client, f"/api/artifact-library/{entry_id}/deletion-impact")
        trash_command = _command(preview, "restart-trash-garden")
        response = await client.post(f"/api/artifact-library/{entry_id}/trash", json=trash_command)
        assert response.status_code == 200
        item = response.json()
        deletion_id = item["deletion_id"]

    if action == "expired-purge":
        deadline = datetime.fromisoformat(item["purge_after"])

        class DeadlineClock(datetime):
            @classmethod
            def __get_pydantic_core_schema__(cls, _source, handler):
                return handler(datetime)

            @classmethod
            def now(cls, tz=None):
                return deadline.astimezone(tz) if tz else deadline.replace(tzinfo=None)

        monkeypatch.setattr(recovery_api, "datetime", DeadlineClock)

    async with _open(settings) as (app, client):
        page = await client.get("/api/recovery-items", params={"kind": "media_library_entry"})
        assert page.status_code == 200
        assert page.json()["items"] == [item]
        repeated = await client.post(f"/api/artifact-library/{entry_id}/trash", json=trash_command)
        assert repeated.status_code == 200 and repeated.json() == item
        with SessionLocal() as session:
            entry = session.get(ArtifactLibraryEntry, entry_id)
            artifact = session.get(Artifact, artifact_id)
            assert entry is not None and artifact is not None
            assert entry.state == "trashed" and entry.favorite and artifact.favorite
            assert app.state.services.artifacts.resolve(artifact).read_bytes() == CONTENT
            assert (
                session.get(MediaCollectionMembership, (collection_id, entry_id)).note
                == "Keep original spacing"
            )
            assert session.get(MediaTagAssignment, (tag_id, entry_id)) is not None
        impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
        command = _command(impact, "restart-restore-garden")
        endpoint = "restore"
        if action == "expired-purge":
            assert impact["available_actions"] == ["purge"]
            refused = await client.post(f"/api/recovery-items/{deletion_id}/restore", json=command)
            assert (
                refused.status_code == 409 and refused.json()["code"] == "recovery-window-expired"
            )
            with SessionLocal() as session:
                row = session.get(RecoveryItem, deletion_id)
                assert row is not None
                assert row.purge_after.replace(tzinfo=UTC) == deadline
                assert row.state == "recoverable"
            command = _command(impact, "restart-purge-garden") | {
                "acknowledgement": "permanently-delete"
            }
            endpoint = "purge"
        result = await client.post(f"/api/recovery-items/{deletion_id}/{endpoint}", json=command)
        assert result.status_code == 200, result.text
        result_body = result.json()
        assert result_body["reclaimed_bytes"] == 0

    async with _open(settings) as (app, client):
        assert (await client.get("/api/recovery-items")).json()["items"] == []
        replay = await client.post(f"/api/recovery-items/{deletion_id}/{endpoint}", json=command)
        assert replay.status_code == 200 and replay.json() == result_body
        with SessionLocal() as session:
            artifact = session.get(Artifact, artifact_id)
            assert artifact is not None
            assert app.state.services.artifacts.resolve(artifact).read_bytes() == CONTENT
            entry = session.get(ArtifactLibraryEntry, entry_id)
            if action == "restore":
                assert entry is not None and entry.state == "visible" and entry.favorite
                assert entry.id == entry_id and entry.artifact_id == artifact_id
                assert session.get(RecoveryItem, deletion_id) is None
                assert (
                    session.get(MediaCollectionMembership, (collection_id, entry_id)).note
                    == "Keep original spacing"
                )
                assert session.get(MediaTagAssignment, (tag_id, entry_id)) is not None
            else:
                assert entry is None and not artifact.favorite
                row = session.get(RecoveryItem, deletion_id)
                assert row is not None and row.state == "purged"
                assert row.purge_after.replace(tzinfo=UTC) == deadline
            assert session.scalar(select(func.count()).select_from(Job)) == 0
