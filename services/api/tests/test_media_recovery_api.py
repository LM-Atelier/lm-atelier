"""Recovery routes hide only library membership and retain independently used files."""

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_recovery import _command, _impact
from test_media_recovery import CONTENT, _seed

from local_lm.db import SessionLocal
from local_lm.models import Artifact, ArtifactLibraryEntry, RecoveryItem, RecoveryPreviewRecord
from local_lm.schemas import EventOut


def _media(app: FastAPI) -> tuple[str, str]:
    with SessionLocal() as session:
        artifact, entry, _collection, _tag = _seed(app.state.services.artifacts, session)
        return entry.id, artifact.id


@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_media_recovery_routes_keep_identity_replay_and_publish_after_commit(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    entry_id, artifact_id = _media(app)
    observed: list[tuple[str, str | None]] = []
    publish: Callable[[str, str | None, dict[str, Any] | None], Awaitable[EventOut]] = (
        app.state.services.events.publish
    )

    async def observe(
        event_type: str, entity_id: str | None = None, payload: dict[str, Any] | None = None
    ) -> EventOut:
        if event_type == "recovery.updated":
            assert entity_id is not None and payload == {}
            with SessionLocal() as session:
                item = session.get(RecoveryItem, entity_id)
                observed.append((entity_id, item.state if item else None))
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(app.state.services.events, "publish", observe)
    preview_response = await client.get(f"/api/artifact-library/{entry_id}/deletion-impact")
    assert preview_response.status_code == 200, preview_response.text
    assert preview_response.headers["cache-control"] == "no-store"
    preview = preview_response.json()
    assert preview["kind"] == "media_library_entry"
    stale = _command(preview, "stale-media") | {"expected_revision": "a" * 64}
    assert (
        await client.post(f"/api/artifact-library/{entry_id}/trash", json=stale)
    ).status_code == 409
    assert observed == []
    command = _command(preview, "trash-media")
    trashed = await client.post(f"/api/artifact-library/{entry_id}/trash", json=command)
    assert trashed.status_code == 200, trashed.text
    item = trashed.json()
    deletion_id = item["deletion_id"]
    assert observed == [(deletion_id, "recoverable")]
    replay = await client.post(f"/api/artifact-library/{entry_id}/trash", json=command)
    assert replay.status_code == 200 and replay.json() == item
    assert not (await client.get("/api/artifact-library")).json()["items"]
    recovery = await client.get("/api/recovery-items", params={"kind": "media_library_entry"})
    assert recovery.status_code == 200
    assert [row["subject_id"] for row in recovery.json()["items"]] == [entry_id]
    assert recovery.json()["items"][0]["counts"]["retained_bytes"] == len(CONTENT)
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    command = _command(preview, f"{action}-media")
    if action == "purge":
        command["acknowledgement"] = "permanently-delete"
    result = await client.post(f"/api/recovery-items/{deletion_id}/{action}", json=command)
    assert result.status_code == 200, result.text
    assert result.json()["kind"] == "media_library_entry"
    assert result.json()["reclaimed_bytes"] == 0
    repeated = await client.post(f"/api/recovery-items/{deletion_id}/{action}", json=command)
    assert repeated.status_code == 200 and repeated.json() == result.json()
    assert observed[-1] == (deletion_id, None if action == "restore" else "purged")
    assert not (await client.get("/api/recovery-items")).json()["items"]
    with SessionLocal() as session:
        entry = session.get(ArtifactLibraryEntry, entry_id)
        assert (entry is not None) == (action == "restore")
        if entry is not None:
            assert entry.state == "visible" and entry.favorite
        artifact = session.get(Artifact, artifact_id)
        assert artifact is not None
        assert app.state.services.artifacts.resolve(artifact).read_bytes() == CONTENT


async def test_mixed_recovery_pages_bind_type_filters_before_seek_paging(
    app: FastAPI, client: AsyncClient
) -> None:
    entry_id, _artifact_id = _media(app)
    preview = await _impact(client, f"/api/artifact-library/{entry_id}/deletion-impact")
    media = await client.post(
        f"/api/artifact-library/{entry_id}/trash", json=_command(preview, "trash-mixed-media")
    )
    assert media.status_code == 200
    for number in range(2):
        chat = await client.post("/api/chats", json={"title": f"Garden chat {number}"})
        assert chat.status_code == 201
        identity = chat.json()["id"]
        preview = await _impact(client, f"/api/chats/{identity}/deletion-impact")
        assert (
            await client.post(
                f"/api/chats/{identity}/trash", json=_command(preview, f"trash-mixed-chat-{number}")
            )
        ).status_code == 200
    with SessionLocal() as session:
        previews_before = set(session.scalars(select(RecoveryPreviewRecord.revision)))
    first = await client.get("/api/recovery-items", params={"limit": 2})
    assert first.status_code == 200
    assert [item["kind"] for item in first.json()["items"]] == ["chat", "chat"]
    cursor = first.json()["next_cursor"]
    assert cursor is not None
    changed = await client.get("/api/recovery-items", params={"cursor": cursor, "kind": "chat"})
    assert changed.status_code == 400 and changed.json()["code"] == "recovery-cursor-invalid"
    last = await client.get("/api/recovery-items", params={"cursor": cursor})
    assert last.status_code == 200
    assert [item["subject_id"] for item in last.json()["items"]] == [entry_id]
    assert last.json()["next_cursor"] is None
    only_media = await client.get(
        "/api/recovery-items", params={"kind": "media_library_entry", "limit": 1}
    )
    assert only_media.status_code == 200
    assert [item["subject_id"] for item in only_media.json()["items"]] == [entry_id]
    assert only_media.json()["next_cursor"] is None
    assert (await client.get("/api/recovery-items", params={"kind": "project"})).json()[
        "items"
    ] == []
    with SessionLocal() as session:
        assert set(session.scalars(select(RecoveryPreviewRecord.revision))) == previews_before


async def test_media_route_missing_subject_and_unknown_payload_are_inert(
    app: FastAPI, client: AsyncClient
) -> None:
    entry_id, _artifact_id = _media(app)
    assert (await client.get("/api/artifact-library/missing/deletion-impact")).status_code == 404
    preview = await _impact(client, f"/api/artifact-library/{entry_id}/deletion-impact")
    invalid = _command(preview, "invalid-media") | {"delete_generated_media": True}
    assert (
        await client.post(f"/api/artifact-library/{entry_id}/trash", json=invalid)
    ).status_code == 422
    with SessionLocal() as session:
        entry = session.get(ArtifactLibraryEntry, entry_id)
        assert entry is not None
        assert entry.state == "visible"
        assert session.scalar(select(RecoveryItem)) is None
