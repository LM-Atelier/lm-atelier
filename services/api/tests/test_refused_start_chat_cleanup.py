"""A chat made for a start that is refused stays blank, so the client can remove it."""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from test_output_recipe_adaptation import _adapt
from test_output_recipe_replay import _profile as _recorded_profile
from test_output_recipe_replay import _recorded, _replay, _resealed
from test_picture_remix import SETTINGS, _png, _preview, _profile, _queue, _revision, _uploaded

from local_lm import api as api_module


async def _new_chat(client: AsyncClient) -> str:
    # Made as the client makes one for a start: titled as any new chat, in no project.
    response = await client.post("/api/chats", json={"title": "New chat", "project_id": None})
    assert response.status_code in (200, 201), response.text
    return str(response.json()["id"])


async def _removed_as_blank(client: AsyncClient, chat_id: str) -> None:
    # The one-chat request the client sends after a refused start, at any age.
    choice = {
        "chat_ids": [chat_id],
        "min_age_hours": 0,
        "include_archived": False,
        "include_configured": False,
    }
    preview = await client.post("/api/maintenance/empty-chats/preview", json=choice)
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert (body["strict_count"], body["configured_count"], body["conflicts"]) == (1, 0, [])

    removed = await client.post(
        "/api/maintenance/empty-chats/execute",
        json={
            **choice,
            "operation_id": str(uuid.uuid4()),
            "preview_id": body["preview_id"],
            "digest": body["digest"],
            "acknowledged_count": 1,
            "acknowledged_configured": False,
        },
    )

    assert removed.status_code == 200, removed.text
    assert removed.json()["deleted_ids"] == [chat_id]
    assert (await client.get(f"/api/chats/{chat_id}")).status_code == 404


async def test_a_refused_replay_leaves_its_chat_blank(client: AsyncClient) -> None:
    _, content = await _recorded(client, _recorded_profile())

    def without_cfg(record: dict[str, Any]) -> None:
        # Admission would fill the missing setting, so the run would not be the record's.
        del record["settings"]["unbound"]["cfg"]

    chat_id = await _new_chat(client)
    response = await _replay(client, chat_id, _resealed(content, without_cfg))

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "replay-differs"
    await _removed_as_blank(client, chat_id)


async def test_a_refused_adaptation_leaves_its_chat_blank(client: AsyncClient) -> None:
    profile_id = _recorded_profile()
    _, content = await _recorded(client, profile_id)

    def three(record: dict[str, Any]) -> None:
        record["settings"]["unbound"]["batch_size"] = 3

    chat_id = await _new_chat(client)
    response = await _adapt(
        client, chat_id, _resealed(content, three), [("profile_id", profile_id)]
    )

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "adaptation-output-count"
    await _removed_as_blank(client, chat_id)


async def test_a_refused_remix_leaves_its_chat_blank(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")
    preview = (await _preview(client, artifact_id, revision_id, profile_id, ["steps"])).json()
    real = api_module.preview_remix

    async def shown_otherwise(*args: Any, **kwargs: Any) -> Any:
        # Refused only once the turn is being built, after the chat was checked as clean.
        resolved = await real(*args, **kwargs)
        settings = {**resolved.settings, "steps": resolved.settings["steps"] + 1}
        return dataclasses.replace(resolved, settings=settings)

    monkeypatch.setattr(api_module, "preview_remix", shown_otherwise)
    chat_id = await _new_chat(client)
    async with app.state.services.scheduler.lease("primary"):
        response = await _queue(client, chat_id, artifact_id, preview, ["steps"])

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "remix-differs"
    await _removed_as_blank(client, chat_id)
