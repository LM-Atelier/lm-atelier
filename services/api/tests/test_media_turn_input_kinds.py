"""A media turn from pictures is refused at once when ComfyUI would be handed another file."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import Artifact

REFUSAL = "This media step requires image-compatible inputs."


def _stored(artifact_id: str, media_type: str, digit: str) -> str:
    with SessionLocal() as session:
        session.add(
            Artifact(
                id=artifact_id,
                sha256=digit * 64,
                kind="video" if media_type.startswith("video/") else "image",
                media_type=media_type,
                size_bytes=4,
                relative_path=artifact_id,
            )
        )
        session.commit()
    return artifact_id


async def _turn(client: AsyncClient, **fields: Any) -> tuple[int, dict[str, Any], str]:
    chat = (await client.post("/api/chats", json={"title": "Input kinds"})).json()
    response = await client.post(f"/api/chats/{chat['id']}/turns", json=fields)
    return response.status_code, response.json(), chat["id"]


@pytest.mark.parametrize(
    "mode,inputs",
    [
        ("video", ["guide-video"]),
        ("image", ["guide-video"]),
        ("image", ["still-picture", "guide-video"]),
    ],
    ids=["video-from-a-video", "image-from-a-video", "a-picture-and-a-video"],
)
async def test_comfyui_media_turns_from_another_kind_of_file_are_refused(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    inputs: list[str],
) -> None:
    """Without this, the turn is queued and only fails once ComfyUI is handed the file."""
    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    _stored("guide-video", "video/mp4", "a")
    _stored("still-picture", "image/png", "b")

    status, body, chat_id = await _turn(
        client,
        text="Make it move gently.",
        mode=mode,
        input_artifact_ids=inputs,
        confirm_media=True,
    )

    assert status == 422, body
    assert body["code"] == "turn-invalid"
    assert body["detail"] == REFUSAL
    assert (await client.get(f"/api/chats/{chat_id}")).json()["messages"] == []


async def test_an_ordered_plan_whose_first_step_would_be_handed_a_video_is_refused(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first step of a plan receives every input, so a picture beside a video is no help."""
    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    _stored("guide-video", "video/mp4", "a")
    _stored("still-picture", "image/png", "b")

    status, body, chat_id = await _turn(
        client,
        text="Create an image of a paper boat, then write a short story about it",
        mode="auto",
        input_artifact_ids=["still-picture", "guide-video"],
        confirm_media=True,
    )

    assert status == 422, body
    assert body["code"] == "turn-invalid"
    assert body["detail"] == "The first media step requires image-compatible inputs."
    assert (await client.get(f"/api/chats/{chat_id}")).json()["messages"] == []
    plans = await client.get("/api/work-plans", params={"chat_id": chat_id})
    assert plans.json() == []


async def test_a_conversation_turn_may_still_carry_a_video(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    _stored("guide-video", "video/mp4", "a")

    status, body, _ = await _turn(
        client, text="What happens in this clip?", mode="text", input_artifact_ids=["guide-video"]
    )

    assert body.get("detail") != REFUSAL
    assert status == 202, body


async def test_an_engine_that_may_take_other_files_is_not_refused(client: AsyncClient) -> None:
    """The mock engine, like an adapter that reads video, is left to decide for itself."""
    _stored("guide-video", "video/mp4", "a")

    status, body, _ = await _turn(
        client,
        text="Make it move gently.",
        mode="video",
        input_artifact_ids=["guide-video"],
        confirm_media=True,
    )

    assert body.get("detail") != REFUSAL
    assert status == 202, body
