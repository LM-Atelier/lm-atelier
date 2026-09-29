"""Stopping an edit that a studio session is running."""

from __future__ import annotations

import io

from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (120, 160, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


async def test_a_studio_edit_stops_and_the_session_shows_it_stopped(
    app: FastAPI, client: AsyncClient
) -> None:
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("scene.png", _png(), "image/png")}
    )
    assert uploaded.status_code == 201, uploaded.text
    source_id = uploaded.json()["id"]
    opened = await client.post("/api/studio/sessions", json={"source_artifact_id": source_id})
    assert opened.status_code == 200, opened.text
    session_id = opened.json()["id"]
    # Holding the media lease keeps the edit queued, as a busy runtime would.
    async with app.state.services.scheduler.lease("primary"):
        turn = await client.post(
            f"/api/chats/{session_id}/turns",
            json={"text": "Make it warmer.", "mode": "image", "input_artifact_ids": [source_id]},
        )
        assert turn.status_code == 202, turn.text
        answer_id = turn.json()["assistant_message"]["id"]

        stopped = await client.post(f"/api/chats/{session_id}/cancel")

        assert stopped.status_code == 200, stopped.text
        assert stopped.json()["status"] == "cancelled"
    session = (await client.get(f"/api/studio/sessions/{session_id}")).json()
    answer = next(message for message in session["messages"] if message["id"] == answer_id)
    assert answer["status"] == "cancelled"
    assert not any(message["status"] == "pending" for message in session["messages"])
    # Stopping again finds nothing running, which the studio treats as already done.
    again = await client.post(f"/api/chats/{session_id}/cancel")
    assert again.status_code == 409
