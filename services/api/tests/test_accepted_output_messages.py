from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import AsyncClient


@pytest.mark.parametrize("mode,count", [("text", 1), ("image", 3), ("video", 3), ("ordered", 4)])
async def test_acceptance_returns_the_complete_output_chain_in_transcript_order(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, mode: str, count: int
) -> None:
    monkeypatch.setattr(type(app.state.services.orchestrator), "start", lambda self, *args: None)
    chat = (await client.post("/api/chats", json={"title": "Output continuity"})).json()
    payload = {
        "text": "Write a short story about a paper boat, then create an image based on it, "
        "then animate the image into a video, then summarize the video"
        if mode == "ordered"
        else "A blue paper boat",
        "mode": "auto" if mode == "ordered" else mode,
        "confirm_media": True,
        "idempotency_key": "output-continuity",
        **({"output_count": count} if mode in {"image", "video"} else {}),
    }
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
        assert response.status_code == 202, response.text
        accepted = response.json()
        outputs = accepted["assistant_messages"]
        assert len(outputs) == count
        assert outputs[0] == accepted["assistant_message"]
        assert outputs[0]["id"] == accepted["run"]["assistant_message_id"]
        assert outputs[0]["parent_id"] == accepted["user_message"]["id"]
        assert [item["parent_id"] for item in outputs[1:]] == [item["id"] for item in outputs[:-1]]
        assert {item["chat_id"] for item in outputs} == {chat["id"]}
        metadata = (await client.get(f"/api/chats/{chat['id']}/metadata")).json()
        assert metadata["active_head_message_id"] == outputs[-1]["id"]
        replay = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
        assert replay.status_code == 202
        replayed = replay.json()["assistant_messages"]
        for original, current in zip(outputs, replayed, strict=True):
            assert current["updated_at"] >= original["updated_at"]
            assert current == {**original, "updated_at": current["updated_at"]}
