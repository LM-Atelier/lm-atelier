from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

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
            _accept_replay(original, current)


def _accept_replay(original: Mapping[str, object], current: Mapping[str, object]) -> None:
    """Compare replay time as an instant, then require every other field to match."""

    assert datetime.fromisoformat(str(current["updated_at"])) >= datetime.fromisoformat(
        str(original["updated_at"])
    )
    assert current == {**original, "updated_at": current["updated_at"]}


def test_a_whole_second_replay_is_not_treated_as_earlier() -> None:
    original = {
        "id": "message-1",
        "body": "A neutral fixture.",
        "updated_at": "2026-01-01T00:00:01.000",
    }
    same_instant = {**original, "updated_at": "2026-01-01T00:00:01"}
    later = {**original, "updated_at": "2026-01-01T00:00:01.001000"}
    earlier = {**original, "updated_at": "2026-01-01T00:00:00.900000"}
    changed = {**same_instant, "body": "A different fixture."}

    _accept_replay(original, same_instant)
    _accept_replay(
        {**original, "updated_at": "2026-01-01T00:00:01"},
        later,
    )
    with pytest.raises(AssertionError):
        _accept_replay(original, earlier)
    with pytest.raises(AssertionError):
        _accept_replay(original, changed)
