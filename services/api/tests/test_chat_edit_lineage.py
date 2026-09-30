"""Keep edit sources reachable without loading the surrounding transcript."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event

from local_lm.db import SessionLocal
from local_lm.models import Artifact, Chat, Message, MessagePart
from local_lm.prompt_helpers import PROMPT_HELPER_SCOPE


def _seed(count: int = 5) -> list[str]:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    ids = [f"msg_edit_{index:04d}" for index in range(count * 2)]
    with SessionLocal() as session:
        session.add(Chat(id="chat_edit"))
        session.add_all(
            Artifact(
                id=f"image_{index}",
                sha256=f"{index:064x}",
                kind="image",
                media_type="image/png",
                size_bytes=1,
                relative_path=f"image_{index}.png",
            )
            for index in range(count + 1)
        )
        session.flush()
        for index, identity in enumerate(ids):
            session.add(
                Message(
                    id=identity,
                    chat_id="chat_edit",
                    parent_id=ids[index - 1] if index else None,
                    role="user" if index % 2 == 0 else "assistant",
                    status="complete",
                    created_at=stamp,
                )
            )
            session.flush()
            session.add(
                MessagePart(
                    message_id=identity,
                    position=0,
                    type="image",
                    artifact_id=f"image_{(index + 1) // 2}",
                    metadata_json={"input_reference": True} if index % 2 == 0 else {},
                )
            )
            if index % 2 == 0:
                session.add(
                    MessagePart(
                        message_id=identity,
                        position=1,
                        type="text",
                        text=f"Adjust color {index // 2}",
                    )
                )
        session.commit()
    return ids


async def test_edit_history_pages_reach_every_step_without_message_hydration(
    client: AsyncClient,
) -> None:
    ids = _seed(105)
    loaded: list[str] = []

    def record(instance: object, _context: object) -> None:
        loaded.append(type(instance).__name__)

    for model in (Message, MessagePart, Artifact):
        event.listen(model, "load", record)
    try:
        full = await client.get("/api/chats/chat_edit")
        assert full.status_code == 200 and loaded.count("Message") == 210
        loaded.clear()
        before = None
        seen = []
        for _ in range(16):
            params: dict[str, str | int] = {"limit": 7}
            if before is not None:
                params["before"] = before
            response = await client.get(
                f"/api/chats/chat_edit/messages/{ids[-1]}/lineage", params=params
            )
            assert response.status_code == 200
            page = response.json()
            assert page["chat_id"] == "chat_edit" and page["result_message_id"] == ids[-1]
            assert len(page["steps"]) <= 7
            seen.extend(page["steps"])
            before = page["next_before"]
            if before is None:
                break
        assert before is None
        assert [step["message_id"] for step in seen] == ids[-2::-2]
        assert seen[0] == {
            "message_id": ids[-2],
            "artifact_id": "image_104",
            "instruction": "Adjust color 104",
        }
        assert seen[-1]["artifact_id"] == "image_0"
        assert loaded == []
    finally:
        for model in (Message, MessagePart, Artifact):
            event.remove(model, "load", record)


async def test_edit_sources_use_only_the_result_branch(client: AsyncClient) -> None:
    ids = _seed()
    with SessionLocal() as session:
        session.add(Message(id="sibling", chat_id="chat_edit", parent_id=ids[0], role="user"))
        session.flush()
        session.add(
            MessagePart(
                message_id="sibling",
                position=0,
                type="image",
                artifact_id="image_5",
                metadata_json={"input_reference": True},
            )
        )
        session.commit()
    response = await client.get(
        f"/api/chats/chat_edit/messages/{ids[-1]}/lineage", params={"limit": 2}
    )
    assert response.status_code == 200
    assert [step["artifact_id"] for step in response.json()["steps"]] == ["image_4", "image_3"]
    assert response.json()["next_before"] == ids[-4]
    foreign = await client.get(
        f"/api/chats/chat_edit/messages/{ids[-1]}/lineage", params={"before": "sibling"}
    )
    assert foreign.status_code == 400


@pytest.mark.parametrize("mode", ["text-only", "hidden", "removed"])
async def test_missing_or_unavailable_input_never_reuses_an_earlier_turn(
    client: AsyncClient, mode: str
) -> None:
    ids = _seed()
    with SessionLocal() as session:
        user = session.get(Message, ids[-2])
        assert user is not None
        if mode == "text-only":
            user.parts[0].metadata_json = {}
        elif mode == "hidden":
            user.transcript_visible = False
        else:
            user.parts.clear()
            session.flush()
            user.content_removed_at = datetime(2026, 1, 2, tzinfo=UTC)
        session.commit()
    response = await client.get(f"/api/chats/chat_edit/messages/{ids[-1]}/lineage")
    assert response.status_code == 200
    assert response.json()["steps"] == []
    assert response.json()["next_before"] is None


@pytest.mark.parametrize("mode", ["preview", "input", "hidden", "removed"])
async def test_unavailable_producer_ends_the_chain(client: AsyncClient, mode: str) -> None:
    ids = _seed()
    with SessionLocal() as session:
        producer = session.get(Message, ids[-3])
        assert producer is not None
        if mode in ("preview", "input"):
            producer.parts[0].metadata_json = {
                "preview" if mode == "preview" else "input_reference": True
            }
        elif mode == "hidden":
            producer.transcript_visible = False
        else:
            producer.parts.clear()
            session.flush()
            producer.content_removed_at = datetime(2026, 1, 2, tzinfo=UTC)
        session.commit()
    response = await client.get(f"/api/chats/chat_edit/messages/{ids[-1]}/lineage")
    assert response.status_code == 200
    assert len(response.json()["steps"]) == 1
    assert response.json()["next_before"] is None


@pytest.mark.parametrize(
    "mode", ["missing-chat", "helper", "foreign-result", "user-result", "missing-cursor"]
)
async def test_edit_history_refuses_unavailable_scope(client: AsyncClient, mode: str) -> None:
    ids = _seed()
    with SessionLocal() as session:
        session.add(Chat(id="other"))
        if mode == "helper":
            chat = session.get(Chat, "chat_edit")
            assert chat is not None
            chat.scope = PROMPT_HELPER_SCOPE
        session.commit()
    chat_id = (
        "missing"
        if mode == "missing-chat"
        else "other"
        if mode == "foreign-result"
        else "chat_edit"
    )
    result_id = ids[-2] if mode == "user-result" else ids[-1]
    response = await client.get(
        f"/api/chats/{chat_id}/messages/{result_id}/lineage",
        params={"before": "missing"} if mode == "missing-cursor" else {},
    )
    assert response.status_code == 404
    assert response.json().get("code") in {"chat-not-found", "message-not-found"}


@pytest.mark.parametrize("limit", [0, 101])
async def test_edit_history_rejects_unbounded_pages(client: AsyncClient, limit: int) -> None:
    ids = _seed()
    response = await client.get(
        f"/api/chats/chat_edit/messages/{ids[-1]}/lineage", params={"limit": limit}
    )
    assert response.status_code == 400
