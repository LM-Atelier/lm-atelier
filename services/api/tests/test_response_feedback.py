"""Preference feedback: a local verdict with provenance, never a training run."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient

pytestmark = pytest.mark.asyncio


def _seed_exchange() -> tuple[str, str, str]:
    from local_lm.db import SessionLocal
    from local_lm.domain import MessageRole, MessageStatus
    from local_lm.models import Chat, Message, ResponseRevision

    with SessionLocal() as session:
        chat = Chat(title="Feedback chat")
        session.add(chat)
        session.flush()
        message = Message(
            chat_id=chat.id,
            role=MessageRole.ASSISTANT.value,
            status=MessageStatus.COMPLETE.value,
        )
        session.add(message)
        session.flush()
        revision = ResponseRevision(
            message_id=message.id,
            sequence=1,
            status=MessageStatus.COMPLETE.value,
        )
        session.add(revision)
        session.commit()
        return chat.id, message.id, revision.id


async def test_a_verdict_sets_changes_and_clears(client: AsyncClient) -> None:
    chat_id, message_id, revision_id = _seed_exchange()

    liked = await client.put(f"/api/messages/{message_id}/feedback", json={"rating": "up"})
    assert liked.status_code == 200
    assert liked.json()["rating"] == "up"

    changed = await client.put(f"/api/messages/{message_id}/feedback", json={"rating": "down"})
    assert changed.json()["rating"] == "down"

    detail = await client.get(f"/api/chats/{chat_id}")
    message = next(item for item in detail.json()["messages"] if item["id"] == message_id)
    assert message["feedback"] == "down"
    # The base verdict and a revision verdict are independent targets.
    assert message["response_revisions"][0]["feedback"] is None

    revised = await client.put(
        f"/api/messages/{message_id}/feedback",
        json={"rating": "up", "response_revision_id": revision_id},
    )
    assert revised.json()["rating"] == "up"
    detail = await client.get(f"/api/chats/{chat_id}")
    message = next(item for item in detail.json()["messages"] if item["id"] == message_id)
    assert message["feedback"] == "down"
    assert message["response_revisions"][0]["feedback"] == "up"

    cleared = await client.put(f"/api/messages/{message_id}/feedback", json={"rating": None})
    assert cleared.json()["rating"] is None
    detail = await client.get(f"/api/chats/{chat_id}")
    message = next(item for item in detail.json()["messages"] if item["id"] == message_id)
    assert message["feedback"] is None
    assert message["response_revisions"][0]["feedback"] == "up"


async def test_missing_targets_refuse_with_stable_codes(client: AsyncClient) -> None:
    _, message_id, _ = _seed_exchange()

    absent_message = await client.put("/api/messages/absent/feedback", json={"rating": "up"})
    assert absent_message.status_code == 404
    assert absent_message.json()["code"] == "message-not-found"

    absent_revision = await client.put(
        f"/api/messages/{message_id}/feedback",
        json={"rating": "up", "response_revision_id": "absent"},
    )
    assert absent_revision.status_code == 404
    assert absent_revision.json()["code"] == "revision-not-found"
