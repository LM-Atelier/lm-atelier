"""Keep the beginning of a long branch reachable through transcript pages."""

from __future__ import annotations

from datetime import timedelta

import pytest
from httpx2 import AsyncClient
from sqlalchemy import insert

from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import Chat, Message


async def test_oldest_messages_remain_reachable_in_a_long_branch(client: AsyncClient) -> None:
    ids = [f"msg_long_branch_{index:05d}" for index in range(10_050)]
    started = utcnow() - timedelta(days=1)
    with SessionLocal() as session:
        session.add(Chat(id="chat_long_branch", active_head_message_id=ids[-1]))
        session.flush()
        session.execute(
            insert(Message),
            [
                {
                    "id": identity,
                    "chat_id": "chat_long_branch",
                    "parent_id": ids[index - 1] if index else None,
                    "created_at": started + timedelta(seconds=index),
                }
                for index, identity in enumerate(ids)
            ],
        )
        session.commit()

    path = "/api/chats/chat_long_branch/messages"
    latest = await client.get(path, params={"head_id": ids[-1], "limit": 5})
    assert latest.status_code == 200
    assert [message["id"] for message in latest.json()["messages"]] == ids[-5:]

    oldest = await client.get(path, params={"head_id": ids[-1], "before": ids[5], "limit": 5})

    assert oldest.status_code == 200
    assert [message["id"] for message in oldest.json()["messages"]] == ids[:5]
    assert oldest.json()["has_older"] is False


@pytest.mark.parametrize("count", [1, 2])
async def test_a_cyclic_branch_returns_each_message_once(client: AsyncClient, count: int) -> None:
    ids = [f"msg_cycle_{index}" for index in range(count)]
    with SessionLocal() as session:
        session.add(Chat(id="chat_cycle"))
        session.flush()
        session.add_all(Message(id=identity, chat_id="chat_cycle") for identity in ids)
        session.flush()
        for index, identity in enumerate(ids):
            message = session.get(Message, identity)
            assert message is not None
            message.parent_id = ids[index - 1]
        session.commit()

    response = await client.get(
        "/api/chats/chat_cycle/messages", params={"head_id": ids[-1], "limit": 5}
    )

    assert response.status_code == 200
    window = response.json()
    assert sorted(message["id"] for message in window["messages"]) == ids
    assert window["has_older"] is False
    assert window["has_newer"] is False
