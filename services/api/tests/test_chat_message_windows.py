"""Reading a long conversation without loading all of it."""

from __future__ import annotations

from datetime import timedelta

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.domain import MessageRole, utcnow
from local_lm.message_window_v1 import MAX_WINDOW
from local_lm.models import Chat, Message


def _conversation(count: int, chat_id: str = "chat_window") -> list[str]:
    """A chat of ``count`` messages, oldest first, with distinct times."""

    started = utcnow() - timedelta(minutes=count + 1)
    ids: list[str] = []
    with SessionLocal() as session:
        session.add(Chat(id=chat_id, title="Window"))
        session.flush()
        for index in range(count):
            message = Message(
                id=f"msg_{chat_id}_{index:04d}",
                chat_id=chat_id,
                role=MessageRole.USER.value if index % 2 == 0 else MessageRole.ASSISTANT.value,
                created_at=started + timedelta(minutes=index),
            )
            session.add(message)
            ids.append(message.id)
        session.commit()
    return ids


async def test_opening_a_conversation_returns_its_newest_page(client: AsyncClient) -> None:
    ids = _conversation(90)

    answered = await client.get("/api/chats/chat_window/messages", params={"limit": 20})

    assert answered.status_code == 200, answered.text
    window = answered.json()
    assert [message["id"] for message in window["messages"]] == ids[-20:]
    assert window["has_older"] is True
    assert window["has_newer"] is False
    assert window["chat_id"] == "chat_window"


async def test_a_conversation_shorter_than_the_page_has_nothing_either_side(
    client: AsyncClient,
) -> None:
    ids = _conversation(3)

    window = (await client.get("/api/chats/chat_window/messages", params={"limit": 20})).json()

    assert [message["id"] for message in window["messages"]] == ids
    assert window["has_older"] is False and window["has_newer"] is False


async def test_older_pages_walk_backwards_without_repeating_the_anchor(
    client: AsyncClient,
) -> None:
    ids = _conversation(50)

    first = (await client.get("/api/chats/chat_window/messages", params={"limit": 10})).json()
    older = (
        await client.get(
            "/api/chats/chat_window/messages",
            params={"before": first["messages"][0]["id"], "limit": 10},
        )
    ).json()

    assert [message["id"] for message in older["messages"]] == ids[30:40]
    assert older["has_older"] is True
    assert older["has_newer"] is True
    assert not {message["id"] for message in older["messages"]} & {
        message["id"] for message in first["messages"]
    }


async def test_newer_pages_walk_forwards_from_an_anchor(client: AsyncClient) -> None:
    ids = _conversation(40)

    window = (
        await client.get("/api/chats/chat_window/messages", params={"after": ids[9], "limit": 10})
    ).json()

    assert [message["id"] for message in window["messages"]] == ids[10:20]
    assert window["has_newer"] is True


async def test_a_page_around_a_message_holds_it_with_both_sides(client: AsyncClient) -> None:
    ids = _conversation(40)

    window = (
        await client.get("/api/chats/chat_window/messages", params={"around": ids[20], "limit": 11})
    ).json()

    shown = [message["id"] for message in window["messages"]]
    assert ids[20] in shown
    assert len(shown) == 11
    assert shown == ids[15:26]
    assert window["has_older"] is True and window["has_newer"] is True


async def test_reading_a_long_conversation_reads_only_its_page(client: AsyncClient) -> None:
    """The point of the endpoint: cost follows the page, not the transcript."""

    _conversation(400)

    window = (await client.get("/api/chats/chat_window/messages", params={"limit": 25})).json()

    assert len(window["messages"]) == 25
    assert window["has_older"] is True


@pytest.mark.parametrize("limit", [0, -1, MAX_WINDOW + 1])
async def test_a_page_outside_the_bounds_is_refused(client: AsyncClient, limit: int) -> None:
    _conversation(5)

    refused = await client.get("/api/chats/chat_window/messages", params={"limit": limit})

    assert refused.status_code == 400
    assert refused.json()["code"] == "chat-window-invalid"


async def test_asking_for_two_anchors_at_once_is_refused(client: AsyncClient) -> None:
    ids = _conversation(5)

    refused = await client.get(
        "/api/chats/chat_window/messages", params={"before": ids[2], "after": ids[1]}
    )

    assert refused.status_code == 400
    assert refused.json()["code"] == "chat-window-ambiguous"


async def test_an_anchor_from_another_conversation_is_not_found(client: AsyncClient) -> None:
    _conversation(4)
    _conversation(4, chat_id="chat_other")

    refused = await client.get(
        "/api/chats/chat_window/messages", params={"before": "msg_chat_window_0001"}
    )

    assert refused.status_code == 200, "an anchor of this conversation is fine"

    with SessionLocal() as session:
        session.add(Message(id="msg_elsewhere", chat_id="chat_other", role=MessageRole.USER.value))
        session.commit()
    missing = await client.get(
        "/api/chats/chat_window/messages", params={"before": "msg_elsewhere"}
    )

    assert missing.status_code == 404
    assert missing.json()["code"] == "message-not-found"


async def test_an_unknown_conversation_is_not_found(client: AsyncClient) -> None:
    missing = await client.get("/api/chats/chat_missing/messages")

    assert missing.status_code == 404
    assert missing.json()["code"] == "chat-not-found"
