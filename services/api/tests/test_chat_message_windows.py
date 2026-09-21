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


def _branched(chat_id: str = "chat_branch", *, early_sibling: bool = False) -> dict[str, str]:
    """A conversation that forked: two children of one parent, one abandoned.

    Returns the ids by name. The abandoned branch is created LAST, so it is the
    newest by time while not being on the kept lineage at all.
    """

    started = utcnow() - timedelta(minutes=10)
    with SessionLocal() as session:
        session.add(Chat(id=chat_id, title="Branched"))
        session.flush()
        rows = [
            ("root", None, 0),
            ("kept_one", "root", 1),
            ("kept_two", "kept_one", 2),
            ("abandoned", "root", 3),
        ]
        if early_sibling:
            # An abandoned message OLDER than the anchor. Without it, a page
            # asking for what came before the anchor cannot tell a lineage from
            # a stretch of time, because the only other branch is the newest
            # thing in the conversation.
            rows.insert(1, ("abandoned_early", "root", 0.5))
        for name, parent, offset in rows:
            session.add(
                Message(
                    id=f"msg_{chat_id}_{name}",
                    chat_id=chat_id,
                    parent_id=f"msg_{chat_id}_{parent}" if parent else None,
                    role=MessageRole.USER.value,
                    created_at=started + timedelta(minutes=float(offset)),
                )
            )
        session.commit()
    return {name: f"msg_{chat_id}_{name}" for name, _parent, _offset in rows}


async def test_a_page_of_a_branch_holds_that_branch_and_not_the_other(
    client: AsyncClient,
) -> None:
    """The reason head_id exists: a transcript is a lineage, not everything.

    Paged by time alone, the newest page of this conversation ends with the
    abandoned message, which the reader of the kept branch never sees, and the
    lineage the reader does see is incomplete.
    """

    ids = _branched()

    by_time = (await client.get("/api/chats/chat_branch/messages")).json()
    along_branch = (
        await client.get("/api/chats/chat_branch/messages", params={"head_id": ids["kept_two"]})
    ).json()

    assert ids["abandoned"] in [message["id"] for message in by_time["messages"]]
    assert [message["id"] for message in along_branch["messages"]] == [
        ids["root"],
        ids["kept_one"],
        ids["kept_two"],
    ]


async def test_a_branch_page_bounds_itself_like_any_other(client: AsyncClient) -> None:
    ids = _branched()

    window = (
        await client.get(
            "/api/chats/chat_branch/messages",
            params={"head_id": ids["kept_two"], "limit": 2},
        )
    ).json()

    assert [message["id"] for message in window["messages"]] == [
        ids["kept_one"],
        ids["kept_two"],
    ]
    assert window["has_older"] is True
    assert window["has_newer"] is False


async def test_an_older_page_of_a_branch_stays_on_it(client: AsyncClient) -> None:
    """Asking for what came before an anchor stays on the branch as well.

    The abandoned message here is older than the anchor, so a page taken by
    time alone would hold it. Only a page that follows the lineage leaves it
    out.
    """

    ids = _branched("chat_branch_early", early_sibling=True)

    window = (
        await client.get(
            "/api/chats/chat_branch_early/messages",
            params={"head_id": ids["kept_two"], "before": ids["kept_one"], "limit": 5},
        )
    ).json()

    assert [message["id"] for message in window["messages"]] == [ids["root"]]
    assert ids["abandoned_early"] not in [message["id"] for message in window["messages"]]


async def test_a_head_from_another_conversation_is_not_found(client: AsyncClient) -> None:
    _branched()
    _conversation(2)

    missing = await client.get(
        "/api/chats/chat_window/messages", params={"head_id": "msg_chat_branch_root"}
    )

    assert missing.status_code == 404
    assert missing.json()["code"] == "message-not-found"
