"""Compatibility chat reads can page messages before loading their contents."""

from __future__ import annotations

from collections import Counter

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event, select
from sqlalchemy.orm import Session
from test_chat_search_pages import _seed

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message, MessagePart, Run
from local_lm.prompt_helpers import PROMPT_HELPER_SCOPE


async def test_chat_detail_pages_preserve_metadata_and_search_associations(
    client: AsyncClient,
) -> None:
    runs = _seed(7)
    whole = await client.get("/api/chats/chat_search_page")
    assert whole.status_code == 200
    original = whole.json()
    message_ids = [message["id"] for message in original["messages"]]
    seen_messages: list[str] = []
    seen_runs: list[str] = []
    for offset in (0, 3, 6):
        response = await client.get(
            "/api/chats/chat_search_page", params={"limit": 3, "offset": offset}
        )
        assert response.status_code == 200
        page = response.json()
        assert page["messages"] == original["messages"][offset : offset + 3]
        assert page["web_searches"] == original["web_searches"][offset : offset + 3]
        assert {
            key: value for key, value in page.items() if key not in {"messages", "web_searches"}
        } == {
            key: value for key, value in original.items() if key not in {"messages", "web_searches"}
        }
        seen_messages.extend(message["id"] for message in page["messages"])
        seen_runs.extend(search["run_id"] for search in page["web_searches"])
    assert seen_messages == message_ids
    assert seen_runs == runs


async def test_a_small_chat_detail_page_loads_only_its_messages_parts_and_searches(
    client: AsyncClient,
) -> None:
    runs = _seed(300)
    with SessionLocal() as session:
        messages = session.scalars(
            select(Message).where(Message.chat_id == "chat_search_page").order_by(Message.id)
        ).all()
        expected = [message.id for message in messages[210:213]]
        for index, message in enumerate(messages):
            session.add(
                MessagePart(
                    id=f"part_detail_{index}",
                    message_id=message.id,
                    position=0,
                    type="text",
                    text="A neutral example.",
                )
            )
        session.commit()
    loaded: Counter[str] = Counter()

    def count(_session: Session, row: object) -> None:
        if isinstance(row, (Message, MessagePart, Run)):
            loaded[type(row).__name__] += 1

    event.listen(SessionLocal, "loaded_as_persistent", count)
    try:
        response = await client.get(
            "/api/chats/chat_search_page", params={"limit": 3, "offset": 210}
        )
    finally:
        event.remove(SessionLocal, "loaded_as_persistent", count)
    assert response.status_code == 200
    assert [message["id"] for message in response.json()["messages"]] == expected
    assert [search["run_id"] for search in response.json()["web_searches"]] == runs[210:213]
    assert loaded == {"Message": 3, "MessagePart": 3, "Run": 3}


async def test_chat_detail_offset_without_limit_uses_a_bounded_page(client: AsyncClient) -> None:
    runs = _seed(65)
    response = await client.get("/api/chats/chat_search_page", params={"offset": 2})
    assert response.status_code == 200
    assert len(response.json()["messages"]) == 50
    assert [search["run_id"] for search in response.json()["web_searches"]] == runs[2:52]


async def test_a_chat_detail_page_after_the_end_has_no_search_history(client: AsyncClient) -> None:
    _seed(4)
    response = await client.get(
        "/api/chats/chat_search_page", params={"limit": 2, "offset": 2**63 - 1}
    )
    assert response.status_code == 200
    assert response.json()["messages"] == response.json()["web_searches"] == []


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": -1}, {"limit": 201}, {"offset": -1}, {"offset": 2**63}]
)
async def test_chat_detail_paging_rejects_invalid_bounds(
    client: AsyncClient, params: dict[str, int]
) -> None:
    _seed(2)
    response = await client.get("/api/chats/chat_search_page", params=params)
    assert response.status_code == 422


@pytest.mark.parametrize("chat_id", ["missing", "chat_search_page"])
async def test_chat_detail_paging_preserves_chat_visibility(
    client: AsyncClient, chat_id: str
) -> None:
    _seed(2)
    with SessionLocal() as session:
        chat = session.get(Chat, "chat_search_page")
        assert chat is not None
        chat.scope = PROMPT_HELPER_SCOPE
        session.commit()
    response = await client.get(f"/api/chats/{chat_id}", params={"limit": 2})
    assert response.status_code == 404
    assert response.json()["code"] == "chat-not-found"
