"""Read conversation settings without hydrating the transcript."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message
from local_lm.prompt_helpers import PROMPT_HELPER_SCOPE


@pytest.mark.parametrize("archived", [False, True])
async def test_metadata_keeps_chat_settings_without_loading_messages(
    client: AsyncClient, archived: bool
) -> None:
    with SessionLocal() as session:
        session.add(
            Chat(
                id="chat_metadata",
                title="Harbor sketches",
                archived=archived,
                pinned=True,
                routing_mode="image",
                active_head_message_id="msg_metadata_119",
                generation_settings_json={"image": {"width": 768}},
            )
        )
        session.flush()
        session.add_all(
            Message(id=f"msg_metadata_{index:03d}", chat_id="chat_metadata") for index in range(120)
        )
        session.commit()

    loaded: list[str] = []

    def record_load(message: Message, _context: object) -> None:
        loaded.append(message.id)

    event.listen(Message, "load", record_load)
    try:
        full = await client.get("/api/chats/chat_metadata")
        assert full.status_code == 200
        assert len(loaded) == 120
        loaded.clear()
        response = await client.get("/api/chats/chat_metadata/metadata")
    finally:
        event.remove(Message, "load", record_load)

    assert response.status_code == 200
    expected = {
        key: value for key, value in full.json().items() if key not in {"messages", "web_searches"}
    }
    assert response.json() == expected
    assert loaded == []


@pytest.mark.parametrize("kind", ["missing", "helper"])
async def test_metadata_hides_unknown_and_internal_conversations(
    client: AsyncClient, kind: str
) -> None:
    if kind == "helper":
        with SessionLocal() as session:
            session.add(Chat(id="chat_metadata_hidden", scope=PROMPT_HELPER_SCOPE))
            session.commit()

    response = await client.get("/api/chats/chat_metadata_hidden/metadata")

    assert response.status_code == 404
    assert response.json().get("code") == "chat-not-found"
