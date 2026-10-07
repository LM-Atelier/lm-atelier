"""Keep branch-wide composer facts without reading message payloads."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event, insert

from local_lm.db import SessionLocal
from local_lm.models import Artifact, Chat, Message, MessagePart, ResponseRevision
from local_lm.prompt_helpers import PROMPT_HELPER_SCOPE


def _chain(
    *,
    kind: str = "image",
    pending: str = "none",
    preview: object = False,
    visible: bool = True,
    sibling: bool = False,
) -> list[str]:
    ids = [f"msg_context_{index:03d}" for index in range(80)]
    with SessionLocal() as session:
        session.add(Chat(id="chat_context", active_head_message_id=ids[-1]))
        session.flush()
        session.execute(
            insert(Message),
            [
                {
                    "id": identity,
                    "chat_id": "chat_context",
                    "parent_id": ids[index - 1] if index else None,
                    "role": "assistant",
                }
                for index, identity in enumerate(ids)
            ],
        )
        target = ids[0]
        if sibling:
            target = "msg_context_sibling"
            session.add(
                Message(id=target, chat_id="chat_context", parent_id=ids[0], role="assistant")
            )
            session.flush()
        message = session.get(Message, target)
        assert message is not None
        message.transcript_visible = visible
        message.status = "pending" if pending == "message" else "complete"
        session.add(
            Artifact(
                id="artifact_context",
                sha256="a" * 64,
                kind=kind,
                media_type="image/png" if kind == "image" else "video/mp4",
                size_bytes=1,
                relative_path="context.bin",
            )
        )
        session.flush()
        session.add(
            MessagePart(
                message_id=target,
                position=0,
                type=kind,
                artifact_id="artifact_context",
                metadata_json={"preview": preview},
            )
        )
        if pending == "revision":
            session.add(ResponseRevision(message_id=target, sequence=1, status="pending"))
        session.commit()
    return ids


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize("pending", ["none", "message", "revision"])
async def test_context_keeps_off_page_visuals_and_pending_work_without_hydration(
    client: AsyncClient, kind: str, pending: str
) -> None:
    ids = _chain(kind=kind, pending=pending)
    loaded: list[str] = []

    def record_load(instance: object, _context: object) -> None:
        loaded.append(type(instance).__name__)

    models = (Message, MessagePart, ResponseRevision, Artifact)
    for model in models:
        event.listen(model, "load", record_load)
    try:
        full = await client.get("/api/chats/chat_context")
        assert full.status_code == 200
        assert loaded.count("Message") == 80
        assert "MessagePart" in loaded and "Artifact" in loaded
        loaded.clear()
        response = await client.get("/api/chats/chat_context/context", params={"head_id": ids[-1]})
    finally:
        for model in models:
            event.remove(model, "load", record_load)

    assert response.status_code == 200
    assert response.json() == {
        "chat_id": "chat_context",
        "head_id": ids[-1],
        "has_prior_visual": True,
        "has_prior_image": kind == "image",
        "has_pending_response": pending != "none",
    }
    assert loaded == []


@pytest.mark.parametrize("preview", [True, False, None, 1, "true"])
async def test_only_a_boolean_preview_flag_excludes_a_visual(
    client: AsyncClient, preview: object
) -> None:
    ids = _chain(preview=preview)
    response = await client.get("/api/chats/chat_context/context", params={"head_id": ids[-1]})

    assert response.status_code == 200
    assert response.json()["has_prior_visual"] is (preview is not True)
    assert response.json()["has_prior_image"] is (preview is not True)


async def test_hidden_messages_do_not_supply_visuals_or_pending_work(client: AsyncClient) -> None:
    ids = _chain(visible=False, pending="revision")
    response = await client.get("/api/chats/chat_context/context", params={"head_id": ids[-1]})

    assert response.status_code == 200
    assert response.json()["has_prior_visual"] is False
    assert response.json()["has_prior_image"] is False
    assert response.json()["has_pending_response"] is False


async def test_context_follows_the_selected_branch_and_all_messages_without_a_head(
    client: AsyncClient,
) -> None:
    ids = _chain(sibling=True, pending="revision")
    selected = await client.get("/api/chats/chat_context/context", params={"head_id": ids[-1]})
    all_messages = await client.get("/api/chats/chat_context/context")

    assert selected.status_code == all_messages.status_code == 200
    for field in ("has_prior_visual", "has_prior_image", "has_pending_response"):
        assert selected.json()[field] is False
        assert all_messages.json()[field] is True
    assert all_messages.json()["head_id"] is None


@pytest.mark.parametrize("kind", ["missing", "helper", "foreign-head"])
async def test_context_refuses_missing_internal_or_mismatched_conversations(
    client: AsyncClient, kind: str
) -> None:
    parameters: dict[str, str] = {}
    with SessionLocal() as session:
        if kind == "helper":
            session.add(Chat(id="chat_unavailable", scope=PROMPT_HELPER_SCOPE))
        elif kind == "foreign-head":
            session.add_all([Chat(id="chat_unavailable"), Chat(id="chat_foreign")])
            session.flush()
            session.add(Message(id="msg_foreign", chat_id="chat_foreign"))
            parameters["head_id"] = "msg_foreign"
        session.commit()

    response = await client.get("/api/chats/chat_unavailable/context", params=parameters)

    assert response.status_code == 404
    expected = "message-not-found" if kind == "foreign-head" else "chat-not-found"
    assert response.json().get("code") == expected
