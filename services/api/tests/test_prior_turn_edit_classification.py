from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message, MessagePart, WorkPlan


@pytest.mark.parametrize(
    "case",
    [
        "original",
        "accepted_edit",
        "source_changed",
        "wrong_digest",
        "wrong_chat",
        "wrong_run",
        "parent_conflict",
    ],
)
async def test_edit_classification_uses_bound_source_context_without_mutation(
    app: FastAPI, client: AsyncClient, case: str
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Source classifier"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat['id']}/turns", json={"text": "A blue paper boat", "mode": "text"}
        )
        assert source.status_code == 202, source.text
        if case == "accepted_edit":
            source = await client.post(
                f"/api/messages/{source.json()['user_message']['id']}/edits",
                json={
                    "text": "A green paper boat",
                    "idempotency_key": "source-classification-edit",
                },
            )
            assert source.status_code == 202, source.text
        source_id = source.json()["user_message"]["id"]
        loaded = await client.get(f"/api/messages/{source_id}/edit-source")
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["context_messages"] == []
        binding = {
            "source_message_id": source_id,
            "source_run_id": source.json()["run"]["id"],
            "source_snapshot_sha256": loaded.json()["source_snapshot_sha256"],
        }
        with SessionLocal() as session:
            later = Message(
                chat_id=chat["id"],
                parent_id=None,
                role="assistant",
                status="complete",
                parts=[
                    MessagePart(position=0, type="text", text="A glass orchard above a quiet sea.")
                ],
            )
            session.add(later)
            session.flush()
            stored_chat = session.get(Chat, chat["id"])
            assert stored_chat is not None
            stored_chat.active_head_message_id = later.id
            head_id = later.id
            if case == "source_changed":
                original = session.get(Message, source_id)
                assert original is not None
                original.parts[0].text = "A different boat"
            session.commit()
            plan_ids = list(
                session.scalars(select(WorkPlan.id).where(WorkPlan.chat_id == chat["id"]))
            )
        text = "Recolor the previous image from the previous story"
        ordinary = await client.post(
            f"/api/chats/{chat['id']}/classify-draft",
            json={"text": text, "mode": "image", "parent_message_id": head_id},
        )
        assert ordinary.status_code == 200, ordinary.text
        assert ordinary.json()["references_prior_visual"] is False
        target_chat_id = chat["id"]
        if case == "wrong_chat":
            target_chat_id = (await client.post("/api/chats", json={"title": "Other chat"})).json()[
                "id"
            ]
        elif case == "wrong_run":
            binding["source_run_id"] = "missing-source-run"
        elif case == "wrong_digest":
            binding["source_snapshot_sha256"] = "b" * 64
        payload: dict[str, Any] = {"text": text, "mode": "image", "edit_source": binding}
        if case == "parent_conflict":
            payload["parent_message_id"] = head_id
        classified = await client.post(f"/api/chats/{target_chat_id}/classify-draft", json=payload)
        status = (
            409
            if case in {"source_changed", "wrong_digest"}
            else 404
            if case in {"wrong_chat", "wrong_run"}
            else 422
            if case == "parent_conflict"
            else 200
        )
        assert classified.status_code == status, classified.text
        if status == 200:
            assert classified.json()["references_prior_visual"] is True
        elif status == 409:
            assert classified.json()["code"] == "edit-source-unavailable"
        with SessionLocal() as session:
            stored_chat = session.get(Chat, chat["id"])
            assert stored_chat is not None and stored_chat.active_head_message_id == head_id
            assert (
                list(session.scalars(select(WorkPlan.id).where(WorkPlan.chat_id == chat["id"])))
                == plan_ids
            )
