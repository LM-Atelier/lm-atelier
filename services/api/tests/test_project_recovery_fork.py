"""Fork retained conversations unfiled when their project is deleted."""

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_deletion import _text_exchange
from test_chat_recovery import _command, _history, _impact
from test_media_recovery import CONTENT, _seed
from test_project_recovery_api import _project, _stored_trash
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message, MessagePart


@pytest.mark.parametrize("state", ["live", "trashed", "restored", "purged"])
async def test_fork_preserves_shared_media_and_projects_only_live_filing(
    app: FastAPI, client: AsyncClient, state: str
) -> None:
    project_id, chat_id = await _project(client)
    exchange = await _text_exchange(client, chat_id, "Keep the garden paths clear")
    leaf_id = exchange["assistant_message"]["id"]
    with SessionLocal() as session:
        artifact, _entry, _collection, _tag = _seed(app.state.services.artifacts, session)
        artifact_id = artifact.id
        leaf = session.get(Message, leaf_id)
        assert leaf is not None
        position = len(leaf.parts)
        session.add(
            MessagePart(
                message_id=leaf_id, position=position, type="image", artifact_id=artifact_id
            )
        )
        session.commit()
    if state != "live":
        item = _stored_trash(project_id, datetime.now(UTC))
        if state in ("restored", "purged"):
            action = "restore" if state == "restored" else "purge"
            preview = await _impact(client, f"/api/recovery-items/{item.deletion_id}/impact")
            command = _command(preview, f"{action}-fork-project")
            if action == "purge":
                command["acknowledgement"] = "permanently-delete"
            assert (
                await client.post(f"/api/recovery-items/{item.deletion_id}/{action}", json=command)
            ).status_code == 200
    before = _history(chat_id)
    response = await client.post(f"/api/messages/{leaf_id}/fork")
    assert response.status_code == 201, response.text
    fork = response.json()
    expected = project_id if state in ("live", "restored") else None
    assert fork["id"] != chat_id and fork["project_id"] == expected
    assert _history(chat_id) == before
    with SessionLocal() as session:
        assert session.scalar(select(Chat.project_id).where(Chat.id == fork["id"])) == expected
        assert session.scalar(select(Chat.project_id).where(Chat.id == chat_id)) == (
            None if state == "purged" else project_id
        )
        original_ids = set(session.scalars(select(Message.id).where(Message.chat_id == chat_id)))
        copied_ids = set(session.scalars(select(Message.id).where(Message.chat_id == fork["id"])))
        assert len(copied_ids) == len(original_ids) == 2 and copied_ids.isdisjoint(original_ids)
        copied_parts = list(
            session.scalars(select(MessagePart).where(MessagePart.message_id.in_(copied_ids)))
        )
        assert any(part.artifact_id == artifact_id for part in copied_parts)
        assert any(part.text == "Keep the garden paths clear" for part in copied_parts)
    content = await client.get(f"/api/artifacts/{artifact_id}/content")
    assert content.status_code == 200 and content.content == CONTENT
