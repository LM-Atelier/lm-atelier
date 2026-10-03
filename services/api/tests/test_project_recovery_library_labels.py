"""Deleted project labels disappear from shared media without hiding its references."""

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_recovery import _command, _impact
from test_media_recovery import CONTENT, _seed
from test_project_recovery_api import _project, _stored_trash
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message, MessagePart


@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_library_keeps_shared_references_but_omits_a_deleted_projects_identity(
    app: FastAPI,
    client: AsyncClient,
    action: str,
) -> None:
    project_id, chat_id = await _project(client)
    other_project_id, other_id = await _project(client, "Garden paths")
    with SessionLocal() as session:
        artifact, _entry, _collection, _tag = _seed(app.state.services.artifacts, session)
        artifact_id = artifact.id
        messages = [
            Message(chat_id=identity, role="assistant", status="complete")
            for identity in (chat_id, other_id)
        ]
        session.add_all(messages)
        session.flush()
        session.add_all(
            MessagePart(message_id=message.id, position=0, type="image", artifact_id=artifact_id)
            for message in messages
        )
        session.commit()
    before = (await client.get("/api/artifacts")).json()
    assert len(before) == 1 and before[0]["reference_count"] == 2
    assert before[0]["project_ids"] == sorted([project_id, other_project_id])
    item = _stored_trash(project_id, datetime.now(UTC))
    response = await client.get("/api/artifacts", params={"kind": "image", "limit": 1})
    assert response.status_code == 200
    current = response.json()
    assert len(current) == 1 and current[0]["id"] == artifact_id
    assert current[0]["reference_count"] == 2
    assert current[0]["chat_ids"] == sorted([chat_id, other_id])
    assert current[0]["project_ids"] == [other_project_id]
    assert (
        await client.get("/api/artifacts", params={"project_id": project_id, "limit": 1})
    ).json() == []
    assert [
        row["id"]
        for row in (
            await client.get("/api/artifacts", params={"project_id": other_project_id, "limit": 1})
        ).json()
    ] == [artifact_id]
    assert [
        row["id"]
        for row in (await client.get("/api/artifacts", params={"chat_id": chat_id})).json()
    ] == [artifact_id]
    with SessionLocal() as session:
        assert session.scalar(select(Chat.project_id).where(Chat.id == chat_id)) == project_id
    preview = await _impact(client, f"/api/recovery-items/{item.deletion_id}/impact")
    command = _command(preview, f"{action}-library-project")
    if action == "purge":
        command["acknowledgement"] = "permanently-delete"
    assert (
        await client.post(f"/api/recovery-items/{item.deletion_id}/{action}", json=command)
    ).status_code == 200
    after = (await client.get("/api/artifacts")).json()
    assert after == (before if action == "restore" else current)
    content = await client.get(f"/api/artifacts/{artifact_id}/content")
    assert content.status_code == 200 and content.content == CONTENT
