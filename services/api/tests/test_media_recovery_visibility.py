"""Library pickers hide removed membership while existing chat media keeps working."""

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_chat_recovery import _command, _impact
from test_media_recovery_api import _media

from local_lm.db import SessionLocal
from local_lm.models import Artifact, Chat, Message, MessagePart, Run


@pytest.mark.parametrize("action", ["trash", "purge"])
async def test_removed_membership_is_absent_from_library_pickers_but_shared_chat_media_survives(
    app: FastAPI, client: AsyncClient, action: str
) -> None:
    entry_id, artifact_id = _media(app)
    created = await client.post("/api/chats", json={"title": "Shared garden drawing"})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    with SessionLocal() as session:
        message = Message(chat_id=chat_id, role="assistant", status="complete")
        session.add(message)
        session.flush()
        session.add(
            MessagePart(message_id=message.id, position=0, type="image", artifact_id=artifact_id)
        )
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.active_head_message_id = message.id
        session.commit()
    before = await client.get("/api/artifacts", params={"kind": "image", "limit": 1})
    assert before.status_code == 200
    assert [row["id"] for row in before.json()] == [artifact_id]
    preview = await _impact(client, f"/api/artifact-library/{entry_id}/deletion-impact")
    trashed = await client.post(
        f"/api/artifact-library/{entry_id}/trash", json=_command(preview, "trash-picker-garden")
    )
    assert trashed.status_code == 200
    deletion_id = trashed.json()["deletion_id"]
    if action == "purge":
        preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
        command = _command(preview, "purge-picker-garden") | {
            "acknowledgement": "permanently-delete"
        }
        assert (
            await client.post(f"/api/recovery-items/{deletion_id}/purge", json=command)
        ).status_code == 200
    transcript = await client.get(f"/api/chats/{chat_id}/messages")
    assert transcript.status_code == 200
    assert [
        part["artifact_id"] for row in transcript.json()["messages"] for part in row["parts"]
    ] == [artifact_id]
    assert (await client.get(f"/api/artifacts/{artifact_id}/content")).status_code == 200
    picker_filters: tuple[dict[str, str | int], ...] = (
        {"kind": "image", "limit": 1},
        {"kind": "image", "chat_id": chat_id},
        {"kind": "image", "query": "Garden"},
    )
    for params in picker_filters:
        picker = await client.get("/api/artifacts", params=params)
        assert picker.status_code == 200
        assert artifact_id not in [row["id"] for row in picker.json()]
    if action == "trash":
        preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
        restored = await client.post(
            f"/api/recovery-items/{deletion_id}/restore",
            json=_command(preview, "restore-picker-garden"),
        )
        assert restored.status_code == 200
        picker = await client.get("/api/artifacts", params={"kind": "image", "chat_id": chat_id})
        assert [row["id"] for row in picker.json()] == [artifact_id]


async def test_picker_applies_visible_membership_before_the_page_limit(
    app: FastAPI, client: AsyncClient
) -> None:
    visible = await client.post(
        "/api/artifacts?kind=image",
        files={"file": ("Older garden.png", b"older constructed image", "image/png")},
    )
    assert visible.status_code == 201
    entry_id, _artifact_id = _media(app)
    preview = await _impact(client, f"/api/artifact-library/{entry_id}/deletion-impact")
    assert (
        await client.post(
            f"/api/artifact-library/{entry_id}/trash",
            json=_command(preview, "trash-before-picker-limit"),
        )
    ).status_code == 200
    unpublished = await client.post(
        "/api/artifacts",
        files={"file": ("Input garden.png", b"unpublished constructed image", "image/png")},
    )
    assert unpublished.status_code == 201
    picker = await client.get("/api/artifacts", params={"kind": "image", "limit": 1})
    assert picker.status_code == 200
    assert [row["id"] for row in picker.json()] == [visible.json()["id"]]
    assert (
        await client.get(f"/api/artifacts/{unpublished.json()['id']}/content")
    ).status_code == 200


async def test_picker_omits_deleted_chat_references_and_generation_labels_until_restore(
    app: FastAPI, client: AsyncClient
) -> None:
    _entry_id, artifact_id = _media(app)
    project = await client.post("/api/projects", json={"name": "Garden project"})
    assert project.status_code == 201
    project_id = project.json()["id"]
    removed = await client.post(
        "/api/chats", json={"title": "Original garden", "project_id": project_id}
    )
    live = await client.post("/api/chats", json={"title": "Shared garden"})
    assert removed.status_code == live.status_code == 201
    removed_id, live_id = removed.json()["id"], live.json()["id"]
    with SessionLocal() as session:
        original = Message(chat_id=removed_id, role="user", status="complete")
        assistant = Message(chat_id=removed_id, role="assistant", status="complete")
        shared = Message(chat_id=live_id, role="assistant", status="complete")
        session.add_all([original, assistant, shared])
        session.flush()
        session.add_all(
            [
                MessagePart(
                    message_id=message.id, position=0, type="image", artifact_id=artifact_id
                )
                for message in (assistant, shared)
            ]
        )
        run = Run(
            chat_id=removed_id,
            user_message_id=original.id,
            assistant_message_id=assistant.id,
            operation="text_to_image",
            status="complete",
            provenance_json={"model": {"profile_name": "Garden model"}},
        )
        session.add(run)
        session.flush()
        artifact = session.get(Artifact, artifact_id)
        assert artifact is not None
        artifact.metadata_json = dict(artifact.metadata_json, run_id=run.id)
        session.commit()
    before = (await client.get("/api/artifacts")).json()[0]
    assert before["reference_count"] == 2
    assert before["generation_identity"]["model_profile_name"] == "Garden model"
    preview = await _impact(client, f"/api/chats/{removed_id}/deletion-impact")
    trashed = await client.post(
        f"/api/chats/{removed_id}/trash", json=_command(preview, "trash-picker-chat")
    )
    assert trashed.status_code == 200, trashed.text
    after = (await client.get("/api/artifacts")).json()[0]
    assert after["id"] == artifact_id and after["reference_count"] == 1
    assert after["chat_ids"] == [live_id] and after["project_ids"] == []
    assert after["generation_identity"] is None
    for params in ({"chat_id": removed_id}, {"project_id": project_id}):
        assert (await client.get("/api/artifacts", params=params)).json() == []
    assert (await client.get(f"/api/artifacts/{artifact_id}/content")).status_code == 200
    deletion_id = trashed.json()["deletion_id"]
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    assert (
        await client.post(
            f"/api/recovery-items/{deletion_id}/restore",
            json=_command(preview, "restore-picker-chat"),
        )
    ).status_code == 200
    restored = (await client.get("/api/artifacts")).json()[0]
    assert restored == before
