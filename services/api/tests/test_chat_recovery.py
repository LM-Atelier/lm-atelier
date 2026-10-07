"""Trashed conversations keep their history until permanent deletion."""

from __future__ import annotations

from typing import Any, cast

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_deletion import _text_exchange
from test_deletion_reliability import _seed_shared_generated_image

from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus
from local_lm.models import Artifact, Chat, Job, Message, MessagePart, Run


async def _chat(client: AsyncClient) -> str:
    created = await client.post("/api/chats", json={"title": "Garden notes"})
    assert created.status_code == 201
    chat_id = str(created.json()["id"])
    pinned = await client.patch(f"/api/chats/{chat_id}", json={"pinned": True})
    assert pinned.status_code == 200
    return chat_id


def _history(chat_id: str) -> dict[str, Any]:
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        messages = list(
            session.scalars(select(Message).where(Message.chat_id == chat_id).order_by(Message.id))
        )
        runs = list(session.scalars(select(Run).where(Run.chat_id == chat_id).order_by(Run.id)))
        return {
            "chat": (
                chat.id,
                chat.title,
                chat.project_id,
                chat.archived,
                chat.pinned,
                chat.active_head_message_id,
                chat.generation_settings_json,
                chat.generation_preset_ids_json,
            ),
            "messages": [
                (message.id, message.parent_id, message.role, message.status)
                for message in messages
            ],
            "parts": [
                (part.id, part.message_id, part.position, part.type, part.text, part.artifact_id)
                for part in session.scalars(
                    select(MessagePart)
                    .where(MessagePart.message_id.in_([message.id for message in messages]))
                    .order_by(MessagePart.id)
                )
            ],
            "runs": [(run.id, run.user_message_id, run.assistant_message_id) for run in runs],
            "jobs": [
                (job.id, job.status, job.attempt)
                for job in session.scalars(
                    select(Job).where(Job.run_id.in_([run.id for run in runs])).order_by(Job.id)
                )
            ],
        }


async def _impact(client: AsyncClient, path: str) -> dict[str, Any]:
    response = await client.get(path)
    assert response.status_code == 200, response.text
    return cast(dict[str, Any], response.json())


def _command(impact: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "expected_revision": impact["revision"],
        "impact_sha256": impact["impact_sha256"],
        "operation_key": key,
    }


async def test_trash_and_restore_keep_the_same_conversation_and_history(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client)
    await _text_exchange(client, chat_id, "Record the spacing between the garden beds")
    for suffix in ("", "/messages", "/context", "/composer-draft"):
        assert (await client.get(f"/api/chats/{chat_id}{suffix}")).status_code == 200
    before = _history(chat_id)
    impact = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    command = _command(impact, "trash-garden-notes")

    trashed = await client.post(f"/api/chats/{chat_id}/trash", json=command)

    assert trashed.status_code == 200, trashed.text
    item = trashed.json()
    assert item["subject_id"] == chat_id
    assert item["kind"] == "chat"
    assert item["state"] == "recoverable"
    assert _history(chat_id) == before
    repeated = await client.post(f"/api/chats/{chat_id}/trash", json=command)
    assert repeated.status_code == 200
    assert repeated.json() == item
    assert all(row["id"] != chat_id for row in (await client.get("/api/chats")).json())
    assert all(row["id"] != chat_id for row in (await client.get("/api/chats/summaries")).json())
    assert (await client.get(f"/api/chats/{chat_id}")).status_code == 404
    assert (await client.get(f"/api/chats/{chat_id}/messages")).status_code == 404
    assert (await client.get(f"/api/chats/{chat_id}/context")).status_code == 404
    assert (await client.get(f"/api/chats/{chat_id}/composer-draft")).status_code == 404
    refused = await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": "Add another bed", "mode": "text"}
    )
    assert refused.status_code == 404
    assert _history(chat_id) == before

    deletion_id = item["deletion_id"]
    restore_impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    restore = await client.post(
        f"/api/recovery-items/{deletion_id}/restore",
        json=_command(restore_impact, "restore-garden-notes"),
    )

    assert restore.status_code == 200, restore.text
    assert (await client.get(f"/api/chats/{chat_id}")).json()["id"] == chat_id
    assert _history(chat_id) == before


async def test_a_changed_conversation_refuses_its_old_deletion_preview(client: AsyncClient) -> None:
    chat_id = await _chat(client)
    impact = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    await _text_exchange(client, chat_id, "Keep this newly accepted garden note")
    before = _history(chat_id)

    refused = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(impact, "stale-garden-preview")
    )

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "recovery-impact-stale"
    assert _history(chat_id) == before
    assert (await client.get(f"/api/chats/{chat_id}")).status_code == 200


async def test_deleted_history_cannot_be_read_or_restarted_by_its_saved_work_ids(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client)
    exchange = await _text_exchange(client, chat_id, "Keep the garden layout unchanged")
    run_id = exchange["run"]["id"]
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None and run.work_plan_id and run.work_step_id
        plan_id, step_id = run.work_plan_id, run.work_step_id
        job_id = session.scalar(select(Job.id).where(Job.run_id == run_id))
        assert job_id is not None
    paths = [
        f"/api/runs/{run_id}",
        f"/api/work-plans/{plan_id}",
        f"/api/work-steps/{step_id}",
        f"/api/messages/{exchange['user_message']['id']}",
        f"/api/messages/{exchange['assistant_message']['id']}",
    ]
    for path in paths:
        assert (await client.get(path)).status_code == 200
    assert any(row["id"] == job_id for row in (await client.get("/api/jobs")).json())
    assert any(row["id"] == plan_id for row in (await client.get("/api/work-plans")).json())
    impact = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    before = _history(chat_id)
    response = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(impact, "trash-saved-work")
    )
    assert response.status_code == 200, response.text
    for path in paths:
        assert (await client.get(path)).status_code == 404
    assert all(row["id"] != job_id for row in (await client.get("/api/jobs")).json())
    assert all(row["id"] != plan_id for row in (await client.get("/api/work-plans")).json())
    for entity, identity in (("jobs", job_id), ("work-plans", plan_id), ("work-steps", step_id)):
        for action in ("retry", "cancel"):
            refused = await client.post(f"/api/{entity}/{identity}/{action}")
            assert refused.status_code == 404, refused.text
    assert _history(chat_id) == before


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED])
async def test_trash_refuses_active_work_without_cancelling_it(
    client: AsyncClient, status: JobStatus
) -> None:
    chat_id = await _chat(client)
    exchange = await _text_exchange(client, chat_id, "Keep this garden note")
    with SessionLocal() as session:
        job = Job(kind="chat", status=status.value, run_id=exchange["run"]["id"], payload_json={})
        session.add(job)
        session.commit()
        job_id = job.id
    before = _history(chat_id)
    impact = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")

    refused = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(impact, f"busy-garden-{status.value}")
    )

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "chat-recovery-active-work"
    assert _history(chat_id) == before
    with SessionLocal() as session:
        retained = session.get(Job, job_id)
        assert retained is not None and retained.status == status.value


async def test_a_runless_verification_blocks_trash_without_losing_its_source(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client)
    exchange = await _text_exchange(client, chat_id, "Keep the source of this garden check")
    with SessionLocal() as session:
        job = Job(
            kind=JobKind.EDIT_VERIFY.value,
            status=JobStatus.QUEUED.value,
            run_id=None,
            payload_json={"chat_id": chat_id, "source_run_id": exchange["run"]["id"]},
        )
        session.add(job)
        session.commit()
        job_id = job.id
    before = _history(chat_id)
    impact = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")

    refused = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(impact, "busy-source-verification")
    )

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "chat-recovery-active-work"
    assert _history(chat_id) == before
    with SessionLocal() as session:
        retained = session.get(Job, job_id)
        assert retained is not None and retained.status == JobStatus.QUEUED.value


async def test_media_purge_intent_keeps_another_conversations_picture(
    client: AsyncClient,
) -> None:
    removed_id = await _chat(client)
    kept_id = await _chat(client)
    artifact_id = _seed_shared_generated_image([removed_id, kept_id])
    retained_history = _history(kept_id)
    impact = await _impact(
        client, f"/api/chats/{removed_id}/deletion-impact?delete_generated_media=true"
    )
    trashed = await client.post(
        f"/api/chats/{removed_id}/trash",
        json={**_command(impact, "trash-shared-picture"), "delete_generated_media": True},
    )

    assert trashed.status_code == 200, trashed.text
    with SessionLocal() as session:
        assert session.get(Artifact, artifact_id) is not None
        assert session.get(Chat, removed_id) is not None

    deletion_id = trashed.json()["deletion_id"]
    purge_impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    purged = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={
            **_command(purge_impact, "purge-shared-picture"),
            "acknowledgement": "permanently-delete",
        },
    )

    assert purged.status_code == 200, purged.text
    assert purged.json()["reclaimed_bytes"] == 0
    assert _history(kept_id) == retained_history
    with SessionLocal() as session:
        assert session.get(Chat, removed_id) is None
        assert session.get(Artifact, artifact_id) is not None
        assert (
            session.scalar(
                select(MessagePart.id)
                .join(Message, Message.id == MessagePart.message_id)
                .where(Message.chat_id == kept_id, MessagePart.artifact_id == artifact_id)
            )
            is not None
        )
