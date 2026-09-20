from __future__ import annotations

import io
import json
import zipfile
from datetime import timedelta
from typing import Any

from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import func, select

from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import (
    Chat,
    ChatActivityEvent,
    Job,
    Message,
    MessagePart,
    ResponseRevision,
    Run,
)
from local_lm.scheduler import JobClaim
from local_lm.schemas import ResponseRevisionOut


def seed_running() -> tuple[str, str, str, str]:
    with SessionLocal() as session:
        chat = Chat(title="Color study")
        session.add(chat)
        session.flush()
        user = Message(chat_id=chat.id, role="user")
        message = Message(
            chat_id=chat.id,
            role="assistant",
            status="pending",
            parts=[MessagePart(position=0, type="text", text="Neutral partial response")],
        )
        session.add_all([user, message])
        session.flush()
        run = Run(
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=message.id,
            status="running",
            operation="text",
        )
        session.add(run)
        session.flush()
        job = Job(
            run_id=run.id,
            kind="chat",
            status="running",
            attempt=2,
            claim_owner="current-attempt",
            claim_expires_at=utcnow() + timedelta(minutes=5),
        )
        session.add(job)
        session.commit()
        return chat.id, message.id, run.id, job.id


async def test_completed_output_binds_summary_to_the_rendered_revision(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Color study"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Explain the color blue.", "mode": "text"}
    )
    assert response.status_code == 202
    accepted = response.json()

    async def read_run() -> dict[str, Any]:
        value: dict[str, Any] = (await client.get(f"/api/runs/{accepted['run']['id']}")).json()
        return value

    await wait_for_terminal_status(read_run, what="chat activity output", expected="complete")
    message = (await client.get(f"/api/messages/{accepted['assistant_message']['id']}")).json()
    revision = next(
        item
        for item in message["response_revisions"]
        if item["id"] == message["active_response_revision_id"]
    )
    summary = (await client.get("/api/chats/summaries")).json()[0]
    activity = summary["activity"]
    assert activity["active_work_count"] == activity["unresolved_failed_count"] == 0
    assert activity["last_failure"] is None
    assert activity["last_output"] == revision["activity"]
    assert activity["last_output"]["message_id"] == message["id"]
    assert activity["last_output"]["occurred_at"].endswith("+00:00")


async def test_stale_claim_cannot_record_failure_activity(
    client: AsyncClient, app: FastAPI
) -> None:
    chat_id, message_id, run_id, job_id = seed_running()
    orchestrator = app.state.services.orchestrator
    await orchestrator._fail(
        job_id, run_id, "Neutral failed attempt", claim=JobClaim(token="expired-attempt", attempt=1)
    )
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(ChatActivityEvent)) == 0
        assert session.get(Run, run_id).status == "running"
    await orchestrator._fail(
        job_id, run_id, "Neutral failed attempt", claim=JobClaim(token="current-attempt", attempt=2)
    )
    summary = (await client.get("/api/chats/summaries")).json()[0]
    assert summary["id"] == chat_id
    assert summary["activity"]["active_work_count"] == 0
    assert summary["activity"]["unresolved_failed_count"] == 1
    failure = summary["activity"]["last_failure"]
    assert failure["message_id"] == message_id
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(ChatActivityEvent)) == 1


async def test_restart_recovery_records_one_failure_without_a_live_claim(
    client: AsyncClient, app: FastAPI
) -> None:
    _, message_id, run_id, job_id = seed_running()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        job.claim_owner = None
        job.claim_expires_at = None
        session.commit()
    app.state.services.orchestrator.recover_interrupted()
    app.state.services.orchestrator.recover_interrupted()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        run = session.get(Run, run_id)
        assert job is not None and run is not None
        assert job.status == "interrupted" and run.status == "failed"
        assert session.scalar(select(func.count()).select_from(ChatActivityEvent)) == 1
    message = (await client.get(f"/api/messages/{message_id}")).json()
    revision = message["response_revisions"][0]
    assert revision["status"] == "failed" and revision["activity"] is not None
    assert [part["type"] for part in message["parts"]] == ["text", "error"]


async def test_an_older_response_snapshot_does_not_acquire_a_new_activity_identity(
    client: AsyncClient, app: FastAPI
) -> None:
    _, message_id, run_id, job_id = seed_running()
    with SessionLocal() as session:
        revision = ResponseRevision(
            message_id=message_id, run_id=run_id, sequence=1, status="pending"
        )
        session.add(revision)
        session.flush()
        message = session.get(Message, message_id)
        assert message is not None
        message.active_response_revision_id = revision.id
        session.commit()
        revision_id = revision.id
    with SessionLocal() as older:
        snapshot = older.get(ResponseRevision, revision_id)
        assert snapshot is not None and snapshot.activity is None
        await app.state.services.orchestrator._fail(
            job_id,
            run_id,
            "Neutral failed attempt",
            claim=JobClaim(token="current-attempt", attempt=2),
        )
        stale = ResponseRevisionOut.model_validate(snapshot)
        assert stale.status == "pending" and stale.activity is None
    current = (await client.get(f"/api/messages/{message_id}")).json()
    assert current["response_revisions"][0]["activity"] is not None


async def test_export_and_import_do_not_carry_local_activity_identities(
    client: AsyncClient, app: FastAPI
) -> None:
    project = (await client.post("/api/projects", json={"name": "Color studies"})).json()
    chat_id, _, run_id, job_id = seed_running()
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.project_id = project["id"]
        session.commit()
    await app.state.services.orchestrator._fail(
        job_id, run_id, "Neutral failed attempt", claim=JobClaim(token="current-attempt", attempt=2)
    )
    exported = await client.post(
        f"/api/projects/{project['id']}/export", params={"include_media": False}
    )
    assert exported.status_code == 201
    archive = await client.get(exported.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    revisions = [
        revision
        for chat in manifest["chats"]
        for message in chat["messages"]
        for revision in message.get("response_revisions", [])
    ]
    assert revisions and all("activity" not in revision for revision in revisions)
    imported = await client.post(
        "/api/projects/import",
        files={"archive": ("color-studies.zip", archive.content, "application/zip")},
    )
    assert imported.status_code == 201, imported.text
    summaries = (
        await client.get("/api/chats/summaries", params={"project_id": imported.json()["id"]})
    ).json()
    assert summaries and all(
        row["activity"]["last_output"] is None and row["activity"]["last_failure"] is None
        for row in summaries
    )
