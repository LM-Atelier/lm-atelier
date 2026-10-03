"""Unpreviewed chat deletion cannot discard history or generated media."""

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_deletion import _image_exchange, _text_exchange
from test_chat_recovery import _chat, _history
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm.db import SessionLocal
from local_lm.models import Artifact, Job, RecoveryItem, RecoveryOperation


@pytest.mark.parametrize("delete_generated_media", [False, True])
async def test_unpreviewed_chat_delete_preserves_the_completed_history_and_image(
    client: AsyncClient, app: FastAPI, delete_generated_media: bool
) -> None:
    chat_id = await _chat(client)
    await _image_exchange(client, chat_id, "Create a picture of garden beds")
    before = _history(chat_id)
    artifact_id = next(part[-1] for part in before["parts"] if part[-1])
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact is not None
        path = app.state.services.artifacts.resolve(artifact)
        content = path.read_bytes()

    refused = await client.delete(
        f"/api/chats/{chat_id}", params={"delete_generated_media": delete_generated_media}
    )

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "recovery-preview-required"
    assert _history(chat_id) == before
    assert path.read_bytes() == content
    assert (await client.get(f"/api/chats/{chat_id}")).status_code == 200
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 200
    with SessionLocal() as session:
        assert session.scalar(select(RecoveryItem)) is None
        assert session.scalar(select(RecoveryOperation)) is None
    assert (await client.delete("/api/chats/nonexistent-garden")).status_code == 404


@pytest.mark.parametrize("status", ["queued", "running", "paused"])
@pytest.mark.parametrize("runless", [False, True])
async def test_unpreviewed_chat_delete_never_cancels_accepted_work(
    client: AsyncClient, status: str, runless: bool
) -> None:
    chat_id = await _chat(client)
    exchange = await _text_exchange(client, chat_id, "Keep these garden notes")
    with SessionLocal() as session:
        job = Job(
            kind="edit_verify" if runless else "chat",
            status=status,
            phase=status,
            run_id=None if runless else exchange["run"]["id"],
            payload_json={"chat_id": chat_id, "source_run_id": exchange["run"]["id"]},
        )
        session.add(job)
        session.commit()
        job_id = job.id
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        before_job = tuple(getattr(job, column.name) for column in Job.__table__.columns)
    before = _history(chat_id)

    refused = await client.delete(f"/api/chats/{chat_id}")

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "recovery-preview-required"
    assert _history(chat_id) == before
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert tuple(getattr(job, column.name) for column in Job.__table__.columns) == before_job
        assert session.scalar(select(RecoveryItem)) is None
        assert session.scalar(select(RecoveryOperation)) is None
