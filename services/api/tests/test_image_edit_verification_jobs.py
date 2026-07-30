from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm.adapters.base import ChatEvent, ChatRequest, GeneratedAsset, MediaEvent, MediaRequest
from local_lm.adapters.mock import MockChatAdapter, MockMediaAdapter
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus
from local_lm.models import (
    Chat,
    Job,
    Message,
    ModelInstall,
    ModelProfile,
    Run,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)
from local_lm.schemas import EngineCapabilities


def _png(color: tuple[int, int, int]) -> bytes:
    content = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(content, format="PNG")
    return content.getvalue()


async def _wait_for_job(client: AsyncClient, kind: str) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        jobs = (await client.get("/api/jobs")).json()
        matching = [job for job in jobs if job["kind"] == kind]
        if matching and matching[0]["status"] in {
            JobStatus.COMPLETE.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.INTERRUPTED.value,
        }:
            return matching[0]
        await asyncio.sleep(0.03)
    raise AssertionError(f"{kind} job did not finish")


@pytest.mark.parametrize(
    ("assessment_raw", "expected_status", "expected_reason"),
    [
        (
            json.dumps(
                {
                    "requested_change_visible": False,
                    "unrelated_content_preserved": True,
                    "retry_recommended": True,
                    "direction": "increase",
                    "confidence": 0.94,
                }
            ),
            "complete",
            "eligible",
        ),
        ("not-json", "skipped", "invalid_assessment"),
    ],
)
async def test_image_edit_verification_is_dependent_at_most_once_and_non_destructive(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    assessment_raw: str,
    expected_status: str,
    expected_reason: str,
) -> None:
    original_capabilities = MockChatAdapter.capabilities
    captured: list[ChatRequest] = []
    source_complete_when_streamed: list[bool] = []
    result_png = _png((20, 180, 80))

    async def vision_capabilities(adapter: MockChatAdapter) -> EngineCapabilities:
        capabilities = await original_capabilities(adapter)
        return capabilities.model_copy(update={"input_modalities": ["text", "image"]})

    async def assessment_stream(
        _adapter: MockChatAdapter,
        request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        captured.append(request)
        with SessionLocal() as session:
            verification = session.get(Job, request.run_id)
            source_job_id = verification.payload_json["source_job_id"] if verification else None
            source_job = session.get(Job, source_job_id) if source_job_id else None
            source_step = (
                session.get(WorkStep, source_job.work_step_id)
                if source_job and source_job.work_step_id
                else None
            )
            source_complete_when_streamed.append(
                bool(
                    source_job
                    and source_job.status == JobStatus.COMPLETE.value
                    and source_step
                    and source_step.status == JobStatus.COMPLETE.value
                )
            )
        yield ChatEvent(type="delta", text=assessment_raw)
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    async def edited_media(
        _adapter: MockMediaAdapter,
        request: MediaRequest,
    ) -> AsyncIterator[MediaEvent]:
        yield MediaEvent(
            type="complete",
            progress=1,
            phase="complete",
            assets=[
                GeneratedAsset(
                    content=result_png,
                    media_type="image/png",
                    kind="image",
                    name="synthetic-edit.png",
                    metadata={"synthetic": True},
                )
            ],
        )

    monkeypatch.setattr(MockChatAdapter, "capabilities", vision_capabilities)
    monkeypatch.setattr(MockChatAdapter, "stream", assessment_stream)
    monkeypatch.setattr(MockMediaAdapter, "generate", edited_media)
    orchestrator = app.state.services.orchestrator
    monkeypatch.setattr(
        orchestrator,
        "_profile_has_verified_vision",
        lambda _session, _profile: True,
    )

    upload = await client.post(
        "/api/artifacts",
        files={"file": ("source.png", _png((180, 20, 20)), "image/png")},
    )
    assert upload.status_code == 201
    source_artifact_id = upload.json()["id"]
    chat = (
        await client.post(
            "/api/chats",
            json={
                "title": "Synthetic edit verification",
                "vision_settings_json": {"verify_image_edits": True},
            },
        )
    ).json()
    with SessionLocal() as session:
        install = ModelInstall(
            id="model_synthetic_vision",
            name="Synthetic vision",
            role="chat",
            engine="mock",
            local_path="synthetic",
            active=True,
        )
        profile = ModelProfile(
            id="profile_synthetic_vision",
            model_install_id=install.id,
            name="Synthetic vision",
            role="chat",
            engine="mock",
        )
        stored_chat = session.get(Chat, chat["id"])
        assert stored_chat
        stored_chat.active_vision_profile_id = profile.id
        session.add_all([install, profile])
        session.commit()

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Make the square green.",
            "mode": "image",
            "input_artifact_ids": [source_artifact_id],
        },
    )
    assert accepted.status_code == 202
    verification_job = await _wait_for_job(client, JobKind.EDIT_VERIFY.value)

    assert verification_job["run_id"] is None
    assert verification_job["status"] == JobStatus.COMPLETE.value
    assert verification_job["result_json"]["status"] == expected_status
    assert verification_job["result_json"]["reason"] == expected_reason
    assert verification_job["result_json"]["automatic_retry_executed"] is False
    assert source_complete_when_streamed == [True]
    assert len(captured) == 1
    content = captured[0].messages[-1]["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["text", "image_url", "image_url"]

    run_id = accepted.json()["run"]["id"]
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        message = session.get(Message, accepted.json()["assistant_message"]["id"])
        job = session.get(Job, verification_job["id"])
        assert run and message and job and run.work_step_id and job.work_step_id
        image_ids = [part.artifact_id for part in message.parts if part.type == "image"]
        revision_id = message.active_response_revision_id
        dependency = session.scalar(
            select(WorkStepDependency).where(
                WorkStepDependency.step_id == job.work_step_id,
                WorkStepDependency.depends_on_step_id == run.work_step_id,
            )
        )
        assert image_ids == [job.payload_json["result_artifact_id"]]
        assert revision_id is not None
        assert dependency is not None
        duplicate = orchestrator._queue_image_edit_verification(
            session,
            run,
            job.payload_json["source_job_id"],
            image_ids,
        )
        assert duplicate == job.id
        verification_jobs = session.scalars(
            select(Job).where(Job.kind == JobKind.EDIT_VERIFY.value)
        ).all()
        assert [
            candidate
            for candidate in verification_jobs
            if candidate.payload_json.get("source_run_id") == run.id
        ] == [job]


async def test_restart_requeues_waiting_verification_and_interrupts_active_one(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del client
    with SessionLocal() as session:
        chat = Chat(id="chat_verification_recovery", title="Recovery")
        plan = WorkPlan(
            id="plan_verification_recovery",
            chat_id=chat.id,
            transcript_sequence=1,
            summary_json={},
        )
        queued_step = WorkStep(
            id="step_verification_queued",
            plan=plan,
            ordinal=1,
            operation=JobKind.EDIT_VERIFY.value,
            status=JobStatus.QUEUED.value,
        )
        running_step = WorkStep(
            id="step_verification_running",
            plan=plan,
            ordinal=2,
            operation=JobKind.EDIT_VERIFY.value,
            status=JobStatus.RUNNING.value,
        )
        base_payload = {
            "version": "image-edit-verification-v1",
            "chat_id": chat.id,
            "source_run_id": "run_missing",
            "source_job_id": "job_missing",
            "source_artifact_id": "source_missing",
            "result_artifact_id": "result_missing",
            "vision_profile_id": "profile_missing",
            "attempt": 0,
        }
        queued = Job(
            id="job_verification_queued",
            kind=JobKind.EDIT_VERIFY.value,
            status=JobStatus.QUEUED.value,
            work_plan_id=plan.id,
            work_step_id=queued_step.id,
            queue_group="primary",
            payload_json=base_payload,
        )
        running = Job(
            id="job_verification_running",
            kind=JobKind.EDIT_VERIFY.value,
            status=JobStatus.RUNNING.value,
            work_plan_id=plan.id,
            work_step_id=running_step.id,
            queue_group="primary",
            payload_json=base_payload,
        )
        session.add_all([chat, plan])
        session.flush()
        session.add_all([queued_step, running_step, queued, running])
        session.commit()
        queued_id = queued.id
        running_id = running.id
        running_step_id = running_step.id

    restarted: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app.state.services.orchestrator,
        "start",
        lambda job_id, run_id: restarted.append((job_id, run_id)),
    )
    app.state.services.orchestrator.recover_interrupted()

    assert restarted == [(queued_id, None)]
    with SessionLocal() as session:
        queued_job = session.get(Job, queued_id)
        active_job = session.get(Job, running_id)
        active_step = session.get(WorkStep, running_step_id)
        assert queued_job and active_job and active_step
        assert queued_job.status == JobStatus.QUEUED.value
        assert active_job.status == JobStatus.INTERRUPTED.value
        assert active_step.status == JobStatus.INTERRUPTED.value
        assert active_job.result_json["reason"] == "assessment_interrupted"


async def test_cancelled_verification_retains_output_and_cannot_be_retried(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "version": "image-edit-verification-v1",
        "chat_id": "chat_verification_cancel",
        "source_run_id": "run_source",
        "source_job_id": "job_source",
        "source_artifact_id": "artifact_source",
        "result_artifact_id": "artifact_result",
        "vision_profile_id": "profile_vision",
        "attempt": 0,
    }
    with SessionLocal() as session:
        chat = Chat(id=payload["chat_id"], title="Cancel verification")
        plan = WorkPlan(
            id="plan_verification_cancel",
            chat_id=chat.id,
            transcript_sequence=1,
            summary_json={},
        )
        step = WorkStep(
            id="step_verification_cancel",
            plan=plan,
            ordinal=1,
            operation=JobKind.EDIT_VERIFY.value,
            status=JobStatus.QUEUED.value,
        )
        job = Job(
            id="job_verification_cancel",
            kind=JobKind.EDIT_VERIFY.value,
            status=JobStatus.QUEUED.value,
            work_plan_id=plan.id,
            work_step_id=step.id,
            queue_group="primary",
            payload_json=payload,
        )
        session.add_all([chat, plan])
        session.flush()
        session.add_all([step, job])
        session.commit()
        job_id = job.id
        step_id = step.id

    cancelled_engine_jobs: list[str] = []

    async def cancel_engine_job(candidate_job_id: str) -> None:
        cancelled_engine_jobs.append(candidate_job_id)

    monkeypatch.setattr(app.state.services.engines.chat, "cancel", cancel_engine_job)

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == JobStatus.CANCELLED.value
    assert cancelled_engine_jobs == [job_id]
    retry = await client.post(f"/api/jobs/{job_id}/retry")
    assert retry.status_code == 422
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        step = session.get(WorkStep, step_id)
        assert job and step
        assert job.status == JobStatus.CANCELLED.value
        assert step.status == JobStatus.CANCELLED.value
        assert job.result_json["reason"] == "cancelled"
        assert job.result_json["automatic_retry_executed"] is False
