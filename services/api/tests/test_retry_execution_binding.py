"""The durable retry binding must survive a record that says it did not run.

The verification record REPLACES the source's, so a truthful not-started
outcome that omits the retry's identity erases the binding it exists to
preserve. Each case here refuses the retry twice, requires the binding
still findable, and then converges healthily on the SAME retry - which is
the property the record is for, and the one a shorter control would miss.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from typing import Any, cast

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
    ModelInstall,
    ModelProfile,
    Run,
    WorkStep,
)
from local_lm.schemas import EngineCapabilities, JobOut


def _png(color: tuple[int, int, int]) -> bytes:
    content = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(content, format="PNG")
    return content.getvalue()


async def _wait_for_job(client: AsyncClient, kind: str) -> dict:  # type: ignore[type-arg]
    del client
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        with SessionLocal() as session:
            matching = session.scalar(
                select(Job).where(Job.kind == kind).order_by(Job.created_at.desc())
            )
            if matching and matching.status in {
                JobStatus.COMPLETE.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
                JobStatus.INTERRUPTED.value,
            }:
                return JobOut.model_validate(matching).model_dump(mode="json")
        await asyncio.sleep(0.03)
    raise AssertionError(f"{kind} job did not finish")


_RETRY_ASSESSMENT = json.dumps(
    {
        "requested_change_visible": False,
        "unrelated_content_preserved": True,
        "retry_recommended": True,
        "direction": "increase",
        "confidence": 0.94,
    }
)


async def _wait_for_run(client: AsyncClient, run_id: str) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        run = (await client.get(f"/api/runs/{run_id}")).json()
        if run["status"] in {
            JobStatus.COMPLETE.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.INTERRUPTED.value,
        }:
            assert run["status"] == JobStatus.COMPLETE.value
            return cast(dict[str, Any], run)
        await asyncio.sleep(0.03)
    raise AssertionError(f"run {run_id} did not finish")


@pytest.mark.parametrize("refusal_point", ["announcement", "start"])
async def test_retry_binding_survives_failed_execution_record(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, refusal_point: str
) -> None:
    """A truthful not-started record must leave the same durable retry recoverable."""
    assessment_raw = _RETRY_ASSESSMENT
    turn_settings: dict[str, object] = {}
    announcement_fails = True
    failures = {"count": 0}
    original_capabilities = MockChatAdapter.capabilities
    refuse_announcements = {"armed": False}
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
        # From here on the source turn is finished and anything announcing a
        # plan is the retry. Arming the refusal earlier would refuse the
        # source's own announcement and no verification would run at all.
        refuse_announcements["armed"] = announcement_fails
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
    if announcement_fails:
        # Every announcement of the RETRY's plan fails, so its start is never
        # reached - not on creation and not on the convergence that recovers
        # from creation. Armed by the assessment stream, so the source turn's
        # own plan announcement still succeeds and there is a verification to
        # run. Only the plan event is refused; nothing else is disturbed.
        real_publish = orchestrator.events.publish

        async def refuse_plan_announcements(name: str, *args: object, **kwargs: object) -> None:
            if (
                name == "work_plan.created"
                and refuse_announcements["armed"]
                and refusal_point == "announcement"
            ):
                failures["count"] += 1
                raise RuntimeError("the event broker was unavailable")
            await real_publish(name, *args, **kwargs)

        monkeypatch.setattr(orchestrator.events, "publish", refuse_plan_announcements)

    real_start = type(orchestrator).start

    def refuse_retry_start(self: Any, job_id: str, run_id: str) -> None:
        if refuse_announcements["armed"] and refusal_point == "start":
            failures["count"] += 1
            raise RuntimeError("constructed retry start was unavailable")
        real_start(self, job_id, run_id)

    monkeypatch.setattr(type(orchestrator), "start", refuse_retry_start)

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
            "settings": turn_settings,
        },
    )
    assert accepted.status_code == 202
    verification_job = await _wait_for_job(client, JobKind.EDIT_VERIFY.value)

    assert verification_job["status"] == JobStatus.COMPLETE.value
    assert verification_job["result_json"]["status"] == "complete"
    assert failures["count"] >= 2, "the initial and recovery failures were not exercised"
    source_id = accepted.json()["run"]["id"]
    with SessionLocal() as session:
        source = session.get(Run, source_id)
        assert source is not None
        bound = orchestrator._bound_retry(session, source)
        assert bound is not None, "the final verification record erased the durable retry binding"
        retained_id = bound.run.id
        job = session.get(Job, verification_job["id"])
        assert job is not None
        from local_lm.image_edit_verification import ImageEditVerificationJobPayload

        payload = ImageEditVerificationJobPayload.model_validate(job.payload_json)

    refuse_announcements["armed"] = False
    starts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        type(orchestrator), "start", lambda self, job_id, run_id: starts.append((job_id, run_id))
    )
    recovered = await orchestrator._converge_on_bound_retry(payload)
    retry = recovered[0] if isinstance(recovered, tuple) else recovered
    assert retry is not None and retry.run.id == retained_id
    assert starts, "the same retained retry could not be started on a later healthy convergence"


async def test_retry_binding_survives_an_unavailable_verification(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read that fails after the retry is durable must not erase its identity.

    The convergence that recovers a bound retry is best-effort at every step
    it takes, but the read that FINDS the retry happens before any of that
    care. When it fails transiently the exception leaves the whole
    verification, which records the run unavailable - and that record replaces
    the source's, taking the retry's identity with it. Nothing can converge on
    the retry afterwards, because nothing can name it any more.
    """
    assessment_raw = _RETRY_ASSESSMENT
    announcement_failures = {"count": 0}
    reconstruction_failures = {"count": 0}
    refuse = {"announcements": False, "reconstruction": False}
    bound_before_failure: list[str] = []
    original_capabilities = MockChatAdapter.capabilities
    result_png = _png((20, 180, 80))

    async def vision_capabilities(adapter: MockChatAdapter) -> EngineCapabilities:
        capabilities = await original_capabilities(adapter)
        return capabilities.model_copy(update={"input_modalities": ["text", "image"]})

    async def assessment_stream(
        _adapter: MockChatAdapter,
        request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        del request
        # The source turn has announced its own plan by now, so refusing from
        # here refuses only the retry's announcement and a verification still
        # runs. Arming earlier would leave nothing to verify.
        refuse["announcements"] = True
        yield ChatEvent(type="delta", text=assessment_raw)
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    async def edited_media(
        _adapter: MockMediaAdapter,
        request: MediaRequest,
    ) -> AsyncIterator[MediaEvent]:
        del request
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

    real_publish = orchestrator.events.publish

    async def refuse_plan_announcements(name: str, *args: object, **kwargs: object) -> None:
        if name == "work_plan.created" and refuse["announcements"]:
            announcement_failures["count"] += 1
            # The retry is durable before this point, so refusing its
            # announcement sends the execution into the recovery that reads
            # the binding back - which is the read this case fails.
            refuse["reconstruction"] = True
            raise RuntimeError("the event broker was unavailable")
        await real_publish(name, *args, **kwargs)

    monkeypatch.setattr(orchestrator.events, "publish", refuse_plan_announcements)

    real_bound_retry = type(orchestrator)._bound_retry

    def transient_source_read(self: Any, session: Any, source_run: Any) -> Any:
        if refuse["reconstruction"]:
            reconstruction_failures["count"] += 1
            refuse["reconstruction"] = False
            raise RuntimeError("the source run could not be read")
        return real_bound_retry(self, session, source_run)

    monkeypatch.setattr(type(orchestrator), "_bound_retry", transient_source_read)

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
            "settings": {},
        },
    )
    assert accepted.status_code == 202
    verification_job = await _wait_for_job(client, JobKind.EDIT_VERIFY.value)

    assert announcement_failures["count"] >= 1, "the retry announcement was never refused"
    assert reconstruction_failures["count"] == 1, "the bound retry was never read back"
    assert verification_job["result_json"]["status"] == "skipped"
    assert verification_job["result_json"]["reason"] == "assessment_unavailable"
    assert verification_job["result_json"]["automatic_retry_executed"] is False

    refuse["announcements"] = False
    source_id = accepted.json()["run"]["id"]
    with SessionLocal() as session:
        source = session.get(Run, source_id)
        assert source is not None
        record = source.provenance_json["image_edit_verification"]
        # The unavailable outcome is recorded truthfully AND the identity of
        # the retry that already exists is still there to be found.
        assert record["reason"] == "assessment_unavailable"
        bound = real_bound_retry(orchestrator, session, source)
        assert bound is not None, "the unavailable record erased the durable retry binding"
        bound_before_failure.append(bound.run.id)
        job = session.get(Job, verification_job["id"])
        assert job is not None
        from local_lm.image_edit_verification import ImageEditVerificationJobPayload

        payload = ImageEditVerificationJobPayload.model_validate(job.payload_json)

    starts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        type(orchestrator), "start", lambda self, job_id, run_id: starts.append((job_id, run_id))
    )
    recovered = await orchestrator._converge_on_bound_retry(payload)
    retry = recovered[0] if isinstance(recovered, tuple) else recovered
    assert retry is not None and retry.run.id == bound_before_failure[0]
    assert starts, "the same retry could not be started once the read recovered"
