from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from local_lm.domain import JobKind, JobStatus, Operation, PartType, RoutingMode, RunStatus
from local_lm.image_edit_verification import (
    ImageEditRetryDecision,
    ImageEditVerificationJobPayload,
    VerificationReason,
    image_edit_verification_job_id,
)
from local_lm.models import (
    Artifact,
    Chat,
    Job,
    Message,
    MessagePart,
    ModelInstall,
    ModelProfile,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.scheduler import JobClaim
from local_lm.schemas import WorkerStatus
from local_lm.vision import VisionInputError


def _orchestrator(*, session_factory=None) -> ConversationOrchestrator:  # type: ignore[no-untyped-def]
    session_factory = session_factory or Mock()
    return ConversationOrchestrator(
        engines=SimpleNamespace(
            settings=SimpleNamespace(),
            chat=SimpleNamespace(cancel=AsyncMock()),
            media=SimpleNamespace(cancel=AsyncMock()),
        ),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(publish_job=AsyncMock()),
        processes=SimpleNamespace(
            statuses=Mock(
                return_value=[
                    WorkerStatus(
                        name="chat",
                        state="stopped",
                        managed=False,
                        running=False,
                    )
                ]
            )
        ),
        session_factory=session_factory,
    )


_TEST_CLAIM = JobClaim(token="attempt-token-a", attempt=1)


def test_successful_edit_queues_one_detached_low_priority_verifier(monkeypatch) -> None:
    source = SimpleNamespace(id="artifact-source", media_type="image/png")
    result = SimpleNamespace(id="artifact-result", media_type="image/png")
    chat = SimpleNamespace(
        id="chat-verify",
        vision_settings_json={"verify_image_edits": True},
    )
    profile = SimpleNamespace(id="profile-vision")
    plan = SimpleNamespace(id="plan-edit", source_action="send", summary_json={})
    run = SimpleNamespace(
        id="run-edit",
        chat_id=chat.id,
        operation=Operation.IMAGE_TO_IMAGE.value,
        work_plan_id=plan.id,
        work_step_id="step-source",
        provenance_json={
            "image_edit": {
                "strength": {
                    "mode": "auto",
                    "parameter": "denoise",
                    "value": 0.66,
                    "applied_bounds": {"minimum": 0.3, "maximum": 0.8},
                }
            }
        },
    )
    stored: dict[tuple[object, str], object] = {
        (Chat, chat.id): chat,
        (Artifact, source.id): source,
        (Artifact, result.id): result,
        (WorkPlan, plan.id): plan,
    }
    added: list[object] = []

    class FakeSession:
        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return stored.get((model, identity))

        def add(self, value: object) -> None:
            added.append(value)
            if isinstance(value, (Job, WorkStep)):
                stored[(type(value), value.id)] = value

        def flush(self) -> None:
            return None

        def scalar(self, _statement):  # type: ignore[no-untyped-def]
            return 1

    orchestrator = _orchestrator()
    orchestrator.input_artifact_ids_for_run = Mock(return_value=[source.id])  # type: ignore[method-assign]
    orchestrator._vision_profile_for_chat = Mock(return_value=profile)  # type: ignore[method-assign]
    monkeypatch.setattr("local_lm.orchestrator.refresh_plan_status", Mock())
    monkeypatch.setattr(
        "local_lm.orchestrator.plan_status_summary",
        Mock(return_value={"complete": 1, "queued": 1}),
    )

    job_id = orchestrator._queue_image_edit_verification(
        FakeSession(),  # type: ignore[arg-type]
        run,  # type: ignore[arg-type]
        "job-source",
        [result.id],
    )
    duplicate_id = orchestrator._queue_image_edit_verification(
        FakeSession(),  # type: ignore[arg-type]
        run,  # type: ignore[arg-type]
        "job-source",
        [result.id],
    )

    assert job_id == duplicate_id == image_edit_verification_job_id(run.id)
    jobs = [value for value in added if isinstance(value, Job)]
    steps = [value for value in added if isinstance(value, WorkStep)]
    dependencies = [value for value in added if isinstance(value, WorkStepDependency)]
    assert len(jobs) == 1
    assert steps == []
    assert dependencies == []
    job = jobs[0]
    assert (
        job.kind,
        job.run_id,
        job.work_plan_id,
        job.work_step_id,
        job.queue_resource,
        job.queue_group,
        job.queue_priority,
    ) == (
        JobKind.EDIT_VERIFY.value,
        None,
        None,
        None,
        "interactive_compute",
        "primary",
        -10,
    )
    payload = ImageEditVerificationJobPayload.model_validate(job.payload_json)
    assert (payload.source_artifact_id, payload.result_artifact_id) == (
        source.id,
        result.id,
    )
    assert (
        payload.strength_parameter,
        payload.current_strength,
        payload.minimum,
        payload.maximum,
    ) == ("denoise", 0.66, 0.3, 0.8)
    assert payload.automatic_strength is True


async def test_verifier_starts_after_media_handoff_and_inside_primary_lease(
    monkeypatch,
) -> None:
    order: list[str] = []
    job = SimpleNamespace(
        id="job-source",
        kind=JobKind.IMAGE.value,
        status=JobStatus.QUEUED.value,
        queue_resource="media_compute",
        queue_group="primary",
        queue_priority=0,
        started_at=None,
    )
    run = SimpleNamespace(
        id="run-source",
        operation=Operation.IMAGE_TO_IMAGE.value,
        status=RunStatus.QUEUED.value,
        started_at=None,
        chat_id="chat-source",
        work_plan_id="plan-source",
        work_step_id="step-source",
        standalone_prompt="Make the top red",
    )

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (Job, job.id): job,
                (Run, run.id): run,
            }.get((model, identity))

        def commit(self) -> None:
            return None

        def in_transaction(self) -> bool:
            return True

        def execute(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(rowcount=1)

    @asynccontextmanager
    async def lease(*_args, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {"resource": "media_compute", "group": "primary", "priority": 0}
        order.append("lease entered")
        yield
        order.append("lease released")

    orchestrator = _orchestrator(session_factory=FakeSession)
    orchestrator.scheduler.job_lease = lease
    orchestrator._resolve_step_inputs = Mock()  # type: ignore[method-assign]
    orchestrator._set_work_status = Mock()  # type: ignore[method-assign]
    orchestrator._arm_step_prewarm = Mock()  # type: ignore[method-assign]
    orchestrator._prepare_device_handoff = AsyncMock(return_value="profile-chat")  # type: ignore[method-assign]

    async def execute_media(_job_id: str, _run_id: str, _claim: object) -> str:
        order.append("media completed")
        return "job-verifier"

    async def complete_handoff(_profile_id: str) -> None:
        order.append("chat restored")

    orchestrator._execute_media = execute_media  # type: ignore[method-assign]
    orchestrator._complete_media_handoff = complete_handoff  # type: ignore[method-assign]
    orchestrator.start = Mock(side_effect=lambda *_args: order.append("verifier started"))  # type: ignore[method-assign]
    orchestrator._finalize_setup_verification_run = AsyncMock()  # type: ignore[method-assign]
    monkeypatch.setattr("local_lm.orchestrator.mark_setup_verification_running", Mock())

    await orchestrator._execute(job.id, run.id)

    assert order == [
        "lease entered",
        "media completed",
        "chat restored",
        "verifier started",
        "lease released",
    ]


async def test_foreground_dispatch_preempts_a_running_image_edit_check() -> None:
    verification_job = Job(
        id="job-running-check",
        kind=JobKind.EDIT_VERIFY.value,
        status=JobStatus.RUNNING.value,
        progress=0,
        progress_json={},
        payload_json={},
    )
    started = asyncio.Event()

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == verification_job.id:
                return verification_job
            return None

        def commit(self) -> None:
            return None

        def execute(self, _statement):  # type: ignore[no-untyped-def]
            # The claim-bound terminal transition is a conditional UPDATE;
            # one owned row answers it.
            return SimpleNamespace(rowcount=1)

    @asynccontextmanager
    async def lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        yield _TEST_CLAIM

    async def block(_job_id: str, _claim: JobClaim) -> None:
        started.set()
        await asyncio.Event().wait()

    orchestrator = _orchestrator(session_factory=FakeSession)
    orchestrator.scheduler.job_lease = lease
    orchestrator._execute_image_edit_verification = block  # type: ignore[method-assign]
    task = asyncio.create_task(orchestrator._execute(verification_job.id, None))
    await started.wait()

    orchestrator._tasks[verification_job.id] = task
    orchestrator._preempted_image_edit_verifications.add(verification_job.id)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert verification_job.status == JobStatus.COMPLETE.value
    assert verification_job.result_json == {
        "version": "image-edit-verification-v1",
        "status": "skipped",
        "reason": VerificationReason.ASSESSMENT_INTERRUPTED.value,
        "automatic_retry_executed": False,
    }


async def test_preemption_cancels_the_running_check_before_foreground_execution() -> None:
    verification_job_id = "job-running-check"
    blocker = asyncio.create_task(asyncio.Event().wait())

    class Result:
        @staticmethod
        def all() -> list[str]:
            return [verification_job_id]

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def scalars(self, _statement):  # type: ignore[no-untyped-def]
            return Result()

    orchestrator = _orchestrator(session_factory=FakeSession)
    orchestrator._tasks[verification_job_id] = blocker

    await orchestrator._preempt_running_image_edit_verifications()

    assert blocker.cancelled()
    orchestrator.engines.chat.cancel.assert_awaited_once_with(verification_job_id)
    assert orchestrator._preempted_image_edit_verifications == set()


async def test_automatic_edit_retry_does_not_preempt_its_own_check() -> None:
    orchestrator = _orchestrator()
    orchestrator._is_image_edit_verification_retry = Mock(return_value=True)  # type: ignore[method-assign]
    orchestrator._preempt_running_image_edit_verifications = AsyncMock()  # type: ignore[method-assign]
    orchestrator._execute = AsyncMock()  # type: ignore[method-assign]

    await orchestrator._execute_after_preempting_verification("job-retry", "run-retry")

    orchestrator._preempt_running_image_edit_verifications.assert_not_awaited()
    orchestrator._execute.assert_awaited_once_with("job-retry", "run-retry")


async def test_invalid_verifier_payload_falls_back_without_a_failed_job() -> None:
    job = Job(
        id="job-invalid-verifier",
        kind=JobKind.EDIT_VERIFY.value,
        status=JobStatus.RUNNING.value,
        progress=0,
        progress_json={},
        payload_json={},
    )

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return job if model is Job and identity == job.id else None

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

        def execute(self, statement):  # type: ignore[no-untyped-def]
            # The claim-bound terminal transition is a conditional UPDATE;
            # one owned row answers it, and the values land on the object
            # the way a live session's identity map would carry them.
            for key, value in statement.compile().params.items():
                # SET-clause binds carry the bare column name; the WHERE
                # binds are suffixed, and must not be mirrored.
                if key == "status" and isinstance(value, str):
                    job.status = value
                if key == "completed_at":
                    job.completed_at = value
            # The read-only ownership probe selects (status, attempt, owner);
            # this one owned row answers for the test claim.
            return SimpleNamespace(
                rowcount=1,
                first=lambda: (job.status, _TEST_CLAIM.attempt, _TEST_CLAIM.token),
            )

    orchestrator = _orchestrator(session_factory=FakeSession)

    await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)

    assert job.status == JobStatus.COMPLETE.value
    assert job.result_json == {
        "version": "image-edit-verification-v1",
        "status": "skipped",
        "reason": VerificationReason.ASSESSMENT_UNAVAILABLE.value,
        "automatic_retry_executed": False,
    }


async def test_cancelling_runless_verifier_emits_no_run_event() -> None:
    job = Job(
        id="job-cancel-verifier",
        kind=JobKind.EDIT_VERIFY.value,
        status=JobStatus.RUNNING.value,
        progress=0,
        progress_json={},
        payload_json={},
    )

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return job if model is Job and identity == job.id else None

        def commit(self) -> None:
            return None

    orchestrator = _orchestrator(session_factory=FakeSession)

    assert await orchestrator.cancel(job.id) is True
    assert job.status == JobStatus.CANCELLED.value
    orchestrator.engines.chat.cancel.assert_awaited_once_with(job.id)
    orchestrator.engines.media.cancel.assert_not_called()
    orchestrator.events.publish.assert_not_awaited()


def test_late_verification_updates_its_revision_without_replacing_current_media() -> None:
    run = SimpleNamespace(
        id="run-old-revision",
        assistant_message_id="message-visible",
        provenance_json={"outputs": [{"artifact_id": "artifact-first"}]},
    )
    payload = ImageEditVerificationJobPayload(
        chat_id="chat-revision",
        source_run_id=run.id,
        source_job_id="job-source",
        source_artifact_id="artifact-source",
        result_artifact_id="artifact-first",
        vision_profile_id="profile-vision",
    )
    job = Job(
        id=image_edit_verification_job_id(run.id),
        kind=JobKind.EDIT_VERIFY.value,
        status=JobStatus.RUNNING.value,
        progress=0,
        progress_json={},
        payload_json=payload.model_dump(mode="json"),
    )
    revision = ResponseRevision(
        id="revision-old",
        message_id=run.assistant_message_id,
        run_id=run.id,
        sequence=1,
        status="complete",
        parts=[
            ResponseRevisionPart(
                position=0,
                type=PartType.IMAGE.value,
                artifact_id="artifact-first",
            ),
            ResponseRevisionPart(
                position=1,
                type=PartType.GENERATION_METADATA.value,
                metadata_json={"provenance": {"old": True}},
            ),
        ],
    )
    message = Message(
        id=run.assistant_message_id,
        chat_id="chat-revision",
        role="assistant",
        status="complete",
        active_response_revision_id="revision-new",
        parts=[
            MessagePart(
                position=0,
                type=PartType.IMAGE.value,
                artifact_id="artifact-new",
            ),
            MessagePart(
                position=1,
                type=PartType.GENERATION_METADATA.value,
                metadata_json={"provenance": {"current": True}},
            ),
        ],
    )

    class FakeSession:
        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (Run, run.id): run,
                (Message, message.id): message,
            }.get((model, identity))

        def scalar(self, _statement):  # type: ignore[no-untyped-def]
            return revision

    result = {
        "version": "image-edit-verification-v1",
        "status": "complete",
        "assessment": {
            "requested_change_visible": True,
            "unrelated_content_preserved": True,
            "retry_recommended": False,
            "direction": "none",
            "confidence": 0.94,
        },
        "retry": False,
        "reason": "accepted",
        "attempt": 0,
        "automatic_retry_executed": False,
    }
    orchestrator = _orchestrator()

    orchestrator._persist_image_edit_verification(
        FakeSession(),  # type: ignore[arg-type]
        job,
        result,
    )

    assert revision.parts[0].artifact_id == "artifact-first"
    assert revision.parts[1].metadata_json["provenance"]["image_edit_verification"] == result
    assert message.parts[0].artifact_id == "artifact-new"
    assert message.parts[1].metadata_json == {"provenance": {"current": True}}


def test_retry_output_is_not_verified_a_second_time() -> None:
    plan = SimpleNamespace(id="plan-retry", source_action="image_edit_verification_retry")
    run = SimpleNamespace(
        id="run-retry",
        chat_id="chat-retry",
        operation=Operation.IMAGE_TO_IMAGE.value,
        work_plan_id=plan.id,
        provenance_json={"existing": True},
    )

    class FakeSession:
        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return plan if model is WorkPlan and identity == plan.id else None

    orchestrator = _orchestrator()

    assert (
        orchestrator._queue_image_edit_verification(
            FakeSession(),  # type: ignore[arg-type]
            run,  # type: ignore[arg-type]
            "job-retry",
            ["artifact-retry"],
        )
        is None
    )
    assert run.provenance_json == {
        "existing": True,
        "image_edit_verification": {
            "version": "image-edit-verification-v1",
            "status": "skipped",
            "reason": VerificationReason.RETRY_LIMIT_REACHED.value,
            "automatic_retry_executed": False,
        },
    }


async def test_automatic_retry_reuses_source_turn_as_a_response_revision() -> None:
    source_user = Message(
        id="message-user",
        chat_id="chat-retry",
        parent_id="message-before",
        role="user",
        status="complete",
        parts=[MessagePart(position=0, type=PartType.TEXT.value, text="Make the mug green")],
    )
    source_assistant = Message(
        id="message-assistant",
        chat_id="chat-retry",
        parent_id=source_user.id,
        role="assistant",
        status="complete",
    )
    source_run = SimpleNamespace(
        id="run-source",
        chat_id="chat-retry",
        user_message_id=source_user.id,
        assistant_message_id=source_assistant.id,
        operation=Operation.IMAGE_TO_IMAGE.value,
        workflow_revision_id="workflow-revision",
        profile_id="profile-image",
        settings_json={"steps": 8, "denoise": 0.5},
        provenance_json={
            "image_edit": {
                "strength": {
                    "mode": "auto",
                    "parameter": "denoise",
                    "value": 0.5,
                    "scope": "localized",
                    "confidence": "high",
                }
            }
        },
    )
    retry_run = SimpleNamespace(id="run-retry", work_plan_id="plan-retry", provenance_json={})
    workflow_revision = SimpleNamespace(input_schema_json={"type": "object"})
    profile = SimpleNamespace(engine="comfyui")
    accepted_run = SimpleNamespace(
        id=retry_run.id,
        work_plan_id="plan-retry",
        provenance_json={
            "response_replacement": {
                "message_id": source_assistant.id,
                "revision_id": "revision-retry",
            }
        },
    )
    accepted = SimpleNamespace(run=accepted_run)
    verification_job = SimpleNamespace(status=JobStatus.RUNNING.value)
    commits: list[bool] = []

    class FakeSession:
        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (Run, source_run.id): source_run,
                (Run, retry_run.id): retry_run,
                (Job, image_edit_verification_job_id(source_run.id)): verification_job,
                (Message, source_assistant.id): source_assistant,
                (WorkflowRevision, source_run.workflow_revision_id): workflow_revision,
                (ModelProfile, source_run.profile_id): profile,
            }.get((model, identity))

        def scalar(self, _statement):  # type: ignore[no-untyped-def]
            return source_user

        def in_transaction(self) -> bool:
            return True

        def commit(self) -> None:
            commits.append(True)

        def execute(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(rowcount=1)

    payload = ImageEditVerificationJobPayload(
        chat_id=source_run.chat_id,
        source_run_id=source_run.id,
        source_job_id="job-source",
        source_artifact_id="artifact-original",
        result_artifact_id="artifact-first-result",
        vision_profile_id="profile-vision",
        automatic_strength=True,
        strength_parameter="denoise",
        current_strength=0.5,
        minimum=0.3,
        maximum=0.8,
    )
    decision = ImageEditRetryDecision(
        retry=True,
        reason=VerificationReason.ELIGIBLE,
        attempt=1,
        parameter="denoise",
        value_before=0.5,
        value_after=0.62,
        minimum=0.3,
        maximum=0.8,
    )
    orchestrator = _orchestrator()
    orchestrator.request_settings_for_operation = AsyncMock(  # type: ignore[method-assign]
        return_value={"steps": 8, "denoise": 0.5}
    )
    orchestrator.input_artifact_ids_for_run = Mock(  # type: ignore[method-assign]
        return_value=["artifact-original"]
    )
    orchestrator.create_turn = AsyncMock(return_value=accepted)  # type: ignore[method-assign]

    result = await orchestrator._create_image_edit_verification_retry(
        FakeSession(),  # type: ignore[arg-type]
        payload,
        decision,
        claim=_TEST_CLAIM,
    )

    assert result is accepted
    assert commits == [True], "the retry provenance is written in the creation transaction"
    call = orchestrator.create_turn.await_args
    hook = call.kwargs.get("before_commit")
    assert callable(hook), "the retry did not bind its durable turn to the claim"
    hook(FakeSession(), retry_run)
    assert retry_run.provenance_json["image_edit_verification_retry"]["attempt"] == 1
    assert retry_run.provenance_json["image_edit_verification_retry"]["strength_after"] == 0.62
    assert call.args[1] == source_run.chat_id
    request = call.args[2]
    assert request.text == "Make the mug green"
    assert request.mode == RoutingMode.IMAGE
    assert request.parent_message_id == source_user.parent_id
    assert request.input_artifact_ids == ["artifact-original"]
    assert request.settings == {"steps": 8, "denoise": 0.62}
    assert {k: v for k, v in call.kwargs.items() if k != "before_commit"} == {
        "use_explicit_parent": True,
        "replacement_message_id": source_assistant.id,
        "source_action": "image_edit_verification_retry",
        "reference_source_message_id": source_user.id,
        "inherited_image_edit_strength": {
            "mode": "auto",
            "parameter": "denoise",
            "value": 0.62,
            "scope": "localized",
            "confidence": "high",
        },
    }
    assert retry_run.provenance_json["image_edit_verification_retry"] == {
        "version": "image-edit-verification-v1",
        "source_run_id": source_run.id,
        "source_job_id": payload.source_job_id,
        "source_verification_job_id": image_edit_verification_job_id(source_run.id),
        "attempt": 1,
        "strength_parameter": "denoise",
        "strength_before": 0.5,
        "strength_after": 0.62,
    }


def _verification_world(*, lose_at: str):  # type: ignore[no-untyped-def]
    """A verification job with a complete source, a verifying chat, both
    artifacts and a verified vision profile, over a fake session whose
    ownership probe stops answering for the test claim once ``lose_at`` is
    reached: "preparation", "stop", "workers", "assessment" or "restore".
    Returns (job, orchestrator, world)."""

    job = Job(
        id="job-verify-owned",
        kind=JobKind.EDIT_VERIFY.value,
        status=JobStatus.RUNNING.value,
        progress=0,
        progress_json={},
        payload_json={
            "chat_id": "chat-v",
            "source_run_id": "run-source",
            "source_job_id": "job-source",
            "source_artifact_id": "artifact-source",
            "result_artifact_id": "artifact-result",
            "vision_profile_id": "profile-vision",
            "automatic_strength": True,
            "strength_parameter": "denoise",
            "current_strength": 0.5,
            "minimum": 0.3,
            "maximum": 0.8,
        },
    )
    run = SimpleNamespace(
        id="run-source",
        status=RunStatus.COMPLETE.value,
        chat_id="chat-v",
        standalone_prompt="make the mug green",
    )
    chat = SimpleNamespace(id="chat-v", vision_settings_json={"verify_image_edits": True})
    source = SimpleNamespace(id="artifact-source")
    result = SimpleNamespace(id="artifact-result")
    install = SimpleNamespace(id="install-vision", active=True)
    profile = SimpleNamespace(id="profile-vision", model_install_id=install.id)
    previous = SimpleNamespace(id="profile-chat", model_install_id="install-chat")
    previous_install = SimpleNamespace(id="install-chat", active=True)
    world = {"owned": True, "lost_at": None}

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (Job, job.id): job,
                (Run, run.id): run,
                (Chat, chat.id): chat,
                (Artifact, source.id): source,
                (Artifact, result.id): result,
                (ModelProfile, profile.id): profile,
                (ModelProfile, previous.id): previous,
                (ModelInstall, install.id): install,
                (ModelInstall, previous_install.id): previous_install,
            }.get((model, identity))

        def expunge(self, _instance):  # type: ignore[no-untyped-def]
            return None

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

        def execute(self, _statement):  # type: ignore[no-untyped-def]
            owned = world["owned"]
            return SimpleNamespace(
                rowcount=1 if owned else 0,
                first=lambda: (
                    job.status,
                    _TEST_CLAIM.attempt,
                    _TEST_CLAIM.token if owned else "a-later-attempt",
                ),
            )

    orchestrator = _orchestrator(session_factory=FakeSession)
    orchestrator.engines = SimpleNamespace(
        settings=SimpleNamespace(
            chat_engine="llama.cpp", media_engine="comfyui", vision_bridge_max_tokens=256
        ),
        chat=SimpleNamespace(cancel=AsyncMock()),
        media=SimpleNamespace(cancel=AsyncMock()),
    )

    loads: list[object] = []

    async def load_chat(profile_argument: object = None, *_args: object, **_kwargs: object) -> None:
        loads.append(profile_argument)
        if lose_at == "workers":
            world["owned"] = False
            world["lost_at"] = "workers"
        # "restore" is the teardown's own load, which is the second one: the
        # verification loads its vision profile first and puts the previous
        # chat profile back afterwards.
        if lose_at == "restore" and len(loads) == 2:
            world["owned"] = False
            world["lost_at"] = "restore"

    async def stop_worker(*_args: object, **_kwargs: object) -> None:
        if lose_at == "stop":
            world["owned"] = False
            world["lost_at"] = "stop"

    world["loads"] = loads

    orchestrator.processes = SimpleNamespace(
        statuses=Mock(
            return_value=[
                WorkerStatus(
                    name="chat", state="ready", managed=True, running=True, profile_id=previous.id
                ),
                WorkerStatus(name="media", state="ready", managed=True, running=True),
            ]
        ),
        stop=AsyncMock(side_effect=stop_worker),
        load_chat=AsyncMock(side_effect=load_chat),
    )
    orchestrator._profile_has_verified_vision = Mock(return_value=True)  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._release_deferred_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._persist_image_edit_verification = Mock(return_value=True)  # type: ignore[method-assign]

    async def prepare(*_args: object, **_kwargs: object) -> object:
        if lose_at == "preparation":
            world["owned"] = False
            world["lost_at"] = "preparation"
        return SimpleNamespace(inspected_artifact_ids=[source.id, result.id])

    orchestrator.vision = SimpleNamespace(  # type: ignore[assignment]
        prepare=AsyncMock(side_effect=prepare),
        attach_to_latest_user=Mock(side_effect=lambda messages, _visual: messages),
    )

    async def chat_capabilities() -> object:
        return SimpleNamespace(input_modalities=["text", "image"])

    orchestrator.engines.chat_capabilities = chat_capabilities  # type: ignore[attr-defined]
    consumed: list[str] = []

    async def stream(_request):  # type: ignore[no-untyped-def]
        from local_lm.adapters.base import ChatEvent

        consumed.append("first")
        yield ChatEvent(type="delta", text="{", data={})
        if lose_at == "assessment":
            world["owned"] = False
            world["lost_at"] = "assessment"
        consumed.append("second")
        yield ChatEvent(type="delta", text="}", data={})
        consumed.append("complete")
        yield ChatEvent(type="complete", text="", data={})

    orchestrator.engines.chat.stream = stream  # type: ignore[attr-defined]
    world["consumed"] = consumed
    return job, orchestrator, world


async def test_a_verification_that_never_loaded_chat_restores_nothing() -> None:
    """A verification that returns before loading chat must not put chat back.

    The profile to restore is chosen from a reading taken at entry, long before
    the load, so it says who WOULD be put back rather than that anything was
    displaced. When the vision preparation refuses the images the execution
    returns early, having moved nothing - and the teardown would still stop the
    chat worker and cold-start it with the profile it is already running.
    """
    job, orchestrator, world = _verification_world(lose_at="never")

    async def refuse(*_args: object, **_kwargs: object) -> object:
        raise VisionInputError("the source and result could not be read together")

    orchestrator.vision.prepare = AsyncMock(side_effect=refuse)  # type: ignore[attr-defined]

    await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)

    assert world["loads"] == [], (
        "a verification that moved no worker cold-reloaded the running chat profile"
    )
    orchestrator.processes.stop.assert_not_awaited()


async def test_a_verification_that_loses_its_claim_after_preparation_moves_no_worker() -> None:
    """The claim is lost during the vision preparation: the execution stops
    before any worker is stopped or loaded, the assessment never streams,
    and the teardown restores nothing - the workers are the successor's to
    move."""

    from local_lm.orchestrator import ClaimLost

    job, orchestrator, world = _verification_world(lose_at="preparation")

    with pytest.raises(ClaimLost, match="after preparation"):
        await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)

    orchestrator.processes.stop.assert_not_awaited()
    orchestrator.processes.load_chat.assert_not_awaited()
    assert world["consumed"] == []
    orchestrator._schedule_media_restart.assert_not_called()
    orchestrator._release_deferred_media_restart.assert_not_called()
    orchestrator._persist_image_edit_verification.assert_not_called()
    assert job.status == JobStatus.RUNNING.value, "a lost claim settled the row"


async def test_a_verification_that_loses_its_claim_mid_assessment_stops_and_restores_nothing() -> (
    None
):
    """The claim is lost after the assessment's first token: the next event
    is refused, the stream is not consumed further, nothing is persisted,
    and the teardown neither restores chat nor schedules media - even
    though this execution stopped media and loaded chat for its own run."""

    from local_lm.orchestrator import ClaimLost

    job, orchestrator, world = _verification_world(lose_at="assessment")

    with pytest.raises(ClaimLost, match="mid-assessment"):
        await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)

    orchestrator.processes.stop.assert_awaited_once_with("media")
    assert orchestrator.processes.load_chat.await_count == 1, "the teardown restored chat"
    assert world["consumed"] == ["first", "second"]
    orchestrator._schedule_media_restart.assert_not_called()
    orchestrator._release_deferred_media_restart.assert_not_called()
    orchestrator._persist_image_edit_verification.assert_not_called()


async def test_a_verification_that_keeps_its_claim_restores_the_workers() -> None:
    """The same world without a loss: the assessment completes, and the
    teardown restores the previous chat profile and schedules media back
    - the recovery an owning attempt still performs."""

    job, orchestrator, _world = _verification_world(lose_at="never")

    await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)

    assert orchestrator.processes.load_chat.await_count == 2
    orchestrator._schedule_media_restart.assert_called_once()


async def test_a_verification_that_loses_its_claim_after_its_workers_streams_nothing() -> None:
    """The claim is lost while the chat model loads for the assessment:
    the capabilities are never read, the assessment never streams, and
    the teardown restores nothing - the chat this execution loaded is the
    successor's to keep or move."""

    from local_lm.orchestrator import ClaimLost

    job, orchestrator, world = _verification_world(lose_at="workers")

    with pytest.raises(ClaimLost, match="after its workers"):
        await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)

    assert orchestrator.processes.load_chat.await_count == 1, "the teardown restored chat"
    assert world["consumed"] == []
    orchestrator._schedule_media_restart.assert_not_called()
    orchestrator._persist_image_edit_verification.assert_not_called()


async def test_a_verification_that_loses_its_claim_stopping_media_loads_no_chat() -> None:
    """The claim is lost inside the awaited media stop.

    Stopping media and loading chat are two effects with an await between
    them. A successor that claims the row during that await is already using
    the global chat worker, so the losing execution must not load over it.
    The media worker it stopped is the successor's to bring back, so this
    execution schedules no restart either.
    """

    from local_lm.orchestrator import ClaimLost

    job, orchestrator, world = _verification_world(lose_at="stop")

    with pytest.raises(ClaimLost, match="after stopping media"):
        await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)

    orchestrator.processes.stop.assert_awaited_once()
    orchestrator.processes.load_chat.assert_not_awaited()
    assert world["lost_at"] == "stop"
    assert world["consumed"] == []
    orchestrator._schedule_media_restart.assert_not_called()
    orchestrator._release_deferred_media_restart.assert_not_called()
    orchestrator._persist_image_edit_verification.assert_not_called()
    assert job.status == JobStatus.RUNNING.value, "a lost claim settled the row"


async def test_a_verification_that_loses_its_claim_restoring_chat_schedules_no_restart() -> None:
    """The claim is lost inside the awaited teardown restore.

    The teardown reads ownership, then awaits the chat restore, then decides
    the media restart. A single reading taken before that await decides the
    restart on what was true beforehand; the restart must be bound to
    ownership at the moment it is scheduled, not to a stale answer.
    """

    job, orchestrator, world = _verification_world(lose_at="restore")

    await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)

    assert world["lost_at"] == "restore"
    assert len(world["loads"]) == 2, "the teardown restore did not run"
    orchestrator._schedule_media_restart.assert_not_called()
    orchestrator._release_deferred_media_restart.assert_not_called()


async def test_failed_verification_load_restores_displaced_chat() -> None:
    """A startup failure can follow stopping the previous chat worker."""
    job, orchestrator, world = _verification_world(lose_at="never")
    running_profile = "profile-chat"

    async def load(profile, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal running_profile
        world["loads"].append(profile)
        running_profile = None
        if profile.id == "profile-vision":
            raise RuntimeError("neutral synthetic worker startup failure")
        running_profile = profile.id

    orchestrator.processes.load_chat = AsyncMock(side_effect=load)
    await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)

    assert running_profile == "profile-chat", "failed verification left the previous chat down"
    assert [profile.id for profile in world["loads"]] == ["profile-vision", "profile-chat"]


async def test_verification_artifact_mismatch_leaves_chat_untouched() -> None:
    job, orchestrator, world = _verification_world(lose_at="never")
    orchestrator.vision.prepare = AsyncMock(
        return_value=SimpleNamespace(inspected_artifact_ids=["artifact-source"])
    )
    await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)
    assert world["loads"] == []
    orchestrator.processes.stop.assert_not_awaited()


async def test_shutdown_does_not_restore_chat_from_active_verification() -> None:
    job, orchestrator, world = _verification_world(lose_at="never")
    entered = asyncio.Event()

    async def stream(_request):  # type: ignore[no-untyped-def]
        entered.set()
        await asyncio.Event().wait()
        yield

    orchestrator.engines.chat.stream = stream
    task = asyncio.create_task(orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM))
    orchestrator._tasks[job.id] = task
    await entered.wait()
    assert [profile.id for profile in world["loads"]] == ["profile-vision"]

    await orchestrator.close()

    assert [profile.id for profile in world["loads"]] == ["profile-vision"], (
        "shutdown restored chat from verification only to destroy it next"
    )
