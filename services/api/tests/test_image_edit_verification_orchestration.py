from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from local_lm.domain import JobKind, JobStatus, Operation, PartType, RunStatus
from local_lm.image_edit_verification import (
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
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.schemas import WorkerStatus


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


def test_successful_edit_queues_one_dependent_low_priority_verifier(monkeypatch) -> None:
    source = SimpleNamespace(id="artifact-source", media_type="image/png")
    result = SimpleNamespace(id="artifact-result", media_type="image/png")
    chat = SimpleNamespace(
        id="chat-verify",
        vision_settings_json={"verify_image_edits": True},
    )
    profile = SimpleNamespace(id="profile-vision")
    plan = SimpleNamespace(id="plan-edit", summary_json={})
    run = SimpleNamespace(
        id="run-edit",
        chat_id=chat.id,
        operation=Operation.IMAGE_TO_IMAGE.value,
        work_plan_id=plan.id,
        work_step_id="step-source",
        provenance_json={
            "image_edit": {
                "strength": {
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
    assert len(jobs) == len(steps) == len(dependencies) == 1
    job = jobs[0]
    assert (job.kind, job.run_id, job.queue_resource, job.queue_group, job.queue_priority) == (
        JobKind.EDIT_VERIFY.value,
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
    assert dependencies[0].depends_on_step_id == run.work_step_id


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

    async def execute_media(_job_id: str, _run_id: str) -> str:
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

    orchestrator = _orchestrator(session_factory=FakeSession)

    await orchestrator._execute_image_edit_verification(job.id)

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
