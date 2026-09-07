from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from local_lm.adapters.base import ChatEvent, MediaEvent, estimate_chat_tokens
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_templates import COMFY_TEMPLATE_COMPILER_VERSION
from local_lm.models import (
    Job,
    Message,
    MessagePart,
    ModelInstall,
    ModelProfile,
    Run,
    WorkflowActivation,
    WorkflowRevision,
    WorkStep,
)
from local_lm.orchestrator import ClaimLost, ConversationOrchestrator, _queued_workflow_activation
from local_lm.scheduler import JobClaim
from local_lm.schemas import EngineCapabilities, WorkerStatus
from local_lm.workflow_activations import WorkflowActivationLaunchScope

_TEST_CLAIM = JobClaim(token="attempt-token-a", attempt=1)


def test_contract_backed_queue_freezes_only_the_ready_active_activation() -> None:
    revision = SimpleNamespace(id="wfrev-one", dependency_contract_sha256="a" * 64)
    activation = SimpleNamespace(
        id="wfact-one",
        resolver_version="workflow-activation-v1",
        dependency_contract_sha256="a" * 64,
        binding_sha256="b" * 64,
        details_json={"launch_sha256": "c" * 64},
    )

    class FakeSession:
        def scalar(self, _query):  # type: ignore[no-untyped-def]
            return activation

    assert _queued_workflow_activation(FakeSession(), revision) == {
        "id": "wfact-one",
        "resolver_version": "workflow-activation-v1",
        "dependency_contract_sha256": "a" * 64,
        "binding_sha256": "b" * 64,
        "launch_sha256": "c" * 64,
    }
    activation.details_json = {}
    with pytest.raises(ValueError, match="dependencies are not ready"):
        _queued_workflow_activation(FakeSession(), revision)
    assert _queued_workflow_activation(FakeSession(), None) is None


def test_media_prompt_uses_the_frozen_combined_trigger_words() -> None:
    run = SimpleNamespace(
        operation="text_to_image",
        standalone_prompt="A candid portrait",
        provenance_json={
            "auxiliary_assets": {
                "model_trigger_words_applied": ["portrait-style"],
                "lora_trigger_words_applied": ["atelier ink"],
                "trigger_words_applied": ["portrait-style", "atelier ink"],
            }
        },
    )

    assert ConversationOrchestrator._media_prompt(run) == (
        "A candid portrait, portrait-style, atelier ink"
    )


def test_successful_media_evidence_requires_an_exact_official_contract(monkeypatch) -> None:
    template_sha256 = "b" * 64
    profile = SimpleNamespace(
        id="profile-image",
        model_install_id="install-image",
        role="image",
        engine="comfyui",
    )
    install = SimpleNamespace(
        id="install-image",
        active=True,
        role="image",
        engine="comfyui",
        manifest_json={
            "expected_sha256": {"model.safetensors": "a" * 64},
            "workflow_template_id": "image_edit",
            "workflow_template_sha256": template_sha256,
        },
    )
    performance = {"version": 1, "signals": [{"kind": "model-cache"}]}
    revision = SimpleNamespace(
        id="revision-image",
        trusted=True,
        engine="comfyui",
        artifact_sha256="d" * 64,
        input_schema_json={"x-lm-atelier-workflow-performance": performance},
        dependencies_json={
            "model_install_ids": [install.id],
            "compiler_version": COMFY_TEMPLATE_COMPILER_VERSION,
            "template_id": "image_edit",
            "template_sha256": template_sha256,
        },
    )
    run = SimpleNamespace(
        operation="image_to_image",
        profile_id=profile.id,
        workflow_revision_id=revision.id,
    )

    class FakeSession:
        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (ModelProfile, profile.id): profile,
                (ModelInstall, install.id): install,
                (WorkflowRevision, revision.id): revision,
            }.get((model, identity))

    recorder = Mock(return_value=SimpleNamespace(evidence_key="evidence-key"))
    monkeypatch.setattr("local_lm.orchestrator.record_capability_evidence", recorder)
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=SimpleNamespace(runtimes=None),
    )
    capabilities = EngineCapabilities(
        engine="comfyui",
        version="comfy-test",
        roles=["image"],
        operations=["image_to_image"],
        formats=["png"],
        devices=["gpu:0"],
        streaming=True,
        tool_calling=False,
        settings=[],
        healthy=True,
    )

    assert (
        orchestrator._record_successful_media_evidence(
            FakeSession(),
            run,
            capabilities,
            output_count=1,
        )
        == "evidence-key"
    )
    assert recorder.call_args.kwargs["component_hashes"] == {"model.safetensors": "a" * 64}
    # Evidence is keyed on what the revision executes, not on the template it was
    # compiled from, so a recompile that changes nothing keeps the proof.
    assert recorder.call_args.kwargs["workflow_contract_version"] == revision.artifact_sha256
    assert recorder.call_args.kwargs["details"] == {
        "probe": "successful_media_output",
        "operation": "image_to_image",
        "workflow_revision_id": revision.id,
        "workflow_template_id": "image_edit",
        "workflow_performance": performance,
        "output_count": 1,
    }

    assert (
        orchestrator._record_successful_media_evidence(
            FakeSession(),
            run,
            capabilities.model_copy(update={"engine": "mock"}),
            output_count=1,
        )
        is None
    )
    assert (
        orchestrator._record_successful_media_evidence(
            FakeSession(),
            run,
            capabilities.model_copy(update={"healthy": False}),
            output_count=1,
        )
        is None
    )
    assert (
        orchestrator._record_successful_media_evidence(
            FakeSession(),
            run,
            capabilities,
            output_count=0,
        )
        is None
    )
    assert recorder.call_count == 1


async def test_context_folding_preserves_system_and_current_messages() -> None:
    messages = [
        {"role": "system", "content": "Keep the project instruction."},
        {"role": "system", "content": "Keep the scoped instruction too."},
        {"role": "user", "content": "Old user detail " * 30},
        {"role": "assistant", "content": "Old assistant detail " * 30},
        {"role": "user", "content": "Current request must remain."},
    ]
    source_ids = [None, None, "old-user", "old-assistant", "current-user"]
    engines = SimpleNamespace(
        chat=SimpleNamespace(
            count_tokens=AsyncMock(side_effect=lambda value: estimate_chat_tokens(value))
        ),
        settings=SimpleNamespace(),
    )
    orchestrator = ConversationOrchestrator(
        engines=engines,
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=Mock(),
    )

    fitted, tokens, omitted, compaction, fitted_source_ids = await orchestrator._fit_chat_context(
        messages,
        source_ids,
        input_budget=160,
    )

    assert tokens <= 160
    assert fitted[0] == messages[0]
    assert fitted[1] == messages[1]
    assert fitted[-1] == messages[-1]
    assert fitted[2]["role"] == "assistant"
    assert "Earlier conversation compacted" in fitted[2]["content"]
    assert omitted == 2
    assert compaction["active"] is True
    assert compaction["source_message_ids"] == ["old-user", "old-assistant"]
    assert compaction["transcript_preserved"] is True
    assert fitted_source_ids == [None, None, None, "current-user"]


async def test_managed_chat_worker_is_aligned_to_the_run_profile() -> None:
    run = SimpleNamespace(profile_id="profile-selected")
    profile = SimpleNamespace(id="profile-selected", model_install_id="install-selected")
    install = SimpleNamespace(id="install-selected")

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (Run, "run-1"): run,
                (ModelProfile, "profile-selected"): profile,
                (ModelInstall, "install-selected"): install,
            }.get((model, identity))

        def expunge(self, _value) -> None:  # type: ignore[no-untyped-def]
            return None

    previous = WorkerStatus(
        name="chat",
        state="ready",
        managed=True,
        running=True,
        pid=11,
        profile_id="profile-previous",
    )
    aligned = previous.model_copy(update={"pid": 12, "profile_id": "profile-selected"})
    processes = SimpleNamespace(
        settings=SimpleNamespace(llama_executable=Path("llama-server")),
        statuses=Mock(return_value=[previous]),
        load_chat=AsyncMock(return_value=aligned),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(chat_engine="llama.cpp")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
        session_factory=FakeSession,
    )

    result = await orchestrator._ensure_chat_worker("run-1")

    assert result == aligned
    processes.load_chat.assert_awaited_once_with(profile, install)


async def test_engine_cancel_runs_after_the_database_session_closes() -> None:
    # `attempt` is part of a Job and the cancel now reads it, so the stand-in
    # carries it too rather than the production code learning to do without.
    job = SimpleNamespace(id="job-cancel", status="running", run_id="run-cancel", attempt=0)
    run = SimpleNamespace(id="run-cancel", operation="text")
    session_closed = False

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            nonlocal session_closed
            session_closed = True

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (Job, "job-cancel"): job,
                (Run, "run-cancel"): run,
            }.get((model, identity))

        def commit(self) -> None:
            return None

    async def cancel(run_id: str) -> None:
        assert run_id == "run-cancel"
        assert session_closed is True

    engines = SimpleNamespace(
        settings=SimpleNamespace(),
        chat=SimpleNamespace(cancel=cancel),
        media=SimpleNamespace(cancel=AsyncMock()),
    )
    orchestrator = ConversationOrchestrator(
        engines=engines,
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(publish_job=AsyncMock()),
        processes=Mock(),
        session_factory=FakeSession,
    )
    orchestrator._mark_cancelled = Mock()  # type: ignore[method-assign]

    assert await orchestrator.cancel("job-cancel") is True
    orchestrator._mark_cancelled.assert_called_once()  # type: ignore[attr-defined]


async def test_chat_worker_resume_runs_after_the_database_session_closes() -> None:
    profile = SimpleNamespace(id="profile-resume", model_install_id="install-resume")
    install = SimpleNamespace(id="install-resume")
    session_closed = False

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            nonlocal session_closed
            session_closed = True

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (ModelProfile, "profile-resume"): profile,
                (ModelInstall, "install-resume"): install,
            }.get((model, identity))

        def expunge(self, _value) -> None:  # type: ignore[no-untyped-def]
            return None

    async def load_chat(selected_profile, selected_install):  # type: ignore[no-untyped-def]
        assert session_closed is True
        assert selected_profile is profile
        assert selected_install is install

    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=SimpleNamespace(load_chat=load_chat),
        session_factory=FakeSession,
    )

    await orchestrator._resume_chat_worker("profile-resume")


async def test_media_handoff_recycles_managed_comfy_before_chat_resume() -> None:
    order: list[str] = []
    media = WorkerStatus(
        name="media",
        state="ready",
        managed=True,
        running=True,
        pid=22,
    )

    async def stop(name: str) -> None:
        assert name == "media"
        order.append("stop media")

    async def start_media() -> None:
        order.append("start media")

    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        stop=stop,
        start_media=start_media,
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )

    async def resume(profile_id: str) -> None:
        assert profile_id == "profile-chat"
        order.append("resume chat")

    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")
    if orchestrator._media_restart_task:
        await orchestrator._media_restart_task

    assert order[0] == "stop media"
    assert order[1:] == ["resume chat", "start media"]


async def test_media_handoff_preloads_the_next_queued_text_profile() -> None:
    media = WorkerStatus(
        name="media",
        state="ready",
        managed=True,
        running=True,
        pid=22,
    )
    next_run = SimpleNamespace(operation="text", profile_id="profile-next")

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Run and identity == "run-next":
                return next_run
            return None

    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        stop=AsyncMock(),
        start_media=AsyncMock(),
    )
    scheduler = SimpleNamespace(
        peek_next_eligible_job=Mock(return_value=("job-next", "run-next")),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=scheduler,
        processes=processes,
        session_factory=FakeSession,
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-previous")

    resume.assert_awaited_once_with("profile-next")
    assert orchestrator._media_restart_after_chat_activity is True
    processes.start_media.assert_not_awaited()

    orchestrator._release_deferred_media_restart()
    if orchestrator._media_restart_task:
        await orchestrator._media_restart_task

    assert orchestrator._media_restart_after_chat_activity is False
    processes.start_media.assert_awaited_once()


async def test_cancelled_chat_release_restores_planner_readiness() -> None:
    chat = WorkerStatus(
        name="chat",
        state="ready",
        managed=True,
        running=True,
        pid=21,
        profile_id="profile-chat",
    )

    async def cancelled_stop(_name: str) -> None:
        raise asyncio.CancelledError

    processes = SimpleNamespace(
        settings=SimpleNamespace(auto_unload_chat_for_media=True),
        statuses=Mock(return_value=[chat]),
        stop=cancelled_stop,
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._prepare_device_handoff("text_to_image", claim=_TEST_CLAIM)

    assert orchestrator._chat_planner_ready.is_set()


async def test_a_refused_chat_release_restores_planner_readiness() -> None:
    """A refused phase must leave the planner usable, as a cancelled stop does.

    The release clears the planner latch and then announces itself, and that
    announcement is a claim-bound write that RAISES when the row belongs to a
    later attempt. It needs no cancellation and no scheduling window. Left
    unrestored the latch stays cleared for the life of the process, because the
    only other place that sets it is the resume the caller runs after this
    method returns - and this method never returned.
    """
    chat = WorkerStatus(
        name="chat",
        state="ready",
        managed=True,
        running=True,
        pid=21,
        profile_id="profile-chat",
    )
    stop = AsyncMock()
    processes = SimpleNamespace(
        settings=SimpleNamespace(auto_unload_chat_for_media=True),
        statuses=Mock(return_value=[chat]),
        stop=stop,
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    # The phase writer answers False exactly as it does for a row a later
    # attempt now owns, which is what `_require_phase` turns into a stop.
    orchestrator._set_media_phase = AsyncMock(return_value=False)  # type: ignore[method-assign]

    with pytest.raises(ClaimLost, match="Releasing chat model"):
        await orchestrator._prepare_device_handoff(
            "text_to_image",
            claim=_TEST_CLAIM,
            job_id="job-release",
            run_id="run-release",
        )

    assert orchestrator._chat_planner_ready.is_set(), (
        "a refused release left the chat planner unavailable for the rest of the process"
    )
    stop.assert_not_awaited()


async def test_media_worker_startup_forwards_truthful_phases() -> None:
    media = WorkerStatus(
        name="media",
        state="stopped",
        managed=True,
        running=False,
    )

    async def start_media(*, phase_callback) -> None:  # type: ignore[no-untyped-def]
        for phase in (
            "Provisioning media runtime",
            "Validating media dependencies",
            "Starting media runtime",
        ):
            await phase_callback(phase)

    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=SimpleNamespace(
            statuses=Mock(return_value=[media]),
            start_media=start_media,
        ),
    )
    phases = AsyncMock()
    orchestrator._set_media_phase = phases  # type: ignore[method-assign]

    await orchestrator._ensure_media_worker(claim=_TEST_CLAIM, job_id="job-1", run_id="run-1")

    assert [call.args[2] for call in phases.await_args_list] == [
        "Starting media worker",
        "Provisioning media runtime",
        "Validating media dependencies",
        "Starting media runtime",
    ]


async def test_ready_media_worker_still_receives_an_exact_activation_scope() -> None:
    media = WorkerStatus(
        name="media",
        state="ready",
        managed=True,
        running=True,
        pid=22,
    )
    scope = WorkflowActivationLaunchScope(
        "wfact-one",
        "wfrev-one",
        "a" * 64,
        "b" * 64,
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
    )
    start_media = AsyncMock()
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=SimpleNamespace(
            statuses=Mock(return_value=[media]),
            start_media=start_media,
        ),
    )

    await orchestrator._ensure_media_worker(claim=_TEST_CLAIM, activation_scope=scope)

    start_media.assert_awaited_once_with(activation_scope=scope)


def test_media_execution_revalidates_the_exact_queued_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = {
        "id": "wfact-one",
        "resolver_version": "workflow-activation-v1",
        "dependency_contract_sha256": "a" * 64,
        "binding_sha256": "b" * 64,
        "launch_sha256": "c" * 64,
    }
    revision = SimpleNamespace(id="wfrev-one", dependency_contract_sha256="a" * 64)
    activation = SimpleNamespace(
        id="wfact-one",
        workflow_revision_id="wfrev-one",
        resolver_version="workflow-activation-v1",
        dependency_contract_sha256="a" * 64,
        binding_sha256="b" * 64,
    )
    run = SimpleNamespace(
        workflow_revision_id="wfrev-one",
        provenance_json={"workflow": {"activation": snapshot}},
    )
    scope = WorkflowActivationLaunchScope(
        "wfact-one",
        "wfrev-one",
        "b" * 64,
        "c" * 64,
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
    )

    class FakeSession:
        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (WorkflowRevision, "wfrev-one"): revision,
                (WorkflowActivation, "wfact-one"): activation,
            }.get((model, identity))

    revalidate = Mock(return_value=scope)
    monkeypatch.setattr("local_lm.orchestrator.revalidate_workflow_activation", revalidate)
    settings = SimpleNamespace(
        custom_node_dir=tmp_path / "nodes",
        state_dir=tmp_path / "state",
        registry_dir=tmp_path / "registry",
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=settings),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=SimpleNamespace(runtimes=None),
    )

    assert orchestrator._media_activation_scope(FakeSession(), run) is scope
    assert revalidate.call_args.kwargs == {
        "runtime_materializer": None,
        "custom_node_root": settings.custom_node_dir,
        "registry_environment_root": registry_wheel_environment_root(settings.registry_dir),
    }

    run.provenance_json["workflow"]["activation"]["launch_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="launch identity changed"):
        orchestrator._media_activation_scope(FakeSession(), run)


async def test_scoped_media_handoff_does_not_restart_a_broad_worker() -> None:
    media = WorkerStatus(
        name="media",
        state="ready",
        managed=True,
        running=True,
        pid=22,
    )
    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        launch_scope_sha256=Mock(return_value="a" * 64),
        stop=AsyncMock(),
        start_media=AsyncMock(),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    orchestrator._resume_chat_worker = AsyncMock()  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    processes.stop.assert_awaited_once_with("media")
    orchestrator._resume_chat_worker.assert_awaited_once_with("profile-chat")
    processes.start_media.assert_not_awaited()
    assert orchestrator._media_restart_task is None


async def test_media_execution_awaits_inflight_handoff_restart() -> None:
    restart_release = asyncio.Event()
    restart_entered = asyncio.Event()
    start_calls = 0

    async def start_media() -> None:
        nonlocal start_calls
        start_calls += 1
        restart_entered.set()
        await restart_release.wait()

    processes = SimpleNamespace(
        statuses=Mock(
            return_value=[
                WorkerStatus(
                    name="media",
                    state="ready",
                    managed=True,
                    running=True,
                    pid=22,
                )
            ]
        ),
        start_media=start_media,
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    orchestrator._schedule_media_restart()
    await restart_entered.wait()
    ensure_task = asyncio.create_task(orchestrator._ensure_media_worker(claim=_TEST_CLAIM))
    await asyncio.sleep(0)

    assert start_calls == 1

    restart_release.set()
    await ensure_task

    assert start_calls == 1


def _plan_prewarm_orchestrator(  # type: ignore[no-untyped-def]
    next_step,
    *,
    media_running: bool = False,
    start_media=None,
    job=None,
    next_revision=None,
):
    current_step = SimpleNamespace(ordinal=1)

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (WorkStep, "step-current"): current_step,
                (Job, "job-current"): job,
                (WorkflowRevision, "revision-next"): next_revision,
            }.get((model, identity))

        def scalar(self, _query):  # type: ignore[no-untyped-def]
            return next_step

    processes = SimpleNamespace(
        statuses=Mock(
            return_value=[
                WorkerStatus(
                    name="media",
                    state="ready" if media_running else "stopped",
                    managed=media_running,
                    running=media_running,
                )
            ]
        ),
        start_media=start_media or AsyncMock(),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
        session_factory=FakeSession,
    )
    return orchestrator, processes, FakeSession


def _ordered_text_run():  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        operation="text",
        work_plan_id="plan-1",
        work_step_id="step-current",
    )


async def test_ordered_text_step_prewarms_the_following_media_step() -> None:
    next_step = SimpleNamespace(operation="text_to_image", status="queued")
    orchestrator, processes, session_factory = _plan_prewarm_orchestrator(next_step)

    with session_factory() as session:
        orchestrator._arm_step_prewarm(session, _ordered_text_run())

    assert orchestrator._step_prewarm_plan_id == "plan-1"
    processes.start_media.assert_not_awaited()

    orchestrator._begin_step_prewarm()
    prewarm_task = orchestrator._step_prewarm_task
    assert prewarm_task is not None
    assert prewarm_task is orchestrator._media_restart_task
    await prewarm_task

    processes.start_media.assert_awaited_once()


async def test_ordered_text_step_does_not_prewarm_a_text_successor() -> None:
    next_step = SimpleNamespace(operation="text", status="queued")
    orchestrator, processes, session_factory = _plan_prewarm_orchestrator(next_step)

    with session_factory() as session:
        orchestrator._arm_step_prewarm(session, _ordered_text_run())
    orchestrator._begin_step_prewarm()

    assert orchestrator._step_prewarm_plan_id is None
    assert orchestrator._step_prewarm_task is None
    processes.start_media.assert_not_awaited()


async def test_ordered_text_step_does_not_broadly_prewarm_a_contract_workflow() -> None:
    next_step = SimpleNamespace(
        operation="text_to_image",
        status="queued",
        workflow_revision_id="revision-next",
    )
    next_revision = SimpleNamespace(dependency_contract_sha256="a" * 64)
    orchestrator, processes, session_factory = _plan_prewarm_orchestrator(
        next_step,
        next_revision=next_revision,
    )

    with session_factory() as session:
        orchestrator._arm_step_prewarm(session, _ordered_text_run())
    orchestrator._begin_step_prewarm()

    assert orchestrator._step_prewarm_plan_id is None
    processes.start_media.assert_not_awaited()


async def test_single_step_turns_never_arm_a_prewarm() -> None:
    orchestrator, processes, session_factory = _plan_prewarm_orchestrator(None)
    run = SimpleNamespace(operation="text", work_plan_id=None, work_step_id=None)

    with session_factory() as session:
        orchestrator._arm_step_prewarm(session, run)
    orchestrator._begin_step_prewarm()

    assert orchestrator._step_prewarm_plan_id is None
    processes.start_media.assert_not_awaited()


async def test_running_media_worker_skips_the_prewarm() -> None:
    next_step = SimpleNamespace(operation="text_to_image", status="queued")
    orchestrator, processes, session_factory = _plan_prewarm_orchestrator(
        next_step, media_running=True
    )

    with session_factory() as session:
        orchestrator._arm_step_prewarm(session, _ordered_text_run())
    orchestrator._begin_step_prewarm()

    assert orchestrator._step_prewarm_plan_id is None
    processes.start_media.assert_not_awaited()


async def test_completed_step_without_streaming_still_prewarms() -> None:
    next_step = SimpleNamespace(operation="text_to_video", status="queued")
    job = SimpleNamespace(status="complete")
    orchestrator, processes, session_factory = _plan_prewarm_orchestrator(next_step, job=job)

    with session_factory() as session:
        orchestrator._arm_step_prewarm(session, _ordered_text_run())
    await orchestrator._settle_step_prewarm("job-current")
    if orchestrator._media_restart_task:
        await orchestrator._media_restart_task

    assert orchestrator._step_prewarm_plan_id is None
    processes.start_media.assert_awaited_once()


@pytest.mark.parametrize("job_status", ["cancelled", "failed"])
async def test_unsuccessful_step_stops_the_inflight_prewarm(job_status: str) -> None:
    start_entered = asyncio.Event()
    start_release = asyncio.Event()
    start_finished = False

    async def start_media() -> None:
        nonlocal start_finished
        start_entered.set()
        await start_release.wait()
        start_finished = True

    next_step = SimpleNamespace(operation="text_to_image", status="queued")
    job = SimpleNamespace(status=job_status)
    orchestrator, _processes, session_factory = _plan_prewarm_orchestrator(
        next_step, start_media=start_media, job=job
    )

    with session_factory() as session:
        orchestrator._arm_step_prewarm(session, _ordered_text_run())
    orchestrator._begin_step_prewarm()
    prewarm_task = orchestrator._step_prewarm_task
    assert prewarm_task is not None
    await start_entered.wait()

    await orchestrator._settle_step_prewarm("job-current")

    assert prewarm_task.cancelled()
    assert start_finished is False
    assert orchestrator._step_prewarm_plan_id is None
    assert orchestrator._step_prewarm_task is None
    assert orchestrator._media_restart_task is None


async def test_a_failing_step_leaves_the_give_back_released_beside_the_prewarm() -> None:
    """The prewarm owns only the restart it started, so a failed step stops only that.

    The two calls here are the two statements the first delta runs, in order.
    When a handoff owed the media worker back, the release schedules that
    give-back and the prewarm then finds a restart already in flight. Owning it
    would mean cancelling somebody else's obligation on the way out - and the
    release has already cleared the flag that would re-arm it, so nothing would
    bring the media worker back at all.
    """
    start_entered = asyncio.Event()
    start_release = asyncio.Event()
    start_finished = False

    async def start_media() -> None:
        nonlocal start_finished
        start_entered.set()
        await start_release.wait()
        start_finished = True

    next_step = SimpleNamespace(operation="text_to_image", status="queued")
    job = SimpleNamespace(status="failed")
    orchestrator, _processes, session_factory = _plan_prewarm_orchestrator(
        next_step, start_media=start_media, job=job
    )

    # A completed media handoff owed the worker back and deferred it until chat
    # went quiet, which is the state the first delta arrives in.
    orchestrator._media_restart_after_chat_activity = True
    with session_factory() as session:
        orchestrator._arm_step_prewarm(session, _ordered_text_run())

    orchestrator._release_deferred_media_restart()
    give_back = orchestrator._media_restart_task
    assert give_back is not None, "the deferred give-back was never released"
    orchestrator._begin_step_prewarm()
    await start_entered.wait()

    assert orchestrator._step_prewarm_task is None, (
        "the prewarm adopted the give-back it did not create"
    )

    await orchestrator._settle_step_prewarm("job-current")

    assert not give_back.cancelled(), "a failing step cancelled the give-back"
    assert orchestrator._media_restart_task is give_back
    assert orchestrator._media_restart_after_chat_activity is False

    start_release.set()
    await give_back
    assert start_finished is True, "the media worker was never brought back"


async def test_external_media_handoff_only_resumes_chat() -> None:
    media = WorkerStatus(
        name="media",
        state="ready",
        managed=False,
        running=True,
        pid=22,
    )
    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        stop=AsyncMock(),
        start_media=AsyncMock(),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    processes.stop.assert_not_awaited()
    processes.start_media.assert_not_awaited()
    resume.assert_awaited_once_with("profile-chat")


async def test_chat_planner_falls_back_during_media_handoff() -> None:
    ready = WorkerStatus(
        name="chat",
        state="ready",
        managed=True,
        running=True,
        pid=12,
        profile_id="profile-selected",
    )
    processes = SimpleNamespace(
        settings=SimpleNamespace(
            llama_executable=Path("llama-server"),
            worker_startup_seconds=60,
        ),
        statuses=Mock(return_value=[ready]),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(chat_engine="llama.cpp")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    orchestrator._chat_planner_ready.clear()

    assert await orchestrator._chat_planner_available() is False
    orchestrator._chat_planner_ready.set()
    assert await orchestrator._chat_planner_available() is True


@pytest.mark.parametrize("cancelled", [False, True])
async def test_vision_bridge_restores_the_text_profile_after_completion_or_cancellation(
    cancelled: bool,
) -> None:
    run = SimpleNamespace(
        id="run-vision",
        chat_id="chat-vision",
        user_message_id="message-vision",
        profile_id="profile-text",
        vision_profile_id="profile-vision",
        standalone_prompt="What is visible?",
    )
    text_profile = SimpleNamespace(id="profile-text", model_install_id="install-text")
    vision_profile = SimpleNamespace(id="profile-vision", model_install_id="install-vision")
    text_install = SimpleNamespace(id="install-text")
    vision_install = SimpleNamespace(id="install-vision")

    class FakeSession:
        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (ModelProfile, "profile-text"): text_profile,
                (ModelProfile, "profile-vision"): vision_profile,
                (ModelInstall, "install-text"): text_install,
                (ModelInstall, "install-vision"): vision_install,
            }.get((model, identity))

        def expunge(self, _value) -> None:  # type: ignore[no-untyped-def]
            return None

        def scalar(self, _statement):  # type: ignore[no-untyped-def]
            return "job-vision"

        def in_transaction(self) -> bool:
            return False

    async def stream(_request):  # type: ignore[no-untyped-def]
        if cancelled:
            raise asyncio.CancelledError
        yield ChatEvent(type="delta", text="A green apple.")
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    processes = SimpleNamespace(
        settings=SimpleNamespace(),
        runtimes=None,
        load_chat=AsyncMock(),
        stop=AsyncMock(),
    )
    engines = SimpleNamespace(
        settings=SimpleNamespace(vision_bridge_max_tokens=128),
        chat_capabilities=AsyncMock(
            return_value=SimpleNamespace(input_modalities=["text", "image"])
        ),
        chat=SimpleNamespace(stream=stream),
    )
    orchestrator = ConversationOrchestrator(
        engines=engines,
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    orchestrator._set_chat_phase = AsyncMock()  # type: ignore[method-assign]
    orchestrator._attach_visual_context = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            [{"role": "user", "content": [{"type": "text", "text": "Question"}]}],
            {
                "available": True,
                "images_included": 1,
                "artifact_ids": ["sha256:image"],
                "visual_contents_inspected": True,
            },
        )
    )

    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await orchestrator._bridge_visual_context(
                _TEST_CLAIM,
                FakeSession(),  # type: ignore[arg-type]
                run,  # type: ignore[arg-type]
                [SimpleNamespace(id="sha256:image")],  # type: ignore[list-item]
            )
    else:
        observation, metadata = await orchestrator._bridge_visual_context(
            _TEST_CLAIM,
            FakeSession(),  # type: ignore[arg-type]
            run,  # type: ignore[arg-type]
            [SimpleNamespace(id="sha256:image")],  # type: ignore[list-item]
        )
        assert observation == "A green apple."
        assert metadata["completion"]["finish_reason"] == "stop"

    assert [call.args[0].id for call in processes.load_chat.await_args_list] == [
        "profile-vision",
        "profile-text",
    ]


def test_media_progress_preserves_the_latest_preview() -> None:
    message = Message(chat_id="chat-1", role="assistant", status="pending")
    message.parts = [
        MessagePart(
            position=0,
            type="progress",
            text="Preview",
            metadata_json={"progress": 0.4, "phase": "preview"},
        ),
        MessagePart(
            position=1,
            type="image",
            artifact_id="sha256:preview",
            metadata_json={"preview": True},
        ),
    ]

    parts = ConversationOrchestrator._media_progress_parts(
        message,
        MediaEvent(type="progress", progress=0.5, phase="sampling"),
    )

    assert [(part.position, part.type) for part in parts] == [(0, "progress"), (1, "image")]
    assert parts[0].text == "Sampling"
    assert parts[0].metadata_json == {
        "progress": 0.5,
        "phase": "sampling",
        "indeterminate": False,
    }
    assert parts[1].artifact_id == "sha256:preview"
    assert parts[1].metadata_json == {"preview": True}


def test_indeterminate_media_phase_is_explicit() -> None:
    message = Message(chat_id="chat-1", role="assistant", status="pending")

    parts = ConversationOrchestrator._media_progress_parts(
        message,
        MediaEvent(
            type="progress",
            phase="Staging media inputs",
            data={"indeterminate": True},
        ),
    )

    assert parts[0].metadata_json == {
        "progress": 0,
        "phase": "Staging media inputs",
        "indeterminate": True,
    }


def test_chat_progress_is_removed_without_discarding_text() -> None:
    text = MessagePart(position=0, type="text", text="Partial response")
    progress = MessagePart(
        position=1,
        type="progress",
        text="Waiting for first token",
        metadata_json={"activity": "chat", "phase": "waiting for first token"},
    )
    message = Message(
        chat_id="chat-1",
        role="assistant",
        status="pending",
        parts=[text, progress],
    )

    ConversationOrchestrator._remove_chat_progress(message)

    assert message.parts == [text]


async def test_chat_phase_advances_when_the_first_token_arrives() -> None:
    """The phase used to say "waiting" for the whole generation.

    It was set once before the stream and never updated, so a long answer that
    was streaming perfectly well read as a stall - which is the same failure the
    setup work was about: the interface saying nothing is happening while
    something is.
    """
    from local_lm.adapters.base import ChatEvent

    async def stream(_request):  # type: ignore[no-untyped-def]
        yield ChatEvent(type="delta", text="Hello", data={})
        yield ChatEvent(type="delta", text=" again", data={})
        yield ChatEvent(type="error", text="", data={"error": "stop here"})

    run = SimpleNamespace(
        id="run-phase",
        assistant_message_id="assistant-phase",
        provenance_json={},
        # The turn reads its chat to learn whether web access was granted.
        # This fake session returns nothing for a Chat, so the gate stays shut.
        chat_id="chat-phase",
    )

    class FakeSession:
        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return run if model is Run else None

        def commit(self):  # type: ignore[no-untyped-def]
            return None

        def rollback(self):  # type: ignore[no-untyped-def]
            return None

        def execute(self, _statement):  # type: ignore[no-untyped-def]
            # The ownership probe and the claim-bound writes are conditional
            # UPDATEs; one owned row answers them.
            return SimpleNamespace(rowcount=1)

        def in_transaction(self) -> bool:
            # The owned commit asserts ownership inside the transaction it
            # commits; this fake is always mid-transaction.
            return True

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_exc):  # type: ignore[no-untyped-def]
            return False

    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(chat=SimpleNamespace(stream=stream), settings=SimpleNamespace()),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=Mock(),
        processes=SimpleNamespace(runtimes=None),
    )
    phases = AsyncMock()
    orchestrator._set_chat_phase = phases  # type: ignore[method-assign]
    orchestrator._ensure_chat_worker = AsyncMock(return_value=None)  # type: ignore[method-assign]
    orchestrator.session_factory = FakeSession  # type: ignore[assignment]
    orchestrator._prepare_chat_context = AsyncMock(  # type: ignore[method-assign]
        return_value=([], {}, {}, False)
    )
    orchestrator._persist_streamed_text = Mock()  # type: ignore[method-assign]
    orchestrator._release_deferred_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._claim_still_owns = Mock(return_value=True)  # type: ignore[method-assign]
    orchestrator._begin_step_prewarm = Mock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await orchestrator._execute_chat(
            "job-phase", "run-phase", SimpleNamespace(token="claim-test", attempt=1)
        )

    labels = [call.args[2] for call in phases.await_args_list]
    assert "Waiting for first token" in labels
    assert "Writing the response" in labels
    assert labels.index("Writing the response") > labels.index("Waiting for first token")
    # Announced once, not per token: each phase change commits.
    assert labels.count("Writing the response") == 1


def _queued_media_orchestrator(
    peeks: list[object],
    next_run: object,
) -> tuple[ConversationOrchestrator, SimpleNamespace]:
    """A handoff whose scheduler is about to hand back another job.

    `peeks` is consumed one entry per handoff, so a test can express a run of
    consecutive jobs without reaching back into the scheduler between them.
    """

    media = WorkerStatus(name="media", state="ready", managed=True, running=True, pid=22)

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Run and identity == "run-next":
                return next_run
            return None

    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        stop=AsyncMock(),
        start_media=AsyncMock(),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=SimpleNamespace(peek_next_eligible_job=Mock(side_effect=list(peeks))),
        processes=processes,
        session_factory=FakeSession,
    )
    return orchestrator, processes


async def test_a_queued_image_keeps_the_media_worker_and_leaves_chat_down() -> None:
    """The recycle is skipped when the worker is about to be needed again.

    A queue of images otherwise pays a full cold start of BOTH workers per
    image: this handoff stops the media worker that the next job needs, and
    reloads a chat model that the next job's own handoff immediately unloads.
    """

    orchestrator, processes = _queued_media_orchestrator(
        [("job-next", "run-next")],
        SimpleNamespace(operation="text_to_image", profile_id=None),
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    processes.stop.assert_not_awaited()
    resume.assert_not_awaited()
    assert orchestrator._media_restart_task is None
    assert orchestrator._media_restart_after_chat_activity is False


async def test_skipping_the_resume_releases_the_chat_readiness_latch() -> None:
    """The latch marks a handoff in flight, and this handoff is over.

    Leaving it held would outlast the media run: `_ensure_chat_worker` loads the
    model for a later text job without touching the latch, so the planner would
    report itself unavailable against a chat worker that is ready.
    """

    orchestrator, _ = _queued_media_orchestrator(
        [("job-next", "run-next")],
        SimpleNamespace(operation="image_to_video", profile_id=None),
    )
    orchestrator._resume_chat_worker = AsyncMock()  # type: ignore[method-assign]
    orchestrator._chat_planner_ready.clear()

    await orchestrator._complete_media_handoff("profile-chat")

    assert orchestrator._chat_planner_ready.is_set()


async def test_a_recycle_whose_stop_fails_leaves_chat_down_and_the_profile_owed() -> None:
    """The stop is what makes the room, so a failed stop must not be followed
    by the load it was making room for.

    A false `recycle_managed_media` meant two unrelated things - there was
    nothing to recycle, and the recycle failed - and both were answered by
    resuming chat. Only the first wants that. On the second the worker may
    still hold the allocations the chat model would be loaded beside, which is
    the contention the recycle exists to prevent.

    The profile stays on the books so a later handoff still owes the restore,
    and the readiness latch is released for the reason the queued-media skip
    records: this handoff is over either way.
    """

    orchestrator, processes = _queued_media_orchestrator(
        [None],
        SimpleNamespace(operation="text", profile_id=None),
    )
    processes.stop = AsyncMock(side_effect=RuntimeError("the media worker would not stop"))
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._chat_planner_ready.clear()

    await orchestrator._complete_media_handoff("profile-chat")

    processes.stop.assert_awaited_once_with("media")
    resume.assert_not_awaited()
    processes.start_media.assert_not_awaited()
    assert orchestrator._displaced_chat_profile_id == "profile-chat"
    assert orchestrator._chat_planner_ready.is_set()
    assert orchestrator._media_restart_task is None
    assert orchestrator._media_restart_after_chat_activity is False


async def test_nothing_to_recycle_still_restores_the_chat_model() -> None:
    """The other half of the split, held in place.

    Separating a failed recycle from an absent one must not make an absent one
    behave like a failure. An unmanaged media worker holds nothing this handoff
    needs to release, so chat comes back and nothing is owed.
    """

    orchestrator, processes = _queued_media_orchestrator(
        [None],
        SimpleNamespace(operation="text", profile_id=None),
    )
    processes.statuses = Mock(
        return_value=[
            WorkerStatus(name="media", state="ready", managed=False, running=True, pid=22)
        ]
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    processes.stop.assert_not_awaited()
    resume.assert_awaited_once_with("profile-chat")
    assert orchestrator._displaced_chat_profile_id is None


async def test_a_failure_choosing_the_target_still_leaves_the_profile_owed() -> None:
    """The obligation has to outlive the method that discovers it.

    `_prepare_device_handoff` hands the displaced profile back as a return
    value, so until this method writes it down the only record of it is a local
    in the caller's frame. `_handoff_chat_target` guards its scheduler peek and
    then opens a session and reads two rows unguarded, so a database failure
    there ends this method - and used to end it before the books were touched,
    leaving the chat model down with nothing that knows it is owed.

    The failure is put in that exact window: after the peek, inside the read.
    """

    orchestrator, processes = _queued_media_orchestrator(
        [("job-next", "run-next")],
        SimpleNamespace(operation="text", profile_id=None),
    )

    def unreadable_session() -> object:
        raise RuntimeError("the database would not answer during the handoff")

    orchestrator.session_factory = unreadable_session  # type: ignore[method-assign]
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="would not answer"):
        await orchestrator._complete_media_handoff("profile-chat")

    assert orchestrator._displaced_chat_profile_id == "profile-chat", (
        "the displaced chat model was left with no owner"
    )
    processes.stop.assert_not_awaited()
    resume.assert_not_awaited()


async def test_a_handoff_that_restores_chat_owes_nothing_afterwards() -> None:
    """The other half: an obligation recorded up front must still be paid off.

    Making the debt the default is only safe if every path that restores chat
    clears it. A handoff that leaves it standing would send a later one to
    reload a model that is already up.
    """

    orchestrator, processes = _queued_media_orchestrator(
        [None],
        SimpleNamespace(operation="text", profile_id=None),
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    processes.stop.assert_awaited_once_with("media")
    resume.assert_awaited_once_with("profile-chat")
    assert orchestrator._displaced_chat_profile_id is None


async def test_a_later_image_in_the_run_still_reaches_the_terminal_handoff() -> None:
    """Only the first image of a run is handed a profile to restore.

    `_prepare_device_handoff` returns None as soon as chat is down, so every
    image after the first - the last one included - passes nothing to the
    handoff. If the run did not remember what it displaced, the final image
    would skip the handoff entirely and chat would stay unloaded until some
    later text execution happened to load it.
    """

    orchestrator, processes = _queued_media_orchestrator(
        [("job-next", "run-next"), None],
        SimpleNamespace(operation="text_to_image", profile_id=None),
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    first = orchestrator._pending_chat_restore("profile-chat")
    assert first == "profile-chat"
    await orchestrator._complete_media_handoff(first)
    processes.stop.assert_not_awaited()

    # What the dispatch computes for every later image in the run.
    last = orchestrator._pending_chat_restore(None)
    assert last == "profile-chat"
    await orchestrator._complete_media_handoff(last)
    if orchestrator._media_restart_task:
        await orchestrator._media_restart_task

    processes.stop.assert_awaited_once_with("media")
    resume.assert_awaited_once_with("profile-chat")
    assert orchestrator._pending_chat_restore(None) is None


async def test_a_run_that_displaced_nothing_owes_nothing() -> None:
    """The remembered profile is an obligation, so it must not be invented."""

    orchestrator, processes = _queued_media_orchestrator(
        [None],
        SimpleNamespace(operation="text_to_image", profile_id=None),
    )

    assert orchestrator._pending_chat_restore(None) is None
    processes.stop.assert_not_awaited()


async def test_a_queued_text_job_still_recycles_before_its_model_loads() -> None:
    """Only another media job keeps the worker; text is why the recycle exists."""

    orchestrator, processes = _queued_media_orchestrator(
        [("job-next", "run-next")],
        SimpleNamespace(operation="text", profile_id="profile-next"),
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-previous")

    processes.stop.assert_awaited_once_with("media")
    resume.assert_awaited_once_with("profile-next")
