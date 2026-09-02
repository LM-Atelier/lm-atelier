from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from local_lm.adapters.base import ChatEvent, MediaEvent, estimate_chat_tokens
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_templates import COMFY_TEMPLATE_COMPILER_VERSION
from local_lm.config import Settings
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
from local_lm.orchestrator import (
    ConversationOrchestrator,
    _DeferredHandoff,
    _queued_workflow_activation,
)
from local_lm.schemas import EngineCapabilities, WorkerStatus
from local_lm.workflow_activations import WorkflowActivationLaunchScope


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
        scheduler=_mock_scheduler(),
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
        scheduler=_mock_scheduler(),
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
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(chat_engine="llama.cpp")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=_mock_scheduler(),
        processes=processes,
        session_factory=FakeSession,
    )

    result = await orchestrator._ensure_chat_worker("run-1")

    assert result == aligned
    processes.load_chat.assert_awaited_once_with(profile, install)


async def test_engine_cancel_runs_after_the_database_session_closes() -> None:
    job = SimpleNamespace(id="job-cancel", status="running", run_id="run-cancel")
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
        scheduler=SimpleNamespace(try_lease=_fake_try_lease, publish_job=AsyncMock()),
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
        scheduler=_mock_scheduler(),
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
        launch_scope_sha256=Mock(return_value=None),
    )
    _follow_media_lifecycle(processes)
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=_mock_scheduler(),
        processes=processes,
    )

    async def resume(profile_id: str) -> bool:
        assert profile_id == "profile-chat"
        order.append("resume chat")
        return True

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
        launch_scope_sha256=Mock(return_value=None),
    )
    _follow_media_lifecycle(processes)
    scheduler = SimpleNamespace(
        try_lease=_fake_try_lease,
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
    assert orchestrator._media_restart_intent == "job-next", (
        "the give-back must be bound to the exact peeked text job, or a "
        "stranger could later consume it into a broad start"
    )
    processes.start_media.assert_not_awaited()

    # Only the OWNER fires it; a different job leaves it alone.
    orchestrator._discharge_media_restart_intent("job-other", fire=True)
    assert orchestrator._media_restart_intent == "job-next"
    processes.start_media.assert_not_awaited()

    orchestrator._discharge_media_restart_intent("job-next", fire=True)
    if orchestrator._media_restart_task:
        await orchestrator._media_restart_task

    assert orchestrator._media_restart_intent is None
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
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=_mock_scheduler(),
        processes=processes,
    )

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._prepare_device_handoff("text_to_image")

    assert orchestrator._chat_planner_ready.is_set()


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
        scheduler=_mock_scheduler(),
        processes=SimpleNamespace(
            statuses=Mock(return_value=[media]),
            start_media=start_media,
        ),
    )
    phases = AsyncMock()
    orchestrator._set_media_phase = phases  # type: ignore[method-assign]

    await orchestrator._ensure_media_worker(job_id="job-1", run_id="run-1")

    assert [call.args[2] for call in phases.await_args_list] == [
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
        scheduler=_mock_scheduler(),
        processes=SimpleNamespace(
            statuses=Mock(return_value=[media]),
            start_media=start_media,
        ),
    )

    await orchestrator._ensure_media_worker(activation_scope=scope)

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
        scheduler=_mock_scheduler(),
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
        scheduler=_mock_scheduler(),
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
    running = {"media": False}

    async def start_media() -> None:
        nonlocal start_calls
        start_calls += 1
        restart_entered.set()
        await restart_release.wait()
        running["media"] = True

    def statuses() -> list[WorkerStatus]:
        return [
            WorkerStatus(
                name="media",
                state="ready" if running["media"] else "stopped",
                managed=True,
                running=running["media"],
                pid=22,
            )
        ]

    processes = SimpleNamespace(
        statuses=Mock(side_effect=statuses),
        start_media=start_media,
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=_mock_scheduler(),
        processes=processes,
    )
    orchestrator._schedule_media_restart()
    await restart_entered.wait()
    ensure_task = asyncio.create_task(orchestrator._ensure_media_worker())
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
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=_mock_scheduler(),
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
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=_mock_scheduler(),
        processes=processes,
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    processes.stop.assert_not_awaited()
    processes.start_media.assert_not_awaited()
    resume.assert_awaited_once_with("profile-chat")


async def test_a_failed_recycle_keeps_chat_out_until_the_recycle_is_paid() -> None:
    """A media stop that raises may have left the worker holding its
    allocations; the report of the failure is not permission to load the
    chat model beside them. Chat stays down, the whole obligation - the
    recycle and the restore - is retained, and the settlement pump owns it
    and pays the recycle before the restore."""

    media = WorkerStatus(
        name="media",
        state="ready",
        managed=True,
        running=True,
        pid=22,
    )
    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        stop=AsyncMock(side_effect=RuntimeError("recycle failed")),
        start_media=AsyncMock(),
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=_mock_scheduler(),
        processes=processes,
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    resume.assert_not_awaited()
    processes.start_media.assert_not_awaited()
    debt = orchestrator._deferred_handoff
    assert debt is not None and debt.profile_id == "profile-chat", (
        "a failed recycle restored chat beside a worker that may still hold its allocations"
    )
    assert debt.recycle_paid is False
    orchestrator._arm_settlement_retry.assert_called_once()


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
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(chat_engine="llama.cpp")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=_mock_scheduler(),
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
        launch_scope_sha256=Mock(return_value=None),
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
        scheduler=_mock_scheduler(),
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
                FakeSession(),  # type: ignore[arg-type]
                run,  # type: ignore[arg-type]
                [SimpleNamespace(id="sha256:image")],  # type: ignore[list-item]
            )
    else:
        observation, metadata = await orchestrator._bridge_visual_context(
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

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_exc):  # type: ignore[no-untyped-def]
            return False

    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(chat=SimpleNamespace(stream=stream), settings=SimpleNamespace()),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=_mock_scheduler(),
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
    orchestrator._discharge_media_restart_intent = Mock()  # type: ignore[method-assign]
    orchestrator._begin_step_prewarm = Mock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await orchestrator._execute_chat("job-phase", "run-phase")

    labels = [call.args[2] for call in phases.await_args_list]
    assert "Waiting for first token" in labels
    assert "Writing the response" in labels
    assert labels.index("Writing the response") > labels.index("Waiting for first token")
    # Announced once, not per token: each phase change commits.
    assert labels.count("Writing the response") == 1


def _mock_scheduler() -> Mock:
    """A scheduler that answers every call with a Mock and holds a free device:
    the give-back's restart takes the device lease at its effect, and a bare
    Mock cannot be entered as an async context."""

    scheduler = Mock()
    scheduler.try_lease = _fake_try_lease
    return scheduler


def _fake_try_lease(device_id: str = "primary"):  # type: ignore[no-untyped-def]  # noqa: ARG001
    """A free device: settlement acquires immediately, releases on exit."""

    class _Held:
        async def __aenter__(self):  # type: ignore[no-untyped-def]
            return True

        async def __aexit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

    return _Held()


def _follow_media_lifecycle(processes: SimpleNamespace) -> None:
    """Make a fake supervisor's media status follow its own stop and start.

    The production supervisor tears the worker record down on stop and
    records a new one on start, so a status read after either answers for
    the worker that exists now. A fake that keeps answering "running" after
    it was stopped describes a worker that no longer exists.
    """

    original = list(processes.statuses())
    others = [item for item in original if item.name != "media"]
    state = {"media": next((item for item in original if item.name == "media"), None)}
    inner_stop = processes.stop
    inner_start = processes.start_media

    async def stop(name: str) -> None:
        await inner_stop(name)
        if name == "media" and state["media"] is not None:
            state["media"] = WorkerStatus(
                name="media", state="stopped", managed=True, running=False
            )

    async def start_media(*args: object, **kwargs: object) -> None:
        await inner_start(*args, **kwargs)
        state["media"] = WorkerStatus(name="media", state="ready", managed=True, running=True)

    processes.stop = AsyncMock(side_effect=stop)
    processes.start_media = AsyncMock(side_effect=start_media)
    processes.statuses = Mock(
        side_effect=lambda: others + ([state["media"]] if state["media"] is not None else [])
    )


def _handoff_orchestrator(peek, next_run=None, media_running=True):  # type: ignore[no-untyped-def]
    """An orchestrator wired just far enough to drive the media handoff."""

    media = WorkerStatus(name="media", state="ready", managed=True, running=media_running)

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Run and next_run is not None and identity == "run-next":
                return next_run
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(all=lambda: [])

    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        stop=AsyncMock(),
        start_media=AsyncMock(),
        launch_scope_sha256=Mock(return_value=None),
    )
    _follow_media_lifecycle(processes)
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=SimpleNamespace(
            peek_next_eligible_job=Mock(return_value=peek),
            try_lease=_fake_try_lease,
        ),
        processes=processes,
        session_factory=FakeSession,
    )
    return orchestrator, processes


async def test_a_following_media_job_keeps_the_worker_and_leaves_chat_down() -> None:
    """A queue of images must not recycle between every pair of them.

    Both halves of the handoff are skipped together and that coupling is the
    design, not an optimisation. Skipping only the recycle would leave ComfyUI's
    retained allocations in place while a large chat model loads onto the same
    device, which is the paging failure the recycle exists to prevent - so
    skipping the halves separately trades one failure for the other.

    The displaced profile has to be remembered here because this is the only job
    that will ever see the chat worker running: every later job in the queue
    finds it already stopped.
    """

    next_run = SimpleNamespace(operation="text_to_image", profile_id=None)
    orchestrator, processes = _handoff_orchestrator(("job-next", "run-next"), next_run)
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    processes.stop.assert_not_awaited()
    resume.assert_not_awaited()
    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-chat", "job-next", scoped=False
    ), (
        "the displaced profile was not remembered - bound to the continuation it "
        "was deferred FOR - so nothing later in the queue can put the chat worker "
        "back, or a stranger's exit could consume it"
    )


async def test_the_displaced_profile_survives_to_the_next_job_in_the_queue() -> None:
    """Without this the chat worker never comes back, and nothing reports it.

    `_complete_media_handoff` runs only when `_prepare_device_handoff` returned a
    profile, and that returns None when the chat worker is not running - which is
    exactly the state the previous job deliberately left behind. So the second
    media job in a queue would skip the completion path entirely, and so would
    every job after it, and the profile would be lost with no error anywhere.
    """

    orchestrator, _processes = _handoff_orchestrator(None, media_running=False)
    orchestrator.processes = SimpleNamespace(  # type: ignore[assignment]
        statuses=Mock(
            return_value=[WorkerStatus(name="chat", state="stopped", managed=True, running=False)]
        ),
        settings=SimpleNamespace(auto_unload_chat_for_media=True),
    )
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    carried = await orchestrator._prepare_device_handoff("text_to_image")

    assert carried == "profile-owed", (
        "a job that found chat already down did not pick up the profile the "
        "previous job left owing, so the completion chain stops here"
    )


async def test_an_idle_queue_still_recycles_and_restores() -> None:
    """An idle queue still releases and restores.

    Releasing ComfyUI's allocations is most worth doing when nothing else is
    queued, and it costs nothing there. Reading "not text next" as
    "media is next" would stop an idle queue releasing anything - turning
    the release into the very paging failure it exists to prevent.

    An unclassifiable answer takes this same branch deliberately: deferring
    the media restart on a misread would leave the machine unrestored, so
    only a provably media-next queue defers.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")
    if orchestrator._media_restart_task:
        await orchestrator._media_restart_task

    processes.stop.assert_awaited_once()
    resume.assert_awaited_once_with("profile-chat")
    assert orchestrator._deferred_handoff is None
    assert orchestrator._media_restart_intent is None, (
        "an idle queue deferred the media restart, which is the behaviour a text "
        "job gets; unclassified and empty must keep scheduling it immediately"
    )


async def test_a_broken_chain_recycles_media_before_the_chat_model_loads() -> None:
    """The deferral owes a recycle, and a text job is where it comes due.

    While the chain holds, leaving the media worker up is the entire point. If
    it BREAKS - the peeked media job fails before dispatch, or a text job
    overtakes it - then a chat model is about to load onto a device where
    ComfyUI still holds the finished generation's allocations. That is the
    paging failure the recycle exists to prevent, so the recycle owed by the
    deferral has to be paid before the load.

    Discharging this in the dispatch `finally` would run AFTER `_execute_chat`
    has already loaded the model and would not recycle at all: it would
    reload the chat worker the text job had just loaded, killing and
    relaunching the server the user was mid-conversation with.

    The planner event matters as much as the recycle. `_prepare_device_handoff`
    cleared it before stopping chat and the deferral skipped the resume that
    sets it, so without this it stays cleared with no owner and one AUTO turn
    silently drops to the heuristic router.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)
    orchestrator._chat_planner_ready.clear()

    await orchestrator._release_media_device_for_chat("job-text")

    processes.stop.assert_awaited_once_with("media")
    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-owed", "job-text", recycle_paid=True, scoped=False
    ), (
        "the release must REBIND the debt to the takeover job, not clear it: "
        "the obligation survives until the chat load actually commits, and it "
        "must record the recycle that just committed"
    )
    assert orchestrator._chat_planner_ready.is_set(), (
        "the planner event was left cleared with no owner, so a ready chat "
        "worker reads as unavailable"
    )
    orchestrator._discharge_deferred_handoff("job-text")
    assert orchestrator._deferred_handoff is None
    (
        resume.assert_not_awaited(),
        (
            "the release reloaded the chat worker; the text job's own load is what "
            "should bring it back, on the profile that job actually wants"
        ),
    )

    # Idempotent: the ordinary path, with nothing deferred, must pay nothing.
    processes.stop.reset_mock()
    await orchestrator._release_media_device_for_chat("job-text")
    processes.stop.assert_not_awaited()


@pytest.mark.parametrize(
    ("scoped", "expected_restarts"),
    [(False, 1), (True, 0)],
    ids=["unscoped-gives-back", "scoped-stays-down"],
)
async def test_a_text_takeover_over_a_down_worker_gives_back_only_when_unscoped(
    scoped: bool, expected_restarts: int
) -> None:
    """The already-down text-takeover branch owes the same give-back settlement does.

    A partial or cancelled stop can leave the media worker DOWN with an
    unpaid obligation before a later text dispatch takes the lease. That
    dispatch's release sees no running worker and pays the recycle half
    trivially - but for an UNSCOPED context the give-back is still owed, and
    the successful text takeover discharges the paid debt through the
    dispatch finally. Unless the down branch arms the broad-restart intent
    exactly as the running branch does, the discharge clears the debt and
    the finally finds nothing to fire, leaving media down where
    `_settle_deferred_handoff` over the same obligation would have restarted
    it. A scoped context, by contrast, must schedule nothing.
    """

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._resume_chat_worker = AsyncMock()  # type: ignore[method-assign]
    restart = Mock()
    orchestrator._schedule_media_restart = restart  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=scoped)

    # The text takeover pays the recycle before its chat load...
    await orchestrator._release_media_device_for_chat("job-text")
    processes.stop.assert_not_awaited()
    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-owed", "job-text", recycle_paid=True, scoped=scoped
    )

    # ...then the chat takeover COMMITS: the debt discharges and the dispatch
    # finally fires the intent for a completed text job, exactly as _execute
    # sequences them.
    orchestrator._discharge_deferred_handoff("job-text")
    assert orchestrator._deferred_handoff is None
    orchestrator._discharge_media_restart_intent("job-text", fire=True, allow_vacated=True)

    assert restart.call_count == expected_restarts, (
        "an unscoped down obligation must schedule exactly one give-back and a scoped one none"
    )


async def test_an_empty_warm_media_worker_is_left_alone() -> None:
    """Only a DEFERRED handoff owes a recycle; a warm empty worker does not.

    Stopping media unconditionally before every chat load would tear down the
    worker the step prewarm deliberately starts empty, and an empty ComfyUI
    beside a chat model is fine - it defers model loads until a prompt executes.
    The flag is the discriminator: it means a generation finished and was
    deliberately not recycled.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    assert orchestrator._deferred_handoff is None

    await orchestrator._release_media_device_for_chat()

    processes.stop.assert_not_awaited()


async def test_the_media_release_happens_before_the_chat_job_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TEXT job with a deferral outstanding stops the media worker before
    its chat model loads: the release runs inside dispatch, ahead of
    `_execute_chat`, so the retained allocations are gone before the load."""

    order: list[str] = []
    job = SimpleNamespace(
        id="job-text",
        kind="chat",
        status="queued",
        queue_resource=None,
        queue_group="primary",
        queue_priority=0,
        started_at=None,
    )
    run = SimpleNamespace(
        id="run-text",
        operation="text",
        status="queued",
        started_at=None,
        work_plan_id=None,
        work_step_id=None,
        standalone_prompt="hello",
        chat_id="chat-1",
    )

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == "job-text":
                return job
            if model is Run and identity == "run-text":
                return run
            return None

        def commit(self) -> None:
            return None

        def scalar(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            # No verification row exists for this job; None is the answer.
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]

            return SimpleNamespace(all=lambda: [])

    @asynccontextmanager
    async def lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        yield

    media = WorkerStatus(name="media", state="ready", managed=True, running=True)

    async def record_stop(name: str) -> None:
        order.append(f"stop {name}")

    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        stop=record_stop,
        start_media=AsyncMock(),
        launch_scope_sha256=Mock(return_value=None),
    )
    _follow_media_lifecycle(processes)
    events = SimpleNamespace(publish=AsyncMock())
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=events,
        scheduler=SimpleNamespace(
            job_lease=lease,
            peek_next_eligible_job=Mock(return_value=None),
            try_lease=_fake_try_lease,
        ),
        processes=processes,
        session_factory=FakeSession,
    )
    orchestrator._resolve_step_inputs = Mock()  # type: ignore[method-assign]
    orchestrator._set_work_status = Mock()  # type: ignore[method-assign]
    orchestrator._arm_step_prewarm = Mock()  # type: ignore[method-assign]
    orchestrator._settle_step_prewarm = AsyncMock()  # type: ignore[method-assign]
    monkeypatch.setattr("local_lm.orchestrator.mark_setup_verification_running", Mock())

    async def record_chat(_job_id: str, _run_id: str) -> None:
        order.append("chat executed")

    orchestrator._execute_chat = record_chat  # type: ignore[method-assign]
    # A previous media job deferred its handoff: the worker is still up holding
    # that generation's allocations, and the chat profile is owed back.
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    await orchestrator._execute("job-text", "run-text")

    assert order == ["stop media", "chat executed"], (
        f"expected the media device to be released before the chat job ran, got {order}. "
        "An empty stop means the release is not wired into dispatch; a trailing "
        "stop means it runs after the chat model has already loaded."
    )
    assert orchestrator._deferred_handoff is None
    assert orchestrator._media_restart_intent is None, (
        "the give-back should have been discharged by the finally once the text job finished"
    )
    # A cleared intent is not the give-back; the restart is.
    if orchestrator._media_restart_task:
        await orchestrator._media_restart_task
    processes.start_media.assert_awaited_once()


@pytest.mark.parametrize(
    ("carried_scope", "expected_restarts"),
    [(True, 0), (None, 0), (False, 1)],
    ids=["scoped-stays-down", "unknown-fails-closed", "unscoped-gives-back"],
)
async def test_a_down_worker_takeover_through_dispatch_honors_the_carried_scope(
    monkeypatch: pytest.MonkeyPatch,
    carried_scope: bool | None,
    expected_restarts: int,
) -> None:
    """A text takeover over a down worker restarts media only for an
    unscoped carried scope.

    A dead waiter's broad intent is still armed and a scoped stop left the
    media worker down with its scope carried on the debt. The takeover's
    release finds no running worker, the chat load succeeds, and the
    dispatch finally discharges whatever intent remains. Scoped and unknown
    carried scopes leave nothing to fire and schedule no restart; an
    unscoped one schedules exactly one.
    """

    job = SimpleNamespace(
        id="job-text",
        kind="chat",
        status="queued",
        queue_resource=None,
        queue_group="primary",
        queue_priority=0,
        started_at=None,
    )
    run = SimpleNamespace(
        id="run-text",
        operation="text",
        status="queued",
        started_at=None,
        work_plan_id=None,
        work_step_id=None,
        standalone_prompt="hello",
        chat_id="chat-1",
    )

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == "job-text":
                return job
            if model is Run and identity == "run-text":
                return run
            return None

        def commit(self) -> None:
            return None

        def scalar(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(all=lambda: [])

    @asynccontextmanager
    async def lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        yield

    down = WorkerStatus(name="media", state="stopped", managed=True, running=False)
    processes = SimpleNamespace(
        statuses=Mock(return_value=[down]),
        stop=AsyncMock(),
        start_media=AsyncMock(),
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(
            job_lease=lease,
            peek_next_eligible_job=Mock(return_value=None),
            try_lease=_fake_try_lease,
        ),
        processes=processes,
        session_factory=FakeSession,
    )
    orchestrator._resolve_step_inputs = Mock()  # type: ignore[method-assign]
    orchestrator._set_work_status = Mock()  # type: ignore[method-assign]
    orchestrator._arm_step_prewarm = Mock()  # type: ignore[method-assign]
    orchestrator._settle_step_prewarm = AsyncMock()  # type: ignore[method-assign]
    orchestrator._execute_chat = AsyncMock()  # type: ignore[method-assign]
    restart = Mock()
    orchestrator._schedule_media_restart = restart  # type: ignore[method-assign]
    monkeypatch.setattr("local_lm.orchestrator.mark_setup_verification_running", Mock())

    # The stale broad intent of a waiter that can no longer pay, and the
    # debt a scoped stop left behind with its scope carried.
    orchestrator._media_restart_intent = "job-dead-waiter"
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=carried_scope)

    await orchestrator._execute("job-text", "run-text")

    processes.stop.assert_not_awaited()
    assert orchestrator._deferred_handoff is None, "the takeover did not discharge the debt"
    assert orchestrator._media_restart_intent is None, "an intent survived the dispatch finally"
    assert restart.call_count == expected_restarts, (
        "a scoped or unknown context must schedule no restart and an unscoped "
        "one exactly one; the stale broad intent must never be the one fired"
    )


async def test_a_cancelled_next_job_gives_both_workers_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deferral the queue never redeems must be given back, not erased.

    The peek describes one instant, not a promise. If the job it saw is
    cancelled before dispatch, `_execute` returns at the status check - before
    `_prepare_device_handoff` - so no handoff runs for it, and none runs for
    anything after it either.

    Stopping the media worker WITHOUT restoring anything behind it would be
    the worst of both: chat ALREADY down because the previous job put it
    down, media now down too, the owed profile erased, and on an idle queue
    nothing to bring either back.
    Settlement has to RESTORE, and in order: the recycle the deferral skipped is
    paid first - the worker is still up holding the finished generation's
    allocations, and loading the displaced chat model beside them is the paging
    failure the recycle exists to prevent - then chat comes back on the profile
    that was displaced, then media is scheduled to come back behind it.
    """

    order: list[str] = []
    job = SimpleNamespace(
        id="job-gone",
        kind="chat",
        status="cancelled",
        queue_resource=None,
        queue_group="primary",
        queue_priority=0,
        started_at=None,
    )
    run = SimpleNamespace(id="run-gone", operation="text", status="queued")

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == "job-gone":
                return job
            if model is Run and identity == "run-gone":
                return run
            return None

        def commit(self) -> None:
            return None

        def scalar(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]

            return SimpleNamespace(all=lambda: [])

    @asynccontextmanager
    async def lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        yield

    async def record_stop(name: str) -> None:
        order.append(f"stop {name}")

    processes = SimpleNamespace(
        statuses=Mock(
            return_value=[WorkerStatus(name="media", state="ready", managed=True, running=True)]
        ),
        stop=record_stop,
        start_media=AsyncMock(),
        launch_scope_sha256=Mock(return_value=None),
    )
    _follow_media_lifecycle(processes)
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(
            job_lease=lease,
            peek_next_eligible_job=Mock(return_value=None),
            try_lease=_fake_try_lease,
        ),
        processes=processes,
        session_factory=FakeSession,
    )

    async def record_resume(profile_id: str) -> bool:
        order.append(f"resume {profile_id}")
        return True

    orchestrator._resume_chat_worker = record_resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock(  # type: ignore[method-assign]
        side_effect=lambda: order.append("schedule media")
    )
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    await orchestrator._execute("job-gone", "run-gone")

    assert order == ["stop media", "resume profile-owed", "schedule media"], (
        f"expected the retained allocations recycled, then the owed chat profile "
        f"restored, then media scheduled behind it, got {order}. Chat restored "
        f"first means the model loads beside the allocations the recycle exists "
        f"to clear; no stop at all means the recycle debt was silently dropped."
    )
    assert orchestrator._deferred_handoff is None


async def test_cancellation_while_waiting_settles_the_deferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary cancellation window, which a terminal-status job does not model.

    A job is far more likely to be cancelled while waiting for the primary lease,
    or between the two event publications before the device handoff, than to be
    found already terminal. Both raise CancelledError past every return in the
    body, so the settlement has to live in the handler; anywhere else and these
    windows leave chat down and media holding a finished generation's
    allocations, with an outstanding debt nobody owns.
    """

    order: list[str] = []

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def commit(self) -> None:
            return None

    @asynccontextmanager
    async def cancelling_lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        # Exactly the shape the scheduler produces when a waiter is cancelled
        # before the lease is granted.
        raise asyncio.CancelledError
        yield  # pragma: no cover

    job = SimpleNamespace(
        id="job-wait",
        kind="chat",
        status="queued",
        queue_resource=None,
        queue_group="primary",
        queue_priority=0,
        started_at=None,
    )
    run = SimpleNamespace(id="run-wait", operation="text", status="queued")

    class LookupSession(FakeSession):
        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == "job-wait":
                return job
            if model is Run and identity == "run-wait":
                return run
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(all=lambda: [])

    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(
            job_lease=cancelling_lease,
            peek_next_eligible_job=Mock(return_value=None),
            try_lease=_fake_try_lease,
        ),
        processes=SimpleNamespace(
            statuses=Mock(return_value=[]), stop=AsyncMock(), start_media=AsyncMock()
        ),
        session_factory=LookupSession,
    )

    async def record_resume(profile_id: str) -> bool:
        order.append(f"resume {profile_id}")
        return True

    orchestrator._resume_chat_worker = record_resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock(  # type: ignore[method-assign]
        side_effect=lambda: order.append("schedule media")
    )
    orchestrator._mark_cancelled = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._execute("job-wait", "run-wait")

    assert order == ["resume profile-owed", "schedule media"], (
        f"cancellation while waiting for the lease left the deferral unsettled, got {order}"
    )
    assert orchestrator._deferred_handoff is None


async def test_shutdown_does_not_leave_a_deferral_for_the_next_process() -> None:
    """A debt outstanding at shutdown is owed to nobody.

    The process is going away and the workers with it, so carrying the flag
    forward would make a restarted orchestrator recycle a freshly started media
    worker on its first text job, for a generation that finished before the
    previous process exited.
    """

    orchestrator, _processes = _handoff_orchestrator(None)
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    await orchestrator.close()

    assert orchestrator._deferred_handoff is None


async def test_settlement_pays_the_recycle_before_restoring_chat() -> None:
    """The ORDER inside settlement is the guarantee, not just its contents.

    A deferral deliberately leaves the media worker running with the finished
    generation's allocations retained, and settlement is the case where no chat
    job is coming to pay that recycle through `_release_media_device_for_chat`.
    Restoring chat first loads the displaced model beside those allocations -
    the same paging failure the recycle exists to prevent, relocated onto the
    recovery path where nothing would ever report it.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    order: list[str] = []

    async def record_stop(name: str) -> None:
        order.append(f"stop {name}")

    processes.stop = record_stop
    _follow_media_lifecycle(processes)

    async def record_resume(profile_id: str) -> bool:
        order.append(f"resume {profile_id}")
        return True

    orchestrator._resume_chat_worker = record_resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock(  # type: ignore[method-assign]
        side_effect=lambda: order.append("schedule media")
    )
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    await orchestrator._settle_deferred_handoff()

    assert order == ["stop media", "resume profile-owed", "schedule media"], (
        f"expected recycle, then restore, then restart, got {order}"
    )
    assert orchestrator._deferred_handoff is None

    # Idempotent: a second settlement finds the flag taken and pays nothing.
    order.clear()
    await orchestrator._settle_deferred_handoff()
    assert order == []


async def test_settlement_preserves_an_activation_scoped_launch() -> None:
    """A scoped worker is recycled but not broadly restarted.

    The same rule `_complete_media_handoff` applies: a broad empty restart
    would expose dependencies outside the activation that just ran, so after
    the recycle the next contract-backed media step revalidates and starts its
    own exact scope instead of inheriting a wide one from the recovery path.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    processes.launch_scope_sha256 = Mock(return_value="digest")
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    await orchestrator._settle_deferred_handoff()

    processes.stop.assert_awaited_once_with("media")
    resume.assert_awaited_once_with("profile-owed")
    orchestrator._schedule_media_restart.assert_not_called()


@pytest.mark.parametrize("failure", [asyncio.CancelledError, RuntimeError])
async def test_a_failed_chat_stop_records_the_displaced_profile_as_the_debt(
    failure: type[BaseException],
) -> None:
    """A partial stop must not lose the profile it displaced.

    `processes.stop` awaits a live process: it can raise after
    the worker is already down, or be cancelled midway. At that moment the
    displaced profile exists only in a local variable of a frame that is about
    to unwind, and the caller's `resume_chat_profile` was never assigned. So
    preparation records it as the deferral debt before re-raising - the
    dispatch finally has not seen its flag flip, and its settlement is the one
    place that knows how to restore a recorded debt.
    """

    chat = WorkerStatus(
        name="chat", state="ready", managed=True, running=True, profile_id="profile-chat"
    )
    processes = SimpleNamespace(
        statuses=Mock(return_value=[chat]),
        stop=AsyncMock(side_effect=failure),
        settings=SimpleNamespace(auto_unload_chat_for_media=True),
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=SimpleNamespace(
            peek_next_eligible_job=Mock(return_value=None),
            try_lease=_fake_try_lease,
        ),
        processes=processes,
        session_factory=Mock(),
    )

    with pytest.raises(failure):
        await orchestrator._prepare_device_handoff("text_to_image")

    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-chat", None, recycle_paid=False, scoped=False
    ), (
        "the displaced profile was dropped on the floor; nothing anywhere "
        "remembers what the chat worker was running"
    )
    assert orchestrator._chat_planner_ready.is_set(), (
        "the planner event was left cleared with no owner"
    )


async def test_a_failure_inside_preparation_still_settles_the_recorded_debt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure inside handoff preparation must still reach settlement.

    If `handoff_prepared` flips before `_prepare_device_handoff` is awaited,
    a failure inside it suppresses the settlement finally entirely:
    chat stopped or half-stopped, the displaced profile lost with the unwound
    frame, and on an idle queue both workers down with nobody left to notice.

    This drives `_execute` itself for a MEDIA job whose chat stop fails:
    preparation records the displaced profile as the deferral debt, the flag
    never flips, and settlement pays the recycle first, restores chat on the
    recorded profile, then schedules the media worker back.
    """

    order: list[str] = []
    job = SimpleNamespace(
        id="job-media",
        kind="media",
        status="queued",
        queue_resource=None,
        queue_group="primary",
        queue_priority=0,
        started_at=None,
    )
    run = SimpleNamespace(
        id="run-media",
        operation="text_to_image",
        status="queued",
        started_at=None,
        work_plan_id=None,
        work_step_id=None,
        standalone_prompt="a picture",
        chat_id="chat-1",
    )

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == "job-media":
                return job
            if model is Run and identity == "run-media":
                return run
            return None

        def commit(self) -> None:
            return None

        def scalar(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]

            return SimpleNamespace(all=lambda: [])

    @asynccontextmanager
    async def lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        yield

    chat = WorkerStatus(
        name="chat", state="ready", managed=True, running=True, profile_id="profile-chat"
    )
    media = WorkerStatus(name="media", state="ready", managed=True, running=True)

    async def failing_stop(name: str) -> None:
        if name == "chat":
            raise RuntimeError("the stop failed partway")
        order.append(f"stop {name}")

    processes = SimpleNamespace(
        statuses=Mock(return_value=[chat, media]),
        stop=failing_stop,
        start_media=AsyncMock(),
        settings=SimpleNamespace(auto_unload_chat_for_media=True),
        launch_scope_sha256=Mock(return_value=None),
    )
    _follow_media_lifecycle(processes)
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(
            job_lease=lease,
            peek_next_eligible_job=Mock(return_value=None),
            try_lease=_fake_try_lease,
        ),
        processes=processes,
        session_factory=FakeSession,
    )
    orchestrator._resolve_step_inputs = Mock()  # type: ignore[method-assign]
    orchestrator._set_work_status = Mock()  # type: ignore[method-assign]
    orchestrator._arm_step_prewarm = Mock()  # type: ignore[method-assign]
    orchestrator._set_media_phase = AsyncMock()  # type: ignore[method-assign]
    orchestrator._fail = AsyncMock()  # type: ignore[method-assign]
    monkeypatch.setattr("local_lm.orchestrator.mark_setup_verification_running", Mock())

    async def record_resume(profile_id: str) -> bool:
        order.append(f"resume {profile_id}")
        return True

    orchestrator._resume_chat_worker = record_resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock(  # type: ignore[method-assign]
        side_effect=lambda: order.append("schedule media")
    )

    await orchestrator._execute("job-media", "run-media")

    assert order == ["stop media", "resume profile-chat", "schedule media"], (
        f"expected the recorded debt settled as recycle, restore, restart, got {order}. "
        f"An empty list means the failure inside preparation suppressed settlement - "
        f"the exact suppression the flag placement exists to prevent."
    )
    assert orchestrator._deferred_handoff is None
    orchestrator._fail.assert_awaited_once()
    assert orchestrator._chat_planner_ready.is_set()


async def test_cancellation_between_the_publications_settles_the_deferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation between the lease grant and the device handoff.

    After the lease is granted, `run.created` has been published and
    `plan.selected` is the next await: a cancellation landing exactly
    there finds a dispatch that owns the device and has settled nothing
    yet. The cancel is driven into the second publication itself, and
    the deferral must still settle.
    """

    order: list[str] = []
    published: list[str] = []
    job = SimpleNamespace(
        id="job-text",
        kind="chat",
        status="queued",
        queue_resource=None,
        queue_group="primary",
        queue_priority=0,
        started_at=None,
    )
    run = SimpleNamespace(
        id="run-text",
        operation="text",
        status="queued",
        started_at=None,
        work_plan_id=None,
        work_step_id=None,
        standalone_prompt="hello",
        chat_id="chat-1",
    )

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == "job-text":
                return job
            if model is Run and identity == "run-text":
                return run
            return None

        def commit(self) -> None:
            return None

        def scalar(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]

            return SimpleNamespace(all=lambda: [])

    @asynccontextmanager
    async def lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        yield

    async def publish(topic: str, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        published.append(topic)
        if topic == "plan.selected":
            raise asyncio.CancelledError

    async def record_stop(name: str) -> None:
        order.append(f"stop {name}")

    processes = SimpleNamespace(
        statuses=Mock(
            return_value=[WorkerStatus(name="media", state="ready", managed=True, running=True)]
        ),
        stop=record_stop,
        start_media=AsyncMock(),
        launch_scope_sha256=Mock(return_value=None),
    )
    _follow_media_lifecycle(processes)
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=SimpleNamespace(publish=publish),
        scheduler=SimpleNamespace(
            job_lease=lease,
            peek_next_eligible_job=Mock(return_value=None),
            try_lease=_fake_try_lease,
        ),
        processes=processes,
        session_factory=FakeSession,
    )
    orchestrator._resolve_step_inputs = Mock()  # type: ignore[method-assign]
    orchestrator._set_work_status = Mock()  # type: ignore[method-assign]
    orchestrator._arm_step_prewarm = Mock()  # type: ignore[method-assign]
    orchestrator._mark_cancelled = Mock()  # type: ignore[method-assign]
    monkeypatch.setattr("local_lm.orchestrator.mark_setup_verification_running", Mock())

    async def record_resume(profile_id: str) -> bool:
        order.append(f"resume {profile_id}")
        return True

    orchestrator._resume_chat_worker = record_resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock(  # type: ignore[method-assign]
        side_effect=lambda: order.append("schedule media")
    )
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._execute("job-text", "run-text")

    assert published == ["run.created", "plan.selected"], (
        f"the cancel was meant to land inside the second publication, got {published}"
    )
    assert order == ["stop media", "resume profile-owed", "schedule media"], (
        f"cancellation between the publications left the deferral unsettled, got {order}"
    )
    assert orchestrator._deferred_handoff is None


async def test_a_cancelled_settlement_restores_the_debt_it_could_not_pay() -> None:
    """Taking the debt first is for concurrency; losing it on failure is a bug.

    Settlement awaits a media stop and a chat load, and either can be cancelled
    or fail. With the debt already erased, a cancellation mid-settlement leaves
    a worker down and nothing owed anywhere - the same stranded shape the
    settlement exists to prevent, created by the settlement itself. So the debt
    is taken first and RESTORED when paying it raises, and a later settlement
    completes the transition.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        side_effect=asyncio.CancelledError
    )
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    debt = _DeferredHandoff("profile-owed", None, scoped=False)
    orchestrator._deferred_handoff = debt

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._settle_deferred_handoff()

    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-owed", None, recycle_paid=True, scoped=False
    ), (
        "a cancelled settlement erased the debt it failed to pay - the record "
        "keeps the recycle payment that committed before the cancel"
    )
    orchestrator._schedule_media_restart.assert_not_called()

    # The obligation is still payable: a later settlement completes it.
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    await orchestrator._settle_deferred_handoff()
    resume.assert_awaited_once_with("profile-owed")
    assert orchestrator._deferred_handoff is None


async def test_a_failed_media_release_keeps_the_obligation_and_stops_the_takeover() -> None:
    """A stop failure is not permission to load chat beside the worker.

    The release raises so the text job fails honestly instead of running
    beside the retained allocations; the obligation stays on the books with
    the recycle unpaid, no give-back is armed for a stop that never
    happened, and the settlement pump owns the retry.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    debt = _DeferredHandoff("profile-owed", None, scoped=False)
    orchestrator._deferred_handoff = debt
    orchestrator._chat_planner_ready.clear()

    with pytest.raises(RuntimeError, match="could not be recycled"):
        await orchestrator._release_media_device_for_chat()

    retained = orchestrator._deferred_handoff
    assert retained is not None and retained.profile_id == "profile-owed", (
        "the media stop failed and the recycle obligation was erased with it"
    )
    assert retained.recycle_paid is False
    assert orchestrator._media_restart_intent is None, (
        "a give-back was armed for a stop that never happened"
    )
    orchestrator._arm_settlement_retry.assert_called_once()
    assert orchestrator._chat_planner_ready.is_set()


async def test_a_scoped_deferral_taken_over_by_text_gets_no_broad_restart() -> None:
    """The takeover path must not broadly restart a scoped worker.

    Completion and settlement both consult the launch scope before scheduling
    the worker back; a takeover arming the broad restart unconditionally
    is the one path around that consultation. A scoped worker is still recycled - that is the
    point of the release - but the give-back is left to the next
    contract-backed media step, which revalidates and starts its own exact
    scope rather than inheriting a broad one from the recovery path.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    processes.launch_scope_sha256 = Mock(return_value="digest")
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    await orchestrator._release_media_device_for_chat()

    processes.stop.assert_awaited_once_with("media")
    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-owed", None, recycle_paid=True, scoped=True
    ), "the rebound debt must survive the release until the takeover commits"
    assert orchestrator._media_restart_intent is None, (
        "a scoped launch was armed for a broad restart through the takeover path"
    )


async def test_a_strangers_cancellation_cannot_consume_anothers_debt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The debt is bound to the continuation it was deferred FOR.

    Every queued job runs its own dispatch task, and any of them can be
    cancelled while waiting for the lease. Such a task exits through the same
    finally that settles deferrals - and with a global unbound debt it would
    restore chat directly in front of the queued media job the deferral was
    taken for, which then immediately unloads it: recycle-thrash resurfacing
    through a bystander's cancellation.

    So: a distinct cancelled waiter leaves the debt alone while the bound
    continuation is still queued, and the continuation itself settles freely.
    """

    order: list[str] = []
    next_media_job = SimpleNamespace(id="job-media-next", status="queued")

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == "job-media-next":
                return next_media_job
            if model is Job and identity == "job-waiter":
                return SimpleNamespace(
                    id="job-waiter",
                    kind="chat",
                    status="queued",
                    queue_resource=None,
                    queue_group="primary",
                    queue_priority=0,
                    started_at=None,
                )
            if model is Run and identity == "run-waiter":
                return SimpleNamespace(id="run-waiter", operation="text", status="queued")
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(all=lambda: [])

        def commit(self) -> None:
            return None

    @asynccontextmanager
    async def cancelling_lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise asyncio.CancelledError
        yield  # pragma: no cover

    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(
            job_lease=cancelling_lease,
            peek_next_eligible_job=Mock(return_value=None),
            try_lease=_fake_try_lease,
        ),
        processes=SimpleNamespace(
            statuses=Mock(
                return_value=[WorkerStatus(name="media", state="ready", managed=True, running=True)]
            ),
            stop=AsyncMock(),
            start_media=AsyncMock(),
        ),
        session_factory=FakeSession,
    )

    async def record_resume(profile_id: str) -> bool:
        order.append(f"resume {profile_id}")
        return True

    orchestrator._resume_chat_worker = record_resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock(  # type: ignore[method-assign]
        side_effect=lambda: order.append("schedule media")
    )
    orchestrator._mark_cancelled = Mock()  # type: ignore[method-assign]
    debt = _DeferredHandoff("profile-owed", "job-media-next", scoped=False)
    orchestrator._deferred_handoff = debt

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._execute("job-waiter", "run-waiter")

    assert orchestrator._deferred_handoff == debt, (
        "a cancelled bystander consumed the debt bound to the queued media "
        "continuation, loading chat in front of the job that will unload it"
    )
    assert order == [], f"the bystander settled anyway: {order}"

    # The bound continuation itself settles without obstruction.
    await orchestrator._settle_deferred_handoff(job_id="job-media-next")
    assert order[0] == "resume profile-owed"
    assert orchestrator._deferred_handoff is None


async def test_a_dead_continuations_debt_is_settled_by_anyone() -> None:
    """Binding must vacate when the bound job can no longer pay.

    A continuation that was cancelled and reaped, or never dispatched again,
    would otherwise pin the debt forever: chat stays down on an idle queue
    precisely because the one job allowed to fix it no longer exists.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", "job-vanished", scoped=False)

    # The session knows no such job: the bound continuation is gone.
    await orchestrator._settle_deferred_handoff(job_id="job-someone-else")

    resume.assert_awaited_once_with("profile-owed")
    assert orchestrator._deferred_handoff is None


async def test_cancellation_after_preparation_leaves_a_durable_debt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suppression window: preparation succeeded, so the finally is silent.

    For a TEXT job, preparation returns immediately and the ownership flag
    flips, so the dispatch finally will never settle - by design, because the
    release inside the dispatch is what pays this path. If the release itself
    is cancelled mid-stop, the debt must survive the task's death through the
    release's own restore, or it is gone with nothing owed anywhere.
    """

    job = SimpleNamespace(
        id="job-text",
        kind="chat",
        status="queued",
        queue_resource=None,
        queue_group="primary",
        queue_priority=0,
        started_at=None,
    )
    run = SimpleNamespace(
        id="run-text",
        operation="text",
        status="queued",
        started_at=None,
        work_plan_id=None,
        work_step_id=None,
        standalone_prompt="hello",
        chat_id="chat-1",
    )

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == "job-text":
                return job
            if model is Run and identity == "run-text":
                return run
            return None

        def commit(self) -> None:
            return None

        def scalar(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]

            return SimpleNamespace(all=lambda: [])

    @asynccontextmanager
    async def lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        yield

    processes = SimpleNamespace(
        statuses=Mock(
            return_value=[WorkerStatus(name="media", state="ready", managed=True, running=True)]
        ),
        stop=AsyncMock(side_effect=asyncio.CancelledError),
        start_media=AsyncMock(),
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(
            job_lease=lease,
            peek_next_eligible_job=Mock(return_value=None),
            try_lease=_fake_try_lease,
        ),
        processes=processes,
        session_factory=FakeSession,
    )
    orchestrator._resolve_step_inputs = Mock()  # type: ignore[method-assign]
    orchestrator._set_work_status = Mock()  # type: ignore[method-assign]
    orchestrator._arm_step_prewarm = Mock()  # type: ignore[method-assign]
    orchestrator._settle_step_prewarm = AsyncMock()  # type: ignore[method-assign]
    orchestrator._execute_chat = AsyncMock()  # type: ignore[method-assign]
    orchestrator._mark_cancelled = Mock()  # type: ignore[method-assign]
    monkeypatch.setattr("local_lm.orchestrator.mark_setup_verification_running", Mock())
    debt = _DeferredHandoff("profile-owed", None, scoped=False)
    orchestrator._deferred_handoff = debt

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._execute("job-text", "run-text")

    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-owed", "job-text", recycle_paid=False, scoped=False
    ), (
        "cancellation inside the release erased the debt, and the suppressed "
        "dispatch finally means nothing else will ever restore it; the rebind "
        "to the takeover job is what the vacancy rule later frees"
    )
    orchestrator._execute_chat.assert_not_awaited()


async def test_a_cancelled_completion_records_what_it_was_about_to_restore() -> None:
    """Completion is the media job's own dispatch; its finally will not settle.

    Preparation succeeded long before, so a cancellation inside the completion
    tail - stopping the worker for the recycle, or reloading the chat model -
    dies in a task whose settlement is deliberately suppressed. The profile the
    completion was about to restore becomes the recorded debt, unbound, so the
    next settler of any kind may pay it.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        side_effect=asyncio.CancelledError
    )

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._complete_media_handoff("profile-chat")

    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-chat", None, recycle_paid=True, scoped=False
    ), "the cancelled completion recorded no debt; chat stays down with nothing owed anywhere"


def _text_dispatch_orchestrator(monkeypatch, order):  # type: ignore[no-untyped-def]
    """A TEXT dispatch with the chat execution path intact and only its leaf
    awaits replaced."""

    job = SimpleNamespace(
        id="job-text",
        kind="chat",
        status="queued",
        queue_resource=None,
        queue_group="primary",
        queue_priority=0,
        started_at=None,
    )
    run = SimpleNamespace(
        id="run-text",
        operation="text",
        status="queued",
        started_at=None,
        work_plan_id=None,
        work_step_id=None,
        standalone_prompt="hello",
        chat_id="chat-1",
    )

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == "job-text":
                return job
            if model is Run and identity == "run-text":
                return run
            return None

        def commit(self) -> None:
            return None

        def scalar(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]

            return SimpleNamespace(all=lambda: [])

    @asynccontextmanager
    async def lease(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        yield

    async def record_stop(name: str) -> None:
        order.append(f"stop {name}")

    processes = SimpleNamespace(
        statuses=Mock(
            return_value=[WorkerStatus(name="media", state="ready", managed=True, running=True)]
        ),
        stop=record_stop,
        start_media=AsyncMock(),
        launch_scope_sha256=Mock(return_value=None),
    )
    _follow_media_lifecycle(processes)
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(
            job_lease=lease,
            peek_next_eligible_job=Mock(return_value=None),
            try_lease=_fake_try_lease,
        ),
        processes=processes,
        session_factory=FakeSession,
    )
    orchestrator._resolve_step_inputs = Mock()  # type: ignore[method-assign]
    orchestrator._set_work_status = Mock()  # type: ignore[method-assign]
    orchestrator._arm_step_prewarm = Mock()  # type: ignore[method-assign]
    orchestrator._settle_step_prewarm = AsyncMock()  # type: ignore[method-assign]
    orchestrator._mark_cancelled = Mock()  # type: ignore[method-assign]
    monkeypatch.setattr("local_lm.orchestrator.mark_setup_verification_running", Mock())
    return orchestrator, processes


async def test_cancellation_at_the_chat_phase_write_keeps_the_debt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first await past the release is the phase write, and it can die.

    The release has already paid the recycle and `handoff_prepared` is true,
    so the outer settlement is suppressed. Only the debt REBOUND by the
    release remembers the displaced profile through this window; clearing at
    the release would forget it here, one await later.
    """

    order: list[str] = []
    orchestrator, processes = _text_dispatch_orchestrator(monkeypatch, order)
    orchestrator._set_chat_phase = AsyncMock(  # type: ignore[method-assign]
        side_effect=asyncio.CancelledError
    )
    orchestrator._ensure_chat_worker = AsyncMock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._execute("job-text", "run-text")

    assert order == ["stop media"], "the release should have paid the recycle first"
    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-owed", "job-text", recycle_paid=True, scoped=False
    ), "cancellation at the phase write forgot the displaced profile"
    assert orchestrator._media_restart_intent is None, (
        "a failed text path must drop its own intent, not leave it stale"
    )
    processes.start_media.assert_not_awaited()


async def test_cancellation_inside_the_real_ensure_path_keeps_the_debt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ensure/load await is the second barrier past the release."""

    order: list[str] = []
    orchestrator, _processes = _text_dispatch_orchestrator(monkeypatch, order)
    orchestrator._set_chat_phase = AsyncMock()  # type: ignore[method-assign]
    orchestrator._ensure_chat_worker = AsyncMock(  # type: ignore[method-assign]
        side_effect=asyncio.CancelledError
    )
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._execute("job-text", "run-text")

    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-owed", "job-text", recycle_paid=True, scoped=False
    ), "cancellation inside the ensure/load path forgot the displaced profile"


async def test_an_ordinary_stop_failure_retains_the_debt_and_schedules_nothing() -> None:
    """A swallowed recycle failure must not read as payment.

    The stop's exception is logged, not raised, so no restoring handler ever
    sees it - the explicit payment outcome is what keeps the debt. And the
    restart must NOT be scheduled: a start before the recycle is paid inverts
    the recycle -> restore -> restart order settlement guarantees.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    debt = _DeferredHandoff("profile-owed", None, scoped=False)
    orchestrator._deferred_handoff = debt

    await orchestrator._settle_deferred_handoff()

    assert orchestrator._deferred_handoff == debt, (
        "a swallowed stop failure erased the debt with the recycle unpaid"
    )
    orchestrator._schedule_media_restart.assert_not_called()


async def test_a_reported_load_failure_through_the_real_helper_retains_the_debt() -> None:
    """Missing load inputs make the resume helper answer False.

    A session that knows no model profile is the missing-profile case the
    helper reports as False without raising, and a settlement must not
    count that answer as success.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    debt = _DeferredHandoff("profile-gone", None, scoped=False)
    orchestrator._deferred_handoff = debt

    await orchestrator._settle_deferred_handoff()

    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-gone", None, recycle_paid=True, scoped=False
    ), (
        "a resume that reported failure was counted as a restoration - and the "
        "retained record must carry the recycle payment that DID commit"
    )
    orchestrator._schedule_media_restart.assert_not_called()
    assert orchestrator._chat_planner_ready.is_set()
    # An idle queue is not a permanent stall: the delayed retry is armed.
    retry = orchestrator._settlement_retry_task
    assert retry is not None and not retry.done(), (
        "the retained debt has no recovery owner on an idle queue"
    )
    retry.cancel()
    with suppress(asyncio.CancelledError):
        await retry


async def test_a_live_dispatch_task_outranks_a_terminal_database_status() -> None:
    """COMPLETE is committed BEFORE the handoff completion runs.

    A media job writes COMPLETE, then awaits publications, and only then pays
    its handoff in the lease-held finally. A waiter cancelled during that
    window must not read the durable status as death and settle concurrently
    with the rightful owner. The in-process dispatch task is the signal that
    stays live through finalization.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]

    complete_job = SimpleNamespace(id="job-y", status="complete")

    class SessionWithJob:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if model is Job and identity == "job-y":
                return complete_job
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(all=lambda: [])

    orchestrator.session_factory = SessionWithJob  # type: ignore[assignment]
    debt = _DeferredHandoff("profile-owed", "job-y", scoped=False)
    orchestrator._deferred_handoff = debt
    # The owner's dispatch task is still finalizing its handoff.
    orchestrator._tasks["job-y"] = SimpleNamespace(done=lambda: False)  # type: ignore[assignment]

    await orchestrator._settle_deferred_handoff(job_id="job-waiter")

    assert orchestrator._deferred_handoff == debt, (
        "a stranger consumed the debt while its owner was mid-finalization, "
        "because COMPLETE was read as death"
    )
    resume.assert_not_awaited()

    # Once the owner's task is genuinely gone, the terminal status vacates
    # the binding and anyone may settle.
    orchestrator._tasks["job-y"] = SimpleNamespace(done=lambda: True)  # type: ignore[assignment]
    await orchestrator._settle_deferred_handoff(job_id="job-waiter")
    resume.assert_awaited_once_with("profile-owed")
    assert orchestrator._deferred_handoff is None


async def test_scoped_truth_supersedes_a_stale_broad_intent() -> None:
    """The cancelled-peeked-text, scoped-media, later-text sequence.

    A cancelled waiter's broad intent survives its death. The next SCOPED
    media completion must supersede it, so the text job after that cannot
    consume stale broad state into a start_media the scoped context never
    asked for.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.launch_scope_sha256 = Mock(return_value="digest")
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    # A broad intent left behind by a cancelled text waiter.
    orchestrator._media_restart_intent = "job-cancelled-waiter"

    await orchestrator._complete_media_handoff("profile-chat")

    assert orchestrator._media_restart_intent is None, (
        "the scoped completion left the stale broad intent alive"
    )
    # The later text job finds nothing to consume.
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._discharge_media_restart_intent("job-later-text", fire=True, allow_vacated=True)
    orchestrator._schedule_media_restart.assert_not_called()


async def test_scope_introspection_failure_reads_as_scoped() -> None:
    """Unknown scope must never earn a broad restart, anywhere.

    The helper fails closed on a raising or absent introspection, so the
    release does not arm an intent and the settlement schedules nothing
    broad - the next contract-backed step starts its own exact scope.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.launch_scope_sha256 = Mock(side_effect=RuntimeError("introspection broke"))
    assert orchestrator._media_launch_scoped() is True

    del processes.launch_scope_sha256
    assert orchestrator._media_launch_scoped() is True

    processes.stop = AsyncMock()
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    await orchestrator._settle_deferred_handoff()

    assert orchestrator._deferred_handoff is None, "payment itself must still complete"
    orchestrator._schedule_media_restart.assert_not_called()


async def test_a_vacated_owners_intent_is_fired_by_the_backstop() -> None:
    """Ownership must not strand the give-back when the owner died.

    A stale intent whose owner has no task and no live job may be fired by a
    completion backstop with allow_vacated - media comes back - while the hot
    path without allow_vacated leaves foreign intents alone entirely.
    """

    orchestrator, _processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._media_restart_intent = "job-dead"

    # Hot path, foreign owner: untouched, no database read taken.
    orchestrator._discharge_media_restart_intent("job-other", fire=True)
    assert orchestrator._media_restart_intent == "job-dead"
    orchestrator._schedule_media_restart.assert_not_called()

    # Backstop: the session knows no such job, so the owner has vacated
    # and the give-back fires.
    orchestrator._discharge_media_restart_intent("job-other", fire=True, allow_vacated=True)
    assert orchestrator._media_restart_intent is None
    orchestrator._schedule_media_restart.assert_called_once()


async def test_a_completion_resume_failure_records_the_debt() -> None:
    """Chat down with nothing owed is the one state nothing recovers from.

    The completion tail's resume can report failure without raising; the
    profile it failed to restore becomes the recorded debt so a later settler
    retries, instead of the failure being permanent because it was logged and
    forgotten.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    resume = AsyncMock(return_value=False)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-chat", None, recycle_paid=True, scoped=False
    ), "the reported resume failure recorded no debt"


async def test_a_text_takeover_over_a_failed_stop_loads_no_chat_and_keeps_the_recycle_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release cannot stop the media worker before the chat model loads.

    The worker may still hold its allocations, so the text job does not run
    beside them: it fails honestly, the rebound debt keeps the unpaid recycle
    on the books, and the pump owns the recovery.
    """

    order: list[str] = []
    orchestrator, processes = _text_dispatch_orchestrator(monkeypatch, order)

    async def failing_stop(name: str) -> None:
        raise RuntimeError("stop failed under the takeover")

    processes.stop = failing_stop
    orchestrator._set_chat_phase = AsyncMock()  # type: ignore[method-assign]
    orchestrator._ensure_chat_worker = AsyncMock()  # type: ignore[method-assign]
    orchestrator._execute_chat = AsyncMock()  # type: ignore[method-assign]
    orchestrator._fail = AsyncMock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    await orchestrator._execute("job-text", "run-text")

    orchestrator._ensure_chat_worker.assert_not_awaited()
    orchestrator._execute_chat.assert_not_awaited()
    orchestrator._fail.assert_awaited_once()
    assert orchestrator._fail.await_args.args[:2] == ("job-text", "run-text")
    debt = orchestrator._deferred_handoff
    assert debt is not None and debt.recycle_paid is False, (
        "the failed takeover erased the unpaid recycle"
    )
    retry = orchestrator._settlement_retry_task
    assert retry is not None and not retry.done(), "the retained obligation has no recovery owner"
    retry.cancel()
    with suppress(asyncio.CancelledError):
        await retry
    processes.start_media.assert_not_awaited()


async def test_an_active_media_dispatch_blocks_every_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The device belongs to the generation that is running on it.

    A debt - bound or not - must not authorize a bystander's exit to stop the
    active media worker or load chat beside it. While a running media job's
    dispatch task is live, settlement returns without consuming anything; the
    job's own completion handoff pays or supersedes what is owed.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]

    running_media = SimpleNamespace(id="job-media-live", status="running")

    class BusySession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(all=lambda: ["job-media-live"])

    orchestrator.session_factory = BusySession  # type: ignore[assignment]
    orchestrator._tasks["job-media-live"] = SimpleNamespace(done=lambda: False)  # type: ignore[assignment]
    debt = _DeferredHandoff(None, None, recycle_paid=False, scoped=False)
    orchestrator._deferred_handoff = debt

    await orchestrator._settle_deferred_handoff(job_id="job-cancelled-bystander")

    assert orchestrator._deferred_handoff == debt, (
        "a bystander consumed an obligation while an active generation owned the device"
    )
    processes.stop.assert_not_awaited()
    resume.assert_not_awaited()
    del running_media


async def test_a_dead_intent_owner_fires_the_give_back_from_task_teardown() -> None:
    """Owner death must not strand the media worker on an idle queue.

    The done-callback is the recovery owner: an intent whose owning task ended
    without discharging fires the give-back immediately, because the give-back
    is owed regardless of how the owner died.
    """

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._media_restart_intent = "job-dead-owner"
    finished = SimpleNamespace(cancelled=lambda: True, exception=lambda: None)

    orchestrator._task_done("job-dead-owner", finished)  # type: ignore[arg-type]

    assert orchestrator._media_restart_intent is None
    orchestrator._schedule_media_restart.assert_called_once()


async def test_a_dead_debt_owner_gets_a_recovery_settlement() -> None:
    """A debt bound to a task that just ended is settled in its name.

    The binding that protects a LIVE owner from bystanders would otherwise
    protect a DEAD one from recovery: the done-callback spawns a settlement
    carrying the dead owner's id, so the binding check passes and the books
    are paid without waiting for a later job that may never come.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", "job-dead", scoped=False)
    finished = SimpleNamespace(cancelled=lambda: True, exception=lambda: None)

    orchestrator._task_done("job-dead", finished)  # type: ignore[arg-type]
    recovery = orchestrator._settlement_retry_task
    assert recovery is not None, "no recovery settlement was spawned for the dead owner"
    await recovery

    resume.assert_awaited_once_with("profile-owed")
    assert orchestrator._deferred_handoff is None


async def test_a_scoped_debt_with_media_down_never_earns_a_broad_restart() -> None:
    """Scope truth discarded after a stop must have travelled with the debt.

    Cancellation after a SCOPED worker was stopped leaves a settler that
    observes media already down and cannot re-read the scope of a process that
    no longer exists. The debt carries it: settlement pays the restore, clears
    any stale broad intent a dead waiter left behind, and schedules nothing
    broad. The unknown-scope debt behaves identically, because unknown fails
    closed.
    """

    for carried in (True, None):
        orchestrator, _processes = _handoff_orchestrator(None, media_running=False)
        resume = AsyncMock(return_value=True)
        orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
        orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
        orchestrator._media_restart_intent = "job-stale-waiter"
        orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=carried)

        await orchestrator._settle_deferred_handoff()

        assert orchestrator._deferred_handoff is None
        resume.assert_awaited_once_with("profile-owed")
        orchestrator._schedule_media_restart.assert_not_called()
        assert orchestrator._media_restart_intent is None, (
            f"scoped={carried}: the stale broad intent survived settlement"
        )


async def test_a_media_job_taking_the_device_claims_the_outstanding_debt() -> None:
    """Preparation REBINDS what it inherits, so bystanders cannot race it.

    A media job that finds chat already down is redeeming someone's deferral;
    from that moment the debt is its own, and a bystander cancelled during the
    generation must be refused by the ordinary binding instead of finding an
    unbound obligation it may consume.
    """

    orchestrator, _processes = _handoff_orchestrator(None, media_running=False)
    orchestrator.processes = SimpleNamespace(  # type: ignore[assignment]
        statuses=Mock(
            return_value=[WorkerStatus(name="chat", state="stopped", managed=True, running=False)]
        ),
        settings=SimpleNamespace(auto_unload_chat_for_media=True),
        launch_scope_sha256=Mock(return_value=None),
    )
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    returned = await orchestrator._prepare_device_handoff(
        "text_to_image", job_id="job-media-taker", run_id="run-x"
    )

    assert returned == "profile-owed"
    assert orchestrator._deferred_handoff == _DeferredHandoff(
        "profile-owed", "job-media-taker", recycle_paid=False, scoped=False
    ), "the inherited debt was not claimed by the job taking the device"


async def test_a_failed_delayed_settlement_is_retried_until_it_pays() -> None:
    """The pump owns the debt across rounds; a one-shot retry strands it.

    A failed payment must not leave a done retry pointer beside an unpaid
    debt on an idle queue: the first recycle fails, the second pays and
    restores, and one pump task carries both rounds.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    processes.stop = AsyncMock(side_effect=[RuntimeError("stop failed"), None])
    _follow_media_lifecycle(processes)
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    orchestrator._arm_settlement_retry()
    pump = orchestrator._settlement_retry_task
    assert pump is not None
    await asyncio.wait_for(pump, timeout=10)

    assert processes.stop.await_count == 2, (
        "the first failed settlement was never retried: the pump is one-shot"
    )
    assert orchestrator._deferred_handoff is None, "the second round did not pay"
    assert orchestrator._settlement_retry_task is pump, (
        "a second retry owner was spawned beside the pump"
    )
    orchestrator._schedule_media_restart.assert_called_once_with()


async def test_persistent_settlement_failure_keeps_one_bounded_owner() -> None:
    """A worker that keeps failing is retried at a bounded rate by ONE task.

    Duplicate concurrent owners would race each other's payments; a task
    that abandons after one try strands the debt. The loop must retry at
    least three times, and the retry task's identity must not change.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    third_attempt = asyncio.Event()
    attempts = 0

    async def failing_stop(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts >= 3:
            third_attempt.set()
        raise RuntimeError("stop failed")

    processes.stop = failing_stop
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    orchestrator._arm_settlement_retry()
    pump = orchestrator._settlement_retry_task
    assert pump is not None
    await asyncio.wait_for(third_attempt.wait(), timeout=10)

    assert orchestrator._settlement_retry_task is pump and not pump.done(), (
        "the pump abandoned a still-unpaid debt, or a duplicate owner appeared"
    )
    assert orchestrator._deferred_handoff is not None, "a failing payment erased the debt"
    pump.cancel()
    with suppress(asyncio.CancelledError):
        await pump
    assert orchestrator._deferred_handoff is not None, (
        "cancellation erased the debt instead of retaining it"
    )


async def test_a_failed_recovery_payment_keeps_retrying() -> None:
    """Dead-owner recovery gets the same pump, with an immediate first try.

    A bare one-shot settlement task in the recovery slot cannot re-arm past
    its own live task after a failed payment, and the debt of a dead owner
    then stalls forever on an idle queue.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    processes.stop = AsyncMock(side_effect=[RuntimeError("stop failed"), None])
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", "job-dead", scoped=False)

    orchestrator._recover_owned_media_state("job-dead")
    pump = orchestrator._settlement_retry_task
    assert pump is not None, "a dead debt owner got no recovery owner"
    await asyncio.wait_for(pump, timeout=10)

    assert processes.stop.await_count == 2, "a failed recovery payment was never retried"
    assert orchestrator._deferred_handoff is None


async def test_cancelling_the_pump_retains_the_debt_and_spawns_nothing() -> None:
    """Cancellation ends the owner without erasing the books.

    The debt outlives the pump - a later job's own settlement, or a re-arm,
    still finds it - and cancellation must not leave a hidden successor task.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    debt = _DeferredHandoff("profile-owed", None, scoped=False)
    orchestrator._deferred_handoff = debt

    orchestrator._arm_settlement_retry()  # default interval: cancelled mid-sleep
    pump = orchestrator._settlement_retry_task
    assert pump is not None
    pump.cancel()
    with suppress(asyncio.CancelledError):
        await pump

    assert orchestrator._deferred_handoff == debt, "cancellation erased the debt"
    assert processes.stop.await_count == 0
    assert orchestrator._settlement_retry_task is pump and pump.cancelled(), (
        "cancellation spawned a replacement owner"
    )


async def test_close_cancels_the_settlement_pump_cleanly() -> None:
    """Shutdown owns the end of the pump; nothing respawns afterwards."""

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)
    orchestrator._arm_settlement_retry()
    pump = orchestrator._settlement_retry_task
    assert pump is not None

    await orchestrator.close()

    assert pump.done(), "close() left the settlement pump running"
    assert orchestrator._settlement_retry_task is None
    assert orchestrator._deferred_handoff is None, (
        "close() retained a debt against workers that no longer exist"
    )


async def test_settlement_defers_while_a_live_dispatch_owns_the_device() -> None:
    """An active media generation OWNS the device; settlement must wait.

    Stopping the worker under a running generation, or loading chat beside
    it, is the interference the binding exists to prevent, and an unbound
    debt is not a loophole: a RUNNING media row whose dispatch task is live
    in _tasks owns the device.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]

    class OccupiedSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(all=lambda: ["job-live-media"])

    orchestrator.session_factory = OccupiedSession  # type: ignore[assignment]
    holder = asyncio.get_running_loop().create_task(asyncio.sleep(60))
    orchestrator._tasks["job-live-media"] = holder
    debt = _DeferredHandoff("profile-owed", None, scoped=False)
    orchestrator._deferred_handoff = debt

    try:
        await orchestrator._settle_deferred_handoff()
    finally:
        holder.cancel()
        with suppress(asyncio.CancelledError):
            await holder

    assert orchestrator._deferred_handoff == debt, (
        "settlement consumed the debt while a live dispatch owned the device"
    )
    assert processes.stop.await_count == 0, (
        "settlement stopped the worker under a running generation"
    )
    orchestrator._resume_chat_worker.assert_not_awaited()
    orchestrator._schedule_media_restart.assert_not_called()


async def test_the_pump_defers_to_a_queued_continuation() -> None:
    """A QUEUED continuation intends to run; its debt is not the pump's.

    A pump that settles with job_id=debt.continuation_job_id is accepted by
    the continuation-authority guard by construction - impersonating the
    very job whose payment it steals. Under its own
    (absent) identity, the guard's durable-status half must defer, round
    after round, until the continuation provably cannot pay.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    processes.stop = AsyncMock()
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]

    durable = {"status": "queued"}

    class ContinuationSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            if identity == "job-q" and durable["status"] is not None:
                return SimpleNamespace(status=durable["status"])
            return None

        def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(all=lambda: [])

    orchestrator.session_factory = ContinuationSession  # type: ignore[assignment]
    debt = _DeferredHandoff("profile-owed", "job-q", scoped=False)
    orchestrator._deferred_handoff = debt

    orchestrator._arm_settlement_retry()
    pump = orchestrator._settlement_retry_task
    assert pump is not None
    await asyncio.sleep(0.05)

    assert not pump.done() and orchestrator._deferred_handoff == debt
    assert processes.stop.await_count == 0, (
        "the pump paid a QUEUED continuation's debt out from under it"
    )

    durable["status"] = None  # the continuation is gone; now the debt is orphaned
    await asyncio.wait_for(pump, timeout=10)
    assert orchestrator._deferred_handoff is None


async def test_the_pump_defers_to_a_complete_but_live_continuation() -> None:
    """COMPLETE commits BEFORE the lease-held handoff completion runs.

    While the dispatch task is still alive in _tasks, the continuation is
    mid-settlement of its own affairs; the pump paying its debt during that
    interval is exactly the bystander-settles-concurrently corruption the
    binding exists to prevent.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    processes.stop = AsyncMock()
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    holder = asyncio.get_running_loop().create_task(asyncio.sleep(60))
    orchestrator._tasks["job-c"] = holder
    debt = _DeferredHandoff("profile-owed", "job-c", scoped=False)
    orchestrator._deferred_handoff = debt

    orchestrator._arm_settlement_retry()
    pump = orchestrator._settlement_retry_task
    assert pump is not None
    await asyncio.sleep(0.05)

    try:
        assert not pump.done() and orchestrator._deferred_handoff == debt
        assert processes.stop.await_count == 0, (
            "the pump paid a COMPLETE-but-live continuation's debt"
        )
    finally:
        holder.cancel()
        with suppress(asyncio.CancelledError):
            await holder
    orchestrator._tasks.pop("job-c", None)

    await asyncio.wait_for(pump, timeout=10)
    assert orchestrator._deferred_handoff is None


async def test_the_pump_waits_out_a_foreground_checkout_and_its_rollback() -> None:
    """The checkout window: debt None does not mean debt PAID.

    A foreground settler clears the shared slot before its awaits; the pump
    observing None during that window must wait it out, because a rollback
    restores a debt that would otherwise have no owner left alive.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}
    processes.stop = AsyncMock()

    # The raise path runs through the restore half: a stop error is a
    # REPORTED failure the settle swallows into quiet retention, so only a
    # resume raise exercises the rollback this barrier is about.
    async def gated_resume(profile_id: str) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            entered.set()
            await release.wait()
            raise RuntimeError("foreground settlement dies mid-payment")
        return True

    orchestrator._resume_chat_worker = gated_resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    foreground = asyncio.get_running_loop().create_task(orchestrator._settle_deferred_handoff())
    await asyncio.wait_for(entered.wait(), timeout=10)
    assert orchestrator._deferred_handoff is None
    assert len(orchestrator._settlement_checkouts) == 1, "the checkout is unrecorded"

    orchestrator._arm_settlement_retry()
    pump = orchestrator._settlement_retry_task
    assert pump is not None
    await asyncio.sleep(0.05)
    assert not pump.done(), "the pump mistook a foreground checkout for payment and left"

    release.set()
    with pytest.raises(RuntimeError):
        await foreground
    assert orchestrator._deferred_handoff is not None, "the rollback lost the debt"

    await asyncio.wait_for(pump, timeout=10)
    assert orchestrator._deferred_handoff is None, "the pump never paid the restored debt"
    assert not orchestrator._settlement_checkouts
    assert calls["n"] >= 2


async def test_a_dying_foreground_settlement_arms_an_owner_for_its_restored_debt() -> None:
    """The except path restores the debt; restoration without an owner is
    a debt no task will ever retry. Task-done recovery matches only
    BOUND debts, so the unbound case must arm its own pump."""

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    processes.stop = AsyncMock()
    # A stop error is swallowed into quiet retention; the restore half is
    # what raises here.
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        side_effect=[RuntimeError("dies"), True]
    )
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    with pytest.raises(RuntimeError):
        await orchestrator._settle_deferred_handoff()

    pump = orchestrator._settlement_retry_task
    assert pump is not None and not pump.done(), "the restored debt has no owner on an idle queue"
    await asyncio.wait_for(pump, timeout=10)
    assert orchestrator._deferred_handoff is None


async def test_a_raised_payment_error_does_not_kill_the_pump() -> None:
    """An exception must not do what a reported failure is not allowed to
    do: end the retries. The pump logs and takes the next round."""

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    processes.stop = AsyncMock()
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        side_effect=[RuntimeError("resume raises"), True]
    )
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    orchestrator._arm_settlement_retry()
    pump = orchestrator._settlement_retry_task
    assert pump is not None
    await asyncio.wait_for(pump, timeout=10)

    assert orchestrator._resume_chat_worker.await_count == 2, (
        "the raised first payment ended the pump instead of being retried"
    )
    assert orchestrator._deferred_handoff is None


async def test_a_newer_debt_recorded_during_payment_stands_and_is_paid() -> None:
    """Success must not clobber newer truth, and the pump must then own it."""

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    processes.stop = AsyncMock()
    newer = _DeferredHandoff("profile-newer", None, scoped=False)
    seen: list[object] = []

    async def recording_resume(profile_id: str) -> bool:
        seen.append(profile_id)
        if len(seen) == 1:
            # A dispatch records a NEW debt while the first payment is in
            # flight; the first success must not erase it.
            orchestrator._deferred_handoff = newer
        return True

    orchestrator._resume_chat_worker = recording_resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    orchestrator._arm_settlement_retry()
    pump = orchestrator._settlement_retry_task
    assert pump is not None
    await asyncio.wait_for(pump, timeout=10)

    assert seen == ["profile-owed", "profile-newer"], (
        "the newer debt recorded during payment was clobbered or never paid"
    )
    assert orchestrator._deferred_handoff is None


async def test_close_gates_new_retry_and_restart_tasks() -> None:
    """Teardown ownership is single: while close() awaits one task, nothing
    may mint a replacement into a slot close has already passed. The gate
    refuses arming and scheduling outright once closing begins."""

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    orchestrator._closing = True

    orchestrator._schedule_media_restart()
    assert orchestrator._media_restart_task is None, (
        "a restart task was scheduled during close and would escape teardown"
    )
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)
    orchestrator._arm_settlement_retry()
    assert orchestrator._settlement_retry_task is None, (
        "a retry pump was armed during close and would escape teardown"
    )
    orchestrator._deferred_handoff = None


async def test_a_settler_dying_under_close_leaves_no_live_task_behind() -> None:
    """close() awaits the cancelled pump whose settlement is mid-payment;
    the rollback arms nothing while closing, and after close no settlement
    or restart task survives anywhere."""

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    entered = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_stop(*args: object, **kwargs: object) -> None:
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await release.wait()
            raise

    processes.stop = stubborn_stop
    orchestrator._resume_chat_worker = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)
    orchestrator._arm_settlement_retry()
    pump = orchestrator._settlement_retry_task
    assert pump is not None
    await asyncio.wait_for(entered.wait(), timeout=10)

    closer = asyncio.get_running_loop().create_task(orchestrator.close())
    await asyncio.sleep(0.01)  # close is cancelling; the pump holds via stubborn_stop
    release.set()
    await asyncio.wait_for(closer, timeout=10)

    strays = [
        task
        for task in asyncio.all_tasks()
        if task.get_name() in {"deferred-handoff-settlement-retry", "media-worker-handoff-restart"}
        and not task.done()
    ]
    for task in strays:
        task.cancel()
    assert strays == [], "a replacement task escaped close()'s teardown"
    assert orchestrator._settlement_retry_task is None
    assert orchestrator._deferred_handoff is None


def _fake_busy_try_lease(device_id: str = "primary"):  # type: ignore[no-untyped-def]  # noqa: ARG001
    """A device someone else owns: settlement must defer, never wait."""

    class _Busy:
        async def __aenter__(self):  # type: ignore[no-untyped-def]
            return False

        async def __aexit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

    return _Busy()


class _MediaKindSession:
    """A session whose kind query reports exactly the wired media job ids -
    the intersection of live task ids and media kind the query answers."""

    def __init__(self, media_ids: list[str]) -> None:
        self._media_ids = media_ids

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        return None

    def get(self, model, identity):  # type: ignore[no-untyped-def]
        return None

    def scalars(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(all=lambda: list(self._media_ids))


async def test_a_complete_but_live_stranger_blocks_settlement() -> None:
    """The cross-product a durable filter misses: COMPLETE, live, unbound.

    A media job commits COMPLETE before its handoff completion runs. While
    that finalizing task is live it still owns the device, even though it no
    longer owns the DEBT - a check selecting only durable RUNNING rows lets
    a settlement racing a stranger's finalization stop the worker under
    it. Kind and liveness decide; durable status does not appear.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    factory = _MediaKindSession(["job-media-finalizing"])
    orchestrator.session_factory = lambda: factory  # type: ignore[assignment]
    orchestrator._tasks["job-media-finalizing"] = SimpleNamespace(done=lambda: False)  # type: ignore[assignment]
    debt = _DeferredHandoff(None, None, recycle_paid=False, scoped=False)
    orchestrator._deferred_handoff = debt

    await orchestrator._settle_deferred_handoff(job_id="job-cancelled-bystander")

    assert orchestrator._deferred_handoff == debt, (
        "settlement consumed the debt while a COMPLETE-but-live media task owned the device"
    )
    processes.stop.assert_not_awaited()
    resume.assert_not_awaited()


async def test_a_queued_media_lease_waiter_blocks_settlement() -> None:
    """The other cross-product: QUEUED, dispatch task already live.

    A queued media job whose dispatch task is waiting on the lease intends to
    run and will take the device the moment it is free; settlement stopping
    the worker in front of it recycles the model that job is about to use.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    factory = _MediaKindSession(["job-media-queued"])
    orchestrator.session_factory = lambda: factory  # type: ignore[assignment]
    orchestrator._tasks["job-media-queued"] = SimpleNamespace(done=lambda: False)  # type: ignore[assignment]
    debt = _DeferredHandoff("profile-owed", None, recycle_paid=False, scoped=False)
    orchestrator._deferred_handoff = debt

    await orchestrator._settle_deferred_handoff()

    assert orchestrator._deferred_handoff == debt
    processes.stop.assert_not_awaited()
    resume.assert_not_awaited()


async def test_a_live_text_task_does_not_block_settlement() -> None:
    """Liveness alone is not media ownership: a live task of non-media kind
    reports no media ids from the kind query, and settlement proceeds."""

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    factory = _MediaKindSession([])
    orchestrator.session_factory = lambda: factory  # type: ignore[assignment]
    orchestrator._tasks["job-text-live"] = SimpleNamespace(done=lambda: False)  # type: ignore[assignment]
    orchestrator._deferred_handoff = _DeferredHandoff(
        "profile-owed", None, recycle_paid=False, scoped=False
    )

    await orchestrator._settle_deferred_handoff()

    assert orchestrator._deferred_handoff is None
    resume.assert_awaited_once_with("profile-owed")


async def test_a_busy_device_defers_settlement_and_arms_the_pump() -> None:
    """Settlement takes the SAME lock dispatch holds, without queueing.

    Busy means a live dispatch, a borrowed device hold, or a concurrent settler
    owns the device; acting on state read before acquisition is the race the
    lock exists to close. The debt stays on the books untouched and the
    bounded pump is armed, so a borrow that pays nothing cannot strand the
    obligation on an idle queue.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator.scheduler.try_lease = _fake_busy_try_lease
    debt = _DeferredHandoff("profile-owed", None, recycle_paid=False, scoped=False)
    orchestrator._deferred_handoff = debt

    await orchestrator._settle_deferred_handoff()
    pump = orchestrator._settlement_retry_task
    assert pump is not None, "a busy device must arm the pump, not drop the debt"
    pump.cancel()
    await asyncio.gather(pump, return_exceptions=True)

    assert orchestrator._deferred_handoff == debt
    processes.stop.assert_not_awaited()
    resume.assert_not_awaited()


async def test_settlement_revalidates_the_debt_under_the_device_lock() -> None:
    """Everything sampled before acquisition is stale by the time it is held.

    The fast path reads the slot before taking the lock; an owner can pay the
    debt in between. A settlement that trusted its pre-acquisition read would
    stop the worker for an obligation that no longer exists.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.stop = AsyncMock()
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff(
        "profile-owed", None, recycle_paid=False, scoped=False
    )

    def _pay_before_yield():  # type: ignore[no-untyped-def]
        class _Held:
            async def __aenter__(self):  # type: ignore[no-untyped-def]
                orchestrator._deferred_handoff = None
                return True

            async def __aexit__(self, *_args):  # type: ignore[no-untyped-def]
                return None

        return _Held()

    orchestrator.scheduler.try_lease = lambda *a, **k: _pay_before_yield()  # noqa: ARG005

    await orchestrator._settle_deferred_handoff()

    processes.stop.assert_not_awaited()
    resume.assert_not_awaited()
    assert not orchestrator._settlement_checkouts


async def test_reverse_order_rollbacks_keep_the_newest_obligation() -> None:
    """Two failed checkouts must merge in EITHER completion order.

    A bare occupancy check loses whichever rollback runs second: old D1
    restored first leaves new D2's rollback seeing the slot taken, and the
    newer obligation vanishes. Generation ordering keeps the newer record standing
    in both orders, with the unpaid halves of both surviving into it.
    """

    orchestrator, _processes = _handoff_orchestrator(None)
    d1 = orchestrator._minted_debt("profile-old", None, recycle_paid=True, scoped=None)
    d2 = orchestrator._minted_debt("profile-new", "job-next", recycle_paid=False, scoped=True)
    assert d1.generation < d2.generation

    # Order one: older rollback lands first, newer must still win the slot.
    orchestrator._deferred_handoff = None
    orchestrator._restore_debt(d1)
    orchestrator._restore_debt(d2)
    merged = orchestrator._deferred_handoff
    assert merged is not None
    assert merged.generation == d2.generation
    assert merged.profile_id == "profile-new"
    assert merged.recycle_paid is False, "an unpaid recycle was erased by the merge"
    assert merged.scoped is True

    # Order two: newer rollback lands first, the late older return must not
    # displace it - and its unpaid half must still narrow the record.
    d3 = orchestrator._minted_debt("profile-a", None, recycle_paid=False, scoped=None)
    d4 = orchestrator._minted_debt("profile-b", None, recycle_paid=True, scoped=False)
    orchestrator._deferred_handoff = None
    orchestrator._restore_debt(d4)
    orchestrator._restore_debt(d3)
    merged = orchestrator._deferred_handoff
    assert merged is not None
    assert merged.generation == d4.generation
    assert merged.profile_id == "profile-b"
    assert merged.recycle_paid is False
    assert merged.scoped is False


async def test_a_rollback_under_a_newer_mint_merges_instead_of_skipping() -> None:
    """The ordering rule under a failed payment.

    While a settlement holds its checkout, a cancellation-path mint records a
    newer debt. The settlement's rollback must neither skip on occupancy
    nor clobber the newer record: the newer truth stands and the
    returning obligation's unpaid recycle survives into it.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    entered = asyncio.Event()
    release = asyncio.Event()
    # The stop FAILS quietly, so the checkout returns with its recycle still
    # unpaid - the merge must carry that into the newer record.
    processes.stop = AsyncMock(side_effect=RuntimeError("stop refused"))

    async def gated_resume(profile_id: str) -> bool:  # noqa: ARG001
        entered.set()
        await release.wait()
        raise RuntimeError("payment dies while a newer debt is minted")

    orchestrator._resume_chat_worker = gated_resume  # type: ignore[method-assign]
    orchestrator._deferred_handoff = orchestrator._minted_debt(
        "profile-old", None, recycle_paid=False, scoped=False
    )

    settler = asyncio.get_running_loop().create_task(orchestrator._settle_deferred_handoff())
    await asyncio.wait_for(entered.wait(), timeout=10)
    newer = orchestrator._minted_debt("profile-new", None, recycle_paid=True, scoped=True)
    orchestrator._deferred_handoff = newer
    release.set()
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(settler, timeout=10)

    merged = orchestrator._deferred_handoff
    assert merged is not None
    assert merged.generation == newer.generation
    assert merged.profile_id == "profile-new"
    assert merged.recycle_paid is False, (
        "the returning checkout's unpaid recycle was erased by the newer record"
    )
    assert not orchestrator._settlement_checkouts


async def test_a_dispatch_started_after_close_snapshots_is_refused() -> None:
    """The late-dispatch barrier: past admission is not past the mint gate.

    A create_turn that passed admission and then awaited its event
    publications can call start() after close() flipped the flag and
    snapshotted the live tasks. A bare create_task there is a dispatch
    nothing ever cancels; the close-aware mint refuses it and the job stays
    QUEUED for the next process.
    """

    orchestrator, _processes = _handoff_orchestrator(None)
    await orchestrator.close()

    orchestrator.start("job-late-dispatch", "run-late")

    assert "job-late-dispatch" not in orchestrator._tasks


async def test_recovery_minted_after_teardown_passes_the_slot_is_refused() -> None:
    """The direct-recovery barrier: a done-callback racing close().

    _task_done fires while close() gathers the dispatches it cancelled; a
    recovery pump created after close() has already passed the retry-task
    slot would survive teardown unowned. The mint gate refuses it
    deterministically once _closing is set.
    """

    orchestrator, _processes = _handoff_orchestrator(None)
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", "job-dead", scoped=False)
    orchestrator._closing = True
    finished = SimpleNamespace(cancelled=lambda: True, exception=lambda: None)

    orchestrator._task_done("job-dead", finished)  # type: ignore[arg-type]

    assert orchestrator._settlement_retry_task is None


async def test_every_background_mint_is_refused_while_closing() -> None:
    """Minting refuses while closing: restart, retry pump, dispatch, recovery.

    Refusal must be a property of the mint path itself, not of each
    caller remembering to check: a guard a caller owns is a guard a
    caller can lack.
    """

    orchestrator, _processes = _handoff_orchestrator(None)
    orchestrator._closing = True

    orchestrator._schedule_media_restart()
    assert orchestrator._media_restart_task is None

    orchestrator._arm_settlement_retry()
    assert orchestrator._settlement_retry_task is None

    orchestrator.start("job-while-closing", None)
    assert "job-while-closing" not in orchestrator._tasks


async def test_a_scoped_stop_with_failed_restore_never_broad_restarts() -> None:
    """Observation must survive the rollback.

    An older BROAD debt is settled against a worker OBSERVED scoped; the
    stop commits, the restore reports failure, and the retained debt must
    carry the observed scope - not revert to the remembered broad one. The
    worker-down retry can no longer read the stopped worker's scope, so a
    reverted false would authorize exactly the broad restart the scoped
    context requires to stay down.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.launch_scope_sha256 = Mock(return_value="scope-digest")
    processes.stop = AsyncMock()
    restart = Mock()
    orchestrator._schedule_media_restart = restart  # type: ignore[method-assign]
    resume = AsyncMock(return_value=False)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._deferred_handoff = orchestrator._minted_debt(
        "profile-owed", None, recycle_paid=False, scoped=False
    )

    await orchestrator._settle_deferred_handoff()
    pump = orchestrator._settlement_retry_task
    if pump is not None:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)

    retained = orchestrator._deferred_handoff
    assert retained is not None, "the failed restore erased the debt"
    assert retained.recycle_paid is True, "the committed stop was forgotten"
    assert retained.scoped is True, (
        "the rollback reverted to the remembered broad scope instead of "
        "carrying the observed scoped truth"
    )

    # The worker is down now; the retry leg completes the restore and must
    # schedule NOTHING broad for the scoped context.
    down = WorkerStatus(name="media", state="stopped", managed=True, running=False)
    processes.statuses = Mock(return_value=[down])
    resume.return_value = True

    await orchestrator._settle_deferred_handoff()
    pump = orchestrator._settlement_retry_task
    if pump is not None:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)

    assert orchestrator._deferred_handoff is None, "the retry did not settle"
    restart.assert_not_called()


async def test_a_stop_that_fails_midway_keeps_the_freshly_observed_scope() -> None:
    """The scope observed before the stop survives the stop failing.

    The stop tears down the live worker record before its fallible
    termination await, so a stop that raises leaves a worker that can no
    longer answer for its own scope. The observation must already be on the
    debt when the stop begins: a debt still carrying its older broad answer
    would send the worker-down retry down the broad-restart branch for a
    context observed scoped the moment before.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.launch_scope_sha256 = Mock(return_value="scope-digest")
    orchestrator._deferred_handoff = orchestrator._minted_debt(
        "profile-owed", None, recycle_paid=False, scoped=False
    )
    seen_at_stop: list[object] = []

    async def partial_stop(name: str) -> None:
        seen_at_stop.append(orchestrator._deferred_handoff)
        down = WorkerStatus(name="media", state="stopped", managed=True, running=False)
        processes.statuses = Mock(return_value=[down])
        raise RuntimeError("terminate failed mid-stop")

    processes.stop = AsyncMock(side_effect=partial_stop)

    with pytest.raises(RuntimeError, match="could not be recycled"):
        await orchestrator._release_media_device_for_chat("job-text")

    at_stop = seen_at_stop[0]
    assert at_stop is not None, "the debt was cleared before the stop"
    assert at_stop.scoped is True, (
        "the freshly observed scope must be on the debt BEFORE the fallible "
        "stop; after a partial stop the worker cannot be asked again"
    )
    assert at_stop.recycle_paid is False, (
        "a rebind ahead of the stop must not record a recycle that has not happened yet"
    )
    retained = orchestrator._deferred_handoff
    assert retained is not None, "the failed stop erased the debt"
    assert retained.scoped is True, "the partial stop reverted to the older scope"
    assert retained.recycle_paid is False, "an unfinished stop must not read as paid"
    assert retained.continuation_job_id == "job-text"

    # The text job failed honestly rather than load beside the worker; the
    # settlement is the recovery, and the worker is down now - the debt's
    # carried scope is all there is, and it must not authorize a broad
    # restart.
    restart = Mock()
    orchestrator._schedule_media_restart = restart  # type: ignore[method-assign]
    orchestrator._resume_chat_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

    await orchestrator._settle_deferred_handoff("job-text")
    pump = orchestrator._settlement_retry_task
    if pump is not None:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)

    assert orchestrator._deferred_handoff is None, "the retry did not settle"
    restart.assert_not_called()


async def test_a_cancelled_stop_keeps_the_freshly_observed_scope_for_recovery() -> None:
    """Cancellation mid-stop leaves the fresh scope for the next settler.

    A cancelled takeover task never runs its text job, so the rebound debt
    is the only record of the displaced obligation, and the ownerless
    settler that eventually pays it can no longer read the stopped worker.
    The scope observed while the worker lived must ride the debt through
    the cancellation, and the worker-down retry must stay scoped.
    """

    orchestrator, processes = _handoff_orchestrator(None)
    processes.launch_scope_sha256 = Mock(return_value="scope-digest")
    orchestrator._deferred_handoff = orchestrator._minted_debt(
        "profile-owed", None, recycle_paid=False, scoped=False
    )
    seen_at_stop: list[object] = []

    async def cancelled_stop(name: str) -> None:
        seen_at_stop.append(orchestrator._deferred_handoff)
        down = WorkerStatus(name="media", state="stopped", managed=True, running=False)
        processes.statuses = Mock(return_value=[down])
        raise asyncio.CancelledError

    processes.stop = AsyncMock(side_effect=cancelled_stop)

    with pytest.raises(asyncio.CancelledError):
        await orchestrator._release_media_device_for_chat("job-text")

    at_stop = seen_at_stop[0]
    assert at_stop is not None and at_stop.scoped is True, (
        "the freshly observed scope must be on the debt BEFORE the stop can be cancelled"
    )
    retained = orchestrator._deferred_handoff
    assert retained is not None, "cancellation erased the debt"
    assert retained.scoped is True, "cancellation reverted to the older scope"
    assert retained.recycle_paid is False, "a cancelled stop must not read as paid"

    # The vacancy settler (no job id) pays next: the bound takeover job is
    # gone, the worker is down, and the carried scope must keep the broad
    # restart off the books.
    restart = Mock()
    orchestrator._schedule_media_restart = restart  # type: ignore[method-assign]
    orchestrator._resume_chat_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

    await orchestrator._settle_deferred_handoff()
    pump = orchestrator._settlement_retry_task
    if pump is not None:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)

    assert orchestrator._deferred_handoff is None, "the vacancy settler did not settle"
    restart.assert_not_called()


async def test_media_ownership_reads_real_status_and_kind_rows(
    settings: Settings,
) -> None:
    """The ownership cross-products, on persisted rows.

    A COMPLETE media row with a live task, a QUEUED media row, and a live
    text dispatch must classify through the rows as the database holds
    them: kind and liveness decide, and durable status alone never
    answers.
    """

    from local_lm.db import SessionLocal, configure_database, init_db
    from local_lm.models import Job

    settings.prepare()
    configure_database(settings)
    init_db()
    with SessionLocal() as session:
        session.add_all(
            [
                Job(
                    id="job-media-complete",
                    kind="image",
                    status="complete",
                    phase="complete",
                    payload_json={},
                ),
                Job(
                    id="job-media-queued",
                    kind="image",
                    status="queued",
                    phase="queued",
                    payload_json={},
                ),
                Job(
                    id="job-text-running",
                    kind="chat",
                    status="running",
                    phase="running",
                    payload_json={},
                ),
            ]
        )
        session.commit()

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator.session_factory = SessionLocal  # type: ignore[assignment]
    live = SimpleNamespace(done=lambda: False)

    # COMPLETE-but-live media: the durable status must not matter.
    orchestrator._tasks = {"job-media-complete": live}  # type: ignore[assignment]
    assert orchestrator._media_dispatch_is_active(), (
        "a COMPLETE-but-live media task read as not owning the device"
    )
    # QUEUED media whose dispatch task already waits on the lease.
    orchestrator._tasks = {"job-media-queued": live}  # type: ignore[assignment]
    assert orchestrator._media_dispatch_is_active(), (
        "a QUEUED lease-waiter read as not owning the device"
    )
    # A live TEXT task: kind decides ownership, not liveness.
    orchestrator._tasks = {"job-text-running": live}  # type: ignore[assignment]
    assert not orchestrator._media_dispatch_is_active(), "a live text task read as media ownership"


async def test_settlement_defers_to_real_media_rows_and_proceeds_past_text(
    settings: Settings,
) -> None:
    """Settlement over persisted rows: a live media row defers it, a queued
    lease-waiter defers it, and a live text row does not."""

    from local_lm.db import SessionLocal, configure_database, init_db
    from local_lm.models import Job

    settings.prepare()
    configure_database(settings)
    init_db()
    with SessionLocal() as session:
        session.add_all(
            [
                Job(
                    id="job-media-complete",
                    kind="image",
                    status="complete",
                    phase="complete",
                    payload_json={},
                ),
                Job(
                    id="job-text-running",
                    kind="chat",
                    status="running",
                    phase="running",
                    payload_json={},
                ),
            ]
        )
        session.commit()

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator.session_factory = SessionLocal  # type: ignore[assignment]
    processes.stop = AsyncMock()
    resume = AsyncMock(return_value=True)
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    live = SimpleNamespace(done=lambda: False)

    orchestrator._tasks = {"job-media-complete": live}  # type: ignore[assignment]
    debt = orchestrator._minted_debt(None, None, recycle_paid=False, scoped=False)
    orchestrator._deferred_handoff = debt
    await orchestrator._settle_deferred_handoff()
    assert orchestrator._deferred_handoff == debt, (
        "settlement consumed the debt under a COMPLETE-but-live media row"
    )
    resume.assert_not_awaited()

    orchestrator._tasks = {"job-text-running": live}  # type: ignore[assignment]
    orchestrator._deferred_handoff = orchestrator._minted_debt(
        "profile-owed", None, recycle_paid=False, scoped=False
    )
    await orchestrator._settle_deferred_handoff()
    assert orchestrator._deferred_handoff is None, (
        "settlement would not proceed past a live text row"
    )
    resume.assert_awaited_once_with("profile-owed")


# The give-back's lifetime: decided at fire time, retained on failure,
# superseded by a scoped launch.


async def test_a_dead_waiters_broad_intent_never_replaces_a_running_scoped_worker() -> None:
    """A broad intent armed for a waiter that later died fires only into a
    down worker: a media worker running now under an activation scope is
    newer truth, and the recovery for the dead owner clears the intent
    without a restart."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=True)
    processes.launch_scope_sha256 = Mock(return_value="scope-live")
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._media_restart_intent = "job-dead-waiter"
    orchestrator._deferred_handoff = None
    finished = SimpleNamespace(cancelled=lambda: True, exception=lambda: None)

    orchestrator._task_done("job-dead-waiter", finished)  # type: ignore[arg-type]

    assert orchestrator._media_restart_intent is None
    orchestrator._schedule_media_restart.assert_not_called()


async def test_a_dead_waiters_broad_intent_never_replaces_a_running_broad_worker() -> None:
    """A worker already running broad is the give-back already made; the
    recovery fires nothing into it."""

    orchestrator, _processes = _handoff_orchestrator(None, media_running=True)
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._media_restart_intent = "job-dead-waiter"

    orchestrator._recover_owned_media_state("job-dead-waiter")

    assert orchestrator._media_restart_intent is None
    orchestrator._schedule_media_restart.assert_not_called()


async def test_a_queued_media_dispatch_is_warmed_not_raced() -> None:
    """With the worker down and a media job queued on the device, the
    give-back proceeds: the worker it starts is what that job waits for,
    launches serialize in the supervisor, and a scoped launch replaces a
    broad worker. Only a worker running now refuses the restart."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._media_dispatch_is_active = Mock(return_value=True)  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._media_restart_intent = "job-dead-waiter"

    orchestrator._recover_owned_media_state("job-dead-waiter")

    orchestrator._schedule_media_restart.assert_called_once()
    _follow_media_lifecycle(processes)
    await processes.start_media()
    orchestrator._schedule_give_back()
    orchestrator._schedule_media_restart.assert_called_once()


async def test_the_idle_queue_give_back_survives_the_completing_jobs_own_task() -> None:
    """A broad media job completing on an idle queue gives the device back
    even though its own dispatch task is still live in the task table while
    its handoff runs."""

    orchestrator, processes = _handoff_orchestrator(None)
    restart = Mock()
    orchestrator._schedule_media_restart = restart  # type: ignore[method-assign]
    orchestrator._resume_chat_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
    # The kind query answers for the job's own live task, as it would in
    # production while that task runs its handoff.
    factory = _MediaKindSession(["job-media"])
    orchestrator.session_factory = lambda: factory  # type: ignore[assignment]

    async def still_running() -> None:
        await asyncio.sleep(3600)

    own_task = asyncio.create_task(still_running())
    orchestrator._tasks["job-media"] = own_task
    try:
        await orchestrator._complete_media_handoff("profile-chat", "job-media")
    finally:
        own_task.cancel()
        await asyncio.gather(own_task, return_exceptions=True)

    processes.stop.assert_awaited_once_with("media")
    restart.assert_called_once()
    assert orchestrator._media_restart_intent is None
    assert orchestrator._deferred_handoff is None


async def test_scoped_truth_supersedes_a_retained_give_back() -> None:
    """A restart the supervisor refused stays owed only until scoped truth
    arrives: a scoped stop or a scoped launch clears the obligation, so no
    later pump round can broad-start the scoped context."""

    orchestrator, processes = _handoff_orchestrator(None)
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._resume_chat_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
    processes.launch_scope_sha256 = Mock(return_value="scope-live")
    orchestrator._media_give_back_owed = True
    orchestrator._media_give_back_attempts = 1

    await orchestrator._complete_media_handoff("profile-chat", "job-media")

    assert orchestrator._media_give_back_owed is False
    assert orchestrator._media_give_back_attempts == 0
    assert not orchestrator._give_back_pending()
    orchestrator._schedule_media_restart.assert_not_called()

    orchestrator._media_give_back_owed = True
    orchestrator._media_give_back_attempts = 1
    _follow_media_lifecycle(processes)
    await processes.stop("media")
    scope = SimpleNamespace(launch_sha256="scope-next")
    await orchestrator._ensure_media_worker(activation_scope=scope)  # type: ignore[arg-type]

    assert orchestrator._media_give_back_owed is False
    orchestrator._schedule_give_back(retry=True)
    orchestrator._schedule_media_restart.assert_not_called()


async def test_a_running_worker_satisfies_a_retained_give_back() -> None:
    """A worker running when the retry fires is the device given back: the
    obligation is met, the pump has nothing left to own, and no restart
    replaces the worker."""

    orchestrator, _processes = _handoff_orchestrator(None, media_running=True)
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._media_give_back_owed = True
    orchestrator._media_give_back_attempts = 2

    orchestrator._schedule_give_back(retry=True)

    orchestrator._schedule_media_restart.assert_not_called()
    assert orchestrator._media_give_back_owed is False
    assert orchestrator._media_give_back_attempts == 0
    assert not orchestrator._give_back_pending()


async def test_any_task_teardown_fires_a_vacated_owners_intent() -> None:
    """An intent whose owner can no longer pay is fired by whichever task
    ends next, not only by the owner's own teardown."""

    orchestrator, _processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._media_restart_intent = "job-gone"

    orchestrator._recover_owned_media_state("job-other")

    assert orchestrator._media_restart_intent is None
    orchestrator._schedule_media_restart.assert_called_once()


async def test_a_live_owners_intent_is_left_to_its_owner() -> None:
    """A stranger's teardown does not consume an intent whose owner is still
    running; the owner discharges it at its own pace."""

    orchestrator, _processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._media_restart_intent = "job-live"

    async def running() -> None:
        await asyncio.sleep(3600)

    live = asyncio.create_task(running())
    orchestrator._tasks["job-live"] = live
    try:
        orchestrator._recover_owned_media_state("job-other")
    finally:
        live.cancel()
        await asyncio.gather(live, return_exceptions=True)

    assert orchestrator._media_restart_intent == "job-live"
    orchestrator._schedule_media_restart.assert_not_called()


async def test_a_managed_worker_that_is_not_running_never_earns_a_broad_intent() -> None:
    """A managed media worker that has exited cannot answer for its scope;
    the handoff treats it as scoped, arms nothing, and leaves any older
    intent superseded."""

    next_run = SimpleNamespace(operation="text", profile_id="profile-text")
    orchestrator, processes = _handoff_orchestrator(
        ("job-next", "run-next"), next_run, media_running=False
    )
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._resume_chat_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
    orchestrator._media_restart_intent = "job-dead-waiter"

    await orchestrator._finish_media_handoff("profile-chat", "text", "job-next", "job-media")

    processes.stop.assert_not_awaited()
    assert orchestrator._media_restart_intent is None
    orchestrator._schedule_media_restart.assert_not_called()


async def test_a_scoped_launch_supersedes_a_stale_broad_intent() -> None:
    """The scoped worker about to run replaces any broad intent a dead waiter
    left behind, before the launch itself."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._media_restart_intent = "job-dead-waiter"
    scope = SimpleNamespace(launch_sha256="scope-next")

    await orchestrator._ensure_media_worker(activation_scope=scope)  # type: ignore[arg-type]

    assert orchestrator._media_restart_intent is None
    processes.start_media.assert_awaited_once_with(activation_scope=scope)


async def test_the_scoped_replacement_holds_through_the_real_task_teardown() -> None:
    """An unscoped completion binds a broad intent to a queued verification,
    a foreground media job then launches scoped, the verification is
    cancelled, and its task teardown must not replace the scoped
    worker."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._media_restart_intent = "job-verify"
    scope = SimpleNamespace(launch_sha256="scope-foreground")
    await orchestrator._ensure_media_worker(activation_scope=scope)  # type: ignore[arg-type]
    processes.launch_scope_sha256 = Mock(return_value="scope-foreground")
    processes.start_media.reset_mock()

    async def waiting() -> None:
        await asyncio.sleep(3600)

    task = asyncio.create_task(waiting())
    orchestrator._tasks["job-verify"] = task
    task.add_done_callback(lambda finished: orchestrator._task_done("job-verify", finished))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert orchestrator._media_restart_intent is None
    processes.start_media.assert_not_awaited()
    assert orchestrator._media_restart_task is None


@pytest.mark.parametrize("failure", ["error", "cancelled"], ids=["stream-error", "cancelled"])
async def test_a_worker_down_takeover_gives_back_exactly_once_across_a_failing_stream(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """The first delta gives the device back; when the stream then fails, the
    retained debt's settlement finds the worker already back and restarts
    nothing a second time."""

    from local_lm.adapters.base import ChatEvent

    order: list[str] = []
    orchestrator, processes = _text_dispatch_orchestrator(monkeypatch, order)
    with orchestrator.session_factory() as session:
        session.get(Run, "run-text").provenance_json = {}
        session.get(Run, "run-text").assistant_message_id = "assistant-text"
        session.get(Job, "job-text").progress_json = {}
    await processes.stop("media")
    processes.stop.reset_mock()

    async def stream(_request):  # type: ignore[no-untyped-def]
        yield ChatEvent(type="delta", text="Hello", data={})
        restart = orchestrator._media_restart_task
        assert restart is not None, "the first delta did not give the device back"
        await restart
        if failure == "cancelled":
            raise asyncio.CancelledError
        yield ChatEvent(type="error", text="", data={"error": "stream failed"})

    orchestrator.engines = SimpleNamespace(
        chat=SimpleNamespace(stream=stream), settings=SimpleNamespace(media_engine="comfyui")
    )
    orchestrator._set_chat_phase = AsyncMock()  # type: ignore[method-assign]
    orchestrator._ensure_chat_worker = AsyncMock(return_value=None)  # type: ignore[method-assign]
    orchestrator._prepare_chat_context = AsyncMock(  # type: ignore[method-assign]
        return_value=([], {}, {}, False)
    )
    orchestrator._persist_streamed_text = Mock()  # type: ignore[method-assign]
    orchestrator._resume_chat_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
    orchestrator.scheduler.publish_job = AsyncMock()
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", None, scoped=False)

    if failure == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await orchestrator._execute("job-text", "run-text")
    else:
        with suppress(RuntimeError):
            await orchestrator._execute("job-text", "run-text")
    processes.start_media.assert_awaited_once()
    assert orchestrator._deferred_handoff is not None, "a failed stream must retain the debt"
    # The dead owner's durable status is terminal by the time the recovery
    # pump settles in nobody's name.
    with orchestrator.session_factory() as session:
        session.get(Job, "job-text").status = failure

    await orchestrator._settle_deferred_handoff()

    assert orchestrator._media_restart_task is None, (
        "settlement scheduled a restart for a worker that is already back"
    )
    processes.start_media.assert_awaited_once()
    assert orchestrator._deferred_handoff is None
    assert orchestrator._media_restart_intent is None


async def test_a_failed_give_back_is_retried_until_the_worker_returns() -> None:
    """A restart the supervisor refuses stays owed: the pump retries it at its
    bounded rate through the same gate, and a later success discharges it."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    processes.start_media = AsyncMock(side_effect=[RuntimeError("start failed"), None])
    _follow_media_lifecycle(processes)

    orchestrator._schedule_give_back()
    restart = orchestrator._media_restart_task
    assert restart is not None
    await restart
    await asyncio.sleep(0)

    assert orchestrator._media_give_back_owed is True
    assert orchestrator._media_give_back_attempts == 1
    pump = orchestrator._settlement_retry_task
    assert pump is not None, "a refused give-back must arm the retry pump"
    await asyncio.wait_for(pump, timeout=5)

    assert processes.start_media.await_count == 2
    assert orchestrator._media_give_back_owed is False
    assert orchestrator._media_give_back_attempts == 0
    assert processes.statuses()[0].running is True


async def test_a_give_back_that_keeps_failing_is_bounded_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The retry is bounded: after the configured attempts the obligation is
    reported as abandoned and stays visible, and nothing schedules further."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._SETTLEMENT_RETRY_SECONDS = 0.001
    processes.start_media = AsyncMock(side_effect=RuntimeError("start failed"))
    _follow_media_lifecycle(processes)

    with caplog.at_level(logging.ERROR, logger="local_lm.orchestrator"):
        orchestrator._schedule_give_back()
        restart = orchestrator._media_restart_task
        assert restart is not None
        await restart
        await asyncio.sleep(0)
        pump = orchestrator._settlement_retry_task
        assert pump is not None
        await asyncio.wait_for(pump, timeout=5)
        await asyncio.sleep(0)

    assert processes.start_media.await_count == orchestrator._GIVE_BACK_ATTEMPTS
    assert orchestrator._media_give_back_owed is True
    assert not orchestrator._give_back_pending()
    assert any("give-back was abandoned" in record.getMessage() for record in caplog.records)
    orchestrator._schedule_give_back(retry=True)
    assert orchestrator._media_restart_task is None or orchestrator._media_restart_task.done()
    assert processes.start_media.await_count == orchestrator._GIVE_BACK_ATTEMPTS


async def test_verification_restarts_only_an_unscoped_worker_it_stopped() -> None:
    """A media worker stopped for verification comes back broad only if it
    was broad; a scoped worker stays down and supersedes any remembered
    give-back."""

    orchestrator, _processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._media_restart_intent = "job-dead-waiter"
    orchestrator._media_give_back_owed = True

    orchestrator._restore_media_after_verification(scoped=True)

    orchestrator._schedule_media_restart.assert_not_called()
    assert orchestrator._media_restart_intent is None
    assert orchestrator._media_give_back_owed is False

    orchestrator._restore_media_after_verification(scoped=False)

    orchestrator._schedule_media_restart.assert_called_once()


async def test_the_step_prewarm_never_replaces_a_running_worker() -> None:
    """The next step's prewarm goes through the same gate as every other
    restart: a media worker running now is left alone, and a worker that is
    down is started."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=True)
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._step_prewarm_plan_id = "plan-next"

    orchestrator._begin_step_prewarm()

    orchestrator._schedule_media_restart.assert_not_called()
    _follow_media_lifecycle(processes)
    await processes.stop("media")
    orchestrator._step_prewarm_task = None
    orchestrator._begin_step_prewarm()
    orchestrator._schedule_media_restart.assert_called_once()


async def test_an_unreadable_status_after_a_broad_handoff_keeps_the_give_back_owed() -> None:
    """The broad worker was stopped and chat restored; when the give-back is
    decided, the supervisor's status cannot be read. Unknown is not a
    running worker: nothing starts, the obligation is neither met nor
    erased, and the pump owns it until a status can be read - a readable
    down worker then earns the restart."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=True)
    orchestrator._resume_chat_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
    orchestrator._schedule_media_restart = Mock()  # type: ignore[method-assign]
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    readable = processes.statuses
    stopped = {"n": 0}
    inner_stop = processes.stop

    async def stop_then_hide_the_status(name: str) -> None:
        await inner_stop(name)
        stopped["n"] += 1
        processes.statuses = Mock(side_effect=RuntimeError("supervisor unavailable"))

    processes.stop = AsyncMock(side_effect=stop_then_hide_the_status)

    await orchestrator._finish_media_handoff("profile-chat", "idle", None, job_id="job-media")

    assert stopped["n"] == 1
    orchestrator._schedule_media_restart.assert_not_called()
    assert orchestrator._media_give_back_owed is True
    assert orchestrator._media_give_back_attempts == 0
    assert orchestrator._media_restart_intent is None
    orchestrator._arm_settlement_retry.assert_called_once()

    processes.statuses = readable
    orchestrator._schedule_give_back(retry=True)

    orchestrator._schedule_media_restart.assert_called_once_with()
    assert orchestrator._media_give_back_owed is True, (
        "a scheduled restart pays on success, not now"
    )


async def test_a_continuation_failing_before_its_first_delta_still_gives_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed broad handoff bound its restart intent to the queued text
    job. When that job's stream raises before any delta, the intent is
    fired at the job's teardown - the handoff was fully paid, so nobody
    else owes the device back - and the media worker comes back."""

    from local_lm.adapters.base import ChatEvent

    order: list[str] = []
    orchestrator, processes = _text_dispatch_orchestrator(monkeypatch, order)
    with orchestrator.session_factory() as session:
        session.get(Run, "run-text").provenance_json = {}
        session.get(Run, "run-text").assistant_message_id = "assistant-text"
        session.get(Job, "job-text").progress_json = {}
    await processes.stop("media")
    processes.stop.reset_mock()
    orchestrator._media_restart_intent = "job-text"
    orchestrator._deferred_handoff = None

    async def stream(_request):  # type: ignore[no-untyped-def]
        raise RuntimeError("the model failed before its first token")
        yield ChatEvent(type="delta", text="", data={})  # pragma: no cover

    orchestrator.engines = SimpleNamespace(
        chat=SimpleNamespace(stream=stream), settings=SimpleNamespace(media_engine="comfyui")
    )
    orchestrator._set_chat_phase = AsyncMock()  # type: ignore[method-assign]
    orchestrator._ensure_chat_worker = AsyncMock(return_value=None)  # type: ignore[method-assign]
    orchestrator._prepare_chat_context = AsyncMock(  # type: ignore[method-assign]
        return_value=([], {}, {}, False)
    )
    orchestrator._persist_streamed_text = Mock()  # type: ignore[method-assign]
    orchestrator.scheduler.publish_job = AsyncMock()

    with suppress(RuntimeError):
        await orchestrator._execute("job-text", "run-text")
    restart = orchestrator._media_restart_task
    assert restart is not None, "the failed continuation did not give the device back"
    await restart

    assert orchestrator._media_restart_intent is None
    processes.start_media.assert_awaited_once()
    assert orchestrator._media_give_back_owed is False


async def test_a_continuation_failing_under_a_retained_debt_leaves_recovery_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a debt bound to the failing continuation is still retained, the
    settlement owns recovery: the intent is dropped without a start, and a
    broad restart does not race the settlement."""

    from local_lm.adapters.base import ChatEvent

    order: list[str] = []
    orchestrator, processes = _text_dispatch_orchestrator(monkeypatch, order)
    with orchestrator.session_factory() as session:
        session.get(Run, "run-text").provenance_json = {}
        session.get(Run, "run-text").assistant_message_id = "assistant-text"
        session.get(Job, "job-text").progress_json = {}
    await processes.stop("media")
    processes.stop.reset_mock()
    orchestrator._media_restart_intent = "job-text"
    orchestrator._deferred_handoff = _DeferredHandoff("profile-owed", "job-text", scoped=False)
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]

    async def stream(_request):  # type: ignore[no-untyped-def]
        raise RuntimeError("the model failed before its first token")
        yield ChatEvent(type="delta", text="", data={})  # pragma: no cover

    orchestrator.engines = SimpleNamespace(
        chat=SimpleNamespace(stream=stream), settings=SimpleNamespace(media_engine="comfyui")
    )
    orchestrator._set_chat_phase = AsyncMock()  # type: ignore[method-assign]
    orchestrator._ensure_chat_worker = AsyncMock(return_value=None)  # type: ignore[method-assign]
    orchestrator._prepare_chat_context = AsyncMock(  # type: ignore[method-assign]
        return_value=([], {}, {}, False)
    )
    orchestrator._persist_streamed_text = Mock()  # type: ignore[method-assign]
    orchestrator.scheduler.publish_job = AsyncMock()

    with suppress(RuntimeError):
        await orchestrator._execute("job-text", "run-text")

    assert orchestrator._media_restart_intent is None
    assert orchestrator._media_restart_task is None
    processes.start_media.assert_not_awaited()
    assert orchestrator._deferred_handoff is not None, "the retained debt must survive the failure"


async def test_a_prewarm_never_adopts_a_give_back_in_flight() -> None:
    """A give-back task minted by the first delta is owed to the device, not
    to the plan: a prewarm that finds it in flight does not become its
    owner, and the step's failure settles the prewarm without cancelling
    it."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    release = asyncio.Event()

    async def start_media(*_args: object, **_kwargs: object) -> None:
        await release.wait()

    processes.start_media = AsyncMock(side_effect=start_media)
    orchestrator._schedule_give_back()
    give_back = orchestrator._media_restart_task
    assert give_back is not None and not give_back.done()
    assert orchestrator._media_restart_owner == "give-back"

    orchestrator._step_prewarm_plan_id = "plan-next"
    orchestrator._begin_step_prewarm()
    assert orchestrator._step_prewarm_task is None, (
        "the prewarm adopted a give-back it did not mint"
    )
    assert orchestrator._media_restart_task is give_back

    await orchestrator._settle_step_prewarm("job-failed-step")
    await asyncio.sleep(0)
    assert not give_back.cancelled(), "the failed step cancelled the device's give-back"
    assert orchestrator._media_restart_task is give_back
    release.set()
    await give_back
    processes.start_media.assert_awaited_once()


async def test_a_cancelled_speculative_prewarm_mints_no_give_back() -> None:
    """A prewarm minted only to warm a queued successor is cut short when its
    step fails. No device was displaced: nothing is owed back, no pump is
    armed, and the successor is not held behind a restart it never
    needed."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    release = asyncio.Event()

    async def start_media(*_args: object, **_kwargs: object) -> None:
        await release.wait()

    processes.start_media = AsyncMock(side_effect=start_media)
    orchestrator._step_prewarm_plan_id = "plan-next"
    orchestrator._begin_step_prewarm()
    prewarm = orchestrator._step_prewarm_task
    assert prewarm is not None and orchestrator._media_restart_task is prewarm
    assert orchestrator._media_restart_owner == "prewarm"

    await orchestrator._settle_step_prewarm("job-failed-step")
    await asyncio.sleep(0)

    assert prewarm.cancelled()
    assert orchestrator._media_restart_task is None
    assert orchestrator._media_give_back_owed is False
    assert orchestrator._media_give_back_attempts == 0
    orchestrator._arm_settlement_retry.assert_not_called()


async def test_a_give_back_that_joined_a_prewarm_survives_its_cancellation() -> None:
    """A give-back that finds the prewarm's task in flight joins it. When the
    prewarm is cut short the device is still owed back: the joined give-back
    keeps the obligation, spends no attempt, and the pump owns it."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    release = asyncio.Event()

    async def start_media(*_args: object, **_kwargs: object) -> None:
        await release.wait()

    processes.start_media = AsyncMock(side_effect=start_media)
    orchestrator._step_prewarm_plan_id = "plan-next"
    orchestrator._begin_step_prewarm()
    prewarm = orchestrator._step_prewarm_task
    assert prewarm is not None and orchestrator._media_restart_task is prewarm

    orchestrator._schedule_give_back()

    assert orchestrator._media_restart_task is prewarm, "the give-back joined the prewarm"
    assert orchestrator._media_restart_owner == "prewarm"
    assert orchestrator._media_give_back_joined is True

    await orchestrator._settle_step_prewarm("job-failed-step")
    await asyncio.sleep(0)

    assert prewarm.cancelled()
    assert orchestrator._media_restart_task is None
    assert orchestrator._media_give_back_owed is True
    assert orchestrator._media_give_back_attempts == 0
    orchestrator._arm_settlement_retry.assert_called_once()

    # The join belongs to that task alone: once the give-back is paid, a
    # fresh speculative prewarm cut short still mints nothing.
    orchestrator._media_give_back_owed = False
    orchestrator._arm_settlement_retry.reset_mock()
    orchestrator._step_prewarm_plan_id = "plan-after"
    orchestrator._begin_step_prewarm()
    again = orchestrator._step_prewarm_task
    assert again is not None and again is not prewarm
    assert orchestrator._media_give_back_joined is False

    await orchestrator._settle_step_prewarm("job-failed-again")
    await asyncio.sleep(0)

    assert again.cancelled()
    assert orchestrator._media_give_back_owed is False
    orchestrator._arm_settlement_retry.assert_not_called()


async def test_a_broad_restart_superseded_in_flight_owes_nothing_to_the_scoped_start() -> None:
    """A scoped ensure supersedes a broad restart already in flight, waits
    for it, and starts its own scope. When that broad restart then fails,
    it belongs to a past generation: no debt is written back, no pump is
    armed, and the scoped start is the only start that follows."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    release = asyncio.Event()
    starts: list[object] = []

    async def start_media(*_args: object, **kwargs: object) -> None:
        starts.append(kwargs.get("activation_scope"))
        if kwargs.get("activation_scope") is None:
            await release.wait()
            raise RuntimeError("the broad restart failed after it was superseded")

    processes.start_media = AsyncMock(side_effect=start_media)
    orchestrator._schedule_give_back()
    broad = orchestrator._media_restart_task
    assert broad is not None and not broad.done()
    generation_before = orchestrator._media_give_back_generation

    scope = SimpleNamespace(launch_sha256="scope-next")
    ensure = asyncio.create_task(
        orchestrator._ensure_media_worker(activation_scope=scope)  # type: ignore[arg-type]
    )
    await asyncio.sleep(0)
    assert orchestrator._media_give_back_generation > generation_before
    release.set()
    await ensure

    assert broad.done() and not broad.cancelled()
    assert orchestrator._media_give_back_owed is False
    assert orchestrator._media_give_back_attempts == 0
    orchestrator._arm_settlement_retry.assert_not_called()
    assert not orchestrator._give_back_pending()
    assert starts == [None, scope]


async def test_a_superseded_restarts_failure_writes_no_debt_back() -> None:
    """Scoped truth arrives while a broad restart is in flight - a scoped
    stop or launch supersedes the give-back - and the broad restart then
    fails. It belonged to a past generation: no debt is written back, no
    attempt is spent, and nothing is pending for the pump."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    release = asyncio.Event()
    started = asyncio.Event()

    async def start_media(*_args: object, **_kwargs: object) -> None:
        started.set()
        await release.wait()
        raise RuntimeError("the broad restart failed after it was superseded")

    processes.start_media = AsyncMock(side_effect=start_media)
    orchestrator._schedule_give_back()
    broad = orchestrator._media_restart_task
    assert broad is not None and not broad.done()
    # The start is in flight - past the fence read under the lease - when
    # scoped truth arrives.
    await asyncio.wait_for(started.wait(), timeout=5)

    orchestrator._supersede_give_back()
    release.set()
    await asyncio.gather(broad, return_exceptions=True)
    await asyncio.sleep(0)

    assert orchestrator._media_give_back_owed is False
    assert orchestrator._media_give_back_attempts == 0
    assert not orchestrator._give_back_pending()
    orchestrator._arm_settlement_retry.assert_not_called()


async def _settle(task: asyncio.Task[None]) -> None:
    """Await a restart task and let its done callback run."""

    await task
    await asyncio.sleep(0)


async def test_a_worker_running_by_the_time_the_restart_fires_is_the_device_given_back() -> None:
    """The give-back was scheduled against a worker that was down; by the
    time its task runs a worker is running. The fence is read at the
    effect, under the device lease: nothing starts, nothing is replaced,
    and the obligation is met."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    orchestrator._media_give_back_owed = True

    orchestrator._schedule_give_back(retry=True)
    task = orchestrator._media_restart_task
    assert task is not None and not task.done()
    processes.statuses = Mock(
        return_value=[WorkerStatus(name="media", state="ready", managed=True, running=True)]
    )
    await _settle(task)

    processes.start_media.assert_not_awaited()
    processes.stop.assert_not_awaited()
    assert orchestrator._media_give_back_owed is False
    assert orchestrator._media_give_back_attempts == 0
    orchestrator._arm_settlement_retry.assert_not_called()


async def test_a_broad_start_that_succeeds_for_a_superseded_generation_is_undone() -> None:
    """Scoped truth arrives while the broad start is in flight. The start
    succeeds for a generation nobody wants any more: the broad worker it
    produced is stopped rather than left standing in for a scope it does
    not carry, and the superseded task writes nothing back."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    orchestrator._media_give_back_owed = True
    release = asyncio.Event()
    inner_start = processes.start_media

    async def start_media(*args: object, **kwargs: object) -> None:
        await release.wait()
        await inner_start(*args, **kwargs)

    processes.start_media = AsyncMock(side_effect=start_media)

    orchestrator._schedule_give_back(retry=True)
    task = orchestrator._media_restart_task
    assert task is not None
    for _ in range(5):
        await asyncio.sleep(0)
    assert not task.done()
    orchestrator._supersede_give_back()
    release.set()
    await _settle(task)

    processes.start_media.assert_awaited_once()
    processes.stop.assert_awaited_once_with("media")
    assert orchestrator._media_give_back_owed is False
    assert orchestrator._media_give_back_attempts == 0
    orchestrator._arm_settlement_retry.assert_not_called()


async def test_a_leased_device_defers_the_give_back_to_the_pump() -> None:
    """Under the real scheduler's lock, a dispatch holds the primary lease
    while the give-back's task fires. The task starts nothing beside the
    holder: the obligation stays owed, spends no attempt, and is handed to
    the pump, which pays it once the device is free."""

    from local_lm.scheduler import ResourceScheduler

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    scheduler = ResourceScheduler(None, session_factory=lambda: None)  # type: ignore[arg-type,return-value]
    orchestrator.scheduler.try_lease = scheduler.try_lease
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    orchestrator._media_give_back_owed = True

    async with scheduler.lease("primary"):
        orchestrator._schedule_give_back(retry=True)
        task = orchestrator._media_restart_task
        assert task is not None
        await _settle(task)
        processes.start_media.assert_not_awaited()
        assert orchestrator._media_give_back_owed is True
        assert orchestrator._media_give_back_attempts == 0
        orchestrator._arm_settlement_retry.assert_called_once()

    orchestrator._schedule_give_back(retry=True)
    task = orchestrator._media_restart_task
    assert task is not None
    await _settle(task)

    processes.start_media.assert_awaited_once()
    assert orchestrator._media_give_back_owed is False
    assert orchestrator._media_give_back_attempts == 0


async def test_an_unreadable_status_at_the_effect_keeps_the_give_back_owed() -> None:
    """The status was readable when the give-back was scheduled and cannot
    be read when its task fires. Unknown at the effect is not a down
    worker: nothing starts, no attempt is spent, and the pump reads again."""

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    orchestrator._media_give_back_owed = True

    orchestrator._schedule_give_back(retry=True)
    task = orchestrator._media_restart_task
    assert task is not None and not task.done()
    processes.statuses = Mock(side_effect=RuntimeError("supervisor unavailable"))
    await _settle(task)

    processes.start_media.assert_not_awaited()
    assert orchestrator._media_give_back_owed is True
    assert orchestrator._media_give_back_attempts == 0
    orchestrator._arm_settlement_retry.assert_called_once()


async def test_a_paid_debt_with_an_unreadable_status_keeps_its_give_back_through_a_takeover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recycle was paid earlier - media is already down - and only the
    status read fails under the text takeover. Chat may load, and the
    takeover's discharge clears the paid debt; the device is still owed
    back, so the give-back is retained on its own books, the pump owns it,
    and exactly one restart follows once the status can be read."""

    order: list[str] = []
    orchestrator, processes = _text_dispatch_orchestrator(monkeypatch, order)
    processes.statuses = Mock(side_effect=RuntimeError("supervisor unavailable"))
    orchestrator._set_chat_phase = AsyncMock()  # type: ignore[method-assign]
    orchestrator._ensure_chat_worker = AsyncMock()  # type: ignore[method-assign]
    orchestrator._execute_chat = AsyncMock()  # type: ignore[method-assign]
    orchestrator._deferred_handoff = _DeferredHandoff(
        "profile-owed", None, recycle_paid=True, scoped=False
    )

    await orchestrator._execute("job-text", "run-text")

    orchestrator._execute_chat.assert_awaited_once()
    assert orchestrator._deferred_handoff is None, "the paid debt is discharged by the takeover"
    assert orchestrator._media_give_back_owed is True, "the give-back died with the paid debt"
    retry = orchestrator._settlement_retry_task
    assert retry is not None and not retry.done()
    retry.cancel()
    with suppress(asyncio.CancelledError):
        await retry
    processes.start_media.assert_not_awaited()

    processes.statuses = Mock(
        return_value=[WorkerStatus(name="media", state="stopped", managed=True, running=False)]
    )
    orchestrator._schedule_give_back(retry=True)
    task = orchestrator._media_restart_task
    assert task is not None
    await _settle(task)

    processes.start_media.assert_awaited_once()
    assert orchestrator._media_give_back_owed is False


async def test_a_leased_device_defers_the_give_back_even_at_the_attempt_bound() -> None:
    """A deferral spends nothing: with every attempt already spent, a
    give-back that finds the device leased is still handed to the pump,
    where a spent attempt would have been abandoned."""

    from local_lm.scheduler import ResourceScheduler

    orchestrator, processes = _handoff_orchestrator(None, media_running=False)
    scheduler = ResourceScheduler(None, session_factory=lambda: None)  # type: ignore[arg-type,return-value]
    orchestrator.scheduler.try_lease = scheduler.try_lease
    orchestrator._arm_settlement_retry = Mock()  # type: ignore[method-assign]
    orchestrator._media_give_back_owed = True
    orchestrator._media_give_back_attempts = orchestrator._GIVE_BACK_ATTEMPTS

    async with scheduler.lease("primary"):
        orchestrator._schedule_media_restart()
        task = orchestrator._media_restart_task
        assert task is not None
        await _settle(task)

    processes.start_media.assert_not_awaited()
    assert orchestrator._media_give_back_owed is True
    assert orchestrator._media_give_back_attempts == orchestrator._GIVE_BACK_ATTEMPTS
    orchestrator._arm_settlement_retry.assert_called_once()
