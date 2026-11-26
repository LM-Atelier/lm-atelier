"""The orchestrator names why it stops and starts the media worker.

A run's own handoff recycles ComfyUI to give the device back to chat, and how
the run ended decides the code: a run that did not finish with its result and
was not allowed to keep its worker is a failed run's handoff, and one that was
allowed but could not keep it is an ordinary handoff. A hold that a run's
handoff opens for someone else's use is recycled under that handoff's code once
they are done, while a window kept for a following media run, including one
held while that run waits, ends as its expiry. A chat model load inside a
window, an edit check loading its vision model, the restart that brings the
worker back, and a run or the step before it starting the worker each name
their own.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from run_waits import PATIENCE_SECONDS
from test_image_edit_verification_orchestration import _TEST_CLAIM as VERIFICATION_CLAIM
from test_image_edit_verification_orchestration import _verification_world
from test_media_warm_window import _settled, _warm_orchestrator

from local_lm.models import Run
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.scheduler import JobClaim
from local_lm.schemas import WorkerStartReason, WorkerStatus, WorkerStopReason
from local_lm.workflow_activations import WorkflowActivationLaunchScope

_CLAIM = JobClaim(token="neutral-attempt", attempt=1)
_TEXT_NEXT = SimpleNamespace(operation="text", profile_id="profile-next")


def _double(**members: object) -> Mock:
    return Mock(spec_set=list(members), **members)


def _handoff_world(
    monkeypatch: pytest.MonkeyPatch,
    *,
    next_run: object | None = None,
    keep_warm_seconds: float = 0,
) -> tuple[ConversationOrchestrator, Mock]:
    """A ready managed ComfyUI worker, chat down, and the next queued job if any."""

    chat = WorkerStatus(name="chat", state="stopped", managed=True, running=False)
    media = WorkerStatus(name="media", state="ready", managed=True, running=True, pid=22)

    class FakeSession:
        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, model: object, identity: str) -> object | None:
            return next_run if model is Run and identity == "run-next" else None

    @asynccontextmanager
    async def lease(_device: str = "primary") -> AsyncIterator[None]:
        yield

    processes = _double(
        statuses=Mock(return_value=[chat, media]), stop=AsyncMock(), start_media=AsyncMock()
    )
    scheduler = _double(
        peek_next_eligible_job=Mock(
            return_value=("job-next", "run-next") if next_run is not None else None
        ),
        lease=lease,
    )
    orchestrator = ConversationOrchestrator(
        engines=_double(
            settings=SimpleNamespace(
                media_engine="comfyui", media_keep_warm_seconds=keep_warm_seconds
            )
        ),
        artifacts=Mock(),
        events=Mock(),
        scheduler=scheduler,
        processes=processes,
        session_factory=cast(Any, FakeSession),
    )
    monkeypatch.setattr(orchestrator, "_resume_chat_worker", AsyncMock())
    return orchestrator, processes


async def _restart_settled(orchestrator: ConversationOrchestrator) -> None:
    task = orchestrator._media_restart_task
    if task is not None:
        await task


async def test_a_run_not_allowed_to_keep_its_worker_is_recycled_as_a_failed_run_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, processes = _handoff_world(monkeypatch, keep_warm_seconds=120)

    await orchestrator._complete_media_handoff("profile-chat", keep_warm=False)
    await _restart_settled(orchestrator)

    processes.stop.assert_awaited_once_with("media", reason=WorkerStopReason.FAILED_RUN_HANDOFF)
    processes.start_media.assert_awaited_once_with(reason=WorkerStartReason.HANDOFF_RESTART)


@pytest.mark.parametrize(
    ("next_run", "keep_warm_seconds"),
    [(_TEXT_NEXT, 120), (None, 0)],
    ids=["text-run-next", "keeping-warm-turned-off"],
)
async def test_a_run_whose_worker_cannot_be_kept_is_recycled_as_a_run_handoff(
    monkeypatch: pytest.MonkeyPatch, next_run: object | None, keep_warm_seconds: float
) -> None:
    orchestrator, processes = _handoff_world(
        monkeypatch, next_run=next_run, keep_warm_seconds=keep_warm_seconds
    )

    await orchestrator._complete_media_handoff("profile-chat", keep_warm=True)
    await _restart_settled(orchestrator)

    processes.stop.assert_awaited_once_with("media", reason=WorkerStopReason.RUN_HANDOFF)
    if next_run is None:
        processes.start_media.assert_awaited_once_with(reason=WorkerStartReason.HANDOFF_RESTART)
    else:
        # The text run goes first, and the worker is owed back once it is active.
        processes.start_media.assert_not_awaited()
        assert orchestrator._media_restart_after_chat_activity is True


async def test_a_retried_attempt_recycled_for_a_text_run_queued_next_is_a_run_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, processes = _handoff_world(monkeypatch, next_run=_TEXT_NEXT)

    await orchestrator._hold_media_handoff_for_retry("profile-chat")

    processes.stop.assert_awaited_once_with("media", reason=WorkerStopReason.RUN_HANDOFF)


@pytest.mark.parametrize("owed", [None, "profile-chat"], ids=["nothing-owed", "chat-owed"])
async def test_the_end_of_a_warm_window_is_counted_as_its_expiry(
    monkeypatch: pytest.MonkeyPatch, owed: str | None
) -> None:
    orchestrator, processes = _handoff_world(monkeypatch)
    orchestrator._media_warm_window = 1
    orchestrator._media_warm_launch = 22
    orchestrator._displaced_chat_profile_id = owed

    await orchestrator._end_media_warm_window()
    await _restart_settled(orchestrator)

    processes.stop.assert_awaited_once_with("media", reason=WorkerStopReason.KEEP_WARM_EXPIRY)
    processes.start_media.assert_awaited_once_with(reason=WorkerStartReason.HANDOFF_RESTART)


async def test_a_chat_model_load_inside_a_warm_window_is_counted_as_a_chat_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, processes = _handoff_world(monkeypatch)
    orchestrator._media_warm_window = 1
    orchestrator._media_warm_generation = 1
    orchestrator._media_warm_launch = 22

    await orchestrator.release_media_for_chat()
    settlement = orchestrator._media_warm_task
    if settlement is not None:
        await settlement

    processes.stop.assert_awaited_once_with("media", reason=WorkerStopReason.CHAT_LOAD)


async def test_an_edit_check_stopping_comfyui_for_its_vision_model_is_a_chat_load() -> None:
    job, orchestrator, _world = _verification_world(lose_at="never")

    await orchestrator._execute_image_edit_verification(job.id, VERIFICATION_CLAIM)

    orchestrator.processes.stop.assert_awaited_once_with("media", reason=WorkerStopReason.CHAT_LOAD)


@pytest.mark.parametrize("scoped", [False, True], ids=["any-workflow", "one-activation"])
@pytest.mark.parametrize("announced", [False, True], ids=["unannounced", "announced"])
async def test_a_run_starts_the_media_worker_under_run(
    monkeypatch: pytest.MonkeyPatch, scoped: bool, announced: bool
) -> None:
    media = WorkerStatus(name="media", state="stopped", managed=False, running=False)
    processes = _double(statuses=Mock(return_value=[media]), start_media=AsyncMock())
    orchestrator = ConversationOrchestrator(
        engines=_double(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    monkeypatch.setattr(orchestrator, "_set_media_phase", AsyncMock(return_value=True))
    scope = (
        WorkflowActivationLaunchScope(
            "wfact-one", "wfrev-one", "a" * 64, "b" * 64, (), (), (), (), (), (), (), (), (), ()
        )
        if scoped
        else None
    )

    await orchestrator._ensure_media_worker(
        claim=_CLAIM,
        job_id="job-media" if announced else None,
        run_id="run-media" if announced else None,
        activation_scope=scope,
    )

    processes.start_media.assert_awaited_once()
    options = processes.start_media.await_args.kwargs
    assert options["reason"] is WorkerStartReason.RUN
    assert options.get("activation_scope") is scope


async def test_starting_the_worker_for_the_next_step_of_a_plan_is_counted_as_a_run() -> None:
    """The prewarm serves the media step that follows, not a device chat borrowed."""

    media = WorkerStatus(name="media", state="stopped", managed=False, running=False)
    processes = _double(statuses=Mock(return_value=[media]), start_media=AsyncMock())
    orchestrator = ConversationOrchestrator(
        engines=_double(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    orchestrator._step_prewarm_plan_id = "plan-next"

    orchestrator._begin_step_prewarm()
    prewarm = orchestrator._step_prewarm_task
    assert prewarm is not None
    await prewarm

    processes.start_media.assert_awaited_once_with(reason=WorkerStartReason.RUN)


def _recorded_stop_reasons(
    orchestrator: ConversationOrchestrator, monkeypatch: pytest.MonkeyPatch
) -> list[object]:
    reasons: list[object] = []
    stop = orchestrator.processes.stop

    async def recording_stop(name: str, **options: Any) -> None:
        reasons.append(options.get("reason"))
        await stop(name, **options)

    monkeypatch.setattr(orchestrator.processes, "stop", recording_stop)
    return reasons


@pytest.mark.parametrize(
    ("keep_warm_seconds", "finished", "counted"),
    [
        (60, False, WorkerStopReason.FAILED_RUN_HANDOFF),
        (0, True, WorkerStopReason.RUN_HANDOFF),
    ],
    ids=["after-a-failed-run", "keeping-warm-turned-off"],
)
async def test_a_worker_held_for_someone_else_is_recycled_as_the_handoff_it_deferred(
    monkeypatch: pytest.MonkeyPatch,
    keep_warm_seconds: float,
    finished: bool,
    counted: WorkerStopReason,
) -> None:
    monkeypatch.setattr("local_lm.orchestrator.MEDIA_BUSY_RECHECK_SECONDS", 0.01)
    orchestrator, record, _ = _warm_orchestrator(keep_warm_seconds, counts_foreign=True)
    reasons = _recorded_stop_reasons(orchestrator, monkeypatch)
    # Someone queued a prompt in ComfyUI's own page while the run was going.
    record["comfy_queue"] = 1
    record["foreign"] = 1

    await orchestrator._complete_media_handoff("profile-chat", keep_warm=finished)

    assert record["order"] == [], "the worker was recycled under someone else's work"
    assert orchestrator._media_warm_window is not None
    record["foreign"] = 0
    await asyncio.wait_for(_settled(orchestrator), PATIENCE_SECONDS)
    assert record["order"] == ["stop media", "resume profile-chat", "start media"]
    assert reasons == [counted]
    await orchestrator.close()


async def test_a_window_kept_for_a_following_media_run_still_ends_as_its_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, record, _ = _warm_orchestrator(0.01)
    reasons = _recorded_stop_reasons(orchestrator, monkeypatch)

    await orchestrator._complete_media_handoff("profile-chat")
    await asyncio.wait_for(_settled(orchestrator), PATIENCE_SECONDS)

    assert record["order"] == ["stop media", "resume profile-chat", "start media"]
    assert reasons == [WorkerStopReason.KEEP_WARM_EXPIRY]
    await orchestrator.close()
