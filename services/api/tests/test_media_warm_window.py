"""A media worker stays warm for a following media turn instead of reloading per turn.

Each media run that ended with nothing queued handed the device straight back:
ComfyUI was stopped and the chat model it displaced was loaded again. A sequence
of video turns, each sent as the previous one finished, then paid a chat model
load, a ComfyUI start and a cold load of every model it used, every time. The
worker is now kept for a while after a successful run. The window's expiry hands
the device back as before, and the next media run takes the worker over. A chat
model load from any path recycles the worker first, so a chat model is never
loaded beside ComfyUI's resident models, and once the device is free whatever
that load left owed is given back. The window keeps one launch and no other: a
worker someone else stopped, restarted or is using directly is left to them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS, wait_until
from test_chat_handoff_across_retries import ChatWorker, finished
from test_generation_retry import choose_retries, failing_media, read_run, start_media
from test_workflow_editor_api import _create_workflow, _ready_editor

from local_lm.adapters.base import ChatEvent, ChatRequest, MediaEvent, MediaRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import ModelInstall, ModelProfile
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.processes import ProcessSupervisor
from local_lm.schemas import WorkerStatus


def _warm_orchestrator(
    seconds: float, *, scope: str | None = None
) -> tuple[ConversationOrchestrator, dict[str, Any], asyncio.Lock]:
    record: dict[str, Any] = {
        "order": [],
        "chat_running": False,
        "media_running": True,
        "media_pid": 31,
        "comfy_queue": 0,
        "stop_gate": None,
        "probe_gate": None,
        "probes": 0,
        "media_state": None,
    }
    # Nothing is queued, and the device lease is a lock a test holds the way a
    # running job, or any caller loading a chat model, holds it.
    device = asyncio.Lock()

    @asynccontextmanager
    async def lease(_device_id: str = "primary") -> AsyncIterator[None]:
        async with device:
            yield

    def worker(name: str) -> WorkerStatus:
        running = record[f"{name}_running"]
        pid = (30 if name == "chat" else record["media_pid"]) if running else None
        state = record["media_state"] if name == "media" and record["media_state"] else None
        return WorkerStatus(
            name=name,
            state=state or ("ready" if running else "stopped"),
            managed=True,
            running=running,
            pid=pid,
        )

    async def stop(name: str) -> None:
        record["order"].append(f"stop {name}")
        gate = record["stop_gate"]
        if name == "media" and gate is not None:
            await gate.wait()
        record[f"{name}_running"] = False

    async def start_media() -> None:
        record["order"].append("start media")
        record["media_running"] = True
        record["media_pid"] += 1

    async def queue_remaining() -> int:
        record["probes"] += 1
        gate = record["probe_gate"]
        if gate is not None:
            await gate.wait()
        return int(record["comfy_queue"])

    processes = Mock(
        spec_set=["settings", "statuses", "stop", "start_media", "launch_scope_sha256"],
        settings=SimpleNamespace(auto_unload_chat_for_media=True),
        statuses=lambda: [worker("chat"), worker("media")],
        stop=stop,
        start_media=start_media,
        # Like the supervisor, a scope is reported only for a live worker.
        launch_scope_sha256=lambda _name: scope if record["media_running"] else None,
    )
    orchestrator = ConversationOrchestrator(
        engines=Mock(
            spec_set=["settings", "media"],
            settings=SimpleNamespace(media_engine="comfyui", media_keep_warm_seconds=seconds),
            media=Mock(spec_set=["queue_remaining"], queue_remaining=queue_remaining),
        ),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(
            spec_set=["lease", "peek_next_eligible_job"],
            lease=lease,
            peek_next_eligible_job=lambda _group: None,
        ),
        processes=processes,
    )

    async def resume(profile_id: str) -> None:
        record["order"].append(f"resume {profile_id}")
        record["chat_running"] = True

    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]
    return orchestrator, record, device


async def _settled(orchestrator: ConversationOrchestrator) -> None:
    """Let the window's own task, and any media restart it started, finish."""

    task = orchestrator._media_warm_task
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)
    restart = orchestrator._media_restart_task
    if restart is not None:
        await restart


async def test_a_media_run_with_nothing_queued_keeps_the_worker_warm() -> None:
    orchestrator, record, _ = _warm_orchestrator(60)
    # The run that displaced chat cleared the readiness latch, as it does.
    orchestrator._chat_planner_ready.clear()

    await orchestrator._complete_media_handoff("profile-chat")

    # Nothing was recycled and chat was not reloaded; it stays owed.
    assert record["order"] == []
    assert orchestrator._media_warm_window is not None
    assert orchestrator._displaced_chat_profile_id == "profile-chat"
    # The handoff is over, so planning reads the chat worker's real state again.
    assert orchestrator._chat_planner_ready.is_set()
    await orchestrator.close()


async def test_an_unused_window_hands_the_device_back_to_chat() -> None:
    orchestrator, record, _ = _warm_orchestrator(0.01)

    await orchestrator._complete_media_handoff("profile-chat")
    await _settled(orchestrator)

    assert record["order"] == ["stop media", "resume profile-chat", "start media"]
    assert orchestrator._displaced_chat_profile_id is None
    assert orchestrator._media_warm_window is None
    await orchestrator.close()


async def test_an_expiry_with_nothing_owed_still_recycles_the_warm_worker() -> None:
    orchestrator, record, _ = _warm_orchestrator(0.01)
    await orchestrator._complete_media_handoff("profile-chat")
    # A text turn that never reached a chat load clears the record on its way out.
    orchestrator._displaced_chat_profile_id = None

    await _settled(orchestrator)

    # The resident models still go, so no later chat load lands beside them.
    assert record["order"] == ["stop media", "start media"]
    assert orchestrator._media_warm_window is None
    await orchestrator.close()


async def test_the_next_media_run_takes_the_warm_worker_over() -> None:
    orchestrator, record, _ = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")
    timer = orchestrator._media_warm_task
    assert timer is not None
    # Owed by an earlier recycle; the run starts its worker itself.
    orchestrator._media_restart_after_chat_activity = True

    displaced = await orchestrator._prepare_device_handoff(
        Operation.TEXT_TO_VIDEO.value, claim=Mock()
    )
    await asyncio.gather(timer, return_exceptions=True)

    # Chat was already down, and nothing may recycle the worker under the run.
    assert displaced is None
    assert timer.cancelled()
    assert orchestrator._media_warm_window is None
    assert orchestrator._media_restart_after_chat_activity is False
    assert record["order"] == []
    # The run's own handoff keeps the worker for the turn after it.
    await orchestrator._complete_media_handoff(
        orchestrator._pending_chat_restore(displaced) or "missing"
    )
    assert orchestrator._media_warm_window is not None
    assert orchestrator._displaced_chat_profile_id == "profile-chat"
    assert record["order"] == []
    await orchestrator.close()


async def test_a_text_run_leaves_the_window_to_its_chat_load() -> None:
    orchestrator, record, _ = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")
    window = orchestrator._media_warm_window

    assert await orchestrator._prepare_device_handoff(Operation.TEXT.value, claim=Mock()) is None

    assert orchestrator._media_warm_window == window
    assert record["order"] == []
    await orchestrator.close()


async def test_a_stale_expiry_moves_no_worker() -> None:
    orchestrator, record, _ = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")
    current = orchestrator._media_warm_window
    assert current is not None

    await orchestrator._expire_media_warm_window(current - 1, 0)

    assert record["order"] == []
    assert orchestrator._media_warm_window == current
    await orchestrator.close()


async def test_a_chat_load_inside_the_window_recycles_the_worker_first() -> None:
    orchestrator, record, device = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")
    timer = orchestrator._media_warm_task
    assert timer is not None

    async with device:
        await orchestrator.release_media_for_chat()
        await asyncio.gather(timer, return_exceptions=True)
        assert record["order"] == ["stop media"]
        assert timer.cancelled()
        assert orchestrator._media_warm_window is None
        assert orchestrator._media_restart_after_chat_activity is True
        # A second load finds no window and moves nothing.
        await orchestrator.release_media_for_chat()
        assert record["order"] == ["stop media"]
        # The load chose and loaded its own model, as an explicit load does.
        record["chat_running"] = True
        orchestrator.forget_displaced_chat_profile()
    await _settled(orchestrator)

    # Once the device is free the empty worker comes back, as after any recycle.
    assert record["order"] == ["stop media", "start media"]
    assert orchestrator._media_restart_after_chat_activity is False
    await orchestrator.close()


async def test_a_load_that_only_borrowed_the_device_gives_both_workers_back() -> None:
    orchestrator, record, device = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")

    async with device:
        # Like proving a newly installed model: chat is down again afterwards.
        await orchestrator.release_media_for_chat()
        assert record["order"] == ["stop media"]
    await _settled(orchestrator)

    assert record["order"] == ["stop media", "resume profile-chat", "start media"]
    assert orchestrator._displaced_chat_profile_id is None
    await orchestrator.close()


async def test_a_chat_load_cancelled_while_comfyui_stops_still_settles() -> None:
    orchestrator, record, device = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")
    record["stop_gate"] = asyncio.Event()

    async with device:
        release = asyncio.create_task(orchestrator.release_media_for_chat())
        while record["order"] != ["stop media"]:
            await asyncio.sleep(0)
        # Stop is pressed on the text turn while ComfyUI is still going down.
        release.cancel()
        await asyncio.gather(release, return_exceptions=True)
        assert release.cancelled()
        # The restart was owed and the settlement made before the stop was awaited.
        assert orchestrator._media_restart_after_chat_activity is True
        assert orchestrator._media_warm_task is not None
        # The text turn's own exit then releases the restart and clears the record.
        record["media_running"] = False
        orchestrator._release_deferred_media_restart()
        orchestrator._displaced_chat_profile_id = None
    await _settled(orchestrator)

    # Chat comes back with the profile owed when the load began, and so does media.
    assert "resume profile-chat" in record["order"]
    assert "start media" in record["order"]
    assert record["chat_running"] and record["media_running"]
    await orchestrator.close()


async def test_a_scoped_worker_is_not_kept_warm() -> None:
    orchestrator, record, _ = _warm_orchestrator(60, scope="a" * 64)

    await orchestrator._complete_media_handoff("profile-chat")

    # The ordinary handoff: recycled, chat back, and no broad worker in its place.
    assert record["order"] == ["stop media", "resume profile-chat"]
    assert orchestrator._media_warm_window is None
    assert orchestrator._media_restart_task is None
    await orchestrator.close()


async def test_a_worker_another_caller_stopped_is_left_to_that_caller() -> None:
    orchestrator, record, device = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")

    async with device:
        # The caller borrowed the device itself and brings the worker back itself.
        record["media_running"] = False
        await orchestrator.release_media_for_chat()
        record["chat_running"] = True
        orchestrator.forget_displaced_chat_profile()
    await _settled(orchestrator)

    assert record["order"] == []
    assert orchestrator._media_restart_after_chat_activity is False
    await orchestrator.close()


async def test_a_chat_load_leaves_a_worker_someone_restarted_alone() -> None:
    orchestrator, record, device = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")
    # Restarted by hand inside the window: a new, empty launch.
    record["media_pid"] = 99

    async with device:
        await orchestrator.release_media_for_chat()
        record["chat_running"] = True
        orchestrator.forget_displaced_chat_profile()
    await _settled(orchestrator)

    assert record["order"] == []
    assert record["media_running"] and record["media_pid"] == 99
    await orchestrator.close()


async def test_a_worker_restarted_inside_the_window_is_left_running() -> None:
    orchestrator, record, _ = _warm_orchestrator(0.01)
    await orchestrator._complete_media_handoff("profile-chat")
    record["media_pid"] = 99

    await _settled(orchestrator)

    # Only the displaced chat model is owed; the new launch is not recycled.
    assert record["order"] == ["resume profile-chat"]
    assert record["media_running"] and record["media_pid"] == 99
    assert orchestrator._media_warm_window is None
    await orchestrator.close()


async def test_a_worker_stopped_inside_the_window_stays_stopped() -> None:
    orchestrator, record, _ = _warm_orchestrator(0.01)
    await orchestrator._complete_media_handoff("profile-chat")
    # Stopped by hand inside the window.
    record["media_running"] = False

    await _settled(orchestrator)

    assert record["order"] == ["resume profile-chat"]
    assert not record["media_running"]
    await orchestrator.close()


async def test_the_window_waits_while_comfyui_runs_a_prompt_of_its_own() -> None:
    orchestrator, record, _ = _warm_orchestrator(0.01)
    await orchestrator._complete_media_handoff("profile-chat")
    # Someone queued a prompt in ComfyUI's own page.
    record["comfy_queue"] = 1

    await asyncio.sleep(0.1)
    assert record["order"] == []
    assert orchestrator._media_warm_window is not None

    record["comfy_queue"] = 0
    await _settled(orchestrator)
    assert record["order"] == ["stop media", "resume profile-chat", "start media"]
    await orchestrator.close()


async def test_a_media_run_supersedes_what_a_chat_load_left_to_settle() -> None:
    orchestrator, record, device = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")

    async with device:
        await orchestrator.release_media_for_chat()
        settle = orchestrator._media_warm_task
        assert settle is not None
        await orchestrator._prepare_device_handoff(Operation.TEXT_TO_VIDEO.value, claim=Mock())
    await asyncio.gather(settle, return_exceptions=True)

    assert settle.cancelled()
    assert record["order"] == ["stop media"]
    assert orchestrator._media_restart_after_chat_activity is False
    await orchestrator.close()


async def test_a_window_needs_a_ready_managed_worker_and_a_positive_length() -> None:
    orchestrator, record, _ = _warm_orchestrator(0)
    await orchestrator._complete_media_handoff("profile-chat")
    # Turned off: the device goes straight back, as before.
    assert record["order"][:2] == ["stop media", "resume profile-chat"]
    assert orchestrator._media_warm_window is None
    await orchestrator.close()

    orchestrator, record, _ = _warm_orchestrator(60)
    record["media_running"] = False
    await orchestrator._complete_media_handoff("profile-chat")
    # Nothing warm to keep, so chat comes back at once.
    assert "resume profile-chat" in record["order"]
    assert orchestrator._media_warm_window is None
    await orchestrator.close()


async def test_a_failed_run_hands_the_device_back_at_once() -> None:
    orchestrator, record, _ = _warm_orchestrator(60)

    await orchestrator._complete_media_handoff("profile-chat", keep_warm=False)

    assert record["order"][:2] == ["stop media", "resume profile-chat"]
    assert orchestrator._media_warm_window is None
    await orchestrator.close()


async def test_only_a_run_that_succeeded_may_keep_its_worker_warm(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await choose_retries(client, 0)
    chat = ChatWorker(app, monkeypatch)
    failing_media(app, monkeypatch, 1)
    orchestrator = app.state.services.orchestrator
    original = orchestrator._complete_media_handoff
    handoffs: list[bool] = []

    async def complete(chat_profile_id: str, *, keep_warm: bool = True) -> None:
        handoffs.append(keep_warm)
        await original(chat_profile_id, keep_warm=keep_warm)

    monkeypatch.setattr(orchestrator, "_complete_media_handoff", complete)

    await finished(client, await start_media(client, "video"), "failed")
    assert handoffs == [False]
    await finished(client, await start_media(client, "video"))
    await chat.settled_order()

    assert handoffs == [False, True]


async def test_shutdown_cancels_an_open_window() -> None:
    orchestrator, record, _ = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")
    timer = orchestrator._media_warm_task
    assert timer is not None

    await orchestrator.close()

    assert timer.cancelled()
    assert orchestrator._media_warm_window is None
    assert record["order"] == []


async def test_every_chat_load_awaits_the_supervisors_hook_first(settings: Settings) -> None:
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    calls: list[str] = []

    async def before() -> None:
        calls.append("released")

    supervisor.before_chat_load = before
    install = ModelInstall(
        name="Unmanaged chat model", role="chat", engine="mock", local_path="managed/unmanaged"
    )
    profile = ModelProfile(name="Unmanaged chat profile", role="chat", engine="mock")

    # The hook runs before the supervisor even decides whether it can load the profile.
    with pytest.raises(ValueError, match="not a managed chat profile"):
        await supervisor.load_chat(profile, install)

    assert calls == ["released"]


def test_the_application_releases_warm_media_before_chat_loads(app: FastAPI) -> None:
    services = app.state.services
    assert services.processes.before_chat_load == services.orchestrator.release_media_for_chat


async def test_an_explicit_chat_load_owes_no_displaced_profile_back(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SessionLocal() as session:
        install = ModelInstall(
            name="Chosen chat model",
            role="chat",
            engine=app.state.services.settings.chat_engine,
            local_path="managed/chosen-chat-model",
            active=True,
        )
        session.add(install)
        session.flush()
        profile = ModelProfile(
            name="Chosen chat profile",
            role="chat",
            engine=app.state.services.settings.chat_engine,
            model_install_id=install.id,
        )
        session.add(profile)
        session.commit()
        profile_id = profile.id

    async def load_chat(chosen: Any, _install: Any) -> WorkerStatus:
        return WorkerStatus(
            name="chat", state="ready", managed=True, running=True, profile_id=chosen.id
        )

    monkeypatch.setattr(app.state.services.processes, "load_chat", load_chat)
    orchestrator = app.state.services.orchestrator
    orchestrator._displaced_chat_profile_id = "profile-displaced-earlier"

    loaded = await client.post(f"/api/workers/chat/load/{profile_id}")

    assert loaded.status_code == 200, loaded.text
    assert orchestrator._displaced_chat_profile_id is None


async def test_a_chat_load_leaves_a_busy_worker_and_its_window_alone() -> None:
    orchestrator, record, device = _warm_orchestrator(0.01)
    await orchestrator._complete_media_handoff("profile-chat")
    # Someone queued a prompt in ComfyUI's own page.
    record["comfy_queue"] = 1

    async with device:
        await orchestrator.release_media_for_chat()
        # A chat model loads beside it: slower, but nobody's work is ended.
        assert record["order"] == []
        assert orchestrator._media_warm_window is not None
        record["chat_running"] = True
        orchestrator.forget_displaced_chat_profile()
    await asyncio.sleep(0.05)
    assert record["order"] == []

    # Once they are done, the window's expiry recycles the resident models.
    record["comfy_queue"] = 0
    await _settled(orchestrator)
    assert record["order"] == ["stop media", "start media"]
    await orchestrator.close()


async def test_an_open_editor_session_keeps_the_warm_worker() -> None:
    orchestrator, record, device = _warm_orchestrator(0.01)
    await orchestrator._complete_media_handoff("profile-chat")
    editing = {"open": True}
    orchestrator.media_worker_in_use = lambda: editing["open"]

    await asyncio.sleep(0.05)
    async with device:
        await orchestrator.release_media_for_chat()
    # Neither the expiry nor a chat load replaced the launch the editor is bound to.
    assert record["order"] == []
    assert record["media_pid"] == 31
    assert orchestrator._media_warm_window is not None

    editing["open"] = False
    await _settled(orchestrator)
    assert record["order"] == ["stop media", "resume profile-chat", "start media"]
    await orchestrator.close()


async def test_a_worker_still_stopping_is_not_the_warm_launch() -> None:
    orchestrator, record, _ = _warm_orchestrator(0.01)
    await orchestrator._complete_media_handoff("profile-chat")
    # A reset is taking it down when the window ends.
    record["media_state"] = "stopping"

    await _settled(orchestrator)

    assert record["order"] == ["resume profile-chat"]
    await orchestrator.close()


async def test_an_owed_restart_does_not_replace_a_worker_someone_started() -> None:
    orchestrator, record, device = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")

    async with device:
        await orchestrator.release_media_for_chat()
        record["chat_running"] = True
        orchestrator.forget_displaced_chat_profile()
    # Before the settlement gets the device, someone starts media by hand.
    record["media_running"] = True
    record["media_pid"] = 77
    await _settled(orchestrator)

    assert record["order"] == ["stop media"]
    assert record["media_pid"] == 77
    assert orchestrator._media_restart_after_chat_activity is False
    await orchestrator.close()


async def test_a_cancel_during_the_busy_check_leaves_the_window_to_its_expiry() -> None:
    orchestrator, record, device = _warm_orchestrator(60)
    await orchestrator._complete_media_handoff("profile-chat")
    timer = orchestrator._media_warm_task
    assert timer is not None
    record["probe_gate"] = asyncio.Event()

    async with device:
        release = asyncio.create_task(orchestrator.release_media_for_chat())
        while not record["probes"]:
            await asyncio.sleep(0)
        release.cancel()
        await asyncio.gather(release, return_exceptions=True)

    # Nothing moved and nothing was settled, so the expiry still owns the window.
    assert release.cancelled()
    assert record["order"] == []
    assert orchestrator._media_warm_window is not None
    assert orchestrator._media_warm_task is timer and not timer.done()
    await orchestrator.close()


async def test_the_application_counts_open_editor_sessions_as_use(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready_editor(app, monkeypatch)
    orchestrator = app.state.services.orchestrator
    workflow = await _create_workflow(client)
    assert orchestrator.media_worker_in_use() is False

    started = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")
    assert started.status_code == 201, started.text
    assert orchestrator.media_worker_in_use() is True

    session = started.json()
    cancelled = await client.post(
        f"/api/workflows/{workflow['id']}/editor-sessions/{session['id']}/cancel",
        json={"nonce": session["nonce"]},
    )
    assert cancelled.status_code < 300, cancelled.text
    assert orchestrator.media_worker_in_use() is False


@pytest.mark.parametrize(
    ("turn_fails", "chat_loaded", "still_owed"),
    [(False, False, False), (True, False, True), (True, True, False)],
)
async def test_a_text_turn_settles_the_record_unless_it_ended_with_chat_down(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    turn_fails: bool,
    chat_loaded: bool,
    still_owed: bool,
) -> None:
    chat = ChatWorker(app, monkeypatch)
    chat.running = chat_loaded
    services = app.state.services
    orchestrator = services.orchestrator
    orchestrator._displaced_chat_profile_id = "profile-chat"
    if turn_fails:

        async def stream(_request: ChatRequest) -> AsyncIterator[ChatEvent]:
            raise RuntimeError("The chat model could not be prepared")
            yield ChatEvent(type="done")

        monkeypatch.setattr(services.engines.chat, "stream", stream)
    created = await client.post("/api/chats", json={"title": "Question"})
    assert created.status_code == 201

    turn = await client.post(
        f"/api/chats/{created.json()['id']}/turns",
        json={"text": "Name three primary colors.", "mode": "text"},
    )
    assert turn.status_code == 202, turn.text
    expected = "failed" if turn_fails else "complete"
    assert (await finished(client, turn.json()["run"]["id"], expected))["status"] == expected

    async def running_jobs() -> int:
        return len(orchestrator._tasks)

    await wait_until(running_jobs, lambda count: count == 0, what="the text turn to exit")
    # A completed turn owes nothing; one that ended with chat down leaves it owed.
    assert orchestrator._displaced_chat_profile_id == ("profile-chat" if still_owed else None)


async def test_a_stopped_run_does_not_keep_its_worker_warm(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ChatWorker(app, monkeypatch)
    services = app.state.services
    orchestrator = services.orchestrator
    generating = asyncio.Event()

    async def generate(_request: MediaRequest) -> AsyncIterator[MediaEvent]:
        generating.set()
        await asyncio.Event().wait()
        yield MediaEvent(type="complete")

    monkeypatch.setattr(services.engines.media, "generate", generate)
    original = orchestrator._complete_media_handoff
    handoffs: list[bool] = []

    async def complete(chat_profile_id: str, *, keep_warm: bool = True) -> None:
        handoffs.append(keep_warm)
        await original(chat_profile_id, keep_warm=keep_warm)

    monkeypatch.setattr(orchestrator, "_complete_media_handoff", complete)

    run_id = await start_media(client, "video")
    await asyncio.wait_for(generating.wait(), timeout=PATIENCE_SECONDS)
    chat_id = (await read_run(client, run_id))["chat_id"]
    stopped = await client.post(f"/api/chats/{chat_id}/cancel")
    assert stopped.status_code == 200, stopped.text
    await finished(client, run_id, "cancelled")

    async def handed_off() -> list[bool]:
        return list(handoffs)

    # ComfyUI may still be finishing the interrupted prompt: hand the device back.
    assert await wait_until(handed_off, bool, what="the stopped run's handoff") == [False]
