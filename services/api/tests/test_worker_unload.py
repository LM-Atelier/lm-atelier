"""One call releases both workers' models and keeps them released.

Someone who wants the graphics card for another program, such as training a
LoRA, while LM Atelier goes on answering everything that needs no model, had to
pause generation, stop chat, stop ComfyUI, wait out the window that keeps
ComfyUI warm after a media turn, and stop whatever giving the device back then
started again. The release does it at once. Everything that would load a model
back on its own is stood down first, and both workers then stop, each stop
counted as manual. It is refused while any job is running or could start, and
while someone else is using ComfyUI. Filling a prompt template's model-guided
slots is not a job, so a fill in progress is interrupted instead. The next
turn, run or load starts its model as usual, and so does a restart.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from run_waits import PATIENCE_SECONDS, wait_for_terminal_status, wait_until
from sqlalchemy import select
from test_asset_adoption import _loras, _safetensors
from test_media_worker_start_and_stop_reasons import (
    HEALTH,
    LAUNCH,
    NO_STARTS,
    NO_STOPS,
    _chat_profile,
    _fake_workers,
    _Process,
    _running_record,
    _status,
)
from test_prompt_library_api import (
    _fill_slots_with,
    _installed_chat_profile,
    _model_slot_batch,
    _select_chat_profile,
    _stopped_workers,
)
from test_workflow_editor_api import _create_workflow, _ready_editor

from local_lm import main as application
from local_lm import worker_startup
from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import Chat, Job, WebSearchProposal
from local_lm.processes import ProcessSupervisor, WorkerRecord
from local_lm.scheduler import JobClaim
from local_lm.schemas import WorkerStartReason, WorkerStatus, WorkerStopReason
from local_lm.web_search import CrwSearchProvider
from local_lm.web_search_configuration import search_provider_revision
from local_lm.web_search_consent import pause_for_search
from local_lm.workflow_compatibility import mirror_legacy_chat_workflow_selections

UNLOAD = "/api/workers/unload"


def _loaded_workers(app: FastAPI, settings: Settings) -> tuple[_Process, _Process]:
    """A chat and a media worker the supervisor manages, both ready."""

    supervisor: ProcessSupervisor = app.state.services.processes
    chat_record, chat_process = _running_record(settings, "chat", 987_654_501)
    media_record, media_process = _running_record(settings, "media", 987_654_502)
    supervisor._workers["chat"] = chat_record
    supervisor._workers["media"] = media_record
    return chat_process, media_process


def _recorded_loads(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every media worker start and every chat model given back from here on."""

    services = app.state.services
    supervisor: ProcessSupervisor = services.processes
    loads: list[str] = []

    async def start_media(**options: Any) -> WorkerStatus:
        loads.append(f"start media ({options.get('reason')})")
        return _status(supervisor)

    async def resume(profile_id: str) -> None:
        loads.append(f"give back {profile_id}")

    monkeypatch.setattr(supervisor, "start_media", start_media)
    monkeypatch.setattr(services.orchestrator, "_resume_chat_worker", resume)
    return loads


def _unchanged(app: FastAPI) -> None:
    """Neither worker was stopped, and nothing was stood down for a release."""

    supervisor: ProcessSupervisor = app.state.services.processes
    for name in ("chat", "media"):
        worker = _status(supervisor, name)
        assert worker.running, f"the {name} worker was stopped"
        assert worker.stops_by_reason == NO_STOPS
    assert app.state.services.orchestrator.release_generation == 0


def _add_job(kind: str, status: str) -> None:
    with SessionLocal() as session:
        session.add(
            Job(
                kind=kind,
                status=status,
                phase=status,
                queue_resource="interactive_compute" if kind == "chat" else "media_compute",
                queue_group="primary",
                queue_ticket="neutral-ticket",
                enqueued_at=utcnow(),
                payload_json={},
            )
        )
        session.commit()


def _counted_lease(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The name of each task that asks for the device lease from here on, in order."""

    scheduler = app.state.services.scheduler
    lease = scheduler.lease
    asked: list[str] = []

    @asynccontextmanager
    async def counted(device_id: str = "primary") -> AsyncIterator[None]:
        task = asyncio.current_task()
        asked.append(task.get_name() if task is not None else "")
        async with lease(device_id):
            yield

    monkeypatch.setattr(scheduler, "lease", counted)
    return asked


@asynccontextmanager
async def _device_held(app: FastAPI) -> AsyncIterator[None]:
    """Hold the device lease from a task of its own, as a job that has not finished does."""

    scheduler = app.state.services.scheduler
    holding, let_go = asyncio.Event(), asyncio.Event()

    async def hold() -> None:
        async with scheduler.lease("primary"):
            holding.set()
            await let_go.wait()

    holder = asyncio.create_task(hold(), name="neutral-device-holder")
    try:
        await asyncio.wait_for(holding.wait(), timeout=PATIENCE_SECONDS)
        yield
    finally:
        let_go.set()
        await asyncio.wait_for(holder, timeout=PATIENCE_SECONDS)


async def test_unloading_stops_both_workers_and_counts_each_stop_as_manual(
    app: FastAPI, client: AsyncClient, settings: Settings
) -> None:
    chat_process, media_process = _loaded_workers(app, settings)

    unloaded = await client.post(UNLOAD)

    assert unloaded.status_code == 200, unloaded.text
    assert chat_process.returncode is not None
    assert media_process.returncode is not None
    workers = {worker["name"]: worker for worker in unloaded.json()}
    assert sorted(workers) == ["chat", "media"]
    for worker in workers.values():
        assert worker["running"] is False
        assert worker["stops_by_reason"] == {
            **{reason.value: 0 for reason in WorkerStopReason},
            "manual": 1,
        }
        assert worker["last_stop_reason"] == "manual"
        assert worker["active_jobs"] == 0
        assert worker["queued_jobs"] == 0


async def test_unloading_needs_the_session_token(
    app: FastAPI, client: AsyncClient, settings: Settings
) -> None:
    _loaded_workers(app, settings)

    refused = await client.post(UNLOAD, headers={"x-local-lm-csrf": ""})

    assert refused.status_code == 403
    assert refused.json()["code"] == "csrf-invalid"
    _unchanged(app)


@pytest.mark.parametrize(
    ("kind", "status"),
    [("image", "running"), ("chat", "queued")],
    ids=["running", "ready-to-start"],
)
async def test_work_running_or_ready_to_start_keeps_the_models_loaded(
    app: FastAPI, client: AsyncClient, settings: Settings, kind: str, status: str
) -> None:
    _loaded_workers(app, settings)
    _add_job(kind, status)

    refused = await client.post(UNLOAD)

    assert refused.status_code == 409, refused.text
    body = refused.json()
    assert body["code"] == "workers-busy"
    assert body["busy_jobs"] == 1
    assert "running or ready to start" in body["detail"]
    _unchanged(app)


async def test_a_job_that_holds_the_device_is_answered_without_waiting_for_it(
    app: FastAPI, client: AsyncClient, settings: Settings
) -> None:
    """A job that never finishes holds the device lease, so jobs are looked for before it."""

    _loaded_workers(app, settings)
    _add_job("image", "running")

    async with _device_held(app):
        refused = await asyncio.wait_for(client.post(UNLOAD), timeout=PATIENCE_SECONDS)

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "workers-busy"
    _unchanged(app)


async def test_work_accepted_while_the_unload_waits_for_the_device_keeps_the_models_loaded(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _loaded_workers(app, settings)
    asked = _counted_lease(app, monkeypatch)

    async with _device_held(app):
        unload = asyncio.create_task(client.post(UNLOAD))

        async def unload_waiting() -> bool:
            return len(asked) == 2 or unload.done()

        await wait_until(unload_waiting, bool, what="the unload to wait for the device")
        assert not unload.done(), (await unload).text
        # A turn accepted meanwhile could start as soon as the device is free.
        _add_job("chat", "queued")
    refused = await asyncio.wait_for(unload, timeout=PATIENCE_SECONDS)

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "workers-busy"
    _unchanged(app)


class _AutomaticSearch:
    """A turn whose chat model asks to search the web, in a chat that allows it without asking.

    The turn pauses with its search due a few seconds later, and waits out that
    delay until the test ends it. Nothing is searched, and each time the turn
    runs, its job is recorded in `executions`.
    """

    def __init__(self, app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        orchestrator = app.state.services.orchestrator
        monkeypatch.setattr(settings, "web_access_enabled", True)
        monkeypatch.setattr(settings, "crw_endpoint", "https://search.example.test")
        provider = CrwSearchProvider("https://search.example.test")
        self.executions: list[str] = []
        self.waiting = asyncio.Event()
        self._delay_over = asyncio.Event()

        async def search_first(runner: Any, job_id: str, _run_id: str, claim: JobClaim) -> None:
            self.executions.append(job_id)
            # What a chat model that asks to search the web leaves behind: the
            # turn paused with its search due shortly, and the wait for it begun.
            with SessionLocal() as session:
                pause_for_search(
                    session,
                    job_id,
                    claim,
                    query="Compare two neutral materials",
                    provider_endpoint=provider.endpoint,
                    provider_revision=search_provider_revision(provider),
                    installation_enabled=True,
                )
            runner.resume_search(job_id)

        async def search_delay(_runner: Any, _deadline: Any) -> None:
            self.waiting.set()
            await self._delay_over.wait()

        monkeypatch.setattr(type(orchestrator), "_execute_chat", search_first)
        monkeypatch.setattr(type(orchestrator), "_wait_for_search_dispatch", search_delay)

    async def send(self, client: AsyncClient) -> str:
        """Send the turn, and return its job once it is waiting out the delay."""

        created = await client.post("/api/chats", json={"title": "Searches on its own"})
        assert created.status_code == 201, created.text
        with SessionLocal() as session:
            chat = session.get(Chat, created.json()["id"])
            assert chat is not None
            chat.web_settings_json = {"allow_search": True, "allow_search_without_asking": True}
            session.commit()
        accepted = await client.post(
            f"/api/chats/{created.json()['id']}/turns",
            json={"text": "Compare two neutral materials", "mode": "text"},
        )
        assert accepted.status_code == 202, accepted.text
        await asyncio.wait_for(self.waiting.wait(), timeout=PATIENCE_SECONDS)
        [job_id] = self.executions
        return job_id

    def end_the_delay(self) -> None:
        """Let the delay end, as its few seconds passing would."""

        with SessionLocal() as session:
            proposal = session.scalar(select(WebSearchProposal))
            assert proposal is not None
            proposal.dispatch_after = utcnow() - timedelta(seconds=1)
            session.commit()
        self._delay_over.set()


async def test_a_turn_waiting_out_its_automatic_search_delay_keeps_the_models_loaded(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn that searches the web on its own pauses, and goes on by itself a few seconds later.

    An unload in between would see it go on and load its chat model again.
    """

    search = _AutomaticSearch(app, settings, monkeypatch)
    _loaded_workers(app, settings)
    job_id = await search.send(client)

    refused = await client.post(UNLOAD)

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "workers-busy"
    assert refused.json()["busy_jobs"] == 1
    _unchanged(app)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.status == "paused"


async def test_a_turn_waiting_out_its_search_delay_in_a_paused_category_does_not_keep_the_models(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once its delay is over, a turn whose category is paused is only queued again.

    It then waits for its category to resume, like any other work in it, so the
    models need not stay loaded for it.
    """

    services = app.state.services
    supervisor: ProcessSupervisor = services.processes
    search = _AutomaticSearch(app, settings, monkeypatch)
    _loaded_workers(app, settings)
    job_id = await search.send(client)
    paused = await client.post(
        "/api/queue/lanes/generation/pause-after-current",
        json={"expected_revision": 0, "idempotency_key": "neutral-pause"},
    )
    assert paused.status_code == 200, paused.text
    assert paused.json()["dispatch_state"] == "paused"

    unloaded = await client.post(UNLOAD)

    assert unloaded.status_code == 200, unloaded.text
    search.end_the_delay()

    async def job_status() -> str:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            return job.status

    await wait_until(job_status, lambda status: status == "queued", what="the turn to be queued")
    # It waits for Generation to resume, and has not run again to load a model.
    assert services.scheduler.startable_jobs([job_id]) == frozenset()
    assert search.executions == [job_id]
    for name in ("chat", "media"):
        worker = _status(supervisor, name)
        assert not worker.running
        assert worker.starts_by_reason == NO_STARTS


async def test_work_waiting_in_a_paused_category_does_not_keep_the_models_loaded(
    app: FastAPI, client: AsyncClient, settings: Settings
) -> None:
    """Pausing is how the models are released while work waits its turn."""

    _loaded_workers(app, settings)
    _add_job("chat", "queued")
    paused = await client.post(
        "/api/queue/lanes/generation/pause-after-current",
        json={"expected_revision": 0, "idempotency_key": "neutral-pause"},
    )
    assert paused.status_code == 200, paused.text
    assert paused.json()["dispatch_state"] == "paused"

    unloaded = await client.post(UNLOAD)

    assert unloaded.status_code == 200, unloaded.text
    workers = {worker["name"]: worker for worker in unloaded.json()}
    assert workers["chat"]["running"] is False
    assert workers["media"]["running"] is False
    # Answered as the worker list answers, so the waiting job is still shown.
    assert workers["chat"]["queued_jobs"] == 1


@pytest.mark.parametrize("use", ["editor", "prompts"])
async def test_comfyui_in_use_keeps_the_models_loaded(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    use: str,
) -> None:
    services = app.state.services
    _loaded_workers(app, settings)
    if use == "editor":
        _ready_editor(app, monkeypatch)
        workflow = await _create_workflow(client)
        opened = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")
        assert opened.status_code == 201, opened.text
    else:
        # A prompt queued from ComfyUI's own page.
        monkeypatch.setattr(
            services.engines.media,
            "foreign_prompt_count",
            AsyncMock(return_value=1),
            raising=False,
        )

    refused = await client.post(UNLOAD)

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "media-worker-in-use"
    assert refused.json()["use"] == use
    _unchanged(app)


async def test_comfyui_use_that_begins_after_the_first_check_is_refused_at_the_stop(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = app.state.services
    supervisor: ProcessSupervisor = services.processes
    chat_process, media_process = _loaded_workers(app, settings)
    answers = iter([0, 1])
    asked_under_lock: list[bool] = []

    async def foreign_prompt_count() -> int:
        asked_under_lock.append(supervisor._locks["media"].locked())
        return next(answers)

    monkeypatch.setattr(
        services.engines.media, "foreign_prompt_count", foreign_prompt_count, raising=False
    )

    refused = await client.post(UNLOAD)

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "media-worker-in-use"
    # The second answer is the one the stop itself asked for, under the worker's lock.
    assert asked_under_lock == [False, True]
    assert media_process.returncode is None
    assert chat_process.returncode is None, "chat was stopped before ComfyUI was refused"
    assert _status(supervisor).stops_by_reason == NO_STOPS
    assert _status(supervisor, "chat").stops_by_reason == NO_STOPS


async def test_an_unload_ends_a_media_worker_restart_in_flight_as_a_manual_stop(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = app.state.services
    supervisor: ProcessSupervisor = services.processes
    spawned = _fake_workers(supervisor, monkeypatch)
    starting = asyncio.Event()

    async def still_starting(_record: WorkerRecord, _url: str) -> None:
        starting.set()
        await asyncio.Event().wait()

    async def start_media(**options: Any) -> WorkerStatus:
        # The launch ComfyUI would get is beside the point; the start is the supervisor's.
        await supervisor._replace("media", LAUNCH, HEALTH, reason=options["reason"])
        return _status(supervisor)

    monkeypatch.setattr(supervisor, "_wait_healthy", still_starting)
    monkeypatch.setattr(supervisor, "start_media", start_media)
    # Brought back after something borrowed the device, and still starting.
    services.orchestrator.schedule_media_restart()
    restart = services.orchestrator._media_restart_task
    assert restart is not None
    try:
        await asyncio.wait_for(starting.wait(), timeout=PATIENCE_SECONDS)
        # Bounded: an unload that waited for the restart instead of ending it
        # would wait for as long as the worker took to become ready.
        unloaded = await asyncio.wait_for(client.post(UNLOAD), timeout=PATIENCE_SECONDS)
    finally:
        if not restart.done():
            restart.cancel()
        await asyncio.gather(restart, return_exceptions=True)

    assert unloaded.status_code == 200, unloaded.text
    assert spawned[0].returncode is not None
    assert len(spawned) == 1
    media = _status(supervisor)
    assert not media.running
    assert media.starts_by_reason == {**NO_STARTS, WorkerStartReason.HANDOFF_RESTART: 1}
    assert media.stops_by_reason == {**NO_STOPS, WorkerStopReason.MANUAL: 1}


async def test_after_an_unload_an_open_warm_window_and_what_it_owes_load_nothing(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = app.state.services.orchestrator
    monkeypatch.setattr(settings, "media_engine", "comfyui")
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 60)
    _chat, media_process = _loaded_workers(app, settings)
    loads = _recorded_loads(app, monkeypatch)
    # A media run that displaced chat has ended, and its worker is kept for the next one.
    await orchestrator._complete_media_handoff("profile-chat")
    window = orchestrator._media_warm_window
    expiry = orchestrator._media_warm_task
    assert window is not None and expiry is not None
    # A restart owed once chat activity starts, and the next ordered step's prewarm.
    orchestrator._media_restart_after_chat_activity = True
    orchestrator._step_prewarm_plan_id = "plan-neutral"

    unloaded = await client.post(UNLOAD)

    assert unloaded.status_code == 200, unloaded.text
    await asyncio.gather(expiry, return_exceptions=True)
    assert expiry.cancelled()
    # The window's expiry waking anyway, chat activity starting, and the text
    # step streaming find nothing to load.
    await orchestrator._expire_media_warm_window(window, 0)
    orchestrator._release_deferred_media_restart()
    orchestrator._begin_step_prewarm()
    assert orchestrator._media_restart_task is None
    assert loads == []
    assert media_process.returncode is not None
    # Nor does the next media run owe the chat model it displaced back.
    assert orchestrator._pending_chat_restore(None) is None


async def test_a_chat_load_inside_a_warm_window_leaves_nothing_to_settle_after_an_unload(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A chat model loaded from the worker controls while the unload waits for the device.

    The load stops the warm ComfyUI first and leaves its settlement waiting for
    the device behind the unload: chat back if it went down again, and the
    empty ComfyUI back once chat is active.
    """

    services = app.state.services
    supervisor: ProcessSupervisor = services.processes
    orchestrator = services.orchestrator
    _fake_workers(supervisor, monkeypatch)
    runtime = tmp_path / "neutral-llama-server.exe"
    runtime.write_bytes(b"neutral runtime")
    monkeypatch.setattr(settings, "llama_executable", runtime)
    profile_id = _chat_profile(settings)
    monkeypatch.setattr(settings, "media_engine", "comfyui")
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 60)
    await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
    await orchestrator._complete_media_handoff(profile_id)
    window = orchestrator._media_warm_window
    assert window is not None
    loads = _recorded_loads(app, monkeypatch)

    lease = services.scheduler.lease
    asked: list[int] = []

    @asynccontextmanager
    async def counted_lease(device_id: str = "primary") -> AsyncIterator[None]:
        asked.append(len(asked))
        async with lease(device_id):
            yield

    monkeypatch.setattr(services.scheduler, "lease", counted_lease)
    # What every chat model load awaits first.
    release = orchestrator.release_media_for_chat
    assert supervisor.before_chat_load == release
    at_release = asyncio.Event()
    go = asyncio.Event()
    settlements: list[asyncio.Task[None] | None] = []

    async def release_when_told() -> None:
        at_release.set()
        await go.wait()
        await release()
        settlements.append(orchestrator._media_warm_task)

    monkeypatch.setattr(supervisor, "before_chat_load", release_when_told)

    load = asyncio.create_task(client.post(f"/api/workers/chat/load/{profile_id}"))
    try:
        await asyncio.wait_for(at_release.wait(), timeout=PATIENCE_SECONDS)
        unload = asyncio.create_task(client.post(UNLOAD))

        async def unload_waiting() -> bool:
            return len(asked) == 2 or unload.done()

        # The load holds the device, and the unload is waiting for it.
        await wait_until(unload_waiting, bool, what="the unload to wait for the device")
        assert not unload.done(), (await unload).text
    finally:
        go.set()
    loaded = await asyncio.wait_for(load, timeout=PATIENCE_SECONDS)
    unloaded = await asyncio.wait_for(unload, timeout=PATIENCE_SECONDS)

    assert loaded.status_code == 200, loaded.text
    assert unloaded.status_code == 200, unloaded.text
    [settlement] = settlements
    assert settlement is not None
    await asyncio.gather(settlement, return_exceptions=True)
    assert settlement.cancelled()
    # A settlement that woke anyway finds itself stale, and chat activity owes nothing.
    await orchestrator._settle_chat_borrow(window, profile_id)
    orchestrator._release_deferred_media_restart()
    assert loads == []
    chat, media = _status(supervisor, "chat"), _status(supervisor)
    assert not chat.running and not media.running
    assert chat.starts_by_reason == {**NO_STARTS, WorkerStartReason.MANUAL: 1}
    assert chat.stops_by_reason == {**NO_STOPS, WorkerStopReason.MANUAL: 1}
    assert media.starts_by_reason == {**NO_STARTS, WorkerStartReason.RUN: 1}
    assert media.stops_by_reason == {**NO_STOPS, WorkerStopReason.CHAT_LOAD: 1}


@asynccontextmanager
async def _session_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        session = await client.post("/api/session")
        client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]
        yield client


class _StartupRestore:
    """The application's own restore of both workers, held until a test lets it go on.

    It is held before it asks for the device. Both workers are set up to be
    restored once the application has started, and nothing is really started:
    `start_media` and `load_chat` record what the restore asks for.
    """

    def __init__(self, app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
        services = app.state.services
        restore = worker_startup.restore_configured_workers
        self.go = asyncio.Event()
        self.done = asyncio.Event()
        self.start_media = AsyncMock()
        self.load_chat = AsyncMock()
        self.reconciled_by: list[str] = []

        async def restore_when_told(restoring: Any, **options: Any) -> None:
            await self.go.wait()
            try:
                await restore(restoring, **options)
            finally:
                self.done.set()

        async def reconcile(*_args: object, **_options: object) -> None:
            # Startup recovery reconciles too, from a task of its own.
            task = asyncio.current_task()
            self.reconciled_by.append(task.get_name() if task is not None else "")

        monkeypatch.setattr(application, "restore_configured_workers", restore_when_told)
        monkeypatch.setattr(services.processes, "start_media", self.start_media)
        monkeypatch.setattr(services.processes, "load_chat", self.load_chat)
        monkeypatch.setattr(
            services.downloads, "refresh_installed_media_workflows", AsyncMock(return_value=0)
        )
        monkeypatch.setattr(services.downloads, "reconcile_workflow_install_offers", reconcile)

    @staticmethod
    def configure_both(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set both workers up to be restored, once the application has started."""

        monkeypatch.setattr(settings, "media_engine", "comfyui")
        monkeypatch.setattr(worker_startup, "_media_worker_should_restore", lambda _services: True)
        monkeypatch.setattr(settings, "chat_engine", "llama.cpp")
        _chat_profile(settings)
        assert worker_startup.chat_profile_to_restore() is not None


async def test_a_startup_restore_that_goes_on_after_an_unload_leaves_both_workers_down(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    restore = _StartupRestore(app, monkeypatch)

    async with app.router.lifespan_context(app), _session_client(app) as client:
        restore.configure_both(settings, monkeypatch)
        # Held before it asks for the device, so it goes on only once the
        # unload has finished.
        unloaded = await client.post(UNLOAD)
        assert unloaded.status_code == 200, unloaded.text
        restore.go.set()
        await asyncio.wait_for(restore.done.wait(), timeout=PATIENCE_SECONDS)

    restore.start_media.assert_not_awaited()
    restore.load_chat.assert_not_awaited()
    # The rest of the restore still goes ahead.
    assert restore.reconciled_by.count("restore-configured-workers") == 1


async def test_a_startup_restore_that_asks_for_the_device_during_an_unload_loads_nothing(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restore asks for the device while the unload holds it, before the release is counted.

    It learns of the release only once it has the device, which is where it
    has to ask whether one was asked for.
    """

    services = app.state.services
    restore = _StartupRestore(app, monkeypatch)
    looking, look_done = asyncio.Event(), asyncio.Event()

    async def foreign_prompt_count() -> int:
        # The unload's first look at ComfyUI, under the device lease and before
        # it stands anything down. The stop's own look answers at once.
        if not looking.is_set():
            looking.set()
            await look_done.wait()
        return 0

    monkeypatch.setattr(
        services.engines.media, "foreign_prompt_count", foreign_prompt_count, raising=False
    )

    async with app.router.lifespan_context(app), _session_client(app) as client:
        restore.configure_both(settings, monkeypatch)
        _loaded_workers(app, settings)
        asked = _counted_lease(app, monkeypatch)
        unload = asyncio.create_task(client.post(UNLOAD))
        try:
            await asyncio.wait_for(looking.wait(), timeout=PATIENCE_SECONDS)
            restore.go.set()

            async def restore_waiting() -> bool:
                return "restore-configured-workers" in asked or restore.done.is_set()

            await wait_until(restore_waiting, bool, what="the restore to ask for the device")
            assert not restore.done.is_set()
            assert services.orchestrator.release_generation == 0
        finally:
            look_done.set()
        unloaded = await asyncio.wait_for(unload, timeout=PATIENCE_SECONDS)
        assert unloaded.status_code == 200, unloaded.text
        await asyncio.wait_for(restore.done.wait(), timeout=PATIENCE_SECONDS)

    restore.start_media.assert_not_awaited()
    restore.load_chat.assert_not_awaited()


async def test_an_unload_interrupts_a_model_guided_prompt_fill_and_leaves_comfyui_stopped(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filling a template's model-guided slots is not a job, so an unload does not wait for it.

    The fill stops ComfyUI and loads chat, and gives ComfyUI back once the model
    has answered. An unload while the model answers stops chat under it: the
    fill fails and can be tried again, and ComfyUI is not given back.
    """

    services = app.state.services
    profile_id = _installed_chat_profile(app)
    state = _stopped_workers(app, monkeypatch, media_running=True)
    unloads: list[int] = []

    async def answer_until_the_unload(_request: ChatRequest) -> AsyncIterator[ChatEvent]:
        unloads.append((await client.post(UNLOAD)).status_code)
        # The chat worker the answer was streaming from has stopped.
        state["chat"] = WorkerStatus(name="chat", state="stopped", managed=True, running=False)
        raise ConnectionError("The chat worker stopped while it was answering.")
        yield ChatEvent(type="done")

    monkeypatch.setattr(services.engines.chat, "stream", answer_until_the_unload)
    chat = (await client.post("/api/chats", json={"title": "Unloaded while it fills"})).json()
    _select_chat_profile(chat["id"], profile_id)

    interrupted = await _model_slot_batch(client, chat["id"], "unloaded-while-it-fills")

    assert unloads == [200]
    assert interrupted.status_code == 503, interrupted.text
    assert interrupted.json()["code"] == "prompt-model-invocation-failed"
    assert "Retry" in interrupted.json()["detail"]
    # The fill stopped ComfyUI to load chat, then the unload stopped both.
    assert state["stopped"] == ["media", "media", "chat"]
    assert state["stop_reasons"] == [
        WorkerStopReason.CHAT_LOAD,
        WorkerStopReason.MANUAL,
        WorkerStopReason.MANUAL,
    ]
    assert state["restarts"] == 0, "ComfyUI was given back after the unload"

    # Tried again, the fill loads chat once more and fills the slots.
    _fill_slots_with(monkeypatch)
    retried = await _model_slot_batch(client, chat["id"], "filled-after-the-unload")

    assert retried.status_code == 201, retried.text
    assert state["loaded"] == [profile_id, profile_id]
    assert state["restarts"] == 0


async def test_the_next_turn_after_an_unload_starts_its_model_as_usual(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    services = app.state.services
    supervisor: ProcessSupervisor = services.processes
    orchestrator = services.orchestrator
    # Read while the test's chat engine is still the active one: loading the
    # managed model below makes llama.cpp the active engine.
    fields = await orchestrator.engines.settings_for_role("chat", engine="mock")
    monkeypatch.setattr(orchestrator.engines, "settings_for_role", AsyncMock(return_value=fields))
    _fake_workers(supervisor, monkeypatch)
    runtime = tmp_path / "neutral-llama-server.exe"
    runtime.write_bytes(b"neutral runtime")
    monkeypatch.setattr(settings, "llama_executable", runtime)
    profile_id = _chat_profile(settings)
    loaded = await client.post(f"/api/workers/chat/load/{profile_id}")
    assert loaded.status_code == 200, loaded.text

    unloaded = await client.post(UNLOAD)
    assert unloaded.status_code == 200, unloaded.text
    assert not _status(supervisor, "chat").running

    # The chat the turn is sent in uses the managed chat model.
    assert settings.chat_engine == "llama.cpp"
    chat = (await client.post("/api/chats", json={"title": "After the unload"})).json()
    with SessionLocal() as session:
        stored = session.get(Chat, chat["id"])
        assert stored is not None
        stored.active_chat_profile_id = profile_id
        mirror_legacy_chat_workflow_selections(session, stored, ["chat"])
        session.commit()

    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Name three primary colors.", "mode": "text"},
    )
    assert turn.status_code == 202, turn.text
    run_id = turn.json()["run"]["id"]

    async def read_run() -> dict[str, Any]:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        return dict(response.json())

    await wait_for_terminal_status(read_run, what="the turn after the unload")

    worker = _status(supervisor, "chat")
    assert worker.running
    assert worker.profile_id == profile_id
    # Loaded from the controls before the unload, and by the turn after it.
    assert sum(worker.starts_by_reason.values()) == 2
    assert not _status(supervisor).running


async def test_the_routes_that_need_no_model_answer_while_nothing_is_loaded(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _loaded_workers(app, settings)
    unloaded = await client.post(UNLOAD)
    assert unloaded.status_code == 200, unloaded.text
    loads = _recorded_loads(app, monkeypatch)
    monkeypatch.setattr(settings, "comfy_directory", tmp_path / "comfy")
    _safetensors(_loras(settings) / "neutral-style.safetensors")

    registered = await client.post(
        "/api/model-assets", json={"kind": "lora", "comfy_name": "neutral-style.safetensors"}
    )
    listed = await client.get("/api/model-assets", params={"kind": "lora"})

    assert registered.status_code == 201, registered.text
    assert listed.status_code == 200, listed.text
    assert [asset["id"] for asset in listed.json()] == [registered.json()["id"]]
    assert loads == []
    supervisor: ProcessSupervisor = app.state.services.processes
    assert not _status(supervisor, "chat").running
    assert not _status(supervisor).running
