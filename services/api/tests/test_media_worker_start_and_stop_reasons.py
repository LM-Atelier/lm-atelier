"""Every media worker start and stop is counted under a fixed reason.

Telling why ComfyUI restarted used to mean reading gaps between its health
probes. The supervisor now counts each process it spawns under the reason its
caller names, each running process it ends under the reason for the stop, and
a process that ended while nothing was stopping it as an exit without a stop.
An emptying of the running worker's memory in place is counted apart, under
the reason of the stop it stood in for. The worker status reports the counts
since the service started, and each counted event writes one log line made of
the worker and the code alone.

Only what happened is counted. A ready launch that is kept, a start refused
before its process exists and a stop of a stopped worker count nothing, and
one process is counted once however many paths notice that it ended.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from run_waits import PATIENCE_SECONDS, wait_for_terminal_status, wait_until
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from test_project_portability import _manifest, _rewrite_archive

import local_lm.workflow_completion_jobs as workflow_completion_jobs_module
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.events import EventBroker
from local_lm.main import create_app
from local_lm.models import Chat, Job, ModelInstall, ModelProfile, Run
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.processes import (
    ProcessSupervisor,
    WorkerRecord,
    WorkerStopIncomplete,
    _RotatingWorkerLog,
)
from local_lm.schemas import WorkerStartReason, WorkerStatus, WorkerStopReason
from local_lm.workflow_package_runtime import workflow_package_runtime

HEALTH = "http://127.0.0.1:9/health"
SCOPE = "a" * 64
OTHER_SCOPE = "b" * 64
LAUNCH = ["neutral-media-worker"]
OTHER_LAUNCH = ["neutral-media-worker", "--another-option"]
NO_STARTS = dict.fromkeys(WorkerStartReason, 0)
NO_STOPS = dict.fromkeys(WorkerStopReason, 0)
#: Counts that describe the service run, never one model run or a bundle.
SERVICE_WIDE_FIELDS = frozenset(
    {
        "starts_by_reason",
        "stops_by_reason",
        "emptied_by_reason",
        "exits_without_stop",
        "last_start_reason",
        "last_stop_reason",
        "counted_since",
    }
)
_LIFECYCLE_LINE = re.compile(r"^(Chat|Media) worker (started|stopped|exited)")


class _Process:
    """A worker process that runs until it is ended, or ends by itself."""

    stdout = None
    stderr = None

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self._ended = asyncio.Event()

    def end(self, code: int) -> None:
        self.returncode = code
        self._ended.set()

    def terminate(self) -> None:
        self.end(0)

    def kill(self) -> None:
        self.end(-9)

    async def wait(self) -> int:
        await self._ended.wait()
        assert self.returncode is not None
        return self.returncode


class _StubbornProcess(_Process):
    """A worker process that ignores being asked to stop, until it ends by itself."""

    def __init__(self, pid: int) -> None:
        super().__init__(pid)
        self.asked_to_stop = asyncio.Event()

    def terminate(self) -> None:
        self.asked_to_stop.set()

    def kill(self) -> None:
        self.asked_to_stop.set()


def _fake_workers(
    supervisor: ProcessSupervisor,
    monkeypatch: pytest.MonkeyPatch,
    process_type: type[_Process] = _Process,
) -> list[_Process]:
    """Make the supervisor's workers fake processes that answer as healthy."""

    spawned: list[_Process] = []

    async def spawn(*_command: str, **_options: object) -> _Process:
        process = process_type(987_654_300 + len(spawned))
        spawned.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(supervisor, "_wait_healthy", AsyncMock())
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())
    monkeypatch.setattr(supervisor, "_reclaim_port_from_our_own_children", AsyncMock())
    monkeypatch.setattr(supervisor, "_record_worker_process_tree", lambda _name, _pid: None)
    monkeypatch.setattr(supervisor, "_probe_worker_health", AsyncMock(return_value=True))
    return spawned


def _supervisor(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    process_type: type[_Process] = _Process,
    **options: Any,
) -> tuple[ProcessSupervisor, list[_Process]]:
    """A supervisor whose workers are fake processes that answer as healthy."""

    settings.prepare()
    supervisor = ProcessSupervisor(settings, **options)
    return supervisor, _fake_workers(supervisor, monkeypatch, process_type)


def _status(supervisor: ProcessSupervisor, name: str = "media") -> WorkerStatus:
    return next(status for status in supervisor.statuses() if status.name == name)


def _lifecycle_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "local_lm.processes" and _LIFECYCLE_LINE.match(record.getMessage())
    ]


def _exited_record(settings: Settings, name: str, pid: int, code: int) -> WorkerRecord:
    process = _Process(pid)
    process.end(code)
    return WorkerRecord(
        name,
        cast(Any, process),
        [f"neutral-{name}-worker"],
        _RotatingWorkerLog(settings.log_dir / f"neutral-{name}-exited.log"),
        state="ready",
    )


def _running_record(settings: Settings, name: str, pid: int) -> tuple[WorkerRecord, _Process]:
    process = _Process(pid)
    record = WorkerRecord(
        name,
        cast(Any, process),
        [f"neutral-{name}-worker"],
        _RotatingWorkerLog(settings.log_dir / f"neutral-{name}-running.log"),
        state="ready",
    )
    return record, process


async def test_a_spawned_start_is_counted_once_under_its_reason(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="local_lm.processes")
    supervisor, spawned = _supervisor(settings, monkeypatch)
    try:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)

        media = _status(supervisor)
        assert len(spawned) == 1
        assert media.starts_by_reason == {**NO_STARTS, WorkerStartReason.RUN: 1}
        assert media.last_start_reason is WorkerStartReason.RUN
        assert media.stops_by_reason == NO_STOPS
        assert media.last_stop_reason is None
        assert media.exits_without_stop == 0
        assert _lifecycle_lines(caplog) == ["Media worker started (reason: run)"]
    finally:
        await supervisor.close()


async def test_start_media_counts_the_reason_its_caller_names(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason reaches the spawn through the media start itself, not only `_replace`."""

    runtime = tmp_path / "ComfyUI"
    runtime.mkdir()
    (runtime / "main.py").write_bytes(b"")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"runtime")
    settings.comfy_directory = runtime
    settings.comfy_executable = executable
    model_paths = tmp_path / "extra-model-paths.yaml"
    model_paths.write_text("{}", encoding="utf-8")
    supervisor, spawned = _supervisor(settings, monkeypatch)
    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", AsyncMock(return_value=[]))
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda *_args: model_paths)
    try:
        status = await supervisor.start_media(reason=WorkerStartReason.PACKAGE_PREPARATION)

        assert len(spawned) == 1
        assert status.starts_by_reason == {**NO_STARTS, WorkerStartReason.PACKAGE_PREPARATION: 1}
        assert status.last_start_reason is WorkerStartReason.PACKAGE_PREPARATION
    finally:
        await supervisor.close()


async def test_a_ready_launch_that_is_kept_counts_nothing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spawned = _supervisor(settings, monkeypatch)
    try:
        for _ in range(2):
            await supervisor._replace(
                "media",
                list(LAUNCH),
                HEALTH,
                launch_scope_sha256=SCOPE,
                reason=WorkerStartReason.RUN,
            )

        media = _status(supervisor)
        assert len(spawned) == 1, "the identical ready launch was replaced"
        assert media.starts_by_reason == {**NO_STARTS, WorkerStartReason.RUN: 1}
        assert media.stops_by_reason == NO_STOPS
    finally:
        await supervisor.close()


@pytest.mark.parametrize("refusal", ["claim", "port"])
async def test_a_start_refused_before_its_process_exists_counts_nothing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    supervisor, spawned = _supervisor(settings, monkeypatch)
    if refusal == "port":
        monkeypatch.setattr(
            supervisor,
            "_ensure_port_available",
            AsyncMock(side_effect=OSError("neutral port refusal")),
        )

    def refuse_the_claim() -> None:
        raise RuntimeError("neutral claim refusal")

    try:
        with pytest.raises((OSError, RuntimeError), match="neutral"):
            await supervisor._replace(
                "media",
                LAUNCH,
                HEALTH,
                before_replace=refuse_the_claim if refusal == "claim" else None,
                reason=WorkerStartReason.MANUAL,
            )

        media = _status(supervisor)
        assert spawned == []
        assert media.starts_by_reason == NO_STARTS
        assert media.last_start_reason is None
    finally:
        await supervisor.close()


async def test_a_stop_is_counted_when_it_ends_a_running_worker_and_never_again(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="local_lm.processes")
    supervisor, spawned = _supervisor(settings, monkeypatch)
    try:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)

        stopped = await supervisor.stop("media", reason=WorkerStopReason.CHAT_LOAD)
        again = await supervisor.stop("media", reason=WorkerStopReason.MANUAL)

        assert spawned[0].returncode is not None
        assert stopped.stops_by_reason == {**NO_STOPS, WorkerStopReason.CHAT_LOAD: 1}
        assert stopped.last_stop_reason is WorkerStopReason.CHAT_LOAD
        assert again.stops_by_reason == stopped.stops_by_reason, "a stopped worker was counted"
        assert again.last_stop_reason is WorkerStopReason.CHAT_LOAD
        assert again.exits_without_stop == 0
        assert _lifecycle_lines(caplog) == [
            "Media worker started (reason: run)",
            "Media worker stopped (reason: chat_load)",
        ]
    finally:
        await supervisor.close()


@pytest.mark.parametrize(
    ("scope", "launch", "counted"),
    [
        # Another command too, so the scope is what decides it.
        (OTHER_SCOPE, OTHER_LAUNCH, WorkerStopReason.REPLACED_SCOPE),
        (None, LAUNCH, WorkerStopReason.REPLACED_SCOPE),
        (SCOPE, OTHER_LAUNCH, WorkerStopReason.REPLACED_COMMAND),
    ],
    ids=["another-scope", "no-scope", "another-command"],
)
async def test_a_live_worker_replaced_by_another_launch_counts_how_the_launch_differs(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    scope: str | None,
    launch: list[str],
    counted: WorkerStopReason,
) -> None:
    supervisor, spawned = _supervisor(settings, monkeypatch)
    try:
        await supervisor._replace(
            "media", LAUNCH, HEALTH, launch_scope_sha256=SCOPE, reason=WorkerStartReason.RUN
        )
        await supervisor._replace(
            "media", launch, HEALTH, launch_scope_sha256=scope, reason=WorkerStartReason.RUN
        )

        media = _status(supervisor)
        assert len(spawned) == 2
        assert spawned[0].returncode is not None
        assert media.stops_by_reason == {**NO_STOPS, counted: 1}
        assert media.starts_by_reason == {**NO_STARTS, WorkerStartReason.RUN: 2}
    finally:
        await supervisor.close()


@pytest.mark.parametrize(
    ("started", "stopped"),
    [
        (WorkerStartReason.MANUAL, WorkerStopReason.MANUAL),
        (WorkerStartReason.PACKAGE_PREPARATION, WorkerStopReason.PACKAGE_PREPARATION),
        (WorkerStartReason.MODEL_CHANGE, WorkerStopReason.MODEL_CHANGE),
        (WorkerStartReason.RUN, WorkerStopReason.UNATTRIBUTED),
    ],
)
async def test_the_same_launch_started_again_counts_the_stop_its_reason_causes(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    started: WorkerStartReason,
    stopped: WorkerStopReason,
) -> None:
    """A launch for no scope is never kept, so starting it again ends the live one."""

    supervisor, spawned = _supervisor(settings, monkeypatch)
    try:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=started)
        await supervisor._replace("media", list(LAUNCH), HEALTH, reason=started)

        media = _status(supervisor)
        assert len(spawned) == 2
        assert media.stops_by_reason == {**NO_STOPS, stopped: 1}
        assert media.starts_by_reason == {**NO_STARTS, started: 2}
    finally:
        await supervisor.close()


async def test_a_worker_that_ends_by_itself_is_one_exit_without_a_stop(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="local_lm.processes")
    events = EventBroker()
    supervisor, spawned = _supervisor(
        settings, monkeypatch, events=events, liveness_interval_seconds=0.01
    )

    async def exit_published() -> bool:
        return any(event.type == "worker.exited" for event in events.since(0))

    try:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
        spawned[0].end(9)
        await wait_until(exit_published, bool, what="the published media worker exit")

        # Counted by the supervision that noticed it, before any status read.
        assert _lifecycle_lines(caplog) == [
            "Media worker started (reason: run)",
            "Media worker exited without a stop",
        ]
        first = _status(supervisor)
        second = _status(supervisor)
        stopped = await supervisor.stop("media", reason=WorkerStopReason.MANUAL)

        assert first.state == "exited"
        for status in (first, second, stopped):
            assert status.exits_without_stop == 1
            assert status.stops_by_reason == NO_STOPS
            assert status.last_stop_reason is None
        assert len(_lifecycle_lines(caplog)) == 2
    finally:
        await supervisor.close()


async def test_an_exit_a_status_read_notices_first_is_counted_once(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, _spawned = _supervisor(settings, monkeypatch)
    supervisor._workers["media"] = _exited_record(settings, "media", 987_654_398, 3)
    try:
        assert _status(supervisor).exits_without_stop == 1
        assert _status(supervisor).exits_without_stop == 1

        stopped = await supervisor.stop("media", reason=WorkerStopReason.MANUAL)

        assert stopped.exits_without_stop == 1
        assert stopped.stops_by_reason == NO_STOPS
    finally:
        await supervisor.close()


@pytest.mark.parametrize("died", [False, True], ids=["never-ready", "died-starting"])
async def test_a_start_that_never_becomes_ready_is_ended_as_unhealthy_unless_it_died(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, died: bool
) -> None:
    supervisor, spawned = _supervisor(settings, monkeypatch)

    async def never_healthy(_record: WorkerRecord, _url: str) -> None:
        if died:
            spawned[-1].end(1)
            raise RuntimeError("neutral worker exited while starting")
        raise TimeoutError("neutral worker did not become healthy")

    monkeypatch.setattr(supervisor, "_wait_healthy", never_healthy)
    try:
        with pytest.raises((RuntimeError, TimeoutError), match="neutral worker"):
            await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)

        media = _status(supervisor)
        assert spawned[0].returncode is not None
        assert media.starts_by_reason == {**NO_STARTS, WorkerStartReason.RUN: 1}
        if died:
            assert media.stops_by_reason == NO_STOPS
            assert media.exits_without_stop == 1
        else:
            assert media.stops_by_reason == {**NO_STOPS, WorkerStopReason.UNHEALTHY: 1}
            assert media.exits_without_stop == 0
    finally:
        await supervisor.close()


@pytest.mark.parametrize(
    ("started", "stopped"),
    [
        (WorkerStartReason.MANUAL, WorkerStopReason.MANUAL),
        (WorkerStartReason.RUN, WorkerStopReason.UNATTRIBUTED),
    ],
)
async def test_a_start_abandoned_by_its_caller_counts_the_stop_its_reason_causes(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    started: WorkerStartReason,
    stopped: WorkerStopReason,
) -> None:
    supervisor, spawned = _supervisor(settings, monkeypatch)
    entered = asyncio.Event()

    async def still_starting(_record: WorkerRecord, _url: str) -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(supervisor, "_wait_healthy", still_starting)
    start = asyncio.create_task(supervisor._replace("media", LAUNCH, HEALTH, reason=started))
    try:
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start

        media = _status(supervisor)
        assert spawned[0].returncode is not None
        assert media.state == "stopped"
        assert media.starts_by_reason == {**NO_STARTS, started: 1}
        assert media.stops_by_reason == {**NO_STOPS, stopped: 1}
    finally:
        if not start.done():
            start.cancel()
            await asyncio.gather(start, return_exceptions=True)
        await supervisor.close()


async def test_the_liveness_monitor_counts_the_worker_it_ends_as_unhealthy(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = EventBroker()
    supervisor, spawned = _supervisor(
        settings,
        monkeypatch,
        events=events,
        liveness_interval_seconds=0.001,
        liveness_failure_threshold=1,
    )
    monkeypatch.setattr(supervisor, "_probe_worker_health", AsyncMock(return_value=False))

    async def unhealthy_published() -> bool:
        return any(event.type == "worker.unhealthy" for event in events.since(0))

    try:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
        await wait_until(unhealthy_published, bool, what="the unhealthy media worker report")

        media = _status(supervisor)
        assert spawned[0].returncode is not None
        assert media.stops_by_reason == {**NO_STOPS, WorkerStopReason.UNHEALTHY: 1}
        assert media.exits_without_stop == 0
    finally:
        await supervisor.close()


async def test_disabling_cancelled_extensions_counts_its_stop_as_package_preparation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spawned = _supervisor(settings, monkeypatch)
    monkeypatch.setattr(
        workflow_completion_jobs_module,
        "cancelled_workflow_activation_packages",
        lambda _session: ("neutral-cancelled-extension",),
    )
    disabled = Mock()
    monkeypatch.setattr(
        workflow_completion_jobs_module, "deactivate_cancelled_workflow_activations", disabled
    )
    try:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)

        await supervisor._clear_cancelled_workflow_activations()

        disabled.assert_called_once()
        assert spawned[0].returncode is not None
        assert _status(supervisor).stops_by_reason == {
            **NO_STOPS,
            WorkerStopReason.PACKAGE_PREPARATION: 1,
        }
    finally:
        await supervisor.close()


async def test_package_setup_stops_and_restores_the_running_worker_as_package_preparation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spawned = _supervisor(settings, monkeypatch)
    try:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
        # Starting ComfyUI is out of reach here; the stop is the supervisor's own.
        restored = AsyncMock(side_effect=lambda **_options: _status(supervisor))
        monkeypatch.setattr(supervisor, "start_media", restored)

        async with workflow_package_runtime(supervisor):
            assert spawned[0].returncode is not None, "the setup ran beside the worker"

        assert _status(supervisor).stops_by_reason == {
            **NO_STOPS,
            WorkerStopReason.PACKAGE_PREPARATION: 1,
        }
        restored.assert_awaited_once_with(reason=WorkerStartReason.PACKAGE_PREPARATION)
    finally:
        await supervisor.close()


async def test_closing_the_supervisor_counts_a_shutdown_for_each_running_worker(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chat worker is counted by the same supervisor, under the reasons it is given."""

    supervisor, _spawned = _supervisor(settings, monkeypatch)
    await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.APP_START)
    await supervisor._replace("chat", ["neutral-chat-worker"], HEALTH)

    await supervisor.close()

    chat, media = supervisor.statuses()
    assert media.starts_by_reason == {**NO_STARTS, WorkerStartReason.APP_START: 1}
    assert media.stops_by_reason == {**NO_STOPS, WorkerStopReason.SHUTDOWN: 1}
    assert chat.starts_by_reason == {**NO_STARTS, WorkerStartReason.UNATTRIBUTED: 1}
    assert chat.stops_by_reason == {**NO_STOPS, WorkerStopReason.SHUTDOWN: 1}


async def test_the_status_reports_every_code_the_last_reasons_and_when_counting_began(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    began = datetime.now(UTC)
    supervisor, _spawned = _supervisor(settings, monkeypatch)
    constructed = datetime.now(UTC)
    try:
        for status in supervisor.statuses():
            assert status.starts_by_reason == NO_STARTS
            assert status.stops_by_reason == NO_STOPS
            assert status.exits_without_stop == 0
            assert status.last_start_reason is None
            assert status.last_stop_reason is None
            assert status.counted_since is not None
            assert began <= status.counted_since <= constructed

        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.HANDOFF_RESTART)
        await supervisor.stop("media", reason=WorkerStopReason.KEEP_WARM_EXPIRY)

        chat, media = supervisor.statuses()
        assert media.last_start_reason is WorkerStartReason.HANDOFF_RESTART
        assert media.last_stop_reason is WorkerStopReason.KEEP_WARM_EXPIRY
        assert chat.starts_by_reason == NO_STARTS, "one worker's counts reached the other"
        assert chat.last_start_reason is None
        assert chat.counted_since == media.counted_since
    finally:
        await supervisor.close()


async def test_the_worker_routes_count_their_stops_and_starts_as_manual(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    services = app.state.services
    supervisor: ProcessSupervisor = services.processes

    listed = await client.get("/api/workers")
    assert listed.status_code == 200, listed.text
    for worker in listed.json():
        assert worker["starts_by_reason"] == {reason.value: 0 for reason in WorkerStartReason}
        assert worker["stops_by_reason"] == {reason.value: 0 for reason in WorkerStopReason}
        assert worker["emptied_by_reason"] == {reason.value: 0 for reason in WorkerStopReason}
        assert worker["exits_without_stop"] == 0
        assert worker["last_start_reason"] is None
        assert worker["last_stop_reason"] is None
        assert datetime.fromisoformat(worker["counted_since"]).utcoffset() is not None

    media_record, media_process = _running_record(settings, "media", 987_654_391)
    supervisor._workers["media"] = media_record
    stopped = await client.post("/api/workers/media/stop")
    assert stopped.status_code == 200, stopped.text
    assert media_process.returncode is not None
    assert stopped.json()["stops_by_reason"]["manual"] == 1
    assert stopped.json()["last_stop_reason"] == "manual"

    chat_record, chat_process = _running_record(settings, "chat", 987_654_392)
    supervisor._workers["chat"] = chat_record
    reset = await client.post("/api/workers/chat/reset")
    assert reset.status_code == 200, reset.text
    assert chat_process.returncode is not None
    assert reset.json()["worker"]["stops_by_reason"]["manual"] == 1

    # Starting ComfyUI is out of reach here, so the start itself is a double;
    # the stop in the restart is the supervisor's own.
    services.settings.media_engine = "comfyui"
    start_media = AsyncMock(side_effect=lambda **_options: _status(supervisor))
    monkeypatch.setattr(supervisor, "start_media", start_media)
    media_record, media_process = _running_record(settings, "media", 987_654_393)
    supervisor._workers["media"] = media_record
    restarted = await client.post("/api/workers/media/restart")
    assert restarted.status_code == 200, restarted.text
    assert media_process.returncode is not None
    started = await client.post("/api/workers/media/start")
    assert started.status_code == 200, started.text

    assert [call.kwargs for call in start_media.await_args_list] == [
        {"reason": WorkerStartReason.MANUAL},
        {"reason": WorkerStartReason.MANUAL},
    ]
    workers = {worker["name"]: worker for worker in (await client.get("/api/workers")).json()}
    assert workers["media"]["stops_by_reason"]["manual"] == 2
    assert workers["chat"]["stops_by_reason"]["manual"] == 1
    assert workers["media"]["exits_without_stop"] == 0


async def test_a_spawn_that_fails_counts_no_start(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, _spawned = _supervisor(settings, monkeypatch)

    async def refuse_to_spawn(*_command: str, **_options: object) -> _Process:
        raise OSError("neutral spawn failure")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse_to_spawn)
    try:
        with pytest.raises(OSError, match="neutral spawn failure"):
            await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)

        media = _status(supervisor)
        assert media.starts_by_reason == NO_STARTS
        assert media.last_start_reason is None
    finally:
        await supervisor.close()


async def test_a_start_refused_once_the_old_process_ended_counts_its_stop_and_no_start(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spawned = _supervisor(settings, monkeypatch)
    asked: list[bool] = []

    def refuse_once_it_ended() -> None:
        # Asked before the live worker is stopped, and again after it ended.
        asked.append(spawned[0].returncode is not None)
        if asked[-1]:
            raise RuntimeError("neutral claim refusal")

    try:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
        with pytest.raises(RuntimeError, match="neutral claim refusal"):
            await supervisor._replace(
                "media",
                OTHER_LAUNCH,
                HEALTH,
                before_replace=refuse_once_it_ended,
                reason=WorkerStartReason.RUN,
            )

        media = _status(supervisor)
        assert asked == [False, True]
        assert len(spawned) == 1
        assert media.starts_by_reason == {**NO_STARTS, WorkerStartReason.RUN: 1}
        assert media.stops_by_reason == {**NO_STOPS, WorkerStopReason.REPLACED_COMMAND: 1}
    finally:
        await supervisor.close()


@pytest.mark.parametrize("cut_short", ["cannot-confirm", "cancelled"])
async def test_a_stop_cut_short_counts_the_end_once_under_its_own_reason(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, cut_short: str
) -> None:
    """Nothing is counted while the process runs on, and its end is that stop's."""

    settings.worker_shutdown_seconds = 1 if cut_short == "cannot-confirm" else PATIENCE_SECONDS
    supervisor, spawned = _supervisor(settings, monkeypatch, _StubbornProcess)
    try:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
        process = spawned[0]
        assert isinstance(process, _StubbornProcess)
        if cut_short == "cannot-confirm":
            with pytest.raises(WorkerStopIncomplete):
                await supervisor.stop("media", reason=WorkerStopReason.CHAT_LOAD)
        else:
            stopping = asyncio.create_task(
                supervisor.stop("media", reason=WorkerStopReason.CHAT_LOAD)
            )
            await asyncio.wait_for(process.asked_to_stop.wait(), timeout=PATIENCE_SECONDS)
            stopping.cancel()
            with pytest.raises(asyncio.CancelledError):
                await stopping

        still_running = _status(supervisor)
        assert still_running.stops_by_reason == NO_STOPS, "counted while the process ran on"
        assert still_running.exits_without_stop == 0

        process.end(0)
        seen = _status(supervisor)
        stopped_later = await supervisor.stop("media", reason=WorkerStopReason.MANUAL)

        for status in (seen, stopped_later):
            assert status.stops_by_reason == {**NO_STOPS, WorkerStopReason.CHAT_LOAD: 1}
            assert status.last_stop_reason is WorkerStopReason.CHAT_LOAD
            assert status.exits_without_stop == 0
    finally:
        if spawned and spawned[0].returncode is None:
            spawned[0].end(0)
        await supervisor.close()


async def test_a_chat_load_cancelled_while_it_releases_the_media_worker_counts_chat_load(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.worker_shutdown_seconds = PATIENCE_SECONDS
    supervisor, spawned = _supervisor(settings, monkeypatch, _StubbornProcess)

    @asynccontextmanager
    async def lease(_device: str = "primary") -> AsyncIterator[None]:
        yield

    orchestrator = ConversationOrchestrator(
        engines=Mock(
            spec_set=["settings"],
            settings=SimpleNamespace(media_engine="comfyui", media_keep_warm_seconds=60),
        ),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(
            spec_set=["lease", "peek_next_eligible_job"],
            lease=lease,
            peek_next_eligible_job=lambda _group: None,
        ),
        processes=supervisor,
    )
    try:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
        process = spawned[0]
        assert isinstance(process, _StubbornProcess)
        # Kept warm after the run, as its handoff leaves it.
        orchestrator._media_warm_window = 1
        orchestrator._media_warm_generation = 1
        orchestrator._media_warm_launch = process.pid

        release = asyncio.create_task(orchestrator.release_media_for_chat())
        await asyncio.wait_for(process.asked_to_stop.wait(), timeout=PATIENCE_SECONDS)
        release.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release
        settlement = orchestrator._media_warm_task
        if settlement is not None:
            await asyncio.gather(settlement, return_exceptions=True)

        assert _status(supervisor).stops_by_reason == NO_STOPS
        process.end(0)
        assert _status(supervisor).stops_by_reason == {
            **NO_STOPS,
            WorkerStopReason.CHAT_LOAD: 1,
        }
    finally:
        if spawned and spawned[0].returncode is None:
            spawned[0].end(0)
        await supervisor.close()


async def test_a_manual_start_over_a_live_worker_of_another_scope_is_a_manual_stop(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spawned = _supervisor(settings, monkeypatch)
    try:
        await supervisor._replace(
            "media", LAUNCH, HEALTH, launch_scope_sha256=SCOPE, reason=WorkerStartReason.RUN
        )
        await supervisor._replace("media", OTHER_LAUNCH, HEALTH, reason=WorkerStartReason.MANUAL)

        media = _status(supervisor)
        assert spawned[0].returncode is not None
        assert media.stops_by_reason == {**NO_STOPS, WorkerStopReason.MANUAL: 1}
    finally:
        await supervisor.close()


def _chat_profile(settings: Settings) -> str:
    model = settings.model_dir / "neutral-chat" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"neutral model bytes")
    with SessionLocal() as session:
        install = ModelInstall(
            name="Neutral chat model",
            role="chat",
            engine="llama.cpp",
            local_path=str(model),
            active=True,
        )
        session.add(install)
        session.flush()
        profile = ModelProfile(
            name="Neutral chat profile",
            role="chat",
            engine="llama.cpp",
            model_install_id=install.id,
        )
        session.add(profile)
        session.commit()
        return profile.id


@pytest.mark.parametrize("loaded_by", ["the-controls", "a-scoped-run"])
async def test_restarting_chat_from_the_worker_controls_counts_manual(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loaded_by: str,
) -> None:
    supervisor: ProcessSupervisor = app.state.services.processes
    _fake_workers(supervisor, monkeypatch)
    runtime = tmp_path / "neutral-llama-server.exe"
    runtime.write_bytes(b"neutral runtime")
    monkeypatch.setattr(settings, "llama_executable", runtime)
    profile_id = _chat_profile(settings)
    if loaded_by == "the-controls":
        loaded = await client.post(f"/api/workers/chat/load/{profile_id}")
        assert loaded.status_code == 200, loaded.text
    else:
        with SessionLocal() as session:
            profile = session.get(ModelProfile, profile_id)
            assert profile is not None and profile.model_install_id is not None
            install = session.get(ModelInstall, profile.model_install_id)
            assert install is not None
            session.expunge(profile)
            session.expunge(install)
        # What a run's own load leaves: the scoped launch its accepted model
        # configuration names.
        await supervisor.load_chat(profile, install, launch_scope_sha256=SCOPE)

    restarted = await client.post("/api/workers/chat/restart")

    assert restarted.status_code == 200, restarted.text
    chat = restarted.json()
    assert chat["stops_by_reason"] == {
        **{reason.value: 0 for reason in WorkerStopReason},
        "manual": 1,
    }
    assert chat["starts_by_reason"]["manual"] == (2 if loaded_by == "the-controls" else 1)
    assert chat["last_start_reason"] == "manual"
    assert chat["last_stop_reason"] == "manual"


async def test_resetting_media_while_a_job_starts_it_counts_a_manual_stop(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = app.state.services
    supervisor: ProcessSupervisor = services.processes
    spawned = _fake_workers(supervisor, monkeypatch)
    entered = asyncio.Event()

    async def still_starting(_record: WorkerRecord, _url: str) -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(supervisor, "_wait_healthy", still_starting)
    with SessionLocal() as session:
        job = Job(kind="image", status="running", phase="running", payload_json={})
        session.add(job)
        session.commit()
        job_id = job.id
    # The job's execution is starting the worker for its run when the reset comes.
    start = asyncio.create_task(
        supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
    )
    services.orchestrator._tasks[job_id] = start
    try:
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)

        reset = await client.post("/api/workers/media/reset")

        assert reset.status_code == 200, reset.text
        assert reset.json()["cancelled_jobs"] == 1
        await asyncio.gather(start, return_exceptions=True)
        media = _status(supervisor)
        assert spawned[0].returncode is not None
        assert media.stops_by_reason == {**NO_STOPS, WorkerStopReason.MANUAL: 1}
    finally:
        if not start.done():
            start.cancel()
            await asyncio.gather(start, return_exceptions=True)
        services.orchestrator._tasks.pop(job_id, None)


async def test_a_reset_that_fails_before_its_stop_names_no_stop_for_a_later_start(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = app.state.services
    supervisor: ProcessSupervisor = services.processes
    spawned = _fake_workers(supervisor, monkeypatch)
    entered = asyncio.Event()

    async def still_starting(_record: WorkerRecord, _url: str) -> None:
        entered.set()
        await asyncio.Event().wait()

    async def held_database(_job_id: str) -> bool:
        raise OperationalError("UPDATE jobs", {}, Exception("database is locked"))

    monkeypatch.setattr(supervisor, "_wait_healthy", still_starting)
    monkeypatch.setattr(services.orchestrator, "cancel", held_database)
    with SessionLocal() as session:
        session.add(Job(kind="image", status="running", phase="running", payload_json={}))
        session.commit()

    with pytest.raises(OperationalError):
        await client.post("/api/workers/media/reset")

    # A start abandoned afterwards is counted under the stop its own reason causes.
    start = asyncio.create_task(
        supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.PACKAGE_PREPARATION)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start

        assert spawned[0].returncode is not None
        assert _status(supervisor).stops_by_reason == {
            **NO_STOPS,
            WorkerStopReason.PACKAGE_PREPARATION: 1,
        }
    finally:
        if not start.done():
            start.cancel()
            await asyncio.gather(start, return_exceptions=True)


async def test_shutting_down_while_a_restart_starts_the_media_worker_counts_a_shutdown(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = app.state.services
    supervisor: ProcessSupervisor = services.processes
    entered = asyncio.Event()

    async def still_starting(_record: WorkerRecord, _url: str) -> None:
        entered.set()
        await asyncio.Event().wait()

    async def start_media(**options: Any) -> WorkerStatus:
        # The launch ComfyUI would get is beside the point; the start is the supervisor's.
        await supervisor._replace("media", LAUNCH, HEALTH, reason=options["reason"])
        return _status(supervisor)

    async with app.router.lifespan_context(app):
        spawned = _fake_workers(supervisor, monkeypatch)
        monkeypatch.setattr(supervisor, "_wait_healthy", still_starting)
        monkeypatch.setattr(supervisor, "start_media", start_media)
        services.orchestrator.schedule_media_restart()
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)

    media = _status(supervisor)
    assert spawned[0].returncode is not None
    assert media.starts_by_reason == {**NO_STARTS, WorkerStartReason.HANDOFF_RESTART: 1}
    assert media.stops_by_reason == {**NO_STOPS, WorkerStopReason.SHUTDOWN: 1}


async def _finished_text_run(client: AsyncClient, project_id: str | None = None) -> dict[str, Any]:
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Neutral chat", **({"project_id": project_id} if project_id else {})},
        )
    ).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Name three primary colors.", "mode": "text"},
    )
    assert turn.status_code == 202, turn.text
    run_id = turn.json()["run"]["id"]

    async def read() -> dict[str, Any]:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        return cast(dict[str, Any], response.json())

    return dict(await wait_for_terminal_status(read, what=f"run {run_id}"))


async def test_a_chat_run_records_its_worker_without_the_service_wide_counts(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = app.state.services
    supervisor: ProcessSupervisor = services.processes
    _fake_workers(supervisor, monkeypatch)
    await supervisor._replace(
        "chat", ["neutral-chat-worker"], HEALTH, reason=WorkerStartReason.MANUAL
    )

    async def managed_chat_worker(_run_id: str) -> WorkerStatus:
        # What loading a managed chat profile returns: the supervisor's own status.
        return _status(supervisor, "chat")

    monkeypatch.setattr(services.orchestrator, "_ensure_chat_worker", managed_chat_worker)

    run = await _finished_text_run(client)

    worker = run["provenance_json"]["worker"]
    assert worker["name"] == "chat"
    assert worker["running"] is True
    assert SERVICE_WIDE_FIELDS.isdisjoint(worker), sorted(SERVICE_WIDE_FIELDS & set(worker))


async def test_a_project_bundle_carries_no_service_wide_counts(
    client: AsyncClient, tmp_path: Path
) -> None:
    project = (await client.post("/api/projects", json={"name": "Neutral project"})).json()
    run = await _finished_text_run(client, project["id"])
    # A worker record that carries the service-wide counts.
    counted = {
        "starts_by_reason": {"manual": 1},
        "stops_by_reason": {"manual": 0},
        "emptied_by_reason": {"run_handoff": 1},
        "exits_without_stop": 0,
        "last_start_reason": "manual",
        "last_stop_reason": None,
        "counted_since": "2026-01-01T00:00:00Z",
    }
    portable = {"name": "chat", "state": "ready"}
    with SessionLocal() as session:
        stored = session.get(Run, run["id"])
        assert stored is not None
        stored.provenance_json = {**stored.provenance_json, "worker": {**portable, **counted}}
        session.commit()

    exported = await client.post(f"/api/projects/{project['id']}/export")
    assert exported.status_code == 201, exported.text
    archive = await client.get(exported.json()["url"])
    manifest = _manifest(archive.content)
    assert manifest["runs"][0]["provenance_json"]["worker"] == portable

    # Importing a bundle whose worker records carry them drops them.
    manifest["runs"][0]["provenance_json"]["worker"] = {**portable, **counted}
    bundle = _rewrite_archive(archive.content, manifest)
    target_app = create_app(
        Settings(
            data_dir=tmp_path / "bundle-target", dev=True, chat_engine="mock", media_engine="mock"
        )
    )
    async with (
        target_app.router.lifespan_context(target_app),
        AsyncClient(
            transport=ASGITransport(app=target_app), base_url="http://testserver"
        ) as target,
    ):
        session_response = await target.post("/api/session")
        target.headers["x-local-lm-csrf"] = session_response.json()["csrf_token"]
        imported = await target.post(
            "/api/projects/import",
            files={"archive": ("neutral.lm-atelier.zip", bundle, "application/zip")},
        )
        assert imported.status_code == 201, imported.text
        with SessionLocal() as session:
            imported_chat = session.scalar(
                select(Chat).where(Chat.project_id == imported.json()["id"])
            )
            assert imported_chat is not None
            imported_run = session.scalar(select(Run).where(Run.chat_id == imported_chat.id))
            assert imported_run is not None
            assert imported_run.provenance_json["worker"] == portable
