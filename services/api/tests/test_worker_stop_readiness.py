from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import cast
from unittest.mock import AsyncMock

import psutil
import pytest
from fastapi import FastAPI, Request

from local_lm.api import _media_worker_truly_stopped, _require_media_worker_stopped
from local_lm.api_errors import ApiError
from local_lm.comfy_editor_bridge import ComfyEditorBridgeSupport
from local_lm.config import Settings
from local_lm.events import EventBroker
from local_lm.main import Services
from local_lm.processes import ProcessSupervisor, WorkerRecord


@pytest.fixture(autouse=True)
def _neutral_media_endpoint(settings: Settings, unused_tcp_port: int) -> None:
    settings.comfy_url = f"http://127.0.0.1:{unused_tcp_port}"


async def _start_neutral_worker(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exit_after_seconds: float | None = None,
) -> tuple[ProcessSupervisor, WorkerRecord, ComfyEditorBridgeSupport]:
    supervisor = ProcessSupervisor(settings, liveness_interval_seconds=60)
    ready_path = settings.data_dir / "neutral-worker-ready.json"

    async def wait_for_child_start(record: WorkerRecord, _health_url: str) -> None:
        async with asyncio.timeout(30):
            while True:
                try:
                    started = json.loads(ready_path.read_text())
                    child = psutil.Process(int(started["pid"]))
                    if child.create_time() == float(started["create_time"]) and (
                        child.pid == record.process.pid
                        or record.process.pid in {parent.pid for parent in child.parents()}
                    ):
                        return
                except (OSError, ValueError, KeyError, psutil.Error):
                    pass
                await asyncio.sleep(0.01)

    monkeypatch.setattr(supervisor, "_wait_healthy", AsyncMock(side_effect=wait_for_child_start))
    support = ComfyEditorBridgeSupport(
        True, "ready", "Native workflow editing is available.", "0.28.0", "1.45.21"
    )
    await supervisor._replace(
        "media",
        [
            sys.executable,
            "-c",
            "import json, pathlib, psutil, sys, time; process = psutil.Process(); "
            "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
            "'pid': process.pid, 'create_time': process.create_time()})); "
            + (
                "time.sleep(60)"
                if exit_after_seconds is None
                else f"time.sleep({exit_after_seconds}); raise SystemExit(9)"
            ),
            str(ready_path),
        ],
        f"{settings.comfy_url}/health",
        launch_scope_sha256="a" * 64,
        editor_bridge_support=support,
    )
    record = supervisor._workers["media"]
    assert supervisor.launch_scope_sha256("media") == "a" * 64
    assert supervisor.workflow_editor_runtime_identity()
    assert supervisor.workflow_editor_bridge_support() == support
    return supervisor, record, support


def _assert_stopping(supervisor: ProcessSupervisor, app: FastAPI, *, running: bool) -> None:
    status = next(item for item in supervisor.statuses() if item.name == "media")
    assert status.state == "stopping"
    assert status.running is running
    assert status.managed
    assert status.failure_detail is None
    assert status.failure_code is None
    assert status.stderr_tail is None
    assert supervisor.launch_scope_sha256("media") is None
    assert supervisor.workflow_editor_runtime_identity() is None
    assert supervisor.workflow_editor_bridge_support() is None
    services = cast(Services, app.state.services)
    services.processes = supervisor
    assert not _media_worker_truly_stopped(services)
    with pytest.raises(ApiError) as refused:
        _require_media_worker_stopped(Request({"type": "http", "app": app}))
    assert refused.value.status_code == 409
    assert refused.value.code == "media-worker-running"


@pytest.mark.parametrize("phase", ["before_terminate", "output_drain"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_a_stopping_worker_cannot_be_reused_during_cleanup(
    settings: Settings,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    cancel: bool,
) -> None:
    supervisor, record, support = await _start_neutral_worker(settings, monkeypatch)
    first_identity = record.editor_bridge_launch_id
    entered = asyncio.Event()
    release = asyncio.Event()
    thread_release = threading.Event()
    original_descendants = supervisor._descendant_processes
    loop = asyncio.get_running_loop()

    def paused_descendants(pid: int) -> list[psutil.Process]:
        loop.call_soon_threadsafe(entered.set)
        if not thread_release.wait(timeout=30):
            raise RuntimeError("neutral shutdown barrier expired")
        return original_descendants(pid)

    if phase == "before_terminate":
        monkeypatch.setattr(supervisor, "_descendant_processes", paused_descendants)
    else:
        assert record.output_task is not None
        original_output_task = record.output_task

        async def finish_output() -> None:
            await original_output_task
            await record.process.wait()
            entered.set()
            await release.wait()

        record.output_task = asyncio.create_task(finish_output())

    task = asyncio.create_task(supervisor.stop("media"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=30)
        _assert_stopping(supervisor, app, running=phase == "before_terminate")
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert supervisor._workers.get("media") is record
            _assert_stopping(supervisor, app, running=phase == "before_terminate")
            if phase == "before_terminate":
                record.log.write(b"neutral output after cancelled stop\n")
            else:
                assert record.output_task is not None and not record.output_task.done()
        thread_release.set()
        release.set()
        if not cancel:
            stopped = await asyncio.wait_for(task, timeout=30)
            assert stopped.state == "stopped"
            assert not stopped.running and not stopped.managed
            assert _media_worker_truly_stopped(cast(Services, app.state.services))
        await supervisor._replace(
            "media",
            list(record.command),
            f"{settings.comfy_url}/health",
            launch_scope_sha256="a" * 64,
            editor_bridge_support=support,
        )
        replacement = supervisor._workers["media"]
        assert replacement is not record
        assert record.process.returncode is not None
        assert record.monitor_task is not None and record.monitor_task.done()
        assert record.output_task is not None and record.output_task.done()
        assert replacement.monitor_task is not None and not replacement.monitor_task.done()
        assert supervisor.launch_scope_sha256("media") == "a" * 64
        assert supervisor.workflow_editor_runtime_identity() != first_identity
        assert supervisor.workflow_editor_bridge_support() == support
        assert next(item for item in supervisor.statuses() if item.name == "media").state == "ready"
    finally:
        thread_release.set()
        release.set()
        monkeypatch.setattr(supervisor, "_descendant_processes", original_descendants)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await supervisor.close()


async def test_a_failed_termination_revokes_readiness_until_stop_is_retried(
    settings: Settings, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, record, _support = await _start_neutral_worker(settings, monkeypatch)
    original_terminate = record.process.terminate

    def fail_termination() -> None:
        raise RuntimeError("neutral termination failure")

    monkeypatch.setattr(record.process, "terminate", fail_termination)
    try:
        with pytest.raises(RuntimeError, match="neutral termination failure"):
            await supervisor.stop("media")
        assert supervisor._workers.get("media") is record
        _assert_stopping(supervisor, app, running=True)
        record.log.write(b"neutral output after failed stop\n")
        monkeypatch.setattr(record.process, "terminate", original_terminate)
        status = await supervisor.stop("media")
        assert status.state == "stopped" and not status.managed
        assert record.process.returncode is not None
    finally:
        monkeypatch.setattr(record.process, "terminate", original_terminate)
        await supervisor.close()


async def test_an_exit_event_waits_for_worker_cleanup_to_finish(
    settings: Settings, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, record, _support = await _start_neutral_worker(
        settings, monkeypatch, exit_after_seconds=2
    )
    events = EventBroker()
    supervisor.events = events
    entered = asyncio.Event()
    release = threading.Event()
    original_matches = supervisor._matching_worker_processes
    loop = asyncio.get_running_loop()

    def paused_matches(name: str) -> list[psutil.Process]:
        if name != "media":
            return original_matches(name)
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(timeout=30):
            raise RuntimeError("neutral exit cleanup barrier expired")
        return original_matches(name)

    monkeypatch.setattr(supervisor, "_matching_worker_processes", paused_matches)
    try:
        await asyncio.wait_for(entered.wait(), timeout=30)
        _assert_stopping(supervisor, app, running=False)
        assert not any(event.type == "worker.exited" for event in events.since(0))
        release.set()
        assert record.monitor_task is not None
        await asyncio.wait_for(record.monitor_task, timeout=30)
        status = next(item for item in supervisor.statuses() if item.name == "media")
        assert status.state == "exited" and status.exit_code == record.process.returncode
        exited = [event for event in events.since(0) if event.type == "worker.exited"]
        assert len(exited) == 1
        assert exited[0].payload == {
            "name": "media",
            "state": "exited",
            "exit_code": record.process.returncode,
        }
    finally:
        release.set()
        monkeypatch.setattr(supervisor, "_matching_worker_processes", original_matches)
        await supervisor.close()


@pytest.mark.parametrize("invalid", ["birth_time", "parent"])
async def test_a_startup_marker_must_belong_to_the_current_worker(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    supervisor, record, _support = await _start_neutral_worker(settings, monkeypatch)
    ready_path = settings.data_dir / "neutral-worker-ready.json"
    root = psutil.Process(record.process.pid)
    genuine = (
        ready_path.read_text()
        if ready_path.exists()
        else json.dumps({"pid": root.pid, "create_time": root.create_time()})
    )
    if invalid == "birth_time":
        marker = {"pid": root.pid, "create_time": root.create_time() + 1}
    else:
        unrelated = psutil.Process(os.getpid())
        marker = {"pid": unrelated.pid, "create_time": unrelated.create_time()}
    ready_path.write_text(json.dumps(marker))
    check = asyncio.create_task(supervisor._wait_healthy(record, f"{settings.comfy_url}/health"))
    try:
        await asyncio.sleep(0.05)
        assert not check.done()
        ready_path.write_text(genuine)
        await asyncio.wait_for(check, timeout=5)
    finally:
        ready_path.write_text(genuine)
        if not check.done():
            check.cancel()
            await asyncio.gather(check, return_exceptions=True)
        await supervisor.close()
