from __future__ import annotations

import asyncio
import json
from typing import cast
from unittest.mock import AsyncMock

import psutil
import pytest
from run_waits import PATIENCE_SECONDS

from local_lm.config import Settings
from local_lm.processes import (
    ProcessSupervisor,
    WorkerRecord,
    _ProcessIdentity,
    _RotatingWorkerLog,
)
from local_lm.schemas import WorkerStatus


class _Survivor:
    pid = 987_654_321

    def __init__(self, *, denied: bool = False) -> None:
        self.denied = denied
        self.terminated = False
        self.killed = False

    def children(self, *, recursive: bool) -> list[psutil.Process]:
        assert recursive
        return []

    def terminate(self) -> None:
        self.terminated = True
        if self.denied:
            raise psutil.AccessDenied(self.pid)

    def kill(self) -> None:
        self.killed = True
        if self.denied:
            raise psutil.AccessDenied(self.pid)


@pytest.mark.parametrize("denied", [False, True])
def test_shutdown_reports_a_process_that_survives_both_waits(
    monkeypatch: pytest.MonkeyPatch, denied: bool
) -> None:
    process = _Survivor(denied=denied)
    waits: list[list[psutil.Process]] = []

    def wait(
        processes: list[psutil.Process], *, timeout: float
    ) -> tuple[list[psutil.Process], list[psutil.Process]]:
        assert timeout == 0.01
        waits.append(processes)
        return [], processes

    monkeypatch.setattr(psutil, "wait_procs", wait)
    with pytest.raises(ValueError, match="Worker shutdown did not complete"):
        ProcessSupervisor._terminate_processes([cast(psutil.Process, process)], 0.01)
    assert process.terminated and process.killed
    assert waits == [[process], [process]]


def test_startup_keeps_an_unstoppable_worker_identity_for_recovery(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.prepare()
    identity = _ProcessIdentity(_Survivor.pid, 123.0)
    payload = {"version": 1, "workers": {"media": [{"pid": identity.pid, "create_time": 123.0}]}}
    path = settings.state_dir / "worker-processes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    process = _Survivor()
    monkeypatch.setattr(
        ProcessSupervisor, "_matching_process", staticmethod(lambda _identity: process)
    )
    monkeypatch.setattr(
        ProcessSupervisor, "_identity_may_still_exist", staticmethod(lambda _identity: True)
    )
    monkeypatch.setattr(psutil, "wait_procs", lambda processes, **_kwargs: ([], processes))

    supervisor = ProcessSupervisor(settings)

    assert supervisor._worker_identities == {"media": [identity]}
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert supervisor.statuses()[0].name == "chat"


async def test_stop_refuses_when_a_persisted_identity_cannot_be_inspected(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = ProcessSupervisor(settings)
    identity = _ProcessIdentity(_Survivor.pid, 123.0)
    supervisor._worker_identities["media"] = [identity]
    monkeypatch.setattr(supervisor, "_matching_process", lambda _identity: None)
    monkeypatch.setattr(supervisor, "_identity_may_still_exist", lambda _identity: True)
    with pytest.raises(ValueError, match="Worker shutdown did not complete"):
        await supervisor.stop("media")
    assert supervisor._worker_identities == {"media": [identity]}


class _WaitingProcess:
    pid = 987_654_322
    returncode: int | None = None
    stdout = None
    stderr = None

    def __init__(self) -> None:
        self.waits = 0
        self.killed = False
        self.released = False

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.waits += 1
        if not self.released:
            await asyncio.Future[None]()
        self.returncode = 0
        return 0


async def test_an_incomplete_primary_shutdown_is_bounded_and_can_be_retried(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.prepare()
    settings.worker_shutdown_seconds = 0.01
    supervisor = ProcessSupervisor(settings)
    process = _WaitingProcess()
    record = WorkerRecord(
        "media",
        cast(asyncio.subprocess.Process, process),
        ["neutral-worker"],
        _RotatingWorkerLog(settings.log_dir / "neutral-stop.log"),
        state="ready",
        launch_scope_sha256="a" * 64,
    )
    supervisor._workers["media"] = record
    monkeypatch.setattr(supervisor, "_descendant_processes", lambda _pid: [])
    monkeypatch.setattr(supervisor, "_matching_worker_processes", lambda _name: [])
    try:
        with pytest.raises(ValueError, match="Worker shutdown did not complete"):
            await asyncio.wait_for(supervisor.stop("media"), timeout=1)
        assert process.waits == 2 and process.killed
        assert supervisor._workers["media"] is record
        assert supervisor.launch_scope_sha256("media") is None
        status = next(item for item in supervisor.statuses() if item.name == "media")
        assert status.state == "stopping" and status.managed and status.running
        assert status.failure_detail and "Worker shutdown did not complete" in status.failure_detail
        assert status.failure_remedy and "Try stopping" in status.failure_remedy
        record.log.write(b"neutral output after incomplete shutdown\n")
        process.released = True
        status = await supervisor.stop("media")
        assert status.state == "stopped" and not status.managed
    finally:
        process.released = True
        record.log.close()


async def test_cancelled_start_keeps_ownership_when_its_cleanup_fails(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    process = _WaitingProcess()
    entered = asyncio.Event()

    async def wait_healthy(_record: WorkerRecord, _url: str) -> None:
        entered.set()
        await asyncio.Future[None]()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(supervisor, "_wait_healthy", wait_healthy)
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())
    monkeypatch.setattr(supervisor, "_reclaim_port_from_our_own_children", AsyncMock())
    monkeypatch.setattr(supervisor, "_record_worker_process_tree", lambda _name, _pid: None)
    monkeypatch.setattr(
        supervisor,
        "_terminate_record",
        AsyncMock(side_effect=ValueError("neutral cleanup failure")),
    )
    task = asyncio.create_task(
        supervisor._replace("chat", ["neutral-worker"], "http://127.0.0.1:12341/health")
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
        record = supervisor._workers["chat"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert supervisor._workers.get("chat") is record
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for record in supervisor._workers.values():
            if record.output_task:
                await record.output_task
            record.log.close()


async def test_an_exited_primary_handle_does_not_hide_a_surviving_native_process(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    process = _WaitingProcess()
    process.pid = _Survivor.pid
    process.returncode = 1
    survivor = _Survivor()
    record = WorkerRecord(
        "media",
        cast(asyncio.subprocess.Process, process),
        ["neutral-worker"],
        _RotatingWorkerLog(settings.log_dir / "neutral-exited-stop.log"),
        state="ready",
    )
    supervisor._workers["media"] = record
    monkeypatch.setattr(supervisor, "_matching_worker_processes", lambda _name: [survivor])
    monkeypatch.setattr(psutil, "wait_procs", lambda processes, **_kwargs: ([], processes))
    try:
        with pytest.raises(ValueError, match="Worker shutdown did not complete"):
            await supervisor.stop("media")
        assert survivor.terminated and survivor.killed
        assert supervisor._workers["media"] is record
        assert record.stopping
    finally:
        record.log.close()


def test_shutdown_reports_an_owned_tree_that_cannot_be_enumerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Survivor()

    def children(*, recursive: bool) -> list[psutil.Process]:
        assert recursive
        raise psutil.AccessDenied(process.pid)

    monkeypatch.setattr(process, "children", children)
    monkeypatch.setattr(psutil, "wait_procs", lambda processes, **_kwargs: (processes, []))
    with pytest.raises(ValueError, match="Worker shutdown did not complete"):
        ProcessSupervisor._terminate_processes([cast(psutil.Process, process)], 0.01)
    assert process.terminated


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
async def test_closing_waits_for_every_stop_before_reporting_one_that_failed(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    supervisor = ProcessSupervisor(settings)
    waiting = asyncio.Event()
    failed = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def stop(name: str) -> None:
        if name == "media":
            await waiting.wait()
            failed.set()
            raise error_type("neutral shutdown failure")
        waiting.set()
        await release.wait()
        finished.set()

    monkeypatch.setattr(supervisor, "_stop_unlocked", stop)
    monkeypatch.setattr(
        supervisor,
        "statuses",
        lambda: [
            WorkerStatus(name=name, managed=False, running=False) for name in ("chat", "media")
        ],
    )
    closing = asyncio.create_task(supervisor.close())
    try:
        await asyncio.wait_for(failed.wait(), timeout=PATIENCE_SECONDS)
        for _ in range(10):
            await asyncio.sleep(0)
        assert not closing.done(), "Closing abandoned a worker whose stop was still running"
        assert not finished.is_set()
        release.set()
        with pytest.raises(error_type, match="neutral shutdown failure"):
            await closing
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(closing, return_exceptions=True)
        await asyncio.wait_for(finished.wait(), timeout=PATIENCE_SECONDS)
