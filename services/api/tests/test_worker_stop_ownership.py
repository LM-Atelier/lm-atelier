from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import Mock

import pytest

from local_lm.config import Settings
from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog


def _running_record(settings: Settings, name: str) -> WorkerRecord:
    settings.prepare()
    process = cast(
        asyncio.subprocess.Process,
        Mock(spec=asyncio.subprocess.Process, pid=987_654_321, returncode=None),
    )
    return WorkerRecord(
        name=name,
        process=process,
        command=["neutral-worker"],
        log=_RotatingWorkerLog(settings.log_dir / f"{name}-stop.log"),
        state="ready",
    )


@pytest.mark.parametrize("name", ["chat", "media"])
async def test_a_failed_stop_keeps_the_worker_record_for_retry(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    supervisor = ProcessSupervisor(settings)
    record = _running_record(settings, name)
    supervisor._workers[name] = record
    calls: list[WorkerRecord] = []

    async def terminate(current: WorkerRecord) -> None:
        calls.append(current)
        if len(calls) == 1:
            raise RuntimeError("neutral shutdown failure")

    monkeypatch.setattr(supervisor, "_terminate_record", terminate)
    try:
        with pytest.raises(RuntimeError, match="neutral shutdown failure"):
            await supervisor._stop_unlocked(name)
        assert supervisor._workers.get(name) is record
        await supervisor._stop_unlocked(name)
        assert calls == [record, record]
        assert name not in supervisor._workers
    finally:
        record.log.close()


@pytest.mark.parametrize("name", ["chat", "media"])
async def test_a_cancelled_stop_keeps_the_worker_record_for_retry(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    supervisor = ProcessSupervisor(settings)
    record = _running_record(settings, name)
    supervisor._workers[name] = record
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[WorkerRecord] = []

    async def terminate(current: WorkerRecord) -> None:
        calls.append(current)
        entered.set()
        await release.wait()

    monkeypatch.setattr(supervisor, "_terminate_record", terminate)
    task = asyncio.create_task(supervisor._stop_unlocked(name))
    try:
        await asyncio.wait_for(entered.wait(), timeout=30)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert supervisor._workers.get(name) is record
        release.set()
        await supervisor._stop_unlocked(name)
        assert calls == [record, record]
        assert name not in supervisor._workers
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        record.log.close()


@pytest.mark.parametrize("name", ["chat", "media"])
async def test_a_completed_stop_removes_only_its_worker_record(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    supervisor = ProcessSupervisor(settings)
    record = _running_record(settings, name)
    replacement = _running_record(settings, name)
    supervisor._workers[name] = record

    async def terminate(current: WorkerRecord) -> None:
        assert current is record
        supervisor._workers[name] = replacement

    monkeypatch.setattr(supervisor, "_terminate_record", terminate)
    try:
        await supervisor._stop_unlocked(name)
        assert supervisor._workers.get(name) is replacement
    finally:
        record.log.close()
        replacement.log.close()
