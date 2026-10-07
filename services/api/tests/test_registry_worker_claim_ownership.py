"""Registry setup rechecks its claim after waiting for the media worker lock."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_activation_worker_replacement import _ControlledProcess, _ObservedLock
from test_install_queue_registry import accept
from test_install_queue_registry import configured_registry as configured_registry

from local_lm import api as api_module
from local_lm.comfy_registry_lifecycle import ComfyRegistryPreparation
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import Job
from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog
from local_lm.schemas import WorkerStatus


def _snapshot(job_id: str) -> dict[str, object]:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        return deepcopy(
            {column.name: getattr(job, column.name) for column in Job.__table__.columns}
        )


@pytest.mark.parametrize("boundary", ["initial", "restore"])
@pytest.mark.parametrize("disposition", ["cleared", "replaced", "owned"])
async def test_registry_setup_preserves_a_replacement_worker_after_waiting_for_its_lock(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    disposition: str,
) -> None:
    supervisor = ProcessSupervisor(settings)
    monkeypatch.setattr(supervisor, "_matching_worker_processes", lambda _name: [])
    monkeypatch.setattr(supervisor, "_descendant_processes", lambda _pid: [])
    monkeypatch.setattr(supervisor, "_refresh_worker_identities_after_stop", lambda _name: None)
    monkeypatch.setattr(supervisor, "_process_tree_rss", lambda _pid: None)
    monkeypatch.setattr(supervisor, "_port_listener_description", lambda _host, _port: None)
    monkeypatch.setattr(app.state.services, "processes", supervisor)
    lock = _ObservedLock()
    supervisor._locks["media"] = lock
    await lock.acquire()
    lock.entered.clear()
    records: list[WorkerRecord] = []

    def record(label: str) -> WorkerRecord:
        worker = WorkerRecord(
            "media",
            _ControlledProcess(),
            ["neutral-media-worker"],
            _RotatingWorkerLog(tmp_path / f"{label}.log"),
            state="ready",
        )
        records.append(worker)
        return worker

    if boundary == "initial":
        supervisor._workers["media"] = record("prior-worker")
    prepared, release = asyncio.Event(), asyncio.Event()
    if boundary == "initial":
        release.set()

    async def prepare(*_args: object, **_kwargs: object) -> ComfyRegistryPreparation:
        prepared.set()
        await release.wait()
        return ComfyRegistryPreparation(
            "neutral-install",
            "neutral-nodes",
            "neutral-environment",
            "a" * 64,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            False,
        )

    async def start_media(**_kwargs: object) -> WorkerStatus:
        supervisor._workers["media"] = record("restored-worker")
        return next(status for status in supervisor.statuses() if status.name == "media")

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    monkeypatch.setattr(supervisor, "start_media", start_media)
    job_id = await accept(client)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    replacement = record("replacement-worker")
    try:
        if boundary == "restore":
            await asyncio.wait_for(prepared.wait(), timeout=3)
            release.set()
        await asyncio.wait_for(lock.entered.wait(), timeout=3)
        assert not task.done()
        supervisor._workers["media"] = replacement
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.claim_owner is not None
            if disposition != "owned":
                job.claim_owner = "replacement-registry" if disposition == "replaced" else None
                if disposition == "replaced":
                    job.attempt += 1
                session.commit()
        before = _snapshot(job_id)
        lock.release()
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        process = replacement.process
        assert isinstance(process, _ControlledProcess)
        if disposition == "owned":
            assert process.terminated
            assert _snapshot(job_id)["status"] == "complete"
        else:
            assert not process.terminated
            assert supervisor._workers["media"] is replacement
            assert replacement.state == "ready" and not replacement.stopping
            assert _snapshot(job_id) == before
    finally:
        release.set()
        task.cancel()
        if lock.locked():
            lock.release()
        await asyncio.gather(task, return_exceptions=True)
        await api_module.shutdown_registry_preparations()
        for worker in records:
            worker.log.close()
