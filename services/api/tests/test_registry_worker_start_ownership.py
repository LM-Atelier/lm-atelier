"""Registry restoration preserves the worker and configuration of a newer claim."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_activation_worker_replacement import _ControlledProcess, _ObservedLock
from test_install_queue_registry import accept
from test_install_queue_registry import configured_registry as configured_registry
from test_registry_worker_claim_ownership import _snapshot

from local_lm import api as api_module
from local_lm import processes as processes_module
from local_lm.comfy_editor_bridge import ComfyEditorBridgeSupport, PreparedComfyEditorBridge
from local_lm.comfy_registry_installs import ComfyRegistryLaunchContract
from local_lm.comfy_registry_lifecycle import ComfyRegistryPreparation
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import Job
from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog


@pytest.mark.parametrize("boundary", ["lock", "model_paths"])
@pytest.mark.parametrize("disposition", ["cleared", "replaced", "owned"])
async def test_registry_restoration_rechecks_the_claim_inside_worker_startup(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    disposition: str,
) -> None:
    executable = tmp_path / "neutral-runtime.exe"
    executable.write_bytes(b"neutral runtime marker")
    runtime = tmp_path / "neutral-media-runtime"
    runtime.mkdir()
    (runtime / "main.py").write_bytes(b"")
    settings.comfy_executable = executable
    settings.comfy_directory = runtime
    supervisor = ProcessSupervisor(settings)
    monkeypatch.setattr(app.state.services, "processes", supervisor)
    monkeypatch.setattr(supervisor, "_matching_worker_processes", lambda _name: [])
    monkeypatch.setattr(supervisor, "_descendant_processes", lambda _pid: [])
    monkeypatch.setattr(supervisor, "_refresh_worker_identities_after_stop", lambda _name: None)
    monkeypatch.setattr(supervisor, "_record_worker_process_tree", lambda _name, _pid: None)
    monkeypatch.setattr(supervisor, "_process_tree_rss", lambda _pid: None)
    monkeypatch.setattr(supervisor, "_port_listener_description", lambda _host, _port: None)
    for name in (
        "_reclaim_port_from_our_own_children",
        "_ensure_port_available",
        "_wait_healthy",
        "_capture_process_output",
        "_monitor_worker",
        "_clear_cancelled_workflow_activations",
    ):
        monkeypatch.setattr(supervisor, name, AsyncMock())
    monkeypatch.setattr(
        supervisor,
        "_verified_comfy_registry_contract",
        AsyncMock(return_value=(ComfyRegistryLaunchContract((), (), ()), None)),
    )
    monkeypatch.setattr(
        processes_module,
        "prepare_comfy_editor_bridge",
        lambda **_kwargs: PreparedComfyEditorBridge(
            ComfyEditorBridgeSupport(False, "workflow-editor-runtime-unavailable", "Unavailable"),
            None,
        ),
    )
    dependencies_entered, dependencies_release = asyncio.Event(), asyncio.Event()

    async def trusted_nodes() -> list[str]:
        dependencies_entered.set()
        await dependencies_release.wait()
        return []

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    created: list[_ControlledProcess] = []

    async def create_process(*_command: str, **_kwargs: object) -> _ControlledProcess:
        process = _ControlledProcess()
        created.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
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

    supervisor._workers["media"] = record("prior-worker")
    lock = _ObservedLock()
    supervisor._locks["media"] = lock

    async def prepare(*_args: object, **_kwargs: object) -> ComfyRegistryPreparation:
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

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    job_id = await accept(client)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    replacement = record("replacement-worker")
    model_paths = settings.state_dir / "comfy-extra-model-paths.yaml"
    replacement_paths = b"replacement configuration marker"
    try:
        await asyncio.wait_for(dependencies_entered.wait(), timeout=3)
        if boundary == "lock":
            await lock.acquire()
            lock.entered.clear()
            dependencies_release.set()
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
        if boundary == "model_paths":
            model_paths.write_bytes(replacement_paths)
            dependencies_release.set()
        else:
            lock.release()
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        process = replacement.process
        assert isinstance(process, _ControlledProcess)
        if disposition == "owned":
            assert process.terminated and len(created) == 1
            assert supervisor._workers["media"].process is created[0]
            assert supervisor._workers["media"].state == "ready"
            assert _snapshot(job_id)["status"] == "complete"
            if boundary == "model_paths":
                assert model_paths.read_bytes() != replacement_paths
        else:
            if boundary == "model_paths":
                assert model_paths.read_bytes() == replacement_paths
            assert not process.terminated
            assert not created
            assert supervisor._workers["media"] is replacement
            assert replacement.state == "ready" and not replacement.stopping
            assert _snapshot(job_id) == before
    finally:
        task.cancel()
        dependencies_release.set()
        if lock.locked():
            lock.release()
        await asyncio.gather(task, return_exceptions=True)
        await api_module.shutdown_registry_preparations()
        await supervisor.stop("media")
        for worker in records:
            worker.log.close()
