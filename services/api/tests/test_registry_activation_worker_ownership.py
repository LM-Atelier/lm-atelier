"""Package activation and compensation retain their authority inside startup waits."""

from __future__ import annotations

import asyncio
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from sqlalchemy import select
from test_activation_worker_replacement import _ControlledProcess, _ObservedLock
from test_install_queue_registry import accept
from test_install_queue_registry import configured_registry as configured_registry
from test_workflow_package_execution_plan import _inputs

from local_lm import api as api_module
from local_lm import processes as processes_module
from local_lm.comfy_editor_bridge import ComfyEditorBridgeSupport, PreparedComfyEditorBridge
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall, Job
from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog


def _state(job_id: str) -> tuple[dict[str, object], ...]:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        install = session.scalar(select(ComfyRegistryInstall))
        assert job is not None and install is not None
        return deepcopy(
            tuple(
                {column.name: getattr(row, column.name) for column in row.__table__.columns}
                for row in (job, install)
            )
        )


@pytest.mark.parametrize("phase", ["activation", "compensation"])
@pytest.mark.parametrize("boundary", ["lock", "model_paths"])
@pytest.mark.parametrize("disposition", ["cleared", "replaced", "cancelled", "owned"])
async def test_prepared_package_startup_preserves_claimed_workers_and_cancelled_cleanup(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    boundary: str,
    disposition: str,
) -> None:
    inputs = _inputs()
    wheels = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    )
    settings.comfy_executable = Path(sys.executable)
    runtime = tmp_path / "neutral-media-runtime"
    runtime.mkdir()
    (runtime / "main.py").write_bytes(b"")
    (runtime / "custom_nodes").mkdir()
    settings.comfy_directory = runtime
    monkeypatch.setattr(
        api_module, "probe_comfy_registry_runtime_target", inputs["interpreter_probe"]
    )
    for name, value in (
        ("ComfyRegistryClient", inputs["registry_client"]),
        ("ComfyRegistryWheelProjectClient", inputs["project_client"]),
        ("ComfyRegistryWheelMetadataClient", inputs["metadata_client"]),
        ("ComfyRegistryArchiveDownloader", inputs["archive_downloader"]),
        ("ComfyRegistryWheelDownloader", wheels),
    ):
        monkeypatch.setattr(api_module, name, lambda value=value: value)
    supervisor = ProcessSupervisor(settings)
    monkeypatch.setattr(
        supervisor,
        "comfy_node_inventory",
        AsyncMock(return_value=frozenset(inputs["selected"].node_types)),
    )
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
    ):
        monkeypatch.setattr(supervisor, name, AsyncMock())
    monkeypatch.setattr(
        processes_module,
        "prepare_comfy_editor_bridge",
        lambda **_kwargs: PreparedComfyEditorBridge(
            ComfyEditorBridgeSupport(False, "workflow-editor-runtime-unavailable", "Unavailable"),
            None,
        ),
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original_nodes = supervisor._trusted_comfy_node_folders
    starts = 0

    async def trusted_nodes() -> list[str]:
        nonlocal starts
        result = await original_nodes()
        starts += 1
        if starts == (1 if phase == "activation" else 2):
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    created: list[_ControlledProcess] = []
    launched_active: list[bool] = []
    attempts = 0

    async def create_process(*_command: str, **_kwargs: object) -> _ControlledProcess:
        nonlocal attempts
        attempts += 1
        if phase == "compensation" and attempts == 1:
            raise RuntimeError("Constructed media startup failure")
        with SessionLocal() as session:
            install = session.scalar(select(ComfyRegistryInstall))
            assert install is not None
            launched_active.append(install.active)
        process = _ControlledProcess()
        created.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    replacement = WorkerRecord(
        "media",
        _ControlledProcess(),
        ["neutral-replacement-worker"],
        _RotatingWorkerLog(tmp_path / "replacement.log"),
        state="ready",
    )
    lock = _ObservedLock()
    supervisor._locks["media"] = lock
    job_id = await accept(client)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    model_paths = settings.state_dir / "comfy-extra-model-paths.yaml"
    marker = b"replacement configuration marker"
    try:
        async with asyncio.timeout(30):
            while not entered.is_set():
                assert not task.done(), _state(job_id)[0]["phase"]
                await asyncio.sleep(0.01)
        assert starts == (1 if phase == "activation" else 2)
        with SessionLocal() as session:
            install = session.scalar(select(ComfyRegistryInstall))
            assert install is not None and install.trusted
            assert install.active is (phase == "activation")
        if boundary == "lock":
            await lock.acquire()
            lock.entered.clear()
            release.set()
            await asyncio.wait_for(lock.entered.wait(), timeout=3)
        supervisor._workers["media"] = replacement
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.claim_owner is not None
            if disposition in {"cleared", "replaced"}:
                job.claim_owner = None if disposition == "cleared" else "replacement-package"
            elif disposition == "cancelled":
                job.status = "cancelled"
            session.commit()
        before = _state(job_id)
        if boundary == "model_paths":
            model_paths.write_bytes(marker)
            release.set()
        else:
            lock.release()
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        process = replacement.process
        assert isinstance(process, _ControlledProcess)
        if disposition in {"cleared", "replaced"}:
            if boundary == "model_paths":
                assert model_paths.read_bytes() == marker
            assert not process.terminated
            assert supervisor._workers["media"] is replacement
            assert replacement.state == "ready" and not replacement.stopping
            assert not created
            assert _state(job_id) == before
        elif disposition == "cancelled":
            assert launched_active == [False]
            assert process.terminated
            assert _state(job_id)[0]["status"] == "cancelled"
            assert _state(job_id)[1]["active"] is False
        else:
            assert process.terminated
            assert launched_active == [phase == "activation"]
            assert _state(job_id)[0]["status"] == (
                "complete" if phase == "activation" else "failed"
            )
            assert _state(job_id)[1]["active"] is (phase == "activation")
        assert not next(
            status for status in supervisor.statuses() if status.name == "media"
        ).running or disposition in {"cleared", "replaced"}
    finally:
        task.cancel()
        release.set()
        if lock.locked():
            lock.release()
        await asyncio.gather(task, return_exceptions=True)
        await api_module.shutdown_registry_preparations()
        await supervisor.stop("media")
        for process in created:
            process.terminate()
        replacement.log.close()
        await inputs["archive_downloader"].close()
        await wheels.close()
