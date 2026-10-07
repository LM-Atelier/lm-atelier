"""Source startup keeps its current claim through worker and configuration waits."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_activation_worker_replacement import _ControlledProcess, _ObservedLock
from test_workflow_completion_jobs import _accept
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm import processes as processes_module
from local_lm.comfy_editor_bridge import (
    ComfyEditorBridgeSupport,
    PreparedComfyEditorBridge,
)
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.models import Job, WorkflowInstallOffer
from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog
from local_lm.workflow_activations import (
    WorkflowMediaLaunchScope,
    WorkflowSourceLaunchScope,
    materialize_comfy_runtime_dependency,
)
from local_lm.workflow_package_preparation import PreparationContext
from local_lm.workflow_source_launch import prepare_workflow_source_launch_scope


def _state(offer_id: str, job_id: str) -> tuple[dict[str, object], dict[str, object]]:
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        job = session.get(Job, job_id)
        assert offer is not None and job is not None
        return deepcopy(
            (
                {
                    column.name: getattr(offer, column.name)
                    for column in WorkflowInstallOffer.__table__.columns
                },
                {column.name: getattr(job, column.name) for column in Job.__table__.columns},
            )
        )


@pytest.mark.parametrize("boundary", ["lock", "model_paths"])
@pytest.mark.parametrize("disposition", ["cleared", "token_replaced", "attempt_replaced", "owned"])
async def test_source_startup_preserves_the_worker_and_configuration_of_a_newer_claim(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    disposition: str,
) -> None:
    manager: DownloadManager = app.state.services.downloads
    previous = app.state.services.processes
    supervisor = ProcessSupervisor(settings, runtimes=previous.runtimes)
    monkeypatch.setattr(app.state.services, "processes", supervisor)
    monkeypatch.setattr(manager, "processes", supervisor)
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
    scopes: list[WorkflowSourceLaunchScope] = []
    original_nodes = supervisor._scoped_comfy_node_folders

    async def scoped_nodes(
        scope: WorkflowMediaLaunchScope,
    ) -> tuple[list[str], tuple[str, ...]]:
        result = await original_nodes(scope)
        if isinstance(scope, WorkflowSourceLaunchScope) and not scopes:
            scopes.append(scope)
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(supervisor, "_scoped_comfy_node_folders", scoped_nodes)
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
    with monkeypatch.context() as deferred:
        deferred.setattr(manager, "start_workflow_installation", lambda _id: None)
        _, offer_id, job_id = await _accept(client)
    manager.start_workflow_installation(offer_id)
    task = manager._offer_tasks[offer_id]
    replacement = record("replacement-worker")
    replacement_paths = b"replacement configuration marker"
    try:
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
        assert len(scopes) == 1
        if boundary == "lock":
            await lock.acquire()
            lock.entered.clear()
            release.set()
            await asyncio.wait_for(lock.entered.wait(), timeout=PATIENCE_SECONDS)
        assert not task.done()
        supervisor._workers["media"] = replacement
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.claim_owner is not None
            if disposition != "owned":
                job.claim_owner = None if disposition == "cleared" else "replacement-workflow"
                if disposition == "attempt_replaced":
                    job.attempt += 1
                session.commit()
        before = _state(offer_id, job_id)
        provisioner = supervisor.runtimes
        assert provisioner is not None
        current_scope = prepare_workflow_source_launch_scope(
            SessionLocal,
            offer_id,
            context=PreparationContext.from_settings(settings),
            runtime_materializer=lambda requirement, selection: (
                materialize_comfy_runtime_dependency(provisioner, requirement, selection)
            ),
        )
        assert (current_scope.launch_sha256 != scopes[0].launch_sha256) is (
            disposition == "attempt_replaced"
        )
        model_paths = settings.state_dir / "comfy-launch" / f"{current_scope.launch_sha256}.yaml"
        if boundary == "model_paths":
            model_paths.parent.mkdir(exist_ok=True)
            model_paths.write_bytes(replacement_paths)
            release.set()
        else:
            lock.release()
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True),
            timeout=PATIENCE_SECONDS,
        )
        process = replacement.process
        assert isinstance(process, _ControlledProcess)
        if disposition == "owned":
            assert process.terminated
            assert _state(offer_id, job_id)[0]["status"] == "completed"
            assert _state(offer_id, job_id)[1]["status"] == "complete"
            if boundary == "model_paths":
                assert model_paths.read_bytes() != replacement_paths
        else:
            if boundary == "model_paths":
                assert model_paths.read_bytes() == replacement_paths
            assert not process.terminated
            assert not created
            assert supervisor._workers["media"] is replacement
            assert replacement.state == "ready" and not replacement.stopping
            assert _state(offer_id, job_id) == before
    finally:
        task.cancel()
        release.set()
        if lock.locked():
            lock.release()
        await asyncio.gather(task, return_exceptions=True)
        await manager.close()
        await supervisor.stop("media")
        for worker in records:
            worker.log.close()
