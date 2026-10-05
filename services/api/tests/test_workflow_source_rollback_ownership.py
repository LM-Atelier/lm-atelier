"""A failed extension batch must not stop a worker belonging to another claim."""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from test_activation_worker_replacement import _ControlledProcess, _ObservedLock
from test_workflow_offer_packages import _setup
from test_workflow_package_execution_plan import _inputs
from test_workflow_package_import_endpoint import _ui_graph
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm import processes as processes_module
from local_lm import workflow_source_runtime
from local_lm.comfy_editor_bridge import ComfyEditorBridgeSupport, PreparedComfyEditorBridge
from local_lm.comfy_registry_downloads import ComfyRegistryArchiveDownloader
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.models import ComfyRegistryInstall, Job, WorkflowInstallOffer
from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog
from local_lm.workflow_completion_jobs import stage_workflow_completion_job
from local_lm.workflow_package_preparation import PreparationContext
from local_lm.workflow_source_extensions import (
    ExtensionPreparationServices,
    prepare_workflow_source_extensions,
)


def _state(offer_id: str, job_id: str) -> tuple[dict[str, object], ...]:
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        job = session.get(Job, job_id)
        assert offer is not None and job is not None
        rows = [offer, job, *session.scalars(select(ComfyRegistryInstall))]
        return deepcopy(
            tuple(
                {column.name: getattr(row, column.name) for column in row.__table__.columns}
                for row in rows
            )
        )


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "cancelled", "owned"])
async def test_source_batch_rollback_keeps_a_replacement_worker_after_waiting_for_its_lock(
    app: FastAPI,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    disposition: str,
) -> None:
    services = app.state.services
    manager: DownloadManager = services.downloads
    settings.comfy_executable = Path(sys.executable)
    supervisor = ProcessSupervisor(settings, runtimes=services.processes.runtimes)
    monkeypatch.setattr(services, "processes", supervisor)
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
    inputs = _inputs()
    original_probe = inputs["interpreter_probe"]

    async def probe(executable: Path) -> Any:
        environment, tags = await original_probe(executable)
        return environment, tags, ()

    inputs["interpreter_probe"] = probe
    provisioner = supervisor.runtimes
    assert provisioner is not None
    context = PreparationContext.from_settings(settings)
    _, context, offer_id, _saved = await _setup(
        tmp_path,
        package_inputs=[inputs],
        source_payload={
            "name": "Neutral rollback source",
            "operation": "text_to_image",
            "ui_graph": _ui_graph(),
            "dependencies": {"version": 1, "slots": []},
            "selections": [],
        },
        preparation_context=context,
        runtime_plan=provisioner.preflight("comfyui"),
        available_nodes=frozenset(source_runtime),
    )
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        job_id = stage_workflow_completion_job(session, offer).id
        session.commit()
    for node_type in inputs["selected"].node_types:
        source_runtime[node_type] = {
            "python_module": "custom_nodes.example",
            "input": {"required": {}},
            "output": [],
        }

    async def inventory() -> frozenset[str]:
        return frozenset(source_runtime)

    monkeypatch.setattr(supervisor, "comfy_node_inventory", inventory)
    entered, release = asyncio.Event(), asyncio.Event()

    async def validate(_graph: dict[str, Any]) -> list[str]:
        entered.set()
        await release.wait()
        return ["Constructed validation failure"]

    monkeypatch.setattr(services.engines.media, "validate_workflow", validate)
    created: list[_ControlledProcess] = []
    previous: WorkerRecord | None = None

    async def create_process(*_command: str, **_kwargs: object) -> _ControlledProcess:
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
    async with AsyncExitStack() as cleanup:
        archive = ComfyRegistryArchiveDownloader(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, content=inputs["state"]["content"])
            )
        )
        cleanup.push_async_callback(archive.close)
        wheels = ComfyRegistryWheelDownloader(
            transport=httpx.MockTransport(lambda _request: httpx.Response(500))
        )
        cleanup.push_async_callback(wheels.close)

        async def prepare_with_local_transport(*args: Any, **kwargs: Any) -> Any:
            return await prepare_workflow_source_extensions(
                *args,
                **kwargs,
                services=ExtensionPreparationServices(
                    inputs["interpreter_probe"],
                    inputs["registry_client"],
                    inputs["project_client"],
                    inputs["metadata_client"],
                    archive,
                    wheels,
                ),
            )

        monkeypatch.setattr(
            workflow_source_runtime,
            "prepare_workflow_source_extensions",
            prepare_with_local_transport,
        )
        manager.start_workflow_installation(offer_id)
        task = manager._offer_tasks[offer_id]
        try:
            await asyncio.wait_for(entered.wait(), timeout=6)
            assert len(created) == 1
            with SessionLocal() as session:
                install = session.scalar(select(ComfyRegistryInstall))
                assert install is not None and install.active and install.trusted
                assert install.review_json["activation_batch_v1"]["state"] == "verified"
            await lock.acquire()
            lock.entered.clear()
            release.set()
            await asyncio.wait_for(lock.entered.wait(), timeout=3)
            assert not task.done()
            previous = supervisor._workers["media"]
            supervisor._workers["media"] = replacement
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None and job.claim_owner is not None
                if disposition in {"cleared", "replaced"}:
                    job.claim_owner = None if disposition == "cleared" else "replacement-source"
                elif disposition == "cancelled":
                    job.status = "cancelled"
                session.commit()
            before = _state(offer_id, job_id)
            lock.release()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=6)
            process = replacement.process
            assert isinstance(process, _ControlledProcess)
            if disposition in {"cleared", "replaced"}:
                assert not process.terminated
                assert supervisor._workers["media"] is replacement
                assert replacement.state == "ready" and not replacement.stopping
                assert _state(offer_id, job_id) == before
            else:
                assert process.terminated
                with SessionLocal() as session:
                    install = session.scalar(select(ComfyRegistryInstall))
                    assert install is not None and not install.active and install.trusted
                    assert install.review_json["activation_batch_v1"]["state"] == "failed"
                    job = session.get(Job, job_id)
                    assert job is not None
                    assert job.status == ("cancelled" if disposition == "cancelled" else "failed")
                assert not next(
                    status for status in supervisor.statuses() if status.name == "media"
                ).running
            assert len(created) == 1
        finally:
            task.cancel()
            release.set()
            if lock.locked():
                lock.release()
            await asyncio.gather(task, return_exceptions=True)
            await manager.close()
            await supervisor.stop("media")
            for process in created:
                process.terminate()
            if previous is not None:
                previous.log.close()
            replacement.log.close()
