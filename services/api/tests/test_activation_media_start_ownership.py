"""A waiting media activation must preserve a newer worker."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, Mock

import pytest
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_activation_claim_ownership import _move_claim, _state
from test_activation_worker_replacement import _ControlledProcess, _ObservedLock

from local_lm import processes as processes_module
from local_lm.adapters.base import MediaRequest
from local_lm.comfy_editor_bridge import ComfyEditorBridgeSupport, PreparedComfyEditorBridge
from local_lm.comfy_registry_installs import ComfyRegistryLaunchContract
from local_lm.comfy_templates import ComfyModelDependency, ComfyTemplate, CompiledComfyTemplate
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import (
    ComfyRegistryInstall,
    Job,
    ModelInstall,
    WorkflowDefinition,
    WorkflowInstallOffer,
    WorkflowInstallOfferPackage,
    WorkflowRevision,
)
from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog
from local_lm.scheduler import ResourceScheduler
from local_lm.workflow_completion_jobs import (
    cancelled_workflow_activation_packages,
    stage_workflow_completion_job,
)


def _cancelled_package(settings: Settings) -> str:
    with SessionLocal() as session:
        definition = WorkflowDefinition(
            name="Neutral cancelled workflow", operation="text_to_image"
        )
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(workflow_id=definition.id, version=1)
        session.add(revision)
        session.flush()
        offer = WorkflowInstallOffer(
            workflow_revision_id=revision.id,
            workflow_artifact_sha256="a" * 64,
            dependency_contract_sha256="b" * 64,
            binding_plan_sha256="c" * 64,
            offer_sha256="d" * 64,
            plan_count=1,
            total_bytes=1,
            status="queued",
        )
        registry = ComfyRegistryInstall(
            package_id="neutral-cancelled-extension",
            package_version="1.0.0",
            registry_record_id="neutral-registry-record",
            repository_url="https://example.invalid/neutral-extension",
            download_url="https://example.invalid/neutral-extension.zip",
            archive_sha256="e" * 64,
            manifest_sha256="f" * 64,
            installed_path=str(settings.custom_node_dir / "neutral-extension"),
            review_json={"activation_batch_v1": {"state": "pending"}},
            active=True,
        )
        session.add_all([offer, registry])
        session.flush()
        job = stage_workflow_completion_job(session, offer)
        job.status = JobStatus.CANCELLED.value
        session.add(
            WorkflowInstallOfferPackage(
                offer_id=offer.id,
                offer_sha256=offer.offer_sha256,
                package_id=registry.package_id,
                execution_plan_sha256="a" * 64,
                registry_install_id=registry.id,
            )
        )
        session.commit()
        assert cancelled_workflow_activation_packages(session) == (registry.id,)
        return registry.id


@pytest.mark.parametrize("boundary", ["lock", "model_paths", "cancelled_workflows"])
@pytest.mark.parametrize("role", ["image", "video"])
@pytest.mark.parametrize("disposition", ["cleared", "replaced", "owned"])
async def test_waiting_media_activation_checks_the_claim_after_the_worker_lock(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: Literal["image", "video"],
    disposition: str,
    boundary: str,
) -> None:
    executable = tmp_path / "neutral-runtime.exe"
    executable.write_bytes(b"neutral runtime marker")
    runtime = tmp_path / "neutral-media-runtime"
    runtime.mkdir()
    (runtime / "main.py").write_bytes(b"")
    settings.comfy_executable = executable
    settings.comfy_directory = runtime
    model_root = settings.model_dir / "activation-fixture"
    model_root.mkdir(parents=True)
    (model_root / "fixture.safetensors").write_bytes(b"neutral tensor fixture")
    supervisor = ProcessSupervisor(settings)
    monkeypatch.setattr(supervisor, "_matching_worker_processes", lambda _name: [])
    monkeypatch.setattr(supervisor, "_descendant_processes", lambda _pid: [])
    monkeypatch.setattr(supervisor, "_refresh_worker_identities_after_stop", lambda _name: None)
    monkeypatch.setattr(supervisor, "_record_worker_process_tree", lambda _name, _pid: None)
    monkeypatch.setattr(supervisor, "_process_tree_rss", lambda _pid: None)
    monkeypatch.setattr(supervisor, "_reclaim_port_from_our_own_children", AsyncMock())
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())
    monkeypatch.setattr(supervisor, "_wait_healthy", AsyncMock())
    monkeypatch.setattr(supervisor, "_capture_process_output", AsyncMock())
    monkeypatch.setattr(supervisor, "_monitor_worker", AsyncMock())
    cancelled_package_id = (
        _cancelled_package(settings) if boundary == "cancelled_workflows" else None
    )
    if cancelled_package_id is None:
        monkeypatch.setattr(supervisor, "_clear_cancelled_workflow_activations", AsyncMock())
    dependencies_entered = asyncio.Event()
    dependencies_release = asyncio.Event()

    async def trusted_nodes() -> list[str]:
        if boundary == "model_paths":
            dependencies_entered.set()
            await dependencies_release.wait()
        return []

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
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
    created: list[_ControlledProcess] = []

    async def create_process(*_command: str, **_kwargs: object) -> _ControlledProcess:
        process = _ControlledProcess()
        created.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    lock = _ObservedLock()
    supervisor._locks["media"] = lock
    await lock.acquire()
    lock.entered.clear()
    if boundary == "model_paths":
        lock.release()
    operation = "text_to_image" if role == "image" else "text_to_video"
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="neutral-activation-" + role,
            path=settings.state_dir / "neutral-template.json",
            role=role,
            operation=operation,
            score=1,
            sha256="a" * 64,
            dependencies=(
                ComfyModelDependency(
                    remote_id="neutral/media",
                    revision="neutral",
                    path="fixture.safetensors",
                    directory="checkpoints",
                    name="fixture.safetensors",
                    url="",
                ),
            ),
        ),
        ui_graph={"nodes": []},
        api_graph={"latent": {"class_type": "EmptyLatentImage", "inputs": {}}},
        input_schema={"type": "object", "properties": {}},
    )

    class Media:
        async def object_info(self) -> dict[str, object]:
            return {}

        async def validate_workflow(self, _graph: dict[str, object]) -> list[str]:
            return []

        async def probe_workflow(self, request: MediaRequest, *, timeout_seconds: float) -> None:
            assert request.operation == operation and timeout_seconds == 300

        async def capabilities(self) -> SimpleNamespace:
            return SimpleNamespace(healthy=True, version="neutral-media-runtime")

    manager = DownloadManager(
        settings,
        EventBroker(),
        media_adapter=Mock(spec_set=Media, wraps=Media()),
        processes=supervisor,
        scheduler=ResourceScheduler(),
    )
    monkeypatch.setattr(manager.comfy_templates, "compile", lambda *_args, **_kwargs: compiled)
    with SessionLocal() as session:
        session.add_all(
            [
                ModelInstall(
                    id="activation-model",
                    name="Neutral media activation",
                    role=role,
                    engine="comfyui",
                    local_path=str(model_root),
                    manifest_json={
                        "expected_sha256": {"fixture.safetensors": "ab" * 32},
                        "workflow_template_id": compiled.template.id,
                        "workflow_template_sha256": compiled.template.sha256,
                        "files": ["fixture.safetensors"],
                        "remote_id": "neutral/media",
                        "revision": "neutral",
                        "comfy_paths": {"checkpoints": "."},
                    },
                    active=False,
                ),
                Job(
                    id="activation-claim",
                    kind=JobKind.ACTIVATE.value,
                    status=JobStatus.QUEUED.value,
                    payload_json={"install_id": "activation-model"},
                    queue_resource="primary_compute",
                    queue_group="primary",
                    enqueued_at=utcnow(),
                ),
            ]
        )
        session.commit()
    manager.start_activation("activation-claim")
    task = manager._tasks["activation-claim"]
    replacement = _ControlledProcess()
    record = WorkerRecord(
        "media",
        replacement,
        ["neutral-replacement-worker"],
        _RotatingWorkerLog(tmp_path / "replacement-worker.log"),
        state="ready",
    )
    model_paths = settings.state_dir / "comfy-extra-model-paths.yaml"
    replacement_paths = b"replacement configuration marker"
    try:
        entered = dependencies_entered if boundary == "model_paths" else lock.entered
        await asyncio.wait_for(entered.wait(), timeout=30)
        assert not task.done()
        supervisor._workers["media"] = record
        if disposition != "owned":
            _move_claim(disposition)
        before = _state()
        if boundary != "model_paths":
            lock.release()
        else:
            model_paths.write_bytes(replacement_paths)
            dependencies_release.set()
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        if disposition == "owned":
            assert replacement.terminated and len(created) == 1
            assert supervisor._workers["media"].process is created[0]
            assert supervisor._workers["media"].state == "ready"
            if boundary == "model_paths":
                assert model_paths.read_bytes() != replacement_paths
            with SessionLocal() as session:
                job = session.get(Job, "activation-claim")
                assert job is not None and job.status == JobStatus.COMPLETE.value
        else:
            if boundary == "model_paths":
                assert model_paths.read_bytes() == replacement_paths
            assert not replacement.terminated
            assert not created
            assert supervisor._workers["media"] is record
            assert record.state == "ready" and not record.stopping
            assert _state() == before
        if cancelled_package_id is not None:
            with SessionLocal() as session:
                package = session.get(ComfyRegistryInstall, cancelled_package_id)
                assert package is not None
                if disposition == "owned":
                    assert not package.active
                    assert package.review_json["activation_failure_code"] == "registry_batch_failed"
                else:
                    assert package.active
                    assert package.review_json == {"activation_batch_v1": {"state": "pending"}}
    finally:
        task.cancel()
        dependencies_release.set()
        if lock.locked():
            lock.release()
        await asyncio.gather(task, return_exceptions=True)
        await supervisor.stop("media")
        record.log.close()
