"""An activation cannot publish a runtime after its scheduler claim changes."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from httpx2 import AsyncClient
from test_activation_claim_ownership import _move_claim, _state
from test_activation_worker_replacement import _ControlledProcess
from test_runtime_provisioning import _write_manifest, _zip_bytes
from test_scheduler_claim_hold import _until

from local_lm import processes as processes_module
from local_lm import runtime_provisioning
from local_lm import scheduler as scheduler_module
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
    Job,
    ModelInstall,
)
from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog
from local_lm.runtime_config import runtime_config_path
from local_lm.runtime_provisioning import RuntimeProvisioner
from local_lm.scheduler import ResourceScheduler


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "owned"])
async def test_media_activation_runtime_publication_requires_the_same_claim(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
    shared: bool = False,
) -> None:
    role: Literal["image"] = "image"
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.02 if shared else 60)
    settings.comfy_executable = None
    settings.comfy_directory = None
    content = _zip_bytes(
        {
            "python/python.exe": b"neutral runtime executable",
            "python/Lib/site-packages/example-1.0.dist-info/METADATA": (
                b"Name: example\nVersion: 1.0\n"
            ),
            "ComfyUI/main.py": b"neutral runtime source",
            "ComfyUI/custom_nodes/.keep": b"",
        }
    )
    manifest = tmp_path / "engines.json"
    _write_manifest(
        manifest, llama_content=b"unused", comfy_content=content, comfy_release="v0.28.0"
    )
    probe = {"python": "3.13.14", "comfyui": "0.28.0", "packages": {"example": "1.0"}}

    def run(command: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=runtime_provisioning._RUNTIME_PROBE_SENTINEL + json.dumps(probe) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    requests: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "runtime.test"
        requests.append(request)
        return httpx.Response(200, content=content)

    transport = httpx.AsyncClient(transport=httpx.MockTransport(serve))
    provisioner = RuntimeProvisioner(
        settings,
        manifest_path=manifest,
        client=transport,
        environment={},
        platform_key="test-platform",
        allowed_download_hosts={"runtime.test"},
    )
    runtime_entered, runtime_release, runtime_finished = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_install = provisioner._install_archive

    def install(*args: Any, **kwargs: Any) -> dict[str, Path]:
        result = original_install(*args, **kwargs)
        runtime_entered.set()
        try:
            assert runtime_release.wait(timeout=30), (
                "The installed runtime thread was not released."
            )
        finally:
            runtime_finished.set()
        return result

    monkeypatch.setattr(provisioner, "_install_archive", install)
    model_root = settings.model_dir / "activation-fixture"
    model_root.mkdir(parents=True)
    (model_root / "fixture.safetensors").write_bytes(b"neutral tensor fixture")
    supervisor = ProcessSupervisor(settings)
    supervisor.runtimes = provisioner
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
    monkeypatch.setattr(supervisor, "_clear_cancelled_workflow_activations", AsyncMock())

    async def trusted_nodes() -> list[str]:
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
    lock = asyncio.Lock()
    supervisor._locks["media"] = lock
    await lock.acquire()
    operation = "text_to_image"
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
    shared_task: asyncio.Task[Any] | None = None
    ensure_entered = asyncio.Event()
    if shared:
        provisioner.start("comfyui")
        await _until(runtime_entered.is_set)
        shared_task = provisioner._tasks["comfyui"]
        original_ensure = provisioner.ensure

        async def ensure(*args: Any, **kwargs: Any) -> Any:
            ensure_entered.set()
            return await original_ensure(*args, **kwargs)

        monkeypatch.setattr(provisioner, "ensure", ensure)
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
    try:
        await _until(runtime_entered.is_set)
        if shared:
            await asyncio.wait_for(ensure_entered.wait(), timeout=2)
        heartbeat = next(
            item
            for item in asyncio.all_tasks()
            if item.get_name() == "job-heartbeat-activation-claim"
        )
        assert not runtime_finished.is_set()
        assert len(requests) == 1 and settings.comfy_executable is None
        assert not task.done()
        supervisor._workers["media"] = record
        if disposition != "owned":
            _move_claim(disposition)
        before = _state()
        saved_config = runtime_config_path(settings.data_dir)
        sentinel = b'{"LOCAL_LM_COMFY_EXECUTABLE": "neutral-replacement-runtime"}\n'
        if disposition != "owned" and not shared:
            saved_config.write_bytes(sentinel)
        lock.release()
        if shared and disposition != "owned":
            await _until(heartbeat.done)
            assert shared_task is not None and not shared_task.cancelling()
        runtime_release.set()
        await _until(runtime_finished.is_set)
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=6)
        if shared:
            assert shared_task is not None
            await asyncio.wait_for(asyncio.shield(shared_task), timeout=6)
            assert shared_task.done() and not shared_task.cancelled()
            assert shared_task.exception() is None
            assert settings.comfy_executable is not None and settings.comfy_directory is not None
            assert provisioner.preflight("comfyui").operation == "reuse_managed"
        if disposition == "owned":
            assert settings.comfy_executable is not None and settings.comfy_directory is not None
            assert provisioner.preflight("comfyui").operation == "reuse_managed"
            assert json.loads(saved_config.read_bytes())["LOCAL_LM_COMFY_EXECUTABLE"] == str(
                settings.comfy_executable
            )
            assert replacement.terminated and len(created) == 1
            assert supervisor._workers["media"].process is created[0]
            assert supervisor._workers["media"].state == "ready"
            with SessionLocal() as session:
                job = session.get(Job, "activation-claim")
                assert job is not None and job.status == JobStatus.COMPLETE.value
        else:
            if not shared:
                assert saved_config.read_bytes() == sentinel
                assert settings.comfy_executable is None and settings.comfy_directory is None
            assert not replacement.terminated
            assert not created
            assert supervisor._workers["media"] is record
            assert record.state == "ready" and not record.stopping
            assert _state() == before
    finally:
        runtime_release.set()
        task.cancel()
        if lock.locked():
            lock.release()
        await asyncio.gather(task, return_exceptions=True)
        await supervisor.stop("media")
        record.log.close()
        await provisioner.close()
        await transport.aclose()


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "owned"])
async def test_a_claim_change_cannot_cancel_an_existing_shared_runtime_setup(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
) -> None:
    await test_media_activation_runtime_publication_requires_the_same_claim(
        client, settings, tmp_path, monkeypatch, disposition, shared=True
    )
