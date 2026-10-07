"""Completed runtime files cannot replace configuration after their claim changes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from sqlalchemy import select
from test_runtime_provisioning import _write_manifest, _zip_bytes
from test_scheduler_claim_hold import _until
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import source_runtime as source_runtime
from test_workflow_source_fresh_runtime_live import _source_graph
from test_workflow_source_thread_ownership import _state

from local_lm import runtime_provisioning
from local_lm import scheduler as scheduler_module
from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.models import Job, WorkflowInstallOffer
from local_lm.runtime_config import runtime_config_path
from local_lm.runtime_provisioning import RuntimeProvisioner

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "cancelled", "owned"])
async def test_fresh_runtime_publication_requires_the_same_claim_after_installation(
    client: AsyncClient,
    app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    disposition: str,
) -> None:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 60)
    services = app.state.services
    manager: DownloadManager = services.downloads
    settings = services.settings
    await services.processes.stop("media")
    settings.comfy_executable = None
    settings.comfy_directory = None
    source_runtime.clear()
    source_runtime.update(
        {
            "EmptyImage": {
                "python_module": "nodes",
                "input": {
                    "required": {
                        "width": ["INT", {"default": 64}],
                        "height": ["INT", {"default": 96}],
                        "batch_size": ["INT", {"default": 1}],
                        "color": ["INT", {"default": 0}],
                    }
                },
                "input_order": {"required": ["width", "height", "batch_size", "color"]},
                "output": ["IMAGE"],
            },
            "SaveImage": {
                "python_module": "nodes",
                "input": {
                    "required": {
                        "images": ["IMAGE"],
                        "filename_prefix": ["STRING", {"default": "constructed"}],
                    }
                },
                "input_order": {"required": ["images", "filename_prefix"]},
                "output": [],
                "output_node": True,
            },
        }
    )
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

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as transport:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=transport,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        monkeypatch.setattr(services, "runtimes", provisioner)
        monkeypatch.setattr(services.processes, "runtimes", provisioner)
        asset = provisioner._definition("comfyui")["runtime_assets"]["test-platform"]
        catalog = {
            "version": 1,
            "engine": "comfyui",
            "release": "v0.28.0",
            "asset_key": "test-platform",
            "archive_sha256": asset["sha256"],
            "runtime_contract_sha256": provisioner._runtime_contract_sha256(asset),
            "nodes": {"EmptyImage": "nodes", "SaveImage": "nodes"},
        }
        relative = "runtime-reviews/constructed-nodes.json"
        catalog_bytes = json.dumps(catalog).encode()
        (manifest.parent / relative).write_bytes(catalog_bytes)
        asset["workflow_node_inventory"] = {
            "file": relative,
            "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "asset_key": "test-platform",
        }
        original_install = provisioner._install_archive

        def install(*args: Any, **kwargs: Any) -> dict[str, Path]:
            result = original_install(*args, **kwargs)
            entered.set()
            try:
                assert release.wait(timeout=30), "The installed runtime thread was not released."
            finally:
                finished.set()
            return result

        monkeypatch.setattr(provisioner, "_install_archive", install)
        with monkeypatch.context() as deferred:
            deferred.setattr(manager, "start_workflow_installation", lambda _id: None)
            preview = await client.post(
                "/api/workflows/packages/install-plans",
                json={
                    "name": "Constructed runtime publication",
                    "operation": "text_to_image",
                    "ui_graph": _source_graph(),
                    "dependencies": {"version": 1, "slots": []},
                    "selections": [],
                },
            )
            assert preview.status_code == 201 and preview.json()["can_accept"], preview.text
            assert preview.json()["runtime_plan"]["operation"] == "install_managed"
            plan_id = preview.json()["id"]
            accepted = await client.post(f"/api/workflow-install-offers/{plan_id}/install")
            assert accepted.status_code == 202, accepted.text
            assert len(accepted.json()) == 1 and accepted.json()[0]["kind"] == "workflow_install"
            job_id = accepted.json()[0]["id"]
        with SessionLocal() as session:
            offer = session.scalar(
                select(WorkflowInstallOffer).where(WorkflowInstallOffer.source_plan_id == plan_id)
            )
            assert offer is not None and offer.completion_job_id == job_id
            offer_id = offer.id
        manager.start_workflow_installation(offer_id)
        task = manager._offer_tasks[offer_id]
        saved_config = runtime_config_path(settings.data_dir)
        sentinel = b'{"LOCAL_LM_COMFY_EXECUTABLE": "neutral-replacement-runtime"}\n'
        try:
            await _until(entered.is_set, timeout=30)
            assert not task.done() and not finished.is_set()
            assert len(requests) == 1 and settings.comfy_executable is None
            heartbeat = next(
                item for item in asyncio.all_tasks() if item.get_name() == f"job-heartbeat-{job_id}"
            )
            if disposition in {"cleared", "replaced"}:
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    assert job is not None and job.claim_owner is not None
                    job.claim_owner = None if disposition == "cleared" else "replacement-runtime"
                    if disposition == "replaced":
                        job.attempt += 1
                    session.commit()
                saved_config.write_bytes(sentinel)
            elif disposition == "cancelled":
                response = await client.post(f"/api/jobs/{job_id}/cancel")
                assert response.status_code == 200 and response.json()["status"] == "cancelled"
            before = _state(offer_id, job_id)
            assert not heartbeat.done()
            release.set()
            await _until(finished.is_set)
            await asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True), timeout=PATIENCE_SECONDS
            )
            assert len(requests) == 1
            if disposition in {"cleared", "replaced"}:
                assert saved_config.read_bytes() == sentinel
                assert settings.comfy_executable is None and settings.comfy_directory is None
                assert _state(offer_id, job_id) == before
            else:
                assert (
                    settings.comfy_executable is not None and settings.comfy_directory is not None
                )
                assert provisioner.preflight("comfyui").operation == "reuse_managed"
                assert json.loads(saved_config.read_bytes())["LOCAL_LM_COMFY_EXECUTABLE"] == str(
                    settings.comfy_executable
                )
                current = _state(offer_id, job_id)
                assert current[1]["status"] == (
                    "cancelled" if disposition == "cancelled" else "complete"
                )
                assert current[0]["status"] == (
                    "queued" if disposition == "cancelled" else "completed"
                )
        finally:
            release.set()
            await manager.close()
            await provisioner.close()
