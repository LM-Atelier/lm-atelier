"""Failed preparation retains its installation claim until disk cleanup finishes."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_comfy_registry_lifecycle import _closure
from test_install_queue_recovery import control
from test_install_queue_registry import accept, state
from test_install_queue_registry import configured_registry as configured_registry
from test_workflow_package_execution_plan import _inputs

from local_lm import api as api_module
from local_lm import comfy_registry_installs as registry_installs
from local_lm import comfy_registry_lifecycle as lifecycle
from local_lm import workflow_package_preparation as composition
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall
from local_lm.queue_lane_policy import read_lane_policy
from local_lm.workflow_package_activation import WorkflowPackageActivation


@pytest.mark.parametrize("boundary", ["node", "environment", "renewal"])
@pytest.mark.parametrize("ending", ["cancel", "shutdown", "failure"])
async def test_registry_cleanup_retains_its_claim_through_cancellation(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    ending: str,
) -> None:
    inputs = _inputs()
    wheels = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    )
    monkeypatch.setattr(settings, "comfy_executable", Path(sys.executable))
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

    async def ready(*_args: Any, **_kwargs: Any) -> WorkflowPackageActivation:
        return WorkflowPackageActivation("review_required", "unreviewed_source", "Review code", ())

    monkeypatch.setattr(api_module, "activate_prepared_workflow_package", ready)
    old_environment: Path | None = None
    old_hashes: dict[str, str] = {}
    prior_binding: tuple[Any, ...] | None = None
    if boundary == "renewal":
        initial_job_id = await accept(client)
        await api_module._REGISTRY_PREPARE_TASKS[initial_job_id]
        assert state(initial_job_id)[:3] == ("complete", None, 1)
        with SessionLocal() as session:
            install = session.scalar(select(ComfyRegistryInstall))
            assert install is not None and install.wheel_environment_path is not None
            install.trusted = True
            install_id = install.id
            prior_binding = (
                install.wheel_environment_path,
                install.wheel_environment_sha256,
                install.wheel_closure_sha256,
                install.review_json,
            )
            old_environment = (
                registry_wheel_environment_root(settings.registry_dir)
                / install.wheel_environment_path
            )
            session.commit()
        old_hashes = {
            str(path.relative_to(old_environment)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in old_environment.rglob("*")
            if path.is_file()
        }

        async def closure(*_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                closure=_closure(
                    inputs["selected"],
                    runtime_distributions=(
                        ComfyRegistryRuntimeDistribution("torch", "2.13.0+cu130"),
                    ),
                )
            )

        monkeypatch.setattr(composition, "drive_comfy_registry_wheel_closure", closure)
    verify = registry_installs.verify_comfy_registry_wheel_binding
    verifications = 0

    def refuse_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal verifications
        result = verify(*args, **kwargs)
        verifications += 1
        if verifications == 1:
            raise lifecycle.ComfyRegistryLifecycleError(
                "binding_refused", "Neutral binding refusal"
            )
        return result

    monkeypatch.setattr(lifecycle, "verify_comfy_registry_wheel_binding", refuse_once)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    target_root = (
        settings.custom_node_dir
        if boundary == "node"
        else registry_wheel_environment_root(settings.registry_dir)
    )
    remove_tree = shutil.rmtree

    def remove(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path).parent == target_root and not entered.is_set():
            entered.set()
            try:
                assert release.wait(30), "Registry cleanup was not released"
                return remove_tree(path, *args, **kwargs)
            finally:
                finished.set()
        return remove_tree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", remove)
    release_job = app.state.services.scheduler._release_job
    early_releases: list[str] = []

    async def observed_release(identifier: str, *args: Any, **kwargs: Any) -> None:
        if identifier == job_id and not finished.is_set():
            early_releases.append(identifier)
        await release_job(identifier, *args, **kwargs)

    monkeypatch.setattr(app.state.services.scheduler, "_release_job", observed_release)
    if boundary == "renewal":
        response = await client.post(f"/api/workflows/packages/installs/{install_id}/renew")
        assert response.status_code == 202, response.text
        job_id = str(response.json()["id"])
    else:
        job_id = await accept(client)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    cancellation: asyncio.Task[Any] | None = None
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                assert not task.done(), "Preparation stopped before disk cleanup"
                await asyncio.sleep(0.01)
        policy = control("pause_after_current", 0)
        assert policy.dispatch_state == "draining" and policy.running_jobs == 1
        if ending == "cancel":
            cancellation = asyncio.create_task(client.post(f"/api/jobs/{job_id}/cancel"))
        elif ending == "shutdown":
            cancellation = asyncio.create_task(api_module.shutdown_registry_preparations())
        if cancellation is not None:
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.wait({task}, timeout=0.25)
            assert not task.done(), "Preparation exited while its deletion thread was live"
        release.set()
        await asyncio.gather(
            task, *([cancellation] if cancellation else []), return_exceptions=True
        )
        async with asyncio.timeout(10):
            while not finished.is_set():
                await asyncio.sleep(0.01)
        assert early_releases == [], "The durable claim released while file deletion was live"
        with SessionLocal() as session:
            policy = read_lane_policy(session, "install")
            assert policy.dispatch_state == "paused" and policy.running_jobs == 0
        environments = list(
            registry_wheel_environment_root(settings.registry_dir).glob("registry-wheels-*")
        )
        if boundary == "renewal":
            assert environments == [old_environment]
            assert old_environment is not None
            assert {
                str(path.relative_to(old_environment)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in old_environment.rglob("*")
                if path.is_file()
            } == old_hashes
            with SessionLocal() as session:
                install = session.get(ComfyRegistryInstall, install_id)
                assert install is not None and install.trusted and not install.active
                assert (
                    install.wheel_environment_path,
                    install.wheel_environment_sha256,
                    install.wheel_closure_sha256,
                    install.review_json,
                ) == prior_binding
            response = await client.post(f"/api/workflows/packages/installs/{install_id}/renew")
            assert response.status_code == 202, response.text
            next_job_id = str(response.json()["id"])
        else:
            assert not list(settings.custom_node_dir.glob("*/__init__.py"))
            assert environments == []
            next_job_id = await accept(client)
        next_task = api_module._REGISTRY_PREPARE_TASKS[next_job_id]
        control("resume", policy.revision)
        await app.state.services.scheduler.queue_control_changed("install")
        await asyncio.wait_for(next_task, timeout=30)
        assert state(next_job_id)[:3] == ("complete", None, 1)
    finally:
        release.set()
        await asyncio.gather(
            task, *([cancellation] if cancellation else []), return_exceptions=True
        )
        async with asyncio.timeout(30):
            while entered.is_set() and not finished.is_set():
                await asyncio.sleep(0.01)
        await api_module.shutdown_registry_preparations()
        await inputs["archive_downloader"].close()
        await wheels.close()
