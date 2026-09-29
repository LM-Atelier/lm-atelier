"""Registry preparation commits only while its durable execution still owns the work."""

from __future__ import annotations

import asyncio
import hashlib
import sys
import threading
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_comfy_registry_lifecycle import _closure
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
from local_lm.domain import utcnow
from local_lm.models import ComfyRegistryInstall, Job
from local_lm.scheduler import ResourceScheduler
from local_lm.workflow_package_activation import WorkflowPackageActivation


@pytest.mark.parametrize("replacement", ["token", "attempt", "released", "expiry", "none"])
@pytest.mark.parametrize("renew", [False, True])
async def test_registry_preparation_rechecks_its_claim_before_persisting(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replacement: str,
    renew: bool,
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
    entered, release = threading.Event(), threading.Event()
    original = registry_installs.verify_comfy_registry_wheel_binding

    def verify(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(30), "Verified Registry preparation was not released"
        return result

    activated: list[str] = []

    async def ready(_session: Any, preparation: Any, *_args: Any, **_kwargs: Any) -> Any:
        activated.append(preparation.install_id)
        return WorkflowPackageActivation("review_required", "unreviewed_source", "Review code", ())

    monkeypatch.setattr(api_module, "activate_prepared_workflow_package", ready)
    original_binding: tuple[Any, ...] | None = None
    old_environment: Path | None = None
    old_files: dict[str, str] = {}
    if renew:
        initial_job_id = await accept(client)
        await api_module._REGISTRY_PREPARE_TASKS[initial_job_id]
        assert state(initial_job_id)[:3] == ("complete", None, 1)
        with SessionLocal() as session:
            install = session.scalar(select(ComfyRegistryInstall))
            assert install is not None and install.wheel_environment_path is not None
            install.trusted = True
            install_id = install.id
            original_binding = (
                install.installed_path,
                install.wheel_environment_path,
                install.wheel_closure_sha256,
                install.wheel_environment_sha256,
                install.review_json,
            )
            old_environment = (
                registry_wheel_environment_root(settings.registry_dir)
                / install.wheel_environment_path
            )
            session.commit()
        old_files = {
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
        activated.clear()
        monkeypatch.setattr(lifecycle, "verify_comfy_registry_wheel_binding", verify)
        response = await client.post(f"/api/workflows/packages/installs/{install_id}/renew")
        assert response.status_code == 202, response.text
        job_id = str(response.json()["id"])
    else:
        monkeypatch.setattr(lifecycle, "verify_comfy_registry_wheel_binding", verify)
        job_id = await accept(client)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    request_task: asyncio.Task[Any] | None = None
    retains_claim = replacement in {"none", "review", "second-renewal"}
    try:
        async with asyncio.timeout(30):
            while not entered.is_set():
                assert not task.done(), f"Preparation stopped before verification: {state(job_id)}"
                await asyncio.sleep(0.01)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.status == "running" and job.claim_owner is not None
            if replacement == "token":
                job.claim_owner = "replacement-claim"
            elif replacement == "attempt":
                job.attempt += 1
            elif replacement == "released":
                job.claim_owner = None
            elif replacement == "expiry":
                job.claim_expires_at = utcnow() - timedelta(seconds=30)
            session.commit()
        if replacement == "expiry":
            assert ResourceScheduler()._expire_foreign_claims("primary") == [job_id]
        if replacement in {"review", "second-renewal"}:
            waiting, acquired = asyncio.Event(), asyncio.Event()
            scheduler = app.state.services.scheduler
            original_lease = scheduler.lease

            @asynccontextmanager
            async def observe_lease(device_id: str = "primary") -> Any:
                waiting.set()
                async with original_lease(device_id):
                    acquired.set()
                    yield

            monkeypatch.setattr(scheduler, "lease", observe_lease)
            request_task = asyncio.create_task(
                client.post(
                    f"/api/workflows/packages/installs/{install_id}/review",
                    json={"trusted": False},
                )
                if replacement == "review"
                else client.post(f"/api/workflows/packages/installs/{install_id}/renew")
            )
            await asyncio.wait_for(waiting.wait(), timeout=10)
            assert not acquired.is_set() and not request_task.done()
            assert old_environment is not None and old_environment.is_dir()
            with SessionLocal() as session:
                install = session.get(ComfyRegistryInstall, install_id)
                assert install is not None and install.trusted and not install.active
        release.set()
        await asyncio.wait_for(task, timeout=30)
        if request_task is not None:
            response = await asyncio.wait_for(request_task, timeout=30)
            assert response.status_code == (200 if replacement == "review" else 202), response.text
            if replacement == "second-renewal":
                next_job_id = str(response.json()["id"])
                await asyncio.wait_for(api_module._REGISTRY_PREPARE_TASKS[next_job_id], timeout=30)
                assert state(next_job_id)[:3] == ("complete", None, 1)
        with SessionLocal() as session:
            installed = list(session.scalars(select(ComfyRegistryInstall)))
            assert len(installed) == (1 if renew or replacement == "none" else 0)
            if renew:
                install = installed[0]
                assert install.trusted == (replacement != "review") and not install.active
                current_binding = (
                    install.installed_path,
                    install.wheel_environment_path,
                    install.wheel_closure_sha256,
                    install.wheel_environment_sha256,
                    install.review_json,
                )
                if retains_claim:
                    assert current_binding != original_binding
                else:
                    assert current_binding == original_binding
        assert len(activated) == (1 if not renew and replacement == "none" else 0)
        if retains_claim:
            assert state(job_id)[:3] == ("complete", None, 1)
            if renew:
                assert old_environment is not None and not old_environment.exists()
        elif renew:
            assert old_environment is not None and old_environment.is_dir()
            assert {
                str(path.relative_to(old_environment)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in old_environment.rglob("*")
                if path.is_file()
            } == old_files
        else:
            assert not list((tmp_path / "ComfyUI" / "custom_nodes").glob("*/__init__.py"))
    finally:
        release.set()
        if request_task is not None:
            await asyncio.gather(request_task, return_exceptions=True)
        await api_module.shutdown_registry_preparations()
        await inputs["archive_downloader"].close()
        await wheels.close()


@pytest.mark.parametrize("action", ["review", "second-renewal"])
async def test_registry_renewal_serializes_review_and_repeat_requests(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
) -> None:
    await test_registry_preparation_rechecks_its_claim_before_persisting(
        client, app, settings, monkeypatch, tmp_path, replacement=action, renew=True
    )
