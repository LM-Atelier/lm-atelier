"""Package activation preserves replacement state after execution ownership changes."""

from __future__ import annotations

import asyncio
import sys
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_install_queue_registry import accept, state
from test_install_queue_registry import configured_registry as configured_registry
from test_workflow_package_execution_plan import _inputs

from local_lm import api as api_module
from local_lm.comfy_registry_target_verification import ComfyRegistryVerificationTarget
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ComfyRegistryInstall, Job
from local_lm.scheduler import ResourceScheduler


@pytest.mark.parametrize("boundary", ["trust", "activation", "completion", "failure"])
@pytest.mark.parametrize("replacement", ["token", "attempt", "released", "expiry", "none"])
@pytest.mark.parametrize("assertion", ["state", "runtime"])
async def test_registry_activation_preserves_state_after_claim_replacement(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    replacement: str,
    assertion: str,
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

    processes = app.state.services.processes
    initial_statuses = processes.statuses()
    running = False
    starts = 0
    stops = 0

    def statuses() -> list[Any]:
        return [
            worker.model_copy(
                update={"running": running, "state": "ready" if running else "stopped"}
            )
            if worker.name == "media"
            else worker
            for worker in initial_statuses
        ]

    entered, release = asyncio.Event(), asyncio.Event()

    async def start(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal running, starts
        starts += 1
        if boundary == "failure" and starts == 1:
            entered.set()
            await release.wait()
            raise RuntimeError("Synthetic startup failure")
        running = True
        return next(worker for worker in statuses() if worker.name == "media")

    async def stop(_name: str) -> Any:
        nonlocal running, stops
        stops += 1
        running = False
        return next(worker for worker in statuses() if worker.name == "media")

    monkeypatch.setattr(processes, "statuses", statuses)
    monkeypatch.setattr(processes, "start_media", start)
    monkeypatch.setattr(processes, "stop", stop)
    original_verify = ComfyRegistryVerificationTarget.verify
    verifications = 0

    async def verify(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal verifications
        result = await original_verify(self, *args, **kwargs)
        verifications += 1
        if verifications == {"trust": 1, "activation": 2, "completion": 3}.get(boundary):
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(ComfyRegistryVerificationTarget, "verify", verify)
    job_id = await accept(client)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    try:
        async with asyncio.timeout(30):
            while not entered.is_set():
                assert not task.done(), f"Activation ended before its boundary: {state(job_id)}"
                await asyncio.sleep(0.01)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            install = session.scalar(select(ComfyRegistryInstall))
            assert job is not None and job.status == "running" and job.claim_owner is not None
            assert install is not None
            if replacement == "token":
                job.claim_owner = "replacement-claim"
            elif replacement == "attempt":
                job.attempt += 1
            elif replacement == "released":
                job.claim_owner = None
            elif replacement == "expiry":
                job.claim_expires_at = utcnow() - timedelta(seconds=30)
            expected = (install.trusted, install.active, deepcopy(install.review_json))
            session.commit()
        if replacement == "expiry":
            assert ResourceScheduler()._expire_foreign_claims("primary") == [job_id]
        expected_job = state(job_id)[:3]
        if replacement == "attempt":
            # The attempt fence rejects writes, but the lease still releases
            # its original token when no different claim has replaced it.
            expected_job = (expected_job[0], None, expected_job[2])
        expected_calls = (starts, stops)
        release.set()
        await asyncio.wait_for(task, timeout=30)
        with SessionLocal() as session:
            install = session.scalar(select(ComfyRegistryInstall))
            assert install is not None
            if replacement != "none":
                if assertion == "state":
                    assert (install.trusted, install.active, install.review_json) == expected
                else:
                    assert (starts, stops) == expected_calls
                assert state(job_id)[:3] == expected_job
            elif boundary == "failure":
                assert not install.active
                assert state(job_id)[:3] == ("failed", None, 1)
            else:
                assert install.trusted and install.active
                assert state(job_id)[:3] == ("complete", None, 1)
    finally:
        release.set()
        await api_module.shutdown_registry_preparations()
        await inputs["archive_downloader"].close()
        await wheels.close()
