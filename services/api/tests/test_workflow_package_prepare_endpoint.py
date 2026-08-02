"""The prepare endpoint: a durable job that refuses honestly."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.config import Settings

pytestmark = pytest.mark.asyncio


async def test_an_unconfigured_runtime_refuses_before_queueing(client: AsyncClient) -> None:
    """A job that must fail on its first step is worse than a typed 422."""

    response = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "example-pack", "version": "1.2.3"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "managed_runtime_unavailable"
    jobs = (await client.get("/api/jobs")).json()
    assert all(job["kind"] != "registry_prepare" for job in jobs)


async def test_a_configured_runtime_queues_and_fails_closed_on_the_probe(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Until a verified target probe exists, preparation refuses with its code.

    The job still runs end to end as a job: queued, leased, failed with the
    stable code recorded - never a silent death.
    """

    executable = tmp_path / "python.exe"
    executable.write_bytes(b"")
    monkeypatch.setattr(settings, "comfy_executable", executable)
    monkeypatch.setattr(settings, "comfy_directory", tmp_path / "ComfyUI")

    from local_lm import api as api_module
    from local_lm.workflow_package_preparation import WorkflowPackagePreparationError

    async def refusing_preparation(*args: object, **kwargs: object) -> object:
        raise WorkflowPackagePreparationError(
            "interpreter_probe_unavailable", "no verified probe yet"
        )

    monkeypatch.setattr(api_module, "prepare_workflow_package", refusing_preparation)

    response = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "example-pack", "version": "1.2.3"},
    )

    assert response.status_code == 202
    job_id = response.json()["id"]
    assert response.json()["kind"] == "registry_prepare"

    import asyncio

    job = None
    for _ in range(200):
        jobs = (await client.get("/api/jobs")).json()
        job = next(item for item in jobs if item["id"] == job_id)
        if job["status"] in {"failed", "complete", "cancelled"}:
            break
        await asyncio.sleep(0.05)
    assert job is not None
    assert job["status"] == "failed"
    assert job["payload_json"]["error_code"] == "interpreter_probe_unavailable"
    assert job["error"]
