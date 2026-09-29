"""Installation pause and drain hold across a real worker restart and image render."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_review_live_comfy import settings as settings
from test_workflow_source_live_comfy import (
    test_one_source_approval_installs_and_renders_with_real_comfy as _install_and_render,
)

from local_lm.db import SessionLocal
from local_lm.models import Job


@pytest.mark.asyncio
async def test_installation_pause_and_drain_with_a_real_media_worker(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_url = "/api/queue/lanes/install"

    async def command(action: str, revision: int) -> dict[str, Any]:
        response = await client.post(
            f"{policy_url}/{action}",
            json={"expected_revision": revision, "idempotency_key": f"live-install-{revision}"},
        )
        assert response.status_code == 200, response.text
        return response.json()

    policy = await command("pause-after-current", 0)
    assert policy["dispatch_state"] == "paused" and policy["running_jobs"] == 0
    cleanup_entered, release_cleanup = asyncio.Event(), asyncio.Event()
    processes = app.state.services.processes
    stop = processes.stop
    job_id: str | None = None

    async def hold_completed_cleanup(name: str, *args: Any, **kwargs: Any) -> Any:
        with SessionLocal() as session:
            job = session.get(Job, job_id) if job_id is not None else None
            completed_claim = (
                job is not None and job.status == "complete" and job.claim_owner is not None
            )
        if name == "media" and completed_claim and not cleanup_entered.is_set():
            cleanup_entered.set()
            await release_cleanup.wait()
        return await stop(name, *args, **kwargs)

    monkeypatch.setattr(processes, "stop", hold_completed_cleanup)
    installation = asyncio.create_task(_install_and_render(client, app, monkeypatch))
    try:
        async with asyncio.timeout(180):
            while True:
                assert not installation.done(), "Live installation ended before the paused queue"
                with SessionLocal() as session:
                    job = session.scalar(select(Job).where(Job.kind == "workflow_install"))
                    if job is not None and job.phase == "install paused":
                        assert (
                            job.status == "queued" and job.claim_owner is None and job.attempt == 0
                        )
                        job_id = job.id
                        break
                await asyncio.sleep(0.05)
        policy = await command("resume", policy["revision"])
        assert policy["dispatch_state"] == "open"
        await asyncio.wait_for(cleanup_entered.wait(), timeout=180)
        policy = await command("pause-after-current", policy["revision"])
        assert policy["dispatch_state"] == "draining" and policy["running_jobs"] == 1
        assert not installation.done()
        release_cleanup.set()
        # The shared acceptance helper renders and verifies a neutral image after
        # restoring the worker, while installation dispatch remains paused.
        await asyncio.wait_for(installation, timeout=180)
        response = await client.get(policy_url)
        assert response.status_code == 200
        assert response.json()["dispatch_state"] == "paused"
        assert response.json()["running_jobs"] == 0
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.status == "complete" and job.claim_owner is None
            assert job.attempt == 1
    finally:
        release_cleanup.set()
        if not installation.done():
            installation.cancel()
        await asyncio.gather(installation, return_exceptions=True)
        await app.state.services.downloads.close()
