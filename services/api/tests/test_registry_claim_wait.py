"""A silent preparation wait ends when the registry job loses its claim."""

from __future__ import annotations

import asyncio

import pytest
from httpx2 import AsyncClient
from sqlalchemy import delete
from test_install_queue_registry import accept, state
from test_install_queue_registry import configured_registry as configured_registry
from test_scheduler_claim_hold import _until

from local_lm import api as api_module
from local_lm import scheduler as scheduler_module
from local_lm.comfy_registry_lifecycle import ComfyRegistryPreparation
from local_lm.db import SessionLocal
from local_lm.models import Job


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "removed", "owned"])
async def test_a_registry_preparation_wait_retains_only_its_own_claim(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, disposition: str
) -> None:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.02)
    entered, release = asyncio.Event(), asyncio.Event()
    ended = asyncio.Event()

    async def prepare(*args: object, **kwargs: object) -> ComfyRegistryPreparation:
        entered.set()
        try:
            await release.wait()
            return ComfyRegistryPreparation(
                "neutral-install",
                "neutral-nodes",
                "neutral-environment",
                "a" * 64,
                "b" * 64,
                "c" * 64,
                "d" * 64,
                False,
            )
        finally:
            ended.set()

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    job_id = await accept(client)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        heartbeat = next(
            task for task in asyncio.all_tasks() if task.get_name() == f"job-heartbeat-{job_id}"
        )
        if disposition == "owned":
            release.set()
            await asyncio.wait_for(task, timeout=3)
            assert state(job_id)[:3] == ("complete", None, 1)
        else:
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None and job.claim_owner is not None
                if disposition == "removed":
                    session.execute(delete(Job).where(Job.id == job_id))
                else:
                    job.claim_owner = "replacement-registry" if disposition == "replaced" else None
                    if disposition == "replaced":
                        job.attempt = 2
                session.commit()
            before = state(job_id) if disposition != "removed" else None
            await _until(heartbeat.done)
            await _until(task.done, timeout=1)
            await asyncio.gather(task, return_exceptions=True)
            assert ended.is_set()
            if before is not None:
                assert state(job_id) == before
            else:
                with SessionLocal() as session:
                    assert session.get(Job, job_id) is None
    finally:
        release.set()
        await api_module.shutdown_registry_preparations()
