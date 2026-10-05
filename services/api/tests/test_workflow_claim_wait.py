"""Source completion stops a silent backend wait after its claim is displaced."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import delete, func, select
from test_scheduler_claim_hold import _until
from test_workflow_completion_jobs import _accept
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm import scheduler as scheduler_module
from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.models import Job, WorkflowActivation, WorkflowInstallOffer


def _state(offer_id: str, job_id: str) -> tuple[object, ...]:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        return (
            (
                job.status,
                job.claim_owner,
                job.attempt,
                job.completed_at,
                job.error,
                dict(job.result_json),
            )
            if job is not None
            else None,
            offer.status,
            offer.workflow_revision_id,
            offer.completion_error_code,
            session.scalar(select(func.count()).select_from(WorkflowActivation)),
        )


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "removed", "owned"])
async def test_source_completion_keeps_only_its_owned_validation_wait(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, disposition: str
) -> None:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.02)
    entered, release, ended = asyncio.Event(), asyncio.Event(), asyncio.Event()
    manager: DownloadManager = app.state.services.downloads

    async def validate(graph: object) -> list[str]:
        entered.set()
        try:
            await release.wait()
            return []
        finally:
            ended.set()

    monkeypatch.setattr(app.state.services.engines.media, "validate_workflow", validate)
    with monkeypatch.context() as deferred:
        deferred.setattr(manager, "start_workflow_installation", lambda _id: None)
        _, offer_id, job_id = await _accept(client)
    manager.start_workflow_installation(offer_id)
    task = manager._offer_tasks[offer_id]
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        heartbeat = next(
            task for task in asyncio.all_tasks() if task.get_name() == f"job-heartbeat-{job_id}"
        )
        if disposition == "owned":
            release.set()
            await asyncio.wait_for(task, timeout=5)
            current = _state(offer_id, job_id)
            assert isinstance(current[0], tuple) and current[0][0] == "complete"
            assert current[1] == "completed" and current[4] == 1
        else:
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None and job.claim_owner is not None
                if disposition == "removed":
                    offer = session.get(WorkflowInstallOffer, offer_id)
                    assert offer is not None
                    offer.completion_job_id = None
                    session.flush()
                    session.execute(delete(Job).where(Job.id == job_id))
                else:
                    job.claim_owner = "replacement-workflow" if disposition == "replaced" else None
                    if disposition == "replaced":
                        job.attempt = 2
                session.commit()
            before = _state(offer_id, job_id)
            await _until(heartbeat.done)
            await _until(task.done, timeout=1)
            await asyncio.gather(task, return_exceptions=True)
            assert ended.is_set()
            assert _state(offer_id, job_id) == before
    finally:
        release.set()
        await manager.close()
