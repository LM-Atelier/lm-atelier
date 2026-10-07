"""Source inspection retains its compute slot until its actual thread has finished."""

from __future__ import annotations

import asyncio
import threading
from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_scheduler_claim_hold import _until
from test_workflow_completion_jobs import _accept
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm import runtime_provisioning_recovery, workflow_source_completion
from local_lm import scheduler as scheduler_module
from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.models import Job, WorkflowInstallOffer
from local_lm.scheduler import JobClaim

pytestmark = pytest.mark.asyncio


def _state(offer_id: str, job_id: str) -> tuple[dict[str, object], ...]:
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        job = session.get(Job, job_id)
        assert offer is not None and job is not None
        return deepcopy(
            tuple(
                {column.name: getattr(row, column.name) for column in row.__table__.columns}
                for row in (offer, job)
            )
        )


@pytest.mark.parametrize("boundary", ["source", "preflight", "runtime_plan", "provision"])
@pytest.mark.parametrize("disposition", ["cleared", "replaced", "cancelled", "owned"])
async def test_source_inspection_keeps_its_slot_until_the_displaced_thread_exits(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    disposition: str,
) -> None:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.02)
    manager: DownloadManager = app.state.services.downloads
    provisioner = app.state.services.processes.runtimes
    assert provisioner is not None
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    provisioned = False

    def hold_thread() -> None:
        entered.set()
        try:
            assert release.wait(timeout=30), "The source inspection thread was not released."
        finally:
            finished.set()

    read_source = workflow_source_completion._read_source
    require_plan = workflow_source_completion._require_runtime_plan
    preflight = provisioner.preflight
    provision = provisioner.provision
    recover = runtime_provisioning_recovery.recover_approved_runtime

    def read(identifier: str, claim: JobClaim) -> Any:
        result = read_source(identifier, claim)
        assert result is not None
        if boundary == "source":
            hold_thread()
        return result

    def inspect(*args: Any, **kwargs: Any) -> Any:
        result = preflight(*args, **kwargs)
        if boundary == "preflight" and provisioned and not entered.is_set():
            hold_thread()
        return result

    async def prepare(*args: Any, **kwargs: Any) -> Any:
        nonlocal provisioned
        result = await provision(*args, **kwargs)
        provisioned = True
        return result

    def check_plan(*args: Any, **kwargs: Any) -> None:
        require_plan(*args, **kwargs)
        if boundary == "runtime_plan" and not entered.is_set():
            hold_thread()

    def recover_runtime(*args: Any, **kwargs: Any) -> Any:
        result = recover(*args, **kwargs)
        if boundary == "provision":
            hold_thread()
        return result

    monkeypatch.setattr(workflow_source_completion, "_read_source", read)
    monkeypatch.setattr(workflow_source_completion, "_require_runtime_plan", check_plan)
    monkeypatch.setattr(provisioner, "preflight", inspect)
    monkeypatch.setattr(provisioner, "provision", prepare)
    monkeypatch.setattr(runtime_provisioning_recovery, "recover_approved_runtime", recover_runtime)
    with monkeypatch.context() as deferred:
        deferred.setattr(manager, "start_workflow_installation", lambda _id: None)
        _, offer_id, job_id = await _accept(client)
    manager.start_workflow_installation(offer_id)
    task = manager._offer_tasks[offer_id]
    acquired, waiter_release = asyncio.Event(), asyncio.Event()

    async def next_installation() -> None:
        async with manager.scheduler.lease("primary"):
            acquired.set()
            await waiter_release.wait()

    waiter: asyncio.Task[None] | None = None
    try:
        await _until(entered.is_set, timeout=PATIENCE_SECONDS)
        assert not finished.is_set() and not task.done()
        heartbeat = next(
            item for item in asyncio.all_tasks() if item.get_name() == f"job-heartbeat-{job_id}"
        )
        waiter = asyncio.create_task(next_installation())
        if disposition in {"cleared", "replaced"}:
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None and job.claim_owner is not None
                job.claim_owner = None if disposition == "cleared" else "replacement-source"
                if disposition == "replaced":
                    job.attempt += 1
                session.commit()
            before = _state(offer_id, job_id)
            await _until(heartbeat.done, timeout=PATIENCE_SECONDS)
        elif disposition == "cancelled":
            response = await client.post(f"/api/jobs/{job_id}/cancel")
            assert response.status_code == 200 and response.json()["status"] == "cancelled"
            before = _state(offer_id, job_id)
        else:
            before = _state(offer_id, job_id)
        await asyncio.wait({task, waiter}, timeout=0.15)
        assert not finished.is_set()
        if boundary == "provision":
            assert provisioner._locks["comfyui"].locked(), (
                "Runtime provisioning released its lock before its thread ended."
            )
        assert not acquired.is_set(), "The next installation acquired the slot before I/O ended."
        assert not task.done(), "The installation exited while its source thread was still running."
        release.set()
        await _until(finished.is_set, timeout=PATIENCE_SECONDS)
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True),
            timeout=PATIENCE_SECONDS,
        )
        await _until(acquired.is_set, timeout=PATIENCE_SECONDS)
        after = _state(offer_id, job_id)
        result = after[1]["result_json"]
        assert isinstance(result, dict)
        if disposition in {"cleared", "replaced"}:
            assert after == before
        elif disposition == "cancelled":
            assert after[0]["status"] == "queued" and after[1]["status"] == "cancelled"
            assert after[1]["attempt"] == before[1]["attempt"]
            assert result.get("activation_id") is None
        else:
            assert after[0]["status"] == "completed" and after[1]["status"] == "complete"
            assert result.get("activation_id") is not None
    finally:
        release.set()
        waiter_release.set()
        if waiter is not None:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        await _until(finished.is_set, timeout=PATIENCE_SECONDS)
        await manager.close()
