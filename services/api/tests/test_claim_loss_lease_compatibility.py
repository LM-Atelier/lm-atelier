"""Default lease callers preserve their execution and a replacement claim."""

from __future__ import annotations

import asyncio

import pytest
from run_waits import PATIENCE_SECONDS

from local_lm import scheduler as scheduler_module
from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.domain import JobStatus, utcnow
from local_lm.models import Job
from local_lm.scheduler import ResourceScheduler


def _queued_job(settings: Settings, job_id: str) -> ResourceScheduler:
    settings.prepare()
    configure_database(settings)
    init_db()
    with SessionLocal() as session:
        session.add(
            Job(
                id=job_id,
                status=JobStatus.QUEUED.value,
                queue_group="primary",
                queue_ticket=job_id,
                enqueued_at=utcnow(),
            )
        )
        session.commit()
    return ResourceScheduler(session_factory=SessionLocal)


@pytest.mark.parametrize("replacement", [False, True])
async def test_a_default_lease_caller_keeps_running_after_ownership_moves(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, replacement: bool
) -> None:
    job_id = "job_default_lease"
    scheduler = _queued_job(settings, job_id)
    entered, finish = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.05)

    async def work() -> None:
        async with scheduler.job_lease(job_id, resource="media_compute", group="primary"):
            entered.set()
            await finish.wait()

    execution = asyncio.create_task(work())
    try:
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
        heartbeat = next(
            task for task in asyncio.all_tasks() if task.get_name() == f"job-heartbeat-{job_id}"
        )
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.claim_owner is not None
            job.claim_owner = "garden-default-replacement" if replacement else None
            job.attempt += 1
            expected = (
                job.claim_owner,
                job.attempt,
                job.status,
                job.claim_expires_at,
                job.heartbeat_at,
            )
            session.commit()
        await asyncio.wait_for(asyncio.shield(heartbeat), timeout=PATIENCE_SECONDS)
        assert not execution.done()
        finish.set()
        await asyncio.wait_for(execution, timeout=PATIENCE_SECONDS)
        assert not execution.cancelled() and not scheduler._lock("primary", 1).locked()
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            assert (
                job.claim_owner,
                job.attempt,
                job.status,
                job.claim_expires_at,
                job.heartbeat_at,
            ) == expected
    finally:
        finish.set()
        if not execution.done():
            execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)


@pytest.mark.parametrize("replacement", [False, True])
async def test_normal_lease_teardown_clears_only_its_own_claim(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, replacement: bool
) -> None:
    job_id = "job_normal_teardown"
    scheduler = _queued_job(settings, job_id)
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 3600)
    async with scheduler.job_lease(job_id, resource="media_compute", group="primary") as claim:
        heartbeat = next(
            task for task in asyncio.all_tasks() if task.get_name() == f"job-heartbeat-{job_id}"
        )
        await asyncio.sleep(0)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.claim_owner == claim.token
            if replacement:
                job.claim_owner = "garden-teardown-replacement"
                job.attempt += 1
            expected = (
                job.claim_owner,
                job.attempt,
                job.status,
                job.claim_expires_at,
                job.heartbeat_at,
            )
            session.commit()
    assert heartbeat.cancelled() and not scheduler._lock("primary", 1).locked()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        if replacement:
            assert (
                job.claim_owner,
                job.attempt,
                job.status,
                job.claim_expires_at,
                job.heartbeat_at,
            ) == expected
        else:
            assert job.claim_owner is None and job.claim_expires_at is None
            assert job.heartbeat_at is None
            assert (job.attempt, job.status) == (expected[1], expected[2])
