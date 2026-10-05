"""Standalone activations retain their queue policy across shutdown and startup."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import Job
from local_lm.queue_lane_policy import Action, LanePolicy, change_lane_policy, read_lane_policy
from local_lm.scheduler import JobClaim
from local_lm.schemas import QueueControlCommand


def control(action: Action, revision: int) -> LanePolicy:
    with SessionLocal() as session:
        return change_lane_policy(
            session,
            "install",
            action,
            QueueControlCommand(expected_revision=revision, idempotency_key=f"command-{revision}"),
        )


async def test_restart_recovers_activations_without_reopening_a_paused_lane(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = app.state.services.downloads
    entered: list[str] = []

    async def finish(job_id: str, *, claim: JobClaim | None = None) -> None:
        entered.append(job_id)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and claim is not None
            assert (job.claim_owner, job.attempt) == (claim.token, claim.attempt)
            assert job.status == "running"
            job.status = "complete"
            job.completed_at = utcnow()
            session.commit()

    monkeypatch.setattr(manager, "_reactivate_claimed", finish)
    with SessionLocal() as session:
        session.add_all(
            Job(
                id=name,
                kind="activate",
                status=status,
                claim_owner="departed-process" if status == "running" else None,
                queue_group="primary",
                queue_resource="primary_compute",
                attempt=2,
            )
            for name, status in (
                ("recovered", "running"),
                ("queued", "queued"),
                ("manual", "paused"),
            )
        )
        session.commit()
    assert control("pause_after_current", 0).dispatch_state == "draining"
    async with app.router.lifespan_context(app):
        # The retention sweep runs after startup and holds the database writer
        # while a batch works, on a loaded machine for longer than the queue
        # control below may wait. Start from the settled store, as the client
        # fixture does.
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        async with asyncio.timeout(10):
            while True:
                with SessionLocal() as session:
                    jobs = [session.get(Job, name) for name in ("recovered", "queued")]
                    assert all(
                        job is not None and job.status == "queued" and job.claim_owner is None
                        for job in jobs
                    )
                    if all(job is not None and job.phase == "install paused" for job in jobs):
                        break
                await asyncio.sleep(0.01)
        assert entered == []
        with SessionLocal() as session:
            policy = read_lane_policy(session, "install")
            assert policy.dispatch_state == "paused" and policy.running_jobs == 0
            manual = session.get(Job, "manual")
            assert manual is not None and manual.status == "paused" and manual.attempt == 2
        tasks = [manager._tasks[name] for name in ("recovered", "queued")]
        control("resume", policy.revision)
        await manager.scheduler.queue_control_changed("install")
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        assert set(entered) == {"recovered", "queued"}
        with SessionLocal() as session:
            for name in entered:
                job = session.get(Job, name)
                assert job is not None and job.status == "complete" and job.attempt == 3


async def test_shutdown_interrupts_an_activation_and_releases_the_draining_lane(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = app.state.services.downloads
    entered = asyncio.Event()

    async def wait_for_shutdown(_job_id: str, *, claim: JobClaim | None = None) -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_reactivate_claimed", wait_for_shutdown)
    with SessionLocal() as session:
        session.add(Job(id="active-install", kind="activate", status="queued"))
        session.commit()
    try:
        manager.start_activation("active-install")
        await asyncio.wait_for(entered.wait(), timeout=10)
        policy = control("pause_after_current", 0)
        assert policy.dispatch_state == "draining" and policy.running_jobs == 1
        await manager.close()
        with SessionLocal() as session:
            job = session.get(Job, "active-install")
            assert job is not None and job.status == "interrupted" and job.claim_owner is None
            policy = read_lane_policy(session, "install")
            assert policy.dispatch_state == "paused" and policy.running_jobs == 0
    finally:
        await manager.close()
