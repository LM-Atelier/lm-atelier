"""An installation retry does not detach the execution that is still cleaning up."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker
from test_generation_queue_pause import sessions as sessions

from local_lm import scheduler as scheduler_module
from local_lm.domain import utcnow
from local_lm.models import Job
from local_lm.queue_lane_policy import change_lane_policy, read_lane_policy
from local_lm.scheduler import ResourceScheduler
from local_lm.schemas import QueueControlCommand


def test_an_expired_queued_install_claim_releases_the_draining_lane(
    sessions: sessionmaker[Session],
) -> None:
    with sessions() as session:
        session.add(
            Job(
                id="retried-install",
                kind="workflow_install",
                status="queued",
                queue_group="primary",
                claim_owner="departed-worker",
                claim_expires_at=utcnow() - timedelta(seconds=1),
            )
        )
        session.commit()
    with sessions() as session:
        policy = change_lane_policy(
            session,
            "install",
            "pause_after_current",
            QueueControlCommand(expected_revision=0, idempotency_key="pause"),
        )
        assert policy.dispatch_state == "draining" and policy.running_jobs == 1
    scheduler = ResourceScheduler(session_factory=sessions)
    assert scheduler._expire_foreign_claims("primary") == ["retried-install"]
    with sessions() as session:
        policy = read_lane_policy(session, "install")
        assert policy.dispatch_state == "paused" and policy.running_jobs == 0
        job = session.get(Job, "retried-install")
        assert job is not None and job.status == "interrupted" and job.claim_owner is None


async def test_a_queued_retry_keeps_renewing_its_existing_claim_during_cleanup(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    before = utcnow() - timedelta(seconds=60)
    with sessions() as session:
        session.add(
            Job(
                id="retried-install",
                kind="workflow_install",
                status="queued",
                queue_group="primary",
                claim_owner="live-owner",
                claim_expires_at=before,
                heartbeat_at=before,
            )
        )
        session.commit()
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0)
    scheduler = ResourceScheduler(session_factory=sessions)
    heartbeat = asyncio.create_task(scheduler._heartbeat("retried-install", "live-owner"))
    try:
        async with asyncio.timeout(5):
            while True:
                with sessions() as session:
                    job = session.get(Job, "retried-install")
                    assert job is not None and job.claim_owner == "live-owner"
                    if job.heartbeat_at is not None and job.heartbeat_at > before.replace(
                        tzinfo=None
                    ):
                        assert job.claim_expires_at is not None
                        assert job.claim_expires_at > job.heartbeat_at
                        break
                await asyncio.sleep(0)
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat


async def test_a_claim_retained_for_install_cleanup_blocks_another_dispatcher(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    with sessions() as session:
        session.add_all(
            [
                Job(
                    id="retried-install",
                    kind="workflow_install",
                    status="queued",
                    queue_group="primary",
                    claim_owner="live-owner",
                    claim_expires_at=utcnow() + timedelta(minutes=5),
                ),
                Job(id="next-generation", kind="image", status="queued", queue_group="primary"),
            ]
        )
        session.commit()
    scheduler = ResourceScheduler(session_factory=sessions)
    observed_wait = asyncio.Event()
    claimed = asyncio.Event()
    release = asyncio.Event()
    expire = scheduler._expire_foreign_claims
    passes = 0

    def scanned(group: str) -> list[str]:
        nonlocal passes
        passes += 1
        if passes >= 2:
            observed_wait.set()
        return expire(group)

    monkeypatch.setattr(scheduler, "_expire_foreign_claims", scanned)

    async def next_job() -> None:
        async with scheduler.job_lease(
            "next-generation", resource="media_compute", group="primary"
        ):
            claimed.set()
            await release.wait()

    execution = asyncio.create_task(next_job())
    observations = [asyncio.create_task(event.wait()) for event in (observed_wait, claimed)]
    try:
        await asyncio.wait_for(
            asyncio.wait(observations, return_when=asyncio.FIRST_COMPLETED), timeout=5
        )
        assert observed_wait.is_set() and not claimed.is_set()
        await scheduler._release_job("retried-install", "live-owner", "primary")
        await asyncio.wait_for(claimed.wait(), timeout=5)
        release.set()
        await asyncio.wait_for(execution, timeout=5)
    finally:
        release.set()
        for task in [execution, *observations]:
            if not task.done():
                task.cancel()
        await asyncio.gather(execution, *observations, return_exceptions=True)
