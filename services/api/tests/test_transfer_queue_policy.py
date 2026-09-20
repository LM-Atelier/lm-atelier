from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session, sessionmaker
from test_generation_queue_pause import sessions as sessions
from test_transfer_queue_pause import seed_downloads

from local_lm.domain import utcnow
from local_lm.models import GenerationQueueReceipt, Job
from local_lm.queue_lane_policy import (
    Action,
    Lane,
    LanePolicy,
    QueueLaneConflict,
    change_lane_policy,
    read_lane_policy,
    recover_queue_lanes,
)
from local_lm.scheduler import ResourceScheduler
from local_lm.schemas import QueueControlCommand


def change(
    sessions: sessionmaker[Session],
    action: Action,
    revision: int,
    key: str,
    lane: Lane = "transfer",
) -> LanePolicy:
    with sessions() as session:
        return change_lane_policy(
            session,
            lane,
            action,
            QueueControlCommand(expected_revision=revision, idempotency_key=key),
        )


def read(sessions: sessionmaker[Session], lane: Lane = "transfer") -> LanePolicy:
    with sessions() as session:
        return read_lane_policy(session, lane)


def test_transfer_pause_preserves_manual_pause_and_separate_generation_state(
    sessions: sessionmaker[Session],
) -> None:
    seed_downloads(sessions)
    scheduler = ResourceScheduler(session_factory=sessions)
    original = change(sessions, "pause_after_current", 0, "same-key")
    assert original.dispatch_state == "paused"
    assert scheduler.peek_next_eligible_job("network") is None
    assert read(sessions, "generation").dispatch_state == "open"
    change(sessions, "pause_after_current", 0, "same-key", "generation")
    change(sessions, "resume", 1, "resume-transfer")
    assert change(sessions, "pause_after_current", 0, "same-key") == original
    assert read(sessions).dispatch_state == "open"
    assert read(sessions, "generation").dispatch_state == "paused"
    assert scheduler.peek_next_eligible_job("network") is not None
    with sessions() as session:
        manual = session.get(Job, "manual")
        assert manual is not None and manual.status == "paused"


async def test_transfer_claims_drain_through_nested_compute_and_terminal_handoff(
    sessions: sessionmaker[Session],
) -> None:
    seed_downloads(sessions)
    scheduler = ResourceScheduler(session_factory=sessions)
    async with scheduler.job_lease(
        "first", resource="network_transfer", group="network", capacity=2
    ):
        async with scheduler.job_lease(
            "second", resource="network_transfer", group="network", capacity=2
        ):
            current = change(sessions, "pause_after_current", 0, "pause-two")
            assert current.dispatch_state == "draining" and current.running_jobs == 2
            async with scheduler.lease("primary"):
                assert read(sessions).running_jobs == 2
                with sessions() as session:
                    job = session.get(Job, "second")
                    assert job is not None
                    job.status = "complete"
                    session.commit()
                assert read(sessions).running_jobs == 2
        current = read(sessions)
        assert current.dispatch_state == "draining" and current.running_jobs == 1
    current = read(sessions)
    assert current.dispatch_state == "paused" and current.running_jobs == 0
    assert current.revision == 2


@pytest.mark.parametrize("initial_revision", [0, 2])
@pytest.mark.parametrize("resume_before_claim", [False, True])
async def test_transfer_pause_wins_at_the_actual_final_claim_update(
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    initial_revision: int,
    resume_before_claim: bool,
) -> None:
    seed_downloads(sessions)
    if initial_revision:
        change(sessions, "pause_after_current", 0, "establish-pause")
        change(sessions, "resume", 1, "establish-open")
    scheduler = ResourceScheduler(session_factory=sessions)
    intercepted = False
    passes = 0

    class OnePass(Exception):
        pass

    def one_pass(_group: str) -> list[str]:
        nonlocal passes
        passes += 1
        if passes > 1:
            raise OnePass
        return []

    def before(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _many: bool,
    ) -> None:
        nonlocal intercepted
        if (
            not intercepted
            and statement.startswith("UPDATE jobs SET")
            and "claim_owner=" in statement
            and "attempt=" in statement
        ):
            intercepted = True
            paused = change(sessions, "pause_after_current", initial_revision, "race-pause")
            assert paused.dispatch_state == "paused"
            if resume_before_claim:
                change(sessions, "resume", initial_revision + 1, "race-resume")

    monkeypatch.setattr(scheduler, "_expire_foreign_claims", one_pass)
    engine = sessions.kw["bind"]
    event.listen(engine, "before_cursor_execute", before)
    try:
        with pytest.raises(OnePass):
            await scheduler._acquire_job(
                "first",
                resource="network_transfer",
                group="network",
                priority=0,
                capacity=1,
                local_lock=asyncio.Semaphore(1),
            )
    finally:
        event.remove(engine, "before_cursor_execute", before)
    assert intercepted
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None and job.status == "queued" and job.claim_owner is None
    assert read(sessions).revision == initial_revision + (2 if resume_before_claim else 1)


@pytest.mark.parametrize("status", ["complete", "failed", "cancelled", "interrupted"])
def test_transfer_recovery_releases_terminal_claims_without_changing_results(
    sessions: sessionmaker[Session],
    status: str,
) -> None:
    seed_downloads(sessions)
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None
        job.status = status
        job.claim_owner = "departed-worker"
        session.commit()
    original = change(sessions, "pause_after_current", 0, "recover-pause")
    assert original.dispatch_state == "draining"
    with sessions() as session:
        recover_queue_lanes(session)
        session.commit()
    assert read(sessions).dispatch_state == "paused"
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None and job.status == status and job.claim_owner is None
    assert change(sessions, "pause_after_current", 0, "recover-pause") == original


def test_foreign_transfer_claim_expiry_finishes_draining(sessions: sessionmaker[Session]) -> None:
    seed_downloads(sessions)
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None
        job.status = "running"
        job.claim_owner = "expired-worker"
        job.claim_expires_at = utcnow() - timedelta(seconds=1)
        session.commit()
    assert change(sessions, "pause_after_current", 0, "expire-pause").dispatch_state == "draining"
    scheduler = ResourceScheduler(session_factory=sessions)
    assert scheduler._expire_foreign_claims("network") == ["first"]
    assert read(sessions).dispatch_state == "paused"
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None and job.status == "interrupted" and job.claim_owner is None


def test_transfer_retry_refuses_a_receipt_for_another_lane(sessions: sessionmaker[Session]) -> None:
    paused = change(sessions, "pause_after_current", 0, "receipt")
    with sessions() as session:
        receipt = session.get(GenerationQueueReceipt, ("transfer", "receipt"))
        assert receipt is not None
        receipt.response_json = {**receipt.response_json, "lane": "generation"}
        session.commit()
    with pytest.raises(QueueLaneConflict):
        change(sessions, "pause_after_current", 0, "receipt")
    assert read(sessions) == paused
