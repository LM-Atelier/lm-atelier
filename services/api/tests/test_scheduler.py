from __future__ import annotations

import asyncio
import time as _real_time
from datetime import timedelta
from typing import Any

from sqlalchemy import select

from local_lm import scheduler as scheduler_module
from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.domain import JobStatus, utcnow
from local_lm.models import Chat, Job, WorkPlan, WorkStep, WorkStepDependency
from local_lm.scheduler import _ELIGIBILITY_SHARE_SECONDS, ResourceScheduler


class _FrozenClock:
    """A stand-in for the `time` module whose monotonic clock the test drives.

    Patching `monotonic` ON the real `time` module would freeze it for the whole
    process, including the asyncio event loop, whose timers are scheduled
    against it. Any test that then runs the loop waits forever instead of
    failing. Replacing the scheduler module's own reference to `time` keeps the
    freeze inside the code under test, and every other attribute still comes
    from the real module.
    """

    def __init__(self, now: dict[str, float]) -> None:
        self._now = now

    def monotonic(self) -> float:
        return self._now["t"]

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_time, name)


def test_queue_order_uses_priority_aging_and_stable_tickets(settings: Settings) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    with SessionLocal() as session:
        session.add_all(
            [
                Job(
                    id="job_aged",
                    status=JobStatus.QUEUED.value,
                    queue_group="primary",
                    queue_priority=0,
                    queue_ticket="ticket-b",
                    enqueued_at=now - timedelta(seconds=61),
                ),
                Job(
                    id="job_priority",
                    status=JobStatus.QUEUED.value,
                    queue_group="primary",
                    queue_priority=1,
                    queue_ticket="ticket-a",
                    enqueued_at=now,
                ),
                Job(
                    id="job_later",
                    status=JobStatus.QUEUED.value,
                    queue_group="primary",
                    queue_priority=0,
                    queue_ticket="ticket-c",
                    enqueued_at=now,
                ),
            ]
        )
        session.flush()

        ordered = ResourceScheduler._eligible_jobs(session, "primary", now)

    assert [job.id for job in ordered] == [
        "job_aged",
        "job_priority",
        "job_later",
    ]


def test_image_edit_checks_never_age_ahead_of_foreground_work(settings: Settings) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    with SessionLocal() as session:
        session.add_all(
            [
                Job(
                    id="job_background_check",
                    kind="edit_verify",
                    status=JobStatus.QUEUED.value,
                    queue_group="primary",
                    queue_priority=10_000,
                    enqueued_at=now - timedelta(days=30),
                ),
                Job(
                    id="job_foreground",
                    kind="image",
                    status=JobStatus.QUEUED.value,
                    queue_group="primary",
                    queue_priority=-10_000,
                    enqueued_at=now,
                ),
            ]
        )
        session.flush()

        ordered = ResourceScheduler._eligible_jobs(session, "primary", now)

    assert [job.id for job in ordered] == ["job_foreground", "job_background_check"]


def test_peek_next_eligible_job_does_not_claim_or_change_it(settings: Settings) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    with SessionLocal() as session:
        session.add_all(
            [
                Job(
                    id="job_second",
                    status=JobStatus.QUEUED.value,
                    queue_group="primary",
                    queue_ticket="ticket-b",
                    enqueued_at=now,
                ),
                Job(
                    id="job_first",
                    status=JobStatus.QUEUED.value,
                    queue_group="primary",
                    queue_ticket="ticket-a",
                    enqueued_at=now,
                ),
            ]
        )
        session.commit()

    scheduler = ResourceScheduler(session_factory=SessionLocal)

    assert scheduler.peek_next_eligible_job("primary") == ("job_first", None)
    with SessionLocal() as session:
        job = session.get(Job, "job_first")
        assert job is not None
        assert job.status == JobStatus.QUEUED.value
        assert job.claim_owner is None


def test_dependent_step_is_not_dispatchable_until_dependency_completes(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    with SessionLocal() as session:
        chat = Chat(id="chat_scheduler", title="Scheduler")
        plan = WorkPlan(
            id="plan_scheduler",
            chat_id=chat.id,
            transcript_sequence=1,
        )
        first = WorkStep(
            id="step_first",
            plan=plan,
            ordinal=1,
            operation="text",
        )
        second = WorkStep(
            id="step_second",
            plan=plan,
            ordinal=2,
            operation="text",
        )
        first_job = Job(
            id="job_first",
            status=JobStatus.RUNNING.value,
            work_step_id=first.id,
            queue_group="primary",
            queue_ticket="ticket-first",
            enqueued_at=now,
        )
        second_job = Job(
            id="job_second",
            status=JobStatus.QUEUED.value,
            work_step_id=second.id,
            queue_group="primary",
            queue_ticket="ticket-second",
            enqueued_at=now,
        )
        session.add_all([chat, plan])
        session.flush()
        session.add_all([first, second])
        session.flush()
        session.add_all(
            [
                WorkStepDependency(step_id=second.id, depends_on_step_id=first.id),
                first_job,
                second_job,
            ]
        )
        session.flush()

        assert ResourceScheduler._eligible_jobs(session, "primary", now) == []
        assert ResourceScheduler._blocking_steps(session, second.id) == [first.id]

        first_job.status = JobStatus.COMPLETE.value
        first.status = JobStatus.COMPLETE.value
        session.flush()

        eligible = ResourceScheduler._eligible_jobs(session, "primary", now)
        assert [job.id for job in eligible] == [second_job.id]

        first.status = JobStatus.CANCELLED.value
        session.flush()
        assert ResourceScheduler._failed_dependencies(session, second.id) == [first.id]


def test_expired_foreign_claim_is_interrupted_without_replay(settings: Settings) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    scheduler = ResourceScheduler()
    now = utcnow()
    with SessionLocal() as session:
        session.add_all(
            [
                Job(
                    id="job_abandoned",
                    status=JobStatus.RUNNING.value,
                    queue_group="primary",
                    claim_owner="other-dispatcher_token",
                    claim_expires_at=now - timedelta(seconds=1),
                ),
                Job(
                    id="job_local",
                    status=JobStatus.RUNNING.value,
                    queue_group="primary",
                    claim_owner=f"{scheduler._owner}_token",
                    claim_expires_at=now - timedelta(seconds=1),
                ),
            ]
        )
        session.commit()

    assert scheduler._expire_foreign_claims("primary") == ["job_abandoned"]

    with SessionLocal() as session:
        abandoned = session.get(Job, "job_abandoned")
        local = session.get(Job, "job_local")
        assert abandoned and local
        assert abandoned.status == JobStatus.INTERRUPTED.value
        assert abandoned.claim_owner is None
        assert abandoned.error == "The dispatcher lease expired before this job completed."
        assert local.status == JobStatus.RUNNING.value


def test_expiring_a_dispatched_turn_replaces_its_progress_part_in_one_order(
    settings: Settings,
) -> None:
    """A dispatched turn's message carries a progress part at position 0.

    Interruption removes that part and records an error part in the same
    position; the removal must reach the database before the insert, or the
    (message, position) uniqueness refuses the interruption itself and the
    abandoned claim is never expired.
    """

    from local_lm.domain import PartType
    from local_lm.models import Chat, Message, MessagePart, Run

    settings.prepare()
    configure_database(settings)
    init_db()
    scheduler = ResourceScheduler()
    now = utcnow()
    with SessionLocal() as session:
        chat = Chat()
        session.add(chat)
        session.flush()
        user = Message(chat_id=chat.id, role="user")
        assistant = Message(chat_id=chat.id, role="assistant", status="pending")
        session.add_all([user, assistant])
        session.flush()
        assistant.parts.append(
            MessagePart(
                position=0,
                type=PartType.PROGRESS.value,
                text="Rendering",
                metadata_json={"progress": 0.4, "phase": "running"},
            )
        )
        run = Run(
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            status="running",
            operation="text_to_image",
        )
        session.add(run)
        session.flush()
        session.add(
            Job(
                id="job_dispatched",
                kind="image",
                status=JobStatus.RUNNING.value,
                queue_group="primary",
                run_id=run.id,
                claim_owner="other-dispatcher_token",
                claim_expires_at=now - timedelta(seconds=1),
            )
        )
        session.commit()
        assistant_id = assistant.id

    assert scheduler._expire_foreign_claims("primary") == ["job_dispatched"]

    with SessionLocal() as session:
        job = session.get(Job, "job_dispatched")
        message = session.get(Message, assistant_id)
        assert job is not None and message is not None
        assert job.status == JobStatus.INTERRUPTED.value
        types = [part.type for part in message.parts]
        assert PartType.PROGRESS.value not in types, "the progress part survived interruption"
        assert PartType.ERROR.value in types, "the interruption recorded no error part"
        assert len({part.position for part in message.parts}) == len(message.parts)


async def test_waiting_for_local_capacity_does_not_keep_a_database_session_open(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_waiting_for_capacity",
                status=JobStatus.QUEUED.value,
                queue_group="primary",
                queue_ticket="ticket-waiting",
                enqueued_at=now,
            )
        )
        session.commit()

    opened = 0
    closed = 0

    class TrackedSession:
        def __init__(self) -> None:
            nonlocal opened
            opened += 1
            self.session = SessionLocal()

        def __enter__(self) -> Any:
            return self.session

        def __exit__(self, *args: object) -> None:
            nonlocal closed
            self.session.close()
            closed += 1

    scheduler = ResourceScheduler(session_factory=TrackedSession)  # type: ignore[arg-type]
    unavailable_slot = asyncio.Semaphore(0)
    task = asyncio.create_task(
        scheduler._acquire_job(
            "job_waiting_for_capacity",
            resource="interactive_compute",
            group="primary",
            priority=0,
            capacity=1,
            local_lock=unavailable_slot,
        )
    )
    try:
        for _ in range(20):
            await asyncio.sleep(0)
            if opened:
                break
        assert not task.done()
        assert opened >= 1
        assert opened == closed
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_expired_runless_claim_interrupts_its_work_step(settings: Settings) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    scheduler = ResourceScheduler()
    now = utcnow()
    with SessionLocal() as session:
        chat = Chat(id="chat_runless", title="Runless")
        plan = WorkPlan(
            id="plan_runless",
            chat_id=chat.id,
            transcript_sequence=1,
            summary_json={},
        )
        step = WorkStep(
            id="step_runless",
            plan=plan,
            ordinal=1,
            operation="edit_verify",
            status=JobStatus.RUNNING.value,
        )
        job = Job(
            id="job_runless",
            kind="edit_verify",
            status=JobStatus.RUNNING.value,
            work_plan_id=plan.id,
            work_step_id=step.id,
            queue_group="primary",
            claim_owner="other-dispatcher_token",
            claim_expires_at=now - timedelta(seconds=1),
        )
        session.add_all([chat, plan])
        session.flush()
        session.add_all([step, job])
        session.commit()
        job_id = job.id
        step_id = step.id
        plan_id = plan.id

    assert scheduler._expire_foreign_claims("primary") == [job_id]
    with SessionLocal() as session:
        stored_job = session.get(Job, job_id)
        stored_step = session.get(WorkStep, step_id)
        stored_plan = session.get(WorkPlan, plan_id)
        assert stored_job and stored_step and stored_plan
        assert stored_job.status == JobStatus.INTERRUPTED.value
        assert stored_step.status == JobStatus.INTERRUPTED.value
        assert stored_plan.summary_json["status_counts"] == {"interrupted": 1}


def _seed_queue(session: Any, depth: int, now: Any, prefix: str) -> None:
    """Queue `depth` jobs that each own a real WorkStep with a real dependency.

    A job with no work_step_id makes `_blocking_steps` return immediately, so a
    counter over it measures helper CALLS and no database work at all. Every job
    here owns a step, and every step depends on one shared upstream step, so each
    inspection issues the dependency query the production path issues.
    """

    for job in session.scalars(select(Job)).all():
        session.delete(job)
    session.flush()
    chat = Chat(id=f"chat_{prefix}", title="Scheduler")
    plan = WorkPlan(id=f"plan_{prefix}", chat_id=chat.id, transcript_sequence=1)
    session.add_all([chat, plan])
    session.flush()
    # COMPLETE, so the dependency is queried on every inspection but satisfied.
    # An incomplete upstream blocks every job and _eligible_jobs returns nothing,
    # which measures a different thing entirely.
    upstream = WorkStep(
        id=f"step_{prefix}_up",
        plan=plan,
        ordinal=0,
        operation="text",
        status=JobStatus.COMPLETE.value,
    )
    session.add(upstream)
    session.flush()
    steps = [
        WorkStep(id=f"step_{prefix}_{i:03d}", plan=plan, ordinal=i + 1, operation="text")
        for i in range(depth)
    ]
    session.add_all(steps)
    session.flush()
    session.add_all(
        [WorkStepDependency(step_id=s.id, depends_on_step_id=upstream.id) for s in steps]
    )
    session.add_all(
        [
            Job(
                id=f"{prefix}_{i:03d}",
                status=JobStatus.QUEUED.value,
                work_step_id=steps[i].id,
                queue_group="primary",
                queue_priority=0,
                queue_ticket=f"{prefix}-ticket-{i:03d}",
                enqueued_at=now,
            )
            for i in range(depth)
        ]
    )
    session.flush()


def test_one_eligibility_pass_queries_dependencies_once_per_queued_job(
    settings: Settings,
    monkeypatch: Any,
) -> None:
    """One eligibility pass issues one dependency query per queued job.

    `_acquire_job` runs one task per QUEUED job and each pass calls
    `_eligible_jobs`, which builds its blocked set by inspecting every queued
    job. A single pass is therefore already linear in depth, and N waiters each
    doing that is quadratic in queue depth.

    The count is the assertion, deliberately, not elapsed time: a timing
    assertion passes or fails on machine speed and says nothing about the
    algorithm. Every job here owns a real WorkStep with a real dependency, so the
    number counts database work rather than short-circuited helper calls.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    original = ResourceScheduler._blocking_steps
    observed: dict[int, int] = {}

    for depth in (1, 8, 32):
        with SessionLocal() as session:
            _seed_queue(session, depth, now, f"pass{depth}")
            queries = 0

            def counting(
                inner_session: Any,
                work_step_id: str | None,
                _original: Any = original,
            ) -> list[str]:
                nonlocal queries
                assert work_step_id is not None, "fixture must exercise the query path"
                queries += 1
                blocking: list[str] = _original(inner_session, work_step_id)
                return blocking

            monkeypatch.setattr(ResourceScheduler, "_blocking_steps", staticmethod(counting))
            eligible = ResourceScheduler._eligible_jobs(session, "primary", now)
            monkeypatch.undo()

            assert len(eligible) == depth
            observed[depth] = queries

    assert observed == {1: 1, 8: 8, 32: 32}


def test_one_queue_epoch_is_bounded_linearly_in_queue_depth(
    settings: Settings,
    monkeypatch: Any,
) -> None:
    """An epoch - every waiter taking one pass - costs O(N), not O(N * N).

    Without sharing, N waiters each run the linear scan above, so an epoch costs
    N * N dependency queries: 1,024 at depth 32 against 32 at depth 1, five times
    a second while the queue sits still. Sharing one scan per poll window makes
    an epoch cost N.

    The clock is FAKE here. The share has a real time bound, so a test using the
    real clock would pass or fail on how long 32 sessions take on the runner -
    which is the machine dependence this assertion exists to exclude.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    original = ResourceScheduler._blocking_steps
    observed: dict[int, int] = {}

    for depth in (1, 8, 32):
        with SessionLocal() as session:
            _seed_queue(session, depth, now, f"epoch{depth}")
            clock = {"t": 1000.0}
            monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))
            queries = 0

            def counting(
                inner_session: Any,
                work_step_id: str | None,
                _original: Any = original,
            ) -> list[str]:
                nonlocal queries
                queries += 1
                blocking: list[str] = _original(inner_session, work_step_id)
                return blocking

            monkeypatch.setattr(ResourceScheduler, "_blocking_steps", staticmethod(counting))
            scheduler = ResourceScheduler(session_factory=SessionLocal)
            for _waiter in range(depth):
                scheduler._eligible_job_ids(session, "primary", now)
            monkeypatch.undo()

            observed[depth] = queries

    assert observed == {1: 1, 8: 8, 32: 32}
    for depth, queries in observed.items():
        assert queries <= 2 * depth, (
            f"depth {depth} cost {queries} dependency queries in one epoch; the bound "
            f"is 2 * depth, and exceeding it means waiter work depends on queue depth "
            f"again"
        )


def test_a_slow_scan_is_still_shared_across_waiters(
    settings: Settings,
    monkeypatch: Any,
) -> None:
    """A scan slower than the share window must still be shared, not rescanned.

    The share is published with a timestamp. Stamping when the scan STARTED
    publishes an entry already stale by however long the scan took, so a scan
    costing more than the window expires before the next waiter reaches it and
    every waiter rescans - restoring the quadratic precisely in the case the
    sharing exists for, because a slow scan is when sharing matters.

    Deterministic and clock-free: module `monotonic` is replaced by a fake clock
    that advances only when a scan COMPLETES, by one window plus a millisecond.
    Real sleeping would make this a timing test and prove nothing.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    depth = 8

    with SessionLocal() as session:
        _seed_queue(session, depth, now, "slow")
        session.commit()

        clock = {"t": 1000.0}
        monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))

        scans = 0
        real_eligible = ResourceScheduler._eligible_jobs

        def slow_scan(inner_session: Any, group: str, when: Any) -> list[Job]:
            nonlocal scans
            scans += 1
            result = real_eligible(inner_session, group, when)
            clock["t"] += _ELIGIBILITY_SHARE_SECONDS + 0.001
            return result

        monkeypatch.setattr(ResourceScheduler, "_eligible_jobs", staticmethod(slow_scan))
        scheduler = ResourceScheduler(session_factory=SessionLocal)
        for _waiter in range(depth):
            scheduler._eligible_job_ids(session, "primary", now)

    assert scans == 1, (
        f"{scans} scans for {depth} waiters. Stamping the scan START publishes an "
        f"entry already expired when the scan is slower than the window, so every "
        f"waiter rescans."
    )


class _ClaimScanReached(Exception):
    """Raised from the scan probe to end the acquire loop without a timer."""


def test_a_shared_scan_can_outlive_the_fact_it_recorded(
    settings: Settings,
    monkeypatch: Any,
) -> None:
    """The share is bounded by time, so it can name a job that is no longer eligible.

    A job is eligible, the share records that, and then its upstream stops being
    complete. Until the window elapses, the shared answer still names the job.
    That is tolerable for a progress display and wrong for a decision to start
    the job, so the fresh scan must disagree with the share here, and must
    replace it.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()

    with SessionLocal() as session:
        _seed_queue(session, 1, now, "stale")
        clock = {"t": 1000.0}
        monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))
        scheduler = ResourceScheduler(session_factory=SessionLocal)

        assert scheduler._eligible_job_ids(session, "primary", now) == ("stale_000",)

        upstream = session.get(WorkStep, "step_stale_up")
        assert upstream is not None
        upstream.status = JobStatus.RUNNING.value
        session.flush()

        assert scheduler._eligible_job_ids(session, "primary", now) == ("stale_000",), (
            "the share is meant to be held for the window; if it re-scanned here "
            "this test is no longer describing the shared path"
        )
        assert scheduler._fresh_eligible_job_ids(session, "primary", now) == ()
        assert scheduler._eligible_job_ids(session, "primary", now) == (), (
            "a fresh scan must replace the share, not sit beside it"
        )


def test_the_claim_path_scans_fresh_rather_than_trusting_the_share(
    settings: Settings,
    monkeypatch: Any,
) -> None:
    """Starting a job must never be authorized by a shared snapshot.

    The share is warm and the clock is frozen, so the progress path inside
    `_acquire_job` answers from the share without scanning. Any scan that then
    happens is the claim path scanning fresh. The probe raises at that scan,
    which both records that it happened and ends the loop without a timer.

    Trusting the share here would let a job START on an answer up to one scan
    plus one window old, while the claim statement guards only that the row is
    still QUEUED and unclaimed - not that its dependencies are still complete,
    and not that no higher-priority job has since been enqueued ahead of it.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()

    with SessionLocal() as session:
        _seed_queue(session, 1, now, "claimfresh")
        session.commit()

    clock = {"t": 1000.0}
    monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))

    async def _no_publish(self: Any, job_id: str) -> None:
        return None

    monkeypatch.setattr(ResourceScheduler, "_publish_job", _no_publish)
    scheduler = ResourceScheduler(session_factory=SessionLocal)

    with SessionLocal() as session:
        assert scheduler._eligible_job_ids(session, "primary", now) == ("claimfresh_000",)

    def probe(inner_session: Any, group: str, when: Any) -> list[Job]:
        raise _ClaimScanReached

    monkeypatch.setattr(ResourceScheduler, "_eligible_jobs", staticmethod(probe))

    async def drive() -> str:
        return await scheduler._acquire_job(
            "claimfresh_000",
            resource="gpu",
            group="primary",
            priority=0,
            capacity=1,
            local_lock=asyncio.Semaphore(1),
        )

    try:
        asyncio.run(drive())
    except _ClaimScanReached:
        pass
    else:
        raise AssertionError(
            "the job was claimed without any fresh scan, so a shared snapshot authorized the start"
        )

    with SessionLocal() as session:
        job = session.get(Job, "claimfresh_000")
        assert job is not None
        assert job.status == JobStatus.QUEUED.value


def _acquire_one_pass(
    scheduler: ResourceScheduler,
    monkeypatch: Any,
    job_id: str,
    capacity: int,
) -> str | None:
    """Drive `_acquire_job` through exactly one pass; return the token, or None.

    `_expire_foreign_claims` runs at the top of every pass, so counting it bounds
    the loop deterministically, with no timer and no dependence on how fast the
    runner is. A pass that claims returns its token before the second pass
    begins, so None means the pass declined to claim.
    """

    passes = {"n": 0}

    def stop_after_one(self: Any, group: str) -> list[str]:
        passes["n"] += 1
        if passes["n"] > 1:
            raise _ClaimScanReached
        return []

    async def _no_publish(self: Any, publish_job_id: str) -> None:
        return None

    monkeypatch.setattr(ResourceScheduler, "_expire_foreign_claims", stop_after_one)
    monkeypatch.setattr(ResourceScheduler, "_publish_job", _no_publish)

    async def drive() -> str:
        return await scheduler._acquire_job(
            job_id,
            resource="gpu",
            group="primary",
            priority=0,
            capacity=capacity,
            local_lock=asyncio.Semaphore(1),
        )

    try:
        return asyncio.run(drive())
    except _ClaimScanReached:
        return None


def test_the_share_expires_at_the_window_boundary(
    settings: Settings,
    monkeypatch: Any,
) -> None:
    """The share must be held for the window and dropped after it.

    Both halves are asserted. Without the second, an implementation that caches
    the first answer forever passes every other test here, and a queue whose
    contents changed would be answered from a snapshot that never expires.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()

    with SessionLocal() as session:
        _seed_queue(session, 1, now, "expiry")
        clock = {"t": 1000.0}
        monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))

        scans = 0
        real_eligible = ResourceScheduler._eligible_jobs

        def counting(inner_session: Any, group: str, when: Any) -> list[Job]:
            nonlocal scans
            scans += 1
            return real_eligible(inner_session, group, when)

        monkeypatch.setattr(ResourceScheduler, "_eligible_jobs", staticmethod(counting))
        scheduler = ResourceScheduler(session_factory=SessionLocal)

        scheduler._eligible_job_ids(session, "primary", now)
        assert scans == 1

        clock["t"] += _ELIGIBILITY_SHARE_SECONDS / 2
        scheduler._eligible_job_ids(session, "primary", now)
        assert scans == 1, "inside the window the scan must be shared, not repeated"

        clock["t"] += _ELIGIBILITY_SHARE_SECONDS
        scheduler._eligible_job_ids(session, "primary", now)
        assert scans == 2, (
            "past the window the share must be dropped; a share that never "
            "expires answers from a snapshot of a queue that has since changed"
        )


def test_a_job_whose_dependency_stopped_being_complete_is_not_claimed(
    settings: Settings,
    monkeypatch: Any,
) -> None:
    """The outcome, not just the mechanism: no token, and the row stays QUEUED.

    The share is warmed while the job is eligible and the clock is frozen, so
    the shared answer still names the job when the claim is attempted. Its
    prerequisite is no longer complete by then, and the claim statement guards
    only that the row is still QUEUED and unclaimed, so nothing downstream would
    catch it. Only a fresh scan at the claim can.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()

    with SessionLocal() as session:
        _seed_queue(session, 1, now, "dropdep")
        session.commit()

    clock = {"t": 1000.0}
    monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))
    scheduler = ResourceScheduler(session_factory=SessionLocal)

    with SessionLocal() as session:
        assert scheduler._eligible_job_ids(session, "primary", now) == ("dropdep_000",)
        upstream = session.get(WorkStep, "step_dropdep_up")
        assert upstream is not None
        upstream.status = JobStatus.RUNNING.value
        session.commit()

    token = _acquire_one_pass(scheduler, monkeypatch, "dropdep_000", capacity=1)

    assert token is None, "a job started while its prerequisite was incomplete"
    with SessionLocal() as session:
        job = session.get(Job, "dropdep_000")
        assert job is not None
        assert job.status == JobStatus.QUEUED.value
        assert job.claim_owner is None


def test_a_job_overtaken_during_the_share_window_is_not_claimed(
    settings: Settings,
    monkeypatch: Any,
) -> None:
    """Fresh dependency status alone would not protect ORDER.

    A higher-priority job is enqueued after the share is taken. At capacity one
    the waiting job is no longer at the front, so it must not start, even though
    its own prerequisites are still perfectly satisfied and the shared answer
    still puts it first.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()

    with SessionLocal() as session:
        _seed_queue(session, 1, now, "overtaken")
        session.commit()

    clock = {"t": 1000.0}
    monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))
    scheduler = ResourceScheduler(session_factory=SessionLocal)

    with SessionLocal() as session:
        assert scheduler._eligible_job_ids(session, "primary", now) == ("overtaken_000",)
        session.add(
            Job(
                id="overtaken_urgent",
                status=JobStatus.QUEUED.value,
                queue_group="primary",
                queue_priority=5,
                queue_ticket="overtaken-ticket-urgent",
                enqueued_at=now,
            )
        )
        session.commit()

    token = _acquire_one_pass(scheduler, monkeypatch, "overtaken_000", capacity=1)

    assert token is None, "a job started ahead of a higher-priority job"
    with SessionLocal() as session:
        job = session.get(Job, "overtaken_000")
        assert job is not None
        assert job.status == JobStatus.QUEUED.value
        assert job.claim_owner is None
