from __future__ import annotations

import asyncio
import sqlite3
import time as _real_time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from run_waits import PATIENCE_SECONDS, wait_until
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from local_lm import scheduler as scheduler_module
from local_lm.config import Settings
from local_lm.db import (
    SessionLocal,
    configure_database,
    database_is_contended,
    init_db,
)
from local_lm.domain import JobStatus, utcnow
from local_lm.models import (
    Chat,
    GenerationQueuePolicy,
    Job,
    Message,
    Run,
    WorkPlan,
    WorkPlanControl,
    WorkStep,
    WorkStepDependency,
)
from local_lm.power_inhibition import PowerInhibitor
from local_lm.scheduler import (
    _ELIGIBILITY_SHARE_SECONDS,
    AdmissionCheck,
    JobClaim,
    ResourceScheduler,
)


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
        chat = Chat()
        session.add(chat)
        session.flush()
        user = Message(chat_id=chat.id, role="user")
        assistant = Message(chat_id=chat.id, role="assistant")
        session.add_all([user, assistant])
        session.flush()
        source = Run(
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            operation="image_edit",
            status="complete",
        )
        session.add(source)
        session.flush()
        session.add_all(
            [
                Job(
                    id="job_background_check",
                    kind="edit_verify",
                    payload_json={"source_run_id": source.id},
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


def _real_sqlite_error(tmp_path: Path, *, contended: bool) -> OperationalError:
    """Produce a genuine sqlite3 failure of the requested kind, and wrap it.

    Genuine on both sides on purpose. A constructed error with an error code
    assigned by hand would only prove that the test and the predicate agree
    about an attribute the test set, which is no evidence about what SQLite
    does. A busy timeout of a tenth of a second keeps the contended case fast.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    database = tmp_path / "contended.sqlite3"
    holder = sqlite3.connect(database, timeout=5)
    try:
        holder.execute("PRAGMA journal_mode=WAL")
        holder.execute("CREATE TABLE waiting (id INTEGER PRIMARY KEY)")
        holder.commit()
        if contended:
            holder.execute("BEGIN IMMEDIATE")
            holder.execute("INSERT INTO waiting (id) VALUES (1)")
        other = sqlite3.connect(database, timeout=0.1)
        try:
            if contended:
                other.execute("BEGIN IMMEDIATE")
                other.execute("INSERT INTO waiting (id) VALUES (2)")
            else:
                other.execute("SELECT missing FROM waiting")
        except sqlite3.OperationalError as error:
            return OperationalError("statement", {}, error)
        finally:
            other.close()
        raise AssertionError("sqlite did not fail as this helper requires")
    finally:
        holder.rollback()
        holder.close()


def test_a_held_database_is_told_apart_from_a_broken_one(tmp_path: Path) -> None:
    """The difference is SQLite's own code, not the message."""

    contended = _real_sqlite_error(tmp_path / "busy", contended=True)
    broken = _real_sqlite_error(tmp_path / "broken", contended=False)
    assert database_is_contended(contended) is True
    assert database_is_contended(broken) is False


def _one_queued_job(settings: Settings) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_waiting",
                kind="image",
                status=JobStatus.QUEUED.value,
                queue_group="primary",
                queue_ticket="ticket-a",
                enqueued_at=utcnow(),
            )
        )
        session.commit()


async def _claim(scheduler: ResourceScheduler) -> Any:
    return await asyncio.wait_for(
        scheduler._acquire_job(
            "job_waiting",
            resource="primary",
            group="primary",
            priority=0,
            capacity=1,
            local_lock=asyncio.Semaphore(1),
        ),
        timeout=PATIENCE_SECONDS,
    )


async def test_a_queued_job_waits_out_a_held_database_rather_than_failing(
    settings: Settings, tmp_path: Path
) -> None:
    """A sweep holding the writer must not end the jobs that are queued behind it.

    The startup retention sweep holds SQLite's writer for most of every batch,
    batch after batch. A job that is only waiting its turn writes its queue
    bookkeeping on its first pass, and a read a few lines later flushes it, so
    the wait is where the lock lands. Waiting longer is the answer; failing the
    job is not, and the queue loop already knows how to wait.
    """

    _one_queued_job(settings)
    scheduler = ResourceScheduler(session_factory=SessionLocal)
    original = scheduler._eligible_job_ids
    refusals: list[int] = []

    def contended_once(*args: Any, **kwargs: Any) -> Any:
        if not refusals:
            refusals.append(1)
            raise _real_sqlite_error(tmp_path / "busy", contended=True)
        return original(*args, **kwargs)

    scheduler._eligible_job_ids = contended_once  # type: ignore[method-assign]

    claim = await _claim(scheduler)

    assert refusals == [1], "the contended pass has to have happened"
    assert claim.token
    with SessionLocal() as session:
        job = session.get(Job, "job_waiting")
        assert job is not None
        assert job.status == JobStatus.RUNNING.value


async def test_a_database_error_that_is_not_contention_still_reaches_the_caller(
    settings: Settings, tmp_path: Path
) -> None:
    """Waiting is right for a held database and wrong for a broken one.

    Without this the repair above could be a bare except that swallows every
    database failure, and a queue would wait forever on damage nobody reports.
    """

    _one_queued_job(settings)
    scheduler = ResourceScheduler(session_factory=SessionLocal)
    broken = _real_sqlite_error(tmp_path / "broken", contended=False)

    def always_broken(*args: Any, **kwargs: Any) -> Any:
        raise broken

    scheduler._eligible_job_ids = always_broken  # type: ignore[method-assign]

    with pytest.raises(OperationalError):
        await _claim(scheduler)


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

    async def drive() -> JobClaim:
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
) -> JobClaim | None:
    """Drive `_acquire_job` through exactly one pass; return the claim, or None.

    `_expire_foreign_claims` runs at the top of every pass, so counting it bounds
    the loop deterministically, with no timer and no dependence on how fast the
    runner is. A pass that claims returns its claim before the second pass
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

    async def drive() -> JobClaim:
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


async def test_a_failed_release_still_lets_the_next_job_take_its_turn(
    settings: Settings,
) -> None:
    """A release that cannot be written must not hold the group's only slot.

    The slot is in this process's memory and the release is a database write.
    When a long writer holds the database past the busy timeout, that write
    fails; if the slot went with it, every later job in the group would wait on
    a lock nothing will ever free, and only a restart would clear it.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    with SessionLocal() as session:
        for index in (1, 2):
            session.add(
                Job(
                    id=f"job_release_{index}",
                    status=JobStatus.QUEUED.value,
                    queue_group="primary",
                    queue_ticket=f"ticket-release-{index}",
                    enqueued_at=now,
                )
            )
        session.commit()

    scheduler = ResourceScheduler(session_factory=SessionLocal)

    async def refusing_release(job_id: str, token: str, group: str) -> None:
        raise RuntimeError("database is locked")

    scheduler._release_job = refusing_release  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        async with scheduler.job_lease(
            "job_release_1", resource="media_compute", group="primary", capacity=1
        ):
            pass

    second = asyncio.create_task(
        scheduler._acquire_job(
            "job_release_2",
            resource="media_compute",
            group="primary",
            priority=0,
            capacity=1,
            local_lock=scheduler._lock("primary", 1),
        )
    )
    try:
        claim = await asyncio.wait_for(second, timeout=PATIENCE_SECONDS)
        assert claim.token
    finally:
        if not second.done():
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)


async def test_a_failed_heartbeat_still_releases_the_slot_and_the_claim(
    settings: Settings,
) -> None:
    """The heartbeat writes too, so it fails the same way a release does.

    Waiting for it on the way out used to raise before the release ran, which
    left the claim owned and the slot held: the same stall, reached through the
    other write.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_heartbeat_failure",
                status=JobStatus.QUEUED.value,
                queue_group="primary",
                queue_ticket="ticket-heartbeat",
                enqueued_at=now,
            )
        )
        session.commit()

    scheduler = ResourceScheduler(session_factory=SessionLocal)

    async def failing_heartbeat(
        job_id: str, token: str, *, on_claim_lost: Callable[[], None] | None = None
    ) -> None:
        raise RuntimeError("database is locked")

    scheduler._heartbeat = failing_heartbeat  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        async with scheduler.job_lease(
            "job_heartbeat_failure", resource="media_compute", group="primary", capacity=1
        ):
            await asyncio.sleep(0)

    assert not scheduler._lock("primary", 1).locked()
    with SessionLocal() as session:
        job = session.get(Job, "job_heartbeat_failure")
        assert job is not None
        assert job.claim_owner is None


async def test_an_abandoned_claim_survives_a_failed_expiry_write(settings: Settings) -> None:
    """Forgetting the token before its row is cleared would strand it for good.

    The expiry pass skips this dispatcher's own claims, so the abandoned set is
    the only thing that brings one back. If a failed clearing write dropped the
    token anyway, the row would keep its claim and no later pass would look at
    it again.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    now = utcnow()
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_expiry_failure",
                status=JobStatus.QUEUED.value,
                queue_group="primary",
                queue_ticket="ticket-expiry",
                enqueued_at=now,
            )
        )
        session.commit()

    scheduler = ResourceScheduler(session_factory=SessionLocal)

    async def refusing_release(job_id: str, token: str, group: str) -> None:
        raise RuntimeError("database is locked")

    scheduler._release_job = refusing_release  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        async with scheduler.job_lease(
            "job_expiry_failure", resource="media_compute", group="primary", capacity=1
        ):
            pass

    refusals = {"left": 1}
    real_commit = Session.commit

    def refusing_commit(self: Session) -> None:
        if refusals["left"]:
            refusals["left"] -= 1
            raise RuntimeError("database is locked")
        real_commit(self)

    Session.commit = refusing_commit  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError):
            scheduler._expire_foreign_claims("primary")
    finally:
        Session.commit = real_commit  # type: ignore[method-assign]

    with SessionLocal() as session:
        job = session.get(Job, "job_expiry_failure")
        assert job is not None
        assert job.claim_owner is not None

    assert scheduler._expire_foreign_claims("primary") == ["job_expiry_failure"]
    with SessionLocal() as session:
        job = session.get(Job, "job_expiry_failure")
        assert job is not None
        assert job.claim_owner is None


#: The stage an admission check gives the job it holds back, in the tests below.
_WAITING = "waiting for other work"


class _RecordingPower:
    """A keep-awake backend that records every request it is asked to make."""

    name = "recording"
    supported = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def acquire(self, reason: str) -> object:
        self.calls.append("acquire")
        return object()

    def release(self, handle: object) -> None:
        self.calls.append("release")


class _Passes:
    """Counts the scheduler's passes over the queue.

    Every pass expires stale claims first, once, so counting that call counts
    passes without a timer. With no event broker attached and a check that
    answers at once, a pass never yields before its wait for the next poll, so
    a count the test reads always belongs to finished passes.
    """

    def __init__(self, scheduler: ResourceScheduler, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        expire = scheduler._expire_foreign_claims

        def counted(group: str) -> list[str]:
            self.count += 1
            return expire(group)

        monkeypatch.setattr(scheduler, "_expire_foreign_claims", counted)

    async def wait(self, more: int) -> None:
        """Return once the scheduler has passed over the queue `more` more times."""

        target = self.count + more

        async def read() -> int:
            return self.count

        await wait_until(read, lambda count: count >= target, what="passes over the queue")


def _queue_media_jobs(settings: Settings, *job_ids: str, plan_id: str | None = None) -> None:
    """Queue media jobs in one group, oldest first.

    Under a named work plan, each job also owns a step that depends on the
    plan's upstream step, which starts complete.
    """

    settings.prepare()
    configure_database(settings)
    init_db()
    first_enqueued = utcnow() - timedelta(seconds=len(job_ids))
    with SessionLocal() as session:
        if plan_id is not None:
            chat = Chat(id="chat_admission", title="Admission")
            plan = WorkPlan(id=plan_id, chat_id=chat.id, transcript_sequence=1, summary_json={})
            session.add_all([chat, plan])
            session.flush()
            upstream = WorkStep(
                id=f"{plan_id}_upstream",
                plan=plan,
                ordinal=0,
                operation="text",
                status=JobStatus.COMPLETE.value,
            )
            steps = [
                WorkStep(id=f"{plan_id}_{job_id}", plan=plan, ordinal=index + 1, operation="text")
                for index, job_id in enumerate(job_ids)
            ]
            session.add_all([upstream, *steps])
            session.flush()
            session.add_all(
                WorkStepDependency(step_id=step.id, depends_on_step_id=upstream.id)
                for step in steps
            )
        session.add_all(
            Job(
                id=job_id,
                kind="image",
                status=JobStatus.QUEUED.value,
                work_plan_id=plan_id,
                work_step_id=None if plan_id is None else f"{plan_id}_{job_id}",
                queue_group="primary",
                queue_ticket=f"ticket-{index}",
                enqueued_at=first_enqueued + timedelta(seconds=index),
            )
            for index, job_id in enumerate(job_ids)
        )
        session.commit()


async def _queue_state(job_id: str) -> dict[str, Any]:
    """What the queue records for a job: its status, claim, attempt, stage and place."""

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        progress = job.progress_json if isinstance(job.progress_json, dict) else {}
        return {
            "status": job.status,
            "claimed": job.claim_owner is not None,
            "attempt": job.attempt,
            "stage": progress.get("stage"),
            "position": progress.get("queue_position"),
        }


async def _shows(job_id: str, stage: str, waiting: asyncio.Task[None]) -> dict[str, Any]:
    """Wait for the job to show `stage`, failing at once with the lease's own error."""

    async def read() -> dict[str, Any]:
        if waiting.done():
            waiting.result()
        return await _queue_state(job_id)

    return await wait_until(
        read, lambda state: state["stage"] == stage, what=f"the stage {job_id} shows"
    )


async def test_a_job_its_admission_holds_back_stays_queued_while_the_next_job_runs(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job that may not start yet waits unclaimed, and the next job goes first.

    Its admission check answers with a stage to show instead of admitting it.
    Keeping its place at the head of the queue, or the group's only slot, while
    it waits would hold every other job in the group up behind work that is not
    theirs. Once the check admits it, it takes its turn like any other job.
    """

    _queue_media_jobs(settings, "job_a", "job_b")
    monkeypatch.setattr(scheduler_module, "_ADMISSION_RECHECK_SECONDS", 0.05)
    scheduler = ResourceScheduler(session_factory=SessionLocal)
    slot = scheduler._lock("primary", 1)
    busy = True
    asked_holding_the_slot: list[bool] = []

    async def admission() -> str | None:
        asked_holding_the_slot.append(slot.locked())
        # A real check waits for an answer from somewhere, so this one gives way
        # once too. A scheduler that went on asking without waiting for its
        # next poll then fails this test within its patience rather than
        # stalling it.
        await asyncio.sleep(0)
        return _WAITING if busy else None

    started = asyncio.Event()
    finish = asyncio.Event()
    claims: list[JobClaim] = []

    async def held_back_job() -> None:
        async with scheduler.job_lease(
            "job_a", resource="media_compute", group="primary", admission=admission
        ) as claim:
            claims.append(claim)
            started.set()
            await finish.wait()

    waiting = asyncio.create_task(held_back_job())
    try:
        held_back = await _shows("job_a", _WAITING, waiting)
        assert held_back == {
            "status": JobStatus.QUEUED.value,
            "claimed": False,
            "attempt": 0,
            "stage": _WAITING,
            "position": None,
        }
        assert asked_holding_the_slot[0], "the check just before the claim ran without the slot"
        assert scheduler.peek_next_eligible_job("primary") == ("job_b", None)

        async with asyncio.timeout(PATIENCE_SECONDS):
            async with scheduler.job_lease("job_b", resource="media_compute", group="primary"):
                assert (await _queue_state("job_b"))["status"] == JobStatus.RUNNING.value
                assert await _queue_state("job_a") == held_back
                assert scheduler.peek_next_eligible_job("primary") is None

                busy = False
                back_in_line = await _shows("job_a", "queued", waiting)
                assert back_in_line == {
                    "status": JobStatus.QUEUED.value,
                    "claimed": False,
                    "attempt": 0,
                    "stage": "queued",
                    "position": 0,
                }
                assert scheduler.peek_next_eligible_job("primary") == ("job_a", None)
                assert not started.is_set(), "the admitted job started while the slot was taken"

        await asyncio.wait_for(started.wait(), timeout=PATIENCE_SECONDS)
        assert claims[0].attempt == 1
        assert (await _queue_state("job_a"))["status"] == JobStatus.RUNNING.value
    finally:
        finish.set()
        if not started.is_set():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.parametrize(
    "ending", ["refused-at-once", "refused-while-waiting", "cancelled-while-waiting"]
)
async def test_however_a_held_back_wait_ends_it_leaves_nothing_claimed_or_held(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    """A refusal or a cancellation ends the wait with nothing to undo.

    The job never claimed, so its attempt count is unchanged, the slot is free
    for the rest of the group and nothing keeps the computer awake. It is next
    in line again, rather than passed over for a check that no longer runs.
    """

    _queue_media_jobs(settings, "job_a", "job_b")
    monkeypatch.setattr(scheduler_module, "_ADMISSION_RECHECK_SECONDS", 0.05)
    keep_awake = _RecordingPower()
    power = PowerInhibitor(keep_awake)
    scheduler = ResourceScheduler(session_factory=SessionLocal, power=power)

    class Refused(Exception):
        pass

    asked = 0

    async def admission() -> str | None:
        nonlocal asked
        asked += 1
        if ending == "refused-at-once" or (ending == "refused-while-waiting" and asked > 1):
            raise Refused
        return _WAITING

    async def held_back_job() -> None:
        async with scheduler.job_lease(
            "job_a", resource="media_compute", group="primary", admission=admission
        ):
            raise AssertionError("a job its admission check did not admit was claimed")

    waiting = asyncio.create_task(held_back_job())
    if ending == "cancelled-while-waiting":
        await _shows("job_a", _WAITING, waiting)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        assert waiting.cancelled()
    else:
        with pytest.raises(Refused):
            await asyncio.wait_for(waiting, timeout=PATIENCE_SECONDS)
        assert asked == (1 if ending == "refused-at-once" else 2)

    state = await _queue_state("job_a")
    assert (state["status"], state["claimed"], state["attempt"]) == (
        JobStatus.QUEUED.value,
        False,
        0,
    )
    assert not scheduler._lock("primary", 1).locked()
    assert keep_awake.calls == []
    assert power.state().holder_count == 0
    assert scheduler.peek_next_eligible_job("primary") == ("job_a", None)


@pytest.mark.parametrize(
    ("stop", "stage"),
    [("hold", "held"), ("pause", "generation paused"), ("dependency", _WAITING)],
    ids=["hold", "pause", "dependency"],
)
async def test_a_held_back_job_is_asked_again_only_once_a_hold_pause_or_dependency_ends(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, stop: str, stage: str
) -> None:
    """A hold, a paused lane or an unfinished dependency stops the check being asked.

    No answer could start the job meanwhile, and a check that bounds its own
    wait would spend it on a stretch no answer could end. Another job holding
    the slot is no such stop: the job stood aside for that job and is asked
    all the while. A hold or a pause is what the job shows while it lasts;
    behind an unfinished dependency it goes on showing the check's stage. Once
    the stop ends, the check is asked again.
    """

    _queue_media_jobs(settings, "job_a", plan_id="plan_admission")
    monkeypatch.setattr(scheduler_module, "_ADMISSION_RECHECK_SECONDS", 0.05)
    scheduler = ResourceScheduler(session_factory=SessionLocal)
    passes = _Passes(scheduler, monkeypatch)
    asked = 0

    async def admission() -> str | None:
        nonlocal asked
        asked += 1
        return _WAITING

    async def times_asked() -> int:
        return asked

    def stopped(active: bool) -> None:
        with SessionLocal() as session:
            if stop == "hold":
                session.merge(
                    WorkPlanControl(
                        plan_id="plan_admission",
                        state="held" if active else "eligible",
                        revision=1 if active else 2,
                        eligible_since=None if active else utcnow(),
                    )
                )
            elif stop == "pause":
                session.merge(
                    GenerationQueuePolicy(
                        lane="generation",
                        dispatch_state="paused" if active else "open",
                        revision=1 if active else 2,
                    )
                )
            else:
                upstream = session.get(WorkStep, "plan_admission_upstream")
                assert upstream is not None
                upstream.status = JobStatus.RUNNING.value if active else JobStatus.COMPLETE.value
            session.commit()

    async def held_back_job() -> None:
        async with scheduler.job_lease(
            "job_a", resource="media_compute", group="primary", admission=admission
        ):
            raise AssertionError("a job its admission check did not admit was claimed")

    waiting = asyncio.create_task(held_back_job())
    try:
        await _shows("job_a", _WAITING, waiting)
        await wait_until(times_asked, lambda count: count >= 2, what="asks while held back")
        stopped(True)
        await _shows("job_a", stage, waiting)
        # Every pass reads the stop by now: a pass scans afresh once the last
        # shared scan is older than a quarter of the poll interval.
        await passes.wait(2)
        asked_when_stopped = asked
        await passes.wait(3)
        assert asked == asked_when_stopped, f"the check was asked while stopped by a {stop}"

        stopped(False)
        await _shows("job_a", _WAITING, waiting)
        await wait_until(
            times_asked, lambda count: count > asked_when_stopped, what="asks once it ended"
        )
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def test_a_held_back_job_asks_again_at_its_own_pace_without_the_slot(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job asks again every few seconds, not every time it looks at the queue.

    It looks five times a second, and the check may have to ask another program
    each time, so the check keeps a slower pace of its own. Only the check just
    before a claim holds the slot; asking again while held back holds nothing.
    """

    _queue_media_jobs(settings, "job_a")
    clock = {"t": 1000.0}
    monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))
    scheduler = ResourceScheduler(session_factory=SessionLocal)
    passes = _Passes(scheduler, monkeypatch)
    slot = scheduler._lock("primary", 1)
    asked_holding_the_slot: list[bool] = []

    async def admission() -> str | None:
        asked_holding_the_slot.append(slot.locked())
        return _WAITING

    async def held_back_job() -> None:
        async with scheduler.job_lease(
            "job_a", resource="media_compute", group="primary", admission=admission
        ):
            raise AssertionError("a job its admission check did not admit was claimed")

    waiting = asyncio.create_task(held_back_job())
    try:
        await _shows("job_a", _WAITING, waiting)
        await passes.wait(3)
        assert asked_holding_the_slot == [True]

        clock["t"] = 1000.0 + scheduler_module._ADMISSION_RECHECK_SECONDS - 0.1
        await passes.wait(3)
        assert asked_holding_the_slot == [True], "the job asked again before its pace allowed"

        clock["t"] = 1000.0 + scheduler_module._ADMISSION_RECHECK_SECONDS + 0.1
        await passes.wait(3)
        assert asked_holding_the_slot == [True, False]
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def test_a_dispatcher_sharing_the_pool_passes_over_the_same_held_back_job(
    settings: Settings,
) -> None:
    """Every dispatcher on one pool passes over a job any of them is holding back.

    They share the slots, so a dispatcher that did not know of the hold would
    keep a place for a job that cannot start yet, and wait behind it.
    """

    _queue_media_jobs(settings, "job_a", "job_b")
    pool = ResourceScheduler(session_factory=SessionLocal)
    scheduler = ResourceScheduler(session_factory=SessionLocal, resource_pool=pool)

    async def admission() -> str | None:
        return _WAITING

    async def held_back_job() -> None:
        async with pool.job_lease(
            "job_a", resource="media_compute", group="primary", admission=admission
        ):
            raise AssertionError("a job its admission check did not admit was claimed")

    waiting = asyncio.create_task(held_back_job())
    try:
        await _shows("job_a", _WAITING, waiting)
        assert scheduler.peek_next_eligible_job("primary") == ("job_b", None)
        async with asyncio.timeout(PATIENCE_SECONDS):
            async with scheduler.job_lease("job_b", resource="media_compute", group="primary"):
                assert (await _queue_state("job_b"))["status"] == JobStatus.RUNNING.value
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


class _Runs:
    """Media jobs that each keep their claim until the test lets them all finish.

    Records the order the jobs start in, and every answer their admission
    checks give: the job, the answer, and whether the slot was held when the
    check was asked. A check answers at once, so an answer recorded here has
    already been acted on by the time the test reads it.
    """

    def __init__(self, scheduler: ResourceScheduler) -> None:
        self.scheduler = scheduler
        self.slot = scheduler._lock("primary", 1)
        self.started: list[str] = []
        self.asked: list[tuple[str, str | None, bool]] = []
        self.finish = asyncio.Event()
        self.tasks: list[asyncio.Task[None]] = []

    def start(
        self, job_id: str, *, held_back: Callable[[], bool] | None, priority: int = 0
    ) -> asyncio.Task[None]:
        """Start waiting for the job's turn; `held_back` None gives it no check at all."""

        admission = None if held_back is None else self._check(job_id, held_back)

        async def run() -> None:
            async with self.scheduler.job_lease(
                job_id,
                resource="media_compute",
                group="primary",
                priority=priority,
                admission=admission,
            ):
                self.started.append(job_id)
                await self.finish.wait()

        task = asyncio.create_task(run())
        self.tasks.append(task)
        return task

    def _check(self, job_id: str, held_back: Callable[[], bool]) -> AdmissionCheck:
        async def admission() -> str | None:
            answer = _WAITING if held_back() else None
            self.asked.append((job_id, answer, self.slot.locked()))
            return answer

        return admission

    async def times_asked(self, job_id: str) -> int:
        return sum(1 for asked_job, _answer, _slot in self.asked if asked_job == job_id)

    async def first_to_start(self) -> list[str]:
        """Wait for a job to start, failing at once with any job's own error."""

        async def read() -> list[str]:
            for task in self.tasks:
                if task.done():
                    task.result()
            return list(self.started)

        return await wait_until(read, bool, what="a job starting")

    async def finish_all(self) -> None:
        """Let every job finish, and wait for the ones still queued to run too."""

        self.finish.set()
        await asyncio.wait_for(asyncio.gather(*self.tasks), timeout=PATIENCE_SECONDS)

    async def stop(self) -> None:
        self.finish.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


async def _claim_state(job_id: str) -> tuple[str, bool, int]:
    """The job's status, whether it is claimed, and its attempt count."""

    state = await _queue_state(job_id)
    return state["status"], state["claimed"], state["attempt"]


#: What `_claim_state` reads for a job that is still waiting and never claimed.
_UNCLAIMED = (JobStatus.QUEUED.value, False, 0)


@pytest.mark.parametrize(
    ("queued", "priorities"),
    [
        (("job_a", "job_b"), {"job_a": 0, "job_b": 0}),
        (("job_b", "job_a"), {"job_a": 10, "job_b": 0}),
    ],
    ids=["queued-first", "higher-priority"],
)
async def test_jobs_held_back_by_the_same_work_start_in_queue_order_once_it_ends(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    queued: tuple[str, ...],
    priorities: dict[str, int],
) -> None:
    """The job ranked first starts first, whichever of them hears first that it may.

    Each job held back asks again at its own pace, so once the work holding
    both back ends, the job ranked second can be the first to be admitted. It
    must not start ahead of the first, which has not been asked since: the
    first goes back in line, is asked under the slot and starts, and the
    second waits its turn behind it. The same holds whether queue order or
    priority ranks them.
    """

    _queue_media_jobs(settings, *queued)
    clock = {"t": 1000.0}
    monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))
    pace = scheduler_module._ADMISSION_RECHECK_SECONDS
    runs = _Runs(ResourceScheduler(session_factory=SessionLocal))
    busy = True
    try:
        first = runs.start("job_a", held_back=lambda: busy, priority=priorities["job_a"])
        await _shows("job_a", _WAITING, first)
        # job_b is held back later in the same stretch of work, and job_a then
        # asks again, so job_b's next ask falls due before job_a's.
        clock["t"] = 1000.0 + pace - 1.0
        second = runs.start("job_b", held_back=lambda: busy, priority=priorities["job_b"])
        await _shows("job_b", _WAITING, second)
        clock["t"] = 1000.0 + pace + 0.1

        async def job_a_asks() -> int:
            return await runs.times_asked("job_a")

        await wait_until(job_a_asks, lambda count: count >= 2, what="job_a asking again")
        asked_while_busy = len(runs.asked)

        busy = False
        clock["t"] = 1000.0 + 2 * pace - 0.9
        assert await runs.first_to_start() == ["job_a"], "the job ranked second started first"
        assert runs.asked[asked_while_busy] == ("job_b", None, False), (
            "job_b was not the first to hear that the work had ended"
        )
        assert await _claim_state("job_b") == _UNCLAIMED

        await runs.finish_all()
        assert runs.started == ["job_a", "job_b"]
    finally:
        await runs.stop()


@pytest.mark.parametrize("check", ["admits", "none"])
async def test_a_job_never_held_back_starts_behind_one_held_back_ahead_of_it(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, check: str
) -> None:
    """A job that reaches the slot after the work ends starts behind the job held back.

    The held-back job asks again only at its own pace, so a job queued behind
    it can reach the slot after the work holding it back has ended but before
    it asks again. Whether that job's own check admits it or it has no check,
    it puts the held-back job back in line first, and that job starts first.
    """

    _queue_media_jobs(settings, "job_a", "job_c")
    clock = {"t": 1000.0}
    monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))
    runs = _Runs(ResourceScheduler(session_factory=SessionLocal))
    busy = True
    try:
        held_back = runs.start("job_a", held_back=lambda: busy)
        await _shows("job_a", _WAITING, held_back)
        # The work ends long before job_a's own next ask falls due.
        busy = False
        runs.start("job_c", held_back=(lambda: False) if check == "admits" else None)
        assert await runs.first_to_start() == ["job_a"], (
            "the job that was never held back started first"
        )
        assert await _claim_state("job_c") == _UNCLAIMED

        await runs.finish_all()
        assert runs.started == ["job_a", "job_c"]
    finally:
        await runs.stop()


async def test_a_job_still_held_back_ahead_is_asked_once_more_and_no_more(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job still held back ahead is asked once more, under the slot, then passed over.

    Its check has then answered since the job behind it was admitted, so that
    job starts. Asking it again each time the job behind it comes back to the
    slot would keep that job waiting for good.
    """

    _queue_media_jobs(settings, "job_a", "job_b")
    clock = {"t": 1000.0}
    monkeypatch.setattr(scheduler_module, "time", _FrozenClock(clock))
    scheduler = ResourceScheduler(session_factory=SessionLocal)
    passes = _Passes(scheduler, monkeypatch)
    runs = _Runs(scheduler)
    try:
        held_back = runs.start("job_a", held_back=lambda: True)
        await _shows("job_a", _WAITING, held_back)
        runs.start("job_b", held_back=lambda: False)
        assert await runs.first_to_start() == ["job_b"]
        assert runs.asked == [
            ("job_a", _WAITING, True),
            ("job_b", None, True),
            ("job_a", _WAITING, True),
            ("job_b", None, True),
        ], "job_a was not asked exactly once more, under the slot, before job_b started"
        assert await _queue_state("job_a") == {
            "status": JobStatus.QUEUED.value,
            "claimed": False,
            "attempt": 0,
            "stage": _WAITING,
            "position": None,
        }

        await passes.wait(5)
        assert await runs.times_asked("job_a") == 2, "job_a was asked again while job_b ran"
    finally:
        await runs.stop()
