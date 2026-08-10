from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.domain import JobStatus, utcnow
from local_lm.models import Chat, Job, WorkPlan, WorkStep, WorkStepDependency
from local_lm.scheduler import ResourceScheduler


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
