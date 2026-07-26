from __future__ import annotations

import asyncio
import math
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from .db import SessionLocal
from .domain import JobStatus, MessageStatus, PartType, RunStatus, utcnow
from .models import (
    Job,
    Message,
    MessagePart,
    Run,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)
from .progress import update_job_progress
from .schemas import JobOut
from .work_plans import BLOCKED_WORK_STATUS, plan_status_summary, refresh_plan_status

if TYPE_CHECKING:
    from .events import EventBroker

_CLAIM_SECONDS = 15
_HEARTBEAT_SECONDS = 5
_QUEUE_POLL_SECONDS = 0.2
_AGING_SECONDS = 30
_TERMINAL_STATUSES = {
    JobStatus.COMPLETE.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
    JobStatus.INTERRUPTED.value,
}


class ResourceScheduler:
    """Durable job tickets plus legacy leases for non-job administration."""

    def __init__(
        self,
        events: EventBroker | None = None,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        resource_pool: ResourceScheduler | None = None,
    ) -> None:
        self._locks: dict[str, asyncio.Semaphore] = resource_pool._locks if resource_pool else {}
        self._capacities: dict[str, int] = resource_pool._capacities if resource_pool else {}
        self._queue_events: dict[str, asyncio.Event] = {}
        self._owner = f"dispatcher_{secrets.token_hex(16)}"
        self._events = events
        self.session_factory = session_factory

    @asynccontextmanager
    async def lease(self, device_id: str = "primary") -> AsyncIterator[None]:
        """Compatibility lease for bounded operations that have no durable Job."""

        lock = self._lock(device_id, 1)
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()

    @asynccontextmanager
    async def job_lease(
        self,
        job_id: str,
        *,
        resource: str,
        group: str,
        priority: int = 0,
        capacity: int = 1,
    ) -> AsyncIterator[None]:
        lock = self._lock(group, capacity)
        token = await self._acquire_job(
            job_id,
            resource=resource,
            group=group,
            priority=priority,
            capacity=capacity,
            local_lock=lock,
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(job_id, token),
            name=f"job-heartbeat-{job_id}",
        )
        try:
            yield
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            await self._release_job(job_id, token, group)
            lock.release()

    async def _acquire_job(
        self,
        job_id: str,
        *,
        resource: str,
        group: str,
        priority: int,
        capacity: int,
        local_lock: asyncio.Semaphore,
    ) -> str:
        token = f"{self._owner}_{secrets.token_hex(12)}"
        while True:
            for expired_job_id in self._expire_foreign_claims(group):
                await self._publish_job(expired_job_id)
            changed = False
            should_try_claim = False
            with self.session_factory() as session:
                job = session.get(Job, job_id)
                if not job or job.status in _TERMINAL_STATUSES:
                    raise asyncio.CancelledError
                now = utcnow()
                if not job.enqueued_at:
                    job.enqueued_at = now
                job.queue_resource = resource
                job.queue_group = group
                job.queue_priority = priority
                job.queue_ticket = job.queue_ticket or job.id

                candidates = self._eligible_jobs(session, group, now)
                position = next(
                    (index for index, candidate in enumerate(candidates) if candidate.id == job.id),
                    None,
                )
                if position is None:
                    failed_dependencies = self._failed_dependencies(
                        session,
                        job.work_step_id,
                    )
                    blocked_by = self._blocking_steps(session, job.work_step_id)
                    step = session.get(WorkStep, job.work_step_id) if job.work_step_id else None
                    plan = session.get(WorkPlan, job.work_plan_id) if job.work_plan_id else None
                    if step:
                        step.status = (
                            BLOCKED_WORK_STATUS if failed_dependencies else JobStatus.QUEUED.value
                        )
                        step.error = (
                            "Blocked by unsuccessful required work."
                            if failed_dependencies
                            else None
                        )
                    if plan:
                        session.flush()
                        refresh_plan_status(session, plan.id)
                        plan.summary_json = {
                            **plan.summary_json,
                            "status_counts": plan_status_summary(session, plan.id),
                        }
                    update_job_progress(
                        job,
                        stage=(
                            "blocked by unsuccessful dependency"
                            if failed_dependencies
                            else "waiting for dependencies"
                        ),
                        queue_resource=resource,
                        queue_position=None,
                        queue_length=len(candidates),
                        blocked_by=blocked_by,
                        indeterminate=True,
                        now=now,
                    )
                else:
                    step = session.get(WorkStep, job.work_step_id) if job.work_step_id else None
                    if step and step.status == BLOCKED_WORK_STATUS:
                        step.status = JobStatus.QUEUED.value
                        step.error = None
                        if job.work_plan_id:
                            session.flush()
                            refresh_plan_status(session, job.work_plan_id)
                            plan = session.get(WorkPlan, job.work_plan_id)
                            if plan:
                                plan.summary_json = {
                                    **plan.summary_json,
                                    "status_counts": plan_status_summary(session, plan.id),
                                }
                    update_job_progress(
                        job,
                        stage="queued",
                        queue_resource=resource,
                        queue_position=position,
                        queue_length=len(candidates),
                        indeterminate=True,
                        now=now,
                    )
                session.commit()
                changed = True

                active_claims = (
                    session.scalar(
                        select(func.count(Job.id)).where(
                            Job.queue_group == group,
                            Job.status == JobStatus.RUNNING.value,
                            Job.claim_owner.is_not(None),
                        )
                    )
                    or 0
                )
                should_try_claim = position is not None and position < max(
                    0, capacity - active_claims
                )

            if should_try_claim:
                # A compute slot can remain occupied for minutes. Never keep a
                # SQLite session (and its read transaction) open while waiting.
                await local_lock.acquire()
                claimed = False
                try:
                    with self.session_factory() as session:
                        current = session.get(Job, job_id)
                        claimed_at = utcnow()
                        candidates = self._eligible_jobs(session, group, claimed_at)
                        position = next(
                            (
                                index
                                for index, candidate in enumerate(candidates)
                                if current and candidate.id == current.id
                            ),
                            None,
                        )
                        active_claims = (
                            session.scalar(
                                select(func.count(Job.id)).where(
                                    Job.queue_group == group,
                                    Job.status == JobStatus.RUNNING.value,
                                    Job.claim_owner.is_not(None),
                                )
                            )
                            or 0
                        )
                        if position is not None and position < max(0, capacity - active_claims):
                            result = cast(
                                CursorResult[Any],
                                session.execute(
                                    update(Job)
                                    .where(
                                        Job.id == job_id,
                                        Job.status == JobStatus.QUEUED.value,
                                        Job.claim_owner.is_(None),
                                    )
                                    .values(
                                        status=JobStatus.RUNNING.value,
                                        claim_owner=token,
                                        claim_expires_at=claimed_at
                                        + timedelta(seconds=_CLAIM_SECONDS),
                                        heartbeat_at=claimed_at,
                                        started_at=claimed_at,
                                        attempt=Job.attempt + 1,
                                    )
                                ),
                            )
                            if result.rowcount == 1:
                                claimed_job = session.get(Job, job_id)
                                if claimed_job:
                                    update_job_progress(
                                        claimed_job,
                                        stage="starting",
                                        queue_resource=resource,
                                        queue_position=0,
                                        queue_length=len(candidates),
                                        indeterminate=True,
                                        now=claimed_at,
                                    )
                                session.commit()
                                claimed = True
                            else:
                                session.rollback()
                finally:
                    if not claimed:
                        local_lock.release()
                if claimed:
                    await self._publish_job(job_id)
                    return token

            if changed:
                await self._publish_job(job_id)
            event = self._queue_event(group)
            event.clear()
            with suppress(TimeoutError):
                await asyncio.wait_for(event.wait(), timeout=_QUEUE_POLL_SECONDS)

    @staticmethod
    def _eligible_jobs(session: Session, group: str, now: datetime) -> list[Job]:
        jobs = list(
            session.scalars(
                select(Job).where(
                    Job.queue_group == group,
                    Job.status == JobStatus.QUEUED.value,
                    Job.claim_owner.is_(None),
                )
            ).all()
        )
        blocked = {
            job.id for job in jobs if ResourceScheduler._blocking_steps(session, job.work_step_id)
        }

        def rank(job: Job) -> tuple[int, datetime, str, str]:
            enqueued = job.enqueued_at or job.created_at
            if enqueued.tzinfo is None:
                enqueued = enqueued.replace(tzinfo=UTC)
            waited = max(0.0, (now - enqueued).total_seconds())
            effective_priority = job.queue_priority + math.floor(waited / _AGING_SECONDS)
            return (-effective_priority, enqueued, job.queue_ticket or job.id, job.id)

        return sorted((job for job in jobs if job.id not in blocked), key=rank)

    def peek_next_eligible_job(self, group: str) -> tuple[str, str | None] | None:
        """Return the next durable job without claiming or changing it."""

        with self.session_factory() as session:
            candidates = self._eligible_jobs(session, group, utcnow())
            if not candidates:
                return None
            job = candidates[0]
            return job.id, job.run_id

    @staticmethod
    def _blocking_steps(session: Session, work_step_id: str | None) -> list[str]:
        if not work_step_id:
            return []
        dependencies = session.scalars(
            select(WorkStepDependency.depends_on_step_id).where(
                WorkStepDependency.step_id == work_step_id
            )
        ).all()
        return [
            dependency_id
            for dependency_id in dependencies
            if (
                (step := session.get(WorkStep, dependency_id)) is None
                or step.status != JobStatus.COMPLETE.value
            )
        ]

    @staticmethod
    def _failed_dependencies(session: Session, work_step_id: str | None) -> list[str]:
        if not work_step_id:
            return []
        dependency_ids = session.scalars(
            select(WorkStepDependency.depends_on_step_id).where(
                WorkStepDependency.step_id == work_step_id
            )
        ).all()
        unsuccessful = {
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.INTERRUPTED.value,
            BLOCKED_WORK_STATUS,
        }
        return [
            dependency_id
            for dependency_id in dependency_ids
            if (
                (step := session.get(WorkStep, dependency_id)) is None
                or step.status in unsuccessful
            )
        ]

    async def _heartbeat(self, job_id: str, token: str) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            with self.session_factory() as session:
                now = utcnow()
                result = cast(
                    CursorResult[Any],
                    session.execute(
                        update(Job)
                        .where(
                            Job.id == job_id,
                            Job.claim_owner == token,
                            Job.status == JobStatus.RUNNING.value,
                        )
                        .values(
                            heartbeat_at=now,
                            claim_expires_at=now + timedelta(seconds=_CLAIM_SECONDS),
                        )
                    ),
                )
                session.commit()
                if result.rowcount != 1:
                    return

    def _expire_foreign_claims(self, group: str) -> list[str]:
        """Interrupt abandoned work without risking a duplicate backend request."""

        now = utcnow()
        error = "The dispatcher lease expired before this job completed."
        with self.session_factory() as session:
            jobs = session.scalars(
                select(Job).where(
                    Job.queue_group == group,
                    Job.status == JobStatus.RUNNING.value,
                    Job.claim_owner.is_not(None),
                    Job.claim_expires_at.is_not(None),
                    Job.claim_expires_at < now,
                    ~Job.claim_owner.like(f"{self._owner}_%"),
                )
            ).all()
            expired_ids: list[str] = []
            for job in jobs:
                job.status = JobStatus.INTERRUPTED.value
                job.error = error
                job.completed_at = now
                job.claim_owner = None
                job.claim_expires_at = None
                job.heartbeat_at = None
                update_job_progress(
                    job,
                    stage="dispatcher lease expired",
                    queue_resource=job.queue_resource,
                    indeterminate=True,
                    now=now,
                )
                run = session.get(Run, job.run_id) if job.run_id else None
                if run:
                    run.status = RunStatus.FAILED.value
                    run.error = error
                    run.completed_at = now
                    step = session.get(WorkStep, run.work_step_id) if run.work_step_id else None
                    if step:
                        step.status = JobStatus.INTERRUPTED.value
                        step.error = error
                    plan = session.get(WorkPlan, run.work_plan_id) if run.work_plan_id else None
                    if plan:
                        session.flush()
                        refresh_plan_status(session, plan.id)
                        plan.summary_json = {
                            **plan.summary_json,
                            "status_counts": plan_status_summary(session, plan.id),
                        }
                    message = session.get(Message, run.assistant_message_id)
                    if message:
                        message.status = MessageStatus.FAILED.value
                        progress_parts = [
                            part for part in message.parts if part.type == PartType.PROGRESS.value
                        ]
                        for part in progress_parts:
                            message.parts.remove(part)
                        error_part = next(
                            (part for part in message.parts if part.type == PartType.ERROR.value),
                            None,
                        )
                        if error_part:
                            error_part.text = error
                        else:
                            message.parts.append(
                                MessagePart(
                                    position=max(
                                        (part.position for part in message.parts),
                                        default=-1,
                                    )
                                    + 1,
                                    type=PartType.ERROR.value,
                                    text=error,
                                )
                            )
                expired_ids.append(job.id)
            session.commit()
        return expired_ids

    async def _release_job(self, job_id: str, token: str, group: str) -> None:
        with self.session_factory() as session:
            session.execute(
                update(Job)
                .where(Job.id == job_id, Job.claim_owner == token)
                .values(
                    claim_owner=None,
                    claim_expires_at=None,
                    heartbeat_at=None,
                )
            )
            session.commit()
        self._queue_event(group).set()
        await self._publish_job(job_id)

    async def _publish_job(self, job_id: str) -> None:
        if not self._events:
            return
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            payload = JobOut.model_validate(job).model_dump(mode="json")
        await self._events.publish("job.progress", job_id, {"job": payload})

    async def publish_job(self, job_id: str) -> None:
        """Publish the latest persisted job snapshot to connected clients."""

        await self._publish_job(job_id)

    def _lock(self, group: str, capacity: int) -> asyncio.Semaphore:
        normalized_capacity = max(1, capacity)
        existing_capacity = self._capacities.get(group)
        if existing_capacity is not None and existing_capacity != normalized_capacity:
            raise ValueError(f"queue group {group!r} already uses capacity {existing_capacity}")
        self._capacities[group] = normalized_capacity
        return self._locks.setdefault(group, asyncio.Semaphore(normalized_capacity))

    def _queue_event(self, group: str) -> asyncio.Event:
        return self._queue_events.setdefault(group, asyncio.Event())
