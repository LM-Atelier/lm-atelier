from __future__ import annotations

import asyncio
import math
import secrets
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from .db import SessionLocal
from .domain import JobKind, JobStatus, MessageStatus, PartType, RunStatus, utcnow
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
#: How long one eligibility scan may be shared between waiters, a quarter of the
#: poll interval. This bounds RESIDENCY, meaning how long a completed answer may
#: go on being served. It does not bound the AGE of that answer, which is the
#: scan duration plus the residency and has no poll-interval bound at all.
_ELIGIBILITY_SHARE_SECONDS = _QUEUE_POLL_SECONDS / 4
_AGING_SECONDS = 30
_TERMINAL_STATUSES = {
    JobStatus.COMPLETE.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
    JobStatus.INTERRUPTED.value,
}


@dataclass(frozen=True)
class JobClaim:
    """The identity a claim mints: release token plus the claimed attempt.

    An execution that produces engine-provenance writes must present this
    identity, captured AT CLAIM TIME, on every such write. The Job row is
    mutable - the scheduler can expire a foreign claim and a new claimant
    then increments the row's attempt while an old backend request is still
    alive - so a writer that re-reads the row labels a stale producer's late
    event with the NEW attempt, refreshing liveness the new engine never
    produced. The token names the exact claim; the attempt is what the
    provenance stamp binds to.
    """

    token: str
    attempt: int


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
        self._eligibility: dict[str, tuple[float, tuple[str, ...]]] = {}
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
    ) -> AsyncIterator[JobClaim]:
        lock = self._lock(group, capacity)
        claim = await self._acquire_job(
            job_id,
            resource=resource,
            group=group,
            priority=priority,
            capacity=capacity,
            local_lock=lock,
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(job_id, claim.token),
            name=f"job-heartbeat-{job_id}",
        )
        try:
            # The claim identity is YIELDED so the execution can bind its
            # engine-provenance writes to the attempt it was claimed for; a
            # bare `async with` caller that ignores it is unchanged.
            yield claim
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            await self._release_job(job_id, claim.token, group)
            lock.release()

    def _invalidate_eligibility(self, group: str) -> None:
        """Drop the shared scan because the queue actually changed."""

        self._eligibility.pop(group, None)

    def _eligible_job_ids(self, session: Session, group: str, now: datetime) -> tuple[str, ...]:
        """Ordered eligible job ids, shared by every waiter in one poll window.

        One task per QUEUED job calls this on every pass, and the underlying scan
        inspects each queued job's dependencies, so an unshared epoch costs N * N
        dependency inspections at depth N. Waiters wake together, so sharing one
        scan across a window makes an epoch cost N instead.

        The share is bounded by TIME, not by an invalidation the callers must
        remember. A missed invalidation would otherwise pin a stale answer and
        strand a queue forever.

        What the window bounds is RESIDENCY, not the age of the answer. The ids
        are computed before the completion stamp, so a waiter can act on a
        snapshot as old as one scan plus one window, not one window. A slow scan
        makes that answer older without making it unshared, and nothing here
        bounds it against the poll interval; the slow-scan test below permits
        exactly that. It is tolerable only because this answer reports progress.
        `_fresh_eligible_job_ids` is what authorizes a start. Real transitions -
        a claim, a release, an expiry - drop the share immediately, so a freed
        slot is never waited on.

        Only ids are shared. The rows themselves belong to the caller's session
        and must not outlive it.
        """

        cached = self._eligibility.get(group)
        if cached is not None and time.monotonic() - cached[0] < _ELIGIBILITY_SHARE_SECONDS:
            return cached[1]
        return self._fresh_eligible_job_ids(session, group, now)

    def _fresh_eligible_job_ids(
        self, session: Session, group: str, now: datetime
    ) -> tuple[str, ...]:
        """Scan now, ignoring any share, and publish the result as the new share.

        Required wherever the answer AUTHORIZES A STATE TRANSITION rather than
        describing progress. The share is bounded by time rather than by
        invalidation, so a shared answer can be one scan plus one window old -
        long enough for a dependency to stop being complete, or for a
        higher-priority job to be enqueued ahead of this one. The claim
        statement guards only that the row is still QUEUED and unclaimed, so it
        catches neither. A waiter may render a slightly old position; a job may
        not START on one.

        One scan per claim is O(N) once per job rather than once per poll, and
        it authorizes compute lasting seconds to minutes, so it does not
        reintroduce the cost the share exists to remove.
        """

        ids = tuple(job.id for job in self._eligible_jobs(session, group, now))
        # Stamp when the scan COMPLETED, not when it started. Stamping the start
        # publishes an entry that is already stale by however long the scan took,
        # so a scan costing more than the share window would expire before the
        # next waiter could use it - and every waiter would rescan, restoring the
        # quadratic exactly where the sharing is most needed.
        self._eligibility[group] = (time.monotonic(), ids)
        return ids

    async def _acquire_job(
        self,
        job_id: str,
        *,
        resource: str,
        group: str,
        priority: int,
        capacity: int,
        local_lock: asyncio.Semaphore,
    ) -> JobClaim:
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

                candidates = self._eligible_job_ids(session, group, now)
                position = next(
                    (index for index, candidate in enumerate(candidates) if candidate == job.id),
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
                claimed_attempt = 0
                try:
                    with self.session_factory() as session:
                        current = session.get(Job, job_id)
                        claimed_at = utcnow()
                        # Fresh, never shared: this decides whether the job
                        # STARTS, and the update below guards only QUEUED and
                        # unclaimed.
                        candidates = self._fresh_eligible_job_ids(session, group, claimed_at)
                        position = next(
                            (
                                index
                                for index, candidate in enumerate(candidates)
                                if current and candidate == current.id
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
                                    # The ORM-enabled UPDATE synchronizes the
                                    # identity map (evaluate strategy), so the
                                    # cached row already shows the incremented
                                    # attempt; a refresh here would re-read
                                    # what the session already holds.
                                    claimed_attempt = claimed_job.attempt
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
                                self._invalidate_eligibility(group)
                            else:
                                session.rollback()
                finally:
                    if not claimed:
                        local_lock.release()
                if claimed:
                    await self._publish_job(job_id)
                    return JobClaim(token=token, attempt=claimed_attempt)

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

        def rank(job: Job) -> tuple[int, int, datetime, str, str]:
            enqueued = job.enqueued_at or job.created_at
            if enqueued.tzinfo is None:
                enqueued = enqueued.replace(tzinfo=UTC)
            # Verification is best-effort background work. Queue aging may
            # reorder foreground jobs, but can never promote a check ahead of
            # a user-requested generation.
            background = job.kind == JobKind.EDIT_VERIFY.value
            waited = max(0.0, (now - enqueued).total_seconds())
            effective_priority = (
                job.queue_priority
                if background
                else job.queue_priority + math.floor(waited / _AGING_SECONDS)
            )
            return (
                1 if background else 0,
                -effective_priority,
                enqueued,
                job.queue_ticket or job.id,
                job.id,
            )

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
                self._invalidate_eligibility(group)
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
                        # The error part takes the position the progress
                        # part vacated, and the unit of work would otherwise
                        # INSERT it before DELETING the old row: the
                        # (message, position) uniqueness then refuses the
                        # interruption itself. Flushing the removals first
                        # keeps the expiry a single-order write.
                        session.flush()
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
                elif job.work_step_id:
                    step = session.get(WorkStep, job.work_step_id)
                    if step:
                        step.status = JobStatus.INTERRUPTED.value
                        step.error = error
                        session.flush()
                        refresh_plan_status(session, step.plan_id)
                        plan = session.get(WorkPlan, step.plan_id)
                        if plan:
                            plan.summary_json = {
                                **plan.summary_json,
                                "status_counts": plan_status_summary(session, plan.id),
                            }
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
        self._invalidate_eligibility(group)
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
