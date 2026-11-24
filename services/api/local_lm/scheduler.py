from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import and_, case, false, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from .db import SessionLocal, database_is_contended
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
from .power_inhibition import PowerInhibitor, SleepJobKind
from .progress import update_job_progress
from .queue_control import claim_control_predicate, job_controls
from .queue_lane_policy import (
    dispatch_allowed,
    lane_claim_predicate,
    lane_for_kind,
    queue_dispatches,
    reconcile_queue_lanes,
)
from .queue_order import apply_manual_order, effective_priority
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
#: How often a job its admission check holds back asks again. The job waits
#: queued and unclaimed, holding nothing, so a slow pace costs only how soon it
#: notices that what held it back has gone. It never costs the job its place: a
#: job behind it puts it back in line before starting.
_ADMISSION_RECHECK_SECONDS = 5.0

#: Asked before a queued job may claim its slot. None admits the job. A string
#: holds it back, still queued and unclaimed, and is the stage it shows while it
#: waits; the rest of its group goes ahead of it meanwhile. A raise ends the
#: wait with no claim made.
AdmissionCheck = Callable[[], Awaitable[str | None]]

logger = logging.getLogger(__name__)

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


# Durable work long enough that an idle sleep would cut it off. Chat turns and
# short disk work are not held.
_SLEEP_KINDS: dict[str, SleepJobKind] = {
    "media_compute": SleepJobKind.GENERATION,
    "network_transfer": SleepJobKind.DOWNLOAD,
    "primary_compute": SleepJobKind.RUNTIME_PREPARATION,
}


class ResourceScheduler:
    """Durable job tickets plus legacy leases for non-job administration."""

    def __init__(
        self,
        events: EventBroker | None = None,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        resource_pool: ResourceScheduler | None = None,
        power: PowerInhibitor | None = None,
    ) -> None:
        # Shared with the pool, like the slots, so one busy stretch holds one request.
        self._power: PowerInhibitor | None = (
            power if power is not None else (resource_pool._power if resource_pool else None)
        )
        self._locks: dict[str, asyncio.Semaphore] = resource_pool._locks if resource_pool else {}
        self._capacities: dict[str, int] = resource_pool._capacities if resource_pool else {}
        self._queue_events: dict[str, asyncio.Event] = {}
        #: Groups already reported as waiting on a held database.
        self._contended_groups: set[str] = set()
        self._eligibility: dict[str, tuple[float, tuple[str, ...]]] = {}
        self._owner = f"dispatcher_{secrets.token_hex(16)}"
        # Claims this dispatcher took whose release could not be written.
        # They belong to no live lease, so nothing else would ever reclaim
        # them: the expiry below skips this dispatcher's own claims,
        # because a live one is kept current by its heartbeat.
        self._abandoned_claims: set[str] = (
            resource_pool._abandoned_claims if resource_pool else set()
        )
        # Claims whose execution is still recording how it ended, counted by
        # each holder. The expiry below leaves these alone even once abandoned:
        # it would end the job as interrupted, and the outcome being recorded -
        # a failure, its retry or a stop - would then be refused.
        self._kept_claims: Counter[str] = resource_pool._kept_claims if resource_pool else Counter()
        # Jobs an admission check is holding back, by group, with the stage each
        # shows. Shared with the pool like the slots, so no dispatcher waits
        # behind a job that would not start if the slot came free.
        self._deferred: dict[str, dict[str, str]] = resource_pool._deferred if resource_pool else {}
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
        on_claim_lost: Callable[[], None] | None = None,
        admission: AdmissionCheck | None = None,
    ) -> AsyncIterator[JobClaim]:
        lock = self._lock(group, capacity)
        claim = await self._acquire_job(
            job_id,
            resource=resource,
            group=group,
            priority=priority,
            capacity=capacity,
            local_lock=lock,
            admission=admission,
        )
        renewal = (
            self._heartbeat(job_id, claim.token)
            if on_claim_lost is None
            else self._heartbeat(job_id, claim.token, on_claim_lost=on_claim_lost)
        )
        heartbeat = asyncio.create_task(
            renewal,
            name=f"job-heartbeat-{job_id}",
        )
        # Held only once the claim is real, never while the job waits its turn.
        # Held under the claim rather than the job: an older attempt that ends
        # after the job was claimed again must not end the newer attempt's hold.
        sleep_kind = _SLEEP_KINDS.get(resource)
        if self._power is not None and sleep_kind is not None:
            self._power.acquire(claim.token, sleep_kind)
        execution_failed = False
        try:
            # The claim identity is YIELDED so the execution can bind its
            # engine-provenance writes to the attempt it was claimed for; a
            # bare `async with` caller that ignores it is unchanged.
            yield claim
        except BaseException:
            execution_failed = True
            raise
        finally:
            # First, so the hold ends with the work even if releasing the claim fails.
            if self._power is not None:
                self._power.release(claim.token)
            heartbeat.cancel()
            stopped: BaseException | None = None
            try:
                with suppress(asyncio.CancelledError):
                    await heartbeat
            except BaseException as exc:
                # The heartbeat writes too, so it can fail for the same reason
                # the release can. Awaiting it here used to raise before the
                # release ran, which left the claim owned and the slot held.
                # Carry the failure and finish the exit first.
                stopped = exc
            released = False
            try:
                await self._release_job(job_id, claim.token, group)
                released = True
            except OperationalError as error:
                # A held database leaves the row naming this token, which the
                # next pass clears as an abandoned claim. The execution's own
                # outcome stands: a release that cannot write yet must not
                # replace it, least of all a success.
                if not database_is_contended(error):
                    raise
                logger.info(
                    "Job %s keeps its claim until another writer releases the database", job_id
                )
            finally:
                if not released:
                    # The row still names this token although the lease is
                    # over, whether the release raised or was cancelled. Record
                    # it so the next pass can clear it; otherwise the group
                    # counts a claim no execution holds and stops dispatching.
                    self._abandoned_claims.add(claim.token)
                # The slot is this process's own memory; the release is a
                # write. A database that refuses that write still has to let
                # the next job take its turn, or one failed row stops the whole
                # group until somebody restarts the application - which is what
                # a long writer, such as a startup sweep holding the database
                # past the busy timeout, makes likely rather than rare.
                lock.release()
            if stopped is not None:
                if not execution_failed:
                    raise stopped
                logger.warning(
                    "A scheduler heartbeat failed while the execution was already ending."
                )

    async def queue_control_changed(self, plan_id: str) -> None:
        self._eligibility.clear()
        for event in self._queue_events.values():
            event.set()
        if self._events is not None:
            await self._events.publish("queue.control", plan_id)

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

    def _note_contended_queue(self, group: str) -> None:
        """Say once per episode that the queue is waiting for the database.

        Once, because the wait repeats every poll for as long as the writer is
        held, and a line per poll would bury the reason it started. Not once per
        process: the flag is dropped again by the first pass that gets through,
        so a later sweep is reported as its own episode rather than passing in
        silence. Saying nothing at all would leave a queue that is merely
        waiting looking exactly like a queue that is stuck.
        """

        if group in self._contended_groups:
            return
        self._contended_groups.add(group)
        logger.info("Queue %s is waiting for another writer to release the database", group)

    async def _acquire_job(
        self,
        job_id: str,
        *,
        resource: str,
        group: str,
        priority: int,
        capacity: int,
        local_lock: asyncio.Semaphore,
        admission: AdmissionCheck | None = None,
    ) -> JobClaim:
        """Wait for the job's turn, then claim it once `admission` lets it start.

        A job its admission check holds back stands aside while it waits, and the
        rest of its group starts ahead of it. A job about to start first puts any
        job held back ahead of it back in line, to be asked again under the slot,
        because what held that job back may have gone since it last asked. However
        the wait ends - with a claim, a refusal or a cancellation - the job stops
        standing aside, so no job is ever passed over for a check that has stopped
        running.
        """

        try:
            return await self._wait_and_claim(
                job_id,
                resource=resource,
                group=group,
                priority=priority,
                capacity=capacity,
                local_lock=local_lock,
                admission=admission,
            )
        finally:
            self._drop_deferral(group, job_id)

    def _drop_deferral(self, group: str, job_id: str) -> None:
        deferred = self._deferred.get(group)
        if deferred is None:
            return
        deferred.pop(job_id, None)
        if not deferred:
            del self._deferred[group]

    async def _wait_and_claim(
        self,
        job_id: str,
        *,
        resource: str,
        group: str,
        priority: int,
        capacity: int,
        local_lock: asyncio.Semaphore,
        admission: AdmissionCheck | None,
    ) -> JobClaim:
        token = f"{self._owner}_{secrets.token_hex(12)}"
        # When a job its admission check holds back next asks again.
        next_admission = 0.0
        # Whether this job has put the jobs held back ahead of it back in line
        # since it began waiting or its own check last held it back. Once is
        # enough: a job held back again after that has answered since, and asking
        # it once more would only hold this job up behind the same answer.
        released_held_back = False
        while True:
            changed = False
            should_try_claim = False
            admission_due = False
            try:
                # Inside the try with everything else this pass writes: clearing
                # an old claim is a write, and a database another writer holds
                # is a reason to wait for the next pass, never to fail a job
                # that has not started.
                for expired_job_id in self._expire_foreign_claims(group):
                    await self._publish_job(expired_job_id)
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

                    eligible = self._eligible_job_ids(session, group, now)
                    deferred = self._deferred.get(group, {})
                    # A job its admission check holds back never takes a place,
                    # so the next job starts rather than waiting behind it.
                    candidates = tuple(
                        candidate for candidate in eligible if candidate not in deferred
                    )
                    deferred_stage = deferred.get(job.id)
                    position = next(
                        (
                            index
                            for index, candidate in enumerate(candidates)
                            if candidate == job.id
                        ),
                        None,
                    )
                    control = job_controls(session, [job.id])[job.id]
                    dispatches = queue_dispatches(session)
                    lane = lane_for_kind(job.kind)
                    if lane is not None and not dispatches[lane].open:
                        position = None
                        update_job_progress(
                            job,
                            stage=f"{lane} paused",
                            queue_resource=resource,
                            queue_position=None,
                            queue_length=len(candidates),
                            indeterminate=True,
                            now=now,
                        )
                    elif position is None and (not control.valid or control.state == "held"):
                        update_job_progress(
                            job,
                            stage="held" if control.state == "held" else "waiting for queue state",
                            queue_resource=resource,
                            queue_position=None,
                            queue_length=len(candidates),
                            indeterminate=True,
                            now=now,
                        )
                    elif deferred_stage is not None:
                        update_job_progress(
                            job,
                            stage=deferred_stage,
                            queue_resource=resource,
                            queue_position=None,
                            queue_length=len(candidates),
                            indeterminate=True,
                            now=now,
                        )
                        # Asked again unless a hold, a pause or an unfinished
                        # dependency keeps the job waiting, since no answer could
                        # start it then. Another job holding the slot does not
                        # stop the asking: the job stood aside for that job, and a
                        # check that bounds its own wait counts that time too.
                        admission_due = job.id in eligible and time.monotonic() >= next_admission
                    elif position is None:
                        failed_dependencies = self._failed_dependencies(
                            session,
                            job.work_step_id,
                        )
                        blocked_by = self._blocking_steps(session, job.work_step_id)
                        step = session.get(WorkStep, job.work_step_id) if job.work_step_id else None
                        plan = session.get(WorkPlan, job.work_plan_id) if job.work_plan_id else None
                        if step:
                            step.status = (
                                BLOCKED_WORK_STATUS
                                if failed_dependencies
                                else JobStatus.QUEUED.value
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
                        # Asked as late as it can be while the job is still
                        # unclaimed, so the answer is as fresh as it can be when
                        # the job starts, and holding the job back here undoes
                        # nothing: no attempt, heartbeat or keep-awake hold exists
                        # yet. A raise leaves through the release below.
                        waiting_stage = await admission() if admission is not None else None
                        if waiting_stage is not None:
                            self._deferred.setdefault(group, {})[job_id] = waiting_stage
                            next_admission = time.monotonic() + _ADMISSION_RECHECK_SECONDS
                            released_held_back = False
                            # The job no longer stands in front of the rest of
                            # its group, so they need not wait for their next poll.
                            self._queue_event(group).set()
                            # Looked at again at once, so it shows what it waits
                            # for before a job behind it can start: the pass that
                            # brought it here showed it next in line.
                            continue
                        else:
                            with self.session_factory() as session:
                                current = session.get(Job, job_id)
                                control_snapshot = job_controls(session, [job_id]).get(job_id)
                                dispatch_snapshots = queue_dispatches(session)
                                claimed_at = utcnow()
                                # Fresh, never shared: this decides whether the job
                                # STARTS. The final update also compares the control
                                # revision captured before ranking, including Hold/Release.
                                ranked = self._fresh_eligible_job_ids(session, group, claimed_at)
                                deferred = self._deferred.get(group, {})
                                candidates = tuple(
                                    candidate for candidate in ranked if candidate not in deferred
                                )
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
                                            Job.claim_owner.is_not(None),
                                        )
                                    )
                                    or 0
                                )
                                its_turn = position is not None and position < max(
                                    0, capacity - active_claims
                                )
                                # A job held back ahead of this one may be free to
                                # start too: what held it back may have gone since
                                # it last asked, and it asks again only at its own
                                # pace. Back in line, it is asked first, under the
                                # slot, and starts if its check lets it; if the check
                                # holds it back again, this job starts next time.
                                held_back_ahead = (
                                    tuple(
                                        candidate
                                        for candidate in ranked[: ranked.index(job_id)]
                                        if candidate in deferred
                                    )
                                    if its_turn and not released_held_back
                                    else ()
                                )
                                if held_back_ahead:
                                    session.rollback()
                                    for held_back in held_back_ahead:
                                        self._drop_deferral(group, held_back)
                                    released_held_back = True
                                    self._queue_event(group).set()
                                elif its_turn and control_snapshot is not None:
                                    result = cast(
                                        CursorResult[Any],
                                        session.execute(
                                            update(Job)
                                            .where(
                                                Job.id == job_id,
                                                Job.status == JobStatus.QUEUED.value,
                                                Job.claim_owner.is_(None),
                                                claim_control_predicate(control_snapshot),
                                                *[
                                                    lane_claim_predicate(snapshot)
                                                    for snapshot in dispatch_snapshots.values()
                                                ],
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
                                            .execution_options(synchronize_session="fetch")
                                        ),
                                    )
                                    if result.rowcount == 1:
                                        claimed_job = session.get(Job, job_id)
                                        if claimed_job:
                                            # The ORM-enabled UPDATE synchronizes the
                                            # identity map using the returned row, so the
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
            except OperationalError as error:
                if not database_is_contended(error):
                    raise
                # Another writer held the database past the busy timeout:
                # the startup retention sweep does exactly this, batch after
                # batch. For a job that is only waiting its turn that is a
                # reason to wait longer, not a reason to fail, and this loop
                # already knows how to wait. Nothing above was committed, so
                # the next pass reads the queue again rather than trusting
                # anything measured against a transaction that rolled back.
                self._note_contended_queue(group)
            else:
                self._contended_groups.discard(group)
                if changed:
                    await self._publish_job(job_id)
                if admission is not None and admission_due:
                    # Outside any session and without the slot, so the check
                    # holds up neither the database nor the rest of the group.
                    waiting_stage = await admission()
                    if waiting_stage is None:
                        self._drop_deferral(group, job_id)
                        # Back in line now: the group recounts its places at
                        # once, and this job looks for its turn without waiting
                        # out a poll.
                        self._queue_event(group).set()
                        continue
                    self._deferred.setdefault(group, {})[job_id] = waiting_stage
                    next_admission = time.monotonic() + _ADMISSION_RECHECK_SECONDS
                    released_held_back = False
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
        controls = job_controls(session, [job.id for job in jobs])
        dispatches = queue_dispatches(session)
        blocked = {
            job.id for job in jobs if ResourceScheduler._blocking_steps(session, job.work_step_id)
        }

        def rank(job: Job) -> tuple[int, int, datetime, str, str]:
            enqueued = job.enqueued_at or job.created_at
            if enqueued.tzinfo is None:
                enqueued = enqueued.replace(tzinfo=UTC)
            released = controls[job.id].eligible_since
            if released is not None:
                enqueued = max(enqueued, released)
            # Verification is best-effort background work. Queue aging may
            # reorder foreground jobs, but can never promote a check ahead of
            # a user-requested generation.
            background = job.kind == JobKind.EDIT_VERIFY.value
            priority = (
                job.queue_priority
                if background
                else effective_priority(job.queue_priority, enqueued, now)
            )
            return (
                1 if background else 0,
                -priority,
                enqueued,
                job.queue_ticket or job.id,
                job.id,
            )

        ranked = sorted(
            (
                job
                for job in jobs
                if job.id not in blocked
                and controls[job.id].valid
                and controls[job.id].state == "eligible"
                and dispatch_allowed(job.kind, dispatches)
            ),
            key=rank,
        )
        return apply_manual_order(session, ranked, controls, now)

    def peek_next_eligible_job(self, group: str) -> tuple[str, str | None] | None:
        """Return the next durable job without claiming or changing it.

        A job its admission check is holding back is never next: it would not
        start if the slot came free now.
        """

        deferred = self._deferred.get(group, {})
        with self.session_factory() as session:
            job = next(
                (
                    candidate
                    for candidate in self._eligible_jobs(session, group, utcnow())
                    if candidate.id not in deferred
                ),
                None,
            )
            if job is None:
                return None
            return job.id, job.run_id

    def startable_jobs(self, job_ids: Iterable[str]) -> frozenset[str]:
        """Which of these jobs would start now if their turn came.

        A job would start once it is queued and unclaimed, not held, in an open
        lane, and every job it depends on has finished: what makes it eligible
        in its own group. Its place in line is left out, and so is its own
        admission check: a job that check holds back is asked again whenever it
        could start, and starts once the check lets it. A job that is held,
        whose lane is paused or that waits for a job it depends on is not asked
        at all, and starts only once someone releases it or that job finishes.
        """

        wanted = set(job_ids)
        if not wanted:
            return frozenset()
        with self.session_factory() as session:
            groups = {
                group
                for group in session.scalars(select(Job.queue_group).where(Job.id.in_(wanted)))
                if group is not None
            }
            now = utcnow()
            return frozenset(
                job.id
                for group in groups
                for job in self._eligible_jobs(session, group, now)
                if job.id in wanted
            )

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

    async def _heartbeat(
        self, job_id: str, token: str, *, on_claim_lost: Callable[[], None] | None = None
    ) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            try:
                with self.session_factory() as session:
                    now = utcnow()
                    result = cast(
                        CursorResult[Any],
                        session.execute(
                            update(Job)
                            .where(
                                Job.id == job_id,
                                Job.claim_owner == token,
                                # A result or retry may precede a slow handoff.
                                # Keep its ownership live until job_lease releases it.
                                Job.status.in_(
                                    [
                                        JobStatus.QUEUED.value,
                                        JobStatus.RUNNING.value,
                                        *_TERMINAL_STATUSES,
                                    ]
                                ),
                            )
                            .values(
                                heartbeat_at=now,
                                claim_expires_at=now + timedelta(seconds=_CLAIM_SECONDS),
                            )
                        ),
                    )
                    session.commit()
                    if result.rowcount != 1:
                        owner = session.execute(
                            select(Job.claim_owner).where(Job.id == job_id)
                        ).scalar_one_or_none()
                        if owner != token:
                            if on_claim_lost is not None:
                                on_claim_lost()
                            if self._power is not None:
                                # The claim moved to another attempt or was cleared, so
                                # this one no longer keeps the computer awake, however
                                # long its own work takes to stop.
                                self._power.release(token)
                        return
            except OperationalError as error:
                if not database_is_contended(error):
                    raise

    def _expire_foreign_claims(self, group: str) -> list[str]:
        """Interrupt abandoned work without risking a duplicate backend request.

        A claim of another dispatcher counts as abandoned once it stops being
        renewed. One of this dispatcher's own counts as abandoned only when its
        lease ended and the release could not be written, because a live claim
        here is kept current by its heartbeat rather than by its expiry time -
        and not while its execution still keeps it to record how it ended.
        """

        now = utcnow()
        error = "The dispatcher lease expired before this job completed."
        abandoned = sorted(
            claim for claim in self._abandoned_claims if claim not in self._kept_claims
        )
        with self.session_factory() as session:
            jobs = session.scalars(
                select(Job).where(
                    Job.queue_group == group,
                    # A paused job keeps a claim its lease could not clear, from
                    # this dispatcher's release or a dead one's, and nothing else
                    # clears it: the group would count it as running for good.
                    Job.status.in_(
                        [
                            JobStatus.QUEUED.value,
                            JobStatus.RUNNING.value,
                            JobStatus.PAUSED.value,
                            *_TERMINAL_STATUSES,
                        ]
                    ),
                    Job.claim_owner.is_not(None),
                    or_(
                        and_(
                            Job.claim_expires_at.is_not(None),
                            Job.claim_expires_at < now,
                            ~Job.claim_owner.like(f"{self._owner}_%"),
                        ),
                        Job.claim_owner.in_(abandoned) if abandoned else false(),
                    ),
                )
            ).all()
            expired_ids: list[str] = []
            cleared_tokens: list[str] = []
            for job in jobs:
                if job.claim_owner:
                    cleared_tokens.append(job.claim_owner)
                if job.status in _TERMINAL_STATUSES or job.status == JobStatus.PAUSED.value:
                    # The result is settled, or the job waits for someone to
                    # resume it; only an abandoned handoff remains.
                    job.claim_owner = None
                    job.claim_expires_at = None
                    job.heartbeat_at = None
                    self._invalidate_eligibility(group)
                    expired_ids.append(job.id)
                    continue
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
            if expired_ids:
                session.flush()
                reconcile_queue_lanes(session)
            session.commit()
        # Only now, because the flush, the reconciliation and the commit can
        # each fail the same way the release did. Forgetting a token before its
        # row is actually cleared would strand that claim for good: the row
        # still names this dispatcher, which the expiry above otherwise skips.
        for token in cleared_tokens:
            self._abandoned_claims.discard(token)
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
                    queue_group=case(
                        (
                            and_(
                                Job.kind == JobKind.WORKFLOW_INSTALL.value,
                                Job.status == JobStatus.QUEUED.value,
                            ),
                            None,
                        ),
                        else_=Job.queue_group,
                    ),
                )
            )
            reconcile_queue_lanes(session)
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

    def keep_claim(self, token: str) -> None:
        """Leave this claim to its execution while that execution records how it ended.

        Once the claim's release could not be written, the next pass would end
        the job as interrupted, and the outcome the execution is still trying
        to record - its failure, the retry that failure queues, or its stop -
        would then be refused. Each call needs its own `let_claim_go`.
        """

        self._kept_claims[token] += 1

    def let_claim_go(self, token: str) -> None:
        """End one `keep_claim`; once none is left, an abandoned claim is cleared as usual."""

        self._kept_claims[token] -= 1
        if self._kept_claims[token] <= 0:
            del self._kept_claims[token]

    def _lock(self, group: str, capacity: int) -> asyncio.Semaphore:
        normalized_capacity = max(1, capacity)
        existing_capacity = self._capacities.get(group)
        if existing_capacity is not None and existing_capacity != normalized_capacity:
            raise ValueError(f"queue group {group!r} already uses capacity {existing_capacity}")
        self._capacities[group] = normalized_capacity
        return self._locks.setdefault(group, asyncio.Semaphore(normalized_capacity))

    def _queue_event(self, group: str) -> asyncio.Event:
        return self._queue_events.setdefault(group, asyncio.Event())
