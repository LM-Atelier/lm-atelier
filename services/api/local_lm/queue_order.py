"""Compare idle queue snapshots before changing an explicit owner order."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select, text, tuple_
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from .domain import utcnow
from .models import (
    Chat,
    GenerationQueuePolicy,
    Job,
    QueueOrderEntry,
    QueueOrderReceipt,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)
from .queue_control import JobQueueControl, job_controls
from .queue_order_v1 import (
    QueueLane,
    QueueOrderCommand,
    QueueOrderItemOut,
    QueueOrderNeighbours,
    QueueOrderOwner,
    QueueOrderPageOut,
    QueueOrderResultOut,
)

_KINDS: dict[QueueLane, tuple[str, ...]] = {
    "generation": ("chat", "image", "video", "edit_verify"),
    "transfer": ("download", "export"),
    "install": ("activate", "registry_prepare", "workflow_install"),
    "utility": ("media_utility",),
}
_TERMINAL = ("complete", "failed", "cancelled", "interrupted")
_LABELS = {
    "download": "Download",
    "export": "Export",
    "activate": "Model preparation",
    "registry_prepare": "Package preparation",
    "workflow_install": "Workflow installation",
    "media_utility": "Video utility",
}
_MAX_REVISION = 9_223_372_036_854_775_807
_MAX_JOBS = 10_000
AGING_SECONDS = 30
OwnerKey = tuple[Literal["work_plan", "job"], str]
CohortKey = tuple[str, str, int, int]


class QueueOrderConflict(Exception):
    """The lane or observed neighbourhood cannot accept this relative move."""


class QueueOrderLimit(QueueOrderConflict):
    """The current category exceeds the supported size for manual ordering."""

    def __init__(self, maximum_jobs: int) -> None:
        self.maximum_jobs = maximum_jobs
        super().__init__("Maximum queue ordering size exceeded.")


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def effective_priority(priority: int, enqueued: datetime, now: datetime) -> int:
    return priority + math.floor(
        max(0.0, (_utc(now) - _utc(enqueued)).total_seconds()) / AGING_SECONDS
    )


def manual_order_positions(
    session: Session, jobs: list[Job], controls: dict[str, JobQueueControl]
) -> dict[str, tuple[int, int]]:
    """Read current positions without scanning records retained for retries."""
    lookup_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for job in jobs:
        job_lane = next((lane for lane, kinds in _KINDS.items() if job.kind in kinds), None)
        if job_lane is None or job.kind == "edit_verify":
            continue
        owner = controls[job.id].owner_id
        lookup_ids[(job_lane, "work_plan" if owner else "job")].add(owner or job.id)
    if not lookup_ids:
        return {}
    entries: dict[tuple[str, str, str], QueueOrderEntry] = {}
    # Keep every primary-key prefix bound so retained history does not become
    # a table scan on each dispatch. Batch identifiers without dropping any.
    for (entry_lane, entry_type), identifiers in lookup_ids.items():
        ordered_ids = sorted(identifiers)
        for offset in range(0, len(ordered_ids), 500):
            for stored_entry in session.scalars(
                select(QueueOrderEntry).where(
                    QueueOrderEntry.lane == entry_lane,
                    QueueOrderEntry.owner_type == entry_type,
                    QueueOrderEntry.owner_id.in_(ordered_ids[offset : offset + 500]),
                )
            ):
                entries[(stored_entry.lane, stored_entry.owner_type, stored_entry.owner_id)] = (
                    stored_entry
                )
    result = {}
    for job in jobs:
        owner = controls[job.id].owner_id
        lane = next((lane for lane, kinds in _KINDS.items() if job.kind in kinds), None)
        if lane is None:
            continue
        entry = entries.get((lane, "work_plan" if owner else "job", owner or job.id))
        if (
            entry is not None
            and job.kind != "edit_verify"
            and entry.queue_group == job.queue_group
            and entry.queue_resource == job.queue_resource
            and entry.priority == job.queue_priority
        ):
            result[job.id] = (0, entry.position)
    return result


def apply_manual_order(
    session: Session,
    jobs: list[Job],
    controls: dict[str, JobQueueControl],
    now: datetime,
) -> list[Job]:
    """Permute existing slots only within one resource, lane and priority cohort."""
    positions = manual_order_positions(session, jobs, controls)
    slots: dict[tuple[str, str | None, str | None, int, int], list[int]] = defaultdict(list)
    for index, job in enumerate(jobs):
        lane = next((lane for lane, kinds in _KINDS.items() if job.kind in kinds), None)
        if lane is None or job.kind == "edit_verify":
            continue
        enqueued = _utc(job.enqueued_at or job.created_at)
        released = controls[job.id].eligible_since
        if released:
            enqueued = max(enqueued, _utc(released))
        key = (
            lane,
            job.queue_group,
            job.queue_resource,
            job.queue_priority,
            effective_priority(job.queue_priority, enqueued, now),
        )
        slots[key].append(index)
    result = list(jobs)
    for indices in slots.values():
        members = sorted(
            (jobs[index] for index in indices), key=lambda job: positions.get(job.id, (1, 0))
        )
        for index, job in zip(indices, members, strict=True):
            result[index] = job
    return result


@dataclass(frozen=True)
class _Job:
    id: str
    kind: str
    status: str
    work_plan_id: str | None
    work_step_id: str | None
    group: str | None
    resource: str | None
    priority: int
    enqueued: datetime | None
    created: datetime
    claim: str | None
    ticket: str | None


@dataclass
class _Snapshot:
    revision: int
    idle: bool
    items: list[QueueOrderItemOut]
    groups: dict[str, list[OwnerKey]]
    keys: dict[str, CohortKey]


def _key(owner: QueueOrderOwner) -> OwnerKey:
    return owner.type, owner.id


def _owner(key: OwnerKey) -> QueueOrderOwner:
    return QueueOrderOwner(type=key[0], id=key[1])


def _neighbours(ordered: list[OwnerKey], index: int) -> QueueOrderNeighbours:
    return QueueOrderNeighbours(
        before=_owner(ordered[index - 1]) if index else None,
        after=_owner(ordered[index + 1]) if index + 1 < len(ordered) else None,
    )


def _snapshot(session: Session, lane: QueueLane, now: datetime) -> _Snapshot:
    policy = session.get(GenerationQueuePolicy, lane)
    revision = policy.revision if policy else 0
    if (
        revision < 0
        or revision >= _MAX_REVISION
        or (policy and policy.dispatch_state not in {"open", "draining", "paused"})
    ):
        raise QueueOrderConflict
    idle = (
        session.scalar(
            select(Job.id).where(Job.kind.in_(_KINDS[lane]), Job.claim_owner.is_not(None)).limit(1)
        )
        is None
    )
    rows = session.execute(
        select(
            Job.id,
            Job.kind,
            Job.status,
            Job.work_plan_id,
            Job.work_step_id,
            Job.queue_group,
            Job.queue_resource,
            Job.queue_priority,
            Job.enqueued_at,
            Job.created_at,
            Job.claim_owner,
            Job.queue_ticket,
        )
        .where(Job.kind.in_(_KINDS[lane]), Job.status.not_in(_TERMINAL))
        .limit(_MAX_JOBS + 1)
    ).all()
    if len(rows) > _MAX_JOBS:
        raise QueueOrderLimit(_MAX_JOBS)
    jobs = [_Job(*row) for row in rows]
    controls = job_controls(session, [job.id for job in jobs])
    plan_ids = {job.work_plan_id for job in jobs if job.work_plan_id}
    visible_plans: dict[str, str] = (
        dict(
            session.execute(
                select(WorkPlan.id, Chat.title)
                .join(Chat, Chat.id == WorkPlan.chat_id)
                .where(
                    WorkPlan.id.in_(plan_ids),
                    WorkPlan.persistence_scope == "durable",
                    Chat.scope == "standard",
                )
            )
            .tuples()
            .all()
        )
        if plan_ids
        else {}
    )
    step_ids = {job.work_step_id for job in jobs if job.work_step_id}
    dependencies = (
        list(
            session.execute(
                select(WorkStepDependency.step_id, WorkStep.status)
                .outerjoin(WorkStep, WorkStep.id == WorkStepDependency.depends_on_step_id)
                .where(WorkStepDependency.step_id.in_(step_ids))
            )
        )
        if step_ids
        else []
    )
    blocked_steps = {step_id for step_id, status in dependencies if status != "complete"}
    owners: dict[OwnerKey, list[_Job]] = defaultdict(list)
    for job in jobs:
        if (
            job.kind == "edit_verify"
            or (lane == "generation" and job.work_plan_id is None)
            or (job.work_plan_id and job.work_plan_id not in visible_plans)
        ):
            continue
        owners[("work_plan", job.work_plan_id) if job.work_plan_id else ("job", job.id)].append(job)
    entries = {
        (entry.owner_type, entry.owner_id): entry
        for entry in session.scalars(
            select(QueueOrderEntry).where(
                QueueOrderEntry.lane == lane,
                tuple_(QueueOrderEntry.owner_type, QueueOrderEntry.owner_id).in_(owners),
            )
        )
    }
    cohorts: dict[CohortKey, list[tuple[OwnerKey, datetime, str, int | None]]] = defaultdict(list)
    unavailable = []
    ranks: dict[OwnerKey, tuple[str, int, datetime, str, str]] = {}
    for owner_key, members in owners.items():
        reason: Literal["lane-busy", "held", "blocked", "mixed-resources", "unsupported"] | None = (
            None
        )
        runnable = [
            job
            for job in members
            if job.status == "queued"
            and job.claim is None
            and job.work_step_id not in blocked_steps
            and controls[job.id].valid
            and controls[job.id].state == "eligible"
        ]
        if any(controls[job.id].state == "held" for job in members):
            reason = "held"
        elif not runnable:
            reason = "blocked"
        elif any(not job.group or not job.resource for job in runnable):
            reason = "unsupported"
        keys = {(job.group, job.resource, job.priority) for job in runnable}
        if reason is None and len(keys) != 1:
            reason = "mixed-resources"
        if reason is not None:
            unavailable.append(
                QueueOrderItemOut(
                    owner=_owner(owner_key),
                    label=visible_plans.get(owner_key[1])
                    or _LABELS.get(members[0].kind, "Submitted work"),
                    queued_at=min(_utc(job.enqueued or job.created) for job in members),
                    priority=None,
                    cohort_id=None,
                    position=None,
                    cohort_length=0,
                    neighbors=QueueOrderNeighbours(before=None, after=None),
                    unavailable_reason=reason,
                )
            )
            continue

        def eligible_time(job: _Job) -> datetime:
            released = controls[job.id].eligible_since
            enqueued = _utc(job.enqueued or job.created)
            return max(enqueued, _utc(released)) if released else enqueued

        first = min(
            runnable,
            key=lambda job: (
                -effective_priority(job.priority, eligible_time(job), now),
                eligible_time(job),
                job.ticket or job.id,
                job.id,
            ),
        )
        assert first.group is not None and first.resource is not None
        cohort_key = (
            first.group,
            first.resource,
            first.priority,
            effective_priority(first.priority, eligible_time(first), now),
        )
        ranks[owner_key] = (
            first.group,
            -cohort_key[3],
            eligible_time(first),
            first.ticket or first.id,
            first.id,
        )
        entry = entries.get(owner_key)
        position = (
            entry.position
            if entry is not None
            and (entry.queue_group, entry.queue_resource, entry.priority) == cohort_key[:3]
            else None
        )
        cohorts[cohort_key].append(
            (owner_key, eligible_time(first), first.ticket or first.id, position)
        )
    items = []
    groups = {}
    group_keys = {}
    membership = hashlib.sha256(
        json.dumps(
            [
                [job.id, job.status, job.claim, job.work_step_id, controls[job.id].revision]
                for job in sorted(jobs, key=lambda job: job.id)
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    for cohort_key, cohort_members in cohorts.items():
        cohort_members.sort(
            key=lambda item: (item[3] is None, item[3] or 0, item[1], item[2], item[0])
        )
        ordered = [item[0] for item in cohort_members]
        witness = [lane, revision, cohort_key, ordered, membership]
        cohort_id = hashlib.sha256(json.dumps(witness, separators=(",", ":")).encode()).hexdigest()
        groups[cohort_id] = ordered
        group_keys[cohort_id] = cohort_key

        for index, owner_key in enumerate(ordered):
            items.append(
                QueueOrderItemOut(
                    owner=_owner(owner_key),
                    label=visible_plans.get(owner_key[1])
                    or _LABELS.get(owners[owner_key][0].kind, "Submitted work"),
                    queued_at=min(_utc(job.enqueued or job.created) for job in owners[owner_key]),
                    priority=cohort_key[2],
                    cohort_id=cohort_id,
                    position=index + 1,
                    cohort_length=len(ordered),
                    neighbors=_neighbours(ordered, index),
                    before_neighbors=_neighbours(ordered, index - 1) if index else None,
                    after_neighbors=_neighbours(ordered, index + 1)
                    if index + 1 < len(ordered)
                    else None,
                    unavailable_reason=None if idle else "lane-busy",
                )
            )
    by_owner = {_key(item.owner): item for item in items}
    ranked = sorted(by_owner, key=ranks.__getitem__)
    slots: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(ranked):
        row_cohort_id = by_owner[key].cohort_id
        assert row_cohort_id is not None
        slots[row_cohort_id].append(index)
    for cohort_id, indices in slots.items():
        for index, key in zip(indices, groups[cohort_id], strict=True):
            ranked[index] = key
    return _Snapshot(
        revision, idle, [*[by_owner[key] for key in ranked], *unavailable], groups, group_keys
    )


def read_queue_order(
    session: Session, lane: QueueLane, *, limit: int = 50, cursor: str | None = None
) -> QueueOrderPageOut:
    # SQLite's legacy driver does not start a read transaction for SELECT.
    # Pin all constituent queries to one snapshot before reading the revision.
    connection = session.connection()
    if connection.dialect.name == "sqlite":
        driver = connection.connection.driver_connection
        if not bool(getattr(driver, "in_transaction", False)):
            connection.exec_driver_sql("BEGIN")
    snapshot = _snapshot(session, lane, utcnow())
    # Page cursors protect traversal order. Aging may refresh cohorts without
    # moving rows; move commands still compare the current cohort and neighbours.
    digest = hashlib.sha256(
        json.dumps(
            [[item.owner.type, item.owner.id, item.unavailable_reason] for item in snapshot.items]
        ).encode()
    ).hexdigest()
    offset = 0
    if cursor:
        try:
            value = json.loads(base64.urlsafe_b64decode(cursor.encode()))
            if (
                not isinstance(value, list)
                or len(value) != 3
                or value[:2] != [snapshot.revision, digest]
                or type(value[2]) is not int
            ):
                raise QueueOrderConflict
            offset = value[2]
            if not 0 <= offset < len(snapshot.items):
                raise QueueOrderConflict
        except (ValueError, TypeError, UnicodeError) as exc:
            raise QueueOrderConflict from exc
    next_offset = offset + limit
    next_cursor = (
        base64.urlsafe_b64encode(
            json.dumps([snapshot.revision, digest, next_offset]).encode()
        ).decode()
        if next_offset < len(snapshot.items)
        else None
    )
    return QueueOrderPageOut(
        lane=lane,
        revision=snapshot.revision,
        items=snapshot.items[offset:next_offset],
        total=len(snapshot.items),
        next_cursor=next_cursor,
    )


def change_queue_order(
    session: Session, lane: QueueLane, command: QueueOrderCommand
) -> QueueOrderResultOut:
    """Replay successful moves even after the work completes or the queue grows."""
    if session.in_transaction():
        raise QueueOrderConflict
    try:
        session.execute(text("BEGIN IMMEDIATE"))
        receipt = session.get(QueueOrderReceipt, (lane, command.idempotency_key))
        command_json = command.model_dump(mode="json")
        if receipt is not None:
            if receipt.command_json != command_json:
                raise QueueOrderConflict
            result = QueueOrderResultOut.model_validate(receipt.response_json)
            session.rollback()
            return result
        snapshot = _snapshot(session, lane, utcnow())
        if not snapshot.idle or snapshot.revision != command.expected_revision:
            raise QueueOrderConflict
        ordered = snapshot.groups.get(command.cohort_id)
        anchor = command.before or command.after
        assert anchor is not None
        item_key, anchor_key = _key(command.owner), _key(anchor)
        if ordered is None or item_key not in ordered or anchor_key not in ordered:
            raise QueueOrderConflict
        observed = {_key(item.owner): item for item in snapshot.items}
        if (
            observed[item_key].neighbors != command.expected_item_neighbors
            or observed[anchor_key].neighbors != command.expected_anchor_neighbors
        ):
            raise QueueOrderConflict
        ordered.remove(item_key)
        ordered.insert(ordered.index(anchor_key) + (command.after is not None), item_key)
        group, resource, priority, _aged = snapshot.keys[command.cohort_id]
        for position, key in enumerate(ordered):
            entry = session.get(QueueOrderEntry, (lane, *key))
            if entry is None:
                entry = QueueOrderEntry(lane=lane, owner_type=key[0], owner_id=key[1])
                session.add(entry)
            entry.queue_group, entry.queue_resource, entry.priority, entry.position = (
                group,
                resource,
                priority,
                position,
            )
        policy = session.get(GenerationQueuePolicy, lane)
        if policy is None:
            policy = GenerationQueuePolicy(lane=lane, dispatch_state="open")
            session.add(policy)
        policy.revision = snapshot.revision + 1
        session.flush()
        result = QueueOrderResultOut(lane=lane, revision=policy.revision, owner=command.owner)
        session.add(
            QueueOrderReceipt(
                lane=lane,
                command_key=command.idempotency_key,
                command_json=command_json,
                response_json=result.model_dump(mode="json"),
            )
        )
        session.commit()
        return result
    except OperationalError as exc:
        session.rollback()
        code = getattr(exc.orig, "sqlite_errorcode", None)
        if isinstance(code, int) and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise QueueOrderConflict from exc
        raise
    except BaseException:
        session.rollback()
        raise
