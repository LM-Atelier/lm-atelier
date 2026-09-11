"""Bounded, scalar-only view of accepted work; never grants dispatch authority."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
from datetime import UTC, datetime
from typing import Literal, cast

from sqlalchemy import case, exists, func, literal, or_, select, tuple_, union_all
from sqlalchemy.orm import Session, aliased

from .domain import JobStatus
from .models import Chat, Job, WorkPlan, WorkStep, WorkStepDependency
from .schemas import (
    QueueActivityItemOut,
    QueueActivityPageOut,
    QueueLaneCountsOut,
    QueuePlanStepsOut,
    QueueStepOut,
    WorkStepStatus,
)

Lane = Literal["generation", "transfer", "install"]
Key = tuple[datetime, str, str]
_ACTIVE = ("queued", "running", "paused")
_PLAN_ACTIVE = (*_ACTIVE, "blocked")
_VISIBLE = ("chat", "image", "video", "download", "export", "activate", "registry_prepare")
_LABELS = {
    "chat": "Chat generation",
    "image": "Image generation",
    "video": "Video generation",
    "download": "Download",
    "export": "Export",
    "activate": "Model preparation",
    "registry_prepare": "Package preparation",
}


class QueueActivityCursorError(ValueError):
    def __init__(self) -> None:
        super().__init__("The accepted work page request is invalid.")


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _key(value: object) -> Key:
    if not isinstance(value, list) or len(value) != 3:
        raise QueueActivityCursorError()
    stamp, owner_type, owner_id = value
    if (
        not isinstance(stamp, str)
        or owner_type not in ("job", "work_plan")
        or not isinstance(owner_id, str)
        or not 1 <= len(owner_id) <= 40
    ):
        raise QueueActivityCursorError()
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is not None:
        raise QueueActivityCursorError()
    return parsed, cast(str, owner_type), owner_id


def _decode(cursor: str, key: bytes, lane: Lane | None) -> tuple[Key, Key]:
    try:
        if len(cursor) > 2048:
            raise QueueActivityCursorError()
        encoded, signature = cursor.split(".")
        payload = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
        canonical = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
        if canonical != encoded or not hmac.compare_digest(
            signature, hmac.new(key, payload, hashlib.sha256).hexdigest()
        ):
            raise QueueActivityCursorError()
        value = json.loads(payload)
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "lane", "anchor", "after"}
            or value["version"] != 1
            or value["lane"] != lane
        ):
            raise QueueActivityCursorError()
        anchor, after = _key(value["anchor"]), _key(value["after"])
        if after > anchor:
            raise QueueActivityCursorError()
        return anchor, after
    except (ValueError, TypeError, UnicodeError, binascii.Error, OverflowError) as exc:
        raise QueueActivityCursorError() from exc


def _encode(anchor: Key, after: Key, key: bytes, lane: Lane | None) -> str:
    def pack(value: Key) -> list[str]:
        return [value[0].isoformat(), value[1], value[2]]

    payload = json.dumps(
        {"version": 1, "lane": lane, "anchor": pack(anchor), "after": pack(after)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    return encoded + "." + hmac.new(key, payload, hashlib.sha256).hexdigest()


def _begin_read_snapshot(session: Session) -> None:
    connection = session.connection()
    if connection.dialect.name == "sqlite":
        driver = connection.connection.driver_connection
        if not bool(getattr(driver, "in_transaction", False)):
            connection.exec_driver_sql("BEGIN")


def list_queue_activity(
    session: Session,
    *,
    signing_key: bytes,
    limit: int = 50,
    lane: Lane | None = None,
    cursor: str | None = None,
) -> QueueActivityPageOut:
    """Read one acceptance-ordered page without hydrating jobs or plan payloads."""
    if not 1 <= limit <= 100 or lane not in (None, "generation", "transfer", "install"):
        raise ValueError("Invalid accepted work query.")
    decoded = _decode(cursor, signing_key, lane) if cursor is not None else None
    _begin_read_snapshot(session)
    with session.no_autoflush:
        counts = (
            select(
                Job.work_plan_id.label("plan_id"),
                func.count().label("active_jobs"),
                func.sum(case((Job.status == "running", 1), else_=0)).label("running_jobs"),
                func.sum(case((Job.status == "queued", 1), else_=0)).label("queued_jobs"),
                func.sum(case((Job.status == "paused", 1), else_=0)).label("paused_jobs"),
                func.max(Job.updated_at).label("updated_at"),
            )
            .where(Job.status.in_(_ACTIVE), Job.kind.in_(_VISIBLE), Job.work_plan_id.is_not(None))
            .group_by(Job.work_plan_id)
            .cte("queue_job_counts")
        )
        public_chat = (WorkPlan.persistence_scope == "durable") & (Chat.scope == "standard")
        plans = (
            select(
                literal("work_plan").label("owner_type"),
                WorkPlan.id.label("owner_id"),
                WorkPlan.created_at.label("created_at"),
                case(
                    (counts.c.updated_at > WorkPlan.updated_at, counts.c.updated_at),
                    else_=WorkPlan.updated_at,
                ).label("updated_at"),
                literal("generation").label("lane"),
                WorkPlan.status.label("status"),
                literal("Submitted work").label("label"),
                case((public_chat, Chat.id), else_=None).label("chat_id"),
                case((public_chat, Chat.title), else_=None).label("chat_title"),
                literal(None).label("progress"),
                *[
                    func.coalesce(counts.c[name], 0).label(name)
                    for name in ("active_jobs", "running_jobs", "queued_jobs", "paused_jobs")
                ],
            )
            .outerjoin(counts, counts.c.plan_id == WorkPlan.id)
            .outerjoin(Chat, Chat.id == WorkPlan.chat_id)
            .where(or_(WorkPlan.status.in_(_PLAN_ACTIVE), counts.c.active_jobs > 0))
        )
        standalone = select(
            literal("job").label("owner_type"),
            Job.id.label("owner_id"),
            Job.created_at.label("created_at"),
            Job.updated_at.label("updated_at"),
            case(
                (Job.kind.in_(("download", "export")), "transfer"),
                (Job.kind.in_(("activate", "registry_prepare")), "install"),
                else_="generation",
            ).label("lane"),
            Job.status.label("status"),
            case(_LABELS, value=Job.kind, else_="Work").label("label"),
            literal(None).label("chat_id"),
            literal(None).label("chat_title"),
            Job.progress.label("progress"),
            literal(1).label("active_jobs"),
            case((Job.status == "running", 1), else_=0).label("running_jobs"),
            case((Job.status == "queued", 1), else_=0).label("queued_jobs"),
            case((Job.status == "paused", 1), else_=0).label("paused_jobs"),
        ).where(
            Job.status.in_(_ACTIVE),
            Job.kind.in_(_VISIBLE),
            or_(
                Job.work_plan_id.is_(None),
                ~exists(select(WorkPlan.id).where(WorkPlan.id == Job.work_plan_id)),
            ),
        )
        owners = union_all(plans, standalone).cte("queue_owners")
        ordering = (owners.c.created_at, owners.c.owner_type, owners.c.owner_id)
        if decoded is None:
            newest = session.execute(
                select(*ordering).order_by(*(column.desc() for column in ordering)).limit(1)
            ).first()
            if newest is None:
                return QueueActivityPageOut(
                    items=[],
                    total=0,
                    lane_counts=QueueLaneCountsOut(),
                    next_cursor=None,
                    observed_at=datetime.now(UTC),
                )
            anchor: Key = (newest[0], newest[1], newest[2])
            after = None
        else:
            anchor, after = decoded
        within = tuple_(*ordering) <= anchor
        lane_counts = QueueLaneCountsOut(
            **{
                row[0]: row[1]
                for row in session.execute(
                    select(owners.c.lane, func.count()).where(within).group_by(owners.c.lane)
                )
            }
        )
        total = (
            getattr(lane_counts, lane)
            if lane is not None
            else lane_counts.generation + lane_counts.transfer + lane_counts.install
        )
        query = select(owners).where(within)
        if lane is not None:
            query = query.where(owners.c.lane == lane)
        if after is not None:
            query = query.where(tuple_(*ordering) > after)
        page = session.execute(query.order_by(*ordering).limit(limit + 1)).all()
        selected = page[:limit]
        plan_ids = [row.owner_id for row in selected if row.owner_type == "work_plan"]
        step_counts: dict[str, tuple[int, int, int]] = {}
        if plan_ids:
            dependency = aliased(WorkStep)
            blocked = exists(
                select(WorkStepDependency.step_id)
                .outerjoin(dependency, dependency.id == WorkStepDependency.depends_on_step_id)
                .where(
                    WorkStepDependency.step_id == WorkStep.id,
                    or_(dependency.id.is_(None), dependency.status != "complete"),
                )
            )
            for row in session.execute(
                select(
                    WorkStep.plan_id,
                    func.count(),
                    func.sum(case((WorkStep.status == "complete", 1), else_=0)),
                    func.sum(
                        case(
                            (
                                or_(
                                    WorkStep.status == "blocked",
                                    (WorkStep.status == "queued") & blocked,
                                ),
                                1,
                            ),
                            else_=0,
                        )
                    ),
                )
                .where(WorkStep.plan_id.in_(plan_ids))
                .group_by(WorkStep.plan_id)
            ):
                step_counts[row[0]] = row[1], row[2], row[3]
        items: list[QueueActivityItemOut] = []
        for row in selected:
            steps, completed, blocked_count = step_counts.get(row.owner_id, (0, 0, 0))
            status = (
                "running"
                if row.running_jobs
                else "queued"
                if row.queued_jobs
                else "paused"
                if row.paused_jobs
                else row.status
            )
            if status == "queued" and steps > completed and blocked_count == steps - completed:
                status = "blocked"
            progress = row.progress
            if progress is not None:
                progress = max(0.0, min(1.0, progress)) if math.isfinite(progress) else None
            items.append(
                QueueActivityItemOut(
                    owner_type=row.owner_type,
                    owner_id=row.owner_id,
                    label=row.label,
                    lane=row.lane,
                    status=status,
                    chat_id=row.chat_id,
                    chat_title=row.chat_title,
                    created_at=_utc(row.created_at),
                    updated_at=_utc(row.updated_at),
                    step_count=steps,
                    completed_steps=completed,
                    blocked_steps=blocked_count,
                    active_jobs=row.active_jobs,
                    running_jobs=row.running_jobs,
                    queued_jobs=row.queued_jobs,
                    paused_jobs=row.paused_jobs,
                    progress=progress,
                )
            )
        next_cursor = None
        if len(page) > limit:
            last = selected[-1]
            next_cursor = _encode(
                anchor, (last.created_at, last.owner_type, last.owner_id), signing_key, lane
            )
        return QueueActivityPageOut(
            items=items,
            total=total,
            lane_counts=lane_counts,
            next_cursor=next_cursor,
            observed_at=datetime.now(UTC),
        )


class QueueStepStateError(ValueError):
    def __init__(self) -> None:
        super().__init__("Stored work step state is invalid.")


def list_queue_plan_steps(
    session: Session,
    plan_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> QueuePlanStepsOut | None:
    """Page immutable accepted step order; return only safe progress metadata."""
    if not 1 <= limit <= 100 or not 0 <= offset <= 2**63 - 1:
        raise ValueError("Invalid work step page.")
    _begin_read_snapshot(session)
    with session.no_autoflush:
        if session.scalar(select(WorkPlan.id).where(WorkPlan.id == plan_id)) is None:
            return None
        total = int(
            session.scalar(select(func.count(WorkStep.id)).where(WorkStep.plan_id == plan_id)) or 0
        )
        dependency = aliased(WorkStep)
        unfinished = (
            select(func.count(WorkStepDependency.step_id))
            .outerjoin(dependency, dependency.id == WorkStepDependency.depends_on_step_id)
            .where(
                WorkStepDependency.step_id == WorkStep.id,
                or_(dependency.id.is_(None), dependency.status != "complete"),
            )
            .correlate(WorkStep)
            .scalar_subquery()
        )

        # SQLite JSON extraction turns booleans into integers. Check JSON types
        # before using values, and never fall back to the legacy monotonic counter.
        overall_type = func.json_type(Job.progress_json, "$.overall_progress")
        use_stage = or_(overall_type.is_(None), overall_type == "null")
        reported = case(
            (use_stage, Job.progress_json["stage_progress"].as_float()),
            else_=Job.progress_json["overall_progress"].as_float(),
        )
        reported_type = case(
            (use_stage, func.json_type(Job.progress_json, "$.stage_progress")),
            else_=overall_type,
        )
        valid_report = (
            (WorkStep.status == "running")
            & (func.json_type(Job.progress_json, "$.version") == "integer")
            & (Job.progress_json["version"].as_integer() == 2)
            & (func.json_type(Job.progress_json, "$.indeterminate") == "false")
            & reported_type.in_(("integer", "real"))
            & (reported >= 0)
            & (reported <= 1)
        )
        # Count every matching job, including those without valid progress.
        # The aggregate returns one scalar only when ownership is unambiguous.
        report_query = (
            select(Job.id)
            .where(
                Job.work_step_id == WorkStep.id,
                Job.work_plan_id == plan_id,
                Job.kind.in_(_VISIBLE),
                Job.status == "running",
            )
            .having(func.count(Job.id) == 1)
            .correlate(WorkStep)
        )
        progress = report_query.with_only_columns(
            func.min(case((valid_report, reported)))
        ).scalar_subquery()
        progress_scope = report_query.with_only_columns(
            func.min(case((valid_report, case((use_stage, "stage"), else_="overall"))))
        ).scalar_subquery()
        rows = session.execute(
            select(
                WorkStep.id,
                WorkStep.ordinal,
                WorkStep.operation,
                WorkStep.status,
                progress.label("progress"),
                progress_scope.label("progress_scope"),
                case((WorkStep.status.in_(("queued", "blocked")), unfinished), else_=0).label(
                    "blocked_by"
                ),
            )
            .where(WorkStep.plan_id == plan_id)
            .order_by(WorkStep.ordinal, WorkStep.id)
            .offset(offset)
            .limit(limit)
        ).all()
        labels = {
            **_LABELS,
            "text": "Chat generation",
            "text_to_image": "Image generation",
            "image_to_image": "Image editing",
            "text_to_video": "Video generation",
            "image_to_video": "Image to video",
            "video_to_video": "Video editing",
        }
        items: list[QueueStepOut] = []
        for row in rows:
            status = "blocked" if row.status == "queued" and row.blocked_by else row.status
            if status not in {*JobStatus, "blocked"}:
                raise QueueStepStateError()
            items.append(
                QueueStepOut(
                    id=row.id,
                    ordinal=row.ordinal,
                    label=labels.get(row.operation, "Work step"),
                    status=cast(WorkStepStatus, status),
                    blocked_by=row.blocked_by,
                    progress=row.progress,
                    progress_scope=row.progress_scope,
                )
            )
        next_offset = offset + len(items) if offset + len(items) < total else None
        return QueuePlanStepsOut(
            plan_id=plan_id,
            items=items,
            total=total,
            next_offset=next_offset,
            observed_at=datetime.now(UTC),
        )
