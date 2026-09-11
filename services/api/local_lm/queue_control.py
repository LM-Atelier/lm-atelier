"""Durable queue controls; execution status and accepted work remain unchanged."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from sqlalchemy import and_, case, exists, func, literal, or_, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .domain import utcnow
from .models import Chat, Job, Run, WorkPlan, WorkPlanControl, WorkPlanControlReceipt
from .schemas import QueueControlCommand, QueueControlResultOut

_TERMINAL = ("complete", "failed", "cancelled", "interrupted")
_CONTROLLED_PLAN_STATUSES = ("queued", "running", "paused", "blocked")
_MAX_REVISION = 9_223_372_036_854_775_807


class QueueControlMissing(Exception):
    """The visible durable owner does not exist."""


class QueueControlConflict(Exception):
    """The command cannot take effect against current durable state."""


def _verification_owner() -> ColumnElement[str | None]:
    return (
        select(Run.work_plan_id)
        .where(Run.id == Job.payload_json["source_run_id"].as_string())
        .correlate(Job)
        .scalar_subquery()
    )


def job_owner_id() -> ColumnElement[str | None]:
    # Direct ownership is normal. Internal verification jobs retain their source
    # run even after their source Job is deleted.
    return func.coalesce(
        Job.work_plan_id,
        case((Job.kind == "edit_verify", _verification_owner()), else_=None),
    )


def job_owner_valid() -> ColumnElement[bool]:
    return or_(
        Job.kind != "edit_verify",
        and_(
            func.json_type(Job.payload_json, "$.source_run_id") == "text",
            exists(
                select(Run.id)
                .where(
                    Run.id == Job.payload_json["source_run_id"].as_string(),
                    or_(Job.work_plan_id.is_(None), Job.work_plan_id == Run.work_plan_id),
                )
                .correlate(Job)
            ),
        ),
    )


def plan_descendants(plan_id: str | ColumnElement[str]) -> ColumnElement[bool]:
    return or_(
        Job.work_plan_id == plan_id,
        and_(Job.kind == "edit_verify", _verification_owner() == plan_id),
    )


def _utc(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


@dataclass(frozen=True)
class JobQueueControl:
    owner_id: str | None
    valid: bool
    state: str
    revision: int
    eligible_since: datetime | None


def job_controls(session: Session, job_ids: list[str]) -> dict[str, JobQueueControl]:
    if not job_ids:
        return {}
    rows = session.execute(
        select(
            Job.id,
            job_owner_id(),
            job_owner_valid(),
            WorkPlanControl.state,
            WorkPlanControl.revision,
            WorkPlanControl.eligible_since,
        )
        .outerjoin(WorkPlanControl, WorkPlanControl.plan_id == job_owner_id())
        .where(Job.id.in_(job_ids))
    )
    return {
        identifier: JobQueueControl(
            owner_id=owner,
            valid=bool(valid) and (state is None or state in ("eligible", "held")),
            state=state or "eligible",
            revision=revision or 0,
            eligible_since=_utc(since),
        )
        for identifier, owner, valid, state, revision, since in rows
    }


def claim_control_predicate(snapshot: JobQueueControl) -> ColumnElement[bool]:
    """Compare the pre-ranking revision in the final claim's database UPDATE."""
    owner = job_owner_id()
    same_owner = owner.is_(None) if snapshot.owner_id is None else owner == snapshot.owner_id
    unchanged = (
        ~exists(select(WorkPlanControl.plan_id).where(WorkPlanControl.plan_id == owner))
        if snapshot.revision == 0
        else exists(
            select(WorkPlanControl.plan_id).where(
                WorkPlanControl.plan_id == owner,
                WorkPlanControl.revision == snapshot.revision,
                WorkPlanControl.state == "eligible",
            )
        )
    )
    return and_(
        job_owner_valid(),
        same_owner,
        unchanged,
        literal(snapshot.valid and snapshot.state == "eligible"),
    )


def change_plan_control(
    session: Session,
    plan_id: str,
    action: Literal["hold", "release"],
    command: QueueControlCommand,
) -> QueueControlResultOut:
    """Use one fresh transaction for the transition and its exact retry result.

    BEGIN IMMEDIATE serializes this decision with scheduler claim writes, including
    the first transition when no control row exists. No process lock is authority.
    Successful receipts live until their plan is deleted, even after completion.
    """
    if session.in_transaction():
        raise QueueControlConflict
    try:
        session.execute(text("BEGIN IMMEDIATE"))
        owner = session.execute(
            select(WorkPlan.id, WorkPlan.status)
            .join(Chat, Chat.id == WorkPlan.chat_id)
            .where(
                WorkPlan.id == plan_id,
                WorkPlan.persistence_scope == "durable",
                Chat.scope == "standard",
            )
        ).one_or_none()
        if owner is None:
            raise QueueControlMissing
        receipt = session.get(WorkPlanControlReceipt, (plan_id, command.idempotency_key))
        if receipt is not None:
            if receipt.action != action or receipt.expected_revision != command.expected_revision:
                raise QueueControlConflict
            result = QueueControlResultOut.model_validate(receipt.response_json)
            session.rollback()
            return result
        control = session.get(WorkPlanControl, plan_id)
        revision = control.revision if control else 0
        state = control.state if control else "eligible"
        if (
            owner.status not in _CONTROLLED_PLAN_STATUSES
            or revision != command.expected_revision
            or revision >= _MAX_REVISION
            or state != ("eligible" if action == "hold" else "held")
        ):
            raise QueueControlConflict
        remaining = session.execute(
            select(Job.status, Job.claim_owner, job_owner_valid()).where(
                plan_descendants(plan_id), Job.status.not_in(_TERMINAL)
            )
        ).all()
        if not remaining or any(
            status != "queued" or claimant is not None or not valid
            for status, claimant, valid in remaining
        ):
            raise QueueControlConflict
        if control is None:
            control = WorkPlanControl(plan_id=plan_id, state="eligible", revision=0)
            session.add(control)
        next_state: Literal["eligible", "held"] = "held" if action == "hold" else "eligible"
        control.state = next_state
        control.revision = revision + 1
        if action == "release":
            control.eligible_since = utcnow()
        session.flush()
        result = QueueControlResultOut(
            owner_id=plan_id,
            control_state=next_state,
            control_revision=control.revision,
            eligible_since=_utc(control.eligible_since),
        )
        session.add(
            WorkPlanControlReceipt(
                plan_id=plan_id,
                command_key=command.idempotency_key,
                action=action,
                expected_revision=command.expected_revision,
                response_json=result.model_dump(mode="json"),
            )
        )
        session.commit()
        return result
    except OperationalError as exc:
        session.rollback()
        code = getattr(exc.orig, "sqlite_errorcode", None)
        if isinstance(code, int) and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise QueueControlConflict from exc
        raise
    except BaseException:
        session.rollback()
        raise


@dataclass(frozen=True)
class QueueControlView:
    state: Literal["eligible", "held"] | None = None
    revision: int | None = None
    allowed_actions: tuple[Literal["hold", "release"], ...] = ()


def plan_controls(session: Session, plan_ids: list[str]) -> dict[str, QueueControlView]:
    """Return scalar capabilities in one query within the queue read snapshot."""
    if not plan_ids:
        return {}
    descendants = plan_descendants(WorkPlan.id.__clause_element__())
    queued = exists(
        select(Job.id).where(descendants, Job.status == "queued", Job.claim_owner.is_(None))
    )
    unsafe = exists(
        select(Job.id).where(
            descendants,
            Job.status.not_in(_TERMINAL),
            or_(
                Job.status != "queued",
                Job.claim_owner.is_not(None),
                job_owner_valid().is_not(True),
            ),
        )
    )
    rows = session.execute(
        select(
            WorkPlan.id,
            WorkPlan.status,
            WorkPlan.persistence_scope,
            Chat.scope,
            WorkPlanControl.state,
            WorkPlanControl.revision,
            queued,
            unsafe,
        )
        .outerjoin(Chat, Chat.id == WorkPlan.chat_id)
        .outerjoin(WorkPlanControl, WorkPlanControl.plan_id == WorkPlan.id)
        .where(WorkPlan.id.in_(plan_ids))
    )
    result: dict[str, QueueControlView] = {}
    for identifier, status, scope, chat_scope, state, revision, has_work, unsafe_work in rows:
        if scope != "durable" or chat_scope != "standard":
            result[identifier] = QueueControlView()
            continue
        if state is None and revision is None:
            state, revision = "eligible", 0
        if (
            state not in ("eligible", "held")
            or not isinstance(revision, int)
            or not 0 <= revision < _MAX_REVISION
        ):
            result[identifier] = QueueControlView()
            continue
        control_state = cast(Literal["eligible", "held"], state)
        action: Literal["hold", "release"] = "hold" if control_state == "eligible" else "release"
        result[identifier] = QueueControlView(
            state=control_state,
            revision=revision,
            allowed_actions=(action,)
            if has_work and not unsafe_work and status in _CONTROLLED_PLAN_STATUSES
            else (),
        )
    return result
