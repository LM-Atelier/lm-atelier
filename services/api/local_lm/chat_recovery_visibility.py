"""Exclude recoverable conversations from ordinary workspace reads."""

from __future__ import annotations

from sqlalchemy import SQLColumnExpression, and_, exists, func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .models import Job, RecoveryItem, Run, WorkPlan, WorkStep
from .recovery_v1 import RecoveryKind


def visible_chat(chat_id: SQLColumnExpression[str]) -> ColumnElement[bool]:
    """Membership hides every deletion state, including blocked or purged records."""

    return ~exists(
        select(RecoveryItem.subject_id).where(
            RecoveryItem.kind == RecoveryKind.CHAT.value, RecoveryItem.subject_id == chat_id
        )
    )


def chat_is_deleted(session: Session, chat_id: str) -> bool:
    """Check the database rather than a possibly cached Chat object."""

    return (
        session.scalar(
            select(RecoveryItem.subject_id).where(
                RecoveryItem.kind == RecoveryKind.CHAT.value, RecoveryItem.subject_id == chat_id
            )
        )
        is not None
    )


def visible_job() -> ColumnElement[bool]:
    """Use current owners before a historical source when hiding a job."""

    run_chat = select(Run.chat_id).where(Run.id == Job.run_id).correlate(Job).scalar_subquery()
    plan_chat = (
        select(WorkPlan.chat_id)
        .where(WorkPlan.id == Job.work_plan_id)
        .correlate(Job)
        .scalar_subquery()
    )
    step_chat = (
        select(WorkPlan.chat_id)
        .join(WorkStep, WorkStep.plan_id == WorkPlan.id)
        .where(WorkStep.id == Job.work_step_id)
        .correlate(Job)
        .scalar_subquery()
    )
    named_chat = func.json_extract(Job.payload_json, "$.chat_id")
    source_chat = (
        select(Run.chat_id)
        .where(Run.id == func.json_extract(Job.payload_json, "$.source_run_id"))
        .correlate(Job)
        .scalar_subquery()
    )
    deleted_owner = or_(
        RecoveryItem.subject_id == run_chat,
        RecoveryItem.subject_id == plan_chat,
        RecoveryItem.subject_id == step_chat,
        RecoveryItem.subject_id == named_chat,
        and_(
            Job.run_id.is_(None),
            Job.work_plan_id.is_(None),
            Job.work_step_id.is_(None),
            named_chat.is_(None),
            RecoveryItem.subject_id == source_chat,
        ),
    )
    return ~exists(
        select(RecoveryItem.subject_id)
        .where(RecoveryItem.kind == RecoveryKind.CHAT.value, deleted_owner)
        .correlate(Job)
    )


def job_is_deleted(session: Session, job_id: str) -> bool:
    return session.scalar(select(Job.id).where(Job.id == job_id, ~visible_job())) is not None
