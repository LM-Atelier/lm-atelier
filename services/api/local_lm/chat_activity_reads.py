"""Count visible work without reading prompts, settings, or error descriptions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import case, exists, func, literal, or_, select, union, union_all
from sqlalchemy.orm import Session, aliased

from .models import Chat, Job, Message, ResponseRevision, Run, WorkPlan, WorkStep
from .prompt_helpers import STANDARD_CHAT_SCOPE

_ACTIVE = ("queued", "running", "paused", "blocked")
_FAILED = ("failed", "interrupted")
_GENERATION = ("chat", "image", "video")


@dataclass(frozen=True)
class ChatWorkCounts:
    active_work_count: int = 0
    unresolved_failed_count: int = 0


def chat_work_counts(session: Session, chat_ids: Sequence[str]) -> dict[str, ChatWorkCounts]:
    """Group planned work once and count failures on the latest visible response."""
    identities = tuple(dict.fromkeys(chat_ids))
    if len(identities) > 200:
        raise ValueError("Too many chats were requested.")
    if not identities:
        return {}
    values = {identity: [0, 0] for identity in identities}
    visible_job = exists(
        select(Job.id).where(
            Job.work_plan_id == WorkPlan.id,
            Job.kind.in_(_GENERATION),
            Job.status.in_(_ACTIVE),
        )
    )
    unbound_failed_step = exists(
        select(WorkStep.id).where(
            WorkStep.plan_id == WorkPlan.id,
            WorkStep.status.in_(_FAILED),
            ~exists(select(Run.id).where(Run.id == WorkStep.run_id)),
        )
    )
    plans = (
        select(
            WorkPlan.chat_id.label("chat_id"),
            func.sum(case((or_(WorkPlan.status.in_(_ACTIVE), visible_job), 1), else_=0)).label(
                "active"
            ),
            literal(0).label("failed"),
        )
        .join(Chat, Chat.id == WorkPlan.chat_id)
        .where(
            WorkPlan.chat_id.in_(identities),
            WorkPlan.persistence_scope == "durable",
            Chat.scope == STANDARD_CHAT_SCOPE,
        )
        .group_by(WorkPlan.chat_id)
    )
    target = aliased(Message)
    attempts = (
        select(
            Run.chat_id.label("chat_id"),
            Run.id.label("run_id"),
            WorkPlan.id.label("plan_id"),
            func.coalesce(Job.status, Run.status).label("status"),
            case(
                (WorkPlan.id.is_not(None), literal("plan:") + WorkPlan.id),
                else_=literal("run:") + Run.id,
            ).label("owner_id"),
            func.row_number()
            .over(
                partition_by=target.id,
                order_by=(
                    func.coalesce(ResponseRevision.sequence, 0).desc(),
                    Run.created_at.desc(),
                    Run.id.desc(),
                    Job.attempt.desc(),
                    Job.id.desc(),
                ),
            )
            .label("target_rank"),
        )
        .outerjoin(Job, Job.run_id == Run.id)
        .outerjoin(WorkPlan, WorkPlan.id == Run.work_plan_id)
        .outerjoin(ResponseRevision, ResponseRevision.run_id == Run.id)
        .join(
            target,
            target.id == func.coalesce(ResponseRevision.message_id, Run.assistant_message_id),
        )
        .join(Chat, Chat.id == Run.chat_id)
        .where(
            Run.chat_id.in_(identities),
            Chat.scope == STANDARD_CHAT_SCOPE,
            target.chat_id == Run.chat_id,
            target.role == "assistant",
            target.transcript_visible.is_(True),
            target.content_removed_at.is_(None),
            or_(Job.id.is_(None), Job.kind.in_(_GENERATION)),
            or_(
                WorkPlan.id.is_(None),
                WorkPlan.persistence_scope == "durable",
            ),
        )
        .subquery()
    )
    with session.no_autoflush:
        active_runs = (
            select(attempts.c.chat_id, func.count(func.distinct(attempts.c.run_id)), literal(0))
            .where(attempts.c.plan_id.is_(None), attempts.c.status.in_(_ACTIVE))
            .group_by(attempts.c.chat_id)
        )
        failed_owners = union(
            select(attempts.c.chat_id, attempts.c.owner_id).where(
                attempts.c.target_rank == 1, attempts.c.status.in_(_FAILED)
            ),
            select(WorkPlan.chat_id, literal("plan:") + WorkPlan.id)
            .join(Chat, Chat.id == WorkPlan.chat_id)
            .where(
                WorkPlan.chat_id.in_(identities),
                WorkPlan.persistence_scope == "durable",
                Chat.scope == STANDARD_CHAT_SCOPE,
                or_(
                    unbound_failed_step,
                    WorkPlan.status.in_(_FAILED)
                    & ~exists(select(Run.id).where(Run.work_plan_id == WorkPlan.id)),
                ),
            ),
        ).subquery()
        failures = select(failed_owners.c.chat_id, literal(0), func.count()).group_by(
            failed_owners.c.chat_id
        )
        counts = union_all(plans, active_runs, failures).subquery()
        totals = select(
            counts.c.chat_id, func.sum(counts.c.active), func.sum(counts.c.failed)
        ).group_by(counts.c.chat_id)
        for identity, active, failed in session.execute(totals):
            values[identity] = [int(active or 0), int(failed or 0)]
    return {identity: ChatWorkCounts(*counts) for identity, counts in values.items()}
