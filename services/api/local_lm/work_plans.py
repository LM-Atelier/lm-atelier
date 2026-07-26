from __future__ import annotations

from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import JobStatus
from .models import WorkPlan, WorkStep

BLOCKED_WORK_STATUS = "blocked"
_ACTIVE_ORDER = (
    JobStatus.RUNNING.value,
    JobStatus.QUEUED.value,
    JobStatus.PAUSED.value,
    BLOCKED_WORK_STATUS,
)


def refresh_plan_status(session: Session, plan_id: str | None) -> str | None:
    if not plan_id:
        return None
    plan = session.get(WorkPlan, plan_id)
    if not plan:
        return None
    statuses = list(
        session.scalars(
            select(WorkStep.status).where(WorkStep.plan_id == plan_id).order_by(WorkStep.ordinal)
        ).all()
    )
    if not statuses:
        return plan.status
    for active in _ACTIVE_ORDER:
        if active in statuses:
            plan.status = active
            return active
    if all(status == JobStatus.COMPLETE.value for status in statuses):
        plan.status = JobStatus.COMPLETE.value
    elif len(set(statuses)) == 1:
        plan.status = statuses[0]
    else:
        plan.status = "partial"
    return plan.status


def plan_status_summary(session: Session, plan_id: str) -> dict[str, int]:
    counts = Counter(
        session.scalars(select(WorkStep.status).where(WorkStep.plan_id == plan_id)).all()
    )
    return dict(sorted(counts.items()))
