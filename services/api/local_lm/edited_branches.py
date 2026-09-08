"""Discover accepted alternate versions and explicitly select a completed branch."""

from __future__ import annotations

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, selectinload

from .models import Chat, Job, Message, Run, WorkPlan, WorkStep
from .prior_turn_edits import EditRequestConflict
from .schemas import EditedBranchOut, EditedBranchPage, JobOut, WorkPlanOut


def _entries(session: Session, plans: list[WorkPlan]) -> list[EditedBranchOut]:
    if not plans:
        return []
    plan_ids = [plan.id for plan in plans]
    runs = list(
        session.scalars(
            select(Run)
            .join(WorkStep, WorkStep.id == Run.work_step_id)
            .where(Run.work_plan_id.in_(plan_ids))
            .order_by(WorkStep.ordinal, Run.id)
        )
    )
    message_ids = {
        message_id
        for run in runs
        for message_id in (run.user_message_id, run.assistant_message_id)
        if message_id is not None
    }
    for plan in plans:
        source = plan.summary_json.get("edit_source") or {}
        if isinstance(source.get("source_message_id"), str):
            message_ids.add(source["source_message_id"])
    messages = {
        message.id: message
        for message in session.scalars(select(Message).where(Message.id.in_(message_ids)))
    }
    jobs = list(
        session.scalars(
            select(Job)
            .where(Job.work_plan_id.in_(plan_ids), Job.kind != "edit_verify")
            .order_by(Job.created_at, Job.id)
        )
    )
    result = []
    for plan in plans:
        source = plan.summary_json["edit_source"]
        head_id = plan.summary_json.get("branch_head_message_id")
        if not isinstance(head_id, str):
            head_id = ""
        branch_runs = [run for run in runs if run.work_plan_id == plan.id]
        target_ids = {
            message_id
            for run in branch_runs
            for message_id in (run.user_message_id, run.assistant_message_id)
        }
        head = messages.get(head_id)
        source_message = messages.get(source["source_message_id"])
        available = bool(
            branch_runs
            and plan.steps
            and len(branch_runs) == len(plan.steps)
            and plan.status == "complete"
            and all(step.status == "complete" for step in plan.steps)
            and all(run.status == "complete" and run.chat_id == plan.chat_id for run in branch_runs)
            and branch_runs[-1].assistant_message_id == head_id
            and head is not None
            and head.role == "assistant"
            and head.transcript_visible
            and all(
                (message := messages.get(message_id)) is not None
                and message.chat_id == plan.chat_id
                and message.content_removed_at is None
                and message.status == "complete"
                for message_id in target_ids
            )
        )
        result.append(
            EditedBranchOut(
                source_message_id=source["source_message_id"],
                source_run_id=source["source_run_id"],
                branch_head_message_id=head_id,
                source_available=bool(
                    source_message is not None
                    and source_message.chat_id == plan.chat_id
                    and source_message.transcript_visible
                    and source_message.content_removed_at is None
                ),
                can_continue=available,
                plan=WorkPlanOut.model_validate(plan),
                jobs=[JobOut.model_validate(job) for job in jobs if job.work_plan_id == plan.id],
            )
        )
    return result


def list_edited_branches(
    session: Session, chat_id: str, *, limit: int, cursor: str | None
) -> EditedBranchPage:
    if session.get(Chat, chat_id) is None:
        raise LookupError("Chat not found.")
    query = select(WorkPlan).where(
        WorkPlan.chat_id == chat_id,
        WorkPlan.source_action == "edit_and_branch",
        WorkPlan.summary_json["edit_source"]["source_message_id"].as_string().is_not(None),
    )
    if cursor is not None:
        anchor = session.scalar(query.where(WorkPlan.id == cursor))
        if anchor is None:
            raise LookupError("Edited branch cursor not found in this chat.")
        query = query.where(
            or_(
                WorkPlan.created_at < anchor.created_at,
                and_(WorkPlan.created_at == anchor.created_at, WorkPlan.id < anchor.id),
            )
        )
    plans = list(
        session.scalars(
            query.options(selectinload(WorkPlan.steps))
            .order_by(WorkPlan.created_at.desc(), WorkPlan.id.desc())
            .limit(limit + 1)
        )
    )
    return EditedBranchPage(
        items=_entries(session, plans[:limit]),
        next_cursor=plans[limit - 1].id if len(plans) > limit else None,
    )


def activate_edited_branch(
    session: Session, chat_id: str, plan_id: str, expected_active_head_message_id: str | None
) -> str:
    """Caller holds the chat graph guard and commits this head change."""
    chat = session.get(Chat, chat_id)
    plan = session.scalar(
        select(WorkPlan)
        .options(selectinload(WorkPlan.steps))
        .where(
            WorkPlan.id == plan_id,
            WorkPlan.chat_id == chat_id,
            WorkPlan.source_action == "edit_and_branch",
            WorkPlan.summary_json["edit_source"]["source_message_id"].as_string().is_not(None),
        )
    )
    if chat is None or plan is None:
        raise LookupError("Edited branch not found in this chat.")
    branch = _entries(session, [plan])[0]
    if not branch.can_continue:
        raise EditRequestConflict("This edited branch is not available to continue.")
    if chat.active_head_message_id not in (
        expected_active_head_message_id,
        branch.branch_head_message_id,
    ):
        raise EditRequestConflict(
            "The active conversation changed. Select the edited version again."
        )
    chat.active_head_message_id = branch.branch_head_message_id
    return branch.branch_head_message_id
