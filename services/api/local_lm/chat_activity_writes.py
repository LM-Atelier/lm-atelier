"""Record visible terminal activity inside the transaction that settles a job."""

from __future__ import annotations

from sqlalchemy import exists, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from .domain import new_id
from .models import (
    Chat,
    ChatActivityEvent,
    Job,
    Message,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    WorkPlan,
)
from .prompt_helpers import STANDARD_CHAT_SCOPE


def record_response_activity(
    session: Session, run: Run, message: Message, revision: ResponseRevision
) -> None:
    job = session.scalar(
        select(Job)
        .where(Job.run_id == run.id, Job.kind.in_(("chat", "image", "video")))
        .order_by(Job.attempt.desc(), Job.created_at.desc(), Job.id.desc())
        .limit(1)
    )
    if job is not None:
        record_chat_activity(session, job=job, run=run, message=message, revision=revision)


def record_chat_activity(
    session: Session,
    *,
    job: Job,
    run: Run,
    message: Message,
    revision: ResponseRevision,
) -> ChatActivityEvent | None:
    """Persist one event per attempt after the caller owns its terminal transition.

    This does not claim a job or commit. The caller's terminal write and this
    insert must succeed or roll back together, including during startup recovery.
    """
    if (
        job.run_id != run.id
        or job.kind not in {"chat", "image", "video"}
        or revision.run_id != run.id
        or revision.status != run.status
        or revision.message_id != message.id
        or message.chat_id != run.chat_id
        or message.role != "assistant"
        or not message.transcript_visible
        or message.content_removed_at is not None
        or job.completed_at is None
    ):
        return None
    if run.status == "failed" and job.status in {"failed", "interrupted"}:
        kind = "failure"
    elif run.status in {"complete", "cancelled"} and job.status == run.status:
        kind = "output"
    else:
        return None
    session.flush()
    if session.scalar(select(Chat.scope).where(Chat.id == run.chat_id)) != STANDARD_CHAT_SCOPE:
        return None
    if run.work_plan_id is not None:
        plan = session.execute(
            select(WorkPlan.chat_id, WorkPlan.persistence_scope).where(
                WorkPlan.id == run.work_plan_id
            )
        ).one_or_none()
        if plan is None or plan.chat_id != run.chat_id or plan.persistence_scope != "durable":
            return None
    if kind == "output" and not session.scalar(
        select(
            exists().where(
                ResponseRevisionPart.response_revision_id == revision.id,
                ResponseRevisionPart.metadata_json["preview"].as_boolean().is_not(True),
                (
                    (ResponseRevisionPart.type == "text")
                    & ResponseRevisionPart.text.is_not(None)
                    & (ResponseRevisionPart.text != "")
                )
                | (
                    ResponseRevisionPart.type.in_(("image", "video", "audio"))
                    & ResponseRevisionPart.artifact_id.is_not(None)
                ),
            )
        )
    ):
        return None
    session.execute(
        insert(ChatActivityEvent)
        .values(
            id=new_id("act"),
            chat_id=run.chat_id,
            message_id=message.id,
            response_revision_id=revision.id,
            job_id=job.id,
            attempt=job.attempt,
            kind=kind,
            occurred_at=job.completed_at,
        )
        .on_conflict_do_nothing(index_elements=["job_id", "attempt"])
    )
    activity = session.scalar(
        select(ChatActivityEvent).where(
            ChatActivityEvent.job_id == job.id, ChatActivityEvent.attempt == job.attempt
        )
    )
    if activity is not None:
        revision.activity_json = {
            "id": activity.id,
            "sequence": activity.sequence,
            "message_id": activity.message_id,
            "response_revision_id": activity.response_revision_id,
            "occurred_at": activity.occurred_at.isoformat(),
        }
    return activity
