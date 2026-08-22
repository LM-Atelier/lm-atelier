"""Delete one user turn and everything the exchange produced.

The service half of deleting a turn: given a user message,
remove it, its assistant answer (every branch step and response revision),
the runs and jobs that produced them, and the exchange's work plan - and
release the generated media so the existing retention sweep reclaims it.
Artifacts are content-addressed and reference-counted through message parts,
so this module never touches files: it deletes references and reports which
artifacts became unreferenced.

Two refusals are deliberate product decisions, not limitations:

- An exchange with a user reply anywhere below it is refused. `parent_id`
  is SET NULL on delete, so deleting the middle of a branch would silently
  turn replies into orphaned roots; "delete the replies first" keeps lineage
  honest.
- An exchange with queued or running jobs is refused. Cancellation is the
  job queue's business (and the worker reset control exists for wedges);
  deletion only ever removes finished history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import JobKind, JobStatus, MessageRole
from .models import (
    Chat,
    Job,
    Message,
    MessagePart,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    WorkPlan,
    WorkStep,
)

_ACTIVE_JOB_STATUSES = frozenset(
    {JobStatus.QUEUED.value, JobStatus.RUNNING.value, JobStatus.PAUSED.value}
)


class ExchangeDeletionError(ValueError):
    """Base class so callers can map every refusal to one error family."""


class ExchangeNotFound(ExchangeDeletionError):
    def __init__(self) -> None:
        super().__init__("the user message was not found")


class ExchangeHasReplies(ExchangeDeletionError):
    def __init__(self, reply_count: int) -> None:
        self.reply_count = reply_count
        super().__init__(
            f"this turn has {reply_count} later "
            f"{'reply' if reply_count == 1 else 'replies'}; delete those first"
        )


class ExchangeBusy(ExchangeDeletionError):
    def __init__(self, job_count: int) -> None:
        self.job_count = job_count
        super().__init__(
            f"this turn still has {job_count} queued or running "
            f"{'job' if job_count == 1 else 'jobs'}; wait or cancel them first"
        )


@dataclass(frozen=True)
class ExchangeDeletion:
    """What a deletion removed - computed before anything is deleted."""

    chat_id: str
    user_message_id: str
    message_ids: list[str]
    run_ids: list[str]
    job_ids: list[str]
    work_plan_ids: list[str]
    # Artifacts the exchange referenced, split by whether any reference
    # survives elsewhere. Unreferenced ones are reclaimed by retention later.
    released_artifact_ids: list[str] = field(default_factory=list)
    retained_artifact_ids: list[str] = field(default_factory=list)
    new_head_message_id: str | None = None


def delete_exchange(session: Session, user_message_id: str) -> ExchangeDeletion:
    """Delete the exchange rooted at one user message and summarize it."""

    user = session.get(Message, user_message_id)
    if not user or user.role != MessageRole.USER.value:
        raise ExchangeNotFound

    descendants = _descendants(session, user)
    replies = [item for item in descendants if item.role == MessageRole.USER.value]
    if replies:
        raise ExchangeHasReplies(len(replies))

    messages = [user, *descendants]
    message_ids = [item.id for item in messages]
    runs = list(
        session.scalars(
            select(Run).where(
                (Run.user_message_id == user.id) | (Run.assistant_message_id.in_(message_ids))
            )
        )
    )
    run_ids = sorted({run.id for run in runs})
    jobs = (
        list(
            session.scalars(
                select(Job).where(
                    (Job.run_id.in_(run_ids))
                    | (
                        (Job.kind == JobKind.EDIT_VERIFY.value)
                        & (Job.payload_json["source_run_id"].as_string().in_(run_ids))
                    )
                )
            )
        )
        if run_ids
        else []
    )
    busy = [job for job in jobs if job.status in _ACTIVE_JOB_STATUSES]
    if busy:
        raise ExchangeBusy(len(busy))
    work_plan_ids = sorted({run.work_plan_id for run in runs if run.work_plan_id})

    referenced = _artifact_ids_for_messages(session, message_ids)

    chat = session.get(Chat, user.chat_id)
    new_head: str | None = None
    if chat and chat.active_head_message_id in set(message_ids):
        new_head = user.parent_id
        chat.active_head_message_id = new_head

    for job in jobs:
        session.delete(job)
    for run in runs:
        session.delete(run)
    # Children before parents: Message.parent_id is SET NULL on delete, and a
    # half-deleted chain must never leave an orphan behind even briefly.
    for message in reversed(messages):
        session.delete(message)
    if work_plan_ids:
        for step in session.scalars(select(WorkStep).where(WorkStep.plan_id.in_(work_plan_ids))):
            session.delete(step)
        for plan in session.scalars(select(WorkPlan).where(WorkPlan.id.in_(work_plan_ids))):
            session.delete(plan)
    session.flush()

    released, retained = _split_by_surviving_references(session, referenced)
    return ExchangeDeletion(
        chat_id=user.chat_id,
        user_message_id=user.id,
        message_ids=message_ids,
        run_ids=run_ids,
        job_ids=sorted(job.id for job in jobs),
        work_plan_ids=work_plan_ids,
        released_artifact_ids=released,
        retained_artifact_ids=retained,
        new_head_message_id=new_head,
    )


def _descendants(session: Session, root: Message) -> list[Message]:
    """Every message below the root, breadth-first, within its chat."""

    collected: list[Message] = []
    frontier = [root.id]
    seen = {root.id}
    while frontier:
        children = list(
            session.scalars(
                select(Message).where(
                    Message.chat_id == root.chat_id, Message.parent_id.in_(frontier)
                )
            )
        )
        frontier = []
        for child in children:
            if child.id in seen:
                continue
            seen.add(child.id)
            collected.append(child)
            frontier.append(child.id)
    return collected


def _artifact_ids_for_messages(session: Session, message_ids: list[str]) -> list[str]:
    direct = session.scalars(
        select(MessagePart.artifact_id).where(
            MessagePart.message_id.in_(message_ids), MessagePart.artifact_id.is_not(None)
        )
    )
    revisions = session.scalars(
        select(ResponseRevisionPart.artifact_id)
        .join(
            ResponseRevision,
            ResponseRevisionPart.response_revision_id == ResponseRevision.id,
        )
        .where(
            ResponseRevision.message_id.in_(message_ids),
            ResponseRevisionPart.artifact_id.is_not(None),
        )
    )
    return sorted({value for value in [*direct, *revisions] if value})


def _split_by_surviving_references(
    session: Session, artifact_ids: list[str]
) -> tuple[list[str], list[str]]:
    released: list[str] = []
    retained: list[str] = []
    for artifact_id in artifact_ids:
        still_referenced = session.scalar(
            select(MessagePart.id).where(MessagePart.artifact_id == artifact_id).limit(1)
        ) or session.scalar(
            select(ResponseRevisionPart.id)
            .where(ResponseRevisionPart.artifact_id == artifact_id)
            .limit(1)
        )
        (retained if still_referenced else released).append(artifact_id)
    return released, retained
