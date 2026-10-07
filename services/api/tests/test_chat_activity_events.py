from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, delete, event, func, select
from sqlalchemy.orm import Session

from local_lm.chat_activity_history import ChatActivityHistory, chat_activity_history
from local_lm.chat_activity_writes import record_chat_activity
from local_lm.db import Base
from local_lm.models import (
    Chat,
    ChatActivityEvent,
    Job,
    Message,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    WorkPlan,
)


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value
    engine.dispose()


def execution(
    session: Session, identity: str, *, status: str = "complete", target: Message | None = None
) -> tuple[Job, Run, Message, ResponseRevision]:
    chat = session.get(Chat, "chat")
    if chat is None:
        session.add(Chat(id="chat", title="Color study"))
        session.flush()
    user = Message(id="user-" + identity, chat_id="chat", role="user")
    staged = Message(
        id="staged-" + identity, chat_id="chat", role="assistant", transcript_visible=target is None
    )
    session.add_all([user, staged])
    session.flush()
    run = Run(
        id="run-" + identity,
        chat_id="chat",
        user_message_id=user.id,
        assistant_message_id=staged.id,
        status=status,
    )
    session.add(run)
    session.flush()
    message = target or staged
    revision = ResponseRevision(
        id="revision-" + identity,
        message_id=message.id,
        run_id=run.id,
        sequence=2 if target else 1,
        status=status,
        parts=[ResponseRevisionPart(position=0, type="text", text="A neutral color study.")],
    )
    job = Job(
        id="job-" + identity,
        run_id=run.id,
        kind="chat",
        status=status,
        attempt=1,
        completed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    session.add_all([revision, job])
    session.commit()
    return job, run, message, revision


def record(
    session: Session, execution: tuple[Job, Run, Message, ResponseRevision]
) -> ChatActivityEvent | None:
    job, run, message, revision = execution
    return record_chat_activity(session, job=job, run=run, message=message, revision=revision)


def test_completion_order_is_durable_and_does_not_follow_acceptance(session: Session) -> None:
    earlier = execution(session, "earlier")
    later = execution(session, "later")
    second = record(session, later)
    first = record(session, earlier)
    assert first is not None and second is not None
    assert first.sequence > second.sequence
    session.commit()
    session.expire_all()
    assert (
        session.scalar(select(ChatActivityEvent.id).order_by(ChatActivityEvent.sequence.desc()))
        == first.id
    )


def test_replayed_attempt_keeps_its_identity_and_a_retry_gets_a_new_one(session: Session) -> None:
    items = execution(session, "retry", status="failed")
    first = record(session, items)
    assert first is not None and first.kind == "failure"
    session.commit()
    duplicate = record(session, items)
    assert (
        duplicate is not None and duplicate.id == first.id and duplicate.sequence == first.sequence
    )
    job, run, _, revision = items
    job.attempt = 2
    job.status = run.status = revision.status = "complete"
    retried = record(session, items)
    assert retried is not None and retried.kind == "output"
    assert retried.sequence > first.sequence and retried.id != first.id
    assert session.scalar(select(func.count()).select_from(ChatActivityEvent)) == 2


def test_rollback_publishes_no_activity_and_deleted_sequences_are_not_reused(
    session: Session,
) -> None:
    items = execution(session, "first")
    rolled_back = record(session, items)
    assert rolled_back is not None
    session.rollback()
    assert session.scalar(select(func.count()).select_from(ChatActivityEvent)) == 0
    first = record(session, items)
    assert first is not None
    sequence, identity = first.sequence, first.id
    session.commit()
    session.execute(delete(ChatActivityEvent))
    session.commit()
    replacement = record(session, execution(session, "second"))
    assert replacement is not None
    assert replacement.sequence > sequence and replacement.id != identity


def test_replacement_activity_binds_the_visible_target(session: Session) -> None:
    original = execution(session, "original")
    replacement = execution(session, "replacement", target=original[2])
    activity = record(session, replacement)
    assert activity is not None
    assert activity.message_id == original[2].id
    assert activity.response_revision_id == replacement[3].id
    assert activity.message_id != replacement[1].assistant_message_id


@pytest.mark.parametrize(
    "reason",
    [
        "hidden-chat",
        "hidden-message",
        "removed-message",
        "internal-job",
        "running",
        "private-plan",
        "wrong-revision",
        "empty-cancel",
    ],
)
def test_non_visible_or_unsettled_work_does_not_publish_activity(
    session: Session, reason: str
) -> None:
    items = execution(session, "excluded")
    job, run, message, revision = items
    if reason == "hidden-chat":
        chat = session.get(Chat, "chat")
        assert chat is not None
        chat.scope = "prompt-helper"
    elif reason == "hidden-message":
        message.transcript_visible = False
    elif reason == "removed-message":
        revision.parts.clear()
        session.flush()
        message.content_removed_at = datetime(2026, 9, 1, tzinfo=UTC)
    elif reason == "internal-job":
        job.kind = "edit_verify"
    elif reason == "running":
        job.status = run.status = "running"
    elif reason == "private-plan":
        session.add(
            WorkPlan(
                id="private", chat_id="chat", transcript_sequence=1, persistence_scope="incognito"
            )
        )
        session.flush()
        run.work_plan_id = "private"
    elif reason == "wrong-revision":
        revision.run_id = None
    elif reason == "empty-cancel":
        job.status = run.status = revision.status = "cancelled"
        revision.parts.clear()
    assert record(session, items) is None
    assert session.scalar(select(func.count()).select_from(ChatActivityEvent)) == 0


def test_partial_cancelled_output_gets_an_activity_identity(session: Session) -> None:
    activity = record(session, execution(session, "partial", status="cancelled"))
    assert activity is not None and activity.kind == "output"


def test_history_returns_only_the_latest_scalar_identities_per_kind(session: Session) -> None:
    record(session, execution(session, "old-output"))
    failed = record(session, execution(session, "failed", status="failed"))
    latest = record(session, execution(session, "latest"))
    assert failed is not None and latest is not None
    identities = failed.id, latest.id
    session.commit()
    session.expunge_all()
    hydrated: list[object] = []

    def loaded(_session: Session, instance: object) -> None:
        hydrated.append(instance)

    event.listen(session, "loaded_as_persistent", loaded)
    try:
        values = chat_activity_history(session, ["chat", "missing", "chat"])
    finally:
        event.remove(session, "loaded_as_persistent", loaded)
    history = values["chat"]
    assert history.last_failure is not None and history.last_output is not None
    assert (history.last_failure.id, history.last_output.id) == identities
    assert values["missing"] == ChatActivityHistory()
    assert hydrated == []
    assert chat_activity_history(session, []) == {}
    with pytest.raises(ValueError, match="Too many chats"):
        chat_activity_history(session, [str(index) for index in range(201)])


@pytest.mark.parametrize("change", ["hidden", "removed", "cleared-output", "private-chat"])
def test_history_rechecks_current_visibility(session: Session, change: str) -> None:
    items = execution(session, "output")
    assert record(session, items) is not None
    session.commit()
    _, _, message, revision = items
    if change == "hidden":
        message.transcript_visible = False
    elif change == "private-chat":
        chat = session.get(Chat, "chat")
        assert chat is not None
        chat.scope = "prompt-helper"
    else:
        revision.parts.clear()
        session.flush()
        if change == "removed":
            message.content_removed_at = datetime(2026, 9, 1, tzinfo=UTC)
    session.commit()
    assert chat_activity_history(session, ["chat"]) == {"chat": ChatActivityHistory()}


def test_history_does_not_report_a_failure_replaced_by_a_successful_retry(session: Session) -> None:
    items = execution(session, "retry", status="failed")
    assert record(session, items) is not None
    job, run, _, revision = items
    job.attempt += 1
    job.status = run.status = revision.status = "complete"
    output = record(session, items)
    assert output is not None
    session.commit()
    history = chat_activity_history(session, ["chat"])["chat"]
    assert history.last_failure is None
    assert history.last_output is not None and history.last_output.id == output.id
