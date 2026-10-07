from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from local_lm.chat_activity_reads import ChatWorkCounts, chat_work_counts
from local_lm.db import Base
from local_lm.models import Chat, Job, Message, ResponseRevision, Run, WorkPlan, WorkStep


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        value.add(Chat(id="chat", title="Color study"))
        value.commit()
        yield value
    engine.dispose()


def plan(session: Session, identity: str, *, status: str, scope: str = "durable") -> None:
    session.add(
        WorkPlan(
            id=identity,
            chat_id="chat",
            transcript_sequence=(
                session.scalar(select(func.max(WorkPlan.transcript_sequence))) or 0
            )
            + 1,
            status=status,
            persistence_scope=scope,
        )
    )
    session.flush()


def legacy(
    session: Session,
    identity: str,
    *,
    status: str,
    target: str | None = None,
    sequence: int | None = None,
    kind: str = "image",
) -> None:
    user = Message(id="user-" + identity, chat_id="chat", role="user")
    answer = Message(
        id="answer-" + identity,
        chat_id="chat",
        role="assistant",
        transcript_visible=target is None,
    )
    session.add_all([user, answer])
    session.flush()
    run = Run(
        id="run-" + identity,
        chat_id="chat",
        user_message_id=user.id,
        assistant_message_id=answer.id,
        status="failed" if status == "interrupted" else status,
    )
    session.add(run)
    session.flush()
    session.add(Job(id="job-" + identity, run_id=run.id, kind=kind, status=status))
    if sequence is not None:
        session.add(
            ResponseRevision(
                id="revision-" + identity,
                message_id=target or answer.id,
                run_id=run.id,
                sequence=sequence,
                status="failed" if status in {"failed", "interrupted"} else "complete",
            )
        )
    session.flush()


def test_plans_count_once_and_keep_partial_failures_until_retry(session: Session) -> None:
    plan(session, "active", status="running")
    plan(session, "partial", status="partial")
    session.add_all(
        [
            Job(id="first", kind="image", status="running", work_plan_id="active"),
            Job(id="second", kind="image", status="queued", work_plan_id="active"),
            WorkStep(
                id="failed-step", plan_id="partial", ordinal=1, operation="image", status="failed"
            ),
            WorkStep(
                id="done-step", plan_id="partial", ordinal=2, operation="image", status="complete"
            ),
        ]
    )
    session.commit()
    assert chat_work_counts(session, ["chat"]) == {"chat": ChatWorkCounts(1, 1)}
    step = session.get(WorkStep, "failed-step")
    assert step is not None
    step.status = "queued"
    retry = session.get(WorkPlan, "partial")
    assert retry is not None
    retry.status = "queued"
    session.commit()
    assert chat_work_counts(session, ["chat"]) == {"chat": ChatWorkCounts(2, 0)}


def test_private_plans_and_internal_jobs_do_not_add_chat_activity(session: Session) -> None:
    plan(session, "private-plan", status="running", scope="incognito")
    plan(session, "complete-plan", status="complete")
    session.add(
        Job(id="verification", kind="edit_verify", status="running", work_plan_id="complete-plan")
    )
    legacy(session, "internal", status="running", kind="edit_verify")
    legacy(session, "visible", status="paused")
    session.commit()
    assert chat_work_counts(session, ["chat"]) == {"chat": ChatWorkCounts(1, 0)}


def test_newer_replacement_resolves_the_older_visible_target_failure(session: Session) -> None:
    legacy(session, "first", status="failed", sequence=1)
    legacy(session, "replacement", status="complete", target="answer-first", sequence=2)
    session.commit()
    assert chat_work_counts(session, ["chat"]) == {"chat": ChatWorkCounts()}
    job = session.get(Job, "job-replacement")
    assert job is not None
    job.status = "interrupted"
    session.commit()
    assert chat_work_counts(session, ["chat"]) == {"chat": ChatWorkCounts(0, 1)}


def test_removed_targets_and_hidden_chats_are_excluded_without_hydration(session: Session) -> None:
    legacy(session, "removed", status="failed")
    message = session.get(Message, "answer-removed")
    assert message is not None
    message.content_removed_at = datetime(2026, 9, 1, tzinfo=UTC)
    session.commit()
    session.expunge_all()
    loaded: list[object] = []

    def track(_session: Session, instance: object) -> None:
        loaded.append(instance)

    event.listen(session, "loaded_as_persistent", track)
    try:
        assert chat_work_counts(session, ["chat"]) == {"chat": ChatWorkCounts()}
    finally:
        event.remove(session, "loaded_as_persistent", track)
    assert loaded == []
    chat = session.get(Chat, "chat")
    assert chat is not None
    chat.scope = "prompt-helper"
    plan(session, "hidden-plan", status="running")
    session.commit()
    assert chat_work_counts(session, ["chat"]) == {"chat": ChatWorkCounts()}


@pytest.mark.parametrize("original_planned", [False, True])
def test_planned_replacement_resolves_a_failure_from_an_earlier_submission(
    session: Session, original_planned: bool
) -> None:
    legacy(session, "first", status="failed", sequence=1)
    legacy(session, "replacement", status="complete", target="answer-first", sequence=2)
    plan(session, "replacement-plan", status="complete")
    replacement = session.get(Run, "run-replacement")
    assert replacement is not None
    replacement.work_plan_id = "replacement-plan"
    if original_planned:
        plan(session, "original-plan", status="failed")
        original = session.get(Run, "run-first")
        assert original is not None
        original.work_plan_id = "original-plan"
    session.commit()
    assert chat_work_counts(session, ["chat"]) == {"chat": ChatWorkCounts()}


def test_multiple_failed_outputs_count_as_one_submission(session: Session) -> None:
    plan(session, "outputs", status="partial")
    for identity in ("one", "two"):
        legacy(session, identity, status="failed", sequence=1)
        run = session.get(Run, "run-" + identity)
        job = session.get(Job, "job-" + identity)
        assert run is not None and job is not None
        run.work_plan_id = "outputs"
        job.work_plan_id = "outputs"
    session.commit()
    assert chat_work_counts(session, ["chat", "chat"]) == {"chat": ChatWorkCounts(0, 1)}


def test_activity_is_bounded_to_the_requested_chat_page(session: Session) -> None:
    plan(session, "active", status="running")
    session.commit()
    assert chat_work_counts(session, []) == {}
    assert chat_work_counts(session, ["missing"]) == {"missing": ChatWorkCounts()}
    with pytest.raises(ValueError, match="Too many chats"):
        chat_work_counts(session, [str(index) for index in range(201)])
