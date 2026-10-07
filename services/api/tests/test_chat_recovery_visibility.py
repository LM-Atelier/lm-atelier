"""Deleted chat membership hides list, search and activity reads without losing history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_chat_recovery_graph import graph_session as graph_session

from local_lm.api_errors import ApiError
from local_lm.chat_activity_reads import chat_work_counts
from local_lm.chat_edit_lineage import read_edit_lineage
from local_lm.chat_recovery_visibility import (
    chat_is_deleted,
    job_is_deleted,
    visible_chat,
    visible_job,
)
from local_lm.chat_search_pages import read_search_page
from local_lm.chat_summary_reads import list_chat_summary_rows
from local_lm.models import Chat, Job, RecoveryItem
from local_lm.recovery_v1 import RecoveryState


def _membership(session: Session, state: str = "recoverable", kind: str = "chat") -> None:
    now = datetime.now(UTC)
    session.add(
        RecoveryItem(
            kind=kind,
            subject_id="chat-garden",
            state=state,
            display_label="Garden notes",
            deleted_at=now,
            purge_after=now + timedelta(days=30),
            subject_revision="a" * 64,
        )
    )
    session.commit()


@pytest.mark.parametrize("state", list(RecoveryState))
def test_every_deletion_state_hides_the_chat_before_paging(
    graph_session: Session, state: RecoveryState
) -> None:
    session = graph_session
    garden = session.get(Chat, "chat-garden")
    assert garden is not None
    garden.pinned = True
    session.commit()
    assert [row.id for row in list_chat_summary_rows(session, limit=1)] == ["chat-garden"]
    _membership(session, state.value)
    assert list(session.scalars(select(Chat.id).where(visible_chat(Chat.id)))) == ["chat-other"]
    assert chat_is_deleted(session, "chat-garden")
    assert not chat_is_deleted(session, "chat-other")
    assert [row.id for row in list_chat_summary_rows(session, limit=1)] == ["chat-other"]
    assert list_chat_summary_rows(session, query="Garden") == []
    assert list_chat_summary_rows(session, query="Garden", search_projects=True) == []
    assert session.get(Chat, "chat-garden") is garden
    assert garden.title == "Garden notes"


def test_recovery_membership_kind_cannot_hide_a_different_resource(graph_session: Session) -> None:
    _membership(graph_session, kind="project")
    assert not chat_is_deleted(graph_session, "chat-garden")
    assert {row.id for row in list_chat_summary_rows(graph_session)} == {
        "chat-garden",
        "chat-other",
    }


def test_direct_search_and_edit_lineage_refuse_deleted_chat_ids(graph_session: Session) -> None:
    session = graph_session
    live = read_search_page(
        session,
        "chat-garden",
        head_id=None,
        oldest_message_id=None,
        before=None,
        pending_only=False,
        limit=20,
    )
    assert live is not None
    _membership(session)
    with pytest.raises(ApiError) as searches:
        read_search_page(
            session,
            "chat-garden",
            head_id=None,
            oldest_message_id=None,
            before=None,
            pending_only=False,
            limit=20,
        )
    assert searches.value.status_code == 404 and searches.value.code == "chat-not-found"
    with pytest.raises(ApiError) as lineage:
        read_edit_lineage(session, "chat-garden", "assistant-garden", before=None, limit=20)
    assert lineage.value.status_code == 404 and lineage.value.code == "chat-not-found"


def test_deleted_work_is_not_counted_as_sidebar_failure_activity(graph_session: Session) -> None:
    session = graph_session
    job = session.get(Job, "job-garden")
    assert job is not None
    job.status = "failed"
    session.commit()
    live = chat_work_counts(session, ["chat-garden", "chat-other"])
    assert live["chat-garden"].unresolved_failed_count == 1
    _membership(session)
    hidden = chat_work_counts(session, ["chat-garden", "chat-other"])
    assert hidden["chat-garden"].unresolved_failed_count == 0
    assert hidden["chat-other"] == live["chat-other"]


@pytest.mark.parametrize("binding", ["run", "plan", "step", "chat", "source"])
def test_job_visibility_covers_every_current_owner_and_runless_source(
    graph_session: Session, binding: str
) -> None:
    session = graph_session
    values: dict[str, dict[str, object]] = {
        "run": {"run_id": "run-garden"},
        "plan": {"work_plan_id": "plan-garden"},
        "step": {"work_step_id": "step-garden"},
        "chat": {"payload_json": {"chat_id": "chat-garden"}},
        "source": {"payload_json": {"source_run_id": "run-garden"}},
    }
    session.add(Job(id="owned-job", status="complete", **values[binding]))
    session.commit()
    assert not job_is_deleted(session, "owned-job")
    _membership(session)
    assert job_is_deleted(session, "owned-job")
    assert "owned-job" not in set(session.scalars(select(Job.id).where(visible_job())))


@pytest.mark.parametrize("binding", ["run", "chat"])
def test_a_historical_source_does_not_hide_another_conversations_job(
    graph_session: Session, binding: str
) -> None:
    session = graph_session
    payload = {"source_run_id": "run-garden"}
    if binding == "chat":
        payload["chat_id"] = "chat-other"
    session.add(
        Job(
            id="foreign-job",
            status="complete",
            payload_json=payload,
            run_id="run-other" if binding == "run" else None,
        )
    )
    session.commit()
    _membership(session)
    assert not job_is_deleted(session, "foreign-job")
    assert "foreign-job" in set(session.scalars(select(Job.id).where(visible_job())))
