"""Project recovery preserves canonical history and later filing decisions."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session
from test_project_recovery_schema import project_session as project_session

from local_lm import chat_recovery_graph, project_recovery
from local_lm.db import Base
from local_lm.models import (
    Chat,
    Job,
    Message,
    MessagePart,
    Project,
    RecoveryItem,
    RecoveryOperation,
    Run,
)
from local_lm.project_recovery import (
    preview_project_recovery,
    preview_project_trash,
    purge_project,
    restore_project,
    trash_project,
)
from local_lm.recovery_previews import RecoveryPreviewConflict
from local_lm.recovery_v1 import (
    PurgeRecoveryV1,
    RecoveryCommandV1,
    RecoveryImpactV1,
    RecoveryItemV1,
    RestoreRecoveryV1,
)

NOW = datetime(2026, 10, 2, tzinfo=UTC)


@pytest.mark.parametrize("bound", ["MAX_GRAPH_ROWS", "MAX_GRAPH_BYTES"])
def test_project_preview_refuses_an_oversized_graph_without_creating_a_receipt(
    project_session: Session, monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    session = project_session
    _seed(session)
    before = _rows(session, "projects", "chats", "recovery_previews", "recovery_operations")
    monkeypatch.setattr(chat_recovery_graph, bound, 1)
    with pytest.raises(RecoveryPreviewConflict, match="^project-recovery-graph-invalid$"):
        preview_project_trash(session, "project-garden", NOW)
    session.rollback()
    assert _rows(session, *before) == before


def _command(preview: RecoveryImpactV1, key: str) -> dict[str, str]:
    return {
        "expected_revision": preview.revision,
        "impact_sha256": preview.impact_sha256,
        "operation_key": key,
    }


def _seed(session: Session) -> None:
    session.execute(delete(RecoveryItem))
    project = session.get(Project, "project-garden")
    assert project is not None
    project.instructions = "Use evenly spaced garden beds"
    project.generation_settings_json = {"image": {"seed": 17}}
    session.add(Chat(id="chat-moved", project_id=project.id, title="Moved garden notes"))
    session.add_all(
        [
            Message(id="message-user", chat_id="chat-garden", role="user"),
            Message(id="message-answer", chat_id="chat-garden", role="assistant"),
        ]
    )
    session.flush()
    session.add_all(
        [
            MessagePart(
                id="part-user",
                message_id="message-user",
                position=0,
                text="Keep the garden spacing",
            ),
            Run(
                id="run-garden",
                chat_id="chat-garden",
                user_message_id="message-user",
                assistant_message_id="message-answer",
                status="pending",
            ),
        ]
    )
    session.flush()
    session.add(Job(id="job-garden", kind="chat", status="queued", run_id="run-garden"))
    session.commit()


def _rows(session: Session, *names: str) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            dict(row)
            for row in session.execute(
                select(Base.metadata.tables[name]).order_by(
                    *Base.metadata.tables[name].primary_key.columns
                )
            ).mappings()
        ]
        for name in names
    }


def _trash(session: Session) -> tuple[RecoveryCommandV1, RecoveryItemV1]:
    preview = preview_project_trash(session, "project-garden", NOW)
    session.commit()
    command = RecoveryCommandV1(**_command(preview, "trash-garden"))
    result = trash_project(session, "project-garden", command, NOW)
    session.commit()
    return command, result


@pytest.mark.parametrize("action", ["restore", "purge"])
def test_project_recovery_keeps_history_active_work_and_later_filing_moves(
    project_session: Session, action: str
) -> None:
    session = project_session
    _seed(session)
    history = _rows(session, "messages", "message_parts", "runs", "jobs")
    original = _rows(session, "projects")["projects"][0]
    command, item = _trash(session)
    assert item.counts.chats == 2 and item.delete_generated_media is False
    assert trash_project(session, "project-garden", command, NOW) == item
    session.commit()
    assert _rows(session, "messages", "message_parts", "runs", "jobs") == history
    assert _rows(session, "projects")["projects"][0] == original
    session.execute(update(Chat).where(Chat.id == "chat-moved").values(project_id="project-live"))
    session.commit()
    preview = preview_project_recovery(session, item.deletion_id, NOW)
    assert preview.counts.chats == 1
    session.commit()
    if action == "restore":
        transition = RestoreRecoveryV1(**_command(preview, "restore-garden"))
        result = restore_project(session, item.deletion_id, transition, NOW)
        session.commit()
        assert (
            restore_project(session, item.deletion_id, transition, NOW + timedelta(days=60))
            == result
        )
        session.commit()
        assert _rows(session, "projects")["projects"][0] == original
        expected_project = "project-garden"
    else:
        purge = PurgeRecoveryV1(
            **_command(preview, "purge-garden"), acknowledgement="permanently-delete"
        )
        result = purge_project(session, item.deletion_id, purge, NOW)
        session.commit()
        assert result.reclaimed_bytes == 0
        assert purge_project(session, item.deletion_id, purge, NOW + timedelta(days=60)) == result
        session.commit()
        assert session.get(Project, "project-garden") is None
        expected_project = None
    assert (
        session.scalar(select(Chat.project_id).where(Chat.id == "chat-garden")) == expected_project
    )
    assert session.scalar(select(Chat.project_id).where(Chat.id == "chat-moved")) == "project-live"
    assert _rows(session, "messages", "message_parts", "runs", "jobs") == history
    assert len(session.scalars(select(RecoveryOperation)).all()) == 2


@pytest.mark.parametrize("change", ["configuration", "filing"])
def test_project_trash_refuses_changed_impact_without_partial_deletion(
    project_session: Session, change: str
) -> None:
    session = project_session
    _seed(session)
    preview = preview_project_trash(session, "project-garden", NOW)
    session.commit()
    if change == "configuration":
        session.execute(
            update(Project)
            .where(Project.id == "project-garden")
            .values(instructions="Keep the garden path wide")
        )
    else:
        session.execute(
            update(Chat).where(Chat.id == "chat-moved").values(project_id="project-live")
        )
    session.commit()
    before = _rows(session, "projects", "chats", "messages", "runs", "jobs")
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-impact-stale$"):
        trash_project(
            session, "project-garden", RecoveryCommandV1(**_command(preview, "trash-stale")), NOW
        )
    session.rollback()
    assert _rows(session, "projects", "chats", "messages", "runs", "jobs") == before
    assert session.scalar(select(RecoveryItem)) is None
    assert session.scalar(select(RecoveryOperation)) is None


def test_unrelated_chat_edits_do_not_invalidate_a_project_filing_preview(
    project_session: Session,
) -> None:
    session = project_session
    _seed(session)
    preview = preview_project_trash(session, "project-garden", NOW)
    session.commit()
    session.execute(
        update(Chat)
        .where(Chat.id == "chat-garden")
        .values(draft_prompt="Keep the garden path wide")
    )
    session.commit()
    item = trash_project(
        session, "project-garden", RecoveryCommandV1(**_command(preview, "trash-after-draft")), NOW
    )
    session.commit()
    assert item.counts.chats == 2
    assert (
        session.scalar(select(Chat.draft_prompt).where(Chat.id == "chat-garden"))
        == "Keep the garden path wide"
    )


def test_failed_project_purge_rolls_back_parent_cascades_and_receipts(
    project_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = project_session
    _seed(session)
    _command_trash, item = _trash(session)
    preview = preview_project_recovery(session, item.deletion_id, NOW)
    session.commit()
    before = _rows(
        session,
        "projects",
        "chats",
        "messages",
        "jobs",
        "project_workflow_selections",
        "project_workflow_use_case_selections",
        "recovery_items",
        "recovery_operations",
    )

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ValueError("constructed receipt failure")

    monkeypatch.setattr(project_recovery, "_remember", refuse)
    with pytest.raises(ValueError, match="^constructed receipt failure$"):
        purge_project(
            session,
            item.deletion_id,
            PurgeRecoveryV1(
                **_command(preview, "purge-failed"), acknowledgement="permanently-delete"
            ),
            NOW,
        )
    session.rollback()
    assert _rows(session, *before) == before


def test_project_restore_keeps_the_original_deadline_and_refuses_exact_expiry(
    project_session: Session,
) -> None:
    session = project_session
    _seed(session)
    _command_trash, item = _trash(session)
    assert item.purge_after == NOW + timedelta(days=30)
    preview = preview_project_recovery(session, item.deletion_id, item.purge_after)
    session.commit()
    command = RestoreRecoveryV1(**_command(preview, "restore-expired"))
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-window-expired$"):
        restore_project(session, item.deletion_id, command, item.purge_after)
    session.rollback()
    retained = session.get(RecoveryItem, item.deletion_id)
    assert retained is not None and retained.state == "recoverable"
    assert retained.purge_after.replace(tzinfo=UTC) == item.purge_after


def test_project_purge_clears_a_trashed_child_link_without_restoring_its_history(
    project_session: Session,
) -> None:
    session = project_session
    _seed(session)
    _command_trash, item = _trash(session)
    session.add(
        RecoveryItem(
            kind="chat",
            subject_id="chat-moved",
            display_label="Moved garden notes",
            original_project_id="project-garden",
            deleted_at=NOW,
            purge_after=NOW + timedelta(days=30),
            state="recoverable",
            subject_revision="d" * 64,
        )
    )
    session.commit()
    before = _rows(session, "messages", "runs", "jobs")
    preview = preview_project_recovery(session, item.deletion_id, NOW)
    session.commit()
    purge_project(
        session,
        item.deletion_id,
        PurgeRecoveryV1(
            **_command(preview, "purge-with-child"), acknowledgement="permanently-delete"
        ),
        NOW,
    )
    session.commit()
    assert session.scalar(select(Chat.project_id).where(Chat.id == "chat-moved")) is None
    assert (
        session.scalar(
            select(RecoveryItem.state).where(
                RecoveryItem.kind == "chat", RecoveryItem.subject_id == "chat-moved"
            )
        )
        == "recoverable"
    )
    assert _rows(session, "messages", "runs", "jobs") == before
