"""Chat recovery transitions preserve canonical history and refuse stale commands."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from test_chat_recovery_graph import graph_session as graph_session

from local_lm.chat_recovery import (
    RECOVERY_WINDOW,
    preview_chat_recovery,
    preview_chat_trash,
    purge_chat,
    restore_chat,
    trash_chat,
)
from local_lm.db import Base
from local_lm.models import Artifact, Chat, Job, Project, RecoveryItem, RecoveryOperation
from local_lm.recovery_previews import RecoveryPreviewConflict
from local_lm.recovery_v1 import (
    PurgeRecoveryV1,
    RecoveryAction,
    RecoveryConflict,
    RecoveryImpactV1,
    RecoveryState,
    RestoreRecoveryV1,
    TrashChatV1,
)

NOW = datetime(2026, 10, 2, tzinfo=UTC)


def _fields(impact: RecoveryImpactV1, key: str) -> dict[str, str]:
    return {
        "expected_revision": impact.revision,
        "impact_sha256": impact.impact_sha256,
        "operation_key": key,
    }


def _trash(session: Session, key: str = "trash-garden"):
    preview = preview_chat_trash(session, "chat-garden", NOW)
    session.commit()
    command = TrashChatV1.model_validate(_fields(preview, key))
    result = trash_chat(session, "chat-garden", command, NOW)
    session.commit()
    return result, command


def _history(session: Session):
    names = (
        "chats",
        "messages",
        "message_parts",
        "response_revisions",
        "response_revision_parts",
        "runs",
        "jobs",
        "work_plans",
        "work_steps",
        "chat_composer_drafts",
        "run_context_snapshots",
        "run_context_artifacts",
    )
    return {
        name: tuple(
            session.execute(
                select(Base.metadata.tables[name]).order_by(
                    *Base.metadata.tables[name].primary_key.columns
                )
            )
        )
        for name in names
    }


def test_trash_and_restore_keep_every_canonical_row_and_exact_replay(
    graph_session: Session,
) -> None:
    session = graph_session
    before = _history(session)
    item, command = _trash(session)
    assert _history(session) == before
    assert item.counts.artifacts == 5 and item.purge_after == NOW + RECOVERY_WINDOW
    assert trash_chat(session, "chat-garden", command, NOW) == item
    session.commit()
    preview = preview_chat_recovery(session, item.deletion_id, NOW)
    session.commit()
    restore = RestoreRecoveryV1.model_validate(_fields(preview, "restore-garden"))
    result = restore_chat(session, item.deletion_id, restore, NOW)
    session.commit()
    assert result.action == RecoveryAction.RESTORE
    assert session.get(RecoveryItem, item.deletion_id) is None
    assert _history(session) == before
    assert restore_chat(session, item.deletion_id, restore, NOW) == result
    session.commit()
    assert _history(session) == before


@pytest.mark.parametrize(
    ("status", "active_work"),
    [("queued", 1), ("running", 1), ("paused", 1), ("interrupted", 0), ("unknown", 1)],
)
def test_trash_preserves_runless_work_and_refuses_active_jobs(
    graph_session: Session, status: str, active_work: int
) -> None:
    session = graph_session
    session.add(Job(id="verification", status=status, payload_json={"source_run_id": "run-garden"}))
    session.commit()
    before = _history(session)
    preview = preview_chat_trash(session, "chat-garden", NOW)
    session.commit()
    assert preview.counts.active_work == active_work
    if active_work:
        assert preview.available_actions == ()
        with pytest.raises(RecoveryPreviewConflict, match="^chat-recovery-active-work$"):
            trash_chat(
                session, "chat-garden", TrashChatV1.model_validate(_fields(preview, "busy")), NOW
            )
        session.rollback()
        assert not list(session.scalars(select(RecoveryItem)))
    else:
        assert preview.available_actions == (RecoveryAction.TRASH,)
        item = trash_chat(
            session, "chat-garden", TrashChatV1.model_validate(_fields(preview, "terminal")), NOW
        )
        session.commit()
        assert item.state == RecoveryState.RECOVERABLE
    assert _history(session) == before


def test_changed_private_content_and_changed_media_intent_refuse_trash(
    graph_session: Session,
) -> None:
    session = graph_session
    preview = preview_chat_trash(session, "chat-garden", NOW)
    session.commit()
    command = TrashChatV1.model_validate(
        {**_fields(preview, "different-intent"), "delete_generated_media": True}
    )
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-impact-stale$"):
        trash_chat(session, "chat-garden", command, NOW)
    session.rollback()
    chat = session.get(Chat, "chat-garden")
    assert chat is not None
    chat.title = "Updated garden notes"
    session.commit()
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-impact-stale$"):
        trash_chat(
            session,
            "chat-garden",
            TrashChatV1.model_validate(_fields(preview, "changed-title")),
            NOW,
        )
    session.rollback()
    assert not list(session.scalars(select(RecoveryItem)))


def test_operation_key_cannot_be_reused_for_changed_intent_or_action(
    graph_session: Session,
) -> None:
    session = graph_session
    item, command = _trash(session)
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-operation-conflict$"):
        trash_chat(
            session, "chat-garden", command.model_copy(update={"delete_generated_media": True}), NOW
        )
    session.rollback()
    preview = preview_chat_recovery(session, item.deletion_id, NOW)
    session.commit()
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-operation-conflict$"):
        restore_chat(
            session,
            item.deletion_id,
            RestoreRecoveryV1.model_validate(_fields(preview, command.operation_key)),
            NOW,
        )
    session.rollback()
    assert session.get(RecoveryItem, item.deletion_id) is not None


def test_purge_deletes_owned_jobs_before_the_graph_and_preserves_foreign_history(
    graph_session: Session,
) -> None:
    session = graph_session
    session.add(
        Job(
            id="job-other",
            run_id="run-other",
            status="complete",
            payload_json={"source_run_id": "run-garden"},
        )
    )
    session.add(
        Job(id="job-source", status="complete", payload_json={"source_run_id": "run-garden"})
    )
    session.commit()
    other = session.get(Chat, "chat-other")
    assert other is not None
    other_title = other.title
    item, _ = _trash(session)
    preview = preview_chat_recovery(session, item.deletion_id, NOW)
    session.commit()
    command = PurgeRecoveryV1.model_validate(
        {**_fields(preview, "purge-garden"), "acknowledgement": "permanently-delete"}
    )
    result = purge_chat(session, item.deletion_id, command, NOW)
    session.commit()
    assert result.reclaimed_bytes == 0 and result.action == RecoveryAction.PURGE
    assert session.get(Chat, "chat-garden") is None
    assert session.get(Job, "job-garden") is None and session.get(Job, "job-source") is None
    assert session.get(Job, "job-other") is not None
    assert session.get(Chat, "chat-other").title == other_title
    assert len(list(session.scalars(select(Artifact)))) == 5
    membership = session.get(RecoveryItem, item.deletion_id)
    assert membership is not None and membership.state == RecoveryState.PURGED.value
    assert purge_chat(session, item.deletion_id, command, NOW) == result
    session.commit()


def test_transition_never_commits_the_callers_transaction(graph_session: Session) -> None:
    session = graph_session
    before = _history(session)
    preview = preview_chat_trash(session, "chat-garden", NOW)
    session.commit()
    trash_chat(
        session, "chat-garden", TrashChatV1.model_validate(_fields(preview, "rollback-trash")), NOW
    )
    session.rollback()
    assert _history(session) == before
    assert not list(session.scalars(select(RecoveryItem)))
    assert not list(session.scalars(select(RecoveryOperation)))


def test_missing_original_project_needs_an_explicit_unfiled_restore(graph_session: Session) -> None:
    session = graph_session
    session.add(Project(id="garden-project", name="Garden project"))
    session.flush()
    chat = session.get(Chat, "chat-garden")
    assert chat is not None
    chat.project_id = "garden-project"
    session.commit()
    item, _ = _trash(session)
    session.execute(delete(Project).where(Project.id == "garden-project"))
    session.commit()
    preview = preview_chat_recovery(session, item.deletion_id, NOW)
    session.commit()
    assert RecoveryConflict.ORIGINAL_PROJECT_MISSING in preview.conflicts
    fields = _fields(preview, "unfiled-restore")
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-original-project-missing$"):
        restore_chat(session, item.deletion_id, RestoreRecoveryV1.model_validate(fields), NOW)
    session.rollback()
    result = restore_chat(
        session,
        item.deletion_id,
        RestoreRecoveryV1.model_validate({**fields, "restore_unfiled": True}),
        NOW,
    )
    session.commit()
    assert result.action == RecoveryAction.RESTORE
    assert session.get(Chat, "chat-garden").project_id is None


def test_expired_window_cannot_restore_but_can_be_previewed_for_purge(
    graph_session: Session,
) -> None:
    session = graph_session
    item, _ = _trash(session)
    expiry = NOW + RECOVERY_WINDOW
    preview = preview_chat_recovery(session, item.deletion_id, expiry)
    session.commit()
    assert preview.available_actions == (RecoveryAction.PURGE,)
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-window-expired$"):
        restore_chat(
            session,
            item.deletion_id,
            RestoreRecoveryV1.model_validate(_fields(preview, "expired")),
            expiry,
        )
    session.rollback()
    assert session.get(RecoveryItem, item.deletion_id) is not None
