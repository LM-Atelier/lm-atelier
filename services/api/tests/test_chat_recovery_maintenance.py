"""Empty-chat cleanup cannot bypass the recovery window."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session
from test_empty_chat_deletion import session as session

from local_lm.empty_chats import (
    EmptyChatExecuteError,
    EmptyChatExecuteRefusal,
    EmptyChatSelectionFilters,
    empty_chat_page,
    execute_deletion,
    issue_preview,
    preview_selection,
)
from local_lm.models import Chat, RecoveryItem
from local_lm.recovery_v1 import RecoveryState

NOW = datetime(2026, 10, 2, tzinfo=UTC)
FILTERS = EmptyChatSelectionFilters(
    minimum_age_hours=0, include_archived=True, include_configured=True
)


def _membership(session: Session, state: str) -> RecoveryItem:
    item = RecoveryItem(
        kind="chat",
        subject_id="chat-recovery",
        display_label="Garden notes",
        state=state,
        deleted_at=NOW,
        purge_after=NOW + timedelta(days=30),
        subject_revision="a" * 64,
    )
    session.add(item)
    session.commit()
    return item


@pytest.mark.parametrize("state", list(RecoveryState))
def test_deleted_blank_chats_are_excluded_before_cleanup_paging(
    session: Session, state: RecoveryState
) -> None:
    session.add_all([Chat(id="chat-recovery"), Chat(id="chat-unrelated")])
    session.commit()
    live = empty_chat_page(session, limit=1, include_configured=True)
    assert [entry.chat_id for entry in live.entries] == ["chat-recovery"]
    item = _membership(session, state.value)
    hidden = empty_chat_page(session, limit=1, include_configured=True)
    assert [entry.chat_id for entry in hidden.entries] == ["chat-unrelated"]
    preview = preview_selection(session, chat_ids=["chat-recovery"], filters=FILTERS, now=NOW)
    assert preview.strict_count == preview.configured_count == 0
    assert [(entry.chat_id, entry.reason) for entry in preview.conflicts] == [
        ("chat-recovery", "missing")
    ]
    assert session.get(Chat, "chat-recovery") is not None
    assert session.get(RecoveryItem, item.deletion_id) is item


def test_a_cleanup_preview_is_invalidated_by_later_recovery_membership(session: Session) -> None:
    session.add_all([Chat(id="chat-recovery"), Chat(id="chat-unrelated")])
    session.commit()
    identities = ["chat-recovery", "chat-unrelated"]
    preview = preview_selection(session, chat_ids=identities, filters=FILTERS, now=NOW)
    assert not preview.conflicts
    preview_id, _ = issue_preview(
        session, digest=preview.digest, chat_states=preview.chat_states, now=NOW
    )
    session.commit()
    item = _membership(session, "recoverable")
    with pytest.raises(EmptyChatExecuteError) as refused:
        execute_deletion(
            session,
            operation_id="cleanup-before-trash",
            preview_id=preview_id,
            digest=preview.digest,
            acknowledged_count=2,
            acknowledged_configured=True,
            chat_ids=identities,
            filters=FILTERS,
            now=NOW,
        )
    assert refused.value.refusal == EmptyChatExecuteRefusal.SELECTION_DRIFTED
    session.rollback()
    assert session.get(Chat, "chat-recovery") is not None
    assert session.get(Chat, "chat-unrelated") is not None
    assert session.get(RecoveryItem, item.deletion_id) is not None
