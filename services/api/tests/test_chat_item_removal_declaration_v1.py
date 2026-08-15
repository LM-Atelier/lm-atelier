from __future__ import annotations

import pytest

from local_lm.chat_item_removal_declaration_v1 import (
    INVALID_REMOVAL,
    MAX_ID,
    ChatItemRemovalDeclarationError,
    ChatItemRemovalDeclarationV1,
    declare_chat_item_removal,
)


def _facts(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "message_id": "m1",
        "expected_message_id": "m1",
        "event_revision_id": "e1",
        "expected_event_revision_id": "e1",
        "idempotency_key": "k1",
        "has_replies": True,
        "content_already_removed": False,
        "chat_has_active_work": False,
        "parts_count": 2,
        "references_count": 1,
        "revision_parts_count": 1,
        "source_content_backs_regeneration": False,
        "same_chat": True,
    }
    base.update(over)
    return base


def test_candidate_never_authorizes() -> None:
    declaration = declare_chat_item_removal(_facts())
    assert declaration.candidate_code == "candidate_content_tombstone"
    assert declaration.allowed is False
    assert declaration.message_id_bound is False
    assert declaration.repository_snapshot_verified is False
    assert declaration.execution_authorized is False
    assert declaration.future_context_omission_verified is False
    assert declaration.planned_preserve_message_id is True
    assert declaration.cascade_delete_replies is False


def test_refuses() -> None:
    assert (
        declare_chat_item_removal(_facts(chat_has_active_work=True)).candidate_code
        == "refuse_active_work"
    )
    assert (
        declare_chat_item_removal(_facts(content_already_removed=True)).candidate_code
        == "refuse_already_removed"
    )
    assert (
        declare_chat_item_removal(_facts(message_id="other")).candidate_code
        == "refuse_identity_mismatch"
    )
    assert (
        declare_chat_item_removal(_facts(same_chat=False)).candidate_code
        == "refuse_identity_mismatch"
    )


def test_invalid() -> None:
    with pytest.raises(ChatItemRemovalDeclarationError, match=INVALID_REMOVAL):
        declare_chat_item_removal(_facts(parts_count=-1))
    with pytest.raises(ChatItemRemovalDeclarationError, match=INVALID_REMOVAL):
        declare_chat_item_removal(_facts(message_id="bad id"))
    with pytest.raises(ChatItemRemovalDeclarationError, match=INVALID_REMOVAL):
        declare_chat_item_removal({"message_id": "m1"})
    with pytest.raises(ChatItemRemovalDeclarationError, match=INVALID_REMOVAL):
        declare_chat_item_removal(_facts(message_id="m" * (MAX_ID + 1)))
    with pytest.raises(ChatItemRemovalDeclarationError, match=INVALID_REMOVAL):
        declare_chat_item_removal(_facts(message_id="m" * 10_000))
    with pytest.raises(ChatItemRemovalDeclarationError, match=INVALID_REMOVAL):
        declare_chat_item_removal(_facts(parts_count=True))


def test_public_constructor_cannot_authorize() -> None:
    with pytest.raises(ChatItemRemovalDeclarationError, match=INVALID_REMOVAL):
        ChatItemRemovalDeclarationV1()
    with pytest.raises(TypeError):
        ChatItemRemovalDeclarationV1(
            schema="lm-atelier-chat-item-removal-declaration-v1",
            schema_version=1,
            allowed=True,
        )
