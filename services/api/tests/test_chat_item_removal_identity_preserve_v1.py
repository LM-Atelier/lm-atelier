from __future__ import annotations

import pytest

from local_lm.chat_item_removal_identity_preserve_v1 import (
    INVALID_IDENTITY_PRESERVE,
    MAX_ID,
    ChatItemRemovalIdentityPreserveError,
    ChatItemRemovalIdentityPreserveV1,
    declare_chat_item_removal_identity_preserve,
)


def test_records_requested_policy_without_deleting() -> None:
    preserve = declare_chat_item_removal_identity_preserve(message_id="m1")
    assert preserve.requested_policy == "preserve_message_id"
    assert preserve.message_delete_authorized is False
    assert preserve.identity_rewrite_authorized is False


def test_refuses_unbounded_ids() -> None:
    with pytest.raises(ChatItemRemovalIdentityPreserveError, match=INVALID_IDENTITY_PRESERVE):
        declare_chat_item_removal_identity_preserve(message_id="bad id")
    with pytest.raises(ChatItemRemovalIdentityPreserveError, match=INVALID_IDENTITY_PRESERVE):
        declare_chat_item_removal_identity_preserve(message_id="m" * (MAX_ID + 1))
    with pytest.raises(ChatItemRemovalIdentityPreserveError, match=INVALID_IDENTITY_PRESERVE):
        declare_chat_item_removal_identity_preserve(message_id="m" * 10_000)
    with pytest.raises(ChatItemRemovalIdentityPreserveError, match=INVALID_IDENTITY_PRESERVE):
        declare_chat_item_removal_identity_preserve(message_id=True)


def test_public_constructor_cannot_authorize_delete() -> None:
    with pytest.raises(ChatItemRemovalIdentityPreserveError, match=INVALID_IDENTITY_PRESERVE):
        ChatItemRemovalIdentityPreserveV1()
    with pytest.raises(TypeError):
        ChatItemRemovalIdentityPreserveV1(
            schema="lm-atelier-chat-item-removal-identity-preserve-v1",
            schema_version=1,
            message_id="m1",
            message_delete_authorized=True,
        )
