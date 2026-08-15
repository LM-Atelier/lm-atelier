from __future__ import annotations

import pytest

from local_lm.chat_item_removal_reply_preserve_v1 import (
    INVALID_REPLY_PRESERVE,
    MAX_ID,
    ChatItemRemovalReplyPreserveError,
    ChatItemRemovalReplyPreserveV1,
    declare_chat_item_removal_reply_preserve,
)


def test_records_requested_policy_without_rewriting_edges() -> None:
    preserve = declare_chat_item_removal_reply_preserve(message_id="m1")
    assert preserve.requested_policy == "preserve_reply_edges"
    assert preserve.reply_rewrite_authorized is False
    assert preserve.cascade_delete_replies is False


def test_refuses_unbounded_ids() -> None:
    with pytest.raises(ChatItemRemovalReplyPreserveError, match=INVALID_REPLY_PRESERVE):
        declare_chat_item_removal_reply_preserve(message_id="bad id")
    with pytest.raises(ChatItemRemovalReplyPreserveError, match=INVALID_REPLY_PRESERVE):
        declare_chat_item_removal_reply_preserve(message_id="m" * (MAX_ID + 1))
    with pytest.raises(ChatItemRemovalReplyPreserveError, match=INVALID_REPLY_PRESERVE):
        declare_chat_item_removal_reply_preserve(message_id="m" * 10_000)
    with pytest.raises(ChatItemRemovalReplyPreserveError, match=INVALID_REPLY_PRESERVE):
        declare_chat_item_removal_reply_preserve(message_id=True)


def test_public_constructor_cannot_authorize_cascade() -> None:
    with pytest.raises(ChatItemRemovalReplyPreserveError, match=INVALID_REPLY_PRESERVE):
        ChatItemRemovalReplyPreserveV1()
    with pytest.raises(TypeError):
        ChatItemRemovalReplyPreserveV1(
            schema="lm-atelier-chat-item-removal-reply-preserve-v1",
            schema_version=1,
            message_id="m1",
            cascade_delete_replies=True,
        )
