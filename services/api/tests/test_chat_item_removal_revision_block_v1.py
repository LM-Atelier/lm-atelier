from __future__ import annotations

import pytest

from local_lm.chat_item_removal_revision_block_v1 import (
    INVALID_REVISION_BLOCK,
    MAX_ID,
    ChatItemRemovalRevisionBlockError,
    ChatItemRemovalRevisionBlockV1,
    declare_chat_item_removal_revision_block,
)


def test_records_requested_policy_without_blocking() -> None:
    block = declare_chat_item_removal_revision_block(message_id="m1")
    assert block.requested_policy == "block_revision_repopulation"
    assert block.revision_repopulation_authorized is False
    assert block.execution_authorized is False


def test_refuses_unbounded_ids() -> None:
    with pytest.raises(ChatItemRemovalRevisionBlockError, match=INVALID_REVISION_BLOCK):
        declare_chat_item_removal_revision_block(message_id="bad id")
    with pytest.raises(ChatItemRemovalRevisionBlockError, match=INVALID_REVISION_BLOCK):
        declare_chat_item_removal_revision_block(message_id="m" * (MAX_ID + 1))
    with pytest.raises(ChatItemRemovalRevisionBlockError, match=INVALID_REVISION_BLOCK):
        declare_chat_item_removal_revision_block(message_id="m" * 10_000)
    with pytest.raises(ChatItemRemovalRevisionBlockError, match=INVALID_REVISION_BLOCK):
        declare_chat_item_removal_revision_block(message_id=True)


def test_public_constructor_cannot_authorize_repopulation() -> None:
    with pytest.raises(ChatItemRemovalRevisionBlockError, match=INVALID_REVISION_BLOCK):
        ChatItemRemovalRevisionBlockV1()
    with pytest.raises(TypeError):
        ChatItemRemovalRevisionBlockV1(
            schema="lm-atelier-chat-item-removal-revision-block-v1",
            schema_version=1,
            message_id="m1",
            revision_repopulation_authorized=True,
        )
