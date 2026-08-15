from __future__ import annotations

import pytest

from local_lm.chat_item_removal_replay_refuse_v1 import (
    INVALID_REPLAY_REFUSE,
    MAX_ID,
    ChatItemRemovalReplayRefuseError,
    ChatItemRemovalReplayRefuseV1,
    declare_chat_item_removal_replay_refuse,
)


def test_records_requested_policy_without_omitting_context() -> None:
    refuse = declare_chat_item_removal_replay_refuse(message_id="m1")
    assert refuse.requested_policy == "source_content_removed"
    assert refuse.replay_authorized is False
    assert refuse.context_omission_verified is False
    assert refuse.forensic_erasure is False


def test_refuses_unbounded_ids() -> None:
    with pytest.raises(ChatItemRemovalReplayRefuseError, match=INVALID_REPLAY_REFUSE):
        declare_chat_item_removal_replay_refuse(message_id="bad id")
    with pytest.raises(ChatItemRemovalReplayRefuseError, match=INVALID_REPLAY_REFUSE):
        declare_chat_item_removal_replay_refuse(message_id="m" * (MAX_ID + 1))
    with pytest.raises(ChatItemRemovalReplayRefuseError, match=INVALID_REPLAY_REFUSE):
        declare_chat_item_removal_replay_refuse(message_id="m" * 10_000)
    with pytest.raises(ChatItemRemovalReplayRefuseError, match=INVALID_REPLAY_REFUSE):
        declare_chat_item_removal_replay_refuse(message_id=True)


def test_public_constructor_cannot_authorize_replay() -> None:
    with pytest.raises(ChatItemRemovalReplayRefuseError, match=INVALID_REPLAY_REFUSE):
        ChatItemRemovalReplayRefuseV1()
    with pytest.raises(TypeError):
        ChatItemRemovalReplayRefuseV1(
            schema="lm-atelier-chat-item-removal-replay-refuse-v1",
            schema_version=1,
            message_id="m1",
            replay_authorized=True,
        )
