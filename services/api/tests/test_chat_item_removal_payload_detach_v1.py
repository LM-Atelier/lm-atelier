from __future__ import annotations

import pytest

from local_lm.chat_item_removal_payload_detach_v1 import (
    INVALID_PAYLOAD_DETACH,
    MAX_ID,
    ChatItemRemovalPayloadDetachError,
    ChatItemRemovalPayloadDetachV1,
    declare_chat_item_removal_payload_detach,
)


def test_records_requested_policy_without_detaching() -> None:
    detach = declare_chat_item_removal_payload_detach(message_id="m1")
    assert detach.requested_policy == "detach_payload_parts"
    assert detach.payload_detach_authorized is False
    assert detach.message_delete_authorized is False
    assert detach.forensic_erasure is False


def test_refuses_unbounded_ids() -> None:
    with pytest.raises(ChatItemRemovalPayloadDetachError, match=INVALID_PAYLOAD_DETACH):
        declare_chat_item_removal_payload_detach(message_id="bad id")
    with pytest.raises(ChatItemRemovalPayloadDetachError, match=INVALID_PAYLOAD_DETACH):
        declare_chat_item_removal_payload_detach(message_id="m" * (MAX_ID + 1))
    with pytest.raises(ChatItemRemovalPayloadDetachError, match=INVALID_PAYLOAD_DETACH):
        declare_chat_item_removal_payload_detach(message_id="m" * 10_000)
    with pytest.raises(ChatItemRemovalPayloadDetachError, match=INVALID_PAYLOAD_DETACH):
        declare_chat_item_removal_payload_detach(message_id=True)


def test_public_constructor_cannot_authorize_detach() -> None:
    with pytest.raises(ChatItemRemovalPayloadDetachError, match=INVALID_PAYLOAD_DETACH):
        ChatItemRemovalPayloadDetachV1()
    with pytest.raises(TypeError):
        ChatItemRemovalPayloadDetachV1(
            schema="lm-atelier-chat-item-removal-payload-detach-v1",
            schema_version=1,
            message_id="m1",
            payload_detach_authorized=True,
        )
