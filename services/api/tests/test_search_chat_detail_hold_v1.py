from __future__ import annotations

import pytest

from local_lm.search_chat_detail_hold_v1 import (
    INVALID_HOLD,
    MAX_CONSUMERS,
    SearchChatDetailHoldError,
    SearchChatDetailHoldV1,
    declare_search_chat_detail_hold,
)


def test_records_requested_count_without_removal() -> None:
    hold = declare_search_chat_detail_hold(requested_consumer_count=3)
    assert hold.requested_policy == "retain_legacy_chat_detail"
    assert hold.requested_consumer_count == 3
    assert hold.removal_authorized is False
    assert not hasattr(hold, "remaining_consumers")
    assert not hasattr(hold, "loads_entire_chat")
    assert not hasattr(hold, "legacy_endpoint_retained")


def test_refuses_empty_and_unbounded_counts() -> None:
    with pytest.raises(SearchChatDetailHoldError, match=INVALID_HOLD):
        declare_search_chat_detail_hold(requested_consumer_count=0)
    with pytest.raises(SearchChatDetailHoldError, match=INVALID_HOLD):
        declare_search_chat_detail_hold(requested_consumer_count=MAX_CONSUMERS + 1)
    with pytest.raises(SearchChatDetailHoldError, match=INVALID_HOLD):
        declare_search_chat_detail_hold(requested_consumer_count=True)


def test_public_constructor_cannot_authorize_removal() -> None:
    with pytest.raises(SearchChatDetailHoldError, match=INVALID_HOLD):
        SearchChatDetailHoldV1()
    with pytest.raises(TypeError):
        SearchChatDetailHoldV1(
            schema="lm-atelier-search-chat-detail-hold-v1",
            schema_version=1,
            requested_consumer_count=1,
            removal_authorized=True,
        )
