from __future__ import annotations

import pytest

from local_lm.conversation_search_http_message_window_v1 import (
    INVALID_HTTP_MESSAGE_WINDOW,
    POST_METHOD,
    ConversationSearchHttpMessageWindowV1,
    SearchHttpMessageWindowError,
    bind_search_http_message_window,
)
from local_lm.message_window_v1 import MAX_INPUT_IDS

IDS = ("msg-01", "msg-02", "msg-03", "msg-04", "msg-05")


def _kwargs(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "method": POST_METHOD,
        "path": "/api/conversation-search/message-window",
        "query_string": "",
        "ordered_ids": IDS,
        "mode": "latest",
        "limit": 3,
    }
    base.update(over)
    return base


def test_message_window_does_not_load_or_activate() -> None:
    bound = bind_search_http_message_window(**_kwargs())
    assert bound.method == POST_METHOD
    assert bound.mode == "latest"
    assert bound.message_ids == ("msg-03", "msg-04", "msg-05")
    assert bound.anchor_id is None
    assert bound.loads_entire_chat is False
    assert bound.branch_activation_authorized is False
    assert bound.query_in_url_authorized is False


def test_refuses_get_head_one_char_query_and_cap_plus_one_ids() -> None:
    with pytest.raises(SearchHttpMessageWindowError, match=INVALID_HTTP_MESSAGE_WINDOW):
        bind_search_http_message_window(**_kwargs(method="GET"))
    with pytest.raises(SearchHttpMessageWindowError, match=INVALID_HTTP_MESSAGE_WINDOW):
        bind_search_http_message_window(**_kwargs(method="HEAD"))
    with pytest.raises(SearchHttpMessageWindowError, match=INVALID_HTTP_MESSAGE_WINDOW):
        bind_search_http_message_window(**_kwargs(query_string="q=secret"))
    with pytest.raises(SearchHttpMessageWindowError, match=INVALID_HTTP_MESSAGE_WINDOW):
        bind_search_http_message_window(**_kwargs(query_string="x"))
    oversize = tuple(f"msg-{i:03d}" for i in range(MAX_INPUT_IDS + 1))
    with pytest.raises(SearchHttpMessageWindowError, match=INVALID_HTTP_MESSAGE_WINDOW):
        bind_search_http_message_window(**_kwargs(ordered_ids=oversize))


def test_public_constructor_cannot_load_chat() -> None:
    with pytest.raises(SearchHttpMessageWindowError, match=INVALID_HTTP_MESSAGE_WINDOW):
        ConversationSearchHttpMessageWindowV1()
    with pytest.raises(TypeError):
        ConversationSearchHttpMessageWindowV1(
            schema="lm-atelier-conversation-search-http-message-window-v1",
            schema_version=1,
            loads_entire_chat=True,
        )
