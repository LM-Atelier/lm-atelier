from __future__ import annotations

import pytest

from local_lm.conversation_search_http_message_anchor_v1 import (
    GET_METHOD,
    INVALID_HTTP_ANCHOR,
    ConversationSearchHttpMessageAnchorV1,
    SearchHttpMessageAnchorError,
    bind_search_http_message_anchor,
)

MESSAGE_ID = "abababababababababababababababab"


def _kwargs(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "method": GET_METHOD,
        "path": f"/api/messages/{MESSAGE_ID}/anchor",
        "query_string": "",
    }
    base.update(over)
    return base


def test_anchor_does_not_write_history() -> None:
    bound = bind_search_http_message_anchor(**_kwargs())
    assert bound.method == GET_METHOD
    assert bound.message_id == MESSAGE_ID
    assert bound.fragment == f"msg={MESSAGE_ID}"
    assert bound.changes_active_branch is False
    assert bound.history_write_authorized is False
    assert bound.query_in_url_authorized is False


def test_refuses_post_put_and_one_char_query() -> None:
    with pytest.raises(SearchHttpMessageAnchorError, match=INVALID_HTTP_ANCHOR):
        bind_search_http_message_anchor(**_kwargs(method="POST"))
    with pytest.raises(SearchHttpMessageAnchorError, match=INVALID_HTTP_ANCHOR):
        bind_search_http_message_anchor(**_kwargs(method="PUT"))
    with pytest.raises(SearchHttpMessageAnchorError, match=INVALID_HTTP_ANCHOR):
        bind_search_http_message_anchor(**_kwargs(query_string="q=secret"))
    with pytest.raises(SearchHttpMessageAnchorError, match=INVALID_HTTP_ANCHOR):
        bind_search_http_message_anchor(**_kwargs(query_string="x"))
    with pytest.raises(SearchHttpMessageAnchorError, match=INVALID_HTTP_ANCHOR):
        bind_search_http_message_anchor(**_kwargs(path="/api/messages/m#1/anchor"))


def test_public_constructor_cannot_write_history() -> None:
    with pytest.raises(SearchHttpMessageAnchorError, match=INVALID_HTTP_ANCHOR):
        ConversationSearchHttpMessageAnchorV1()
    with pytest.raises(TypeError):
        ConversationSearchHttpMessageAnchorV1(
            schema="lm-atelier-conversation-search-http-message-anchor-v1",
            schema_version=1,
            history_write_authorized=True,
        )
