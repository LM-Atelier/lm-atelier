from __future__ import annotations

import pytest

from local_lm.conversation_search_http_snippet_policy_v1 import (
    INVALID_HTTP_SNIPPET_POLICY,
    POST_METHOD,
    ConversationSearchHttpSnippetPolicyV1,
    SearchHttpSnippetPolicyError,
    bind_search_http_snippet_policy,
)


def _kwargs(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "method": POST_METHOD,
        "path": "/api/conversation-search/snippet-policy",
        "query_string": "",
        "body": "prefix hello world suffix",
        "query": "hello",
    }
    base.update(over)
    return base


def test_snippet_policy_does_not_emit_html_or_load() -> None:
    bound = bind_search_http_snippet_policy(**_kwargs())
    assert bound.method == POST_METHOD
    assert bound.segment_count >= 1
    assert bound.html_authorized is False
    assert bound.loads_entire_chat is False
    assert bound.query_in_url_authorized is False


def test_refuses_get_head_one_char_query_and_empty_body() -> None:
    with pytest.raises(SearchHttpSnippetPolicyError, match=INVALID_HTTP_SNIPPET_POLICY):
        bind_search_http_snippet_policy(**_kwargs(method="GET"))
    with pytest.raises(SearchHttpSnippetPolicyError, match=INVALID_HTTP_SNIPPET_POLICY):
        bind_search_http_snippet_policy(**_kwargs(method="HEAD"))
    with pytest.raises(SearchHttpSnippetPolicyError, match=INVALID_HTTP_SNIPPET_POLICY):
        bind_search_http_snippet_policy(**_kwargs(query_string="q=secret"))
    with pytest.raises(SearchHttpSnippetPolicyError, match=INVALID_HTTP_SNIPPET_POLICY):
        bind_search_http_snippet_policy(**_kwargs(query_string="x"))
    with pytest.raises(SearchHttpSnippetPolicyError, match=INVALID_HTTP_SNIPPET_POLICY):
        bind_search_http_snippet_policy(**_kwargs(body=""))


def test_public_constructor_cannot_emit_html() -> None:
    with pytest.raises(SearchHttpSnippetPolicyError, match=INVALID_HTTP_SNIPPET_POLICY):
        ConversationSearchHttpSnippetPolicyV1()
    with pytest.raises(TypeError):
        ConversationSearchHttpSnippetPolicyV1(
            schema="lm-atelier-conversation-search-http-snippet-policy-v1",
            schema_version=1,
            html_authorized=True,
        )
