"""Pure conversation-search HTTP snippet-policy facts.

Binds POST /api/conversation-search/snippet-policy with an empty
query string. Passes body and query through the parent bound-string
check. Never copies a collection or emits HTML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

from .search_snippet_v1 import SearchSnippetError, SearchSnippetV1, build_search_snippet
from .search_text_v1 import require_bounded_exact_str

SCHEMA_ID: Final = "lm-atelier-conversation-search-http-snippet-policy-v1"
SCHEMA_VERSION: Final = 1
INVALID_HTTP_SNIPPET_POLICY: Final = "search http-snippet-policy facts are invalid"
POST_METHOD: Final = "POST"
METHODS: Final = frozenset({POST_METHOD})
SNIPPET_PATH: Final = "/api/conversation-search/snippet-policy"
_HTTP_WITNESS = object()


class SearchHttpSnippetPolicyError(ValueError):
    """Fixed non-echoing refusal for invalid HTTP snippet-policy facts."""


@dataclass(frozen=True, slots=True)
class ConversationSearchHttpSnippetPolicyV1:
    schema: Literal["lm-atelier-conversation-search-http-snippet-policy-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    method: Literal["POST"] = field(init=False)
    segment_count: int = field(init=False)
    html_authorized: Literal[False] = field(init=False)
    loads_entire_chat: Literal[False] = field(init=False)
    query_in_url_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise SearchHttpSnippetPolicyError(INVALID_HTTP_SNIPPET_POLICY)


def _invalid() -> NoReturn:
    raise SearchHttpSnippetPolicyError(INVALID_HTTP_SNIPPET_POLICY)


def _refuse_surrogates(text: str) -> None:
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        _invalid()


def _require_method(value: object) -> str:
    text = require_bounded_exact_str(value, max_len=len(POST_METHOD), refuse=_invalid)
    _refuse_surrogates(text)
    if text not in METHODS:
        _invalid()
    return text


def _require_empty_query_string(value: object) -> str:
    if value is None:
        return ""
    text = require_bounded_exact_str(value, max_len=1, refuse=_invalid, allow_empty=True)
    _refuse_surrogates(text)
    if text:
        _invalid()
    return text


def _require_path(value: object) -> str:
    text = require_bounded_exact_str(value, max_len=len(SNIPPET_PATH), refuse=_invalid)
    _refuse_surrogates(text)
    if any(ch in "#=?&%" for ch in text):
        _invalid()
    if text != SNIPPET_PATH:
        _invalid()
    return text


def _http_from_evaluator(
    *,
    witness: object,
    method: str,
    snippet: SearchSnippetV1,
) -> ConversationSearchHttpSnippetPolicyV1:
    if witness is not _HTTP_WITNESS:
        _invalid()
    bound = object.__new__(ConversationSearchHttpSnippetPolicyV1)
    object.__setattr__(bound, "schema", SCHEMA_ID)
    object.__setattr__(bound, "schema_version", SCHEMA_VERSION)
    object.__setattr__(bound, "method", method)
    object.__setattr__(bound, "segment_count", len(snippet.segments))
    object.__setattr__(bound, "html_authorized", False)
    object.__setattr__(bound, "loads_entire_chat", False)
    object.__setattr__(bound, "query_in_url_authorized", False)
    return bound


def bind_search_http_snippet_policy(
    *,
    method: object,
    path: object,
    query_string: object = "",
    body: object,
    query: object,
    radius: object = 48,
) -> ConversationSearchHttpSnippetPolicyV1:
    """Bind snippet segments without copying a collection."""
    bound_method = _require_method(method)
    _require_path(path)
    _require_empty_query_string(query_string)
    try:
        snippet = build_search_snippet(body, query, radius=radius)
    except SearchSnippetError:
        _invalid()
    return _http_from_evaluator(witness=_HTTP_WITNESS, method=bound_method, snippet=snippet)
