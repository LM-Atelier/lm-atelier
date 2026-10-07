"""Pure conversation-search HTTP message-anchor facts.

Binds GET /api/messages/{message_id}/anchor with an empty query
string. Never writes history or changes the active branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

from .message_anchor_v1 import (
    MAX_ID_CHARS,
    AnchorError,
    MessageAnchorV1,
    parse_message_anchor,
)
from .search_text_v1 import require_bounded_exact_str

SCHEMA_ID: Final = "lm-atelier-conversation-search-http-message-anchor-v1"
SCHEMA_VERSION: Final = 1
INVALID_HTTP_ANCHOR: Final = "search http-message-anchor facts are invalid"
GET_METHOD: Final = "GET"
METHODS: Final = frozenset({GET_METHOD})
PATH_PREFIX: Final = "/api/messages/"
PATH_SUFFIX: Final = "/anchor"
PATH_CEILING: Final = len(PATH_PREFIX) + MAX_ID_CHARS + len(PATH_SUFFIX)
_HTTP_WITNESS = object()


class SearchHttpMessageAnchorError(ValueError):
    """Fixed non-echoing refusal for invalid HTTP message-anchor facts."""


@dataclass(frozen=True, slots=True)
class ConversationSearchHttpMessageAnchorV1:
    schema: Literal["lm-atelier-conversation-search-http-message-anchor-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    method: Literal["GET"] = field(init=False)
    message_id: str = field(init=False)
    fragment: str = field(init=False)
    changes_active_branch: Literal[False] = field(init=False)
    history_write_authorized: Literal[False] = field(init=False)
    query_in_url_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise SearchHttpMessageAnchorError(INVALID_HTTP_ANCHOR)


def _invalid() -> NoReturn:
    raise SearchHttpMessageAnchorError(INVALID_HTTP_ANCHOR)


def _refuse_surrogates(text: str) -> None:
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        _invalid()


def _require_method(value: object) -> str:
    text = require_bounded_exact_str(value, max_len=len(GET_METHOD), refuse=_invalid)
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


def _parse_path(value: object) -> str:
    text = require_bounded_exact_str(value, max_len=PATH_CEILING, refuse=_invalid)
    _refuse_surrogates(text)
    if any(ch in "#=?&%" for ch in text):
        _invalid()
    if not text.startswith(PATH_PREFIX) or not text.endswith(PATH_SUFFIX):
        _invalid()
    message_id = text[len(PATH_PREFIX) : len(text) - len(PATH_SUFFIX)]
    message_id = require_bounded_exact_str(message_id, max_len=MAX_ID_CHARS, refuse=_invalid)
    _refuse_surrogates(message_id)
    if message_id.strip() != message_id or any(ch.isspace() for ch in message_id):
        _invalid()
    if "/" in message_id:
        _invalid()
    return message_id


def _http_from_evaluator(
    *,
    witness: object,
    method: str,
    anchor: MessageAnchorV1,
) -> ConversationSearchHttpMessageAnchorV1:
    if witness is not _HTTP_WITNESS:
        _invalid()
    bound = object.__new__(ConversationSearchHttpMessageAnchorV1)
    object.__setattr__(bound, "schema", SCHEMA_ID)
    object.__setattr__(bound, "schema_version", SCHEMA_VERSION)
    object.__setattr__(bound, "method", method)
    object.__setattr__(bound, "message_id", anchor.message_id)
    object.__setattr__(bound, "fragment", f"msg={anchor.message_id}")
    object.__setattr__(bound, "changes_active_branch", False)
    object.__setattr__(bound, "history_write_authorized", False)
    object.__setattr__(bound, "query_in_url_authorized", False)
    return bound


def bind_search_http_message_anchor(
    *,
    method: object,
    path: object,
    query_string: object = "",
) -> ConversationSearchHttpMessageAnchorV1:
    """Bind a device-local fragment without writing history."""
    bound_method = _require_method(method)
    message_id = _parse_path(path)
    _require_empty_query_string(query_string)
    try:
        anchor = parse_message_anchor(f"msg={message_id}")
    except AnchorError:
        _invalid()
    return _http_from_evaluator(witness=_HTTP_WITNESS, method=bound_method, anchor=anchor)
