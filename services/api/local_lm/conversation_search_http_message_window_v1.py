"""Pure conversation-search HTTP message-window facts.

Binds POST /api/conversation-search/message-window with an empty
query string. Never loads a chat or activates a branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

from .message_window_v1 import (
    MAX_INPUT_IDS,
    MessageWindowError,
    MessageWindowPlan,
    plan_message_window,
)
from .search_text_v1 import require_bounded_exact_str

SCHEMA_ID: Final = "lm-atelier-conversation-search-http-message-window-v1"
SCHEMA_VERSION: Final = 1
INVALID_HTTP_MESSAGE_WINDOW: Final = "search http-message-window facts are invalid"
POST_METHOD: Final = "POST"
METHODS: Final = frozenset({POST_METHOD})
WINDOW_PATH: Final = "/api/conversation-search/message-window"
_HTTP_WITNESS = object()


class SearchHttpMessageWindowError(ValueError):
    """Fixed non-echoing refusal for invalid HTTP message-window facts."""


@dataclass(frozen=True, slots=True)
class ConversationSearchHttpMessageWindowV1:
    schema: Literal["lm-atelier-conversation-search-http-message-window-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    method: Literal["POST"] = field(init=False)
    mode: str = field(init=False)
    message_ids: tuple[str, ...] = field(init=False)
    anchor_id: str | None = field(init=False)
    loads_entire_chat: Literal[False] = field(init=False)
    branch_activation_authorized: Literal[False] = field(init=False)
    query_in_url_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise SearchHttpMessageWindowError(INVALID_HTTP_MESSAGE_WINDOW)


def _invalid() -> NoReturn:
    raise SearchHttpMessageWindowError(INVALID_HTTP_MESSAGE_WINDOW)


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
    text = require_bounded_exact_str(value, max_len=len(WINDOW_PATH), refuse=_invalid)
    _refuse_surrogates(text)
    if any(ch in "#=?&%" for ch in text):
        _invalid()
    if text != WINDOW_PATH:
        _invalid()
    return text


def _require_ordered_ids(value: object) -> object:
    if type(value) is not list and type(value) is not tuple:
        _invalid()
    if len(value) > MAX_INPUT_IDS:
        _invalid()
    return value


def _http_from_evaluator(
    *,
    witness: object,
    method: str,
    plan: MessageWindowPlan,
) -> ConversationSearchHttpMessageWindowV1:
    if witness is not _HTTP_WITNESS:
        _invalid()
    bound = object.__new__(ConversationSearchHttpMessageWindowV1)
    object.__setattr__(bound, "schema", SCHEMA_ID)
    object.__setattr__(bound, "schema_version", SCHEMA_VERSION)
    object.__setattr__(bound, "method", method)
    object.__setattr__(bound, "mode", plan.mode)
    object.__setattr__(bound, "message_ids", plan.message_ids)
    object.__setattr__(bound, "anchor_id", plan.anchor_id)
    object.__setattr__(bound, "loads_entire_chat", False)
    object.__setattr__(bound, "branch_activation_authorized", False)
    object.__setattr__(bound, "query_in_url_authorized", False)
    return bound


def bind_search_http_message_window(
    *,
    method: object,
    path: object,
    query_string: object = "",
    ordered_ids: object,
    mode: object,
    anchor_id: object = None,
    limit: object = 40,
) -> ConversationSearchHttpMessageWindowV1:
    """Plan a window without loading a chat or copying unbounded ids."""
    bound_method = _require_method(method)
    _require_path(path)
    _require_empty_query_string(query_string)
    bounded_ids = _require_ordered_ids(ordered_ids)
    try:
        plan = plan_message_window(bounded_ids, mode=mode, anchor_id=anchor_id, limit=limit)
    except MessageWindowError:
        _invalid()
    return _http_from_evaluator(witness=_HTTP_WITNESS, method=bound_method, plan=plan)
