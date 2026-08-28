"""Pure opaque conversation-search cursor facts.

Binds index generation, query digest, and offset only. Never carries raw
query text or grants query execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

from .search_text_v1 import require_bounded_exact_str

SCHEMA_ID: Final = "lm-atelier-search-cursor-v1"
SCHEMA_VERSION: Final = 1
INVALID_CURSOR: Final = "search cursor facts are invalid"
MAX_GENERATION: Final = 2**31 - 1
MAX_OFFSET: Final = 100_000
MAX_EXPIRES: Final = 2**31 - 1
MAX_DIGEST_CHARS: Final = 64
MAX_TOKEN_CHARS: Final = 128
MAX_INT_DIGITS: Final = 10
DIGEST_CHARS: Final = frozenset("0123456789abcdef")
DIGIT_CHARS: Final = frozenset("0123456789")
TOKEN_PREFIX: Final = "v1"
_CURSOR_WITNESS = object()


class SearchCursorError(ValueError):
    """Fixed non-echoing refusal for invalid cursor facts."""


@dataclass(frozen=True, slots=True)
class SearchCursorV1:
    schema: Literal["lm-atelier-search-cursor-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    index_generation: int = field(init=False)
    query_digest: str = field(init=False)
    offset: int = field(init=False)
    expires_at_unix: int = field(init=False)
    contains_query_text: Literal[False] = field(init=False)
    query_execution_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise SearchCursorError(INVALID_CURSOR)


def _invalid() -> NoReturn:
    raise SearchCursorError(INVALID_CURSOR)


def _require_int(value: object, *, maximum: int, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        _invalid()
    return value


def _require_digest(value: object) -> str:
    text = require_bounded_exact_str(value, max_len=MAX_DIGEST_CHARS, refuse=_invalid)
    if len(text) != MAX_DIGEST_CHARS:
        _invalid()
    if any(ch not in DIGEST_CHARS for ch in text):
        _invalid()
    return text


def _require_int_text(value: str, *, maximum: int) -> int:
    if type(value) is not str:
        _invalid()
    if len(value) > MAX_INT_DIGITS:
        _invalid()
    if not value or any(ch not in DIGIT_CHARS for ch in value):
        _invalid()
    if len(value) > 1 and value[0] == "0":
        _invalid()
    parsed = int(value)
    if parsed > maximum:
        _invalid()
    return parsed


def _cursor_from_evaluator(
    *,
    witness: object,
    index_generation: int,
    query_digest: str,
    offset: int,
    expires_at_unix: int,
) -> SearchCursorV1:
    if witness is not _CURSOR_WITNESS:
        _invalid()
    cursor = object.__new__(SearchCursorV1)
    object.__setattr__(cursor, "schema", SCHEMA_ID)
    object.__setattr__(cursor, "schema_version", SCHEMA_VERSION)
    object.__setattr__(cursor, "index_generation", index_generation)
    object.__setattr__(cursor, "query_digest", query_digest)
    object.__setattr__(cursor, "offset", offset)
    object.__setattr__(cursor, "expires_at_unix", expires_at_unix)
    object.__setattr__(cursor, "contains_query_text", False)
    object.__setattr__(cursor, "query_execution_authorized", False)
    return cursor


def bind_search_cursor(
    *,
    index_generation: object,
    query_digest: object,
    offset: object,
    expires_at_unix: object,
    now_unix: object,
) -> SearchCursorV1:
    """Bind an opaque cursor without retaining query text."""
    generation = _require_int(index_generation, maximum=MAX_GENERATION)
    digest = _require_digest(query_digest)
    start = _require_int(offset, maximum=MAX_OFFSET)
    expires = _require_int(expires_at_unix, maximum=MAX_EXPIRES, minimum=1)
    now = _require_int(now_unix, maximum=MAX_EXPIRES)
    if now >= expires:
        _invalid()
    return _cursor_from_evaluator(
        witness=_CURSOR_WITNESS,
        index_generation=generation,
        query_digest=digest,
        offset=start,
        expires_at_unix=expires,
    )


def encode_search_cursor(cursor: SearchCursorV1) -> str:
    """Return a bounded opaque token with no query text."""
    if type(cursor) is not SearchCursorV1:
        _invalid()
    token = (
        f"{TOKEN_PREFIX}.{cursor.index_generation}.{cursor.offset}."
        f"{cursor.expires_at_unix}.{cursor.query_digest}"
    )
    if len(token) > MAX_TOKEN_CHARS:
        _invalid()
    return token


def decode_search_cursor(
    token: object,
    *,
    now_unix: object,
    index_generation: object,
    query_digest: object,
) -> SearchCursorV1:
    """Restore a cursor and refuse stale generation, digest, or expiry."""
    text = require_bounded_exact_str(token, max_len=MAX_TOKEN_CHARS, refuse=_invalid)
    parts = text.split(".", 4)
    if len(parts) != 5:
        _invalid()
    prefix, generation_text, offset_text, expires_text, digest_text = parts
    if prefix != TOKEN_PREFIX:
        _invalid()
    expected_generation = _require_int(index_generation, maximum=MAX_GENERATION)
    expected_digest = _require_digest(query_digest)
    parsed = bind_search_cursor(
        index_generation=_require_int_text(generation_text, maximum=MAX_GENERATION),
        query_digest=digest_text,
        offset=_require_int_text(offset_text, maximum=MAX_OFFSET),
        expires_at_unix=_require_int_text(expires_text, maximum=MAX_EXPIRES),
        now_unix=now_unix,
    )
    if parsed.index_generation != expected_generation or parsed.query_digest != expected_digest:
        _invalid()
    return parsed
