"""Pure result pages over already-ranked search hits.

A page is a window, a cursor for the next window, and how many rows were
offered. It performs no I/O, ranks nothing, and cannot be talked into claiming
it loaded an entire chat.

The cursor is an offset serialized as a decimal string, and parsing it is not
this module's job. `search_cursor_v1.require_int_text` is the one parser for a
bounded decimal offset in this tree; `_offset` calls it and translates its
refusal into this module's fixed one, so a future cursor rule has exactly one
implementation to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

from .search_cursor_v1 import MAX_OFFSET, SearchCursorError, require_int_text
from .search_hit_v1 import SearchHitV1

SCHEMA_ID: Final = "lm-atelier-search-page-v1"
SCHEMA_VERSION: Final = 1
INVALID_PAGE: Final = "search page facts are invalid"
MAX_PAGE_SIZE: Final = 50
DEFAULT_PAGE_SIZE: Final = 20


class SearchPageError(ValueError):
    """Fixed non-echoing refusal for invalid page facts."""


@dataclass(frozen=True, slots=True)
class SearchPageV1:
    schema: Literal["lm-atelier-search-page-v1"]
    schema_version: Literal[1]
    hits: tuple[SearchHitV1, ...]
    next_cursor: str | None
    total_reported: int
    loads_entire_chat: Literal[False] = field(default=False, init=False)


def _refuse() -> NoReturn:
    raise SearchPageError(INVALID_PAGE)


def _offset(cursor: object) -> int:
    """Delegate to the canonical parser and translate its refusal.

    There is one bounded-decimal parser in the tree and it lives in
    `search_cursor_v1`. This does not restate its rules, because a restatement
    is a second implementation however carefully it is kept in step: an
    equivalence test over a finite table proves the two agree on the inputs
    sampled, and says nothing about the next rule added to one of them.

    What is left here is the only thing that is genuinely this module's
    business - that a caller of the page sees the page's own fixed refusal
    rather than a cursor error from a module it never called.
    """
    if cursor is None:
        return 0
    try:
        return require_int_text(cursor, maximum=MAX_OFFSET)
    except SearchCursorError as error:
        raise SearchPageError(INVALID_PAGE) from error


def build_search_page(
    hits: object,
    *,
    limit: object = DEFAULT_PAGE_SIZE,
    cursor: object = None,
) -> SearchPageV1:
    """Window already-ranked hits, and say where the next window starts.

    Ranking belongs to whoever built the hits. This slices what it is given in
    the order it was given, so a page is reproducible from its cursor alone.

    `total_reported` counts the rows offered to this call, not matches in the
    corpus. It is an `int` on every path; an optional type here would only be a
    `None` no caller can ever observe.
    """
    if type(hits) is not list and type(hits) is not tuple:
        _refuse()
    # Refuse more rows than a cursor can address, before looking at the rows
    # themselves. Whatever this hands out it has to accept back, and with more
    # than MAX_OFFSET rows the next offset could exceed the ceiling `_offset`
    # enforces - handing the caller a cursor this module refuses would strand
    # them on a page they cannot leave. Bounding the input makes that
    # impossible by construction rather than conditional on arithmetic, which
    # is the difference between an invariant and a guard nothing can reach.
    if len(hits) > MAX_OFFSET:
        _refuse()
    if type(limit) is not int or limit < 1 or limit > MAX_PAGE_SIZE:
        _refuse()
    offset = _offset(cursor)
    rows = tuple(hits)
    for item in rows:
        if not isinstance(item, SearchHitV1):
            _refuse()
    end = offset + limit
    more = end < len(rows)
    return SearchPageV1(
        schema="lm-atelier-search-page-v1",
        schema_version=1,
        hits=rows[offset:end],
        next_cursor=str(end) if more else None,
        total_reported=len(rows),
    )
