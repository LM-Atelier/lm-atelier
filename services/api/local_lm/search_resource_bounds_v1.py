"""Pure conversation-search resource bounds.

Fixed caps for a later FTS/API layer. No I/O, and the record never grants
permission to execute a query or write an index - both of those are
`Literal[False]` facts a caller cannot set.

The caps agree with the modules that enforce them. A declaration permitting more
than the implementation accepts converts a caller's correct behaviour into a
refusal: this once declared a 256-character query acceptable while
conversation_search_query_v1 and the already-landed search_snippet_v1 both
refuse anything over 200.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-search-resource-bounds-v1"
SCHEMA_VERSION: Final = 1
INVALID_BOUNDS: Final = "search resource bounds are invalid"
MAX_LIMIT: Final = 50
MAX_WINDOW: Final = 64
MAX_SNIPPET: Final = 280
MAX_QUERY_CHARS: Final = 200
MAX_TERMS: Final = 16


class SearchResourceBoundsError(ValueError):
    """Fixed non-echoing refusal."""


@dataclass(frozen=True, slots=True)
class SearchResourceBoundsV1:
    schema: Literal["lm-atelier-search-resource-bounds-v1"]
    schema_version: Literal[1]
    limit: int
    window: int
    snippet_chars: int
    query_chars: int
    term_count: int
    query_execution_authorized: Literal[False] = field(default=False, init=False)
    fts_write_authorized: Literal[False] = field(default=False, init=False)


def _invalid() -> NoReturn:
    raise SearchResourceBoundsError(INVALID_BOUNDS)


def _configured(value: object, *, maximum: int) -> int:
    """A configured ceiling: at least one, because zero of it means no search."""
    if type(value) is not int or isinstance(value, bool) or value < 1 or value > maximum:
        _invalid()
    return value


def _measured(value: object, *, maximum: int) -> int:
    """A count taken FROM a request, so zero is a legitimate measurement.

    query_chars and term_count are supplied by the caller as what the request
    actually contained, not as ceilings it may use - an empty query genuinely
    measures zero of both. That is why they accept zero where limit, window and
    snippet_chars require at least one; the asymmetry is the difference between
    describing a request and configuring a search, not an oversight.
    """
    if type(value) is not int or isinstance(value, bool) or value < 0 or value > maximum:
        _invalid()
    return value


def declare_search_resource_bounds(
    *,
    limit: object = 20,
    window: object = 32,
    snippet_chars: object = 160,
    query_chars: object,
    term_count: object,
) -> SearchResourceBoundsV1:
    """Declare the bounds a search may use, and what the request measured."""
    return SearchResourceBoundsV1(
        schema="lm-atelier-search-resource-bounds-v1",
        schema_version=1,
        limit=_configured(limit, maximum=MAX_LIMIT),
        window=_configured(window, maximum=MAX_WINDOW),
        snippet_chars=_configured(snippet_chars, maximum=MAX_SNIPPET),
        query_chars=_measured(query_chars, maximum=MAX_QUERY_CHARS),
        term_count=_measured(term_count, maximum=MAX_TERMS),
    )
