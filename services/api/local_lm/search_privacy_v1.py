"""Pure search privacy facts for a conversation-search request.

Classifies what may be logged or echoed from a request. Never stores, logs, or
returns the raw query string: the facts below describe a query without carrying
it, so a caller can record them anywhere the query itself must not go.

The caps match the modules that do the work rather than exceeding them. A
declaration that permits more than the implementation accepts turns a caller's
correct behaviour into a refusal - it would report a 220-character query as
acceptable while the query and snippet modules refuse anything over 200, and
report a term count of 128 for a request that is rejected at 17.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-search-privacy-v1"
SCHEMA_VERSION: Final = 1
INVALID_PRIVACY: Final = "search privacy facts are invalid"

# Both agree with conversation_search_query_v1 and with search_snippet_v1, which
# is already on trunk. Four modules holding the same two numbers is still four
# places they can drift apart; they are written here rather than imported
# because this module deliberately depends on nothing.
MAX_PRIVACY_QUERY_CHARS: Final = 200
MAX_PRIVACY_TERMS: Final = 16


class SearchPrivacyError(ValueError):
    """Fixed non-echoing refusal."""


@dataclass(frozen=True, slots=True)
class SearchPrivacyV1:
    """Facts about a query, never the query.

    The three permissions are `Literal[False]` with `init=False` so that no
    caller can construct a record claiming the raw query may be logged, placed
    in a URL, or echoed back. A permission a constructor can be talked into is
    not a permission this module is willing to describe.
    """

    schema: Literal["lm-atelier-search-privacy-v1"]
    schema_version: Literal[1]
    term_count: int
    query_char_count: int
    may_log_raw_query: Literal[False] = field(default=False, init=False)
    may_put_raw_query_in_url: Literal[False] = field(default=False, init=False)
    may_echo_raw_query: Literal[False] = field(default=False, init=False)


def _invalid() -> NoReturn:
    raise SearchPrivacyError(INVALID_PRIVACY)


def classify_search_privacy(query: object) -> SearchPrivacyV1:
    """Return non-echoing privacy facts for a query, without its text."""
    if type(query) is not str:
        _invalid()
    query_char_count = len(query)
    if query_char_count > MAX_PRIVACY_QUERY_CHARS:
        _invalid()
    parts = [part for part in query.split() if part]
    if len(parts) > MAX_PRIVACY_TERMS:
        _invalid()
    return SearchPrivacyV1(
        schema="lm-atelier-search-privacy-v1",
        schema_version=1,
        term_count=len(parts),
        query_char_count=query_char_count,
    )
