"""Pure conversation search query normalization, scoring, and hit locations.

No database, FTS, API, index write, or branch activation. Callers supply text
they have already established is visible; this module only normalizes queries
into terms, scores supplied bodies against them, and builds hit locations that
cannot activate a branch or load a whole chat.

Snippets are deliberately NOT here. `search_snippet_v1` already returns bounded
segments and maps a casefolded match index back through an offset map before
slicing. An earlier version of this module carried its own `build_snippet`,
which found a match index in casefolded text and then sliced the ORIGINAL
string: `str.casefold()` is not length preserving, so every expanding character
before the match shifted the window one place, and past a drift of `radius` the
snippet no longer contained the term it had matched on. Deleting the duplicate
removes that defect rather than repairing it, and callers gain the other
module's bounds for free.

Ranking is bounded in three ways because an unbounded pure helper is still a
denial of service in the caller's process: a cap on rows accepted, a cap on any
single body, and a cap on the total text scanned. Without them this ranked and
materialized an entire supplied corpus before the caller ever sliced a page.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

MAX_QUERY_CHARS: Final = 200
MAX_TERMS: Final = 16
MAX_ID_CHARS: Final = 40

# Matches search_snippet_v1, which callers pair with this module. A body larger
# than that module will accept is not worth scoring here either.
MAX_BODY_CHARS: Final = 8192

# Ranking is a pure helper over rows the caller already loaded. Paging is the
# caller's job, so accepting an unbounded sequence only moves the cost rather
# than removing it.
MAX_RANKED_ROWS: Final = 500
MAX_TOTAL_BODY_CHARS: Final = 1_048_576

PROJECTION_SCHEMA: Final = "conversation-fts-v1"


class ConversationSearchError(ValueError):
    """Fixed refusal for invalid search query facts."""


INVALID_QUERY: Final = "invalid-conversation-search-query"
INVALID_LOCATION: Final = "invalid-conversation-search-location"


def normalize_search_query(raw: object) -> str:
    """Return a bounded, whitespace-collapsed query, or refuse."""
    if type(raw) is not str:
        raise ConversationSearchError(INVALID_QUERY)
    if len(raw) > MAX_QUERY_CHARS:
        raise ConversationSearchError(INVALID_QUERY)
    text = " ".join(raw.split())
    if not text or len(text) > MAX_QUERY_CHARS:
        raise ConversationSearchError(INVALID_QUERY)
    return text


def query_terms(query: object) -> tuple[str, ...]:
    """Split a normalized query into at most MAX_TERMS casefolded terms."""
    normalized = normalize_search_query(query)
    terms = tuple(part.casefold() for part in normalized.split() if part)
    if not terms or len(terms) > MAX_TERMS:
        raise ConversationSearchError(INVALID_QUERY)
    return terms


@dataclass(frozen=True, slots=True)
class SearchHitLocation:
    """Where a hit is, and two facts a caller can never set to True.

    Both are `Literal[False]` with `init=False` rather than ordinary bools:
    a fixed fact that a constructor can be talked out of is not a fact, and a
    caller that could pass `activates_branch=True` would be able to describe a
    location this module has no authority to produce.
    """

    message_id: str
    chat_id: str
    project_id: str | None
    activates_branch: Literal[False] = field(default=False, init=False)
    loads_entire_chat: Literal[False] = field(default=False, init=False)


def _require_body(value: object) -> str:
    if type(value) is not str or not value or len(value) > MAX_BODY_CHARS:
        raise ConversationSearchError(INVALID_QUERY)
    return value


def term_coverage_score(body: object, query: object) -> int:
    """Count the distinct query terms present in body, from 0 to len(terms)."""
    text = _require_body(body)
    terms = query_terms(query)
    folded = text.casefold()
    return len({term for term in terms if term in folded})


def _require_id(value: object) -> str:
    if type(value) is not str or not value or len(value) > MAX_ID_CHARS:
        raise ConversationSearchError(INVALID_LOCATION)
    if any(character.isspace() for character in value):
        raise ConversationSearchError(INVALID_LOCATION)
    return value


def hit_location(
    *,
    message_id: object,
    chat_id: object,
    project_id: object = None,
) -> SearchHitLocation:
    """Build a location that never auto-activates a branch or full-chat load."""
    return SearchHitLocation(
        message_id=_require_id(message_id),
        chat_id=_require_id(chat_id),
        project_id=None if project_id is None else _require_id(project_id),
    )


def _require_rows(candidates: object, *, width: int) -> tuple[tuple[object, ...], ...]:
    """Accept a bounded sequence of fixed-width tuples, or refuse.

    The row cap is checked before anything is scored, so an oversized corpus
    costs one length check rather than a full ranking pass.
    """
    if type(candidates) is not list and type(candidates) is not tuple:
        raise ConversationSearchError(INVALID_QUERY)
    if len(candidates) > MAX_RANKED_ROWS:
        raise ConversationSearchError(INVALID_QUERY)
    rows: list[tuple[object, ...]] = []
    total = 0
    for item in candidates:
        if type(item) is not tuple or len(item) != width:
            raise ConversationSearchError(INVALID_QUERY)
        body = item[width - 1]
        total += len(body) if type(body) is str else 0
        if total > MAX_TOTAL_BODY_CHARS:
            raise ConversationSearchError(INVALID_QUERY)
        rows.append(item)
    return tuple(rows)


def rank_candidate_bodies(
    candidates: Sequence[tuple[str, str]],
    query: object,
) -> tuple[tuple[str, str, int], ...]:
    """Rank (id, body) rows by term coverage descending, then id ascending."""
    rows = _require_rows(candidates, width=2)
    scored: list[tuple[str, str, int]] = []
    for row in rows:
        identifier = _require_id(row[0])
        body = _require_body(row[1])
        score = term_coverage_score(body, query)
        if score <= 0:
            continue
        scored.append((identifier, body, score))
    scored.sort(key=lambda row: (-row[2], row[0]))
    return tuple(scored)


def rank_identity_bodies(
    candidates: Sequence[tuple[str, str, str]],
    query: object,
) -> tuple[tuple[str, str, str, int], ...]:
    """Rank (message_id, chat_id, body) rows, keeping identity on each result.

    The full identity travels with every row on purpose. Keying results by
    message id alone lets two rows that share an id collapse onto one mapping,
    so both hits resolve to whichever chat was seen last.
    """
    rows = _require_rows(candidates, width=3)
    scored: list[tuple[str, str, str, int]] = []
    for row in rows:
        message_id = _require_id(row[0])
        chat_id = _require_id(row[1])
        body = _require_body(row[2])
        score = term_coverage_score(body, query)
        if score <= 0:
            continue
        scored.append((message_id, chat_id, body, score))
    scored.sort(key=lambda row: (-row[3], row[0], row[1]))
    return tuple(scored)
