"""Pure search hit assembly over the landed query and snippet modules.

A hit is a location, a bounded snippet and a score. It performs no I/O, and the
record cannot be talked into claiming it loaded an entire chat or activated a
branch: both are `Literal[False]` with `init=False`.

The snippet is the structured `SearchSnippetV1` rather than a rendered string.
An earlier shape of this module carried `snippet: str` built by a helper that
has since been deleted, which meant a caller received text it had to render and
this module had no way to say the segments were never HTML. Segments carry their
own `html_authorized` fact, so passing the record through keeps that guarantee
attached to the thing it describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from .conversation_search_query_v1 import (
    SearchHitLocation,
    hit_location,
    term_coverage_score,
)
from .search_snippet_v1 import SearchSnippetV1, build_search_snippet

SCHEMA_ID: Final = "lm-atelier-search-hit-v1"
SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class SearchHitV1:
    schema: Literal["lm-atelier-search-hit-v1"]
    schema_version: Literal[1]
    location: SearchHitLocation
    snippet: SearchSnippetV1
    score: int
    loads_entire_chat: Literal[False] = field(default=False, init=False)
    activates_branch: Literal[False] = field(default=False, init=False)


def build_search_hit(
    *,
    message_id: object,
    chat_id: object,
    body: object,
    query: object,
    project_id: object = None,
) -> SearchHitV1 | None:
    """Return a hit when the body covers at least one query term, else None.

    Refusals from the query and snippet modules are allowed to propagate rather
    than being wrapped in a third error type. Which module refused is the useful
    part of the answer - a caller that cannot build a location has a different
    problem from one whose body exceeds the snippet ceiling - and a wrapper here
    would erase exactly that distinction while adding no information.

    Scoring runs before anything is built. A body that covers no term is not a
    hit, and there is no reason to construct a location or slice a snippet for
    a row that will be discarded.
    """
    score = term_coverage_score(body, query)
    if score <= 0:
        return None
    return SearchHitV1(
        schema="lm-atelier-search-hit-v1",
        schema_version=1,
        location=hit_location(message_id=message_id, chat_id=chat_id, project_id=project_id),
        snippet=build_search_snippet(body, query),
        score=score,
    )
