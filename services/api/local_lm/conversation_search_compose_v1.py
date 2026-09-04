"""Pure conversation-search compose path.

Wires visibility filter, rank, hit assembly, paging, privacy, and resource
bounds. Caller supplies already-visible row facts including chat_id. No DB,
FTS, or API I/O. Never grants query execution or index write authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from .conversation_search_query_v1 import (
    MAX_ID_CHARS,
    ConversationSearchError,
    rank_identity_bodies,
)
from .search_hit_v1 import SearchHitV1, build_search_hit
from .search_page_v1 import SearchPageError, SearchPageV1, build_search_page
from .search_privacy_v1 import SearchPrivacyError, SearchPrivacyV1, classify_search_privacy
from .search_resource_bounds_v1 import (
    SearchResourceBoundsError,
    SearchResourceBoundsV1,
    declare_search_resource_bounds,
)
from .search_visibility_v1 import SearchVisibilityError, filter_indexable_bodies

SCHEMA_ID: Final = "lm-atelier-search-compose-v1"
SCHEMA_VERSION: Final = 1
INVALID_COMPOSE: Final = "search compose facts are invalid"
MAX_COMPOSE_ROWS: Final = 64
MAX_COMPOSE_BODY_CHARS: Final = 8192
MAX_COMPOSE_TEXT_CHARS: Final = 65536
MAX_COMPOSE_WORK: Final = 512
MAX_COMPOSE_KEYS: Final = 16
COMPOSE_ROW_KEYS: Final = frozenset(
    {
        "message_id",
        "chat_id",
        "body",
        "transcript_visible",
        "content_removed",
        "private_session",
        "helper_session",
        "secret_payload",
    }
)


class SearchComposeError(ValueError):
    """Fixed non-echoing refusal for invalid compose facts."""


class _RowTooWide(SearchComposeError):
    """A row carrying more keys than acquisition is allowed to read.

    Private, and deliberately not a second public error: it subclasses the
    fixed refusal and carries the same message, so every caller sees exactly
    what it saw before and nothing about the input is echoed.

    It exists because the key ceiling is otherwise unobservable. A row with too
    many keys is refused either way - once by this bound, or later because its
    key set is wrong - so a test could not tell bounded acquisition from
    unbounded acquisition without timing the difference, and a timing assertion
    is not a proof. Naming the refusal makes the bound provable by behaviour.
    """


@dataclass(frozen=True, slots=True)
class SearchComposeResultV1:
    schema: Literal["lm-atelier-search-compose-v1"]
    schema_version: Literal[1]
    page: SearchPageV1
    privacy: SearchPrivacyV1
    bounds: SearchResourceBoundsV1
    considered: int
    eligible: int
    query_execution_authorized: Literal[False] = field(default=False, init=False)
    fts_write_authorized: Literal[False] = field(default=False, init=False)
    index_rebuild_authorized: Literal[False] = field(default=False, init=False)


def _bound_id(value: object) -> str:
    if type(value) is not str or not value:
        raise SearchComposeError(INVALID_COMPOSE)
    if value.strip() != value or any(ch.isspace() for ch in value):
        raise SearchComposeError(INVALID_COMPOSE)
    if len(value) > MAX_ID_CHARS:
        raise SearchComposeError(INVALID_COMPOSE)
    return value


def _owned_compose_row(row: object) -> dict[str, object]:
    if type(row) is not dict:
        raise SearchComposeError(INVALID_COMPOSE)
    if len(row) > MAX_COMPOSE_KEYS:
        # Before the copy below, not after it. A wider row is refused without
        # reading it, so per-row work stays bounded by the ceiling rather than
        # by whatever the caller sent.
        raise _RowTooWide(INVALID_COMPOSE)
    owned: dict[str, object] = {}
    for key, value in row.items():
        if type(key) is not str:
            raise SearchComposeError(INVALID_COMPOSE)
        owned[key] = value
    if set(owned) != COMPOSE_ROW_KEYS:
        raise SearchComposeError(INVALID_COMPOSE)
    return owned


def compose_conversation_search(
    rows: object,
    query: object,
    *,
    limit: object = 20,
) -> SearchComposeResultV1:
    """Compose a page of hits from declared visible rows.

    Ineligible rows (tombstone / private / helper / secret) never rank.
    Duplicate message ids are refused. Each eligible hit binds the caller
    chat_id for that unique message. Raw query is not stored on the result.
    """
    try:
        return _compose_conversation_search(rows, query, limit=limit)
    except _RowTooWide:
        # Recorded, not re-raised from here. The subclass exists so a test can
        # see WHICH internal refusal happened; letting any trace of it out
        # would make it part of the public contract, which is the opposite of
        # the point.
        #
        # The raise has to happen AFTER this handler has exited. `from None`
        # is not enough: it clears __cause__ and suppresses the traceback
        # display, but __context__ still holds the private exception and that
        # attribute is public. Raising once the handler is done means there is
        # no active exception to become the context at all.
        pass
    except SearchComposeError:
        raise
    except (
        ConversationSearchError,
        SearchPageError,
        SearchPrivacyError,
        SearchResourceBoundsError,
        SearchVisibilityError,
    ) as exc:
        raise SearchComposeError(INVALID_COMPOSE) from exc

    # Reached only through the ceiling handler above, which records the
    # refusal and deliberately does not raise while it is still active.
    raise SearchComposeError(INVALID_COMPOSE)


def _compose_conversation_search(
    rows: object,
    query: object,
    *,
    limit: object,
) -> SearchComposeResultV1:
    if type(rows) is not list and type(rows) is not tuple:
        raise SearchComposeError(INVALID_COMPOSE)
    if len(rows) > MAX_COMPOSE_ROWS:
        raise SearchComposeError(INVALID_COMPOSE)
    if type(query) is not str:
        raise SearchComposeError(INVALID_COMPOSE)
    if type(limit) is not int:
        raise SearchComposeError(INVALID_COMPOSE)

    identities: dict[str, str] = {}
    owned_rows: list[dict[str, object]] = []
    total_text = 0
    for row in rows:
        owned = _owned_compose_row(row)
        mid = _bound_id(owned["message_id"])
        if mid in identities:
            raise SearchComposeError(INVALID_COMPOSE)
        chat_id = _bound_id(owned["chat_id"])
        body = owned["body"]
        if type(body) is not str:
            raise SearchComposeError(INVALID_COMPOSE)
        if len(body) > MAX_COMPOSE_BODY_CHARS:
            raise SearchComposeError(INVALID_COMPOSE)
        total_text += len(body)
        if total_text > MAX_COMPOSE_TEXT_CHARS:
            raise SearchComposeError(INVALID_COMPOSE)
        identities[mid] = chat_id
        owned_rows.append(owned)

    privacy = classify_search_privacy(query)
    work = len(rows) * privacy.term_count
    if work > MAX_COMPOSE_WORK:
        raise SearchComposeError(INVALID_COMPOSE)
    bounds = declare_search_resource_bounds(
        limit=limit,
        query_chars=privacy.query_char_count,
        term_count=privacy.term_count,
    )
    eligible = filter_indexable_bodies(owned_rows)
    identity_rows = tuple((mid, identities[mid], body) for mid, body in eligible)
    ranked = rank_identity_bodies(identity_rows, query)
    hits: list[SearchHitV1] = []
    for mid, chat_id, body, _score in ranked:
        hit = build_search_hit(
            message_id=mid,
            chat_id=chat_id,
            body=body,
            query=query,
        )
        if hit is not None:
            hits.append(hit)
    page = build_search_page(hits, limit=bounds.limit)
    return SearchComposeResultV1(
        schema="lm-atelier-search-compose-v1",
        schema_version=1,
        page=page,
        privacy=privacy,
        bounds=bounds,
        considered=len(rows),
        eligible=len(eligible),
    )
