"""Pure conversation search filter validation.

Validates caller-supplied filter facts only. No SQL, FTS, or chat mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

MAX_ID_CHARS: Final = 40
MAX_TIME_CHARS: Final = 40
MAX_FILTER_KEYS: Final = 8
INVALID_FILTER: Final = "invalid-conversation-search-filter"
ROLES: Final = frozenset({"user", "assistant", "system", "tool"})
ALLOWED_KEYS: Final = frozenset({"project_id", "chat_id", "role", "has_media", "since", "until"})


class SearchFilterError(ValueError):
    """Fixed refusal for invalid search filters."""


@dataclass(frozen=True, slots=True)
class SearchFiltersV1:
    project_id: str | None
    chat_id: str | None
    role: str | None
    has_media: bool | None
    since: str | None
    until: str | None
    query_execution_authorized: Literal[False] = field(default=False, init=False)
    fts_write_authorized: Literal[False] = field(default=False, init=False)


def _opt_id(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value or len(value) > MAX_ID_CHARS:
        raise SearchFilterError(INVALID_FILTER)
    if any(ch.isspace() for ch in value):
        raise SearchFilterError(INVALID_FILTER)
    return value


def _opt_bool(value: object) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise SearchFilterError(INVALID_FILTER)
    return value


def _opt_time(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value or len(value) > MAX_TIME_CHARS:
        raise SearchFilterError(INVALID_FILTER)
    if len(value) < 10 or value[4] != "-" or value[7] != "-":
        raise SearchFilterError(INVALID_FILTER)
    return value


def validate_search_filters(raw: object) -> SearchFiltersV1:
    if type(raw) is not dict or len(raw) > MAX_FILTER_KEYS:
        raise SearchFilterError(INVALID_FILTER)
    owned: dict[str, object] = {}
    for key, value in raw.items():
        if type(key) is not str:
            raise SearchFilterError(INVALID_FILTER)
        owned[key] = value
    if set(owned) - ALLOWED_KEYS:
        raise SearchFilterError(INVALID_FILTER)
    role = owned.get("role")
    if role is not None and (type(role) is not str or role not in ROLES):
        raise SearchFilterError(INVALID_FILTER)
    return SearchFiltersV1(
        project_id=_opt_id(owned.get("project_id")),
        chat_id=_opt_id(owned.get("chat_id")),
        role=role if type(role) is str else None,
        has_media=_opt_bool(owned.get("has_media")),
        since=_opt_time(owned.get("since")),
        until=_opt_time(owned.get("until")),
    )
