"""Pure conversation-search index status facts (item 41).

Reports ready/building/degraded from caller-supplied generation facts.
Never writes FTS, rebuilds an index, or echoes content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-search-index-status-v1"
SCHEMA_VERSION: Final = 1
INVALID_STATUS: Final = "search index status facts are invalid"
MAX_GENERATION: Final = 2**31 - 1
MAX_SEQUENCE: Final = 2**31 - 1
MAX_DETAIL_CHARS: Final = 40
IndexState = Literal["ready", "building", "degraded"]
STATES: Final = frozenset({"ready", "building", "degraded"})
DETAIL_CODES: Final = frozenset(
    {
        "ok",
        "building",
        "version_mismatch",
        "integrity_failure",
        "missing_projection",
    }
)
_STATUS_WITNESS = object()


class SearchIndexStatusError(ValueError):
    """Fixed non-echoing refusal for invalid index status facts."""


@dataclass(frozen=True, slots=True)
class SearchIndexStatusV1:
    schema: Literal["lm-atelier-search-index-status-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    state: IndexState = field(init=False)
    generation: int = field(init=False)
    indexed_through: int = field(init=False)
    detail_code: str = field(init=False)
    query_execution_authorized: Literal[False] = field(init=False)
    fts_write_authorized: Literal[False] = field(init=False)
    index_rebuild_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise SearchIndexStatusError(INVALID_STATUS)


def _invalid() -> NoReturn:
    raise SearchIndexStatusError(INVALID_STATUS)


def _status_from_evaluator(
    *,
    witness: object,
    state: IndexState,
    generation: int,
    indexed_through: int,
    detail_code: str,
) -> SearchIndexStatusV1:
    if witness is not _STATUS_WITNESS:
        _invalid()
    status = object.__new__(SearchIndexStatusV1)
    object.__setattr__(status, "schema", SCHEMA_ID)
    object.__setattr__(status, "schema_version", SCHEMA_VERSION)
    object.__setattr__(status, "state", state)
    object.__setattr__(status, "generation", generation)
    object.__setattr__(status, "indexed_through", indexed_through)
    object.__setattr__(status, "detail_code", detail_code)
    object.__setattr__(status, "query_execution_authorized", False)
    object.__setattr__(status, "fts_write_authorized", False)
    object.__setattr__(status, "index_rebuild_authorized", False)
    return status


def declare_search_index_status(
    *,
    state: object,
    generation: object,
    indexed_through: object,
    detail_code: object,
) -> SearchIndexStatusV1:
    """Classify index readiness from already-measured generation facts."""
    if type(state) is not str:
        _invalid()
    # Equality against short literals rather than frozenset membership: an
    # equality check refuses an attacker-sized string without hashing it
    # first, and each branch proves the Literal to mypy instead of
    # asserting it past a membership test it cannot narrow. The else arm
    # keeps an unknown state fail-closed on its own, so losing any one
    # branch refuses that state rather than minting it.
    narrowed: IndexState
    if state == "ready":
        narrowed = "ready"
    elif state == "building":
        narrowed = "building"
    elif state == "degraded":
        narrowed = "degraded"
    else:
        _invalid()
    if type(generation) is not int or generation < 0 or generation > MAX_GENERATION:
        _invalid()
    if type(indexed_through) is not int or indexed_through < 0 or indexed_through > MAX_SEQUENCE:
        _invalid()
    # Length before membership: DETAIL_CODES is a frozenset, and hashing is the
    # one operation here whose cost scales with attacker-sized input.
    if type(detail_code) is not str or len(detail_code) > MAX_DETAIL_CHARS:
        _invalid()
    if detail_code not in DETAIL_CODES:
        _invalid()
    if narrowed == "ready" and detail_code != "ok":
        _invalid()
    if narrowed == "building" and detail_code != "building":
        _invalid()
    if narrowed == "degraded" and detail_code in {"ok", "building"}:
        _invalid()
    return _status_from_evaluator(
        witness=_STATUS_WITNESS,
        state=narrowed,
        generation=generation,
        indexed_through=indexed_through,
        detail_code=detail_code,
    )
