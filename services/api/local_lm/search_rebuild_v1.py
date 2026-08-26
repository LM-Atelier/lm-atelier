"""Pure conversation-search rebuild watermark facts.

Records shadow / catch-up / swap identity only. Never writes FTS or swaps
a live generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

from .search_text_v1 import MAX_SEARCH_TOKEN_CHARS, require_bounded_exact_str

SCHEMA_ID: Final = "lm-atelier-search-rebuild-v1"
SCHEMA_VERSION: Final = 1
INVALID_REBUILD: Final = "search rebuild facts are invalid"
MAX_GENERATION: Final = 2**31 - 1
MAX_SEQUENCE: Final = 2**31 - 1
RebuildPhase = Literal["shadow", "catch_up", "swap"]
PHASES: Final = frozenset({"shadow", "catch_up", "swap"})
_REBUILD_WITNESS = object()


class SearchRebuildError(ValueError):
    """Fixed non-echoing refusal for invalid rebuild facts."""


@dataclass(frozen=True, slots=True)
class SearchRebuildV1:
    schema: Literal["lm-atelier-search-rebuild-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    phase: RebuildPhase = field(init=False)
    start_sequence: int = field(init=False)
    end_sequence: int = field(init=False)
    from_generation: int = field(init=False)
    to_generation: int = field(init=False)
    query_execution_authorized: Literal[False] = field(init=False)
    fts_write_authorized: Literal[False] = field(init=False)
    generation_swap_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise SearchRebuildError(INVALID_REBUILD)


def _invalid() -> NoReturn:
    raise SearchRebuildError(INVALID_REBUILD)


def _require_int(value: object, *, maximum: int, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        _invalid()
    return value


def _rebuild_from_evaluator(
    *,
    witness: object,
    phase: RebuildPhase,
    start_sequence: int,
    end_sequence: int,
    from_generation: int,
    to_generation: int,
) -> SearchRebuildV1:
    if witness is not _REBUILD_WITNESS:
        _invalid()
    rebuild = object.__new__(SearchRebuildV1)
    object.__setattr__(rebuild, "schema", SCHEMA_ID)
    object.__setattr__(rebuild, "schema_version", SCHEMA_VERSION)
    object.__setattr__(rebuild, "phase", phase)
    object.__setattr__(rebuild, "start_sequence", start_sequence)
    object.__setattr__(rebuild, "end_sequence", end_sequence)
    object.__setattr__(rebuild, "from_generation", from_generation)
    object.__setattr__(rebuild, "to_generation", to_generation)
    object.__setattr__(rebuild, "query_execution_authorized", False)
    object.__setattr__(rebuild, "fts_write_authorized", False)
    object.__setattr__(rebuild, "generation_swap_authorized", False)
    return rebuild


def declare_search_rebuild(
    *,
    phase: object,
    start_sequence: object,
    end_sequence: object,
    from_generation: object,
    to_generation: object,
) -> SearchRebuildV1:
    """Bind rebuild watermarks without granting write or swap authority."""
    phase_text = require_bounded_exact_str(phase, max_len=MAX_SEARCH_TOKEN_CHARS, refuse=_invalid)
    if phase_text not in PHASES:
        _invalid()
    start = _require_int(start_sequence, maximum=MAX_SEQUENCE)
    end = _require_int(end_sequence, maximum=MAX_SEQUENCE)
    source = _require_int(from_generation, maximum=MAX_GENERATION)
    target = _require_int(to_generation, maximum=MAX_GENERATION)
    if end < start or target <= source:
        _invalid()
    planned: RebuildPhase
    if phase_text == "shadow":
        planned = "shadow"
    elif phase_text == "catch_up":
        planned = "catch_up"
    else:
        planned = "swap"
    return _rebuild_from_evaluator(
        witness=_REBUILD_WITNESS,
        phase=planned,
        start_sequence=start,
        end_sequence=end,
        from_generation=source,
        to_generation=target,
    )
