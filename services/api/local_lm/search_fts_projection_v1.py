"""Pure conversation-search FTS projection identity.

Names the local rebuildable cache only. Never exports and never writes FTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

from .search_text_v1 import MAX_SEARCH_TOKEN_CHARS, require_bounded_exact_str

SCHEMA_ID: Final = "lm-atelier-search-fts-projection-v1"
SCHEMA_VERSION: Final = 1
INVALID_PROJECTION: Final = "search fts projection facts are invalid"
MAX_GENERATION: Final = 2**31 - 1
MAX_DOCUMENTS: Final = 100_000
PROJECTION_NAMES: Final = frozenset({"conversation-fts-v1"})
ProjectionName = Literal["conversation-fts-v1"]
_PROJECTION_WITNESS = object()


class SearchFtsProjectionError(ValueError):
    """Fixed non-echoing refusal for invalid projection facts."""


@dataclass(frozen=True, slots=True)
class SearchFtsProjectionV1:
    schema: Literal["lm-atelier-search-fts-projection-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    projection_name: ProjectionName = field(init=False)
    generation: int = field(init=False)
    document_count: int = field(init=False)
    export_authorized: Literal[False] = field(init=False)
    fts_write_authorized: Literal[False] = field(init=False)
    query_execution_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise SearchFtsProjectionError(INVALID_PROJECTION)


def _invalid() -> NoReturn:
    raise SearchFtsProjectionError(INVALID_PROJECTION)


def _projection_from_evaluator(
    *,
    witness: object,
    projection_name: ProjectionName,
    generation: int,
    document_count: int,
) -> SearchFtsProjectionV1:
    if witness is not _PROJECTION_WITNESS:
        _invalid()
    projection = object.__new__(SearchFtsProjectionV1)
    object.__setattr__(projection, "schema", SCHEMA_ID)
    object.__setattr__(projection, "schema_version", SCHEMA_VERSION)
    object.__setattr__(projection, "projection_name", projection_name)
    object.__setattr__(projection, "generation", generation)
    object.__setattr__(projection, "document_count", document_count)
    object.__setattr__(projection, "export_authorized", False)
    object.__setattr__(projection, "fts_write_authorized", False)
    object.__setattr__(projection, "query_execution_authorized", False)
    return projection


def declare_search_fts_projection(
    *,
    projection_name: object,
    generation: object,
    document_count: object,
) -> SearchFtsProjectionV1:
    """Bind local FTS cache identity without export or write authority."""
    name = require_bounded_exact_str(
        projection_name,
        max_len=MAX_SEARCH_TOKEN_CHARS,
        refuse=_invalid,
    )
    if name not in PROJECTION_NAMES:
        _invalid()
    if type(generation) is not int or generation < 0 or generation > MAX_GENERATION:
        _invalid()
    if type(document_count) is not int or document_count < 0 or document_count > MAX_DOCUMENTS:
        _invalid()
    return _projection_from_evaluator(
        witness=_PROJECTION_WITNESS,
        projection_name="conversation-fts-v1",
        generation=generation,
        document_count=document_count,
    )
