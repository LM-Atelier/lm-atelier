"""Requested source-preserve policy for prior-turn edit (item 40).

Records requested_policy only. This module does not mutate or cancel
the source turn.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-prior-turn-edit-source-preserve-v1"
SCHEMA_VERSION: Final = 1
INVALID_SOURCE_PRESERVE: Final = "prior turn edit source preserve facts are invalid"
MAX_ID: Final = 128
RequestedPolicy = Literal["no_source_mutation"]
_PRESERVE_WITNESS = object()


class PriorTurnEditSourcePreserveError(ValueError):
    """Fixed non-echoing refusal for invalid source-preserve policy facts."""


@dataclass(frozen=True, slots=True)
class PriorTurnEditSourcePreserveV1:
    schema: Literal["lm-atelier-prior-turn-edit-source-preserve-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    source_message_id: str = field(init=False)
    requested_policy: RequestedPolicy = field(init=False)
    source_mutation_authorized: Literal[False] = field(init=False)
    cancel_source_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise PriorTurnEditSourcePreserveError(INVALID_SOURCE_PRESERVE)


def _invalid() -> NoReturn:
    raise PriorTurnEditSourcePreserveError(INVALID_SOURCE_PRESERVE)


def _require_id(value: object) -> str:
    if type(value) is not str:
        _invalid()
    if not value or len(value) > MAX_ID:
        _invalid()
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        _invalid()
    for ch in value:
        category = unicodedata.category(ch)
        if category == "Cc" or category == "Cf":
            _invalid()
    if value.strip() != value or any(ch.isspace() for ch in value):
        _invalid()
    if "/" in value or "\\" in value:
        _invalid()
    if value in {".", ".."}:
        _invalid()
    return value


def _preserve_from_evaluator(
    *,
    witness: object,
    source_message_id: str,
) -> PriorTurnEditSourcePreserveV1:
    if witness is not _PRESERVE_WITNESS:
        _invalid()
    preserve = object.__new__(PriorTurnEditSourcePreserveV1)
    object.__setattr__(preserve, "schema", SCHEMA_ID)
    object.__setattr__(preserve, "schema_version", SCHEMA_VERSION)
    object.__setattr__(preserve, "source_message_id", source_message_id)
    object.__setattr__(preserve, "requested_policy", "no_source_mutation")
    object.__setattr__(preserve, "source_mutation_authorized", False)
    object.__setattr__(preserve, "cancel_source_authorized", False)
    return preserve


def declare_prior_turn_edit_source_preserve(
    *,
    source_message_id: object,
) -> PriorTurnEditSourcePreserveV1:
    """Record a requested source preserve without mutating the source."""
    return _preserve_from_evaluator(
        witness=_PRESERVE_WITNESS, source_message_id=_require_id(source_message_id)
    )
