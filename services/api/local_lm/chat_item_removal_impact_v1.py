"""Requested removal-impact preview facts (item 39).

Records caller-declared reply, regeneration, and proposed detach
identities. This module does not verify a repository snapshot or
execute removal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-chat-item-removal-impact-v1"
SCHEMA_VERSION: Final = 1
INVALID_IMPACT: Final = "chat item removal impact facts are invalid"
MAX_ID: Final = 128
MAX_REF_COUNT: Final = 32
MAX_REF_BYTES: Final = 1024
_IMPACT_WITNESS = object()


class ChatItemRemovalImpactError(ValueError):
    """Fixed non-echoing refusal for invalid removal-impact facts."""


@dataclass(frozen=True, slots=True)
class ChatItemRemovalImpactV1:
    schema: Literal["lm-atelier-chat-item-removal-impact-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    message_id: str = field(init=False)
    declared_has_replies: bool = field(init=False)
    declared_source_backs_regeneration: bool = field(init=False)
    proposed_detached_ref_ids: tuple[str, ...] = field(init=False)
    repository_snapshot_verified: Literal[False] = field(init=False)
    message_id_bound: Literal[False] = field(init=False)
    impact_verified: Literal[False] = field(init=False)
    execute_authorized: Literal[False] = field(init=False)
    chat_lock_acquired: Literal[False] = field(init=False)
    mutation_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise ChatItemRemovalImpactError(INVALID_IMPACT)


def _invalid() -> NoReturn:
    raise ChatItemRemovalImpactError(INVALID_IMPACT)


def _utf8_size(value: str) -> int:
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        _invalid()
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        _invalid()


def _require_id(value: object) -> str:
    if type(value) is not str:
        _invalid()
    if not value or len(value) > MAX_ID:
        _invalid()
    if value.strip() != value or any(ch.isspace() for ch in value):
        _invalid()
    if any(ord(ch) < 32 or ord(ch) == 127 or 0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        _invalid()
    return value


def _require_ref_ids(raw_ids: object) -> tuple[str, ...]:
    if type(raw_ids) is not list and type(raw_ids) is not tuple:
        _invalid()
    if len(raw_ids) > MAX_REF_COUNT:
        _invalid()
    total = 0
    for item in raw_ids:
        if type(item) is not str:
            _invalid()
        total += _utf8_size(item)
        if total > MAX_REF_BYTES:
            _invalid()
    collected: list[str] = []
    for item in raw_ids:
        collected.append(_require_id(item))
    ids = tuple(collected)
    if len(ids) != len(set(ids)):
        _invalid()
    return ids


def _impact_from_evaluator(
    *,
    witness: object,
    message_id: str,
    declared_has_replies: bool,
    declared_source_backs_regeneration: bool,
    proposed_detached_ref_ids: tuple[str, ...],
) -> ChatItemRemovalImpactV1:
    if witness is not _IMPACT_WITNESS:
        _invalid()
    impact = object.__new__(ChatItemRemovalImpactV1)
    object.__setattr__(impact, "schema", SCHEMA_ID)
    object.__setattr__(impact, "schema_version", SCHEMA_VERSION)
    object.__setattr__(impact, "message_id", message_id)
    object.__setattr__(impact, "declared_has_replies", declared_has_replies)
    object.__setattr__(
        impact, "declared_source_backs_regeneration", declared_source_backs_regeneration
    )
    object.__setattr__(impact, "proposed_detached_ref_ids", proposed_detached_ref_ids)
    object.__setattr__(impact, "repository_snapshot_verified", False)
    object.__setattr__(impact, "message_id_bound", False)
    object.__setattr__(impact, "impact_verified", False)
    object.__setattr__(impact, "execute_authorized", False)
    object.__setattr__(impact, "chat_lock_acquired", False)
    object.__setattr__(impact, "mutation_authorized", False)
    return impact


def declare_chat_item_removal_impact(
    *,
    message_id: object,
    declared_has_replies: object,
    declared_source_backs_regeneration: object,
    proposed_detached_ref_ids: object,
) -> ChatItemRemovalImpactV1:
    """Record a declared impact preview without verifying the repository."""
    if type(declared_has_replies) is not bool:
        _invalid()
    if type(declared_source_backs_regeneration) is not bool:
        _invalid()
    return _impact_from_evaluator(
        witness=_IMPACT_WITNESS,
        message_id=_require_id(message_id),
        declared_has_replies=declared_has_replies,
        declared_source_backs_regeneration=declared_source_backs_regeneration,
        proposed_detached_ref_ids=_require_ref_ids(proposed_detached_ref_ids),
    )
