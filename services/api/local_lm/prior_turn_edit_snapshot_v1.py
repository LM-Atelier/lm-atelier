"""Requested acceptance-snapshot facts for prior-turn edit.

Records a digest and bounded context identities. This descriptive value is not
an in-process authorization boundary and does not accept, persist, or authorize
late-bound execution.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-prior-turn-edit-snapshot-v1"
SCHEMA_VERSION: Final = 1
INVALID_SNAPSHOT: Final = "prior turn edit snapshot facts are invalid"
MAX_ID: Final = 128
MAX_CONTEXT_COUNT: Final = 32
MAX_CONTEXT_BYTES: Final = 1024
DIGEST_LEN: Final = 64
DIGEST_CHARS: Final = frozenset("0123456789abcdef")
_SNAPSHOT_WITNESS = object()


class PriorTurnEditSnapshotError(ValueError):
    """Fixed non-echoing refusal for invalid edit-snapshot facts."""


@dataclass(frozen=True, slots=True)
class PriorTurnEditSnapshotV1:
    schema: Literal["lm-atelier-prior-turn-edit-snapshot-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    source_message_id: str = field(init=False)
    snapshot_digest: str = field(init=False)
    context_message_ids: tuple[str, ...] = field(init=False)
    accepted: Literal[False] = field(init=False)
    execution_authorized: Literal[False] = field(init=False)
    late_bind_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise PriorTurnEditSnapshotError(INVALID_SNAPSHOT)


def _invalid() -> NoReturn:
    raise PriorTurnEditSnapshotError(INVALID_SNAPSHOT)


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


def _require_digest(value: object) -> str:
    if type(value) is not str:
        _invalid()
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        _invalid()
    if len(value) != DIGEST_LEN:
        _invalid()
    if any(ch not in DIGEST_CHARS for ch in value):
        _invalid()
    return value


def _require_context_ids(raw_ids: object) -> tuple[str, ...]:
    if type(raw_ids) is not list and type(raw_ids) is not tuple:
        _invalid()
    if len(raw_ids) > MAX_CONTEXT_COUNT:
        _invalid()
    captured_ids = tuple(raw_ids)
    if len(captured_ids) > MAX_CONTEXT_COUNT:
        _invalid()

    total = 0
    collected: list[str] = []
    for raw_item in captured_ids:
        if len(collected) >= MAX_CONTEXT_COUNT:
            _invalid()
        item = _require_id(raw_item)
        total += _utf8_size(item)
        if total > MAX_CONTEXT_BYTES:
            _invalid()
        collected.append(item)
    ids = tuple(collected)
    if len(ids) != len(set(ids)):
        _invalid()
    return ids


def _snapshot_from_evaluator(
    *,
    witness: object,
    source_message_id: str,
    snapshot_digest: str,
    context_message_ids: tuple[str, ...],
) -> PriorTurnEditSnapshotV1:
    if witness is not _SNAPSHOT_WITNESS:
        _invalid()
    snapshot = object.__new__(PriorTurnEditSnapshotV1)
    object.__setattr__(snapshot, "schema", SCHEMA_ID)
    object.__setattr__(snapshot, "schema_version", SCHEMA_VERSION)
    object.__setattr__(snapshot, "source_message_id", source_message_id)
    object.__setattr__(snapshot, "snapshot_digest", snapshot_digest)
    object.__setattr__(snapshot, "context_message_ids", context_message_ids)
    object.__setattr__(snapshot, "accepted", False)
    object.__setattr__(snapshot, "execution_authorized", False)
    object.__setattr__(snapshot, "late_bind_authorized", False)
    return snapshot


def declare_prior_turn_edit_snapshot(
    *,
    source_message_id: object,
    snapshot_digest: object,
    context_message_ids: object,
) -> PriorTurnEditSnapshotV1:
    """Record a snapshot digest without accepting or late-binding execution."""
    return _snapshot_from_evaluator(
        witness=_SNAPSHOT_WITNESS,
        source_message_id=_require_id(source_message_id),
        snapshot_digest=_require_digest(snapshot_digest),
        context_message_ids=_require_context_ids(context_message_ids),
    )
