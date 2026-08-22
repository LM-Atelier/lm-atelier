"""Requested revision-repopulation block after content tombstone (item 39).

Records requested_policy only. This module does not select or block a
revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-chat-item-removal-revision-block-v1"
SCHEMA_VERSION: Final = 1
INVALID_REVISION_BLOCK: Final = "chat item removal revision block facts are invalid"
MAX_ID: Final = 128
RequestedPolicy = Literal["block_revision_repopulation"]
_BLOCK_WITNESS = object()


class ChatItemRemovalRevisionBlockError(ValueError):
    """Fixed non-echoing refusal for invalid revision-block policy facts."""


@dataclass(frozen=True, slots=True)
class ChatItemRemovalRevisionBlockV1:
    schema: Literal["lm-atelier-chat-item-removal-revision-block-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    message_id: str = field(init=False)
    requested_policy: RequestedPolicy = field(init=False)
    revision_repopulation_authorized: Literal[False] = field(init=False)
    execution_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise ChatItemRemovalRevisionBlockError(INVALID_REVISION_BLOCK)


def _invalid() -> NoReturn:
    raise ChatItemRemovalRevisionBlockError(INVALID_REVISION_BLOCK)


def _require_id(value: object) -> str:
    if type(value) is not str:
        _invalid()
    if not value or len(value) > MAX_ID:
        _invalid()
    if value.strip() != value or any(ch.isspace() for ch in value):
        _invalid()
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        _invalid()
    return value


def _block_from_evaluator(
    *,
    witness: object,
    message_id: str,
) -> ChatItemRemovalRevisionBlockV1:
    if witness is not _BLOCK_WITNESS:
        _invalid()
    block = object.__new__(ChatItemRemovalRevisionBlockV1)
    object.__setattr__(block, "schema", SCHEMA_ID)
    object.__setattr__(block, "schema_version", SCHEMA_VERSION)
    object.__setattr__(block, "message_id", message_id)
    object.__setattr__(block, "requested_policy", "block_revision_repopulation")
    object.__setattr__(block, "revision_repopulation_authorized", False)
    object.__setattr__(block, "execution_authorized", False)
    return block


def declare_chat_item_removal_revision_block(
    *,
    message_id: object,
) -> ChatItemRemovalRevisionBlockV1:
    """Record a requested revision block without mutating revisions."""
    return _block_from_evaluator(witness=_BLOCK_WITNESS, message_id=_require_id(message_id))
