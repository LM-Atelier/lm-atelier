"""Requested replay-refuse policy after content tombstone.

Records requested_policy only. This module does not omit context or
block a replay path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-chat-item-removal-replay-refuse-v1"
SCHEMA_VERSION: Final = 1
INVALID_REPLAY_REFUSE: Final = "chat item removal replay refuse facts are invalid"
MAX_ID: Final = 128
RequestedPolicy = Literal["source_content_removed"]
_REPLAY_WITNESS = object()


class ChatItemRemovalReplayRefuseError(ValueError):
    """Fixed non-echoing refusal for invalid replay-refuse policy facts."""


@dataclass(frozen=True, slots=True)
class ChatItemRemovalReplayRefuseV1:
    schema: Literal["lm-atelier-chat-item-removal-replay-refuse-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    message_id: str = field(init=False)
    requested_policy: RequestedPolicy = field(init=False)
    replay_authorized: Literal[False] = field(init=False)
    context_omission_verified: Literal[False] = field(init=False)
    forensic_erasure: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise ChatItemRemovalReplayRefuseError(INVALID_REPLAY_REFUSE)


def _invalid() -> NoReturn:
    raise ChatItemRemovalReplayRefuseError(INVALID_REPLAY_REFUSE)


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


def _refuse_from_evaluator(*, witness: object, message_id: str) -> ChatItemRemovalReplayRefuseV1:
    if witness is not _REPLAY_WITNESS:
        _invalid()
    refuse = object.__new__(ChatItemRemovalReplayRefuseV1)
    object.__setattr__(refuse, "schema", SCHEMA_ID)
    object.__setattr__(refuse, "schema_version", SCHEMA_VERSION)
    object.__setattr__(refuse, "message_id", message_id)
    object.__setattr__(refuse, "requested_policy", "source_content_removed")
    object.__setattr__(refuse, "replay_authorized", False)
    object.__setattr__(refuse, "context_omission_verified", False)
    object.__setattr__(refuse, "forensic_erasure", False)
    return refuse


def declare_chat_item_removal_replay_refuse(*, message_id: object) -> ChatItemRemovalReplayRefuseV1:
    """Record a requested replay refuse without omitting context."""
    return _refuse_from_evaluator(witness=_REPLAY_WITNESS, message_id=_require_id(message_id))
