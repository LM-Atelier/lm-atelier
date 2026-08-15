"""Requested message-identity preserve policy after content tombstone (item 39).

Records requested_policy only. This module does not delete or rewrite a
message identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-chat-item-removal-identity-preserve-v1"
SCHEMA_VERSION: Final = 1
INVALID_IDENTITY_PRESERVE: Final = "chat item removal identity preserve facts are invalid"
MAX_ID: Final = 128
RequestedPolicy = Literal["preserve_message_id"]
_PRESERVE_WITNESS = object()


class ChatItemRemovalIdentityPreserveError(ValueError):
    """Fixed non-echoing refusal for invalid identity-preserve policy facts."""


@dataclass(frozen=True, slots=True)
class ChatItemRemovalIdentityPreserveV1:
    schema: Literal["lm-atelier-chat-item-removal-identity-preserve-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    message_id: str = field(init=False)
    requested_policy: RequestedPolicy = field(init=False)
    message_delete_authorized: Literal[False] = field(init=False)
    identity_rewrite_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise ChatItemRemovalIdentityPreserveError(INVALID_IDENTITY_PRESERVE)


def _invalid() -> NoReturn:
    raise ChatItemRemovalIdentityPreserveError(INVALID_IDENTITY_PRESERVE)


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


def _preserve_from_evaluator(
    *,
    witness: object,
    message_id: str,
) -> ChatItemRemovalIdentityPreserveV1:
    if witness is not _PRESERVE_WITNESS:
        _invalid()
    preserve = object.__new__(ChatItemRemovalIdentityPreserveV1)
    object.__setattr__(preserve, "schema", SCHEMA_ID)
    object.__setattr__(preserve, "schema_version", SCHEMA_VERSION)
    object.__setattr__(preserve, "message_id", message_id)
    object.__setattr__(preserve, "requested_policy", "preserve_message_id")
    object.__setattr__(preserve, "message_delete_authorized", False)
    object.__setattr__(preserve, "identity_rewrite_authorized", False)
    return preserve


def declare_chat_item_removal_identity_preserve(
    *,
    message_id: object,
) -> ChatItemRemovalIdentityPreserveV1:
    """Record a requested identity preserve without deleting the message."""
    return _preserve_from_evaluator(witness=_PRESERVE_WITNESS, message_id=_require_id(message_id))
