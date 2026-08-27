"""Requested payload-detach policy after content tombstone.

Records requested_policy only. This module does not detach parts or
delete a message identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-chat-item-removal-payload-detach-v1"
SCHEMA_VERSION: Final = 1
INVALID_PAYLOAD_DETACH: Final = "chat item removal payload detach facts are invalid"
MAX_ID: Final = 128
RequestedPolicy = Literal["detach_payload_parts"]
_DETACH_WITNESS = object()


class ChatItemRemovalPayloadDetachError(ValueError):
    """Fixed non-echoing refusal for invalid payload-detach policy facts."""


@dataclass(frozen=True, slots=True)
class ChatItemRemovalPayloadDetachV1:
    schema: Literal["lm-atelier-chat-item-removal-payload-detach-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    message_id: str = field(init=False)
    requested_policy: RequestedPolicy = field(init=False)
    payload_detach_authorized: Literal[False] = field(init=False)
    message_delete_authorized: Literal[False] = field(init=False)
    forensic_erasure: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise ChatItemRemovalPayloadDetachError(INVALID_PAYLOAD_DETACH)


def _invalid() -> NoReturn:
    raise ChatItemRemovalPayloadDetachError(INVALID_PAYLOAD_DETACH)


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


def _detach_from_evaluator(
    *,
    witness: object,
    message_id: str,
) -> ChatItemRemovalPayloadDetachV1:
    if witness is not _DETACH_WITNESS:
        _invalid()
    detach = object.__new__(ChatItemRemovalPayloadDetachV1)
    object.__setattr__(detach, "schema", SCHEMA_ID)
    object.__setattr__(detach, "schema_version", SCHEMA_VERSION)
    object.__setattr__(detach, "message_id", message_id)
    object.__setattr__(detach, "requested_policy", "detach_payload_parts")
    object.__setattr__(detach, "payload_detach_authorized", False)
    object.__setattr__(detach, "message_delete_authorized", False)
    object.__setattr__(detach, "forensic_erasure", False)
    return detach


def declare_chat_item_removal_payload_detach(
    *,
    message_id: object,
) -> ChatItemRemovalPayloadDetachV1:
    """Record a requested payload detach without mutating parts."""
    return _detach_from_evaluator(witness=_DETACH_WITNESS, message_id=_require_id(message_id))
