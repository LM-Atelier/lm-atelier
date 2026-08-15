"""Requested queue-order policy for prior-turn edit (item 40).

Records requested_policy only. This module does not reorder or activate
the live queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-prior-turn-edit-queue-order-v1"
SCHEMA_VERSION: Final = 1
INVALID_QUEUE_ORDER: Final = "prior turn edit queue order facts are invalid"
MAX_ID: Final = 128
RequestedPolicy = Literal["append_new_ordinal"]
_ORDER_WITNESS = object()


class PriorTurnEditQueueOrderError(ValueError):
    """Fixed non-echoing refusal for invalid queue-order policy facts."""


@dataclass(frozen=True, slots=True)
class PriorTurnEditQueueOrderV1:
    schema: Literal["lm-atelier-prior-turn-edit-queue-order-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    chat_id: str = field(init=False)
    requested_policy: RequestedPolicy = field(init=False)
    reorder_queue_authorized: Literal[False] = field(init=False)
    activate_branch: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise PriorTurnEditQueueOrderError(INVALID_QUEUE_ORDER)


def _invalid() -> NoReturn:
    raise PriorTurnEditQueueOrderError(INVALID_QUEUE_ORDER)


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


def _order_from_evaluator(*, witness: object, chat_id: str) -> PriorTurnEditQueueOrderV1:
    if witness is not _ORDER_WITNESS:
        _invalid()
    order = object.__new__(PriorTurnEditQueueOrderV1)
    object.__setattr__(order, "schema", SCHEMA_ID)
    object.__setattr__(order, "schema_version", SCHEMA_VERSION)
    object.__setattr__(order, "chat_id", chat_id)
    object.__setattr__(order, "requested_policy", "append_new_ordinal")
    object.__setattr__(order, "reorder_queue_authorized", False)
    object.__setattr__(order, "activate_branch", False)
    return order


def declare_prior_turn_edit_queue_order(*, chat_id: object) -> PriorTurnEditQueueOrderV1:
    """Record a requested new ordinal without reordering the live queue."""
    return _order_from_evaluator(witness=_ORDER_WITNESS, chat_id=_require_id(chat_id))
