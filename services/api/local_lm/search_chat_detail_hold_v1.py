"""Requested legacy ChatDetail retention policy.

Records requested_policy and a caller-declared consumer count only.
This module does not inventory consumers or remove an endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-search-chat-detail-hold-v1"
SCHEMA_VERSION: Final = 1
INVALID_HOLD: Final = "search chat-detail hold facts are invalid"
MAX_CONSUMERS: Final = 256
RequestedPolicy = Literal["retain_legacy_chat_detail"]
_HOLD_WITNESS = object()


class SearchChatDetailHoldError(ValueError):
    """Fixed non-echoing refusal for invalid hold policy facts."""


@dataclass(frozen=True, slots=True)
class SearchChatDetailHoldV1:
    schema: Literal["lm-atelier-search-chat-detail-hold-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    requested_consumer_count: int = field(init=False)
    requested_policy: RequestedPolicy = field(init=False)
    removal_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise SearchChatDetailHoldError(INVALID_HOLD)


def _invalid() -> NoReturn:
    raise SearchChatDetailHoldError(INVALID_HOLD)


def _hold_from_evaluator(
    *,
    witness: object,
    requested_consumer_count: int,
) -> SearchChatDetailHoldV1:
    if witness is not _HOLD_WITNESS:
        _invalid()
    hold = object.__new__(SearchChatDetailHoldV1)
    object.__setattr__(hold, "schema", SCHEMA_ID)
    object.__setattr__(hold, "schema_version", SCHEMA_VERSION)
    object.__setattr__(hold, "requested_consumer_count", requested_consumer_count)
    object.__setattr__(hold, "requested_policy", "retain_legacy_chat_detail")
    object.__setattr__(hold, "removal_authorized", False)
    return hold


def declare_search_chat_detail_hold(
    *,
    requested_consumer_count: object,
) -> SearchChatDetailHoldV1:
    """Record a requested ChatDetail hold without authorizing removal."""
    if (
        type(requested_consumer_count) is not int
        or requested_consumer_count < 1
        or requested_consumer_count > MAX_CONSUMERS
    ):
        _invalid()
    return _hold_from_evaluator(
        witness=_HOLD_WITNESS,
        requested_consumer_count=requested_consumer_count,
    )
