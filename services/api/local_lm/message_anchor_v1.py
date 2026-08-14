"""Pure chat message URL anchor encoding (item 41).

Builds and parses device-local fragment anchors. Does not touch history API or
change active branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

MAX_ID_CHARS: Final = 40
MAX_FRAGMENT_CHARS: Final = 48
INVALID_ANCHOR: Final = "invalid-message-anchor"


class AnchorError(ValueError):
    """Fixed refusal for invalid anchors."""


@dataclass(frozen=True, slots=True)
class MessageAnchorV1:
    message_id: str
    changes_active_branch: Literal[False] = field(default=False, init=False)
    history_write_authorized: Literal[False] = field(default=False, init=False)


def _require_id(value: object) -> str:
    if type(value) is not str or not value or len(value) > MAX_ID_CHARS:
        raise AnchorError(INVALID_ANCHOR)
    if any(ch.isspace() for ch in value) or "#" in value or "/" in value or "=" in value:
        raise AnchorError(INVALID_ANCHOR)
    return value


def encode_message_anchor(message_id: object) -> str:
    mid = _require_id(message_id)
    return f"msg={mid}"


def parse_message_anchor(fragment: object) -> MessageAnchorV1:
    if type(fragment) is not str or not fragment:
        raise AnchorError(INVALID_ANCHOR)
    if len(fragment) > MAX_FRAGMENT_CHARS:
        raise AnchorError(INVALID_ANCHOR)
    text = fragment[1:] if fragment.startswith("#") else fragment
    if not text.startswith("msg=") or len(text) <= 4:
        raise AnchorError(INVALID_ANCHOR)
    mid = _require_id(text[4:])
    return MessageAnchorV1(message_id=mid)
