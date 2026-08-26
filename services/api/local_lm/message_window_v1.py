"""Pure bounded chat message window planner.

Plans latest/older/newer/around windows from caller-supplied ordered message
ids. No DB, branch activation, or full-chat load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

SCHEMA_ID: Final = "lm-atelier-message-window-v1"
SCHEMA_VERSION: Final = 1
MAX_WINDOW: Final = 100
MAX_INPUT_IDS: Final = 256
MAX_INPUT_CHARS: Final = 4096
MAX_ID_CHARS: Final = 40
DEFAULT_WINDOW: Final = 40
INVALID_WINDOW: Final = "invalid-message-window"
WindowMode = Literal["latest", "older", "newer", "around"]


class MessageWindowError(ValueError):
    """Fixed non-echoing refusal for invalid window plans."""


@dataclass(frozen=True, slots=True)
class MessageWindowPlan:
    schema: Literal["lm-atelier-message-window-v1"]
    schema_version: Literal[1]
    mode: WindowMode
    message_ids: tuple[str, ...]
    anchor_id: str | None
    loads_entire_chat: Literal[False] = field(default=False, init=False)
    branch_activation_authorized: Literal[False] = field(default=False, init=False)


def _require_id(value: object) -> str:
    if type(value) is not str or not value:
        raise MessageWindowError(INVALID_WINDOW)
    if value.strip() != value or any(ch.isspace() for ch in value):
        raise MessageWindowError(INVALID_WINDOW)
    if len(value) > MAX_ID_CHARS:
        raise MessageWindowError(INVALID_WINDOW)
    return value


def plan_message_window(
    ordered_ids: object,
    *,
    mode: object,
    anchor_id: object = None,
    limit: object = DEFAULT_WINDOW,
) -> MessageWindowPlan:
    """Return a bounded window over already-ordered visible message ids.

    ordered_ids is oldest-first. Modes: latest | older | newer | around.
    Duplicate ids are refused. Never loads an entire chat.
    """
    if type(ordered_ids) is not list and type(ordered_ids) is not tuple:
        raise MessageWindowError(INVALID_WINDOW)
    if not ordered_ids or len(ordered_ids) > MAX_INPUT_IDS:
        raise MessageWindowError(INVALID_WINDOW)
    if type(mode) is not str or mode not in {"latest", "older", "newer", "around"}:
        raise MessageWindowError(INVALID_WINDOW)
    if type(limit) is not int or limit < 1 or limit > MAX_WINDOW:
        raise MessageWindowError(INVALID_WINDOW)

    total_chars = 0
    for item in ordered_ids:
        if type(item) is not str:
            raise MessageWindowError(INVALID_WINDOW)
        total_chars += len(item)
        if total_chars > MAX_INPUT_CHARS:
            raise MessageWindowError(INVALID_WINDOW)

    seen: set[str] = set()
    ids: list[str] = []
    for item in ordered_ids:
        mid = _require_id(item)
        if mid in seen:
            raise MessageWindowError(INVALID_WINDOW)
        seen.add(mid)
        ids.append(mid)
    id_tuple = tuple(ids)

    if mode == "latest":
        if anchor_id is not None:
            raise MessageWindowError(INVALID_WINDOW)
        return MessageWindowPlan(
            schema=SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            mode="latest",
            message_ids=id_tuple[-limit:],
            anchor_id=None,
        )

    anchor = _require_id(anchor_id)
    if anchor not in seen:
        raise MessageWindowError(INVALID_WINDOW)
    idx = ids.index(anchor)
    window: tuple[str, ...]
    planned: WindowMode
    if mode == "older":
        start = max(0, idx - limit)
        window = id_tuple[start:idx]
        planned = "older"
    elif mode == "newer":
        end = min(len(ids), idx + 1 + limit)
        window = id_tuple[idx + 1 : end]
        planned = "newer"
    else:
        before = limit // 2
        after = limit - before - 1
        start = max(0, idx - before)
        end = min(len(ids), idx + 1 + after)
        while end - start < min(limit, len(ids)) and start > 0:
            start -= 1
        while end - start < min(limit, len(ids)) and end < len(ids):
            end += 1
        window = id_tuple[start:end]
        planned = "around"
    return MessageWindowPlan(
        schema=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        mode=planned,
        message_ids=window,
        anchor_id=anchor,
    )
