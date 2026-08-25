from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

ADAPTER_MESSAGE_PROJECTION_VERSION = "coalesce-adjacent-role-v1"
ADAPTER_MESSAGE_SEPARATOR = "\n\n"
MAX_ADAPTER_PROJECTION_SOURCE_IDS = 64
_COALESCIBLE_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class AdapterMessageProjection:
    messages: list[dict[str, Any]]
    source_message_ids: list[tuple[str, ...]]
    coalesced_message_count: int

    def metadata(self, *, original_message_count: int) -> dict[str, Any]:
        remaining = MAX_ADAPTER_PROJECTION_SOURCE_IDS
        visible_groups: list[list[str]] = []
        source_count = 0
        for group in self.source_message_ids:
            source_count += len(group)
            visible = list(group[: max(0, remaining)])
            visible_groups.append(visible)
            remaining = max(0, remaining - len(visible))
        return {
            "version": ADAPTER_MESSAGE_PROJECTION_VERSION,
            "separator": ADAPTER_MESSAGE_SEPARATOR,
            "original_message_count": original_message_count,
            "projected_message_count": len(self.messages),
            "coalesced_message_count": self.coalesced_message_count,
            "source_message_count": source_count,
            "source_message_ids": visible_groups,
            "source_ids_truncated": source_count > MAX_ADAPTER_PROJECTION_SOURCE_IDS,
        }


def project_chat_messages(
    messages: Sequence[dict[str, Any]],
    source_message_ids: Sequence[str | None] = (),
) -> AdapterMessageProjection:
    """Coalesce adjacent ordinary chat roles without changing stored messages."""

    projected: list[dict[str, Any]] = []
    projected_sources: list[tuple[str, ...]] = []
    coalesced = 0
    for index, message in enumerate(messages):
        candidate = dict(message)
        source_id = source_message_ids[index] if index < len(source_message_ids) else None
        sources = (source_id,) if isinstance(source_id, str) else ()
        if projected and _can_coalesce(projected[-1], candidate):
            merged_content = _merge_content(projected[-1].get("content"), candidate.get("content"))
            if merged_content is not None:
                projected[-1]["content"] = merged_content
                projected_sources[-1] = (*projected_sources[-1], *sources)
                coalesced += 1
                continue
        projected.append(candidate)
        projected_sources.append(sources)
    return AdapterMessageProjection(projected, projected_sources, coalesced)


def _can_coalesce(left: dict[str, Any], right: dict[str, Any]) -> bool:
    role = left.get("role")
    return (
        isinstance(role, str)
        and role in _COALESCIBLE_ROLES
        and right.get("role") == role
        and set(left) <= {"role", "content"}
        and set(right) <= {"role", "content"}
    )


def _merge_content(left: object, right: object) -> str | list[dict[str, Any]] | None:
    if isinstance(left, str) and isinstance(right, str):
        return f"{left}{ADAPTER_MESSAGE_SEPARATOR}{right}"
    left_parts = _content_parts(left)
    right_parts = _content_parts(right)
    if left_parts is None or right_parts is None:
        return None
    return [
        *left_parts,
        {"type": "text", "text": ADAPTER_MESSAGE_SEPARATOR},
        *right_parts,
    ]


def _content_parts(value: object) -> list[dict[str, Any]] | None:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [dict(item) for item in value]
    return None
