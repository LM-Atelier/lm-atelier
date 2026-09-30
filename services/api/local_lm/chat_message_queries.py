"""Select complete conversation branches without loading message payloads."""

from __future__ import annotations

from sqlalchemy import func, literal, select
from sqlalchemy.sql.selectable import CTE

from .models import Message


def message_ancestry(chat_id: str, head_id: str) -> CTE:
    """Visit each parent identity once, even when the graph contains a cycle."""
    ancestry = (
        select(Message.id, Message.parent_id)
        .where(Message.id == head_id, Message.chat_id == chat_id)
        .cte("message_ancestry", recursive=True)
    )
    return ancestry.union(
        select(Message.id, Message.parent_id)
        .join(ancestry, Message.id == ancestry.c.parent_id)
        .where(Message.chat_id == chat_id)
    )


def message_ancestry_positions(chat_id: str, head_id: str) -> CTE:
    """Count steps from the head, stopping before a parent identity repeats."""
    ancestry = message_ancestry(chat_id, head_id)
    size = select(func.count()).select_from(ancestry).scalar_subquery()
    positions = (
        select(Message.id, Message.parent_id, literal(0).label("depth"))
        .where(Message.id == head_id, Message.chat_id == chat_id)
        .cte("message_ancestry_positions", recursive=True)
    )
    # Each message has one parent, so the unique ancestry count bounds this walk
    # even when the head leads into a cycle that does not contain the head itself.
    return positions.union_all(
        select(Message.id, Message.parent_id, (positions.c.depth + 1).label("depth"))
        .join(positions, Message.id == positions.c.parent_id)
        .where(Message.chat_id == chat_id, positions.c.depth + 1 < size)
    )
