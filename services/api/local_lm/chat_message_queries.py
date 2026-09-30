"""Select complete conversation branches without loading message payloads."""

from __future__ import annotations

from sqlalchemy import select
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
