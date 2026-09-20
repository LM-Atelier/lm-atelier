"""Read sidebar rows without loading conversation settings or message content."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import islice

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Chat, Project
from .prompt_helpers import STANDARD_CHAT_SCOPE


@dataclass(frozen=True)
class ChatSummaryRow:
    id: str
    project_id: str | None
    title: str
    archived: bool
    pinned: bool
    created_at: datetime
    updated_at: datetime


def list_chat_summary_rows(
    session: Session,
    *,
    project_id: str | None = None,
    include_archived: bool = False,
    query: str = "",
    search_projects: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[ChatSummaryRow]:
    """Select one bounded page, including Unicode project-name search."""
    if not 1 <= limit <= 200 or not 0 <= offset <= 9_223_372_036_854_775_807:
        raise ValueError("The chat summary page request is invalid.")
    if len(query) > 500:
        raise ValueError("The chat summary search is too long.")
    statement = (
        select(
            Chat.id,
            Chat.project_id,
            Chat.title,
            Chat.archived,
            Chat.pinned,
            Chat.created_at,
            Chat.updated_at,
        )
        .where(Chat.scope == STANDARD_CHAT_SCOPE)
        .order_by(Chat.pinned.desc(), Chat.updated_at.desc(), Chat.id.desc())
    )
    if project_id:
        statement = statement.where(Chat.project_id == project_id)
    if not include_archived:
        statement = statement.where(Chat.archived.is_(False))
    normalized = query.strip().lower()
    with session.no_autoflush:
        if normalized and search_projects:
            names = statement.add_columns(Project.name).outerjoin(
                Project, Project.id == Chat.project_id
            )
            with session.execute(names.execution_options(yield_per=200)) as candidates:
                matching = (
                    ChatSummaryRow(*row[:7])
                    for row in candidates
                    if normalized in row.title.lower() or normalized in (row.name or "").lower()
                )
                return list(islice(islice(matching, offset, None), limit))
        if normalized:
            statement = statement.where(Chat.title.ilike(f"%{query.strip()}%"))
        return [
            ChatSummaryRow(*row) for row in session.execute(statement.limit(limit).offset(offset))
        ]
