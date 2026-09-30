"""Read bounded search history and keep every pending decision reachable."""

from __future__ import annotations

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from .api_errors import api_error
from .chat_message_queries import message_ancestry
from .models import Chat, Job, Message, Run, WebSearchProposal
from .prompt_helpers import STANDARD_CHAT_SCOPE
from .schemas import ChatSearchPage
from .web_search_projection import _project


def read_search_page(
    session: Session,
    chat_id: str,
    *,
    head_id: str | None,
    oldest_message_id: str | None,
    before: str | None,
    pending_only: bool,
    limit: int,
) -> ChatSearchPage:
    if not 1 <= limit <= 100:
        raise api_error(
            400, "chat-search-window-invalid", "A page holds between 1 and 100 searches."
        )
    if pending_only and (head_id is not None or oldest_message_id is not None):
        raise api_error(
            400,
            "chat-search-window-invalid",
            "Pending decisions include every conversation branch.",
        )
    if (
        session.scalar(select(Chat.id).where(Chat.id == chat_id, Chat.scope == STANDARD_CHAT_SCOPE))
        is None
    ):
        raise api_error(404, "chat-not-found", "chat not found")

    messages = select(Message.id).where(Message.chat_id == chat_id)
    if not pending_only:
        messages = messages.where(
            Message.transcript_visible.is_(True), Message.content_removed_at.is_(None)
        )
    if head_id is not None:
        if (
            session.scalar(
                select(Message.id).where(Message.id == head_id, Message.chat_id == chat_id)
            )
            is None
        ):
            raise api_error(404, "message-not-found", "This message is not in this conversation")
        ancestry = message_ancestry(chat_id, head_id)
        messages = messages.where(Message.id.in_(select(ancestry.c.id)))
    if oldest_message_id is not None:
        oldest = session.execute(
            select(Message.id, Message.created_at).where(
                Message.id == oldest_message_id, Message.chat_id == chat_id
            )
        ).one_or_none()
        if oldest is None:
            raise api_error(404, "message-not-found", "This message is not in this conversation")
        if (
            head_id is not None
            and session.scalar(select(ancestry.c.id).where(ancestry.c.id == oldest_message_id))
            is None
        ):
            raise api_error(
                400, "chat-search-window-invalid", "The anchor is not in the selected branch."
            )
        messages = messages.where(
            or_(
                Message.created_at > oldest.created_at,
                and_(Message.created_at == oldest.created_at, Message.id >= oldest.id),
            )
        )

    scope = (
        Run.chat_id == chat_id,
        Run.operation == "text",
        Run.assistant_message_id.in_(messages),
    )
    query = (
        select(Run, WebSearchProposal, Job.status)
        .outerjoin(WebSearchProposal, WebSearchProposal.run_id == Run.id)
        .outerjoin(Job, Job.id == WebSearchProposal.job_id)
        .where(*scope)
    )
    if pending_only:
        query = query.where(
            WebSearchProposal.state.in_(("awaiting_approval", "approved", "scheduled")),
            Job.status.in_(("running", "queued", "paused")),
        )
    else:
        query = query.where(
            or_(
                WebSearchProposal.id.is_not(None),
                func.json_type(Run.provenance_json, "$.web_search") == "object",
            )
        )
    if before is not None:
        # A decision can finish between pages. Its cursor still orders history.
        anchor = session.execute(
            select(Run.id, Run.created_at).where(*scope, Run.id == before)
        ).one_or_none()
        if anchor is None:
            raise api_error(404, "search-not-found", "This search is not in the requested history.")
        query = query.where(
            or_(
                Run.created_at < anchor.created_at,
                and_(Run.created_at == anchor.created_at, Run.id < anchor.id),
            )
        )
    rows = session.execute(
        query.order_by(Run.created_at.desc(), Run.id.desc()).limit(limit + 1)
    ).all()
    page = rows[:limit]
    searches = []
    for run, proposal, status in reversed(page):
        projected = _project(run, proposal, status)
        if projected is not None:
            searches.append(projected)
    return ChatSearchPage(
        chat_id=chat_id,
        searches=searches,
        # Raw rows advance the page even when every projection is malformed.
        next_before=page[-1][0].id if len(rows) > limit else None,
    )
