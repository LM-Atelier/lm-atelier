"""Page image edit history without hydrating the conversation."""

from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .api_errors import api_error
from .chat_message_queries import message_ancestry
from .chat_recovery_visibility import visible_chat
from .models import Chat, Message, MessagePart
from .prompt_helpers import STANDARD_CHAT_SCOPE
from .schemas import ChatEditLineagePage, ChatEditLineageStep


def _nearest_ancestor(
    session: Session, chat_id: str, start_id: str, match: ColumnElement[bool]
) -> str | None:
    """Follow parent links until the first match, visiting cyclic identities only once."""
    walk = (
        select(Message.id, Message.parent_id, match.label("matched"))
        .where(Message.chat_id == chat_id, Message.id == start_id)
        .cte("edit_ancestor", recursive=True)
    )
    walk = walk.union(
        select(Message.id, Message.parent_id, match.label("matched"))
        .join(walk, Message.id == walk.c.parent_id)
        .where(Message.chat_id == chat_id, walk.c.matched.is_(False))
    )
    return session.scalar(select(walk.c.id).where(walk.c.matched.is_(True)))


def read_edit_lineage(
    session: Session, chat_id: str, result_message_id: str, *, before: str | None, limit: int
) -> ChatEditLineagePage:
    """Return newest steps first; the last returned user message anchors the next page."""
    if not 1 <= limit <= 100:
        raise api_error(400, "chat-lineage-window-invalid", "A page holds between 1 and 100 edits.")
    if (
        session.scalar(
            select(Chat.id).where(
                Chat.id == chat_id, Chat.scope == STANDARD_CHAT_SCOPE, visible_chat(Chat.id)
            )
        )
        is None
    ):
        raise api_error(404, "chat-not-found", "chat not found")
    result = session.scalar(
        select(Message.id).where(
            Message.id == result_message_id,
            Message.chat_id == chat_id,
            Message.role == "assistant",
            Message.transcript_visible.is_(True),
            Message.content_removed_at.is_(None),
        )
    )
    if result is None:
        raise api_error(404, "message-not-found", "This result is not in this conversation.")
    ancestry = message_ancestry(chat_id, result_message_id)
    terminal = session.scalar(
        select(ancestry.c.id)
        .where(
            or_(
                ancestry.c.parent_id.is_(None),
                ancestry.c.parent_id.not_in(select(ancestry.c.id)),
            )
        )
        .limit(1)
    )
    if terminal is None:
        raise api_error(409, "chat-lineage-cycle", "This edit history contains a parent cycle.")

    def input_reference(user_id: str) -> str | None:
        return session.scalar(
            select(MessagePart.artifact_id)
            .where(
                MessagePart.message_id == user_id,
                MessagePart.type == "image",
                MessagePart.artifact_id.is_not(None),
                func.json_type(MessagePart.metadata_json, "$.input_reference") == "true",
            )
            .order_by(MessagePart.position.desc())
            .limit(1)
        )

    def producer(user_id: str, artifact_id: str) -> str | None:
        output = (
            select(MessagePart.id)
            .where(
                MessagePart.message_id == Message.id,
                MessagePart.type == "image",
                MessagePart.artifact_id == artifact_id,
                or_(
                    func.json_type(MessagePart.metadata_json, "$.input_reference").is_(None),
                    func.json_type(MessagePart.metadata_json, "$.input_reference") != "true",
                ),
                or_(
                    func.json_type(MessagePart.metadata_json, "$.preview").is_(None),
                    func.json_type(MessagePart.metadata_json, "$.preview") != "true",
                ),
            )
            .exists()
        )
        return _nearest_ancestor(
            session,
            chat_id,
            user_id,
            (Message.role == "assistant")
            & Message.transcript_visible.is_(True)
            & Message.content_removed_at.is_(None)
            & output,
        )

    position: str | None = result
    if before is not None:
        anchor = session.scalar(
            select(Message.id).where(
                Message.chat_id == chat_id,
                Message.id == before,
                Message.role == "user",
            )
        )
        if anchor is None:
            raise api_error(404, "message-not-found", "This edit is not in this conversation.")
        if session.scalar(select(ancestry.c.id).where(ancestry.c.id == before)) is None:
            raise api_error(
                400, "chat-lineage-window-invalid", "The anchor is not in the result's branch."
            )
        available = session.scalar(
            select(Message.id).where(
                Message.id == before,
                Message.transcript_visible.is_(True),
                Message.content_removed_at.is_(None),
            )
        )
        reference = input_reference(before) if available is not None else None
        position = producer(anchor, reference) if reference else None

    steps: list[ChatEditLineageStep] = []
    while position is not None and len(steps) <= limit:
        user_id = _nearest_ancestor(session, chat_id, position, Message.role == "user")
        user = session.execute(
            select(
                Message.id,
                Message.transcript_visible,
                Message.content_removed_at,
            ).where(
                Message.chat_id == chat_id,
                Message.id == user_id,
            )
        ).one_or_none()
        # A hidden or removed input is a boundary, not permission to use another turn.
        if user is None or not user.transcript_visible or user.content_removed_at is not None:
            break
        reference = input_reference(user.id)
        if reference is None:
            break
        instruction = session.scalar(
            select(MessagePart.text)
            .where(
                MessagePart.message_id == user.id,
                MessagePart.type == "text",
                MessagePart.text != "",
            )
            .order_by(MessagePart.position)
            .limit(1)
        )
        steps.append(
            ChatEditLineageStep(
                message_id=user.id,
                artifact_id=reference,
                instruction=instruction or "",
            )
        )
        position = producer(user.id, reference)
    return ChatEditLineagePage(
        chat_id=chat_id,
        result_message_id=result_message_id,
        steps=steps[:limit],
        next_before=steps[limit - 1].message_id if len(steps) > limit else None,
    )
