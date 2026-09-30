"""Read branch-wide composer facts without hydrating conversation payloads."""

from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .chat_message_queries import message_ancestry
from .models import Message, MessagePart, ResponseRevision
from .schemas import ChatTranscriptContext


def read_transcript_context(
    session: Session, chat_id: str, head_id: str | None
) -> ChatTranscriptContext:
    visible = select(Message.id).where(
        Message.chat_id == chat_id,
        Message.transcript_visible.is_(True),
        Message.content_removed_at.is_(None),
    )
    if head_id is not None:
        ancestry = message_ancestry(chat_id, head_id)
        visible = visible.where(Message.id.in_(select(ancestry.c.id)))
    visible = visible.correlate(None)
    preview_type = func.json_type(MessagePart.metadata_json, "$.preview")
    visual = select(MessagePart.id).where(
        MessagePart.message_id.in_(visible),
        MessagePart.artifact_id.is_not(None),
        MessagePart.type.in_(("image", "video")),
        or_(preview_type.is_(None), preview_type != "true"),
    )
    pending_revision = (
        select(ResponseRevision.id)
        .where(ResponseRevision.message_id == Message.id, ResponseRevision.status == "pending")
        .correlate(Message)
        .exists()
    )
    pending = select(Message.id).where(
        Message.id.in_(visible), or_(Message.status == "pending", pending_revision)
    )
    has_visual, has_image, has_pending = session.execute(
        select(
            visual.exists(), visual.where(MessagePart.type == "image").exists(), pending.exists()
        )
    ).one()
    return ChatTranscriptContext(
        chat_id=chat_id,
        head_id=head_id,
        has_prior_visual=has_visual,
        has_prior_image=has_image,
        has_pending_response=has_pending,
    )
