"""Read bounded terminal identities without loading conversation content."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from .models import Chat, ChatActivityEvent, Message, ResponseRevision, ResponseRevisionPart
from .prompt_helpers import STANDARD_CHAT_SCOPE


@dataclass(frozen=True)
class ChatActivityReference:
    id: str
    sequence: int
    message_id: str
    response_revision_id: str
    occurred_at: datetime


@dataclass(frozen=True)
class ChatActivityHistory:
    last_output: ChatActivityReference | None = None
    last_failure: ChatActivityReference | None = None


def chat_activity_history(
    session: Session, chat_ids: Sequence[str]
) -> dict[str, ChatActivityHistory]:
    """Return at most two scalar records per standard chat in the requested page."""
    identities = tuple(dict.fromkeys(chat_ids))
    if len(identities) > 200:
        raise ValueError("Too many chats were requested.")
    if not identities:
        return {}
    has_output = exists().where(
        ResponseRevisionPart.response_revision_id == ResponseRevision.id,
        ResponseRevisionPart.metadata_json["preview"].as_boolean().is_not(True),
        ((ResponseRevisionPart.type == "text") & (ResponseRevisionPart.text != ""))
        | (
            ResponseRevisionPart.type.in_(("image", "video", "audio"))
            & ResponseRevisionPart.artifact_id.is_not(None)
        ),
    )
    latest = (
        select(
            ChatActivityEvent.chat_id,
            ChatActivityEvent.kind,
            ChatActivityEvent.id,
            ChatActivityEvent.sequence,
            ChatActivityEvent.message_id,
            ChatActivityEvent.response_revision_id,
            ChatActivityEvent.occurred_at,
            func.row_number()
            .over(
                partition_by=(ChatActivityEvent.chat_id, ChatActivityEvent.kind),
                order_by=ChatActivityEvent.sequence.desc(),
            )
            .label("position"),
        )
        .join(Chat, Chat.id == ChatActivityEvent.chat_id)
        .join(Message, Message.id == ChatActivityEvent.message_id)
        .join(ResponseRevision, ResponseRevision.id == ChatActivityEvent.response_revision_id)
        .where(
            ChatActivityEvent.chat_id.in_(identities),
            ChatActivityEvent.kind.in_(("output", "failure")),
            Chat.scope == STANDARD_CHAT_SCOPE,
            Message.chat_id == Chat.id,
            Message.role == "assistant",
            Message.transcript_visible.is_(True),
            Message.content_removed_at.is_(None),
            ResponseRevision.message_id == Message.id,
            ((ChatActivityEvent.kind == "failure") & (ResponseRevision.status == "failed"))
            | (
                (ChatActivityEvent.kind == "output")
                & ResponseRevision.status.in_(("complete", "cancelled"))
                & has_output
            ),
        )
        .subquery()
    )
    values: dict[str, dict[str, ChatActivityReference]] = {identity: {} for identity in identities}
    with session.no_autoflush:
        for row in session.execute(select(latest).where(latest.c.position == 1)):
            values[row.chat_id][row.kind] = ChatActivityReference(
                id=row.id,
                sequence=row.sequence,
                message_id=row.message_id,
                response_revision_id=row.response_revision_id,
                occurred_at=row.occurred_at,
            )
    return {
        identity: ChatActivityHistory(items.get("output"), items.get("failure"))
        for identity, items in values.items()
    }
