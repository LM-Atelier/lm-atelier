"""Studio sessions: hidden chats behind the canvas-first editing surface.

A studio session is a chat with `scope = "studio"` - invisible in the chat
sidebar, never a transcript on screen - whose turns are the applies the
studio executes. One session exists per source image; opening the same
image again resumes its history, which is exactly what makes the filmstrip
a real, durable edit history rather than view state.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Artifact, Chat

STUDIO_SCOPE = "studio"


def find_studio_session(session: Session, source_artifact_id: str) -> Chat | None:
    """The existing session for one source image, if any.

    Sessions are keyed by the source artifact recorded in `origin_json` at
    creation. The studio population is small and local, so matching in
    Python over the scoped rows keeps the key out of SQL-JSON territory.
    """
    for chat in session.scalars(select(Chat).where(Chat.scope == STUDIO_SCOPE)):
        origin = chat.origin_json
        if isinstance(origin, dict) and origin.get("source_artifact_id") == source_artifact_id:
            return chat
    return None


def studio_session_title(artifact: Artifact) -> str:
    name = artifact.original_name or f"{artifact.kind} {artifact.sha256[:12]}"
    return f"Studio - {name}"[:240]
