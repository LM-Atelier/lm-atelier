"""A picture from earlier is context, not a permanent attachment."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from local_lm.db import Base
from local_lm.domain import PartType
from local_lm.models import Artifact, Chat, Message, Run
from local_lm.orchestrator import ConversationOrchestrator


def _session() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _conversation(session: Session, *, turns_after_image: int) -> Run:
    chat = Chat(id="chat-1", title="Long thread")
    session.add(chat)
    session.flush()

    artifact = Artifact(
        id="artifact-1",
        sha256="a" * 64,
        media_type="image/png",
        size_bytes=32,
        kind="image",
        original_name="earlier.png",
        relative_path="images/earlier.png",
    )
    session.add(artifact)
    session.flush()

    parent: str | None = None
    with_image = Message(id="m-image", chat_id=chat.id, role="assistant", parent_id=parent)
    session.add(with_image)
    session.flush()
    session.add(
        Message.__mapper__.relationships["parts"].mapper.class_(
            id="part-image",
            message_id=with_image.id,
            position=0,
            type=PartType.IMAGE.value,
            artifact_id=artifact.id,
        )
    )
    parent = with_image.id

    for index in range(turns_after_image):
        message = Message(id=f"m-{index}", chat_id=chat.id, role="user", parent_id=parent)
        session.add(message)
        session.flush()
        parent = message.id

    current = Message(id="m-current", chat_id=chat.id, role="user", parent_id=parent)
    answer = Message(id="m-answer", chat_id=chat.id, role="assistant", parent_id=current.id)
    session.add_all([current, answer])
    session.flush()
    run = Run(
        id="run-1",
        chat_id=chat.id,
        user_message_id=current.id,
        assistant_message_id=answer.id,
        operation="text",
        status="running",
    )
    session.add(run)
    session.commit()
    return run


def test_a_recent_picture_is_still_what_this_turn_is_about() -> None:
    session = _session()
    run = _conversation(session, turns_after_image=1)

    found = ConversationOrchestrator._visual_context_artifacts(session, run, lookback=4)

    # "Make it brighter" right after a picture means that picture.
    assert [artifact.original_name for artifact in found] == ["earlier.png"]


def test_a_picture_from_far_back_is_not_dragged_into_every_message() -> None:
    session = _session()
    run = _conversation(session, turns_after_image=20)

    found = ConversationOrchestrator._visual_context_artifacts(session, run, lookback=4)

    # The climb used to be unbounded, so any chat that had ever contained a
    # picture paid to decode, resize, and re-send it on every message after -
    # and the model was asked to look at something nobody was discussing.
    assert found == []


def test_a_lookback_of_zero_keeps_only_what_this_message_carries() -> None:
    session = _session()
    run = _conversation(session, turns_after_image=1)

    assert ConversationOrchestrator._visual_context_artifacts(session, run, lookback=0) == []
