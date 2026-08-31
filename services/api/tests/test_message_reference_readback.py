"""A message hands back what it referred to, as it stood at the time.

The transcript needs this to render mentions from the record rather than by
scanning the message text. Scanning is the failure this contract refuses: a
message can contain `@ada-lovelace` that nobody chose - typed by hand, pasted,
or left after the reference was dropped from the draft - and highlighting it
would claim a binding that does not exist.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.message_references import record_message_references, resolve_reference_requests
from local_lm.models import Chat, Message, MessageReference
from local_lm.reference_library import create_subject
from local_lm.references import ReferenceRequest


@pytest.fixture()
def session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _foreign_keys(connection, _record) -> None:  # type: ignore[no-untyped-def]
        # SQLite ignores foreign keys unless asked, and the cascade from a
        # message to its references is the thing under test here.
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as opened:
        yield opened


def _turn(session: Session, *, subject_ids: list[str]) -> Message:
    chat = Chat(title="Chat", routing_mode="auto")
    session.add(chat)
    session.flush()
    message = Message(chat_id=chat.id, role="user", status="complete")
    session.add(message)
    session.flush()
    resolved = resolve_reference_requests(
        session, [ReferenceRequest(reference_subject_id=one) for one in subject_ids]
    )
    record_message_references(session, message.id, resolved)
    session.flush()
    return message


def test_a_message_hands_back_what_it_referred_to(session: Session) -> None:
    ada = create_subject(session, name="Ada Lovelace", kind="person")
    message = _turn(session, subject_ids=[ada.id])

    session.expire(message)

    assert [one.subject_name for one in message.references] == ["Ada Lovelace"]
    assert [one.mention_slug for one in message.references] == ["ada-lovelace"]


def test_the_record_keeps_the_name_the_turn_used(session: Session) -> None:
    """A rename must not rewrite what an old message meant."""

    ada = create_subject(session, name="Ada Lovelace", kind="person")
    message = _turn(session, subject_ids=[ada.id])

    ada.name = "Augusta Ada King"
    session.flush()
    session.expire(message)

    assert [one.subject_name for one in message.references] == ["Ada Lovelace"]


def test_the_record_survives_deleting_the_subject(session: Session) -> None:
    """The question this answers - why does that picture look like that - is
    asked most often after the subject is gone."""

    ada = create_subject(session, name="Ada Lovelace", kind="person")
    message = _turn(session, subject_ids=[ada.id])
    subject_id = ada.id

    session.delete(ada)
    session.flush()
    session.expire(message)

    assert [one.reference_subject_id for one in message.references] == [subject_id]
    assert [one.subject_name for one in message.references] == ["Ada Lovelace"]


def test_references_come_back_in_the_order_the_turn_named_them(session: Session) -> None:
    first = create_subject(session, name="Ada Lovelace", kind="person")
    second = create_subject(session, name="Grace Hopper", kind="person")
    message = _turn(session, subject_ids=[second.id, first.id])

    session.expire(message)

    assert [one.subject_name for one in message.references] == ["Grace Hopper", "Ada Lovelace"]


def test_a_message_that_named_nothing_hands_back_nothing(session: Session) -> None:
    message = _turn(session, subject_ids=[])

    session.expire(message)

    assert list(message.references) == []


def test_deleting_the_message_takes_its_record_with_it(session: Session) -> None:
    """A reference record outliving its own message would be an orphan nobody
    could interpret."""

    ada = create_subject(session, name="Ada Lovelace", kind="person")
    message = _turn(session, subject_ids=[ada.id])
    session.commit()

    session.delete(message)
    session.commit()

    assert session.query(MessageReference).count() == 0
