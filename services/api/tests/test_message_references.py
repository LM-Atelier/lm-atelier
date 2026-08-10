from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.domain import MessageRole, MessageStatus
from local_lm.message_references import (
    ResolvedReference,
    carry_message_references_if_absent,
    message_references,
    record_message_references,
    resolve_reference_requests,
)
from local_lm.models import Artifact, Chat, Message, MessageReference, ReferenceAsset
from local_lm.reference_library import create_subject, rename_subject
from local_lm.references import (
    MentionSource,
    ReferenceError,
    ReferenceNotFoundError,
    ReferenceRequest,
)


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _enforce_foreign_keys(connection, _record):  # type: ignore[no-untyped-def]
        # SQLite ignores foreign keys unless asked, so a cascade this file
        # asserts on would silently do nothing without it.
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


_artifacts = 0


def _artifact(session: Session) -> Artifact:
    global _artifacts
    _artifacts += 1
    artifact = Artifact(
        id=f"mr_art_{_artifacts}",
        sha256=f"{_artifacts:064x}",
        kind="image",
        media_type="image/png",
        size_bytes=1,
        relative_path=f"mr/{_artifacts}",
    )
    session.add(artifact)
    session.flush()
    return artifact


def _message(session: Session) -> Message:
    chat = Chat(title="Referring")
    session.add(chat)
    session.flush()
    message = Message(
        chat_id=chat.id, role=MessageRole.USER.value, status=MessageStatus.COMPLETE.value
    )
    session.add(message)
    session.flush()
    return message


def test_a_request_resolves_to_the_subject_as_it_stands(session: Session) -> None:
    subject = create_subject(session, name="Ada Lovelace", kind="person")

    resolved = resolve_reference_requests(session, [ReferenceRequest(subject.id)])

    assert len(resolved) == 1
    assert resolved[0].subject_name == "Ada Lovelace"
    assert resolved[0].mention_slug == "ada-lovelace"
    assert resolved[0].subject_kind == "person"
    assert resolved[0].source is MentionSource.MENTION


def test_a_subject_that_no_longer_exists_refuses_the_turn(session: Session) -> None:
    """Dropping it would generate an image the request did not ask for and then
    report success, which is worse than refusing."""

    with pytest.raises(ReferenceNotFoundError, match="no longer exists"):
        resolve_reference_requests(session, [ReferenceRequest("refsubject_missing")])


def test_an_image_belonging_to_another_subject_is_refused(session: Session) -> None:
    """Accepting it would let a turn pull an image out of a subject it never
    named."""

    named = create_subject(session, name="Ada Lovelace", kind="person")
    other = create_subject(session, name="Grace Hopper", kind="person")
    artifact = _artifact(session)
    asset = ReferenceAsset(reference_subject_id=other.id, artifact_id=artifact.id)
    session.add(asset)
    session.flush()

    with pytest.raises(ReferenceError, match="does not hold image"):
        resolve_reference_requests(
            session, [ReferenceRequest(named.id, selected_asset_ids=(asset.id,))]
        )


def test_selected_images_are_recorded_in_the_order_they_were_chosen(session: Session) -> None:
    subject = create_subject(session, name="Ada Lovelace", kind="person")
    first, second = _artifact(session), _artifact(session)
    one = ReferenceAsset(reference_subject_id=subject.id, artifact_id=first.id)
    two = ReferenceAsset(reference_subject_id=subject.id, artifact_id=second.id)
    session.add_all([one, two])
    session.flush()

    resolved = resolve_reference_requests(
        session, [ReferenceRequest(subject.id, selected_asset_ids=(two.id, one.id))]
    )

    assert resolved[0].reference_asset_ids == (two.id, one.id)
    assert resolved[0].artifact_ids == (second.id, first.id)


def test_what_a_turn_referred_to_is_read_back_from_its_own_snapshot(session: Session) -> None:
    subject = create_subject(session, name="Ada Lovelace", kind="person")
    message = _message(session)

    record_message_references(
        session, message.id, resolve_reference_requests(session, [ReferenceRequest(subject.id)])
    )
    session.flush()

    read = message_references(session, message.id)
    assert [one.subject_name for one in read] == ["Ada Lovelace"]


def test_a_later_rename_does_not_rewrite_what_a_past_turn_recorded(session: Session) -> None:
    """The record answers what a turn used at the time. If a rename moved it,
    the answer would change every time somebody tidied up their library."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")
    message = _message(session)
    record_message_references(
        session, message.id, resolve_reference_requests(session, [ReferenceRequest(subject.id)])
    )
    session.flush()

    rename_subject(session, subject, name="Augusta Ada King")
    session.flush()

    read = message_references(session, message.id)
    assert read[0].subject_name == "Ada Lovelace"
    assert read[0].mention_slug == "ada-lovelace"


def test_deleting_the_subject_does_not_destroy_the_history(session: Session) -> None:
    """A live foreign key would erase this at exactly the moment someone asked
    why an old picture looks the way it does."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")
    subject_id = subject.id
    message = _message(session)
    record_message_references(
        session, message.id, resolve_reference_requests(session, [ReferenceRequest(subject.id)])
    )
    session.flush()

    session.delete(subject)
    session.flush()

    read = message_references(session, message.id)
    assert len(read) == 1
    assert read[0].reference_subject_id == subject_id
    assert read[0].subject_name == "Ada Lovelace"


def test_deleting_the_turn_removes_what_it_recorded(session: Session) -> None:
    """The opposite direction: deleting a user's turn is meant to remove what it
    produced, and an orphan record nobody can interpret is not history."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")
    message = _message(session)
    record_message_references(
        session, message.id, resolve_reference_requests(session, [ReferenceRequest(subject.id)])
    )
    session.flush()

    session.delete(message)
    session.flush()

    assert session.scalars(select(MessageReference)).all() == []


def test_recording_a_message_twice_is_refused(session: Session) -> None:
    subject = create_subject(session, name="Ada Lovelace", kind="person")
    message = _message(session)
    resolved = resolve_reference_requests(session, [ReferenceRequest(subject.id)])
    record_message_references(session, message.id, resolved)
    session.flush()

    with pytest.raises(ReferenceError, match="already records"):
        record_message_references(session, message.id, resolved)


def test_recording_nothing_writes_nothing(session: Session) -> None:
    message = _message(session)
    assert record_message_references(session, message.id, []) == ()
    session.flush()
    assert session.scalars(select(MessageReference)).all() == []


def test_a_regeneration_carries_the_original_references_verbatim(session: Session) -> None:
    """Copied rather than re-resolved: a subject renamed or deleted in between
    must not change, or refuse, a repeat of something that already ran."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")
    original = _message(session)
    repeat = _message(session)
    record_message_references(
        session, original.id, resolve_reference_requests(session, [ReferenceRequest(subject.id)])
    )
    session.flush()

    rename_subject(session, subject, name="Augusta Ada King")
    session.delete(subject)
    session.flush()

    carry_message_references_if_absent(
        session, source_message_id=original.id, target_message_id=repeat.id
    )
    session.flush()

    assert message_references(session, repeat.id) == message_references(session, original.id)
    assert message_references(session, repeat.id)[0].subject_name == "Ada Lovelace"


def test_an_identical_retry_carry_is_a_no_op(session: Session) -> None:
    subject = create_subject(session, name="Ada Lovelace", kind="person")
    original = _message(session)
    repeat = _message(session)
    resolved = resolve_reference_requests(session, [ReferenceRequest(subject.id)])
    record_message_references(session, original.id, resolved)
    record_message_references(session, repeat.id, resolved)
    session.flush()

    carry_message_references_if_absent(
        session, source_message_id=original.id, target_message_id=repeat.id
    )

    assert message_references(session, repeat.id) == message_references(session, original.id)


def test_a_retry_carry_refuses_conflicting_existing_provenance(session: Session) -> None:
    original_subject = create_subject(session, name="Ada Lovelace", kind="person")
    other_subject = create_subject(session, name="Grace Hopper", kind="person")
    original = _message(session)
    repeat = _message(session)
    record_message_references(
        session,
        original.id,
        resolve_reference_requests(session, [ReferenceRequest(original_subject.id)]),
    )
    record_message_references(
        session,
        repeat.id,
        resolve_reference_requests(session, [ReferenceRequest(other_subject.id)]),
    )
    session.flush()

    with pytest.raises(ReferenceError, match="already records different references"):
        carry_message_references_if_absent(
            session, source_message_id=original.id, target_message_id=repeat.id
        )


def test_the_source_of_a_reference_survives_the_round_trip(session: Session) -> None:
    """A typed mention and something inherited from context differ in how much
    the user actually asserted, so the difference has to survive storage."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")
    message = _message(session)
    record_message_references(
        session,
        message.id,
        (
            ResolvedReference(
                reference_subject_id=subject.id,
                mention_slug=subject.mention_slug,
                subject_name=subject.name,
                subject_kind=subject.kind,
                role="subject",
                strength=0.65,
                source=MentionSource.INHERITED_CONTEXT,
            ),
        ),
    )
    session.flush()

    read = message_references(session, message.id)
    assert read[0].source is MentionSource.INHERITED_CONTEXT
    assert read[0].role == "subject"
    assert read[0].strength == pytest.approx(0.65)
