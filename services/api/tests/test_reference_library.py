from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.models import Artifact, ReferenceAsset, ReferenceSubject
from local_lm.reference_library import (
    MAX_PAGE,
    create_subject,
    deletion_impact,
    list_subjects,
    rename_subject,
    set_archived,
    set_favorite,
)
from local_lm.references import ReferenceError, ReferenceKind


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _enforce_foreign_keys(connection, _record):  # type: ignore[no-untyped-def]
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


_artifacts = 0


def _artifact(session: Session) -> Artifact:
    global _artifacts
    _artifacts += 1
    artifact = Artifact(
        id=f"art_{_artifacts}",
        sha256=f"{_artifacts:064x}",
        kind="image",
        media_type="image/png",
        size_bytes=1,
        relative_path=f"{_artifacts}/a",
    )
    session.add(artifact)
    session.flush()
    return artifact


def _attach(session: Session, subject: ReferenceSubject, artifact: Artifact) -> None:
    session.add(ReferenceAsset(reference_subject_id=subject.id, artifact_id=artifact.id))
    session.flush()


def test_a_subject_gets_a_mention_derived_from_its_name(session: Session) -> None:
    subject = create_subject(session, name="Ada Lovelace", kind=ReferenceKind.PERSON)
    assert subject.mention_slug == "ada-lovelace"
    assert subject.kind == "person"


def test_two_people_may_share_a_name_but_not_a_mention(session: Session) -> None:
    """The collision is usually two real people called the same thing, not a
    mistake. Refusing would push the user into inventing a worse name."""

    first = create_subject(session, name="Ada Lovelace", kind="person")
    second = create_subject(session, name="Ada Lovelace", kind="person")
    assert first.name == second.name
    assert (first.mention_slug, second.mention_slug) == ("ada-lovelace", "ada-lovelace-2")


def test_a_mention_the_user_chose_is_refused_rather_than_renumbered(session: Session) -> None:
    """Silently handing back a different mention than the one asked for would
    mean the user's chats reference something that does not exist."""

    create_subject(session, name="Ada", kind="person", mention_slug="ada")
    with pytest.raises(ReferenceError) as caught:
        create_subject(session, name="Someone else", kind="person", mention_slug="ada")
    assert "already answers to @ada" in str(caught.value)


@pytest.mark.parametrize("bad", ["Ada Lovelace", "ada_lovelace", "-ada", "", "a" * 80])
def test_an_unusable_mention_is_refused(session: Session, bad: str) -> None:
    with pytest.raises(ReferenceError):
        create_subject(session, name="Ada", kind="person", mention_slug=bad)


def test_renaming_does_not_move_the_mention_by_default(session: Session) -> None:
    """A live chat draft may already hold the mention. Changing the addressing
    token underneath someone mid-sentence is worse than letting the two drift."""

    subject = create_subject(session, name="Ada Lovelace", kind="person")
    rename_subject(session, subject, name="Augusta Ada King")
    assert subject.name == "Augusta Ada King"
    assert subject.mention_slug == "ada-lovelace"

    rename_subject(session, subject, name="Augusta Ada King", follow_mention=True)
    assert subject.mention_slug == "augusta-ada-king"


def test_a_following_mention_still_avoids_a_collision(session: Session) -> None:
    create_subject(session, name="Grace Hopper", kind="person")
    subject = create_subject(session, name="Ada", kind="person")
    rename_subject(session, subject, name="Grace Hopper", follow_mention=True)
    assert subject.mention_slug == "grace-hopper-2"


def test_archiving_hides_a_subject_without_destroying_it(session: Session) -> None:
    """Archiving is the removal a user wants; it has to be reversible."""

    subject = create_subject(session, name="Ada", kind="person")
    set_archived(session, subject, True)
    visible, total = list_subjects(session)
    assert visible == [] and total == 0

    hidden, hidden_total = list_subjects(session, include_archived=True)
    assert [item.id for item in hidden] == [subject.id] and hidden_total == 1

    set_archived(session, subject, False)
    restored, _ = list_subjects(session)
    assert [item.id for item in restored] == [subject.id]


def test_favourites_sort_first_without_being_a_quality_signal(session: Session) -> None:
    """Organisation only - it says someone wanted this near the top, not that
    its images are good."""

    plain = create_subject(session, name="Bravo", kind="person")
    marked = create_subject(session, name="Alpha", kind="person")
    set_favorite(session, marked, True)
    listed, _ = list_subjects(session)
    assert listed[0].id == marked.id
    assert plain.favorite is False


def test_listing_filters_by_kind_and_name(session: Session) -> None:
    create_subject(session, name="Ada Lovelace", kind="person")
    create_subject(session, name="Brass Lamp", kind="object")

    people, total = list_subjects(session, kind="person")
    assert total == 1 and people[0].name == "Ada Lovelace"

    found, _ = list_subjects(session, search="LAMP")
    assert [item.name for item in found] == ["Brass Lamp"]

    assert list_subjects(session, search="nothing here")[1] == 0


def test_a_page_is_bounded(session: Session) -> None:
    for index in range(5):
        create_subject(session, name=f"Subject {index}", kind="person")
    page, total = list_subjects(session, limit=2)
    assert len(page) == 2 and total == 5
    assert len(list_subjects(session, limit=2, offset=4)[0]) == 1

    for bad in (0, -1, MAX_PAGE + 1):
        with pytest.raises(ReferenceError):
            list_subjects(session, limit=bad)
    with pytest.raises(ReferenceError):
        list_subjects(session, offset=-1)


def test_deletion_impact_counts_only_what_is_this_subject_s_to_lose(
    session: Session,
) -> None:
    """A photograph showing two subjects belongs to both. Removing one of them
    is not permission to delete the picture."""

    subject = create_subject(session, name="Ada", kind="person")
    other = create_subject(session, name="Grace", kind="person")

    private_image = _artifact(session)
    shared_image = _artifact(session)
    _attach(session, subject, private_image)
    _attach(session, subject, shared_image)
    _attach(session, other, shared_image)

    impact = deletion_impact(session, subject)
    assert impact.asset_count == 2
    assert impact.exclusive_artifact_ids == (private_image.id,)
    assert impact.shared_artifact_count == 1
    assert impact.name == "Ada"


def test_deletion_impact_is_computed_before_anything_is_destroyed(
    session: Session,
) -> None:
    subject = create_subject(session, name="Ada", kind="person")
    _attach(session, subject, _artifact(session))
    session.commit()

    deletion_impact(session, subject)

    assert session.query(ReferenceSubject).count() == 1
    assert session.query(ReferenceAsset).count() == 1
    assert session.query(Artifact).count() == 1
