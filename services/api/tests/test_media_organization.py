from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm.artifact_library import ensure_library_entry
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base, create_database_engine
from local_lm.domain import ArtifactKind
from local_lm.media_organization import (
    MEDIA_ORGANIZATION_CONFLICT,
    MEDIA_ORGANIZATION_INVALID,
    MediaOrganizationConflict,
    MediaOrganizationError,
    add_manual_membership,
    assign_media_tag,
    create_manual_collection,
    create_media_tag,
    media_tag_slug,
)
from local_lm.models import (
    ArtifactLibraryEntry,
    MediaCollectionMembership,
    MediaTag,
    MediaTagAssignment,
)


@pytest.fixture
def organization_session(tmp_path: Path) -> Iterator[tuple[ArtifactStore, Session]]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    try:
        yield ArtifactStore(settings), session
    finally:
        session.close()
        engine.dispose()


def _entry(store: ArtifactStore, session: Session, marker: bytes) -> ArtifactLibraryEntry:
    artifact = store.ingest_bytes(
        session,
        marker,
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name=f"{marker.hex()}.png",
    )
    entry = ensure_library_entry(session, artifact)
    assert entry is not None
    session.flush()
    return entry


def test_manual_memberships_are_ordered_unique_strong_entry_references(
    organization_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = organization_session
    first = _entry(store, session, b"first")
    second = _entry(store, session, b"second")
    collection = create_manual_collection(
        session, name="Portrait studies", description="Exact selected media"
    )

    one = add_manual_membership(
        session,
        collection_id=collection.id,
        entry_id=first.id,
        expected_version=1,
        note="keeper",
    )
    two = add_manual_membership(
        session,
        collection_id=collection.id,
        entry_id=second.id,
        expected_version=2,
    )
    session.commit()

    assert (one.position, two.position, collection.version) == (0, 1, 3)
    assert session.scalars(
        select(MediaCollectionMembership)
        .where(MediaCollectionMembership.collection_id == collection.id)
        .order_by(MediaCollectionMembership.position)
    ).all() == [one, two]
    with pytest.raises(IntegrityError):
        session.execute(delete(ArtifactLibraryEntry).where(ArtifactLibraryEntry.id == first.id))
        session.flush()
    session.rollback()


def test_collection_delete_removes_only_memberships_and_never_entries(
    organization_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = organization_session
    entry = _entry(store, session, b"collection-cascade")
    collection = create_manual_collection(session, name="Album")
    add_manual_membership(
        session,
        collection_id=collection.id,
        entry_id=entry.id,
        expected_version=1,
    )
    session.commit()

    session.delete(collection)
    session.commit()
    assert session.scalars(select(MediaCollectionMembership)).all() == []
    assert session.get(ArtifactLibraryEntry, entry.id) is not None


def test_raw_membership_delete_advances_parent_version(
    organization_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = organization_session
    entry = _entry(store, session, b"membership-delete")
    collection = create_manual_collection(session, name="Album")
    add_manual_membership(
        session,
        collection_id=collection.id,
        entry_id=entry.id,
        expected_version=1,
    )
    session.commit()

    session.execute(
        delete(MediaCollectionMembership).where(
            MediaCollectionMembership.collection_id == collection.id
        )
    )
    session.commit()
    session.refresh(collection)
    assert collection.version == 3


def test_tags_normalize_once_and_assign_explicitly(
    organization_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = organization_session
    entry = _entry(store, session, b"tagged")
    tag = create_media_tag(session, label="Portrait Study", color="#a1b2c3")
    assignment = assign_media_tag(
        session,
        tag_id=tag.id,
        entry_id=entry.id,
        expected_version=1,
    )
    session.commit()

    assert tag.slug == "portrait-study"
    assert tag.version == 2
    assert session.get(MediaTagAssignment, (tag.id, entry.id)) is assignment
    session.delete(tag)
    session.commit()
    assert session.scalars(select(MediaTagAssignment)).all() == []
    assert session.get(ArtifactLibraryEntry, entry.id) is not None


@pytest.mark.parametrize(
    "value",
    [
        "",
        " leading",
        "trailing ",
        "bad_control\n",
        "naïve",
        "a" * 81,
        True,
        1,
        ["tag"],
    ],
)
def test_tag_normalization_is_total_fixed_and_ascii_slug_bounded(value: object) -> None:
    with pytest.raises(MediaOrganizationError) as caught:
        media_tag_slug(value)
    assert str(caught.value) == MEDIA_ORGANIZATION_INVALID


def test_duplicate_tag_and_stale_versions_fail_closed(
    organization_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = organization_session
    entry = _entry(store, session, b"conflicts")
    create_media_tag(session, label="Portrait")
    session.commit()
    with pytest.raises(MediaOrganizationConflict) as duplicate:
        create_media_tag(session, label="portrait")
    assert str(duplicate.value) == MEDIA_ORGANIZATION_CONFLICT
    session.rollback()

    collection = create_manual_collection(session, name="Album")
    session.commit()
    with pytest.raises(MediaOrganizationConflict) as stale:
        add_manual_membership(
            session,
            collection_id=collection.id,
            entry_id=entry.id,
            expected_version=2,
        )
    assert str(stale.value) == MEDIA_ORGANIZATION_CONFLICT
    assert session.scalars(select(MediaCollectionMembership)).all() == []


def test_raw_sql_cannot_forge_identities_versions_or_dangling_references(
    organization_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = organization_session
    entry = _entry(store, session, b"raw-guards")
    note_entry = _entry(store, session, b"raw-note-guards")
    trashed = _entry(store, session, b"raw-trashed")
    collection = create_manual_collection(session, name="Album")
    tag = create_media_tag(session, label="Safe Tag")
    add_manual_membership(
        session,
        collection_id=collection.id,
        entry_id=entry.id,
        expected_version=1,
    )
    assign_media_tag(session, tag_id=tag.id, entry_id=entry.id, expected_version=1)
    session.commit()
    session.execute(
        text(
            "UPDATE artifact_library_entries "
            "SET state='trashed',deleted_at=CURRENT_TIMESTAMP,recovery_id=:recovery,"
            "version=version+1,updated_at=CURRENT_TIMESTAMP WHERE id=:entry"
        ),
        {"entry": trashed.id, "recovery": "recovery_" + "e" * 32},
    )
    session.commit()

    statements: tuple[tuple[str, dict[str, object]], ...] = (
        (
            "INSERT INTO media_collections "
            "(id,kind,name,description,version,created_at,updated_at) "
            "VALUES ('collection_bad','manual','Bad','',1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
            {},
        ),
        (
            "INSERT INTO media_collections "
            "(id,kind,name,description,version,created_at,updated_at) "
            "VALUES (:id,'manual','Version Nine','',9,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
            {"id": "collection_" + "9" * 32},
        ),
        (
            "INSERT INTO media_collections "
            "(id,kind,name,description,version,created_at,updated_at) "
            "VALUES (:id,'manual',:name,'',1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
            {"id": "collection_" + "8" * 32, "name": "bad‮"},
        ),
        (
            "UPDATE media_collections SET version=version+2 WHERE id=:id",
            {"id": collection.id},
        ),
        (
            "INSERT INTO media_collection_memberships "
            "(collection_id,entry_id,position,note,added_at) "
            "VALUES (:collection,'libentry:sha256:' || :digest,0,NULL,CURRENT_TIMESTAMP)",
            {"collection": collection.id, "digest": "0" * 64},
        ),
        (
            "INSERT INTO media_collection_memberships "
            "(collection_id,entry_id,position,note,added_at) "
            "VALUES (:collection,:entry,4,NULL,CURRENT_TIMESTAMP)",
            {"collection": collection.id, "entry": trashed.id},
        ),
        (
            "INSERT INTO media_collection_memberships "
            "(collection_id,entry_id,position,note,added_at) "
            "VALUES (:collection,:entry,5,:note,CURRENT_TIMESTAMP)",
            {"collection": collection.id, "entry": note_entry.id, "note": " leading"},
        ),
        (
            "INSERT INTO media_collection_memberships "
            "(collection_id,entry_id,position,note,added_at) "
            "VALUES (:collection,:entry,6,:note,CURRENT_TIMESTAMP)",
            {"collection": collection.id, "entry": note_entry.id, "note": "trailing "},
        ),
        (
            "UPDATE media_tags SET slug='changed',version=version+1 WHERE id=:id",
            {"id": tag.id},
        ),
        (
            "UPDATE media_collection_memberships SET note='changed' "
            "WHERE collection_id=:collection",
            {"collection": collection.id},
        ),
        (
            "UPDATE media_tag_assignments SET added_at=added_at WHERE tag_id=:tag",
            {"tag": tag.id},
        ),
        (
            "INSERT INTO media_tags "
            "(id,slug,label,color,version,created_at,updated_at) "
            "VALUES (:id,'control','bad' || char(10),NULL,1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
            {"id": "mediatag_" + "f" * 32},
        ),
        (
            "INSERT INTO media_tags "
            "(id,slug,label,color,version,created_at,updated_at) "
            "VALUES (:id,'version-nine','Version Nine',NULL,9,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
            {"id": "mediatag_" + "9" * 32},
        ),
        (
            "INSERT INTO media_tag_assignments (tag_id,entry_id,added_at) "
            "VALUES (:tag,:entry,CURRENT_TIMESTAMP)",
            {"tag": tag.id, "entry": trashed.id},
        ),
    )
    for statement, parameters in statements:
        with pytest.raises(IntegrityError):
            session.execute(text(statement), parameters)
            session.flush()
        session.rollback()


def test_schema_rejects_duplicate_order_and_invalid_tag_slug(
    organization_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = organization_session
    first = _entry(store, session, b"one")
    second = _entry(store, session, b"two")
    collection = create_manual_collection(session, name="Album")
    add_manual_membership(
        session,
        collection_id=collection.id,
        entry_id=first.id,
        expected_version=1,
    )
    session.commit()

    session.add(
        MediaCollectionMembership(
            collection_id=collection.id,
            entry_id=second.id,
            position=0,
            note=None,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(
        MediaTag(
            id="mediatag_" + "a" * 32,
            slug="Bad Tag",
            label="Bad Tag",
            color=None,
            version=1,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
