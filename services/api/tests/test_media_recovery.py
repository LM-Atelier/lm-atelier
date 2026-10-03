"""Library recovery preserves identity and releases only membership-owned pins."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from test_artifact_library_entries import library_session as library_session

from local_lm.artifact_library import ensure_library_entry, set_library_favorite
from local_lm.artifacts import ArtifactStore
from local_lm.domain import ArtifactKind
from local_lm.media_organization import (
    add_manual_membership,
    assign_media_tag,
    create_manual_collection,
    create_media_tag,
)
from local_lm.media_recovery import (
    RECOVERY_WINDOW,
    preview_media_recovery,
    preview_media_trash,
    purge_media,
    restore_media,
    trash_media,
)
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    MediaCollection,
    MediaCollectionMembership,
    MediaTag,
    MediaTagAssignment,
    RecoveryItem,
    RecoveryOperation,
    ReferenceAsset,
    ReferenceSubject,
)
from local_lm.recovery_previews import RecoveryPreviewConflict
from local_lm.recovery_v1 import (
    PurgeRecoveryV1,
    RecoveryAction,
    RecoveryCommandV1,
    RecoveryImpactV1,
    RestoreRecoveryV1,
)

NOW = datetime(2026, 10, 2, tzinfo=UTC)
CONTENT = b"constructed garden recovery media"


def _command(preview: RecoveryImpactV1, key: str) -> dict[str, str]:
    return {
        "expected_revision": preview.revision,
        "impact_sha256": preview.impact_sha256,
        "operation_key": key,
    }


def _seed(store: ArtifactStore, session: Session):
    artifact = store.ingest_bytes(
        session,
        CONTENT,
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="Garden layout",
    )
    entry = ensure_library_entry(session, artifact)
    assert entry is not None
    set_library_favorite(session, artifact, True)
    collection = create_manual_collection(session, name="Garden archive")
    tag = create_media_tag(session, label="Garden")
    session.commit()
    add_manual_membership(
        session,
        collection_id=collection.id,
        entry_id=entry.id,
        expected_version=collection.version,
        note="Keep original spacing",
    )
    session.commit()
    assign_media_tag(session, tag_id=tag.id, entry_id=entry.id, expected_version=tag.version)
    session.commit()
    return artifact, entry, collection, tag


def _trash(session: Session, entry_id: str, key: str = "trash-garden"):
    preview = preview_media_trash(session, entry_id, NOW)
    session.commit()
    command = RecoveryCommandV1.model_validate(_command(preview, key))
    result = trash_media(session, entry_id, command, NOW)
    session.commit()
    return result, command


def test_trash_and_restore_keep_favorite_organization_identity_and_bytes(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact, entry, collection, tag = _seed(store, session)
    ids = artifact.id, entry.id, collection.id, tag.id
    item, command = _trash(session, entry.id)
    assert entry.state == "trashed" and entry.favorite and artifact.favorite
    assert entry.recovery_id == item.deletion_id
    assert (
        session.get(MediaCollectionMembership, (collection.id, entry.id)).note
        == "Keep original spacing"
    )
    assert session.get(MediaTagAssignment, (tag.id, entry.id)) is not None
    assert store.resolve(artifact).read_bytes() == CONTENT
    assert trash_media(session, entry.id, command, NOW) == item
    session.commit()
    preview = preview_media_recovery(session, item.deletion_id, NOW)
    session.commit()
    restore_command = RestoreRecoveryV1.model_validate(_command(preview, "restore-garden"))
    restored = restore_media(session, item.deletion_id, restore_command, NOW)
    session.commit()
    assert restore_media(session, item.deletion_id, restore_command, NOW) == restored
    session.commit()
    assert (artifact.id, entry.id, collection.id, tag.id) == ids
    assert entry.state == "visible" and entry.favorite and artifact.favorite
    assert entry.recovery_id is entry.deleted_at is None
    assert (
        session.get(MediaCollectionMembership, (collection.id, entry.id)).note
        == "Keep original spacing"
    )
    assert session.get(MediaTagAssignment, (tag.id, entry.id)) is not None
    assert store.resolve(artifact).read_bytes() == CONTENT


@pytest.mark.parametrize("reference", [False, True])
def test_purge_releases_membership_favorite_and_organization_but_defers_byte_removal(
    library_session: tuple[ArtifactStore, Session], reference: bool
) -> None:
    store, session = library_session
    artifact, entry, collection, tag = _seed(store, session)
    entry_id, artifact_id = entry.id, artifact.id
    if reference:
        subject = ReferenceSubject(
            name="Garden structure", mention_slug="garden-structure", kind="object"
        )
        session.add(subject)
        session.flush()
        session.add(
            ReferenceAsset(
                reference_subject_id=subject.id, artifact_id=artifact.id, purpose="other"
            )
        )
        session.commit()
    item, _ = _trash(session, entry_id)
    preview = preview_media_recovery(session, item.deletion_id, NOW)
    session.commit()
    command = PurgeRecoveryV1.model_validate(
        {**_command(preview, "purge-garden"), "acknowledgement": "permanently-delete"}
    )
    result = purge_media(session, item.deletion_id, command, NOW)
    session.commit()
    assert result.reclaimed_bytes == 0
    assert session.get(ArtifactLibraryEntry, entry_id) is None
    assert session.get(MediaCollectionMembership, (collection.id, entry_id)) is None
    assert session.get(MediaTagAssignment, (tag.id, entry_id)) is None
    assert session.get(MediaCollection, collection.id) is not None
    assert session.get(MediaTag, tag.id) is not None
    session.refresh(artifact)
    assert artifact.favorite is False
    assert store.resolve(artifact).read_bytes() == CONTENT
    assert purge_media(session, item.deletion_id, command, NOW) == result
    session.commit()
    later = NOW + timedelta(days=120)
    marked = store.cleanup_retention(
        session, retention_days=1, temporary_hours=1, dry_run=False, now=later
    )
    session.commit()
    assert marked.marked_count == (0 if reference else 1)
    assert marked.removed_count == 0
    assert store.resolve(artifact).read_bytes() == CONTENT
    swept = store.cleanup_retention(
        session,
        retention_days=1,
        temporary_hours=1,
        dry_run=False,
        now=later + timedelta(days=1, microseconds=1),
    )
    session.commit()
    assert swept.removed_count == (0 if reference else 1)
    assert swept.reclaimed_bytes == (0 if reference else len(CONTENT))
    assert (session.get(Artifact, artifact_id) is not None) == reference


def test_stale_name_and_favorite_preview_refuses_without_hiding_membership(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact, entry, _collection, _tag = _seed(store, session)
    preview = preview_media_trash(session, entry.id, NOW)
    session.commit()
    set_library_favorite(session, artifact, False)
    session.commit()
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-impact-stale$"):
        trash_media(
            session,
            entry.id,
            RecoveryCommandV1.model_validate(_command(preview, "stale-garden")),
            NOW,
        )
    session.rollback()
    assert entry.state == "visible"
    assert session.scalar(select(func.count()).select_from(RecoveryItem)) == 0


@pytest.mark.parametrize("target", ["entry", "favorite", "collection", "tag", "delete"])
def test_bulk_writers_cannot_change_a_recoverable_library_membership(
    library_session: tuple[ArtifactStore, Session], target: str
) -> None:
    store, session = library_session
    artifact, entry, _collection, _tag = _seed(store, session)
    _trash(session, entry.id)
    statements = {
        "entry": update(ArtifactLibraryEntry)
        .where(ArtifactLibraryEntry.id == entry.id)
        .values(display_name="Changed garden", version=entry.version + 1),
        "favorite": update(Artifact).where(Artifact.id == artifact.id).values(favorite=False),
        "collection": delete(MediaCollectionMembership).where(
            MediaCollectionMembership.entry_id == entry.id
        ),
        "tag": delete(MediaTagAssignment).where(MediaTagAssignment.entry_id == entry.id),
        "delete": delete(ArtifactLibraryEntry).where(ArtifactLibraryEntry.id == entry.id),
    }
    with pytest.raises(IntegrityError):
        session.execute(statements[target])
        session.flush()
    session.rollback()
    assert entry.state == "trashed" and entry.favorite
    assert store.resolve(artifact).read_bytes() == CONTENT


def test_restore_refuses_at_exact_expiry_and_keeps_its_membership(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    _artifact, entry, _collection, _tag = _seed(store, session)
    item, _ = _trash(session, entry.id)
    deadline = NOW + RECOVERY_WINDOW
    preview = preview_media_recovery(session, item.deletion_id, deadline)
    session.commit()
    assert preview.available_actions == (RecoveryAction.PURGE,)
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-window-expired$"):
        restore_media(
            session,
            item.deletion_id,
            RestoreRecoveryV1.model_validate(_command(preview, "expired-garden")),
            deadline,
        )
    session.rollback()
    assert entry.state == "trashed"


def test_rollback_keeps_membership_and_leaves_no_applied_operation(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact, entry, _collection, _tag = _seed(store, session)
    preview = preview_media_trash(session, entry.id, NOW)
    session.commit()
    trash_media(
        session,
        entry.id,
        RecoveryCommandV1.model_validate(_command(preview, "rollback-garden")),
        NOW,
    )
    session.rollback()
    assert entry.state == "visible" and entry.favorite
    assert session.scalar(select(func.count()).select_from(RecoveryItem)) == 0
    assert session.scalar(select(func.count()).select_from(RecoveryOperation)) == 0
    assert store.resolve(artifact).read_bytes() == CONTENT


def test_reimport_can_start_a_new_recovery_cycle_without_replaying_the_old_purge(
    library_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = library_session
    artifact, entry, _collection, _tag = _seed(store, session)
    entry_id = entry.id
    first, _ = _trash(session, entry_id)
    preview = preview_media_recovery(session, first.deletion_id, NOW)
    session.commit()
    command = PurgeRecoveryV1.model_validate(
        {**_command(preview, "old-purge"), "acknowledgement": "permanently-delete"}
    )
    purged = purge_media(session, first.deletion_id, command, NOW)
    session.commit()
    reimported = store.ingest_bytes(
        session, CONTENT, kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    fresh = ensure_library_entry(session, reimported)
    assert fresh is not None and fresh.id == entry_id and reimported.id == artifact.id
    session.commit()
    second, _ = _trash(session, entry_id, "second-trash")
    assert second.deletion_id != first.deletion_id
    assert purge_media(session, first.deletion_id, command, NOW) == purged
    session.commit()
    assert session.get(ArtifactLibraryEntry, entry_id).recovery_id == second.deletion_id
    assert store.resolve(artifact).read_bytes() == CONTENT
