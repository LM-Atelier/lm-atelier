"""Recover library membership while shared media remains in the content store."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from .artifact_library import (
    ArtifactLibraryDataError,
    ArtifactReferenceDataError,
    _mapping,
    _optional_id,
    _validate_library_row,
    referenced_artifact_ids,
)
from .artifact_library_schema import ARTIFACT_METADATA_REFERENCE_KEYS
from .chat_recovery_graph import ChatRecoveryGraphError, _GraphReader
from .models import (
    Artifact,
    ArtifactLibraryEntry,
    MediaCollectionMembership,
    MediaTagAssignment,
    RecoveryItem,
    RecoveryOperation,
    RecoveryPreviewRecord,
)
from .recovery_operations import _remember, _replay, forget_recovery_labels
from .recovery_previews import (
    RecoveryPreviewConflict,
    RecoverySnapshot,
    _utc,
    issue_recovery_preview,
    require_current_recovery_preview,
    reserve_recovery_write,
)
from .recovery_v1 import (
    PurgeRecoveryV1,
    RecoveryAction,
    RecoveryCommandV1,
    RecoveryCountsV1,
    RecoveryImpactV1,
    RecoveryItemV1,
    RecoveryKind,
    RecoveryLocationV1,
    RecoveryResultV1,
    RecoveryState,
    RestoreRecoveryV1,
)

RECOVERY_WINDOW = timedelta(days=30)
_KIND = RecoveryKind.MEDIA_LIBRARY_ENTRY


def _membership(session: Session, entry_id: str) -> RecoveryItem | None:
    return session.scalar(
        select(RecoveryItem).where(
            RecoveryItem.kind == _KIND.value, RecoveryItem.subject_id == entry_id
        )
    )


def _entry(session: Session, entry_id: str) -> ArtifactLibraryEntry:
    entry = session.get(ArtifactLibraryEntry, entry_id)
    if entry is None:
        raise RecoveryPreviewConflict("recovery-subject-not-found")
    artifact = session.get(Artifact, entry.artifact_id)
    if artifact is None:
        raise RecoveryPreviewConflict("media-recovery-graph-invalid")
    try:
        _validate_library_row(entry, artifact)
    except ArtifactLibraryDataError:
        raise RecoveryPreviewConflict("media-recovery-graph-invalid") from None
    return entry


def _snapshot(session: Session, entry_id: str, now: datetime) -> RecoverySnapshot:
    entry = _entry(session, entry_id)
    item = _membership(session, entry_id)
    if (entry.state == "trashed") != (item is not None and item.state != "purged") or (
        entry.state == "trashed" and (item is None or item.deletion_id != entry.recovery_id)
    ):
        raise RecoveryPreviewConflict("media-recovery-membership-invalid")
    reader = _GraphReader(session)
    try:
        reader.read("artifact_library_entries", ArtifactLibraryEntry.id == entry_id)
        reader.read(
            "recovery_items",
            (RecoveryItem.kind == _KIND.value) & (RecoveryItem.subject_id == entry_id),
        )
        reader.read("media_collection_memberships", MediaCollectionMembership.entry_id == entry_id)
        reader.read("media_tag_assignments", MediaTagAssignment.entry_id == entry_id)
        pending = {entry.artifact_id}
        visited: set[str] = set()
        retained_bytes = 0
        while frontier := sorted(pending - visited):
            for start in range(0, len(frontier), 500):
                batch = frontier[start : start + 500]
                artifacts = reader.read("artifacts", Artifact.id.in_(batch))
                if len(artifacts) != len(batch):
                    raise ChatRecoveryGraphError()
                for artifact in artifacts:
                    size = artifact["size_bytes"]
                    if type(size) is not int or size < 0:
                        raise ChatRecoveryGraphError()
                    retained_bytes += size
                    metadata = _mapping(artifact["metadata_json"])
                    for key in ARTIFACT_METADATA_REFERENCE_KEYS:
                        if key in metadata:
                            pending.update(_optional_id(metadata[key]))
                visited.update(batch)
        # Membership purge leaves bytes to retention; validate its complete
        # reference graph before any pins are changed.
        referenced_artifact_ids(session, for_deletion=True)
    except (ArtifactReferenceDataError, ChatRecoveryGraphError):
        raise RecoveryPreviewConflict("media-recovery-graph-invalid") from None
    actions: tuple[RecoveryAction, ...] = ()
    if entry.state == "visible":
        actions = (RecoveryAction.TRASH,)
    elif item and item.state == "recoverable":
        actions = (
            (RecoveryAction.RESTORE, RecoveryAction.PURGE)
            if _utc(now) < _utc(item.purge_after)
            else (RecoveryAction.PURGE,)
        )
    return RecoverySnapshot(
        fingerprint=reader.digest.hexdigest(),
        counts=RecoveryCountsV1(artifacts=len(visited), retained_bytes=retained_bytes),
        available_actions=actions,
    )


def preview_media_trash(session: Session, entry_id: str, now: datetime) -> RecoveryImpactV1:
    return issue_recovery_preview(
        session,
        kind=_KIND,
        subject_id=entry_id,
        deletion_id=None,
        inspect=lambda current: _snapshot(current, entry_id, now),
        now=now,
    )


def _item(session: Session, deletion_id: str) -> RecoveryItem:
    item = session.get(RecoveryItem, deletion_id)
    if item is None or item.kind != _KIND.value:
        raise RecoveryPreviewConflict("recovery-item-not-found")
    if item.state != RecoveryState.RECOVERABLE.value:
        raise RecoveryPreviewConflict("recovery-item-not-recoverable")
    return item


def preview_media_recovery(session: Session, deletion_id: str, now: datetime) -> RecoveryImpactV1:
    reserve_recovery_write(session)
    item = _item(session, deletion_id)
    return issue_recovery_preview(
        session,
        kind=_KIND,
        subject_id=item.subject_id,
        deletion_id=deletion_id,
        inspect=lambda current: _snapshot(current, item.subject_id, now),
        now=now,
    )


def trash_media(
    session: Session, entry_id: str, command: RecoveryCommandV1, now: datetime
) -> RecoveryItemV1:
    """Hide membership and keep favorite and organization; the caller commits."""
    reserve_recovery_write(session)
    replay = _replay(
        session.get(RecoveryOperation, (_KIND.value, entry_id, command.operation_key)),
        RecoveryAction.TRASH,
        command,
    )
    if replay is not None:
        if not isinstance(replay, RecoveryItemV1):
            raise RecoveryPreviewConflict("recovery-operation-invalid")
        return replay
    snapshot = require_current_recovery_preview(
        session,
        kind=_KIND,
        subject_id=entry_id,
        deletion_id=None,
        command=command,
        inspect=lambda current: _snapshot(current, entry_id, now),
        now=now,
    )
    if RecoveryAction.TRASH not in snapshot.available_actions:
        raise RecoveryPreviewConflict("media-already-trashed")
    previous = _membership(session, entry_id)
    if previous is not None:
        if previous.state != RecoveryState.PURGED.value:
            raise RecoveryPreviewConflict("media-already-trashed")
        session.delete(previous)
        session.flush()
    entry = _entry(session, entry_id)
    item = RecoveryItem(
        kind=_KIND.value,
        subject_id=entry_id,
        display_label=entry.display_name[:240],
        deleted_at=_utc(now),
        purge_after=_utc(now) + RECOVERY_WINDOW,
        state=RecoveryState.RECOVERABLE.value,
        subject_revision=secrets.token_hex(32),
        revision=1,
        delete_generated_media=False,
    )
    session.add(item)
    session.flush()
    entry.state = "trashed"
    entry.deleted_at = item.deleted_at
    entry.recovery_id = item.deletion_id
    entry.version += 1
    session.flush()
    result = RecoveryItemV1(
        deletion_id=item.deletion_id,
        kind=_KIND,
        subject_id=entry_id,
        display_label=item.display_label,
        original_location=RecoveryLocationV1(),
        deleted_at=item.deleted_at,
        purge_after=item.purge_after,
        state=RecoveryState.RECOVERABLE,
        revision=item.subject_revision,
        counts=snapshot.counts,
    )
    _remember(session, RecoveryAction.TRASH, command, result)
    return result


def _transition(
    session: Session, deletion_id: str, command: RestoreRecoveryV1 | PurgeRecoveryV1, now: datetime
) -> RecoveryResultV1:
    reserve_recovery_write(session)
    action = (
        RecoveryAction.RESTORE if isinstance(command, RestoreRecoveryV1) else RecoveryAction.PURGE
    )
    operation = session.scalar(
        select(RecoveryOperation).where(
            RecoveryOperation.kind == _KIND.value,
            RecoveryOperation.deletion_id == deletion_id,
            RecoveryOperation.operation_key == command.operation_key,
        )
    )
    replay = _replay(operation, action, command)
    if replay is not None:
        if not isinstance(replay, RecoveryResultV1):
            raise RecoveryPreviewConflict("recovery-operation-invalid")
        return replay
    item = _item(session, deletion_id)
    if (
        session.get(RecoveryOperation, (_KIND.value, item.subject_id, command.operation_key))
        is not None
    ):
        raise RecoveryPreviewConflict("recovery-operation-conflict")
    require_current_recovery_preview(
        session,
        kind=_KIND,
        subject_id=item.subject_id,
        deletion_id=deletion_id,
        command=command,
        inspect=lambda current: _snapshot(current, item.subject_id, now),
        now=now,
    )
    entry = _entry(session, item.subject_id)
    if action == RecoveryAction.RESTORE:
        if _utc(now) >= _utc(item.purge_after):
            raise RecoveryPreviewConflict("recovery-window-expired")
        item.state = RecoveryState.RESTORING.value
        session.flush()
        entry.state, entry.deleted_at, entry.recovery_id = "visible", None, None
        entry.version += 1
        session.flush()
        session.delete(item)
        session.flush()
    else:
        item.state = RecoveryState.PURGING.value
        session.flush()
        session.execute(
            delete(MediaCollectionMembership).where(MediaCollectionMembership.entry_id == entry.id)
        )
        session.execute(delete(MediaTagAssignment).where(MediaTagAssignment.entry_id == entry.id))
        session.execute(
            update(Artifact).where(Artifact.id == entry.artifact_id).values(favorite=False)
        )
        session.delete(entry)
        session.flush()
        item.state = RecoveryState.PURGED.value
        forget_recovery_labels(session, item)
        item.subject_revision = secrets.token_hex(32)
        item.revision += 1
    session.execute(
        delete(RecoveryPreviewRecord).where(
            RecoveryPreviewRecord.kind == _KIND.value,
            RecoveryPreviewRecord.subject_id == item.subject_id,
        )
    )
    result = RecoveryResultV1(
        deletion_id=deletion_id,
        kind=_KIND,
        subject_id=item.subject_id,
        action=action,
        reclaimed_bytes=0,
    )
    _remember(session, action, command, result)
    return result


def restore_media(
    session: Session, deletion_id: str, command: RestoreRecoveryV1, now: datetime
) -> RecoveryResultV1:
    """Restore the same membership without changing any byte consumer."""
    return _transition(session, deletion_id, command, now)


def purge_media(
    session: Session, deletion_id: str, command: PurgeRecoveryV1, now: datetime
) -> RecoveryResultV1:
    """Remove membership-only pins and defer byte removal to ordinary retention."""
    return _transition(session, deletion_id, command, now)
