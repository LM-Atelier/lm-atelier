"""Release exclusive generated library membership after a chat's canonical rows leave."""

import secrets

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .artifact_library import (
    ArtifactLibraryDataError,
    ArtifactReferenceDataError,
    _mapping,
    _validate_library_row,
    referenced_artifact_ids,
)
from .chat_recovery_graph import ChatRecoveryGraph
from .models import (
    Artifact,
    ArtifactLibraryEntry,
    MediaCollectionMembership,
    MediaTagAssignment,
    RecoveryItem,
    RecoveryPreviewRecord,
)
from .recovery_operations import forget_recovery_labels
from .recovery_previews import RecoveryPreviewConflict


def generated_media_ids(graph: ChatRecoveryGraph) -> frozenset[str]:
    """Select assistant image/video outputs and their saved revisions, excluding inputs."""
    messages = {row["id"] for row in graph.rows["messages"] if row["role"] == "assistant"}
    revisions = {
        row["id"] for row in graph.rows["response_revisions"] if row["message_id"] in messages
    }
    return frozenset(
        str(row["artifact_id"])
        for table, owner, ids in (
            ("message_parts", "message_id", messages),
            ("response_revision_parts", "response_revision_id", revisions),
        )
        for row in graph.rows[table]
        if row[owner] in ids and row["type"] in {"image", "video"} and row["artifact_id"]
    )


def release_generated_membership(
    session: Session, artifact_ids: frozenset[str], chat_item: RecoveryItem
) -> None:
    """Retire only unprotected membership under the same reserved writer as chat purge."""
    if not artifact_ids:
        return
    if (
        chat_item.kind != "chat"
        or chat_item.state != "purging"
        or not chat_item.delete_generated_media
    ):
        raise RecoveryPreviewConflict("chat-recovery-media-intent-invalid")
    entries: list[ArtifactLibraryEntry] = []
    ordered = sorted(artifact_ids)
    for start in range(0, len(ordered), 500):
        for entry, artifact in session.execute(
            select(ArtifactLibraryEntry, Artifact)
            .join(Artifact, Artifact.id == ArtifactLibraryEntry.artifact_id)
            .where(
                ArtifactLibraryEntry.artifact_id.in_(ordered[start : start + 500]),
                ArtifactLibraryEntry.state == "visible",
                ArtifactLibraryEntry.favorite.is_(False),
                Artifact.favorite.is_(False),
                ~select(MediaCollectionMembership.entry_id)
                .where(MediaCollectionMembership.entry_id == ArtifactLibraryEntry.id)
                .exists(),
                ~select(MediaTagAssignment.entry_id)
                .where(MediaTagAssignment.entry_id == ArtifactLibraryEntry.id)
                .exists(),
            )
        ):
            try:
                _validate_library_row(entry, artifact)
                metadata = _mapping(artifact.metadata_json)
            except ArtifactLibraryDataError:
                raise RecoveryPreviewConflict("media-recovery-graph-invalid") from None
            except ArtifactReferenceDataError:
                raise RecoveryPreviewConflict("chat-recovery-graph-invalid") from None
            if "uploaded" in metadata and type(metadata["uploaded"]) is not bool:
                raise RecoveryPreviewConflict("chat-recovery-graph-invalid")
            if metadata.get("uploaded") is True:
                continue
            entries.append(entry)
    try:
        # Validate every stored reference before previewing membership-only pin release.
        referenced_artifact_ids(session, for_deletion=True)
        retained = referenced_artifact_ids(
            session,
            exclude_library_membership_for=frozenset(entry.artifact_id for entry in entries),
        )
    except ArtifactReferenceDataError:
        raise RecoveryPreviewConflict("chat-recovery-graph-invalid") from None
    for entry in entries:
        if entry.artifact_id in retained:
            continue
        previous = session.scalar(
            select(RecoveryItem).where(
                RecoveryItem.kind == "media_library_entry", RecoveryItem.subject_id == entry.id
            )
        )
        if previous is not None:
            if previous.state != "purged":
                raise RecoveryPreviewConflict("media-recovery-membership-invalid")
            session.delete(previous)
            session.flush()
        item = RecoveryItem(
            kind="media_library_entry",
            subject_id=entry.id,
            display_label="Deleted item",
            deleted_at=chat_item.deleted_at,
            purge_after=chat_item.purge_after,
            state="purging",
            subject_revision=secrets.token_hex(32),
            revision=1,
            delete_generated_media=False,
        )
        session.add(item)
        session.flush()
        entry.state, entry.deleted_at, entry.recovery_id = (
            "trashed",
            item.deleted_at,
            item.deletion_id,
        )
        entry.version += 1
        session.flush()
        session.delete(entry)
        session.flush()
        item.state = "purged"
        forget_recovery_labels(session, item)
        item.revision += 1
        session.execute(
            delete(RecoveryPreviewRecord).where(
                RecoveryPreviewRecord.kind == "media_library_entry",
                RecoveryPreviewRecord.subject_id == entry.id,
            )
        )
