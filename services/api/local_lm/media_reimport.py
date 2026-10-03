"""Recover existing Media Library membership during an explicit file import."""

from datetime import datetime

from sqlalchemy.orm import Session

from .artifact_library import ensure_library_entry
from .domain import new_id
from .media_recovery import preview_media_recovery, restore_media
from .models import Artifact
from .recovery_previews import RecoveryPreviewConflict
from .recovery_v1 import RestoreRecoveryV1


def recover_imported_membership(session: Session, artifact: Artifact, now: datetime) -> str | None:
    """Restore the original membership under the import's writer; the caller commits."""
    entry = ensure_library_entry(session, artifact)
    if entry is None or entry.state == "visible":
        return None
    if entry.state != "trashed" or entry.recovery_id is None:
        raise RecoveryPreviewConflict("media-recovery-membership-invalid")
    deletion_id = entry.recovery_id
    preview = preview_media_recovery(session, deletion_id, now)
    restore_media(
        session,
        deletion_id,
        RestoreRecoveryV1(
            expected_revision=preview.revision,
            impact_sha256=preview.impact_sha256,
            operation_key=new_id("reimport"),
        ),
        now,
    )
    return deletion_id
