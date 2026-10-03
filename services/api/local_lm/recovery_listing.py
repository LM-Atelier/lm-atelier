"""Seek through current recovery memberships with filters bound to each cursor."""

from __future__ import annotations

import base64
import json
from datetime import datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from .chat_recovery import _snapshot as inspect_chat
from .media_recovery import _snapshot as inspect_media
from .models import ArtifactLibraryEntry, Chat, Project, RecoveryItem, WorkflowFamily
from .project_recovery import inspect_project_recovery
from .prompt_helpers import STANDARD_CHAT_SCOPE
from .recovery_previews import RecoveryPreviewConflict, _utc, reserve_recovery_write
from .recovery_v1 import (
    RecoveryItemV1,
    RecoveryKind,
    RecoveryLocationV1,
    RecoveryPageV1,
    RecoveryState,
)
from .workflow_recovery_graph import inspect_workflow_recovery


def _page_cursor(
    item: RecoveryItem,
    state: RecoveryState | None,
    since: datetime | None,
    kind: RecoveryKind | None,
) -> str:
    payload = [
        "recovery-page-v1",
        _utc(item.deleted_at).isoformat(),
        item.deletion_id,
        state.value if state else None,
        _utc(since).isoformat() if since else None,
        kind.value if kind else None,
    ]
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def _read_page_cursor(
    cursor: str, state: RecoveryState | None, since: datetime | None, kind: RecoveryKind | None
) -> tuple[datetime, str]:
    try:
        if len(cursor) > 2048:
            raise ValueError
        payload = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
        if not isinstance(payload, list) or len(payload) != 6:
            raise ValueError
        version, timestamp, identity, saved_state, saved_since, saved_kind = payload
        if (
            version != "recovery-page-v1"
            or not isinstance(timestamp, str)
            or not isinstance(identity, str)
            or not 1 <= len(identity) <= 128
            or saved_kind != (kind.value if kind else None)
            or saved_state != (state.value if state else None)
            or saved_since != (_utc(since).isoformat() if since else None)
        ):
            raise ValueError
        date = datetime.fromisoformat(timestamp)
        if date.tzinfo is None:
            raise ValueError
        return _utc(date), identity
    except (ValueError, TypeError, UnicodeError):
        raise RecoveryPreviewConflict("recovery-cursor-invalid") from None


def recovery_page(
    session: Session,
    now: datetime,
    *,
    limit: int = 10,
    cursor: str | None = None,
    state: RecoveryState | None = None,
    kind: RecoveryKind | None = None,
    deleted_since: datetime | None = None,
) -> RecoveryPageV1:
    """Read a bounded, consistent page without creating action previews."""

    if not 1 <= limit <= 20:
        raise RecoveryPreviewConflict("recovery-page-size-invalid")
    after = _read_page_cursor(cursor, state, deleted_since, kind) if cursor else None
    reserve_recovery_write(session)
    statement = select(RecoveryItem).where(
        RecoveryItem.state != RecoveryState.PURGED.value,
        or_(
            and_(
                RecoveryItem.kind == RecoveryKind.WORKFLOW_FAMILY.value,
                select(WorkflowFamily.id)
                .where(WorkflowFamily.id == RecoveryItem.subject_id)
                .exists(),
            ),
            and_(
                RecoveryItem.kind == RecoveryKind.PROJECT.value,
                select(Project.id).where(Project.id == RecoveryItem.subject_id).exists(),
            ),
            and_(
                RecoveryItem.kind == RecoveryKind.CHAT.value,
                select(Chat.id)
                .where(
                    Chat.id == RecoveryItem.subject_id,
                    Chat.scope == STANDARD_CHAT_SCOPE,
                )
                .exists(),
            ),
            and_(
                RecoveryItem.kind == RecoveryKind.MEDIA_LIBRARY_ENTRY.value,
                select(ArtifactLibraryEntry.id)
                .where(
                    ArtifactLibraryEntry.id == RecoveryItem.subject_id,
                    ArtifactLibraryEntry.state == "trashed",
                    ArtifactLibraryEntry.recovery_id == RecoveryItem.deletion_id,
                )
                .exists(),
            ),
        ),
    )
    if kind is not None:
        statement = statement.where(RecoveryItem.kind == kind.value)
    if state is not None:
        statement = statement.where(RecoveryItem.state == state.value)
    if deleted_since is not None:
        statement = statement.where(RecoveryItem.deleted_at >= _utc(deleted_since))
    if after is not None:
        date, identity = after
        statement = statement.where(
            or_(
                RecoveryItem.deleted_at < date,
                and_(RecoveryItem.deleted_at == date, RecoveryItem.deletion_id < identity),
            )
        )
    rows = list(
        session.scalars(
            statement.order_by(
                RecoveryItem.deleted_at.desc(), RecoveryItem.deletion_id.desc()
            ).limit(limit + 1)
        )
    )
    items: list[RecoveryItemV1] = []
    for item in rows[:limit]:
        inspect = {
            RecoveryKind.CHAT.value: inspect_chat,
            RecoveryKind.PROJECT.value: inspect_project_recovery,
            RecoveryKind.MEDIA_LIBRARY_ENTRY.value: inspect_media,
            RecoveryKind.WORKFLOW_FAMILY.value: inspect_workflow_recovery,
        }[item.kind]
        snapshot = inspect(session, item.subject_id, now)
        items.append(
            RecoveryItemV1(
                deletion_id=item.deletion_id,
                kind=RecoveryKind(item.kind),
                subject_id=item.subject_id,
                display_label=item.display_label,
                original_location=RecoveryLocationV1(
                    project_id=item.original_project_id, project_label=item.original_project_label
                ),
                deleted_at=item.deleted_at,
                purge_after=item.purge_after,
                state=RecoveryState(item.state),
                revision=item.subject_revision,
                counts=snapshot.counts,
                restore_conflicts=snapshot.conflicts,
                delete_generated_media=item.delete_generated_media,
            )
        )
    return RecoveryPageV1(
        items=tuple(items),
        next_cursor=(
            _page_cursor(rows[limit - 1], state, deleted_since, kind) if len(rows) > limit else None
        ),
    )
