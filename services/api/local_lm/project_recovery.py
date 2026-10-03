"""Keep a deleted project's identity without undoing later conversation moves."""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .chat_recovery import RECOVERY_WINDOW
from .chat_recovery_graph import ChatRecoveryGraphError, _GraphReader
from .db import Base
from .models import Project, RecoveryItem, RecoveryOperation, RecoveryPreviewRecord
from .recovery_operations import _remember, _replay, forget_recovery_labels
from .recovery_previews import (
    RecoveryPreviewConflict,
    RecoverySnapshot,
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


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _membership(session: Session, project_id: str) -> RecoveryItem | None:
    return session.scalar(
        select(RecoveryItem).where(
            RecoveryItem.kind == RecoveryKind.PROJECT.value, RecoveryItem.subject_id == project_id
        )
    )


def _item(session: Session, deletion_id: str) -> RecoveryItem:
    item = session.get(RecoveryItem, deletion_id)
    if item is None or item.kind != RecoveryKind.PROJECT.value:
        raise RecoveryPreviewConflict("recovery-item-not-found")
    if item.state != RecoveryState.RECOVERABLE.value:
        raise RecoveryPreviewConflict("recovery-item-not-recoverable")
    return item


def inspect_project_recovery(session: Session, project_id: str, now: datetime) -> RecoverySnapshot:
    """Compare configuration and filing links, without hashing unrelated chat edits."""
    try:
        return _snapshot(session, project_id, now)
    except ChatRecoveryGraphError:
        raise RecoveryPreviewConflict("project-recovery-graph-invalid") from None


def _snapshot(session: Session, project_id: str, now: datetime) -> RecoverySnapshot:
    reader = _GraphReader(session)
    projects = Base.metadata.tables["projects"]
    if not reader.read("projects", projects.c.id == project_id):
        raise RecoveryPreviewConflict("recovery-subject-not-found")
    for name in ("project_workflow_selections", "project_workflow_use_case_selections"):
        table = Base.metadata.tables[name]
        reader.read(name, table.c.project_id == project_id)
    chats = Base.metadata.tables["chats"]
    children = reader.read(
        "chats", chats.c.project_id == project_id, columns=("id", "project_id", "scope")
    )
    recoveries = Base.metadata.tables["recovery_items"]
    reader.read(
        "recovery_items",
        (recoveries.c.kind == RecoveryKind.PROJECT.value) & (recoveries.c.subject_id == project_id),
    )
    item = _membership(session, project_id)
    actions: tuple[RecoveryAction, ...]
    if item is None:
        actions = (RecoveryAction.TRASH,)
    elif item.state != RecoveryState.RECOVERABLE.value:
        actions = ()
    elif _utc(now) < _utc(item.purge_after):
        actions = (RecoveryAction.RESTORE, RecoveryAction.PURGE)
    else:
        actions = (RecoveryAction.PURGE,)
    return RecoverySnapshot(
        fingerprint=hashlib.sha256(
            b"project-recovery-canonical-v1\0" + reader.digest.digest()
        ).hexdigest(),
        counts=RecoveryCountsV1(chats=len(children)),
        available_actions=actions,
    )


def preview_project_trash(session: Session, project_id: str, now: datetime) -> RecoveryImpactV1:
    def inspect(current: Session) -> RecoverySnapshot:
        if _membership(current, project_id) is not None:
            raise RecoveryPreviewConflict("project-already-trashed")
        return inspect_project_recovery(current, project_id, now)

    return issue_recovery_preview(
        session,
        kind=RecoveryKind.PROJECT,
        subject_id=project_id,
        deletion_id=None,
        inspect=inspect,
        now=now,
    )


def preview_project_recovery(session: Session, deletion_id: str, now: datetime) -> RecoveryImpactV1:
    reserve_recovery_write(session)
    item = _item(session, deletion_id)
    return issue_recovery_preview(
        session,
        kind=RecoveryKind.PROJECT,
        subject_id=item.subject_id,
        deletion_id=deletion_id,
        inspect=lambda current: inspect_project_recovery(current, item.subject_id, now),
        now=now,
    )


def trash_project(
    session: Session,
    project_id: str,
    command: RecoveryCommandV1,
    now: datetime,
    *,
    before_trash: Callable[[Session, str], None] | None = None,
) -> RecoveryItemV1:
    """Retain canonical configuration and child links; the caller owns the commit."""
    reserve_recovery_write(session)
    replay = _replay(
        session.get(
            RecoveryOperation, (RecoveryKind.PROJECT.value, project_id, command.operation_key)
        ),
        RecoveryAction.TRASH,
        command,
    )
    if replay is not None:
        if not isinstance(replay, RecoveryItemV1):
            raise RecoveryPreviewConflict("recovery-operation-invalid")
        return replay
    if _membership(session, project_id) is not None:
        raise RecoveryPreviewConflict("project-already-trashed")
    snapshot = require_current_recovery_preview(
        session,
        kind=RecoveryKind.PROJECT,
        subject_id=project_id,
        deletion_id=None,
        command=command,
        inspect=lambda current: inspect_project_recovery(current, project_id, now),
        now=now,
    )
    project = session.get(Project, project_id)
    if project is None:
        raise RecoveryPreviewConflict("recovery-subject-not-found")
    if before_trash is not None:
        before_trash(session, project_id)
    item = RecoveryItem(
        kind=RecoveryKind.PROJECT.value,
        subject_id=project_id,
        display_label=project.name.strip()[:240] or "Untitled project",
        deleted_at=_utc(now),
        purge_after=_utc(now) + RECOVERY_WINDOW,
        state=RecoveryState.RECOVERABLE.value,
        subject_revision=secrets.token_hex(32),
        delete_generated_media=False,
    )
    session.add(item)
    session.flush()
    result = RecoveryItemV1(
        deletion_id=item.deletion_id,
        kind=RecoveryKind.PROJECT,
        subject_id=project_id,
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
    replay = _replay(
        session.scalar(
            select(RecoveryOperation).where(
                RecoveryOperation.kind == RecoveryKind.PROJECT.value,
                RecoveryOperation.deletion_id == deletion_id,
                RecoveryOperation.operation_key == command.operation_key,
            )
        ),
        action,
        command,
    )
    if replay is not None:
        if not isinstance(replay, RecoveryResultV1):
            raise RecoveryPreviewConflict("recovery-operation-invalid")
        return replay
    item = _item(session, deletion_id)
    project_id = item.subject_id
    if (
        session.get(
            RecoveryOperation, (RecoveryKind.PROJECT.value, project_id, command.operation_key)
        )
        is not None
    ):
        raise RecoveryPreviewConflict("recovery-operation-conflict")
    require_current_recovery_preview(
        session,
        kind=RecoveryKind.PROJECT,
        subject_id=project_id,
        deletion_id=deletion_id,
        command=command,
        inspect=lambda current: inspect_project_recovery(current, project_id, now),
        now=now,
    )
    if action == RecoveryAction.RESTORE:
        if _utc(now) >= _utc(item.purge_after):
            raise RecoveryPreviewConflict("recovery-window-expired")
        session.delete(item)
        session.flush()
    else:
        item.state = RecoveryState.PURGING.value
        session.flush()
        # SQLite clears only still-linked conversations and removes project-local selections.
        session.execute(delete(Project).where(Project.id == project_id))
        item.state = RecoveryState.PURGED.value
        forget_recovery_labels(session, item)
        item.subject_revision = secrets.token_hex(32)
        item.revision += 1
    session.execute(
        delete(RecoveryPreviewRecord).where(
            RecoveryPreviewRecord.kind == RecoveryKind.PROJECT.value,
            RecoveryPreviewRecord.subject_id == project_id,
        )
    )
    result = RecoveryResultV1(
        deletion_id=deletion_id,
        kind=RecoveryKind.PROJECT,
        subject_id=project_id,
        action=action,
        reclaimed_bytes=0,
    )
    _remember(session, action, command, result)
    return result


def restore_project(
    session: Session, deletion_id: str, command: RestoreRecoveryV1, now: datetime
) -> RecoveryResultV1:
    """Make the original project visible without reversing any later filing move."""
    return _transition(session, deletion_id, command, now)


def purge_project(
    session: Session, deletion_id: str, command: PurgeRecoveryV1, now: datetime
) -> RecoveryResultV1:
    """Remove only project configuration; child history and media remain canonical."""
    return _transition(session, deletion_id, command, now)
