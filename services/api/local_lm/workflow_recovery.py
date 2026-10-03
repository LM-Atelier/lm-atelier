"""Retain deleted workflow identity without granting execution or trust on restore."""

from __future__ import annotations

import secrets
from datetime import datetime

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from .chat_recovery import RECOVERY_WINDOW
from .models import (
    RecoveryItem,
    RecoveryOperation,
    RecoveryPreviewRecord,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
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
    RecoveryImpactV1,
    RecoveryItemV1,
    RecoveryKind,
    RecoveryLocationV1,
    RecoveryResultV1,
    RecoveryState,
    RestoreRecoveryV1,
)
from .workflow_recovery_graph import inspect_workflow_recovery
from .workflow_recovery_history import retain_workflow_history

KIND = RecoveryKind.WORKFLOW_FAMILY


def _membership(session: Session, family_id: str) -> RecoveryItem | None:
    return session.scalar(
        select(RecoveryItem).where(
            RecoveryItem.kind == KIND.value,
            RecoveryItem.subject_id == family_id,
        )
    )


def _item(session: Session, deletion_id: str) -> RecoveryItem:
    item = session.get(RecoveryItem, deletion_id)
    if item is None or item.kind != KIND.value:
        raise RecoveryPreviewConflict("recovery-item-not-found")
    if item.state != RecoveryState.RECOVERABLE.value:
        raise RecoveryPreviewConflict("recovery-item-not-recoverable")
    return item


def preview_workflow_trash(session: Session, family_id: str, now: datetime) -> RecoveryImpactV1:
    def inspect(current: Session) -> RecoverySnapshot:
        if _membership(current, family_id) is not None:
            raise RecoveryPreviewConflict("workflow-already-trashed")
        return inspect_workflow_recovery(current, family_id, now)

    return issue_recovery_preview(
        session, kind=KIND, subject_id=family_id, deletion_id=None, inspect=inspect, now=now
    )


def preview_workflow_recovery(
    session: Session, deletion_id: str, now: datetime
) -> RecoveryImpactV1:
    reserve_recovery_write(session)
    item = _item(session, deletion_id)
    return issue_recovery_preview(
        session,
        kind=KIND,
        subject_id=item.subject_id,
        deletion_id=deletion_id,
        inspect=lambda current: inspect_workflow_recovery(current, item.subject_id, now),
        now=now,
    )


def trash_workflow(
    session: Session, family_id: str, command: RecoveryCommandV1, now: datetime
) -> RecoveryItemV1:
    reserve_recovery_write(session)
    replay = _replay(
        session.get(RecoveryOperation, (KIND.value, family_id, command.operation_key)),
        RecoveryAction.TRASH,
        command,
    )
    if replay is not None:
        if not isinstance(replay, RecoveryItemV1):
            raise RecoveryPreviewConflict("recovery-operation-invalid")
        return replay
    if _membership(session, family_id) is not None:
        raise RecoveryPreviewConflict("workflow-already-trashed")
    snapshot = require_current_recovery_preview(
        session,
        kind=KIND,
        subject_id=family_id,
        deletion_id=None,
        command=command,
        inspect=lambda current: inspect_workflow_recovery(current, family_id, now),
        now=now,
    )
    if RecoveryAction.TRASH not in snapshot.available_actions:
        raise RecoveryPreviewConflict("workflow-recovery-in-use")
    family = session.get(WorkflowFamily, family_id)
    if family is None:
        raise RecoveryPreviewConflict("recovery-subject-not-found")
    family.enabled = False
    session.execute(
        update(WorkflowPreference)
        .where(WorkflowPreference.workflow_family_id == family_id)
        .values(enabled=False, is_default=False)
    )
    revisions = (
        select(WorkflowRevision.id)
        .join(WorkflowDefinition, WorkflowDefinition.id == WorkflowRevision.workflow_id)
        .where(WorkflowDefinition.family_id == family_id)
    )
    session.execute(
        update(WorkflowActivation)
        .where(WorkflowActivation.workflow_revision_id.in_(revisions))
        .values(
            is_active=False,
            state="disabled",
            invalidated_at=_utc(now),
            invalidation_code="workflow_deleted",
            invalidation_reason="The workflow was moved to Recently Deleted.",
        )
    )
    item = RecoveryItem(
        kind=KIND.value,
        subject_id=family_id,
        display_label=family.name.strip()[:240] or "Untitled workflow",
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
        kind=KIND,
        subject_id=family_id,
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
                RecoveryOperation.kind == KIND.value,
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
    family_id = item.subject_id
    if session.get(RecoveryOperation, (KIND.value, family_id, command.operation_key)) is not None:
        raise RecoveryPreviewConflict("recovery-operation-conflict")
    snapshot = require_current_recovery_preview(
        session,
        kind=KIND,
        subject_id=family_id,
        deletion_id=deletion_id,
        command=command,
        inspect=lambda current: inspect_workflow_recovery(current, family_id, now),
        now=now,
    )
    if action == RecoveryAction.RESTORE and _utc(now) >= _utc(item.purge_after):
        raise RecoveryPreviewConflict("recovery-window-expired")
    if action not in snapshot.available_actions:
        raise RecoveryPreviewConflict("workflow-recovery-in-use")
    if action == RecoveryAction.RESTORE:
        session.delete(item)
        session.flush()
        session.execute(
            update(WorkflowFamily).where(WorkflowFamily.id == family_id).values(enabled=False)
        )
        session.execute(
            update(WorkflowPreference)
            .where(WorkflowPreference.workflow_family_id == family_id)
            .values(enabled=False, is_default=False)
        )
    else:
        item.state = RecoveryState.PURGING.value
        session.flush()
        if snapshot.counts.references:
            retain_workflow_history(session, family_id, now)
        else:
            session.execute(
                delete(WorkflowDefinition).where(WorkflowDefinition.family_id == family_id)
            )
            session.execute(delete(WorkflowFamily).where(WorkflowFamily.id == family_id))
        item.state = RecoveryState.PURGED.value
        forget_recovery_labels(session, item)
        item.subject_revision = secrets.token_hex(32)
        item.revision += 1
    session.execute(
        delete(RecoveryPreviewRecord).where(
            RecoveryPreviewRecord.kind == KIND.value, RecoveryPreviewRecord.subject_id == family_id
        )
    )
    result = RecoveryResultV1(
        deletion_id=deletion_id, kind=KIND, subject_id=family_id, action=action, reclaimed_bytes=0
    )
    _remember(session, action, command, result)
    return result


def restore_workflow(
    session: Session, deletion_id: str, command: RestoreRecoveryV1, now: datetime
) -> RecoveryResultV1:
    return _transition(session, deletion_id, command, now)


def purge_workflow(
    session: Session, deletion_id: str, command: PurgeRecoveryV1, now: datetime
) -> RecoveryResultV1:
    return _transition(session, deletion_id, command, now)
