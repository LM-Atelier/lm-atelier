"""Keep a conversation's canonical rows through recoverable deletion and restore."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .chat_media_purge import generated_media_ids, release_generated_membership
from .chat_recovery_graph import inspect_chat_recovery_graph
from .models import Chat, Job, Project, RecoveryItem, RecoveryOperation, RecoveryPreviewRecord
from .profile_service import reset_unavailable_chat_profiles
from .prompt_helpers import STANDARD_CHAT_SCOPE
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
    RecoveryConflict,
    RecoveryImpactV1,
    RecoveryItemV1,
    RecoveryKind,
    RecoveryLocationV1,
    RecoveryPageV1,
    RecoveryResultV1,
    RecoveryState,
    RestoreRecoveryV1,
    TrashChatV1,
)

RECOVERY_WINDOW = timedelta(days=30)


def chat_recovery_page(
    session: Session,
    now: datetime,
    *,
    limit: int = 10,
    cursor: str | None = None,
    state: RecoveryState | None = None,
    deleted_since: datetime | None = None,
) -> RecoveryPageV1:
    """Read only chat memberships through the shared recovery page."""
    from .recovery_listing import recovery_page

    return recovery_page(
        session,
        now,
        limit=limit,
        cursor=cursor,
        state=state,
        deleted_since=deleted_since,
        kind=RecoveryKind.CHAT,
    )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _membership(session: Session, chat_id: str) -> RecoveryItem | None:
    return session.scalar(
        select(RecoveryItem).where(
            RecoveryItem.kind == RecoveryKind.CHAT.value, RecoveryItem.subject_id == chat_id
        )
    )


def _require_standard_chat(session: Session, chat_id: str) -> None:
    scope = session.scalar(select(Chat.scope).where(Chat.id == chat_id))
    # A purged chat is absent, but its exact saved result remains replayable.
    if scope is not None and scope != STANDARD_CHAT_SCOPE:
        raise RecoveryPreviewConflict("recovery-item-not-found")


def _item(session: Session, deletion_id: str) -> RecoveryItem:
    item = session.get(RecoveryItem, deletion_id)
    if item is None or item.kind != RecoveryKind.CHAT.value:
        raise RecoveryPreviewConflict("recovery-item-not-found")
    _require_standard_chat(session, item.subject_id)
    if item.state != RecoveryState.RECOVERABLE.value:
        raise RecoveryPreviewConflict("recovery-item-not-recoverable")
    return item


def _snapshot(session: Session, chat_id: str, now: datetime) -> RecoverySnapshot:
    _require_standard_chat(session, chat_id)
    graph = inspect_chat_recovery_graph(session, chat_id)
    item = _membership(session, chat_id)
    chat = session.get(Chat, chat_id)
    if chat is None:
        raise RecoveryPreviewConflict("recovery-subject-not-found")
    project_id = item.original_project_id if item else chat.project_id
    project = session.get(Project, project_id) if project_id else None
    project_state = (project_id, project.name if project else None)
    fingerprint = hashlib.sha256(
        (graph.snapshot.fingerprint + json.dumps(project_state)).encode("utf-8")
    ).hexdigest()
    conflicts = graph.snapshot.conflicts
    if item and project_id and project is None:
        conflicts += (RecoveryConflict.ORIGINAL_PROJECT_MISSING,)
    actions: tuple[RecoveryAction, ...] = ()
    if not graph.snapshot.counts.active_work:
        if item is None:
            actions = (RecoveryAction.TRASH,)
        elif item.state == RecoveryState.RECOVERABLE.value:
            actions = (
                (RecoveryAction.RESTORE, RecoveryAction.PURGE)
                if _utc(now) < _utc(item.purge_after)
                else (RecoveryAction.PURGE,)
            )
    return replace(
        graph.snapshot, fingerprint=fingerprint, conflicts=conflicts, available_actions=actions
    )


def preview_chat_trash(
    session: Session, chat_id: str, now: datetime, *, delete_generated_media: bool = False
) -> RecoveryImpactV1:
    def inspect(current: Session) -> RecoverySnapshot:
        _require_standard_chat(current, chat_id)
        if _membership(current, chat_id) is not None:
            raise RecoveryPreviewConflict("chat-already-trashed")
        return _snapshot(current, chat_id, now)

    return issue_recovery_preview(
        session,
        kind=RecoveryKind.CHAT,
        subject_id=chat_id,
        deletion_id=None,
        inspect=inspect,
        now=now,
        delete_generated_media=delete_generated_media,
    )


def preview_chat_recovery(session: Session, deletion_id: str, now: datetime) -> RecoveryImpactV1:
    reserve_recovery_write(session)
    item = _item(session, deletion_id)
    return issue_recovery_preview(
        session,
        kind=RecoveryKind.CHAT,
        subject_id=item.subject_id,
        deletion_id=deletion_id,
        inspect=lambda current: _snapshot(current, item.subject_id, now),
        now=now,
        delete_generated_media=item.delete_generated_media,
    )


def trash_chat(
    session: Session, chat_id: str, command: TrashChatV1, now: datetime
) -> RecoveryItemV1:
    """Preserve history; its caller holds the chat lock and commits the whole transition."""

    reserve_recovery_write(session)
    _require_standard_chat(session, chat_id)
    operation = session.get(
        RecoveryOperation, (RecoveryKind.CHAT.value, chat_id, command.operation_key)
    )
    replay = _replay(operation, RecoveryAction.TRASH, command)
    if replay is not None:
        if not isinstance(replay, RecoveryItemV1):
            raise RecoveryPreviewConflict("recovery-operation-invalid")
        return replay
    if _membership(session, chat_id) is not None:
        raise RecoveryPreviewConflict("chat-already-trashed")
    snapshot = require_current_recovery_preview(
        session,
        kind=RecoveryKind.CHAT,
        subject_id=chat_id,
        deletion_id=None,
        command=command,
        inspect=lambda current: _snapshot(current, chat_id, now),
        now=now,
        delete_generated_media=command.delete_generated_media,
    )
    if snapshot.counts.active_work:
        raise RecoveryPreviewConflict("chat-recovery-active-work")
    chat = session.get(Chat, chat_id)
    if chat is None:
        raise RecoveryPreviewConflict("recovery-subject-not-found")
    project = session.get(Project, chat.project_id) if chat.project_id else None
    item = RecoveryItem(
        kind=RecoveryKind.CHAT.value,
        subject_id=chat_id,
        display_label=chat.title.strip()[:240] or "Untitled chat",
        original_project_id=chat.project_id,
        original_project_label=project.name if project else None,
        deleted_at=_utc(now),
        purge_after=_utc(now) + RECOVERY_WINDOW,
        state=RecoveryState.RECOVERABLE.value,
        subject_revision=secrets.token_hex(32),
        delete_generated_media=command.delete_generated_media,
    )
    session.add(item)
    session.flush()
    result = RecoveryItemV1(
        deletion_id=item.deletion_id,
        kind=RecoveryKind.CHAT,
        subject_id=chat_id,
        display_label=item.display_label,
        original_location=RecoveryLocationV1(
            project_id=item.original_project_id, project_label=item.original_project_label
        ),
        deleted_at=item.deleted_at,
        purge_after=item.purge_after,
        state=RecoveryState.RECOVERABLE,
        revision=item.subject_revision,
        counts=snapshot.counts,
        delete_generated_media=item.delete_generated_media,
    )
    _remember(session, RecoveryAction.TRASH, command, result)
    return result


def _transition(
    session: Session,
    deletion_id: str,
    command: RestoreRecoveryV1 | PurgeRecoveryV1,
    now: datetime,
) -> RecoveryResultV1:
    reserve_recovery_write(session)
    action = (
        RecoveryAction.RESTORE if isinstance(command, RestoreRecoveryV1) else RecoveryAction.PURGE
    )
    operation = session.scalar(
        select(RecoveryOperation).where(
            RecoveryOperation.kind == RecoveryKind.CHAT.value,
            RecoveryOperation.deletion_id == deletion_id,
            RecoveryOperation.operation_key == command.operation_key,
        )
    )
    if operation is not None:
        _require_standard_chat(session, operation.subject_id)
    replay = _replay(operation, action, command)
    if replay is not None:
        if not isinstance(replay, RecoveryResultV1):
            raise RecoveryPreviewConflict("recovery-operation-invalid")
        return replay
    item = _item(session, deletion_id)
    other_operation = session.get(
        RecoveryOperation, (RecoveryKind.CHAT.value, item.subject_id, command.operation_key)
    )
    if other_operation is not None:
        raise RecoveryPreviewConflict("recovery-operation-conflict")
    chat_id = item.subject_id
    snapshot = require_current_recovery_preview(
        session,
        kind=RecoveryKind.CHAT,
        subject_id=chat_id,
        deletion_id=deletion_id,
        command=command,
        inspect=lambda current: _snapshot(current, chat_id, now),
        now=now,
        delete_generated_media=item.delete_generated_media,
    )
    if snapshot.counts.active_work:
        raise RecoveryPreviewConflict("chat-recovery-active-work")
    if action == RecoveryAction.RESTORE:
        if _utc(now) >= _utc(item.purge_after):
            raise RecoveryPreviewConflict("recovery-window-expired")
        if RecoveryConflict.ORIGINAL_PROJECT_MISSING in snapshot.conflicts and (
            not isinstance(command, RestoreRecoveryV1) or not command.restore_unfiled
        ):
            raise RecoveryPreviewConflict("recovery-original-project-missing")
        session.delete(item)
        session.flush()
        chat = session.get(Chat, chat_id)
        if chat is not None:
            reset_unavailable_chat_profiles(session, chat)
    else:
        graph = inspect_chat_recovery_graph(session, chat_id)
        media_ids = generated_media_ids(graph) if item.delete_generated_media else frozenset()
        item.state = RecoveryState.PURGING.value
        session.flush()
        # Delete owned jobs before the run and plan foreign keys disappear.
        for start in range(0, len(graph.owned_job_ids), 500):
            session.execute(delete(Job).where(Job.id.in_(graph.owned_job_ids[start : start + 500])))
        session.execute(delete(Chat).where(Chat.id == chat_id))
        if item.delete_generated_media:
            release_generated_membership(session, media_ids, item)
        item.state = RecoveryState.PURGED.value
        forget_recovery_labels(session, item)
        item.subject_revision = secrets.token_hex(32)
        item.revision += 1
    session.execute(
        delete(RecoveryPreviewRecord).where(
            RecoveryPreviewRecord.kind == RecoveryKind.CHAT.value,
            RecoveryPreviewRecord.subject_id == chat_id,
        )
    )
    result = RecoveryResultV1(
        deletion_id=deletion_id,
        kind=RecoveryKind.CHAT,
        subject_id=chat_id,
        action=action,
        # Shared or retained media continues through ordinary reference-aware retention.
        reclaimed_bytes=0,
    )
    _remember(session, action, command, result)
    return result


def restore_chat(
    session: Session, deletion_id: str, command: RestoreRecoveryV1, now: datetime
) -> RecoveryResultV1:
    """Restore the same chat without starting work; the caller commits under its chat lock."""

    return _transition(session, deletion_id, command, now)


def purge_chat(
    session: Session, deletion_id: str, command: PurgeRecoveryV1, now: datetime
) -> RecoveryResultV1:
    """Purge canonical rows atomically; the caller commits under the chat lock."""

    return _transition(session, deletion_id, command, now)
