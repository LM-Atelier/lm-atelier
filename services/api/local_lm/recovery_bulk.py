"""Apply a materialized recovery selection in one caller-owned transaction."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .chat_recovery import preview_chat_recovery, purge_chat, restore_chat
from .domain import new_id
from .media_recovery import preview_media_recovery, purge_media, restore_media
from .models import Chat, RecoveryBatchRecord, RecoveryItem, RecoveryPreviewRecord
from .project_recovery import preview_project_recovery, purge_project, restore_project
from .prompt_helpers import STANDARD_CHAT_SCOPE
from .recovery_bulk_v1 import (
    RecoveryBatchApplyV1,
    RecoveryBatchMemberV1,
    RecoveryBatchPreviewV1,
    RecoveryBatchResultV1,
    RecoveryBatchSelectionV1,
)
from .recovery_previews import (
    PREVIEW_LIFETIME,
    RecoveryPreviewConflict,
    _utc,
    reserve_recovery_write,
)
from .recovery_v1 import (
    PurgeRecoveryV1,
    RecoveryAction,
    RecoveryConflict,
    RecoveryImpactV1,
    RecoveryKind,
    RecoveryResultV1,
    RestoreRecoveryV1,
)
from .workflow_recovery import preview_workflow_recovery, purge_workflow, restore_workflow


def batch_record(
    session: Session, batch_id: str
) -> tuple[RecoveryBatchRecord, RecoveryBatchPreviewV1]:
    record = session.get(RecoveryBatchRecord, batch_id)
    if record is None:
        raise RecoveryPreviewConflict("recovery-batch-not-found")
    try:
        preview = RecoveryBatchPreviewV1.model_validate(record.preview_json)
    except ValidationError:
        raise RecoveryPreviewConflict("recovery-operation-invalid") from None
    if preview.batch_id != record.id or len({item.deletion_id for item in preview.items}) != len(
        preview.items
    ):
        raise RecoveryPreviewConflict("recovery-operation-invalid")
    expected_digest = hashlib.sha256(
        preview.model_copy(update={"impact_sha256": "0" * 64}).model_dump_json().encode()
    ).hexdigest()
    if (
        not hmac.compare_digest(preview.impact_sha256, expected_digest)
        or not isinstance(record.fingerprints_json, dict)
        or set(record.fingerprints_json) != {item.deletion_id for item in preview.items}
    ):
        raise RecoveryPreviewConflict("recovery-operation-invalid")
    return record, preview


def batch_chat_ids(session: Session, deletion_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            session.scalars(
                select(RecoveryItem.subject_id).where(
                    RecoveryItem.deletion_id.in_(deletion_ids),
                    RecoveryItem.kind == RecoveryKind.CHAT.value,
                )
            )
        )
    )


def _member(session: Session, deletion_id: str, now: datetime) -> tuple[RecoveryBatchMemberV1, str]:
    item = session.get(RecoveryItem, deletion_id)
    if item is None:
        raise RecoveryPreviewConflict("recovery-impact-stale")
    try:
        kind = RecoveryKind(item.kind)
    except ValueError:
        raise RecoveryPreviewConflict("recovery-item-not-found") from None
    if kind == RecoveryKind.CHAT:
        if (
            session.scalar(select(Chat.scope).where(Chat.id == item.subject_id))
            != STANDARD_CHAT_SCOPE
        ):
            raise RecoveryPreviewConflict("recovery-item-not-found")
        impact = preview_chat_recovery(session, deletion_id, now)
    elif kind == RecoveryKind.PROJECT:
        impact = preview_project_recovery(session, deletion_id, now)
    elif kind == RecoveryKind.MEDIA_LIBRARY_ENTRY:
        impact = preview_media_recovery(session, deletion_id, now)
    else:
        impact = preview_workflow_recovery(session, deletion_id, now)
    member = RecoveryBatchMemberV1(
        deletion_id=deletion_id,
        display_label=item.display_label,
        purge_after=_utc(item.purge_after),
        impact=impact,
    )
    record = session.get(RecoveryPreviewRecord, impact.revision)
    if record is None:
        raise RecoveryPreviewConflict("recovery-operation-invalid")
    return member, record.subject_fingerprint


def _allowed(impact: RecoveryImpactV1, selection: RecoveryBatchSelectionV1) -> bool:
    return RecoveryAction(selection.action) in impact.available_actions and not (
        selection.action == "restore"
        and not selection.restore_unfiled
        and RecoveryConflict.ORIGINAL_PROJECT_MISSING in impact.conflicts
    )


def materialize_batch(
    session: Session, selection: RecoveryBatchSelectionV1, now: datetime
) -> RecoveryBatchPreviewV1:
    reserve_recovery_write(session)
    members = [_member(session, identity, now) for identity in sorted(selection.deletion_ids)]
    draft = RecoveryBatchPreviewV1(
        batch_id=new_id("rcbatch"),
        revision=secrets.token_hex(32),
        impact_sha256="0" * 64,
        action=selection.action,
        restore_unfiled=selection.restore_unfiled,
        available=all(_allowed(member.impact, selection) for member, _ in members),
        expires_at=_utc(now) + PREVIEW_LIFETIME,
        items=tuple(member for member, _ in members),
    )
    digest = hashlib.sha256(draft.model_dump_json().encode()).hexdigest()
    preview = draft.model_copy(update={"impact_sha256": digest})
    session.execute(
        delete(RecoveryBatchRecord).where(
            RecoveryBatchRecord.expires_at <= _utc(now), RecoveryBatchRecord.operation_key.is_(None)
        )
    )
    session.add(
        RecoveryBatchRecord(
            id=preview.batch_id,
            preview_json=preview.model_dump(mode="json"),
            fingerprints_json={member.deletion_id: fingerprint for member, fingerprint in members},
            expires_at=preview.expires_at,
        )
    )
    session.flush()
    return preview


def _comparable(member: RecoveryBatchMemberV1) -> dict[str, Any]:
    value = member.model_dump(mode="json")
    value["impact"].pop("revision")
    value["impact"].pop("impact_sha256")
    return value


def _transition(
    session: Session,
    member: RecoveryBatchMemberV1,
    selection: RecoveryBatchSelectionV1,
    batch_id: str,
    operation_key: str,
    now: datetime,
) -> RecoveryResultV1:
    current, _ = _member(session, member.deletion_id, now)
    if not _allowed(current.impact, selection):
        raise RecoveryPreviewConflict("recovery-batch-unavailable")
    key = (
        "batch_"
        + hashlib.sha256(f"{batch_id}:{operation_key}:{member.deletion_id}".encode()).hexdigest()
    )
    fields = dict(
        expected_revision=current.impact.revision,
        impact_sha256=current.impact.impact_sha256,
        operation_key=key,
    )
    if selection.action == "restore":
        restore = RestoreRecoveryV1(**fields, restore_unfiled=selection.restore_unfiled)
        if current.impact.kind == RecoveryKind.CHAT:
            return restore_chat(session, member.deletion_id, restore, now)
        if current.impact.kind == RecoveryKind.PROJECT:
            return restore_project(session, member.deletion_id, restore, now)
        if current.impact.kind == RecoveryKind.MEDIA_LIBRARY_ENTRY:
            return restore_media(session, member.deletion_id, restore, now)
        return restore_workflow(session, member.deletion_id, restore, now)
    purge = PurgeRecoveryV1(**fields, acknowledgement="permanently-delete")
    if current.impact.kind == RecoveryKind.CHAT:
        return purge_chat(session, member.deletion_id, purge, now)
    if current.impact.kind == RecoveryKind.PROJECT:
        return purge_project(session, member.deletion_id, purge, now)
    if current.impact.kind == RecoveryKind.MEDIA_LIBRARY_ENTRY:
        return purge_media(session, member.deletion_id, purge, now)
    return purge_workflow(session, member.deletion_id, purge, now)


def apply_batch(
    session: Session, batch_id: str, command: RecoveryBatchApplyV1, now: datetime
) -> RecoveryBatchResultV1:
    reserve_recovery_write(session)
    record, preview = batch_record(session, batch_id)
    digest = hashlib.sha256(command.model_dump_json().encode()).hexdigest()
    if record.operation_key is not None:
        if (
            record.operation_key != command.operation_key
            or record.request_sha256 is None
            or not hmac.compare_digest(record.request_sha256, digest)
        ):
            raise RecoveryPreviewConflict("recovery-operation-conflict")
        try:
            result = RecoveryBatchResultV1.model_validate(record.response_json)
        except ValidationError:
            raise RecoveryPreviewConflict("recovery-operation-invalid") from None
        expected = {
            (item.deletion_id, item.impact.kind, item.impact.subject_id) for item in preview.items
        }
        actual = {(item.deletion_id, item.kind, item.subject_id) for item in result.results}
        if (
            result.batch_id != batch_id
            or result.action != preview.action
            or len(result.results) != len(expected)
            or actual != expected
            or any(item.action.value != preview.action for item in result.results)
            or result.reclaimed_bytes != sum(item.reclaimed_bytes for item in result.results)
        ):
            raise RecoveryPreviewConflict("recovery-operation-invalid")
        return result
    if preview.action == "purge" and command.acknowledgement != "permanently-delete":
        raise RecoveryPreviewConflict("recovery-batch-acknowledgement-required")
    if preview.action == "restore" and command.acknowledgement is not None:
        raise RecoveryPreviewConflict("recovery-batch-acknowledgement-invalid")
    if (
        session.scalar(
            select(RecoveryBatchRecord.id).where(
                RecoveryBatchRecord.operation_key == command.operation_key
            )
        )
        is not None
    ):
        raise RecoveryPreviewConflict("recovery-operation-conflict")
    if (
        _utc(record.expires_at) <= _utc(now)
        or command.expected_revision != preview.revision
        or not hmac.compare_digest(command.impact_sha256, preview.impact_sha256)
    ):
        raise RecoveryPreviewConflict("recovery-impact-stale")
    selection = RecoveryBatchSelectionV1(
        deletion_ids=tuple(item.deletion_id for item in preview.items),
        action=preview.action,
        restore_unfiled=preview.restore_unfiled,
    )
    for member in preview.items:
        fresh, fingerprint = _member(session, member.deletion_id, now)
        if record.fingerprints_json.get(member.deletion_id) != fingerprint or _comparable(
            fresh
        ) != _comparable(member):
            raise RecoveryPreviewConflict("recovery-impact-stale")
        if not _allowed(fresh.impact, selection):
            raise RecoveryPreviewConflict("recovery-batch-unavailable")
    # Validate the whole initial snapshot before planned members can change each
    # other's references. Every renewed item preview stays inside this writer.
    first = RecoveryKind.PROJECT if preview.action == "restore" else RecoveryKind.CHAT
    ordered = sorted(
        preview.items, key=lambda member: (member.impact.kind != first, member.deletion_id)
    )
    results = {
        member.deletion_id: _transition(
            session, member, selection, batch_id, command.operation_key, now
        )
        for member in ordered
    }
    result = RecoveryBatchResultV1(
        batch_id=batch_id,
        action=preview.action,
        reclaimed_bytes=sum(value.reclaimed_bytes for value in results.values()),
        results=tuple(results[member.deletion_id] for member in preview.items),
    )
    if len(result.model_dump_json().encode()) > 32 * 1024:
        raise RecoveryPreviewConflict("recovery-operation-invalid")
    record.operation_key = command.operation_key
    record.request_sha256 = digest
    record.response_json = result.model_dump(mode="json")
    session.flush()
    return result
