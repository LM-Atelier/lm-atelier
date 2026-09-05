"""Preview and atomically detach one chat item's owned payload."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.orm import Session

from .artifact_library import referenced_artifact_ids
from .domain import JobKind, JobStatus, utcnow
from .models import (
    ChatItemRemovalReceipt,
    Job,
    Message,
    MessagePart,
    MessageReference,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
)
from .operation_idempotency_declaration_v1 import classify_idempotency_digest

MAX_PREVIEW_REFERENCES = 32
MAX_PREVIEW_REFERENCE_ID_BYTES = 1024
MAX_PREVIEW_ARTIFACT_IDS = 256
MAX_PREVIEW_ARTIFACT_ID_BYTES = 16_384

# These records are outside logical transcript-item removal. The response
# names the boundary explicitly so callers cannot mistake this operation for
# privacy deletion or forensic erasure.
RETAINED_WITNESS_CLASSES = (
    "chat_title",
    "run_prompt_settings_and_provenance",
    "work_plan_and_step_records",
    "job_records",
    "artifact_metadata_and_library_membership",
    "pre_removal_backups_exports_and_forks",
    "event_history",
)


class ChatItemRemovalNotFound(ValueError):
    def __init__(self) -> None:
        super().__init__("message not found")


class ChatItemRemovalIdentityMismatch(ValueError):
    def __init__(self) -> None:
        super().__init__("message identity does not match the removal request")


class ChatItemRemovalAlreadyRemoved(ValueError):
    def __init__(self) -> None:
        super().__init__("message content is already removed")


class ChatItemRemovalRevisionConflict(ValueError):
    def __init__(self) -> None:
        super().__init__("message removal preview is stale")


class ChatItemRemovalIdempotencyConflict(ValueError):
    def __init__(self) -> None:
        super().__init__("operation key was already used for a different removal request")


class ChatItemRemovalActiveWork(ValueError):
    def __init__(self, job_count: int) -> None:
        self.job_count = job_count
        super().__init__("chat has active work")


@dataclass(frozen=True)
class DetachedMessageReference:
    id: str
    subject_name: str
    mention_slug: str
    subject_kind: str


@dataclass(frozen=True)
class ChatItemRemovalImpact:
    chat_id: str
    message_id: str
    message_revision_id: str
    role: str
    already_removed: bool
    has_replies: bool
    source_backs_regeneration: bool
    detached_message_part_count: int
    detached_response_revision_part_count: int
    detached_reference_count: int
    detached_references: list[DetachedMessageReference] = field(default_factory=list)
    detached_references_truncated: bool = False
    released_artifact_count: int = 0
    released_artifact_ids: list[str] = field(default_factory=list)
    released_artifacts_truncated: bool = False
    retained_artifact_count: int = 0
    retained_artifact_ids: list[str] = field(default_factory=list)
    retained_artifacts_truncated: bool = False
    retained_witness_classes: list[str] = field(default_factory=list)
    forensic_erasure: bool = False
    execute_authorized: bool = False


@dataclass(frozen=True)
class ChatItemRemovalExecution:
    operation_key: str
    chat_id: str
    message_id: str
    message_revision_id: str
    content_removed_at: datetime
    replayed: bool


def preview_chat_item_removal(session: Session, message_id: str) -> ChatItemRemovalImpact:
    """Derive a bounded impact snapshot without mutating or authorizing removal."""

    message = session.get(Message, message_id)
    if message is None:
        raise ChatItemRemovalNotFound

    references = list(
        session.scalars(
            select(MessageReference)
            .where(MessageReference.message_id == message.id)
            .order_by(MessageReference.position, MessageReference.id)
        )
    )
    detached_references = _bounded_references(references)

    # Validate the complete current graph first; the second pass then asks
    # which target-owned roots still have any independently retained edge.
    referenced_artifact_ids(session)
    surviving_artifacts = referenced_artifact_ids(
        session,
        exclude_message_payload_for=message.id,
    )
    target_artifacts = _target_artifact_ids(session, message.id, references)
    released_artifacts = sorted(target_artifacts - surviving_artifacts)
    retained_artifacts = sorted(target_artifacts & surviving_artifacts)
    released_sample = _bounded_ids(released_artifacts)
    retained_sample = _bounded_ids(retained_artifacts)

    source_run_ids = set(
        session.scalars(
            select(Run.id).where(
                or_(Run.user_message_id == message.id, Run.assistant_message_id == message.id)
            )
        )
    )
    source_run_ids.update(
        value
        for value in session.scalars(
            select(ResponseRevision.run_id).where(
                ResponseRevision.message_id == message.id,
                ResponseRevision.run_id.is_not(None),
            )
        )
        if value
    )

    return ChatItemRemovalImpact(
        chat_id=message.chat_id,
        message_id=message.id,
        message_revision_id=_message_revision_id(
            session,
            message,
            references=references,
            target_artifacts=target_artifacts,
            surviving_artifacts=surviving_artifacts,
            source_run_ids=source_run_ids,
        ),
        role=message.role,
        already_removed=message.content_removed_at is not None,
        has_replies=session.scalar(
            select(Message.id)
            .where(Message.chat_id == message.chat_id, Message.parent_id == message.id)
            .limit(1)
        )
        is not None,
        source_backs_regeneration=bool(source_run_ids),
        detached_message_part_count=_count_message_parts(session, message.id),
        detached_response_revision_part_count=_count_revision_parts(session, message.id),
        detached_reference_count=len(references),
        detached_references=detached_references,
        detached_references_truncated=len(detached_references) != len(references),
        released_artifact_count=len(released_artifacts),
        released_artifact_ids=released_sample,
        released_artifacts_truncated=len(released_sample) != len(released_artifacts),
        retained_artifact_count=len(retained_artifacts),
        retained_artifact_ids=retained_sample,
        retained_artifacts_truncated=len(retained_sample) != len(retained_artifacts),
        retained_witness_classes=list(RETAINED_WITNESS_CLASSES),
    )


def execute_chat_item_removal(
    session: Session,
    message_id: str,
    *,
    expected_message_id: str,
    expected_revision_id: str,
    operation_key: str,
) -> ChatItemRemovalExecution:
    """Detach target-owned payload after the endpoint acquires the chat lock."""

    if expected_message_id != message_id:
        raise ChatItemRemovalIdentityMismatch
    message = session.get(Message, message_id)
    if message is None:
        raise ChatItemRemovalNotFound

    request_sha256 = _request_sha256(
        expected_message_id=expected_message_id,
        expected_revision_id=expected_revision_id,
    )
    receipt = session.scalar(
        select(ChatItemRemovalReceipt).where(
            ChatItemRemovalReceipt.chat_id == message.chat_id,
            ChatItemRemovalReceipt.operation_key == operation_key,
        )
    )
    if receipt is not None:
        comparison = classify_idempotency_digest(
            left_digest=receipt.request_sha256,
            right_digest=request_sha256,
        )
        if comparison != "same_digest":
            raise ChatItemRemovalIdempotencyConflict
        return _execution_from_receipt(receipt, replayed=True)

    if message.content_removed_at is not None:
        raise ChatItemRemovalAlreadyRemoved

    active_job_count = _active_job_count(session, message.chat_id)
    if active_job_count:
        raise ChatItemRemovalActiveWork(active_job_count)

    impact = preview_chat_item_removal(session, message.id)
    if impact.message_revision_id != expected_revision_id:
        raise ChatItemRemovalRevisionConflict

    revision_ids = select(ResponseRevision.id).where(ResponseRevision.message_id == message.id)
    session.execute(
        delete(ResponseRevisionPart).where(
            ResponseRevisionPart.response_revision_id.in_(revision_ids)
        )
    )
    session.execute(delete(MessagePart).where(MessagePart.message_id == message.id))
    session.execute(delete(MessageReference).where(MessageReference.message_id == message.id))
    removed_at = utcnow()
    message.content_removed_at = removed_at
    session.add(
        ChatItemRemovalReceipt(
            chat_id=message.chat_id,
            operation_key=operation_key,
            message_id=message.id,
            request_sha256=request_sha256,
            message_revision_id=impact.message_revision_id,
            content_removed_at=removed_at,
        )
    )
    session.flush()
    return ChatItemRemovalExecution(
        operation_key=operation_key,
        chat_id=message.chat_id,
        message_id=message.id,
        message_revision_id=impact.message_revision_id,
        content_removed_at=removed_at,
        replayed=False,
    )


def _request_sha256(*, expected_message_id: str, expected_revision_id: str) -> str:
    encoded = json.dumps(
        {
            "expected_message_id": expected_message_id,
            "expected_revision_id": expected_revision_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _execution_from_receipt(
    receipt: ChatItemRemovalReceipt,
    *,
    replayed: bool,
) -> ChatItemRemovalExecution:
    return ChatItemRemovalExecution(
        operation_key=receipt.operation_key,
        chat_id=receipt.chat_id,
        message_id=receipt.message_id,
        message_revision_id=receipt.message_revision_id,
        content_removed_at=_as_utc(receipt.content_removed_at),
        replayed=replayed,
    )


def _active_job_count(session: Session, chat_id: str) -> int:
    active_statuses = (
        JobStatus.QUEUED.value,
        JobStatus.RUNNING.value,
        JobStatus.PAUSED.value,
    )
    return (
        session.scalar(
            select(func.count(Job.id))
            .outerjoin(Run, Job.run_id == Run.id)
            .where(
                Job.status.in_(active_statuses),
                or_(
                    Run.chat_id == chat_id,
                    and_(
                        Job.run_id.is_(None),
                        Job.kind == JobKind.EDIT_VERIFY.value,
                        Job.payload_json["chat_id"].as_string() == chat_id,
                    ),
                ),
            )
        )
        or 0
    )


def _message_revision_id(
    session: Session,
    message: Message,
    *,
    references: list[MessageReference],
    target_artifacts: set[str],
    surviving_artifacts: AbstractSet[str],
    source_run_ids: set[str],
) -> str:
    """Hash structural authority only; never hash short message content."""

    message_parts = list(
        session.execute(
            select(
                MessagePart.id,
                MessagePart.position,
                MessagePart.type,
                MessagePart.artifact_id,
                MessagePart.updated_at,
            )
            .where(MessagePart.message_id == message.id)
            .order_by(MessagePart.position, MessagePart.id)
        ).all()
    )
    revisions = list(
        session.execute(
            select(
                ResponseRevision.id,
                ResponseRevision.sequence,
                ResponseRevision.status,
                ResponseRevision.run_id,
                ResponseRevision.updated_at,
            )
            .where(ResponseRevision.message_id == message.id)
            .order_by(ResponseRevision.sequence, ResponseRevision.id)
        ).all()
    )
    revision_parts = list(
        session.execute(
            select(
                ResponseRevisionPart.response_revision_id,
                ResponseRevisionPart.id,
                ResponseRevisionPart.position,
                ResponseRevisionPart.type,
                ResponseRevisionPart.artifact_id,
                ResponseRevisionPart.updated_at,
            )
            .select_from(ResponseRevisionPart)
            .join(
                ResponseRevision,
                ResponseRevisionPart.response_revision_id == ResponseRevision.id,
            )
            .where(ResponseRevision.message_id == message.id)
            .order_by(
                ResponseRevisionPart.response_revision_id,
                ResponseRevisionPart.position,
                ResponseRevisionPart.id,
            )
        ).all()
    )
    reply_ids = list(
        session.scalars(
            select(Message.id)
            .where(Message.chat_id == message.chat_id, Message.parent_id == message.id)
            .order_by(Message.id)
        )
    )
    authority: dict[str, Any] = {
        "message": [
            message.id,
            message.chat_id,
            message.parent_id,
            message.role,
            message.status,
            message.transcript_visible,
            message.active_response_revision_id,
            _timestamp(message.updated_at),
        ],
        "message_parts": [[*row[:-1], _timestamp(row[-1])] for row in message_parts],
        "revisions": [[*row[:-1], _timestamp(row[-1])] for row in revisions],
        "revision_parts": [
            [
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                _timestamp(row[5]),
            ]
            for row in revision_parts
        ],
        "references": [
            [
                reference.id,
                reference.position,
                reference.reference_subject_id,
                reference.mention_slug,
                reference.subject_name,
                reference.subject_kind,
                reference.role,
                reference.strength,
                reference.source,
                reference.reference_asset_ids_json,
                reference.artifact_ids_json,
                _timestamp(reference.updated_at),
            ]
            for reference in references
        ],
        "reply_ids": reply_ids,
        "source_run_ids": sorted(source_run_ids),
        "target_artifacts": sorted(target_artifacts),
        "retained_target_artifacts": sorted(target_artifacts & surviving_artifacts),
    }
    encoded = json.dumps(authority, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: datetime) -> str:
    return value.isoformat()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _bounded_references(
    references: list[MessageReference],
) -> list[DetachedMessageReference]:
    bounded: list[DetachedMessageReference] = []
    id_bytes = 0
    for reference in references:
        encoded_size = len(reference.id.encode("utf-8"))
        if len(bounded) >= MAX_PREVIEW_REFERENCES:
            break
        if id_bytes + encoded_size > MAX_PREVIEW_REFERENCE_ID_BYTES:
            break
        id_bytes += encoded_size
        bounded.append(
            DetachedMessageReference(
                id=reference.id,
                subject_name=reference.subject_name,
                mention_slug=reference.mention_slug,
                subject_kind=reference.subject_kind,
            )
        )
    return bounded


def _bounded_ids(values: list[str]) -> list[str]:
    bounded: list[str] = []
    id_bytes = 0
    for value in values:
        encoded_size = len(value.encode("utf-8"))
        if len(bounded) >= MAX_PREVIEW_ARTIFACT_IDS:
            break
        if id_bytes + encoded_size > MAX_PREVIEW_ARTIFACT_ID_BYTES:
            break
        id_bytes += encoded_size
        bounded.append(value)
    return bounded


def _count_message_parts(session: Session, message_id: str) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(MessagePart)
            .where(MessagePart.message_id == message_id)
        )
        or 0
    )


def _count_revision_parts(session: Session, message_id: str) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(ResponseRevisionPart)
            .join(
                ResponseRevision,
                ResponseRevisionPart.response_revision_id == ResponseRevision.id,
            )
            .where(ResponseRevision.message_id == message_id)
        )
        or 0
    )


def _target_artifact_ids(
    session: Session,
    message_id: str,
    references: list[MessageReference],
) -> set[str]:
    found = {
        value
        for value in session.scalars(
            select(MessagePart.artifact_id).where(
                MessagePart.message_id == message_id,
                MessagePart.artifact_id.is_not(None),
            )
        )
        if value
    }
    found.update(
        value
        for value in session.scalars(
            select(ResponseRevisionPart.artifact_id)
            .join(
                ResponseRevision,
                ResponseRevisionPart.response_revision_id == ResponseRevision.id,
            )
            .where(
                ResponseRevision.message_id == message_id,
                ResponseRevisionPart.artifact_id.is_not(None),
            )
        )
        if value
    )
    for reference in references:
        found.update(reference.artifact_ids_json)
    return found
