"""Read-only impact preview for removing one chat item's payload."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .artifact_library import referenced_artifact_ids
from .models import (
    Message,
    MessagePart,
    MessageReference,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
)

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
