"""Expire recoverable memberships under their ordinary lifecycle boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import SQLAlchemyError

from .chat_recovery import preview_chat_recovery, purge_chat
from .chat_recovery_graph import ChatRecoveryGraphError
from .db import SessionLocal
from .media_recovery import preview_media_recovery, purge_media
from .models import RecoveryItem
from .project_recovery import preview_project_recovery, purge_project
from .recovery_previews import RecoveryPreviewConflict, _utc, reserve_recovery_write
from .recovery_v1 import PurgeRecoveryV1, RecoveryKind
from .workflow_recovery import preview_workflow_recovery, purge_workflow

if TYPE_CHECKING:
    from .main import Services

logger = logging.getLogger("local_lm")
EXPIRY_INTERVAL_SECONDS = 60.0
EXPIRY_BATCH_SIZE = 10
_SUPPORTED = (
    RecoveryKind.CHAT.value,
    RecoveryKind.PROJECT.value,
    RecoveryKind.MEDIA_LIBRARY_ENTRY.value,
    RecoveryKind.WORKFLOW_FAMILY.value,
)


@dataclass(frozen=True)
class ExpiryCursor:
    purge_after: datetime
    deletion_id: str


@dataclass(frozen=True)
class ExpiryBatch:
    examined: int
    purged: int
    deferred: int
    next_cursor: ExpiryCursor | None


@dataclass(frozen=True)
class _Candidate:
    deletion_id: str
    kind: str
    subject_id: str
    purge_after: datetime
    revision: str


def _candidates(now: datetime, after: ExpiryCursor | None, limit: int) -> list[_Candidate]:
    statement = select(RecoveryItem).where(
        RecoveryItem.kind.in_(_SUPPORTED),
        RecoveryItem.state == "recoverable",
        RecoveryItem.purge_after <= _utc(now),
    )
    if after is not None:
        statement = statement.where(
            or_(
                RecoveryItem.purge_after > _utc(after.purge_after),
                and_(
                    RecoveryItem.purge_after == _utc(after.purge_after),
                    RecoveryItem.deletion_id > after.deletion_id,
                ),
            )
        )
    with SessionLocal() as session:
        return [
            _Candidate(
                row.deletion_id,
                row.kind,
                row.subject_id,
                _utc(row.purge_after),
                row.subject_revision,
            )
            for row in session.scalars(
                statement.order_by(RecoveryItem.purge_after, RecoveryItem.deletion_id).limit(limit)
            )
        ]


def _purge(candidate: _Candidate, now: datetime) -> bool:
    with SessionLocal() as session:
        try:
            reserve_recovery_write(session)
            item = session.get(RecoveryItem, candidate.deletion_id)
            if item is None or (
                item.kind != candidate.kind
                or item.subject_id != candidate.subject_id
                or item.subject_revision != candidate.revision
                or item.state != "recoverable"
                or _utc(item.purge_after) != candidate.purge_after
                or _utc(item.purge_after) > _utc(now)
            ):
                session.rollback()
                return False
            preview_action = {
                RecoveryKind.CHAT.value: preview_chat_recovery,
                RecoveryKind.PROJECT.value: preview_project_recovery,
                RecoveryKind.MEDIA_LIBRARY_ENTRY.value: preview_media_recovery,
                RecoveryKind.WORKFLOW_FAMILY.value: preview_workflow_recovery,
            }[item.kind]
            preview = preview_action(session, item.deletion_id, now)
            key = hashlib.sha256(
                f"{item.deletion_id}:{candidate.purge_after.isoformat()}".encode("ascii")
            ).hexdigest()
            command = PurgeRecoveryV1(
                expected_revision=preview.revision,
                impact_sha256=preview.impact_sha256,
                operation_key=f"expiry-{key}",
                acknowledgement="permanently-delete",
            )
            if item.kind == RecoveryKind.CHAT.value:
                purge_chat(session, item.deletion_id, command, now)
            elif item.kind == RecoveryKind.PROJECT.value:
                purge_project(session, item.deletion_id, command, now)
            elif item.kind == RecoveryKind.MEDIA_LIBRARY_ENTRY.value:
                purge_media(session, item.deletion_id, command, now)
            else:
                purge_workflow(session, item.deletion_id, command, now)
            session.commit()
            return True
        except (RecoveryPreviewConflict, ChatRecoveryGraphError, SQLAlchemyError):
            session.rollback()
            return False


async def _finish_thread[Result](operation: asyncio.Task[Result]) -> Result:
    """Finish a database operation before releasing its lifecycle lock on shutdown."""
    interrupted = False
    while True:
        try:
            result = await asyncio.shield(operation)
            if interrupted:
                raise asyncio.CancelledError
            return result
        except asyncio.CancelledError:
            if operation.done():
                raise
            interrupted = True


async def expire_recovery_batch(
    services: Services,
    now: datetime,
    *,
    after: ExpiryCursor | None = None,
    limit: int = EXPIRY_BATCH_SIZE,
) -> ExpiryBatch:
    """Purge a bounded seek page; each membership commits or rolls back as one item."""
    if not 1 <= limit <= 20:
        raise ValueError("Recovery expiry batches contain between one and twenty items.")
    candidates = await _finish_thread(
        asyncio.create_task(asyncio.to_thread(_candidates, now, after, limit))
    )
    purged = 0
    for candidate in candidates:

        async def perform(current: _Candidate = candidate) -> bool:
            return await _finish_thread(
                asyncio.create_task(asyncio.to_thread(_purge, current, now))
            )

        if candidate.kind == RecoveryKind.CHAT.value:
            async with services.orchestrator.chat_guard(candidate.subject_id):
                changed = await perform()
        else:
            changed = await perform()
        if changed:
            purged += 1
            if candidate.kind == RecoveryKind.PROJECT.value:
                await services.events.publish("project.updated", candidate.subject_id, {})
            await services.events.publish("recovery.updated", candidate.deletion_id, {})
    last = candidates[-1] if candidates else None
    cursor = (
        ExpiryCursor(last.purge_after, last.deletion_id)
        if last is not None and len(candidates) == limit
        else None
    )
    return ExpiryBatch(len(candidates), purged, len(candidates) - purged, cursor)


async def maintain_recovery_expiry(
    services: Services, *, interval_seconds: float = EXPIRY_INTERVAL_SECONDS
) -> None:
    """Resume from durable deadlines on startup without creating generation jobs."""
    if interval_seconds <= 0:
        raise ValueError("Recovery expiry checks need a positive interval.")
    cursor: ExpiryCursor | None = None
    while True:
        try:
            result = await expire_recovery_batch(services, datetime.now(UTC), after=cursor)
            cursor = result.next_cursor
            if result.purged or result.deferred:
                logger.info(
                    "Recovery expiry examined %s item(s), purged %s, deferred %s",
                    result.examined,
                    result.purged,
                    result.deferred,
                )
        except Exception:
            logger.error("Recovery expiry check could not finish; it will retry later.")
            cursor = None
        await asyncio.sleep(interval_seconds)
