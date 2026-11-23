"""Expire recoverable memberships under their ordinary lifecycle boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from .artifact_deletion_authority import ArtifactReferenceIndexStale
from .chat_recovery import preview_chat_recovery, purge_chat
from .chat_recovery_graph import ChatRecoveryGraphError
from .db import SessionLocal, database_is_contended
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
#: A due item that keeps failing is tried again after a wait that doubles from
#: the first to the longest. Each try takes SQLite's writer for as long as the
#: item takes to check, so an item that cannot be purged must not take it every
#: minute for as long as it stays stuck.
EXPIRY_RETRY_FIRST_SECONDS = 120.0
EXPIRY_RETRY_LONGEST_SECONDS = 6 * 60 * 60.0
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
    #: Due items passed over because an earlier failure's wait is not over.
    waiting: int = 0


@dataclass(frozen=True)
class ExpiryWait:
    """How many times a due item has failed in a row, and when it may be tried again."""

    failures: int
    retry_at: datetime


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


def _purge(candidate: _Candidate, now: datetime) -> str | None:
    """Purge one due item: None once it is purged, otherwise the fixed reason it waits."""

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
                return "recovery-item-changed"
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
            return None
        # Each waits for a later try; an index catching up is ready by then. The
        # reason is a fixed code, never anything the item holds.
        except RecoveryPreviewConflict as error:
            session.rollback()
            return error.code
        except ChatRecoveryGraphError:
            session.rollback()
            return "chat-recovery-graph-invalid"
        except ArtifactReferenceIndexStale:
            session.rollback()
            return "artifact-reference-index-stale"
        except SQLAlchemyError as error:
            session.rollback()
            if isinstance(error, OperationalError) and database_is_contended(error):
                return "recovery-database-busy"
            return "recovery-database-error"


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
    waits: dict[str, ExpiryWait] | None = None,
) -> ExpiryBatch:
    """Purge a bounded seek page; each membership commits or rolls back as one item.

    With `waits`, an item that failed before is passed over until its wait is
    over, and each failure in a row doubles the wait up to the longest; a purge
    clears it. The cursor still moves past a passed-over item, so one stuck
    item never holds up the items after it.
    """
    if not 1 <= limit <= 20:
        raise ValueError("Recovery expiry batches contain between one and twenty items.")
    candidates = await _finish_thread(
        asyncio.create_task(asyncio.to_thread(_candidates, now, after, limit))
    )
    purged = deferred = waiting = 0
    for candidate in candidates:
        earlier = waits.get(candidate.deletion_id) if waits is not None else None
        if earlier is not None and now < earlier.retry_at:
            waiting += 1
            continue

        async def perform(current: _Candidate = candidate) -> str | None:
            return await _finish_thread(
                asyncio.create_task(asyncio.to_thread(_purge, current, now))
            )

        if candidate.kind == RecoveryKind.CHAT.value:
            async with services.orchestrator.chat_guard(candidate.subject_id):
                reason = await perform()
        else:
            reason = await perform()
        if reason is not None:
            deferred += 1
            if waits is not None:
                failures = (earlier.failures if earlier is not None else 0) + 1
                delay = min(
                    EXPIRY_RETRY_FIRST_SECONDS * 2 ** (failures - 1), EXPIRY_RETRY_LONGEST_SECONDS
                )
                waits[candidate.deletion_id] = ExpiryWait(failures, now + timedelta(seconds=delay))
                logger.info(
                    "Recovery expiry will try %s again in %.0f s (%s)",
                    candidate.deletion_id,
                    delay,
                    reason,
                )
            continue
        purged += 1
        if waits is not None:
            waits.pop(candidate.deletion_id, None)
        if candidate.kind == RecoveryKind.PROJECT.value:
            await services.events.publish("project.updated", candidate.subject_id, {})
        await services.events.publish("recovery.updated", candidate.deletion_id, {})
    last = candidates[-1] if candidates else None
    cursor = (
        ExpiryCursor(last.purge_after, last.deletion_id)
        if last is not None and len(candidates) == limit
        else None
    )
    return ExpiryBatch(len(candidates) - waiting, purged, deferred, cursor, waiting)


async def maintain_recovery_expiry(
    services: Services, *, interval_seconds: float = EXPIRY_INTERVAL_SECONDS
) -> None:
    """Resume from durable deadlines on startup without creating generation jobs."""
    if interval_seconds <= 0:
        raise ValueError("Recovery expiry checks need a positive interval.")
    cursor: ExpiryCursor | None = None
    # Kept for the life of the process: a restart tries every due item once more.
    waits: dict[str, ExpiryWait] = {}
    while True:
        try:
            result = await expire_recovery_batch(
                services, datetime.now(UTC), after=cursor, waits=waits
            )
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
