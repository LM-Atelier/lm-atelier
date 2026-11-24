"""Bind short-lived deletion previews to a fresh, reserved database snapshot."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from .db import database_is_contended
from .models import RecoveryPreviewRecord
from .recovery_v1 import (
    RecoveryAction,
    RecoveryCommandV1,
    RecoveryConflict,
    RecoveryCountsV1,
    RecoveryImpactV1,
    RecoveryKind,
)

PREVIEW_LIFETIME = timedelta(minutes=15)
#: The checks a snapshot purge has read, with the transaction they were read in.
_SNAPSHOT_INSPECTIONS = "lm_atelier_recovery_snapshot_inspections"


class RecoveryPreviewConflict(ValueError):
    """A fixed reason for refusing a preview, without stored resource content."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class RecoverySnapshot:
    """Keep the private comparison separate from the counts shown to the user."""

    fingerprint: str = field(repr=False)
    counts: RecoveryCountsV1
    conflicts: tuple[RecoveryConflict, ...] = ()
    available_actions: tuple[RecoveryAction, ...] = ()


RecoveryInspector = Callable[[Session], RecoverySnapshot]


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def reserve_recovery_write(session: Session) -> None:
    """Reserve the writer before reading, including an existing deferred transaction.

    A read transaction alone does not exclude another writer. A zero-row write
    reserves SQLite's writer without changing a preview or committing the caller's
    transaction. A stale WAL snapshot refuses rather than being treated as fresh.
    Loaded ORM values are expired only after the reservation succeeds.
    """

    if session.new or session.dirty or session.deleted:
        raise RecoveryPreviewConflict("recovery-transaction-not-clean")
    if session.get_bind().dialect.name != "sqlite":
        raise RecoveryPreviewConflict("recovery-database-unsupported")
    try:
        with session.no_autoflush:
            session.connection().exec_driver_sql(
                "UPDATE recovery_previews SET revision = revision WHERE 0"
            )
    except OperationalError as error:
        if _SNAPSHOT_INSPECTIONS in session.info and snapshot_moved_on(error):
            # Something was committed after this purge began reading, so what it
            # read may no longer be what it would delete.
            raise RecoveryPreviewConflict("recovery-snapshot-stale") from None
        if database_is_contended(error):
            raise RecoveryPreviewConflict("recovery-database-busy") from None
        raise
    session.expire_all()


def snapshot_moved_on(error: OperationalError) -> bool:
    """Whether SQLite refused the writer because a read snapshot is out of date."""

    return getattr(error.orig, "sqlite_errorcode", None) == sqlite3.SQLITE_BUSY_SNAPSHOT


def begin_recovery_snapshot(session: Session) -> None:
    """Read a purge's checks from one snapshot, without holding SQLite's writer.

    Python's sqlite3 begins a transaction only before a write, so on their own
    the reads of a check would each see the database as it was at that moment.
    An explicit BEGIN makes every read until the writer is reserved share the
    snapshot of the first. SQLite refuses to reserve the writer from a snapshot
    once anything has been committed after it (SQLITE_BUSY_SNAPSHOT), so a
    reservation that succeeds proves every check read before it is still
    current, and `remember_recovery_inspection` keeps those checks for the rest
    of the transaction instead of reading them again under the writer. SQLite
    does not wait for another writer here either: a transaction that has read
    is refused at once (SQLITE_BUSY) while another writer holds the database.
    """

    if session.get_bind().dialect.name != "sqlite":
        raise RecoveryPreviewConflict("recovery-database-unsupported")
    if session.new or session.dirty or session.deleted or session.in_transaction():
        raise RecoveryPreviewConflict("recovery-transaction-not-clean")
    connection = session.connection()
    connection.exec_driver_sql("BEGIN")
    session.info[_SNAPSHOT_INSPECTIONS] = (session.get_transaction(), {})


def end_recovery_snapshot(session: Session) -> None:
    """Forget the checks a snapshot purge read; its transaction is over."""

    session.info.pop(_SNAPSHOT_INSPECTIONS, None)


def remember_recovery_inspection[Inspection](
    session: Session, key: Hashable, inspect: Callable[[], Inspection]
) -> Inspection:
    """Check once per snapshot purge, and every time anywhere else.

    A check is kept only for the transaction that read it: once that ends, or
    outside a snapshot purge, it is read again.
    """

    held = session.info.get(_SNAPSHOT_INSPECTIONS)
    if held is None or held[0] is None or held[0] is not session.get_transaction():
        return inspect()
    inspections: dict[Hashable, Inspection] = held[1]
    if key not in inspections:
        inspections[key] = inspect()
    return inspections[key]


def _impact(
    snapshot: RecoverySnapshot,
    *,
    kind: RecoveryKind,
    subject_id: str,
    revision: str,
    delete_generated_media: bool,
) -> RecoveryImpactV1:
    if len(snapshot.fingerprint) != 64 or any(
        value not in "0123456789abcdef" for value in snapshot.fingerprint
    ):
        raise RecoveryPreviewConflict("recovery-snapshot-invalid")
    fields = {
        "kind": kind,
        "subject_id": subject_id,
        "revision": revision,
        "impact_sha256": "0" * 64,
        "counts": snapshot.counts,
        "conflicts": snapshot.conflicts,
        "available_actions": snapshot.available_actions,
        "delete_generated_media": delete_generated_media,
    }
    draft = RecoveryImpactV1.model_validate(fields)
    digest = hashlib.sha256(draft.model_dump_json().encode("utf-8")).hexdigest()
    return draft.model_copy(update={"impact_sha256": digest})


def issue_recovery_preview(
    session: Session,
    *,
    kind: RecoveryKind,
    subject_id: str,
    deletion_id: str | None,
    inspect: RecoveryInspector,
    now: datetime,
    delete_generated_media: bool = False,
) -> RecoveryImpactV1:
    """Create an opaque preview; its caller owns the transaction and commit."""

    reserve_recovery_write(session)
    snapshot = inspect(session)
    revision = secrets.token_hex(32)
    impact = _impact(
        snapshot,
        kind=kind,
        subject_id=subject_id,
        revision=revision,
        delete_generated_media=delete_generated_media,
    )
    session.execute(
        delete(RecoveryPreviewRecord).where(RecoveryPreviewRecord.expires_at <= _utc(now))
    )
    session.add(
        RecoveryPreviewRecord(
            revision=revision,
            kind=kind.value,
            subject_id=subject_id,
            deletion_id=deletion_id,
            subject_fingerprint=snapshot.fingerprint,
            impact_sha256=impact.impact_sha256,
            expires_at=_utc(now) + PREVIEW_LIFETIME,
        )
    )
    session.flush()
    return impact


def require_current_recovery_preview(
    session: Session,
    *,
    kind: RecoveryKind,
    subject_id: str,
    deletion_id: str | None,
    command: RecoveryCommandV1,
    inspect: RecoveryInspector,
    now: datetime,
    delete_generated_media: bool = False,
) -> RecoverySnapshot:
    """Re-read the resource under the same reservation used by the later transition."""

    reserve_recovery_write(session)
    preview = session.scalar(
        select(RecoveryPreviewRecord).where(
            RecoveryPreviewRecord.revision == command.expected_revision,
            RecoveryPreviewRecord.kind == kind.value,
            RecoveryPreviewRecord.subject_id == subject_id,
            RecoveryPreviewRecord.deletion_id == deletion_id,
        )
    )
    if (
        preview is None
        or _utc(preview.expires_at) <= _utc(now)
        or not hmac.compare_digest(preview.impact_sha256, command.impact_sha256)
    ):
        raise RecoveryPreviewConflict("recovery-impact-stale")
    snapshot = inspect(session)
    current = _impact(
        snapshot,
        kind=kind,
        subject_id=subject_id,
        revision=preview.revision,
        delete_generated_media=delete_generated_media,
    )
    if not hmac.compare_digest(preview.subject_fingerprint, snapshot.fingerprint) or not (
        hmac.compare_digest(preview.impact_sha256, current.impact_sha256)
    ):
        raise RecoveryPreviewConflict("recovery-impact-stale")
    return snapshot
