"""A deletion preview stays private and cannot outlive its reserved resource state."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NoReturn, TypedDict

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from local_lm.config import Settings
from local_lm.db import Base, create_database_engine, database_is_contended
from local_lm.models import Chat, RecoveryPreviewRecord
from local_lm.recovery_previews import RecoveryInspector, RecoverySnapshot
from local_lm.recovery_v1 import (
    RecoveryAction,
    RecoveryCommandV1,
    RecoveryCountsV1,
    RecoveryImpactV1,
    RecoveryKind,
)

NOW = datetime(2026, 10, 2, tzinfo=UTC)


class PreviewChanges(TypedDict, total=False):
    kind: RecoveryKind
    subject_id: str
    deletion_id: str | None
    delete_generated_media: bool
    now: datetime


@pytest.fixture
def preview_session(tmp_path: Path) -> Iterator[Session]:
    settings = Settings(data_dir=tmp_path / "data", chat_engine="mock", media_engine="mock")
    settings.prepare()
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Chat(id="chat-garden", title="Garden notes"))
        session.commit()
        yield session
    engine.dispose()


def _snapshot(session: Session) -> RecoverySnapshot:
    chat = session.get(Chat, "chat-garden")
    assert chat is not None
    return RecoverySnapshot(
        fingerprint=hashlib.sha256(chat.title.encode()).hexdigest(),
        counts=RecoveryCountsV1(messages=1),
        available_actions=(RecoveryAction.TRASH,),
    )


def _preview(session: Session, *, now: datetime = NOW) -> RecoveryImpactV1:
    from local_lm.recovery_previews import issue_recovery_preview

    return issue_recovery_preview(
        session,
        kind=RecoveryKind.CHAT,
        subject_id="chat-garden",
        deletion_id=None,
        inspect=_snapshot,
        now=now,
    )


def _require(
    session: Session,
    command: RecoveryCommandV1,
    *,
    kind: RecoveryKind = RecoveryKind.CHAT,
    subject_id: str = "chat-garden",
    deletion_id: str | None = None,
    inspect: RecoveryInspector = _snapshot,
    now: datetime = NOW,
    delete_generated_media: bool = False,
) -> RecoverySnapshot:
    from local_lm.recovery_previews import require_current_recovery_preview

    return require_current_recovery_preview(
        session,
        kind=kind,
        subject_id=subject_id,
        deletion_id=deletion_id,
        inspect=inspect,
        now=now,
        command=command,
        delete_generated_media=delete_generated_media,
    )


def _command(impact: RecoveryImpactV1) -> RecoveryCommandV1:
    return RecoveryCommandV1(
        expected_revision=impact.revision,
        impact_sha256=impact.impact_sha256,
        operation_key="trash-garden",
    )


def test_equal_previews_use_different_opaque_revisions(preview_session: Session) -> None:
    first = _preview(preview_session)
    second = _preview(preview_session)
    fingerprint = _snapshot(preview_session).fingerprint

    assert first.counts == second.counts
    assert first.revision != second.revision
    assert first.impact_sha256 != second.impact_sha256
    assert fingerprint not in first.model_dump_json()
    assert fingerprint not in repr(_snapshot(preview_session))
    assert "Garden notes" not in first.model_dump_json()
    assert _require(preview_session, _command(first)).counts.messages == 1
    preview_session.commit()
    with Session(preview_session.get_bind()) as reader:
        record = reader.get(RecoveryPreviewRecord, first.revision)
        assert record is not None and record.subject_fingerprint == fingerprint


def test_a_changed_private_value_refuses_an_unchanged_public_count(
    preview_session: Session,
) -> None:
    from local_lm.recovery_previews import RecoveryPreviewConflict

    impact = _preview(preview_session)
    preview_session.commit()
    preview_session.execute(update(Chat).values(title="Updated garden notes"))
    preview_session.commit()

    with pytest.raises(RecoveryPreviewConflict, match="^recovery-impact-stale$"):
        _require(preview_session, _command(impact))
    assert _snapshot(preview_session).counts == impact.counts
    preview_session.rollback()


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": RecoveryKind.PROJECT},
        {"subject_id": "chat-other"},
        {"deletion_id": "recovery-other"},
        {"delete_generated_media": True},
        {"now": NOW + timedelta(minutes=15)},
    ],
)
def test_a_preview_refuses_a_different_resource_intent_or_expired_deadline(
    preview_session: Session, changes: PreviewChanges
) -> None:
    from local_lm.recovery_previews import RecoveryPreviewConflict

    impact = _preview(preview_session)
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-impact-stale$"):
        _require(preview_session, _command(impact), **changes)
    preview_session.rollback()


def test_a_forged_public_digest_is_refused_before_inspection(preview_session: Session) -> None:
    from local_lm.recovery_previews import RecoveryPreviewConflict

    impact = _preview(preview_session)

    def must_not_read(_session: Session) -> NoReturn:
        pytest.fail("A forged preview must be refused before reading the resource.")

    with pytest.raises(RecoveryPreviewConflict, match="^recovery-impact-stale$"):
        _require(
            preview_session,
            _command(impact).model_copy(update={"impact_sha256": "f" * 64}),
            inspect=must_not_read,
        )
    preview_session.rollback()


def test_preview_creation_does_not_commit_the_callers_transaction(
    preview_session: Session,
) -> None:
    impact = _preview(preview_session)
    preview_session.rollback()
    with Session(preview_session.get_bind()) as reader:
        assert reader.get(RecoveryPreviewRecord, impact.revision) is None


def test_the_writer_is_reserved_before_the_inspector_reads(preview_session: Session) -> None:
    from local_lm.recovery_previews import issue_recovery_preview

    def inspect_under_reservation(session: Session) -> RecoverySnapshot:
        with Session(session.get_bind()) as contender:
            contender.connection().exec_driver_sql("PRAGMA busy_timeout=0")
            with pytest.raises(OperationalError) as failure:
                contender.execute(update(Chat).values(title="Concurrent garden notes"))
            assert database_is_contended(failure.value)
            contender.rollback()
        return _snapshot(session)

    issue_recovery_preview(
        preview_session,
        kind=RecoveryKind.CHAT,
        subject_id="chat-garden",
        deletion_id=None,
        inspect=inspect_under_reservation,
        now=NOW,
    )
    preview_session.rollback()
    with Session(preview_session.get_bind()) as writer:
        writer.execute(update(Chat).values(title="Later garden notes"))
        writer.commit()


def test_an_old_wal_read_snapshot_cannot_be_promoted_to_a_fresh_preview(
    preview_session: Session,
) -> None:
    from local_lm.recovery_previews import RecoveryPreviewConflict

    impact = _preview(preview_session)
    preview_session.commit()
    preview_session.connection().exec_driver_sql("BEGIN")
    assert preview_session.scalar(select(Chat.title)) == "Garden notes"
    with Session(preview_session.get_bind()) as writer:
        writer.execute(update(Chat).values(title="New garden notes"))
        writer.commit()

    with pytest.raises(RecoveryPreviewConflict, match="^recovery-database-busy$"):
        _require(preview_session, _command(impact))
    preview_session.rollback()
    with pytest.raises(RecoveryPreviewConflict, match="^recovery-impact-stale$"):
        _require(preview_session, _command(impact))
    preview_session.rollback()


def test_a_cached_orm_value_is_expired_after_reserving_the_writer(
    preview_session: Session,
) -> None:
    from local_lm.recovery_previews import RecoveryPreviewConflict

    impact = _preview(preview_session)
    preview_session.commit()
    cached = preview_session.get(Chat, "chat-garden")
    assert cached is not None and cached.title == "Garden notes"
    with Session(preview_session.get_bind()) as writer:
        writer.execute(update(Chat).values(title="Revised garden notes"))
        writer.commit()

    with pytest.raises(RecoveryPreviewConflict, match="^recovery-impact-stale$"):
        _require(preview_session, _command(impact))
    assert cached.title == "Revised garden notes"
    preview_session.rollback()


def test_pending_resource_writes_are_not_discarded_or_committed(preview_session: Session) -> None:
    from local_lm.recovery_previews import RecoveryPreviewConflict

    chat = preview_session.get(Chat, "chat-garden")
    assert chat is not None
    chat.title = "Unsaved garden notes"

    with pytest.raises(RecoveryPreviewConflict, match="^recovery-transaction-not-clean$"):
        _preview(preview_session)
    assert chat.title == "Unsaved garden notes" and chat in preview_session.dirty
    preview_session.rollback()
    assert chat.title == "Garden notes"


def test_expired_preview_cleanup_rolls_back_with_its_replacement(
    preview_session: Session,
) -> None:
    first = _preview(preview_session)
    preview_session.commit()
    replacement = _preview(preview_session, now=NOW + timedelta(minutes=15))
    assert preview_session.get(RecoveryPreviewRecord, first.revision) is None
    preview_session.rollback()
    assert preview_session.get(RecoveryPreviewRecord, first.revision) is not None
    assert preview_session.get(RecoveryPreviewRecord, replacement.revision) is None
