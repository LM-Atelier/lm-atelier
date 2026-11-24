"""An expiry purge reads what it deletes without holding SQLite's writer.

Checking a deleted conversation with many renders can take tens of seconds, and
every other writer kept waiting that long fails. The purge reads its checks from
one snapshot first and reserves the writer only afterwards. If anything was
committed in between, or another writer holds the database at that moment, a
quick check is read again at once under the writer and a long one waits for a
later try. A Media Library entry's check takes the writer part-way, where it
proves the entry's references, and is refused there the same way.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy import select
from test_recovery_expiry import DEADLINE, NOW, _trash
from test_recovery_expiry import manual_expiry as manual_expiry
from test_workflow_recovery_expiry import _trash as _trash_workflow

from local_lm import (
    chat_recovery,
    media_recovery,
    project_recovery,
    recovery_maintenance,
    recovery_previews,
    workflow_recovery_graph,
)
from local_lm.chat_recovery import preview_chat_recovery, purge_chat
from local_lm.db import SessionLocal, shorten_busy_timeout
from local_lm.models import ArtifactLibraryEntry, Chat, Project, RecoveryItem, WorkflowFamily
from local_lm.project_recovery import preview_project_trash, trash_project
from local_lm.recovery_maintenance import _Candidate
from local_lm.recovery_previews import RecoveryPreviewConflict, reserve_recovery_write
from local_lm.recovery_v1 import PurgeRecoveryV1, RecoveryCommandV1

#: Where each kind's check is read, below the point a purge remembers it.
_CHECKS: dict[str, tuple[Any, str]] = {
    "chat": (chat_recovery, "_inspected"),
    "media_library_entry": (media_recovery, "_snapshot"),
    "project": (project_recovery, "_snapshot"),
    "workflow_family": (workflow_recovery_graph, "_snapshot"),
}


def _database_path() -> str:
    with SessionLocal() as session:
        path = session.connection().engine.url.database
    assert path
    return path


def _writer_is_free() -> bool:
    """Whether another connection could take the writer this moment, without waiting."""

    connection = sqlite3.connect(_database_path(), timeout=0, isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("ROLLBACK")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        connection.close()


class _HeldWriter:
    """Another connection holding the writer, without committing, until released."""

    def __init__(self) -> None:
        self.connection: sqlite3.Connection | None = None
        self.lock = threading.Lock()

    def take(self) -> bool:
        self.connection = sqlite3.connect(
            _database_path(), timeout=0, isolation_level=None, check_same_thread=False
        )
        self.connection.execute("BEGIN IMMEDIATE")
        return True

    def release(self) -> None:
        with self.lock:
            if self.connection is not None:
                self.connection.execute("ROLLBACK")
                self.connection.close()
                self.connection = None


def _commit_elsewhere() -> None:
    """Commit an unrelated change from another connection, without waiting for the writer."""

    with SessionLocal() as other:
        shorten_busy_timeout(other, 0)
        other.add(Chat(title="Unrelated planting notes"))
        other.commit()


class _ObservedCheck:
    """Count a kind's reads and run something as chosen reads begin."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        kind: str,
        during: dict[int, Callable[[], object]] | None = None,
    ) -> None:
        module, name = _CHECKS[kind]
        real = getattr(module, name)
        self.reads = 0
        self.observed: list[object] = []

        def observed(*args: Any, **kwargs: Any) -> Any:
            self.reads += 1
            action = (during or {}).get(self.reads)
            if action is not None:
                self.observed.append(action())
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, observed)


def _trash_project() -> tuple[str, str]:
    with SessionLocal() as session:
        project = Project(name="Garden plans")
        session.add(project)
        session.commit()
        preview = preview_project_trash(session, project.id, NOW)
        item = trash_project(
            session,
            project.id,
            RecoveryCommandV1(
                expected_revision=preview.revision,
                impact_sha256=preview.impact_sha256,
                operation_key="trash-expiring-project",
            ),
            NOW,
        )
        session.commit()
        return item.deletion_id, project.id


async def _due(app: FastAPI, kind: str) -> tuple[_Candidate, str]:
    if kind in {"chat", "media_library_entry"}:
        deletion_id, subject_id, _artifact_id = await _trash(app, kind)
    elif kind == "project":
        deletion_id, subject_id = _trash_project()
    else:
        deletion_id, subject_id, _definition_id, _revision_id = _trash_workflow()
    (candidate,) = recovery_maintenance._candidates(DEADLINE, None, 10)
    assert candidate.deletion_id == deletion_id
    return candidate, subject_id


def _subject_exists(kind: str, subject_id: str) -> bool:
    model = {
        "chat": Chat,
        "media_library_entry": ArtifactLibraryEntry,
        "project": Project,
        "workflow_family": WorkflowFamily,
    }[kind]
    with SessionLocal() as session:
        return session.get(model, subject_id) is not None


def _state(deletion_id: str) -> str:
    with SessionLocal() as session:
        item = session.get(RecoveryItem, deletion_id)
        assert item is not None
        return item.state


@pytest.mark.parametrize("kind", list(_CHECKS))
@pytest.mark.usefixtures("client")
async def test_an_expiry_purge_reads_its_checks_once_without_holding_the_writer(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    candidate, subject_id = await _due(app, kind)
    check = _ObservedCheck(monkeypatch, kind, {1: _writer_is_free})

    assert recovery_maintenance._purge(candidate, DEADLINE) is None

    assert check.observed == [True]
    assert check.reads == 1
    assert _state(candidate.deletion_id) == "purged"
    assert not _subject_exists(kind, subject_id)


@pytest.mark.parametrize("kind", list(_CHECKS))
@pytest.mark.usefixtures("client")
async def test_a_commit_while_a_long_check_reads_gives_up_that_try(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    candidate, subject_id = await _due(app, kind)
    # Any check counts as long, so none is read again under the writer.
    monkeypatch.setattr(recovery_maintenance, "QUICK_CHECK_SECONDS", 0.0)
    check = _ObservedCheck(monkeypatch, kind, {1: _commit_elsewhere})

    assert recovery_maintenance._purge(candidate, DEADLINE) == "recovery-snapshot-stale"

    assert check.observed == [None]
    assert check.reads == 1
    assert _state(candidate.deletion_id) == "recoverable"
    assert _subject_exists(kind, subject_id)
    # Nothing is committed during the next try, which purges.
    assert recovery_maintenance._purge(candidate, DEADLINE) is None
    assert _state(candidate.deletion_id) == "purged"
    assert not _subject_exists(kind, subject_id)


@pytest.mark.parametrize("kind", list(_CHECKS))
@pytest.mark.usefixtures("client")
async def test_a_quick_check_a_commit_refused_is_read_again_under_the_writer(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    candidate, subject_id = await _due(app, kind)
    # Any check counts as quick, however long the test machine takes.
    monkeypatch.setattr(recovery_maintenance, "QUICK_CHECK_SECONDS", 3600.0)
    check = _ObservedCheck(monkeypatch, kind, {1: _commit_elsewhere, 2: _writer_is_free})

    assert recovery_maintenance._purge(candidate, DEADLINE) is None

    assert check.observed == [None, False]
    assert check.reads == 2
    assert _state(candidate.deletion_id) == "purged"
    assert not _subject_exists(kind, subject_id)


@pytest.mark.parametrize("kind", list(_CHECKS))
@pytest.mark.usefixtures("client")
async def test_a_quick_check_that_meets_another_writer_waits_for_it_under_the_writer(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """SQLite does not wait for another writer once a transaction has read.

    It refuses the writer at once, so the try under the writer is the one that waits.
    """

    candidate, subject_id = await _due(app, kind)
    monkeypatch.setattr(recovery_maintenance, "QUICK_CHECK_SECONDS", 3600.0)
    held = _HeldWriter()
    tries: list[bool] = []
    real_once = recovery_maintenance._purge_once

    def once(candidate: _Candidate, now: datetime, *, reserve_first: bool) -> str | None:
        tries.append(reserve_first)
        if reserve_first:
            # The other writer finishes without committing while this try waits.
            threading.Timer(0.2, held.release).start()
        return real_once(candidate, now, reserve_first=reserve_first)

    monkeypatch.setattr(recovery_maintenance, "_purge_once", once)
    check = _ObservedCheck(monkeypatch, kind, {1: held.take, 2: _writer_is_free})
    try:
        assert recovery_maintenance._purge(candidate, DEADLINE) is None
    finally:
        held.release()

    assert tries == [False, True]
    assert check.observed == [True, False]
    assert check.reads == 2
    assert _state(candidate.deletion_id) == "purged"
    assert not _subject_exists(kind, subject_id)


@pytest.mark.usefixtures("client")
async def test_a_purge_asked_for_by_hand_still_checks_under_the_writer(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, subject_id = await _due(app, "chat")
    with SessionLocal() as session:
        preview = preview_chat_recovery(session, candidate.deletion_id, DEADLINE)
        session.commit()
    check = _ObservedCheck(monkeypatch, "chat", {1: _writer_is_free})
    with SessionLocal() as session:
        purge_chat(
            session,
            candidate.deletion_id,
            PurgeRecoveryV1(
                expected_revision=preview.revision,
                impact_sha256=preview.impact_sha256,
                operation_key="purge-by-hand",
                acknowledgement="permanently-delete",
            ),
            DEADLINE,
        )
        session.commit()
    assert check.observed == [False]
    assert not _subject_exists("chat", subject_id)


@pytest.mark.usefixtures("client")
async def test_a_check_is_kept_only_for_the_snapshot_purge_that_read_it() -> None:
    reads: list[str] = []

    def inspect() -> str:
        reads.append("read")
        return f"check {len(reads)}"

    with SessionLocal() as session:
        assert recovery_previews.remember_recovery_inspection(session, "key", inspect) == "check 1"
        assert recovery_previews.remember_recovery_inspection(session, "key", inspect) == "check 2"
        session.rollback()
        recovery_previews.begin_recovery_snapshot(session)
        assert recovery_previews.remember_recovery_inspection(session, "key", inspect) == "check 3"
        assert recovery_previews.remember_recovery_inspection(session, "key", inspect) == "check 3"
        assert (
            recovery_previews.remember_recovery_inspection(session, "other", inspect) == "check 4"
        )
        session.rollback()
        # A later transaction of the same session reads again.
        assert recovery_previews.remember_recovery_inspection(session, "key", inspect) == "check 5"
        session.rollback()
        recovery_previews.begin_recovery_snapshot(session)
        assert recovery_previews.remember_recovery_inspection(session, "key", inspect) == "check 6"
        recovery_previews.end_recovery_snapshot(session)
        assert recovery_previews.remember_recovery_inspection(session, "key", inspect) == "check 7"
        session.rollback()


@pytest.mark.usefixtures("client")
async def test_a_snapshot_purge_starts_only_in_a_clean_session() -> None:
    with SessionLocal() as session:
        session.add(Chat(title="Unsaved garden notes"))
        with pytest.raises(RecoveryPreviewConflict) as pending:
            recovery_previews.begin_recovery_snapshot(session)
        assert pending.value.code == "recovery-transaction-not-clean"
        session.rollback()
        session.scalar(select(Chat.id).limit(1))
        with pytest.raises(RecoveryPreviewConflict) as started:
            recovery_previews.begin_recovery_snapshot(session)
        assert started.value.code == "recovery-transaction-not-clean"
        session.rollback()


@pytest.mark.usefixtures("client")
async def test_only_a_snapshot_purge_calls_a_stale_snapshot_by_that_name() -> None:
    """Elsewhere an old snapshot keeps the code a busy database always had."""

    for snapshot_purge, expected in (
        (True, "recovery-snapshot-stale"),
        (False, "recovery-database-busy"),
    ):
        with SessionLocal() as session:
            if snapshot_purge:
                recovery_previews.begin_recovery_snapshot(session)
            else:
                session.connection().exec_driver_sql("BEGIN")
            session.scalar(select(Chat.id).limit(1))
            _commit_elsewhere()
            with pytest.raises(RecoveryPreviewConflict) as refused:
                reserve_recovery_write(session)
            assert refused.value.code == expected
            session.rollback()
            recovery_previews.end_recovery_snapshot(session)
