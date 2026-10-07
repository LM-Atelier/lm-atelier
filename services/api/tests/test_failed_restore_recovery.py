"""A restore someone asked for that cannot be applied leaves the data, and LM Atelier starts."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from httpx2 import ASGITransport, AsyncClient

from local_lm.backups import BackupManager
from local_lm.config import Settings
from local_lm.main import create_app

KNOWN_REVISION = "266b3b9df743"
BACKUP_NAME = "local-lm-20260110T120000Z-00000001.sqlite3"


def _write_database(path: Path, marker: str, revision: str = KNOWN_REVISION) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE alembic_version (version_num TEXT PRIMARY KEY);
            CREATE TABLE artifacts (
                sha256 TEXT PRIMARY KEY,
                size_bytes INTEGER NOT NULL,
                relative_path TEXT NOT NULL,
                kind TEXT NOT NULL
            );
            CREATE TABLE chats (id TEXT PRIMARY KEY);
            CREATE TABLE message_parts (id TEXT PRIMARY KEY);
            CREATE TABLE messages (id TEXT PRIMARY KEY);
            CREATE TABLE projects (marker TEXT NOT NULL);
            """
        )
        connection.execute("INSERT INTO alembic_version VALUES (?)", (revision,))
        connection.execute("INSERT INTO projects VALUES (?)", (marker,))
        connection.commit()


def _prepared(tmp_path: Path) -> tuple[Settings, BackupManager, Path, Path]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    live = settings.state_dir / "local-lm.sqlite3"
    _write_database(live, "live")
    backup = settings.backup_dir / BACKUP_NAME
    _write_database(backup, "backup")
    return settings, BackupManager(settings), live, backup


def _marker(settings: Settings) -> Path:
    return settings.state_dir / "restore-on-next-start.json"


def test_a_requested_restore_whose_backup_went_missing_keeps_the_data_and_says_so(
    tmp_path: Path,
) -> None:
    settings, manager, live, backup = _prepared(tmp_path)
    manager.request_restore(BACKUP_NAME, requested=True)
    before = live.read_bytes()
    backup.unlink()

    assert manager.apply_pending_restore() is False

    assert live.read_bytes() == before
    assert not _marker(settings).exists()
    state = manager.restore_state()
    assert (state.state, state.reason, state.backup) == ("failed", "backup-missing", BACKUP_NAME)
    assert state.failed_at is not None
    # The next start has nothing left to try and nothing to fail on.
    assert manager.apply_pending_restore() is False


def test_a_requested_restore_whose_backup_went_bad_keeps_the_data(tmp_path: Path) -> None:
    settings, manager, live, backup = _prepared(tmp_path)
    manager.request_restore(BACKUP_NAME, requested=True)
    before = live.read_bytes()
    backup.write_bytes(b"neutral bytes that are not a database")

    assert manager.apply_pending_restore() is False

    assert live.read_bytes() == before
    assert manager.restore_state().reason == "backup-invalid"


def test_a_requested_restore_of_a_newer_backup_is_named_for_what_it_is(tmp_path: Path) -> None:
    settings, manager, live, backup = _prepared(tmp_path)
    manager.request_restore(BACKUP_NAME, requested=True)
    with closing(sqlite3.connect(backup)) as connection:
        connection.execute("UPDATE alembic_version SET version_num = 'from_a_newer_build'")
        connection.commit()

    assert manager.apply_pending_restore() is False

    assert manager.restore_state().reason == "backup-newer"
    assert (settings.state_dir / "local-lm.sqlite3").read_bytes() == live.read_bytes()


def test_a_requested_restore_that_cannot_replace_the_data_keeps_it_and_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, manager, live, _backup = _prepared(tmp_path)
    manager.request_restore(BACKUP_NAME, requested=True)
    log = live.with_name(live.name + "-wal")
    log.write_bytes(b"neutral uncheckpointed frames")
    before = live.read_bytes()
    real_replace = os.replace

    def refusing(source: str | Path, target: str | Path) -> None:
        if Path(target) == live:
            raise PermissionError("the database file is in use")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", refusing)

    assert manager.apply_pending_restore() is False

    assert live.read_bytes() == before
    assert log.read_bytes() == b"neutral uncheckpointed frames"
    assert manager.restore_state().reason == "restore-failed"


def _projects(database: Path) -> list[str]:
    with closing(sqlite3.connect(database)) as connection:
        return [row[0] for row in connection.execute("SELECT marker FROM projects")]


def test_a_requested_restore_whose_log_cannot_be_put_back_stops_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, manager, live, _backup = _prepared(tmp_path)
    manager.request_restore(BACKUP_NAME, requested=True)
    log = live.with_name(live.name + "-wal")
    log.write_bytes(b"neutral uncheckpointed frames")
    real_replace = os.replace

    def refusing(source: str | Path, target: str | Path) -> None:
        # The database cannot be replaced, and its log cannot go back.
        if Path(target) in (live, log):
            raise PermissionError("the file is in use")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", refusing)

    with pytest.raises(OSError):
        manager.apply_pending_restore()

    # Starting now would open the database without its log, so nothing says
    # the restore was set aside: it is still pending, and the log is kept.
    assert _marker(settings).exists()
    assert not (settings.state_dir / "restore-failed.json").exists()
    assert log.with_name(log.name + ".restore-aside").read_bytes() == (
        b"neutral uncheckpointed frames"
    )


def test_a_restore_that_was_applied_is_not_called_failed_when_a_set_aside_log_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, manager, live, _backup = _prepared(tmp_path)
    manager.request_restore(BACKUP_NAME, requested=True)
    live.with_name(live.name + "-wal").write_bytes(b"neutral uncheckpointed frames")
    real_unlink = Path.unlink

    def stuck(self: Path, missing_ok: bool = False) -> None:
        if self.name.endswith(".restore-aside"):
            raise PermissionError("the file is in use")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", stuck)

    assert manager.apply_pending_restore() is True

    assert _projects(live) == ["backup"]
    assert not _marker(settings).exists()
    assert manager.restore_state().state == "none"


@pytest.mark.parametrize("kind", ["hard", "symbolic"])
def test_a_failure_record_name_that_links_elsewhere_is_not_written_through(
    tmp_path: Path, kind: str
) -> None:
    settings, manager, _live, backup = _prepared(tmp_path)
    manager.request_restore(BACKUP_NAME, requested=True)
    backup.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("neutral outside content", encoding="utf-8")
    record = settings.state_dir / "restore-failed.json"
    try:
        if kind == "hard":
            os.link(outside, record)
        else:
            record.symlink_to(outside)
    except OSError:
        pytest.skip(f"this platform cannot make a {kind} link here")

    assert manager.apply_pending_restore() is False

    assert outside.read_text(encoding="utf-8") == "neutral outside content"
    assert not _marker(settings).exists()


def test_a_failure_record_that_cannot_be_cleared_schedules_nothing(tmp_path: Path) -> None:
    settings, manager, _live, _backup = _prepared(tmp_path)
    (settings.state_dir / "restore-failed.json").mkdir()

    with pytest.raises(OSError):
        manager.request_restore(BACKUP_NAME, requested=True)

    assert not _marker(settings).exists()


@pytest.mark.parametrize(
    "payload", [{"backup": BACKUP_NAME}, {"backup": BACKUP_NAME, "requested": False}]
)
def test_a_restore_nobody_asked_for_still_stops_the_start_when_it_fails(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    settings, manager, live, backup = _prepared(tmp_path)
    # What an upgrade arms before it begins, or what an earlier build wrote.
    _marker(settings).write_text(json.dumps(payload), encoding="utf-8")
    backup.unlink()

    with pytest.raises(FileNotFoundError):
        manager.apply_pending_restore()

    assert _marker(settings).exists()
    assert manager.restore_state().state == "pending"


def test_asking_again_replaces_a_failure_and_a_failure_can_be_dismissed(tmp_path: Path) -> None:
    settings, manager, _live, backup = _prepared(tmp_path)
    manager.request_restore(BACKUP_NAME, requested=True)
    backup.unlink()
    manager.apply_pending_restore()
    assert manager.restore_state().state == "failed"

    _write_database(backup, "backup again")
    manager.request_restore(BACKUP_NAME, requested=True)
    assert (manager.restore_state().state, manager.restore_state().backup) == (
        "pending",
        BACKUP_NAME,
    )
    assert manager.apply_pending_restore() is True
    assert manager.restore_state().state == "none"

    manager.request_restore(BACKUP_NAME, requested=True)
    backup.unlink()
    manager.apply_pending_restore()
    assert manager.dismiss_failed_restore() is True
    assert manager.restore_state().state == "none"
    assert manager.dismiss_failed_restore() is False


def test_a_damaged_failure_record_still_reads_as_a_failure(tmp_path: Path) -> None:
    settings, manager, _live, _backup = _prepared(tmp_path)
    (settings.state_dir / "restore-failed.json").write_text("not json", encoding="utf-8")

    state = manager.restore_state()

    assert (state.state, state.reason, state.backup) == ("failed", "restore-failed", None)


async def test_the_application_starts_after_a_requested_restore_fails_and_reports_it(
    settings: Settings,
) -> None:
    settings.backup_dir.mkdir(parents=True, exist_ok=True)
    # A restore asked for through the API, whose backup is gone by the next start.
    _marker(settings).write_text(
        json.dumps({"backup": BACKUP_NAME, "requested": True}), encoding="utf-8"
    )

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            session = await client.post("/api/session")
            client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]

            state = (await client.get("/api/backups/restore-state")).json()
            assert (state["state"], state["reason"], state["backup"]) == (
                "failed",
                "backup-missing",
                BACKUP_NAME,
            )
            assert (await client.get("/api/chats")).status_code == 200

            assert (await client.post("/api/backups/restore-state/dismiss")).status_code == 204
            assert (await client.get("/api/backups/restore-state")).json() == {
                "state": "none",
                "backup": None,
                "reason": None,
                "failed_at": None,
                "encrypted": False,
            }


async def test_a_restore_asked_for_through_the_api_is_marked_as_asked_for(
    client: AsyncClient, settings: Settings
) -> None:
    created = await client.post("/api/backups")
    assert created.status_code == 201, created.text

    scheduled = await client.post(f"/api/backups/{created.json()['name']}/restore")

    assert scheduled.status_code == 200, scheduled.text
    assert json.loads(_marker(settings).read_text(encoding="utf-8")) == {
        "backup": created.json()["name"],
        "requested": True,
    }
    state = (await client.get("/api/backups/restore-state")).json()
    assert (state["state"], state["backup"]) == ("pending", created.json()["name"])
