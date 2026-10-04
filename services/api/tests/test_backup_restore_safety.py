"""A restore never puts data this build cannot open in place, and never loses the live log."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from httpx2 import AsyncClient

from local_lm import backups as backups_module
from local_lm.backups import BackupManager
from local_lm.config import Settings
from local_lm.database_migrations import DatabaseVersionError, upgrade_database

KNOWN_REVISION = "266b3b9df743"
NEWER_REVISION = "from_a_newer_build"
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


def _marker(path: Path) -> str:
    with closing(sqlite3.connect(path)) as connection:
        return str(connection.execute("SELECT marker FROM projects").fetchone()[0])


def _prepared(tmp_path: Path) -> tuple[Settings, BackupManager, Path]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    live = settings.state_dir / "local-lm.sqlite3"
    _write_database(live, "live")
    return settings, BackupManager(settings), live


def _arm(settings: Settings, name: str = BACKUP_NAME) -> Path:
    marker = settings.state_dir / "restore-on-next-start.json"
    marker.write_text(json.dumps({"backup": name}), encoding="utf-8")
    return marker


def _set_revision(path: Path, revision: str) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("UPDATE alembic_version SET version_num = ?", (revision,))
        connection.commit()


def test_a_backup_from_a_newer_build_is_refused_before_a_restore_is_scheduled(
    tmp_path: Path,
) -> None:
    settings, manager, _live = _prepared(tmp_path)
    _write_database(settings.backup_dir / BACKUP_NAME, "newer", NEWER_REVISION)

    with pytest.raises(ValueError, match="newer version of LM Atelier"):
        manager.request_restore(BACKUP_NAME)

    assert not (settings.state_dir / "restore-on-next-start.json").exists()


async def test_the_restore_route_refuses_a_backup_from_a_newer_build(
    client: AsyncClient, settings: Settings
) -> None:
    created = await client.post("/api/backups")
    assert created.status_code == 201, created.text
    _set_revision(settings.backup_dir / created.json()["name"], NEWER_REVISION)

    refused = await client.post(f"/api/backups/{created.json()['name']}/restore")

    assert (refused.status_code, refused.json()["code"]) == (422, "backup-invalid")
    assert "newer version of LM Atelier" in refused.json()["detail"]
    assert not (settings.state_dir / "restore-on-next-start.json").exists()


def test_a_scheduled_restore_of_a_newer_backup_leaves_the_live_database_in_place(
    tmp_path: Path,
) -> None:
    settings, manager, live = _prepared(tmp_path)
    _write_database(settings.backup_dir / BACKUP_NAME, "newer", NEWER_REVISION)
    # Armed as an earlier build, which did not check, would have armed it.
    _arm(settings)
    before = live.read_bytes()

    with pytest.raises(ValueError, match="newer version of LM Atelier"):
        manager.apply_pending_restore()

    assert live.read_bytes() == before
    assert _marker(live) == "live"


def test_a_backup_recording_one_unknown_revision_among_known_ones_is_refused(
    tmp_path: Path,
) -> None:
    settings, manager, live = _prepared(tmp_path)
    backup = settings.backup_dir / BACKUP_NAME
    _write_database(backup, "mixed")
    # Data on two migration branches records one revision for each.
    with closing(sqlite3.connect(backup)) as connection:
        connection.execute("INSERT INTO alembic_version VALUES (?)", (NEWER_REVISION,))
        connection.commit()
    before = live.read_bytes()

    with pytest.raises(ValueError, match="newer version of LM Atelier"):
        manager.request_restore(BACKUP_NAME)
    assert not (settings.state_dir / "restore-on-next-start.json").exists()

    _arm(settings)
    with pytest.raises(ValueError, match="newer version of LM Atelier"):
        manager.apply_pending_restore()
    assert live.read_bytes() == before


def test_a_replace_that_fails_keeps_the_live_write_ahead_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, manager, live = _prepared(tmp_path)
    _write_database(settings.backup_dir / BACKUP_NAME, "backup")
    manager.request_restore(BACKUP_NAME)
    # What SQLite leaves beside a database that was not closed cleanly: the
    # log may hold writes that never reached the database file.
    log = {"-wal": b"neutral uncheckpointed frames", "-shm": b"neutral shared memory"}
    for suffix, content in log.items():
        live.with_name(live.name + suffix).write_bytes(content)
    real_replace = os.replace

    def refusing(source: str | Path, target: str | Path) -> None:
        if Path(target) == live:
            raise PermissionError("the database file is in use")
        real_replace(source, target)

    monkeypatch.setattr(backups_module.os, "replace", refusing)

    with pytest.raises(PermissionError):
        manager.apply_pending_restore()

    for suffix, content in log.items():
        assert live.with_name(live.name + suffix).read_bytes() == content
    assert not list(live.parent.glob("*.restore-aside"))
    assert _marker(live) == "live"
    assert (settings.state_dir / "restore-on-next-start.json").exists()


def test_a_restore_leaves_nothing_of_the_replaced_log_behind(tmp_path: Path) -> None:
    settings, manager, live = _prepared(tmp_path)
    _write_database(settings.backup_dir / BACKUP_NAME, "backup")
    manager.request_restore(BACKUP_NAME)
    for suffix in ("-wal", "-shm"):
        live.with_name(live.name + suffix).write_bytes(b"neutral stale log")

    assert manager.apply_pending_restore() is True

    assert _marker(live) == "backup"
    assert not live.with_name(live.name + "-wal").exists()
    assert not live.with_name(live.name + "-shm").exists()
    assert not list(live.parent.glob("*.restore-aside"))
    assert not (settings.state_dir / "restore-on-next-start.json").exists()


def test_data_from_a_newer_build_is_not_copied_before_the_upgrade_refuses_it(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "newer-build")
    settings.prepare()
    upgrade_database(settings)
    database = settings.state_dir / "local-lm.sqlite3"
    _set_revision(database, NEWER_REVISION)
    before = sorted(path.name for path in settings.backup_dir.iterdir())

    with pytest.raises(DatabaseVersionError):
        upgrade_database(settings)

    # A copy this build could never restore is not worth the disk it takes.
    assert sorted(path.name for path in settings.backup_dir.iterdir()) == before
    assert not (settings.state_dir / "restore-on-next-start.json").exists()
