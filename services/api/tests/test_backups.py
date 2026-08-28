from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.backups import BackupManager
from local_lm.config import Settings
from local_lm.main import (
    create_app,
    ensure_automatic_recovery_backup,
    maintain_automatic_recovery_backups,
)
from local_lm.runtime_provisioning import RuntimeProvisioner
from local_lm.schemas import BackupInfo


class VerificationConnection:
    def __init__(self) -> None:
        self.closed = False
        self.query = ""

    def execute(self, query: str) -> VerificationConnection:
        self.query = query
        return self

    def fetchone(self) -> tuple[str] | None:
        if "integrity_check" in self.query:
            return ("ok",)
        if "foreign_key_check" in self.query:
            return None
        if "alembic_version" in self.query:
            return ("266b3b9df743",)
        raise AssertionError(self.query)

    def fetchall(self) -> list[tuple[str]]:
        if "sqlite_master" not in self.query:
            raise AssertionError(self.query)
        return [
            ("alembic_version",),
            ("artifacts",),
            ("chats",),
            ("message_parts",),
            ("messages",),
            ("projects",),
        ]

    def close(self) -> None:
        self.closed = True


def test_backup_verification_closes_sqlite_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    connection = VerificationConnection()
    monkeypatch.setattr("local_lm.backups.sqlite3.connect", lambda *_args, **_kwargs: connection)

    BackupManager._verify_path(tmp_path / "backup.sqlite3")

    assert connection.closed is True


def test_backup_retention_keeps_daily_and_older_weekly_snapshots(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", backup_daily_count=2, backup_weekly_count=1)
    settings.prepare()
    names = [
        "local-lm-20260110T120000Z-00000001.sqlite3",
        "local-lm-20260109T120000Z-00000002.sqlite3",
        "local-lm-20260103T120000Z-00000003.sqlite3",
        "local-lm-20260102T120000Z-00000004.sqlite3",
        "local-lm-20251220T120000Z-00000005.sqlite3",
    ]
    for name in names:
        path = settings.backup_dir / name
        path.write_bytes(b"backup")
        path.with_name(f"{name}.media.zip").write_bytes(b"media")

    removed = BackupManager(settings).prune()

    assert removed == 2
    remaining = {path.name for path in settings.backup_dir.glob("*.sqlite3")}
    assert remaining == set(names[:3])
    assert not (settings.backup_dir / f"{names[3]}.media.zip").exists()


def _write_test_database(
    path: Path,
    marker: str,
    *,
    artifact: tuple[str, int, str] | None = None,
) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE alembic_version (version_num TEXT PRIMARY KEY);
            INSERT INTO alembic_version VALUES ('266b3b9df743');
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
        connection.execute("INSERT INTO projects VALUES (?)", (marker,))
        if artifact:
            connection.execute("INSERT INTO artifacts VALUES (?, ?, ?, 'image')", artifact)
        connection.commit()


def _database_marker(path: Path) -> str:
    with closing(sqlite3.connect(path)) as connection:
        return str(connection.execute("SELECT marker FROM projects").fetchone()[0])


def test_daily_backup_is_verified_metadata_only_and_idempotent(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    _write_test_database(settings.state_dir / "local-lm.sqlite3", "current")
    manager = BackupManager(settings)
    today = datetime(2026, 7, 25, 1, 2, 3, tzinfo=UTC)

    first = manager.ensure_daily_backup(now=today)
    repeated = manager.ensure_daily_backup(now=today + timedelta(hours=20))

    assert repeated.name == first.name
    assert first.verified is True
    assert repeated.verified is True
    assert first.media_included is False
    assert not manager._media_path(settings.backup_dir / first.name).exists()
    assert len(list(settings.backup_dir.glob("*.sqlite3"))) == 1
    assert manager.verify(first.name).verified is True


def test_daily_backup_creates_a_new_snapshot_after_utc_day_changes(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    _write_test_database(settings.state_dir / "local-lm.sqlite3", "current")
    manager = BackupManager(settings)

    first = manager.ensure_daily_backup(
        now=datetime(2026, 7, 25, 23, 59, 59, tzinfo=UTC),
    )
    second = manager.ensure_daily_backup(
        now=datetime(2026, 7, 26, 0, 0, 1, tzinfo=UTC),
    )

    assert second.name != first.name
    assert second.verified is True
    assert len(list(settings.backup_dir.glob("*.sqlite3"))) == 2


def test_simultaneous_daily_checks_create_only_one_snapshot(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    _write_test_database(settings.state_dir / "local-lm.sqlite3", "current")
    manager = BackupManager(settings)
    now = datetime(2026, 7, 25, 12, tzinfo=UTC)
    callers = 8
    barrier = threading.Barrier(callers)

    def check() -> str:
        barrier.wait(timeout=2)
        return manager.ensure_daily_backup(now=now).name

    with ThreadPoolExecutor(max_workers=callers) as executor:
        names = list(executor.map(lambda _index: check(), range(callers)))

    assert len(set(names)) == 1
    assert len(list(settings.backup_dir.glob("*.sqlite3"))) == 1


async def test_app_startup_creates_and_stops_automatic_backup_maintenance(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        maintenance = next(
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "maintain-automatic-recovery-backups"
        )

        # The backup is no longer awaited before the lifespan yields, so it
        # exists shortly after startup rather than at the moment of entry.
        async def created() -> list[BackupInfo]:
            while True:
                existing = BackupManager(settings).list()
                if existing:
                    return existing
                await asyncio.sleep(0.05)

        backups = await asyncio.wait_for(created(), timeout=30)
        assert len(backups) == 1
        assert backups[0].media_included is False
        assert BackupManager(settings).verify(backups[0].name).verified is True
        assert maintenance.done() is False

    assert maintenance.cancelled() is True


async def test_app_serves_while_daily_backup_verification_is_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_ensure_daily_backup(self: BackupManager) -> None:
        entered.set()
        release.wait(timeout=30)

    monkeypatch.setattr(BackupManager, "ensure_daily_backup", blocking_ensure_daily_backup)
    settings = Settings(
        data_dir=tmp_path / "data",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    app = create_app(settings)

    lifespan = app.router.lifespan_context(app)

    # Startup must not wait on the verification. Bounded, because the failure
    # this guards against is startup blocking for as long as the check takes,
    # and an unbounded await would hang here rather than fail.
    await asyncio.wait_for(lifespan.__aenter__(), timeout=10)

    # Startup completed - the port would be open - while the verification is
    # still running in its worker thread. This is the whole point of the
    # change: an integrity check, a foreign-key check and a whole-file digest
    # once held the port closed for as long as they took.
    assert await asyncio.to_thread(entered.wait, 30) is True
    maintenance = next(
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "maintain-automatic-recovery-backups"
    )
    assert maintenance.done() is False

    # Shutdown must not cut a filesystem/SQLite operation midway: it waits for
    # the shielded operation to finish rather than returning while it runs.
    shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
    await asyncio.sleep(0.2)
    assert shutdown.done() is False

    release.set()
    await asyncio.wait_for(shutdown, timeout=30)
    assert maintenance.done() is True


async def test_app_serves_while_managed_runtime_verification_is_pending(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    started = asyncio.Event()
    release = asyncio.Event()

    def start_slow_restore(self: RuntimeProvisioner) -> asyncio.Task[None]:
        async def restore() -> None:
            started.set()
            await release.wait()

        task = asyncio.create_task(restore(), name="verify-managed-runtimes")
        self._restore_task = task
        return task

    monkeypatch.setattr(RuntimeProvisioner, "start_restore", start_slow_restore)
    settings = Settings(
        data_dir=tmp_path / "data",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(started.wait(), timeout=1)
        restore = app.state.services.runtimes._restore_task
        assert restore is not None
        assert restore.done() is False
        release.set()

    assert restore.done() is True


async def test_app_startup_reconciles_orphaned_model_quarantine(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    settings.prepare()
    quarantine = settings.model_dir / ".delete-pending" / f"delete_{'a' * 32}"
    payload = quarantine / "payload"
    payload.mkdir(parents=True)
    (quarantine / ".model-id").write_text("model_already_deleted", encoding="utf-8")
    (payload / "model.gguf").write_bytes(b"orphaned")
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        assert not quarantine.exists()

    assert not (settings.model_dir / ".delete-pending").exists()


async def test_backup_api_offloads_every_filesystem_operation(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = app.state.services.backups
    original_list = manager.list
    observed_threads: dict[str, int] = {}
    main_thread = threading.get_ident()

    def record(name: str, result):  # type: ignore[no-untyped-def]
        def operation(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            observed_threads[name] = threading.get_ident()
            return result

        return operation

    existing = original_list()[0]
    monkeypatch.setattr(manager, "list", record("list", [existing]))
    monkeypatch.setattr(manager, "create", record("create", existing))
    monkeypatch.setattr(
        manager,
        "verify",
        record("verify", existing.model_copy(update={"verified": True})),
    )
    monkeypatch.setattr(
        manager,
        "request_restore",
        record("restore", existing.model_copy(update={"restore_pending": True})),
    )
    monkeypatch.setattr(manager, "delete", record("delete", None))

    assert (await client.get("/api/backups")).status_code == 200
    assert (await client.post("/api/backups")).status_code == 201
    assert (await client.post(f"/api/backups/{existing.name}/verify")).status_code == 200
    assert (await client.post(f"/api/backups/{existing.name}/restore")).status_code == 200
    assert (await client.delete(f"/api/backups/{existing.name}")).status_code == 204

    assert set(observed_threads) == {"list", "create", "verify", "restore", "delete"}
    assert all(thread_id != main_thread for thread_id in observed_threads.values())


async def test_backup_cadence_checks_long_running_sessions() -> None:
    checked = threading.Event()

    class CountingBackups:
        def ensure_daily_backup(self) -> None:
            checked.set()

    maintenance = asyncio.create_task(
        maintain_automatic_recovery_backups(  # type: ignore[arg-type]
            CountingBackups(),
            interval_seconds=0.01,
        )
    )
    try:
        observed = await asyncio.wait_for(asyncio.to_thread(checked.wait, 1), timeout=2)
        assert observed is True
    finally:
        maintenance.cancel()
        with pytest.raises(asyncio.CancelledError):
            await maintenance


async def test_cancelling_backup_check_waits_for_filesystem_transaction() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingBackups:
        def ensure_daily_backup(self) -> None:
            started.set()
            assert release.wait(timeout=2)

    check = asyncio.create_task(
        ensure_automatic_recovery_backup(BlockingBackups()),  # type: ignore[arg-type]
    )
    observed = await asyncio.wait_for(asyncio.to_thread(started.wait, 1), timeout=2)
    assert observed is True
    check.cancel()
    await asyncio.sleep(0)
    assert check.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await check


async def test_automatic_backup_failure_is_logged_without_crashing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingBackups:
        def ensure_daily_backup(self) -> None:
            raise OSError("simulated backup failure")

    with caplog.at_level("ERROR", logger="local_lm"):
        await ensure_automatic_recovery_backup(FailingBackups())  # type: ignore[arg-type]

    assert "Could not maintain the automatic LM Atelier recovery backup" in caplog.text


def test_failed_media_restore_leaves_current_database_and_marker_intact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    manager = BackupManager(settings)
    destination = settings.state_dir / "local-lm.sqlite3"
    _write_test_database(destination, "current")
    backup_name = "local-lm-20260110T120000Z-00000001.sqlite3"
    backup = settings.backup_dir / backup_name
    _write_test_database(backup, "restore")
    manager._create_media_archive(backup)
    marker = settings.state_dir / "restore-on-next-start.json"
    marker.write_text(json.dumps({"backup": backup_name}), encoding="utf-8")

    def fail_restore(_path: Path, _database: Path) -> None:
        raise OSError("simulated media write failure")

    monkeypatch.setattr(manager, "_restore_media_archive", fail_restore)

    with pytest.raises(OSError, match="simulated media write failure"):
        manager.apply_pending_restore()

    assert _database_marker(destination) == "current"
    assert marker.is_file()


def test_successful_restore_replaces_database_and_media_then_clears_marker(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    manager = BackupManager(settings)
    destination = settings.state_dir / "local-lm.sqlite3"
    _write_test_database(destination, "current")
    content = b"restored media"
    digest = hashlib.sha256(content).hexdigest()
    relative = f"{digest[:2]}/{digest[2:4]}/{digest}"
    artifact_path = settings.artifact_dir / relative
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(content)
    backup_name = "local-lm-20260110T120000Z-00000001.sqlite3"
    backup = settings.backup_dir / backup_name
    _write_test_database(
        backup,
        "restore",
        artifact=(digest, len(content), relative),
    )
    manager._create_media_archive(backup)
    artifact_path.unlink()
    marker = settings.state_dir / "restore-on-next-start.json"
    marker.write_text(json.dumps({"backup": backup_name}), encoding="utf-8")
    Path(f"{destination}-wal").write_bytes(b"stale wal")
    Path(f"{destination}-shm").write_bytes(b"stale shm")

    assert manager.apply_pending_restore() is True

    assert _database_marker(destination) == "restore"
    assert artifact_path.read_bytes() == content
    assert not marker.exists()
    assert not Path(f"{destination}-wal").exists()
    assert not Path(f"{destination}-shm").exists()


def test_pending_restore_backup_is_not_pruned_or_deleted(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        backup_daily_count=1,
        backup_weekly_count=0,
    )
    settings.prepare()
    pending_name = "local-lm-20260101T120000Z-00000001.sqlite3"
    newest_name = "local-lm-20260110T120000Z-00000002.sqlite3"
    for name in (pending_name, newest_name):
        (settings.backup_dir / name).write_bytes(b"snapshot")
    (settings.state_dir / "restore-on-next-start.json").write_text(
        json.dumps({"backup": pending_name}),
        encoding="utf-8",
    )
    manager = BackupManager(settings)

    assert manager.prune() == 0
    assert {item.name for item in manager.list()} == {pending_name, newest_name}
    assert next(item for item in manager.list() if item.name == pending_name).restore_pending
    with pytest.raises(ValueError, match="pending restore"):
        manager.delete(pending_name)


def test_backup_delete_restores_media_if_database_staging_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    name = "local-lm-20260110T120000Z-00000001.sqlite3"
    database = settings.backup_dir / name
    media = database.with_name(f"{name}.media.zip")
    database.write_bytes(b"database")
    media.write_bytes(b"media")
    real_replace = os.replace
    calls = 0

    def fail_database_stage(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("simulated locked database")
        real_replace(source, destination)

    monkeypatch.setattr("local_lm.backups.os.replace", fail_database_stage)

    with pytest.raises(PermissionError, match="locked database"):
        BackupManager(settings).delete(name)

    assert database.read_bytes() == b"database"
    assert media.read_bytes() == b"media"
    assert not list(settings.backup_dir.glob(".delete-pending-*"))


@pytest.mark.parametrize("committed", [False, True])
def test_backup_prune_recovers_interrupted_pair_deletion(
    committed: bool,
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    name = "local-lm-20260110T120000Z-00000001.sqlite3"
    transaction = settings.backup_dir / ".delete-pending-test"
    transaction.mkdir()
    (transaction / name).write_bytes(b"database")
    (transaction / f"{name}.media.zip").write_bytes(b"media")
    if committed:
        (transaction / "COMMITTED").write_text(
            "lm-atelier-backup-delete-v1",
            encoding="utf-8",
        )

    BackupManager(settings).prune()

    assert not transaction.exists()
    database = settings.backup_dir / name
    media = settings.backup_dir / f"{name}.media.zip"
    if committed:
        assert not database.exists()
        assert not media.exists()
    else:
        assert database.read_bytes() == b"database"
        assert media.read_bytes() == b"media"


def test_media_archive_must_match_database_artifact_records(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    manager = BackupManager(settings)
    content = b"bound media"
    digest = hashlib.sha256(content).hexdigest()
    relative = f"{digest[:2]}/{digest[2:4]}/{digest}"
    artifact_path = settings.artifact_dir / relative
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(content)
    backup = settings.backup_dir / "local-lm-20260110T120000Z-00000001.sqlite3"
    _write_test_database(backup, "backup", artifact=(digest, len(content), relative))
    manager._create_media_archive(backup)

    with closing(sqlite3.connect(backup)) as connection:
        connection.execute("UPDATE artifacts SET size_bytes = size_bytes + 1")
        connection.commit()

    with pytest.raises(ValueError, match="does not match"):
        manager.verify(backup.name)


def test_media_backup_excludes_diagnostic_and_export_archives(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    manager = BackupManager(settings)
    content = b"nested private export"
    digest = hashlib.sha256(content).hexdigest()
    relative = f"{digest[:2]}/{digest[2:4]}/{digest}"
    artifact_path = settings.artifact_dir / relative
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(content)
    backup = settings.backup_dir / "local-lm-20260110T120000Z-00000001.sqlite3"
    _write_test_database(backup, "backup")
    with closing(sqlite3.connect(backup)) as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?, 'export')",
            (digest, len(content), relative),
        )
        connection.commit()

    manager._create_media_archive(backup)

    with zipfile.ZipFile(manager._media_path(backup)) as archive:
        assert archive.namelist() == ["manifest.json"]


def test_backup_prune_removes_only_stale_managed_transaction_files(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", temporary_retention_hours=24)
    settings.prepare()
    old = datetime.now(UTC) - timedelta(hours=25)
    stale = [
        settings.backup_dir / "backup-abcd.partial",
        settings.backup_dir / "local-lm-20260110T120000Z-00000001.sqlite3.media.partial",
        settings.backup_dir
        / ".local-lm-20260110T120000Z-00000001.sqlite3.media.zip.random.partial",
        settings.state_dir / "restore-abcd.partial",
    ]
    for path in stale:
        path.write_bytes(b"partial")
        os.utime(path, (old.timestamp(), old.timestamp()))
    fresh = settings.backup_dir / "backup-fresh.partial"
    fresh.write_bytes(b"preserve")
    unrelated = settings.backup_dir / "user.partial"
    unrelated.write_bytes(b"preserve")

    BackupManager(settings).prune()

    assert all(not path.exists() for path in stale)
    assert fresh.read_bytes() == b"preserve"
    assert unrelated.read_bytes() == b"preserve"


def _verify_spy(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every structural verification, which is the cost being avoided."""

    seen: list[Path] = []
    original = BackupManager._verify_path

    def spy(path: Path) -> None:
        seen.append(path)
        original(path)

    monkeypatch.setattr(BackupManager, "_verify_path", staticmethod(spy))
    return seen


def test_a_same_day_relaunch_reuses_the_receipt_instead_of_verifying_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: the second start does not walk the database again.

    `PRAGMA integrity_check` is a page-level structural walk and
    `foreign_key_check` scans every foreign key, and both were being repeated
    on every relaunch for a file already checked that day. This asserts the
    structural check does not run the second time, rather than asserting the
    result looks the same - a reused answer and a recomputed one are
    indistinguishable from the outside, which is exactly why the spy is on the
    work and not on the value.
    """
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    _write_test_database(settings.state_dir / "local-lm.sqlite3", "current")
    manager = BackupManager(settings)
    today = datetime(2026, 7, 25, 1, 2, 3, tzinfo=UTC)

    first = manager.ensure_daily_backup(now=today)
    seen = _verify_spy(monkeypatch)
    repeated = manager.ensure_daily_backup(now=today + timedelta(hours=20))

    assert repeated.name == first.name
    assert repeated.verified is True
    assert seen == [], "the structural check ran again despite a matching receipt"


def test_a_replaced_backup_cannot_inherit_the_earlier_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different bytes under the same name must be verified on their own.

    This is the failure the digest binding exists to prevent. A receipt keyed
    on the file name, or on `(st_dev, st_ino)`, would match here: the name is
    unchanged and an inode can be reused. The digest does not, so the
    replacement is checked rather than trusted.
    """
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    _write_test_database(settings.state_dir / "local-lm.sqlite3", "current")
    manager = BackupManager(settings)
    today = datetime(2026, 7, 25, 1, 2, 3, tzinfo=UTC)

    first = manager.ensure_daily_backup(now=today)
    backup_path = settings.backup_dir / first.name
    # Replaced under the same name, which is precisely the case a name-keyed or
    # inode-keyed receipt would wave through.
    backup_path.unlink()
    _write_test_database(backup_path, "replaced")

    seen = _verify_spy(monkeypatch)
    repeated = manager.ensure_daily_backup(now=today + timedelta(hours=1))

    assert seen == [backup_path], "replaced bytes were not re-verified"
    assert repeated.verified is True
    assert _database_marker(settings.backup_dir / repeated.name) == "replaced"


def test_a_damaged_receipt_causes_a_re_verification_rather_than_a_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt that cannot be read is no receipt, not a passing one."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    _write_test_database(settings.state_dir / "local-lm.sqlite3", "current")
    manager = BackupManager(settings)
    today = datetime(2026, 7, 25, 1, 2, 3, tzinfo=UTC)
    first = manager.ensure_daily_backup(now=today)

    receipt = manager._receipt_path(first.sha256)
    assert receipt.is_file()
    for damaged in ("", "not json at all", "[]", '{"schema": "wrong"}'):
        receipt.write_text(damaged, encoding="utf-8")
        seen = _verify_spy(monkeypatch)
        assert manager.ensure_daily_backup(now=today + timedelta(hours=2)).verified is True
        assert seen, f"a receipt reading {damaged!r} was treated as a pass"
        monkeypatch.undo()


def test_a_receipt_that_disagrees_about_size_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every recorded fact has to match, not only the one naming the file."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    _write_test_database(settings.state_dir / "local-lm.sqlite3", "current")
    manager = BackupManager(settings)
    today = datetime(2026, 7, 25, 1, 2, 3, tzinfo=UTC)
    first = manager.ensure_daily_backup(now=today)

    receipt = manager._receipt_path(first.sha256)
    record = json.loads(receipt.read_text(encoding="utf-8"))
    record["size_bytes"] = record["size_bytes"] + 1
    receipt.write_text(json.dumps(record), encoding="utf-8")

    seen = _verify_spy(monkeypatch)
    assert manager.ensure_daily_backup(now=today + timedelta(hours=3)).verified is True
    assert seen, "a receipt with the wrong size was accepted"


def test_no_receipt_is_written_when_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt records a pass, so a failure must leave none behind."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    _write_test_database(settings.state_dir / "local-lm.sqlite3", "current")
    manager = BackupManager(settings)
    today = datetime(2026, 7, 25, 1, 2, 3, tzinfo=UTC)
    first = manager.ensure_daily_backup(now=today)

    for path in manager._receipt_dir().glob("*.json"):
        path.unlink()

    def always_fails(path: Path) -> None:
        raise ValueError("backup failed SQLite integrity verification")

    monkeypatch.setattr(BackupManager, "_verify_path", staticmethod(always_fails))
    # The existing snapshot is skipped because it cannot be verified, so the
    # call falls through to making a new one - and that failure propagates
    # rather than returning an unverified backup.
    with pytest.raises(ValueError):
        manager.ensure_daily_backup(now=today + timedelta(hours=4))
    monkeypatch.undo()

    assert list(manager._receipt_dir().glob("*.json")) == [], (
        "a receipt was written for a backup that did not pass"
    )
    assert first.sha256


def test_receipts_older_than_every_retained_backup_are_pruned(tmp_path: Path) -> None:
    """A receipt outlives its file, so something has to remove it.

    Pruned on time rather than by digest deliberately: matching receipts to
    retained backups would mean re-reading every retained backup in full,
    which is the cost this change removes.
    """
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    _write_test_database(settings.state_dir / "local-lm.sqlite3", "current")
    manager = BackupManager(settings)
    manager.ensure_daily_backup(now=datetime(2026, 7, 25, 1, 0, 0, tzinfo=UTC))

    stale = manager._receipt_path("0" * 64)
    stale.write_text(json.dumps({"schema": "old"}), encoding="utf-8")
    ancient = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    os.utime(stale, (ancient, ancient))
    assert stale.is_file()

    manager.prune()

    assert not stale.exists(), "a receipt predating every retained backup survived"
    assert list(manager._receipt_dir().glob("*.json")), "the live receipt was pruned too"


def test_a_receipt_filed_under_one_digest_but_naming_another_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded digest is checked, not merely the name it is filed under.

    Receipts are content-addressed, so a wrong digest normally finds no file at
    all and the body's `sha256` never gets consulted. That makes the field look
    redundant - removing it passes every other test here, which is how it was
    found. It is not redundant: it is the only thing binding identity if the
    path scheme ever stops carrying the digest. This forges the one case the
    filename cannot catch, a receipt sitting at the right path whose body
    claims different bytes.
    """
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    _write_test_database(settings.state_dir / "local-lm.sqlite3", "current")
    manager = BackupManager(settings)
    today = datetime(2026, 7, 25, 1, 2, 3, tzinfo=UTC)
    first = manager.ensure_daily_backup(now=today)

    receipt = manager._receipt_path(first.sha256)
    record = json.loads(receipt.read_text(encoding="utf-8"))
    record["sha256"] = "f" * 64
    receipt.write_text(json.dumps(record), encoding="utf-8")

    seen = _verify_spy(monkeypatch)
    assert manager.ensure_daily_backup(now=today + timedelta(hours=5)).verified is True
    assert seen, "a receipt naming different bytes was accepted at the right path"
