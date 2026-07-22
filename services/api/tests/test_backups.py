from __future__ import annotations

from pathlib import Path

import pytest

from local_lm.backups import BackupManager
from local_lm.config import Settings


class VerificationConnection:
    def __init__(self) -> None:
        self.closed = False

    def execute(self, _query: str) -> VerificationConnection:
        return self

    def fetchone(self) -> tuple[str]:
        return ("ok",)

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
