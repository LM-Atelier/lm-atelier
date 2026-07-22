from __future__ import annotations

from pathlib import Path

import pytest

from local_lm.backups import BackupManager


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
