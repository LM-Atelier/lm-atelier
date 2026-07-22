from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .schemas import BackupInfo

_BACKUP_NAME = re.compile(r"^local-lm-\d{8}T\d{6}Z-[0-9a-f]{8}\.sqlite3$")


class BackupManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def list(self) -> list[BackupInfo]:
        items = [self._info(path) for path in self.settings.backup_dir.glob("*.sqlite3")]
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    def create(self) -> BackupInfo:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = os.urandom(4).hex()
        destination = self.settings.backup_dir / f"local-lm-{stamp}-{suffix}.sqlite3"
        fd, temporary_name = tempfile.mkstemp(
            prefix="backup-", suffix=".partial", dir=self.settings.backup_dir
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with (
                closing(sqlite3.connect(self._database_path())) as source,
                closing(sqlite3.connect(temporary)) as target,
            ):
                source.backup(target)
            self._verify_path(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return self._info(destination)

    def verify(self, name: str) -> BackupInfo:
        path = self._path(name)
        self._verify_path(path)
        result = self._info(path)
        result.verified = True
        return result

    def request_restore(self, name: str) -> BackupInfo:
        result = self.verify(name)
        marker = self.settings.state_dir / "restore-on-next-start.json"
        temporary = marker.with_suffix(".partial")
        temporary.write_text(json.dumps({"backup": name}), encoding="utf-8")
        os.replace(temporary, marker)
        result.restore_pending = True
        return result

    def delete(self, name: str) -> None:
        self._path(name).unlink()

    def apply_pending_restore(self) -> bool:
        marker = self.settings.state_dir / "restore-on-next-start.json"
        if not marker.is_file():
            return False
        payload = json.loads(marker.read_text(encoding="utf-8"))
        source_path = self._path(str(payload["backup"]))
        self._verify_path(source_path)
        destination = self._database_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            closing(sqlite3.connect(source_path)) as source,
            closing(sqlite3.connect(destination)) as target,
        ):
            source.backup(target)
        marker.unlink()
        return True

    def _path(self, name: str) -> Path:
        if not _BACKUP_NAME.fullmatch(name):
            raise ValueError("invalid backup name")
        path = (self.settings.backup_dir / name).resolve()
        if path.parent != self.settings.backup_dir.resolve() or not path.is_file():
            raise FileNotFoundError(name)
        return path

    def _database_path(self) -> Path:
        return self.settings.state_dir / "local-lm.sqlite3"

    @staticmethod
    def _verify_path(path: Path) -> None:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise ValueError("backup failed SQLite integrity verification")

    @staticmethod
    def _info(path: Path) -> BackupInfo:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        stat = path.stat()
        return BackupInfo(
            name=path.name,
            size_bytes=stat.st_size,
            sha256=digest.hexdigest(),
            created_at=datetime.fromtimestamp(stat.st_mtime, UTC),
        )
