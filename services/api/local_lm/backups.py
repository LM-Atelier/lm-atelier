from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .schemas import BackupInfo

_BACKUP_NAME = re.compile(r"^local-lm-(?P<stamp>\d{8}T\d{6}Z)-[0-9a-f]{8}\.sqlite3$")


class BackupManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def list(self) -> list[BackupInfo]:
        items = [self._info(path) for path in self.settings.backup_dir.glob("*.sqlite3")]
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    def create(self, *, include_media: bool = False) -> BackupInfo:
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
            if include_media:
                self._create_media_archive(destination)
            self.prune()
        except Exception:
            destination.unlink(missing_ok=True)
            self._media_path(destination).unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return self._info(destination)

    def verify(self, name: str) -> BackupInfo:
        path = self._path(name)
        self._verify_path(path)
        media_path = self._media_path(path)
        if media_path.is_file():
            self._verify_media_archive(media_path)
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
        path = self._path(name)
        path.unlink()
        self._media_path(path).unlink(missing_ok=True)

    def prune(self) -> int:
        daily: dict[str, Path] = {}
        candidates: list[tuple[Path, datetime, int]] = []
        for path in self.settings.backup_dir.glob("*.sqlite3"):
            match = _BACKUP_NAME.fullmatch(path.name)
            if not match:
                continue
            created = datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            candidates.append((path, created, path.stat().st_mtime_ns))
        candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
        parsed = [(path, created) for path, created, _mtime in candidates]
        for path, created in parsed:
            daily.setdefault(created.date().isoformat(), path)
        keep = set(list(daily.values())[: self.settings.backup_daily_count])
        daily_dates = sorted(daily, reverse=True)[: self.settings.backup_daily_count]
        oldest_daily = daily_dates[-1] if daily_dates else None
        weekly: set[tuple[int, int]] = set()
        if self.settings.backup_weekly_count:
            for path, created in parsed:
                if oldest_daily and created.date().isoformat() >= oldest_daily:
                    continue
                week = created.isocalendar()[:2]
                if week in weekly:
                    continue
                weekly.add(week)
                keep.add(path)
                if len(weekly) >= self.settings.backup_weekly_count:
                    break
        removed = 0
        for path, _created in parsed:
            if path not in keep:
                path.unlink(missing_ok=True)
                self._media_path(path).unlink(missing_ok=True)
                removed += 1
        return removed

    def apply_pending_restore(self) -> bool:
        marker = self.settings.state_dir / "restore-on-next-start.json"
        if not marker.is_file():
            return False
        payload = json.loads(marker.read_text(encoding="utf-8"))
        source_path = self._path(str(payload["backup"]))
        self._verify_path(source_path)
        media_path = self._media_path(source_path)
        if media_path.is_file():
            self._verify_media_archive(media_path)
        destination = self._database_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            closing(sqlite3.connect(source_path)) as source,
            closing(sqlite3.connect(destination)) as target,
        ):
            source.backup(target)
        if media_path.is_file():
            self._restore_media_archive(media_path)
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
    def _media_path(database_backup: Path) -> Path:
        return database_backup.with_name(f"{database_backup.name}.media.zip")

    def _create_media_archive(self, database_backup: Path) -> None:
        destination = self._media_path(database_backup)
        temporary = destination.with_suffix(".partial")
        with closing(sqlite3.connect(database_backup)) as connection:
            records = connection.execute(
                "SELECT sha256, size_bytes, relative_path FROM artifacts ORDER BY sha256"
            ).fetchall()
        manifest: list[dict[str, object]] = []
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                for sha256, size_bytes, relative_path in records:
                    source = (self.settings.artifact_dir / str(relative_path)).resolve()
                    if self.settings.artifact_dir.resolve() not in source.parents:
                        raise ValueError("artifact path escapes the backup store")
                    if not source.is_file() or source.stat().st_size != int(size_bytes):
                        raise ValueError("artifact file is missing or changed during backup")
                    archive_path = f"artifacts/{relative_path}"
                    archive.write(source, archive_path, compress_type=zipfile.ZIP_STORED)
                    manifest.append(
                        {
                            "sha256": sha256,
                            "size_bytes": size_bytes,
                            "relative_path": relative_path,
                            "archive_path": archive_path,
                        }
                    )
                archive.writestr(
                    "manifest.json",
                    json.dumps(
                        {"format": "lm-atelier-backup-media", "version": 1, "artifacts": manifest}
                    ),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            self._verify_media_archive(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _verify_media_archive(self, path: Path) -> None:
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > self.settings.max_project_archive_entries:
                    raise ValueError("media backup contains too many entries")
                if sum(info.file_size for info in infos) > self.settings.max_project_import_bytes:
                    raise ValueError("media backup expands beyond the configured limit")
                payload = json.loads(archive.read("manifest.json"))
                if (
                    payload.get("format") != "lm-atelier-backup-media"
                    or payload.get("version") != 1
                ):
                    raise ValueError("unsupported media backup format")
                records = payload.get("artifacts")
                if not isinstance(records, list):
                    raise ValueError("invalid media backup manifest")
                names = set(archive.namelist())
                seen: set[str] = set()
                for record in records:
                    if not isinstance(record, dict):
                        raise ValueError("invalid media backup artifact")
                    archive_path = str(record.get("archive_path", ""))
                    relative_path = Path(str(record.get("relative_path", "")))
                    digest_value = record.get("sha256")
                    if (
                        archive_path not in names
                        or relative_path.is_absolute()
                        or ".." in relative_path.parts
                        or archive_path in seen
                        or not isinstance(digest_value, str)
                        or relative_path
                        != Path(digest_value[:2]) / digest_value[2:4] / digest_value
                    ):
                        raise ValueError("unsafe media backup path")
                    seen.add(archive_path)
                    digest = hashlib.sha256()
                    size = 0
                    with archive.open(archive_path) as source:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            size += len(chunk)
                    if digest.hexdigest() != digest_value or size != record.get("size_bytes"):
                        raise ValueError("media backup checksum mismatch")
        except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise ValueError("invalid media backup archive") from exc

    def _restore_media_archive(self, path: Path) -> None:
        self._verify_media_archive(path)
        with zipfile.ZipFile(path) as archive:
            payload = json.loads(archive.read("manifest.json"))
            for record in payload["artifacts"]:
                destination = (self.settings.artifact_dir / record["relative_path"]).resolve()
                if self.settings.artifact_dir.resolve() not in destination.parents:
                    raise ValueError("artifact restore path escapes the store")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".restore-partial")
                try:
                    with (
                        archive.open(record["archive_path"]) as source,
                        temporary.open("wb") as target,
                    ):
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)

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
        media_path = BackupManager._media_path(path)
        return BackupInfo(
            name=path.name,
            size_bytes=stat.st_size,
            sha256=digest.hexdigest(),
            created_at=datetime.fromtimestamp(stat.st_mtime, UTC),
            media_included=media_path.is_file(),
            media_size_bytes=media_path.stat().st_size if media_path.is_file() else 0,
        )
