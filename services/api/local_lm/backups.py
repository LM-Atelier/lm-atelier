from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
import tempfile
import threading
import zipfile
from contextlib import closing, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

from .config import Settings
from .filesystem_links import is_link_or_reparse
from .schemas import BackupInfo

_VERIFICATION_RECEIPT_SCHEMA = "lm-atelier-backup-verification-v1"
_BACKUP_NAME = re.compile(r"^local-lm-(?P<stamp>\d{8}T\d{6}Z)-[0-9a-f]{8}\.sqlite3$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_MEDIA_MANIFEST_BYTES = 16 * 1024 * 1024
_BACKUP_DELETE_MARKER = "lm-atelier-backup-delete-v1"
_REQUIRED_TABLES = {
    "alembic_version",
    "artifacts",
    "chats",
    "message_parts",
    "messages",
    "projects",
}
logger = logging.getLogger(__name__)
_BACKED_UP_ARTIFACT_KINDS = ("image", "video", "thumbnail", "input")


class BackupManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Automatic checks run in a worker thread while backup actions remain
        # available through the API. Serialize the filesystem transaction as
        # well as the daily check/create decision so two checks cannot both
        # decide that today's snapshot is missing.
        self._lock = threading.RLock()

    def list(self) -> list[BackupInfo]:
        with self._lock:
            pending_name = self._pending_backup_name()
            items: list[BackupInfo] = []
            for path in self.settings.backup_dir.glob("*.sqlite3"):
                if not _BACKUP_NAME.fullmatch(path.name) or not self._is_managed_file(path):
                    continue
                info = self._info(path)
                info.restore_pending = path.name == pending_name
                items.append(info)
            return sorted(items, key=lambda item: item.created_at, reverse=True)

    def create(self, *, include_media: bool = False) -> BackupInfo:
        with self._lock:
            return self._create_locked(
                include_media=include_media,
                created_at=datetime.now(UTC),
            )

    def ensure_daily_backup(self, *, now: datetime | None = None) -> BackupInfo:
        """Return one verified metadata-only recovery snapshot for the UTC day."""

        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            raise ValueError("daily backup time must include a timezone")
        current = current.astimezone(UTC)
        with self._lock:
            existing = self._verified_metadata_backup_for_day_locked(current)
            if existing is not None:
                try:
                    self._prune_locked()
                except OSError:
                    logger.warning("Could not prune old LM Atelier backups", exc_info=True)
                return existing
            created = self._create_locked(include_media=False, created_at=current)
            return self._verify_locked(created.name)

    def _create_locked(self, *, include_media: bool, created_at: datetime) -> BackupInfo:
        stamp = created_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
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
        except Exception:
            destination.unlink(missing_ok=True)
            self._media_path(destination).unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        try:
            self._prune_locked()
        except OSError:
            # A valid new backup must not be discarded merely because an older
            # snapshot is temporarily locked by antivirus or another process.
            logger.warning("Could not prune old LM Atelier backups", exc_info=True)
        return self._info(destination)

    def verify(self, name: str) -> BackupInfo:
        with self._lock:
            return self._verify_locked(name)

    def _verify_locked(self, name: str) -> BackupInfo:
        path = self._path(name)
        self._verify_path(path)
        media_path = self._media_path(path)
        if media_path.is_file():
            self._verify_media_archive(media_path, path)
        result = self._info(path)
        # Record the pass here as well as in the same-day lookup, because this
        # is where a freshly created backup is verified. Recording only in the
        # lookup would mean the first start after a backup was made still had
        # to walk it again to earn a receipt.
        self._write_verification_receipt(result)
        result.verified = True
        return result

    def request_restore(self, name: str) -> BackupInfo:
        with self._lock:
            result = self._verify_locked(name)
            marker = self.settings.state_dir / "restore-on-next-start.json"
            fd, temporary_name = tempfile.mkstemp(
                prefix="restore-marker-",
                suffix=".partial",
                dir=self.settings.state_dir,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump({"backup": name}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, marker)
            finally:
                temporary.unlink(missing_ok=True)
            result.restore_pending = True
            return result

    def cancel_restore(self) -> bool:
        """Withdraw a restore that is no longer wanted.

        Pairs with `request_restore` for callers that arm a restore before doing
        something they might not survive, and withdraw it once they have.
        """

        with self._lock:
            marker = self.settings.state_dir / "restore-on-next-start.json"
            existed = marker.is_file()
            marker.unlink(missing_ok=True)
            return existed

    def delete(self, name: str) -> None:
        with self._lock:
            path = self._path(name)
            if name == self._pending_backup_name():
                raise ValueError("backup is pending restore")
            self._delete_backup_pair(path)

    def prune(self) -> int:
        with self._lock:
            return self._prune_locked()

    def _prune_locked(self) -> int:
        self._cleanup_stale_partials()
        daily: dict[str, Path] = {}
        candidates: list[tuple[Path, datetime, int]] = []
        for path in self.settings.backup_dir.glob("*.sqlite3"):
            match = _BACKUP_NAME.fullmatch(path.name)
            if not match or not self._is_managed_file(path):
                continue
            try:
                created = datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=UTC
                )
            except ValueError:
                continue
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
        pending_name = self._pending_backup_name()
        if pending_name:
            keep.update(path for path, _created in parsed if path.name == pending_name)
        removed = 0
        for path, _created in parsed:
            if path not in keep:
                self._delete_backup_pair(path)
                removed += 1
        retained = [created for path, created in parsed if path in keep]
        self._prune_verification_receipts_locked(min(retained) if retained else None)
        return removed

    def _verified_metadata_backup_for_day_locked(
        self,
        current: datetime,
    ) -> BackupInfo | None:
        candidates: list[tuple[Path, datetime, int]] = []
        for path in self.settings.backup_dir.glob("*.sqlite3"):
            match = _BACKUP_NAME.fullmatch(path.name)
            if not match or not self._is_managed_file(path):
                continue
            try:
                created = datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=UTC
                )
                modified = path.stat().st_mtime_ns
            except (OSError, ValueError):
                continue
            if created.date() != current.date():
                continue
            media_path = self._media_path(path)
            if media_path.exists() or self._is_link(media_path):
                continue
            candidates.append((path, created, modified))
        candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
        for path, _created, _modified in candidates:
            try:
                # `_info` is computed first because it already digests the whole
                # file, and that digest is what a receipt is bound to. Ordering
                # it before the structural check makes the receipt free: nothing
                # is read that this method did not already read.
                result = self._info(path)
                if not self._verification_receipt_matches(result):
                    self._verify_path(path)
                    self._write_verification_receipt(result)
            except (OSError, ValueError):
                logger.warning(
                    "Ignoring an invalid recovery backup for the current UTC day",
                    exc_info=True,
                )
                continue
            result.verified = True
            return result
        return None

    def _receipt_dir(self) -> Path:
        return self.settings.state_dir / "backup-verifications"

    def _receipt_path(self, digest: str) -> Path:
        return self._receipt_dir() / f"{digest}.json"

    def _verification_receipt_matches(self, info: BackupInfo) -> bool:
        """True only when a receipt records THESE bytes passing verification.

        Bound to the content digest, not to the file name and not to
        `(st_dev, st_ino)`. A name can be re-pointed at different bytes between
        two starts, and inode numbers are reused, so either would let a
        replaced file inherit an earlier file's result - which is the one thing
        a reused verification must never do. The digest cannot: different bytes
        produce a different digest and therefore find no receipt.

        Reading it costs nothing extra. `_info` already streams the whole file
        to compute that digest, so the receipt removes `PRAGMA integrity_check`
        and `PRAGMA foreign_key_check` - a page-level structural walk and a
        scan across every foreign key - without adding a read.

        Fails closed. A receipt that is missing, unreadable, malformed, or
        disagrees about size is treated as no receipt at all, so the answer is
        a re-verification rather than a wrong reuse.
        """

        try:
            raw = self._receipt_path(info.sha256).read_text(encoding="utf-8")
            record = json.loads(raw)
        except (OSError, ValueError):
            return False
        if type(record) is not dict:
            return False
        return (
            record.get("schema") == _VERIFICATION_RECEIPT_SCHEMA
            and record.get("sha256") == info.sha256
            and record.get("size_bytes") == info.size_bytes
        )

    def _write_verification_receipt(self, info: BackupInfo) -> None:
        """Record a passing verification, and only after it has passed.

        Written to a name derived from the digest, which makes the store
        content-addressed and the record effectively immutable: a receipt for
        different bytes is a different file rather than an overwrite of this
        one. Published by rename so a crash mid-write cannot leave a partial
        record that would later read as a valid one.

        A failure to record is not a failure to verify. The backup was checked
        and is sound; losing the receipt costs one repeated check on the next
        start, which is the cost this exists to avoid rather than an error to
        propagate.
        """

        record = {
            "schema": _VERIFICATION_RECEIPT_SCHEMA,
            "sha256": info.sha256,
            "size_bytes": info.size_bytes,
            "verified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        try:
            directory = self._receipt_dir()
            directory.mkdir(parents=True, exist_ok=True)
            handle, temporary_name = tempfile.mkstemp(
                prefix="receipt-", suffix=".partial", dir=directory
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                    json.dump(record, stream, sort_keys=True)
                os.replace(temporary, self._receipt_path(info.sha256))
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        except OSError:
            logger.warning("Could not record a backup verification receipt", exc_info=True)

    def _prune_verification_receipts_locked(self, oldest_kept: datetime | None) -> None:
        """Drop receipts that predate every retained backup.

        Keyed by digest, so a receipt outlives the file it describes and would
        otherwise accumulate one entry for every backup ever verified.

        Pruned by time rather than by digest on purpose. Matching receipts to
        retained backups would mean digesting each retained backup, which is a
        full read of every one - the cost this whole change exists to remove.
        A receipt is written when its backup is verified, so a receipt older
        than the oldest retained backup cannot belong to one, and dropping it
        is safe. Erring towards keeping is free: a stale receipt is never
        matched, because no file digests to it.
        """

        if oldest_kept is None:
            return
        directory = self._receipt_dir()
        if not directory.is_dir():
            return
        cutoff = oldest_kept.timestamp()
        for path in directory.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue

    def _cleanup_stale_partials(self) -> None:
        self._recover_backup_deletions()
        cutoff = datetime.now(UTC) - timedelta(hours=self.settings.temporary_retention_hours)
        candidates = [
            *self.settings.backup_dir.glob("backup-*.partial"),
            *self.settings.backup_dir.glob("local-lm-*.sqlite3.media.partial"),
            *self.settings.backup_dir.glob(".local-lm-*.sqlite3.media.zip.*.partial"),
            *self.settings.state_dir.glob("restore-*.partial"),
            *self.settings.state_dir.glob("restore-marker-*.partial"),
        ]
        for path in candidates:
            if self._is_link(path) or not path.is_file():
                continue
            try:
                modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
                if modified <= cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove stale backup transaction file", exc_info=True)
        for media_path in self.settings.backup_dir.glob("local-lm-*.sqlite3.media.zip"):
            database_path = media_path.with_name(media_path.name.removesuffix(".media.zip"))
            if self._is_link(media_path) or not media_path.is_file() or database_path.exists():
                continue
            try:
                modified = datetime.fromtimestamp(media_path.stat().st_mtime, UTC)
                if modified <= cutoff:
                    media_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove orphaned backup media", exc_info=True)

    def _delete_backup_pair(self, database_path: Path) -> None:
        token = os.urandom(8).hex()
        transaction = self.settings.backup_dir / f".delete-pending-{token}"
        transaction.mkdir()
        originals = [self._media_path(database_path), database_path]
        staged: list[tuple[Path, Path]] = []
        try:
            for original in originals:
                if not original.exists() and not self._is_link(original):
                    continue
                temporary = transaction / original.name
                os.replace(original, temporary)
                staged.append((original, temporary))
            marker = transaction / "COMMITTED"
            with marker.open("x", encoding="utf-8") as handle:
                handle.write(_BACKUP_DELETE_MARKER)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            for original, temporary in reversed(staged):
                with suppress(OSError):
                    os.replace(temporary, original)
            with suppress(OSError):
                (transaction / "COMMITTED").unlink(missing_ok=True)
            with suppress(OSError):
                transaction.rmdir()
            raise
        for _original, temporary in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not finalize backup deletion", exc_info=True)
        with suppress(OSError):
            (transaction / "COMMITTED").unlink(missing_ok=True)
        with suppress(OSError):
            transaction.rmdir()

    def _recover_backup_deletions(self) -> None:
        for transaction in self.settings.backup_dir.glob(".delete-pending-*"):
            if self._is_link(transaction) or not transaction.is_dir():
                continue
            marker = transaction / "COMMITTED"
            try:
                committed = (
                    not self._is_link(marker)
                    and marker.is_file()
                    and marker.read_text(encoding="utf-8") == _BACKUP_DELETE_MARKER
                )
                entries = [path for path in transaction.iterdir() if path.name != "COMMITTED"]
                if committed:
                    for path in entries:
                        if path.is_file() or self._is_link(path):
                            path.unlink(missing_ok=True)
                else:
                    for path in entries:
                        if self._is_link(path) or not path.is_file():
                            continue
                        original = self.settings.backup_dir / path.name
                        if original.exists() or self._is_link(original):
                            logger.error(
                                "Could not recover interrupted backup deletion for %s",
                                path.name,
                            )
                            continue
                        os.replace(path, original)
                marker.unlink(missing_ok=True)
                transaction.rmdir()
            except OSError:
                logger.warning("Could not reconcile interrupted backup deletion", exc_info=True)

    def apply_pending_restore(self) -> bool:
        marker = self.settings.state_dir / "restore-on-next-start.json"
        if not marker.is_file():
            return False
        if self._is_link(marker):
            raise ValueError("restore marker may not be a filesystem link")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        source_path = self._path(str(payload["backup"]))
        self._verify_path(source_path)
        media_path = self._media_path(source_path)
        if media_path.is_file():
            self._verify_media_archive(media_path, source_path)

        # Materialize and verify the complete database before changing any live
        # state. Media restoration is additive in the content-addressed store,
        # so a media failure can safely leave the current database untouched.
        fd, temporary_name = tempfile.mkstemp(
            prefix="restore-", suffix=".partial", dir=self.settings.state_dir
        )
        os.close(fd)
        restored_database = Path(temporary_name)
        destination = self._database_path()
        try:
            with (
                closing(sqlite3.connect(source_path)) as source,
                closing(sqlite3.connect(restored_database)) as target,
            ):
                source.backup(target)
            self._verify_path(restored_database)
            if media_path.is_file():
                self._restore_media_archive(media_path, restored_database)

            destination.parent.mkdir(parents=True, exist_ok=True)
            # A restored database must never inherit WAL pages from the
            # database it replaces. This runs before SQLAlchemy is configured.
            Path(f"{destination}-wal").unlink(missing_ok=True)
            Path(f"{destination}-shm").unlink(missing_ok=True)
            os.replace(restored_database, destination)
            marker.unlink()
            return True
        finally:
            restored_database.unlink(missing_ok=True)

    def _path(self, name: str) -> Path:
        if not _BACKUP_NAME.fullmatch(name):
            raise ValueError("invalid backup name")
        path = self.settings.backup_dir / name
        if not self._is_managed_file(path):
            raise FileNotFoundError(name)
        return path.resolve()

    def _pending_backup_name(self) -> str | None:
        marker = self.settings.state_dir / "restore-on-next-start.json"
        if not marker.is_file() or self._is_link(marker):
            return None
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            name = payload.get("backup")
        except (OSError, json.JSONDecodeError):
            return None
        return name if isinstance(name, str) and _BACKUP_NAME.fullmatch(name) else None

    def _is_managed_file(self, path: Path) -> bool:
        if self._is_link(path) or not path.is_file():
            return False
        try:
            return path.resolve().parent == self.settings.backup_dir.resolve()
        except OSError:
            return False

    @staticmethod
    def _is_link(path: Path) -> bool:
        return is_link_or_reparse(
            path,
            missing="assume_regular",
            unreadable="assume_link",
        )

    def _database_path(self) -> Path:
        return self.settings.state_dir / "local-lm.sqlite3"

    @staticmethod
    def _media_path(database_backup: Path) -> Path:
        return database_backup.with_name(f"{database_backup.name}.media.zip")

    def _create_media_archive(self, database_backup: Path) -> None:
        destination = self._media_path(database_backup)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".partial",
            dir=self.settings.backup_dir,
        )
        os.close(fd)
        temporary = Path(temporary_name)
        with closing(sqlite3.connect(database_backup)) as connection:
            records = connection.execute(
                """
                SELECT sha256, size_bytes, relative_path
                FROM artifacts
                WHERE kind IN (?, ?, ?, ?)
                ORDER BY sha256
                """,
                _BACKED_UP_ARTIFACT_KINDS,
            ).fetchall()
        manifest: list[dict[str, object]] = []
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                for sha256, size_bytes, relative_path in records:
                    source = self._safe_artifact_path(str(relative_path).replace("\\", "/"))
                    if not source.is_file() or source.stat().st_size != int(size_bytes):
                        raise ValueError("artifact file is missing or changed during backup")
                    portable_relative_path = PurePosixPath(
                        str(relative_path).replace("\\", "/")
                    ).as_posix()
                    archive_path = f"artifacts/{portable_relative_path}"
                    archive.write(source, archive_path, compress_type=zipfile.ZIP_STORED)
                    manifest.append(
                        {
                            "sha256": sha256,
                            "size_bytes": size_bytes,
                            "relative_path": portable_relative_path,
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
            self._verify_media_archive(temporary, database_backup)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _verify_media_archive(self, path: Path, database_backup: Path | None = None) -> None:
        if self._is_link(path) or not path.is_file():
            raise ValueError("invalid media backup archive")
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > self.settings.max_project_archive_entries:
                    raise ValueError("media backup contains too many entries")
                if sum(info.file_size for info in infos) > self.settings.max_project_import_bytes:
                    raise ValueError("media backup expands beyond the configured limit")
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    raise ValueError("media backup contains duplicate entries")
                for info in infos:
                    file_mode = info.external_attr >> 16
                    if (
                        info.is_dir()
                        or info.flag_bits & 0x1
                        or (file_mode and stat.S_ISLNK(file_mode))
                    ):
                        raise ValueError("media backup contains unsupported entries")
                manifest_info = archive.getinfo("manifest.json")
                if manifest_info.file_size > _MAX_MEDIA_MANIFEST_BYTES:
                    raise ValueError("media backup manifest is too large")
                payload = json.loads(archive.read("manifest.json"))
                if (
                    not isinstance(payload, dict)
                    or payload.get("format") != "lm-atelier-backup-media"
                    or payload.get("version") != 1
                ):
                    raise ValueError("unsupported media backup format")
                records = payload.get("artifacts")
                if not isinstance(records, list):
                    raise ValueError("invalid media backup manifest")
                if len(records) > self.settings.max_project_archive_entries - 1:
                    raise ValueError("media backup manifest contains too many artifacts")
                archive_names = set(names)
                seen: set[str] = set()
                verified_records: set[tuple[str, int, str]] = set()
                for record in records:
                    if not isinstance(record, dict):
                        raise ValueError("invalid media backup artifact")
                    archive_path = str(record.get("archive_path", ""))
                    relative_path = PurePosixPath(str(record.get("relative_path", "")))
                    digest_value = record.get("sha256")
                    size_value = record.get("size_bytes")
                    if (
                        archive_path not in archive_names
                        or relative_path.is_absolute()
                        or ".." in relative_path.parts
                        or archive_path in seen
                        or not isinstance(digest_value, str)
                        or not _SHA256.fullmatch(digest_value)
                        or not isinstance(size_value, int)
                        or isinstance(size_value, bool)
                        or size_value < 0
                        or relative_path
                        != PurePosixPath(digest_value[:2]) / digest_value[2:4] / digest_value
                        or archive_path != f"artifacts/{relative_path.as_posix()}"
                    ):
                        raise ValueError("unsafe media backup path")
                    seen.add(archive_path)
                    digest = hashlib.sha256()
                    size = 0
                    with archive.open(archive_path) as source:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            size += len(chunk)
                    if digest.hexdigest() != digest_value or size != size_value:
                        raise ValueError("media backup checksum mismatch")
                    verified_records.add((digest_value, size_value, relative_path.as_posix()))
                if archive_names != {"manifest.json", *seen}:
                    raise ValueError("media backup contains unmanifested entries")
                if database_backup is not None:
                    expected_records = self._database_artifact_records(database_backup)
                    if verified_records != expected_records:
                        raise ValueError("media backup does not match its database backup")
        except (
            KeyError,
            RecursionError,
            RuntimeError,
            json.JSONDecodeError,
            sqlite3.Error,
            zipfile.BadZipFile,
        ) as exc:
            raise ValueError("invalid media backup archive") from exc

    def _restore_media_archive(self, path: Path, database_backup: Path) -> None:
        self._verify_media_archive(path, database_backup)
        with zipfile.ZipFile(path) as archive:
            payload = json.loads(archive.read("manifest.json"))
            for record in payload["artifacts"]:
                destination = self._safe_artifact_path(str(record["relative_path"]))
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".restore-partial",
                    dir=destination.parent,
                )
                os.close(fd)
                temporary = Path(temporary_name)
                try:
                    digest = hashlib.sha256()
                    size = 0
                    with (
                        archive.open(record["archive_path"]) as source,
                        temporary.open("wb") as target,
                    ):
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    if digest.hexdigest() != record["sha256"] or size != record["size_bytes"]:
                        raise ValueError("media backup changed during restore")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)

    @staticmethod
    def _verify_path(path: Path) -> None:
        try:
            with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
                result = connection.execute("PRAGMA integrity_check").fetchone()
                foreign_key_violation = connection.execute("PRAGMA foreign_key_check").fetchone()
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        except sqlite3.Error as exc:
            raise ValueError("backup failed SQLite integrity verification") from exc
        if (
            not result
            or result[0] != "ok"
            or foreign_key_violation
            or not _REQUIRED_TABLES.issubset(tables)
            or not version
            or not isinstance(version[0], str)
            or not version[0]
        ):
            raise ValueError("backup failed SQLite integrity verification")

    def _database_artifact_records(self, path: Path) -> set[tuple[str, int, str]]:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
            rows = connection.execute(
                """
                SELECT sha256, size_bytes, relative_path
                FROM artifacts
                WHERE kind IN (?, ?, ?, ?)
                """,
                _BACKED_UP_ARTIFACT_KINDS,
            ).fetchall()
        return {
            (
                str(sha256),
                int(size_bytes),
                PurePosixPath(str(relative_path).replace("\\", "/")).as_posix(),
            )
            for sha256, size_bytes, relative_path in rows
        }

    def _safe_artifact_path(self, relative_value: str) -> Path:
        relative = PurePosixPath(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact restore path escapes the store")
        root = self.settings.artifact_dir.resolve()
        candidate = root.joinpath(*relative.parts)
        cursor = root
        for part in relative.parts:
            cursor /= part
            if self._is_link(cursor):
                raise ValueError("artifact restore path uses a filesystem link")
        resolved = candidate.resolve()
        if root not in resolved.parents:
            raise ValueError("artifact restore path escapes the store")
        return candidate

    @staticmethod
    def _info(path: Path) -> BackupInfo:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        stat = path.stat()
        media_path = BackupManager._media_path(path)
        media_exists = media_path.is_file() and not BackupManager._is_link(media_path)
        return BackupInfo(
            name=path.name,
            size_bytes=stat.st_size,
            sha256=digest.hexdigest(),
            created_at=datetime.fromtimestamp(stat.st_mtime, UTC),
            media_included=media_exists,
            media_size_bytes=media_path.stat().st_size if media_exists else 0,
        )
