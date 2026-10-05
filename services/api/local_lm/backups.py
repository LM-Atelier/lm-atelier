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
from contextlib import ExitStack, closing, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Literal, get_args

from .archive_key_store import ArchiveKeyMissing, ArchiveKeyStore, RestoreKeyBinding
from .config import Settings
from .filesystem_links import is_link_or_reparse
from .project_archive_encryption import STAGING_FOLDER, STAGING_PREFIX
from .schema_revisions import known_revisions
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
_RESTORE_MARKER = "restore-on-next-start.json"
_FAILED_RESTORE = "restore-failed.json"

#: Why a restore someone asked for could not be applied, as the API reports it.
FailedRestoreReason = Literal[
    "backup-missing", "backup-invalid", "backup-newer", "backup-key-missing", "restore-failed"
]


class BackupFromNewerVersion(ValueError):
    """A backup recording a schema revision this build's migrations do not know."""


class _RestoreNotApplied(Exception):
    """A restore that failed while the live data was still exactly as it was."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


class _LiveLogLeftAside(OSError):
    """The live database's log was moved aside and could not be put back."""


@dataclass(frozen=True)
class RestoreState:
    """A restore waiting for the next start, or why the last one asked for was not applied."""

    state: Literal["none", "pending", "failed"]
    backup: str | None = None
    reason: FailedRestoreReason | None = None
    failed_at: datetime | None = None
    #: Whether the restore is, or was, of an encrypted backup file rather than a backup here.
    encrypted: bool = False


class BackupManager:
    def __init__(self, settings: Settings, archive_keys: ArchiveKeyStore | None = None) -> None:
        self.settings = settings
        self.archive_keys = (
            ArchiveKeyStore(settings.data_dir) if archive_keys is None else archive_keys
        )
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
            self._snapshot_database(temporary)
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

    def snapshot_to(self, database: Path, media: Path | None) -> str:
        """Write a fresh, verified copy of the live state into files the caller made.

        Nothing is written to the backup folder and the managed backups are not
        touched, so a copy made this way never joins their rotation. With
        ``media``, the pictures and videos the copy refers to are written there
        and checked against it. Returns the copy's schema revision.
        """

        self._snapshot_database(database)
        if media is not None:
            self._write_media_archive(database, media)
            self._verify_media_archive(media, database)
        return self.schema_revision(database)

    def verify_files(self, database: Path, media: Path | None) -> int:
        """Check a database copy, and its media against it, as a backup is checked.

        Returns how many pictures and videos the copy refers to. Raises
        ValueError for anything a backup would be refused for.
        """

        self._verify_path(database)
        if media is not None:
            self._verify_media_archive(media, database)
        return len(self._database_artifact_records(database))

    def estimated_bytes(self, *, include_media: bool) -> int:
        """Roughly how large a fresh copy of the live state would be."""

        size = self._database_path().stat().st_size
        if include_media:
            size += sum(size_bytes for _sha, size_bytes, _path in self._live_artifact_records())
        return size

    @staticmethod
    def schema_revision(database: Path) -> str:
        """The schema revision a database copy was made at."""

        try:
            with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
                row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        except sqlite3.Error as exc:
            raise ValueError("backup failed SQLite integrity verification") from exc
        if not row or not isinstance(row[0], str) or not row[0]:
            raise ValueError("backup failed SQLite integrity verification")
        return row[0]

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

    def request_restore(self, name: str, *, requested: bool = False) -> BackupInfo:
        """Ask for `name` to replace the data on the next start.

        `requested` marks a restore a person asked for. If one of those cannot
        be applied, the next start keeps the data it has and says why, rather
        than failing. A restore armed for any other reason, such as the copy an
        upgrade takes before it begins, still stops the start when it fails,
        because the data it was meant to replace may be half changed.
        """

        with self._lock:
            result = self._verify_locked(name)
            self.require_known_revision(self._path(name))
            # Asking again replaces whatever the last failure said. Cleared
            # before the restore is armed, so a record that cannot be cleared
            # leaves nothing scheduled behind the error.
            (self.settings.state_dir / _FAILED_RESTORE).unlink(missing_ok=True)
            replaced = self._pending_encrypted()
            self._write_marker(
                {"backup": name, "requested": True} if requested else {"backup": name}
            )
            # An encrypted restore this one replaced takes its file and key with
            # it, once nothing points at them any more.
            if replaced is not None:
                self._discard_encrypted(replaced)
            result.restore_pending = True
            return result

    def request_encrypted_restore(
        self, binding: RestoreKeyBinding, encrypted: Path, key: bytes
    ) -> None:
        """Ask for an opened and checked encrypted backup to replace the data on the next start.

        ``encrypted`` is moved to where the next start looks for it and ``key``,
        its archive key, waits in the vault; the marker names neither, only the
        identities in ``binding``. Always a restore a person asked for. If the
        key cannot be stored nothing is scheduled, and no other restore that was
        waiting is left in place either.
        """

        with self._lock:
            (self.settings.state_dir / _FAILED_RESTORE).unlink(missing_ok=True)
            self._withdraw_locked()
            staged = self.staged_restore_path(binding.operation)
            os.replace(encrypted, staged)
            try:
                self.archive_keys.put(binding, key)
                self._write_marker({"encrypted": binding.as_marker(), "requested": True})
            except BaseException:
                with suppress(OSError):
                    staged.unlink(missing_ok=True)
                self.archive_keys.discard(binding)
                raise

    def staged_restore_path(self, operation: str) -> Path:
        """Where the encrypted file of the restore ``operation`` waits for the next start.

        It sits in the private archive staging folder under that folder's own
        prefix, so whatever a crash leaves is removed by the same sweep at
        startup, which runs after a waiting restore has been applied.
        """

        if not re.fullmatch(r"[0-9a-f]{32}", operation):
            raise ValueError("invalid restore operation")
        return (
            self.settings.export_dir
            / STAGING_FOLDER
            / f"{STAGING_PREFIX}restore-{operation}.lm-atelier.encrypted"
        )

    def restore_state(self) -> RestoreState:
        """What waits for the next start, or why the last restore asked for was not applied."""

        if self._pending_encrypted() is not None:
            return RestoreState(state="pending", encrypted=True)
        pending = self._pending_backup_name()
        if pending is not None:
            return RestoreState(state="pending", backup=pending)
        record = self.settings.state_dir / _FAILED_RESTORE
        if not record.is_file() or self._is_link(record):
            return RestoreState(state="none")
        try:
            payload = json.loads(record.read_text(encoding="utf-8"))
            reason = payload["reason"]
            failed_at = datetime.fromisoformat(payload["failed_at"])
            backup = payload.get("backup")
            encrypted = payload.get("encrypted") is True
        except (OSError, ValueError, KeyError, TypeError):
            return RestoreState(state="failed", reason="restore-failed")
        return RestoreState(
            state="failed",
            backup=backup if isinstance(backup, str) and _BACKUP_NAME.fullmatch(backup) else None,
            reason=reason if reason in get_args(FailedRestoreReason) else "restore-failed",
            failed_at=failed_at,
            encrypted=encrypted,
        )

    def dismiss_failed_restore(self) -> bool:
        """Forget why the last restore asked for was not applied. Returns whether there was one."""

        with self._lock:
            record = self.settings.state_dir / _FAILED_RESTORE
            existed = record.is_file()
            record.unlink(missing_ok=True)
            return existed

    def cancel_restore(self) -> bool:
        """Withdraw a restore that is no longer wanted.

        Pairs with `request_restore` for callers that arm a restore before doing
        something they might not survive, and withdraw it once they have.
        """

        with self._lock:
            return self._withdraw_locked()

    def withdraw_requested_restore(self) -> bool:
        """Withdraw the restore a person asked for, before it is applied.

        An encrypted one's file and key go with it. A restore armed for any
        other reason is left alone. Returns whether one was withdrawn.
        """

        with self._lock:
            payload = self._pending_marker()
            if payload is None or payload.get("requested") is not True:
                return False
            return self._withdraw_locked()

    def _withdraw_locked(self) -> bool:
        marker = self.settings.state_dir / _RESTORE_MARKER
        existed = marker.is_file()
        encrypted = self._pending_encrypted()
        marker.unlink(missing_ok=True)
        if encrypted is not None:
            self._discard_encrypted(encrypted)
        return existed

    def _discard_encrypted(self, binding: RestoreKeyBinding) -> None:
        """Remove an encrypted restore's file and its key; startup sweeps a file left behind."""

        try:
            self.staged_restore_path(binding.operation).unlink(missing_ok=True)
        except OSError:
            logger.warning("The file of a withdrawn restore could not be removed.", exc_info=True)
        self.archive_keys.discard(binding)

    def _write_marker(self, payload: dict[str, object]) -> None:
        marker = self.settings.state_dir / _RESTORE_MARKER
        fd, temporary_name = tempfile.mkstemp(
            prefix="restore-marker-",
            suffix=".partial",
            dir=self.settings.state_dir,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, marker)
        finally:
            temporary.unlink(missing_ok=True)

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
        # A backup with media holds the only backed-up copy of its pictures and
        # videos, so a later backup that day without media must not replace it.
        # The hourly check makes exactly such a backup whenever the day's only
        # one has media. Each retained day keeps its newest media backup too.
        media_days: set[str] = set()
        for path, created in parsed:
            day = created.date().isoformat()
            if day in daily_dates and day not in media_days and self._has_media(path):
                media_days.add(day)
                keep.add(path)
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

    def _has_media(self, path: Path) -> bool:
        media_path = self._media_path(path)
        return media_path.exists() or self._is_link(media_path)

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
            if self._has_media(path):
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
        a re-verification rather than a wrong reuse. A link at the receipt
        directory, or at the receipt file, is not a receipt either: reading it
        would follow that link.
        """

        path = self._receipt_path(info.sha256)
        if self._is_link(path) or self._is_link(path.parent):
            return False
        try:
            raw = path.read_text(encoding="utf-8")
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
        propagate. A link in place of the receipt directory is left untouched,
        because creating the receipt there would write into the directory that
        link names.
        """

        record = {
            "schema": _VERIFICATION_RECEIPT_SCHEMA,
            "sha256": info.sha256,
            "size_bytes": info.size_bytes,
            "verified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        try:
            directory = self._receipt_dir()
            if self._is_link(directory):
                return
            directory.mkdir(parents=True, exist_ok=True)
            if self._is_link(directory) or not directory.is_dir():
                return
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
        matched, because no file digests to it. A link in place of the receipt
        directory is left untouched. The files it names are outside this store.
        """

        if oldest_kept is None:
            return
        directory = self._receipt_dir()
        if self._is_link(directory) or not directory.is_dir():
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
        """Replace the data with the backup a restore was asked for, before it is opened.

        A restore a person asked for that fails while the current data is still
        exactly as it was records why, withdraws itself and lets the start go
        on, so a backup that went missing or went bad cannot keep LM Atelier
        from starting. A failure after the live data may have changed still
        raises, as does every failure of any other restore.
        """

        marker = self.settings.state_dir / _RESTORE_MARKER
        if not marker.is_file():
            return False
        if self._is_link(marker):
            raise ValueError("restore marker may not be a filesystem link")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        requested = isinstance(payload, dict) and payload.get("requested") is True
        try:
            if requested and "encrypted" in payload:
                return self._apply_encrypted_restore(marker, payload["encrypted"])
            return self._apply_restore(marker, payload)
        except _RestoreNotApplied as failure:
            if not requested:
                raise failure.cause from None
            self._record_failed_restore(marker, payload, failure.cause)
            return False

    def _record_failed_restore(
        self, marker: Path, payload: dict[str, Any], error: Exception
    ) -> None:
        reason: FailedRestoreReason
        if isinstance(error, FileNotFoundError):
            reason = "backup-missing"
        elif isinstance(error, BackupFromNewerVersion):
            reason = "backup-newer"
        elif isinstance(error, ArchiveKeyMissing):
            reason = "backup-key-missing"
        elif isinstance(error, ValueError | KeyError | TypeError):
            reason = "backup-invalid"
        else:
            reason = "restore-failed"
        logger.warning(
            "The restore that was asked for could not be applied (%s), so LM Atelier "
            "started with the data it already had.",
            reason,
            exc_info=error,
        )
        backup = payload.get("backup")
        record = {
            "backup": backup
            if isinstance(backup, str) and _BACKUP_NAME.fullmatch(backup)
            else None,
            "reason": reason,
            "failed_at": datetime.now(UTC).isoformat(),
            "encrypted": "encrypted" in payload,
        }
        try:
            self._write_failure_record(json.dumps(record))
        except OSError:
            logger.warning("Why the restore failed could not be saved.", exc_info=True)
        # Withdrawn even when the record could not be saved: starting matters more.
        marker.unlink()

    def _write_failure_record(self, text: str) -> None:
        """Write the failure record as a new file, never through a link at its name."""

        record = self.settings.state_dir / _FAILED_RESTORE
        if self._is_link(record):
            raise OSError("the restore failure record may not be a filesystem link")
        fd, temporary_name = tempfile.mkstemp(
            prefix="restore-failed-", suffix=".partial", dir=self.settings.state_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            # Replacing the name replaces whatever is there, a link included,
            # rather than writing to where it points.
            os.replace(temporary, record)
        finally:
            temporary.unlink(missing_ok=True)

    def _apply_restore(self, marker: Path, payload: Any) -> bool:
        """Put the backup in place of the live database.

        Every failure that leaves the live data exactly as it was is raised as
        _RestoreNotApplied. Anything else raised here means the live data may
        have changed.
        """

        try:
            source_path = self._path(str(payload["backup"]))
            media_path = self._media_path(source_path)
            media = media_path if media_path.is_file() else None
        except Exception as exc:
            raise _RestoreNotApplied(exc) from exc
        return self._restore_from(marker, source_path, media)

    def _apply_encrypted_restore(self, marker: Path, binding_value: object) -> bool:
        """Open the encrypted backup a restore waits on, with its stored key, and put it in place.

        The opened files are removed when this ends. The encrypted file and its
        key are used once: they go too once the restore was applied or was not.
        Only a failure that may have changed the live data keeps them, because
        then the start stops and the next one must be able to finish the
        restore. Failures are reported exactly as :meth:`_apply_restore` does.
        """

        # Imported here because the encrypted backup module builds on this one.
        from .backup_archives import opened_encrypted_restore

        binding: RestoreKeyBinding | None = None
        settled = False
        try:
            with ExitStack() as opened:
                try:
                    binding = RestoreKeyBinding.from_marker(binding_value)
                    database, media = opened.enter_context(opened_encrypted_restore(self, binding))
                except Exception as exc:
                    raise _RestoreNotApplied(exc) from exc
                applied = self._restore_from(marker, database, media)
            settled = True
            return applied
        except _RestoreNotApplied:
            settled = True
            raise
        finally:
            # A marker that cannot be read names no key, so none is removed:
            # it may be another restore's, and the next restore replaces it.
            if settled and binding is not None:
                self._discard_encrypted(binding)

    def _restore_from(self, marker: Path, source_path: Path, media_path: Path | None) -> bool:
        try:
            self._verify_path(source_path)
            # Checked again here, not only when the restore was asked for: a build
            # that cannot open the backup must not put it in place of data it can.
            self.require_known_revision(source_path)
            if media_path is not None:
                self._verify_media_archive(media_path, source_path)

            # Materialize and verify the complete database before changing any live
            # state. Media restoration is additive in the content-addressed store,
            # so a media failure can safely leave the current database untouched.
            fd, temporary_name = tempfile.mkstemp(
                prefix="restore-", suffix=".partial", dir=self.settings.state_dir
            )
            os.close(fd)
        except Exception as exc:
            raise _RestoreNotApplied(exc) from exc
        restored_database = Path(temporary_name)
        destination = self._database_path()
        try:
            try:
                with (
                    closing(sqlite3.connect(source_path)) as source,
                    closing(sqlite3.connect(restored_database)) as target,
                ):
                    source.backup(target)
                self._verify_path(restored_database)
                if media_path is not None:
                    self._restore_media_archive(media_path, restored_database)
                destination.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                raise _RestoreNotApplied(exc) from exc

            # A restored database must never inherit WAL pages from the
            # database it replaces. This runs before SQLAlchemy is configured.
            # The live log is only set aside until the replace succeeds: it may
            # hold writes not yet in the database file, and a replace that fails
            # leaves that database in use.
            try:
                set_aside = self._set_aside_live_log(destination)
                try:
                    os.replace(restored_database, destination)
                except BaseException:
                    self._put_back_live_log(set_aside)
                    raise
            except _LiveLogLeftAside:
                # The live database is without its log now; starting on it would
                # lose whatever the log held.
                raise
            except OSError as exc:
                raise _RestoreNotApplied(exc) from exc

            # The backup is in place from here, so nothing below may report the
            # restore as not applied.
            for _live, aside in set_aside:
                try:
                    aside.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "A database log set aside for a restore could not be removed.",
                        exc_info=True,
                    )
            marker.unlink()
            return True
        finally:
            restored_database.unlink(missing_ok=True)

    @staticmethod
    def _set_aside_live_log(database: Path) -> tuple[tuple[Path, Path], ...]:
        """Move the live database's WAL and shared-memory files out of SQLite's sight.

        Returns each moved file with the name it was moved to. If one cannot be
        moved, the ones already moved are put back before the failure is raised.
        """

        moved: list[tuple[Path, Path]] = []
        try:
            for suffix in ("-wal", "-shm"):
                live = database.with_name(f"{database.name}{suffix}")
                if not live.exists():
                    continue
                aside = live.with_name(f"{live.name}.restore-aside")
                os.replace(live, aside)
                moved.append((live, aside))
        except BaseException:
            BackupManager._put_back_live_log(tuple(moved))
            raise
        return tuple(moved)

    @staticmethod
    def _put_back_live_log(moved: tuple[tuple[Path, Path], ...]) -> None:
        try:
            for live, aside in reversed(moved):
                os.replace(aside, live)
        except OSError as exc:
            raise _LiveLogLeftAside("the live database log could not be put back") from exc

    @staticmethod
    def require_known_revision(path: Path) -> None:
        """Refuse a backup that records any schema revision this build's migrations do not know.

        Data on more than one migration branch records one revision for each,
        and every one of them must be known for this build to open it.
        """

        try:
            with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
                rows = connection.execute("SELECT version_num FROM alembic_version").fetchall()
        except sqlite3.Error as exc:
            raise ValueError("backup failed SQLite integrity verification") from exc
        recorded = {row[0] for row in rows}
        if not recorded or not recorded <= known_revisions():
            raise BackupFromNewerVersion(
                "backup uses a database schema revision this version does not recognize; "
                "it was likely made by a newer version of LM Atelier"
            )

    def _path(self, name: str) -> Path:
        if not _BACKUP_NAME.fullmatch(name):
            raise ValueError("invalid backup name")
        path = self.settings.backup_dir / name
        if not self._is_managed_file(path):
            raise FileNotFoundError(name)
        return path.resolve()

    def _pending_backup_name(self) -> str | None:
        payload = self._pending_marker()
        name = None if payload is None else payload.get("backup")
        return name if isinstance(name, str) and _BACKUP_NAME.fullmatch(name) else None

    def _pending_encrypted(self) -> RestoreKeyBinding | None:
        payload = self._pending_marker()
        if payload is None or "encrypted" not in payload:
            return None
        try:
            return RestoreKeyBinding.from_marker(payload["encrypted"])
        except ValueError:
            return None

    def _pending_marker(self) -> dict[str, Any] | None:
        marker = self.settings.state_dir / _RESTORE_MARKER
        if not marker.is_file() or self._is_link(marker):
            return None
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

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

    def _snapshot_database(self, target: Path) -> None:
        """Copy the live database into ``target`` with SQLite's online backup, then check it."""

        with (
            closing(sqlite3.connect(self._database_path())) as source,
            closing(sqlite3.connect(target)) as copy,
        ):
            source.backup(copy)
        self._verify_path(target)

    def _live_artifact_records(self) -> set[tuple[str, int, str]]:
        return self._database_artifact_records(self._database_path())

    def _create_media_archive(self, database_backup: Path) -> None:
        destination = self._media_path(database_backup)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".partial",
            dir=self.settings.backup_dir,
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            self._write_media_archive(database_backup, temporary)
            self._verify_media_archive(temporary, database_backup)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_media_archive(self, database_backup: Path, target: Path) -> None:
        """Write the pictures and videos ``database_backup`` refers to into the zip ``target``."""

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
        with zipfile.ZipFile(target, "w", allowZip64=True) as archive:
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
