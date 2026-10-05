"""Encrypted backup files: a copy of this installation sealed with a passphrase.

An encrypted backup is a fresh copy of the database, and optionally of the
pictures and videos it refers to, sealed as one ``BACKUP`` archive. Inside the
archive is one stream laid out exactly as follows, so a later version can still
read a file made by this one:

    b"LMABKUP\\x00"        eight bytes
    header length          four bytes, big-endian, from 1 to 65536
    header                 that many bytes of UTF-8 JSON
    database               the header's database size_bytes
    media zip              the header's media size_bytes, when media is not null

and nothing after it. The header is ``{"format": "lm-atelier-backup",
"version": 1, "created_at", "app_version", "schema_revision", "database":
{"size_bytes", "sha256"}, "media": null | {"size_bytes", "sha256"}}``. It only
lets a damaged stream be refused early: the bytes themselves are authenticated
by the archive and checked by the same checks a backup gets.

Plaintext only ever exists in the private archive staging folder, in files
named with its prefix and removed as soon as the work that needed them ends;
startup removes any a crash left. Nothing here writes to the backup folder, so
these files never join the automatic backups or their rotation.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import struct
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import BinaryIO, Final, cast

from . import __version__
from .backups import BackupManager
from .config import Settings
from .portable_archive_v1 import (
    MAX_PASSPHRASE_BYTES,
    ArchiveKind,
    ArchiveRefused,
    open_archive,
    write_archive,
)
from .project_archive_encryption import ExportUnverified, private_staging, staging_path

PAYLOAD_MAGIC: Final = b"LMABKUP\x00"
PAYLOAD_FORMAT: Final = "lm-atelier-backup"
PAYLOAD_VERSION: Final = 1
MAX_HEADER_BYTES: Final = 64 * 1024
_LENGTH = struct.Struct(">I")
_FIXED_BYTES = len(PAYLOAD_MAGIC) + _LENGTH.size
_HEADER_KEYS = frozenset(
    {"format", "version", "created_at", "app_version", "schema_revision", "database", "media"}
)
_CREATED_AT = "%Y-%m-%dT%H:%M:%SZ"
# Exactly the shape _CREATED_AT writes, which is all a backup's report may carry.
_CREATED_AT_SHAPE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_READ_BYTES = 1024 * 1024
# Room for the media zip's own structure and the stream's framing, beyond the
# database and the pictures and videos themselves.
_STRUCTURE_BYTES = 64 * 1024 * 1024
# Room left on the disk beyond the files themselves, so making or checking a
# backup never fills the disk the live database still writes to.
_SPARE_DISK_BYTES = 256 * 1024 * 1024
# What SQLite may write beside a database copy while it is made or checked.
_SQLITE_SIDE_FILES = ("-journal", "-wal", "-shm")

# One encrypted backup is made or checked at a time: each derives a key that
# holds a quarter of a gigabyte, and each stages a full copy of the data.
_busy = threading.Lock()


class EncryptedBackupBusy(Exception):
    """Another encrypted backup is being made or checked."""


class BackupTooLarge(Exception):
    """The copy is larger than an encrypted backup can be."""


class BackupStorageInsufficient(Exception):
    """The disk does not have room for the copy and its encrypted form."""


class BackupInvalid(Exception):
    """An opened backup that is not a whole, consistent backup."""


@dataclass(frozen=True)
class EncryptedBackupReport:
    created_at: str
    app_version: str
    schema_revision: str
    database_size_bytes: int
    media_included: bool
    media_size_bytes: int | None
    artifact_count: int


def max_plaintext_bytes(settings: Settings) -> int:
    """The longest stream an encrypted backup may hold.

    A database and a media zip may each be as large as a project import, the
    same bound a media backup already has.
    """

    return 2 * settings.max_project_import_bytes + _STRUCTURE_BYTES


def max_encrypted_bytes(settings: Settings) -> int:
    """The largest encrypted backup file: its stream, a 16-byte tag per 16 KiB, a header."""

    plaintext = max_plaintext_bytes(settings)
    return plaintext + plaintext // 1024 + 64 * 1024


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_READ_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _header(database: Path, media: Path | None, schema_revision: str) -> dict[str, object]:
    database_size, database_sha = _digest(database)
    media_part: dict[str, object] | None = None
    if media is not None:
        media_size, media_sha = _digest(media)
        media_part = {"size_bytes": media_size, "sha256": media_sha}
    return {
        "format": PAYLOAD_FORMAT,
        "version": PAYLOAD_VERSION,
        "created_at": datetime.now(UTC).strftime(_CREATED_AT),
        "app_version": __version__,
        "schema_revision": schema_revision,
        "database": {"size_bytes": database_size, "sha256": database_sha},
        "media": media_part,
    }


def _framing(header: dict[str, object]) -> bytes:
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if not 1 <= len(encoded) <= MAX_HEADER_BYTES:
        raise BackupTooLarge
    return PAYLOAD_MAGIC + _LENGTH.pack(len(encoded)) + encoded


def _part(value: object) -> tuple[int, str]:
    if not isinstance(value, dict) or set(value) != {"size_bytes", "sha256"}:
        raise BackupInvalid
    size, sha = value["size_bytes"], value["sha256"]
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(sha, str)
        or len(sha) != 64
        or sha.strip("0123456789abcdef")
    ):
        raise BackupInvalid
    return size, sha


def _parts(header: dict[str, object]) -> list[tuple[str, int, str]]:
    """The parts a header names, in stream order, as (name, size, sha256)."""

    parts = [("database", *_part(header["database"]))]
    if header["media"] is not None:
        parts.append(("media", *_part(header["media"])))
    return parts


def _validated_header(raw: bytes) -> dict[str, object]:
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BackupInvalid from exc
    if (
        not isinstance(header, dict)
        or set(header) != _HEADER_KEYS
        or header["format"] != PAYLOAD_FORMAT
        or isinstance(header["version"], bool)
        or header["version"] != PAYLOAD_VERSION
    ):
        raise BackupInvalid
    for key in ("created_at", "app_version", "schema_revision"):
        if not isinstance(header[key], str) or not 0 < len(header[key]) <= 128:
            raise BackupInvalid
    if not _CREATED_AT_SHAPE.fullmatch(header["created_at"]):
        raise BackupInvalid
    try:
        datetime.strptime(header["created_at"], _CREATED_AT)
    except ValueError as exc:
        raise BackupInvalid from exc
    _parts(header)
    return header


class _Joined:
    """One readable stream: the framing, then the database, then the media zip."""

    def __init__(self, framing: bytes, parts: list[Path]) -> None:
        self._framing = framing
        self._parts = list(parts)
        self._current: BinaryIO | None = None

    def read(self, size: int) -> bytes:
        if self._framing:
            taken, self._framing = self._framing[:size], self._framing[size:]
            return taken
        while True:
            if self._current is None:
                if not self._parts:
                    return b""
                self._current = self._parts.pop(0).open("rb")
            chunk = self._current.read(size)
            if chunk:
                return chunk
            self._current.close()
            self._current = None

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None


class _Split:
    """A destination that takes the opened stream apart as it arrives.

    Each part is hashed as it passes and written to its file when one is
    given. Anything that does not follow the documented layout, or the sizes
    and digests the header names, is refused rather than written on.
    """

    def __init__(self, database: Path | None = None, media: Path | None = None) -> None:
        self._targets = {"database": database, "media": media}
        self._pending = bytearray()
        self.header: dict[str, object] | None = None
        self._order: list[tuple[str, int, str]] = []
        self._index = 0
        self._written = 0
        self._hash = hashlib.sha256()
        self._file: BinaryIO | None = None

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        if self.header is None:
            self._pending.extend(view)
            view = memoryview(self._take_header())
        while view:
            if self._index >= len(self._order):
                raise BackupInvalid
            name, size, _sha = self._order[self._index]
            chunk = view[: size - self._written]
            self._hash.update(chunk)
            target = self._targets[name]
            if self._file is None and target is not None:
                self._file = cast(BinaryIO, target.open("wb"))
            if self._file is not None:
                self._file.write(chunk)
            self._written += len(chunk)
            view = view[len(chunk) :]
            if self._written == size:
                self._end_part()
        return len(data)

    def _take_header(self) -> bytes:
        held = bytes(self._pending[:_FIXED_BYTES])
        if not (held.startswith(PAYLOAD_MAGIC) or PAYLOAD_MAGIC.startswith(held)):
            raise BackupInvalid
        if len(held) < _FIXED_BYTES:
            return b""
        (length,) = _LENGTH.unpack(held[len(PAYLOAD_MAGIC) :])
        if not 1 <= length <= MAX_HEADER_BYTES:
            raise BackupInvalid
        if len(self._pending) < _FIXED_BYTES + length:
            return b""
        self.header = _validated_header(bytes(self._pending[_FIXED_BYTES : _FIXED_BYTES + length]))
        self._order = _parts(self.header)
        rest = bytes(self._pending[_FIXED_BYTES + length :])
        self._pending.clear()
        self._skip_empty_parts()
        return rest

    def _skip_empty_parts(self) -> None:
        while self._index < len(self._order) and self._order[self._index][1] == 0:
            target = self._targets[self._order[self._index][0]]
            if target is not None:
                target.write_bytes(b"")
            self._end_part()

    def _end_part(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._hash.hexdigest() != self._order[self._index][2]:
            raise BackupInvalid
        self._index += 1
        self._written = 0
        self._hash = hashlib.sha256()
        self._skip_empty_parts()

    def finish(self) -> dict[str, object]:
        """The header, once every part has arrived whole and matched its digest."""

        self.close()
        if self.header is None or self._index != len(self._order):
            raise BackupInvalid
        return self.header

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _remove(*paths: Path | None) -> None:
    for path in paths:
        if path is None:
            continue
        for leftover in (path, *(path.with_name(path.name + side) for side in _SQLITE_SIDE_FILES)):
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                # Startup removes staged files that could not be removed now.
                continue


def _require_room(folder: Path, needed: int) -> None:
    if shutil.disk_usage(folder).free < needed + _SPARE_DISK_BYTES:
        raise BackupStorageInsufficient


def _passphrase_in_range(passphrase: bytes) -> None:
    if not 1 <= len(passphrase) <= MAX_PASSPHRASE_BYTES:
        raise ArchiveRefused("archive-passphrase-invalid")


def create_encrypted_backup(
    manager: BackupManager, settings: Settings, passphrase: bytes, *, include_media: bool
) -> Path:
    """Make an encrypted backup in private staging, proven to open, and return its path.

    The copy is sealed, then opened again with the same passphrase and taken
    apart as a check would, every part matching what was written. The
    plaintext is removed before this returns, whatever happens; the encrypted
    file is the caller's to store and remove.
    """

    _passphrase_in_range(passphrase)
    if not _busy.acquire(blocking=False):
        raise EncryptedBackupBusy
    try:
        staging = private_staging(settings.export_dir)
        # The plaintext copy and its encrypted form are both on disk at once.
        _require_room(staging, 2 * manager.estimated_bytes(include_media=include_media))
        database = staging_path(staging, ".sqlite3")
        media = staging_path(staging, ".media.zip") if include_media else None
        sealed: Path | None = None
        try:
            header = _header(database, media, manager.snapshot_to(database, media))
            framing = _framing(header)
            parts_bytes = sum(size for _name, size, _sha in _parts(header))
            if len(framing) + parts_bytes > max_plaintext_bytes(settings):
                raise BackupTooLarge
            sealed = staging_path(staging, ".lm-atelier.encrypted")
            source = _Joined(framing, [database] if media is None else [database, media])
            try:
                with sealed.open("wb") as destination:
                    write_archive(
                        cast(BinaryIO, source),
                        destination,
                        kind=ArchiveKind.BACKUP,
                        passphrase=passphrase,
                    )
                    destination.flush()
                    os.fsync(destination.fileno())
            finally:
                source.close()
            _verify_sealed(sealed, passphrase, header)
        except BaseException:
            _remove(sealed)
            raise
        finally:
            _remove(database, media)
        return sealed
    finally:
        _busy.release()


def _verify_sealed(sealed: Path, passphrase: bytes, header: dict[str, object]) -> None:
    opened = _Split()
    try:
        with sealed.open("rb") as source:
            open_archive(
                source, cast(BinaryIO, opened), kind=ArchiveKind.BACKUP, passphrase=passphrase
            )
        if opened.finish() != header:
            raise ExportUnverified
    except ArchiveRefused as refused:
        # Memory for the check is this computer's limit, not a fault in what
        # was written, and says so the same way it does elsewhere.
        if refused.code == "archive-key-derivation-failed":
            raise
        raise ExportUnverified from refused
    except BackupInvalid as exc:
        raise ExportUnverified from exc


async def check_uploaded_backup(
    manager: BackupManager,
    settings: Settings,
    body: AsyncIterator[bytes],
    passphrase: bytes,
    *,
    declared_bytes: int | None,
) -> EncryptedBackupReport:
    """Receive an encrypted backup into private staging, check it as a backup, keep nothing.

    The upload streams to disk in batches, never held whole, and is refused as
    soon as it passes the largest size an encrypted backup can be. Room on the
    disk is checked before any of it is read, for its declared length when it
    has one, and again before each batch is written, so an upload of unknown
    length stops while what has arrived can still be opened beside it. The
    check runs off the event loop; if the caller stops waiting, it still
    finishes, and the upload and everything opened from it are removed when it
    does.
    """

    _passphrase_in_range(passphrase)
    limit = max_encrypted_bytes(settings)
    if declared_bytes is not None and declared_bytes > limit:
        raise BackupTooLarge
    if not _busy.acquire(blocking=False):
        raise EncryptedBackupBusy
    upload: Path | None = None
    receiving: _Receiving | None = None
    try:
        staging = private_staging(settings.export_dir)
        # The upload and the plaintext it opens to are both on disk at once.
        _require_room(staging, 2 * (declared_bytes or 0))
        upload = staging_path(staging, ".lm-atelier.encrypted")
        receiving = _Receiving(upload)
        await receiving.receive(body, limit)
        checking = asyncio.ensure_future(
            asyncio.to_thread(_check_staged, manager, settings, receiving.path, passphrase)
        )
    except BaseException:
        if receiving is None:
            _remove(upload)
            _busy.release()
        else:
            receiving.abandon()
        raise
    checking.add_done_callback(partial(_end_check, receiving.path))
    return await asyncio.shield(checking)


class _Receiving:
    """An upload being written to its staged file, owned until its last write has ended.

    Each batch is written in a thread. A write already handed to a thread runs
    to its end even if the request is cancelled meanwhile, so giving the upload
    up waits for it: the file stays open, and the busy lock held, until then.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("wb")
        self._received = 0
        self._writing: asyncio.Future[None] | None = None

    async def receive(self, body: AsyncIterator[bytes], limit: int) -> None:
        batch = bytearray()
        async for chunk in body:
            self._received += len(chunk)
            if self._received > limit:
                raise BackupTooLarge
            batch.extend(chunk)
            if len(batch) >= _READ_BYTES:
                await self._write(bytes(batch))
                batch.clear()
        if batch:
            await self._write(bytes(batch))
        self._file.close()

    async def _write(self, data: bytes) -> None:
        self._writing = asyncio.ensure_future(
            asyncio.to_thread(self._write_with_room, data, self._received)
        )
        await asyncio.shield(self._writing)

    def _write_with_room(self, data: bytes, received: int) -> None:
        # After this write the disk must still hold as much again as has
        # arrived, for the plaintext it opens to.
        _require_room(self.path.parent, received + len(data))
        self._file.write(data)

    def abandon(self) -> None:
        """Close and remove the upload, and give up the busy lock, once no write is running."""

        if self._writing is not None and not self._writing.done():
            self._writing.add_done_callback(self._release)
            return
        self._release(self._writing)

    def _release(self, finished: asyncio.Future[None] | None) -> None:
        # Marks a write's outcome seen; whoever was waiting for it has gone.
        if finished is not None and not finished.cancelled():
            finished.exception()
        # A close that fails to flush must not keep the lock; startup removes
        # whatever could not be removed now.
        with contextlib.suppress(OSError):
            self._file.close()
        _remove(self.path)
        _busy.release()


def _end_check(upload: Path, finished: asyncio.Future[EncryptedBackupReport]) -> None:
    # The outcome is the waiting caller's, if one is still waiting; this only
    # marks it seen, so a caller that has gone does not leave it unread.
    if not finished.cancelled():
        finished.exception()
    _remove(upload)
    _busy.release()


def _check_staged(
    manager: BackupManager, settings: Settings, encrypted: Path, passphrase: bytes
) -> EncryptedBackupReport:
    staging = private_staging(settings.export_dir)
    # The encrypted file is already on disk; its plaintext is about as large.
    _require_room(staging, encrypted.stat().st_size)
    database = staging_path(staging, ".sqlite3")
    media = staging_path(staging, ".media.zip")
    try:
        split = _Split(database, media)
        try:
            with encrypted.open("rb") as source:
                open_archive(
                    source, cast(BinaryIO, split), kind=ArchiveKind.BACKUP, passphrase=passphrase
                )
        finally:
            split.close()
        header = split.finish()
        sizes = {name: size for name, size, _sha in _parts(header)}
        try:
            artifact_count = manager.verify_files(database, media if "media" in sizes else None)
            revision = manager.schema_revision(database)
        except ValueError as exc:
            raise BackupInvalid from exc
        if revision != header["schema_revision"]:
            raise BackupInvalid
        return EncryptedBackupReport(
            created_at=str(header["created_at"]),
            app_version=str(header["app_version"]),
            schema_revision=revision,
            database_size_bytes=sizes["database"],
            media_included="media" in sizes,
            media_size_bytes=sizes.get("media"),
            artifact_count=artifact_count,
        )
    finally:
        _remove(database, media)
