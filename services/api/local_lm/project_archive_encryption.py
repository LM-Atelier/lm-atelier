"""Encrypting a project archive as it leaves, and opening one as it arrives.

The plaintext of an encrypted archive only ever exists in one staging folder
inside the export folder, which must be private to this account before any
plaintext is written there, in files named with ``STAGING_PREFIX`` that are
removed as soon as the work that needed them ends. A crash can still leave one
behind, so startup removes any that remain. The content-addressed store only
ever holds the encrypted form.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import IO, BinaryIO, cast

from .api_errors import ApiError, api_error
from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryExists,
    AnchoredEntryKind,
    create_private_directory,
    directory_private_to_current_user,
    list_entries,
    remove_entry,
)
from .portable_archive_v1 import MAGIC, ArchiveKind, ArchiveRefused, open_archive, write_archive

STAGING_PREFIX = "archive-staging-"
STAGING_FOLDER = "archive-staging"
_READ_BYTES = 1024 * 1024
logger = logging.getLogger("local_lm")


# What each refusal says to the person, by status and fixed code. A wrong
# passphrase and a damaged archive stay one answer; a key that could not be
# derived says nothing about the passphrase, so it is not that answer.
_REFUSALS: dict[str, tuple[int, str, str]] = {
    "archive-format-unsupported": (
        422,
        "archive-format-unsupported",
        "This archive's format is not one this version can open.",
    ),
    "archive-kind-mismatch": (
        422,
        "archive-kind-mismatch",
        "This archive is not a project archive.",
    ),
    "archive-limits-exceeded": (
        422,
        "archive-limits-exceeded",
        "This archive asks for settings outside what this version allows.",
    ),
    "archive-passphrase-invalid": (
        422,
        "archive-passphrase-invalid",
        "A passphrase must be between 1 and 1024 bytes.",
    ),
    "archive-key-derivation-failed": (
        503,
        "archive-key-derivation-failed",
        "This computer could not set aside the memory the passphrase needs. "
        "Close other applications and try again.",
    ),
    "invalid_passphrase_or_corrupt": (
        422,
        "archive-passphrase-or-archive-invalid",
        "The passphrase is wrong, or the archive is damaged.",
    ),
}


class ExportUnverified(Exception):
    """An encrypted export that did not open again to exactly what was written."""


class StagingNotPrivate(Exception):
    """A staging folder that other accounts could read, or whose access could not be read."""


def staging_api_error() -> ApiError:
    return api_error(
        409,
        "archive-staging-not-private",
        "An encrypted archive's contents are only written to a folder private to your "
        "account, and the data folder does not allow one. Nothing was written.",
    )


def private_staging(export_dir: Path) -> Path:
    """The staging folder, made private to this account, or a refusal.

    A new file takes its access from the folder it is made in, on Windows from
    the folder's access list, so the folder is what must be private. A missing
    one is created with that access set rather than inherited. Either way it is
    checked before any plaintext is written, and an existing folder is never
    changed: one that others can read, or whose access cannot be read, refuses.
    """

    folder = export_dir / STAGING_FOLDER
    try:
        with AnchoredDirectory(export_dir) as parent, contextlib.suppress(AnchoredEntryExists):
            create_private_directory(parent, STAGING_FOLDER)
        with AnchoredDirectory(folder, read_security=True) as anchor:
            private = directory_private_to_current_user(anchor)
    except AnchoredDirectoryError as exc:
        raise StagingNotPrivate from exc
    if not private:
        raise StagingNotPrivate
    return folder


def archive_api_error(refused: ArchiveRefused) -> ApiError:
    status, code, message = _REFUSALS[refused.code]
    return api_error(status, code, message)


class _Digest:
    """A destination that keeps only the SHA-256 and length of what it is given."""

    def __init__(self) -> None:
        self.hash = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        self.hash.update(data)
        self.size += len(data)
        return len(data)


def is_encrypted(source: IO[bytes]) -> bool:
    """Whether ``source`` starts as an encrypted archive does, leaving it where it was."""

    position = source.tell()
    try:
        return source.read(len(MAGIC)) == MAGIC
    finally:
        source.seek(position)


def staging_path(directory: Path, suffix: str) -> Path:
    """A new, empty file for staged archive bytes, readable by this account only."""

    path = directory / f"{STAGING_PREFIX}{uuid.uuid4().hex}{suffix}"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600
    )
    os.close(descriptor)
    return path


def encrypt_export(plaintext: Path, directory: Path, passphrase: bytes) -> Path:
    """Encrypt a finished project archive and prove it opens to the same bytes.

    The encrypted file is written beside the plaintext, flushed to disk, then
    opened again with the same passphrase. Only when every chunk authenticates
    and the bytes it opens to hash the same as the plaintext is it returned.
    """

    encrypted = staging_path(private_staging(directory), ".lm-atelier.encrypted")
    try:
        with plaintext.open("rb") as source, encrypted.open("wb") as destination:
            write_archive(source, destination, kind=ArchiveKind.PROJECT, passphrase=passphrase)
            destination.flush()
            os.fsync(destination.fileno())
        expected = hashlib.sha256()
        with plaintext.open("rb") as source:
            while chunk := source.read(_READ_BYTES):
                expected.update(chunk)
        opened = _Digest()
        with encrypted.open("rb") as source:
            try:
                # Opening only writes to its destination, which is all this one does.
                open_archive(
                    source, cast(BinaryIO, opened), kind=ArchiveKind.PROJECT, passphrase=passphrase
                )
            except ArchiveRefused as refused:
                # Memory for the check is this computer's limit, not a fault in
                # what was written, and says so the same way it does elsewhere.
                if refused.code == "archive-key-derivation-failed":
                    raise
                raise ExportUnverified from refused
        if opened.hash.digest() != expected.digest():
            raise ExportUnverified
    except BaseException:
        encrypted.unlink(missing_ok=True)
        raise
    return encrypted


def decrypt_import(source: BinaryIO, directory: Path, passphrase: bytes) -> Path:
    """Open an encrypted project archive into a staged file, or refuse and leave nothing.

    Chunks are written as they authenticate, so a refusal part way through
    would otherwise leave a true prefix of the archive behind; it is removed.
    """

    staged = staging_path(private_staging(directory), ".lm-atelier.zip")
    try:
        with staged.open("wb") as destination:
            open_archive(source, destination, kind=ArchiveKind.PROJECT, passphrase=passphrase)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


async def decrypt_import_owned(source: BinaryIO, directory: Path, passphrase: bytes) -> Path:
    """Decrypt off the event loop, and remove what was staged if the caller stops waiting.

    The thread cannot be stopped, so a request cancelled while it runs would
    lose the staged plaintext it was about to be handed. The work is shielded
    instead, and once it finishes, a file staged for a caller that has gone is
    removed rather than left until the next start.
    """

    decrypting = asyncio.ensure_future(
        asyncio.to_thread(decrypt_import, source, directory, passphrase)
    )
    try:
        return await asyncio.shield(decrypting)
    except asyncio.CancelledError:
        decrypting.add_done_callback(_discard_unclaimed)
        raise


def _discard_unclaimed(finished: asyncio.Future[Path]) -> None:
    # A failed decrypt has already removed its own staged file.
    if finished.cancelled() or finished.exception() is not None:
        return
    try:
        finished.result().unlink(missing_ok=True)
    except OSError:
        logger.warning("A staged project archive file could not be removed; startup removes it.")


def sweep_staging(export_dir: Path) -> int:
    """Remove staged archive files a previous run left behind, returning how many.

    Only regular files in the staging folder named with ``STAGING_PREFIX`` are
    removed. A link or a folder with that name is left alone, and nothing is
    followed.
    """

    try:
        anchor = AnchoredDirectory(export_dir / STAGING_FOLDER)
    except AnchoredDirectoryError:
        return 0
    removed = 0
    with anchor:
        try:
            entries = list_entries(anchor, include_metadata=False)
        except (AnchoredDirectoryError, OSError):
            # A folder this module cannot describe is left as it is; starting
            # up matters more than leftovers, and the next start tries again.
            logger.warning("The export folder could not be listed to remove staged archive files.")
            return 0
        for entry in entries:
            if entry.kind is not AnchoredEntryKind.FILE or not entry.name.startswith(
                STAGING_PREFIX
            ):
                continue
            try:
                remove_entry(anchor, entry.name)
            except AnchoredDirectoryError:
                # Starting up matters more than one leftover; the next start tries again.
                logger.warning("A staged project archive file could not be removed at startup.")
                continue
            removed += 1
    return removed
