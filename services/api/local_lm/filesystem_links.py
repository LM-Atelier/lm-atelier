from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import enum
import errno
import os
import stat
import sys
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, NoReturn

LinkInspectionFailure = Literal["assume_link", "assume_regular", "raise"]


def is_link_or_reparse(
    path: Path,
    *,
    missing: LinkInspectionFailure,
    unreadable: LinkInspectionFailure,
) -> bool:
    """Inspect a path without following filesystem links.

    Callers must choose how absence and other inspection failures affect their
    own safety boundary. Destructive and trust-sensitive paths generally fail
    closed; optional discovery paths can treat a missing entry as ordinary.
    """

    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        return _failure_result(missing, exc)
    except OSError as exc:
        return _failure_result(unreadable, exc)
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _failure_result(policy: LinkInspectionFailure, error: OSError) -> bool:
    if policy == "raise":
        raise error
    return policy == "assume_link"


# ---------------------------------------------------------------------------
# Anchored directories.
#
# Inspecting a path and then operating on that path again is two lookups, and
# between them the directory can be replaced. On Windows the replacement needs
# neither a rename nor a delete: FSCTL_SET_REPARSE_POINT converts an existing
# EMPTY directory into a junction in place, unelevated. Holding the directory
# open against rename and delete is therefore not enough, and neither is
# checking twice.
#
# What holds is never consulting a name again. The chain is walked one
# component at a time from the volume root, each component opened relative to
# the handle before it and refused if it carries a reparse point, and every
# later operation resolves against the handle that walk produced. POSIX spells
# this openat; Windows has no dir_fd for these calls, so it uses NtCreateFile
# with OBJECT_ATTRIBUTES.RootDirectory, the same construction one layer down.
#
# Three properties carry the guarantee and were each measured rather than
# assumed: a name opened relative to a held handle is never re-parsed; creating
# relative to a directory already converted returns
# STATUS_REPARSE_POINT_ENCOUNTERED rather than writing through it; and the
# conversion requires an empty directory, so the first entry created closes the
# window permanently.
#
# This layer is deliberately domain-neutral. It knows about directories,
# entries and containment; it knows nothing about what any caller stores.
# ---------------------------------------------------------------------------

_FILE_LIST_DIRECTORY: Final = 0x00000001
_FILE_WRITE_DATA: Final = 0x00000002
_FILE_TRAVERSE: Final = 0x00000020
_FILE_READ_ATTRIBUTES: Final = 0x00000080
_DELETE: Final = 0x00010000
_SYNCHRONIZE: Final = 0x00100000
_FILE_SHARE_READ: Final = 0x00000001
_FILE_SHARE_WRITE: Final = 0x00000002
_FILE_OPEN: Final = 1
_FILE_CREATE: Final = 2
_FILE_OPEN_IF: Final = 3
_FILE_READ_DATA: Final = 0x00000001
_FILE_DIRECTORY_FILE: Final = 0x00000001
_FILE_SYNCHRONOUS_IO_NONALERT: Final = 0x00000020
_FILE_NON_DIRECTORY_FILE: Final = 0x00000040
_FILE_OPEN_REPARSE_POINT: Final = 0x00200000
_OBJ_CASE_INSENSITIVE: Final = 0x40
_FILE_BASIC_INFORMATION_CLASS: Final = 4
_FILE_RENAME_INFORMATION_CLASS: Final = 10
_FILE_LINK_INFORMATION_CLASS: Final = 11
_STATUS_SUCCESS: Final = 0
_STATUS_OBJECT_NAME_COLLISION: Final = 0xC0000035
_STATUS_OBJECT_NAME_NOT_FOUND: Final = 0xC0000034
_STATUS_OBJECT_PATH_NOT_FOUND: Final = 0xC000003A
#: A success status with no handle should be impossible; it is given its own
#: value so it can never be mistaken for either of the not-found statuses.
_STATUS_UNEXPECTED: Final = 0xFFFFFFFF
_FILE_DISPOSITION_INFORMATION_CLASS: Final = 13
#: FileDirectoryInformation. The variable-length record carries the name and
#: the attributes TOGETHER, which is the whole reason enumeration belongs here:
#: a kind read from a second call is a kind read after the name could change.
_FILE_DIRECTORY_INFORMATION_CLASS: Final = 1
_STATUS_NO_MORE_FILES: Final = 0x80000006
#: Fixed part of FILE_DIRECTORY_INFORMATION, before FileName.
_FILE_DIRECTORY_INFORMATION_HEADER: Final = 64
#: One buffer that holds a useful number of records without a growth loop.
_DIRECTORY_QUERY_BUFFER: Final = 64 * 1024
#: A directory larger than this is refused rather than read. Every caller of
#: this primitive walks what it returns, so an unbounded answer is an
#: unbounded amount of someone else's work.
_MAX_LISTED_ENTRIES: Final = 8192
#: linkat flag: oldpath is ignored and olddirfd is the file itself.
_AT_EMPTY_PATH: Final = 0x1000
#: POSIX d_type values. Only the four that map to a distinct kind are named;
#: everything else is OTHER, and DT_UNKNOWN stays UNKNOWN rather than being
#: resolved by a second lookup.
_DT_UNKNOWN: Final = 0
_DT_DIR: Final = 4
_DT_REG: Final = 8
_DT_LNK: Final = 10
#: Linux/glibc 64-bit struct dirent offsets. Claimed for that platform only.
_DIRENT_TYPE_OFFSET: Final = 18
_DIRENT_NAME_OFFSET: Final = 19
#: No real dirent is longer; a larger d_reclen means the bytes are not one.
_DIRENT_RECORD_CEILING: Final = 4096
#: FILE_DIRECTORY_INFORMATION carries these beside the name and attributes, so
#: on Windows the size and the timestamp come from the SAME record as the kind.
_RECORD_LAST_WRITE_OFFSET: Final = 24
_RECORD_END_OF_FILE_OFFSET: Final = 40
#: NT timestamps count 100-nanosecond intervals from this instant.
_FILETIME_EPOCH: Final = datetime(1601, 1, 1, tzinfo=UTC)
#: FileFsSizeInformation. Volume capacity answered for the volume the HANDLE
#: is on, rather than for whatever a pathname resolves to at the moment it is
#: read.
_FILE_FS_SIZE_INFORMATION_CLASS: Final = 3
_MAX_ENTRY_NAME: Final = 260
_NT_NAMESPACE: Final = "\\??\\"

#: dir_fd-relative operations exist on POSIX only; Windows anchors through NT
#: handles instead. Every call this module makes with dir_fd is listed.
_HAS_DIR_FD: Final = (
    os.open in os.supports_dir_fd
    and os.link in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.rename in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
)

CONTAINMENT_REFUSED: Final = "directory containment could not be established"


class AnchoredDirectoryError(Exception):
    """A directory could not be anchored, or an anchored operation failed.

    Carries no path. Callers that promise a fixed non-echoing refusal can
    translate this into their own message without stripping anything.
    """


class AnchoredListingStopped(AnchoredDirectoryError):
    """A held-directory listing observed its caller's stop request.

    This is distinct from a containment refusal so a lifecycle owner can treat
    an expected shutdown as such. Like every other listing failure, it is
    raised before a tuple is returned; callers never receive a partial listing.
    """


class AnchoredDirectoryNotFound(AnchoredDirectoryError):
    """No entry of that name exists, and nothing on the way to it was a link.

    Distinct because "no library has been established here yet" is a
    legitimate answer to a legitimate question, while "something on that path
    is a link" is a refusal. Collapsing the two let a DANGLING linked root be
    reported as ordinary absence - the target is gone, so a following
    existence query says false - which is the escape this module exists to
    close. Only acquisition raises it, and only for genuine absence.
    """


class AnchoredEntryExists(AnchoredDirectoryError):
    """An entry could not be created because that name is already taken.

    Separate from the general refusal on purpose. A caller proving that
    exclusive creation WORKS has to tell a collision apart from an I/O error
    or a containment refusal; treating every failure as a collision turns the
    proof into a tautology that passes when the filesystem is broken.
    """


def _refuse() -> NoReturn:
    raise AnchoredDirectoryError(CONTAINMENT_REFUSED) from None


class AnchoredDirectory:
    """A held reference to a verified directory itself, never to its path.

    The whole ancestry is retained rather than released. Operations performed
    through the anchor do not need it - they resolve against the leaf - but a
    caller that still reads by path gets a path that keeps meaning what it
    meant, because a held directory can be neither renamed nor deleted.
    """

    __slots__ = ("_chain", "_windows", "path")

    def __init__(self, path: Path, *, create: bool = False) -> None:
        self.path = path
        self._chain: list[int] = []
        self._windows = not _HAS_DIR_FD and os.name == "nt"
        try:
            if _HAS_DIR_FD:
                self._chain = _walk_posix(path, create=create)
            elif self._windows:
                self._chain = _walk_windows(path, create=create)
            else:  # pragma: no cover - no third platform is supported
                _refuse()
        except OSError:
            self.close()
            _refuse()
        except AnchoredDirectoryError:
            self.close()
            raise

    @property
    def descriptor(self) -> int | None:
        """The leaf directory descriptor on POSIX, else None."""

        return None if self._windows or not self._chain else self._chain[-1]

    @property
    def handle(self) -> int | None:
        """The leaf directory handle on Windows, else None."""

        return self._chain[-1] if self._windows and self._chain else None

    def close(self) -> None:
        # Deepest first, so an ancestor is never released while something
        # below it is still held.
        while self._chain:
            held = self._chain.pop()
            if self._windows:
                _close_windows_handle(held)
            else:
                with contextlib.suppress(OSError):
                    os.close(held)

    def __enter__(self) -> AnchoredDirectory:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _adopt(path: Path, held: int, windows: bool) -> AnchoredDirectory:
    """Wrap an already-open child handle as its own anchor."""

    anchor = AnchoredDirectory.__new__(AnchoredDirectory)
    anchor.path = path
    anchor._chain = [held]
    anchor._windows = windows
    return anchor


def open_child_directory(
    anchor: AnchoredDirectory, name: str, *, create: bool = False
) -> AnchoredDirectory:
    """Open or create one child directory through the held parent."""

    _require_entry_name(name)
    try:
        if anchor.descriptor is not None:
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(name, 0o700, dir_fd=anchor.descriptor)
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            return _adopt(anchor.path / name, os.open(name, flags, dir_fd=anchor.descriptor), False)
    except OSError:
        _refuse()
    handle = anchor.handle
    if handle is None:
        _refuse()
    child = _nt_open_relative(handle, name, intent="create_dir" if create else "open_dir")
    if _nt_is_reparse(child):
        _close_windows_handle(child)
        _refuse()
    return _adopt(anchor.path / name, child, True)


def create_entry(anchor: AnchoredDirectory, name: str) -> int:
    """Create a new entry inside the held directory and return a descriptor.

    Create-only: an existing name refuses. The descriptor owns the underlying
    resource on both platforms, so closing it is the only cleanup required.
    """

    _require_entry_name(name)
    if anchor.descriptor is not None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            return os.open(name, flags, 0o600, dir_fd=anchor.descriptor)
        except FileExistsError:
            raise AnchoredEntryExists(CONTAINMENT_REFUSED) from None
        except OSError:
            _refuse()
    handle = anchor.handle
    if handle is None:
        _refuse()
    # _nt_open_relative raises AnchoredEntryExists for a collision, which the
    # caller needs in order to tell exclusivity from a broken filesystem.
    created = _nt_open_relative(handle, name, intent="create_file")
    try:
        return _descriptor_from_handle(created)
    except OSError:
        _close_windows_handle(created)
        _refuse()


def open_entry(anchor: AnchoredDirectory, name: str) -> int | None:
    """Open an EXISTING entry through the held directory and hand back a descriptor.

    None means the entry is not there. Everything else - refusing a link,
    refusing a directory, an unreadable entry - refuses.

    This exists because reading an entry as bytes throws away the descriptor,
    and a caller that then wants to check the entry's type or repair its mode
    has to look the name up again. Two lookups are two objects: the bytes
    validated and the thing subsequently modified need not be the same entry.
    Handing back the descriptor makes them one.

    The caller owns the descriptor and must close it.
    """

    _require_entry_name(name)
    if anchor.descriptor is not None:
        # O_NONBLOCK matters as much as O_NOFOLLOW here: opening a named pipe
        # for reading otherwise BLOCKS until a writer appears, so the type
        # check below would never be reached. The flag is dropped again once
        # the entry is known to be a regular file, for which it means nothing.
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=anchor.descriptor)
        except FileNotFoundError:
            return None
        except OSError:
            _refuse()
        return _require_regular(descriptor)
    handle = anchor.handle
    if handle is None:
        _refuse()
    opened, status = _nt_try_open_relative(handle, name, intent="open_file")
    if status in (_STATUS_OBJECT_NAME_NOT_FOUND, _STATUS_OBJECT_PATH_NOT_FOUND):
        return None
    if status != _STATUS_SUCCESS or not opened:
        _refuse()
    try:
        descriptor = _descriptor_from_handle(opened)
    except OSError:
        _close_windows_handle(opened)
        _refuse()
    return _require_regular(descriptor)


def _require_regular(descriptor: int) -> int:
    """Refuse anything that is not a regular file, closing it first.

    The contract this enforces is the caller's whole reason for holding a
    descriptor: it is going to read, stat and possibly chmod THROUGH it, and
    every one of those means something different on a directory, a device or
    a pipe.
    """

    try:
        mode = os.fstat(descriptor).st_mode
    except OSError:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        _refuse()
    if not stat.S_ISREG(mode):
        with contextlib.suppress(OSError):
            os.close(descriptor)
        _refuse()
    return descriptor


def take_regular_file(anchor: AnchoredDirectory, name: str) -> int | None:
    """Open an existing regular file through the held directory, with move rights.

    None means the name is gone. A link, a directory, or anything else refuses.
    The caller owns the descriptor and must close it.

    Distinct from open_entry because a later publish must move THIS object.
    Windows rename needs DELETE on the handle; open_entry's read intent cannot
    rename.
    """

    _require_entry_name(name)
    if anchor.descriptor is not None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=anchor.descriptor)
        except FileNotFoundError:
            return None
        except OSError:
            _refuse()
        return _require_regular(descriptor)
    handle = anchor.handle
    if handle is None:
        _refuse()
    opened, status = _nt_try_open_relative(handle, name, intent="rename_source")
    if status in (_STATUS_OBJECT_NAME_NOT_FOUND, _STATUS_OBJECT_PATH_NOT_FOUND):
        return None
    if status != _STATUS_SUCCESS or not opened:
        _refuse()
    if _nt_is_reparse(opened):
        _close_windows_handle(opened)
        _refuse()
    try:
        descriptor = _descriptor_from_handle(opened)
    except OSError:
        _close_windows_handle(opened)
        _refuse()
    return _require_regular(descriptor)


def publish_opened_file(
    source: AnchoredDirectory,
    name: str,
    opened: int,
    *,
    into: AnchoredDirectory,
    destination: str | None = None,
) -> None:
    """Move the already-open regular file into `into`.

    The object published is the one `opened` refers to, not a later lookup of
    `name`. `name` is used only to drop the source directory entry after a
    POSIX hard-link of that same inode. Destination must not already exist.
    """

    dest_name = name if destination is None else destination
    _require_entry_name(name)
    _require_entry_name(dest_name)
    if source.descriptor is not None:
        if into.descriptor is None:
            _refuse()
        _publish_opened_posix(source.descriptor, name, opened, into.descriptor, dest_name)
        return
    dest_handle = into.handle
    if dest_handle is None:
        _refuse()
    native = _handle_from_descriptor(opened)
    moved = _nt_set_name(
        native,
        dest_handle,
        dest_name,
        _FILE_RENAME_INFORMATION_CLASS,
        replace=False,
    )
    if not moved:
        raise AnchoredEntryExists(CONTAINMENT_REFUSED) from None


def _publish_opened_posix(
    source_dirfd: int,
    source_name: str,
    opened: int,
    dest_dirfd: int,
    dest_name: str,
) -> None:
    before = os.fstat(opened)
    if not stat.S_ISREG(before.st_mode):
        _refuse()
    if _link_opened_posix(opened, dest_dirfd, dest_name):
        dest = os.open(
            dest_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=dest_dirfd,
        )
        try:
            after = os.fstat(dest)
        finally:
            os.close(dest)
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            with contextlib.suppress(OSError):
                os.unlink(dest_name, dir_fd=dest_dirfd)
            _refuse()
    else:
        _copy_opened_posix(opened, dest_dirfd, dest_name)
    try:
        os.unlink(source_name, dir_fd=source_dirfd)
    except OSError:
        _refuse()


def _link_opened_posix(opened: int, dest_dirfd: int, dest_name: str) -> bool:
    """Hard-link the opened inode into dest. False means copy instead.

    Only one linkat failure says anything about containment: EEXIST means the
    destination name is already taken, and taking it anyway is the thing this
    module exists to refuse. Every other failure says that this host, this
    filesystem or this pair of mounts will not hard-link, which is a fact about
    the machine rather than about the file, and copying is the correct answer.

    An allow-list of "expected" errnos gets that backwards. It has to predict
    every way a kernel can decline to link, and the ones it misses turn an
    ordinary install into a refusal: EXDEV when staging and the model directory
    sit on different mounts, ENOENT or EACCES where AT_EMPTY_PATH needs a
    capability the process does not hold, EMLINK on a full link count. None of
    those are unsafe, and none of them are rare.

    Falling back does not weaken the boundary. `_copy_opened_posix` reads from
    this same held descriptor rather than looking the name up again, creates the
    destination with O_EXCL | O_NOFOLLOW so it can neither replace an entry nor
    write through a link, and refuses on any error of its own. What it gives up
    is a shared inode, which nothing here relies on.
    """

    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return False
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        return False
    encoded = os.fsencode(dest_name)
    result = linkat(
        ctypes.c_int(opened),
        b"",
        ctypes.c_int(dest_dirfd),
        encoded,
        ctypes.c_int(_AT_EMPTY_PATH),
    )
    if result == 0:
        return True
    if ctypes.get_errno() == errno.EEXIST:
        raise AnchoredEntryExists(CONTAINMENT_REFUSED) from None
    return False


def _copy_opened_posix(opened: int, dest_dirfd: int, dest_name: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        dest = os.open(dest_name, flags, 0o600, dir_fd=dest_dirfd)
    except FileExistsError:
        raise AnchoredEntryExists(CONTAINMENT_REFUSED) from None
    except OSError:
        _refuse()
    try:
        os.lseek(opened, 0, os.SEEK_SET)
        while True:
            chunk = os.read(opened, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(dest, view)
                if written <= 0:
                    raise OSError("incomplete write")
                view = view[written:]
        os.fsync(dest)
    except OSError:
        os.close(dest)
        with contextlib.suppress(OSError):
            os.unlink(dest_name, dir_fd=dest_dirfd)
        _refuse()
    os.close(dest)


def read_entry(anchor: AnchoredDirectory, name: str) -> bytes | None:
    """Return an entry's bytes, or None when it does not exist."""

    _require_entry_name(name)
    try:
        if anchor.descriptor is not None:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(name, flags, dir_fd=anchor.descriptor)
            except FileNotFoundError:
                return None
            with os.fdopen(descriptor, "rb") as entry:
                return entry.read()
        handle = anchor.handle
        if handle is None:
            _refuse()
        # Absence is decided by the open itself. Testing anchor.path first
        # resolved a name again, which is the whole thing this layer exists to
        # avoid - and an adopted child holds only its own handle, so its
        # ancestors are not held and the name really can change underneath.
        opened, status = _nt_try_open_relative(handle, name, intent="open_file")
        if status in (_STATUS_OBJECT_NAME_NOT_FOUND, _STATUS_OBJECT_PATH_NOT_FOUND):
            return None
        if status != _STATUS_SUCCESS or not opened:
            _refuse()
        with os.fdopen(_descriptor_from_handle(opened), "rb") as entry:
            return entry.read()
    except OSError:
        _refuse()


def link_entry(anchor: AnchoredDirectory, source: str, destination: str) -> bool:
    """Publish `source` under `destination`, create-only.

    Returns False when the destination already exists, which is convergence
    rather than failure for callers racing to publish the same thing.
    """

    _require_entry_name(source)
    _require_entry_name(destination)
    if anchor.descriptor is not None:
        try:
            os.link(
                source,
                destination,
                src_dir_fd=anchor.descriptor,
                dst_dir_fd=anchor.descriptor,
            )
        except FileExistsError:
            return False
        except OSError:
            _refuse()
        return True
    handle = anchor.handle
    if handle is None:
        _refuse()
    opened = _nt_open_relative(handle, source, intent="open_file")
    try:
        return _nt_set_name(opened, handle, destination, _FILE_LINK_INFORMATION_CLASS)
    finally:
        _close_windows_handle(opened)


def rename_entry(
    anchor: AnchoredDirectory,
    source: str,
    destination: str,
    *,
    replace: bool = False,
    into: AnchoredDirectory | None = None,
) -> None:
    """Rename `source` to `destination`, optionally into another held directory.

    `replace` is explicit because the platforms disagree by default and the
    disagreement is silent: os.rename REPLACES an existing destination, while
    the NT call refuses one. A single unqualified call therefore meant
    "publish over the old file" on POSIX and "do nothing" on Windows, and the
    Windows outcome arrived as a return value rather than as a refusal - so
    the atomic-publish pattern this module exists to support worked on Linux
    and silently no-opped on Windows.

    `into` names the destination directory when it is not the source's. It
    exists because content-addressed publication cannot avoid crossing
    directories: the digest is not known until the bytes have been consumed,
    so staging must happen somewhere chosen BEFORE the destination shard is
    known. The alternatives were copying, which defeats atomicity and doubles
    IO on large media, or abandoning containment for that publish.

    BOTH directories are held for the whole operation, so neither end can be
    substituted between the call and the kernel acting on it. Both names are
    still single components: this moves an entry between two verified
    directories and can never be handed a path.

    A refusal RAISES like every other refusal here rather than returning
    something a caller can drop.

    With replace=False the entry must be one that can be linked, which on
    POSIX means a regular file; a directory refuses.
    """

    _require_entry_name(source)
    _require_entry_name(destination)
    target = anchor if into is None else into
    if anchor.descriptor is not None:
        if target.descriptor is None:
            # Mixing a POSIX anchor with a Windows one cannot happen on one
            # host, so this is a programming error rather than a filesystem
            # condition - but it refuses rather than reaching a kernel call
            # with a meaningless descriptor.
            _refuse()
        if replace:
            try:
                os.rename(
                    source,
                    destination,
                    src_dir_fd=anchor.descriptor,
                    dst_dir_fd=target.descriptor,
                )
            except OSError:
                _refuse()
            return
        # POSIX rename ALWAYS replaces, so a non-replacing rename is a link
        # that refuses an existing name followed by dropping the old one.
        # Both steps go through the held directories.
        try:
            os.link(
                source,
                destination,
                src_dir_fd=anchor.descriptor,
                dst_dir_fd=target.descriptor,
            )
        except FileExistsError:
            raise AnchoredEntryExists(CONTAINMENT_REFUSED) from None
        except OSError:
            _refuse()
        try:
            os.unlink(source, dir_fd=anchor.descriptor)
        except OSError:
            _refuse()
        return
    handle = anchor.handle
    destination_handle = target.handle
    if handle is None or destination_handle is None:
        _refuse()
    # A rename changes a directory entry on the SOURCE, so the source handle
    # needs DELETE - which the read intent has no business holding. The
    # DESTINATION directory travels in the information block rather than as a
    # path, which is what makes the cross-directory form possible at all.
    opened = _nt_open_relative(handle, source, intent="rename_source")
    try:
        moved = _nt_set_name(
            opened,
            destination_handle,
            destination,
            _FILE_RENAME_INFORMATION_CLASS,
            replace=replace,
        )
    finally:
        _close_windows_handle(opened)
    if not moved:
        raise AnchoredEntryExists(CONTAINMENT_REFUSED) from None


def remove_entry(anchor: AnchoredDirectory, name: str) -> None:
    """Delete one entry inside the held directory, refusing on failure.

    Absence is not a failure. Anything else is: a caller that cannot remove
    what it staged needs to know, because the alternative is a leftover
    sitting beside something it just reported as successful.
    """

    _require_entry_name(name)
    if anchor.descriptor is not None:
        try:
            os.unlink(name, dir_fd=anchor.descriptor)
        except FileNotFoundError:
            return
        except OSError:
            _refuse()
        return
    handle = anchor.handle
    if handle is None:
        _refuse()
    # Opened relative to the held directory and deleted through that handle.
    # Unlinking by path re-resolved the name, and an adopted child does not
    # hold its ancestors, so the path could mean something else by now.
    opened, status = _nt_try_open_relative(handle, name, intent="delete_source")
    if status in (_STATUS_OBJECT_NAME_NOT_FOUND, _STATUS_OBJECT_PATH_NOT_FOUND):
        return
    if status != _STATUS_SUCCESS or not opened:
        _refuse()
    try:
        _nt_mark_deleted(opened)
    finally:
        _close_windows_handle(opened)


def remove_directory_entry(anchor: AnchoredDirectory, name: str) -> None:
    """Remove one EMPTY child directory through the held parent.

    Deliberately not recursive and deliberately separate from remove_entry. A
    recursive delete that meets a link partway down is the failure this module
    exists to prevent, and a caller that wants recursion should have to say so
    and hold each level itself.

    Absence is success, matching remove_entry: a caller pruning what it just
    listed should not fail because someone else pruned it first. Everything
    else refuses - a file, a link or reparse point, a directory that still has
    entries in it, or any other failure.

    The link case matters most and is why this cannot simply be `rmdir` by
    path. On POSIX `os.rmdir` on a symlink fails because a symlink is not a
    directory. On Windows a junction IS a directory to the ordinary APIs, so
    the entry is opened through the held parent with FILE_OPEN_REPARSE_POINT -
    which opens the link itself - and refused if it carries a reparse point,
    before any disposition is set.
    """

    _require_entry_name(name)
    if anchor.descriptor is not None:
        try:
            os.rmdir(name, dir_fd=anchor.descriptor)
        except FileNotFoundError:
            return
        except OSError:
            # ENOTDIR for a file or a symlink, ENOTEMPTY for a populated
            # directory, anything else for a real failure. All of them refuse:
            # this function removes an empty directory or nothing.
            _refuse()
        return
    handle = anchor.handle
    if handle is None:
        _refuse()
    opened, status = _nt_try_open_relative(handle, name, intent="delete_directory")
    if status in (_STATUS_OBJECT_NAME_NOT_FOUND, _STATUS_OBJECT_PATH_NOT_FOUND):
        return
    if status != _STATUS_SUCCESS or not opened:
        # A file opened with FILE_DIRECTORY_FILE returns
        # STATUS_NOT_A_DIRECTORY, so the file case refuses here rather than
        # needing a separate check.
        _refuse()
    try:
        if _nt_is_reparse(opened):
            _refuse()
        # A non-empty directory returns STATUS_DIRECTORY_NOT_EMPTY from the
        # disposition, which _nt_mark_deleted turns into this module's refusal.
        _nt_mark_deleted(opened)
    finally:
        _close_windows_handle(opened)


def discard_entry(anchor: AnchoredDirectory, name: str) -> None:
    """Best-effort removal, for use while a refusal is already propagating.

    Separate from remove_entry on purpose: suppressing a cleanup failure is
    right when something has already gone wrong and wrong when it has not.
    """

    with contextlib.suppress(AnchoredDirectoryError, OSError):
        remove_entry(anchor, name)


class AnchoredEntryKind(enum.Enum):
    """What an entry is, as the directory's own enumeration record said.

    UNKNOWN is a real answer rather than a failure. A filesystem is allowed to
    return a name without a type, and the honest report is that the type is not
    known - not a second lookup by that name, which is the gap this module
    exists to close.
    """

    FILE = "file"
    DIRECTORY = "directory"
    LINK = "link"
    OTHER = "other"
    UNKNOWN = "unknown"


#: Kinds no caller may treat as ordinary content. A consumer that deletes,
#: reads or trusts what it lists must skip these: a link may point anywhere,
#: and an unknown kind may BE a link.
UNSAFE_ENTRY_KINDS: Final = frozenset(
    {AnchoredEntryKind.LINK, AnchoredEntryKind.UNKNOWN, AnchoredEntryKind.OTHER}
)


@dataclasses.dataclass(frozen=True, slots=True)
class AnchoredEntry:
    """One directory entry: a validated single-component name and its kind.

    Frozen because it is evidence. A caller that could edit the kind after the
    fact could turn a refusal into a permission without touching the
    filesystem, and the record would still look like it came from the kernel.

    `size_bytes` and `modified_at` are populated ONLY for FILE and DIRECTORY,
    and only when they could be established. Both are None together, never one
    without the other, so a consumer has a single question to ask.

    They are absent by design for LINK, OTHER and UNKNOWN. A prune that cannot
    establish an age does not prune; it skips. Attaching metadata to an unsafe
    classification would turn "I do not know what this is" into a permission to
    act on it, which is the whole failure this module exists to prevent.

    `size_bytes` FOR A DIRECTORY MEANS DIFFERENT THINGS ON THE TWO PLATFORMS,
    and it is documented rather than papered over. Windows reports the record's
    EndOfFile, which is 0 for a directory; POSIX reports the directory's own
    on-disk size from `st_size`, typically a block. Neither is the size of what
    the directory contains, and a caller enforcing a byte budget should sum
    FILE entries only. Forcing them to agree would mean inventing a number on
    one platform to match the other.
    """

    name: str
    kind: AnchoredEntryKind
    size_bytes: int | None = None
    modified_at: datetime | None = None

    @property
    def is_safe(self) -> bool:
        """True only for an ordinary file or directory from the record."""

        return self.kind not in UNSAFE_ENTRY_KINDS

    @property
    def has_metadata(self) -> bool:
        """True when size and modification time were both established.

        False is a legitimate answer and not an error: an unsafe kind never
        carries metadata, and a safe entry that vanished or refused
        reacquisition between the enumeration and the measurement does not
        either. A consumer that needs an age must skip an entry that answers
        False rather than substituting a default.
        """

        return self.size_bytes is not None and self.modified_at is not None


def list_entries(
    anchor: AnchoredDirectory,
    *,
    limit: int = _MAX_LISTED_ENTRIES,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[AnchoredEntry, ...]:
    """List a held directory, taking each name and kind from one record.

    The name and the kind come from the SAME enumeration record on both
    platforms, so nothing is looked up by name a second time. That is the whole
    point: a caller that lists names and then stats each one has reopened the
    window between the check and the operation, once per entry.

    `.` and `..` never appear. A duplicate name, a name that is not a single
    component, or more than `limit` entries refuses - a directory that answers
    with a name it should not be able to hold is not one this module describes.

    Neither platform can fall back on the KIND. Windows reads the name and the
    attributes from one FILE_DIRECTORY_INFORMATION record. POSIX reads the name
    and `d_type` from one `dirent`, and a `d_type` of DT_UNKNOWN becomes
    AnchoredEntryKind.UNKNOWN - it is never resolved by asking again, which
    would be the second lookup this primitive exists to remove.

    Size and modification time are different and the difference is stated
    rather than blurred. On Windows they come from that same record. On POSIX
    the `dirent` does not carry them, so a FILE or a DIRECTORY is REACQUIRED
    through the held parent and measured with `fstat` - one anchored lookup
    after the enumeration. Unsafe kinds are never measured on either platform,
    and an entry that vanished or refused carries no metadata.

    When `should_stop` is supplied, it is observed before and after native
    record reads and around POSIX's anchored metadata work. A request raises
    AnchoredListingStopped and abandons the whole result. The callback should
    therefore be a cheap, side-effect-free readiness query such as
    `threading.Event.is_set`.
    """

    if limit < 1:
        _refuse()
    entries: list[AnchoredEntry] = []
    seen: set[str] = set()
    posix_metadata = anchor.descriptor is not None
    records = _iter_anchored_entries(anchor, limit, should_stop)
    with contextlib.closing(records):
        while True:
            # An explicit check BEFORE next() is what prevents a caller that
            # stopped after validating one record from forcing another native
            # record to be requested.
            _raise_if_listing_stopped(should_stop)
            try:
                entry = next(records)
            except StopIteration:
                _raise_if_listing_stopped(should_stop)
                break
            _raise_if_listing_stopped(should_stop)

            # A single-component name is what every other operation on this
            # anchor requires. Refuse it and duplicates before POSIX performs
            # the extra anchored metadata lookup for this record.
            _require_entry_name(entry.name)
            _raise_if_listing_stopped(should_stop)
            if entry.name in seen or len(entries) >= limit:
                _refuse()
            seen.add(entry.name)

            if posix_metadata:
                entry = _with_posix_metadata(anchor, entry, should_stop=should_stop)
            _raise_if_listing_stopped(should_stop)
            entries.append(entry)
    return tuple(entries)


def _raise_if_listing_stopped(
    should_stop: Callable[[], bool] | None,
) -> None:
    """Raise the fixed, distinguishable whole-list stop result when requested."""

    if should_stop is not None and should_stop():
        raise AnchoredListingStopped("directory listing stopped") from None


def _iter_anchored_entries(
    anchor: AnchoredDirectory,
    limit: int,
    should_stop: Callable[[], bool] | None,
) -> Generator[AnchoredEntry, None, None]:
    """Yield native name/kind records without any second lookup by name."""

    descriptor = anchor.descriptor
    if descriptor is not None:
        yield from _iter_posix_entries(descriptor, limit, should_stop)
        return
    handle = anchor.handle
    if handle is None:
        _refuse()
    yield from _iter_windows_entries(handle, limit, should_stop)


def _kind_from_dirent_type(raw: int) -> AnchoredEntryKind:
    """Map a POSIX d_type to a kind, with no filesystem access at all.

    Deliberately a pure function of one integer. `os.scandir` is not used here
    and neither are `DirEntry.is_dir`, `is_file` or `is_symlink`: each of those
    answers from the record when the filesystem supplied a type and otherwise
    performs `fstatat` by name, which is the second lookup this module exists
    to remove. DT_UNKNOWN therefore maps to UNKNOWN and is never resolved.

    Being a pure function is also what makes the unknown case testable. A
    filesystem that omits d_type cannot be conjured in a test, but this
    mapping can be handed DT_UNKNOWN directly, and no fallback can hide in a
    function that takes an int and touches nothing.
    """

    if raw == _DT_REG:
        return AnchoredEntryKind.FILE
    if raw == _DT_DIR:
        return AnchoredEntryKind.DIRECTORY
    if raw == _DT_LNK:
        return AnchoredEntryKind.LINK
    if raw == _DT_UNKNOWN:
        return AnchoredEntryKind.UNKNOWN
    return AnchoredEntryKind.OTHER


def _list_posix(descriptor: int, limit: int) -> list[AnchoredEntry]:
    """Enumerate through the held descriptor, one dirent at a time.

    Name and kind only, from the enumeration record, touching nothing else.
    Metadata is a separate pass - see `_with_posix_metadata` - because a
    `dirent` does not carry it and pretending otherwise is the exact blurring
    this module refuses.
    """

    return [
        AnchoredEntry(name, _kind_from_dirent_type(raw))
        for name, raw in _read_dirents(descriptor, limit)
    ]


def _iter_posix_entries(
    descriptor: int,
    limit: int,
    should_stop: Callable[[], bool] | None,
) -> Generator[AnchoredEntry, None, None]:
    """Yield POSIX name/kind pairs from one dirent apiece."""

    records = _iter_dirents(descriptor, limit, should_stop)
    with contextlib.closing(records):
        for name, raw in records:
            _raise_if_listing_stopped(should_stop)
            yield AnchoredEntry(name, _kind_from_dirent_type(raw))


def _with_posix_metadata(
    anchor: AnchoredDirectory,
    entry: AnchoredEntry,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> AnchoredEntry:
    """Attach size and modification time to a safe entry, or leave it as it is.

    This is where POSIX pays what Windows does not. A `dirent` carries a name
    and a d_type and nothing else, so a FILE or a DIRECTORY is REACQUIRED
    through the held parent - `open_entry` or `open_child_directory` - and
    measured with `fstat` on the descriptor that returns. That is ONE anchored
    lookup after the enumeration, said plainly rather than described as coming
    from the record, because it does not.

    `fstat` resolves no name at all; the reacquisition does, through the held
    parent, which is what keeps it contained. An entry that vanished in
    between, or that refuses, keeps its kind and carries no metadata.
    """

    _raise_if_listing_stopped(should_stop)
    size, modified = _posix_metadata(anchor, entry.name, entry.kind, should_stop=should_stop)
    _raise_if_listing_stopped(should_stop)
    if size is None or modified is None:
        return entry
    return AnchoredEntry(entry.name, entry.kind, size, modified)


def _posix_metadata(
    anchor: AnchoredDirectory,
    name: str,
    kind: AnchoredEntryKind,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[int | None, datetime | None]:
    """Size and modification time for a safe entry, or (None, None).

    Only FILE and DIRECTORY are measured. Everything else is left unmeasured on
    purpose - see AnchoredEntry - and so is anything that has gone away or
    refuses, because a measurement that could not be taken is not a zero.
    """

    _raise_if_listing_stopped(should_stop)
    if kind is AnchoredEntryKind.FILE:
        try:
            opened = open_entry(anchor, name)
        except AnchoredDirectoryError:
            _raise_if_listing_stopped(should_stop)
            return None, None
        if opened is None:
            _raise_if_listing_stopped(should_stop)
            return None, None
        try:
            # open_entry validates the descriptor with fstat. Check here, while
            # the descriptor is owned by this try/finally, so a stop observed
            # during that anchored open closes it before leaving.
            _raise_if_listing_stopped(should_stop)
            measured = os.fstat(opened)
            _raise_if_listing_stopped(should_stop)
        except OSError:
            return None, None
        finally:
            with contextlib.suppress(OSError):
                os.close(opened)
        _raise_if_listing_stopped(should_stop)
        return _from_stat(measured)
    if kind is AnchoredEntryKind.DIRECTORY:
        try:
            child = open_child_directory(anchor, name)
        except AnchoredDirectoryError:
            _raise_if_listing_stopped(should_stop)
            return None, None
        try:
            _raise_if_listing_stopped(should_stop)
            held = child.descriptor
            if held is None:  # pragma: no cover - POSIX branch only
                return None, None
            measured = os.fstat(held)
            _raise_if_listing_stopped(should_stop)
        except OSError:
            return None, None
        finally:
            child.close()
        _raise_if_listing_stopped(should_stop)
        return _from_stat(measured)
    _raise_if_listing_stopped(should_stop)
    return None, None


def _from_stat(measured: os.stat_result) -> tuple[int, datetime]:
    """Size and modification time from one fstat result."""

    return int(measured.st_size), datetime.fromtimestamp(measured.st_mtime, tz=UTC)


def _from_filetime(raw: int) -> datetime | None:
    """An NT timestamp as a datetime, or None when it says nothing.

    Zero is what the record carries for "not recorded", and a timestamp is also
    refused if it is outside the range a datetime can hold. Either way the
    honest answer is that the time is unknown, which the entry reports as
    absent metadata rather than as the year 1601.
    """

    if raw <= 0:
        return None
    try:
        return _FILETIME_EPOCH + timedelta(microseconds=raw // 10)
    except (OverflowError, ValueError):
        return None


def _read_dirents(descriptor: int, limit: int) -> list[tuple[str, int]]:
    """Collect the streaming dirent reader for compatibility with callers."""

    return list(_iter_dirents(descriptor, limit, None))


def _iter_dirents(
    descriptor: int,
    limit: int,
    should_stop: Callable[[], bool] | None,
) -> Generator[tuple[str, int], None, None]:
    """Name and raw d_type per entry, straight from the directory stream.

    `fdopendir` takes ownership of the descriptor it is given and `closedir`
    closes it, so it is never handed the anchor's own descriptor - that would
    release the containment guarantee halfway through reading it.

    It is handed a FRESH descriptor rather than a dup, and the difference is
    not cosmetic. `os.dup` shares the file offset with its original, so the
    first enumeration leaves the anchor's own descriptor sitting at end of
    directory and every later listing returns nothing at all. Opening `.`
    through the held descriptor is the same inode reached through a handle
    that already IS that directory - no name is resolved and nothing can be
    swapped underneath it - and the new descriptor carries its own offset.

    The dirent layout is read by offset rather than through a ctypes Structure
    because only one platform's layout is claimed. Linux/glibc on a 64-bit
    host: d_ino 0..7, d_off 8..15, d_reclen 16..17, d_type 18, d_name from 19,
    NUL-terminated. Any other POSIX platform, and any word size other than 64
    bits, refuses rather than guessing, on the same principle as the rest of
    this module: a wrong offset reads a plausible type out of the wrong byte
    and classifies silently, which is worse than not answering.
    """

    _raise_if_listing_stopped(should_stop)
    if not sys.platform.startswith("linux"):  # pragma: no cover - CI is Linux
        _refuse()

    import ctypes

    if ctypes.sizeof(ctypes.c_void_p) != 8:  # pragma: no cover - CI is 64-bit
        # The offsets above are the 64-bit layout. On a 32-bit host d_ino and
        # d_off are four bytes each, so d_type would be read out of the middle
        # of d_off - a plausible small integer, silently classifying every
        # entry wrongly. Refusing is the only safe answer to a layout this
        # module has not measured.
        _refuse()

    libc: Any = ctypes.CDLL(None, use_errno=True)
    libc.fdopendir.argtypes = [ctypes.c_int]
    libc.fdopendir.restype = ctypes.c_void_p
    libc.readdir.argtypes = [ctypes.c_void_p]
    libc.readdir.restype = ctypes.c_void_p
    libc.closedir.argtypes = [ctypes.c_void_p]
    libc.closedir.restype = ctypes.c_int

    try:
        # "." through the held descriptor: the directory itself, with an offset
        # of its own, leaving the anchor's descriptor where it was.
        owned = os.open(".", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0), dir_fd=descriptor)
    except OSError:
        _refuse()
    stream = libc.fdopendir(owned)
    if not stream:
        os.close(owned)
        _refuse()

    found = 0
    try:
        while True:
            _raise_if_listing_stopped(should_stop)
            ctypes.set_errno(0)
            record = libc.readdir(stream)
            _raise_if_listing_stopped(should_stop)
            if not record:
                # NULL is both end-of-stream and failure; errno separates them,
                # and a partial listing reported as complete would be a caller
                # deciding on a directory it has only half seen.
                if ctypes.get_errno():
                    _refuse()
                return
            # d_reclen first, and read exactly that many bytes. A fixed read
            # of the maximum name length would run past the LAST record in
            # the kernel's buffer, because glibc sizes each record to its
            # own name and the tail entry is short. That overrun is
            # invisible until the page after it is not mapped.
            header = bytes((ctypes.c_ubyte * _DIRENT_NAME_OFFSET).from_address(record))
            length = int.from_bytes(header[16:18], "little")
            if length <= _DIRENT_NAME_OFFSET or length > _DIRENT_RECORD_CEILING:
                _refuse()
            payload = bytes((ctypes.c_ubyte * length).from_address(record))
            entry_type = payload[_DIRENT_TYPE_OFFSET]
            name_bytes = payload[_DIRENT_NAME_OFFSET:].split(b"\x00", 1)[0]
            try:
                # STRICT, not surrogateescape. A POSIX name may be any
                # bytes, and surrogates survive decoding only to raise a
                # raw UnicodeEncodeError later inside the UTF-16 length
                # bound - the exact 'raw conversion error instead of this
                # layer's fixed refusal' that _require_entry_name already
                # warns about. A name this module cannot represent is one
                # it will not describe.
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError:
                _refuse()
            if name in (".", ".."):
                continue
            if found >= limit:
                _refuse()
            found += 1
            _raise_if_listing_stopped(should_stop)
            yield name, entry_type
    finally:
        libc.closedir(stream)


def _list_windows(handle: int, limit: int) -> list[AnchoredEntry]:
    """Collect the streaming Windows reader for compatibility with callers."""

    return list(_iter_windows_entries(handle, limit, None))


def _iter_windows_entries(
    handle: int,
    limit: int,
    should_stop: Callable[[], bool] | None,
) -> Generator[AnchoredEntry, None, None]:
    """Enumerate through the held handle with NtQueryDirectoryFile.

    One buffer, queried until STATUS_NO_MORE_FILES. Each record carries its own
    name and its own FileAttributes, so the kind is decided from the same bytes
    that carried the name.
    """

    api = _windows_api()
    buffer = api.ctypes.create_string_buffer(_DIRECTORY_QUERY_BUFFER)
    status_block = api.IoStatusBlock()
    found = 0
    restart = True
    while True:
        _raise_if_listing_stopped(should_stop)
        status = api.ntdll.NtQueryDirectoryFile(
            api.ctypes.c_void_p(handle),
            None,
            None,
            None,
            api.ctypes.byref(status_block),
            buffer,
            api.ctypes.c_ulong(_DIRECTORY_QUERY_BUFFER),
            api.ctypes.c_ulong(_FILE_DIRECTORY_INFORMATION_CLASS),
            api.ctypes.c_ubyte(0),
            None,
            api.ctypes.c_ubyte(1 if restart else 0),
        )
        _raise_if_listing_stopped(should_stop)
        restart = False
        masked = status & 0xFFFFFFFF
        if masked == _STATUS_NO_MORE_FILES:
            return
        if masked != _STATUS_SUCCESS:
            _refuse()
        records = _iter_directory_records(buffer.raw, limit, found, should_stop)
        with contextlib.closing(records):
            for entry in records:
                found += 1
                yield entry


def _read_directory_records(raw: bytes, limit: int, already: int) -> list[AnchoredEntry]:
    """Collect one streaming Windows record buffer for compatibility."""

    return list(_iter_directory_records(raw, limit, already, None))


def _iter_directory_records(
    raw: bytes,
    limit: int,
    already: int,
    should_stop: Callable[[], bool] | None,
) -> Generator[AnchoredEntry, None, None]:
    """Walk one buffer of FILE_DIRECTORY_INFORMATION records."""

    found = 0
    offset = 0
    while True:
        _raise_if_listing_stopped(should_stop)
        if offset + _FILE_DIRECTORY_INFORMATION_HEADER > len(raw):
            _refuse()
        next_offset = int.from_bytes(raw[offset : offset + 4], "little")
        attributes = int.from_bytes(raw[offset + 56 : offset + 60], "little")
        name_length = int.from_bytes(raw[offset + 60 : offset + 64], "little")
        # Same record, same bytes. The size and the timestamp sit inside the
        # header this parser already reads for the attributes, so on Windows
        # they cost nothing and cannot describe a different entry than the
        # name does.
        written = int.from_bytes(
            raw[offset + _RECORD_LAST_WRITE_OFFSET : offset + _RECORD_LAST_WRITE_OFFSET + 8],
            "little",
            signed=True,
        )
        end_of_file = int.from_bytes(
            raw[offset + _RECORD_END_OF_FILE_OFFSET : offset + _RECORD_END_OF_FILE_OFFSET + 8],
            "little",
            signed=True,
        )
        start = offset + _FILE_DIRECTORY_INFORMATION_HEADER
        end = start + name_length
        if name_length % 2 or end > len(raw):
            _refuse()
        try:
            # The POSIX reader decodes strictly and refuses; this is its
            # mirror. An unpaired surrogate in the record's bytes raises here,
            # and an unguarded raise leaves this module through a codec error
            # instead of its fixed refusal - the same class, on the platform
            # the other fix did not touch.
            name = raw[start:end].decode("utf-16-le")
        except UnicodeDecodeError:
            _refuse()
        if name not in (".", ".."):
            if already + found >= limit:
                _refuse()
            kind = _windows_kind(attributes)
            size, modified = _windows_metadata(kind, end_of_file, written)
            found += 1
            _raise_if_listing_stopped(should_stop)
            yield AnchoredEntry(name, kind, size, modified)
        if next_offset == 0:
            return
        offset += next_offset


def _windows_metadata(
    kind: AnchoredEntryKind, end_of_file: int, written: int
) -> tuple[int | None, datetime | None]:
    """Size and modification time from the record, for safe kinds only.

    An unsafe kind carries no metadata even though the record supplies it. The
    bytes are there and are deliberately dropped: a link's size is the link's,
    not the target's, and reporting it invites a consumer to reason about an
    entry it has been told to skip.

    A negative size, or a timestamp the record did not record, leaves both
    absent - the pair is all-or-nothing so a consumer has one question to ask.
    """

    if kind not in (AnchoredEntryKind.FILE, AnchoredEntryKind.DIRECTORY):
        return None, None
    if end_of_file < 0:
        return None, None
    modified = _from_filetime(written)
    if modified is None:
        return None, None
    return end_of_file, modified


def _windows_kind(attributes: int) -> AnchoredEntryKind:
    """Classify from the attributes in the record, reparse point first.

    Order matters and is not cosmetic. A junction has BOTH the directory and
    the reparse-point attribute set, so testing for a directory first would
    report the one thing every caller of this module must not treat as a
    directory.
    """

    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    directory = int(getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0x10))
    if attributes & reparse:
        return AnchoredEntryKind.LINK
    if attributes & directory:
        return AnchoredEntryKind.DIRECTORY
    return AnchoredEntryKind.FILE


def _require_entry_name(name: str) -> None:
    """One entry inside the held directory - never a path, never a traversal."""

    if (
        not isinstance(name, str)
        or not name
        or len(name) >= _MAX_ENTRY_NAME
        # The native FileName field is 260 WIDE CHARACTERS, and the value is
        # assigned there after being measured in UTF-16. A name under 260 code
        # points can still exceed 260 UTF-16 units - any supplementary
        # character costs two - which overflowed the structure and surfaced a
        # raw conversion error instead of this layer's fixed refusal. Bounded
        # on every platform so a name is not accepted on one and refused on
        # the other.
        or _utf16_length(name) >= _MAX_ENTRY_NAME * 2
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        _refuse()


def _walk_posix(path: Path, *, create: bool = False) -> list[int]:
    """Open each component relative to the last, refusing any symlink.

    Every descriptor is retained, not just the leaf.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parts = path.parts
    chain: list[int] = []
    try:
        chain.append(os.open(parts[0], flags))
        for index, component in enumerate(parts[1:], start=1):
            last = index == len(parts) - 1
            if last and create:
                # Creating the leaf through its parent's descriptor is the
                # point: a path-based mkdir would build directories inside
                # whatever the path means by now.
                with contextlib.suppress(FileExistsError):
                    os.mkdir(component, 0o700, dir_fd=chain[-1])
            chain.append(os.open(component, flags, dir_fd=chain[-1]))
    except FileNotFoundError:
        for held in reversed(chain):
            with contextlib.suppress(OSError):
                os.close(held)
        # O_NOFOLLOW makes a symlink fail with ELOOP even when it dangles, and
        # O_DIRECTORY makes a link-to-file fail with ENOTDIR, so ENOENT here
        # means the entry genuinely is not there rather than that we declined
        # to follow something.
        raise AnchoredDirectoryNotFound(CONTAINMENT_REFUSED) from None
    except OSError:
        for held in reversed(chain):
            with contextlib.suppress(OSError):
                os.close(held)
        _refuse()
    return chain


def _walk_windows(path: Path, *, create: bool = False) -> list[int]:
    """Walk the chain handle-relative, refusing a reparse point at any depth."""

    parts = path.parts
    # The volume root is the one name with no handle to hang it on, so it is
    # the one name given to the object manager directly - and it must be in
    # the NT namespace: measured, "C:\\" is STATUS_OBJECT_PATH_SYNTAX_BAD and
    # "\\??\\C:" is STATUS_ACCESS_DENIED, while "\\??\\C:\\" opens.
    chain = [_nt_open_relative(None, f"{_NT_NAMESPACE}{parts[0]}", intent="open_dir")]
    try:
        for index, component in enumerate(parts[1:], start=1):
            # Validate the component BEFORE the native open, not inside it.
            # This walk is the one caller that reaches _nt_open_relative
            # without going through the entry validator first, so a name the
            # object manager cannot hold would otherwise be handed to it and
            # refused by the kernel rather than by this layer. The volume root
            # is deliberately outside this loop: it carries separators and is
            # the one name given to the object manager directly.
            _require_entry_name(component)
            last = index == len(parts) - 1
            # Only the leaf may be created, and only through its parent's
            # handle. Creating an ancestor would mean deciding, by path, that
            # a directory the caller never named should exist.
            intent = "create_dir" if last and create else "open_dir"
            chain.append(_nt_open_relative(chain[-1], component, intent=intent))
            if _nt_is_reparse(chain[-1]):
                _refuse()
    except AnchoredDirectoryError:
        for held in reversed(chain):
            _close_windows_handle(held)
        raise
    return chain


def _windows_api() -> Any:
    """Bind ntdll and the NT structures.

    Widened to Any deliberately: this module is type-checked for POSIX too,
    where ctypes has no WinDLL at all, and a platform-conditional ignore is
    itself an error on the platform that does have it.
    """

    import ctypes
    import types
    from ctypes import wintypes

    windows: Any = ctypes

    class UnicodeString(ctypes.Structure):
        _fields_ = (
            ("Length", ctypes.c_ushort),
            ("MaximumLength", ctypes.c_ushort),
            ("Buffer", ctypes.c_wchar_p),
        )

    class ObjectAttributes(ctypes.Structure):
        _fields_ = (
            ("Length", ctypes.c_ulong),
            ("RootDirectory", ctypes.c_void_p),
            ("ObjectName", ctypes.POINTER(UnicodeString)),
            ("Attributes", ctypes.c_ulong),
            ("SecurityDescriptor", ctypes.c_void_p),
            ("SecurityQualityOfService", ctypes.c_void_p),
        )

    class IoStatusBlock(ctypes.Structure):
        _fields_ = (("Status", ctypes.c_void_p), ("Information", ctypes.c_void_p))

    class FileBasicInformation(ctypes.Structure):
        _fields_ = (
            ("CreationTime", ctypes.c_longlong),
            ("LastAccessTime", ctypes.c_longlong),
            ("LastWriteTime", ctypes.c_longlong),
            ("ChangeTime", ctypes.c_longlong),
            ("FileAttributes", ctypes.c_ulong),
            ("Reserved", ctypes.c_ulong),
        )

    class FileFsSizeInformation(ctypes.Structure):
        _fields_ = (
            ("TotalAllocationUnits", ctypes.c_longlong),
            ("AvailableAllocationUnits", ctypes.c_longlong),
            ("SectorsPerAllocationUnit", ctypes.c_ulong),
            ("BytesPerSector", ctypes.c_ulong),
        )

    class FileNameInformation(ctypes.Structure):
        # BOOLEAN then HANDLE: the seven bytes of padding are the 64-bit
        # layout the kernel expects, not decoration. The link and rename
        # classes share this shape.
        _fields_ = (
            ("ReplaceIfExists", ctypes.c_ubyte),
            ("Padding", ctypes.c_ubyte * 7),
            ("RootDirectory", ctypes.c_void_p),
            ("FileNameLength", ctypes.c_ulong),
            ("FileName", ctypes.c_wchar * _MAX_ENTRY_NAME),
        )

    return types.SimpleNamespace(
        ctypes=ctypes,
        wintypes=wintypes,
        ntdll=windows.WinDLL("ntdll", use_last_error=True),
        UnicodeString=UnicodeString,
        ObjectAttributes=ObjectAttributes,
        IoStatusBlock=IoStatusBlock,
        FileBasicInformation=FileBasicInformation,
        FileFsSizeInformation=FileFsSizeInformation,
        FileNameInformation=FileNameInformation,
    )


def _utf16_length(name: str) -> int:
    """Bytes this name occupies as UTF-16, which is what NT counts.

    Python's len() counts code points, so anything outside the basic plane is
    two bytes short per character - and a short length makes the object
    manager silently truncate the name, which then resolves to a DIFFERENT
    entry while every path-based check still passes.

    REFUSES a name that cannot be encoded at all, rather than measuring it or
    answering a sentinel. A lone surrogate - which is what any filename that is
    not valid UTF-8 becomes under surrogateescape - raises inside encode, and
    the raw UnicodeEncodeError would escape this module in place of its fixed
    refusal.

    Refusing here rather than returning a large number is the whole point, and
    the reason is the caller set. This helper has three callers and only ONE of
    them is the name validator: `_nt_try_open_relative` forwards the result
    straight into UNICODE_STRING.Length and MaximumLength, and `_nt_set_name`
    into FileNameLength. A sentinel would be presented to the object manager as
    a real buffer length for a name it cannot hold - which is the truncation
    hazard this function exists to prevent, reintroduced by its own guard.
    Refusal happens before any native field is populated.
    """

    try:
        return len(name.encode("utf-16-le"))
    except UnicodeEncodeError:
        _refuse()


def _nt_open_relative(parent: int | None, name: str, *, intent: str) -> int:
    handle, status = _nt_try_open_relative(parent, name, intent=intent)
    if status == _STATUS_OBJECT_NAME_COLLISION:
        raise AnchoredEntryExists(CONTAINMENT_REFUSED) from None
    if status in (_STATUS_OBJECT_NAME_NOT_FOUND, _STATUS_OBJECT_PATH_NOT_FOUND):
        # Every open here sets FILE_OPEN_REPARSE_POINT, so a junction - dangling
        # or not - opens AS the link and is refused by the identity check that
        # follows. These statuses therefore cannot come from a redirection we
        # declined to follow; they mean the entry genuinely is not there.
        raise AnchoredDirectoryNotFound(CONTAINMENT_REFUSED) from None
    if status != _STATUS_SUCCESS or not handle:
        _refuse()
    return handle


def _nt_try_open_relative(parent: int | None, name: str, *, intent: str) -> tuple[int, int]:
    """Open or create one entry relative to a held handle, reporting status.

    The status is returned rather than raised so callers can distinguish an
    absent entry and a name collision from a genuine failure. Everything that
    does not need that distinction goes through _nt_open_relative.

    `intent` is one of open_dir, create_dir, open_file, create_file,
    delete_directory or rename_source. It is spelled out rather than inferred from a flag because
    the access mask and the disposition have to agree, and getting that pair
    wrong fails in ways that look like a filesystem problem rather than a
    coding one.
    """

    api = _windows_api()
    unicode_name = api.UnicodeString()
    unicode_name.Buffer = name
    encoded_length = _utf16_length(name)
    unicode_name.Length = encoded_length
    unicode_name.MaximumLength = encoded_length

    attributes = api.ObjectAttributes()
    attributes.Length = api.ctypes.sizeof(api.ObjectAttributes)
    attributes.RootDirectory = api.ctypes.c_void_p(parent) if parent else None
    attributes.ObjectName = api.ctypes.pointer(unicode_name)
    attributes.Attributes = _OBJ_CASE_INSENSITIVE
    attributes.SecurityDescriptor = None
    attributes.SecurityQualityOfService = None

    access = _FILE_READ_ATTRIBUTES | _SYNCHRONIZE
    options = _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT
    if intent in ("open_dir", "create_dir"):
        access |= _FILE_LIST_DIRECTORY | _FILE_TRAVERSE
        options |= _FILE_DIRECTORY_FILE
        disposition = _FILE_OPEN_IF if intent == "create_dir" else _FILE_OPEN
    elif intent == "create_file":
        access |= _FILE_WRITE_DATA
        options |= _FILE_NON_DIRECTORY_FILE
        disposition = _FILE_CREATE
    elif intent == "open_file":
        access |= _FILE_READ_DATA
        options |= _FILE_NON_DIRECTORY_FILE
        disposition = _FILE_OPEN
    elif intent == "delete_directory":
        # A directory this time, so FILE_DIRECTORY_FILE rather than its
        # opposite - which also makes a file refuse with STATUS_NOT_A_DIRECTORY
        # instead of being opened and then rejected by a second check.
        access |= _DELETE
        options |= _FILE_DIRECTORY_FILE
        disposition = _FILE_OPEN
    elif intent in ("delete_source", "rename_source"):
        # Both change a directory ENTRY rather than file contents, which is
        # what DELETE authorises. Kept off the read path, which has no
        # business holding delete rights.
        access |= _DELETE
        options |= _FILE_NON_DIRECTORY_FILE
        disposition = _FILE_OPEN
    else:  # pragma: no cover - the caller set is closed
        _refuse()

    handle = api.wintypes.HANDLE()
    status_block = api.IoStatusBlock()
    status = api.ntdll.NtCreateFile(
        api.ctypes.byref(handle),
        api.ctypes.c_ulong(access),
        api.ctypes.byref(attributes),
        api.ctypes.byref(status_block),
        None,
        api.ctypes.c_ulong(0),
        api.ctypes.c_ulong(_FILE_SHARE_READ | _FILE_SHARE_WRITE),
        api.ctypes.c_ulong(disposition),
        api.ctypes.c_ulong(options),
        None,
        api.ctypes.c_ulong(0),
    )
    masked = status & 0xFFFFFFFF
    if masked != _STATUS_SUCCESS:
        return 0, masked
    if not handle.value:
        return 0, _STATUS_UNEXPECTED
    return int(handle.value), _STATUS_SUCCESS


def _nt_is_reparse(handle: int) -> bool:
    """True when the held object carries a reparse point, or cannot be read."""

    api = _windows_api()
    information = api.FileBasicInformation()
    status_block = api.IoStatusBlock()
    status = api.ntdll.NtQueryInformationFile(
        api.ctypes.c_void_p(handle),
        api.ctypes.byref(status_block),
        api.ctypes.byref(information),
        api.ctypes.c_ulong(api.ctypes.sizeof(api.FileBasicInformation)),
        api.ctypes.c_ulong(_FILE_BASIC_INFORMATION_CLASS),
    )
    if status & 0xFFFFFFFF != _STATUS_SUCCESS:
        # Fail closed. The handle carries FILE_READ_ATTRIBUTES, so a failure
        # here is real rather than a missing access right - and a silent True
        # would refuse every path while looking exactly like a working guard.
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(information.FileAttributes & reparse_flag)


def _nt_set_name(
    source: int,
    directory: int,
    name: str,
    information_class: int,
    *,
    replace: bool = False,
) -> bool:
    """Link or rename `source` to `name` inside the held directory.

    False means the destination already existed and `replace` was not asked
    for, which callers racing to publish the same entry treat as convergence.
    Any other status refuses, because the operation did not happen and
    pretending otherwise would report an entry nobody can read back.
    """

    api = _windows_api()
    information = api.FileNameInformation()
    information.ReplaceIfExists = 1 if replace else 0
    information.RootDirectory = api.ctypes.c_void_p(directory)
    information.FileNameLength = _utf16_length(name)
    information.FileName = name

    status_block = api.IoStatusBlock()
    status = api.ntdll.NtSetInformationFile(
        api.ctypes.c_void_p(source),
        api.ctypes.byref(status_block),
        api.ctypes.byref(information),
        api.ctypes.c_ulong(api.ctypes.sizeof(api.FileNameInformation)),
        api.ctypes.c_ulong(information_class),
    )
    masked = status & 0xFFFFFFFF
    if masked == _STATUS_SUCCESS:
        return True
    if masked == _STATUS_OBJECT_NAME_COLLISION:
        return False
    _refuse()


def _nt_mark_deleted(handle: int) -> None:
    """Delete the entry this handle refers to, by disposition rather than name."""

    api = _windows_api()
    disposition = api.ctypes.c_ubyte(1)
    status_block = api.IoStatusBlock()
    status = api.ntdll.NtSetInformationFile(
        api.ctypes.c_void_p(handle),
        api.ctypes.byref(status_block),
        api.ctypes.byref(disposition),
        api.ctypes.c_ulong(api.ctypes.sizeof(disposition)),
        api.ctypes.c_ulong(_FILE_DISPOSITION_INFORMATION_CLASS),
    )
    if status & 0xFFFFFFFF != _STATUS_SUCCESS:
        _refuse()


def available_bytes(anchor: AnchoredDirectory) -> int:
    """Free bytes on the volume holding the directory the caller HOLDS.

    Capacity is a volume property, which is exactly why it has to be asked
    through the handle. A pathname query answers for whatever that name means
    when it is read, so a report whose other fields describe the held
    directory could carry a capacity measured on a different volume - one
    result describing two identities.
    """

    if anchor.descriptor is not None:
        fstatvfs = getattr(os, "fstatvfs", None)
        if fstatvfs is None:  # pragma: no cover - POSIX always has it
            _refuse()
        try:
            status = fstatvfs(anchor.descriptor)
        except OSError:
            _refuse()
        # f_bavail is what an unprivileged writer may actually use, which is
        # the honest number here; f_bfree counts reserved blocks it cannot.
        return int(status.f_bavail) * int(status.f_frsize)

    handle = anchor.handle
    if handle is None:
        _refuse()
    api = _windows_api()
    information = api.FileFsSizeInformation()
    status_block = api.IoStatusBlock()
    status = api.ntdll.NtQueryVolumeInformationFile(
        api.ctypes.c_void_p(handle),
        api.ctypes.byref(status_block),
        api.ctypes.byref(information),
        api.ctypes.c_ulong(api.ctypes.sizeof(information)),
        api.ctypes.c_ulong(_FILE_FS_SIZE_INFORMATION_CLASS),
    )
    if status & 0xFFFFFFFF != _STATUS_SUCCESS:
        _refuse()
    return (
        int(information.AvailableAllocationUnits)
        * int(information.SectorsPerAllocationUnit)
        * int(information.BytesPerSector)
    )


def sync_directory(anchor: AnchoredDirectory) -> None:
    """Durably commit the held directory's entries.

    fsyncing a published file leaves the directory entry itself unfenced. POSIX
    exposes a descriptor for this; Windows has no directory sync for these
    calls and journals the metadata operation instead, so it is a no-op there
    rather than a silent gap.
    """

    if anchor.descriptor is None:
        return
    try:
        os.fsync(anchor.descriptor)
    except OSError:
        _refuse()


def _descriptor_from_handle(handle: int) -> int:
    """Hand a native handle to the C runtime, which then owns it.

    Closing the returned descriptor closes the handle, so the caller must not
    also close it - a double close can land on a handle the process has since
    reused for something else.
    """

    import msvcrt

    windows: Any = msvcrt
    return int(windows.open_osfhandle(handle, getattr(os, "O_BINARY", 0)))


def _handle_from_descriptor(descriptor: int) -> int:
    """Recover the native handle the C runtime already owns.

    The descriptor still owns the handle. Do not CloseHandle the result.
    """

    import msvcrt

    windows: Any = msvcrt
    return int(windows.get_osfhandle(descriptor))


def _close_windows_handle(handle: int) -> None:
    import ctypes

    windows: Any = ctypes
    with contextlib.suppress(OSError, AttributeError):
        windows.WinDLL("kernel32", use_last_error=True).CloseHandle(ctypes.c_void_p(handle))
