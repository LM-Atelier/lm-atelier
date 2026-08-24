from __future__ import annotations

import contextlib
import os
import stat
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


def discard_entry(anchor: AnchoredDirectory, name: str) -> None:
    """Best-effort removal, for use while a refusal is already propagating.

    Separate from remove_entry on purpose: suppressing a cleanup failure is
    right when something has already gone wrong and wrong when it has not.
    """

    with contextlib.suppress(AnchoredDirectoryError, OSError):
        remove_entry(anchor, name)


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
    """

    return len(name.encode("utf-16-le"))


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

    `intent` is one of open_dir, create_dir, open_file, create_file or
    rename_source. It is spelled out rather than inferred from a flag because
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


def _close_windows_handle(handle: int) -> None:
    import ctypes

    windows: Any = ctypes
    with contextlib.suppress(OSError, AttributeError):
        windows.WinDLL("kernel32", use_last_error=True).CloseHandle(ctypes.c_void_p(handle))
