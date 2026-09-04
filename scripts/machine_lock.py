"""One kernel-held handle for the one machine-exclusive resource this repo has.

The full verification gate is machine-exclusive: a package-wide type check or
a mutation battery running beside it does not merely slow it down, it can
hang it outright. The exclusion here is a HANDLE, not a record:

- ``acquire`` OPENS the lease file with write access while sharing only
  reads. The kernel enforces the exclusion: a second write-access open fails
  with a sharing violation while any holder is alive. There is nothing to
  renew and nothing to recover - a handle dies with its owner, and the
  machine frees the instant the last holder is gone.
- The handle is INHERITABLE. Children launched by the holder inherit it, so
  the exclusion lives exactly as long as the process tree that can still
  act: a gate whose own process dies mid-stage keeps the machine held while
  the stage child it launched is still running. A grandchild spawned with
  handle inheritance disabled ends before the stage child that owns it in a
  synchronous gate; a deliberately detached daemon is outside this contract.
- The bytes inside the file are DIAGNOSTICS - pid, purpose, acquired_at:
  who to go look at - never authority. Reading them is always allowed;
  taking the machine requires the kernel to grant the write-access open.

The lease file lives beside the repository's COMMON git directory, so the
primary checkout and every linked worktree resolve the same lease with no
configuration. The repository is identified by an explicit ``--repo``
argument or by this module's own location - never the launch cwd - and git
runs with its redirection environment scrubbed, so inherited GIT_DIR state
cannot point the lease at a different repository. The repository directory
is HELD as an object before any name is resolved from it: the common
directory is resolved through the held object's current name, opened and
held in turn, both are read back from their handles after every step that
used a name, and the resolution is repeated through the held repository
after the lease open. A holder keeps that binding and re-verifies it at
every barrier, so a repository whose common directory is replaced under a
holder refuses that holder its next stage rather than letting two holders
proceed under two directories.

Scope honesty: this excludes the tools of ONE repository's worktrees on
one machine. Two independent clones do not exclude each other.

Exit codes: 0 success; 2 refused; 3 usage; 4 stranded - this process may
still hold the machine through a handle it could not close, and the host
must exit rather than continue as if unleased. ``hold`` exits 0 only after
a clean release, and 2 when the acquisition is refused, the binding is lost
while holding, or the release did not complete cleanly.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LEASE_BASENAME = "machine-exclusive.lease"
_REFUSED = 2
_USAGE = 3
_STRANDED = 4
_REVERIFY_SECONDS = 2.0
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_FILE_READ_ATTRIBUTES = 0x80
_SHARE_READ = 0x1
_SHARE_ALL = 0x7
_OPEN_ALWAYS = 4
_OPEN_EXISTING = 3
_ATTRIBUTE_NORMAL = 0x80
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
_HANDLE_FLAG_INHERIT = 0x1
_ERROR_SHARING_VIOLATION = 32
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_SCRUBBED_ENV = (
    "GIT_DIR",
    "GIT_COMMON_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_PREFIX",
)


class LeaseStranded(RuntimeError):
    """This process may hold the machine through a handle it could not close.

    ``kind`` names the object that would not close: ``"descriptor"`` (the C
    runtime descriptor that owned the handle), ``"handle"`` (the raw kernel
    handle before any descriptor owned it) or ``"probe"`` (the status
    probe's write-access handle). ``number`` is that descriptor or handle
    value and ``error`` the Win32 error the close returned. A strand raised
    during an acquisition - a failed initialization or a refused binding -
    is raised FROM that failure, so ``__cause__`` is the error that aborted
    it; a probe strand has no acquisition behind it and no cause. A host
    that sees this must not continue as if it were unleased: the exclusion
    stands until this process exits.

    ``strands`` lists every close the cleanup attempted and the kernel
    refused, the first of them being ``kind``/``number``/``error``: a
    boundary closes each acquired object exactly once and reports all of
    them, so a refused pin behind a refused descriptor is named too.
    """

    def __init__(
        self,
        kind: str,
        number: int,
        error: int,
        *,
        during: str,
        others: tuple[tuple[str, int, int], ...] = (),
    ) -> None:
        self.kind = kind
        self.number = number
        self.error = error
        self.strands = ((kind, number, error), *others)
        detail = ""
        if others:
            detail = "; every refused close: " + ", ".join(
                f"{k} {n} (error {e})" for k, n, e in self.strands
            )
        super().__init__(
            f"the lease {kind} {number} could not be closed after {during} "
            f"(error {error}); this process may still hold the machine until it exits"
            + detail
        )


class LeaseRefused(RuntimeError):
    """Fail-closed refusal: nothing is held afterwards.

    The attempt may have created the lease file - the open uses OPEN_ALWAYS -
    but that file is a diagnostic record, never authority, and no handle
    survives the refusal.
    """


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class _ByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("attributes", ctypes.c_uint32),
        ("creation", _FileTime),
        ("last_access", _FileTime),
        ("last_write", _FileTime),
        ("volume_serial", ctypes.c_uint32),
        ("size_high", ctypes.c_uint32),
        ("size_low", ctypes.c_uint32),
        ("links", ctypes.c_uint32),
        ("index_high", ctypes.c_uint32),
        ("index_low", ctypes.c_uint32),
    ]


_KERNEL32: Any = None


def _kernel32() -> Any:
    """kernel32 bound with per-call error capture.

    ``use_last_error=True`` copies GetLastError immediately after every call
    through this binding, so ``ctypes.get_last_error()`` read right after a
    call is that call's own result - never a stale copy left by an earlier
    call through some other binding.
    """

    global _KERNEL32
    if _KERNEL32 is None:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        kernel.CloseHandle.restype = ctypes.c_int
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.SetHandleInformation.restype = ctypes.c_int
        kernel.SetHandleInformation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        kernel.GetFileInformationByHandle.restype = ctypes.c_int
        kernel.GetFileInformationByHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel.GetFinalPathNameByHandleW.restype = ctypes.c_uint32
        kernel.GetFinalPathNameByHandleW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        _KERNEL32 = kernel
    return _KERNEL32


def _now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _scrubbed_git_env() -> dict[str, str]:
    environment = dict(os.environ)
    for name in _SCRUBBED_ENV:
        environment.pop(name, None)
    return environment


def _repo_root_default() -> Path:
    # The module's own location, never the launch cwd: a tool started from
    # inside some other repository must not point the machine lease there.
    return Path(__file__).resolve().parent


def _common_dir(repo: Path | None) -> Path:
    start = repo if repo is not None else _repo_root_default()
    if not isinstance(start, Path) or not Path(start).is_absolute():
        raise LeaseRefused(f"--repo must be an absolute path: {start}")
    try:
        # git gets its own, already-closed stdin rather than this process's:
        # a hold drains its stdin on another thread, and spawning a child
        # that inherits that pipe while the read is pending blocks the
        # spawn instead of answering.
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(start),
            input="",
            capture_output=True,
            text=True,
            check=True,
            env=_scrubbed_git_env(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LeaseRefused(f"{start} does not resolve a git repository") from exc
    common = Path(result.stdout.strip())
    if (
        not common.is_absolute()
        or not common.is_dir()
        or not (common / "HEAD").is_file()
        or not (common / "config").is_file()
    ):
        raise LeaseRefused(f"resolved common dir is not a git dir: {common}")
    return common


def lease_path(repo: Path | None = None) -> Path:
    return _common_dir(repo) / LEASE_BASENAME


def _identity(handle: int) -> tuple[int, int, int]:
    """(volume serial, file index high, file index low) of a held object."""

    information = _ByHandleFileInformation()
    if not _kernel32().GetFileInformationByHandle(
        ctypes.c_void_p(handle), ctypes.byref(information)
    ):
        raise LeaseRefused(
            f"the held object's identity could not be read (error {ctypes.get_last_error()})"
        )
    return (information.volume_serial, information.index_high, information.index_low)


def _final_path(handle: int) -> str:
    """The held object's current name, from the handle, never from a lookup."""

    buffer = ctypes.create_unicode_buffer(32768)
    length = _kernel32().GetFinalPathNameByHandleW(
        ctypes.c_void_p(handle), buffer, len(buffer), 0
    )
    if length == 0 or length >= len(buffer):
        raise LeaseRefused(
            f"the held directory's name could not be read (error {ctypes.get_last_error()})"
        )
    return buffer.value


def _plain(name: str) -> str:
    """A final name without the long-path prefix, for git and path checks."""

    if name.startswith("\\\\?\\UNC\\"):
        return "\\\\" + name[8:]
    if name.startswith("\\\\?\\"):
        return name[4:]
    return name


def _open_directory(path: Path) -> int:
    handle = _kernel32().CreateFileW(
        str(path),
        _FILE_READ_ATTRIBUTES,
        _SHARE_ALL,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )
    if handle is None or handle == _INVALID_HANDLE:
        raise LeaseRefused(
            f"the directory could not be held: {path} (error {ctypes.get_last_error()})"
        )
    return int(handle)


def _close(handle: int) -> bool:
    return bool(_kernel32().CloseHandle(ctypes.c_void_p(handle)))


@dataclass(frozen=True)
class _Pin:
    """One link of the checkout's resolution chain, held read-share only for
    the lease lifetime: the .git entry under the repository directory, for a
    pointer the private git directory it names and that directory's
    commondir file, and every reparse point on the way to either. While a
    pin lives, any open that would write, rename, delete or retarget the
    link is refused by the kernel, so what the repository names cannot
    change under the holder or under a child. A pin is inheritable exactly
    as the lease is: a stage child launched while the lease is held carries
    every pin for as long as it can act, so the chain stays held for the
    child's lifetime even when the holder dies first."""

    path: str
    role: str
    handle: int


def _abandon(
    *,
    during: str,
    descriptor: int = -1,
    handles: tuple[tuple[str, int], ...] = (),
    pins: tuple[_Pin, ...] = (),
) -> LeaseStranded | None:
    """Close everything a boundary acquired, each exactly once: the C
    runtime descriptor first (it owns its handle), then every raw handle,
    then every pin. Nothing is skipped because an earlier close refused;
    the result names every close the kernel refused, or is None when all
    of them closed. The caller raises it from the failure that reached the
    boundary, so the primary failure and each cleanup failure survive."""

    strands: list[tuple[str, int, int]] = []
    cause: BaseException | None = None
    if descriptor != -1:
        try:
            os.close(descriptor)
        except OSError as refusal:
            strands.append(("descriptor", descriptor, refusal.errno or 0))
            cause = refusal
    for kind, number in handles:
        if not _close(number):
            strands.append((kind, number, ctypes.get_last_error()))
    for pin in pins:
        if not _close(pin.handle):
            strands.append(("pin", pin.handle, ctypes.get_last_error()))
    if not strands:
        return None
    (kind, number, error), *others = strands
    strand = LeaseStranded(kind, number, error, during=during, others=tuple(others))
    strand.__cause__ = cause
    return strand


@dataclass(frozen=True)
class _Binding:
    """The repository object held before resolution, the common directory
    it named, and the pins on every link between them: the identities a
    holder re-verifies at every barrier, and the holds that make the
    re-verification a check rather than the mechanism."""

    anchor: str
    anchor_identity: tuple[int, int, int]
    common: str
    common_identity: tuple[int, int, int]
    pins: tuple[_Pin, ...] = ()


def _directory_identity(path: Path) -> tuple[int, int, int]:
    handle = _open_directory(path)
    try:
        return _identity(handle)
    finally:
        _close(handle)


def _attributes(path: Path) -> int:
    attributes = _kernel32().GetFileAttributesW(str(path))
    if attributes == _INVALID_FILE_ATTRIBUTES:
        raise LeaseRefused(
            f"the resolution chain could not be read: {path} (error {ctypes.get_last_error()})"
        )
    return int(attributes)


def _reparse_links(
    path: Path, role: str, *, itself: bool = False
) -> list[tuple[Path, str]]:
    """Every reparse point among the components of a textual path, root
    first - and the path itself when ``itself`` is set and it is one. Git
    resolves the text at each invocation, so retargeting one of these
    changes what the text names while the object at its end stays held;
    each is pinned as itself. ``..`` components collapse lexically, as
    Win32 collapses them before the kernel sees the name."""

    plain = Path(os.path.normpath(str(path)))
    components = list(reversed(plain.parents))[1:]
    if itself:
        components.append(plain)
    links: list[tuple[Path, str]] = []
    for prefix in components:
        if _attributes(prefix) & _FILE_ATTRIBUTE_REPARSE_POINT:
            links.append((prefix, f"a link on the way to {role}"))
    return links


def _named_directory(base: Path, file: Path) -> Path:
    """The directory a commondir file names, as git reads it: its text,
    relative to the directory holding the file."""

    text = file.read_text(encoding="utf-8").strip()
    named = Path(text)
    return named if named.is_absolute() else base / named


def _resolution_chain(anchor: Path) -> list[tuple[Path, str]]:
    """The links git follows from the repository directory to its common
    directory, each a path whose change would change what the repository
    names: the .git entry; for a pointer file, the private git directory it
    names and that directory's commondir file; for a directory, its
    commondir file if it has one; and every reparse point among the
    components of the paths the pointer and the commondir file name. A
    reparse point is a link as itself."""

    entry = anchor / ".git"
    links = [(entry, "the repository's .git entry")]
    attributes = _attributes(entry)
    if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        return links
    if attributes & _FILE_ATTRIBUTE_DIRECTORY:
        commondir = entry / "commondir"
        if commondir.is_file():
            links.append((commondir, "the commondir file"))
            links.extend(
                _reparse_links(
                    _named_directory(entry, commondir),
                    "the common git directory",
                    itself=True,
                )
            )
        return links
    text = entry.read_text(encoding="utf-8")
    if not text.startswith("gitdir:"):
        raise LeaseRefused(f"the .git file is not a git pointer: {entry}")
    target = Path(text[len("gitdir:") :].strip())
    private = target if target.is_absolute() else anchor / target
    links.extend(_reparse_links(private, "the checkout's private git directory"))
    links.append((private, "the checkout's private git directory"))
    commondir = private / "commondir"
    if commondir.is_file():
        links.append((commondir, "the commondir file"))
        links.extend(
            _reparse_links(
                _named_directory(private, commondir),
                "the common git directory",
                itself=True,
            )
        )
    return links


def _open_pin(path: Path, role: str) -> _Pin:
    """Hold one link against change: read share only. A directory is held
    with backup semantics, a reparse point as itself rather than through
    its target, and a file for reading; the kernel then refuses every other
    open that would write, rename, delete or retarget the link."""

    attributes = _attributes(path)
    flags = _ATTRIBUTE_NORMAL
    if attributes & _FILE_ATTRIBUTE_DIRECTORY:
        flags = _FILE_FLAG_BACKUP_SEMANTICS
    if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        flags |= _FILE_FLAG_OPEN_REPARSE_POINT
    handle = _kernel32().CreateFileW(
        str(path),
        _FILE_READ_ATTRIBUTES | _GENERIC_READ,
        _SHARE_READ,
        None,
        _OPEN_EXISTING,
        flags,
        None,
    )
    if handle is None or handle == _INVALID_HANDLE:
        error = ctypes.get_last_error()
        if error == _ERROR_SHARING_VIOLATION:
            raise LeaseRefused(f"{role} is open for writing elsewhere: {path}")
        raise LeaseRefused(f"{role} could not be held: {path} (error {error})")
    handle = int(handle)
    marked = _kernel32().SetHandleInformation(
        ctypes.c_void_p(handle), _HANDLE_FLAG_INHERIT, _HANDLE_FLAG_INHERIT
    )
    if not marked:
        refusal = LeaseRefused(
            f"{role} could not be held for a child's lifetime: {path} "
            f"(error {ctypes.get_last_error()})"
        )
        strand = _abandon(during="a refused pinning", handles=(("pin", handle),))
        if strand is not None:
            raise strand from refusal
        raise refusal
    return _Pin(str(path), role, handle)


def _open_pins(anchor: Path) -> tuple[_Pin, ...]:
    """Pin every link of the chain; a link that cannot be pinned lets the
    pins already taken go, each closed exactly once, and a close the kernel
    refuses is a strand raised from the refusal."""

    pins: list[_Pin] = []
    try:
        for path, role in _resolution_chain(anchor):
            pins.append(_open_pin(path, role))
    except BaseException as refusal:
        strand = _abandon(during="a refused pinning", pins=tuple(pins))
        if strand is not None:
            raise strand from refusal
        raise
    return tuple(pins)


def _hold_common_dir(repo: Path | None) -> tuple[int, _Binding]:
    """Hold the repository, then the common directory it names.

    The repository directory is opened as an object first; its identity and
    current name come from that handle; the common directory is resolved
    through that name, opened and read back the same way; the repository is
    then re-read to prove it did not move while its name was used, and the
    resolution is repeated through it to prove the held common directory is
    still what the repository names. Returns the held common directory
    handle and the binding; the caller closes the handle.
    """

    start = repo if repo is not None else _repo_root_default()
    if not isinstance(start, Path) or not Path(start).is_absolute():
        raise LeaseRefused(f"--repo must be an absolute path: {start}")
    anchor = _open_directory(start)
    try:
        anchor_identity = _identity(anchor)
        anchor_name = _final_path(anchor)
        common = _common_dir(Path(_plain(anchor_name)))
        directory = _open_directory(common)
        try:
            identity = _identity(directory)
            name = _final_path(directory)
            if (
                _final_path(anchor) != anchor_name
                or _identity(anchor) != anchor_identity
            ):
                raise LeaseRefused(
                    "the repository directory moved while it was being resolved"
                )
            if _directory_identity(_common_dir(Path(_plain(anchor_name)))) != identity:
                raise LeaseRefused(
                    "the repository's common git directory changed while it was being held"
                )
            # Every link between the repository and the held common
            # directory is now pinned; the resolution is repeated once more
            # through the pinned chain, so a change slipped in before the
            # pins took hold is refused and nothing can change after them.
            pins = _open_pins(Path(_plain(anchor_name)))
            try:
                pins = (
                    *pins,
                    _open_pin(Path(_plain(name)), "the common git directory"),
                )
                resolved = _common_dir(Path(_plain(anchor_name)))
                if _directory_identity(resolved) != identity:
                    raise LeaseRefused(
                        "the repository's common git directory changed while it was being pinned"
                    )
            except BaseException as refusal:
                strand = _abandon(during="a refused acquisition", pins=pins)
                if strand is not None:
                    raise strand from refusal
                raise
        except BaseException:
            _close(directory)
            raise
        return directory, _Binding(anchor_name, anchor_identity, name, identity, pins)
    finally:
        _close(anchor)


def _open_lease_handle(
    repo: Path | None, *, access: int, disposition: int
) -> tuple[int, Path, _Binding]:
    """Open the lease file under the HELD common directory.

    The common directory is held through ``_hold_common_dir``, with every
    link of the resolution chain pinned; HEAD and config are checked under
    its held name; the lease is opened under that name; the directory is
    read back from its handle and the repository re-resolved after the
    open, and the binding is verified again after it. Once the pins hold,
    no link of the chain can be renamed, replaced or retargeted - the
    kernel refuses the opens that would do it - so what the repository
    names is fixed for as long as the lease lives. A refusal before or
    after the open closes what was acquired - the lease handle once it
    exists, then every pin - each exactly once, and a close the kernel
    refuses on that path is a strand raised from the refusal naming every
    refused close, never an ordinary refusal that hides a live handle.
    """

    directory, binding = _hold_common_dir(repo)
    handle = -1
    try:
        name = binding.common
        plain = _plain(name)
        if not Path(plain, "HEAD").is_file() or not Path(plain, "config").is_file():
            raise LeaseRefused(f"held common dir is not a git dir: {plain}")
        lease = Path(name, LEASE_BASENAME)
        opened = _kernel32().CreateFileW(
            str(lease),
            access,
            _SHARE_READ,
            None,
            disposition,
            _ATTRIBUTE_NORMAL,
            None,
        )
        if opened is None or opened == _INVALID_HANDLE:
            error = ctypes.get_last_error()
            if error == _ERROR_SHARING_VIOLATION:
                raise LeaseRefused(
                    f"contended: {_holder_line(Path(plain, LEASE_BASENAME))}"
                )
            raise LeaseRefused(f"the lease file could not be opened (error {error})")
        handle = int(opened)
        _assert_binding(binding)
        return handle, Path(plain, LEASE_BASENAME), binding
    except BaseException as refusal:
        strand = _abandon(
            during="a refused acquisition",
            handles=(("handle", handle),) if handle != -1 else (),
            pins=binding.pins,
        )
        if strand is not None:
            raise strand from refusal
        raise
    finally:
        _close(directory)


def _assert_binding(binding: _Binding) -> None:
    """The repository still names the held common directory.

    The repository directory is opened by the name it had when the lease was
    taken and must be the same object; the common directory resolved through
    it must be the object the lease lives in. Either mismatch means the
    repository moved under the holder, and the holder must not proceed as
    if it still held this repository's machine.
    """

    anchor = _open_directory(Path(_plain(binding.anchor)))
    try:
        if _identity(anchor) != binding.anchor_identity:
            raise LeaseRefused(
                f"the repository moved under the lease: {_plain(binding.anchor)} is another object"
            )
        resolved = _common_dir(Path(_plain(binding.anchor)))
    finally:
        _close(anchor)
    if _directory_identity(resolved) != binding.common_identity:
        raise LeaseRefused(
            "the repository moved under the lease: its common git directory is now "
            f"{resolved}, not the one held"
        )


def _holder_line(path: Path) -> str:
    """The recorded holder, read share-friendly; unreadable is still held."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            f"held by pid {payload.get('holder_pid')} "
            f"({payload.get('purpose')}) since {payload.get('acquired_at')}"
        )
    except (OSError, ValueError):
        return "held (record unreadable while the holder writes it)"


@dataclass
class AcquiredLease:
    """An open, inheritable write-access handle: the exclusion itself."""

    descriptor: int
    path: Path
    purpose: str
    binding: _Binding | None = None

    def close(self) -> None:
        """Close the descriptor and every pin; a failure is LOUD, never a
        success report.

        os.close deallocates the descriptor and asks the kernel to drop the
        handle; if that fails, this process may still hold the machine while
        claiming otherwise. The pins are still closed after it, each exactly
        once, and one strand naming every refused close propagates to the
        caller, raised from the descriptor's refusal when there was one.
        The descriptor and the pins are marked closed FIRST either way: the
        numbers are deallocated territory after a failed close, and
        retrying them could close a stranger's handle.
        """

        descriptor, self.descriptor = self.descriptor, -1
        pins: tuple[_Pin, ...] = ()
        if self.binding is not None and self.binding.pins:
            pins, self.binding = (
                self.binding.pins,
                _Binding(
                    self.binding.anchor,
                    self.binding.anchor_identity,
                    self.binding.common,
                    self.binding.common_identity,
                    (),
                ),
            )
        strand = _abandon(during="a release", descriptor=descriptor, pins=pins)
        if strand is not None:
            raise strand

    def assert_bound(self) -> None:
        """The barrier: the descriptor is still open, every pin still answers
        for the link it holds, and the repository still names the held
        common directory. The pins make the last a check of something the
        kernel already forbids from changing; a pin that no longer answers
        means this process let a link go, and the hold is not intact."""

        if self.descriptor == -1:
            raise LeaseRefused("the lease descriptor is closed")
        if self.binding is not None:
            for pin in self.binding.pins:
                if not _kernel32().GetFileInformationByHandle(
                    ctypes.c_void_p(pin.handle),
                    ctypes.byref(_ByHandleFileInformation()),
                ):
                    raise LeaseRefused(f"{pin.role} is no longer held: {pin.path}")
            _assert_binding(self.binding)


def acquire(
    purpose: str, *, repo: Path | None = None, holder_pid: int | None = None
) -> AcquiredLease:
    """Open the machine's one write-access handle, or refuse naming the holder.

    The open IS the acquisition. Success writes the diagnostic record through
    the held handle; refusal reads the record through the read share the
    holder always grants.
    """

    if not isinstance(purpose, str) or not purpose or len(purpose) > 120:
        raise LeaseRefused("purpose must be a short non-empty string")
    if holder_pid is not None and (type(holder_pid) is not int or holder_pid <= 0):
        raise LeaseRefused("holder_pid must be a positive integer")
    if os.name != "nt":
        raise LeaseRefused(
            "the machine lease is implemented for Windows only; "
            "this platform has no executed hold implementation"
        )
    import msvcrt

    handle, path, binding = _open_lease_handle(
        repo, access=_GENERIC_READ | _GENERIC_WRITE, disposition=_OPEN_ALWAYS
    )
    # Everything between the successful kernel open and the return runs
    # under a cleanup guard: the open IS the acquisition, so a failure in
    # initialization would otherwise leave a kernel handle no caller can
    # release - in a long-lived host the machine stays excluded with no
    # holder able to let go.
    descriptor = -1
    try:
        marked = _kernel32().SetHandleInformation(
            ctypes.c_void_p(handle), _HANDLE_FLAG_INHERIT, _HANDLE_FLAG_INHERIT
        )
        if not marked:
            raise LeaseRefused(
                f"the hold could not be marked inheritable (error {ctypes.get_last_error()})"
            )
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDWR)
        record = json.dumps(
            {
                "schema": 2,
                "purpose": purpose,
                "holder_pid": holder_pid if holder_pid is not None else os.getpid(),
                "acquired_at": _now_text(),
            },
            indent=2,
        )
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, record.encode("utf-8"))
    except BaseException as failure:
        # After open_osfhandle the descriptor owns the handle; before it,
        # the raw handle must be closed directly. That close and every pin's
        # close are each attempted exactly once, and every refusal is
        # reported ON TOP of the initialization failure rather than under
        # it: the caller must know both that the acquisition failed and
        # that this process may still hold the machine, or the checkout
        # binding, through what would not close.
        strand = _abandon(
            during="a failed acquisition",
            descriptor=descriptor,
            handles=(("handle", handle),) if descriptor == -1 else (),
            pins=binding.pins,
        )
        if strand is not None:
            raise strand from failure
        raise
    return AcquiredLease(
        descriptor=descriptor, path=path, purpose=purpose, binding=binding
    )


def release(lease: AcquiredLease) -> None:
    """Close the handle; the kernel does the rest. Best-effort tidy of the file.

    A close failure PROPAGATES: the exclusion may still stand, and a release
    that reports success over a live handle would let the gate print green
    while the machine stays held. Only the record removal is best-effort -
    the exclusion never depended on the file's presence.
    """

    lease.close()
    with contextlib.suppress(OSError):
        lease.path.unlink()


@contextlib.contextmanager
def hold_lease(purpose: str, *, repo: Path | None = None) -> Iterator[AcquiredLease]:
    """Hold the machine for the whole block; children inherit the hold.

    The kernel owns the exclusion for exactly as long as a holder lives:
    while this process (or any child that inherited the handle) is alive,
    a contender's write-access open fails at the kernel.
    """

    lease = acquire(purpose, repo=repo)
    try:
        yield lease
    finally:
        release(lease)


def status(repo: Path | None = None) -> str:
    path = lease_path(repo)
    if not path.exists():
        return "free"
    if os.name != "nt":
        return f"record present; holds are Windows-only: {_holder_line(path)}"
    try:
        probe, _path, probe_binding = _open_lease_handle(
            repo, access=_GENERIC_WRITE, disposition=_OPEN_EXISTING
        )
    except LeaseRefused as refusal:
        if str(refusal).startswith("contended: "):
            return _holder_line(path)
        raise
    # The probe took a write-access handle to answer, and pins with it; the
    # probe is closed first and every pin after it, each exactly once. If
    # any of them will not close, this process is now a holder and "free"
    # would be the one wrong answer.
    strand = _abandon(
        during="a status probe", handles=(("probe", probe),), pins=probe_binding.pins
    )
    if strand is not None:
        raise strand
    return f"free (stale record left behind: {_holder_line(path)})"


def _wait_for_stdin_while_bound(lease: AcquiredLease) -> None:
    """Hold until stdin closes.

    The binding cannot change while the hold lives: every link of the
    resolution chain is pinned by the lease, and the kernel refuses the
    writes that would move it. The barrier still runs every
    ``_REVERIFY_SECONDS`` as a self-check that the pins are intact; a
    refusal ends the hold through the caller's release.
    """

    closed = threading.Event()

    def drain() -> None:
        try:
            sys.stdin.read()
        finally:
            closed.set()

    threading.Thread(target=drain, name="machine-lease-stdin", daemon=True).start()
    while not closed.wait(_REVERIFY_SECONDS):
        lease.assert_bound()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    hold_command = commands.add_parser(
        "hold", help="acquire, print READY, hold until stdin closes, release"
    )
    hold_command.add_argument("--purpose", required=True)
    try:
        arguments = parser.parse_args(argv)
    except SystemExit:
        return _USAGE
    try:
        if arguments.command == "status":
            print(status(arguments.repo))
            return 0
        if arguments.command == "hold":
            try:
                with hold_lease(arguments.purpose, repo=arguments.repo) as lease:
                    print("READY", flush=True)
                    _wait_for_stdin_while_bound(lease)
            except OSError as exc:
                print(
                    f"the machine lease did not release cleanly: {exc}",
                    file=sys.stderr,
                )
                return _REFUSED
            return 0
    except LeaseRefused as exc:
        print(f"machine lease refused: {exc}", file=sys.stderr)
        return _REFUSED
    except LeaseStranded as exc:
        # Not contention: this process itself may be the holder now, through
        # a handle it cannot close. The host must exit to free the machine.
        print(f"machine lease stranded: {exc}; exit this process", file=sys.stderr)
        return _STRANDED
    return _USAGE


if __name__ == "__main__":
    raise SystemExit(main())
