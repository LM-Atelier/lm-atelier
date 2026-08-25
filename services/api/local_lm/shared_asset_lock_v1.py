"""A cross-process lock whose ABSENCE is provable.

The Shared Asset Library's read leases turn on one rule from the plan: a lease
is not stolen because its timestamp is old - the holder's OS lock and process
identity must be proven ABSENT. That rules out the pattern this codebase
already uses twice, in `comfy_registry_wheel_environments.py` and
`comfy_registry_wheel_downloads.py`:

    os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)

A create-only sentinel proves somebody once claimed the resource. It cannot
prove they still hold it, because the file outlives the process that made it -
kill the holder and the resource is locked until a human deletes the file.

An OS lock is released by the kernel when the holder dies, including on a hard
kill, so "I could take it" IS the proof that nobody holds it. Measured on
Windows: refused while a child process held it, acquired 0.1s after that child
was killed.

A pid may be recorded beside a lock for DIAGNOSIS - so a refusal can say who
holds it - but never as the proof. Pid numbers are reused.

The lock file is created through a held directory rather than by path, so a
reparse point planted at the parent cannot move it somewhere else.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from typing import Any, Final, NoReturn

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    create_entry,
    open_entry,
)

LOCK_UNAVAILABLE: Final = "shared asset lock is held"
LOCK_INVALID: Final = "shared asset lock is invalid"
LOCK_REPLACED: Final = "shared asset lock entry was replaced"

#: One byte at offset zero. The region is arbitrary; both platforms only need
#: the lock to be per-file, and locking the whole file would mean asking the
#: kernel about a length that changes.
_REGION: Final = 1

_WINDOWS: Final = os.name == "nt"


class SharedAssetLockError(RuntimeError):
    """Fixed, non-echoing refusal for a lock that cannot be taken."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _held() -> NoReturn:
    raise SharedAssetLockError(LOCK_UNAVAILABLE)


def _invalid() -> NoReturn:
    raise SharedAssetLockError(LOCK_INVALID)


def _replaced() -> NoReturn:
    raise SharedAssetLockError(LOCK_REPLACED)


def _take(descriptor: int) -> bool:
    """True when this process now holds the lock, False when somebody else does.

    Non-blocking on both platforms on purpose: a lease decision must not wait
    on a holder that may never let go.
    """

    try:
        if _WINDOWS:
            import msvcrt

            windows: Any = msvcrt
            os.lseek(descriptor, 0, os.SEEK_SET)
            windows.locking(descriptor, windows.LK_NBLCK, _REGION)
        else:
            import fcntl

            posix: Any = fcntl
            posix.flock(descriptor, posix.LOCK_EX | posix.LOCK_NB)
    except OSError:
        # Both platforms report a live holder as an OSError, and neither
        # distinguishes it from a broken filesystem well enough to be worth
        # guessing. The caller learns "not yours", which is the only thing a
        # lease decision needs.
        return False
    return True


def _release(descriptor: int) -> None:
    with contextlib.suppress(OSError):
        if _WINDOWS:
            import msvcrt

            windows: Any = msvcrt
            os.lseek(descriptor, 0, os.SEEK_SET)
            windows.locking(descriptor, windows.LK_UNLCK, _REGION)
        else:
            import fcntl

            posix: Any = fcntl
            posix.flock(descriptor, posix.LOCK_UN)


def _open_lock_entry(anchor: AnchoredDirectory, name: str) -> int:
    """A descriptor on the lock file, created through the held directory.

    Creation races are expected and benign: two processes reaching for the same
    lease will both try to create it, and the loser simply opens what the
    winner made. The lock itself decides who holds it, not who made the file.
    """

    try:
        return create_entry(anchor, name)
    except AnchoredDirectoryError:
        pass
    descriptor = None
    with contextlib.suppress(AnchoredDirectoryError):
        descriptor = open_entry(anchor, name)
    if descriptor is None:
        _invalid()
    return descriptor


def entry_identity(descriptor: int) -> tuple[int, int]:
    """Which object this descriptor holds, independent of its name.

    A lock lives on the OBJECT, not on the directory entry that points at it,
    and on POSIX the two can be separated: unlinking the name does not close
    the object, so a later open at the same name creates a DIFFERENT object
    that can be locked independently while the first holder still owns the
    original. Identity is how a caller tells those apart.
    """

    measured = os.fstat(descriptor)
    return measured.st_dev, measured.st_ino


def _current_entry_identity(anchor: AnchoredDirectory, name: str) -> tuple[int, int] | None:
    """The identity of whatever the NAME points at now, or None if nothing does."""

    try:
        descriptor = open_entry(anchor, name)
    except AnchoredDirectoryError:
        return None
    if descriptor is None:
        return None
    try:
        return entry_identity(descriptor)
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


@contextlib.contextmanager
def hold(
    anchor: AnchoredDirectory, name: str, *, expect: tuple[int, int] | None = None
) -> Iterator[tuple[int, tuple[int, int]]]:
    """Hold the lock for the block, or refuse because somebody else does.

    Yields the descriptor and the identity of the object locked. A caller that
    already knows which object the lock lives on passes `expect`; a mismatch is
    a fixed refusal rather than a successful acquire, because locking a
    replacement object proves nothing about the holder of the original.
    """

    descriptor = _open_lock_entry(anchor, name)
    # Cleanup starts HERE, not after the checks. Everything below can raise -
    # entry_identity() and the post-acquire re-check both call os.fstat - and
    # an exit that skips this leaves a held descriptor nothing can reach, which
    # keeps the lease unavailable until the process ends. Reported as
    # codex/R1908.
    acquired = False
    try:
        identity = entry_identity(descriptor)
        if expect is not None and identity != expect:
            _replaced()
        if not _take(descriptor):
            _held()
        acquired = True
        # After taking it, and not before: between opening and locking, the
        # entry can be replaced, and the object we locked would then no longer
        # be the one the name refers to.
        if _current_entry_identity(anchor, name) != identity:
            _replaced()
        yield descriptor, identity
    finally:
        # Release only what was actually taken, but close in every case.
        if acquired:
            _release(descriptor)
        with contextlib.suppress(OSError):
            os.close(descriptor)


def holder_is_gone(
    anchor: AnchoredDirectory, name: str, *, expect: tuple[int, int] | None = None
) -> bool:
    """True when nobody holds THIS lock object, proven by taking it.

    `expect` names the object the caller means. Without it this answers only
    "is the thing at that name free", which is a different and weaker question:
    if the entry has been replaced, the object at the name is new and unheld
    while the original holder is still running. So a lease MUST pass the
    identity it recorded, and a mismatch answers False - not because the
    original is known to be held, but because this call cannot prove it is not.

    Even with a match there is an unavoidable race: somebody may take the lock
    immediately afterwards. A caller that intends to act must hold() and act
    inside the block. This is for reporting.
    """

    try:
        descriptor = _open_lock_entry(anchor, name)
    except SharedAssetLockError:
        return False
    try:
        if expect is not None and entry_identity(descriptor) != expect:
            return False
        if not _take(descriptor):
            return False
        _release(descriptor)
        return True
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def current_process_identity() -> tuple[int, str]:
    """The pid and the interpreter that owns it, for DIAGNOSIS only.

    Never proof. A pid is reused, so a stored pid matching a live process says
    nothing about whether that process is the one that took the lease.
    """

    return os.getpid(), sys.executable
