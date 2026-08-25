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
import secrets
import sys
from collections.abc import Iterator
from typing import Any, Final, NoReturn

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    create_entry,
    link_entry,
    open_entry,
    remove_entry,
)

#: Which object a descriptor holds: device, inode, and the entry's creation
#: token. A bare (device, inode) pair is NOT an object identity - see
#: `entry_identity`.
EntryIdentity = tuple[int, int, bytes]

LOCK_UNAVAILABLE: Final = "shared asset lock is held"
LOCK_INVALID: Final = "shared asset lock is invalid"
LOCK_REPLACED: Final = "shared asset lock entry was replaced"

#: One byte at offset zero. The region is arbitrary; both platforms only need
#: the lock to be per-file, and locking the whole file would mean asking the
#: kernel about a length that changes.
_REGION: Final = 1

#: The creation token lives immediately after the locked byte, so a mandatory
#: Windows lock on the region can never refuse the read that establishes
#: identity.
_TOKEN_OFFSET: Final = _REGION
_TOKEN_BYTES: Final = 16
#: One byte of padding so the locked region is inside the file rather than
#: past its end. Its value carries no meaning.
_RESERVED: Final = bytes(1)

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
    """A descriptor on a COMPLETE lock entry, opened through the held directory.

    Every entry this module opens carries a creation token, and carries it
    before any other process can observe the name. That is why the entry is
    built under a staging name and published by link rather than created where
    it will live. `create_entry` is exclusive, so exactly one process creates -
    but a second process can open the empty file in the gap between that create
    and the write, read no token, and record an identity that will never match
    again, leaving a lease nothing can take. Publishing an already-complete
    object closes that gap instead of moving it.

    Creation races stay benign, as before. Both processes stage their own
    entry, one wins the link, and the loser opens what the winner published.
    """

    descriptor = _open_existing(anchor, name)
    if descriptor is None:
        _establish(anchor, name)
        descriptor = _open_existing(anchor, name)
    if descriptor is None:
        _invalid()
    return descriptor


def _open_existing(anchor: AnchoredDirectory, name: str) -> int | None:
    """The entry if it is there, None if it is not, a refusal for anything else."""

    try:
        return open_entry(anchor, name)
    except AnchoredDirectoryError:
        _invalid()


def _establish(anchor: AnchoredDirectory, name: str) -> None:
    """Publish a complete lock entry, or leave the winner's in place.

    Losing the link is not a failure and is not reported as one: the caller
    opens the published entry either way. The staging name is removed on every
    path, including the ones that raise, because nothing else will ever know
    it existed.
    """

    staging = f"{name}.{secrets.token_hex(8)}"
    try:
        descriptor = create_entry(anchor, staging)
    except AnchoredDirectoryError:
        _invalid()
    try:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, _RESERVED + secrets.token_bytes(_TOKEN_BYTES))
        finally:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        link_entry(anchor, staging, name)
    except (OSError, AnchoredDirectoryError):
        _invalid()
    finally:
        with contextlib.suppress(AnchoredDirectoryError):
            remove_entry(anchor, staging)


def _read_token(descriptor: int) -> bytes:
    """The entry's creation token, refusing anything this module did not write.

    A short read means a truncated entry or a file somebody else put at that
    name. Neither is something to take a lease on, and guessing a default here
    would reintroduce exactly the false match the token exists to prevent.
    """

    try:
        os.lseek(descriptor, _TOKEN_OFFSET, os.SEEK_SET)
        token = os.read(descriptor, _TOKEN_BYTES)
    except OSError:
        _invalid()
    if len(token) != _TOKEN_BYTES:
        _invalid()
    return token


def entry_identity(descriptor: int) -> EntryIdentity:
    """Which object this descriptor holds, independent of its name.

    A lock lives on the OBJECT, not on the directory entry that points at it,
    and on POSIX the two can be separated: unlinking the name does not close
    the object, so a later open at the same name creates a DIFFERENT object
    that can be locked independently while the first holder still owns the
    original. Identity is how a caller tells those apart.

    THE INODE NUMBER IS NOT THAT IDENTITY, and the gap is not theoretical. A
    filesystem is free to hand a freed inode number straight back to the next
    file created in that directory, and ext4 does it immediately: on the Ubuntu
    leg an entry and its replacement both measured (2049, 9439958). Every local
    Windows run was green because Windows file IDs are not reused that way,
    which is the whole reason this was believed to work. A guard that fires
    where the numbers happen to differ and is silent where they do not reads as
    protection while providing none.

    So identity carries the entry's creation token, which is unique to the act
    of creating it. Device and inode stay: they cost nothing, and they catch a
    different mistake - the same token appearing in a second file.
    """

    return (*_object_numbers(descriptor), _read_token(descriptor))


def _object_numbers(descriptor: int) -> tuple[int, int]:
    """The filesystem's own numbering for this object.

    Its own function so a test can make it answer the way ext4 does - the same
    pair for an entry and its replacement - on a platform whose file IDs are
    never reused. Windows could not falsify the identity guard at all without
    this seam, which is precisely how the reuse reached CI instead of a local
    run: the property was only ever tested where it happened to hold.
    """

    measured = os.fstat(descriptor)
    return measured.st_dev, measured.st_ino


def _current_entry_identity(anchor: AnchoredDirectory, name: str) -> EntryIdentity | None:
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
    anchor: AnchoredDirectory, name: str, *, expect: EntryIdentity | None = None
) -> Iterator[tuple[int, EntryIdentity]]:
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
    anchor: AnchoredDirectory, name: str, *, expect: EntryIdentity | None = None
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
    acquired = False
    try:
        # Read unconditionally, not only when `expect` was given. An entry this
        # module did not write is not a lock, and "nobody holds it" would be a
        # true statement about a file and a false one about a lease. What
        # hold() refuses, this must not report as free.
        identity = entry_identity(descriptor)
        if expect is not None and identity != expect:
            return False
        if not _take(descriptor):
            return False
        acquired = True
        # After taking it, and not before, exactly as hold() does. Between
        # opening the entry and locking it the name can be replaced, and a lock
        # on the object we opened then says nothing about the object the NAME
        # refers to - which is the one a caller about to steal the lease will
        # act on. Reporting absence there proves only that a detached object
        # nobody can reach any more is free. Reported as codex/R1917: the
        # window was closed in hold() and left open in the other public
        # decision helper, which is the one a lease actually asks.
        return _current_entry_identity(anchor, name) == identity
    except SharedAssetLockError:
        return False
    finally:
        if acquired:
            _release(descriptor)
        with contextlib.suppress(OSError):
            os.close(descriptor)


def current_process_identity() -> tuple[int, str]:
    """The pid and the interpreter that owns it, for DIAGNOSIS only.

    Never proof. A pid is reused, so a stored pid matching a live process says
    nothing about whether that process is the one that took the lease.
    """

    return os.getpid(), sys.executable
