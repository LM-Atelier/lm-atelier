from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import stat
from contextlib import suppress
from pathlib import Path

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryExists,
    create_entry,
    discard_entry,
    open_child_directory,
    open_entry,
    rename_entry,
    sync_directory,
)

INSTANCE_ID_HEADER = "X-LM-Atelier-Instance"
_INSTANCE_SEED_NAME = "desktop-instance-seed"
_STATE_DIR_NAME = "state"
#: The record is 64 hex characters. Reading ONE byte past the longest legal
#: value distinguishes valid from overlong, and bounds what a malformed
#: restored seed can cost.
_SEED_MAX_BYTES = 66
_SEED_READ_LIMIT = 67
_SEED_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_CONTEXT = b"lm-atelier-desktop-instance-v1\0"


class InstanceIdentityError(RuntimeError):
    pass


def instance_identity_from_directory(anchor: AnchoredDirectory) -> str:
    """Return the identity of the data root the caller already HOLDS.

    Every step - reaching the state folder, reading the seed, creating one
    when absent - goes through the retained directory rather than through a
    pathname, so nothing occurring between the caller's validation and this
    derivation can substitute a different root underneath it. A pathname
    validated and then used again is two separate resolutions; this is one.

    The anchor belongs to the caller and is not closed here.
    """

    root = anchor.path
    try:
        state = open_child_directory(anchor, _STATE_DIR_NAME, create=True)
    except AnchoredDirectoryError as exc:
        # Reaching the state folder failed. The refusal is deliberately
        # causeless, so it is reported by WHERE it happened rather than by a
        # reason the primitive does not disclose.
        raise InstanceIdentityError(
            "LM Atelier's state folder may not be a filesystem link"
        ) from exc
    try:
        with state:
            for _attempt in range(5):
                seed = _read_seed_entry(state)
                if seed is not None:
                    return _derive_identity(seed, root)
                candidate = secrets.token_bytes(32)
                # Written under an unpredictable staging name and PUBLISHED by
                # rename, so the final name never exists holding partial bytes.
                # Creating the final name directly made it visible before its
                # contents were complete, and a concurrent first start could
                # read empty content and fail instead of converging.
                # Collision-RESISTANT, not merely per-process: a pid and thread
                # id are deterministic and get reused, so a staging entry left
                # by an interrupted earlier start could block a later one and
                # be misreported as a bad final seed. An earlier version of
                # this comment called that name unpredictable, which it was
                # not.
                #
                # Abandoned staging entries are inert: the name is random and
                # never consulted by any reader, so they cost disk and nothing
                # else. Sweeping them would need a directory listing the
                # primitive does not expose, and is deliberately not done here.
                staging = ""
                descriptor = -1
                for _create_attempt in range(5):
                    staging = f"{_INSTANCE_SEED_NAME}.{secrets.token_hex(8)}.tmp"
                    try:
                        descriptor = create_entry(state, staging)
                        break
                    except AnchoredEntryExists:
                        continue
                if descriptor < 0:
                    raise InstanceIdentityError(
                        "LM Atelier could not establish ownership of its data folder"
                    )
                try:
                    handle = os.fdopen(descriptor, "w", encoding="ascii")
                except BaseException:
                    # fdopen takes ownership only once it succeeds, so every
                    # failure BEFORE it has to close the descriptor itself.
                    with suppress(OSError):
                        os.close(descriptor)
                    with suppress(Exception):
                        discard_entry(state, staging)
                    raise
                try:
                    with handle:
                        if hasattr(os, "fchmod"):
                            os.fchmod(handle.fileno(), 0o600)
                        handle.write(candidate.hex())
                        handle.flush()
                        os.fsync(handle.fileno())
                except BaseException:
                    with suppress(Exception):
                        discard_entry(state, staging)
                    raise
                try:
                    rename_entry(state, staging, _INSTANCE_SEED_NAME, replace=False)
                except AnchoredEntryExists:
                    # Another start published first. Drop ours and read theirs,
                    # so both converge on the one identity.
                    with suppress(Exception):
                        discard_entry(state, staging)
                    continue
                # The record is durable; the directory ENTRY naming it is not
                # until the directory itself is synced.
                sync_directory(state)
                return _derive_identity(candidate, root)
    except AnchoredDirectoryError as exc:
        # The state folder was reached; the seed entry itself is unusable -
        # a directory, a link, or otherwise not the regular file it must be.
        raise InstanceIdentityError("LM Atelier's desktop identity is not a regular file") from exc
    raise InstanceIdentityError("LM Atelier could not establish ownership of its data folder")


def load_or_create_instance_identity(data_dir: Path) -> str:
    """Return an opaque identity bound to one resolved LM Atelier data root.

    Establishing the root is this entry point's job, which is why it may
    create one; the derivation itself then runs against the directory that
    was actually established rather than against the name again.
    """

    root = data_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        with AnchoredDirectory(root) as anchor:
            return instance_identity_from_directory(anchor)
    except AnchoredDirectoryError as exc:
        raise InstanceIdentityError("LM Atelier's state folder is outside its data folder") from exc


def _read_bounded(descriptor: int, limit: int) -> bytes:
    """Read up to `limit` bytes, continuing through short reads until EOF.

    One os.read is allowed to return fewer bytes than asked for before EOF, so
    a 64-byte valid-looking prefix of a much longer file could satisfy the
    pattern without the overflow byte ever being read - the bound would have
    been enforced against a number that was never measured.
    """

    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_seed_entry(state: AnchoredDirectory) -> bytes | None:
    """Read, validate and repair the seed through ONE descriptor.

    Reading the entry as bytes and then repairing its mode by name was two
    lookups, so the bytes validated and the entry chmod-ed need not have been
    the same object. One open settles type, contents and mode together.
    """

    descriptor = open_entry(state, _INSTANCE_SEED_NAME)
    if descriptor is None:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise InstanceIdentityError("LM Atelier's desktop identity is not a regular file")
        raw = _read_bounded(descriptor, _SEED_READ_LIMIT)
        if len(raw) > _SEED_MAX_BYTES:
            raise InstanceIdentityError("LM Atelier's desktop identity is invalid")
        try:
            value = raw.decode("ascii").strip()
        except UnicodeError as exc:
            raise InstanceIdentityError("LM Atelier's desktop identity is invalid") from exc
        if not _SEED_PATTERN.fullmatch(value):
            raise InstanceIdentityError("LM Atelier's desktop identity is invalid")
        # Repaired on the SAME descriptor whose bytes were just validated.
        # POSIX only: Windows carries no mode here, which is why the regression
        # asserts it only off Windows.
        if hasattr(os, "fchmod"):
            try:
                os.fchmod(descriptor, 0o600)
            except OSError as exc:
                raise InstanceIdentityError(
                    "LM Atelier could not read its desktop identity"
                ) from exc
        return bytes.fromhex(value)
    except OSError as exc:
        raise InstanceIdentityError("LM Atelier could not read its desktop identity") from exc
    finally:
        os.close(descriptor)


def _derive_identity(seed: bytes, root: Path) -> str:
    normalized_root = os.path.normcase(str(root)).encode("utf-8", errors="surrogatepass")
    return hmac.new(
        seed,
        _IDENTITY_CONTEXT + normalized_root,
        hashlib.sha256,
    ).hexdigest()
