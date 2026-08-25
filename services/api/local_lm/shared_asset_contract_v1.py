"""Shared Asset Library store identity and root contract (item 58, Phase 0).

Completes the store contract over the existing pure slices: a durable
`store.json` identity with reader/writer version negotiation, a
root-acceptance probe battery, and read-only mode detection. Callers pass an
explicit root; this module never discovers the desktop library. No publish,
claim, lease, API mount, Settings surface, or live install mutation.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import stat
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final, NoReturn

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredDirectoryNotFound,
    AnchoredEntryExists,
    available_bytes,
    create_entry,
    discard_entry,
    link_entry,
    open_child_directory,
    read_entry,
    remove_entry,
    rename_entry,
    sync_directory,
)

SCHEMA_ID: Final = "lm-atelier-shared-asset-store-v1"
FORMAT_VERSION: Final = 1
SUPPORTED_READER_VERSION: Final = 1
SUPPORTED_WRITER_VERSION: Final = 1
IDENTITY_LEAF: Final = "store.json"
LOCKS_LEAF: Final = "locks"
_PROBE_RENAME_PREFIX: Final = "probe-rename-"
_PROBE_CREATE_PREFIX: Final = "probe-create-"
# Enough for identity, registry, staging bookkeeping - object payloads carry
# their own preflight space checks at publish time.
MINIMUM_FREE_BYTES: Final = 64 * 1024 * 1024
INVALID_STORE: Final = "shared asset store is invalid"


class SharedAssetContractError(ValueError):
    """Fixed non-echoing refusal for an unusable store root or identity."""


def _invalid() -> NoReturn:
    # `from None` deliberately: most callers refuse from inside an `except`
    # block, and without it the original OSError rides along as __context__
    # and prints the very path this refusal exists to withhold.
    raise SharedAssetContractError(INVALID_STORE) from None


def _is_unc(path: Path) -> bool:
    text = str(path)
    return text.startswith("\\\\") or text.startswith("//") or path.as_posix().startswith("//")


_IDENTITY_STAGING_PREFIX: Final = "identity-"
_HAS_DIR_FD: Final = (
    os.open in os.supports_dir_fd
    and os.link in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
)


def _is_link(path: Path) -> bool:
    """True for any reparse redirection: symlinks everywhere, and on Windows
    every other reparse point too - junctions report is_symlink()=False, so a
    symlink-only check writes straight through them."""
    try:
        if path.is_symlink() or os.path.islink(path):
            return True
        try:
            entry_stat = os.lstat(path)
        except FileNotFoundError:
            return False
        attributes = getattr(entry_stat, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return bool(attributes & reparse_flag)
    except OSError:
        _invalid()


def _require_absolute_root(root: Path) -> Path:
    if not isinstance(root, Path):
        _invalid()
    try:
        chosen = root.expanduser()
    except (OSError, RuntimeError, ValueError):
        _invalid()
    if not str(chosen) or "\x00" in str(chosen) or _is_unc(chosen) or not chosen.is_absolute():
        _invalid()
    return chosen


@dataclasses.dataclass(frozen=True)
class StoreIdentity:
    """The durable facts `store.json` records about one library."""

    library_uuid: str
    format_version: int
    min_reader_version: int
    min_writer_version: int


@dataclasses.dataclass(frozen=True)
class RootProbeReport:
    """Per-check outcome of the root-acceptance battery."""

    directory: bool
    no_reparse_points: bool
    atomic_rename: bool
    exclusive_create: bool
    free_space: bool

    @property
    def usable(self) -> bool:
        return (
            self.directory
            and self.no_reparse_points
            and self.atomic_rename
            and self.exclusive_create
            and self.free_space
        )


def _positive_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        _invalid()
    return value


def _parse_identity(payload: object) -> StoreIdentity:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA_ID:
        _invalid()
    library_uuid = payload.get("library_uuid")
    if not isinstance(library_uuid, str):
        _invalid()
    try:
        canonical = str(uuid.UUID(library_uuid))
    except ValueError:
        _invalid()
    if canonical != library_uuid:
        _invalid()
    return StoreIdentity(
        library_uuid=canonical,
        format_version=_positive_int(payload.get("format_version")),
        min_reader_version=_positive_int(payload.get("min_reader_version")),
        min_writer_version=_positive_int(payload.get("min_writer_version")),
    )


def read_store_identity(*, root: Path) -> StoreIdentity | None:
    """Return the identity at `root`, None when no store exists there.

    A present-but-unreadable or malformed identity is a refusal, never None:
    an existing library must not be silently treated as absent. A root
    reached through any reparse redirection refuses outright - adopting a
    junction target's identity on read is the same escape as writing
    through one.
    """

    store = _require_absolute_root(root)
    # Absence is decided by the acquisition itself. Asking Path.exists() first
    # FOLLOWED the name, so a dangling linked root - a present entry pointing
    # nowhere - answered false and was reported as "no library here" rather
    # than refused. Every other acquisition failure stays a refusal.
    try:
        with _anchored(store, absent_is_refusal=False) as anchor:
            return _read_identity_anchored(anchor)
    except AnchoredDirectoryNotFound:
        # Only acquisition raises this; the anchored read reports a missing
        # record by returning None rather than by raising.
        return None


def _refuse_reparse_chain(path: Path) -> None:
    cursor = path
    while True:
        if _is_link(cursor):
            _invalid()
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


@contextlib.contextmanager
def _anchored(
    store: Path, *, create: bool = False, absent_is_refusal: bool = True
) -> Iterator[AnchoredDirectory]:
    """Hold `store` anchored, translating containment refusals into ours.

    The shared layer refuses with its own neutral error carrying no path.
    This module promises one fixed message, so the translation happens here
    rather than by letting a second error type reach callers.
    """

    try:
        anchor = AnchoredDirectory(store, create=create)
    except AnchoredDirectoryNotFound:
        # Absence is a refusal everywhere except the one reader whose whole
        # question is whether a store is there. Defaulting the other way would
        # have let any caller treat "not there" as an ordinary outcome.
        if absent_is_refusal:
            _invalid()
        raise
    except AnchoredDirectoryError:
        _invalid()
    try:
        yield anchor
    finally:
        anchor.close()


def initialize_store_identity(*, root: Path) -> StoreIdentity:
    """Create the identity at `root`, or return the existing valid one.

    The candidate chain is validated reparse-free BEFORE any read, creation,
    or early return - a junctioned root must neither adopt its target's
    identity nor gain children below the junction. Publication never
    replaces: the record is staged, fsynced, and linked to its destination
    with create-only semantics, so a concurrent initializer cannot overwrite
    an identity another caller has already observed - the loser converges by
    reading the winner. Every filesystem failure along staging, publication,
    AND cleanup surfaces the fixed refusal, never a raw path; only a cleanup
    failure while a refusal is already propagating is suppressed in its
    favor.
    """

    store = _require_absolute_root(root)
    # The anchor comes FIRST, and it creates the leaf through its parent's
    # handle. An earlier version validated the chain, then read, then
    # mkdir'd, then anchored - and every one of those steps resolved the
    # path again, so a redirect arriving in that window built directories in
    # a foreign target and could return a foreign identity. Ordering closes
    # it; another check would not.
    with _anchored(store, create=True) as anchor:
        existing = _read_identity_anchored(anchor)
        if existing is not None:
            return existing
        identity = StoreIdentity(
            library_uuid=str(uuid.uuid4()),
            format_version=FORMAT_VERSION,
            min_reader_version=SUPPORTED_READER_VERSION,
            min_writer_version=SUPPORTED_WRITER_VERSION,
        )
        _stage_and_publish(
            anchor,
            {
                "schema": SCHEMA_ID,
                "format_version": identity.format_version,
                "library_uuid": identity.library_uuid,
                "min_reader_version": identity.min_reader_version,
                "min_writer_version": identity.min_writer_version,
            },
        )
        published = _read_identity_anchored(anchor)
    if published is None:
        _invalid()
    return published


def _stage_and_publish(anchor: AnchoredDirectory, record: dict[str, object]) -> None:
    """Write the record and publish it create-only, inside the held root.

    Creating the staging entry is also what closes the remaining window: the
    in-place conversion this guards against requires an EMPTY directory, so
    once one entry exists the root can no longer become a redirection at all.
    Had it already been one, the create refuses rather than writing through.
    """

    payload = json.dumps(record, sort_keys=True).encode("utf-8")
    staging = f"{_IDENTITY_STAGING_PREFIX}{uuid.uuid4().hex}"
    staged = False
    try:
        descriptor = create_entry(anchor, staging)
        staged = True
        with os.fdopen(descriptor, "wb") as staged_file:
            staged_file.write(payload)
            staged_file.flush()
            os.fsync(staged_file.fileno())
        # A concurrent initializer publishing first is convergence, not
        # failure: the loser reads the winner's identity below. The link is
        # create-only, so it can never replace an identity someone observed.
        link_entry(anchor, staging, IDENTITY_LEAF)
        # fsyncing the record alone leaves the directory ENTRY unfenced, so a
        # crash could lose a published identity that was already observed.
        # POSIX syncs the held directory; Windows journals the metadata and
        # has no directory sync for these calls.
        sync_directory(anchor)
    except (AnchoredDirectoryError, OSError):
        if staged:
            discard_entry(anchor, staging)
        _invalid()
    except SharedAssetContractError:
        if staged:
            discard_entry(anchor, staging)
        raise
    # Success-path cleanup is part of the contract: a staging record that
    # cannot be removed would sit beside a "successful" identity, so this one
    # refuses rather than shrugging.
    try:
        remove_entry(anchor, staging)
    except (AnchoredDirectoryError, OSError):
        # OSError as well as the neutral refusal: the primitive should only
        # raise its own error, and if it ever raises a raw one the fixed
        # non-echoing message is still what a caller of THIS module gets.
        _invalid()


def _read_identity_anchored(anchor: AnchoredDirectory) -> StoreIdentity | None:
    """Read the identity through the held directory, never by path.

    The public reader takes a path because callers have one; this is the
    reader initialization uses, so that establishing a store never consults a
    name after the anchor is held.
    """

    try:
        raw = read_entry(anchor, IDENTITY_LEAF)
    except AnchoredDirectoryError:
        _invalid()
    if raw is None:
        return None
    # Parsed exactly as the public reader parses it, so an identity does not
    # mean two different things depending on which door it came through.
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _invalid()
    return _parse_identity(payload)


def negotiate_store_access(
    identity: StoreIdentity,
    *,
    reader_version: int = SUPPORTED_READER_VERSION,
    writer_version: int = SUPPORTED_WRITER_VERSION,
) -> str:
    """Return "read_write" or "read_only"; refuse an unreadable store.

    A store whose minimum reader is newer than this build refuses outright -
    misreading a newer format is worse than no access. A store whose minimum
    writer is newer degrades to read-only: exact bytes stay reusable while
    every mutation is left to builds that understand the format.
    """

    for value in (reader_version, writer_version):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            _invalid()
    if identity.min_reader_version > reader_version:
        _invalid()
    if identity.min_writer_version > writer_version:
        return "read_only"
    return "read_write"


def _probe_directory(store: Path) -> bool:
    try:
        return store.is_dir() and not _is_link(store)
    except SharedAssetContractError:
        return False


def _probe_reparse_free(store: Path) -> bool:
    cursor = store
    while True:
        try:
            if _is_link(cursor):
                return False
        except SharedAssetContractError:
            return False
        parent = cursor.parent
        if parent == cursor:
            return True
        cursor = parent


def _probe_atomic_rename(anchor: AnchoredDirectory) -> bool:
    """Prove the filesystem renames atomically, writing only inside the anchor.

    Probing is still writing. An earlier version resolved the store path again
    for each probe, so a redirect arriving after the battery's own containment
    check was followed and a foreign directory collected the artifacts while
    the report called the root clean.
    """

    name = f"{_PROBE_RENAME_PREFIX}{uuid.uuid4().hex}"
    renamed = f"{name}.renamed"
    created = False
    try:
        os.close(create_entry(anchor, name))
        created = True
        # A freshly generated destination, so replacement is neither needed
        # nor wanted - and stating it means the probe proves the same thing
        # on both platforms rather than a different thing on each.
        rename_entry(anchor, name, renamed, replace=False)
        return True
    except (OSError, AnchoredDirectoryError):
        return False
    finally:
        if created:
            discard_entry(anchor, name)
        discard_entry(anchor, renamed)


def _probe_exclusive_create(anchor: AnchoredDirectory) -> bool:
    """Prove exclusive creation refuses a second create of the same name.

    The probe writes inside the store's locks directory, which is part of the
    layout rather than a probe artifact and so survives; only its contents are
    transient. Both the directory and the probe file are reached through the
    held root.
    """

    name = f"{_PROBE_CREATE_PREFIX}{uuid.uuid4().hex}"
    try:
        locks = open_child_directory(anchor, LOCKS_LEAF, create=True)
    except AnchoredDirectoryError:
        return False
    created = False
    try:
        os.close(create_entry(locks, name))
        created = True
        try:
            os.close(create_entry(locks, name))
        except AnchoredEntryExists:
            # Only a COLLISION proves exclusive creation. Accepting any
            # refusal here made the probe pass when creation failed for an
            # unrelated reason - a green result from a broken filesystem,
            # which is the opposite of what a battery is for.
            return True
        except AnchoredDirectoryError:
            return False
        return False
    except (OSError, AnchoredDirectoryError):
        # A refusal on the FIRST create is a failed probe, not an escaping
        # error: the battery reports, it does not raise.
        return False
    finally:
        if created:
            discard_entry(locks, name)
        locks.close()


def _probe_free_space(anchor: AnchoredDirectory, minimum_free_bytes: int) -> bool:
    """Capacity for the volume of the directory the report is ABOUT.

    Measured through the held anchor rather than by pathname, so every field
    of the report describes one directory identity. Taking this one field by
    name let a single report combine two: the capability fields answered for
    the held directory while capacity answered for whatever the name meant
    when it was read.
    """

    try:
        return available_bytes(anchor) >= minimum_free_bytes
    except AnchoredDirectoryError:
        return False


def _probe_anchored(anchor: AnchoredDirectory, minimum_free_bytes: int) -> RootProbeReport:
    """Run the write probes against a directory the caller already holds."""

    return RootProbeReport(
        directory=True,
        no_reparse_points=True,
        atomic_rename=_probe_atomic_rename(anchor),
        exclusive_create=_probe_exclusive_create(anchor),
        free_space=_probe_free_space(anchor, minimum_free_bytes),
    )


def probe_store_root(
    *,
    root: Path,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> RootProbeReport:
    """Run the acceptance battery against an existing candidate root.

    The battery is diagnostic: each check reports independently so a chooser
    can explain what failed without echoing the path. Gate with
    `require_usable_root` when only acceptance matters.
    """

    if not isinstance(minimum_free_bytes, int) or isinstance(minimum_free_bytes, bool):
        _invalid()
    if minimum_free_bytes < 0:
        _invalid()
    store = _require_absolute_root(root)
    directory = _probe_directory(store)
    reparse_free = _probe_reparse_free(store) if directory else False
    if not directory or not reparse_free:
        # Never run the write probes against a root that already failed
        # containment - a probe write through a junction is itself an escape.
        return RootProbeReport(
            directory=directory,
            no_reparse_points=reparse_free,
            atomic_rename=False,
            exclusive_create=False,
            free_space=False,
        )
    # The write probes run against a HELD root, not against the path that
    # passed the checks above. Re-resolving here is exactly what let a
    # redirect arriving after the containment check collect the probe
    # artifacts while this report still called the root clean.
    try:
        with _anchored(store) as anchor:
            return _probe_anchored(anchor, minimum_free_bytes)
    except SharedAssetContractError:
        # The battery reports; it does not raise. A root that cannot even be
        # anchored fails containment, which is what the report already says.
        return RootProbeReport(
            directory=directory,
            no_reparse_points=False,
            atomic_rename=False,
            exclusive_create=False,
            free_space=False,
        )


def require_usable_root(
    *,
    root: Path,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> Path:
    """Return the accepted root or raise the fixed refusal."""

    store = _require_absolute_root(root)
    if not probe_store_root(root=store, minimum_free_bytes=minimum_free_bytes).usable:
        _invalid()
    return store


def store_access_mode(
    *,
    root: Path,
    reader_version: int = SUPPORTED_READER_VERSION,
    writer_version: int = SUPPORTED_WRITER_VERSION,
) -> str:
    """Resolve one root to "read_write" or "read_only" from identity + probes.

    The identity must already exist: access-mode resolution never creates a
    library as a side effect. A usable root with a compatible writer is
    read/write; a root whose battery fails writes (or whose format is newer
    than this writer) is read-only; an unreadable identity refuses.
    """

    store = _require_absolute_root(root)
    # One acquisition for the read AND the probes. Reading the identity from
    # one directory and then probing another would resolve two different
    # directories into a single answer, which is exactly the confusion this
    # module exists to prevent.
    with _anchored(store) as anchor:
        identity = _read_identity_anchored(anchor)
        if identity is None:
            _invalid()
        negotiated = negotiate_store_access(
            identity,
            reader_version=reader_version,
            writer_version=writer_version,
        )
        if negotiated == "read_only":
            return "read_only"
        report = _probe_anchored(anchor, MINIMUM_FREE_BYTES)
    if not report.directory or not report.no_reparse_points:
        _invalid()
    if not (report.atomic_rename and report.exclusive_create and report.free_space):
        return "read_only"
    return "read_write"
