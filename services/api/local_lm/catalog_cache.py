from __future__ import annotations

import contextlib
import errno
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntry,
    AnchoredEntryKind,
    create_publishable_entry,
    is_link_or_reparse,
    list_entries,
    open_child_directory,
    open_entry,
    remove_entry,
    rename_entry,
)

#: A key is 64 lowercase hexadecimal characters and the payload is one of two
#: suffixes. Both `path` and the prune read these, so the shape the store will
#: write and the shape it will delete cannot drift apart.
KEY_LENGTH: Final = 64
KEY_ALPHABET: Final = "0123456789abcdef"
SUFFIXES: Final = frozenset({".json", ".bin"})
PARTIAL_SUFFIX: Final = ".partial"

#: Floor for one pass's enumeration ceiling; see `_listing_limit`.
_MIN_LISTING: Final = 8192


def is_cache_name(name: str) -> bool:
    """True only for a name `CatalogCacheStore.path` could have produced.

    The store already refuses to WRITE anything else, but the prune used to
    decide on the suffix alone, so any file whose name merely ended `.json`
    was inside the cache budget and eligible for deletion. Applying the same
    shape on the way out means a pass can only ever delete something this
    store could have written.
    """

    stem, separator, suffix = name.rpartition(".")
    return (
        bool(separator)
        and len(stem) == KEY_LENGTH
        and all(character in KEY_ALPHABET for character in stem)
        and f".{suffix}" in SUFFIXES
    )


def is_partial_name(name: str) -> bool:
    """True only for a leftover `_atomic_write` could actually have staged.

    That method stages `.{key}-{token}.partial` in the held cache root: a dot,
    the 64-character key, a hyphen, a token, and `.partial`. Parsing that
    grammar is the point.

    Accepting anything that merely began with a dot and ended `.partial`
    deleted files this store could not have written - `.download.partial` among
    them. The same suffix-only reasoning is still in place one function
    below it, where it has the same effect.
    """

    if not name.startswith(".") or not name.endswith(PARTIAL_SUFFIX):
        return False
    key, separator, temporary = name[1 : -len(PARTIAL_SUFFIX)].partition("-")
    return (
        bool(separator)
        and bool(temporary)
        and len(key) == KEY_LENGTH
        and all(character in KEY_ALPHABET for character in key)
    )


@dataclass(frozen=True)
class CatalogCachePolicy:
    fresh_seconds: float = 5 * 60
    stale_seconds: float = 7 * 24 * 60 * 60
    partial_seconds: float = 60 * 60
    max_entries: int = 512
    max_bytes: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class _CacheEntry:
    path: Path
    modified_at: float
    size: int


class CatalogCacheStore:
    def __init__(self, root: Path, policy: CatalogCachePolicy | None = None) -> None:
        self.root = root
        self.policy = policy or CatalogCachePolicy()

    def path(self, key: str, *, suffix: str = ".json") -> Path:
        if len(key) != KEY_LENGTH or any(character not in KEY_ALPHABET for character in key):
            raise ValueError("catalog cache key must be lowercase hexadecimal")
        if suffix not in SUFFIXES:
            raise ValueError("catalog cache suffix is unsupported")
        return self.root / f"{key}{suffix}"

    def read_text(self, path: Path, *, max_age_seconds: float | None = None) -> str | None:
        payload = self._read_cached(path, max_age_seconds=max_age_seconds)
        if payload is None:
            return None
        try:
            return payload.decode("utf-8")
        except UnicodeError:
            return None

    def read_bytes(self, path: Path, *, max_age_seconds: float | None = None) -> bytes | None:
        return self._read_cached(path, max_age_seconds=max_age_seconds)

    def write_text(self, path: Path, content: str) -> None:
        try:
            self._atomic_write(path, content.encode("utf-8"))
        except OSError:
            return

    def write_bytes(self, path: Path, content: bytes) -> None:
        try:
            self._atomic_write(path, content)
        except OSError:
            return

    def prune(self, *, protected: Path | None = None) -> None:
        """Bound the cache, deciding and deleting through one held root.

        Every entry used to be resolved by name three more times after
        `iterdir` had already named it - a link check, a stat, and an unlink -
        and each of those reopened the window this pass has to be safe across.
        A name that was an ordinary file when it was checked could be a link by
        the time it was deleted, and the deletion would follow it out of the
        cache.

        The root is held for the whole pass now, so it can be neither renamed
        nor replaced while the pass runs, and a link anywhere on the way to it
        refuses outright instead of being pruned as though it were the cache.
        Each entry's kind, size and modification time come from ONE enumeration
        record, and every deletion goes through the held directory by name.

        Best-effort, as before: a root that cannot be held, or a directory too
        large to enumerate, leaves the cache untouched rather than raising into
        a caller that was only writing a file.
        """

        protected_name = (
            protected.name if protected is not None and protected.parent == self.root else None
        )
        now = self._now()
        try:
            with AnchoredDirectory(self.root) as anchor:
                entries = list_entries(anchor, limit=self._listing_limit())
                kept = self._sweep_by_age(anchor, entries, now=now, protected=protected_name)
                self._enforce_budget(anchor, kept, protected=protected_name)
        except (AnchoredDirectoryError, OSError):
            return

    def _listing_limit(self) -> int:
        """How many entries one pass may enumerate.

        `list_entries` refuses rather than truncating, so a ceiling at or below
        the store's own entry budget would stop the pass working exactly when
        the directory outgrew it - the moment pruning matters most. The floor
        is the primitive's own default, and the multiple leaves room for the
        partials and for entries that arrived since the last pass.
        """

        return max(_MIN_LISTING, self.policy.max_entries * 4)

    def _sweep_by_age(
        self,
        anchor: AnchoredDirectory,
        entries: tuple[AnchoredEntry, ...],
        *,
        now: float,
        protected: str | None,
    ) -> list[_CacheEntry]:
        kept: list[_CacheEntry] = []
        for entry in entries:
            size = entry.size_bytes
            modified = entry.modified_at
            if entry.kind is not AnchoredEntryKind.FILE or size is None or modified is None:
                # An unsafe kind carries no metadata by design, and a safe entry
                # that vanished or refused reacquisition between the enumeration
                # and the measurement carries none either. Neither can be aged,
                # and a pass that cannot establish an age does not delete.
                continue
            modified_at = modified.timestamp()
            age = max(0.0, now - modified_at)
            if is_partial_name(entry.name):
                if age > self.policy.partial_seconds:
                    self._remove(anchor, entry.name)
                continue
            if not is_cache_name(entry.name):
                continue
            expired = age > self.policy.stale_seconds and entry.name != protected
            if expired and self._remove(anchor, entry.name):
                continue
            # Either it is not expired, or its removal was REFUSED - in which
            # case it is still in the directory and the budget pass below has to
            # keep counting it. Dropping it here lost it from the inventory
            # while it still occupied the cache.
            kept.append(_CacheEntry(self.root / entry.name, modified_at, size))
        return kept

    def _enforce_budget(
        self,
        anchor: AnchoredDirectory,
        kept: list[_CacheEntry],
        *,
        protected: str | None,
    ) -> None:
        kept.sort(key=lambda entry: (entry.modified_at, entry.path.name))
        total_bytes = sum(entry.size for entry in kept)
        # Entries whose removal was refused. They stay in the inventory because
        # they are still in the directory, and they stop being candidates so the
        # loop cannot spin on them.
        refused: set[str] = set()
        while len(kept) > self.policy.max_entries or total_bytes > self.policy.max_bytes:
            removable = next(
                (
                    entry
                    for entry in kept
                    if entry.path.name != protected and entry.path.name not in refused
                ),
                None,
            )
            if removable is None:
                break
            # Out of the inventory only when it actually left the directory.
            # Removing it first meant a refusal shrank len(kept) while the file
            # stayed, so an entry-count overflow could end the loop with the
            # cache still over budget and nothing saying so.
            if self._remove(anchor, removable.path.name):
                kept.remove(removable)
                total_bytes -= removable.size
            else:
                refused.add(removable.path.name)

    def _atomic_write(self, path: Path, content: bytes) -> None:
        """Write one cache file through the held root, or leave it untouched.

        A path-based temporary follows a link planted at the cache directory
        and publishes into whatever that link points at. Holding the root
        refuses that link before any byte is written. Missing ancestors are
        created through each held parent, so a source cache one level under
        the shared directory still appears, and a link in that chain is
        refused before the next directory is made. The staged name matches
        `is_partial_name`, so a crash leftover is still the shape prune
        deletes. A write that fails removes that staged file before it
        returns. A root that cannot be held is the same outcome as a failed
        write: the caller asked to store a cache entry, and nothing was stored.
        """

        self._require_cache_path(path)
        partial = f".{path.stem}-{os.urandom(8).hex()}.partial"
        published = False
        try:
            with _held_cache_root(self.root) as anchor:
                descriptor = create_publishable_entry(anchor, partial)
                try:
                    try:
                        _write_all(descriptor, content)
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    _publish_cache_name(anchor, partial, path.name)
                    published = True
                finally:
                    if not published:
                        with contextlib.suppress(AnchoredDirectoryError):
                            remove_entry(anchor, partial)
        except (AnchoredDirectoryError, OSError):
            return
        self.prune(protected=path)

    def _read_cached(self, path: Path, *, max_age_seconds: float | None) -> bytes | None:
        """Read one cache file through the held root, or return nothing.

        A path-based read follows a link planted at the cache directory and
        returns the file that link points at. Holding the root refuses that
        link before any byte is read. A link at the cache name is not an
        entry. A root that cannot be held is the same outcome as a missing
        file.
        """

        self._require_cache_path(path)
        try:
            with AnchoredDirectory(self.root) as anchor:
                if is_link_or_reparse(
                    path,
                    missing="assume_regular",
                    unreadable="assume_link",
                ):
                    return None
                descriptor = open_entry(anchor, path.name)
                if descriptor is None:
                    return None
                try:
                    if max_age_seconds is not None:
                        age = max(0.0, self._now() - os.fstat(descriptor).st_mtime)
                        if age > max_age_seconds:
                            return None
                    return _read_all(descriptor)
                finally:
                    os.close(descriptor)
        except (AnchoredDirectoryError, OSError):
            return None

    def _require_cache_path(self, path: Path) -> None:
        if path.parent != self.root or path.suffix not in {".json", ".bin"}:
            raise ValueError("catalog cache path escaped its root")

    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _remove(anchor: AnchoredDirectory, name: str) -> bool:
        """Delete one entry through the held root, reporting whether it went.

        The byte budget subtracts only what actually left the directory, so a
        refusal has to be distinguishable from a removal. Absence is not a
        refusal - `remove_entry` treats it as success, which is right here:
        something else removing the file first is the outcome this pass wanted.
        """

        try:
            remove_entry(anchor, name)
        except AnchoredDirectoryError:
            return False
        return True


@contextlib.contextmanager
def _held_cache_root(root: Path) -> Iterator[AnchoredDirectory]:
    """Hold `root`, creating each missing ancestor through its held parent.

    Opening the root with `create=True` makes only the final component, and
    only when its parent already exists. A source cache lives at
    `catalog-cache/<source>`, and that parent is not prepared in every
    caller. Each missing name is created from the directory that holds it,
    so a link anywhere in the chain refuses before the next directory exists.
    """

    parts = root.parts
    held: list[AnchoredDirectory] = []
    try:
        base: AnchoredDirectory | None = None
        start = len(parts)
        for count in range(len(parts), 0, -1):
            try:
                base = AnchoredDirectory(Path(*parts[:count]))
            except AnchoredDirectoryError:
                continue
            start = count
            break
        if base is None:
            raise AnchoredDirectoryError
        held.append(base)
        current = base
        for component in parts[start:]:
            current = open_child_directory(current, component, create=True)
            held.append(current)
        yield current
    finally:
        for anchor in reversed(held):
            anchor.close()


def _publish_cache_name(anchor: AnchoredDirectory, partial: str, name: str) -> None:
    """Move the staged file onto `name` inside the held cache root.

    An ordinary file is replaced in one rename. A link refuses that rename on
    some hosts; the link itself is then removed and the staged file takes the
    name. Removing the link does not change the file the link pointed at.
    """

    try:
        rename_entry(anchor, partial, name, replace=True)
    except AnchoredDirectoryError as refusal:
        if _held_kind(anchor, name) is not AnchoredEntryKind.LINK:
            raise refusal
        remove_entry(anchor, name)
        rename_entry(anchor, partial, name, replace=False)


def _held_kind(anchor: AnchoredDirectory, name: str) -> AnchoredEntryKind | None:
    for entry in list_entries(anchor, include_metadata=False):
        if entry.name == name:
            return entry.kind
    return None


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError(errno.EIO, "catalog cache write ended early")
        view = view[written:]
