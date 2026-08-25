from __future__ import annotations

import os
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntry,
    AnchoredEntryKind,
    is_link_or_reparse,
    list_entries,
    remove_entry,
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

    That method stages through `NamedTemporaryFile(prefix=f".{key}-",
    suffix=".partial")`, so the emitted shape is a dot, the 64-character key, a
    hyphen, the temporary's own name, and `.partial`. Parsing that grammar is
    the point.

    Accepting anything that merely began with a dot and ended `.partial`
    deleted files this store could not have written - `.download.partial` among
    them. That is the identical suffix-only reasoning this slice removed from
    the cache branch, left in place one function below it. Reported as
    codex/R1946.
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
        if self._usable_entry(path, max_age_seconds=max_age_seconds) is None:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None

    def read_bytes(self, path: Path, *, max_age_seconds: float | None = None) -> bytes | None:
        if self._usable_entry(path, max_age_seconds=max_age_seconds) is None:
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

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
        self._require_cache_path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.root,
                prefix=f".{path.stem}-",
                suffix=".partial",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                self._unlink(temporary)
        self.prune(protected=path)

    def _usable_entry(
        self,
        path: Path,
        *,
        max_age_seconds: float | None,
    ) -> os.stat_result | None:
        self._require_cache_path(path)
        try:
            if is_link_or_reparse(
                path,
                missing="assume_regular",
                unreadable="assume_link",
            ):
                return None
            metadata = path.stat()
        except OSError:
            return None
        if not stat.S_ISREG(metadata.st_mode):
            return None
        if max_age_seconds is not None:
            age = max(0.0, self._now() - metadata.st_mtime)
            if age > max_age_seconds:
                return None
        return metadata

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

    @staticmethod
    def _unlink(path: Path) -> bool:
        """Remove a staged temporary by path, on the write path only.

        The prune no longer uses this. `_atomic_write` created this file itself
        moments earlier, holds no anchor, and is cleaning up its own failure.
        """

        try:
            path.unlink()
        except OSError:
            return False
        return True
