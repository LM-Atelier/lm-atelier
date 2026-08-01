from __future__ import annotations

import os
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


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
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("catalog cache key must be lowercase hexadecimal")
        if suffix not in {".json", ".bin"}:
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
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return
        now = self._now()
        regular: list[_CacheEntry] = []
        for path in entries:
            try:
                if path.is_symlink():
                    continue
                metadata = path.stat()
            except OSError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            age = max(0.0, now - metadata.st_mtime)
            if path.suffix == ".partial":
                if age > self.policy.partial_seconds:
                    self._unlink(path)
                continue
            if path.suffix not in {".json", ".bin"}:
                continue
            if age > self.policy.stale_seconds and path != protected:
                self._unlink(path)
                continue
            regular.append(_CacheEntry(path, metadata.st_mtime, metadata.st_size))

        regular.sort(key=lambda entry: (entry.modified_at, entry.path.name))
        total_bytes = sum(entry.size for entry in regular)
        while len(regular) > self.policy.max_entries or total_bytes > self.policy.max_bytes:
            removable = next(
                (entry for entry in regular if entry.path != protected),
                None,
            )
            if removable is None:
                break
            regular.remove(removable)
            if self._unlink(removable.path):
                total_bytes -= removable.size

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
            if path.is_symlink():
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
    def _unlink(path: Path) -> bool:
        try:
            path.unlink()
        except OSError:
            return False
        return True
