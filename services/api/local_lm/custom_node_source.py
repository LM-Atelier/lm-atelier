"""Verify manual node source bytes against a pinned Git tree through held reads."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from typing import Never

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryKind,
    list_entries,
    open_child_directory,
    open_entry,
)

_MAX_FILES = 100_000
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_OID = re.compile(r"[0-9a-f]{40}")
_CACHE_NAME = re.compile(r"(.+)\.cpython-[0-9]+(?:\.opt-[0-9]+)?\.pyc")
_REFUSAL = "Custom node source no longer matches its pinned tree. Restore the reviewed package."


def _refuse() -> Never:
    raise ValueError(_REFUSAL)


def _manifest(raw: str, tree_hash: str) -> dict[tuple[str, ...], tuple[str, str]]:
    if not _OID.fullmatch(tree_hash) or len(raw.encode("utf-8")) > _MAX_MANIFEST_BYTES:
        _refuse()
    records = raw.split("\0")
    if not records or records[-1] != "" or len(records) > _MAX_FILES + 1:
        _refuse()
    result: dict[tuple[str, ...], tuple[str, str]] = {}
    folded: set[tuple[str, ...]] = set()
    for record in records[:-1]:
        metadata, separator, name = record.partition("\t")
        fields = metadata.split(" ")
        parts = tuple(name.split("/"))
        if (
            not separator
            or len(fields) != 3
            or fields[0] not in {"100644", "100755"}
            or fields[1] != "blob"
            or not _OID.fullmatch(fields[2])
            or len(parts) > 64
            or any(
                not part
                or part in {".", ".."}
                or "\\" in part
                or ":" in part
                or any(ord(character) < 32 for character in part)
                for part in parts
            )
            or any(part.casefold() == ".git" for part in parts)
        ):
            _refuse()
        key = tuple(part.casefold() for part in parts)
        if key in folded:
            _refuse()
        folded.add(key)
        result[parts] = fields[0], fields[2]
    # Git output is a manifest, not authority by itself. Reconstruct its Merkle
    # root so changed/replaced object metadata cannot redefine the pinned tree.
    directories: dict[tuple[str, ...], dict[str, tuple[str, str]]] = {(): {}}
    for parts, identity in result.items():
        for depth in range(len(parts) - 1):
            directories.setdefault(parts[: depth + 1], {})
        directories[parts[:-1]][parts[-1]] = identity
    for parts in sorted(directories, key=len, reverse=True):
        entries = directories[parts]
        payload = b"".join(
            mode.encode("ascii") + b" " + name.encode("utf-8") + b"\0" + bytes.fromhex(oid)
            for name, (mode, oid) in sorted(
                entries.items(),
                key=lambda item: item[0].encode("utf-8") + (b"/" if item[1][0] == "40000" else b""),
            )
        )
        # Git object IDs prescribe SHA-1; workflow approval uses SHA-256.
        digest = hashlib.sha1(
            b"tree " + str(len(payload)).encode("ascii") + b"\0" + payload,
            usedforsecurity=False,
        ).hexdigest()
        if parts:
            parent = directories[parts[:-1]]
            if parts[-1] in parent:
                _refuse()
            parent[parts[-1]] = "40000", digest
        elif digest != tree_hash:
            _refuse()
    return result


def _runtime_cache(parts: tuple[str, ...], tracked: dict[tuple[str, ...], tuple[str, str]]) -> bool:
    # Same narrow exception as Registry installs: ordinary CPython caches beside
    # reviewed source. Cache contents are not represented as reviewed source.
    if len(parts) < 2 or parts[-2] != "__pycache__":
        return False
    match = _CACHE_NAME.fullmatch(parts[-1])
    return match is not None and (*parts[:-2], match.group(1) + ".py") in tracked


def _blob_digest(
    anchor: AnchoredDirectory, name: str, remaining: int, should_stop: Callable[[], bool]
) -> tuple[str, int]:
    descriptor = open_entry(anchor, name)
    if descriptor is None:
        _refuse()
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > min(_MAX_FILE_BYTES, remaining):
            _refuse()
        digest = hashlib.sha1(
            b"blob " + str(before.st_size).encode("ascii") + b"\0", usedforsecurity=False
        )
        consumed = 0
        while block := stream.read(min(1024 * 1024, before.st_size - consumed + 1)):
            if should_stop():
                _refuse()
            consumed += len(block)
            if consumed > before.st_size:
                _refuse()
            digest.update(block)
        after = os.fstat(stream.fileno())
        if consumed != before.st_size or (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            _refuse()
        return digest.hexdigest(), consumed


def verify_pinned_source(
    package: AnchoredDirectory,
    raw_manifest: str,
    tree_hash: str,
    should_stop: Callable[[], bool],
) -> None:
    """Read only the selected held package; no index, stat cache or ignore rules."""
    tracked = _manifest(raw_manifest, tree_hash)
    seen: set[tuple[str, ...]] = set()
    count = 0
    total = 0

    def visit(anchor: AnchoredDirectory, prefix: tuple[str, ...]) -> None:
        nonlocal count, total
        if len(prefix) > 64:
            _refuse()
        if should_stop():
            _refuse()
        entries = list_entries(anchor, limit=_MAX_FILES, should_stop=should_stop)
        count += len(entries)
        if count > _MAX_FILES:
            _refuse()
        for entry in entries:
            parts = (*prefix, entry.name)
            if not entry.is_safe:
                _refuse()
            if not prefix and entry.name == ".git":
                if entry.kind != AnchoredEntryKind.DIRECTORY:
                    _refuse()
                continue
            if entry.kind == AnchoredEntryKind.DIRECTORY:
                with open_child_directory(anchor, entry.name) as child:
                    visit(child, parts)
            elif entry.kind == AnchoredEntryKind.FILE:
                identity = tracked.get(parts)
                if identity is None:
                    if _runtime_cache(parts, tracked):
                        continue
                    _refuse()
                actual, size = _blob_digest(
                    anchor, entry.name, _MAX_TOTAL_BYTES - total, should_stop
                )
                total += size
                if actual != identity[1]:
                    _refuse()
                seen.add(parts)
            else:
                _refuse()

    try:
        visit(package, ())
        if seen != tracked.keys():
            _refuse()
    except (AnchoredDirectoryError, OSError, UnicodeError):
        raise ValueError(_REFUSAL) from None
