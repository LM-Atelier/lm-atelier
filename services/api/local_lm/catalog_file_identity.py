from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from .catalog import HuggingFaceCatalog
from .workflow_asset_bindings import is_immutable_install_revision

_MAX_FILE_BYTES = 10 * 1024 * 1024
_MAX_TOTAL_BYTES = 40 * 1024 * 1024
_MAX_FILES = 16
_HASH_TIMEOUT_SECONDS = 10.0

VerificationIssue = Literal[
    "size-unavailable",
    "verification-limit",
    "source-unavailable",
    "read-failed",
    "size-changed",
    "timed-out",
]
_ISSUE_DETAILS: dict[VerificationIssue, str] = {
    "size-unavailable": "Some file sizes are unavailable. Choose another version.",
    "verification-limit": (
        "The selected files exceed the verification limit. Choose fewer files or another version."
    ),
    "source-unavailable": "A file's source could not be verified. Choose another version.",
    "read-failed": "Some selected files could not be read. Run the install check again.",
    "size-changed": "A file did not match its reported size. Run the install check again.",
    "timed-out": "File verification timed out. Run the install check again.",
}


@dataclass(frozen=True, slots=True)
class CatalogFileVerification:
    files: list[dict[str, Any]]
    issues: frozenset[VerificationIssue]

    @property
    def detail(self) -> str | None:
        if not self.issues:
            return None
        return "The selected files could not be fully verified. " + " ".join(
            detail for issue, detail in _ISSUE_DETAILS.items() if issue in self.issues
        )


class CatalogFileReader(Protocol):
    async def __call__(
        self, remote_id: str, revision: str, filename: str, *, max_bytes: int
    ) -> bytes: ...


async def hash_selected_catalog_files(
    files: list[dict[str, Any]],
    *,
    provider: str,
    remote_id: str,
    revision: str,
    read: CatalogFileReader,
) -> CatalogFileVerification:
    """Hash complete bounded files and explain why missing evidence could not be verified.

    Provider hashes stay authoritative. Failed reads and limits preserve the
    original metadata; only a complete file at a pinned revision adds a hash.
    """
    result = [dict(item) for item in files]
    issues: set[VerificationIssue] = set()
    if provider != "huggingface":
        return CatalogFileVerification(result, frozenset())
    remaining = _MAX_TOTAL_BYTES
    reads = 0
    try:
        async with asyncio.timeout(_HASH_TIMEOUT_SECONDS):
            for item in result:
                if item.get("sha256") is not None:
                    continue
                size = item.get("size")
                if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                    issues.add("size-unavailable")
                    continue
                if size > _MAX_FILE_BYTES or size + 1 > remaining or reads >= _MAX_FILES:
                    issues.add("verification-limit")
                    continue
                source = (
                    item.get("source_remote_id"),
                    item.get("source_revision"),
                    item.get("source_filename"),
                )
                if all(value is None for value in source):
                    source = (remote_id, revision, item.get("filename"))
                source_remote, source_revision, source_filename = source
                if (
                    not isinstance(source_remote, str)
                    or not HuggingFaceCatalog.validate_item_id(source_remote)
                    or not isinstance(source_revision, str)
                    or not is_immutable_install_revision("huggingface", source_revision)
                    or not isinstance(source_filename, str)
                    or not source_filename
                ):
                    issues.add("source-unavailable")
                    continue
                remaining -= size + 1
                reads += 1
                try:
                    content = await read(
                        source_remote, source_revision, source_filename, max_bytes=size + 1
                    )
                except (httpx.HTTPError, OSError, ValueError):
                    issues.add("read-failed")
                    continue
                # The extra byte distinguishes a whole file from a full prefix.
                if len(content) == size:
                    item["sha256"] = hashlib.sha256(content).hexdigest()
                else:
                    issues.add("size-changed")
    except TimeoutError:
        issues.add("timed-out")
    return CatalogFileVerification(result, frozenset(issues))
