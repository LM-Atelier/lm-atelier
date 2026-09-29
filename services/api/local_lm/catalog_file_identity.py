from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Protocol

import httpx

from .catalog import HuggingFaceCatalog
from .workflow_asset_bindings import is_immutable_install_revision

_MAX_FILE_BYTES = 4 * 1024 * 1024
_MAX_TOTAL_BYTES = 16 * 1024 * 1024
_MAX_FILES = 16
_HASH_TIMEOUT_SECONDS = 10.0


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
) -> list[dict[str, Any]]:
    """Add missing hashes only after reading complete bounded files at pinned revisions.

    Provider hashes stay authoritative. Files outside the read budget retain
    their existing metadata so this enrichment does not change legacy admission.
    """
    result = [dict(item) for item in files]
    if provider != "huggingface":
        return result
    remaining = _MAX_TOTAL_BYTES
    reads = 0
    try:
        async with asyncio.timeout(_HASH_TIMEOUT_SECONDS):
            for item in result:
                size = item.get("size")
                if (
                    item.get("sha256") is not None
                    or not isinstance(size, int)
                    or isinstance(size, bool)
                    or not 0 < size <= _MAX_FILE_BYTES
                    or size + 1 > remaining
                    or reads >= _MAX_FILES
                ):
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
                    continue
                remaining -= size + 1
                reads += 1
                try:
                    content = await read(
                        source_remote, source_revision, source_filename, max_bytes=size + 1
                    )
                except (httpx.HTTPError, OSError, ValueError):
                    continue
                # The extra byte distinguishes a whole file from a full prefix.
                if len(content) == size:
                    item["sha256"] = hashlib.sha256(content).hexdigest()
    except TimeoutError:
        pass
    return result
