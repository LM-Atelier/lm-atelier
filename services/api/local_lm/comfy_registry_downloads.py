from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path

import httpx

from .comfy_registry import ComfyNodeResolution
from .comfy_registry_archives import (
    MAX_ARCHIVE_BYTES,
    ComfyRegistryArchiveReport,
    stage_comfy_registry_archive,
)
from .comfy_registry_sources import (
    ComfyPackageSourceError,
    ComfyPackageSourceIdentity,
    resolve_comfy_package_source,
)
from .network import shared_tls_context

logger = logging.getLogger(__name__)

MAX_REGISTRY_ARCHIVE_DOWNLOAD_BYTES = MAX_ARCHIVE_BYTES
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

DownloadProgress = Callable[[int, int | None], Awaitable[None]]


class ComfyRegistryDownloadError(ValueError):
    pass


class ComfyRegistryArchiveDownloader:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            headers={
                "accept": "application/zip",
                "accept-encoding": "identity",
                "user-agent": "local-lm/0.1",
            },
            timeout=httpx.Timeout(30, read=120),
            follow_redirects=False,
            verify=shared_tls_context(),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def download_and_stage(
        self,
        resolution: ComfyNodeResolution,
        destination: Path,
        *,
        progress: DownloadProgress | None = None,
    ) -> ComfyRegistryArchiveReport:
        source = _resolved_archive_source(resolution)
        if destination.exists() or destination.is_symlink():
            raise ComfyRegistryDownloadError("archive staging destination already exists")
        if not destination.parent.is_dir():
            raise ComfyRegistryDownloadError("archive staging parent does not exist")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix="registry-archive-",
            suffix=".zip.part",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            archive_sha256 = await self._download(source.download_url, temporary, progress)
            return await _stage_without_orphan(
                temporary,
                destination,
                archive_sha256,
                strip_single_root=source.install_kind == "git_commit",
            )
        finally:
            temporary.unlink(missing_ok=True)

    async def _download(
        self,
        url: str,
        temporary: Path,
        progress: DownloadProgress | None,
    ) -> str:
        async with self._client.stream("GET", url) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ComfyRegistryDownloadError(
                    f"registry archive server returned HTTP {response.status_code}"
                )
            encoding = response.headers.get("content-encoding", "identity").strip().lower()
            if encoding not in {"", "identity"}:
                raise ComfyRegistryDownloadError(
                    "registry archive server used unsupported content encoding"
                )
            expected_bytes = _content_length(response)
            await _publish_progress(progress, 0, expected_bytes)

            digest = hashlib.sha256()
            downloaded = 0
            with temporary.open("wb") as output:
                async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_REGISTRY_ARCHIVE_DOWNLOAD_BYTES:
                        raise ComfyRegistryDownloadError(
                            "registry archive download exceeds the size limit"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                    await _publish_progress(progress, downloaded, expected_bytes)
                output.flush()
                os.fsync(output.fileno())
        if expected_bytes is not None and downloaded != expected_bytes:
            raise ComfyRegistryDownloadError(
                "registry archive download size does not match Content-Length"
            )
        return digest.hexdigest()


def _resolved_archive_source(
    resolution: ComfyNodeResolution,
) -> ComfyPackageSourceIdentity:
    try:
        return resolve_comfy_package_source(resolution)
    except ComfyPackageSourceError as exc:
        if "download URL" in str(exc):
            raise ComfyRegistryDownloadError("resolution has an untrusted archive URL") from exc
        raise ComfyRegistryDownloadError(
            "resolution does not identify an exact registry archive or exact commit archive"
        ) from exc


def _content_length(response: httpx.Response) -> int | None:
    value = response.headers.get("content-length")
    if value is None:
        return None
    try:
        expected = int(value)
    except ValueError as exc:
        raise ComfyRegistryDownloadError(
            "registry archive server returned an invalid Content-Length"
        ) from exc
    if expected < 0:
        raise ComfyRegistryDownloadError(
            "registry archive server returned an invalid Content-Length"
        )
    if expected > MAX_REGISTRY_ARCHIVE_DOWNLOAD_BYTES:
        raise ComfyRegistryDownloadError("registry archive download exceeds the size limit")
    return expected


async def _publish_progress(
    callback: DownloadProgress | None,
    downloaded: int,
    total: int | None,
) -> None:
    if callback is None:
        return
    try:
        await callback(downloaded, total)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Could not publish registry archive download progress", exc_info=True)


async def _stage_without_orphan(
    temporary: Path,
    destination: Path,
    archive_sha256: str,
    *,
    strip_single_root: bool,
) -> ComfyRegistryArchiveReport:
    task = asyncio.create_task(
        asyncio.to_thread(
            stage_comfy_registry_archive,
            temporary,
            destination,
            expected_sha256=archive_sha256,
            strip_single_root=strip_single_root,
        )
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        with suppress(Exception):
            task.result()
        shutil.rmtree(destination, ignore_errors=True)
        raise
