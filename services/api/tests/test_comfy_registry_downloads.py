from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_lm import comfy_registry_downloads as registry_downloads
from local_lm.comfy_registry import ComfyNodeResolution
from local_lm.comfy_registry_archives import ComfyRegistryArchiveError
from local_lm.comfy_registry_downloads import (
    ComfyRegistryArchiveDownloader,
    ComfyRegistryDownloadError,
)


def archive_bytes(entries: dict[str, bytes] | None = None) -> bytes:
    output = io.BytesIO()
    values = entries or {"__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in values.items():
            archive.writestr(name, content)
    return output.getvalue()


def resolution(**updates: Any) -> ComfyNodeResolution:
    value = ComfyNodeResolution(
        package_id="example-pack",
        declared_version="1.2.3",
        node_types=("ExampleNode",),
        install_kind="registry_archive",
        repository_url="https://github.com/example/example-pack.git",
        registry_record_id="record-123",
        download_url="https://cdn.comfy.org/example/example-pack/1.2.3/node.zip",
    )
    return replace(value, **updates)


def commit_resolution(**updates: Any) -> ComfyNodeResolution:
    value = ComfyNodeResolution(
        package_id="example-pack",
        declared_version="a" * 40,
        node_types=("ExampleNode",),
        install_kind="git_commit",
        repository_url="https://github.com/example/example-pack.git",
    )
    return replace(value, **updates)


async def download_with(
    handler: httpx.AsyncBaseTransport,
    destination: Path,
    *,
    value: ComfyNodeResolution | None = None,
    progress: registry_downloads.DownloadProgress | None = None,
) -> Any:
    downloader = ComfyRegistryArchiveDownloader(transport=handler)
    try:
        return await downloader.download_and_stage(
            value or resolution(),
            destination,
            progress=progress,
        )
    finally:
        await downloader.close()


async def test_exact_archive_download_is_hashed_staged_and_reported(tmp_path: Path) -> None:
    payload = archive_bytes(
        {
            "__init__.py": b"NODE_CLASS_MAPPINGS = {}\n",
            "prestartup_script.py": b"pass\n",
        }
    )
    requests: list[httpx.Request] = []
    progress: list[tuple[int, int | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=payload)

    async def publish(downloaded: int, total: int | None) -> None:
        progress.append((downloaded, total))

    destination = tmp_path / "staged"
    report = await download_with(
        httpx.MockTransport(handler),
        destination,
        progress=publish,
    )

    assert [request.url for request in requests] == [
        httpx.URL("https://cdn.comfy.org/example/example-pack/1.2.3/node.zip")
    ]
    assert requests[0].headers["accept-encoding"] == "identity"
    assert report.archive_sha256 == hashlib.sha256(payload).hexdigest()
    assert report.startup_hooks == ("prestartup_script.py",)
    assert (destination / "__init__.py").is_file()
    assert progress == [(0, len(payload)), (len(payload), len(payload))]
    assert not list(tmp_path.glob("registry-archive-*"))


async def test_commit_archive_uses_exact_codeload_url_and_removes_wrapper(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    payload = archive_bytes(
        {
            f"example-pack-{revision}/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n",
            f"example-pack-{revision}/requirements.txt": b"pillow>=12\n",
        }
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=payload)

    destination = tmp_path / "staged"
    report = await download_with(
        httpx.MockTransport(handler),
        destination,
        value=commit_resolution(),
    )

    assert [request.url for request in requests] == [
        httpx.URL(f"https://codeload.github.com/example/example-pack/zip/{revision}")
    ]
    assert (destination / "__init__.py").is_file()
    assert (destination / "requirements.txt").is_file()
    assert not (destination / f"example-pack-{revision}").exists()
    assert report.dependency_manifests == ("requirements.txt",)
    assert report.top_level_entries == ("__init__.py", "requirements.txt")


async def test_missing_content_length_reports_indeterminate_progress(tmp_path: Path) -> None:
    payload = archive_bytes()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(payload))

    progress: list[tuple[int, int | None]] = []

    async def publish(downloaded: int, total: int | None) -> None:
        progress.append((downloaded, total))

    await download_with(
        httpx.MockTransport(handler),
        tmp_path / "staged",
        progress=publish,
    )
    assert progress == [(0, None), (len(payload), None)]


@pytest.mark.parametrize(
    "value",
    [
        resolution(install_kind="git_commit"),
        resolution(error_code="registry_version_not_found"),
        resolution(registry_record_id=None),
        resolution(download_url=None),
    ],
)
async def test_incomplete_resolution_is_rejected_before_network(
    tmp_path: Path,
    value: ComfyNodeResolution,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid resolution must not reach the network")

    with pytest.raises(ComfyRegistryDownloadError, match="archive"):
        await download_with(
            httpx.MockTransport(handler),
            tmp_path / "staged",
            value=value,
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.comfy.org/example/node.zip",
        "https://attacker.invalid/example/node.zip",
        "https://user@cdn.comfy.org/example/node.zip",
        "https://cdn.comfy.org:444/example/node.zip",
        "https://cdn.comfy.org/example/node.zip?token=value",
        "https://cdn.comfy.org/example/node.bin",
    ],
)
async def test_untrusted_archive_urls_are_rejected_before_network(
    tmp_path: Path,
    url: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("untrusted URL must not reach the network")

    with pytest.raises(ComfyRegistryDownloadError, match="untrusted archive URL"):
        await download_with(
            httpx.MockTransport(handler),
            tmp_path / "staged",
            value=resolution(download_url=url),
        )


async def test_redirects_are_not_followed_and_temporary_file_is_removed(
    tmp_path: Path,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "https://attacker.invalid/node.zip"},
        )

    with pytest.raises(httpx.HTTPStatusError):
        await download_with(httpx.MockTransport(handler), tmp_path / "staged")
    assert not list(tmp_path.glob("registry-archive-*"))


@pytest.mark.parametrize("value", ["invalid", "-1"])
async def test_invalid_content_length_is_rejected(tmp_path: Path, value: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-length": value}, content=b"")

    with pytest.raises(ComfyRegistryDownloadError, match="invalid Content-Length"):
        await download_with(httpx.MockTransport(handler), tmp_path / "staged")


async def test_declared_or_streamed_oversize_download_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_downloads, "MAX_REGISTRY_ARCHIVE_DOWNLOAD_BYTES", 8)

    def declared(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-length": "9"}, content=b"")

    with pytest.raises(ComfyRegistryDownloadError, match="exceeds the size limit"):
        await download_with(httpx.MockTransport(declared), tmp_path / "declared")

    def streamed(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(b"123456789"))

    with pytest.raises(ComfyRegistryDownloadError, match="exceeds the size limit"):
        await download_with(httpx.MockTransport(streamed), tmp_path / "streamed")
    assert not list(tmp_path.glob("registry-archive-*"))


async def test_truncated_or_encoded_response_is_rejected(tmp_path: Path) -> None:
    payload = archive_bytes()

    def truncated(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": str(len(payload) + 1)},
            stream=httpx.ByteStream(payload),
        )

    with pytest.raises(ComfyRegistryDownloadError, match="does not match Content-Length"):
        await download_with(httpx.MockTransport(truncated), tmp_path / "truncated")

    def encoded(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=httpx.ByteStream(payload),
        )

    with pytest.raises(ComfyRegistryDownloadError, match="content encoding"):
        await download_with(httpx.MockTransport(encoded), tmp_path / "encoded")


async def test_invalid_zip_is_removed_without_staged_content(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not a zip")

    destination = tmp_path / "staged"
    with pytest.raises(ComfyRegistryArchiveError, match="invalid Comfy Registry archive"):
        await download_with(httpx.MockTransport(handler), destination)
    assert not destination.exists()
    assert not list(tmp_path.glob("registry-archive-*"))


async def test_progress_failures_do_not_discard_a_valid_download(tmp_path: Path) -> None:
    payload = archive_bytes()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async def broken_progress(_downloaded: int, _total: int | None) -> None:
        raise RuntimeError("progress sink unavailable")

    report = await download_with(
        httpx.MockTransport(handler),
        tmp_path / "staged",
        progress=broken_progress,
    )
    assert report.archive_sha256 == hashlib.sha256(payload).hexdigest()


async def test_cancellation_removes_partial_download(tmp_path: Path) -> None:
    payload = archive_bytes()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async def cancel(_downloaded: int, _total: int | None) -> None:
        raise asyncio.CancelledError

    destination = tmp_path / "staged"
    with pytest.raises(asyncio.CancelledError):
        await download_with(
            httpx.MockTransport(handler),
            destination,
            progress=cancel,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("registry-archive-*"))


async def test_cancellation_waits_for_staging_and_removes_its_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = archive_bytes()
    started = threading.Event()
    release = threading.Event()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    def slow_stage(
        _archive: Path,
        destination: Path,
        *,
        expected_sha256: str,
        strip_single_root: bool,
    ) -> object:
        assert expected_sha256 == hashlib.sha256(payload).hexdigest()
        assert strip_single_root is False
        destination.mkdir()
        (destination / "partial.py").write_text("partial")
        started.set()
        assert release.wait(5)
        return object()

    monkeypatch.setattr(registry_downloads, "stage_comfy_registry_archive", slow_stage)
    destination = tmp_path / "staged"
    task = asyncio.create_task(download_with(httpx.MockTransport(handler), destination))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not destination.exists()
    assert not list(tmp_path.glob("registry-archive-*"))
