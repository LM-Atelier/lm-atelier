from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from packaging.markers import default_environment

import local_lm.comfy_registry_wheel_downloads as download_module
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactManifest,
    resolve_comfy_registry_wheel_artifacts,
)
from local_lm.comfy_registry_wheel_downloads import (
    ComfyRegistryWheelDownloader,
    ComfyRegistryWheelDownloadError,
)
from local_lm.filesystem_links import AnchoredDirectory, AnchoredDirectoryError
from local_lm.shared_asset_lock_v1 import hold

_TAG = "py3-none-any"


def _manifest(
    entries: list[tuple[str, str, bytes, bytes | None]] | None = None,
) -> ComfyRegistryWheelArtifactManifest:
    selected = (
        entries
        if entries is not None
        else [("example-package", "1.2.3", b"wheel-bytes", b"metadata")]
    )
    declarations: list[str] = []
    documents: dict[str, object] = {}
    for name, version, wheel, metadata in selected:
        filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        declarations.append(f"{name}=={version}")
        documents[name] = {
            "meta": {"api-version": "1.4"},
            "name": name,
            "files": [
                {
                    "filename": filename,
                    "url": f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
                    "hashes": {"sha256": hashlib.sha256(wheel).hexdigest()},
                    "requires-python": ">=3.12",
                    "yanked": False,
                    "size": len(wheel),
                    "core-metadata": (
                        {"sha256": hashlib.sha256(metadata).hexdigest()}
                        if metadata is not None
                        else False
                    ),
                }
            ],
        }
    environment = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = ""
    return resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies(declarations),
        documents,
        marker_environment=environment,
        supported_tags=(_TAG,),
    )


async def _stage_with(
    transport: httpx.AsyncBaseTransport,
    destination: Path,
    *,
    manifest: ComfyRegistryWheelArtifactManifest | None = None,
    progress: download_module.WheelDownloadProgress | None = None,
) -> download_module.ComfyRegistryWheelStageReport:
    downloader = ComfyRegistryWheelDownloader(transport=transport)
    try:
        return await downloader.download_and_stage(
            manifest or _manifest(),
            destination,
            progress=progress,
        )
    finally:
        await downloader.close()


def _assert_stage_lock_is_unheld(parent: Path, destination: Path) -> None:
    """The lock entry stays; the next holder must be able to take it."""

    lock_name = f".{destination.name}.lock"
    assert (parent / lock_name).is_file()
    with AnchoredDirectory(parent) as anchor, hold(anchor, lock_name):
        pass


def _assert_clean(parent: Path, destination: Path) -> None:
    assert not destination.exists()
    lock = parent / f".{destination.name}.lock"
    if lock.exists():
        _assert_stage_lock_is_unheld(parent, destination)
    assert not list(parent.glob(f".registry-wheels-{destination.name}-*"))


async def test_wheel_and_hash_bound_metadata_are_staged_atomically(tmp_path: Path) -> None:
    wheel = b"wheel-bytes"
    metadata = b"metadata"
    requests: list[httpx.Request] = []
    progress: list[tuple[str, int, int | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = metadata if request.url.path.endswith(".metadata") else wheel
        return httpx.Response(200, content=content)

    async def publish(filename: str, downloaded: int, total: int | None) -> None:
        progress.append((filename, downloaded, total))

    destination = tmp_path / "staged"
    report = await _stage_with(
        httpx.MockTransport(handler),
        destination,
        progress=publish,
    )

    filename = "example_package-1.2.3-py3-none-any.whl"
    assert [str(request.url) for request in requests] == [
        f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
        f"https://files.pythonhosted.org/packages/aa/bb/{filename}.metadata",
    ]
    assert all(request.headers["accept-encoding"] == "identity" for request in requests)
    assert (destination / filename).read_bytes() == wheel
    assert (destination / f"{filename}.metadata").read_bytes() == metadata
    staged_manifest = (destination / "stage-manifest.json").read_bytes()
    document = json.loads(staged_manifest)
    assert document["artifact_manifest_sha256"] == report.artifact_manifest_sha256
    assert report.stage_manifest_sha256 == hashlib.sha256(staged_manifest).hexdigest()
    assert report.total_bytes == len(wheel) + len(metadata)
    assert progress == [
        (filename, 0, len(wheel)),
        (filename, len(wheel), len(wheel)),
        (f"{filename}.metadata", 0, len(metadata)),
        (f"{filename}.metadata", len(metadata), len(metadata)),
    ]
    _assert_stage_lock_is_unheld(tmp_path, destination)
    assert not list(tmp_path.glob(".registry-wheels-staged-*"))


async def test_artifact_without_hash_bound_metadata_downloads_only_wheel(
    tmp_path: Path,
) -> None:
    wheel = b"wheel-only"
    manifest = _manifest([("example-package", "1.2.3", wheel, None)])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=wheel)

    destination = tmp_path / "staged"
    report = await _stage_with(
        httpx.MockTransport(handler),
        destination,
        manifest=manifest,
    )

    assert len(requests) == 1
    assert report.artifacts[0].metadata_filename is None
    assert not (destination / f"{report.artifacts[0].filename}.metadata").exists()


async def test_empty_manifest_stages_a_durable_record_without_network(
    tmp_path: Path,
) -> None:
    manifest = _manifest([])

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("empty manifest must not reach the network")

    destination = tmp_path / "staged"
    report = await _stage_with(
        httpx.MockTransport(handler),
        destination,
        manifest=manifest,
    )

    assert report.artifacts == ()
    assert report.total_bytes == 0
    assert (destination / "stage-manifest.json").is_file()


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda manifest: replace(manifest, manifest_sha256="0" * 64),
            "artifact_manifest_hash_mismatch",
        ),
        (
            lambda manifest: replace(manifest, target_sha256="A" * 64),
            "invalid_artifact_manifest",
        ),
        (
            lambda manifest: replace(
                manifest,
                artifacts=(
                    replace(manifest.artifacts[0], url="https://attacker.invalid/file.whl"),
                ),
            ),
            "invalid_wheel_artifact",
        ),
        (
            lambda manifest: replace(
                manifest,
                artifacts=(replace(manifest.artifacts[0], size_bytes=-1),),
            ),
            "invalid_wheel_artifact",
        ),
        (
            lambda manifest: replace(
                manifest,
                artifacts=(replace(manifest.artifacts[0], wheel_tags=("cp311-none-any",)),),
            ),
            "invalid_wheel_artifact",
        ),
    ],
)
async def test_invalid_manifest_and_artifacts_are_rejected_before_network(
    tmp_path: Path,
    mutate: Any,
    code: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid artifact must not reach the network")

    destination = tmp_path / "staged"
    with pytest.raises(ComfyRegistryWheelDownloadError) as raised:
        await _stage_with(
            httpx.MockTransport(handler),
            destination,
            manifest=mutate(_manifest()),
        )
    assert raised.value.code == code
    _assert_clean(tmp_path, destination)


async def test_duplicate_artifacts_and_artifact_limit_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    duplicate = replace(manifest, artifacts=(manifest.artifacts[0], manifest.artifacts[0]))

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid manifest must not reach the network")

    with pytest.raises(ComfyRegistryWheelDownloadError) as duplicate_error:
        await _stage_with(
            httpx.MockTransport(handler),
            tmp_path / "duplicate",
            manifest=duplicate,
        )
    assert duplicate_error.value.code == "invalid_artifact_manifest"

    monkeypatch.setattr(download_module, "MAX_REGISTRY_WHEEL_STAGE_ARTIFACTS", 0)
    with pytest.raises(ComfyRegistryWheelDownloadError) as limit_error:
        await _stage_with(
            httpx.MockTransport(handler),
            tmp_path / "limit",
            manifest=manifest,
        )
    assert limit_error.value.code == "too_many_wheel_artifacts"

    monkeypatch.setattr(download_module, "MAX_REGISTRY_WHEEL_STAGE_ARTIFACTS", 256)
    monkeypatch.setattr(download_module, "MAX_REGISTRY_WHEEL_STAGE_BYTES", 0)
    with pytest.raises(ComfyRegistryWheelDownloadError) as size_error:
        await _stage_with(
            httpx.MockTransport(handler),
            tmp_path / "size",
            manifest=manifest,
        )
    assert size_error.value.code == "wheel_stage_too_large"


async def test_destination_parent_existing_target_and_lock_are_enforced(
    tmp_path: Path,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid destination must not reach the network")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ComfyRegistryWheelDownloadError) as exists_error:
        await _stage_with(httpx.MockTransport(handler), existing)
    assert exists_error.value.code == "stage_destination_exists"

    missing_parent = tmp_path / "missing" / "stage"
    with pytest.raises(ComfyRegistryWheelDownloadError) as parent_error:
        await _stage_with(httpx.MockTransport(handler), missing_parent)
    assert parent_error.value.code == "invalid_stage_destination"


async def test_redirect_partial_and_encoded_responses_fail_without_staging(
    tmp_path: Path,
) -> None:
    handlers = [
        lambda _request: httpx.Response(
            302, headers={"location": "https://attacker.invalid/file.whl"}
        ),
        lambda _request: httpx.Response(206, content=b"wheel-bytes"),
        lambda _request: httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=httpx.ByteStream(b"wheel-bytes"),
        ),
    ]
    for index, handler in enumerate(handlers):
        destination = tmp_path / f"staged-{index}"
        with pytest.raises((ComfyRegistryWheelDownloadError, httpx.HTTPStatusError)):
            await _stage_with(httpx.MockTransport(handler), destination)
        _assert_clean(tmp_path, destination)


@pytest.mark.parametrize("content_length", ["invalid", "-1"])
async def test_invalid_content_length_is_rejected(
    tmp_path: Path,
    content_length: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": content_length},
            stream=httpx.ByteStream(b"wheel-bytes"),
        )

    destination = tmp_path / "staged"
    with pytest.raises(ComfyRegistryWheelDownloadError) as raised:
        await _stage_with(httpx.MockTransport(handler), destination)
    assert raised.value.code == "invalid_content_length"
    _assert_clean(tmp_path, destination)


async def test_declared_streamed_and_locked_size_mismatches_are_rejected(
    tmp_path: Path,
) -> None:
    def declared(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": "12"},
            stream=httpx.ByteStream(b"wheel-bytes"),
        )

    with pytest.raises(ComfyRegistryWheelDownloadError) as declared_error:
        await _stage_with(httpx.MockTransport(declared), tmp_path / "declared")
    assert declared_error.value.code == "download_size_mismatch"

    def truncated(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(b"short"))

    with pytest.raises(ComfyRegistryWheelDownloadError) as truncated_error:
        await _stage_with(httpx.MockTransport(truncated), tmp_path / "truncated")
    assert truncated_error.value.code == "download_size_mismatch"

    def oversized(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(b"wheel-bytes-extra"))

    with pytest.raises(ComfyRegistryWheelDownloadError) as oversized_error:
        await _stage_with(httpx.MockTransport(oversized), tmp_path / "oversized")
    assert oversized_error.value.code == "download_too_large"


async def test_wheel_and_metadata_hash_mismatches_remove_entire_stage(
    tmp_path: Path,
) -> None:
    def wheel_mismatch(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"wrong-bytes")

    with pytest.raises(ComfyRegistryWheelDownloadError) as wheel_error:
        await _stage_with(httpx.MockTransport(wheel_mismatch), tmp_path / "wheel")
    assert wheel_error.value.code == "download_hash_mismatch"

    def metadata_mismatch(request: httpx.Request) -> httpx.Response:
        content = b"wrong---" if request.url.path.endswith(".metadata") else b"wheel-bytes"
        return httpx.Response(200, content=content)

    with pytest.raises(ComfyRegistryWheelDownloadError) as metadata_error:
        await _stage_with(httpx.MockTransport(metadata_mismatch), tmp_path / "metadata")
    assert metadata_error.value.code == "download_hash_mismatch"
    _assert_clean(tmp_path, tmp_path / "metadata")


async def test_metadata_size_limit_is_enforced(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = b"metadata" if request.url.path.endswith(".metadata") else b"wheel-bytes"
        return httpx.Response(200, content=content)

    destination = tmp_path / "staged"
    original = download_module.MAX_REGISTRY_WHEEL_METADATA_BYTES
    download_module.MAX_REGISTRY_WHEEL_METADATA_BYTES = 4
    try:
        with pytest.raises(ComfyRegistryWheelDownloadError) as raised:
            await _stage_with(httpx.MockTransport(handler), destination)
    finally:
        download_module.MAX_REGISTRY_WHEEL_METADATA_BYTES = original
    assert raised.value.code == "download_too_large"
    _assert_clean(tmp_path, destination)


async def test_progress_failure_does_not_discard_verified_bytes(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = b"metadata" if request.url.path.endswith(".metadata") else b"wheel-bytes"
        return httpx.Response(200, content=content)

    async def broken_progress(_filename: str, _downloaded: int, _total: int | None) -> None:
        raise RuntimeError("progress unavailable")

    destination = tmp_path / "staged"
    report = await _stage_with(
        httpx.MockTransport(handler),
        destination,
        progress=broken_progress,
    )
    assert report.total_bytes == len(b"wheel-bytesmetadata")
    assert destination.is_dir()


async def test_cancellation_removes_partial_files_lock_and_destination(
    tmp_path: Path,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"wheel-bytes")

    async def cancel(_filename: str, downloaded: int, _total: int | None) -> None:
        if downloaded:
            raise asyncio.CancelledError

    destination = tmp_path / "staged"
    with pytest.raises(asyncio.CancelledError):
        await _stage_with(
            httpx.MockTransport(handler),
            destination,
            progress=cancel,
        )
    _assert_clean(tmp_path, destination)


async def test_failure_on_second_artifact_never_exposes_first_artifact(
    tmp_path: Path,
) -> None:
    entries = [
        ("alpha", "1.0", b"alpha-wheel", None),
        ("beta", "2.0", b"beta-wheel", None),
    ]
    manifest = _manifest(entries)

    def handler(request: httpx.Request) -> httpx.Response:
        if "alpha" in request.url.path:
            return httpx.Response(200, content=b"alpha-wheel")
        return httpx.Response(200, content=b"wrong-wheel")

    destination = tmp_path / "staged"
    with pytest.raises(ComfyRegistryWheelDownloadError) as raised:
        await _stage_with(
            httpx.MockTransport(handler),
            destination,
            manifest=manifest,
        )
    assert raised.value.code == "download_size_mismatch"
    _assert_clean(tmp_path, destination)


_STAGE_HOLDER = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, sys.argv[1])
    from local_lm.filesystem_links import AnchoredDirectory
    from local_lm.shared_asset_lock_v1 import hold

    with AnchoredDirectory(Path(sys.argv[2])) as anchor:
        with hold(anchor, sys.argv[3]):
            print("HELD", flush=True)
            import time

            time.sleep(300)
    """
)


def _start_stage_holder(tmp_path: Path, lock_name: str) -> subprocess.Popen[str]:
    script = tmp_path / "stage-holder.py"
    script.write_text(_STAGE_HOLDER, encoding="utf-8")
    child = subprocess.Popen(
        [
            sys.executable,
            str(script),
            str(Path(__file__).resolve().parents[1]),
            str(tmp_path),
            lock_name,
        ],
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    try:
        assert child.stdout is not None
        announced = child.stdout.readline().strip()
        assert announced == "HELD", "the holder never took the lock"
        return child
    except BaseException:
        child.kill()
        child.wait(timeout=30)
        raise


async def test_a_live_holder_keeps_the_stage_locked(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("a locked stage must not reach the network")

    destination = tmp_path / "staged"
    lock_name = f".{destination.name}.lock"
    child = _start_stage_holder(tmp_path, lock_name)
    try:
        with pytest.raises(ComfyRegistryWheelDownloadError) as raised:
            await _stage_with(httpx.MockTransport(handler), destination)
        assert raised.value.code == "stage_locked"
    finally:
        child.kill()
        child.wait(timeout=30)


async def test_a_killed_holder_does_not_keep_the_stage_locked(tmp_path: Path) -> None:
    wheel = b"wheel-bytes"
    metadata = b"metadata"

    def handler(request: httpx.Request) -> httpx.Response:
        content = metadata if request.url.path.endswith(".metadata") else wheel
        return httpx.Response(200, content=content)

    destination = tmp_path / "staged"
    lock_name = f".{destination.name}.lock"
    child = _start_stage_holder(tmp_path, lock_name)
    child.kill()
    child.wait(timeout=30)

    deadline = time.monotonic() + 30
    last_error: ComfyRegistryWheelDownloadError | None = None
    while time.monotonic() < deadline:
        try:
            await _stage_with(httpx.MockTransport(handler), destination)
            break
        except ComfyRegistryWheelDownloadError as exc:
            last_error = exc
            if exc.code != "stage_locked":
                raise
            time.sleep(0.1)
    else:
        pytest.fail(
            "stage stayed locked after the holder died"
            if last_error is None
            else f"stage stayed locked after the holder died: {last_error.code}"
        )
    assert destination.is_dir()
    _assert_stage_lock_is_unheld(tmp_path, destination)


async def test_an_unanchorable_parent_fails_closed_as_stage_lock_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anchor failure must refuse, not stage with no lock held."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("an unanchored parent must not reach the network")

    class Boom:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AnchoredDirectoryError("directory containment could not be established")

    monkeypatch.setattr(download_module, "AnchoredDirectory", Boom)
    with pytest.raises(ComfyRegistryWheelDownloadError) as raised:
        await _stage_with(httpx.MockTransport(handler), tmp_path / "staged")
    assert raised.value.code == "stage_lock_failed"
