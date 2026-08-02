from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import parse_tag
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from .comfy_registry_wheel_artifacts import (
    MAX_WHEEL_ARTIFACT_BYTES,
    ComfyRegistryWheelArtifact,
    ComfyRegistryWheelArtifactManifest,
)
from .network import shared_tls_context

logger = logging.getLogger(__name__)

MAX_REGISTRY_WHEEL_STAGE_ARTIFACTS = 256
MAX_REGISTRY_WHEEL_STAGE_BYTES = 32 * 1024 * 1024 * 1024
MAX_REGISTRY_WHEEL_METADATA_BYTES = 1024 * 1024
WHEEL_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_DIGEST_CHARACTERS = frozenset("0123456789abcdef")


class ComfyRegistryWheelDownloadError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ComfyRegistryStagedWheel:
    filename: str
    sha256: str
    size_bytes: int
    metadata_filename: str | None
    metadata_sha256: str | None
    metadata_size_bytes: int | None


@dataclass(frozen=True)
class ComfyRegistryWheelStageReport:
    artifact_manifest_sha256: str
    stage_manifest_sha256: str
    total_bytes: int
    artifacts: tuple[ComfyRegistryStagedWheel, ...]


WheelDownloadProgress = Callable[[str, int, int | None], Awaitable[None]]


class ComfyRegistryWheelDownloader:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            headers={
                "accept": "application/octet-stream, text/plain;q=0.9",
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
        manifest: ComfyRegistryWheelArtifactManifest,
        destination: Path,
        *,
        progress: WheelDownloadProgress | None = None,
    ) -> ComfyRegistryWheelStageReport:
        artifact_payload, artifacts = _validated_manifest(manifest)
        parent, lock = _stage_target(destination)
        lock_descriptor = _acquire_lock(lock)
        staging: Path | None = None
        try:
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".registry-wheels-{destination.name}-",
                    dir=parent,
                )
            )
            staged: list[ComfyRegistryStagedWheel] = []
            total_bytes = 0
            for artifact in artifacts:
                wheel_path = staging / artifact.filename
                wheel_size = await self._download(
                    artifact.url,
                    wheel_path,
                    expected_sha256=artifact.sha256,
                    expected_size=artifact.size_bytes,
                    maximum_size=artifact.size_bytes,
                    progress=progress,
                )
                metadata_filename: str | None = None
                metadata_size: int | None = None
                if artifact.metadata_sha256 is not None:
                    metadata_filename = f"{artifact.filename}.metadata"
                    metadata_size = await self._download(
                        f"{artifact.url}.metadata",
                        staging / metadata_filename,
                        expected_sha256=artifact.metadata_sha256,
                        expected_size=None,
                        maximum_size=MAX_REGISTRY_WHEEL_METADATA_BYTES,
                        progress=progress,
                    )
                total_bytes += wheel_size + (metadata_size or 0)
                staged.append(
                    ComfyRegistryStagedWheel(
                        artifact.filename,
                        artifact.sha256,
                        wheel_size,
                        metadata_filename,
                        artifact.metadata_sha256,
                        metadata_size,
                    )
                )
            report, encoded = _stage_report(manifest, artifact_payload, tuple(staged), total_bytes)
            _write_new_file(staging / "stage-manifest.json", encoded)
            if destination.exists() or destination.is_symlink():
                raise ComfyRegistryWheelDownloadError(
                    "stage_destination_exists",
                    "Registry wheel staging destination already exists",
                )
            os.rename(staging, destination)
            staging = None
            return report
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            os.close(lock_descriptor)
            lock.unlink(missing_ok=True)

    async def _download(
        self,
        url: str,
        destination: Path,
        *,
        expected_sha256: str,
        expected_size: int | None,
        maximum_size: int,
        progress: WheelDownloadProgress | None,
    ) -> int:
        async with self._client.stream("GET", url) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ComfyRegistryWheelDownloadError(
                    "unexpected_download_status",
                    f"Package server returned HTTP {response.status_code}",
                )
            encoding = response.headers.get("content-encoding", "identity").strip().lower()
            if encoding not in {"", "identity"}:
                raise ComfyRegistryWheelDownloadError(
                    "unsupported_content_encoding",
                    "Package server used an unsupported content encoding",
                )
            declared_size = _content_length(response)
            if expected_size is not None and declared_size not in {None, expected_size}:
                raise ComfyRegistryWheelDownloadError(
                    "download_size_mismatch",
                    "Package Content-Length does not match the locked artifact size",
                )
            if declared_size is not None and declared_size > maximum_size:
                raise ComfyRegistryWheelDownloadError(
                    "download_too_large",
                    "Package download exceeds the locked size limit",
                )
            progress_total = expected_size if expected_size is not None else declared_size
            await _publish_progress(progress, destination.name, 0, progress_total)

            digest = hashlib.sha256()
            downloaded = 0
            with destination.open("xb") as output:
                async for chunk in response.aiter_bytes(WHEEL_DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > maximum_size:
                        raise ComfyRegistryWheelDownloadError(
                            "download_too_large",
                            "Package download exceeds the locked size limit",
                        )
                    digest.update(chunk)
                    output.write(chunk)
                    await _publish_progress(
                        progress,
                        destination.name,
                        downloaded,
                        progress_total,
                    )
                output.flush()
                os.fsync(output.fileno())
        if declared_size is not None and downloaded != declared_size:
            raise ComfyRegistryWheelDownloadError(
                "download_size_mismatch",
                "Package bytes do not match Content-Length",
            )
        if expected_size is not None and downloaded != expected_size:
            raise ComfyRegistryWheelDownloadError(
                "download_size_mismatch",
                "Package bytes do not match the locked artifact size",
            )
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            raise ComfyRegistryWheelDownloadError(
                "download_hash_mismatch",
                "Package bytes do not match the locked SHA-256 hash",
            )
        return downloaded


def _validated_manifest(
    manifest: ComfyRegistryWheelArtifactManifest,
) -> tuple[dict[str, object], tuple[ComfyRegistryWheelArtifact, ...]]:
    if not isinstance(manifest, ComfyRegistryWheelArtifactManifest):
        raise ComfyRegistryWheelDownloadError(
            "invalid_artifact_manifest", "Registry wheel artifact manifest is invalid"
        )
    declaration_sha256 = _digest(manifest.declaration_sha256, "declaration")
    target_sha256 = _digest(manifest.target_sha256, "target")
    manifest_sha256 = _digest(manifest.manifest_sha256, "manifest")
    if len(manifest.artifacts) > MAX_REGISTRY_WHEEL_STAGE_ARTIFACTS:
        raise ComfyRegistryWheelDownloadError(
            "too_many_wheel_artifacts", "Registry wheel manifest has too many artifacts"
        )
    artifacts = tuple(_validated_artifact(artifact) for artifact in manifest.artifacts)
    if sum(artifact.size_bytes for artifact in artifacts) > MAX_REGISTRY_WHEEL_STAGE_BYTES:
        raise ComfyRegistryWheelDownloadError(
            "wheel_stage_too_large", "Registry wheel stage exceeds the total size limit"
        )
    if artifacts != tuple(sorted(artifacts, key=lambda item: (item.name, item.requirement))):
        raise ComfyRegistryWheelDownloadError(
            "invalid_artifact_manifest", "Registry wheel artifacts are not canonical"
        )
    names = [artifact.name for artifact in artifacts]
    filenames = [artifact.filename for artifact in artifacts]
    if len(names) != len(set(names)) or len(filenames) != len(set(filenames)):
        raise ComfyRegistryWheelDownloadError(
            "invalid_artifact_manifest", "Registry wheel artifacts contain duplicates"
        )
    payload = {
        "version": 1,
        "declaration_sha256": declaration_sha256,
        "target_sha256": target_sha256,
        "artifacts": [_artifact_payload(artifact) for artifact in artifacts],
    }
    if not hmac.compare_digest(_payload_sha256(payload), manifest_sha256):
        raise ComfyRegistryWheelDownloadError(
            "artifact_manifest_hash_mismatch",
            "Registry wheel artifact manifest hash does not match its contents",
        )
    return payload, artifacts


def _validated_artifact(artifact: object) -> ComfyRegistryWheelArtifact:
    if not isinstance(artifact, ComfyRegistryWheelArtifact):
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry wheel artifact is invalid"
        )
    name = _text(artifact.name, "package name", 200)
    if canonicalize_name(name) != name:
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry wheel package name is not canonical"
        )
    filename = _text(artifact.filename, "wheel filename", 500)
    try:
        wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(filename)
    except InvalidWheelFilename as exc:
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry wheel filename is invalid"
        ) from exc
    try:
        version = Version(_text(artifact.version, "package version", 200))
    except InvalidVersion as exc:
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry wheel version is invalid"
        ) from exc
    if str(wheel_name) != name or wheel_version != version or str(version) != artifact.version:
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry wheel identity does not match its filename"
        )
    try:
        requirement = Requirement(_text(artifact.requirement, "requirement", 1_000))
    except InvalidRequirement as exc:
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry wheel requirement is invalid"
        ) from exc
    if (
        requirement.url is not None
        or canonicalize_name(requirement.name) != name
        or not requirement.specifier.contains(version, prereleases=True)
    ):
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry wheel does not satisfy its requirement"
        )
    wheel_tag_values = tuple(sorted(str(tag) for tag in wheel_tags))
    if artifact.wheel_tags != wheel_tag_values:
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry wheel compatibility tags are invalid"
        )
    compatibility = _text(artifact.compatibility_tag, "compatibility tag", 200)
    try:
        parsed_compatibility = parse_tag(compatibility)
    except ValueError as exc:
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry selected wheel tag is invalid"
        ) from exc
    if len(parsed_compatibility) != 1 or compatibility not in wheel_tag_values:
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry selected wheel tag is invalid"
        )
    size = artifact.size_bytes
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or size > MAX_WHEEL_ARTIFACT_BYTES
    ):
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry wheel size is invalid"
        )
    _wheel_url(artifact.url, filename)
    _digest(artifact.sha256, "wheel")
    if artifact.metadata_sha256 is not None:
        _digest(artifact.metadata_sha256, "wheel metadata")
    return artifact


def _stage_target(destination: Path) -> tuple[Path, Path]:
    if not isinstance(destination, Path) or not destination.name or len(destination.name) > 200:
        raise ComfyRegistryWheelDownloadError(
            "invalid_stage_destination", "Registry wheel staging destination is invalid"
        )
    if _has_control(destination.name):
        raise ComfyRegistryWheelDownloadError(
            "invalid_stage_destination", "Registry wheel staging destination is invalid"
        )
    if destination.exists() or destination.is_symlink():
        raise ComfyRegistryWheelDownloadError(
            "stage_destination_exists", "Registry wheel staging destination already exists"
        )
    parent = destination.parent
    if not parent.is_dir():
        raise ComfyRegistryWheelDownloadError(
            "invalid_stage_destination", "Registry wheel staging parent does not exist"
        )
    return parent, parent / f".{destination.name}.lock"


def _acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ComfyRegistryWheelDownloadError(
            "stage_locked", "Registry wheel staging destination is already in use"
        ) from exc


def _wheel_url(value: object, filename: str) -> str:
    url = _text(value, "wheel URL", 2_000)
    parsed = urlparse(url)
    decoded = [unquote(segment) for segment in parsed.path.split("/") if segment]
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/packages/")
        or not decoded
        or decoded[-1] != filename
        or any(
            segment in {".", ".."} or "/" in segment or chr(92) in segment for segment in decoded
        )
    ):
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", "Registry wheel URL is not allowlisted"
        )
    return url


def _content_length(response: httpx.Response) -> int | None:
    value = response.headers.get("content-length")
    if value is None:
        return None
    try:
        size = int(value)
    except ValueError as exc:
        raise ComfyRegistryWheelDownloadError(
            "invalid_content_length", "Package server returned an invalid Content-Length"
        ) from exc
    if size < 0:
        raise ComfyRegistryWheelDownloadError(
            "invalid_content_length", "Package server returned an invalid Content-Length"
        )
    return size


async def _publish_progress(
    callback: WheelDownloadProgress | None,
    filename: str,
    downloaded: int,
    total: int | None,
) -> None:
    if callback is None:
        return
    try:
        await callback(filename, downloaded, total)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Could not publish Registry wheel download progress", exc_info=True)


def _stage_report(
    manifest: ComfyRegistryWheelArtifactManifest,
    artifact_payload: dict[str, object],
    artifacts: tuple[ComfyRegistryStagedWheel, ...],
    total_bytes: int,
) -> tuple[ComfyRegistryWheelStageReport, bytes]:
    payload = {
        "version": 1,
        "artifact_manifest_sha256": manifest.manifest_sha256,
        "artifact_manifest": artifact_payload,
        "total_bytes": total_bytes,
        "files": [
            {
                "filename": item.filename,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "metadata_filename": item.metadata_filename,
                "metadata_sha256": item.metadata_sha256,
                "metadata_size_bytes": item.metadata_size_bytes,
            }
            for item in artifacts
        ],
    }
    encoded = _encode_payload(payload, trailing_newline=True)
    stage_sha256 = hashlib.sha256(encoded).hexdigest()
    return (
        ComfyRegistryWheelStageReport(
            manifest.manifest_sha256,
            stage_sha256,
            total_bytes,
            artifacts,
        ),
        encoded,
    )


def _write_new_file(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _artifact_payload(artifact: ComfyRegistryWheelArtifact) -> dict[str, object]:
    return {
        "name": artifact.name,
        "version": artifact.version,
        "requirement": artifact.requirement,
        "filename": artifact.filename,
        "url": artifact.url,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "metadata_sha256": artifact.metadata_sha256,
        "compatibility_tag": artifact.compatibility_tag,
        "wheel_tags": list(artifact.wheel_tags),
    }


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_encode_payload(payload, trailing_newline=False)).hexdigest()


def _encode_payload(payload: object, *, trailing_newline: bool) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return (encoded + ("\n" if trailing_newline else "")).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise ComfyRegistryWheelDownloadError(
            "invalid_artifact_manifest", "Registry wheel manifest cannot be encoded"
        ) from exc


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARACTERS for character in value)
    ):
        raise ComfyRegistryWheelDownloadError(
            "invalid_artifact_manifest", f"Registry wheel {label} hash is invalid"
        )
    return value


def _text(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or _has_control(value):
        raise ComfyRegistryWheelDownloadError(
            "invalid_wheel_artifact", f"Registry wheel {label} is invalid"
        )
    return value


def _has_control(value: str) -> bool:
    return any(character < " " or character == "\x7f" for character in value)
