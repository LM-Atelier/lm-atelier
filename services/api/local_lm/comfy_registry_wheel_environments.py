from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import hmac
import io
import json
import os
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .comfy_registry_runtime import (
    ComfyRegistryRuntimeDistribution,
    ComfyRegistryRuntimeError,
    canonical_comfy_registry_runtime_distributions,
    comfy_registry_runtime_distribution_payload,
)
from .comfy_registry_wheel_artifacts import ComfyRegistryWheelArtifact
from .comfy_registry_wheel_closure import (
    ComfyRegistryWheelClosure,
    ComfyRegistryWheelClosureError,
    validate_comfy_registry_wheel_closure,
)
from .filesystem_links import is_link_or_reparse
from .processes import WINDOWS_CREATE_NO_WINDOW
from .subprocess_env import subprocess_environment

MAX_REGISTRY_WHEEL_ENVIRONMENT_FILES = 100_000
MAX_REGISTRY_WHEEL_ENVIRONMENT_BYTES = 32 * 1024 * 1024 * 1024
MAX_REGISTRY_WHEEL_METADATA_FILE_BYTES = 1024 * 1024
MAX_REGISTRY_WHEEL_ENVIRONMENT_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_REGISTRY_WHEEL_ARCHIVE_PATH_CHARACTERS = 1_000
MAX_REGISTRY_WHEEL_ARCHIVE_COMPONENT_CHARACTERS = 255
MAX_REGISTRY_WHEEL_EXPANSION_RATIO = 200
# The ratio only means anything once an entry is big enough to matter. A
# decompression bomb is dangerous because of what it expands *to*, not because
# it compressed well: an entry that unpacks to a few hundred kilobytes cannot
# fill a disk at any ratio, and the whole-environment caps above already bound
# what an archive set may expand to in total.
#
# Without this floor the check refuses ordinary packages for shipping
# compressible test data. pooch - a transitive dependency of scikit-image, and
# so of half the scientific ecosystem - includes `tests/data/large-data.txt`,
# which is 0.1 MB expanded and compresses at 321:1. Nothing about that is a
# bomb, and refusing it made an entire workflow uninstallable.
MAX_REGISTRY_WHEEL_UNCHECKED_ENTRY_BYTES = 64 * 1024 * 1024
WHEEL_HASH_CHUNK_BYTES = 1024 * 1024
WHEEL_INSTALL_TIMEOUT_SECONDS = 600
_DIGEST_CHARACTERS = frozenset("0123456789abcdefABCDEF")
WHEEL_OWNERSHIP_ATTESTATION = "wheel-source-record-v1"
REGISTRY_WHEEL_ENVIRONMENT_PREFIX = "registry-wheels-v3-"
_GENERATED_DISTRIBUTION_FILES = ("INSTALLER", "REQUESTED", "direct_url.json")
_RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class ComfyRegistryWheelEnvironmentError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ComfyRegistryWheelEnvironmentDistribution:
    name: str
    version: str
    dist_info: str


@dataclass(frozen=True)
class ComfyRegistryWheelEnvironmentReport:
    closure_sha256: str
    environment_sha256: str
    artifact_count: int
    file_count: int
    total_bytes: int
    distributions: tuple[ComfyRegistryWheelEnvironmentDistribution, ...]
    runtime_distributions: tuple[ComfyRegistryRuntimeDistribution, ...] = ()


@dataclass(frozen=True, slots=True)
class _PlannedInstalledFile:
    path: str
    source_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _WheelOwnership:
    record_path: str
    files: tuple[_PlannedInstalledFile, ...]


@dataclass(frozen=True, slots=True)
class _WheelOwnershipPlan:
    wheels: tuple[_WheelOwnership, ...]


async def assemble_comfy_registry_wheel_environment(
    closure: ComfyRegistryWheelClosure,
    wheel_files: Mapping[str, Path],
    *,
    python_executable: Path,
    destination: Path,
    media_worker_stopped: bool,
) -> ComfyRegistryWheelEnvironmentReport:
    """Atomically assemble a closed wheel set into an isolated offline overlay."""
    if media_worker_stopped is not True:
        raise ComfyRegistryWheelEnvironmentError(
            "media_worker_running",
            "The media worker must be stopped before changing its dependencies",
        )
    try:
        artifacts = validate_comfy_registry_wheel_closure(closure)
    except ComfyRegistryWheelClosureError as exc:
        raise ComfyRegistryWheelEnvironmentError("invalid_closure", str(exc)) from exc
    if not closure.complete:
        raise ComfyRegistryWheelEnvironmentError(
            "closure_incomplete", "Wheel dependencies must be fully closed before assembly"
        )
    executable = _python_executable(python_executable)
    wheels = await asyncio.to_thread(_wheel_inputs, artifacts, wheel_files)
    parent, lock = _destination(closure, destination)
    lock_descriptor = _acquire_lock(lock)
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}-",
                dir=parent,
            )
        )
        site_packages = staging / "site-packages"
        site_packages.mkdir()
        wheel_staging = staging / "wheels"
        staged_wheels, ownership_plan = await asyncio.to_thread(
            _stage_wheels, wheels, wheel_staging
        )
        if staged_wheels:
            await _run_pip(executable, staged_wheels, site_packages)
        await asyncio.to_thread(_remove_staged_wheels, wheel_staging)
        report, encoded = await asyncio.to_thread(
            _audit_environment,
            closure,
            artifacts,
            site_packages,
            ownership_plan,
        )
        _write_new(staging / "environment-manifest.json", encoded)
        if destination.exists() or destination.is_symlink():
            raise ComfyRegistryWheelEnvironmentError(
                "environment_destination_exists",
                "Wheel environment destination already exists",
            )
        os.rename(staging, destination)
        staging = None
        return report
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        os.close(lock_descriptor)
        lock.unlink(missing_ok=True)


def verify_comfy_registry_wheel_environment(
    destination: Path,
    *,
    expected_closure_sha256: str,
    expected_environment_sha256: str,
) -> ComfyRegistryWheelEnvironmentReport:
    """Revalidate a published inert overlay without importing from it."""
    closure_sha256 = _digest(expected_closure_sha256, "closure")
    environment_sha256 = _digest(expected_environment_sha256, "environment")
    legacy_name = f"registry-wheels-{closure_sha256}"
    expected_name = f"{REGISTRY_WHEEL_ENVIRONMENT_PREFIX}{closure_sha256}"
    if isinstance(destination, Path) and destination.name == legacy_name:
        raise ComfyRegistryWheelEnvironmentError(
            "legacy_environment_manifest",
            "Legacy wheel environments must be renewed before use",
        )
    if (
        not isinstance(destination, Path)
        or destination.name != expected_name
        or _is_link_or_reparse(destination)
        or not destination.is_dir()
    ):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment", "Wheel environment path is invalid"
        )
    manifest_path = destination / "environment-manifest.json"
    site_packages = destination / "site-packages"
    children = {path.name for path in destination.iterdir()}
    if children != {manifest_path.name, site_packages.name} or _is_link_or_reparse(site_packages):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment", "Wheel environment layout is invalid"
        )
    try:
        if manifest_path.stat().st_size > MAX_REGISTRY_WHEEL_ENVIRONMENT_MANIFEST_BYTES:
            raise ComfyRegistryWheelEnvironmentError(
                "invalid_environment_manifest", "Wheel environment manifest is too large"
            )
        encoded = manifest_path.read_bytes()
    except OSError as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_manifest", "Wheel environment manifest is unreadable"
        ) from exc
    if not hmac.compare_digest(hashlib.sha256(encoded).hexdigest(), environment_sha256):
        raise ComfyRegistryWheelEnvironmentError(
            "environment_hash_mismatch", "Wheel environment manifest has changed"
        )
    try:
        payload = json.loads(encoded, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_manifest", "Wheel environment manifest is invalid"
        ) from exc
    if (
        isinstance(payload, dict)
        and type(payload.get("version")) is int
        and payload["version"]
        in {
            1,
            2,
        }
    ):
        raise ComfyRegistryWheelEnvironmentError(
            "legacy_environment_manifest",
            "Legacy wheel environments must be renewed before use",
        )
    required_fields = {
        "version",
        "ownership_attestation",
        "closure_sha256",
        "artifact_count",
        "file_count",
        "total_bytes",
        "distributions",
        "runtime_distributions",
        "inventory",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_manifest", "Wheel environment manifest shape is invalid"
        )
    artifact_count = _count(payload["artifact_count"], "artifact")
    if (
        type(payload["version"]) is not int
        or payload["version"] != 3
        or payload["ownership_attestation"] != WHEEL_OWNERSHIP_ATTESTATION
        or payload["closure_sha256"] != closure_sha256
    ):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_manifest", "Wheel environment manifest identity is invalid"
        )
    runtime_value = payload["runtime_distributions"]
    if not isinstance(runtime_value, list):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_manifest",
            "Wheel environment runtime baseline is invalid",
        )
    try:
        runtime_distributions = canonical_comfy_registry_runtime_distributions(
            tuple(
                ComfyRegistryRuntimeDistribution(item["name"], item["version"])
                for item in runtime_value
                if isinstance(item, dict)
                and set(item) == {"name", "version"}
                and isinstance(item["name"], str)
                and isinstance(item["version"], str)
            )
        )
    except (ComfyRegistryRuntimeError, TypeError) as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_manifest",
            "Wheel environment runtime baseline is invalid",
        ) from exc
    if len(runtime_distributions) != len(runtime_value):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_manifest",
            "Wheel environment runtime baseline is invalid",
        )
    inventory, file_count, total_bytes, distributions = _scan_environment(site_packages)
    verified_payload = _environment_payload(
        closure_sha256,
        artifact_count,
        file_count,
        total_bytes,
        distributions,
        inventory,
        runtime_distributions=runtime_distributions,
    )
    canonical = _encode_environment_payload(verified_payload)
    if not hmac.compare_digest(canonical, encoded):
        raise ComfyRegistryWheelEnvironmentError(
            "environment_inventory_mismatch", "Wheel environment contents have changed"
        )
    return ComfyRegistryWheelEnvironmentReport(
        closure_sha256,
        environment_sha256,
        artifact_count,
        file_count,
        total_bytes,
        distributions,
        runtime_distributions,
    )


def _python_executable(value: Path) -> Path:
    if not isinstance(value, Path):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_python_executable", "Managed Python executable is invalid"
        )
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_python_executable", "Managed Python executable is missing"
        ) from exc
    if not resolved.is_file() or _is_link_or_reparse(resolved):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_python_executable", "Managed Python executable is invalid"
        )
    return resolved


def _wheel_inputs(
    artifacts: Sequence[ComfyRegistryWheelArtifact],
    value: Mapping[str, Path],
) -> tuple[tuple[ComfyRegistryWheelArtifact, Path], ...]:
    if not isinstance(value, Mapping):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_wheel_files", "Wheel file mapping is invalid"
        )
    expected = {artifact.filename: artifact for artifact in artifacts}
    if any(not isinstance(key, str) for key in value):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_wheel_files", "Wheel file mapping is invalid"
        )
    provided = set(value)
    if provided != set(expected):
        missing = sorted(set(expected) - provided)
        code = "missing_wheel_file" if missing else "unexpected_wheel_file"
        detail = missing[0] if missing else sorted(provided - set(expected))[0]
        raise ComfyRegistryWheelEnvironmentError(
            code, f"Wheel files do not match closed artifact {detail}"
        )
    resolved: list[tuple[ComfyRegistryWheelArtifact, Path]] = []
    for filename in sorted(expected):
        path = value[filename]
        if not isinstance(path, Path):
            raise ComfyRegistryWheelEnvironmentError(
                "invalid_wheel_file", f"Wheel path for {filename} is invalid"
            )
        try:
            candidate = path.resolve(strict=True)
        except OSError as exc:
            raise ComfyRegistryWheelEnvironmentError(
                "missing_wheel_file", f"Wheel file {filename} is missing"
            ) from exc
        if (
            _is_link_or_reparse(path)
            or candidate.name != filename
            or not candidate.is_file()
            or _is_link_or_reparse(candidate)
        ):
            raise ComfyRegistryWheelEnvironmentError(
                "invalid_wheel_file", f"Wheel file {filename} is invalid"
            )
        artifact = expected[filename]
        size, digest = _file_identity(candidate)
        if size != artifact.size_bytes:
            raise ComfyRegistryWheelEnvironmentError(
                "wheel_size_mismatch", f"Wheel file {filename} has changed size"
            )
        if digest != artifact.sha256:
            raise ComfyRegistryWheelEnvironmentError(
                "wheel_hash_mismatch", f"Wheel file {filename} has changed contents"
            )
        resolved.append((artifact, candidate))
    return tuple(resolved)


def _stage_wheels(
    wheels: Sequence[tuple[ComfyRegistryWheelArtifact, Path]],
    directory: Path,
) -> tuple[tuple[Path, ...], _WheelOwnershipPlan]:
    directory.mkdir()
    staged: list[Path] = []
    ownership: list[_WheelOwnership] = []
    installed_paths: set[str] = set()
    installed_parent_paths: set[str] = set()
    expanded_files = 0
    expanded_bytes = 0
    for artifact, source in wheels:
        destination = directory / artifact.filename
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as reader, destination.open("xb") as writer:
                while chunk := reader.read(WHEEL_HASH_CHUNK_BYTES):
                    size += len(chunk)
                    digest.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
        except OSError as exc:
            raise ComfyRegistryWheelEnvironmentError(
                "wheel_stage_failed", f"Wheel file {artifact.filename} could not be staged"
            ) from exc
        if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
            raise ComfyRegistryWheelEnvironmentError(
                "wheel_changed_during_staging",
                f"Wheel file {artifact.filename} changed while it was staged",
            )
        archive_files, archive_bytes, wheel_ownership = _inspect_wheel_archive(
            destination,
            installed_paths,
            installed_parent_paths=installed_parent_paths,
            remaining_files=MAX_REGISTRY_WHEEL_ENVIRONMENT_FILES - expanded_files,
            remaining_bytes=MAX_REGISTRY_WHEEL_ENVIRONMENT_BYTES - expanded_bytes,
        )
        expanded_files += archive_files
        expanded_bytes += archive_bytes
        if (
            expanded_files > MAX_REGISTRY_WHEEL_ENVIRONMENT_FILES
            or expanded_bytes > MAX_REGISTRY_WHEEL_ENVIRONMENT_BYTES
        ):
            raise ComfyRegistryWheelEnvironmentError(
                "wheel_archive_too_large", "Closed wheel archives exceed extraction limits"
            )
        staged.append(destination)
        ownership.append(wheel_ownership)
    return tuple(staged), _WheelOwnershipPlan(tuple(ownership))


def _inspect_wheel_archive(
    path: Path,
    installed_paths: set[str],
    *,
    installed_parent_paths: set[str] | None = None,
    remaining_files: int = MAX_REGISTRY_WHEEL_ENVIRONMENT_FILES,
    remaining_bytes: int = MAX_REGISTRY_WHEEL_ENVIRONMENT_BYTES,
) -> tuple[int, int, _WheelOwnership]:
    file_count = 0
    expanded_bytes = 0
    if installed_parent_paths is None:
        installed_parent_paths = {
            parent for existing in installed_paths for parent in _parent_path_keys(existing)
        }
    try:
        with zipfile.ZipFile(path) as archive:
            entries: dict[str, zipfile.ZipInfo] = {}
            archive_keys: set[str] = set()
            archive_files: set[str] = set()
            archive_directories: set[str] = set()
            for entry in archive.infolist():
                relative = _archive_path(entry.filename)
                archive_key = _wheel_path_key(relative)
                parents = _parent_path_keys(archive_key)
                mode = (entry.external_attr >> 16) & 0xFFFF
                if entry.flag_bits & 1 or stat.S_ISLNK(mode):
                    raise ComfyRegistryWheelEnvironmentError(
                        "unsafe_wheel_archive", "Wheel archive contains an unsafe entry"
                    )
                if archive_key in archive_keys:
                    raise ComfyRegistryWheelEnvironmentError(
                        "invalid_wheel_record",
                        "Wheel archive contains aliased or duplicate file paths",
                    )
                archive_keys.add(archive_key)
                if any(parent in archive_files for parent in parents) or (
                    not entry.is_dir() and archive_key in archive_directories
                ):
                    raise ComfyRegistryWheelEnvironmentError(
                        "invalid_wheel_record",
                        "Wheel archive contains a file-directory path collision",
                    )
                archive_directories.update(parents)
                if entry.is_dir():
                    if entry.file_size or entry.compress_size or entry.CRC:
                        raise ComfyRegistryWheelEnvironmentError(
                            "unsafe_wheel_archive",
                            "Wheel archive directory entries must contain no payload",
                        )
                    archive_directories.add(archive_key)
                    continue
                archive_files.add(archive_key)
                entries[relative] = entry
                installed = _installed_wheel_path(relative)
                _claim_installed_path(
                    installed,
                    installed_paths,
                    installed_parent_paths,
                )
                file_count += 1
                expanded_bytes += entry.file_size
                if file_count > remaining_files or expanded_bytes > remaining_bytes:
                    raise ComfyRegistryWheelEnvironmentError(
                        "wheel_archive_too_large",
                        "Closed wheel archives exceed extraction limits",
                    )
                compressed = max(entry.compress_size, 1)
                if (
                    entry.file_size > MAX_REGISTRY_WHEEL_UNCHECKED_ENTRY_BYTES
                    and entry.file_size > compressed * MAX_REGISTRY_WHEEL_EXPANSION_RATIO
                ):
                    raise ComfyRegistryWheelEnvironmentError(
                        "unsafe_wheel_archive",
                        f"{path.name} expands {entry.filename} from "
                        f"{compressed} bytes to {entry.file_size}, which is an unsafe ratio",
                    )
            record_paths = [
                relative
                for relative in entries
                if len(PurePosixPath(relative).parts) == 2
                and PurePosixPath(relative).parts[0].endswith(".dist-info")
                and PurePosixPath(relative).name == "RECORD"
            ]
            if len(record_paths) != 1:
                raise ComfyRegistryWheelEnvironmentError(
                    "invalid_wheel_record",
                    "Wheel archive must contain exactly one distribution RECORD",
                )
            record_path = record_paths[0]
            record_entry = entries[record_path]
            if record_entry.file_size > MAX_REGISTRY_WHEEL_METADATA_FILE_BYTES:
                raise ComfyRegistryWheelEnvironmentError(
                    "invalid_wheel_record", "Wheel distribution RECORD is too large"
                )
            record_bytes = _read_wheel_entry(archive, record_entry)
            recorded = _parse_source_record(record_bytes)
            if set(recorded) != set(entries):
                raise ComfyRegistryWheelEnvironmentError(
                    "invalid_wheel_record",
                    "Wheel distribution RECORD does not cover the archive exactly",
                )
            planned: list[_PlannedInstalledFile] = []
            for relative, entry in entries.items():
                size, digest = _wheel_entry_identity(archive, entry)
                recorded_hash, recorded_size = recorded[relative]
                if relative == record_path:
                    if recorded_hash or recorded_size:
                        raise ComfyRegistryWheelEnvironmentError(
                            "invalid_wheel_record",
                            "Wheel distribution RECORD must leave its own identity blank",
                        )
                elif recorded_hash != digest or recorded_size != size:
                    raise ComfyRegistryWheelEnvironmentError(
                        "invalid_wheel_record",
                        "Wheel distribution RECORD identity does not match the archive",
                    )
                planned.append(
                    _PlannedInstalledFile(
                        _installed_wheel_path(relative),
                        relative,
                        size,
                        digest,
                    )
                )
            dist_info = PurePosixPath(record_path).parent.as_posix()
            for name in _GENERATED_DISTRIBUTION_FILES:
                try:
                    _claim_installed_path(
                        f"{dist_info}/{name}",
                        installed_paths,
                        installed_parent_paths,
                    )
                except ComfyRegistryWheelEnvironmentError as exc:
                    raise ComfyRegistryWheelEnvironmentError(
                        "invalid_wheel_record",
                        "Wheel archive contains pip-generated distribution metadata",
                    ) from exc
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_wheel_archive", f"Wheel file {path.name} is not a valid archive"
        ) from exc
    return (
        file_count,
        expanded_bytes,
        _WheelOwnership(
            _installed_wheel_path(record_path),
            tuple(sorted(planned, key=lambda item: item.path)),
        ),
    )


def _installed_wheel_path(relative: str) -> str:
    parts = PurePosixPath(relative).parts
    if not parts[0].endswith(".data"):
        return relative
    if len(parts) < 3 or parts[1] not in {"purelib", "platlib"}:
        raise ComfyRegistryWheelEnvironmentError(
            "unsupported_wheel_scheme",
            "Wheel archive uses a target scheme whose installed path is not stable",
        )
    return PurePosixPath(*parts[2:]).as_posix()


def _read_wheel_entry(archive: zipfile.ZipFile, entry: zipfile.ZipInfo) -> bytes:
    try:
        with archive.open(entry) as reader:
            value = reader.read(MAX_REGISTRY_WHEEL_METADATA_FILE_BYTES + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_wheel_record", "Wheel distribution RECORD is unreadable"
        ) from exc
    if len(value) > MAX_REGISTRY_WHEEL_METADATA_FILE_BYTES:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_wheel_record", "Wheel distribution RECORD is too large"
        )
    return value


def _parse_source_record(value: bytes) -> dict[str, tuple[str, int | str]]:
    try:
        text = value.decode("utf-8")
        rows = csv.reader(io.StringIO(text, newline=""))
        result: dict[str, tuple[str, int | str]] = {}
        for row in rows:
            if len(result) >= MAX_REGISTRY_WHEEL_ENVIRONMENT_FILES:
                raise ValueError("too many RECORD rows")
            if len(row) != 3:
                raise ValueError("invalid RECORD row")
            relative = _archive_path(row[0])
            if relative in result:
                raise ValueError("duplicate RECORD row")
            if not row[1] and not row[2]:
                identity: tuple[str, int | str] = ("", "")
            else:
                if not row[2].isascii() or not row[2].isdigit():
                    raise ValueError("invalid RECORD size")
                size = int(row[2])
                if size < 0:
                    raise ValueError("invalid RECORD size")
                identity = (_record_sha256(row[1]), size)
            result[relative] = identity
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_wheel_record", "Wheel distribution RECORD is invalid"
        ) from exc
    return result


def _record_sha256(value: str) -> str:
    prefix = "sha256="
    encoded = value[len(prefix) :] if value.startswith(prefix) else ""
    if (
        not encoded
        or "=" in encoded
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in encoded
        )
    ):
        raise ValueError("invalid RECORD digest")
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid RECORD digest") from exc
    if len(decoded) != hashlib.sha256().digest_size:
        raise ValueError("invalid RECORD digest")
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if not hmac.compare_digest(canonical, encoded):
        raise ValueError("invalid RECORD digest")
    return decoded.hex()


def _wheel_entry_identity(
    archive: zipfile.ZipFile,
    entry: zipfile.ZipInfo,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with archive.open(entry) as reader:
            while chunk := reader.read(WHEEL_HASH_CHUNK_BYTES):
                size += len(chunk)
                digest.update(chunk)
                if size > entry.file_size:
                    raise ComfyRegistryWheelEnvironmentError(
                        "invalid_wheel_record",
                        "Wheel archive entry exceeds its declared size",
                    )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_wheel_record", "Wheel archive entry is unreadable"
        ) from exc
    if size != entry.file_size:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_wheel_record", "Wheel archive entry size is inconsistent"
        )
    return size, digest.hexdigest()


def _archive_path(value: str) -> str:
    if (
        not value
        or len(value) > MAX_REGISTRY_WHEEL_ARCHIVE_PATH_CHARACTERS
        or chr(92) in value
        or value.startswith("/")
    ):
        raise ComfyRegistryWheelEnvironmentError(
            "unsafe_wheel_archive", "Wheel archive contains an unsafe path"
        )
    normalized = value[:-1] if value.endswith("/") else value
    parts = normalized.split("/")
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        raise ComfyRegistryWheelEnvironmentError(
            "unsafe_wheel_archive", "Wheel archive contains an unsafe path"
        )
    for part in parts:
        if (
            len(part) > MAX_REGISTRY_WHEEL_ARCHIVE_COMPONENT_CHARACTERS
            or part.rstrip(" .") != part
            or ":" in part
            or any(ord(character) < 32 or ord(character) == 127 for character in part)
            or part.split(".", 1)[0].upper() in _RESERVED_WINDOWS_NAMES
        ):
            raise ComfyRegistryWheelEnvironmentError(
                "unsafe_wheel_archive", "Wheel archive contains an unsafe path"
            )
    path = PurePosixPath(*parts)
    return path.as_posix()


def _wheel_path_key(value: str) -> str:
    return "/".join(
        unicodedata.normalize("NFC", part).casefold() for part in PurePosixPath(value).parts
    )


def _wheel_path_sort_key(value: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parts = PurePosixPath(value).parts
    return (
        tuple(unicodedata.normalize("NFC", part).casefold() for part in parts),
        parts,
    )


def _parent_path_keys(value: str) -> tuple[str, ...]:
    parts = PurePosixPath(value).parts
    return tuple(PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts)))


def _claim_installed_path(
    value: str,
    installed_paths: set[str],
    installed_parent_paths: set[str],
) -> None:
    key = _wheel_path_key(value)
    parents = _parent_path_keys(key)
    if (
        key in installed_paths
        or key in installed_parent_paths
        or any(parent in installed_paths for parent in parents)
    ):
        raise ComfyRegistryWheelEnvironmentError(
            "overlapping_wheel_archives",
            "Closed wheel archives contain overlapping installed paths",
        )
    installed_paths.add(key)
    installed_parent_paths.update(parents)


def _remove_staged_wheels(directory: Path) -> None:
    try:
        shutil.rmtree(directory)
    except OSError as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "wheel_stage_cleanup_failed", "Staged wheel files could not be removed"
        ) from exc


def _destination(
    closure: ComfyRegistryWheelClosure,
    destination: Path,
) -> tuple[Path, Path]:
    if not isinstance(destination, Path):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_destination", "Wheel environment destination is invalid"
        )
    expected_name = f"{REGISTRY_WHEEL_ENVIRONMENT_PREFIX}{closure.closure_sha256}"
    if destination.name != expected_name or destination.exists() or destination.is_symlink():
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_destination",
            "Wheel environment destination is not the closure-addressed path",
        )
    parent = destination.parent
    if not parent.is_dir() or _is_link_or_reparse(parent):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_destination", "Wheel environment parent is invalid"
        )
    return parent, parent / f".{expected_name}.lock"


def _acquire_lock(lock: Path) -> int:
    try:
        return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "environment_locked", "Wheel environment assembly is already running"
        ) from exc
    except OSError as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "environment_lock_failed", "Wheel environment lock could not be created"
        ) from exc


async def _run_pip(
    python_executable: Path,
    wheel_paths: tuple[Path, ...],
    target: Path,
) -> None:
    command = [
        str(python_executable),
        "-I",
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-index",
        "--no-deps",
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--no-compile",
        "--target",
        str(target),
        *(str(path) for path in wheel_paths),
    ]
    environment = subprocess_environment(
        overrides={
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target.parent,
        env=environment,
        creationflags=WINDOWS_CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        _, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=WHEEL_INSTALL_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        await _terminate(process)
        raise
    except TimeoutError as exc:
        await _terminate(process)
        raise ComfyRegistryWheelEnvironmentError(
            "wheel_install_timeout", "Offline wheel assembly timed out"
        ) from exc
    if process.returncode:
        raise ComfyRegistryWheelEnvironmentError(
            "wheel_install_failed",
            "The closed wheel set could not be assembled offline",
        ) from RuntimeError(stderr.decode("utf-8", errors="replace")[-1_000:])


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    await process.wait()


def _audit_environment(
    closure: ComfyRegistryWheelClosure,
    artifacts: Sequence[ComfyRegistryWheelArtifact],
    site_packages: Path,
    ownership_plan: _WheelOwnershipPlan,
) -> tuple[ComfyRegistryWheelEnvironmentReport, bytes]:
    expected = {(artifact.name, artifact.version) for artifact in artifacts}
    inventory, file_count, total_bytes, resolved = _scan_environment(site_packages)
    actual = {(item.name, item.version) for item in resolved}
    if actual != expected or len(resolved) != len(expected):
        raise ComfyRegistryWheelEnvironmentError(
            "distribution_set_mismatch",
            "Wheel environment distributions do not match the closed artifacts",
        )
    _verify_installed_ownership(site_packages, ownership_plan, inventory)
    payload = _environment_payload(
        closure.closure_sha256,
        len(artifacts),
        file_count,
        total_bytes,
        resolved,
        inventory,
        runtime_distributions=closure.runtime_distributions,
    )
    encoded = _encode_environment_payload(payload)
    environment_sha256 = hashlib.sha256(encoded).hexdigest()
    return (
        ComfyRegistryWheelEnvironmentReport(
            closure.closure_sha256,
            environment_sha256,
            len(artifacts),
            file_count,
            total_bytes,
            resolved,
            closure.runtime_distributions,
        ),
        encoded,
    )


def _verify_installed_ownership(
    site_packages: Path,
    plan: _WheelOwnershipPlan,
    inventory: Sequence[dict[str, object]],
) -> None:
    final_files: dict[str, tuple[int, str]] = {}
    for item in inventory:
        if item.get("kind") != "file":
            continue
        path = item.get("path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            type(path) is not str
            or type(size) is not int
            or type(digest) is not str
            or path in final_files
        ):
            raise ComfyRegistryWheelEnvironmentError(
                "wheel_ownership_mismatch",
                "Installed wheel ownership evidence is invalid",
            )
        final_files[path] = (size, digest)

    expected_global: set[str] = set()
    for wheel in plan.wheels:
        source = {item.path: item for item in wheel.files}
        if len(source) != len(wheel.files) or wheel.record_path not in source:
            raise ComfyRegistryWheelEnvironmentError(
                "wheel_ownership_mismatch",
                "Wheel ownership plan is inconsistent",
            )
        dist_info = PurePosixPath(wheel.record_path).parent.as_posix()
        generated = {f"{dist_info}/{name}" for name in _GENERATED_DISTRIBUTION_FILES}
        expected = set(source) | generated
        if expected_global & expected:
            raise ComfyRegistryWheelEnvironmentError(
                "wheel_ownership_mismatch",
                "Installed wheel ownership overlaps another distribution",
            )
        expected_global.update(expected)

        record_file = site_packages.joinpath(*PurePosixPath(wheel.record_path).parts)
        try:
            if (
                _is_link_or_reparse(record_file)
                or not record_file.is_file()
                or record_file.stat().st_size > MAX_REGISTRY_WHEEL_METADATA_FILE_BYTES
            ):
                raise OSError("invalid RECORD")
            recorded = _parse_source_record(record_file.read_bytes())
        except (OSError, ComfyRegistryWheelEnvironmentError) as exc:
            raise ComfyRegistryWheelEnvironmentError(
                "wheel_ownership_mismatch",
                "Installed wheel RECORD is invalid",
            ) from exc
        if set(recorded) != expected:
            raise ComfyRegistryWheelEnvironmentError(
                "wheel_ownership_mismatch",
                "Installed wheel RECORD does not cover its owned files exactly",
            )
        for relative in expected:
            identity = final_files.get(relative)
            if identity is None:
                raise ComfyRegistryWheelEnvironmentError(
                    "wheel_ownership_mismatch",
                    "Installed wheel file is missing",
                )
            recorded_hash, recorded_size = recorded[relative]
            if relative == wheel.record_path:
                if recorded_hash or recorded_size:
                    raise ComfyRegistryWheelEnvironmentError(
                        "wheel_ownership_mismatch",
                        "Installed wheel RECORD must leave its own identity blank",
                    )
                continue
            if (recorded_size, recorded_hash) != identity:
                raise ComfyRegistryWheelEnvironmentError(
                    "wheel_ownership_mismatch",
                    "Installed wheel RECORD identity does not match its file",
                )
            planned = source.get(relative)
            if planned is not None and (
                planned.size_bytes != identity[0] or planned.sha256 != identity[1]
            ):
                raise ComfyRegistryWheelEnvironmentError(
                    "wheel_ownership_mismatch",
                    "Pip changed a source-owned wheel file",
                )
    if set(final_files) != expected_global:
        raise ComfyRegistryWheelEnvironmentError(
            "wheel_ownership_mismatch",
            "Installed wheel file set differs from the source ownership plan",
        )


def _scan_environment(
    site_packages: Path,
) -> tuple[
    list[dict[str, object]],
    int,
    int,
    tuple[ComfyRegistryWheelEnvironmentDistribution, ...],
]:
    distributions: list[ComfyRegistryWheelEnvironmentDistribution] = []
    inventory: list[dict[str, object]] = []
    file_count = 0
    total_bytes = 0
    paths = list(site_packages.rglob("*"))
    paths.sort(key=lambda item: _wheel_path_sort_key(item.relative_to(site_packages).as_posix()))
    for path in paths:
        if _is_link_or_reparse(path):
            raise ComfyRegistryWheelEnvironmentError(
                "unsafe_environment_link", "Wheel environment contains a link"
            )
        relative = path.relative_to(site_packages).as_posix()
        if path.is_dir():
            inventory.append({"path": relative, "kind": "directory"})
            if len(inventory) > MAX_REGISTRY_WHEEL_ENVIRONMENT_FILES:
                raise ComfyRegistryWheelEnvironmentError(
                    "environment_too_large", "Wheel environment exceeds its audited limits"
                )
            continue
        if not path.is_file():
            continue
        # Registry overlays are inserted with sys.path directly, never through
        # site.addsitedir, so .pth files stay inert. Preserve and hash them like
        # every other wheel byte; activation still has to prove the required
        # node inventory before the package can become usable.
        size, digest = _file_identity(path)
        file_count += 1
        total_bytes += size
        inventory.append({"path": relative, "kind": "file", "size_bytes": size, "sha256": digest})
        if (
            len(inventory) > MAX_REGISTRY_WHEEL_ENVIRONMENT_FILES
            or total_bytes > MAX_REGISTRY_WHEEL_ENVIRONMENT_BYTES
        ):
            raise ComfyRegistryWheelEnvironmentError(
                "environment_too_large", "Wheel environment exceeds its audited limits"
            )
    directories = list(site_packages.glob("*.dist-info"))
    directories.sort(key=lambda item: _wheel_path_sort_key(item.name))
    for directory in directories:
        if not directory.is_dir() or _is_link_or_reparse(directory):
            raise ComfyRegistryWheelEnvironmentError(
                "invalid_distribution_metadata", "Wheel distribution metadata is invalid"
            )
        metadata_path = directory / "METADATA"
        if (
            not metadata_path.is_file()
            or metadata_path.stat().st_size > MAX_REGISTRY_WHEEL_METADATA_FILE_BYTES
        ):
            raise ComfyRegistryWheelEnvironmentError(
                "invalid_distribution_metadata", "Wheel distribution METADATA is invalid"
            )
        message = BytesParser(policy=policy.default).parsebytes(
            metadata_path.read_bytes(),
            headersonly=True,
        )
        names = message.get_all("Name")
        versions = message.get_all("Version")
        if (
            message.defects
            or names is None
            or versions is None
            or len(names) != 1
            or len(versions) != 1
        ):
            raise ComfyRegistryWheelEnvironmentError(
                "invalid_distribution_metadata", "Wheel distribution identity is invalid"
            )
        name = canonicalize_name(str(names[0]))
        try:
            version = str(Version(str(versions[0])))
        except InvalidVersion as exc:
            raise ComfyRegistryWheelEnvironmentError(
                "invalid_distribution_metadata", "Wheel distribution version is invalid"
            ) from exc
        distributions.append(
            ComfyRegistryWheelEnvironmentDistribution(name, version, directory.name)
        )
    resolved = tuple(sorted(distributions, key=lambda item: (item.name, item.version)))
    if len(resolved) != len({item.name for item in resolved}):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_distribution_metadata", "Wheel distributions contain duplicate names"
        )
    return inventory, file_count, total_bytes, resolved


def _environment_payload(
    closure_sha256: str,
    artifact_count: int,
    file_count: int,
    total_bytes: int,
    distributions: Sequence[ComfyRegistryWheelEnvironmentDistribution],
    inventory: Sequence[dict[str, object]],
    *,
    runtime_distributions: Sequence[ComfyRegistryRuntimeDistribution] = (),
) -> dict[str, object]:
    runtime = canonical_comfy_registry_runtime_distributions(runtime_distributions)
    payload: dict[str, object] = {
        "version": 3,
        "ownership_attestation": WHEEL_OWNERSHIP_ATTESTATION,
        "closure_sha256": closure_sha256,
        "artifact_count": artifact_count,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "distributions": [
            {"name": item.name, "version": item.version, "dist_info": item.dist_info}
            for item in distributions
        ],
        "runtime_distributions": comfy_registry_runtime_distribution_payload(runtime),
        "inventory": list(inventory),
    }
    return payload


def _encode_environment_payload(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate environment manifest field")
        result[key] = value
    return result


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARACTERS for character in value)
    ):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_identity", f"Wheel environment {label} hash is invalid"
        )
    return value.lower()


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_manifest", f"Wheel environment {label} count is invalid"
        )
    return value


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(WHEEL_HASH_CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _is_link_or_reparse(path: Path) -> bool:
    return is_link_or_reparse(
        path,
        missing="assume_link",
        unreadable="assume_link",
    )


def _write_new(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "environment_manifest_failed",
            "Wheel environment manifest could not be written",
        ) from exc
