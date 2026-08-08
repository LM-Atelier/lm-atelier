from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import shutil
import stat
import tempfile
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
        staged_wheels = await asyncio.to_thread(_stage_wheels, wheels, wheel_staging)
        if staged_wheels:
            await _run_pip(executable, staged_wheels, site_packages)
        await asyncio.to_thread(_remove_staged_wheels, wheel_staging)
        report, encoded = await asyncio.to_thread(
            _audit_environment,
            closure,
            artifacts,
            site_packages,
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
    expected_name = f"registry-wheels-{closure_sha256}"
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
    if not isinstance(payload, dict) or frozenset(payload) not in {
        frozenset(
            {
                "version",
                "closure_sha256",
                "artifact_count",
                "file_count",
                "total_bytes",
                "distributions",
                "inventory",
            }
        ),
        frozenset(
            {
                "version",
                "closure_sha256",
                "artifact_count",
                "file_count",
                "total_bytes",
                "distributions",
                "runtime_distributions",
                "inventory",
            }
        ),
    }:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_manifest", "Wheel environment manifest shape is invalid"
        )
    artifact_count = _count(payload["artifact_count"], "artifact")
    version = payload["version"]
    if (
        version not in {1, 2}
        or (version == 1) != ("runtime_distributions" not in payload)
        or payload["closure_sha256"] != closure_sha256
    ):
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_environment_manifest", "Wheel environment manifest identity is invalid"
        )
    runtime_value = payload.get("runtime_distributions", [])
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
) -> tuple[Path, ...]:
    directory.mkdir()
    staged: list[Path] = []
    installed_paths: set[str] = set()
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
        archive_files, archive_bytes = _inspect_wheel_archive(destination, installed_paths)
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
    return tuple(staged)


def _inspect_wheel_archive(path: Path, installed_paths: set[str]) -> tuple[int, int]:
    file_count = 0
    expanded_bytes = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                relative = _archive_path(entry.filename)
                mode = (entry.external_attr >> 16) & 0xFFFF
                if entry.flag_bits & 1 or stat.S_ISLNK(mode):
                    raise ComfyRegistryWheelEnvironmentError(
                        "unsafe_wheel_archive", "Wheel archive contains an unsafe entry"
                    )
                if entry.is_dir():
                    continue
                if relative in installed_paths:
                    raise ComfyRegistryWheelEnvironmentError(
                        "overlapping_wheel_archives",
                        "Closed wheel archives contain overlapping files",
                    )
                installed_paths.add(relative)
                file_count += 1
                expanded_bytes += entry.file_size
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
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ComfyRegistryWheelEnvironmentError(
            "invalid_wheel_archive", f"Wheel file {path.name} is not a valid archive"
        ) from exc
    return file_count, expanded_bytes


def _archive_path(value: str) -> str:
    if not value or len(value) > MAX_REGISTRY_WHEEL_ARCHIVE_PATH_CHARACTERS or chr(92) in value:
        raise ComfyRegistryWheelEnvironmentError(
            "unsafe_wheel_archive", "Wheel archive contains an unsafe path"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} or ":" in part for part in path.parts):
        raise ComfyRegistryWheelEnvironmentError(
            "unsafe_wheel_archive", "Wheel archive contains an unsafe path"
        )
    return path.as_posix()


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
    expected_name = f"registry-wheels-{closure.closure_sha256}"
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
) -> tuple[ComfyRegistryWheelEnvironmentReport, bytes]:
    expected = {(artifact.name, artifact.version) for artifact in artifacts}
    inventory, file_count, total_bytes, resolved = _scan_environment(site_packages)
    actual = {(item.name, item.version) for item in resolved}
    if actual != expected or len(resolved) != len(expected):
        raise ComfyRegistryWheelEnvironmentError(
            "distribution_set_mismatch",
            "Wheel environment distributions do not match the closed artifacts",
        )
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
    for path in sorted(site_packages.rglob("*")):
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
    for directory in sorted(site_packages.glob("*.dist-info")):
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
        "version": 2 if runtime else 1,
        "closure_sha256": closure_sha256,
        "artifact_count": artifact_count,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "distributions": [
            {"name": item.name, "version": item.version, "dist_info": item.dist_info}
            for item in distributions
        ],
        "inventory": list(inventory),
    }
    if runtime:
        payload["runtime_distributions"] = comfy_registry_runtime_distribution_payload(runtime)
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
