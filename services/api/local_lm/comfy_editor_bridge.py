from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from .filesystem_links import is_link_or_reparse

WORKFLOW_EDITOR_BRIDGE_PROTOCOL_VERSION = 2
SUPPORTED_COMFYUI_VERSION = "0.28.0"
SUPPORTED_COMFYUI_FRONTEND_VERSION = "1.45.21"
MAX_BRIDGE_ASSET_BYTES = 512 * 1024
MAX_DISTRIBUTION_METADATA_BYTES = 64 * 1024
MAX_SITE_PACKAGES_ENTRIES = 4_096

_BRIDGE_ASSET_DIRECTORY = Path(__file__).with_name("comfy_editor_bridge_assets")
_BRIDGE_ASSET_HASHES = {
    "__init__.py": "a236d9d2e96f0857dc38192fef58927b9592886b7d2c7c9e05a941b253887df8",
    "js/lm_atelier_workflow_editor.js": (
        "7df5bd55f11c89bd15eeb8b85f01809f51c04096f611d49ad9cb52d13c8b6fd1"
    ),
}
BRIDGE_COORDINATOR_CONFIG = "js/lm_atelier_workflow_editor_config.js"
_BRIDGE_MANIFEST_SHA256 = hashlib.sha256(
    "".join(
        f"{name}{chr(0)}{digest}{chr(10)}" for name, digest in sorted(_BRIDGE_ASSET_HASHES.items())
    ).encode("utf-8")
).hexdigest()
BRIDGE_DIRECTORY_PREFIX = (
    f"_lm_atelier_workflow_editor_bridge_v{WORKFLOW_EDITOR_BRIDGE_PROTOCOL_VERSION}_"
    f"{_BRIDGE_MANIFEST_SHA256[:12]}"
)


class ComfyEditorBridgeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ComfyEditorBridgeSupport:
    supported: bool
    code: str
    message: str
    comfyui_version: str | None = None
    frontend_version: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedComfyEditorBridge:
    support: ComfyEditorBridgeSupport
    folder: Path | None


def inspect_comfy_editor_bridge_support(
    *,
    comfy_executable: Path | None,
    comfy_directory: Path | None,
) -> ComfyEditorBridgeSupport:
    if comfy_executable is None or comfy_directory is None:
        return ComfyEditorBridgeSupport(
            False,
            "workflow-editor-runtime-unavailable",
            "Install the managed media runtime before opening the workflow editor.",
        )
    try:
        executable = comfy_executable.expanduser().resolve(strict=True)
        directory = comfy_directory.expanduser().resolve(strict=True)
        if not executable.is_file() or not directory.is_dir():
            raise ComfyEditorBridgeError(
                "workflow-editor-runtime-unavailable",
                "Install the managed media runtime before opening the workflow editor.",
            )
        comfyui_version = _read_comfyui_version(directory)
        frontend_version = _read_frontend_version(executable, directory)
    except ComfyEditorBridgeError as exc:
        return ComfyEditorBridgeSupport(False, exc.code, str(exc))
    except OSError:
        return ComfyEditorBridgeSupport(
            False,
            "workflow-editor-runtime-unavailable",
            "The media runtime could not be inspected for workflow editing.",
        )
    if comfyui_version != SUPPORTED_COMFYUI_VERSION:
        return ComfyEditorBridgeSupport(
            False,
            "workflow-editor-comfyui-unsupported",
            (
                "Native workflow editing requires ComfyUI "
                f"{SUPPORTED_COMFYUI_VERSION}; the configured runtime uses {comfyui_version}."
            ),
            comfyui_version,
            frontend_version,
        )
    if frontend_version != SUPPORTED_COMFYUI_FRONTEND_VERSION:
        return ComfyEditorBridgeSupport(
            False,
            "workflow-editor-frontend-unsupported",
            (
                "Native workflow editing requires the ComfyUI frontend "
                f"{SUPPORTED_COMFYUI_FRONTEND_VERSION}; the configured runtime uses "
                f"{frontend_version}."
            ),
            comfyui_version,
            frontend_version,
        )
    return ComfyEditorBridgeSupport(
        True,
        "ready",
        "Native workflow editing is available.",
        comfyui_version,
        frontend_version,
    )


def prepare_comfy_editor_bridge(
    *,
    comfy_executable: Path | None,
    comfy_directory: Path | None,
    custom_node_root: Path,
    coordinator_origins: Iterable[str],
) -> PreparedComfyEditorBridge:
    support = inspect_comfy_editor_bridge_support(
        comfy_executable=comfy_executable,
        comfy_directory=comfy_directory,
    )
    if not support.supported:
        return PreparedComfyEditorBridge(support, None)
    return PreparedComfyEditorBridge(
        support,
        stage_comfy_editor_bridge(
            custom_node_root,
            coordinator_origins=coordinator_origins,
        ),
    )


def bridge_directory_name(coordinator_origins: Iterable[str]) -> str:
    config = _coordinator_config(coordinator_origins)
    digest = hashlib.sha256(config).hexdigest()
    return f"{BRIDGE_DIRECTORY_PREFIX}_{digest[:12]}"


def stage_comfy_editor_bridge(
    custom_node_root: Path,
    *,
    coordinator_origins: Iterable[str],
) -> Path:
    source_files = _verified_asset_files(_BRIDGE_ASSET_DIRECTORY)
    origins = tuple(coordinator_origins)
    coordinator_config = _coordinator_config(origins)
    directory_name = bridge_directory_name(origins)
    staged_hashes = {
        **_BRIDGE_ASSET_HASHES,
        BRIDGE_COORDINATOR_CONFIG: hashlib.sha256(coordinator_config).hexdigest(),
    }
    custom_node_root.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(custom_node_root):
        raise ComfyEditorBridgeError(
            "workflow-editor-bridge-root-invalid",
            "The workflow editor bridge directory is not a regular folder.",
        )
    root = custom_node_root.resolve(strict=True)
    destination = root / directory_name
    if os.path.lexists(destination):
        _verified_asset_files(destination, expected_hashes=staged_hashes)
        return destination

    staging = Path(tempfile.mkdtemp(prefix=f".{directory_name}-", dir=root))
    try:
        for relative, content in {
            **source_files,
            BRIDGE_COORDINATOR_CONFIG: coordinator_config,
        }.items():
            target = staging.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(content)
        _verified_asset_files(staging, expected_hashes=staged_hashes)
        try:
            os.rename(staging, destination)
        except FileExistsError:
            _verified_asset_files(destination, expected_hashes=staged_hashes)
        except OSError:
            if not os.path.lexists(destination):
                raise
            _verified_asset_files(destination, expected_hashes=staged_hashes)
        return destination
    except ComfyEditorBridgeError:
        raise
    except OSError as exc:
        raise ComfyEditorBridgeError(
            "workflow-editor-bridge-staging-failed",
            "The workflow editor bridge could not be staged.",
        ) from exc
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging, ignore_errors=True)


def _read_comfyui_version(directory: Path) -> str:
    path = directory / "comfyui_version.py"
    content = _read_bounded(path, MAX_DISTRIBUTION_METADATA_BYTES)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ComfyEditorBridgeError(
            "workflow-editor-comfyui-version-invalid",
            "The configured ComfyUI version could not be verified.",
        ) from exc
    matches: list[str] = []
    for line in text.splitlines():
        name, separator, raw_value = line.partition("=")
        value = raw_value.strip()
        if (
            separator
            and name.strip() == "__version__"
            and len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            matches.append(value[1:-1])
    if len(matches) != 1 or not matches[0] or len(matches[0]) > 64:
        raise ComfyEditorBridgeError(
            "workflow-editor-comfyui-version-invalid",
            "The configured ComfyUI version could not be verified.",
        )
    return matches[0]


def _read_frontend_version(executable: Path, directory: Path) -> str:
    for root in _site_packages_roots(executable, directory):
        if not root.is_dir() or _is_link_or_reparse(root):
            continue
        candidates: list[Path] = []
        for entries, entry in enumerate(root.iterdir(), start=1):
            if entries > MAX_SITE_PACKAGES_ENTRIES:
                raise ComfyEditorBridgeError(
                    "workflow-editor-frontend-ambiguous",
                    "The ComfyUI frontend installation could not be identified safely.",
                )
            normalized = entry.name.casefold().replace("-", "_").replace(".", "_")
            if normalized.startswith("comfyui_frontend_package_") and normalized.endswith(
                "_dist_info"
            ):
                candidates.append(entry)
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ComfyEditorBridgeError(
                "workflow-editor-frontend-ambiguous",
                "The ComfyUI frontend installation contains conflicting versions.",
            )
        candidate = candidates[0]
        if not candidate.is_dir() or _is_link_or_reparse(candidate):
            raise ComfyEditorBridgeError(
                "workflow-editor-frontend-invalid",
                "The ComfyUI frontend installation could not be verified.",
            )
        metadata = BytesParser(policy=compat32).parsebytes(
            _read_bounded(candidate / "METADATA", MAX_DISTRIBUTION_METADATA_BYTES)
        )
        names = metadata.get_all("Name", [])
        versions = metadata.get_all("Version", [])
        if len(names) != 1 or len(versions) != 1:
            raise ComfyEditorBridgeError(
                "workflow-editor-frontend-invalid",
                "The ComfyUI frontend installation could not be verified.",
            )
        name = str(names[0]).strip().casefold().replace("_", "-").replace(".", "-")
        version = str(versions[0]).strip()
        if name != "comfyui-frontend-package" or not version or len(version) > 64:
            raise ComfyEditorBridgeError(
                "workflow-editor-frontend-invalid",
                "The ComfyUI frontend installation could not be verified.",
            )
        return version
    raise ComfyEditorBridgeError(
        "workflow-editor-frontend-unavailable",
        "The configured ComfyUI frontend package is unavailable.",
    )


def _site_packages_roots(executable: Path, directory: Path) -> tuple[Path, ...]:
    binary = executable.parent
    candidates = [
        binary / "Lib" / "site-packages",
        binary.parent / "Lib" / "site-packages",
        directory / ".venv" / "Lib" / "site-packages",
        directory / "venv" / "Lib" / "site-packages",
        directory.parent / "python_embeded" / "Lib" / "site-packages",
    ]
    candidates.extend(sorted(binary.parent.glob("lib/python*/site-packages")))
    candidates.extend(sorted((directory / ".venv").glob("lib/python*/site-packages")))
    candidates.extend(sorted((directory / "venv").glob("lib/python*/site-packages")))
    ordered: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate.absolute()))
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)
    return tuple(ordered)


def _coordinator_config(coordinator_origins: Iterable[str]) -> bytes:
    origins = sorted({_coordinator_origin(origin) for origin in coordinator_origins})
    if not origins:
        raise ComfyEditorBridgeError(
            "workflow-editor-coordinator-origin-invalid",
            "The workflow editor bridge has no trusted application origin.",
        )
    return (
        "export const COORDINATOR_ORIGINS = Object.freeze("
        + json.dumps(origins, ensure_ascii=True, separators=(",", ":"))
        + ");\n"
    ).encode("utf-8")


def _coordinator_origin(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ComfyEditorBridgeError(
            "workflow-editor-coordinator-origin-invalid",
            "The workflow editor bridge application origin is invalid.",
        )
    try:
        port = parsed.port
        loopback = (
            parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
        )
    except ValueError as exc:
        raise ComfyEditorBridgeError(
            "workflow-editor-coordinator-origin-invalid",
            "The workflow editor bridge application origin is invalid.",
        ) from exc
    if port is None or not loopback:
        raise ComfyEditorBridgeError(
            "workflow-editor-coordinator-origin-invalid",
            "The workflow editor bridge application origin must use an explicit port.",
        )
    return f"{parsed.scheme}://{parsed.netloc}".lower()


def _verified_asset_files(
    root: Path,
    *,
    expected_hashes: Mapping[str, str] = _BRIDGE_ASSET_HASHES,
) -> dict[str, bytes]:
    if not root.is_dir() or _is_link_or_reparse(root):
        raise ComfyEditorBridgeError(
            "workflow-editor-bridge-integrity-failed",
            "The bundled workflow editor bridge failed its integrity check.",
        )
    inventory = _asset_inventory(root)
    if set(inventory) != set(expected_hashes):
        raise ComfyEditorBridgeError(
            "workflow-editor-bridge-integrity-failed",
            "The bundled workflow editor bridge failed its integrity check.",
        )
    files: dict[str, bytes] = {}
    for relative, path in inventory.items():
        content = _read_bounded(path, MAX_BRIDGE_ASSET_BYTES)
        actual = hashlib.sha256(content).hexdigest()
        if not hmac.compare_digest(actual, expected_hashes[relative]):
            raise ComfyEditorBridgeError(
                "workflow-editor-bridge-integrity-failed",
                "The bundled workflow editor bridge failed its integrity check.",
            )
        files[relative] = content
    return files


def _asset_inventory(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    stack = [root]
    while stack:
        directory = stack.pop()
        for path in directory.iterdir():
            if _is_link_or_reparse(path):
                raise ComfyEditorBridgeError(
                    "workflow-editor-bridge-integrity-failed",
                    "The bundled workflow editor bridge failed its integrity check.",
                )
            if path.is_dir():
                stack.append(path)
                continue
            if not path.is_file():
                raise ComfyEditorBridgeError(
                    "workflow-editor-bridge-integrity-failed",
                    "The bundled workflow editor bridge failed its integrity check.",
                )
            relative = path.relative_to(root).as_posix()
            files[relative] = path
            if len(files) > len(_BRIDGE_ASSET_HASHES) + 1:
                return files
    return files


def _read_bounded(path: Path, limit: int) -> bytes:
    if _is_link_or_reparse(path) or not path.is_file():
        raise ComfyEditorBridgeError(
            "workflow-editor-runtime-metadata-invalid",
            "The media runtime metadata could not be verified.",
        )
    try:
        with path.open("rb") as handle:
            content = handle.read(limit + 1)
    except OSError as exc:
        raise ComfyEditorBridgeError(
            "workflow-editor-runtime-metadata-invalid",
            "The media runtime metadata could not be verified.",
        ) from exc
    if len(content) > limit:
        raise ComfyEditorBridgeError(
            "workflow-editor-runtime-metadata-invalid",
            "The media runtime metadata exceeds its size limit.",
        )
    return content


def _is_link_or_reparse(path: Path) -> bool:
    return is_link_or_reparse(
        path,
        missing="assume_regular",
        unreadable="assume_regular",
    )
