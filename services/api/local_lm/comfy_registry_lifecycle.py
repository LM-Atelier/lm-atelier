from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import stat
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from .comfy_registry import ComfyNodeResolution
from .comfy_registry_archives import ComfyRegistryArchiveReport
from .comfy_registry_dependencies import (
    ComfyRegistryDependencyError,
    plan_comfy_registry_dependencies,
)
from .comfy_registry_downloads import DownloadProgress
from .comfy_registry_installs import (
    bind_comfy_registry_wheel_environment,
    persist_comfy_registry_install,
)
from .comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifact,
    ComfyRegistryWheelArtifactManifest,
)
from .comfy_registry_wheel_closure import (
    ComfyRegistryWheelClosure,
    ComfyRegistryWheelClosureError,
    validate_comfy_registry_wheel_closure,
)
from .comfy_registry_wheel_downloads import (
    ComfyRegistryWheelStageReport,
    WheelDownloadProgress,
)
from .comfy_registry_wheel_environments import (
    ComfyRegistryWheelEnvironmentReport,
    assemble_comfy_registry_wheel_environment,
    verify_comfy_registry_wheel_environment,
)
from .models import ComfyRegistryInstall

_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class ComfyRegistryLifecycleError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RegistryArchiveDownloader(Protocol):
    async def download_and_stage(
        self,
        resolution: ComfyNodeResolution,
        destination: Path,
        *,
        progress: DownloadProgress | None = None,
    ) -> ComfyRegistryArchiveReport: ...


class RegistryWheelDownloader(Protocol):
    async def download_and_stage(
        self,
        manifest: ComfyRegistryWheelArtifactManifest,
        destination: Path,
        *,
        progress: WheelDownloadProgress | None = None,
    ) -> ComfyRegistryWheelStageReport: ...


EnvironmentAssembler = Callable[..., Awaitable[ComfyRegistryWheelEnvironmentReport]]


@dataclass(frozen=True)
class ComfyRegistryPreparation:
    install_id: str
    installed_path: str
    wheel_environment_path: str
    archive_sha256: str
    manifest_sha256: str
    wheel_closure_sha256: str
    wheel_environment_sha256: str
    reused_wheel_environment: bool


async def prepare_comfy_registry_install(
    session: Session,
    *,
    resolution: ComfyNodeResolution,
    closure: ComfyRegistryWheelClosure,
    archive_downloader: RegistryArchiveDownloader,
    wheel_downloader: RegistryWheelDownloader,
    python_executable: Path,
    custom_node_root: Path,
    state_root: Path,
    media_worker_stopped: bool,
    archive_progress: DownloadProgress | None = None,
    wheel_progress: WheelDownloadProgress | None = None,
    environment_assembler: EnvironmentAssembler = assemble_comfy_registry_wheel_environment,
) -> ComfyRegistryPreparation:
    """Prepare one exact Registry package without trusting or activating it."""
    if media_worker_stopped is not True:
        raise ComfyRegistryLifecycleError(
            "media_worker_running",
            "The media worker must be stopped before preparing a Registry package",
        )
    package_id, package_version, record_id = _resolution_identity(resolution)
    artifacts = _complete_closure(closure, resolution)
    node_root = _managed_root(custom_node_root, "custom node")
    managed_state = _managed_root(state_root, "state")
    environment_root = _managed_child(managed_state, "registry-wheel-environments")
    staging_root = _managed_child(managed_state, "registry-wheel-staging")
    installed_path = _installed_path(package_id, package_version, record_id)
    node_destination = node_root / installed_path
    environment_destination = environment_root / f"registry-wheels-{closure.closure_sha256}"
    wheel_destination = staging_root / f"registry-wheels-{closure.closure_sha256}"
    if session.scalar(
        select(ComfyRegistryInstall.id).where(ComfyRegistryInstall.registry_record_id == record_id)
    ):
        raise ComfyRegistryLifecycleError(
            "registry_install_exists", "This exact Registry package is already prepared"
        )
    if node_destination.exists() or node_destination.is_symlink():
        raise ComfyRegistryLifecycleError(
            "node_destination_exists", "The managed Registry node destination already exists"
        )
    if wheel_destination.exists() or wheel_destination.is_symlink():
        raise ComfyRegistryLifecycleError(
            "wheel_stage_exists", "Registry wheel staging from another attempt still exists"
        )

    environment_preexisting = (
        environment_destination.exists() or environment_destination.is_symlink()
    )
    try:
        archive = await archive_downloader.download_and_stage(
            resolution,
            node_destination,
            progress=archive_progress,
        )
        environment, reused = await _environment(
            session,
            closure=closure,
            artifacts=artifacts,
            wheel_downloader=wheel_downloader,
            wheel_destination=wheel_destination,
            wheel_progress=wheel_progress,
            python_executable=python_executable,
            environment_destination=environment_destination,
            environment_root=environment_root,
            environment_assembler=environment_assembler,
        )
        await _remove_tree(wheel_destination, staging_root)
        install = persist_comfy_registry_install(
            session,
            resolution=resolution,
            archive=archive,
            installed_path=installed_path,
        )
        bind_comfy_registry_wheel_environment(
            install,
            closure,
            environment,
            environment_destination,
            environment_root=environment_root,
        )
        install.trusted = False
        install.active = False
        if not (
            install.wheel_environment_path
            and install.wheel_closure_sha256
            and install.wheel_environment_sha256
        ):
            raise ComfyRegistryLifecycleError(
                "binding_incomplete", "Registry package environment binding is incomplete"
            )
        session.commit()
        session.refresh(install)
        return ComfyRegistryPreparation(
            install.id,
            install.installed_path,
            install.wheel_environment_path,
            install.archive_sha256,
            install.manifest_sha256,
            install.wheel_closure_sha256,
            install.wheel_environment_sha256,
            reused,
        )
    except (Exception, asyncio.CancelledError):
        session.rollback()
        await _remove_tree(wheel_destination, staging_root)
        if node_destination.exists() or node_destination.is_symlink():
            await _remove_tree(node_destination, node_root)
        if not environment_preexisting and (
            environment_destination.exists() or environment_destination.is_symlink()
        ):
            await _remove_tree(environment_destination, environment_root)
        raise


async def _environment(
    session: Session,
    *,
    closure: ComfyRegistryWheelClosure,
    artifacts: tuple[ComfyRegistryWheelArtifact, ...],
    wheel_downloader: RegistryWheelDownloader,
    wheel_destination: Path,
    wheel_progress: WheelDownloadProgress | None,
    python_executable: Path,
    environment_destination: Path,
    environment_root: Path,
    environment_assembler: EnvironmentAssembler,
) -> tuple[ComfyRegistryWheelEnvironmentReport, bool]:
    if environment_destination.exists() or environment_destination.is_symlink():
        existing = session.scalar(
            select(ComfyRegistryInstall)
            .where(
                ComfyRegistryInstall.wheel_closure_sha256 == closure.closure_sha256,
                ComfyRegistryInstall.wheel_environment_path == environment_destination.name,
            )
            .order_by(ComfyRegistryInstall.id)
        )
        if existing is None or not existing.wheel_environment_sha256:
            raise ComfyRegistryLifecycleError(
                "unbound_wheel_environment",
                "The existing Registry wheel environment has no trusted database identity",
            )
        report = await asyncio.to_thread(
            verify_comfy_registry_wheel_environment,
            environment_destination,
            expected_closure_sha256=closure.closure_sha256,
            expected_environment_sha256=existing.wheel_environment_sha256,
        )
        return report, True

    wheel_files: dict[str, Path] = {}
    if artifacts:
        stage = await wheel_downloader.download_and_stage(
            closure.manifest,
            wheel_destination,
            progress=wheel_progress,
        )
        if stage.artifact_manifest_sha256 != closure.manifest.manifest_sha256:
            raise ComfyRegistryLifecycleError(
                "wheel_stage_identity_mismatch",
                "Registry wheel staging does not match the closed dependency manifest",
            )
        wheel_files = {
            artifact.filename: wheel_destination / artifact.filename for artifact in artifacts
        }
    report = await environment_assembler(
        closure,
        wheel_files,
        python_executable=python_executable,
        destination=environment_destination,
        media_worker_stopped=True,
    )
    return report, False


def _complete_closure(
    closure: ComfyRegistryWheelClosure,
    resolution: ComfyNodeResolution,
) -> tuple[ComfyRegistryWheelArtifact, ...]:
    try:
        artifacts = validate_comfy_registry_wheel_closure(closure)
        dependencies = plan_comfy_registry_dependencies(resolution.pip_dependencies)
    except (ComfyRegistryWheelClosureError, ComfyRegistryDependencyError) as exc:
        raise ComfyRegistryLifecycleError(
            "invalid_dependency_closure", "Registry dependency closure is invalid"
        ) from exc
    if not closure.complete:
        raise ComfyRegistryLifecycleError(
            "dependency_closure_incomplete", "Registry dependency closure is incomplete"
        )
    if dependencies.declaration_sha256 != closure.manifest.declaration_sha256:
        raise ComfyRegistryLifecycleError(
            "dependency_closure_mismatch",
            "Registry dependency closure does not belong to this package",
        )
    return artifacts


def _resolution_identity(resolution: ComfyNodeResolution) -> tuple[str, str, str]:
    if not isinstance(resolution, ComfyNodeResolution) or not resolution.resolved:
        raise ComfyRegistryLifecycleError(
            "invalid_resolution", "Registry package resolution is incomplete"
        )
    package_id = resolution.package_id
    package_version = resolution.declared_version
    record_id = resolution.registry_record_id
    if (
        resolution.install_kind != "registry_archive"
        or not isinstance(package_id, str)
        or not _PACKAGE_ID.fullmatch(package_id)
        or not isinstance(package_version, str)
        or not _SEMANTIC_VERSION.fullmatch(package_version)
        or not isinstance(record_id, str)
        or not record_id
        or len(record_id) > 1_000
        or _has_control(record_id)
    ):
        raise ComfyRegistryLifecycleError(
            "invalid_resolution", "Registry package resolution has an invalid identity"
        )
    return package_id, package_version, record_id


def _installed_path(package_id: str, package_version: str, record_id: str) -> str:
    identity = f"{package_id}{chr(0)}{package_version}{chr(0)}{record_id}".encode()
    return f"lm-atelier-registry_{hashlib.sha256(identity).hexdigest()}"


def _managed_root(path: Path, label: str) -> Path:
    if not isinstance(path, Path) or _is_link_or_reparse(path):
        raise ComfyRegistryLifecycleError(
            "invalid_managed_root", f"Managed Registry {label} root is invalid"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ComfyRegistryLifecycleError(
            "invalid_managed_root", f"Managed Registry {label} root is missing"
        ) from exc
    if not resolved.is_dir() or _is_link_or_reparse(resolved):
        raise ComfyRegistryLifecycleError(
            "invalid_managed_root", f"Managed Registry {label} root is invalid"
        )
    return resolved


def _managed_child(root: Path, name: str) -> Path:
    path = root / name
    try:
        path.mkdir(exist_ok=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ComfyRegistryLifecycleError(
            "managed_root_unavailable", "Managed Registry storage is unavailable"
        ) from exc
    if resolved.parent != root or _is_link_or_reparse(path) or not resolved.is_dir():
        raise ComfyRegistryLifecycleError(
            "invalid_managed_root", "Managed Registry storage is invalid"
        )
    return resolved


async def _remove_tree(path: Path, root: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise ComfyRegistryLifecycleError(
            "cleanup_failed", "Registry preparation cleanup root is unavailable"
        ) from exc
    if resolved_parent != root or _is_link_or_reparse(path):
        raise ComfyRegistryLifecycleError(
            "cleanup_failed", "Registry preparation cleanup path is unsafe"
        )
    await asyncio.to_thread(shutil.rmtree, path)


def _has_control(value: str) -> bool:
    return any(character < " " or character == "\x7f" for character in value)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)
