from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import stat
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from .comfy_registry import ComfyNodeResolution
from .comfy_registry_archives import (
    ComfyRegistryArchiveError,
    ComfyRegistryArchiveReport,
    parse_comfy_registry_runtime_files,
    verify_staged_comfy_registry_archive,
)
from .comfy_registry_dependencies import (
    ComfyRegistryDependencyError,
    plan_comfy_registry_dependencies,
)
from .comfy_registry_downloads import DownloadProgress
from .comfy_registry_installs import (
    bind_comfy_registry_wheel_environment,
    persist_comfy_registry_install,
)
from .comfy_registry_sources import ComfyPackageSourceError, resolve_comfy_package_source
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
    ComfyRegistryWheelEnvironmentError,
    ComfyRegistryWheelEnvironmentReport,
    assemble_comfy_registry_wheel_environment,
    verify_comfy_registry_wheel_environment,
)
from .models import ComfyRegistryInstall
from .source_omission_proof import PendingOmission, record_pending_omission

_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class ComfyRegistryStagedArchive:
    installed_path: str
    destination: Path
    report: ComfyRegistryArchiveReport


async def stage_comfy_registry_install_archive(
    *,
    resolution: ComfyNodeResolution,
    archive_downloader: RegistryArchiveDownloader,
    custom_node_root: Path,
    media_worker_stopped: bool,
    archive_progress: DownloadProgress | None = None,
) -> ComfyRegistryStagedArchive:
    """Stage package code once, still inert and review-required."""
    if media_worker_stopped is not True:
        raise ComfyRegistryLifecycleError(
            "media_worker_running",
            "The media worker must be stopped before preparing a Registry package",
        )
    package_id, package_version, record_id = _resolution_identity(resolution)
    node_root = _managed_root(custom_node_root, "custom node")
    installed_path = _installed_path(package_id, package_version, record_id)
    destination = node_root / installed_path
    if destination.exists() or destination.is_symlink():
        raise ComfyRegistryLifecycleError(
            "node_destination_exists", "The managed Registry node destination already exists"
        )
    try:
        report = await archive_downloader.download_and_stage(
            resolution,
            destination,
            progress=archive_progress,
        )
        staged = ComfyRegistryStagedArchive(installed_path, destination, report)
        _validate_staged_archive(staged, destination, installed_path)
        return staged
    except (Exception, asyncio.CancelledError):
        if destination.exists() or destination.is_symlink():
            await _remove_tree(destination, node_root)
        raise


async def discard_comfy_registry_staged_archive(
    *,
    resolution: ComfyNodeResolution,
    staged_archive: ComfyRegistryStagedArchive,
    custom_node_root: Path,
) -> None:
    """Remove only the deterministic inert tree belonging to this resolution."""
    package_id, package_version, record_id = _resolution_identity(resolution)
    node_root = _managed_root(custom_node_root, "custom node")
    installed_path = _installed_path(package_id, package_version, record_id)
    destination = node_root / installed_path
    if (
        not isinstance(staged_archive, ComfyRegistryStagedArchive)
        or staged_archive.installed_path != installed_path
        or staged_archive.destination != destination
    ):
        raise ComfyRegistryLifecycleError(
            "invalid_staged_archive", "The staged Registry package identity is invalid"
        )
    await _remove_tree(destination, node_root)


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
    staged_archive: ComfyRegistryStagedArchive | None = None,
    environment_assembler: EnvironmentAssembler = assemble_comfy_registry_wheel_environment,
    pending_omission: PendingOmission | None = None,
) -> ComfyRegistryPreparation:
    """Prepare one exact Registry package without trusting or activating it.

    A pending omission is written in the same commit as the install it is
    about. Persisting it afterwards would leave a window - a cancellation, a
    process death, a failed second commit - in which an install whose
    dependencies were skipped exists with nothing recording that they were.
    """
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
    environment_preexisting = (
        environment_destination.exists() or environment_destination.is_symlink()
    )
    owns_node_destination = False
    try:
        if session.scalar(
            select(ComfyRegistryInstall.id).where(
                ComfyRegistryInstall.registry_record_id == record_id
            )
        ):
            raise ComfyRegistryLifecycleError(
                "registry_install_exists", "This exact Registry package is already prepared"
            )
        if staged_archive is None:
            if node_destination.exists() or node_destination.is_symlink():
                raise ComfyRegistryLifecycleError(
                    "node_destination_exists",
                    "The managed Registry node destination already exists",
                )
            owns_node_destination = True
            staged_archive = await stage_comfy_registry_install_archive(
                resolution=resolution,
                archive_downloader=archive_downloader,
                custom_node_root=node_root,
                media_worker_stopped=True,
                archive_progress=archive_progress,
            )
        else:
            _validate_staged_archive(staged_archive, node_destination, installed_path)
            owns_node_destination = True
        if wheel_destination.exists() or wheel_destination.is_symlink():
            raise ComfyRegistryLifecycleError(
                "wheel_stage_exists", "Registry wheel staging from another attempt still exists"
            )
        archive = staged_archive.report
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
        if pending_omission is not None:
            install.review_json = record_pending_omission(
                install.review_json,
                manifest_sha256=install.manifest_sha256,
                omitted_declarations=pending_omission.omitted_declarations,
                workflow_revision_id=pending_omission.workflow_revision_id,
                required_node_types=pending_omission.required_node_types,
            )
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
        if owns_node_destination and (node_destination.exists() or node_destination.is_symlink()):
            await _remove_tree(node_destination, node_root)
        if not environment_preexisting and (
            environment_destination.exists() or environment_destination.is_symlink()
        ):
            await _remove_tree(environment_destination, environment_root)
        raise


async def renew_comfy_registry_install_environment(
    session: Session,
    *,
    install_id: str,
    resolution: ComfyNodeResolution,
    closure: ComfyRegistryWheelClosure,
    wheel_downloader: RegistryWheelDownloader,
    python_executable: Path,
    custom_node_root: Path,
    state_root: Path,
    media_worker_stopped: bool,
    wheel_progress: WheelDownloadProgress | None = None,
    environment_assembler: EnvironmentAssembler = assemble_comfy_registry_wheel_environment,
) -> ComfyRegistryPreparation:
    """Rebuild only an inactive package's target-bound wheel environment.

    The reviewed node archive never changes. A new environment is assembled
    beside the old one, the database binding moves in one commit, and only then
    is an unshared old environment retired. Failed renewal therefore leaves the
    prior verified binding usable.
    """
    if media_worker_stopped is not True:
        raise ComfyRegistryLifecycleError(
            "media_worker_running",
            "The media worker must be stopped before renewing a Registry package",
        )
    install = session.get(ComfyRegistryInstall, install_id)
    if install is None:
        raise ComfyRegistryLifecycleError(
            "registry_install_not_found", "Registry package install was not found"
        )
    if install.active:
        raise ComfyRegistryLifecycleError(
            "registry_install_active",
            "Deactivate the Registry package before refreshing its dependencies",
        )
    _validate_renewal_identity(install, resolution)
    artifacts = _complete_closure(closure, resolution)
    node_root = _managed_root(custom_node_root, "custom node")
    managed_state = _managed_root(state_root, "state")
    environment_root = _managed_child(managed_state, "registry-wheel-environments")
    staging_root = _managed_child(managed_state, "registry-wheel-staging")
    node_destination = _existing_managed_child(node_root, install.installed_path, "node")
    _verify_existing_archive(install, node_destination)
    old_environment_name = install.wheel_environment_path
    if not old_environment_name:
        raise ComfyRegistryLifecycleError(
            "registry_environment_missing",
            "Registry package has no prepared dependency environment",
        )
    old_environment = _existing_managed_child(environment_root, old_environment_name, "environment")
    environment_destination = environment_root / f"registry-wheels-{closure.closure_sha256}"
    wheel_destination = staging_root / f"registry-wheels-{closure.closure_sha256}"
    environment_preexisting = (
        environment_destination.exists() or environment_destination.is_symlink()
    )
    old_shared = session.scalar(
        select(ComfyRegistryInstall.id)
        .where(
            ComfyRegistryInstall.id != install.id,
            ComfyRegistryInstall.wheel_environment_path == old_environment_name,
        )
        .limit(1)
    )
    retirement = environment_root / f".{old_environment_name}.retiring-{install.id}"
    retired = False
    original_trusted = install.trusted
    try:
        if wheel_destination.exists() or wheel_destination.is_symlink():
            raise ComfyRegistryLifecycleError(
                "wheel_stage_exists", "Registry wheel staging from another attempt still exists"
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
        if old_environment != environment_destination and old_shared is None:
            if retirement.exists() or retirement.is_symlink():
                raise ComfyRegistryLifecycleError(
                    "renewal_cleanup_pending",
                    "A previous Registry dependency renewal still needs cleanup",
                )
            old_environment.rename(retirement)
            retired = True
        install.trusted = False
        install.wheel_closure_sha256 = None
        install.wheel_environment_sha256 = None
        install.wheel_environment_path = None
        try:
            bind_comfy_registry_wheel_environment(
                install,
                closure,
                environment,
                environment_destination,
                environment_root=environment_root,
            )
        finally:
            install.trusted = original_trusted
        preparation = ComfyRegistryPreparation(
            install.id,
            install.installed_path,
            install.wheel_environment_path or "",
            install.archive_sha256,
            install.manifest_sha256,
            install.wheel_closure_sha256 or "",
            install.wheel_environment_sha256 or "",
            reused,
        )
        session.commit()
    except (Exception, asyncio.CancelledError):
        session.rollback()
        await _remove_tree(wheel_destination, staging_root)
        if retired and (retirement.exists() or retirement.is_symlink()):
            retirement.rename(old_environment)
        if (
            not environment_preexisting
            and environment_destination != old_environment
            and (environment_destination.exists() or environment_destination.is_symlink())
        ):
            await _remove_tree(environment_destination, environment_root)
        raise
    if retired:
        try:
            await _remove_tree(retirement, environment_root)
        except ComfyRegistryLifecycleError:
            logger.warning(
                "Registry dependency renewal committed but stale environment cleanup is pending"
            )
    return preparation


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
        try:
            report = await asyncio.to_thread(
                verify_comfy_registry_wheel_environment,
                environment_destination,
                expected_closure_sha256=closure.closure_sha256,
                expected_environment_sha256=existing.wheel_environment_sha256,
            )
        except ComfyRegistryWheelEnvironmentError as exc:
            raise ComfyRegistryLifecycleError(exc.code, str(exc)) from exc
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
    try:
        report = await environment_assembler(
            closure,
            wheel_files,
            python_executable=python_executable,
            destination=environment_destination,
            media_worker_stopped=True,
        )
    except ComfyRegistryWheelEnvironmentError as exc:
        raise ComfyRegistryLifecycleError(exc.code, str(exc)) from exc
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
    try:
        source = resolve_comfy_package_source(resolution)
    except ComfyPackageSourceError as exc:
        raise ComfyRegistryLifecycleError("invalid_resolution", str(exc)) from exc
    return source.package_id, source.package_version, source.source_record_id


def _validate_renewal_identity(
    install: ComfyRegistryInstall, resolution: ComfyNodeResolution
) -> None:
    package_id, package_version, record_id = _resolution_identity(resolution)
    expected = (
        package_id,
        package_version,
        record_id,
        resolution.repository_url,
        resolution.download_url,
        tuple(resolution.node_types),
        tuple(resolution.pip_dependencies),
    )
    actual = (
        install.package_id,
        install.package_version,
        install.registry_record_id,
        install.repository_url,
        install.download_url,
        tuple(install.node_types_json),
        tuple(install.pip_dependencies_json),
    )
    if actual != expected:
        raise ComfyRegistryLifecycleError(
            "registry_install_identity_changed",
            "Registry package identity changed; remove and review a fresh package instead",
        )


def _verify_existing_archive(install: ComfyRegistryInstall, destination: Path) -> None:
    review = install.review_json if isinstance(install.review_json, dict) else {}
    file_count = review.get("file_count")
    expanded_bytes = review.get("expanded_bytes")
    if (
        not isinstance(file_count, int)
        or isinstance(file_count, bool)
        or file_count < 1
        or not isinstance(expanded_bytes, int)
        or isinstance(expanded_bytes, bool)
        or expanded_bytes < 1
    ):
        raise ComfyRegistryLifecycleError(
            "registry_install_review_missing",
            "Registry package has no complete archive review evidence",
        )
    try:
        verify_staged_comfy_registry_archive(
            destination,
            expected_manifest_sha256=install.manifest_sha256,
            expected_file_count=file_count,
            expected_expanded_bytes=expanded_bytes,
            runtime_files=parse_comfy_registry_runtime_files(review.get("runtime_files")),
        )
    except ComfyRegistryArchiveError as exc:
        raise ComfyRegistryLifecycleError(
            "registry_install_verification_failed",
            "Registry package files failed verification",
        ) from exc


def _validate_staged_archive(
    staged_archive: ComfyRegistryStagedArchive,
    destination: Path,
    installed_path: str,
) -> None:
    if (
        not isinstance(staged_archive, ComfyRegistryStagedArchive)
        or staged_archive.installed_path != installed_path
        or staged_archive.destination != destination
        or not isinstance(staged_archive.report, ComfyRegistryArchiveReport)
        or staged_archive.report.review_required is not True
    ):
        raise ComfyRegistryLifecycleError(
            "invalid_staged_archive", "The staged Registry package identity is invalid"
        )
    try:
        verify_staged_comfy_registry_archive(
            destination,
            expected_manifest_sha256=staged_archive.report.manifest_sha256,
            expected_file_count=staged_archive.report.file_count,
            expected_expanded_bytes=staged_archive.report.expanded_bytes,
        )
    except ComfyRegistryArchiveError as exc:
        raise ComfyRegistryLifecycleError(
            "invalid_staged_archive", "The staged Registry package files are invalid"
        ) from exc


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


def _existing_managed_child(root: Path, name: str, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ComfyRegistryLifecycleError(
            "invalid_managed_path", f"Managed Registry {label} path is invalid"
        )
    path = root / name
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ComfyRegistryLifecycleError(
            "managed_path_missing", f"Managed Registry {label} path is missing"
        ) from exc
    if resolved.parent != root or _is_link_or_reparse(path) or not resolved.is_dir():
        raise ComfyRegistryLifecycleError(
            "invalid_managed_path", f"Managed Registry {label} path is invalid"
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


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)
