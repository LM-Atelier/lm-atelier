from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import JobKind, JobStatus
from .filesystem_links import is_link_or_reparse
from .models import ComfyRegistryInstall, Job, WorkflowDependencyBinding

logger = logging.getLogger(__name__)

_NODE_PATH = re.compile(r"^lm-atelier-registry_[A-Za-z0-9._-]{1,200}$")
_ENVIRONMENT_PATH = re.compile(r"^registry-wheels-v3-[0-9a-f]{64}$")
_MAX_PENDING_REFRESH_JOBS = 10_000
RegistryInstallDiskStatus = Literal[
    "ready",
    "node_files_missing",
    "wheel_environment_missing",
    "files_missing",
]


class ComfyRegistryReconciliationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class RegistryInstallDiskState:
    status: RegistryInstallDiskStatus
    node_files_present: bool
    wheel_environment_present: bool


def inspect_registry_install_disk_state(
    install: ComfyRegistryInstall,
    *,
    custom_node_root: Path | None,
    environment_root: Path,
) -> RegistryInstallDiskState:
    node_present = _present_managed_child(
        custom_node_root,
        install.installed_path,
        pattern=_NODE_PATH,
    )
    environment_present = _present_managed_child(
        environment_root,
        install.wheel_environment_path,
        pattern=_ENVIRONMENT_PATH,
    )
    if node_present and environment_present:
        status: RegistryInstallDiskStatus = "ready"
    elif node_present:
        status = "wheel_environment_missing"
    elif environment_present:
        status = "node_files_missing"
    else:
        status = "files_missing"
    return RegistryInstallDiskState(status, node_present, environment_present)


def remove_registry_install(
    session: Session,
    *,
    install_id: str,
    custom_node_root: Path | None,
    environment_root: Path,
) -> None:
    """Remove one inactive install's row and exclusively owned managed paths.

    Paths are renamed to hidden siblings before the database commit. A failed
    flush or commit restores those exact directory entries; successful commits
    make the install disappear atomically and then reclaim the quarantines.
    """

    install = session.get(ComfyRegistryInstall, install_id)
    if install is None:
        raise ComfyRegistryReconciliationError(
            "registry_install_not_found",
            "Registry package install was not found",
        )
    if install.active:
        raise ComfyRegistryReconciliationError(
            "registry_install_active",
            "Deactivate the Registry package before removing it",
        )
    if (
        session.scalar(
            select(WorkflowDependencyBinding.id).where(
                WorkflowDependencyBinding.comfy_registry_install_id == install.id
            )
        )
        is not None
    ):
        raise ComfyRegistryReconciliationError(
            "registry_install_in_use",
            "A workflow still depends on this Registry package",
        )
    pending_payloads = session.scalars(
        select(Job.payload_json)
        .where(
            Job.kind == JobKind.REGISTRY_PREPARE.value,
            Job.status.in_((JobStatus.QUEUED.value, JobStatus.RUNNING.value)),
        )
        .limit(_MAX_PENDING_REFRESH_JOBS + 1)
    ).all()
    if len(pending_payloads) > _MAX_PENDING_REFRESH_JOBS:
        raise ComfyRegistryReconciliationError(
            "registry_install_busy",
            "The Registry package refresh queue could not be verified safely",
        )
    if any(
        isinstance(payload, dict) and payload.get("renew_install_id") == install.id
        for payload in pending_payloads
    ):
        raise ComfyRegistryReconciliationError(
            "registry_install_busy",
            "A dependency refresh is still using this Registry package",
        )

    node_path = _managed_child(
        custom_node_root,
        install.installed_path,
        pattern=_NODE_PATH,
    )
    environment_path: Path | None = None
    if install.wheel_environment_path is not None:
        shared = session.scalar(
            select(ComfyRegistryInstall.id).where(
                ComfyRegistryInstall.id != install.id,
                ComfyRegistryInstall.wheel_environment_path == install.wheel_environment_path,
            )
        )
        if shared is None:
            environment_path = _managed_child(
                environment_root,
                install.wheel_environment_path,
                pattern=_ENVIRONMENT_PATH,
            )

    staged: list[tuple[Path, Path]] = []
    try:
        for original in (node_path, environment_path):
            if original is None:
                continue
            quarantine = original.with_name(f".lm-atelier-removing-{uuid4().hex}")
            os.replace(original, quarantine)
            staged.append((original, quarantine))
            if not quarantine.is_dir() or _is_link(quarantine):
                raise ComfyRegistryReconciliationError(
                    "registry_install_path_invalid",
                    "A managed Registry path changed during removal",
                )
        session.delete(install)
        session.flush()
        session.commit()
    except Exception as exc:
        session.rollback()
        _restore_staged_paths(staged)
        if isinstance(exc, ComfyRegistryReconciliationError):
            raise
        raise ComfyRegistryReconciliationError(
            "registry_install_remove_failed",
            "The Registry package could not be removed",
        ) from exc

    for _original, quarantine in staged:
        try:
            shutil.rmtree(quarantine)
        except OSError:
            logger.warning("Could not reclaim removed Registry package files at %s", quarantine)


def _present_managed_child(root: Path | None, name: object, *, pattern: re.Pattern[str]) -> bool:
    try:
        return _managed_child(root, name, pattern=pattern) is not None
    except ComfyRegistryReconciliationError:
        return False


def _managed_child(
    root: Path | None,
    name: object,
    *,
    pattern: re.Pattern[str],
) -> Path | None:
    if not isinstance(name, str) or not pattern.fullmatch(name):
        if name is None:
            return None
        raise ComfyRegistryReconciliationError(
            "registry_install_path_invalid",
            "A managed Registry path is invalid",
        )
    if root is None or not root.exists():
        return None
    if not root.is_dir() or _is_link(root):
        raise ComfyRegistryReconciliationError(
            "registry_install_path_invalid",
            "A managed Registry root is invalid",
        )
    candidate = root / name
    if not candidate.exists():
        return None
    if not candidate.is_dir() or _is_link(candidate):
        raise ComfyRegistryReconciliationError(
            "registry_install_path_invalid",
            "A managed Registry path is invalid",
        )
    try:
        if candidate.resolve(strict=True).parent != root.resolve(strict=True):
            raise ComfyRegistryReconciliationError(
                "registry_install_path_invalid",
                "A managed Registry path is invalid",
            )
    except OSError as exc:
        raise ComfyRegistryReconciliationError(
            "registry_install_path_invalid",
            "A managed Registry path is unreadable",
        ) from exc
    return candidate


def _restore_staged_paths(staged: list[tuple[Path, Path]]) -> None:
    failures: list[Path] = []
    for original, quarantine in reversed(staged):
        try:
            if original.exists():
                failures.append(original)
                continue
            os.replace(quarantine, original)
        except OSError:
            failures.append(quarantine)
    if failures:
        raise ComfyRegistryReconciliationError(
            "registry_install_restore_failed",
            "Registry package removal failed and its files could not be restored",
        )


def _is_link(path: Path) -> bool:
    return is_link_or_reparse(path, missing="assume_link", unreadable="assume_link")
