"""Verify model files before reserving the transaction that activates a workflow."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Never

from sqlalchemy.orm import Session

from . import workflow_activations as activations
from .filesystem_links import _FILE_BASIC_INFORMATION_CLASS, _windows_api

_SEAL = object()
_FileIdentity = tuple[int, int, int, int, int, int]


def _changed_install() -> Never:
    raise activations.WorkflowActivationError(
        "dependency_content_drift", "Selected workflow install changed after verification"
    )


def _identity(path: Path) -> _FileIdentity:
    try:
        value = path.stat()
        changed = value.st_ctime_ns
        if sys.platform == "win32" and stat.S_ISREG(value.st_mode):
            # Windows stat exposes creation time as ctime. Read the actual
            # change time so restoring mtime cannot conceal rewritten bytes.
            with path.open("rb") as opened:
                value = os.fstat(opened.fileno())
                changed = _windows_change_time(opened.fileno())
    except OSError as exc:
        raise activations.WorkflowActivationError(
            "dependency_content_drift", "Selected workflow files are unavailable"
        ) from exc
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        changed,
    )


def _windows_change_time(descriptor: int) -> int:
    if sys.platform != "win32":
        raise OSError("Windows file change time is unavailable")
    import msvcrt

    api = _windows_api()
    information = api.FileBasicInformation()
    status_block = api.IoStatusBlock()
    result = api.ntdll.NtQueryInformationFile(
        api.ctypes.c_void_p(msvcrt.get_osfhandle(descriptor)),
        api.ctypes.byref(status_block),
        api.ctypes.byref(information),
        api.ctypes.c_ulong(api.ctypes.sizeof(information)),
        api.ctypes.c_ulong(_FILE_BASIC_INFORMATION_CLASS),
    )
    if result & 0xFFFFFFFF:
        raise OSError("Windows file change time is unavailable")
    return int(information.ChangeTime)


@dataclass(frozen=True)
class _VerifiedFiles:
    root: Path
    directory: tuple[int, int, int]
    files: tuple[tuple[str, Path, _FileIdentity], ...]

    def require_current(self, base: Path) -> None:
        root = activations._directory(base, "Selected workflow directory is unavailable")
        if root != self.root or _identity(root)[:3] != self.directory:
            raise activations.WorkflowActivationError(
                "dependency_content_drift", "Selected workflow directory changed"
            )
        for relative, expected_path, expected_identity in self.files:
            path = activations._contained_file(root, relative)
            if path != expected_path or _identity(path) != expected_identity:
                raise activations.WorkflowActivationError(
                    "dependency_content_drift", "Selected workflow file changed"
                )


def _verify_files(base: Path, files: Sequence[tuple[str, str]]) -> _VerifiedFiles:
    root = activations._directory(base, "Selected workflow directory is unavailable")
    directory = _identity(root)[:3]
    verified: list[tuple[str, Path, _FileIdentity]] = []
    for relative, digest in files:
        path = activations._contained_file(root, relative)
        before = _identity(path)
        activations._verify_file_digest(path, digest, "Selected workflow file bytes changed")
        if _identity(path) != before:
            raise activations.WorkflowActivationError(
                "dependency_content_drift", "Selected workflow file changed while being read"
            )
        verified.append((relative, path, before))
    result = _VerifiedFiles(root, directory, tuple(verified))
    result.require_current(base)
    return result


@dataclass(frozen=True)
class VerifiedWorkflowFiles:
    models: tuple[tuple[activations._ModelLaunchInput, _VerifiedFiles], ...]
    assets: tuple[tuple[activations._AssetLaunchInput, _VerifiedFiles], ...]
    seal: object

    def model_binding(
        self, session: Session, install_id: str
    ) -> activations.WorkflowModelLaunchBinding:
        current = activations._model_launch_input(session, install_id)
        for expected, files in self.models:
            if expected.binding.model_install_id == install_id:
                self._require_identity(current == expected)
                files.require_current(current.binding.base_path)
                return replace(current.binding, base_path=files.root)
        _changed_install()

    def asset_binding(
        self, session: Session, install_id: str
    ) -> activations.WorkflowAssetLaunchBinding:
        current = activations._asset_launch_input(session, install_id)
        for expected, files in self.assets:
            if expected.binding.model_asset_install_id == install_id:
                self._require_identity(current == expected)
                files.require_current(current.binding.base_path)
                return replace(current.binding, base_path=files.root)
        _changed_install()

    def _require_identity(self, matches: bool) -> None:
        if self.seal is not _SEAL or not matches:
            _changed_install()

    def require_current(self, session: Session) -> None:
        """Compare fresh install rows and contained file identities before commit."""
        self._require_identity(True)
        for model, _files in self.models:
            self.model_binding(session, model.binding.model_install_id)
        for asset, _files in self.assets:
            self.asset_binding(session, asset.binding.model_asset_install_id)


def verify_workflow_files(
    session_factory: Callable[[], Session],
    model_ids: Sequence[str],
    asset_ids: Sequence[str],
) -> VerifiedWorkflowFiles:
    """Close a consistent database read before hashing the selected installation files."""
    with session_factory() as session:
        session.connection().exec_driver_sql("BEGIN")
        models = tuple(
            activations._model_launch_input(session, identifier)
            for identifier in sorted(set(model_ids))
        )
        assets = tuple(
            activations._asset_launch_input(session, identifier)
            for identifier in sorted(set(asset_ids))
        )
    return VerifiedWorkflowFiles(
        tuple((value, _verify_files(value.binding.base_path, value.files)) for value in models),
        tuple(
            (
                value,
                _verify_files(
                    value.binding.base_path,
                    [(value.binding.runtime_reference, value.binding.sha256)],
                ),
            )
            for value in assets
        ),
        _SEAL,
    )
