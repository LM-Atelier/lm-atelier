from __future__ import annotations

import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from .comfy_registry import ComfyNodeResolution
from .comfy_registry_archives import (
    ComfyRegistryArchiveReport,
    verify_staged_comfy_registry_archive,
)
from .comfy_registry_dependencies import (
    ComfyRegistryDependencyError,
    plan_comfy_registry_dependencies,
)
from .comfy_registry_wheel_closure import (
    ComfyRegistryWheelClosure,
    ComfyRegistryWheelClosureError,
    validate_comfy_registry_wheel_closure,
)
from .comfy_registry_wheel_environments import (
    ComfyRegistryWheelEnvironmentError,
    ComfyRegistryWheelEnvironmentReport,
    verify_comfy_registry_wheel_environment,
)
from .models import ComfyRegistryInstall

if TYPE_CHECKING:
    from .workflow_activations import WorkflowRegistryLaunchBinding

_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_DIGEST = re.compile(r"^[0-9a-fA-F]{64}$")
_INSTALL_PATH = re.compile(r"^lm-atelier-registry_[A-Za-z0-9._-]{1,200}$")
_REPOSITORY_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_ENVIRONMENT_PATH = re.compile(r"^registry-wheels-([0-9a-f]{64})$")


class ComfyRegistryInstallError(ValueError):
    pass


@dataclass(frozen=True)
class _InstallIdentity:
    package_id: str
    package_version: str
    registry_record_id: str
    repository_url: str
    download_url: str
    archive_sha256: str
    manifest_sha256: str
    installed_path: str
    node_types: tuple[str, ...]
    pip_dependencies: tuple[str, ...]
    review: dict[str, object]


@dataclass(frozen=True)
class ComfyRegistryLaunchContract:
    custom_node_folders: tuple[str, ...]
    site_packages: tuple[Path, ...]
    node_types: tuple[str, ...]


def installed_comfy_registry_versions(session: Session) -> dict[str, set[str]]:
    """Return exact package versions that are both trusted and active."""
    result: dict[str, set[str]] = {}
    statement = select(
        ComfyRegistryInstall.package_id,
        ComfyRegistryInstall.package_version,
    ).where(
        ComfyRegistryInstall.trusted.is_(True),
        ComfyRegistryInstall.active.is_(True),
    )
    for package_id, package_version in session.execute(statement):
        result.setdefault(package_id, set()).add(package_version)
    return result


def persist_comfy_registry_install(
    session: Session,
    *,
    resolution: ComfyNodeResolution,
    archive: ComfyRegistryArchiveReport,
    installed_path: str,
) -> ComfyRegistryInstall:
    """Persist an inert Registry archive identity without trusting or activating it."""
    identity = _identity(resolution, archive, installed_path)
    existing = session.scalar(
        select(ComfyRegistryInstall).where(
            ComfyRegistryInstall.registry_record_id == identity.registry_record_id
        )
    )
    if existing is not None:
        if not _matches(existing, identity):
            raise ComfyRegistryInstallError(
                "Comfy Registry record identity conflicts with the staged archive"
            )
        return existing
    version_match = session.scalar(
        select(ComfyRegistryInstall).where(
            ComfyRegistryInstall.package_id == identity.package_id,
            ComfyRegistryInstall.package_version == identity.package_version,
        )
    )
    if version_match is not None:
        raise ComfyRegistryInstallError(
            "Comfy Registry package version is already bound to another record"
        )
    path_match = session.scalar(
        select(ComfyRegistryInstall).where(
            ComfyRegistryInstall.installed_path == identity.installed_path
        )
    )
    if path_match is not None:
        raise ComfyRegistryInstallError("managed Registry install path is already recorded")

    install = ComfyRegistryInstall(
        package_id=identity.package_id,
        package_version=identity.package_version,
        registry_record_id=identity.registry_record_id,
        repository_url=identity.repository_url,
        download_url=identity.download_url,
        archive_sha256=identity.archive_sha256,
        manifest_sha256=identity.manifest_sha256,
        installed_path=identity.installed_path,
        node_types_json=list(identity.node_types),
        pip_dependencies_json=list(identity.pip_dependencies),
        review_json=identity.review,
        trusted=False,
        active=False,
    )
    session.add(install)
    session.flush()
    return install


def bind_comfy_registry_wheel_environment(
    install: ComfyRegistryInstall,
    closure: ComfyRegistryWheelClosure,
    report: ComfyRegistryWheelEnvironmentReport,
    destination: Path,
    *,
    environment_root: Path,
) -> None:
    """Bind an inert Registry install to one exact verified dependency overlay."""
    if not isinstance(install, ComfyRegistryInstall) or install.trusted or install.active:
        raise ComfyRegistryInstallError("Registry install must be inactive and untrusted")
    try:
        artifacts = validate_comfy_registry_wheel_closure(closure)
    except ComfyRegistryWheelClosureError as exc:
        raise ComfyRegistryInstallError("Registry wheel closure is invalid") from exc
    if not closure.complete:
        raise ComfyRegistryInstallError("Registry wheel closure is incomplete")
    try:
        dependency_plan = plan_comfy_registry_dependencies(install.pip_dependencies_json)
    except ComfyRegistryDependencyError as exc:
        raise ComfyRegistryInstallError("Registry dependency declarations are invalid") from exc
    if dependency_plan.declaration_sha256 != closure.manifest.declaration_sha256:
        raise ComfyRegistryInstallError("Registry wheel closure does not match the install")
    root = _managed_root(environment_root)
    path = _managed_environment_path(root, destination)
    try:
        verified = verify_comfy_registry_wheel_environment(
            path,
            expected_closure_sha256=closure.closure_sha256,
            expected_environment_sha256=report.environment_sha256,
        )
    except ComfyRegistryWheelEnvironmentError as exc:
        raise ComfyRegistryInstallError("Registry wheel environment is invalid") from exc
    if (
        verified != report
        or report.closure_sha256 != closure.closure_sha256
        or report.artifact_count != len(artifacts)
    ):
        raise ComfyRegistryInstallError("Registry wheel environment identity does not match")
    identity = (closure.closure_sha256, report.environment_sha256, path.name)
    current = (
        install.wheel_closure_sha256,
        install.wheel_environment_sha256,
        install.wheel_environment_path,
    )
    if any(value is not None for value in current) and current != identity:
        raise ComfyRegistryInstallError("Registry install is already bound to another environment")
    (
        install.wheel_closure_sha256,
        install.wheel_environment_sha256,
        install.wheel_environment_path,
    ) = identity


def trusted_comfy_registry_launch_contract(
    session: Session,
    *,
    custom_node_root: Path,
    environment_root: Path,
) -> ComfyRegistryLaunchContract:
    """Revalidate every trusted active Registry package for a stopped-worker launch."""
    installs = session.scalars(
        select(ComfyRegistryInstall)
        .where(
            ComfyRegistryInstall.trusted.is_(True),
            ComfyRegistryInstall.active.is_(True),
        )
        .order_by(ComfyRegistryInstall.installed_path)
    ).all()
    return _verified_comfy_registry_launch_contract(
        installs,
        custom_node_root=custom_node_root,
        environment_root=environment_root,
    )


def scoped_comfy_registry_launch_contract(
    session: Session,
    bindings: Sequence[WorkflowRegistryLaunchBinding],
    *,
    custom_node_root: Path,
    environment_root: Path,
) -> ComfyRegistryLaunchContract:
    """Revalidate only the exact Registry packages frozen into one activation."""

    installs: list[ComfyRegistryInstall] = []
    seen: set[str] = set()
    for binding in bindings:
        if binding.registry_install_id in seen:
            raise ComfyRegistryInstallError("Registry launch scope contains duplicate packages")
        seen.add(binding.registry_install_id)
        install = session.get(ComfyRegistryInstall, binding.registry_install_id)
        if install is None or not install.trusted or not install.active:
            raise ComfyRegistryInstallError("Registry launch scope package is unavailable")
        _assert_scoped_registry_identity(install, binding)
        installs.append(install)
    return _verified_comfy_registry_launch_contract(
        installs,
        custom_node_root=custom_node_root,
        environment_root=environment_root,
        expected_bindings={item.registry_install_id: item for item in bindings},
    )


def _verified_comfy_registry_launch_contract(
    installs: Sequence[ComfyRegistryInstall],
    *,
    custom_node_root: Path,
    environment_root: Path,
    expected_bindings: Mapping[str, WorkflowRegistryLaunchBinding] | None = None,
) -> ComfyRegistryLaunchContract:
    if not installs:
        return ComfyRegistryLaunchContract((), (), ())
    node_root = _managed_root(custom_node_root)
    wheel_root = _managed_root(environment_root)
    folders: list[str] = []
    site_packages: set[Path] = set()
    node_types: set[str] = set()
    for install in installs:
        folder = _registry_node_path(node_root, install.installed_path)
        expected = expected_bindings.get(install.id) if expected_bindings else None
        if expected is not None and folder != expected.installed_path:
            raise ComfyRegistryInstallError("Registry launch scope node path changed")
        review = _activation_review(install.review_json)
        try:
            verify_staged_comfy_registry_archive(
                folder,
                expected_manifest_sha256=install.manifest_sha256,
                expected_file_count=review["file_count"],
                expected_expanded_bytes=review["expanded_bytes"],
            )
        except ValueError as exc:
            raise ComfyRegistryInstallError("Registry node files failed verification") from exc
        if not (
            install.wheel_closure_sha256
            and install.wheel_environment_sha256
            and install.wheel_environment_path
        ):
            raise ComfyRegistryInstallError("Registry install has no verified wheel environment")
        environment = _managed_environment_path(
            wheel_root,
            wheel_root / install.wheel_environment_path,
        )
        if expected is not None and environment / "site-packages" != expected.site_packages:
            raise ComfyRegistryInstallError("Registry launch scope environment path changed")
        try:
            verify_comfy_registry_wheel_environment(
                environment,
                expected_closure_sha256=install.wheel_closure_sha256,
                expected_environment_sha256=install.wheel_environment_sha256,
            )
        except ComfyRegistryWheelEnvironmentError as exc:
            raise ComfyRegistryInstallError(
                "Registry wheel environment failed verification"
            ) from exc
        declared_nodes = _node_types(install.node_types_json)
        folders.append(folder.name)
        site_packages.add(environment / "site-packages")
        node_types.update(declared_nodes)
    return ComfyRegistryLaunchContract(
        tuple(folders),
        tuple(sorted(site_packages)),
        tuple(sorted(node_types)),
    )


def _assert_scoped_registry_identity(
    install: ComfyRegistryInstall,
    binding: WorkflowRegistryLaunchBinding,
) -> None:
    expected = (
        binding.package_id,
        binding.package_version,
        binding.archive_sha256,
        binding.manifest_sha256,
        binding.wheel_closure_sha256,
        binding.wheel_environment_sha256,
        binding.node_types,
    )
    actual = (
        install.package_id,
        install.package_version,
        install.archive_sha256.lower(),
        install.manifest_sha256.lower(),
        str(install.wheel_closure_sha256).lower(),
        str(install.wheel_environment_sha256).lower(),
        tuple(
            sorted(
                _node_types(install.node_types_json),
                key=lambda item: (item.casefold(), item),
            )
        ),
    )
    if actual != expected:
        raise ComfyRegistryInstallError("Registry launch scope identity changed")


def _identity(
    resolution: ComfyNodeResolution,
    archive: ComfyRegistryArchiveReport,
    installed_path: str,
) -> _InstallIdentity:
    if (
        not resolution.resolved
        or resolution.install_kind != "registry_archive"
        or not resolution.package_id
        or not resolution.declared_version
        or not resolution.registry_record_id
        or not resolution.repository_url
        or not resolution.download_url
    ):
        raise ComfyRegistryInstallError(
            "resolution does not identify an exact Comfy Registry archive"
        )
    if not _PACKAGE_ID.fullmatch(resolution.package_id):
        raise ComfyRegistryInstallError("resolution has an invalid Registry package id")
    version = _text(resolution.declared_version, "Registry package version", 100)
    if not _SEMANTIC_VERSION.fullmatch(version):
        raise ComfyRegistryInstallError("resolution has an invalid Registry package version")
    record_id = _text(resolution.registry_record_id, "Registry record id", 1_000)
    repository_url = _repository_url(resolution.repository_url)
    download_url = _download_url(resolution.download_url)
    archive_sha256 = _digest(archive.archive_sha256, "archive hash")
    manifest_sha256 = _digest(archive.manifest_sha256, "manifest hash")
    if not archive.review_required:
        raise ComfyRegistryInstallError("Registry archive must remain review-required")
    if not _INSTALL_PATH.fullmatch(installed_path):
        raise ComfyRegistryInstallError("invalid managed Registry install path")
    node_types = tuple(_text(value, "node type", 200) for value in resolution.node_types)
    if not node_types or len(set(node_types)) != len(node_types):
        raise ComfyRegistryInstallError("resolution has invalid node type identities")
    dependencies = tuple(_text(value, "dependency", 1_000) for value in resolution.pip_dependencies)
    review = _review(archive)
    review["registry_warnings"] = [
        _text(value, "Registry warning", 200) for value in resolution.warnings
    ]
    return _InstallIdentity(
        resolution.package_id,
        version,
        record_id,
        repository_url,
        download_url,
        archive_sha256,
        manifest_sha256,
        installed_path,
        node_types,
        dependencies,
        review,
    )


def _matches(install: ComfyRegistryInstall, identity: _InstallIdentity) -> bool:
    values: Mapping[str, object] = {
        "package_id": identity.package_id,
        "package_version": identity.package_version,
        "registry_record_id": identity.registry_record_id,
        "repository_url": identity.repository_url,
        "download_url": identity.download_url,
        "archive_sha256": identity.archive_sha256,
        "manifest_sha256": identity.manifest_sha256,
        "installed_path": identity.installed_path,
        "node_types_json": list(identity.node_types),
        "pip_dependencies_json": list(identity.pip_dependencies),
        "review_json": identity.review,
    }
    return all(getattr(install, name) == value for name, value in values.items())


def _text(value: object, name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(character < " " or character == "\x7f" for character in value)
    ):
        raise ComfyRegistryInstallError(f"invalid {name}")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ComfyRegistryInstallError(f"invalid {name}")
    return value.lower()


def _repository_url(value: str) -> str:
    source = _text(value, "Registry repository URL", 1_000)
    parsed = urlparse(source)
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or len(parts) != 2
        or not _REPOSITORY_PART.fullmatch(parts[0])
    ):
        raise ComfyRegistryInstallError("invalid Registry repository URL")
    repository = parts[1][:-4] if parts[1].endswith(".git") else ""
    canonical = f"https://github.com/{parts[0]}/{repository}.git"
    if not repository or not _REPOSITORY_PART.fullmatch(repository) or source != canonical:
        raise ComfyRegistryInstallError("invalid Registry repository URL")
    return canonical


def _download_url(value: str) -> str:
    source = _text(value, "Registry download URL", 1_000)
    parsed = urlparse(source)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "cdn.comfy.org"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
        or not parsed.path.endswith(".zip")
        or source != f"https://cdn.comfy.org{parsed.path}"
    ):
        raise ComfyRegistryInstallError("invalid Registry download URL")
    return source


def _review(archive: ComfyRegistryArchiveReport) -> dict[str, object]:
    entry_count = _count(archive.entry_count, "archive entry count")
    file_count = _count(archive.file_count, "archive file count")
    expanded_bytes = _count(archive.expanded_bytes, "archive expanded size")
    python_file_count = _count(archive.python_file_count, "archive Python file count")
    if file_count > entry_count or python_file_count > file_count:
        raise ComfyRegistryInstallError("invalid Registry archive review counts")
    return {
        "entry_count": entry_count,
        "file_count": file_count,
        "expanded_bytes": expanded_bytes,
        "python_file_count": python_file_count,
        "dependency_manifests": _paths(archive.dependency_manifests),
        "install_scripts": _paths(archive.install_scripts),
        "startup_hooks": _paths(archive.startup_hooks),
        "native_files": _paths(archive.native_files),
        "top_level_entries": _paths(archive.top_level_entries),
        "review_required": True,
    }


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComfyRegistryInstallError(f"invalid {name}")
    return value


def _paths(values: object) -> list[str]:
    if not isinstance(values, tuple):
        raise ComfyRegistryInstallError("invalid Registry archive review paths")
    result = [_text(value, "Registry archive review path", 1_024) for value in values]
    if tuple(result) != tuple(sorted(set(result))):
        raise ComfyRegistryInstallError("invalid Registry archive review paths")
    return result


def _managed_root(value: Path) -> Path:
    if not isinstance(value, Path) or _is_link_or_reparse(value):
        raise ComfyRegistryInstallError("managed Registry root is invalid")
    try:
        root = value.resolve(strict=True)
    except OSError as exc:
        raise ComfyRegistryInstallError("managed Registry root is missing") from exc
    if not root.is_dir() or _is_link_or_reparse(root):
        raise ComfyRegistryInstallError("managed Registry root is invalid")
    return root


def _managed_environment_path(root: Path, value: Path) -> Path:
    if not isinstance(value, Path) or not _ENVIRONMENT_PATH.fullmatch(value.name):
        raise ComfyRegistryInstallError("managed Registry wheel environment path is invalid")
    try:
        path = value.resolve(strict=True)
    except OSError as exc:
        raise ComfyRegistryInstallError("managed Registry wheel environment is missing") from exc
    if path.parent != root or _is_link_or_reparse(value) or _is_link_or_reparse(path):
        raise ComfyRegistryInstallError("managed Registry wheel environment path is invalid")
    return path


def _registry_node_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not _INSTALL_PATH.fullmatch(value):
        raise ComfyRegistryInstallError("managed Registry node path is invalid")
    try:
        path = (root / value).resolve(strict=True)
    except OSError as exc:
        raise ComfyRegistryInstallError("managed Registry node files are missing") from exc
    if path.parent != root or _is_link_or_reparse(root / value):
        raise ComfyRegistryInstallError("managed Registry node path is invalid")
    return path


def _activation_review(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or value.get("review_required") is not True:
        raise ComfyRegistryInstallError("Registry install review evidence is invalid")
    return {
        "file_count": _count(value.get("file_count"), "archive file count"),
        "expanded_bytes": _count(value.get("expanded_bytes"), "archive expanded size"),
    }


def _node_types(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > 4_096:
        raise ComfyRegistryInstallError("Registry install node types are invalid")
    nodes = tuple(_text(item, "node type", 200) for item in value)
    if len(nodes) != len(set(nodes)):
        raise ComfyRegistryInstallError("Registry install node types are invalid")
    return nodes


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return True
    return path.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
