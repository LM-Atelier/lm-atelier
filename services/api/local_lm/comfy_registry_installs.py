from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from .comfy_registry import ComfyNodeResolution
from .comfy_registry_archives import ComfyRegistryArchiveReport
from .models import ComfyRegistryInstall

_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_DIGEST = re.compile(r"^[0-9a-fA-F]{64}$")
_INSTALL_PATH = re.compile(r"^lm-atelier-registry_[A-Za-z0-9._-]{1,200}$")
_REPOSITORY_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


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
