from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from .comfy_registry import ComfyNodeResolution

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_REPOSITORY_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")

ComfyPackageSourceKind = Literal["git_commit", "registry_archive"]


class ComfyPackageSourceError(ValueError):
    pass


@dataclass(frozen=True)
class ComfyPackageSourceIdentity:
    install_kind: ComfyPackageSourceKind
    package_id: str
    package_version: str
    source_record_id: str
    repository_url: str
    download_url: str


def resolve_comfy_package_source(
    resolution: ComfyNodeResolution,
) -> ComfyPackageSourceIdentity:
    """Return one exact download and persistence identity for a package."""
    if (
        not isinstance(resolution, ComfyNodeResolution)
        or not resolution.resolved
        or not isinstance(resolution.package_id, str)
        or _PACKAGE_ID.fullmatch(resolution.package_id) is None
        or not isinstance(resolution.declared_version, str)
    ):
        raise ComfyPackageSourceError("package resolution has an invalid identity")
    repository = _repository_url(resolution.repository_url)
    if resolution.install_kind == "registry_archive":
        version = resolution.declared_version
        if _SEMANTIC_VERSION.fullmatch(version) is None:
            raise ComfyPackageSourceError("package resolution has an invalid Registry version")
        record_id = _text(resolution.registry_record_id, "Registry record id", 1_000)
        download_url = _registry_download_url(resolution.download_url)
        return ComfyPackageSourceIdentity(
            "registry_archive",
            resolution.package_id,
            version,
            record_id,
            repository,
            download_url,
        )
    if resolution.install_kind == "git_commit":
        revision = resolution.declared_version
        if _COMMIT.fullmatch(revision) is None:
            raise ComfyPackageSourceError("package resolution has an invalid commit revision")
        digest = hashlib.sha256(f"{repository}{chr(0)}{revision}".encode()).hexdigest()
        record_id = f"github-commit:{digest}"
        download_url = _github_commit_download_url(repository, revision)
        if resolution.registry_record_id not in (
            None,
            record_id,
        ) or resolution.download_url not in (
            None,
            download_url,
        ):
            raise ComfyPackageSourceError("commit resolution contains conflicting source metadata")
        return ComfyPackageSourceIdentity(
            "git_commit",
            resolution.package_id,
            revision,
            record_id,
            repository,
            download_url,
        )
    raise ComfyPackageSourceError("package resolution has an unsupported install kind")


def _repository_url(value: object) -> str:
    source = _text(value, "package repository URL", 1_000)
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
        raise ComfyPackageSourceError("package resolution has an invalid repository URL")
    repository = parts[1][:-4] if parts[1].endswith(".git") else ""
    canonical = f"https://github.com/{parts[0]}/{repository}.git"
    if not repository or not _REPOSITORY_PART.fullmatch(repository) or source != canonical:
        raise ComfyPackageSourceError("package resolution has an invalid repository URL")
    return canonical


def _registry_download_url(value: object) -> str:
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
        raise ComfyPackageSourceError("package resolution has an invalid Registry download URL")
    return source


def _github_commit_download_url(repository: str, revision: str) -> str:
    parsed = urlparse(repository)
    owner, repository_name = [part for part in parsed.path.split("/") if part]
    repository_name = repository_name[:-4]
    return f"https://codeload.github.com/{owner}/{repository_name}/zip/{revision}"


def _text(value: object, name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(character < " " or character == "\x7f" for character in value)
    ):
        raise ComfyPackageSourceError(f"invalid {name}")
    return value
