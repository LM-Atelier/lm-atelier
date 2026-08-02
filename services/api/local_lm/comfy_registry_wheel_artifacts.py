from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag, parse_tag, sys_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from .comfy_registry_dependencies import (
    ComfyRegistryDependency,
    ComfyRegistryDependencyError,
    ComfyRegistryDependencyPlan,
    plan_comfy_registry_dependencies,
)

MAX_PYPI_PROJECT_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_PYPI_PROJECT_FILES = 4_096
MAX_WHEEL_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_WHEEL_FILENAME_CHARACTERS = 500
MAX_WHEEL_URL_CHARACTERS = 2_000
MAX_SUPPORTED_WHEEL_TAGS = 4_096
_DIGEST_CHARACTERS = frozenset("0123456789abcdefABCDEF")
_MARKER_ENVIRONMENT_KEYS = frozenset(
    {
        "implementation_name",
        "implementation_version",
        "os_name",
        "platform_machine",
        "platform_python_implementation",
        "platform_release",
        "platform_system",
        "platform_version",
        "python_full_version",
        "python_version",
        "sys_platform",
    }
)


class ComfyRegistryWheelArtifactError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ComfyRegistryWheelArtifact:
    name: str
    version: str
    requirement: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    metadata_sha256: str | None
    compatibility_tag: str
    wheel_tags: tuple[str, ...]


@dataclass(frozen=True)
class ComfyRegistryWheelArtifactManifest:
    declaration_sha256: str
    target_sha256: str
    artifacts: tuple[ComfyRegistryWheelArtifact, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class _Candidate:
    artifact: ComfyRegistryWheelArtifact
    rank: int
    build: tuple[int, int, str]


def current_comfy_registry_wheel_target() -> tuple[dict[str, str], tuple[str, ...]]:
    """Return an explicit snapshot of the current interpreter's wheel target."""
    environment = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = ""
    return environment, tuple(str(tag) for tag in sys_tags())


def comfy_registry_wheel_target_sha256(
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
) -> str:
    """Validate and identify an explicit wheel compatibility target."""
    _, _, _, target_sha256 = _wheel_target(marker_environment, supported_tags)
    return target_sha256


def select_comfy_registry_wheel_artifact(
    requirement: str,
    project_document: object,
    *,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
) -> ComfyRegistryWheelArtifact:
    """Select the newest compatible binary artifact for one inert requirement."""
    try:
        plan = plan_comfy_registry_dependencies([requirement])
    except ComfyRegistryDependencyError as exc:
        raise ComfyRegistryWheelArtifactError(
            "invalid_wheel_requirement", "Wheel requirement is invalid"
        ) from exc
    dependency = plan.dependencies[0]
    if dependency.marker is not None:
        raise ComfyRegistryWheelArtifactError(
            "invalid_wheel_requirement",
            "Wheel selection requires an already-evaluated requirement",
        )
    environment, tags, ranks, _ = _wheel_target(
        marker_environment,
        supported_tags,
    )
    return _resolve_dependency(
        dependency,
        project_document,
        environment,
        tags,
        ranks,
        prefer_latest_version=True,
    )


def resolve_comfy_registry_wheel_artifacts(
    plan: ComfyRegistryDependencyPlan,
    project_documents: Mapping[str, object],
    *,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
) -> ComfyRegistryWheelArtifactManifest:
    """Bind exact active requirements to immutable compatible PyPI wheel records."""
    if plan.version_resolution_required:
        raise ComfyRegistryWheelArtifactError(
            "version_resolution_required",
            "Registry dependency versions must be exact before resolving wheel artifacts",
        )
    environment, tags, ranks, target_sha256 = _wheel_target(
        marker_environment,
        supported_tags,
    )
    active = _active_dependencies(plan.dependencies, environment)
    documents = _project_documents(project_documents)
    expected = {dependency.name for dependency in active}
    supplied = set(documents)
    if supplied != expected:
        missing = sorted(expected - supplied)
        code = "missing_project_metadata" if missing else "unexpected_project_metadata"
        detail = missing[0] if missing else sorted(supplied - expected)[0]
        raise ComfyRegistryWheelArtifactError(
            code,
            f"PyPI project metadata does not match active dependency {detail}",
        )

    artifacts = tuple(
        _resolve_dependency(dependency, documents[dependency.name], environment, tags, ranks)
        for dependency in active
    )
    manifest_payload = {
        "version": 1,
        "declaration_sha256": plan.declaration_sha256,
        "target_sha256": target_sha256,
        "artifacts": [_artifact_payload(artifact) for artifact in artifacts],
    }
    return ComfyRegistryWheelArtifactManifest(
        plan.declaration_sha256,
        target_sha256,
        artifacts,
        _payload_sha256(manifest_payload, "wheel artifact manifest"),
    )


def _wheel_target(
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
) -> tuple[dict[str, str], tuple[Tag, ...], dict[Tag, int], str]:
    environment = _marker_environment(marker_environment)
    tags, ranks = _supported_tags(supported_tags)
    payload = {
        "version": 1,
        "marker_environment": dict(sorted(environment.items())),
        "supported_tags": [str(tag) for tag in tags],
    }
    return environment, tags, ranks, _payload_sha256(payload, "wheel target")


def _marker_environment(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ComfyRegistryWheelArtifactError(
            "invalid_wheel_target", "Wheel marker environment must be an object"
        )
    environment: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, str)
            or not key
            or len(key) > 100
            or len(item) > 1_000
            or _has_control(key)
            or _has_control(item)
        ):
            raise ComfyRegistryWheelArtifactError(
                "invalid_wheel_target", "Wheel marker environment is invalid"
            )
        environment[key] = item
    missing = _MARKER_ENVIRONMENT_KEYS - environment.keys()
    if missing:
        raise ComfyRegistryWheelArtifactError(
            "invalid_wheel_target", "Wheel marker environment is incomplete"
        )
    environment.setdefault("extra", "")
    try:
        Version(environment["python_full_version"])
    except InvalidVersion as exc:
        raise ComfyRegistryWheelArtifactError(
            "invalid_wheel_target", "Wheel target has an invalid Python version"
        ) from exc
    return environment


def _supported_tags(values: Sequence[str]) -> tuple[tuple[Tag, ...], dict[Tag, int]]:
    if (
        isinstance(values, str | bytes)
        or not isinstance(values, Sequence)
        or not values
        or len(values) > MAX_SUPPORTED_WHEEL_TAGS
    ):
        raise ComfyRegistryWheelArtifactError(
            "invalid_wheel_target", "Supported wheel tags are invalid"
        )
    tags: list[Tag] = []
    seen: set[Tag] = set()
    for value in values:
        if not isinstance(value, str) or not value or len(value) > 200 or _has_control(value):
            raise ComfyRegistryWheelArtifactError(
                "invalid_wheel_target", "Supported wheel tags are invalid"
            )
        try:
            parsed = parse_tag(value)
        except ValueError as exc:
            raise ComfyRegistryWheelArtifactError(
                "invalid_wheel_target", "Supported wheel tags are invalid"
            ) from exc
        if len(parsed) != 1:
            raise ComfyRegistryWheelArtifactError(
                "invalid_wheel_target", "Supported wheel tags must be expanded"
            )
        tag = next(iter(parsed))
        if tag in seen:
            raise ComfyRegistryWheelArtifactError(
                "invalid_wheel_target", "Supported wheel tags contain a duplicate"
            )
        seen.add(tag)
        tags.append(tag)
    resolved = tuple(tags)
    return resolved, {tag: index for index, tag in enumerate(resolved)}


def _active_dependencies(
    dependencies: Sequence[ComfyRegistryDependency],
    environment: Mapping[str, str],
) -> tuple[ComfyRegistryDependency, ...]:
    active: list[ComfyRegistryDependency] = []
    names: set[str] = set()
    for dependency in dependencies:
        if dependency.marker and not Marker(dependency.marker).evaluate(environment=environment):
            continue
        if dependency.name in names:
            raise ComfyRegistryWheelArtifactError(
                "overlapping_dependency_markers",
                f"Multiple Registry requirements target active package {dependency.name}",
            )
        names.add(dependency.name)
        active.append(dependency)
    return tuple(active)


def _project_documents(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", "PyPI project metadata must be an object"
        )
    result: dict[str, object] = {}
    for name, document in value.items():
        if not isinstance(name, str) or not name or len(name) > 200 or _has_control(name):
            raise ComfyRegistryWheelArtifactError(
                "invalid_project_metadata", "PyPI project metadata has an invalid name"
            )
        canonical = canonicalize_name(name)
        if canonical in result:
            raise ComfyRegistryWheelArtifactError(
                "invalid_project_metadata", "PyPI project metadata contains a duplicate project"
            )
        result[canonical] = document
    return result


def _resolve_dependency(
    dependency: ComfyRegistryDependency,
    document: object,
    environment: Mapping[str, str],
    supported_tags: tuple[Tag, ...],
    ranks: Mapping[Tag, int],
    *,
    prefer_latest_version: bool = False,
) -> ComfyRegistryWheelArtifact:
    project = _project_document(dependency.name, document)
    candidates: list[_Candidate] = []
    filenames: set[str] = set()
    urls: set[str] = set()
    for record in project:
        candidate = _candidate(dependency, record, environment, supported_tags, ranks)
        if candidate is None:
            continue
        artifact = candidate.artifact
        if artifact.filename in filenames or artifact.url in urls:
            raise ComfyRegistryWheelArtifactError(
                "duplicate_wheel_artifact",
                f"PyPI metadata repeats a wheel for {dependency.name}",
            )
        filenames.add(artifact.filename)
        urls.add(artifact.url)
        candidates.append(candidate)
    if not candidates:
        raise ComfyRegistryWheelArtifactError(
            "no_compatible_wheel",
            f"No non-yanked, hash-bound compatible wheel exists for {dependency.requirement}",
        )
    if prefer_latest_version:
        specifier = Requirement(dependency.requirement).specifier
        versions = sorted({Version(candidate.artifact.version) for candidate in candidates})
        eligible = tuple(specifier.filter(versions, prereleases=None))
        if not eligible:
            raise ComfyRegistryWheelArtifactError(
                "no_compatible_wheel",
                f"No non-yanked, hash-bound compatible wheel exists for {dependency.requirement}",
            )
        best_version = max(eligible)
        candidates = [
            candidate
            for candidate in candidates
            if Version(candidate.artifact.version) == best_version
        ]
    best_rank = min(candidate.rank for candidate in candidates)
    ranked = [candidate for candidate in candidates if candidate.rank == best_rank]
    best_build = max(candidate.build for candidate in ranked)
    selected = [candidate for candidate in ranked if candidate.build == best_build]
    if len(selected) != 1:
        raise ComfyRegistryWheelArtifactError(
            "ambiguous_wheel_artifact",
            f"Multiple equally preferred wheels exist for {dependency.requirement}",
        )
    return selected[0].artifact


def _project_document(name: str, value: object) -> list[object]:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", f"PyPI metadata for {name} is not valid JSON"
        ) from exc
    if len(encoded) > MAX_PYPI_PROJECT_DOCUMENT_BYTES or not isinstance(value, dict):
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", f"PyPI metadata for {name} is invalid or too large"
        )
    meta = value.get("meta")
    api_version = meta.get("api-version") if isinstance(meta, dict) else None
    if not isinstance(api_version, str) or api_version.split(".", 1)[0] != "1":
        raise ComfyRegistryWheelArtifactError(
            "unsupported_project_api", f"PyPI metadata for {name} uses an unsupported API"
        )
    project_name = value.get("name")
    if not isinstance(project_name, str) or canonicalize_name(project_name) != name:
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", f"PyPI metadata identity does not match {name}"
        )
    files = value.get("files")
    if not isinstance(files, list) or len(files) > MAX_PYPI_PROJECT_FILES:
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", f"PyPI metadata file list for {name} is invalid"
        )
    return files


def _candidate(
    dependency: ComfyRegistryDependency,
    value: object,
    environment: Mapping[str, str],
    supported_tags: tuple[Tag, ...],
    ranks: Mapping[Tag, int],
) -> _Candidate | None:
    if not isinstance(value, dict):
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", f"PyPI file record for {dependency.name} is invalid"
        )
    filename = value.get("filename")
    if (
        not isinstance(filename, str)
        or not filename
        or len(filename) > MAX_WHEEL_FILENAME_CHARACTERS
        or _has_control(filename)
    ):
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", f"PyPI file name for {dependency.name} is invalid"
        )
    if not filename.endswith(".whl"):
        return None
    try:
        wheel_name, version, build, wheel_tags = parse_wheel_filename(filename)
    except InvalidWheelFilename as exc:
        raise ComfyRegistryWheelArtifactError(
            "invalid_wheel_filename", f"PyPI wheel name for {dependency.name} is invalid"
        ) from exc
    if str(wheel_name) != dependency.name:
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata",
            f"PyPI wheel identity does not match {dependency.name}",
        )
    requirement = dependency.requirement
    if not Requirement(requirement).specifier.contains(version, prereleases=True):
        return None
    yanked = value.get("yanked", False)
    if not isinstance(yanked, bool | str):
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", f"PyPI yank state for {dependency.name} is invalid"
        )
    if bool(yanked):
        return None
    if not _supports_python(value.get("requires-python"), environment["python_full_version"]):
        return None
    matching = wheel_tags.intersection(ranks)
    if not matching:
        return None
    rank = min(ranks[tag] for tag in matching)
    compatibility_tag = str(supported_tags[rank])
    artifact = ComfyRegistryWheelArtifact(
        dependency.name,
        str(version),
        requirement,
        filename,
        _wheel_url(value.get("url"), filename),
        _sha256(value.get("hashes"), f"wheel {filename}"),
        _wheel_size(value.get("size"), filename),
        _metadata_sha256(value.get("core-metadata"), filename),
        compatibility_tag,
        tuple(sorted(str(tag) for tag in wheel_tags)),
    )
    build_key = (0, 0, "") if not build else (1, build[0], build[1])
    return _Candidate(artifact, rank, build_key)


def _supports_python(value: object, python_version: str) -> bool:
    if value is None:
        return True
    if not isinstance(value, str) or len(value) > 500 or _has_control(value):
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", "PyPI Requires-Python metadata is invalid"
        )
    try:
        return SpecifierSet(value).contains(Version(python_version), prereleases=True)
    except (InvalidSpecifier, InvalidVersion) as exc:
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", "PyPI Requires-Python metadata is invalid"
        ) from exc


def _wheel_url(value: object, filename: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_WHEEL_URL_CHARACTERS or _has_control(value):
        raise ComfyRegistryWheelArtifactError(
            "invalid_wheel_url", f"PyPI wheel URL for {filename} is invalid"
        )
    parsed = urlparse(value)
    segments = parsed.path.split("/")
    decoded = [unquote(segment) for segment in segments if segment]
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/packages/")
        or not decoded
        or decoded[-1] != filename
        or any(
            segment in {".", ".."} or "/" in segment or chr(92) in segment for segment in decoded
        )
    ):
        raise ComfyRegistryWheelArtifactError(
            "invalid_wheel_url", f"PyPI wheel URL for {filename} is not allowlisted"
        )
    return value


def _sha256(value: object, label: str) -> str:
    digest = value.get("sha256") if isinstance(value, dict) else None
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _DIGEST_CHARACTERS for character in digest)
    ):
        raise ComfyRegistryWheelArtifactError(
            "missing_wheel_hash", f"PyPI {label} lacks a valid SHA-256 hash"
        )
    return digest.lower()


def _wheel_size(value: object, filename: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ComfyRegistryWheelArtifactError(
            "invalid_wheel_size", f"PyPI wheel size for {filename} is invalid"
        )
    if value > MAX_WHEEL_ARTIFACT_BYTES:
        raise ComfyRegistryWheelArtifactError(
            "wheel_too_large", f"PyPI wheel {filename} exceeds the size limit"
        )
    return value


def _metadata_sha256(value: object, filename: str) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, dict):
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", f"PyPI metadata hash for {filename} is invalid"
        )
    return _sha256(value, f"metadata for {filename}")


def _artifact_payload(artifact: ComfyRegistryWheelArtifact) -> dict[str, object]:
    return {
        "name": artifact.name,
        "version": artifact.version,
        "requirement": artifact.requirement,
        "filename": artifact.filename,
        "url": artifact.url,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "metadata_sha256": artifact.metadata_sha256,
        "compatibility_tag": artifact.compatibility_tag,
        "wheel_tags": list(artifact.wheel_tags),
    }


def _payload_sha256(payload: object, label: str) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise ComfyRegistryWheelArtifactError(
            "invalid_project_metadata", f"Could not encode {label}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _has_control(value: str) -> bool:
    return any(character < " " or character == "\x7f" for character in value)
