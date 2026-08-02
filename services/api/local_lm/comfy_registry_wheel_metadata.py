from __future__ import annotations

import hashlib
import hmac
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser

from packaging.markers import Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifact,
    ComfyRegistryWheelArtifactManifest,
)

MAX_WHEEL_CORE_METADATA_BYTES = 1024 * 1024
MAX_WHEEL_CORE_METADATA_LINES = 10_000
MAX_WHEEL_CORE_METADATA_LINE_BYTES = 10_000
MAX_WHEEL_REQUIRES_DIST = 1_024
MAX_WHEEL_TRANSITIVE_REQUIREMENTS = 4_096
MAX_WHEEL_TRANSITIVE_EXTRAS = 1_024
MAX_WHEEL_REQUIREMENT_CHARACTERS = 1_000
MAX_WHEEL_ARTIFACTS = 4_096
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


class ComfyRegistryWheelMetadataError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ComfyRegistryWheelMetadataRequirement:
    source_name: str
    source_version: str
    name: str
    requirement: str
    specifier: str
    marker: str | None
    extras: tuple[str, ...]


@dataclass(frozen=True)
class ComfyRegistryWheelMetadataFrontier:
    name: str
    requirements: tuple[str, ...]
    sources: tuple[str, ...]
    requested_extras: tuple[str, ...]
    locked_version: str | None
    status: str


@dataclass(frozen=True)
class ComfyRegistryWheelMetadataPlan:
    requirements: tuple[ComfyRegistryWheelMetadataRequirement, ...]
    frontier: tuple[ComfyRegistryWheelMetadataFrontier, ...]
    unavailable_metadata: tuple[str, ...]
    resolution_required: bool
    conflicts: tuple[str, ...]
    plan_sha256: str


@dataclass(frozen=True)
class _ParsedRequirement:
    name: str
    requirement: str
    specifier: str
    marker: str | None
    extras: tuple[str, ...]


@dataclass(frozen=True)
class _MetadataRecord:
    artifact: ComfyRegistryWheelArtifact
    requirements: tuple[_ParsedRequirement, ...]


def plan_comfy_registry_wheel_metadata(
    manifest: ComfyRegistryWheelArtifactManifest,
    metadata_documents: Mapping[str, bytes],
    *,
    marker_environment: Mapping[str, str],
) -> ComfyRegistryWheelMetadataPlan:
    """Parse hash-bound wheel metadata into an inert transitive dependency frontier."""
    artifacts = _artifacts(manifest)
    environment = _marker_environment(marker_environment)
    documents = _metadata_documents(metadata_documents)
    expected = {
        artifact.filename: artifact
        for artifact in artifacts
        if artifact.metadata_sha256 is not None
    }
    if set(documents) != set(expected):
        missing = sorted(set(expected) - set(documents))
        code = "missing_core_metadata" if missing else "unexpected_core_metadata"
        detail = missing[0] if missing else sorted(set(documents) - set(expected))[0]
        raise ComfyRegistryWheelMetadataError(
            code, f"Wheel core metadata does not match artifact {detail}"
        )

    records = tuple(
        _metadata_record(artifact, documents[artifact.filename])
        for artifact in artifacts
        if artifact.metadata_sha256 is not None
    )
    unavailable = tuple(
        artifact.filename for artifact in artifacts if artifact.metadata_sha256 is None
    )
    requirements = _active_requirements(records, artifacts, environment)
    frontier, conflicts = _frontier(requirements, artifacts)
    payload = {
        "version": 1,
        "artifact_manifest_sha256": manifest.manifest_sha256,
        "requirements": [_requirement_payload(item) for item in requirements],
        "frontier": [_frontier_payload(item) for item in frontier],
        "unavailable_metadata": list(unavailable),
        "conflicts": list(conflicts),
    }
    return ComfyRegistryWheelMetadataPlan(
        requirements,
        frontier,
        unavailable,
        bool(unavailable) or any(item.status != "satisfied" for item in frontier),
        conflicts,
        _payload_sha256(payload),
    )


def _artifacts(
    manifest: ComfyRegistryWheelArtifactManifest,
) -> tuple[ComfyRegistryWheelArtifact, ...]:
    if not isinstance(manifest, ComfyRegistryWheelArtifactManifest):
        raise ComfyRegistryWheelMetadataError(
            "invalid_artifact_manifest", "Wheel artifact manifest is invalid"
        )
    declaration_sha256 = _digest(manifest.declaration_sha256)
    target_sha256 = _digest(manifest.target_sha256)
    manifest_sha256 = _digest(manifest.manifest_sha256)
    artifacts = manifest.artifacts
    if not isinstance(artifacts, tuple) or len(artifacts) > MAX_WHEEL_ARTIFACTS:
        raise ComfyRegistryWheelMetadataError(
            "invalid_artifact_manifest", "Wheel artifact manifest is invalid"
        )
    names: set[str] = set()
    filenames: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, ComfyRegistryWheelArtifact):
            raise ComfyRegistryWheelMetadataError(
                "invalid_artifact_manifest", "Wheel artifact manifest is invalid"
            )
        _validated_artifact(artifact)
        if artifact.name in names or artifact.filename in filenames:
            raise ComfyRegistryWheelMetadataError(
                "invalid_artifact_manifest", "Wheel artifact manifest contains duplicates"
            )
        names.add(artifact.name)
        filenames.add(artifact.filename)
    if artifacts != tuple(sorted(artifacts, key=lambda item: (item.name, item.requirement))):
        raise ComfyRegistryWheelMetadataError(
            "invalid_artifact_manifest", "Wheel artifacts are not canonical"
        )
    payload = {
        "version": 1,
        "declaration_sha256": declaration_sha256,
        "target_sha256": target_sha256,
        "artifacts": [_artifact_payload(artifact) for artifact in artifacts],
    }
    if not hmac.compare_digest(_payload_sha256(payload), manifest_sha256):
        raise ComfyRegistryWheelMetadataError(
            "artifact_manifest_hash_mismatch",
            "Wheel artifact manifest hash does not match its contents",
        )
    return artifacts


def _validated_artifact(artifact: ComfyRegistryWheelArtifact) -> None:
    name = _text(artifact.name, 200)
    if canonicalize_name(name) != name:
        raise ComfyRegistryWheelMetadataError(
            "invalid_artifact_manifest", "Wheel artifact name is not canonical"
        )
    try:
        version = Version(_text(artifact.version, 200))
        requirement = Requirement(_text(artifact.requirement, 1_000))
    except (InvalidRequirement, InvalidVersion) as exc:
        raise ComfyRegistryWheelMetadataError(
            "invalid_artifact_manifest", "Wheel artifact identity is invalid"
        ) from exc
    if (
        str(version) != artifact.version
        or requirement.url is not None
        or canonicalize_name(requirement.name) != name
        or not requirement.specifier.contains(version, prereleases=True)
    ):
        raise ComfyRegistryWheelMetadataError(
            "invalid_artifact_manifest", "Wheel artifact identity is invalid"
        )
    _text(artifact.filename, 500)
    _text(artifact.url, 2_000)
    _digest(artifact.sha256)
    if artifact.metadata_sha256 is not None:
        _digest(artifact.metadata_sha256)
    if (
        not isinstance(artifact.size_bytes, int)
        or isinstance(artifact.size_bytes, bool)
        or artifact.size_bytes < 0
    ):
        raise ComfyRegistryWheelMetadataError(
            "invalid_artifact_manifest", "Wheel artifact size is invalid"
        )
    _text(artifact.compatibility_tag, 200)
    if (
        not isinstance(artifact.wheel_tags, tuple)
        or not artifact.wheel_tags
        or any(not isinstance(tag, str) or not tag for tag in artifact.wheel_tags)
    ):
        raise ComfyRegistryWheelMetadataError(
            "invalid_artifact_manifest", "Wheel artifact tags are invalid"
        )


def _text(value: object, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or _has_control(value):
        raise ComfyRegistryWheelMetadataError(
            "invalid_artifact_manifest", "Wheel artifact manifest text is invalid"
        )
    return value


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARACTERS for character in value)
    ):
        raise ComfyRegistryWheelMetadataError(
            "invalid_artifact_manifest", "Wheel artifact manifest digest is invalid"
        )
    return value.lower()


def _marker_environment(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ComfyRegistryWheelMetadataError(
            "invalid_marker_environment", "Wheel marker environment must be an object"
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
            raise ComfyRegistryWheelMetadataError(
                "invalid_marker_environment", "Wheel marker environment is invalid"
            )
        environment[key] = item
    if _MARKER_ENVIRONMENT_KEYS - environment.keys():
        raise ComfyRegistryWheelMetadataError(
            "invalid_marker_environment", "Wheel marker environment is incomplete"
        )
    try:
        Version(environment["python_full_version"])
    except InvalidVersion as exc:
        raise ComfyRegistryWheelMetadataError(
            "invalid_marker_environment", "Wheel marker Python version is invalid"
        ) from exc
    environment["extra"] = ""
    return environment


def _metadata_documents(value: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        raise ComfyRegistryWheelMetadataError(
            "invalid_core_metadata", "Wheel core metadata must be an object"
        )
    documents: dict[str, bytes] = {}
    for filename, content in value.items():
        if (
            not isinstance(filename, str)
            or not filename
            or len(filename) > 600
            or _has_control(filename)
            or not isinstance(content, bytes)
        ):
            raise ComfyRegistryWheelMetadataError(
                "invalid_core_metadata", "Wheel core metadata entry is invalid"
            )
        documents[filename] = content
    return documents


def _metadata_record(
    artifact: ComfyRegistryWheelArtifact,
    content: bytes,
) -> _MetadataRecord:
    if artifact.metadata_sha256 is None:
        raise ComfyRegistryWheelMetadataError(
            "invalid_core_metadata", "Wheel does not declare hash-bound core metadata"
        )
    if len(content) > MAX_WHEEL_CORE_METADATA_BYTES:
        raise ComfyRegistryWheelMetadataError(
            "core_metadata_too_large", "Wheel core metadata exceeds the size limit"
        )
    if not hmac.compare_digest(hashlib.sha256(content).hexdigest(), artifact.metadata_sha256):
        raise ComfyRegistryWheelMetadataError(
            "core_metadata_hash_mismatch", "Wheel core metadata hash does not match"
        )
    lines = content.splitlines()
    if (
        len(lines) > MAX_WHEEL_CORE_METADATA_LINES
        or any(len(line) > MAX_WHEEL_CORE_METADATA_LINE_BYTES for line in lines)
        or b"\x00" in content
    ):
        raise ComfyRegistryWheelMetadataError(
            "invalid_core_metadata", "Wheel core metadata structure is invalid"
        )
    message = BytesParser(policy=policy.default).parsebytes(content, headersonly=True)
    if message.defects:
        raise ComfyRegistryWheelMetadataError(
            "invalid_core_metadata", "Wheel core metadata headers are malformed"
        )
    metadata_version = _single_header(message.get_all("Metadata-Version"), "Metadata-Version")
    major = metadata_version.split(".", 1)[0]
    if major not in {"1", "2"}:
        raise ComfyRegistryWheelMetadataError(
            "unsupported_core_metadata", "Wheel core metadata version is unsupported"
        )
    name = canonicalize_name(_single_header(message.get_all("Name"), "Name"))
    try:
        version = Version(_single_header(message.get_all("Version"), "Version"))
    except InvalidVersion as exc:
        raise ComfyRegistryWheelMetadataError(
            "invalid_core_metadata", "Wheel core metadata version is invalid"
        ) from exc
    if name != artifact.name or str(version) != artifact.version:
        raise ComfyRegistryWheelMetadataError(
            "core_metadata_identity_mismatch",
            "Wheel core metadata identity does not match its artifact",
        )
    values = message.get_all("Requires-Dist", [])
    if len(values) > MAX_WHEEL_REQUIRES_DIST:
        raise ComfyRegistryWheelMetadataError(
            "too_many_transitive_requirements",
            f"Wheel {artifact.filename} declares too many dependencies",
        )
    parsed = {_requirement(value) for value in values}
    return _MetadataRecord(
        artifact,
        tuple(sorted(parsed, key=lambda item: (item.name, item.requirement))),
    )


def _single_header(values: Sequence[str] | None, label: str) -> str:
    if values is None or len(values) != 1:
        raise ComfyRegistryWheelMetadataError(
            "invalid_core_metadata", f"Wheel core metadata must declare one {label}"
        )
    value = str(values[0])
    if not value or len(value) > 1_000 or _has_control(value):
        raise ComfyRegistryWheelMetadataError(
            "invalid_core_metadata", f"Wheel core metadata {label} is invalid"
        )
    return value


def _requirement(value: object) -> _ParsedRequirement:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_WHEEL_REQUIREMENT_CHARACTERS
        or _has_control(value)
    ):
        raise ComfyRegistryWheelMetadataError(
            "invalid_transitive_requirement", "Wheel dependency declaration is invalid"
        )
    try:
        parsed = Requirement(value)
    except InvalidRequirement as exc:
        raise ComfyRegistryWheelMetadataError(
            "invalid_transitive_requirement", "Wheel dependency declaration is invalid"
        ) from exc
    if parsed.url is not None:
        raise ComfyRegistryWheelMetadataError(
            "direct_transitive_url", "Wheel dependencies cannot use direct, local, or VCS URLs"
        )
    name = canonicalize_name(parsed.name)
    extras = tuple(sorted(canonicalize_name(extra) for extra in parsed.extras))
    specifier = str(parsed.specifier)
    marker = str(parsed.marker) if parsed.marker is not None else None
    canonical: str = name
    if extras:
        canonical += f"[{','.join(extras)}]"
    canonical += specifier
    if marker:
        canonical += f"; {marker}"
    return _ParsedRequirement(name, canonical, specifier, marker, extras)


def _active_requirements(
    records: Sequence[_MetadataRecord],
    artifacts: Sequence[ComfyRegistryWheelArtifact],
    environment: Mapping[str, str],
) -> tuple[ComfyRegistryWheelMetadataRequirement, ...]:
    locked = {artifact.name for artifact in artifacts}
    requested_extras: dict[str, set[str]] = {}
    for artifact in artifacts:
        try:
            parsed_requirement = Requirement(artifact.requirement)
        except InvalidRequirement as exc:
            raise ComfyRegistryWheelMetadataError(
                "invalid_artifact_manifest",
                "Wheel artifact manifest contains an invalid requirement",
            ) from exc
        if (
            parsed_requirement.url is not None
            or canonicalize_name(parsed_requirement.name) != artifact.name
        ):
            raise ComfyRegistryWheelMetadataError(
                "invalid_artifact_manifest",
                "Wheel artifact manifest contains an invalid requirement",
            )
        requested_extras[artifact.name] = {
            canonicalize_name(extra) for extra in parsed_requirement.extras
        }
    while True:
        active = _evaluate_requirements(records, requested_extras, environment)
        changed = False
        for requirement in active:
            if requirement.name not in locked:
                continue
            target = requested_extras[requirement.name]
            before = len(target)
            target.update(requirement.extras)
            changed = changed or len(target) != before
        if sum(len(values) for values in requested_extras.values()) > MAX_WHEEL_TRANSITIVE_EXTRAS:
            raise ComfyRegistryWheelMetadataError(
                "too_many_transitive_extras", "Wheel dependency extras exceed the size limit"
            )
        if not changed:
            return active


def _evaluate_requirements(
    records: Sequence[_MetadataRecord],
    requested_extras: Mapping[str, set[str]],
    environment: Mapping[str, str],
) -> tuple[ComfyRegistryWheelMetadataRequirement, ...]:
    active: set[ComfyRegistryWheelMetadataRequirement] = set()
    for record in records:
        contexts = ("", *sorted(requested_extras.get(record.artifact.name, set())))
        for item in record.requirements:
            marker = Marker(item.marker) if item.marker else None
            if marker and not any(
                marker.evaluate(environment={**environment, "extra": extra}) for extra in contexts
            ):
                continue
            active.add(
                ComfyRegistryWheelMetadataRequirement(
                    record.artifact.name,
                    record.artifact.version,
                    item.name,
                    item.requirement,
                    item.specifier,
                    item.marker,
                    item.extras,
                )
            )
            if len(active) > MAX_WHEEL_TRANSITIVE_REQUIREMENTS:
                raise ComfyRegistryWheelMetadataError(
                    "too_many_transitive_requirements",
                    "Wheel dependency frontier exceeds the size limit",
                )
    return tuple(sorted(active, key=lambda item: (item.name, item.requirement, item.source_name)))


def _frontier(
    requirements: Sequence[ComfyRegistryWheelMetadataRequirement],
    artifacts: Sequence[ComfyRegistryWheelArtifact],
) -> tuple[
    tuple[ComfyRegistryWheelMetadataFrontier, ...],
    tuple[str, ...],
]:
    locked = {artifact.name: artifact.version for artifact in artifacts}
    grouped: dict[str, list[ComfyRegistryWheelMetadataRequirement]] = defaultdict(list)
    for requirement in requirements:
        grouped[requirement.name].append(requirement)
    frontier: list[ComfyRegistryWheelMetadataFrontier] = []
    conflicts: list[str] = []
    for name in sorted(grouped):
        items = grouped[name]
        declarations = tuple(sorted({item.requirement for item in items}))
        sources = tuple(sorted({item.source_name for item in items}))
        extras = tuple(sorted({extra for item in items for extra in item.extras}))
        locked_version = locked.get(name)
        status = "resolve"
        if locked_version is not None:
            version = Version(locked_version)
            status = (
                "satisfied"
                if all(Requirement(item.requirement).specifier.contains(version) for item in items)
                else "conflict"
            )
        elif _exact_pin_conflict(items):
            status = "conflict"
        if status == "conflict":
            conflicts.append(name)
        frontier.append(
            ComfyRegistryWheelMetadataFrontier(
                name,
                declarations,
                sources,
                extras,
                locked_version,
                status,
            )
        )
    return tuple(frontier), tuple(conflicts)


def _exact_pin_conflict(
    requirements: Sequence[ComfyRegistryWheelMetadataRequirement],
) -> bool:
    pins: set[str] = set()
    for item in requirements:
        parsed = Requirement(item.requirement)
        specifiers = tuple(parsed.specifier)
        if len(specifiers) != 1:
            continue
        specifier = specifiers[0]
        if specifier.operator not in {"==", "==="} or "*" in specifier.version:
            continue
        try:
            pins.add(str(Version(specifier.version)))
        except InvalidVersion:
            pins.add(specifier.version)
    return len(pins) > 1


def _requirement_payload(
    item: ComfyRegistryWheelMetadataRequirement,
) -> dict[str, object]:
    return {
        "source_name": item.source_name,
        "source_version": item.source_version,
        "name": item.name,
        "requirement": item.requirement,
        "specifier": item.specifier,
        "marker": item.marker,
        "extras": list(item.extras),
    }


def _frontier_payload(item: ComfyRegistryWheelMetadataFrontier) -> dict[str, object]:
    return {
        "name": item.name,
        "requirements": list(item.requirements),
        "sources": list(item.sources),
        "requested_extras": list(item.requested_extras),
        "locked_version": item.locked_version,
        "status": item.status,
    }


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


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _has_control(value: str) -> bool:
    return any(character < " " or character == "\x7f" for character in value)
