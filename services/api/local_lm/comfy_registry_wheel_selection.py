from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifact,
    ComfyRegistryWheelArtifactError,
    build_comfy_registry_wheel_artifact_manifest,
    comfy_registry_wheel_target_sha256,
    select_comfy_registry_wheel_artifact,
)
from .comfy_registry_wheel_metadata import (
    ComfyRegistryWheelMetadataFrontier,
    ComfyRegistryWheelMetadataPlan,
    ComfyRegistryWheelMetadataRequirement,
)

MAX_REGISTRY_WHEEL_SELECTION_ARTIFACTS = 256
_DIGEST_CHARACTERS = frozenset("0123456789abcdefABCDEF")
_FRONTIER_STATUSES = frozenset({"satisfied", "resolve", "conflict"})


class ComfyRegistryWheelSelectionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ComfyRegistryWheelSelection:
    artifact_manifest_sha256: str
    metadata_plan_sha256: str
    target_sha256: str
    artifacts: tuple[ComfyRegistryWheelArtifact, ...]
    selection_sha256: str


def select_comfy_registry_wheel_versions(
    plan: ComfyRegistryWheelMetadataPlan,
    project_documents: Mapping[str, object],
    *,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
) -> ComfyRegistryWheelSelection:
    """Select exact compatible wheels for an inert transitive metadata frontier."""
    metadata_plan_sha256 = validate_comfy_registry_wheel_metadata_plan(plan)
    if plan.unavailable_metadata:
        raise ComfyRegistryWheelSelectionError(
            "metadata_unavailable",
            "Hash-bound metadata is required before selecting transitive wheels",
        )
    if plan.conflicts:
        raise ComfyRegistryWheelSelectionError(
            "dependency_conflict",
            "Conflicting transitive requirements cannot be selected",
        )
    unresolved = tuple(item for item in plan.frontier if item.status == "resolve")
    documents = _project_documents(project_documents)
    expected = {item.name for item in unresolved}
    if set(documents) != expected:
        missing = sorted(expected - set(documents))
        code = "missing_project_metadata" if missing else "unexpected_project_metadata"
        detail = missing[0] if missing else sorted(set(documents) - expected)[0]
        raise ComfyRegistryWheelSelectionError(
            code, f"PyPI project metadata does not match unresolved dependency {detail}"
        )
    if len(unresolved) > MAX_REGISTRY_WHEEL_SELECTION_ARTIFACTS:
        raise ComfyRegistryWheelSelectionError(
            "too_many_selection_artifacts",
            "Transitive wheel selection exceeds the artifact limit",
        )
    target_sha256 = comfy_registry_wheel_target_sha256(
        marker_environment,
        supported_tags,
    )
    artifacts: list[ComfyRegistryWheelArtifact] = []
    for item in unresolved:
        declaration = _frontier_declaration(item)
        try:
            artifact = select_comfy_registry_wheel_artifact(
                declaration,
                documents[item.name],
                marker_environment=marker_environment,
                supported_tags=supported_tags,
            )
        except ComfyRegistryWheelArtifactError as exc:
            raise ComfyRegistryWheelSelectionError(exc.code, str(exc)) from exc
        artifacts.append(artifact)
    selected = tuple(sorted(artifacts, key=lambda item: (item.name, item.requirement)))
    payload = {
        "version": 1,
        "artifact_manifest_sha256": plan.artifact_manifest_sha256,
        "metadata_plan_sha256": metadata_plan_sha256,
        "target_sha256": target_sha256,
        "artifacts": [_artifact_payload(artifact) for artifact in selected],
    }
    return ComfyRegistryWheelSelection(
        artifact_manifest_sha256=plan.artifact_manifest_sha256,
        metadata_plan_sha256=metadata_plan_sha256,
        target_sha256=target_sha256,
        artifacts=selected,
        selection_sha256=_payload_sha256(payload),
    )


def validate_comfy_registry_wheel_selection(
    selection: ComfyRegistryWheelSelection,
) -> tuple[ComfyRegistryWheelArtifact, ...]:
    """Revalidate a frozen target-bound transitive artifact selection."""
    if not isinstance(selection, ComfyRegistryWheelSelection):
        raise ComfyRegistryWheelSelectionError(
            "invalid_selection", "Wheel artifact selection is invalid"
        )
    artifact_manifest_sha256 = _selection_digest(selection.artifact_manifest_sha256)
    metadata_plan_sha256 = _selection_digest(selection.metadata_plan_sha256)
    target_sha256 = _selection_digest(selection.target_sha256)
    selection_sha256 = _selection_digest(selection.selection_sha256)
    if (
        not isinstance(selection.artifacts, tuple)
        or len(selection.artifacts) > MAX_REGISTRY_WHEEL_SELECTION_ARTIFACTS
    ):
        raise ComfyRegistryWheelSelectionError(
            "invalid_selection", "Wheel artifact selection is invalid"
        )
    try:
        validated = build_comfy_registry_wheel_artifact_manifest(
            metadata_plan_sha256,
            target_sha256,
            selection.artifacts,
        ).artifacts
    except ComfyRegistryWheelArtifactError as exc:
        raise ComfyRegistryWheelSelectionError(
            "invalid_selection", "Wheel artifact selection is invalid"
        ) from exc
    payload = {
        "version": 1,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "metadata_plan_sha256": metadata_plan_sha256,
        "target_sha256": target_sha256,
        "artifacts": [_artifact_payload(artifact) for artifact in validated],
    }
    if not hmac.compare_digest(_payload_sha256(payload), selection_sha256):
        raise ComfyRegistryWheelSelectionError(
            "selection_hash_mismatch",
            "Wheel artifact selection hash does not match its contents",
        )
    return validated


def validate_comfy_registry_wheel_metadata_plan(
    plan: ComfyRegistryWheelMetadataPlan,
) -> str:
    """Revalidate a frozen metadata dependency plan and return its digest."""
    if not isinstance(plan, ComfyRegistryWheelMetadataPlan):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata plan is invalid"
        )
    artifact_manifest_sha256 = _digest(plan.artifact_manifest_sha256)
    plan_sha256 = _digest(plan.plan_sha256)
    if not isinstance(plan.requirements, tuple) or not isinstance(plan.frontier, tuple):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata plan collections are invalid"
        )
    requirements = tuple(_requirement_payload(item) for item in plan.requirements)
    frontier = tuple(_frontier_payload(item) for item in plan.frontier)
    unavailable = _text_tuple(plan.unavailable_metadata)
    conflicts = _text_tuple(plan.conflicts)
    actual_conflicts = tuple(item.name for item in plan.frontier if item.status == "conflict")
    expected_resolution = bool(unavailable) or any(
        item.status != "satisfied" for item in plan.frontier
    )
    if (
        conflicts != actual_conflicts
        or not isinstance(plan.resolution_required, bool)
        or plan.resolution_required != expected_resolution
    ):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata plan state is inconsistent"
        )
    payload = {
        "version": 1,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "requirements": list(requirements),
        "frontier": list(frontier),
        "unavailable_metadata": list(unavailable),
        "conflicts": list(conflicts),
    }
    if not hmac.compare_digest(_payload_sha256(payload), plan_sha256):
        raise ComfyRegistryWheelSelectionError(
            "metadata_plan_hash_mismatch",
            "Wheel metadata plan hash does not match its contents",
        )
    return plan_sha256


def _requirement_payload(
    item: object,
) -> dict[str, object]:
    if not isinstance(item, ComfyRegistryWheelMetadataRequirement):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata requirement is invalid"
        )
    source_name = _canonical_name(item.source_name)
    source_version = _version(item.source_version)
    name = _canonical_name(item.name)
    requirement = _parsed_requirement(item.requirement, name)
    specifier = str(requirement.specifier)
    marker = str(requirement.marker) if requirement.marker is not None else None
    extras = _canonical_names(item.extras)
    if item.specifier != specifier or item.marker != marker or item.extras != extras:
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata requirement is inconsistent"
        )
    return {
        "source_name": source_name,
        "source_version": source_version,
        "name": name,
        "requirement": item.requirement,
        "specifier": item.specifier,
        "marker": item.marker,
        "extras": list(item.extras),
    }


def _frontier_payload(item: object) -> dict[str, object]:
    if not isinstance(item, ComfyRegistryWheelMetadataFrontier):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata frontier is invalid"
        )
    name = _canonical_name(item.name)
    requirements = _text_tuple(item.requirements)
    if not requirements or any(
        _parsed_requirement(requirement, name).name != name for requirement in requirements
    ):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata frontier requirements are invalid"
        )
    sources = _canonical_names(item.sources)
    extras = _canonical_names(item.requested_extras)
    locked_version = None if item.locked_version is None else _version(item.locked_version)
    if item.status not in _FRONTIER_STATUSES:
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata frontier status is invalid"
        )
    if (item.status == "satisfied") != (locked_version is not None) and not (
        item.status == "conflict" and locked_version is not None
    ):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata frontier lock state is invalid"
        )
    return {
        "name": name,
        "requirements": list(requirements),
        "sources": list(sources),
        "requested_extras": list(extras),
        "locked_version": locked_version,
        "status": item.status,
    }


def _frontier_declaration(item: ComfyRegistryWheelMetadataFrontier) -> str:
    extras = set(item.requested_extras)
    specifiers: list[str] = []
    for value in item.requirements:
        requirement = _parsed_requirement(value, item.name)
        extras.update(requirement.extras)
        if str(requirement.specifier):
            specifiers.append(str(requirement.specifier))
    try:
        combined = str(SpecifierSet(",".join(specifiers)))
    except InvalidSpecifier as exc:
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata constraints are invalid"
        ) from exc
    declaration = item.name
    if extras:
        declaration += f"[{','.join(sorted(extras))}]"
    return declaration + combined


def _project_documents(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ComfyRegistryWheelSelectionError(
            "invalid_project_metadata", "PyPI project metadata must be an object"
        )
    documents: dict[str, object] = {}
    for key, document in value.items():
        name = _canonical_name(key)
        if name in documents:
            raise ComfyRegistryWheelSelectionError(
                "invalid_project_metadata", "PyPI project metadata contains duplicates"
            )
        documents[name] = document
    return documents


def _canonical_name(value: object) -> str:
    text = _text(value)
    if canonicalize_name(text) != text:
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata package name is not canonical"
        )
    return text


def _canonical_names(value: object) -> tuple[str, ...]:
    values = _text_tuple(value)
    canonical = tuple(_canonical_name(item) for item in values)
    if canonical != tuple(sorted(set(canonical))):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata names are not canonical"
        )
    return canonical


def _parsed_requirement(value: object, expected_name: str) -> Requirement:
    text = _text(value)
    try:
        requirement = Requirement(text)
    except InvalidRequirement as exc:
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata requirement is invalid"
        ) from exc
    if requirement.url is not None or canonicalize_name(requirement.name) != expected_name:
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata requirement identity is invalid"
        )
    return requirement


def _version(value: object) -> str:
    text = _text(value)
    try:
        version = Version(text)
    except InvalidVersion as exc:
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata version is invalid"
        ) from exc
    if str(version) != text:
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata version is not canonical"
        )
    return text


def _text_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata collection is invalid"
        )
    values = tuple(_text(item) for item in value)
    if values != tuple(sorted(set(values))):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata collection is not canonical"
        )
    return values


def _text(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2_000
        or any(character < " " or character == "\x7f" for character in value)
    ):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata text is invalid"
        )
    return value


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARACTERS for character in value)
    ):
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel metadata digest is invalid"
        )
    return value.lower()


def _selection_digest(value: object) -> str:
    try:
        return _digest(value)
    except ComfyRegistryWheelSelectionError as exc:
        raise ComfyRegistryWheelSelectionError(
            "invalid_selection", "Wheel artifact selection digest is invalid"
        ) from exc


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
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise ComfyRegistryWheelSelectionError(
            "invalid_metadata_plan", "Wheel selection payload is invalid"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()
