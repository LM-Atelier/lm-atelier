from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass

from packaging.utils import canonicalize_name

from .comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifact,
    ComfyRegistryWheelArtifactError,
    ComfyRegistryWheelArtifactManifest,
    build_comfy_registry_wheel_artifact_manifest,
    validate_comfy_registry_wheel_artifact_manifest,
)
from .comfy_registry_wheel_metadata import (
    ComfyRegistryWheelMetadataError,
    ComfyRegistryWheelMetadataPlan,
    plan_comfy_registry_wheel_metadata,
)
from .comfy_registry_wheel_selection import (
    ComfyRegistryWheelSelection,
    ComfyRegistryWheelSelectionError,
    validate_comfy_registry_wheel_metadata_plan,
    validate_comfy_registry_wheel_selection,
)

MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS = 64
_DIGEST_CHARACTERS = frozenset("0123456789abcdefABCDEF")


class ComfyRegistryWheelClosureError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ComfyRegistryWheelClosure:
    round_number: int
    manifest: ComfyRegistryWheelArtifactManifest
    metadata_plan: ComfyRegistryWheelMetadataPlan
    pending_projects: tuple[str, ...]
    manifest_history: tuple[str, ...]
    complete: bool
    closure_sha256: str


def plan_comfy_registry_wheel_closure(
    manifest: ComfyRegistryWheelArtifactManifest,
    metadata_documents: Mapping[str, bytes],
    *,
    marker_environment: Mapping[str, str],
) -> ComfyRegistryWheelClosure:
    """Plan the first inert dependency-closure round for an exact manifest."""
    return _plan_closure(
        manifest,
        metadata_documents,
        marker_environment,
        prior_history=(),
    )


def advance_comfy_registry_wheel_closure(
    closure: ComfyRegistryWheelClosure,
    selection: ComfyRegistryWheelSelection,
    metadata_documents: Mapping[str, bytes],
    *,
    marker_environment: Mapping[str, str],
) -> ComfyRegistryWheelClosure:
    """Extend one verified closure round and re-plan the complete locked set."""
    validate_comfy_registry_wheel_closure(closure)
    if closure.complete:
        raise ComfyRegistryWheelClosureError(
            "closure_already_complete", "Wheel dependency closure is already complete"
        )
    try:
        selected = validate_comfy_registry_wheel_selection(selection)
    except ComfyRegistryWheelSelectionError as exc:
        raise ComfyRegistryWheelClosureError(exc.code, str(exc)) from exc
    if (
        selection.artifact_manifest_sha256 != closure.manifest.manifest_sha256
        or selection.metadata_plan_sha256 != closure.metadata_plan.plan_sha256
        or selection.target_sha256 != closure.manifest.target_sha256
    ):
        raise ComfyRegistryWheelClosureError(
            "selection_source_mismatch",
            "Wheel selection does not belong to the current closure round",
        )
    selected_names = tuple(artifact.name for artifact in selected)
    if not selected or selected_names != closure.pending_projects:
        raise ComfyRegistryWheelClosureError(
            "selection_frontier_mismatch",
            "Wheel selection does not exactly cover the pending dependency frontier",
        )
    current = validate_comfy_registry_wheel_artifact_manifest(closure.manifest)
    current_names = {artifact.name for artifact in current}
    current_filenames = {artifact.filename for artifact in current}
    if any(
        artifact.name in current_names or artifact.filename in current_filenames
        for artifact in selected
    ):
        raise ComfyRegistryWheelClosureError(
            "closure_no_progress",
            "Wheel selection repeats an artifact already present in the closure",
        )
    try:
        manifest = build_comfy_registry_wheel_artifact_manifest(
            closure.manifest.declaration_sha256,
            closure.manifest.target_sha256,
            tuple(sorted((*current, *selected), key=lambda item: (item.name, item.requirement))),
        )
    except ComfyRegistryWheelArtifactError as exc:
        raise ComfyRegistryWheelClosureError(exc.code, str(exc)) from exc
    return _plan_closure(
        manifest,
        metadata_documents,
        marker_environment,
        prior_history=closure.manifest_history,
    )


def _plan_closure(
    manifest: ComfyRegistryWheelArtifactManifest,
    metadata_documents: Mapping[str, bytes],
    marker_environment: Mapping[str, str],
    *,
    prior_history: tuple[str, ...],
) -> ComfyRegistryWheelClosure:
    if len(prior_history) >= MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS + 1:
        raise ComfyRegistryWheelClosureError(
            "closure_round_limit", "Wheel dependency closure exceeds the round limit"
        )
    try:
        validate_comfy_registry_wheel_artifact_manifest(manifest)
        metadata_plan = plan_comfy_registry_wheel_metadata(
            manifest,
            metadata_documents,
            marker_environment=marker_environment,
        )
    except (ComfyRegistryWheelArtifactError, ComfyRegistryWheelMetadataError) as exc:
        raise ComfyRegistryWheelClosureError(exc.code, str(exc)) from exc
    manifest_sha256 = manifest.manifest_sha256
    history = _history(prior_history)
    if manifest_sha256 in history:
        raise ComfyRegistryWheelClosureError(
            "closure_repeated_state", "Wheel dependency closure repeated a prior manifest"
        )
    if metadata_plan.unavailable_metadata:
        raise ComfyRegistryWheelClosureError(
            "metadata_unavailable",
            "Hash-bound metadata is required to close wheel dependencies",
        )
    if metadata_plan.conflicts:
        raise ComfyRegistryWheelClosureError(
            "dependency_conflict",
            "Wheel dependency closure contains incompatible locked versions",
        )
    pending = tuple(item.name for item in metadata_plan.frontier if item.status == "resolve")
    if pending != tuple(sorted(set(pending))):
        raise ComfyRegistryWheelClosureError(
            "invalid_metadata_plan", "Wheel dependency frontier is not canonical"
        )
    complete = not pending
    if metadata_plan.resolution_required != (not complete):
        raise ComfyRegistryWheelClosureError(
            "invalid_metadata_plan", "Wheel dependency resolution state is inconsistent"
        )
    resolved_history = (*history, manifest_sha256)
    payload = {
        "version": 1,
        "round_number": len(history),
        "manifest_sha256": manifest_sha256,
        "metadata_plan_sha256": metadata_plan.plan_sha256,
        "pending_projects": list(pending),
        "manifest_history": list(resolved_history),
        "complete": complete,
    }
    return ComfyRegistryWheelClosure(
        round_number=len(history),
        manifest=manifest,
        metadata_plan=metadata_plan,
        pending_projects=pending,
        manifest_history=resolved_history,
        complete=complete,
        closure_sha256=_payload_sha256(payload),
    )


def validate_comfy_registry_wheel_closure(
    closure: ComfyRegistryWheelClosure,
) -> tuple[ComfyRegistryWheelArtifact, ...]:
    """Revalidate a frozen dependency closure and return its locked artifacts."""
    if not isinstance(closure, ComfyRegistryWheelClosure):
        raise ComfyRegistryWheelClosureError(
            "invalid_closure", "Wheel dependency closure is invalid"
        )
    try:
        artifacts = validate_comfy_registry_wheel_artifact_manifest(closure.manifest)
    except ComfyRegistryWheelArtifactError as exc:
        raise ComfyRegistryWheelClosureError(
            "invalid_closure", "Wheel dependency closure manifest is invalid"
        ) from exc
    if not isinstance(closure.metadata_plan, ComfyRegistryWheelMetadataPlan):
        raise ComfyRegistryWheelClosureError(
            "invalid_closure", "Wheel dependency closure metadata plan is invalid"
        )
    try:
        validate_comfy_registry_wheel_metadata_plan(closure.metadata_plan)
    except ComfyRegistryWheelSelectionError as exc:
        raise ComfyRegistryWheelClosureError(
            "invalid_closure", "Wheel dependency closure metadata plan is invalid"
        ) from exc
    history = _history(closure.manifest_history)
    pending = _project_names(closure.pending_projects)
    complete = not pending
    if (
        not isinstance(closure.round_number, int)
        or isinstance(closure.round_number, bool)
        or closure.round_number != len(history) - 1
        or not history
        or history[-1] != closure.manifest.manifest_sha256
        or closure.metadata_plan.artifact_manifest_sha256 != closure.manifest.manifest_sha256
        or pending
        != tuple(item.name for item in closure.metadata_plan.frontier if item.status == "resolve")
        or not isinstance(closure.complete, bool)
        or closure.complete != complete
    ):
        raise ComfyRegistryWheelClosureError(
            "invalid_closure", "Wheel dependency closure state is inconsistent"
        )
    payload = {
        "version": 1,
        "round_number": closure.round_number,
        "manifest_sha256": closure.manifest.manifest_sha256,
        "metadata_plan_sha256": closure.metadata_plan.plan_sha256,
        "pending_projects": list(pending),
        "manifest_history": list(history),
        "complete": complete,
    }
    closure_sha256 = _digest(closure.closure_sha256)
    if not hmac.compare_digest(_payload_sha256(payload), closure_sha256):
        raise ComfyRegistryWheelClosureError(
            "closure_hash_mismatch",
            "Wheel dependency closure hash does not match its contents",
        )
    return artifacts


def _history(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or len(value) > MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS + 1:
        raise ComfyRegistryWheelClosureError(
            "invalid_closure", "Wheel dependency closure history is invalid"
        )
    history = tuple(_digest(item) for item in value)
    if len(history) != len(set(history)):
        raise ComfyRegistryWheelClosureError(
            "invalid_closure", "Wheel dependency closure history contains a cycle"
        )
    return history


def _project_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ComfyRegistryWheelClosureError(
            "invalid_closure", "Wheel dependency closure frontier is invalid"
        )
    names: list[str] = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 200
            or canonicalize_name(item) != item
        ):
            raise ComfyRegistryWheelClosureError(
                "invalid_closure", "Wheel dependency closure frontier is invalid"
            )
        names.append(item)
    resolved = tuple(names)
    if resolved != tuple(sorted(set(resolved))):
        raise ComfyRegistryWheelClosureError(
            "invalid_closure", "Wheel dependency closure frontier is not canonical"
        )
    return resolved


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARACTERS for character in value)
    ):
        raise ComfyRegistryWheelClosureError(
            "invalid_closure", "Wheel dependency closure digest is invalid"
        )
    return value.lower()


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
