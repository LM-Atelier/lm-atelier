"""Close mixed wheel dependencies without replacing retained source identities."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .comfy_registry_mixed_wheel_metadata import plan_comfy_registry_mixed_wheel_metadata
from .comfy_registry_runtime import (
    ComfyRegistryRuntimeDistribution,
    ComfyRegistryRuntimeError,
    canonical_comfy_registry_runtime_distributions,
    comfy_registry_runtime_distribution_payload,
)
from .comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactError,
    build_comfy_registry_wheel_artifact_manifest,
)
from .comfy_registry_wheel_closure import (
    MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS,
    ComfyRegistryWheelClosureError,
)
from .comfy_registry_wheel_inputs_v1 import (
    ComfyRegistryWheelInputError,
    ComfyRegistryWheelInputManifest,
    build_comfy_registry_wheel_input_manifest,
    validate_comfy_registry_wheel_input_manifest,
)
from .comfy_registry_wheel_metadata import (
    ComfyRegistryWheelMetadataError,
    ComfyRegistryWheelMetadataPlan,
)
from .comfy_registry_wheel_selection import (
    ComfyRegistryWheelSelection,
    ComfyRegistryWheelSelectionError,
    validate_comfy_registry_wheel_metadata_plan,
    validate_comfy_registry_wheel_selection,
)


@dataclass(frozen=True)
class ComfyRegistryMixedWheelClosure:
    manifest: ComfyRegistryWheelInputManifest
    metadata_plan: ComfyRegistryWheelMetadataPlan
    runtime_distributions: tuple[ComfyRegistryRuntimeDistribution, ...]
    manifest_history: tuple[str, ...]
    round_number: int
    pending_projects: tuple[str, ...]
    complete: bool
    closure_sha256: str


def plan_comfy_registry_mixed_wheel_closure(
    manifest: ComfyRegistryWheelInputManifest,
    metadata_documents: Mapping[str, bytes],
    *,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
    runtime_distributions: (Mapping[str, str] | Sequence[ComfyRegistryRuntimeDistribution]) = (),
) -> ComfyRegistryMixedWheelClosure:
    """Start a target-bound closure over downloaded and reviewed local wheels."""
    return _plan(
        manifest,
        metadata_documents,
        marker_environment=marker_environment,
        supported_tags=supported_tags,
        runtime_distributions=runtime_distributions,
        history=(),
    )


def advance_comfy_registry_mixed_wheel_closure(
    closure: ComfyRegistryMixedWheelClosure,
    selection: ComfyRegistryWheelSelection,
    metadata_documents: Mapping[str, bytes],
    *,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
) -> ComfyRegistryMixedWheelClosure:
    """Add the exact unresolved frontier and re-evaluate all retained metadata."""
    inputs = validate_comfy_registry_mixed_wheel_closure(closure)
    if closure.complete:
        raise ComfyRegistryWheelClosureError(
            "closure_already_complete", "Wheel dependency closure is already complete"
        )
    try:
        selected = validate_comfy_registry_wheel_selection(selection)
        if (
            selection.artifact_manifest_sha256 != inputs.manifest_sha256
            or selection.metadata_plan_sha256 != closure.metadata_plan.plan_sha256
            or selection.target_sha256 != inputs.remote.target_sha256
        ):
            raise ComfyRegistryWheelClosureError(
                "selection_source_mismatch", "Wheel selection belongs to another closure round"
            )
        if not selected or tuple(item.name for item in selected) != closure.pending_projects:
            raise ComfyRegistryWheelClosureError(
                "selection_frontier_mismatch", "Wheel selection does not cover the pending frontier"
            )
        names = {item.name for item in inputs.remote.artifacts} | {
            item.name for item in inputs.reviewed_local
        }
        filenames = {item.filename for item in inputs.remote.artifacts} | {
            item.filename for item in inputs.reviewed_local
        }
        if any(item.name in names or item.filename in filenames for item in selected):
            raise ComfyRegistryWheelClosureError(
                "closure_no_progress", "Wheel selection repeats an existing input"
            )
        remote = build_comfy_registry_wheel_artifact_manifest(
            inputs.remote.declaration_sha256,
            inputs.remote.target_sha256,
            tuple(
                sorted(
                    (*inputs.remote.artifacts, *selected),
                    key=lambda item: (item.name, item.requirement),
                )
            ),
        )
        manifest = build_comfy_registry_wheel_input_manifest(
            inputs.declaration_sha256,
            remote,
            inputs.reviewed_local,
        )
    except (
        ComfyRegistryWheelSelectionError,
        ComfyRegistryWheelArtifactError,
        ComfyRegistryWheelInputError,
    ) as exc:
        raise ComfyRegistryWheelClosureError(exc.code, str(exc)) from exc
    return _plan(
        manifest,
        metadata_documents,
        marker_environment=marker_environment,
        supported_tags=supported_tags,
        runtime_distributions=closure.runtime_distributions,
        history=closure.manifest_history,
    )


def _plan(
    manifest: ComfyRegistryWheelInputManifest,
    documents: Mapping[str, bytes],
    *,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
    runtime_distributions: Mapping[str, str] | Sequence[ComfyRegistryRuntimeDistribution],
    history: tuple[str, ...],
) -> ComfyRegistryMixedWheelClosure:
    if len(history) >= MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS + 1:
        raise ComfyRegistryWheelClosureError(
            "closure_round_limit", "Wheel dependency closure exceeds the round limit"
        )
    try:
        runtime = canonical_comfy_registry_runtime_distributions(runtime_distributions)
        plan = plan_comfy_registry_mixed_wheel_metadata(
            manifest,
            documents,
            marker_environment=marker_environment,
            supported_tags=supported_tags,
            runtime_distributions=runtime,
        )
    except (
        ComfyRegistryRuntimeError,
        ComfyRegistryWheelInputError,
        ComfyRegistryWheelArtifactError,
        ComfyRegistryWheelMetadataError,
    ) as exc:
        raise ComfyRegistryWheelClosureError(exc.code, str(exc)) from exc
    if plan.unavailable_metadata:
        raise ComfyRegistryWheelClosureError(
            "metadata_unavailable", "Hash-bound metadata is required to close wheel dependencies"
        )
    if plan.conflicts:
        raise ComfyRegistryWheelClosureError(
            "dependency_conflict", "Wheel dependencies conflict for " + ", ".join(plan.conflicts)
        )
    if manifest.manifest_sha256 in history:
        raise ComfyRegistryWheelClosureError(
            "closure_repeated_state", "Wheel dependency closure repeated a prior manifest"
        )
    pending = tuple(item.name for item in plan.frontier if item.status == "resolve")
    closure = ComfyRegistryMixedWheelClosure(
        manifest,
        plan,
        runtime,
        (*history, manifest.manifest_sha256),
        len(history),
        pending,
        not pending,
        "",
    )
    closure = replace(closure, closure_sha256=_hash(_payload(closure)))
    validate_comfy_registry_mixed_wheel_closure(closure)
    return closure


def validate_comfy_registry_mixed_wheel_closure(
    closure: ComfyRegistryMixedWheelClosure,
) -> ComfyRegistryWheelInputManifest:
    """Validate a frozen closure's identities and consistency, without renewing reviews."""
    if not isinstance(closure, ComfyRegistryMixedWheelClosure):
        _invalid()
    try:
        inputs = validate_comfy_registry_wheel_input_manifest(closure.manifest)
        runtime = canonical_comfy_registry_runtime_distributions(closure.runtime_distributions)
        validate_comfy_registry_wheel_metadata_plan(closure.metadata_plan)
    except (
        ComfyRegistryWheelInputError,
        ComfyRegistryRuntimeError,
        ComfyRegistryWheelSelectionError,
    ) as exc:
        raise ComfyRegistryWheelClosureError("invalid_closure", "Wheel closure is invalid") from exc
    history = closure.manifest_history
    if (
        not isinstance(history, tuple)
        or not 1 <= len(history) <= MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS + 1
        or any(not _is_digest(item) for item in history)
        or len(history) != len(set(history))
        or type(closure.round_number) is not int
        or closure.round_number != len(history) - 1
        or history[-1] != inputs.manifest_sha256
        or runtime != closure.runtime_distributions
        or not isinstance(closure.runtime_distributions, tuple)
        or closure.metadata_plan.artifact_manifest_sha256 != inputs.manifest_sha256
        or closure.metadata_plan.unavailable_metadata
        or closure.metadata_plan.conflicts
    ):
        _invalid()
    pending = tuple(
        item.name for item in closure.metadata_plan.frontier if item.status == "resolve"
    )
    if (
        pending != tuple(sorted(set(pending)))
        or closure.pending_projects != pending
        or type(closure.complete) is not bool
        or closure.complete != (not pending)
        or closure.metadata_plan.resolution_required != bool(pending)
        or not _is_digest(closure.closure_sha256)
    ):
        _invalid()
    if not hmac.compare_digest(_hash(_payload(closure)), closure.closure_sha256):
        raise ComfyRegistryWheelClosureError(
            "closure_hash_mismatch", "Wheel dependency closure hash does not match its contents"
        )
    return inputs


def _invalid() -> None:
    raise ComfyRegistryWheelClosureError("invalid_closure", "Wheel closure is inconsistent")


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _payload(closure: ComfyRegistryMixedWheelClosure) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "mixed-wheel-inputs",
        "manifest_sha256": closure.manifest.manifest_sha256,
        "metadata_plan_sha256": closure.metadata_plan.plan_sha256,
        "runtime_distributions": comfy_registry_runtime_distribution_payload(
            closure.runtime_distributions
        ),
        "manifest_history": list(closure.manifest_history),
        "round_number": closure.round_number,
        "pending_projects": list(closure.pending_projects),
        "complete": closure.complete,
    }


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
