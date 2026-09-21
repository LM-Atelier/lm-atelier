"""Fetch transitive remote metadata while retaining reviewed local wheel identities."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping, Sequence

from .comfy_registry_closure_driver import (
    MAX_REGISTRY_WHEEL_METADATA_TOTAL_BYTES,
    ComfyRegistryWheelClosureDriverError,
    RegistryClosureProgress,
    RegistryMetadataFetcher,
    RegistryProjectFetcher,
    _driver_error,
    _fetch_metadata,
    _fetch_projects,
    _publish_progress,
)
from .comfy_registry_mixed_wheel_closure import (
    ComfyRegistryMixedWheelClosure,
    advance_comfy_registry_mixed_wheel_closure,
    plan_comfy_registry_mixed_wheel_closure,
)
from .comfy_registry_runtime import (
    ComfyRegistryRuntimeDistribution,
    ComfyRegistryRuntimeError,
    canonical_comfy_registry_runtime_distributions,
)
from .comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactError,
    build_comfy_registry_wheel_artifact_manifest,
    comfy_registry_wheel_target_sha256,
)
from .comfy_registry_wheel_closure import (
    MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS,
    ComfyRegistryWheelClosureError,
)
from .comfy_registry_wheel_inputs_v1 import (
    ComfyRegistryWheelInputError,
    ComfyRegistryWheelInputManifest,
    validate_comfy_registry_wheel_input_manifest,
)
from .comfy_registry_wheel_metadata import MAX_WHEEL_CORE_METADATA_BYTES
from .comfy_registry_wheel_selection import (
    ComfyRegistryWheelSelectionError,
    select_comfy_registry_wheel_versions,
)


async def drive_comfy_registry_mixed_wheel_closure(
    manifest: ComfyRegistryWheelInputManifest,
    local_metadata: Mapping[str, bytes],
    *,
    project_fetcher: RegistryProjectFetcher,
    metadata_fetcher: RegistryMetadataFetcher,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
    runtime_distributions: Mapping[str, str] | Sequence[ComfyRegistryRuntimeDistribution] = (),
    progress: RegistryClosureProgress | None = None,
) -> ComfyRegistryMixedWheelClosure:
    """Close prepared roots; callers must renew local source reviews before installation."""
    environment = dict(marker_environment)
    tags = tuple(supported_tags)
    try:
        inputs = validate_comfy_registry_wheel_input_manifest(manifest)
        target = comfy_registry_wheel_target_sha256(environment, tags)
        if target != inputs.remote.target_sha256:
            raise ComfyRegistryWheelInputError("wheel_input_target_mismatch")
        runtime = canonical_comfy_registry_runtime_distributions(runtime_distributions)
    except (
        ComfyRegistryWheelInputError,
        ComfyRegistryWheelArtifactError,
        ComfyRegistryRuntimeError,
    ) as exc:
        raise _driver_error(exc) from exc
    documents: dict[str, bytes] = {}
    _merge_metadata(documents, local_metadata, {item.filename for item in inputs.reviewed_local})
    for item in inputs.reviewed_local:
        if not hmac.compare_digest(
            hashlib.sha256(documents[item.filename]).hexdigest(), item.metadata_sha256
        ):
            raise ComfyRegistryWheelClosureDriverError(
                "core_metadata_hash_mismatch", "Reviewed wheel core metadata hash does not match"
            )
    remote_metadata = await _fetch_metadata(
        metadata_fetcher, inputs.remote, round_number=0, progress=progress
    )
    _merge_metadata(documents, remote_metadata, {item.filename for item in inputs.remote.artifacts})
    await _publish_progress(progress, "validating_closure", 0, ())
    try:
        closure = plan_comfy_registry_mixed_wheel_closure(
            inputs,
            documents,
            marker_environment=environment,
            supported_tags=tags,
            runtime_distributions=runtime,
        )
    except ComfyRegistryWheelClosureError as exc:
        raise _driver_error(exc) from exc
    for _ in range(MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS):
        if closure.complete:
            return closure
        closure = await _advance_round(
            closure,
            documents,
            project_fetcher=project_fetcher,
            metadata_fetcher=metadata_fetcher,
            marker_environment=environment,
            supported_tags=tags,
            progress=progress,
        )
    if closure.complete:
        return closure
    raise ComfyRegistryWheelClosureDriverError(
        "closure_round_limit", "Wheel dependency closure exceeds the round limit"
    )


async def _advance_round(
    closure: ComfyRegistryMixedWheelClosure,
    documents: dict[str, bytes],
    *,
    project_fetcher: RegistryProjectFetcher,
    metadata_fetcher: RegistryMetadataFetcher,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
    progress: RegistryClosureProgress | None,
) -> ComfyRegistryMixedWheelClosure:
    round_number = closure.round_number + 1
    projects = closure.pending_projects
    await _publish_progress(progress, "fetching_projects", round_number, projects)
    project_documents = await _fetch_projects(project_fetcher, projects)
    await _publish_progress(progress, "selecting_wheels", round_number, projects)
    try:
        selection = select_comfy_registry_wheel_versions(
            closure.metadata_plan,
            project_documents,
            marker_environment=marker_environment,
            supported_tags=supported_tags,
        )
        selected_manifest = build_comfy_registry_wheel_artifact_manifest(
            selection.selection_sha256, selection.target_sha256, selection.artifacts
        )
    except (ComfyRegistryWheelSelectionError, ComfyRegistryWheelArtifactError) as exc:
        raise _driver_error(exc) from exc
    selected_metadata = await _fetch_metadata(
        metadata_fetcher, selected_manifest, round_number=round_number, progress=progress
    )
    _merge_metadata(documents, selected_metadata, {item.filename for item in selection.artifacts})
    await _publish_progress(progress, "validating_closure", round_number, projects)
    try:
        return advance_comfy_registry_mixed_wheel_closure(
            closure,
            selection,
            documents,
            marker_environment=marker_environment,
            supported_tags=supported_tags,
        )
    except ComfyRegistryWheelClosureError as exc:
        raise _driver_error(exc) from exc


def _merge_metadata(
    documents: dict[str, bytes], additions: Mapping[str, bytes], expected: set[str]
) -> None:
    if not isinstance(additions, Mapping) or set(additions) != expected:
        raise ComfyRegistryWheelClosureDriverError(
            "core_metadata_set_mismatch", "Wheel core metadata does not match the requested inputs"
        )
    if set(documents) & expected:
        raise ComfyRegistryWheelClosureDriverError(
            "closure_no_progress",
            "Wheel metadata repeats an artifact already present in the closure",
        )
    total = sum(len(content) for content in documents.values())
    for content in additions.values():
        if not isinstance(content, bytes):
            raise ComfyRegistryWheelClosureDriverError(
                "invalid_core_metadata", "Wheel core metadata must contain bytes"
            )
        if len(content) > MAX_WHEEL_CORE_METADATA_BYTES:
            raise ComfyRegistryWheelClosureDriverError(
                "core_metadata_too_large", "Wheel core metadata exceeds the size limit"
            )
        total += len(content)
        if total > MAX_REGISTRY_WHEEL_METADATA_TOTAL_BYTES:
            raise ComfyRegistryWheelClosureDriverError(
                "metadata_total_too_large", "Wheel core metadata exceeds the aggregate size limit"
            )
    documents.update(additions)
