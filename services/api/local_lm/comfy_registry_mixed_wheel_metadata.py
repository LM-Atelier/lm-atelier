"""Resolve one dependency frontier while preserving mixed wheel provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from .comfy_registry_wheel_artifacts import comfy_registry_wheel_target_sha256
from .comfy_registry_wheel_inputs_v1 import (
    ComfyRegistryWheelInputError,
    ComfyRegistryWheelInputManifest,
    validate_comfy_registry_wheel_input_manifest,
)
from .comfy_registry_wheel_metadata import (
    ComfyRegistryWheelMetadataInput,
    ComfyRegistryWheelMetadataPlan,
    plan_comfy_registry_metadata_inputs,
)


def plan_comfy_registry_mixed_wheel_metadata(
    manifest: ComfyRegistryWheelInputManifest,
    metadata_documents: Mapping[str, bytes],
    *,
    marker_environment: Mapping[str, str],
    supported_tags: Sequence[str],
    runtime_distributions: (Mapping[str, str] | Sequence[ComfyRegistryRuntimeDistribution]) = (),
) -> ComfyRegistryWheelMetadataPlan:
    """Evaluate both sources together; planning does not renew a local source review."""
    inputs = validate_comfy_registry_wheel_input_manifest(manifest)
    target = comfy_registry_wheel_target_sha256(marker_environment, supported_tags)
    if target != inputs.remote.target_sha256:
        raise ComfyRegistryWheelInputError("wheel_input_target_mismatch")
    records = [
        ComfyRegistryWheelMetadataInput(
            item.name, item.version, item.requirement, item.filename, item.metadata_sha256
        )
        for item in inputs.remote.artifacts
    ]
    for item in inputs.reviewed_local:
        source = Requirement(item.declaration)
        extras = sorted(canonicalize_name(extra) for extra in source.extras)
        requirement = item.name
        if extras:
            requirement += f"[{','.join(extras)}]"
        requirement += f"=={item.version}"
        records.append(
            ComfyRegistryWheelMetadataInput(
                item.name, item.version, requirement, item.filename, item.metadata_sha256
            )
        )
    return plan_comfy_registry_metadata_inputs(
        inputs.manifest_sha256,
        tuple(sorted(records, key=lambda item: (item.name, item.requirement))),
        metadata_documents,
        marker_environment=marker_environment,
        runtime_distributions=runtime_distributions,
    )
