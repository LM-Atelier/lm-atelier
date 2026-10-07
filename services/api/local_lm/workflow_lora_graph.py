"""Apply authorized workflow-native LoRA overrides to a detached API graph.

The public envelope deliberately carries no graph locations. Mutation is
possible only when it is paired with the public and server-private products of
one exact slot-extraction pass. This module revalidates that pairing, copies
the source graph, changes only audited scalar leaves, and proves that the final
deep diff is exactly the requested allowlist.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from .comfy_package_widgets import POWER_LORA_LOADER, POWER_LORA_LOADER_CONTRACT
from .lora_constraints import MAX_LORA_STRENGTH
from .workflow_lora_overrides import (
    WORKFLOW_LORA_OVERRIDE_ORIGINS,
    ResolvedWorkflowLoraField,
    ResolvedWorkflowLoraOverride,
    WorkflowLoraOverrideCatalog,
    WorkflowLoraOverrideError,
    WorkflowLoraOverrideOrigin,
    WorkflowLoraOverrideResolution,
    WorkflowLoraOverrideTargetWitness,
    workflow_lora_override_resolution_payload,
    workflow_lora_override_resolution_sha256,
)
from .workflow_lora_slots import (
    CORE_LORA_LOADER_CONTRACT,
    CORE_MODEL_ONLY_LORA_LOADER_CONTRACT,
    WORKFLOW_LORA_SLOT_CONTRACT_VERSION,
    WorkflowLoraAssetBinding,
    WorkflowLoraPrivateEditTarget,
    WorkflowLoraSlot,
    WorkflowLoraSlotExtraction,
    WorkflowLoraSlotExtractionWithPrivateTargets,
)
from .workflow_trust import canonical_graph

WORKFLOW_LORA_GRAPH_PATCH_VERSION = 1

WorkflowLoraGraphField = Literal["enabled", "model_strength", "clip_strength"]

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SLOT_ID = re.compile(r"^wflora_[0-9a-f]{64}$")
_RGTHREE_ENTRY = re.compile(r"^lora_[1-9][0-9]*$")
_FIELDS = frozenset({"enabled", "model_strength", "clip_strength"})
_FIELD_ORDER = {"enabled": 0, "model_strength": 1, "clip_strength": 2}
_ORIGINS = frozenset(WORKFLOW_LORA_OVERRIDE_ORIGINS)


class WorkflowLoraGraphError(ValueError):
    """A typed refusal to locate or mutate a workflow LoRA graph leaf."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class AppliedWorkflowLoraField:
    field: WorkflowLoraGraphField
    authored_value: bool | float
    effective_value: bool | float
    origin: WorkflowLoraOverrideOrigin


@dataclass(frozen=True, slots=True)
class AppliedWorkflowLoraOverride:
    slot_id: str
    loader_contract: str
    loader_authority_sha256: str
    asset_sha256: str
    changes: tuple[AppliedWorkflowLoraField, ...]


@dataclass(frozen=True, slots=True)
class WorkflowLoraGraphResolution:
    """Frozen provenance plus canonical bytes for one native-patched graph."""

    version: int
    source_api_graph_sha256: str
    workflow_effective_graph_sha256: str
    override_resolution_sha256: str
    overrides: tuple[AppliedWorkflowLoraOverride, ...]
    effective_api_graph_json: str


def apply_workflow_lora_overrides(
    *,
    api_graph: Mapping[str, Any],
    catalog: WorkflowLoraOverrideCatalog,
    extraction: WorkflowLoraSlotExtractionWithPrivateTargets,
    resolution: WorkflowLoraOverrideResolution,
) -> WorkflowLoraGraphResolution:
    """Apply one resolved plan to a copied graph under exact private authority."""

    source_json, source_graph, source_sha256 = _canonical_source_graph(api_graph)
    slots, targets = _validated_authority(
        catalog=catalog,
        extraction=extraction,
        resolution=resolution,
        source_sha256=source_sha256,
    )
    effective = cast(dict[str, Any], json.loads(source_json))
    expected_changed_paths: set[tuple[object, ...]] = set()
    applied: list[AppliedWorkflowLoraOverride] = []

    for override in sorted(resolution.overrides, key=lambda item: item.slot_id):
        slot = slots[override.slot_id]
        target = targets[override.slot_id]
        changes, changed_paths = _apply_slot_override(
            graph=effective,
            slot=slot,
            target=target,
            override=override,
        )
        expected_changed_paths.update(changed_paths)
        applied.append(
            AppliedWorkflowLoraOverride(
                slot_id=override.slot_id,
                loader_contract=target.loader_contract,
                loader_authority_sha256=target.loader_authority_sha256,
                asset_sha256=target.asset_sha256,
                changes=changes,
            )
        )

    actual_changed_paths = _diff_paths(source_graph, effective)
    if actual_changed_paths != expected_changed_paths:
        raise WorkflowLoraGraphError(
            "workflow_lora_graph_diff_escape",
            "Workflow LoRA graph edits escaped their audited scalar allowlist",
        )
    effective_json = canonical_graph(effective)
    effective_sha256 = hashlib.sha256(effective_json.encode("utf-8")).hexdigest()
    result = WorkflowLoraGraphResolution(
        version=WORKFLOW_LORA_GRAPH_PATCH_VERSION,
        source_api_graph_sha256=source_sha256,
        workflow_effective_graph_sha256=effective_sha256,
        override_resolution_sha256=workflow_lora_override_resolution_sha256(resolution),
        overrides=tuple(applied),
        effective_api_graph_json=effective_json,
    )
    workflow_lora_graph_resolution_payload(result)
    return result


def workflow_lora_effective_api_graph(
    value: WorkflowLoraGraphResolution,
) -> dict[str, Any]:
    """Return a fresh exact-JSON graph after revalidating frozen result bytes."""

    workflow_lora_graph_resolution_payload(value)
    try:
        graph = json.loads(value.effective_api_graph_json)
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_graph_resolution",
            "Workflow LoRA graph result does not contain valid canonical JSON",
        ) from exc
    if type(graph) is not dict or not _plain_json_value(graph):
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_graph_resolution",
            "Workflow LoRA graph result is not an API graph object",
        )
    return cast(dict[str, Any], graph)


def workflow_lora_graph_resolution_payload(
    value: WorkflowLoraGraphResolution,
) -> dict[str, object]:
    """Return public-safe provenance, excluding graph bytes and private locators."""

    if (
        type(value) is not WorkflowLoraGraphResolution
        or type(value.version) is not int
        or value.version != WORKFLOW_LORA_GRAPH_PATCH_VERSION
        or not _is_digest(value.source_api_graph_sha256)
        or not _is_digest(value.workflow_effective_graph_sha256)
        or not _is_digest(value.override_resolution_sha256)
        or type(value.overrides) is not tuple
        or type(value.effective_api_graph_json) is not str
    ):
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_graph_resolution",
            "Workflow LoRA graph result is malformed",
        )
    try:
        graph = json.loads(value.effective_api_graph_json)
        canonical = canonical_graph(graph)
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_graph_resolution",
            "Workflow LoRA graph result does not contain valid canonical JSON",
        ) from exc
    if (
        type(graph) is not dict
        or not _plain_json_value(graph)
        or canonical != value.effective_api_graph_json
        or hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        != value.workflow_effective_graph_sha256
    ):
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_graph_resolution",
            "Workflow LoRA graph result bytes do not match their digest",
        )

    seen_slots: set[str] = set()
    overrides: list[dict[str, object]] = []
    for override in value.overrides:
        if (
            type(override) is not AppliedWorkflowLoraOverride
            or not _is_slot_id(override.slot_id)
            or type(override.loader_contract) is not str
            or not override.loader_contract
            or not _is_digest(override.loader_authority_sha256)
            or not _is_digest(override.asset_sha256)
            or type(override.changes) is not tuple
            or not override.changes
            or override.slot_id in seen_slots
        ):
            raise WorkflowLoraGraphError(
                "invalid_workflow_lora_graph_resolution",
                "Workflow LoRA graph result contains invalid slot provenance",
            )
        seen_slots.add(override.slot_id)
        seen_fields: set[str] = set()
        changes: list[dict[str, object]] = []
        for change in override.changes:
            if (
                type(change) is not AppliedWorkflowLoraField
                or type(change.field) is not str
                or change.field not in _FIELDS
                or change.field in seen_fields
                or type(change.origin) is not str
                or change.origin not in _ORIGINS
            ):
                raise WorkflowLoraGraphError(
                    "invalid_workflow_lora_graph_resolution",
                    "Workflow LoRA graph result contains invalid field provenance",
                )
            seen_fields.add(change.field)
            authored = _normalized_field_value(change.field, change.authored_value)
            effective_value = _normalized_field_value(change.field, change.effective_value)
            if (
                change.field in {"model_strength", "clip_strength"}
                and abs(cast(float, effective_value)) > MAX_LORA_STRENGTH
            ):
                raise WorkflowLoraGraphError(
                    "invalid_workflow_lora_graph_resolution",
                    "Workflow LoRA graph result contains an out-of-range override",
                )
            changes.append(
                {
                    "field": change.field,
                    "authored_value": authored,
                    "effective_value": effective_value,
                    "origin": change.origin,
                }
            )
        if [item["field"] for item in changes] != sorted(
            (cast(str, item["field"]) for item in changes),
            key=lambda field: _FIELD_ORDER[field],
        ):
            raise WorkflowLoraGraphError(
                "invalid_workflow_lora_graph_resolution",
                "Workflow LoRA graph result fields are not canonical",
            )
        overrides.append(
            {
                "slot_id": override.slot_id,
                "loader_contract": override.loader_contract,
                "loader_authority_sha256": override.loader_authority_sha256,
                "asset_sha256": override.asset_sha256,
                "changes": changes,
            }
        )
    if [item["slot_id"] for item in overrides] != sorted(
        cast(str, item["slot_id"]) for item in overrides
    ):
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_graph_resolution",
            "Workflow LoRA graph result slots are not canonical",
        )
    return {
        "version": WORKFLOW_LORA_GRAPH_PATCH_VERSION,
        "source_api_graph_sha256": value.source_api_graph_sha256,
        "workflow_effective_graph_sha256": value.workflow_effective_graph_sha256,
        "override_resolution_sha256": value.override_resolution_sha256,
        "overrides": overrides,
    }


def workflow_lora_graph_resolution_sha256(value: WorkflowLoraGraphResolution) -> str:
    """Hash only the public-safe native-patch provenance."""

    return hashlib.sha256(
        json.dumps(
            workflow_lora_graph_resolution_payload(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_source_graph(
    api_graph: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str]:
    if type(api_graph) is not dict or not _plain_json_value(api_graph):
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_graph_source",
            "Workflow LoRA graph mutation requires exact inert JSON",
        )
    try:
        encoded = canonical_graph(api_graph)
        detached = json.loads(encoded)
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_graph_source",
            "Workflow LoRA graph mutation requires canonical JSON",
        ) from exc
    if type(detached) is not dict or not _plain_json_value(detached):
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_graph_source",
            "Workflow LoRA graph mutation requires an API graph object",
        )
    return (
        encoded,
        cast(dict[str, Any], detached),
        hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )


def _validated_authority(
    *,
    catalog: WorkflowLoraOverrideCatalog,
    extraction: WorkflowLoraSlotExtractionWithPrivateTargets,
    resolution: WorkflowLoraOverrideResolution,
    source_sha256: str,
) -> tuple[dict[str, WorkflowLoraSlot], dict[str, WorkflowLoraPrivateEditTarget]]:
    try:
        workflow_lora_override_resolution_payload(resolution)
    except WorkflowLoraOverrideError as exc:
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_override_resolution",
            "Workflow LoRA graph mutation requires a valid resolved override plan",
        ) from exc
    if (
        type(catalog) is not WorkflowLoraOverrideCatalog
        or type(catalog.target) is not WorkflowLoraOverrideTargetWitness
        or type(catalog.slots) is not tuple
        or type(extraction) is not WorkflowLoraSlotExtractionWithPrivateTargets
        or type(extraction.public) is not WorkflowLoraSlotExtraction
        or type(extraction.edit_targets) is not tuple
    ):
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_private_targets",
            "Workflow LoRA graph mutation requires one typed extraction pass",
        )
    public = extraction.public
    if (
        type(public.version) is not int
        or public.version != WORKFLOW_LORA_SLOT_CONTRACT_VERSION
        or type(public.revision_scope_sha256) is not str
        or not _is_digest(public.revision_scope_sha256)
        or not _is_digest(public.api_graph_sha256)
        or not _is_digest(public.dependency_contract_sha256)
        or not _is_digest(public.activation_binding_sha256)
        or public.ordering_authority != "presentation_only"
        or type(public.slots) is not tuple
        or source_sha256 != public.api_graph_sha256
    ):
        raise WorkflowLoraGraphError(
            "stale_workflow_lora_graph_source",
            "Workflow LoRA graph bytes no longer match their slot evidence",
        )
    witness = resolution.target
    if (
        catalog.target != witness
        or catalog.slots != public.slots
        or hashlib.sha256(witness.workflow_revision_id.encode("utf-8")).hexdigest()
        != public.revision_scope_sha256
        or witness.slot_contract_version != WORKFLOW_LORA_SLOT_CONTRACT_VERSION
        or witness.revision_scope_sha256 != public.revision_scope_sha256
        or witness.api_graph_sha256 != public.api_graph_sha256
        or witness.dependency_contract_sha256 != public.dependency_contract_sha256
        or witness.activation_binding_sha256 != public.activation_binding_sha256
    ):
        raise WorkflowLoraGraphError(
            "stale_workflow_lora_private_target",
            "Workflow LoRA override evidence does not match the extraction pass",
        )

    slots: dict[str, WorkflowLoraSlot] = {}
    editable_slot_ids: set[str] = set()
    for position, slot in enumerate(public.slots):
        if not _valid_public_slot(slot, position) or slot.slot_id in slots:
            raise WorkflowLoraGraphError(
                "invalid_workflow_lora_private_targets",
                "Workflow LoRA extraction contains invalid public slot evidence",
            )
        slots[slot.slot_id] = slot
        if slot.editability == "editable":
            editable_slot_ids.add(slot.slot_id)

    targets: dict[str, WorkflowLoraPrivateEditTarget] = {}
    target_locators: set[tuple[str, str]] = set()
    authority_digests: set[str] = set()
    for target in extraction.edit_targets:
        if not _valid_private_target(target):
            raise WorkflowLoraGraphError(
                "invalid_workflow_lora_private_targets",
                "Workflow LoRA extraction contains invalid private edit evidence",
            )
        locator = (target.node_id, target.entry_locator)
        if (
            target.slot_id in targets
            or locator in target_locators
            or target.slot_id
            != _expected_slot_id(
                revision_scope=witness.workflow_revision_id,
                graph_sha256=public.api_graph_sha256,
                node_id=target.node_id,
                locator=target.entry_locator,
            )
        ):
            raise WorkflowLoraGraphError(
                "invalid_workflow_lora_private_targets",
                "Workflow LoRA extraction contains invalid private edit evidence",
            )
        target_slot = slots.get(target.slot_id)
        if (
            target_slot is None
            or target_slot.editability != "editable"
            or target_slot.asset_binding is None
            or target.source_api_graph_sha256 != public.api_graph_sha256
            or target.source_dependency_contract_sha256 != public.dependency_contract_sha256
            or target.source_activation_binding_sha256 != public.activation_binding_sha256
            or target.loader_contract != target_slot.loader_contract
            or target.loader_authority_sha256 != target_slot.loader_authority_sha256
            or target.asset_sha256 != target_slot.asset_binding.sha256
            or target.editable_fields != target_slot.editable_fields
        ):
            raise WorkflowLoraGraphError(
                "stale_workflow_lora_private_target",
                "Workflow LoRA private edit evidence no longer matches its slot",
            )
        authority_digests.add(target.source_authority_evidence_sha256)
        target_locators.add(locator)
        targets[target.slot_id] = target
    if set(targets) != editable_slot_ids or (targets and len(authority_digests) != 1):
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_private_targets",
            "Workflow LoRA private edit evidence is incomplete or contradictory",
        )
    if any(override.slot_id not in targets for override in resolution.overrides):
        raise WorkflowLoraGraphError(
            "stale_workflow_lora_private_target",
            "Workflow LoRA override has no exact private edit authority",
        )
    return slots, targets


def _expected_slot_id(
    *,
    revision_scope: str,
    graph_sha256: str,
    node_id: str,
    locator: str,
) -> str:
    """Rebind one private locator to the extractor's opaque occurrence ID."""

    payload = {
        "version": WORKFLOW_LORA_SLOT_CONTRACT_VERSION,
        "revision_scope": revision_scope,
        "api_graph_sha256": graph_sha256,
        "node_id": node_id,
        "locator": locator,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"wflora_{hashlib.sha256(encoded).hexdigest()}"


def _valid_public_slot(value: object, position: int) -> bool:
    if (
        type(value) is not WorkflowLoraSlot
        or not _is_slot_id(value.slot_id)
        or type(value.position) is not int
        or value.position != position
        or type(value.loader_type) is not str
        or not value.loader_type
        or value.editability not in {"editable", "required_locked", "detected_read_only"}
        or (
            value.read_only_reason is not None
            and (type(value.read_only_reason) is not str or not value.read_only_reason)
        )
        or type(value.editable_fields) is not tuple
        or any(type(field) is not str or field not in _FIELDS for field in value.editable_fields)
        or len(set(value.editable_fields)) != len(value.editable_fields)
        or not _valid_optional_loader_evidence(value)
        or not _valid_slot_defaults(value)
        or not _valid_optional_asset_binding(value)
    ):
        return False
    if value.editability != "editable":
        return not value.editable_fields and value.read_only_reason is not None
    binding = value.asset_binding
    return (
        type(value.loader_contract) is str
        and bool(value.loader_contract)
        and _is_digest(value.loader_authority_sha256)
        and value.read_only_reason is None
        and value.dependency_required is False
        and type(binding) is WorkflowLoraAssetBinding
        and type(value.observed_runtime_reference) is str
        and value.observed_runtime_reference == binding.runtime_reference
    )


def _valid_optional_loader_evidence(slot: WorkflowLoraSlot) -> bool:
    if slot.loader_contract is None or slot.loader_authority_sha256 is None:
        return slot.loader_contract is None and slot.loader_authority_sha256 is None
    return (
        type(slot.loader_contract) is str
        and bool(slot.loader_contract)
        and _is_digest(slot.loader_authority_sha256)
    )


def _valid_optional_asset_binding(slot: WorkflowLoraSlot) -> bool:
    binding = slot.asset_binding
    if binding is None:
        return (slot.dependency_required is None or type(slot.dependency_required) is bool) and (
            slot.observed_runtime_reference is None or type(slot.observed_runtime_reference) is str
        )
    return (
        type(binding) is WorkflowLoraAssetBinding
        and type(binding.dependency_slot) is str
        and bool(binding.dependency_slot)
        and type(binding.requirement_key) is str
        and bool(binding.requirement_key)
        and _is_digest(binding.resource_identity_sha256)
        and type(binding.runtime_reference) is str
        and bool(binding.runtime_reference)
        and _is_digest(binding.sha256)
        and type(slot.dependency_required) is bool
        and type(slot.observed_runtime_reference) is str
        and slot.observed_runtime_reference == binding.runtime_reference
    )


def _valid_slot_defaults(slot: WorkflowLoraSlot) -> bool:
    if slot.default_enabled is not None and type(slot.default_enabled) is not bool:
        return False
    for value in (slot.default_model_strength, slot.default_clip_strength):
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            return False
    return slot.strength_mode in {"separate", "coupled", "model_only", "unknown"}


def _valid_private_target(value: object) -> bool:
    return (
        type(value) is WorkflowLoraPrivateEditTarget
        and _is_slot_id(value.slot_id)
        and _is_digest(value.source_api_graph_sha256)
        and _is_digest(value.source_dependency_contract_sha256)
        and _is_digest(value.source_activation_binding_sha256)
        and _is_digest(value.source_authority_evidence_sha256)
        and _is_digest(value.loader_authority_sha256)
        and _is_digest(value.asset_sha256)
        and _bounded_private_text(value.node_id)
        and _bounded_private_text(value.entry_locator)
        and _bounded_private_text(value.loader_contract)
        and type(value.editable_fields) is tuple
        and bool(value.editable_fields)
        and all(type(field) is str and field in _FIELDS for field in value.editable_fields)
        and len(set(value.editable_fields)) == len(value.editable_fields)
    )


def _apply_slot_override(
    *,
    graph: dict[str, Any],
    slot: WorkflowLoraSlot,
    target: WorkflowLoraPrivateEditTarget,
    override: ResolvedWorkflowLoraOverride,
) -> tuple[tuple[AppliedWorkflowLoraField, ...], set[tuple[object, ...]]]:
    if (
        type(override) is not ResolvedWorkflowLoraOverride
        or override.loader_contract != target.loader_contract
        or override.loader_authority_sha256 != target.loader_authority_sha256
        or type(override.changes) is not tuple
        or not override.changes
    ):
        raise WorkflowLoraGraphError(
            "stale_workflow_lora_private_target",
            "Workflow LoRA override does not match its private edit authority",
        )
    container, leaf_names, path_prefix = _patch_container(
        graph=graph,
        slot=slot,
        target=target,
    )
    _validate_all_authored_values(
        slot=slot,
        container=container,
        leaf_names=leaf_names,
    )
    applied: list[AppliedWorkflowLoraField] = []
    changed_paths: set[tuple[object, ...]] = set()
    for change in sorted(override.changes, key=lambda item: _FIELD_ORDER[item.field]):
        if (
            type(change) is not ResolvedWorkflowLoraField
            or type(change.field) is not str
            or change.field not in target.editable_fields
            or change.field not in leaf_names
        ):
            raise WorkflowLoraGraphError(
                "unauthorized_workflow_lora_patch",
                "Workflow LoRA override names an unauthorized graph field",
            )
        leaf = leaf_names[change.field]
        authored = _normalized_field_value(change.field, container.get(leaf))
        expected_authored = _slot_authored_value(slot, change.field)
        if authored != expected_authored or type(change.origin) is not str:
            raise WorkflowLoraGraphError(
                "changed_workflow_lora_authored_value",
                "Workflow LoRA authored value no longer matches its slot evidence",
            )
        effective_value = _normalized_field_value(change.field, change.value)
        if change.field == "enabled" and slot.default_enabled is not True:
            raise WorkflowLoraGraphError(
                "workflow_lora_override_enable_unsupported",
                "Workflow LoRA overrides cannot target an authored-disabled entry",
            )
        if authored != effective_value:
            container[leaf] = effective_value
            changed_paths.add((*path_prefix, leaf))
        applied.append(
            AppliedWorkflowLoraField(
                field=change.field,
                authored_value=authored,
                effective_value=effective_value,
                origin=change.origin,
            )
        )
    return tuple(applied), changed_paths


def _validate_all_authored_values(
    *,
    slot: WorkflowLoraSlot,
    container: dict[str, Any],
    leaf_names: dict[str, str],
) -> None:
    """Prove every audited authored leaf, not only sparsely changed fields."""

    for field, leaf in leaf_names.items():
        authored = _normalized_field_value(field, container.get(leaf))
        if authored != _slot_authored_value(slot, field):
            raise WorkflowLoraGraphError(
                "changed_workflow_lora_authored_value",
                "Workflow LoRA authored values no longer match their slot evidence",
            )


def _patch_container(
    *,
    graph: dict[str, Any],
    slot: WorkflowLoraSlot,
    target: WorkflowLoraPrivateEditTarget,
) -> tuple[dict[str, Any], dict[str, str], tuple[object, ...]]:
    node = graph.get(target.node_id)
    if type(node) is not dict or set(node) - {"class_type", "inputs", "_meta"}:
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_patch_target",
            "Workflow LoRA private target no longer names an audited node",
        )
    inputs = node.get("inputs")
    if type(inputs) is not dict:
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_patch_target",
            "Workflow LoRA private target no longer names audited inputs",
        )
    if target.loader_contract == CORE_LORA_LOADER_CONTRACT:
        expected_inputs = {
            "model",
            "clip",
            "lora_name",
            "strength_model",
            "strength_clip",
        }
        expected_fields = ("model_strength", "clip_strength")
        if (
            node.get("class_type") != "LoraLoader"
            or slot.loader_type != "LoraLoader"
            or set(inputs) != expected_inputs
            or target.entry_locator != "lora_name"
            or target.editable_fields != expected_fields
            or slot.strength_mode != "separate"
            or slot.default_enabled is not True
        ):
            raise WorkflowLoraGraphError(
                "invalid_workflow_lora_patch_target",
                "Workflow LoRA core target no longer matches its audited layout",
            )
        _validate_runtime_reference(inputs.get("lora_name"), slot)
        return (
            inputs,
            {"model_strength": "strength_model", "clip_strength": "strength_clip"},
            (target.node_id, "inputs"),
        )
    if target.loader_contract == CORE_MODEL_ONLY_LORA_LOADER_CONTRACT:
        if (
            node.get("class_type") != "LoraLoaderModelOnly"
            or slot.loader_type != "LoraLoaderModelOnly"
            or set(inputs) != {"model", "lora_name", "strength_model"}
            or target.entry_locator != "lora_name"
            or target.editable_fields != ("model_strength",)
            or slot.strength_mode != "model_only"
            or slot.default_enabled is not True
            or slot.default_clip_strength is not None
        ):
            raise WorkflowLoraGraphError(
                "invalid_workflow_lora_patch_target",
                "Workflow LoRA model-only target no longer matches its audited layout",
            )
        _validate_runtime_reference(inputs.get("lora_name"), slot)
        return (
            inputs,
            {"model_strength": "strength_model"},
            (target.node_id, "inputs"),
        )
    if target.loader_contract != POWER_LORA_LOADER_CONTRACT:
        raise WorkflowLoraGraphError(
            "unauthorized_workflow_lora_patch",
            "Workflow LoRA loader contract has no graph patch adapter",
        )
    entry = inputs.get(target.entry_locator)
    if (
        node.get("class_type") != POWER_LORA_LOADER
        or slot.loader_type != POWER_LORA_LOADER
        or _RGTHREE_ENTRY.fullmatch(target.entry_locator) is None
        or type(entry) is not dict
        or not {"on", "lora", "strength"} <= set(entry)
        or set(entry) - {"on", "lora", "strength", "strengthTwo"}
        or type(entry.get("on")) is not bool
    ):
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_patch_target",
            "Workflow LoRA package target no longer matches its audited layout",
        )
    _validate_runtime_reference(entry.get("lora"), slot)
    _normalized_field_value("model_strength", entry.get("strength"))
    clip = entry.get("strengthTwo")
    rgthree_fields: tuple[str, ...]
    if clip is None:
        rgthree_fields = ("enabled", "model_strength")
        if (
            slot.strength_mode != "coupled"
            or slot.default_model_strength is None
            or slot.default_clip_strength != slot.default_model_strength
        ):
            raise WorkflowLoraGraphError(
                "invalid_workflow_lora_patch_target",
                "Workflow LoRA coupled strength evidence changed",
            )
    else:
        _normalized_field_value("clip_strength", clip)
        rgthree_fields = ("enabled", "model_strength", "clip_strength")
        if slot.strength_mode != "separate":
            raise WorkflowLoraGraphError(
                "invalid_workflow_lora_patch_target",
                "Workflow LoRA separate strength evidence changed",
            )
    if target.editable_fields != rgthree_fields:
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_patch_target",
            "Workflow LoRA package edit fields changed",
        )
    leaf_names = {"enabled": "on", "model_strength": "strength"}
    if "clip_strength" in rgthree_fields:
        leaf_names["clip_strength"] = "strengthTwo"
    return entry, leaf_names, (target.node_id, "inputs", target.entry_locator)


def _validate_runtime_reference(value: object, slot: WorkflowLoraSlot) -> None:
    binding = slot.asset_binding
    if (
        type(value) is not str
        or type(slot.observed_runtime_reference) is not str
        or type(binding) is not WorkflowLoraAssetBinding
        or value != slot.observed_runtime_reference
        or value != binding.runtime_reference
    ):
        raise WorkflowLoraGraphError(
            "changed_workflow_lora_authored_value",
            "Workflow LoRA runtime reference no longer matches its bound asset",
        )


def _slot_authored_value(
    slot: WorkflowLoraSlot,
    field: str,
) -> bool | float:
    value: bool | float | None
    if field == "enabled":
        value = slot.default_enabled
    elif field == "model_strength":
        value = slot.default_model_strength
    else:
        value = slot.default_clip_strength
    if value is None:
        raise WorkflowLoraGraphError(
            "changed_workflow_lora_authored_value",
            "Workflow LoRA slot no longer has an authored value for this field",
        )
    return _normalized_field_value(field, value)


def _normalized_field_value(field: str, value: object) -> bool | float:
    if field == "enabled":
        if type(value) is not bool:
            raise WorkflowLoraGraphError(
                "invalid_workflow_lora_patch_value",
                "Workflow LoRA enabled value must be a boolean",
            )
        return value
    if field not in {"model_strength", "clip_strength"}:
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_patch_value",
            "Workflow LoRA graph field is unsupported",
        )
    if type(value) not in {int, float}:
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_patch_value",
            "Workflow LoRA strength must be numeric",
        )
    try:
        result = float(cast(int | float, value))
    except (OverflowError, ValueError) as exc:
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_patch_value",
            "Workflow LoRA strength is invalid",
        ) from exc
    if not math.isfinite(result):
        raise WorkflowLoraGraphError(
            "invalid_workflow_lora_patch_value",
            "Workflow LoRA strength must be finite",
        )
    return 0.0 if result == 0 else result


def _diff_paths(
    before: object,
    after: object,
    path: tuple[object, ...] = (),
) -> set[tuple[object, ...]]:
    if type(before) is not type(after):
        return {path}
    if type(before) is dict:
        left = cast(dict[str, object], before)
        right = cast(dict[str, object], after)
        if set(left) != set(right):
            return {path}
        paths: set[tuple[object, ...]] = set()
        for key in sorted(left):
            paths.update(_diff_paths(left[key], right[key], (*path, key)))
        return paths
    if type(before) is list:
        left_list = cast(list[object], before)
        right_list = cast(list[object], after)
        if len(left_list) != len(right_list):
            return {path}
        paths = set()
        for index, (left_item, right_item) in enumerate(zip(left_list, right_list, strict=True)):
            paths.update(_diff_paths(left_item, right_item, (*path, index)))
        return paths
    return set() if before == after else {path}


def _plain_json_value(value: object) -> bool:
    stack = [value]
    while stack:
        current = stack.pop()
        if type(current) is dict:
            mapping = cast(dict[object, object], current)
            if any(type(key) is not str for key in mapping):
                return False
            stack.extend(mapping.values())
        elif type(current) is list:
            stack.extend(cast(list[object], current))
        elif current is None or type(current) in {str, int, float, bool}:
            if type(current) is float and not math.isfinite(current):
                return False
        else:
            return False
    return True


def _bounded_private_text(value: object) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= 1_000
        and all(character >= " " and ord(character) != 127 for character in value)
    )


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _is_slot_id(value: object) -> bool:
    return type(value) is str and _SLOT_ID.fullmatch(value) is not None
