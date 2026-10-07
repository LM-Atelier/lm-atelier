"""Compose workflow-native LoRA edits with the existing Added/Auto stack.

This module is the single pure graph boundary for workflow LoRA edits. Native scalar edits
are re-derived under their public/private extraction authority first. The
existing Added/Auto transform is then applied to that detached graph. The
result freezes canonical pre-Added and final graph bytes plus public-safe
provenance; it never treats an already-built graph as mutation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy.orm import Session

from .auxiliary_assets import (
    LORA_GRAPH_TRANSFORM_VERSION,
    MAX_LORA_STACK_SIZE,
    resolve_lora_stack_against_graph,
)
from .comfy_workflow_packages import WorkflowPackageError, validate_bounded_workflow_json
from .lora_constraints import MAX_LORA_STRENGTH
from .models import WorkflowRevision
from .workflow_bindings import WorkflowBindingError, validate_workflow_runtime_reference
from .workflow_lora_graph import (
    WORKFLOW_LORA_GRAPH_PATCH_VERSION,
    apply_workflow_lora_overrides,
    workflow_lora_effective_api_graph,
    workflow_lora_graph_resolution_payload,
    workflow_lora_graph_resolution_sha256,
)
from .workflow_lora_overrides import (
    WORKFLOW_LORA_OVERRIDE_ORIGINS,
    WorkflowLoraOverrideCatalog,
    WorkflowLoraOverrideResolution,
    workflow_lora_override_resolution_payload,
    workflow_lora_override_resolution_sha256,
)
from .workflow_lora_slots import WorkflowLoraSlotExtractionWithPrivateTargets
from .workflow_trust import canonical_graph

WORKFLOW_LORA_COMPOSITION_VERSION = 1

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SLOT_ID = re.compile(r"^wflora_[0-9a-f]{64}$")
_FIELDS = ("enabled", "model_strength", "clip_strength")
_ORIGINS = frozenset(WORKFLOW_LORA_OVERRIDE_ORIGINS)


class WorkflowLoraCompositionError(ValueError):
    """A typed refusal to compose or consume an effective LoRA graph."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class WorkflowLoraComposition:
    """Canonical immutable evidence for native-first, Added-second composition.

    This is an integrity receipt, not a replay capability. Dispatch must rerun
    composition from the immutable revision, recorded activation and frozen
    override plan, then compare this receipt and its full composition digest.
    """

    version: int
    source_api_graph_sha256: str
    workflow_effective_graph_sha256: str
    effective_graph_sha256: str
    workflow_override_resolution_sha256: str | None
    workflow_graph_resolution_sha256: str | None
    added_transform_version: str
    workflow_override_resolution_json: str | None = field(repr=False)
    workflow_graph_resolution_json: str | None = field(repr=False)
    added_settings_json: str = field(repr=False)
    added_provenance_json: str = field(repr=False)
    workflow_effective_api_graph_json: str = field(repr=False)
    effective_api_graph_json: str = field(repr=False)


def compose_workflow_lora_graph(
    session: Session,
    revision: WorkflowRevision,
    *,
    added_loras: object,
    workflow_activation_id: str | None = None,
    override_catalog: WorkflowLoraOverrideCatalog | None = None,
    slot_extraction: WorkflowLoraSlotExtractionWithPrivateTargets | None = None,
    override_resolution: WorkflowLoraOverrideResolution | None = None,
) -> WorkflowLoraComposition:
    """Re-authorize native edits, then apply the existing Added/Auto transform."""

    source_json, source_graph, source_sha256 = _canonical_api_graph(
        revision.api_graph_json,
        label="workflow source graph",
    )
    try:
        detached_added_loras = _detached_added_loras(added_loras)
    except (ValueError, WorkflowLoraCompositionError) as exc:
        raise WorkflowLoraCompositionError(
            "invalid_added_lora_composition",
            "The Added LoRA stack cannot be applied to the selected workflow",
        ) from exc
    native_inputs = (override_catalog, slot_extraction, override_resolution)
    if any(value is not None for value in native_inputs) and not all(
        value is not None for value in native_inputs
    ):
        raise WorkflowLoraCompositionError(
            "incomplete_workflow_lora_native_authority",
            "Workflow-native LoRA composition requires one complete authority set",
        )

    workflow_override_json: str | None = None
    workflow_override_sha256: str | None = None
    workflow_graph_json_receipt: str | None = None
    workflow_graph_receipt_sha256: str | None = None
    workflow_graph = source_graph
    workflow_graph_json = source_json
    workflow_graph_sha256 = source_sha256
    if all(value is not None for value in native_inputs):
        assert override_catalog is not None
        assert slot_extraction is not None
        assert override_resolution is not None
        override_payload = workflow_lora_override_resolution_payload(override_resolution)
        target = override_payload["target"]
        assert type(target) is dict
        if (
            target.get("workflow_revision_id") != revision.id
            or target.get("workflow_definition_id") != revision.workflow_id
            or type(revision.dependency_contract_sha256) is not str
            or target.get("dependency_contract_sha256") != revision.dependency_contract_sha256
            or target.get("api_graph_sha256") != source_sha256
        ):
            raise WorkflowLoraCompositionError(
                "stale_workflow_lora_composition_target",
                "Workflow-native LoRA edits do not belong to the selected revision",
            )
        workflow_override_json = _canonical_json(override_payload)
        workflow_override_sha256 = workflow_lora_override_resolution_sha256(override_resolution)
        native = apply_workflow_lora_overrides(
            api_graph=source_graph,
            catalog=override_catalog,
            extraction=slot_extraction,
            resolution=override_resolution,
        )
        if native.source_api_graph_sha256 != source_sha256:
            raise WorkflowLoraCompositionError(
                "stale_workflow_lora_composition_source",
                "Workflow-native LoRA edits no longer match the selected revision",
            )
        workflow_graph = workflow_lora_effective_api_graph(native)
        workflow_graph_json, workflow_graph, workflow_graph_sha256 = _canonical_api_graph(
            workflow_graph,
            label="workflow-native effective graph",
        )
        if workflow_graph_sha256 != native.workflow_effective_graph_sha256:
            raise WorkflowLoraCompositionError(
                "invalid_workflow_lora_native_result",
                "Workflow-native LoRA graph bytes do not match their evidence",
            )
        native_payload = workflow_lora_graph_resolution_payload(native)
        workflow_graph_json_receipt = _canonical_json(native_payload)
        workflow_graph_receipt_sha256 = workflow_lora_graph_resolution_sha256(native)
        if native.override_resolution_sha256 != workflow_override_sha256:
            raise WorkflowLoraCompositionError(
                "invalid_workflow_lora_native_result",
                "Workflow-native LoRA graph evidence does not match its override plan",
            )

    try:
        added = resolve_lora_stack_against_graph(
            session,
            revision,
            detached_added_loras,
            base_api_graph=workflow_graph,
            workflow_activation_id=workflow_activation_id,
        )
    except ValueError as exc:
        raise WorkflowLoraCompositionError(
            "invalid_added_lora_composition",
            "The Added LoRA stack cannot be applied to the selected workflow",
        ) from exc

    final_json, _, final_sha256 = _canonical_api_graph(
        added.graph,
        label="final effective graph",
    )
    if final_sha256 != added.graph_sha256:
        raise WorkflowLoraCompositionError(
            "invalid_added_lora_result",
            "The Added LoRA graph bytes do not match their evidence",
        )
    added_settings_json = _canonical_json(added.settings)
    added_provenance_json = _canonical_json(added.provenance)
    try:
        normalized_added_settings = _validated_added_settings(added_settings_json)
        _validated_added_provenance(added_provenance_json, normalized_added_settings)
    except WorkflowLoraCompositionError as exc:
        raise WorkflowLoraCompositionError(
            "invalid_added_lora_composition",
            "The Added LoRA stack cannot be applied to the selected workflow",
        ) from exc
    result = WorkflowLoraComposition(
        version=WORKFLOW_LORA_COMPOSITION_VERSION,
        source_api_graph_sha256=source_sha256,
        workflow_effective_graph_sha256=workflow_graph_sha256,
        effective_graph_sha256=final_sha256,
        workflow_override_resolution_sha256=workflow_override_sha256,
        workflow_graph_resolution_sha256=workflow_graph_receipt_sha256,
        added_transform_version=LORA_GRAPH_TRANSFORM_VERSION,
        workflow_override_resolution_json=workflow_override_json,
        workflow_graph_resolution_json=workflow_graph_json_receipt,
        added_settings_json=added_settings_json,
        added_provenance_json=added_provenance_json,
        workflow_effective_api_graph_json=workflow_graph_json,
        effective_api_graph_json=final_json,
    )
    workflow_lora_composition_payload(result)
    return result


def workflow_lora_composition_graph(value: WorkflowLoraComposition) -> dict[str, Any]:
    """Return a fresh final graph from a valid in-process composition receipt.

    A persisted receipt must be rederived before dispatch rather than trusted
    as standalone graph-mutation authority.
    """

    workflow_lora_composition_payload(value)
    return _decoded_graph(value.effective_api_graph_json)


def workflow_lora_composition_workflow_graph(
    value: WorkflowLoraComposition,
) -> dict[str, Any]:
    """Return a fresh pre-Added workflow-effective graph."""

    workflow_lora_composition_payload(value)
    return _decoded_graph(value.workflow_effective_api_graph_json)


def workflow_lora_composition_added_settings(
    value: WorkflowLoraComposition,
) -> list[dict[str, Any]]:
    """Return a fresh normalized Added/Auto settings list."""

    workflow_lora_composition_payload(value)
    return _decoded_object_list(value.added_settings_json)


def workflow_lora_composition_added_provenance(
    value: WorkflowLoraComposition,
) -> list[dict[str, Any]]:
    """Return a fresh normalized Added/Auto provenance list."""

    workflow_lora_composition_payload(value)
    return _decoded_object_list(value.added_provenance_json)


def workflow_lora_composition_payload(
    value: WorkflowLoraComposition,
) -> dict[str, object]:
    """Return public-safe canonical provenance without graph bytes or private locators."""

    if (
        type(value) is not WorkflowLoraComposition
        or type(value.version) is not int
        or value.version != WORKFLOW_LORA_COMPOSITION_VERSION
        or not _is_digest(value.source_api_graph_sha256)
        or not _is_digest(value.workflow_effective_graph_sha256)
        or not _is_digest(value.effective_graph_sha256)
        or type(value.added_transform_version) is not str
        or value.added_transform_version != LORA_GRAPH_TRANSFORM_VERSION
        or type(value.added_settings_json) is not str
        or type(value.added_provenance_json) is not str
        or type(value.workflow_effective_api_graph_json) is not str
        or type(value.effective_api_graph_json) is not str
    ):
        raise _invalid_result()

    workflow_canonical, _, workflow_sha256 = _canonical_api_graph(
        _decoded_json(value.workflow_effective_api_graph_json),
        label="stored workflow-effective graph",
    )
    final_canonical, _, final_sha256 = _canonical_api_graph(
        _decoded_json(value.effective_api_graph_json),
        label="stored final effective graph",
    )
    if (
        workflow_canonical != value.workflow_effective_api_graph_json
        or final_canonical != value.effective_api_graph_json
        or workflow_sha256 != value.workflow_effective_graph_sha256
        or final_sha256 != value.effective_graph_sha256
    ):
        raise _invalid_result()

    settings = _validated_added_settings(value.added_settings_json)
    provenance = _validated_added_provenance(value.added_provenance_json, settings)
    native_payload: dict[str, object] | None
    override_payload: dict[str, object] | None
    if value.workflow_graph_resolution_json is None:
        if (
            value.workflow_override_resolution_json is not None
            or value.workflow_override_resolution_sha256 is not None
            or value.workflow_graph_resolution_sha256 is not None
            or value.workflow_effective_graph_sha256 != value.source_api_graph_sha256
        ):
            raise _invalid_result()
        native_payload = None
        override_payload = None
    else:
        if (
            type(value.workflow_graph_resolution_json) is not str
            or type(value.workflow_override_resolution_json) is not str
            or not _is_digest(value.workflow_graph_resolution_sha256)
            or not _is_digest(value.workflow_override_resolution_sha256)
        ):
            raise _invalid_result()
        override_payload = _validated_override_payload(value.workflow_override_resolution_json)
        native_payload = _validated_native_payload(value.workflow_graph_resolution_json)
        if (
            native_payload["source_api_graph_sha256"] != value.source_api_graph_sha256
            or native_payload["workflow_effective_graph_sha256"]
            != value.workflow_effective_graph_sha256
            or native_payload["override_resolution_sha256"]
            != value.workflow_override_resolution_sha256
            or _sha256(value.workflow_override_resolution_json)
            != value.workflow_override_resolution_sha256
            or _sha256(value.workflow_graph_resolution_json)
            != value.workflow_graph_resolution_sha256
            or not _native_receipts_match(override_payload, native_payload)
            or cast(dict[str, object], override_payload["target"])["api_graph_sha256"]
            != value.source_api_graph_sha256
        ):
            raise _invalid_result()

    return {
        "version": WORKFLOW_LORA_COMPOSITION_VERSION,
        "source_api_graph_sha256": value.source_api_graph_sha256,
        "workflow_effective_graph_sha256": value.workflow_effective_graph_sha256,
        "effective_graph_sha256": value.effective_graph_sha256,
        "workflow_override_resolution_sha256": value.workflow_override_resolution_sha256,
        "workflow_graph_resolution_sha256": value.workflow_graph_resolution_sha256,
        "workflow_native": (
            None
            if native_payload is None
            else {
                "override_resolution": override_payload,
                "graph_resolution": native_payload,
            }
        ),
        "added_transform_version": LORA_GRAPH_TRANSFORM_VERSION,
        "added_settings": settings,
        "added_provenance": provenance,
    }


def workflow_lora_composition_sha256(value: WorkflowLoraComposition) -> str:
    """Digest public-safe composition provenance."""

    return _sha256(_canonical_json(workflow_lora_composition_payload(value)))


def _canonical_api_graph(
    value: object,
    *,
    label: str,
) -> tuple[str, dict[str, Any], str]:
    _validate_exact_json(value)
    try:
        validate_bounded_workflow_json(value)
    except WorkflowPackageError as exc:
        raise WorkflowLoraCompositionError(
            "invalid_workflow_lora_composition_graph",
            f"The {label} is not bounded canonical JSON",
        ) from exc
    if type(value) is not dict:
        raise WorkflowLoraCompositionError(
            "invalid_workflow_lora_composition_graph",
            f"The {label} is not an API graph object",
        )
    graph = cast(dict[str, Any], value)
    for node_id, node in graph.items():
        if (
            type(node_id) is not str
            or not node_id
            or type(node) is not dict
            or type(node.get("class_type")) is not str
            or not node["class_type"]
            or type(node.get("inputs")) is not dict
        ):
            raise WorkflowLoraCompositionError(
                "invalid_workflow_lora_composition_graph",
                f"The {label} contains an invalid API graph node",
            )
    try:
        encoded = canonical_graph(graph)
        detached = json.loads(encoded)
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraCompositionError(
            "invalid_workflow_lora_composition_graph",
            f"The {label} cannot be canonicalized",
        ) from exc
    if type(detached) is not dict:
        raise WorkflowLoraCompositionError(
            "invalid_workflow_lora_composition_graph",
            f"The {label} is not an API graph object",
        )
    return encoded, cast(dict[str, Any], detached), _sha256(encoded)


def _validated_added_settings(encoded: str) -> list[dict[str, object]]:
    raw = _canonical_decoded_list(encoded)
    if len(raw) > MAX_LORA_STACK_SIZE:
        raise _invalid_result()
    result: list[dict[str, object]] = []
    seen_assets: set[str] = set()
    for item in raw:
        if type(item) is not dict or set(item) != {
            "asset_id",
            "model_strength",
            "clip_strength",
            "enabled",
        }:
            raise _invalid_result()
        asset_id = item.get("asset_id")
        if (
            not _bounded_text(asset_id, 200)
            or cast(str, asset_id) in seen_assets
            or type(item.get("model_strength")) is not float
            or type(item.get("clip_strength")) is not float
        ):
            raise _invalid_result()
        assert isinstance(asset_id, str)
        seen_assets.add(asset_id)
        model_strength = _strength(item.get("model_strength"))
        clip_strength = _strength(item.get("clip_strength"))
        enabled = item.get("enabled")
        if type(enabled) is not bool:
            raise _invalid_result()
        result.append(
            {
                "asset_id": asset_id,
                "model_strength": model_strength,
                "clip_strength": clip_strength,
                "enabled": enabled,
            }
        )
    return result


def _detached_added_loras(value: object) -> list[dict[str, object]]:
    _validate_exact_json(value)
    if type(value) is not list or len(value) > MAX_LORA_STACK_SIZE:
        raise ValueError("Added LoRA settings must be a bounded exact list")
    encoded = _canonical_json(value)
    detached = json.loads(encoded)
    if type(detached) is not list or any(type(item) is not dict for item in detached):
        raise ValueError("Added LoRA settings must contain exact objects")
    return cast(list[dict[str, object]], detached)


def _validated_added_provenance(
    encoded: str,
    settings: list[dict[str, object]],
) -> list[dict[str, object]]:
    raw = _canonical_decoded_list(encoded)
    if len(raw) != len(settings):
        raise _invalid_result()
    result: list[dict[str, object]] = []
    for position, (item, setting) in enumerate(zip(raw, settings, strict=True)):
        if type(item) is not dict or set(item) != {
            "asset_id",
            "model_strength",
            "clip_strength",
            "enabled",
            "position",
            "name",
            "family",
            "sha256",
            "comfy_name",
            "trigger_words",
        }:
            raise _invalid_result()
        if any(
            type(item.get(key)) is not type(setting[key]) or item.get(key) != setting[key]
            for key in setting
        ):
            raise _invalid_result()
        family = item.get("family")
        trigger_words = item.get("trigger_words")
        if (
            type(item.get("position")) is not int
            or item["position"] != position
            or not _bounded_text(item.get("name"), 1_000)
            or (family is not None and not _bounded_text(family, 1_000))
            or not _is_digest(item.get("sha256"))
            or not _bounded_text(item.get("comfy_name"), 1_000)
            or type(trigger_words) is not list
            or len(trigger_words) > 100
            or any(not _bounded_text(word, 200) for word in trigger_words)
        ):
            raise _invalid_result()
        try:
            validate_workflow_runtime_reference(item.get("comfy_name"))
        except WorkflowBindingError as exc:
            raise _invalid_result() from exc
        result.append(cast(dict[str, object], item))
    return result


def _validated_override_payload(encoded: str) -> dict[str, object]:
    raw = _canonical_decoded_object(encoded)
    if set(raw) != {"version", "target", "overrides"} or (
        type(raw.get("version")) is not int
        or raw["version"] != 1
        or type(raw.get("target")) is not dict
        or type(raw.get("overrides")) is not list
    ):
        raise _invalid_result()
    target = cast(dict[str, object], raw["target"])
    if set(target) != {
        "workflow_family_id",
        "workflow_definition_id",
        "workflow_variant_key",
        "workflow_revision_id",
        "slot_contract_version",
        "revision_scope_sha256",
        "api_graph_sha256",
        "dependency_contract_sha256",
        "activation_binding_sha256",
        "activation_witness_sha256",
    } or (
        target.get("workflow_family_id") is not None
        and not _bounded_text(target.get("workflow_family_id"), 64)
        or target.get("workflow_variant_key") is not None
        and not _bounded_text(target.get("workflow_variant_key"), 100)
        or not _bounded_text(target.get("workflow_definition_id"), 64)
        or not _bounded_text(target.get("workflow_revision_id"), 40)
        or type(target.get("slot_contract_version")) is not int
        or target["slot_contract_version"] != 1
        or any(
            not _is_digest(target.get(key))
            for key in (
                "revision_scope_sha256",
                "api_graph_sha256",
                "dependency_contract_sha256",
                "activation_binding_sha256",
                "activation_witness_sha256",
            )
        )
    ):
        raise _invalid_result()
    overrides = cast(list[object], raw["overrides"])
    if len(overrides) > 64:
        raise _invalid_result()
    seen_slots: set[str] = set()
    prior_slot = ""
    for override in overrides:
        if type(override) is not dict or set(override) != {
            "slot_id",
            "loader_contract",
            "loader_authority_sha256",
            "changes",
        }:
            raise _invalid_result()
        slot_id = override.get("slot_id")
        changes = override.get("changes")
        if (
            type(slot_id) is not str
            or _SLOT_ID.fullmatch(slot_id) is None
            or slot_id in seen_slots
            or slot_id < prior_slot
            or not _bounded_text(override.get("loader_contract"), 200)
            or not _is_digest(override.get("loader_authority_sha256"))
            or type(changes) is not dict
            or not changes
            or set(changes) - set(_FIELDS)
        ):
            raise _invalid_result()
        seen_slots.add(slot_id)
        prior_slot = slot_id
        change_map = cast(dict[str, object], changes)
        for field_name, field_value in change_map.items():
            if type(field_value) is not dict or set(field_value) != {"value", "origin"}:
                raise _invalid_result()
            normalized = cast(dict[str, object], field_value)
            if normalized.get("origin") not in _ORIGINS:
                raise _invalid_result()
            value = normalized.get("value")
            if field_name == "enabled":
                if type(value) is not bool:
                    raise _invalid_result()
            elif (
                type(value) is not float
                or not math.isfinite(value)
                or abs(value) > MAX_LORA_STRENGTH
            ):
                raise _invalid_result()
    return raw


def _native_receipts_match(
    override_payload: dict[str, object],
    graph_payload: dict[str, object],
) -> bool:
    override_items = cast(list[object], override_payload["overrides"])
    graph_items = cast(list[object], graph_payload["overrides"])
    if len(override_items) != len(graph_items):
        return False
    for override_item, graph_item in zip(override_items, graph_items, strict=True):
        assert type(override_item) is dict
        assert type(graph_item) is dict
        if any(
            override_item.get(key) != graph_item.get(key)
            for key in ("slot_id", "loader_contract", "loader_authority_sha256")
        ):
            return False
        override_changes = cast(dict[str, object], override_item["changes"])
        graph_changes = cast(list[object], graph_item["changes"])
        if len(override_changes) != len(graph_changes):
            return False
        seen_fields: set[str] = set()
        for graph_change in graph_changes:
            assert type(graph_change) is dict
            field_name = cast(str, graph_change["field"])
            override_change = override_changes.get(field_name)
            if type(override_change) is not dict or field_name in seen_fields:
                return False
            seen_fields.add(field_name)
            planned_value = override_change.get("value")
            effective_value = graph_change.get("effective_value")
            if (
                type(planned_value) is not type(effective_value)
                or planned_value != effective_value
                or override_change.get("origin") != graph_change.get("origin")
            ):
                return False
        if seen_fields != set(override_changes):
            return False
    return True


def _validated_native_payload(encoded: str) -> dict[str, object]:
    raw = _canonical_decoded_object(encoded)
    if set(raw) != {
        "version",
        "source_api_graph_sha256",
        "workflow_effective_graph_sha256",
        "override_resolution_sha256",
        "overrides",
    } or (
        type(raw.get("version")) is not int
        or raw["version"] != WORKFLOW_LORA_GRAPH_PATCH_VERSION
        or not _is_digest(raw.get("source_api_graph_sha256"))
        or not _is_digest(raw.get("workflow_effective_graph_sha256"))
        or not _is_digest(raw.get("override_resolution_sha256"))
        or type(raw.get("overrides")) is not list
    ):
        raise _invalid_result()
    overrides = cast(list[object], raw["overrides"])
    if len(overrides) > 64:
        raise _invalid_result()
    seen_slots: set[str] = set()
    prior_slot = ""
    for override in overrides:
        if type(override) is not dict or set(override) != {
            "slot_id",
            "loader_contract",
            "loader_authority_sha256",
            "asset_sha256",
            "changes",
        }:
            raise _invalid_result()
        slot_id = override.get("slot_id")
        changes = override.get("changes")
        if (
            type(slot_id) is not str
            or _SLOT_ID.fullmatch(slot_id) is None
            or slot_id in seen_slots
            or slot_id < prior_slot
            or not _bounded_text(override.get("loader_contract"), 200)
            or not _is_digest(override.get("loader_authority_sha256"))
            or not _is_digest(override.get("asset_sha256"))
            or type(changes) is not list
            or not changes
        ):
            raise _invalid_result()
        seen_slots.add(slot_id)
        prior_slot = slot_id
        seen_fields: set[str] = set()
        prior_field = -1
        for change in changes:
            if type(change) is not dict or set(change) != {
                "field",
                "authored_value",
                "effective_value",
                "origin",
            }:
                raise _invalid_result()
            field = change.get("field")
            origin = change.get("origin")
            if (
                type(field) is not str
                or field not in _FIELDS
                or field in seen_fields
                or _FIELDS.index(field) < prior_field
                or type(origin) is not str
                or origin not in _ORIGINS
            ):
                raise _invalid_result()
            seen_fields.add(field)
            prior_field = _FIELDS.index(field)
            authored = change.get("authored_value")
            effective = change.get("effective_value")
            if field == "enabled":
                if type(authored) is not bool or type(effective) is not bool:
                    raise _invalid_result()
            elif (
                _finite_number(authored) is None
                or _finite_number(effective) is None
                or abs(cast(float, _finite_number(effective))) > MAX_LORA_STRENGTH
            ):
                raise _invalid_result()
    return raw


def _canonical_decoded_list(encoded: str) -> list[object]:
    value = _decoded_json(encoded)
    if type(value) is not list or _canonical_json(value) != encoded:
        raise _invalid_result()
    return cast(list[object], value)


def _canonical_decoded_object(encoded: str) -> dict[str, object]:
    value = _decoded_json(encoded)
    if type(value) is not dict or _canonical_json(value) != encoded:
        raise _invalid_result()
    return cast(dict[str, object], value)


def _decoded_graph(encoded: str) -> dict[str, Any]:
    value = _decoded_json(encoded)
    if type(value) is not dict:
        raise _invalid_result()
    return cast(dict[str, Any], value)


def _decoded_object_list(encoded: str) -> list[dict[str, Any]]:
    value = _decoded_json(encoded)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise _invalid_result()
    return cast(list[dict[str, Any]], value)


def _decoded_json(encoded: str) -> object:
    if type(encoded) is not str:
        raise _invalid_result()
    try:
        value = json.loads(encoded)
    except (RecursionError, TypeError, ValueError) as exc:
        raise _invalid_result() from exc
    _validate_exact_json(value)
    return value


def _canonical_json(value: object) -> str:
    _validate_exact_json(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraCompositionError(
            "invalid_workflow_lora_composition_result",
            "Workflow LoRA composition contains invalid JSON",
        ) from exc


def _validate_exact_json(value: object) -> None:
    stack = [value]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if type(current) is dict:
            identity = id(current)
            if identity in seen:
                raise _invalid_result()
            seen.add(identity)
            mapping = cast(dict[object, object], current)
            if any(type(key) is not str for key in mapping):
                raise _invalid_result()
            stack.extend(mapping.values())
        elif type(current) is list:
            identity = id(current)
            if identity in seen:
                raise _invalid_result()
            seen.add(identity)
            stack.extend(cast(list[object], current))
        elif (
            current is None
            or type(current) in {str, int, bool}
            or type(current) is float
            and math.isfinite(current)
        ):
            continue
        else:
            raise _invalid_result()


def _strength(value: object) -> float:
    normalized = _finite_number(value)
    if normalized is None or abs(normalized) > MAX_LORA_STRENGTH:
        raise _invalid_result()
    return normalized


def _finite_number(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        normalized = float(cast(int | float, value))
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(normalized):
        return None
    return 0.0 if normalized == 0 else normalized


def _bounded_text(value: object, maximum: int) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= maximum
        and all(character >= " " and ord(character) != 127 for character in value)
    )


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _invalid_result() -> WorkflowLoraCompositionError:
    return WorkflowLoraCompositionError(
        "invalid_workflow_lora_composition_result",
        "Workflow LoRA composition evidence is malformed",
    )
