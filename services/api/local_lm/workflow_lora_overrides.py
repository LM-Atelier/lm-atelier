"""Parse and resolve bounded workflow-native LoRA override envelopes.

The browser names only an exact revision/activation witness and an opaque slot.
It never supplies a graph location.  This module deliberately stops before graph
mutation: it normalizes portable settings, binds them to an exact slot catalog,
and resolves field precedence with explicit origins.  A later graph adapter may
consume that result only after independently deriving private patch targets.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal, cast

from .lora_constraints import MAX_LORA_STRENGTH
from .workflow_lora_slots import (
    MAX_WORKFLOW_LORA_SLOTS,
    WORKFLOW_LORA_SLOT_CONTRACT_VERSION,
    WorkflowLoraAssetBinding,
    WorkflowLoraSlot,
)

WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION = 1
MAX_WORKFLOW_LORA_OVERRIDE_TARGETS = 16
MAX_WORKFLOW_LORA_OVERRIDES_PER_TARGET = MAX_WORKFLOW_LORA_SLOTS
MAX_WORKFLOW_LORA_OVERRIDES_TOTAL = 256
MAX_WORKFLOW_LORA_OVERRIDE_CANONICAL_BYTES = 256 * 1024

WorkflowLoraOverrideField = Literal["enabled", "model_strength", "clip_strength"]
WorkflowLoraOverrideOrigin = Literal[
    "profile_request",
    "default_preset",
    "project_preset",
    "project",
    "chat_preset",
    "chat",
    "turn_preset",
    "turn",
]
WorkflowLoraInactiveTargetReason = Literal["different_workflow", "different_revision"]

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SLOT_ID = re.compile(r"^wflora_[0-9a-f]{64}$")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RESOLUTION_FIELDS = frozenset({"version", "target", "overrides"})
_RESOLUTION_SLOT_FIELDS = frozenset(
    {"slot_id", "loader_contract", "loader_authority_sha256", "changes"}
)
_RESOLUTION_CHANGE_FIELDS = frozenset({"value", "origin"})
_TARGET_FIELDS = frozenset(
    {
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
        "overrides",
    }
)
_SLOT_OVERRIDE_FIELDS = frozenset(
    {"slot_id", "loader_contract", "loader_authority_sha256", "changes"}
)
_CHANGE_FIELDS = frozenset({"enabled", "model_strength", "clip_strength"})
_FIELD_ORDER: dict[str, int] = {
    "enabled": 0,
    "model_strength": 1,
    "clip_strength": 2,
}
_ORIGIN_ORDER: dict[str, int] = {
    "profile_request": 0,
    "default_preset": 1,
    "project_preset": 2,
    "project": 3,
    "chat_preset": 4,
    "chat": 5,
    "turn_preset": 6,
    "turn": 7,
}
WORKFLOW_LORA_OVERRIDE_ORIGINS: tuple[WorkflowLoraOverrideOrigin, ...] = cast(
    tuple[WorkflowLoraOverrideOrigin, ...], tuple(_ORIGIN_ORDER)
)


class WorkflowLoraOverrideError(ValueError):
    """A typed refusal to parse or resolve workflow LoRA overrides."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class WorkflowLoraOverrideTargetWitness:
    """Public-safe identity of one exact editable workflow activation."""

    workflow_family_id: str | None
    workflow_definition_id: str
    workflow_variant_key: str | None
    workflow_revision_id: str
    slot_contract_version: int
    revision_scope_sha256: str
    api_graph_sha256: str
    dependency_contract_sha256: str
    activation_binding_sha256: str
    activation_witness_sha256: str


@dataclass(frozen=True, slots=True)
class WorkflowLoraFieldChange:
    field: WorkflowLoraOverrideField
    value: bool | float


@dataclass(frozen=True, slots=True)
class WorkflowLoraSlotOverride:
    slot_id: str
    loader_contract: str
    loader_authority_sha256: str
    changes: tuple[WorkflowLoraFieldChange, ...]


@dataclass(frozen=True, slots=True)
class WorkflowLoraOverrideTarget:
    witness: WorkflowLoraOverrideTargetWitness
    overrides: tuple[WorkflowLoraSlotOverride, ...]


@dataclass(frozen=True, slots=True)
class WorkflowLoraOverrides:
    version: int
    targets: tuple[WorkflowLoraOverrideTarget, ...]


@dataclass(frozen=True, slots=True)
class WorkflowLoraOverrideCatalog:
    """Exact public slot catalog supplied by the server evidence pass.

    This contains no graph location and therefore grants no mutation authority.
    The caller must pair the target only with slots from that target's exact
    public evidence pass. Because opaque slot ids cannot prove that pairing by
    themselves, a later graph adapter must still derive and match the private
    patch target under the same witness before applying this module's result.
    """

    target: WorkflowLoraOverrideTargetWitness
    slots: tuple[WorkflowLoraSlot, ...]


@dataclass(frozen=True, slots=True)
class WorkflowLoraOverrideLayer:
    origin: WorkflowLoraOverrideOrigin
    overrides: WorkflowLoraOverrides


@dataclass(frozen=True, slots=True)
class ResolvedWorkflowLoraField:
    field: WorkflowLoraOverrideField
    value: bool | float
    origin: WorkflowLoraOverrideOrigin


@dataclass(frozen=True, slots=True)
class ResolvedWorkflowLoraOverride:
    slot_id: str
    loader_contract: str
    loader_authority_sha256: str
    changes: tuple[ResolvedWorkflowLoraField, ...]


@dataclass(frozen=True, slots=True)
class InactiveWorkflowLoraOverrideTarget:
    origin: WorkflowLoraOverrideOrigin
    reason: WorkflowLoraInactiveTargetReason
    target: WorkflowLoraOverrideTargetWitness


@dataclass(frozen=True, slots=True)
class WorkflowLoraOverrideResolution:
    """An unapplied field plan, never authority to locate or mutate a graph."""

    version: int
    target: WorkflowLoraOverrideTargetWitness
    overrides: tuple[ResolvedWorkflowLoraOverride, ...]
    inactive_targets: tuple[InactiveWorkflowLoraOverrideTarget, ...]


def parse_workflow_lora_overrides(value: object) -> WorkflowLoraOverrides:
    """Return one canonical, detached override envelope.

    Input must use exact JSON built-ins.  Canonical target and slot ordering is
    semantic-free; duplicates are rejected before sorting rather than collapsed.
    """

    root = _exact_object(value, "Workflow LoRA overrides")
    _exact_keys(root, {"version", "targets"}, "Workflow LoRA overrides")
    _exact_version(root.get("version"), WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION)
    raw_targets = _exact_array(root.get("targets"), "Workflow LoRA override targets")
    if len(raw_targets) > MAX_WORKFLOW_LORA_OVERRIDE_TARGETS:
        raise WorkflowLoraOverrideError(
            "too_many_workflow_lora_override_targets",
            f"Workflow LoRA overrides may target at most {MAX_WORKFLOW_LORA_OVERRIDE_TARGETS} "
            "workflow activations",
        )

    targets: list[WorkflowLoraOverrideTarget] = []
    target_keys: set[tuple[object, ...]] = set()
    override_count = 0
    for raw_target in raw_targets:
        target = _parse_target(raw_target)
        key = _target_identity_key(target.witness)
        if key in target_keys:
            raise WorkflowLoraOverrideError(
                "duplicate_workflow_lora_override_target",
                "One workflow activation is targeted more than once",
            )
        target_keys.add(key)
        override_count += len(target.overrides)
        if override_count > MAX_WORKFLOW_LORA_OVERRIDES_TOTAL:
            raise WorkflowLoraOverrideError(
                "too_many_workflow_lora_overrides",
                f"Workflow LoRA overrides may contain at most "
                f"{MAX_WORKFLOW_LORA_OVERRIDES_TOTAL} slot overrides",
            )
        targets.append(target)

    targets.sort(key=lambda item: _canonical_bytes(_witness_payload(item.witness)))
    envelope = WorkflowLoraOverrides(
        version=WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION,
        targets=tuple(targets),
    )
    _canonical_envelope_bytes(envelope)
    return envelope


def workflow_lora_overrides_payload(value: WorkflowLoraOverrides) -> dict[str, object]:
    """Return the canonical JSON payload, revalidating manually built values."""

    if type(value) is not WorkflowLoraOverrides:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_overrides",
            "Workflow LoRA overrides are not a typed envelope",
        )
    canonical = parse_workflow_lora_overrides(_envelope_payload(value))
    return _envelope_payload(canonical)


def workflow_lora_overrides_sha256(value: WorkflowLoraOverrides) -> str:
    """Hash the normalized portable envelope with deterministic JSON."""

    payload = workflow_lora_overrides_payload(value)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def parse_workflow_lora_override_resolution(
    value: object,
) -> WorkflowLoraOverrideResolution:
    """Parse one persisted public effective plan without granting authority.

    The public payload deliberately has no inactive-target evidence or graph
    locator. Replay must still derive the current catalog and private graph
    targets independently before it can use the returned field plan.
    """

    root = _exact_object(value, "Workflow LoRA override resolution")
    _exact_keys(root, _RESOLUTION_FIELDS, "Workflow LoRA override resolution")
    _exact_version(root.get("version"), WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION)
    raw_target = _exact_object(
        root.get("target"),
        "Workflow LoRA override resolution target",
    )
    _exact_keys(
        raw_target,
        _TARGET_FIELDS - {"overrides"},
        "Workflow LoRA override resolution target",
    )
    target = _parse_witness(raw_target)
    raw_overrides = _exact_array(
        root.get("overrides"),
        "Workflow LoRA resolved overrides",
    )
    if len(raw_overrides) > MAX_WORKFLOW_LORA_SLOTS:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_resolution",
            "Workflow LoRA override resolution has too many slots",
        )

    overrides: list[ResolvedWorkflowLoraOverride] = []
    seen_slots: set[str] = set()
    for raw_override in raw_overrides:
        raw = _exact_object(raw_override, "Workflow LoRA resolved slot override")
        _exact_keys(raw, _RESOLUTION_SLOT_FIELDS, "Workflow LoRA resolved slot override")
        slot_id = _validated_resolution_slot_id(raw.get("slot_id"))
        raw_changes = _exact_object(
            raw.get("changes"),
            "Workflow LoRA resolved changes",
        )
        _reject_unknown_keys(raw_changes, _CHANGE_FIELDS)
        if not raw_changes:
            raise WorkflowLoraOverrideError(
                "invalid_workflow_lora_override_resolution",
                "Workflow LoRA resolved changes cannot be empty",
            )

        values: dict[str, object] = {}
        origins: dict[str, WorkflowLoraOverrideOrigin] = {}
        for field, raw_change in raw_changes.items():
            change = _exact_object(raw_change, "Workflow LoRA resolved field")
            _exact_keys(change, _RESOLUTION_CHANGE_FIELDS, "Workflow LoRA resolved field")
            origin = change.get("origin")
            if type(origin) is not str or origin not in _ORIGIN_ORDER:
                raise WorkflowLoraOverrideError(
                    "invalid_workflow_lora_override_resolution",
                    "Workflow LoRA resolved field has an invalid origin",
                )
            values[field] = change.get("value")
            origins[field] = cast(WorkflowLoraOverrideOrigin, origin)

        normalized = _parse_slot_override(
            {
                "slot_id": slot_id,
                "loader_contract": raw.get("loader_contract"),
                "loader_authority_sha256": raw.get("loader_authority_sha256"),
                "changes": values,
            }
        )
        if normalized.slot_id in seen_slots:
            raise WorkflowLoraOverrideError(
                "duplicate_workflow_lora_override_slot",
                "Workflow LoRA override resolution repeats a slot",
            )
        seen_slots.add(normalized.slot_id)
        overrides.append(
            ResolvedWorkflowLoraOverride(
                slot_id=normalized.slot_id,
                loader_contract=normalized.loader_contract,
                loader_authority_sha256=normalized.loader_authority_sha256,
                changes=tuple(
                    ResolvedWorkflowLoraField(
                        field=change.field,
                        value=change.value,
                        origin=origins[change.field],
                    )
                    for change in normalized.changes
                ),
            )
        )

    resolution = WorkflowLoraOverrideResolution(
        version=WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION,
        target=target,
        overrides=tuple(sorted(overrides, key=lambda item: item.slot_id)),
        inactive_targets=(),
    )
    workflow_lora_override_resolution_payload(resolution)
    return resolution


def workflow_lora_override_resolution_payload(
    value: WorkflowLoraOverrideResolution,
) -> dict[str, object]:
    """Return the canonical unapplied plan, including each winning field origin.

    Inactive targets are intentionally not part of the effective-plan identity:
    they are retained UI evidence but cannot affect the selected workflow.
    """

    if type(value) is not WorkflowLoraOverrideResolution:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_resolution",
            "Workflow LoRA override resolution is not typed",
        )
    _exact_version(value.version, WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION)
    target = _parse_witness(_witness_payload(value.target))
    _validate_inactive_targets(value.inactive_targets, target)
    if type(value.overrides) is not tuple or len(value.overrides) > MAX_WORKFLOW_LORA_SLOTS:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_resolution",
            "Workflow LoRA override resolution has invalid slots",
        )
    seen_slots: set[str] = set()
    overrides: list[dict[str, object]] = []
    for override in value.overrides:
        if (
            type(override) is not ResolvedWorkflowLoraOverride
            or type(override.changes) is not tuple
            or not override.changes
            or len(override.changes) > len(_CHANGE_FIELDS)
        ):
            raise WorkflowLoraOverrideError(
                "invalid_workflow_lora_override_resolution",
                "Workflow LoRA override resolution has invalid changes",
            )
        slot_id = _validated_resolution_slot_id(override.slot_id)
        seen_fields: set[str] = set()
        origins: dict[str, WorkflowLoraOverrideOrigin] = {}
        raw_changes: dict[str, bool | float] = {}
        for change in override.changes:
            if (
                type(change) is not ResolvedWorkflowLoraField
                or type(change.field) is not str
                or change.field not in _CHANGE_FIELDS
                or change.field in seen_fields
                or type(change.origin) is not str
                or change.origin not in _ORIGIN_ORDER
            ):
                raise WorkflowLoraOverrideError(
                    "invalid_workflow_lora_override_resolution",
                    "Workflow LoRA override resolution has an invalid field",
                )
            seen_fields.add(change.field)
            origins[change.field] = change.origin
            raw_changes[change.field] = change.value
        normalized = _parse_slot_override(
            {
                "slot_id": slot_id,
                "loader_contract": override.loader_contract,
                "loader_authority_sha256": override.loader_authority_sha256,
                "changes": raw_changes,
            }
        )
        if normalized.slot_id in seen_slots:
            raise WorkflowLoraOverrideError(
                "duplicate_workflow_lora_override_slot",
                "Workflow LoRA override resolution repeats a slot",
            )
        seen_slots.add(normalized.slot_id)
        overrides.append(
            {
                "slot_id": normalized.slot_id,
                "loader_contract": normalized.loader_contract,
                "loader_authority_sha256": normalized.loader_authority_sha256,
                "changes": {
                    change.field: {
                        "value": change.value,
                        "origin": origins[change.field],
                    }
                    for change in normalized.changes
                },
            }
        )
    overrides.sort(key=lambda item: cast(str, item["slot_id"]))
    payload: dict[str, object] = {
        "version": WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION,
        "target": _witness_payload(target),
        "overrides": overrides,
    }
    if len(_canonical_bytes(payload)) > MAX_WORKFLOW_LORA_OVERRIDE_CANONICAL_BYTES:
        raise WorkflowLoraOverrideError(
            "workflow_lora_overrides_too_large",
            "Workflow LoRA override resolution exceeds the canonical payload limit",
        )
    return payload


def workflow_lora_override_resolution_sha256(
    value: WorkflowLoraOverrideResolution,
) -> str:
    """Hash the exact effective fields and their winning settings origins."""

    return hashlib.sha256(
        _canonical_bytes(workflow_lora_override_resolution_payload(value))
    ).hexdigest()


def resolve_workflow_lora_override_layers(
    *,
    catalog: WorkflowLoraOverrideCatalog,
    layers: tuple[WorkflowLoraOverrideLayer, ...],
) -> WorkflowLoraOverrideResolution:
    """Resolve exact slot fields by the closed settings-layer precedence.

    Targets for another workflow or revision are retained as inactive evidence.
    A target naming the current definition/revision with any different witness
    is stale and refuses.  This function returns values and origins only; it
    never locates or changes a graph field. Its output is unapplied until a
    later adapter independently derives an exact private target for every slot
    under this witness; absence or mismatch there must refuse graph mutation.
    """

    target, slots = _validated_catalog(catalog)
    ordered_layers = _validated_layers(layers)
    winners: dict[
        tuple[str, WorkflowLoraOverrideField],
        tuple[bool | float, WorkflowLoraOverrideOrigin, str, str],
    ] = {}
    inactive: list[InactiveWorkflowLoraOverrideTarget] = []

    for layer in ordered_layers:
        for override_target in layer.overrides.targets:
            candidate = override_target.witness
            if candidate.workflow_definition_id != target.workflow_definition_id:
                inactive.append(
                    InactiveWorkflowLoraOverrideTarget(
                        layer.origin,
                        "different_workflow",
                        candidate,
                    )
                )
                continue
            if candidate.workflow_revision_id != target.workflow_revision_id:
                inactive.append(
                    InactiveWorkflowLoraOverrideTarget(
                        layer.origin,
                        "different_revision",
                        candidate,
                    )
                )
                continue
            if candidate != target:
                raise WorkflowLoraOverrideError(
                    "stale_workflow_lora_override_target",
                    "Workflow LoRA overrides do not match the current revision evidence",
                )
            for slot_override in override_target.overrides:
                slot = slots.get(slot_override.slot_id)
                if slot is None:
                    raise WorkflowLoraOverrideError(
                        "stale_workflow_lora_override_slot",
                        "A workflow LoRA override names a slot outside the current revision",
                    )
                _validate_slot_override(slot, slot_override)
                for change in slot_override.changes:
                    winners[(slot.slot_id, change.field)] = (
                        change.value,
                        layer.origin,
                        slot_override.loader_contract,
                        slot_override.loader_authority_sha256,
                    )

    grouped: dict[str, list[ResolvedWorkflowLoraField]] = {}
    evidence: dict[str, tuple[str, str]] = {}
    for (slot_id, field), (value, origin, loader_contract, authority) in winners.items():
        grouped.setdefault(slot_id, []).append(ResolvedWorkflowLoraField(field, value, origin))
        evidence[slot_id] = (loader_contract, authority)
    resolved: list[ResolvedWorkflowLoraOverride] = []
    for slot_id in sorted(grouped):
        fields = sorted(grouped[slot_id], key=lambda item: _FIELD_ORDER[item.field])
        loader_contract, authority = evidence[slot_id]
        resolved.append(
            ResolvedWorkflowLoraOverride(
                slot_id=slot_id,
                loader_contract=loader_contract,
                loader_authority_sha256=authority,
                changes=tuple(fields),
            )
        )
    return WorkflowLoraOverrideResolution(
        version=WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION,
        target=target,
        overrides=tuple(resolved),
        inactive_targets=tuple(inactive),
    )


def _parse_target(value: object) -> WorkflowLoraOverrideTarget:
    raw = _exact_object(value, "Workflow LoRA override target")
    _exact_keys(raw, _TARGET_FIELDS, "Workflow LoRA override target")
    witness = _parse_witness(raw)
    raw_overrides = _exact_array(raw.get("overrides"), "Workflow LoRA slot overrides")
    if not raw_overrides:
        raise WorkflowLoraOverrideError(
            "empty_workflow_lora_override_target",
            "A workflow LoRA override target must contain at least one slot override",
        )
    if len(raw_overrides) > MAX_WORKFLOW_LORA_OVERRIDES_PER_TARGET:
        raise WorkflowLoraOverrideError(
            "too_many_workflow_lora_overrides_for_target",
            f"One workflow activation may override at most "
            f"{MAX_WORKFLOW_LORA_OVERRIDES_PER_TARGET} LoRA slots",
        )
    overrides: list[WorkflowLoraSlotOverride] = []
    seen: set[str] = set()
    for raw_override in raw_overrides:
        override = _parse_slot_override(raw_override)
        if override.slot_id in seen:
            raise WorkflowLoraOverrideError(
                "duplicate_workflow_lora_override_slot",
                "One workflow LoRA slot is overridden more than once in a target",
            )
        seen.add(override.slot_id)
        overrides.append(override)
    overrides.sort(key=lambda item: item.slot_id)
    return WorkflowLoraOverrideTarget(witness=witness, overrides=tuple(overrides))


def _parse_witness(raw: dict[str, object]) -> WorkflowLoraOverrideTargetWitness:
    family_id = _nullable_token(
        raw.get("workflow_family_id"),
        "workflow family id",
        maximum=64,
    )
    definition_id = _bounded_token(
        raw.get("workflow_definition_id"),
        "workflow definition id",
        maximum=64,
    )
    variant_key = _nullable_token(
        raw.get("workflow_variant_key"),
        "workflow variant key",
        maximum=100,
    )
    revision_id = _bounded_token(
        raw.get("workflow_revision_id"),
        "workflow revision id",
        maximum=40,
    )
    _exact_version(raw.get("slot_contract_version"), WORKFLOW_LORA_SLOT_CONTRACT_VERSION)
    return WorkflowLoraOverrideTargetWitness(
        workflow_family_id=family_id,
        workflow_definition_id=definition_id,
        workflow_variant_key=variant_key,
        workflow_revision_id=revision_id,
        slot_contract_version=WORKFLOW_LORA_SLOT_CONTRACT_VERSION,
        revision_scope_sha256=_digest(raw.get("revision_scope_sha256"), "revision scope"),
        api_graph_sha256=_digest(raw.get("api_graph_sha256"), "API graph"),
        dependency_contract_sha256=_digest(
            raw.get("dependency_contract_sha256"),
            "dependency contract",
        ),
        activation_binding_sha256=_digest(
            raw.get("activation_binding_sha256"),
            "activation binding",
        ),
        activation_witness_sha256=_digest(
            raw.get("activation_witness_sha256"),
            "activation witness",
        ),
    )


def _parse_slot_override(value: object) -> WorkflowLoraSlotOverride:
    raw = _exact_object(value, "Workflow LoRA slot override")
    _exact_keys(raw, _SLOT_OVERRIDE_FIELDS, "Workflow LoRA slot override")
    slot_id = raw.get("slot_id")
    if type(slot_id) is not str or _SLOT_ID.fullmatch(slot_id) is None:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_slot",
            "Workflow LoRA override slot id is invalid",
        )
    loader_contract = _bounded_token(
        raw.get("loader_contract"),
        "workflow LoRA loader contract",
        maximum=200,
    )
    loader_authority = _digest(
        raw.get("loader_authority_sha256"),
        "workflow LoRA loader authority",
    )
    changes = _exact_object(raw.get("changes"), "Workflow LoRA override changes")
    _reject_unknown_keys(changes, _CHANGE_FIELDS)
    if not changes:
        raise WorkflowLoraOverrideError(
            "empty_workflow_lora_override_changes",
            "Workflow LoRA override changes cannot be empty",
        )
    normalized: list[WorkflowLoraFieldChange] = []
    for field in sorted(changes, key=_FIELD_ORDER.__getitem__):
        if field == "enabled":
            enabled = changes[field]
            if type(enabled) is not bool:
                raise WorkflowLoraOverrideError(
                    "invalid_workflow_lora_override_enabled",
                    "Workflow LoRA enabled override must be a boolean",
                )
            normalized.append(WorkflowLoraFieldChange("enabled", enabled))
        else:
            normalized.append(
                WorkflowLoraFieldChange(
                    cast(WorkflowLoraOverrideField, field),
                    _strength(changes[field], field),
                )
            )
    return WorkflowLoraSlotOverride(
        slot_id=slot_id,
        loader_contract=loader_contract,
        loader_authority_sha256=loader_authority,
        changes=tuple(normalized),
    )


def _validated_resolution_slot_id(value: object) -> str:
    if type(value) is not str or _SLOT_ID.fullmatch(value) is None:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_resolution",
            "Workflow LoRA override resolution has invalid slots",
        )
    return value


def _validated_catalog(
    catalog: WorkflowLoraOverrideCatalog,
) -> tuple[WorkflowLoraOverrideTargetWitness, dict[str, WorkflowLoraSlot]]:
    if type(catalog) is not WorkflowLoraOverrideCatalog:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_catalog",
            "Workflow LoRA override catalog is not typed",
        )
    target = _parse_witness(_witness_payload(catalog.target))
    if type(catalog.slots) is not tuple or len(catalog.slots) > MAX_WORKFLOW_LORA_SLOTS:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_catalog",
            "Workflow LoRA override catalog has invalid slots",
        )
    slots: dict[str, WorkflowLoraSlot] = {}
    positions: set[int] = set()
    for slot in catalog.slots:
        _validate_catalog_slot(slot)
        if slot.slot_id in slots:
            raise WorkflowLoraOverrideError(
                "duplicate_workflow_lora_override_catalog_slot",
                "Workflow LoRA override catalog repeats a slot",
            )
        if slot.position in positions:
            raise WorkflowLoraOverrideError(
                "invalid_workflow_lora_override_catalog",
                "Workflow LoRA override catalog repeats a presentation position",
            )
        positions.add(slot.position)
        slots[slot.slot_id] = slot
    return target, slots


def _validate_catalog_slot(value: object) -> None:
    if type(value) is not WorkflowLoraSlot:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_catalog",
            "Workflow LoRA override catalog has a non-exact public slot",
        )
    slot = value
    if (
        type(slot.slot_id) is not str
        or _SLOT_ID.fullmatch(slot.slot_id) is None
        or type(slot.position) is not int
        or not 0 <= slot.position < MAX_WORKFLOW_LORA_SLOTS
        or not _catalog_text(slot.loader_type, maximum=200)
        or (
            slot.loader_contract is not None
            and not _catalog_text(slot.loader_contract, maximum=200)
        )
        or (
            slot.loader_authority_sha256 is not None
            and (
                type(slot.loader_authority_sha256) is not str
                or _DIGEST.fullmatch(slot.loader_authority_sha256) is None
            )
        )
        or type(slot.editability) is not str
        or slot.editability not in {"editable", "required_locked", "detected_read_only"}
        or (
            slot.read_only_reason is not None
            and not _catalog_text(slot.read_only_reason, maximum=100)
        )
        or (slot.dependency_required is not None and type(slot.dependency_required) is not bool)
        or (
            slot.observed_runtime_reference is not None
            and not _catalog_text(slot.observed_runtime_reference, maximum=1_000)
        )
        or (slot.default_enabled is not None and type(slot.default_enabled) is not bool)
        or not _catalog_strength(slot.default_model_strength)
        or not _catalog_strength(slot.default_clip_strength)
        or type(slot.strength_mode) is not str
        or slot.strength_mode not in {"separate", "coupled", "model_only", "unknown"}
        or type(slot.editable_fields) is not tuple
        or len(slot.editable_fields) > len(_CHANGE_FIELDS)
        or any(
            type(field) is not str or field not in _CHANGE_FIELDS for field in slot.editable_fields
        )
        or len(set(slot.editable_fields)) != len(slot.editable_fields)
        or tuple(sorted(slot.editable_fields, key=_FIELD_ORDER.__getitem__)) != slot.editable_fields
    ):
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_catalog",
            "Workflow LoRA override catalog has an invalid public slot",
        )
    binding = slot.asset_binding
    if binding is not None and (
        type(binding) is not WorkflowLoraAssetBinding
        or not _catalog_text(binding.dependency_slot, maximum=100)
        or not _catalog_text(binding.requirement_key, maximum=100)
        or type(binding.resource_identity_sha256) is not str
        or _DIGEST.fullmatch(binding.resource_identity_sha256) is None
        or not _catalog_text(binding.runtime_reference, maximum=1_000)
        or type(binding.sha256) is not str
        or _DIGEST.fullmatch(binding.sha256) is None
    ):
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_catalog",
            "Workflow LoRA override catalog has an invalid asset binding",
        )
    if slot.editability == "editable":
        if (
            slot.loader_contract is None
            or slot.loader_authority_sha256 is None
            or binding is None
            or slot.dependency_required is not False
            or slot.read_only_reason is not None
            or not slot.editable_fields
        ):
            raise WorkflowLoraOverrideError(
                "invalid_workflow_lora_override_catalog",
                "Workflow LoRA override catalog has an unaudited editable slot",
            )
    elif slot.editable_fields:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_catalog",
            "A non-editable workflow LoRA slot cannot advertise editable fields",
        )


def _validate_inactive_targets(
    value: object,
    active_target: WorkflowLoraOverrideTargetWitness,
) -> None:
    maximum = len(_ORIGIN_ORDER) * MAX_WORKFLOW_LORA_OVERRIDE_TARGETS
    if type(value) is not tuple or len(value) > maximum:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_resolution",
            "Workflow LoRA override resolution has invalid inactive targets",
        )
    for item in value:
        if (
            type(item) is not InactiveWorkflowLoraOverrideTarget
            or type(item.origin) is not str
            or item.origin not in _ORIGIN_ORDER
            or type(item.reason) is not str
            or item.reason not in {"different_workflow", "different_revision"}
        ):
            raise WorkflowLoraOverrideError(
                "invalid_workflow_lora_override_resolution",
                "Workflow LoRA override resolution has invalid inactive evidence",
            )
        candidate = _parse_witness(_witness_payload(item.target))
        if item.reason == "different_workflow":
            valid_reason = candidate.workflow_definition_id != active_target.workflow_definition_id
        else:
            valid_reason = (
                candidate.workflow_definition_id == active_target.workflow_definition_id
                and candidate.workflow_revision_id != active_target.workflow_revision_id
            )
        if not valid_reason:
            raise WorkflowLoraOverrideError(
                "invalid_workflow_lora_override_resolution",
                "Workflow LoRA override inactive reason contradicts its target",
            )


def _validated_layers(
    layers: tuple[WorkflowLoraOverrideLayer, ...],
) -> tuple[WorkflowLoraOverrideLayer, ...]:
    if type(layers) is not tuple:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_layers",
            "Workflow LoRA override layers must be a tuple",
        )
    validated: list[WorkflowLoraOverrideLayer] = []
    seen: set[str] = set()
    for layer in layers:
        if (
            type(layer) is not WorkflowLoraOverrideLayer
            or type(layer.origin) is not str
            or layer.origin not in _ORIGIN_ORDER
            or type(layer.overrides) is not WorkflowLoraOverrides
        ):
            raise WorkflowLoraOverrideError(
                "invalid_workflow_lora_override_layer",
                "Workflow LoRA override layer has an invalid origin",
            )
        if layer.origin in seen:
            raise WorkflowLoraOverrideError(
                "duplicate_workflow_lora_override_layer",
                f"Workflow LoRA override origin {layer.origin} occurs more than once",
            )
        seen.add(layer.origin)
        validated.append(
            WorkflowLoraOverrideLayer(
                layer.origin,
                parse_workflow_lora_overrides(_envelope_payload(layer.overrides)),
            )
        )
    validated.sort(key=lambda item: _ORIGIN_ORDER[item.origin])
    return tuple(validated)


def _validate_slot_override(slot: WorkflowLoraSlot, override: WorkflowLoraSlotOverride) -> None:
    if (
        slot.loader_contract != override.loader_contract
        or slot.loader_authority_sha256 != override.loader_authority_sha256
    ):
        raise WorkflowLoraOverrideError(
            "stale_workflow_lora_override_slot",
            "Workflow LoRA override loader evidence no longer matches its slot",
        )
    if (
        slot.editability != "editable"
        or slot.asset_binding is None
        or slot.dependency_required is not False
    ):
        raise WorkflowLoraOverrideError(
            "workflow_lora_override_locked",
            "Workflow LoRA slot is not editable under its current binding",
        )
    editable = set(slot.editable_fields)
    for change in override.changes:
        if change.field not in editable:
            raise WorkflowLoraOverrideError(
                "unsupported_workflow_lora_override_field",
                f"Workflow LoRA slot does not authorize {change.field}",
            )
        if change.field == "enabled":
            if slot.default_enabled is not True:
                raise WorkflowLoraOverrideError(
                    "workflow_lora_override_enable_unsupported",
                    "An authored-disabled workflow LoRA cannot be enabled here",
                )
        elif change.field == "model_strength" and slot.default_model_strength is None:
            raise WorkflowLoraOverrideError(
                "unsupported_workflow_lora_override_field",
                "Workflow LoRA slot has no authored model strength",
            )
        elif change.field == "clip_strength" and slot.default_clip_strength is None:
            raise WorkflowLoraOverrideError(
                "unsupported_workflow_lora_override_field",
                "Workflow LoRA slot has no authored CLIP strength",
            )


def _exact_object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_overrides",
            f"{label} must be a JSON object",
        )
    raw = cast(dict[object, object], value)
    if any(type(key) is not str for key in raw):
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_overrides",
            f"{label} keys must be strings",
        )
    return cast(dict[str, object], raw)


def _exact_array(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_overrides",
            f"{label} must be a JSON array",
        )
    return cast(list[object], value)


def _exact_keys(
    value: dict[str, object],
    expected: set[str] | frozenset[str],
    _label: str,
) -> None:
    if set(value) != expected:
        _raise_invalid_payload_fields()


def _reject_unknown_keys(
    value: dict[str, object],
    allowed: set[str] | frozenset[str],
) -> None:
    if not set(value).issubset(allowed):
        raise WorkflowLoraOverrideError(
            "unsupported_workflow_lora_override_field",
            "Workflow LoRA override payload has unsupported or missing fields",
        )


def _raise_invalid_payload_fields() -> None:
    raise WorkflowLoraOverrideError(
        "invalid_workflow_lora_overrides",
        "Workflow LoRA override payload has unsupported or missing fields",
    )


def _exact_version(value: object, expected: int) -> None:
    if type(value) is not int or value != expected:
        raise WorkflowLoraOverrideError(
            "unsupported_workflow_lora_override_version",
            f"Workflow LoRA override contract version must be {expected}",
        )


def _bounded_text(value: object, label: str, *, maximum: int) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or any(character < " " or ord(character) == 127 for character in value)
    ):
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_identity",
            f"{label} is invalid",
        )
    return value


def _bounded_token(value: object, label: str, *, maximum: int) -> str:
    text = _bounded_text(value, label, maximum=maximum)
    if _SAFE_TOKEN.fullmatch(text) is None:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_identity",
            f"{label} is invalid",
        )
    return text


def _nullable_token(value: object, label: str, *, maximum: int) -> str | None:
    if value is None:
        return None
    return _bounded_token(value, label, maximum=maximum)


def _catalog_text(value: object, *, maximum: int) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= maximum
        and all(character >= " " and ord(character) != 127 for character in value)
    )


def _catalog_strength(value: object) -> bool:
    if value is None:
        return True
    if type(value) is not float:
        return False
    return math.isfinite(value)


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_identity",
            f"{label} digest is invalid",
        )
    return value


def _strength(value: object, field: str) -> float:
    if type(value) not in {int, float}:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_strength",
            f"Workflow LoRA {field} override must be a number",
        )
    try:
        normalized = float(cast(int | float, value))
    except (OverflowError, ValueError) as exc:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_strength",
            f"Workflow LoRA {field} override is outside the supported range",
        ) from exc
    if not math.isfinite(normalized) or abs(normalized) > MAX_LORA_STRENGTH:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_strength",
            f"Workflow LoRA {field} override must be between "
            f"-{MAX_LORA_STRENGTH:g} and {MAX_LORA_STRENGTH:g}",
        )
    return 0.0 if normalized == 0 else normalized


def _target_identity_key(witness: WorkflowLoraOverrideTargetWitness) -> tuple[object, ...]:
    return (
        witness.workflow_family_id,
        witness.workflow_definition_id,
        witness.workflow_variant_key,
        witness.workflow_revision_id,
        witness.slot_contract_version,
        witness.revision_scope_sha256,
        witness.api_graph_sha256,
        witness.dependency_contract_sha256,
        witness.activation_binding_sha256,
        witness.activation_witness_sha256,
    )


def _witness_payload(witness: WorkflowLoraOverrideTargetWitness) -> dict[str, object]:
    if type(witness) is not WorkflowLoraOverrideTargetWitness:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_override_identity",
            "Workflow LoRA override target witness is not typed",
        )
    return {
        "workflow_family_id": witness.workflow_family_id,
        "workflow_definition_id": witness.workflow_definition_id,
        "workflow_variant_key": witness.workflow_variant_key,
        "workflow_revision_id": witness.workflow_revision_id,
        "slot_contract_version": witness.slot_contract_version,
        "revision_scope_sha256": witness.revision_scope_sha256,
        "api_graph_sha256": witness.api_graph_sha256,
        "dependency_contract_sha256": witness.dependency_contract_sha256,
        "activation_binding_sha256": witness.activation_binding_sha256,
        "activation_witness_sha256": witness.activation_witness_sha256,
    }


def _change_payload(change: WorkflowLoraFieldChange) -> tuple[str, bool | float]:
    if (
        type(change) is not WorkflowLoraFieldChange
        or type(change.field) is not str
        or change.field not in _CHANGE_FIELDS
    ):
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_overrides",
            "Workflow LoRA override change is not typed",
        )
    return change.field, change.value


def _slot_override_payload(override: WorkflowLoraSlotOverride) -> dict[str, object]:
    if type(override) is not WorkflowLoraSlotOverride or type(override.changes) is not tuple:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_overrides",
            "Workflow LoRA slot override is not typed",
        )
    if not override.changes or len(override.changes) > len(_CHANGE_FIELDS):
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_overrides",
            "Workflow LoRA slot override has invalid changes",
        )
    changes: dict[str, bool | float] = {}
    for change in override.changes:
        field, value = _change_payload(change)
        if field in changes:
            raise WorkflowLoraOverrideError(
                "duplicate_workflow_lora_override_field",
                "Workflow LoRA slot override repeats a field",
            )
        changes[field] = value
    return {
        "slot_id": override.slot_id,
        "loader_contract": override.loader_contract,
        "loader_authority_sha256": override.loader_authority_sha256,
        "changes": changes,
    }


def _target_payload(target: WorkflowLoraOverrideTarget) -> dict[str, object]:
    if type(target) is not WorkflowLoraOverrideTarget or type(target.overrides) is not tuple:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_overrides",
            "Workflow LoRA override target is not typed",
        )
    return {
        **_witness_payload(target.witness),
        "overrides": [_slot_override_payload(override) for override in target.overrides],
    }


def _envelope_payload(envelope: WorkflowLoraOverrides) -> dict[str, object]:
    if type(envelope) is not WorkflowLoraOverrides or type(envelope.targets) is not tuple:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_overrides",
            "Workflow LoRA override targets are not typed",
        )
    return {
        "version": envelope.version,
        "targets": [_target_payload(target) for target in envelope.targets],
    }


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraOverrideError(
            "invalid_workflow_lora_overrides",
            "Workflow LoRA overrides are not canonical JSON",
        ) from exc


def _canonical_envelope_bytes(envelope: WorkflowLoraOverrides) -> bytes:
    encoded = _canonical_bytes(_envelope_payload(envelope))
    if len(encoded) > MAX_WORKFLOW_LORA_OVERRIDE_CANONICAL_BYTES:
        raise WorkflowLoraOverrideError(
            "workflow_lora_overrides_too_large",
            "Workflow LoRA overrides exceed the canonical payload limit",
        )
    return encoded
