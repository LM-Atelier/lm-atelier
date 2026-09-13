"""Project embedded workflow LoRAs from one persisted revision, without mutation.

The UI and API graphs, the typed dependency contract, and the active binding
snapshot are separate claims.  This service joins them only where their exact
persisted identities agree.  Missing or contradictory authority makes a LoRA
visible but read-only. Exact authority adds only a public witness; graph
locators and the mutation itself remain private to later admission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from .comfy_package_widgets import (
    POWER_LORA_LOADER,
    PackageClaim,
    PackageWidgetError,
    package_widget_contract,
    package_widget_inputs,
)
from .comfy_subgraphs import SubgraphExpansionError, expand_workflow
from .comfy_workflow_packages import (
    WorkflowPackageError,
    analyze_comfyui_workflow_package,
)
from .lora_constraints import MAX_LORA_STRENGTH
from .models import (
    WorkflowDefinition,
    WorkflowDependencySlot,
    WorkflowRevision,
)
from .workflow_activations import WorkflowActivationError
from .workflow_bindings import ResolvedWorkflowBinding
from .workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyError,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_sha256,
)
from .workflow_lora_activation import (
    WorkflowLoraActivationEvidence,
    load_current_workflow_lora_activation_evidence,
)
from .workflow_lora_overrides import (
    WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION,
    WorkflowLoraOverrideCatalog,
    WorkflowLoraOverrideError,
    WorkflowLoraOverrideTargetWitness,
    resolve_workflow_lora_override_layers,
)
from .workflow_lora_slots import (
    CORE_LORA_LOADER_CONTRACT,
    CORE_LORA_LOADER_SCHEMA_SHA256,
    CORE_MODEL_ONLY_LORA_LOADER_CONTRACT,
    CORE_MODEL_ONLY_LORA_LOADER_SCHEMA_SHA256,
    CORE_RUNTIME_ADAPTER_CONTRACT_VERSION,
    MAX_WORKFLOW_LORA_CORE_EVIDENCE,
    MAX_WORKFLOW_LORA_PACKAGE_EVIDENCE,
    WorkflowLoraCoreEvidence,
    WorkflowLoraPackageEvidence,
    WorkflowLoraPrivateEditTarget,
    WorkflowLoraSlot,
    WorkflowLoraSlotExtraction,
    WorkflowLoraSlotExtractionWithPrivateTargets,
    extract_workflow_lora_slots,
    extract_workflow_lora_slots_with_private_targets,
)
from .workflow_trust import canonical_graph

WorkflowLoraEvidenceGap = Literal[
    "dependency_contract_unavailable",
    "dependency_contract_invalid",
    "active_activation_unavailable",
    "active_activation_invalid",
    "ui_graph_provenance_unavailable",
    "core_runtime_evidence_unavailable",
    "core_graph_binding_unavailable",
    "package_binding_evidence_unavailable",
    "package_graph_binding_unavailable",
]

_CORE_LOADERS = frozenset({"LoraLoader", "LoraLoaderModelOnly"})
_EMPTY_CONTRACT = WorkflowDependencyContract(version=1, slots=())


class WorkflowLoraProjectionError(ValueError):
    """The named revision cannot produce even a safe read-only projection."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class WorkflowLoraStrengthBoundsProjection:
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class WorkflowLoraControlsProjection:
    version: int
    override_contract_version: int
    strength_bounds: WorkflowLoraStrengthBoundsProjection
    override_target: WorkflowLoraOverrideTargetWitness | None
    revision_scope_sha256: str
    api_graph_sha256: str
    dependency_contract_sha256: str
    activation_binding_sha256: str | None
    ordering_authority: Literal["presentation_only"]
    evidence_gaps: tuple[WorkflowLoraEvidenceGap, ...]
    slots: tuple[WorkflowLoraSlot, ...]


@dataclass(frozen=True, slots=True)
class _ContractSnapshot:
    contract: WorkflowDependencyContract
    rows_by_id: dict[str, WorkflowDependencySlot]
    verified: bool


@dataclass(frozen=True, slots=True)
class _UiGraphSnapshot:
    nodes: dict[str, Mapping[str, Any]]
    connections: dict[tuple[str, str], list[object]]


@dataclass(frozen=True, slots=True)
class _UiLink:
    identifier: str
    origin: str
    origin_slot: int
    target: str
    target_slot: int


def workflow_lora_controls(
    session: Session,
    *,
    revision_id: str,
) -> WorkflowLoraControlsProjection:
    """Return the public-safe LoRA controls for one exact stored revision."""

    revision = session.get(WorkflowRevision, revision_id)
    if revision is None:
        raise WorkflowLoraProjectionError(
            "workflow_revision_not_found",
            "Workflow revision not found",
        )
    if not isinstance(revision.api_graph_json, Mapping):
        raise WorkflowLoraProjectionError(
            "workflow_lora_projection_unavailable",
            "The stored workflow prompt is not an object",
        )

    gaps: set[WorkflowLoraEvidenceGap] = set()
    contract_snapshot = _load_contract(session, revision, gaps)
    activation_evidence = _load_activation_evidence(
        session,
        revision,
        contract_snapshot,
        gaps,
    )
    activation = activation_evidence.resolution if activation_evidence is not None else None

    # The first extraction validates and hashes the prompt using the pure slot
    # contract itself.  It also remains the correct result when no node-level
    # provenance can be established: every detected entry is still visible,
    # but none receives edit authority.
    detected = extract_workflow_lora_slots(
        revision_scope=revision.id,
        api_graph=revision.api_graph_json,
        dependency_contract=contract_snapshot.contract,
        activation=activation,
    )

    core_evidence: tuple[WorkflowLoraCoreEvidence, ...] = ()
    package_evidence: tuple[WorkflowLoraPackageEvidence, ...] = ()
    loader_types = _loader_types(revision.api_graph_json)
    if loader_types:
        ui_graph = (
            _expanded_ui_graph(revision.ui_graph_json) if revision.engine == "comfyui" else None
        )
        if ui_graph is None:
            gaps.add("ui_graph_provenance_unavailable")
        elif activation is not None:
            if loader_types & _CORE_LOADERS:
                core_evidence = _core_evidence(
                    revision.api_graph_json,
                    ui_graph,
                    activation.bindings,
                    detected.api_graph_sha256,
                    gaps,
                )
            if POWER_LORA_LOADER in loader_types:
                package_evidence = _package_evidence(
                    revision.api_graph_json,
                    ui_graph,
                    activation.bindings,
                    detected.api_graph_sha256,
                    gaps,
                )

    extraction = extract_workflow_lora_slots_with_private_targets(
        revision_scope=revision.id,
        api_graph=revision.api_graph_json,
        dependency_contract=contract_snapshot.contract,
        activation=activation,
        core_evidence=core_evidence,
        package_evidence=package_evidence,
    )
    extracted = extraction.public
    override_target = _override_target(
        session,
        revision,
        activation_evidence,
        extraction,
    )
    return WorkflowLoraControlsProjection(
        version=extracted.version,
        override_contract_version=WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION,
        strength_bounds=WorkflowLoraStrengthBoundsProjection(
            minimum=-MAX_LORA_STRENGTH,
            maximum=MAX_LORA_STRENGTH,
        ),
        override_target=override_target,
        revision_scope_sha256=extracted.revision_scope_sha256,
        api_graph_sha256=extracted.api_graph_sha256,
        dependency_contract_sha256=extracted.dependency_contract_sha256,
        activation_binding_sha256=extracted.activation_binding_sha256,
        ordering_authority=extracted.ordering_authority,
        evidence_gaps=tuple(sorted(gaps)),
        slots=extracted.slots,
    )


def _load_contract(
    session: Session,
    revision: WorkflowRevision,
    gaps: set[WorkflowLoraEvidenceGap],
) -> _ContractSnapshot:
    if revision.dependency_contract_sha256 is None:
        gaps.add("dependency_contract_unavailable")
        return _ContractSnapshot(_EMPTY_CONTRACT, {}, False)
    rows = list(
        session.scalars(
            select(WorkflowDependencySlot)
            .where(WorkflowDependencySlot.workflow_revision_id == revision.id)
            .order_by(WorkflowDependencySlot.ordinal)
        ).all()
    )
    try:
        if [row.ordinal for row in rows] != list(range(len(rows))):
            raise ValueError("non-canonical slot order")
        contract = parse_workflow_dependency_contract(
            {
                "version": 1,
                "slots": [
                    {
                        "name": row.name,
                        "resource_kind": row.resource_kind,
                        "required": row.required,
                        "satisfaction": row.satisfaction,
                        "requirements": row.requirements_json,
                    }
                    for row in rows
                ],
            }
        )
        by_name = {slot.name: slot for slot in contract.slots}
        if any(
            row.contract_sha256 != workflow_dependency_slot_sha256(by_name[row.name])
            for row in rows
        ):
            raise ValueError("slot digest mismatch")
        if revision.dependency_contract_sha256 != workflow_dependency_contract_sha256(contract):
            raise ValueError("contract digest mismatch")
    except (KeyError, TypeError, ValueError, WorkflowDependencyError):
        gaps.add("dependency_contract_invalid")
        return _ContractSnapshot(_EMPTY_CONTRACT, {}, False)
    return _ContractSnapshot(contract, {row.id: row for row in rows}, True)


def _load_activation_evidence(
    session: Session,
    revision: WorkflowRevision,
    snapshot: _ContractSnapshot,
    gaps: set[WorkflowLoraEvidenceGap],
) -> WorkflowLoraActivationEvidence | None:
    if not snapshot.verified:
        return None
    try:
        evidence = load_current_workflow_lora_activation_evidence(session, revision.id)
    except WorkflowActivationError as exc:
        gaps.add(
            "active_activation_unavailable"
            if exc.code == "workflow_activation_unavailable"
            else "active_activation_invalid"
        )
        return None
    if evidence.dependency_contract_sha256 != revision.dependency_contract_sha256:
        gaps.add("active_activation_invalid")
        return None
    return evidence


def _override_target(
    session: Session,
    revision: WorkflowRevision,
    activation: WorkflowLoraActivationEvidence | None,
    extraction: WorkflowLoraSlotExtractionWithPrivateTargets,
) -> WorkflowLoraOverrideTargetWitness | None:
    """Issue public override authority only for one exact audited edit catalog."""

    if activation is None:
        return None
    public = extraction.public
    editable = tuple(slot for slot in public.slots if slot.editability == "editable")
    private = extraction.edit_targets
    if (
        not editable
        or len(editable) != len(private)
        or public.activation_binding_sha256 != activation.binding_sha256
        or public.dependency_contract_sha256 != activation.dependency_contract_sha256
        or any(
            not _private_target_matches_public(slot, target, public)
            for slot, target in zip(editable, private, strict=True)
        )
    ):
        return None

    definition = session.get(WorkflowDefinition, revision.workflow_id)
    if definition is None or definition.id != revision.workflow_id:
        return None
    witness = WorkflowLoraOverrideTargetWitness(
        workflow_family_id=definition.family_id,
        workflow_definition_id=revision.workflow_id,
        workflow_variant_key=definition.variant_key,
        workflow_revision_id=revision.id,
        slot_contract_version=public.version,
        revision_scope_sha256=public.revision_scope_sha256,
        api_graph_sha256=public.api_graph_sha256,
        dependency_contract_sha256=public.dependency_contract_sha256,
        activation_binding_sha256=activation.binding_sha256,
        activation_witness_sha256=activation.activation_witness_sha256,
    )
    try:
        resolution = resolve_workflow_lora_override_layers(
            catalog=WorkflowLoraOverrideCatalog(witness, public.slots),
            layers=(),
        )
    except WorkflowLoraOverrideError:
        return None
    if resolution.target != witness or resolution.overrides or resolution.inactive_targets:
        return None
    return witness


def _private_target_matches_public(
    slot: WorkflowLoraSlot,
    target: WorkflowLoraPrivateEditTarget,
    public: WorkflowLoraSlotExtraction,
) -> bool:
    return (
        type(target) is WorkflowLoraPrivateEditTarget
        and target.slot_id == slot.slot_id
        and target.source_api_graph_sha256 == public.api_graph_sha256
        and target.source_dependency_contract_sha256 == public.dependency_contract_sha256
        and target.source_activation_binding_sha256 == public.activation_binding_sha256
        and target.loader_contract == slot.loader_contract
        and target.loader_authority_sha256 == slot.loader_authority_sha256
        and slot.asset_binding is not None
        and target.asset_sha256 == slot.asset_binding.sha256
        and target.editable_fields == slot.editable_fields
    )


def _loader_types(api_graph: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        loader_type
        for node in api_graph.values()
        if isinstance(node, Mapping)
        and isinstance((loader_type := node.get("class_type")), str)
        and loader_type in _CORE_LOADERS | {POWER_LORA_LOADER}
    )


def _expanded_ui_graph(value: object) -> _UiGraphSnapshot | None:
    if not isinstance(value, Mapping):
        return None
    try:
        analyze_comfyui_workflow_package(cast(Mapping[str, object], value))
        expanded = expand_workflow(value)
        analyze_comfyui_workflow_package(expanded)
    except (RecursionError, TypeError, ValueError, SubgraphExpansionError, WorkflowPackageError):
        return None
    raw_nodes = expanded.get("nodes")
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, str | bytes):
        return None
    nodes: dict[str, Mapping[str, Any]] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping):
            return None
        identifier = _ui_identifier(raw_node.get("id"))
        if identifier is None or identifier in nodes:
            return None
        nodes[identifier] = raw_node
    connections = _ui_connections(expanded, nodes)
    if connections is None:
        return None
    return _UiGraphSnapshot(nodes, connections)


def _ui_connections(
    graph: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], list[object]] | None:
    raw_links = graph.get("links", [])
    if not isinstance(raw_links, Sequence) or isinstance(raw_links, str | bytes):
        return None
    slots = _ui_node_slots(nodes)
    if slots is None:
        return None
    by_id: dict[str, _UiLink] = {}
    result: dict[tuple[str, str], list[object]] = {}
    for raw_link in raw_links:
        if isinstance(raw_link, Mapping):
            identifier = _ui_identifier(raw_link.get("id"))
            origin = _ui_identifier(raw_link.get("origin_id"))
            origin_slot = raw_link.get("origin_slot")
            target = _ui_identifier(raw_link.get("target_id"))
            target_slot = raw_link.get("target_slot")
        elif (
            isinstance(raw_link, Sequence)
            and not isinstance(raw_link, str | bytes)
            and len(raw_link) >= 5
        ):
            identifier = _ui_identifier(raw_link[0])
            origin = _ui_identifier(raw_link[1])
            origin_slot = raw_link[2]
            target = _ui_identifier(raw_link[3])
            target_slot = raw_link[4]
        else:
            return None
        if (
            identifier is None
            or identifier in by_id
            or origin is None
            or target is None
            or origin not in nodes
            or target not in nodes
            or isinstance(origin_slot, bool)
            or not isinstance(origin_slot, int)
            or origin_slot < 0
            or isinstance(target_slot, bool)
            or not isinstance(target_slot, int)
            or target_slot < 0
        ):
            return None
        target_inputs, _ = slots[target]
        _, origin_outputs = slots[origin]
        if target_slot >= len(target_inputs) or origin_slot >= len(origin_outputs):
            return None
        target_input = target_inputs[target_slot]
        origin_output = origin_outputs[origin_slot]
        name = target_input["name"]
        key = (target, name)
        if key in result:
            return None
        declared_target = target_input.get("link")
        if declared_target is not None and _ui_identifier(declared_target) != identifier:
            return None
        declared_origins = origin_output.get("links")
        if declared_origins is not None:
            origin_ids = _ui_link_identifiers(declared_origins)
            if origin_ids is None or identifier not in origin_ids:
                return None
        link = _UiLink(identifier, origin, origin_slot, target, target_slot)
        by_id[identifier] = link
        result[key] = [origin, origin_slot]

    # The raw link table is not compiler authority on its own. Each slot also
    # records its incoming/outgoing link identities; all declared metadata must
    # name the same endpoint and slot, and no declared orphan may survive.
    for node_id, (inputs, outputs) in slots.items():
        for slot_index, value in enumerate(inputs):
            declared = value.get("link")
            if declared is None:
                continue
            link_id = _ui_identifier(declared)
            declared_link = by_id.get(link_id) if link_id is not None else None
            if (
                declared_link is None
                or declared_link.target != node_id
                or declared_link.target_slot != slot_index
            ):
                return None
        for slot_index, value in enumerate(outputs):
            declared = value.get("links")
            if declared is None:
                continue
            link_ids = _ui_link_identifiers(declared)
            if link_ids is None:
                return None
            for link_id in link_ids:
                declared_link = by_id.get(link_id)
                if (
                    declared_link is None
                    or declared_link.origin != node_id
                    or declared_link.origin_slot != slot_index
                ):
                    return None
    return result


def _ui_node_slots(
    nodes: Mapping[str, Mapping[str, Any]],
) -> (
    dict[
        str,
        tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]],
    ]
    | None
):
    result: dict[
        str,
        tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]],
    ] = {}
    for node_id, node in nodes.items():
        raw_inputs = node.get("inputs", [])
        raw_outputs = node.get("outputs", [])
        if (
            not isinstance(raw_inputs, Sequence)
            or isinstance(raw_inputs, str | bytes)
            or not isinstance(raw_outputs, Sequence)
            or isinstance(raw_outputs, str | bytes)
            or any(not isinstance(item, Mapping) for item in (*raw_inputs, *raw_outputs))
        ):
            return None
        inputs = tuple(item for item in raw_inputs if isinstance(item, Mapping))
        outputs = tuple(item for item in raw_outputs if isinstance(item, Mapping))
        names: set[str] = set()
        for value in inputs:
            name = value.get("name")
            if not isinstance(name, str) or not name or name in names:
                return None
            names.add(name)
            widget = value.get("widget")
            if widget is not None and (
                not isinstance(widget, Mapping) or widget.get("name") != name
            ):
                return None
        result[node_id] = (inputs, outputs)
    return result


def _ui_link_identifiers(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    result: list[str] = []
    for raw in value:
        identifier = _ui_identifier(raw)
        if identifier is None:
            return None
        result.append(identifier)
    return tuple(result)


def _ui_identifier(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int | str):
        return None
    result = str(value)
    if (
        not result
        or len(result) > 200
        or any(character < " " or character == "\x7f" for character in result)
    ):
        return None
    return result


def _core_evidence(
    api_graph: Mapping[str, Any],
    ui_graph: _UiGraphSnapshot,
    bindings: Sequence[ResolvedWorkflowBinding],
    graph_sha256: str,
    gaps: set[WorkflowLoraEvidenceGap],
) -> tuple[WorkflowLoraCoreEvidence, ...]:
    runtime = _one_core_runtime(bindings)
    if runtime is None:
        gaps.add("core_runtime_evidence_unavailable")
        return ()
    extension_node_types = _activated_extension_node_types(bindings)
    evidence: list[WorkflowLoraCoreEvidence] = []
    missing_graph_binding = False
    for node_id, api_node in api_graph.items():
        if not isinstance(api_node, Mapping):
            continue
        loader_type = api_node.get("class_type")
        if loader_type not in _CORE_LOADERS:
            continue
        # A graph-local `cnr_id=comfy-core` claim is forgeable. Activated
        # extensions are independent runtime providers, so an advertised
        # collision vetoes core authority. Older/manual custom-node bindings
        # without portable node-type evidence are ambiguous and veto every
        # core loader until a stronger persisted origin witness exists.
        if (
            extension_node_types is None
            or cast(str, loader_type).casefold() in extension_node_types
        ):
            missing_graph_binding = True
            continue
        core_claimed = _core_ui_binding(
            ui_graph.nodes.get(node_id),
            api_node,
            cast(str, loader_type),
            ui_graph.connections,
            node_id,
        )
        if core_claimed is None:
            missing_graph_binding = True
            continue
        loader_contract, schema_sha256 = (
            (CORE_LORA_LOADER_CONTRACT, CORE_LORA_LOADER_SCHEMA_SHA256)
            if loader_type == "LoraLoader"
            else (
                CORE_MODEL_ONLY_LORA_LOADER_CONTRACT,
                CORE_MODEL_ONLY_LORA_LOADER_SCHEMA_SHA256,
            )
        )
        evidence.append(
            WorkflowLoraCoreEvidence(
                api_graph_sha256=graph_sha256,
                node_id=node_id,
                loader_contract=loader_contract,
                adapter_schema_sha256=schema_sha256,
                core_claimed=core_claimed,
                dependency_slot=runtime.slot_name,
                requirement_key=runtime.requirement_key,
                resource_identity_sha256=runtime.resource_identity_sha256,
            )
        )
    if missing_graph_binding or len(evidence) > MAX_WORKFLOW_LORA_CORE_EVIDENCE:
        gaps.add("core_graph_binding_unavailable")
    if len(evidence) > MAX_WORKFLOW_LORA_CORE_EVIDENCE:
        return ()
    return tuple(evidence)


def _activated_extension_node_types(
    bindings: Sequence[ResolvedWorkflowBinding],
) -> frozenset[str] | None:
    result: set[str] = set()
    for binding in bindings:
        if binding.resource_kind == "custom_node":
            # The portable custom-node identity contract does not authenticate
            # provided node types. Extra JSON keys remain hashable, so even a
            # plausible-looking list could be attacker-authored and cannot
            # prove that the extension does not replace a core loader.
            return None
        if binding.resource_kind != "registry_package":
            continue
        identity = binding.identity
        raw_node_types = identity.get("node_types") if isinstance(identity, Mapping) else None
        if (
            not isinstance(raw_node_types, list)
            or not raw_node_types
            or len(raw_node_types) > 4_096
        ):
            return None
        local: set[str] = set()
        for value in raw_node_types:
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 200
                or any(character < " " or character == "\x7f" for character in value)
            ):
                return None
            folded = value.casefold()
            if folded in local:
                return None
            local.add(folded)
            result.add(folded)
    return frozenset(result)


def _one_core_runtime(
    bindings: Sequence[ResolvedWorkflowBinding],
) -> ResolvedWorkflowBinding | None:
    candidates = [
        binding
        for binding in bindings
        if binding.resource_kind == "runtime"
        and isinstance(binding.identity, dict)
        and binding.identity.get("kind") == "runtime"
        and binding.identity.get("engine") == "comfyui"
        and binding.identity.get("adapter_contract_version")
        == CORE_RUNTIME_ADAPTER_CONTRACT_VERSION
    ]
    return candidates[0] if len(candidates) == 1 else None


def _core_ui_binding(
    ui_node: Mapping[str, Any] | None,
    api_node: Mapping[str, Any],
    loader_type: str,
    connections: Mapping[tuple[str, str], list[object]],
    node_id: str,
) -> bool | None:
    if ui_node is None or ui_node.get("type") != loader_type or ui_node.get("mode", 0) != 0:
        return None
    inputs = api_node.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    link_names = ("model", "clip") if loader_type == "LoraLoader" else ("model",)
    if any(
        not _same_json(inputs.get(name), connections.get((node_id, name))) for name in link_names
    ):
        return None
    names = (
        ("lora_name", "strength_model", "strength_clip")
        if loader_type == "LoraLoader"
        else ("lora_name", "strength_model")
    )
    expected = [inputs.get(name) for name in names]
    raw_values = ui_node.get("widgets_values", [])
    if isinstance(raw_values, Mapping):
        if set(raw_values) != set(names):
            return None
        observed = [raw_values.get(name) for name in names]
    elif isinstance(raw_values, Sequence) and not isinstance(raw_values, str | bytes):
        observed = list(raw_values)
    else:
        return None
    if not _same_json(observed, expected):
        return None
    properties = ui_node.get("properties")
    auxiliary_id = properties.get("aux_id") if isinstance(properties, Mapping) else None
    return bool(
        isinstance(properties, Mapping)
        and properties.get("cnr_id") == "comfy-core"
        and (auxiliary_id is None or auxiliary_id == "")
    )


def _package_evidence(
    api_graph: Mapping[str, Any],
    ui_graph: _UiGraphSnapshot,
    bindings: Sequence[ResolvedWorkflowBinding],
    graph_sha256: str,
    gaps: set[WorkflowLoraEvidenceGap],
) -> tuple[WorkflowLoraPackageEvidence, ...]:
    evidence: list[WorkflowLoraPackageEvidence] = []
    missing_binding = False
    missing_graph_binding = False
    for node_id, api_node in api_graph.items():
        if not isinstance(api_node, Mapping) or api_node.get("class_type") != POWER_LORA_LOADER:
            continue
        ui_node = ui_graph.nodes.get(node_id)
        claim = _package_ui_claim(ui_node)
        if claim is None:
            missing_graph_binding = True
            continue
        package = _one_package_binding(bindings, claim)
        if package is None:
            missing_binding = True
            continue
        if not _package_ui_binding(
            ui_node,
            api_node,
            claim,
            ui_graph.connections,
            node_id,
        ):
            missing_graph_binding = True
            continue
        evidence.append(
            WorkflowLoraPackageEvidence(
                api_graph_sha256=graph_sha256,
                node_id=node_id,
                package_claim=claim,
                dependency_slot=package.slot_name,
                requirement_key=package.requirement_key,
                resource_identity_sha256=package.resource_identity_sha256,
            )
        )
    if missing_binding:
        gaps.add("package_binding_evidence_unavailable")
    if missing_graph_binding or len(evidence) > MAX_WORKFLOW_LORA_PACKAGE_EVIDENCE:
        gaps.add("package_graph_binding_unavailable")
    if len(evidence) > MAX_WORKFLOW_LORA_PACKAGE_EVIDENCE:
        return ()
    return tuple(evidence)


def _package_ui_claim(ui_node: Mapping[str, Any] | None) -> PackageClaim | None:
    if ui_node is None or ui_node.get("type") != POWER_LORA_LOADER or ui_node.get("mode", 0) != 0:
        return None
    properties = ui_node.get("properties")
    if not isinstance(properties, Mapping):
        return None
    registry = properties.get("cnr_id")
    repository = properties.get("aux_id")
    revision = properties.get("ver")
    if not isinstance(registry, str) or not registry:
        return None
    return PackageClaim(
        registry_id=registry,
        repository_id=(repository if isinstance(repository, str) and repository else None),
        revision=revision if isinstance(revision, str) else "",
    )


def _one_package_binding(
    bindings: Sequence[ResolvedWorkflowBinding],
    claim: PackageClaim,
) -> ResolvedWorkflowBinding | None:
    candidates = [
        binding
        for binding in bindings
        if binding.resource_kind == "registry_package"
        and isinstance(binding.identity, dict)
        and binding.identity.get("kind") == "registry_package"
        and binding.identity.get("package_id") == claim.registry_id
        and binding.identity.get("package_version") == claim.revision
        and isinstance(binding.identity.get("node_types"), list)
        and POWER_LORA_LOADER in binding.identity["node_types"]
    ]
    return candidates[0] if len(candidates) == 1 else None


def _package_ui_binding(
    ui_node: Mapping[str, Any] | None,
    api_node: Mapping[str, Any],
    claim: PackageClaim,
    connections: Mapping[tuple[str, str], list[object]],
    node_id: str,
) -> bool:
    if ui_node is None:
        return False
    # An unaudited or conflicting claim can be retained as evidence of why the
    # slot is read-only.  It cannot authorize an edit: the pure extractor asks
    # the audited-contract registry again and rejects it before parsing values.
    try:
        audited = package_widget_contract(POWER_LORA_LOADER, claim)
    except PackageWidgetError:
        return True
    if audited is None:
        return False
    raw_values = ui_node.get("widgets_values")
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, str | bytes):
        return False
    try:
        compiled = package_widget_inputs(POWER_LORA_LOADER, claim, raw_values)
    except PackageWidgetError:
        return False
    inputs = api_node.get("inputs")
    if compiled is None or not isinstance(inputs, Mapping):
        return False
    if any(
        name in inputs and not _same_json(inputs.get(name), connections.get((node_id, name)))
        for name in ("model", "clip")
    ):
        return False
    embedded = {key: value for key, value in inputs.items() if key not in {"model", "clip"}}
    return _same_json(embedded, compiled)


def _same_json(first: object, second: object) -> bool:
    try:
        return canonical_graph({"value": first}) == canonical_graph({"value": second})
    except (RecursionError, TypeError, ValueError):
        return False
