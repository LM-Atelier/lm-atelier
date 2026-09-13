"""Extract workflow-native LoRAs without granting guessed graph-edit authority.

The immutable API prompt says where a loader reads its values. Typed workflow
dependency bindings say which verified bytes a portable runtime name denotes.
For package-drawn loaders a third fact is required: graph-bound evidence naming
the exact package revision whose serialization was audited. None of those facts
substitutes for another.

This module is intentionally pure. It neither reads persistence nor mutates the
prompt. Its public result contains opaque, revision-scoped slot identifiers and
portable content identities, never graph node identifiers or machine paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from .comfy_package_widgets import (
    POWER_LORA_LOADER,
    POWER_LORA_LOADER_CONTRACT,
    PackageClaim,
    PackageWidgetError,
    package_widget_contract,
)
from .comfy_workflow_packages import (
    WorkflowPackageError,
    validate_bounded_workflow_json,
)
from .workflow_bindings import (
    ResolvedWorkflowBinding,
    WorkflowActivationResolution,
    WorkflowBindingError,
    validate_workflow_runtime_reference,
    workflow_activation_binding_sha256,
)
from .workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyError,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_payload,
    workflow_dependency_contract_sha256,
)
from .workflow_trust import canonical_graph

WORKFLOW_LORA_SLOT_CONTRACT_VERSION = 1
MAX_WORKFLOW_LORA_SLOTS = 64
MAX_WORKFLOW_LORA_PACKAGE_EVIDENCE = 64
MAX_WORKFLOW_LORA_CORE_EVIDENCE = 64
CORE_RUNTIME_ADAPTER_CONTRACT_VERSION = 1

CORE_LORA_LOADER_CONTRACT = "comfy-core-lora-loader-v1"
CORE_MODEL_ONLY_LORA_LOADER_CONTRACT = "comfy-core-lora-loader-model-only-v1"
CORE_LORA_LOADER_SCHEMA_SHA256 = hashlib.sha256(
    b"comfy-core:LoraLoader:model,clip,lora_name,strength_model,strength_clip:model,clip:v1"
).hexdigest()
CORE_MODEL_ONLY_LORA_LOADER_SCHEMA_SHA256 = hashlib.sha256(
    b"comfy-core:LoraLoaderModelOnly:model,lora_name,strength_model:model:v1"
).hexdigest()

WorkflowLoraEditability = Literal[
    "editable",
    "required_locked",
    "detected_read_only",
]
WorkflowLoraEditableField = Literal[
    "enabled",
    "model_strength",
    "clip_strength",
]
WorkflowLoraStrengthMode = Literal[
    "separate",
    "coupled",
    "model_only",
    "unknown",
]
WorkflowLoraOrderingAuthority = Literal["presentation_only"]

_REVISION_SCOPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RGTHREE_ENTRY = re.compile(r"^lora_([1-9][0-9]*)$")
_PUBLIC_LOADER_TYPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _().,+-]{0,199}$")
_STABLE_IDENTITY_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,199}$")
_CORE_LORA_LOADER = "LoraLoader"
_CORE_MODEL_ONLY_LORA_LOADER = "LoraLoaderModelOnly"
_NODE_FIELDS = frozenset({"class_type", "inputs", "_meta"})
_CORE_LOADER_AUTHORITY = {
    _CORE_LORA_LOADER: (
        CORE_LORA_LOADER_CONTRACT,
        CORE_LORA_LOADER_SCHEMA_SHA256,
    ),
    _CORE_MODEL_ONLY_LORA_LOADER: (
        CORE_MODEL_ONLY_LORA_LOADER_CONTRACT,
        CORE_MODEL_ONLY_LORA_LOADER_SCHEMA_SHA256,
    ),
}


class WorkflowLoraSlotError(ValueError):
    """A typed refusal to derive editable workflow LoRA slots."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class WorkflowLoraAssetBinding:
    """Portable, content-addressed evidence for one loader name."""

    dependency_slot: str
    requirement_key: str
    resource_identity_sha256: str
    runtime_reference: str
    sha256: str


@dataclass(frozen=True, slots=True)
class WorkflowLoraPackageEvidence:
    """Bind one compiled node to package evidence from the same revision.

    Node identifiers are accepted only on this server-side input. They are not
    copied into the extracted slot contract.
    """

    api_graph_sha256: str
    node_id: str
    package_claim: PackageClaim
    dependency_slot: str
    requirement_key: str
    resource_identity_sha256: str


@dataclass(frozen=True, slots=True)
class WorkflowLoraCoreEvidence:
    """Bind one core loader to exact compiler and runtime-adapter evidence."""

    api_graph_sha256: str
    node_id: str
    loader_contract: str
    adapter_schema_sha256: str
    core_claimed: bool
    dependency_slot: str
    requirement_key: str
    resource_identity_sha256: str


@dataclass(frozen=True, slots=True)
class WorkflowLoraSlot:
    """One public-safe projection of an embedded workflow LoRA entry."""

    slot_id: str
    position: int
    loader_type: str
    loader_contract: str | None
    loader_authority_sha256: str | None
    editability: WorkflowLoraEditability
    read_only_reason: str | None
    dependency_required: bool | None
    observed_runtime_reference: str | None
    asset_binding: WorkflowLoraAssetBinding | None
    default_enabled: bool | None
    default_model_strength: float | None
    default_clip_strength: float | None
    strength_mode: WorkflowLoraStrengthMode
    editable_fields: tuple[WorkflowLoraEditableField, ...]


@dataclass(frozen=True, slots=True)
class WorkflowLoraSlotExtraction:
    """Frozen revision evidence and its ordered opaque LoRA slots."""

    version: int
    revision_scope_sha256: str
    api_graph_sha256: str
    dependency_contract_sha256: str
    activation_binding_sha256: str | None
    ordering_authority: WorkflowLoraOrderingAuthority
    slots: tuple[WorkflowLoraSlot, ...]


@dataclass(frozen=True, slots=True)
class WorkflowLoraPrivateEditTarget:
    """Server-only authority for locating one safely editable graph entry.

    This is deliberately not part of the public projection. The source graph
    digest binds the locator to the same canonical bytes that produced the
    opaque slot identifier; callers must still copy and revalidate that graph
    before applying an edit. ``entry_locator`` is the exact key within the
    named node's ``inputs`` object; its versioned loader contract defines how
    each allowed field maps around or within that entry.
    """

    slot_id: str
    source_api_graph_sha256: str
    source_dependency_contract_sha256: str
    source_activation_binding_sha256: str
    source_authority_evidence_sha256: str
    loader_authority_sha256: str
    asset_sha256: str
    node_id: str
    entry_locator: str
    loader_contract: str
    editable_fields: tuple[WorkflowLoraEditableField, ...]


@dataclass(frozen=True, slots=True)
class WorkflowLoraSlotExtractionWithPrivateTargets:
    """One audited pass split into a public projection and private targets."""

    public: WorkflowLoraSlotExtraction
    edit_targets: tuple[WorkflowLoraPrivateEditTarget, ...]


@dataclass(frozen=True, slots=True)
class _AssetEvidence:
    binding: WorkflowLoraAssetBinding
    required: bool


@dataclass(frozen=True, slots=True)
class _ParsedLora:
    locator: str
    runtime_reference: str | None
    loader_contract: str | None
    loader_authority_sha256: str | None
    exact: bool
    read_only_reason: str | None
    enabled: bool | None
    model_strength: float | None
    clip_strength: float | None
    strength_mode: WorkflowLoraStrengthMode
    editable_fields: tuple[WorkflowLoraEditableField, ...]


class _LayoutError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def extract_workflow_lora_slots(
    *,
    revision_scope: str,
    api_graph: Mapping[str, Any],
    dependency_contract: WorkflowDependencyContract,
    activation: WorkflowActivationResolution | None = None,
    core_evidence: Sequence[WorkflowLoraCoreEvidence] = (),
    package_evidence: Sequence[WorkflowLoraPackageEvidence] = (),
) -> WorkflowLoraSlotExtraction:
    """Derive bounded immutable slots from one exact workflow revision.

    Missing authority is a visible read-only result. An absent activation means
    no bindings; a partially resolved activation is deliberately not consumed.
    Contradictory or malformed typed evidence is refused, because silently
    dropping it could make a stale activation look current.
    """

    return _extract_workflow_lora_slots(
        revision_scope=revision_scope,
        api_graph=api_graph,
        dependency_contract=dependency_contract,
        activation=activation,
        core_evidence=core_evidence,
        package_evidence=package_evidence,
        include_private_targets=False,
    ).public


def extract_workflow_lora_slots_with_private_targets(
    *,
    revision_scope: str,
    api_graph: Mapping[str, Any],
    dependency_contract: WorkflowDependencyContract,
    activation: WorkflowActivationResolution | None = None,
    core_evidence: Sequence[WorkflowLoraCoreEvidence] = (),
    package_evidence: Sequence[WorkflowLoraPackageEvidence] = (),
) -> WorkflowLoraSlotExtractionWithPrivateTargets:
    """Return public slots and their server-private graph edit locators.

    Only slots that retain ``editable`` authority receive a target. Required,
    malformed, unbound, and otherwise read-only entries remain visible in the
    public result but have no server-side edit target.
    """

    return _extract_workflow_lora_slots(
        revision_scope=revision_scope,
        api_graph=api_graph,
        dependency_contract=dependency_contract,
        activation=activation,
        core_evidence=core_evidence,
        package_evidence=package_evidence,
        include_private_targets=True,
    )


def _extract_workflow_lora_slots(
    *,
    revision_scope: str,
    api_graph: Mapping[str, Any],
    dependency_contract: WorkflowDependencyContract,
    activation: WorkflowActivationResolution | None,
    core_evidence: Sequence[WorkflowLoraCoreEvidence],
    package_evidence: Sequence[WorkflowLoraPackageEvidence],
    include_private_targets: bool,
) -> WorkflowLoraSlotExtractionWithPrivateTargets:
    if not isinstance(revision_scope, str) or not _REVISION_SCOPE.fullmatch(revision_scope):
        raise WorkflowLoraSlotError(
            "invalid_revision_scope",
            "Workflow LoRA extraction requires one stable revision scope",
        )
    graph_sha256 = _api_graph_sha256(api_graph)
    private_graph_is_canonical = not include_private_targets or (
        type(revision_scope) is str and _plain_json_value(api_graph)
    )
    graph_for_extraction = api_graph
    if include_private_targets and private_graph_is_canonical:
        graph_for_extraction = _private_graph_snapshot(api_graph, graph_sha256)

    contract = _canonical_contract(dependency_contract)
    contract_sha256 = workflow_dependency_contract_sha256(contract)
    bindings, activation_sha256 = _activation_bindings(contract, activation)
    private_activation_is_canonical = not include_private_targets or (
        _private_activation_is_canonical(activation)
    )
    bindings_for_extraction = bindings
    if include_private_targets and private_activation_is_canonical:
        bindings_for_extraction = _private_binding_snapshots(
            contract,
            bindings,
            activation_sha256,
        )

    assets = _asset_index(contract, bindings_for_extraction)
    source_core = _core_evidence_index(graph_for_extraction, core_evidence)
    source_packages = _package_evidence_index(graph_for_extraction, package_evidence)
    private_evidence_is_canonical = not include_private_targets or (
        all(_private_core_evidence_is_canonical(item) for item in source_core.values())
        and all(_private_package_evidence_is_canonical(item) for item in source_packages.values())
    )
    core = source_core
    packages = source_packages
    private_evidence_sha256: str | None = None
    if include_private_targets and private_evidence_is_canonical:
        private_evidence_sha256 = _private_authority_evidence_sha256(
            source_core,
            source_packages,
        )
        core = _private_core_evidence_snapshots(graph_for_extraction, source_core)
        packages = _private_package_evidence_snapshots(
            graph_for_extraction,
            source_packages,
        )
        if _private_authority_evidence_sha256(core, packages) != private_evidence_sha256:
            raise WorkflowLoraSlotError(
                "changed_private_edit_source",
                "Workflow LoRA edit evidence changed during private target extraction",
            )
    private_sources_are_canonical = (
        private_graph_is_canonical
        and private_activation_is_canonical
        and private_evidence_is_canonical
    )
    if not private_sources_are_canonical:
        private_evidence_sha256 = None
    node_order = tuple(sorted(graph_for_extraction))

    slots: list[WorkflowLoraSlot] = []
    edit_targets: list[WorkflowLoraPrivateEditTarget] = []
    for node_id in node_order:
        node = graph_for_extraction[node_id]
        if not isinstance(node, Mapping):
            raise WorkflowLoraSlotError(
                "invalid_api_graph_node",
                "Workflow API graph contains a node that is not an object",
            )
        loader_type = node.get("class_type")
        if not isinstance(loader_type, str) or not loader_type:
            raise WorkflowLoraSlotError(
                "invalid_api_graph_node",
                "Workflow API graph contains a node without a class type",
            )

        if loader_type in {_CORE_LORA_LOADER, _CORE_MODEL_ONLY_LORA_LOADER}:
            loader_contract, core_reason, core_authority_sha256 = _core_authority(
                node_id,
                loader_type,
                graph_sha256,
                core.get(node_id),
                bindings_for_extraction,
            )
            parsed = _core_loras(
                graph_for_extraction,
                node_id,
                node,
                loader_type,
                loader_contract,
                core_reason,
                core_authority_sha256,
            )
        elif loader_type == POWER_LORA_LOADER:
            package_contract, package_reason, package_authority_sha256 = _rgthree_authority(
                node_id,
                graph_sha256,
                packages.get(node_id),
                bindings_for_extraction,
            )
            parsed = _rgthree_loras(
                graph_for_extraction,
                node_id,
                node,
                package_contract,
                package_reason,
                package_authority_sha256,
            )
        else:
            parsed = _detected_loras(node, loader_type, "unsupported_loader_contract")

        for item in parsed:
            if len(slots) >= MAX_WORKFLOW_LORA_SLOTS:
                raise WorkflowLoraSlotError(
                    "too_many_workflow_loras",
                    "Workflow declares more embedded LoRAs than this contract can represent",
                )
            asset = _one_asset(assets, item.runtime_reference)
            slot = _public_slot(
                revision_scope=revision_scope,
                graph_sha256=graph_sha256,
                node_id=node_id,
                loader_type=loader_type,
                parsed=item,
                asset=asset,
                position=len(slots),
            )
            slots.append(slot)
            if include_private_targets and slot.editability == "editable":
                edit_targets.append(
                    _private_edit_target(
                        slot=slot,
                        parsed=item,
                        node_id=node_id,
                        graph_sha256=graph_sha256,
                        contract_sha256=contract_sha256,
                        activation_sha256=activation_sha256,
                        evidence_sha256=private_evidence_sha256,
                        sources_are_canonical=private_sources_are_canonical,
                    )
                )

    if edit_targets:
        _revalidate_private_edit_sources(
            api_graph=api_graph,
            graph_sha256=graph_sha256,
            contract=contract,
            contract_sha256=contract_sha256,
            activation=activation,
            activation_sha256=activation_sha256,
            package_evidence=source_packages,
            core_evidence=source_core,
            evidence_sha256=private_evidence_sha256,
        )

    return WorkflowLoraSlotExtractionWithPrivateTargets(
        public=WorkflowLoraSlotExtraction(
            version=WORKFLOW_LORA_SLOT_CONTRACT_VERSION,
            revision_scope_sha256=hashlib.sha256(revision_scope.encode("utf-8")).hexdigest(),
            api_graph_sha256=graph_sha256,
            dependency_contract_sha256=contract_sha256,
            activation_binding_sha256=activation_sha256,
            ordering_authority="presentation_only",
            slots=tuple(slots),
        ),
        edit_targets=tuple(edit_targets),
    )


def _plain_json_value(value: object) -> bool:
    """Require inert built-in JSON containers before issuing mutation authority."""

    stack = [value]
    while stack:
        current = stack.pop()
        if type(current) is dict:
            mapping = current
            if any(type(key) is not str for key in mapping):
                return False
            stack.extend(mapping.values())
        elif type(current) is list:
            stack.extend(current)
        elif current is None or type(current) in {str, int, float, bool}:
            continue
        else:
            return False
    return True


def _private_graph_snapshot(
    api_graph: Mapping[str, Any],
    expected_sha256: str,
) -> dict[str, Any]:
    """Detach exact JSON bytes so source container methods cannot affect parsing."""

    try:
        encoded = canonical_graph(api_graph)
        detached = json.loads(encoded)
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        ) from exc
    if (
        type(detached) is not dict
        or not _plain_json_value(detached)
        or hashlib.sha256(encoded.encode("utf-8")).hexdigest() != expected_sha256
    ):
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        )
    return cast(dict[str, Any], detached)


def _private_binding_snapshots(
    contract: WorkflowDependencyContract,
    bindings: Sequence[ResolvedWorkflowBinding],
    expected_sha256: str | None,
) -> tuple[ResolvedWorkflowBinding, ...]:
    if expected_sha256 is None:
        if bindings:
            raise WorkflowLoraSlotError(
                "changed_private_edit_source",
                "Workflow LoRA edit evidence changed during private target extraction",
            )
        return ()
    snapshots = tuple(
        ResolvedWorkflowBinding(
            slot_name=binding.slot_name,
            requirement_key=binding.requirement_key,
            resource_kind=binding.resource_kind,
            identity=_private_plain_mapping_snapshot(binding.identity),
            resource_identity_sha256=binding.resource_identity_sha256,
            mount=_private_plain_mapping_snapshot(binding.mount),
        )
        for binding in bindings
    )
    try:
        actual_sha256 = workflow_activation_binding_sha256(contract, snapshots)
    except (AttributeError, RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        ) from exc
    if actual_sha256 != expected_sha256:
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        )
    return snapshots


def _private_plain_mapping_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        detached = json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        ) from exc
    if type(detached) is not dict or not _plain_json_value(detached):
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        )
    return cast(dict[str, Any], detached)


def _private_core_evidence_snapshots(
    graph: Mapping[str, Any],
    evidence: Mapping[str, WorkflowLoraCoreEvidence],
) -> dict[str, WorkflowLoraCoreEvidence]:
    snapshots = {
        node_id: WorkflowLoraCoreEvidence(
            api_graph_sha256=item.api_graph_sha256,
            node_id=item.node_id,
            loader_contract=item.loader_contract,
            adapter_schema_sha256=item.adapter_schema_sha256,
            core_claimed=item.core_claimed,
            dependency_slot=item.dependency_slot,
            requirement_key=item.requirement_key,
            resource_identity_sha256=item.resource_identity_sha256,
        )
        for node_id, item in evidence.items()
    }
    if not all(_private_core_evidence_is_canonical(item) for item in snapshots.values()):
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        )
    return _core_evidence_index(graph, tuple(snapshots.values()))


def _private_package_evidence_snapshots(
    graph: Mapping[str, Any],
    evidence: Mapping[str, WorkflowLoraPackageEvidence],
) -> dict[str, WorkflowLoraPackageEvidence]:
    snapshots = {
        node_id: WorkflowLoraPackageEvidence(
            api_graph_sha256=item.api_graph_sha256,
            node_id=item.node_id,
            package_claim=PackageClaim(
                registry_id=item.package_claim.registry_id,
                repository_id=item.package_claim.repository_id,
                revision=item.package_claim.revision,
            ),
            dependency_slot=item.dependency_slot,
            requirement_key=item.requirement_key,
            resource_identity_sha256=item.resource_identity_sha256,
        )
        for node_id, item in evidence.items()
    }
    if not all(_private_package_evidence_is_canonical(item) for item in snapshots.values()):
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        )
    return _package_evidence_index(graph, tuple(snapshots.values()))


def _private_activation_is_canonical(
    activation: WorkflowActivationResolution | None,
) -> bool:
    if activation is None:
        return True
    return (
        type(activation) is WorkflowActivationResolution
        and type(activation.bindings) is tuple
        and type(activation.issues) is tuple
        and type(activation.missing_required_slots) is tuple
        and type(activation.complete) is bool
        and (activation.binding_sha256 is None or type(activation.binding_sha256) is str)
        and all(type(value) is str for value in activation.missing_required_slots)
        and all(type(binding) is ResolvedWorkflowBinding for binding in activation.bindings)
        and all(
            type(binding.slot_name) is str
            and type(binding.requirement_key) is str
            and type(binding.resource_kind) is str
            and type(binding.resource_identity_sha256) is str
            and _plain_json_value(binding.identity)
            and _plain_json_value(binding.mount)
            for binding in activation.bindings
        )
    )


def _private_core_evidence_is_canonical(evidence: WorkflowLoraCoreEvidence) -> bool:
    return type(evidence) is WorkflowLoraCoreEvidence and all(
        type(value) is expected
        for value, expected in (
            (evidence.api_graph_sha256, str),
            (evidence.node_id, str),
            (evidence.loader_contract, str),
            (evidence.adapter_schema_sha256, str),
            (evidence.core_claimed, bool),
            (evidence.dependency_slot, str),
            (evidence.requirement_key, str),
            (evidence.resource_identity_sha256, str),
        )
    )


def _private_package_evidence_is_canonical(
    evidence: WorkflowLoraPackageEvidence,
) -> bool:
    claim = evidence.package_claim
    return (
        type(evidence) is WorkflowLoraPackageEvidence
        and type(evidence.api_graph_sha256) is str
        and type(evidence.node_id) is str
        and type(evidence.dependency_slot) is str
        and type(evidence.requirement_key) is str
        and type(evidence.resource_identity_sha256) is str
        and type(claim) is PackageClaim
        and (claim.registry_id is None or type(claim.registry_id) is str)
        and (claim.repository_id is None or type(claim.repository_id) is str)
        and type(claim.revision) is str
    )


def _private_edit_target(
    *,
    slot: WorkflowLoraSlot,
    parsed: _ParsedLora,
    node_id: str,
    graph_sha256: str,
    contract_sha256: str,
    activation_sha256: str | None,
    evidence_sha256: str | None,
    sources_are_canonical: bool,
) -> WorkflowLoraPrivateEditTarget:
    if (
        not sources_are_canonical
        or type(node_id) is not str
        or type(parsed.locator) is not str
        or not parsed.exact
        or type(graph_sha256) is not str
        or _DIGEST.fullmatch(graph_sha256) is None
        or type(contract_sha256) is not str
        or _DIGEST.fullmatch(contract_sha256) is None
        or type(activation_sha256) is not str
        or _DIGEST.fullmatch(activation_sha256) is None
        or type(evidence_sha256) is not str
        or _DIGEST.fullmatch(evidence_sha256) is None
        or type(slot.loader_contract) is not str
        or type(slot.loader_authority_sha256) is not str
        or _DIGEST.fullmatch(slot.loader_authority_sha256) is None
        or slot.asset_binding is None
        or type(slot.asset_binding.sha256) is not str
        or _DIGEST.fullmatch(slot.asset_binding.sha256) is None
        or not slot.editable_fields
        or slot.editable_fields != parsed.editable_fields
    ):
        raise WorkflowLoraSlotError(
            "noncanonical_private_edit_source",
            "Workflow LoRA private edit authority requires canonical audited evidence",
        )
    return WorkflowLoraPrivateEditTarget(
        slot_id=slot.slot_id,
        source_api_graph_sha256=graph_sha256,
        source_dependency_contract_sha256=contract_sha256,
        source_activation_binding_sha256=activation_sha256,
        source_authority_evidence_sha256=evidence_sha256,
        loader_authority_sha256=slot.loader_authority_sha256,
        asset_sha256=slot.asset_binding.sha256,
        node_id=node_id,
        entry_locator=parsed.locator,
        loader_contract=slot.loader_contract,
        editable_fields=slot.editable_fields,
    )


def _private_authority_evidence_sha256(
    core_evidence: Mapping[str, WorkflowLoraCoreEvidence],
    package_evidence: Mapping[str, WorkflowLoraPackageEvidence],
) -> str:
    """Hash the exact scalar-only authority records consumed by this pass."""

    payload = {
        "core": [
            {
                "node_id": node_id,
                "api_graph_sha256": item.api_graph_sha256,
                "evidence_node_id": item.node_id,
                "loader_contract": item.loader_contract,
                "adapter_schema_sha256": item.adapter_schema_sha256,
                "core_claimed": item.core_claimed,
                "dependency_slot": item.dependency_slot,
                "requirement_key": item.requirement_key,
                "resource_identity_sha256": item.resource_identity_sha256,
            }
            for node_id, item in sorted(core_evidence.items())
        ],
        "packages": [
            {
                "node_id": node_id,
                "api_graph_sha256": item.api_graph_sha256,
                "evidence_node_id": item.node_id,
                "package_claim": {
                    "registry_id": item.package_claim.registry_id,
                    "repository_id": item.package_claim.repository_id,
                    "revision": item.package_claim.revision,
                },
                "dependency_slot": item.dependency_slot,
                "requirement_key": item.requirement_key,
                "resource_identity_sha256": item.resource_identity_sha256,
            }
            for node_id, item in sorted(package_evidence.items())
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _revalidate_private_edit_sources(
    *,
    api_graph: Mapping[str, Any],
    graph_sha256: str,
    contract: WorkflowDependencyContract,
    contract_sha256: str,
    activation: WorkflowActivationResolution | None,
    activation_sha256: str | None,
    core_evidence: Mapping[str, WorkflowLoraCoreEvidence],
    package_evidence: Mapping[str, WorkflowLoraPackageEvidence],
    evidence_sha256: str | None,
) -> None:
    """Refuse authority if mutable evidence drifted during the extraction pass."""

    if activation is None or activation_sha256 is None or evidence_sha256 is None:
        raise WorkflowLoraSlotError(
            "noncanonical_private_edit_source",
            "Workflow LoRA private edit authority requires canonical audited evidence",
        )
    if (
        not _plain_json_value(api_graph)
        or not _private_activation_is_canonical(activation)
        or not all(_private_core_evidence_is_canonical(item) for item in core_evidence.values())
        or not all(
            _private_package_evidence_is_canonical(item) for item in package_evidence.values()
        )
    ):
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        )
    try:
        current_graph_sha256 = _api_graph_sha256(api_graph)
        current_contract_sha256 = workflow_dependency_contract_sha256(contract)
        current_activation_sha256 = workflow_activation_binding_sha256(
            contract,
            activation.bindings,
        )
        current_evidence_sha256 = _private_authority_evidence_sha256(
            core_evidence,
            package_evidence,
        )
    except (AttributeError, RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        ) from exc
    if (
        current_graph_sha256 != graph_sha256
        or current_contract_sha256 != contract_sha256
        or current_activation_sha256 != activation_sha256
        or current_evidence_sha256 != evidence_sha256
    ):
        raise WorkflowLoraSlotError(
            "changed_private_edit_source",
            "Workflow LoRA edit evidence changed during private target extraction",
        )


def _api_graph_sha256(api_graph: Mapping[str, Any]) -> str:
    if not isinstance(api_graph, Mapping):
        raise WorkflowLoraSlotError(
            "invalid_api_graph",
            "Workflow API graph must be an object",
        )
    try:
        validate_bounded_workflow_json(api_graph)
        canonical = canonical_graph(api_graph)
    except WorkflowPackageError as exc:
        raise WorkflowLoraSlotError(exc.code, str(exc)) from exc
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraSlotError(
            "invalid_api_graph",
            "Workflow API graph is not canonical JSON",
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_contract(contract: WorkflowDependencyContract) -> WorkflowDependencyContract:
    if not isinstance(contract, WorkflowDependencyContract):
        raise WorkflowLoraSlotError(
            "invalid_dependency_contract",
            "Workflow dependency evidence is not a typed contract",
        )
    try:
        # Parsing round-trips every nested constraint through canonical JSON, so
        # the extraction pass owns this copy rather than caller-owned mappings.
        return parse_workflow_dependency_contract(workflow_dependency_contract_payload(contract))
    except WorkflowDependencyError as exc:
        raise WorkflowLoraSlotError(exc.code, str(exc)) from exc
    except (AttributeError, RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraSlotError(
            "invalid_dependency_contract",
            "Workflow dependency evidence contains malformed typed values",
        ) from exc


def _activation_bindings(
    contract: WorkflowDependencyContract,
    activation: WorkflowActivationResolution | None,
) -> tuple[tuple[ResolvedWorkflowBinding, ...], str | None]:
    if activation is None:
        return (), None
    if not isinstance(activation, WorkflowActivationResolution):
        raise WorkflowLoraSlotError(
            "invalid_binding_evidence",
            "Workflow activation evidence is not a typed resolution",
        )
    if (
        not activation.complete
        or activation.issues
        or activation.missing_required_slots
        or not isinstance(activation.binding_sha256, str)
        or not _DIGEST.fullmatch(activation.binding_sha256)
    ):
        raise WorkflowLoraSlotError(
            "incomplete_binding_evidence",
            "Workflow activation evidence is incomplete",
        )
    try:
        digest = workflow_activation_binding_sha256(contract, activation.bindings)
    except WorkflowBindingError as exc:
        raise WorkflowLoraSlotError(exc.code, str(exc)) from exc
    if digest != activation.binding_sha256:
        raise WorkflowLoraSlotError(
            "stale_binding_evidence",
            "Workflow activation binding digest does not match its typed evidence",
        )
    return activation.bindings, digest


def _asset_index(
    contract: WorkflowDependencyContract,
    bindings: Sequence[ResolvedWorkflowBinding],
) -> dict[str, list[_AssetEvidence]]:
    contracts = {slot.name: slot for slot in contract.slots}
    result: dict[str, list[_AssetEvidence]] = {}
    for binding in bindings:
        if binding.resource_kind != "model_asset":
            continue
        identity = binding.identity
        if not isinstance(identity, dict):
            raise WorkflowLoraSlotError(
                "invalid_lora_binding_evidence",
                "Workflow model asset binding has no typed identity",
            )
        if identity.get("asset_kind") != "lora":
            continue
        try:
            runtime_reference = validate_workflow_runtime_reference(
                identity.get("runtime_reference")
            )
        except WorkflowBindingError as exc:
            raise WorkflowLoraSlotError(
                "invalid_lora_binding_evidence",
                "Workflow LoRA binding has no portable runtime reference",
            ) from exc
        digest = identity.get("sha256")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise WorkflowLoraSlotError(
                "invalid_lora_binding_evidence",
                "Workflow LoRA binding is not content-addressed",
            )
        slot = contracts[binding.slot_name]
        evidence = _AssetEvidence(
            binding=WorkflowLoraAssetBinding(
                dependency_slot=binding.slot_name,
                requirement_key=binding.requirement_key,
                resource_identity_sha256=binding.resource_identity_sha256,
                runtime_reference=runtime_reference,
                sha256=digest,
            ),
            required=slot.required,
        )
        result.setdefault(runtime_reference, []).append(evidence)
    return result


def _bounded_evidence_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 100
        and not any(character < " " or ord(character) == 127 for character in value)
    )


def _valid_package_claim(claim: PackageClaim) -> bool:
    optional_names = (claim.registry_id, claim.repository_id)
    return (
        all(
            value is None
            or (
                isinstance(value, str)
                and len(value) <= 1_000
                and not any(character < " " or ord(character) == 127 for character in value)
            )
            for value in optional_names
        )
        and isinstance(claim.revision, str)
        and len(claim.revision) <= 1_000
        and not any(character < " " or ord(character) == 127 for character in claim.revision)
    )


def _exact_comfy_runtime_identity(identity: object) -> bool:
    if not isinstance(identity, Mapping) or set(identity) != {
        "kind",
        "engine",
        "runtime_build",
        "adapter_contract_version",
        "launch_contract_version",
    }:
        return False
    runtime_build = identity.get("runtime_build")
    return (
        identity.get("kind") == "runtime"
        and identity.get("engine") == "comfyui"
        and isinstance(runtime_build, str)
        and 0 < len(runtime_build) <= 200
        and not any(character < " " or ord(character) == 127 for character in runtime_build)
        and type(identity.get("adapter_contract_version")) is int
        and identity["adapter_contract_version"] == CORE_RUNTIME_ADAPTER_CONTRACT_VERSION
        and isinstance(identity.get("launch_contract_version"), str)
        and _STABLE_IDENTITY_TEXT.fullmatch(identity["launch_contract_version"]) is not None
    )


def _core_evidence_index(
    api_graph: Mapping[str, Any],
    evidence: Sequence[WorkflowLoraCoreEvidence],
) -> dict[str, WorkflowLoraCoreEvidence]:
    if isinstance(evidence, str | bytes) or not isinstance(evidence, Sequence):
        raise WorkflowLoraSlotError(
            "invalid_core_evidence",
            "Workflow LoRA core evidence must be an array",
        )
    if len(evidence) > MAX_WORKFLOW_LORA_CORE_EVIDENCE:
        raise WorkflowLoraSlotError(
            "too_much_core_evidence",
            "Workflow declares too many LoRA core evidence records",
        )
    result: dict[str, WorkflowLoraCoreEvidence] = {}
    for item in evidence:
        if not isinstance(item, WorkflowLoraCoreEvidence):
            raise WorkflowLoraSlotError(
                "invalid_core_evidence",
                "Workflow LoRA core evidence is not typed",
            )
        if (
            not isinstance(item.node_id, str)
            or item.node_id not in api_graph
            or not isinstance(item.api_graph_sha256, str)
            or not _DIGEST.fullmatch(item.api_graph_sha256)
            or not isinstance(item.loader_contract, str)
            or not item.loader_contract
            or len(item.loader_contract) > 200
            or not isinstance(item.adapter_schema_sha256, str)
            or not _DIGEST.fullmatch(item.adapter_schema_sha256)
            or type(item.core_claimed) is not bool
            or not _bounded_evidence_name(item.dependency_slot)
            or not _bounded_evidence_name(item.requirement_key)
            or not isinstance(item.resource_identity_sha256, str)
            or not _DIGEST.fullmatch(item.resource_identity_sha256)
        ):
            raise WorkflowLoraSlotError(
                "invalid_core_evidence",
                "Workflow LoRA core evidence is malformed or orphaned",
            )
        if item.node_id in result:
            raise WorkflowLoraSlotError(
                "duplicate_core_evidence",
                "Workflow LoRA node has more than one core evidence record",
            )
        node = api_graph[item.node_id]
        if not isinstance(node, Mapping) or node.get("class_type") not in _CORE_LOADER_AUTHORITY:
            raise WorkflowLoraSlotError(
                "unexpected_core_evidence",
                "Workflow LoRA core evidence names an unsupported node",
            )
        result[item.node_id] = item
    return result


def _package_evidence_index(
    api_graph: Mapping[str, Any],
    evidence: Sequence[WorkflowLoraPackageEvidence],
) -> dict[str, WorkflowLoraPackageEvidence]:
    if isinstance(evidence, str | bytes) or not isinstance(evidence, Sequence):
        raise WorkflowLoraSlotError(
            "invalid_package_evidence",
            "Workflow LoRA package evidence must be an array",
        )
    if len(evidence) > MAX_WORKFLOW_LORA_PACKAGE_EVIDENCE:
        raise WorkflowLoraSlotError(
            "too_much_package_evidence",
            "Workflow declares too many LoRA package evidence records",
        )
    result: dict[str, WorkflowLoraPackageEvidence] = {}
    for item in evidence:
        if not isinstance(item, WorkflowLoraPackageEvidence):
            raise WorkflowLoraSlotError(
                "invalid_package_evidence",
                "Workflow LoRA package evidence is not typed",
            )
        if (
            not isinstance(item.node_id, str)
            or item.node_id not in api_graph
            or not isinstance(item.api_graph_sha256, str)
            or not _DIGEST.fullmatch(item.api_graph_sha256)
            or not isinstance(item.resource_identity_sha256, str)
            or not _DIGEST.fullmatch(item.resource_identity_sha256)
            or not isinstance(item.package_claim, PackageClaim)
            or not _valid_package_claim(item.package_claim)
            or not _bounded_evidence_name(item.dependency_slot)
            or not _bounded_evidence_name(item.requirement_key)
        ):
            raise WorkflowLoraSlotError(
                "invalid_package_evidence",
                "Workflow LoRA package evidence is malformed or orphaned",
            )
        if item.node_id in result:
            raise WorkflowLoraSlotError(
                "duplicate_package_evidence",
                "Workflow LoRA node has more than one package evidence record",
            )
        node = api_graph[item.node_id]
        if not isinstance(node, Mapping) or node.get("class_type") != POWER_LORA_LOADER:
            raise WorkflowLoraSlotError(
                "unexpected_package_evidence",
                "Workflow LoRA package evidence names an unsupported node",
            )
        result[item.node_id] = item
    return result


def _core_authority(
    node_id: str,
    loader_type: str,
    graph_sha256: str,
    evidence: WorkflowLoraCoreEvidence | None,
    bindings: Sequence[ResolvedWorkflowBinding],
) -> tuple[str | None, str | None, str | None]:
    if evidence is None:
        return None, "missing_core_evidence", None
    if evidence.node_id != node_id:
        return None, "stale_core_evidence", None
    if evidence.api_graph_sha256 != graph_sha256:
        return None, "stale_core_evidence", None
    if evidence.core_claimed is not True:
        return None, "node_not_core_claimed", None
    expected = _CORE_LOADER_AUTHORITY[loader_type]
    if evidence.loader_contract != expected[0] or evidence.adapter_schema_sha256 != expected[1]:
        return None, "core_loader_contract_mismatch", None
    matches = [
        binding
        for binding in bindings
        if binding.slot_name == evidence.dependency_slot
        and binding.requirement_key == evidence.requirement_key
    ]
    if len(matches) != 1:
        return None, "missing_runtime_binding", None
    binding = matches[0]
    if (
        binding.resource_kind != "runtime"
        or binding.resource_identity_sha256 != evidence.resource_identity_sha256
    ):
        return None, "stale_runtime_binding", None
    identity = binding.identity
    if not _exact_comfy_runtime_identity(identity):
        return None, "invalid_runtime_binding", None
    authority_payload = {
        "loader_contract": evidence.loader_contract,
        "adapter_schema_sha256": evidence.adapter_schema_sha256,
        "runtime_adapter_contract_version": CORE_RUNTIME_ADAPTER_CONTRACT_VERSION,
        "runtime_resource_identity_sha256": binding.resource_identity_sha256,
    }
    authority_sha256 = hashlib.sha256(
        json.dumps(
            authority_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return evidence.loader_contract, None, authority_sha256


def _rgthree_authority(
    node_id: str,
    graph_sha256: str,
    evidence: WorkflowLoraPackageEvidence | None,
    bindings: Sequence[ResolvedWorkflowBinding],
) -> tuple[str | None, str | None, str | None]:
    if evidence is None:
        return None, "missing_package_evidence", None
    if evidence.api_graph_sha256 != graph_sha256:
        return None, "stale_package_evidence", None
    matches = [
        binding
        for binding in bindings
        if binding.slot_name == evidence.dependency_slot
        and binding.requirement_key == evidence.requirement_key
    ]
    if len(matches) != 1:
        return None, "missing_package_binding", None
    binding = matches[0]
    if (
        binding.resource_kind != "registry_package"
        or binding.resource_identity_sha256 != evidence.resource_identity_sha256
    ):
        return None, "stale_package_binding", None
    identity = binding.identity
    if not isinstance(identity, dict) or not _exact_registry_package_identity(identity):
        return None, "invalid_package_binding", None
    claim = evidence.package_claim
    node_types = identity.get("node_types")
    if (
        identity.get("package_id") != claim.registry_id
        or identity.get("package_version") != claim.revision
        or not isinstance(node_types, list)
        or POWER_LORA_LOADER not in node_types
    ):
        return None, "package_binding_mismatch", None
    try:
        contract = package_widget_contract(POWER_LORA_LOADER, claim)
    except PackageWidgetError as exc:
        return None, exc.code, None
    if contract != POWER_LORA_LOADER_CONTRACT:
        return None, "unsupported_package_contract", None
    return contract, None, binding.resource_identity_sha256


def _exact_registry_package_identity(identity: Mapping[str, Any]) -> bool:
    digest_fields = (
        "archive_sha256",
        "manifest_sha256",
        "wheel_closure_sha256",
        "wheel_environment_sha256",
    )
    node_types = identity.get("node_types")
    registry_record = identity.get("registry_record_id")
    return (
        identity.get("kind") == "registry_package"
        and isinstance(registry_record, str)
        and 0 < len(registry_record) <= 1_000
        and isinstance(node_types, list)
        and bool(node_types)
        and all(isinstance(node_type, str) and node_type for node_type in node_types)
        and all(
            isinstance(identity.get(field), str) and _DIGEST.fullmatch(identity[field]) is not None
            for field in digest_fields
        )
    )


def _core_loras(
    graph: Mapping[str, Any],
    node_id: str,
    node: Mapping[str, Any],
    loader_type: str,
    loader_contract: str | None,
    authority_reason: str | None,
    loader_authority_sha256: str | None,
) -> tuple[_ParsedLora, ...]:
    expected_contract = _CORE_LOADER_AUTHORITY[loader_type][0]
    if loader_contract != expected_contract:
        return _detected_loras(
            node,
            loader_type,
            authority_reason or "unsupported_core_contract",
            include_placeholder=False,
        )
    try:
        inputs = _exact_node_inputs(node)
        model = inputs.get("model")
        _require_link(graph, node_id, model)
        runtime_reference = _required_runtime_reference(inputs.get("lora_name"))
        model_strength = _finite_strength(inputs.get("strength_model"))
        if loader_type == _CORE_MODEL_ONLY_LORA_LOADER:
            if set(inputs) != {"model", "lora_name", "strength_model"}:
                raise _LayoutError("invalid_loader_layout")
            return (
                _ParsedLora(
                    locator="lora_name",
                    runtime_reference=runtime_reference,
                    loader_contract=loader_contract,
                    loader_authority_sha256=loader_authority_sha256,
                    exact=True,
                    read_only_reason=None,
                    enabled=True,
                    model_strength=model_strength,
                    clip_strength=None,
                    strength_mode="model_only",
                    editable_fields=("model_strength",),
                ),
            )
        if set(inputs) != {
            "model",
            "clip",
            "lora_name",
            "strength_model",
            "strength_clip",
        }:
            raise _LayoutError("invalid_loader_layout")
        _require_link(graph, node_id, inputs.get("clip"))
        return (
            _ParsedLora(
                locator="lora_name",
                runtime_reference=runtime_reference,
                loader_contract=loader_contract,
                loader_authority_sha256=loader_authority_sha256,
                exact=True,
                read_only_reason=None,
                enabled=True,
                model_strength=model_strength,
                clip_strength=_finite_strength(inputs.get("strength_clip")),
                strength_mode="separate",
                editable_fields=("model_strength", "clip_strength"),
            ),
        )
    except _LayoutError as exc:
        return _detected_loras(node, loader_type, exc.code, include_placeholder=False)


def _rgthree_loras(
    graph: Mapping[str, Any],
    node_id: str,
    node: Mapping[str, Any],
    package_contract: str | None,
    package_reason: str | None,
    package_authority_sha256: str | None,
) -> tuple[_ParsedLora, ...]:
    if package_contract != POWER_LORA_LOADER_CONTRACT:
        return _detected_loras(
            node,
            POWER_LORA_LOADER,
            package_reason or "unsupported_package_contract",
            include_placeholder=False,
        )
    try:
        inputs = _exact_node_inputs(node)
        for link_name in ("model", "clip"):
            if link_name in inputs:
                _require_link(graph, node_id, inputs[link_name])
        entries: list[tuple[int, str, Mapping[str, Any]]] = []
        for name, value in inputs.items():
            if name in {"model", "clip"}:
                continue
            match = _RGTHREE_ENTRY.fullmatch(name)
            if match is None or not isinstance(value, Mapping):
                raise _LayoutError("invalid_loader_layout")
            entries.append((int(match.group(1)), name, value))
        entries.sort()
        if [index for index, _, _ in entries] != list(range(1, len(entries) + 1)):
            raise _LayoutError("invalid_loader_layout")
        return tuple(
            _rgthree_entry(
                name,
                value,
                package_contract,
                package_authority_sha256,
            )
            for _, name, value in entries
        )
    except _LayoutError as exc:
        return _detected_loras(
            node,
            POWER_LORA_LOADER,
            exc.code,
            include_placeholder=False,
        )


def _rgthree_entry(
    name: str,
    value: Mapping[str, Any],
    package_contract: str,
    package_authority_sha256: str | None,
) -> _ParsedLora:
    keys = set(value)
    if not {"on", "lora", "strength"} <= keys or keys - {
        "on",
        "lora",
        "strength",
        "strengthTwo",
    }:
        raise _LayoutError("invalid_loader_layout")
    enabled = value.get("on")
    if not isinstance(enabled, bool):
        raise _LayoutError("invalid_loader_layout")
    runtime_reference = _required_runtime_reference(value.get("lora"))
    model_strength = _finite_strength(value.get("strength"))
    clip_value = value.get("strengthTwo")
    if clip_value is None:
        clip_strength = model_strength
        strength_mode: WorkflowLoraStrengthMode = "coupled"
        fields: tuple[WorkflowLoraEditableField, ...] = (
            "enabled",
            "model_strength",
        )
    else:
        clip_strength = _finite_strength(clip_value)
        strength_mode = "separate"
        fields = ("enabled", "model_strength", "clip_strength")
    return _ParsedLora(
        locator=name,
        runtime_reference=runtime_reference,
        loader_contract=package_contract,
        loader_authority_sha256=package_authority_sha256,
        exact=True,
        read_only_reason=None,
        enabled=enabled,
        model_strength=model_strength,
        clip_strength=clip_strength,
        strength_mode=strength_mode,
        editable_fields=fields,
    )


def _exact_node_inputs(node: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(node) - _NODE_FIELDS or node.get("class_type") is None:
        raise _LayoutError("invalid_loader_layout")
    inputs = node.get("inputs")
    if not isinstance(inputs, Mapping):
        raise _LayoutError("invalid_loader_layout")
    if any(not isinstance(key, str) for key in inputs):
        raise _LayoutError("invalid_loader_layout")
    return inputs


def _require_link(
    graph: Mapping[str, Any],
    node_id: str,
    value: object,
) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not isinstance(value[0], str)
        or value[0] == node_id
        or value[0] not in graph
        or isinstance(value[1], bool)
        or not isinstance(value[1], int)
        or value[1] < 0
    ):
        raise _LayoutError("invalid_loader_link")


def _required_runtime_reference(value: object) -> str:
    try:
        return validate_workflow_runtime_reference(value)
    except WorkflowBindingError as exc:
        raise _LayoutError("invalid_loader_reference") from exc


def _safe_runtime_reference(value: object) -> str | None:
    try:
        return validate_workflow_runtime_reference(value)
    except WorkflowBindingError:
        return None


def _finite_strength(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _LayoutError("invalid_loader_strength")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise _LayoutError("invalid_loader_strength") from exc
    if not math.isfinite(result):
        raise _LayoutError("invalid_loader_strength")
    return result


def _detected_loras(
    node: Mapping[str, Any],
    loader_type: str,
    reason: str,
    *,
    include_placeholder: bool = True,
) -> tuple[_ParsedLora, ...]:
    inputs = node.get("inputs")
    candidates: list[tuple[str, str | None]] = []
    mentions_lora = False
    if isinstance(inputs, Mapping):
        if "lora_name" in inputs:
            mentions_lora = True
            candidates.append(("lora_name", _safe_runtime_reference(inputs.get("lora_name"))))
        for name in sorted(key for key in inputs if isinstance(key, str)):
            if name == "lora_name":
                continue
            value = inputs[name]
            match = _RGTHREE_ENTRY.fullmatch(name)
            if match is not None:
                mentions_lora = True
                reference = (
                    _safe_runtime_reference(value.get("lora"))
                    if isinstance(value, Mapping)
                    else None
                )
                candidates.append((name, reference))
            elif "lora" in name.casefold():
                mentions_lora = True
    if not candidates and mentions_lora and include_placeholder:
        candidates.append(("detected", None))
    return tuple(
        _ParsedLora(
            locator=locator,
            runtime_reference=runtime_reference,
            loader_contract=None,
            loader_authority_sha256=None,
            exact=False,
            read_only_reason=reason,
            enabled=None,
            model_strength=None,
            clip_strength=None,
            strength_mode="unknown",
            editable_fields=(),
        )
        for locator, runtime_reference in candidates
    )


def _one_asset(
    assets: Mapping[str, list[_AssetEvidence]],
    runtime_reference: str | None,
) -> _AssetEvidence | None:
    if runtime_reference is None:
        return None
    matches = assets.get(runtime_reference, [])
    if len(matches) > 1:
        raise WorkflowLoraSlotError(
            "ambiguous_lora_binding",
            "One workflow LoRA loader name resolves to more than one typed binding",
        )
    return matches[0] if matches else None


def _public_slot(
    *,
    revision_scope: str,
    graph_sha256: str,
    node_id: str,
    loader_type: str,
    parsed: _ParsedLora,
    asset: _AssetEvidence | None,
    position: int,
) -> WorkflowLoraSlot:
    editable_fields = parsed.editable_fields
    editability: WorkflowLoraEditability
    read_only_reason = parsed.read_only_reason
    if parsed.exact and asset is not None:
        if asset.required:
            editability = "required_locked"
            editable_fields = ()
            read_only_reason = "required_workflow_dependency"
        elif editable_fields:
            editability = "editable"
            read_only_reason = None
        else:
            editability = "detected_read_only"
            read_only_reason = "no_safe_edit_fields"
    else:
        editability = "detected_read_only"
        editable_fields = ()
        if parsed.exact:
            read_only_reason = "missing_asset_binding"

    binding = asset.binding if asset is not None else None
    slot_id = _opaque_slot_id(
        revision_scope=revision_scope,
        graph_sha256=graph_sha256,
        node_id=node_id,
        locator=parsed.locator,
    )
    return WorkflowLoraSlot(
        slot_id=slot_id,
        position=position,
        loader_type=(loader_type if _PUBLIC_LOADER_TYPE.fullmatch(loader_type) else "unrecognized"),
        loader_contract=parsed.loader_contract if parsed.exact else None,
        loader_authority_sha256=(parsed.loader_authority_sha256 if parsed.exact else None),
        editability=editability,
        read_only_reason=read_only_reason,
        dependency_required=asset.required if asset is not None else None,
        observed_runtime_reference=parsed.runtime_reference,
        asset_binding=binding,
        default_enabled=parsed.enabled,
        default_model_strength=parsed.model_strength,
        default_clip_strength=parsed.clip_strength,
        strength_mode=parsed.strength_mode,
        editable_fields=editable_fields,
    )


def _opaque_slot_id(
    *,
    revision_scope: str,
    graph_sha256: str,
    node_id: str,
    locator: str,
) -> str:
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
