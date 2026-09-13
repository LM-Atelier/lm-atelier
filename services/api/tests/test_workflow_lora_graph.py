from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any, cast

import pytest

from local_lm.comfy_package_widgets import (
    POWER_LORA_LOADER,
    POWER_LORA_LOADER_CONTRACT,
    PackageClaim,
)
from local_lm.workflow_bindings import (
    ResolvedWorkflowBinding,
    WorkflowActivationResolution,
    workflow_activation_binding_sha256,
    workflow_resource_identity_sha256,
)
from local_lm.workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyRequirement,
    WorkflowDependencySlotContract,
)
from local_lm.workflow_lora_graph import (
    AppliedWorkflowLoraField,
    AppliedWorkflowLoraOverride,
    WorkflowLoraGraphError,
    WorkflowLoraGraphResolution,
    apply_workflow_lora_overrides,
    workflow_lora_effective_api_graph,
    workflow_lora_graph_resolution_payload,
    workflow_lora_graph_resolution_sha256,
)
from local_lm.workflow_lora_overrides import (
    ResolvedWorkflowLoraField,
    ResolvedWorkflowLoraOverride,
    WorkflowLoraOverrideCatalog,
    WorkflowLoraOverrideError,
    WorkflowLoraOverrideLayer,
    WorkflowLoraOverrideResolution,
    WorkflowLoraOverrideTargetWitness,
    parse_workflow_lora_overrides,
    resolve_workflow_lora_override_layers,
    workflow_lora_override_resolution_sha256,
)
from local_lm.workflow_lora_slots import (
    CORE_LORA_LOADER_CONTRACT,
    CORE_MODEL_ONLY_LORA_LOADER_CONTRACT,
    WorkflowLoraAssetBinding,
    WorkflowLoraPackageEvidence,
    WorkflowLoraPrivateEditTarget,
    WorkflowLoraSlot,
    WorkflowLoraSlotExtraction,
    WorkflowLoraSlotExtractionWithPrivateTargets,
    extract_workflow_lora_slots_with_private_targets,
)
from local_lm.workflow_trust import canonical_graph

_MISSING = object()


def _digest(index: int) -> str:
    return f"{index:064x}"


def _graph_sha256(graph: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_graph(graph).encode("utf-8")).hexdigest()


def _slot_id(
    *,
    revision_scope: str,
    graph_sha256: str,
    node_id: str,
    locator: str,
) -> str:
    payload = {
        "version": 1,
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


@dataclass(frozen=True, slots=True)
class _Case:
    graph: dict[str, Any]
    catalog: WorkflowLoraOverrideCatalog
    extraction: WorkflowLoraSlotExtractionWithPrivateTargets


_RGTHREE_REVISION = "6b76ee6f2c5a007710b5a16f97c94330d6ecc871"


def _dependency_slot(
    name: str,
    resource_kind: str,
    *,
    required: bool = True,
) -> WorkflowDependencySlotContract:
    return WorkflowDependencySlotContract(
        name=name,
        resource_kind=cast(Any, resource_kind),
        required=required,
        satisfaction="any_of",
        requirements=(WorkflowDependencyRequirement("default", {}),),
    )


def _resolved_binding(
    *,
    slot_name: str,
    resource_kind: str,
    identity: dict[str, Any],
) -> ResolvedWorkflowBinding:
    return ResolvedWorkflowBinding(
        slot_name=slot_name,
        requirement_key="default",
        resource_kind=cast(Any, resource_kind),
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256(
            cast(Any, resource_kind),
            identity,
        ),
        mount={},
    )


def _authentic_rgthree_case(*, include_read_only: bool = False) -> _Case:
    revision_scope = "wfrev-authentic-rgthree"
    runtime_reference = "styles/shared.safetensors"
    graph = {
        "source": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "model.safetensors"},
        },
        "private-power-node": {
            "class_type": POWER_LORA_LOADER,
            "inputs": {
                "model": ["source", 0],
                "clip": ["source", 1],
                "lora_1": {
                    "on": True,
                    "lora": runtime_reference,
                    "strength": 0.8,
                    "strengthTwo": 0.7,
                },
                "lora_2": {
                    "on": True,
                    "lora": runtime_reference,
                    "strength": 0.8,
                    "strengthTwo": 0.7,
                },
            },
        },
    }
    if include_read_only:
        graph["future-read-only-node"] = {
            "class_type": "FutureLoraLoader",
            "inputs": {
                "model": ["source", 0],
                "lora_name": "styles/future.safetensors",
                "strength_model": 0.5,
            },
        }
    contract = WorkflowDependencyContract(
        version=1,
        slots=(
            _dependency_slot("rgthree", "registry_package"),
            _dependency_slot("styles", "model_asset", required=False),
        ),
    )
    package_identity = {
        "kind": "registry_package",
        "package_id": "rgthree-comfy",
        "package_version": _RGTHREE_REVISION,
        "registry_record_id": f"rgthree-comfy@{_RGTHREE_REVISION}",
        "node_types": [POWER_LORA_LOADER],
        "archive_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "wheel_closure_sha256": "3" * 64,
        "wheel_environment_sha256": "4" * 64,
    }
    package_binding = _resolved_binding(
        slot_name="rgthree",
        resource_kind="registry_package",
        identity=package_identity,
    )
    asset_identity = {
        "kind": "model_asset",
        "asset_kind": "lora",
        "runtime_reference": runtime_reference,
        "sha256": "a" * 64,
    }
    asset_binding = _resolved_binding(
        slot_name="styles",
        resource_kind="model_asset",
        identity=asset_identity,
    )
    bindings = (package_binding, asset_binding)
    activation = WorkflowActivationResolution(
        bindings=bindings,
        issues=(),
        missing_required_slots=(),
        complete=True,
        binding_sha256=workflow_activation_binding_sha256(contract, bindings),
    )
    evidence = WorkflowLoraPackageEvidence(
        api_graph_sha256=_graph_sha256(graph),
        node_id="private-power-node",
        package_claim=PackageClaim(
            registry_id="rgthree-comfy",
            repository_id="rgthree/rgthree-comfy",
            revision=_RGTHREE_REVISION,
        ),
        dependency_slot="rgthree",
        requirement_key="default",
        resource_identity_sha256=package_binding.resource_identity_sha256,
    )
    extraction = extract_workflow_lora_slots_with_private_targets(
        revision_scope=revision_scope,
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    )
    witness = WorkflowLoraOverrideTargetWitness(
        workflow_family_id="family-image",
        workflow_definition_id="workflow-image",
        workflow_variant_key="text_to_image",
        workflow_revision_id=revision_scope,
        slot_contract_version=1,
        revision_scope_sha256=extraction.public.revision_scope_sha256,
        api_graph_sha256=extraction.public.api_graph_sha256,
        dependency_contract_sha256=extraction.public.dependency_contract_sha256,
        activation_binding_sha256=cast(str, extraction.public.activation_binding_sha256),
        activation_witness_sha256=_digest(90),
    )
    return _Case(
        graph,
        WorkflowLoraOverrideCatalog(witness, extraction.public.slots),
        extraction,
    )


def _case(
    *,
    loader_contract: str = CORE_LORA_LOADER_CONTRACT,
    rgthree_clip: object = 0.6,
    enabled: bool = True,
) -> _Case:
    editable_fields: tuple[str, ...]
    if loader_contract == CORE_LORA_LOADER_CONTRACT:
        loader_type = "LoraLoader"
        node_id = "private-core-node"
        locator = "lora_name"
        inputs: dict[str, Any] = {
            "model": ["source", 0],
            "clip": ["source", 1],
            "lora_name": "styles/identity.safetensors",
            "strength_model": 0.8,
            "strength_clip": 0.7,
        }
        editable_fields = ("model_strength", "clip_strength")
        strength_mode = "separate"
        default_enabled = True
        default_model = 0.8
        default_clip = 0.7
    elif loader_contract == CORE_MODEL_ONLY_LORA_LOADER_CONTRACT:
        loader_type = "LoraLoaderModelOnly"
        node_id = "private-model-node"
        locator = "lora_name"
        inputs = {
            "model": ["source", 0],
            "lora_name": "styles/identity.safetensors",
            "strength_model": 0.8,
        }
        editable_fields = ("model_strength",)
        strength_mode = "model_only"
        default_enabled = True
        default_model = 0.8
        default_clip = None
    else:
        assert loader_contract == POWER_LORA_LOADER_CONTRACT
        loader_type = POWER_LORA_LOADER
        node_id = "private-power-node"
        locator = "lora_1"
        entry: dict[str, Any] = {
            "on": enabled,
            "lora": "styles/identity.safetensors",
            "strength": 0.8,
        }
        if rgthree_clip is not _MISSING:
            entry["strengthTwo"] = rgthree_clip
        inputs = {
            "model": ["source", 0],
            "clip": ["source", 1],
            "lora_1": entry,
        }
        default_enabled = enabled
        default_model = 0.8
        if rgthree_clip is _MISSING or rgthree_clip is None:
            editable_fields = ("enabled", "model_strength")
            strength_mode = "coupled"
            default_clip = 0.8
        else:
            editable_fields = ("enabled", "model_strength", "clip_strength")
            strength_mode = "separate"
            default_clip = float(cast(float, rgthree_clip))
    graph = {
        "source": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "model.safetensors"},
        },
        node_id: {
            "class_type": loader_type,
            "inputs": inputs,
            "_meta": {"title": "Authored LoRA"},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {"model": [node_id, 0], "seed": 1},
        },
    }
    graph_sha256 = _graph_sha256(graph)
    revision_scope = "wfrev-image"
    slot_id = _slot_id(
        revision_scope=revision_scope,
        graph_sha256=graph_sha256,
        node_id=node_id,
        locator=locator,
    )
    authority = _digest(12)
    asset_sha256 = _digest(13)
    binding = WorkflowLoraAssetBinding(
        dependency_slot="lora-assets",
        requirement_key="identity",
        resource_identity_sha256=_digest(14),
        runtime_reference="styles/identity.safetensors",
        sha256=asset_sha256,
    )
    slot = WorkflowLoraSlot(
        slot_id=slot_id,
        position=0,
        loader_type=loader_type,
        loader_contract=loader_contract,
        loader_authority_sha256=authority,
        editability="editable",
        read_only_reason=None,
        dependency_required=False,
        observed_runtime_reference=binding.runtime_reference,
        asset_binding=binding,
        default_enabled=default_enabled,
        default_model_strength=default_model,
        default_clip_strength=default_clip,
        strength_mode=cast(Any, strength_mode),
        editable_fields=cast(Any, editable_fields),
    )
    witness = WorkflowLoraOverrideTargetWitness(
        workflow_family_id="family-image",
        workflow_definition_id="workflow-image",
        workflow_variant_key="text_to_image",
        workflow_revision_id=revision_scope,
        slot_contract_version=1,
        revision_scope_sha256=hashlib.sha256(revision_scope.encode("utf-8")).hexdigest(),
        api_graph_sha256=graph_sha256,
        dependency_contract_sha256=_digest(16),
        activation_binding_sha256=_digest(17),
        activation_witness_sha256=_digest(18),
    )
    public = WorkflowLoraSlotExtraction(
        version=1,
        revision_scope_sha256=witness.revision_scope_sha256,
        api_graph_sha256=graph_sha256,
        dependency_contract_sha256=witness.dependency_contract_sha256,
        activation_binding_sha256=witness.activation_binding_sha256,
        ordering_authority="presentation_only",
        slots=(slot,),
    )
    target = WorkflowLoraPrivateEditTarget(
        slot_id=slot_id,
        source_api_graph_sha256=graph_sha256,
        source_dependency_contract_sha256=witness.dependency_contract_sha256,
        source_activation_binding_sha256=witness.activation_binding_sha256,
        source_authority_evidence_sha256=_digest(19),
        loader_authority_sha256=authority,
        asset_sha256=asset_sha256,
        node_id=node_id,
        entry_locator=locator,
        loader_contract=loader_contract,
        editable_fields=cast(Any, editable_fields),
    )
    return _Case(
        graph=graph,
        catalog=WorkflowLoraOverrideCatalog(witness, (slot,)),
        extraction=WorkflowLoraSlotExtractionWithPrivateTargets(public, (target,)),
    )


def _target_payload(case: _Case, changes: dict[str, object]) -> dict[str, object]:
    witness = case.catalog.target
    slot = next(slot for slot in case.catalog.slots if slot.editability == "editable")
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
        "overrides": [
            {
                "slot_id": slot.slot_id,
                "loader_contract": slot.loader_contract,
                "loader_authority_sha256": slot.loader_authority_sha256,
                "changes": changes,
            }
        ],
    }


def _resolution(
    case: _Case,
    changes: dict[str, object],
    *,
    origin: str = "turn",
) -> WorkflowLoraOverrideResolution:
    envelope = parse_workflow_lora_overrides(
        {"version": 1, "targets": [_target_payload(case, changes)]}
    )
    return resolve_workflow_lora_override_layers(
        catalog=case.catalog,
        layers=(WorkflowLoraOverrideLayer(cast(Any, origin), envelope),),
    )


def test_core_full_patch_changes_only_two_scalar_leaves_and_preserves_source() -> None:
    case = _case()
    original = deepcopy(case.graph)
    resolution = _resolution(
        case,
        {"clip_strength": 0.4, "model_strength": 1.25},
        origin="chat",
    )

    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=resolution,
    )
    effective = workflow_lora_effective_api_graph(result)

    assert case.graph == original
    assert effective["private-core-node"]["inputs"]["strength_model"] == 1.25
    assert effective["private-core-node"]["inputs"]["strength_clip"] == 0.4
    assert (
        effective["private-core-node"]["inputs"]["lora_name"]
        == (original["private-core-node"]["inputs"]["lora_name"])
    )
    assert effective["private-core-node"]["inputs"]["model"] == ["source", 0]
    assert effective["sampler"] == original["sampler"]
    assert result.source_api_graph_sha256 == _graph_sha256(original)
    assert result.workflow_effective_graph_sha256 == _graph_sha256(effective)
    assert result.override_resolution_sha256 == workflow_lora_override_resolution_sha256(resolution)
    assert [(change.field, change.origin) for change in result.overrides[0].changes] == [
        ("model_strength", "chat"),
        ("clip_strength", "chat"),
    ]


def test_core_model_only_patch_has_no_clip_or_toggle_surface() -> None:
    case = _case(loader_contract=CORE_MODEL_ONLY_LORA_LOADER_CONTRACT)

    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=_resolution(case, {"model_strength": -0.5}),
    )
    effective = workflow_lora_effective_api_graph(result)

    assert effective["private-model-node"]["inputs"] == {
        "model": ["source", 0],
        "lora_name": "styles/identity.safetensors",
        "strength_model": -0.5,
    }
    assert [change.field for change in result.overrides[0].changes] == ["model_strength"]


def test_rgthree_separate_patch_disables_and_changes_both_strengths() -> None:
    case = _case(loader_contract=POWER_LORA_LOADER_CONTRACT, rgthree_clip=0.6)

    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=_resolution(
            case,
            {"clip_strength": 0.25, "enabled": False, "model_strength": 1.5},
        ),
    )
    entry = workflow_lora_effective_api_graph(result)["private-power-node"]["inputs"]["lora_1"]

    assert entry == {
        "on": False,
        "lora": "styles/identity.safetensors",
        "strength": 1.5,
        "strengthTwo": 0.25,
    }
    assert [change.field for change in result.overrides[0].changes] == [
        "enabled",
        "model_strength",
        "clip_strength",
    ]


@pytest.mark.parametrize("clip_value", [_MISSING, None])
def test_rgthree_coupled_patch_preserves_missing_vs_null_clip_leaf(
    clip_value: object,
) -> None:
    case = _case(loader_contract=POWER_LORA_LOADER_CONTRACT, rgthree_clip=clip_value)
    before = deepcopy(case.graph["private-power-node"]["inputs"]["lora_1"])

    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=_resolution(case, {"model_strength": 0.35}),
    )
    entry = workflow_lora_effective_api_graph(result)["private-power-node"]["inputs"]["lora_1"]

    assert entry["strength"] == 0.35
    assert ("strengthTwo" in entry) == ("strengthTwo" in before)
    if "strengthTwo" in before:
        assert entry["strengthTwo"] is None


def test_explicit_authored_value_keeps_origin_without_graph_byte_change() -> None:
    case = _case()
    resolution = _resolution(case, {"model_strength": 0.8}, origin="project")

    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=resolution,
    )

    assert result.source_api_graph_sha256 == result.workflow_effective_graph_sha256
    change = result.overrides[0].changes[0]
    assert change == AppliedWorkflowLoraField("model_strength", 0.8, 0.8, "project")
    assert workflow_lora_effective_api_graph(result) == case.graph


def test_empty_resolution_returns_a_fresh_unchanged_graph() -> None:
    case = _case()
    resolution = resolve_workflow_lora_override_layers(catalog=case.catalog, layers=())

    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=resolution,
    )
    first = workflow_lora_effective_api_graph(result)
    first["source"]["inputs"]["ckpt_name"] = "changed"

    assert workflow_lora_effective_api_graph(result) == case.graph
    assert result.overrides == ()
    assert result.source_api_graph_sha256 == result.workflow_effective_graph_sha256


def _replace_extraction(
    case: _Case,
    *,
    public: WorkflowLoraSlotExtraction | None = None,
    targets: tuple[WorkflowLoraPrivateEditTarget, ...] | None = None,
) -> WorkflowLoraSlotExtractionWithPrivateTargets:
    return WorkflowLoraSlotExtractionWithPrivateTargets(
        public or case.extraction.public,
        case.extraction.edit_targets if targets is None else targets,
    )


def test_one_rgthree_occurrence_patch_does_not_touch_its_sibling() -> None:
    base = _case(loader_contract=POWER_LORA_LOADER_CONTRACT, rgthree_clip=0.6)
    graph = deepcopy(base.graph)
    graph["private-power-node"]["inputs"]["lora_2"] = {
        "on": True,
        "lora": "styles/second.safetensors",
        "strength": 0.45,
    }
    graph_sha256 = _graph_sha256(graph)
    first = replace(
        base.catalog.slots[0],
        slot_id=_slot_id(
            revision_scope=base.catalog.target.workflow_revision_id,
            graph_sha256=graph_sha256,
            node_id="private-power-node",
            locator="lora_1",
        ),
    )
    second_binding = replace(
        cast(WorkflowLoraAssetBinding, first.asset_binding),
        requirement_key="second",
        runtime_reference="styles/second.safetensors",
        sha256=_digest(31),
    )
    second = replace(
        first,
        slot_id=_slot_id(
            revision_scope=base.catalog.target.workflow_revision_id,
            graph_sha256=graph_sha256,
            node_id="private-power-node",
            locator="lora_2",
        ),
        position=1,
        observed_runtime_reference=second_binding.runtime_reference,
        asset_binding=second_binding,
        default_model_strength=0.45,
        default_clip_strength=0.45,
        strength_mode="coupled",
        editable_fields=("enabled", "model_strength"),
    )
    witness = replace(base.catalog.target, api_graph_sha256=graph_sha256)
    public = replace(
        base.extraction.public,
        api_graph_sha256=graph_sha256,
        slots=(first, second),
    )
    first_target = replace(
        base.extraction.edit_targets[0],
        slot_id=first.slot_id,
        source_api_graph_sha256=graph_sha256,
    )
    second_target = replace(
        first_target,
        slot_id=second.slot_id,
        asset_sha256=second_binding.sha256,
        entry_locator="lora_2",
        editable_fields=second.editable_fields,
    )
    case = _Case(
        graph,
        WorkflowLoraOverrideCatalog(witness, (first, second)),
        WorkflowLoraSlotExtractionWithPrivateTargets(
            public,
            (first_target, second_target),
        ),
    )

    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=_resolution(case, {"model_strength": 1.1}),
    )
    entries = workflow_lora_effective_api_graph(result)["private-power-node"]["inputs"]

    assert entries["lora_1"]["strength"] == 1.1
    assert entries["lora_2"] == graph["private-power-node"]["inputs"]["lora_2"]


@pytest.mark.parametrize("mode", ["swap", "alias"])
def test_authentic_private_targets_bind_each_opaque_slot_to_its_exact_occurrence(
    mode: str,
) -> None:
    case = _authentic_rgthree_case()
    first, second = case.extraction.edit_targets
    if mode == "swap":
        targets = (
            replace(first, entry_locator=second.entry_locator),
            replace(second, entry_locator=first.entry_locator),
        )
    else:
        targets = (first, replace(second, entry_locator=first.entry_locator))
    original = deepcopy(case.graph)

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=case.graph,
            catalog=case.catalog,
            extraction=replace(case.extraction, edit_targets=targets),
            resolution=_resolution(case, {"model_strength": 1.1}),
        )

    assert raised.value.code == "invalid_workflow_lora_private_targets"
    assert case.graph == original
    entries = case.graph["private-power-node"]["inputs"]
    assert entries["lora_1"]["strength"] == 0.8
    assert entries["lora_2"]["strength"] == 0.8


def test_authentic_mixed_editable_and_unknown_read_only_catalog_applies_editable_slot() -> None:
    case = _authentic_rgthree_case(include_read_only=True)
    read_only = [slot for slot in case.catalog.slots if slot.editability == "detected_read_only"]
    assert len(read_only) == 1
    assert read_only[0].loader_contract is None
    assert read_only[0].loader_authority_sha256 is None

    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=_resolution(case, {"model_strength": 1.1}),
    )
    effective = workflow_lora_effective_api_graph(result)

    assert effective["private-power-node"]["inputs"]["lora_1"]["strength"] == 1.1
    assert effective["future-read-only-node"] == case.graph["future-read-only-node"]


def test_coupled_slot_refuses_a_clip_override_before_graph_application() -> None:
    case = _case(loader_contract=POWER_LORA_LOADER_CONTRACT, rgthree_clip=None)

    with pytest.raises(WorkflowLoraOverrideError) as raised:
        _resolution(case, {"clip_strength": 0.4})

    assert raised.value.code == "unsupported_workflow_lora_override_field"


def test_authored_disabled_rgthree_cannot_be_enabled() -> None:
    case = _case(
        loader_contract=POWER_LORA_LOADER_CONTRACT,
        rgthree_clip=0.6,
        enabled=False,
    )

    with pytest.raises(WorkflowLoraOverrideError) as raised:
        _resolution(case, {"enabled": True})

    assert raised.value.code == "workflow_lora_override_enable_unsupported"


@pytest.mark.parametrize("enabled_value", [False, True])
def test_graph_boundary_refuses_any_manually_constructed_enabled_field_for_authored_disabled_slot(
    enabled_value: bool,
) -> None:
    case = _case(
        loader_contract=POWER_LORA_LOADER_CONTRACT,
        rgthree_clip=0.6,
        enabled=False,
    )
    slot = case.catalog.slots[0]
    hostile = WorkflowLoraOverrideResolution(
        version=1,
        target=case.catalog.target,
        overrides=(
            ResolvedWorkflowLoraOverride(
                slot_id=slot.slot_id,
                loader_contract=cast(str, slot.loader_contract),
                loader_authority_sha256=cast(str, slot.loader_authority_sha256),
                changes=(ResolvedWorkflowLoraField("enabled", enabled_value, "turn"),),
            ),
        ),
        inactive_targets=(),
    )

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=case.graph,
            catalog=case.catalog,
            extraction=case.extraction,
            resolution=hostile,
        )

    assert raised.value.code == "workflow_lora_override_enable_unsupported"
    assert case.graph["private-power-node"]["inputs"]["lora_1"]["on"] is False


@pytest.mark.parametrize(
    "loader_contract",
    [CORE_LORA_LOADER_CONTRACT, POWER_LORA_LOADER_CONTRACT],
)
def test_sparse_patch_refuses_stale_unmentioned_authored_slot_fields(
    loader_contract: str,
) -> None:
    case = _case(loader_contract=loader_contract)
    if loader_contract == CORE_LORA_LOADER_CONTRACT:
        changed_slot = replace(case.catalog.slots[0], default_clip_strength=0.123)
    else:
        changed_slot = replace(case.catalog.slots[0], default_enabled=False)
    public = replace(case.extraction.public, slots=(changed_slot,))
    changed = _Case(
        case.graph,
        WorkflowLoraOverrideCatalog(case.catalog.target, (changed_slot,)),
        replace(case.extraction, public=public),
    )

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=changed.graph,
            catalog=changed.catalog,
            extraction=changed.extraction,
            resolution=_resolution(changed, {"model_strength": 0.4}),
        )

    assert raised.value.code == "changed_workflow_lora_authored_value"


def test_patch_refuses_public_loader_type_drift_from_private_graph_target() -> None:
    case = _case()
    changed_slot = replace(case.catalog.slots[0], loader_type="NotTheAuthoredLoader")
    changed = _Case(
        case.graph,
        WorkflowLoraOverrideCatalog(case.catalog.target, (changed_slot,)),
        replace(
            case.extraction,
            public=replace(case.extraction.public, slots=(changed_slot,)),
        ),
    )

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=changed.graph,
            catalog=changed.catalog,
            extraction=changed.extraction,
            resolution=_resolution(changed, {"model_strength": 0.4}),
        )

    assert raised.value.code == "invalid_workflow_lora_patch_target"


def test_changed_source_graph_refuses_before_any_output() -> None:
    case = _case()
    changed = deepcopy(case.graph)
    changed["private-core-node"]["inputs"]["strength_model"] = 0.9
    original = deepcopy(changed)

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=changed,
            catalog=case.catalog,
            extraction=case.extraction,
            resolution=_resolution(case, {"model_strength": 0.4}),
        )

    assert raised.value.code == "stale_workflow_lora_graph_source"
    assert changed == original


@pytest.mark.parametrize(
    "mutate",
    [
        lambda target: replace(target, node_id="sampler"),
        lambda target: replace(target, entry_locator="strength_model"),
        lambda target: replace(target, loader_contract=POWER_LORA_LOADER_CONTRACT),
        lambda target: replace(target, loader_authority_sha256=_digest(41)),
        lambda target: replace(target, asset_sha256=_digest(42)),
        lambda target: replace(target, editable_fields=("clip_strength", "model_strength")),
    ],
)
def test_private_target_drift_refuses_without_mutating_source(
    mutate: Any,
) -> None:
    case = _case()
    original = deepcopy(case.graph)
    target = mutate(case.extraction.edit_targets[0])
    extraction = _replace_extraction(case, targets=(target,))

    with pytest.raises(WorkflowLoraGraphError):
        apply_workflow_lora_overrides(
            api_graph=case.graph,
            catalog=case.catalog,
            extraction=extraction,
            resolution=_resolution(case, {"model_strength": 0.4}),
        )

    assert case.graph == original


def test_missing_and_duplicate_private_targets_fail_closed() -> None:
    case = _case()
    target = case.extraction.edit_targets[0]
    resolution = _resolution(case, {"model_strength": 0.4})

    for targets in ((), (target, target)):
        with pytest.raises(WorkflowLoraGraphError) as raised:
            apply_workflow_lora_overrides(
                api_graph=case.graph,
                catalog=case.catalog,
                extraction=_replace_extraction(case, targets=targets),
                resolution=resolution,
            )
        assert raised.value.code in {
            "invalid_workflow_lora_private_targets",
            "stale_workflow_lora_private_target",
        }


def test_catalog_slot_must_be_the_same_exact_public_extraction() -> None:
    case = _case()
    mismatched = replace(case.catalog.slots[0], default_model_strength=0.9)

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=case.graph,
            catalog=WorkflowLoraOverrideCatalog(case.catalog.target, (mismatched,)),
            extraction=case.extraction,
            resolution=_resolution(case, {"model_strength": 0.4}),
        )

    assert raised.value.code == "stale_workflow_lora_private_target"


def test_witness_component_drift_refuses_private_application() -> None:
    case = _case()
    changed_witness = replace(case.catalog.target, activation_witness_sha256=_digest(50))
    catalog = WorkflowLoraOverrideCatalog(changed_witness, case.catalog.slots)

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=case.graph,
            catalog=catalog,
            extraction=case.extraction,
            resolution=_resolution(case, {"model_strength": 0.4}),
        )

    assert raised.value.code == "stale_workflow_lora_private_target"


@pytest.mark.parametrize(
    "component",
    ["api_graph_sha256", "dependency_contract_sha256", "activation_binding_sha256"],
)
def test_plan_resolved_for_other_evidence_never_applies_to_this_extraction(
    component: str,
) -> None:
    case = _case()
    other = replace(case.catalog.target, **{component: _digest(52)})
    catalog = WorkflowLoraOverrideCatalog(other, case.catalog.slots)
    plan = _resolution(_Case(case.graph, catalog, case.extraction), {"model_strength": 0.4})
    original = deepcopy(case.graph)

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=case.graph,
            catalog=catalog,
            extraction=case.extraction,
            resolution=plan,
        )

    assert raised.value.code in {
        "stale_workflow_lora_graph_source",
        "stale_workflow_lora_private_target",
    }
    assert case.graph == original


def test_private_targets_must_cover_every_editable_slot_from_one_pass() -> None:
    case = _authentic_rgthree_case()
    first, second = case.extraction.edit_targets
    mixed = replace(second, source_authority_evidence_sha256=_digest(53))

    for targets in ((first,), (first, mixed)):
        with pytest.raises(WorkflowLoraGraphError) as raised:
            apply_workflow_lora_overrides(
                api_graph=case.graph,
                catalog=case.catalog,
                extraction=replace(case.extraction, edit_targets=targets),
                resolution=_resolution(case, {"model_strength": 1.1}),
            )
        assert raised.value.code == "invalid_workflow_lora_private_targets"


def test_slot_bound_to_another_file_never_patches_the_authored_loader() -> None:
    case = _case()
    slot = case.catalog.slots[0]
    assert slot.asset_binding is not None
    other_file = "styles/another.safetensors"
    forged = replace(
        slot,
        observed_runtime_reference=other_file,
        asset_binding=replace(slot.asset_binding, runtime_reference=other_file),
    )
    catalog = WorkflowLoraOverrideCatalog(case.catalog.target, (forged,))
    extraction = _replace_extraction(case, public=replace(case.extraction.public, slots=(forged,)))
    forged_case = _Case(case.graph, catalog, extraction)
    original = deepcopy(case.graph)

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=case.graph,
            catalog=catalog,
            extraction=extraction,
            resolution=_resolution(forged_case, {"model_strength": 0.4}),
        )

    assert raised.value.code == "changed_workflow_lora_authored_value"
    assert case.graph == original


class _HostileDict(dict[str, Any]):
    pass


class _HostileTarget(WorkflowLoraPrivateEditTarget):
    pass


class _HostileSlot(WorkflowLoraSlot):
    pass


@pytest.mark.parametrize("nested", [False, True])
def test_graph_container_subclasses_never_receive_mutation_authority(nested: bool) -> None:
    case = _case()
    if nested:
        graph = deepcopy(case.graph)
        graph["private-core-node"] = _HostileDict(graph["private-core-node"])
    else:
        graph = _HostileDict(deepcopy(case.graph))

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=graph,
            catalog=case.catalog,
            extraction=case.extraction,
            resolution=_resolution(case, {"model_strength": 0.4}),
        )

    assert raised.value.code == "invalid_workflow_lora_graph_source"


def test_private_and_public_typed_subclasses_fail_closed() -> None:
    case = _case()
    target = case.extraction.edit_targets[0]
    hostile_target = _HostileTarget(*(getattr(target, name) for name in target.__slots__))
    slot = case.catalog.slots[0]
    hostile_slot = _HostileSlot(*(getattr(slot, name) for name in slot.__slots__))
    original = deepcopy(case.graph)

    with pytest.raises(WorkflowLoraGraphError):
        apply_workflow_lora_overrides(
            api_graph=case.graph,
            catalog=case.catalog,
            extraction=_replace_extraction(case, targets=(hostile_target,)),
            resolution=_resolution(case, {"model_strength": 0.4}),
        )
    with pytest.raises(WorkflowLoraGraphError):
        apply_workflow_lora_overrides(
            api_graph=case.graph,
            catalog=WorkflowLoraOverrideCatalog(case.catalog.target, (hostile_slot,)),
            extraction=_replace_extraction(
                case,
                public=replace(case.extraction.public, slots=(hostile_slot,)),
            ),
            resolution=_resolution(case, {"model_strength": 0.4}),
        )
    assert case.graph == original


def test_untyped_private_target_refuses_with_typed_graph_error() -> None:
    case = _case()
    extraction = WorkflowLoraSlotExtractionWithPrivateTargets(
        case.extraction.public,
        cast(Any, (object(),)),
    )

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=case.graph,
            catalog=case.catalog,
            extraction=extraction,
            resolution=_resolution(case, {"model_strength": 0.4}),
        )

    assert raised.value.code == "invalid_workflow_lora_private_targets"


def test_nonfinite_source_strength_is_never_accepted_as_authored_evidence() -> None:
    case = _case()
    graph = deepcopy(case.graph)
    graph["private-core-node"]["inputs"]["strength_model"] = float("nan")

    with pytest.raises(WorkflowLoraGraphError) as raised:
        apply_workflow_lora_overrides(
            api_graph=graph,
            catalog=case.catalog,
            extraction=case.extraction,
            resolution=_resolution(case, {"model_strength": 0.4}),
        )

    assert raised.value.code == "invalid_workflow_lora_graph_source"


def test_result_provenance_is_public_safe_canonical_and_frozen() -> None:
    case = _case()
    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=_resolution(case, {"model_strength": 0.4}),
    )
    payload = workflow_lora_graph_resolution_payload(result)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    assert "private-core-node" not in encoded
    assert "lora_name" not in encoded
    assert "effective_api_graph_json" not in payload
    assert len(workflow_lora_graph_resolution_sha256(result)) == 64
    assert payload["workflow_effective_graph_sha256"] == _graph_sha256(
        workflow_lora_effective_api_graph(result)
    )
    with pytest.raises(FrozenInstanceError):
        result.__setattr__("version", 2)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: replace(result, workflow_effective_graph_sha256=_digest(60)),
        lambda result: replace(result, effective_api_graph_json="{}"),
        lambda result: replace(result, override_resolution_sha256="BAD"),
        lambda result: replace(result, overrides=(result.overrides[0], result.overrides[0])),
        lambda result: replace(
            result,
            overrides=(
                replace(
                    result.overrides[0],
                    changes=(
                        result.overrides[0].changes[0],
                        result.overrides[0].changes[0],
                    ),
                ),
            ),
        ),
    ],
)
def test_result_tampering_is_refused(mutate: Any) -> None:
    case = _case()
    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=_resolution(case, {"model_strength": 0.4}),
    )

    with pytest.raises(WorkflowLoraGraphError) as raised:
        workflow_lora_graph_resolution_payload(mutate(result))

    assert raised.value.code == "invalid_workflow_lora_graph_resolution"


def test_result_refuses_noncanonical_field_order_and_out_of_range_effective_value() -> None:
    case = _case()
    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=_resolution(
            case,
            {"model_strength": 0.4, "clip_strength": 0.3},
        ),
    )
    override = result.overrides[0]
    reversed_fields = replace(
        result,
        overrides=(replace(override, changes=tuple(reversed(override.changes))),),
    )
    too_large = replace(
        result,
        overrides=(
            replace(
                override,
                changes=(
                    replace(override.changes[0], effective_value=999.0),
                    override.changes[1],
                ),
            ),
        ),
    )

    for hostile in (reversed_fields, too_large):
        with pytest.raises(WorkflowLoraGraphError) as raised:
            workflow_lora_graph_resolution_payload(hostile)
        assert raised.value.code == "invalid_workflow_lora_graph_resolution"


def test_result_typed_subclasses_are_refused() -> None:
    case = _case()
    result = apply_workflow_lora_overrides(
        api_graph=case.graph,
        catalog=case.catalog,
        extraction=case.extraction,
        resolution=_resolution(case, {"model_strength": 0.4}),
    )

    class HostileResult(WorkflowLoraGraphResolution):
        pass

    class HostileOverride(AppliedWorkflowLoraOverride):
        pass

    hostile_result = HostileResult(*(getattr(result, name) for name in result.__slots__))
    override = result.overrides[0]
    hostile_override = HostileOverride(*(getattr(override, name) for name in override.__slots__))

    with pytest.raises(WorkflowLoraGraphError):
        workflow_lora_graph_resolution_payload(hostile_result)
    with pytest.raises(WorkflowLoraGraphError):
        workflow_lora_graph_resolution_payload(replace(result, overrides=(hostile_override,)))
