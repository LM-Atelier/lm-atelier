from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

import local_lm.workflow_lora_slots as workflow_lora_slots_module
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
from local_lm.workflow_lora_slots import (
    CORE_LORA_LOADER_CONTRACT,
    CORE_LORA_LOADER_SCHEMA_SHA256,
    CORE_MODEL_ONLY_LORA_LOADER_CONTRACT,
    CORE_MODEL_ONLY_LORA_LOADER_SCHEMA_SHA256,
    CORE_RUNTIME_ADAPTER_CONTRACT_VERSION,
    MAX_WORKFLOW_LORA_SLOTS,
    WorkflowLoraCoreEvidence,
    WorkflowLoraPackageEvidence,
    WorkflowLoraSlotError,
    extract_workflow_lora_slots,
    extract_workflow_lora_slots_with_private_targets,
)
from local_lm.workflow_trust import canonical_graph

_RGTHREE_REVISION = "6b76ee6f2c5a007710b5a16f97c94330d6ecc871"


def _slot(
    name: str,
    resource_kind: str,
    *,
    required: bool = True,
    satisfaction: str = "any_of",
    requirement_keys: tuple[str, ...] = ("default",),
    constraints: dict[str, Any] | None = None,
) -> WorkflowDependencySlotContract:
    return WorkflowDependencySlotContract(
        name=name,
        resource_kind=resource_kind,  # type: ignore[arg-type]
        required=required,
        satisfaction=satisfaction,  # type: ignore[arg-type]
        requirements=tuple(
            WorkflowDependencyRequirement(key, constraints or {}) for key in requirement_keys
        ),
    )


def _contract(*slots: WorkflowDependencySlotContract) -> WorkflowDependencyContract:
    return WorkflowDependencyContract(version=1, slots=slots)


def _asset_binding(
    slot_name: str,
    runtime_reference: str,
    *,
    requirement_key: str = "default",
    sha256: str = "a" * 64,
) -> ResolvedWorkflowBinding:
    identity = {
        "kind": "model_asset",
        "asset_kind": "lora",
        "runtime_reference": runtime_reference,
        "sha256": sha256,
    }
    return ResolvedWorkflowBinding(
        slot_name=slot_name,
        requirement_key=requirement_key,
        resource_kind="model_asset",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256(
            "model_asset",
            identity,
        ),
        mount={},
    )


def _package_binding(
    *,
    revision: str = _RGTHREE_REVISION,
) -> ResolvedWorkflowBinding:
    identity = {
        "kind": "registry_package",
        "package_id": "rgthree-comfy",
        "package_version": revision,
        "registry_record_id": f"rgthree-comfy@{revision}",
        "node_types": [POWER_LORA_LOADER],
        "archive_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "wheel_closure_sha256": "3" * 64,
        "wheel_environment_sha256": "4" * 64,
    }
    return ResolvedWorkflowBinding(
        slot_name="rgthree",
        requirement_key="default",
        resource_kind="registry_package",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256(
            "registry_package",
            identity,
        ),
        mount={},
    )


def _runtime_binding(
    *,
    adapter_contract_version: int = CORE_RUNTIME_ADAPTER_CONTRACT_VERSION,
) -> ResolvedWorkflowBinding:
    identity = {
        "kind": "runtime",
        "engine": "comfyui",
        "runtime_build": "ComfyUI test build",
        "adapter_contract_version": adapter_contract_version,
        "launch_contract_version": "v1",
    }
    return ResolvedWorkflowBinding(
        slot_name="comfy-runtime",
        requirement_key="default",
        resource_kind="runtime",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256(
            "runtime",
            identity,
        ),
        mount={},
    )


def _activation(
    contract: WorkflowDependencyContract,
    *bindings: ResolvedWorkflowBinding,
) -> WorkflowActivationResolution:
    digest = workflow_activation_binding_sha256(contract, bindings)
    return WorkflowActivationResolution(
        bindings=bindings,
        issues=(),
        missing_required_slots=(),
        complete=True,
        binding_sha256=digest,
    )


def _graph_sha256(graph: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_graph(graph).encode("utf-8")).hexdigest()


def _source() -> dict[str, Any]:
    return {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "base.safetensors"},
        "_meta": {"title": "Base"},
    }


def _core_graph(
    *,
    loader_type: str = "LoraLoader",
    runtime_reference: str = "styles/detail.safetensors",
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "model": ["source", 0],
        "lora_name": runtime_reference,
        "strength_model": 0.75,
    }
    if loader_type == "LoraLoader":
        inputs.update({"clip": ["source", 1], "strength_clip": 0.5})
    return {
        "source": _source(),
        "loader-node-that-must-stay-private": {
            "class_type": loader_type,
            "inputs": inputs,
            "_meta": {"title": "Embedded style"},
        },
    }


def _core_evidence(
    *,
    required: bool = True,
    loader_type: str = "LoraLoader",
    runtime_reference: str = "styles/detail.safetensors",
) -> tuple[
    dict[str, Any],
    WorkflowDependencyContract,
    WorkflowActivationResolution,
    WorkflowLoraCoreEvidence,
]:
    graph = _core_graph(
        loader_type=loader_type,
        runtime_reference=runtime_reference,
    )
    contract = _contract(
        _slot("comfy-runtime", "runtime"),
        _slot("style", "model_asset", required=required),
    )
    runtime = _runtime_binding()
    asset = _asset_binding("style", runtime_reference)
    activation = _activation(contract, runtime, asset)
    if loader_type == "LoraLoader":
        loader_contract = CORE_LORA_LOADER_CONTRACT
        adapter_schema_sha256 = CORE_LORA_LOADER_SCHEMA_SHA256
    else:
        loader_contract = CORE_MODEL_ONLY_LORA_LOADER_CONTRACT
        adapter_schema_sha256 = CORE_MODEL_ONLY_LORA_LOADER_SCHEMA_SHA256
    evidence = WorkflowLoraCoreEvidence(
        api_graph_sha256=_graph_sha256(graph),
        node_id="loader-node-that-must-stay-private",
        loader_contract=loader_contract,
        adapter_schema_sha256=adapter_schema_sha256,
        core_claimed=True,
        dependency_slot="comfy-runtime",
        requirement_key="default",
        resource_identity_sha256=runtime.resource_identity_sha256,
    )
    return graph, contract, activation, evidence


def _rgthree_graph(
    *,
    runtime_reference: str = "styles/power.safetensors",
    separate_clip: float | None = 0.6,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "on": True,
        "lora": runtime_reference,
        "strength": 0.8,
        "strengthTwo": separate_clip,
    }
    return {
        "source": _source(),
        "power-node-that-must-stay-private": {
            "class_type": POWER_LORA_LOADER,
            "inputs": {
                "model": ["source", 0],
                "clip": ["source", 1],
                "lora_1": entry,
            },
            "_meta": {"title": "Power"},
        },
    }


def _rgthree_evidence(
    *,
    required: bool = False,
    revision: str = _RGTHREE_REVISION,
    runtime_reference: str = "styles/power.safetensors",
    separate_clip: float | None = 0.6,
) -> tuple[
    dict[str, Any],
    WorkflowDependencyContract,
    WorkflowActivationResolution,
    WorkflowLoraPackageEvidence,
]:
    graph = _rgthree_graph(
        runtime_reference=runtime_reference,
        separate_clip=separate_clip,
    )
    contract = _contract(
        _slot("rgthree", "registry_package"),
        _slot("style", "model_asset", required=required),
    )
    package = _package_binding(revision=revision)
    asset = _asset_binding("style", runtime_reference)
    activation = _activation(contract, package, asset)
    evidence = WorkflowLoraPackageEvidence(
        api_graph_sha256=_graph_sha256(graph),
        node_id="power-node-that-must-stay-private",
        package_claim=PackageClaim(
            registry_id="rgthree-comfy",
            repository_id="rgthree/rgthree-comfy",
            revision=revision,
        ),
        dependency_slot="rgthree",
        requirement_key="default",
        resource_identity_sha256=package.resource_identity_sha256,
    )
    return graph, contract, activation, evidence


def test_core_loader_extracts_frozen_opaque_content_addressed_slot() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    original = deepcopy(graph)

    extracted = extract_workflow_lora_slots(
        revision_scope="wfrev_exact",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )

    assert graph == original
    assert extracted.api_graph_sha256 == _graph_sha256(graph)
    assert extracted.activation_binding_sha256 == activation.binding_sha256
    assert len(extracted.slots) == 1
    slot = extracted.slots[0]
    assert slot.slot_id.startswith("wflora_")
    assert "loader-node-that-must-stay-private" not in repr(slot)
    assert slot.loader_contract == CORE_LORA_LOADER_CONTRACT
    assert slot.editability == "editable"
    assert slot.dependency_required is False
    assert slot.editable_fields == ("model_strength", "clip_strength")
    assert slot.default_enabled is True
    assert slot.default_model_strength == 0.75
    assert slot.default_clip_strength == 0.5
    assert slot.asset_binding is not None
    assert slot.asset_binding.sha256 == "a" * 64
    with pytest.raises(FrozenInstanceError):
        slot.position = 9  # type: ignore[misc]


def test_model_only_loader_has_no_invented_clip_or_enable_control() -> None:
    graph, contract, activation, evidence = _core_evidence(
        required=False,
        loader_type="LoraLoaderModelOnly",
    )

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_model_only",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    ).slots[0]

    assert slot.loader_contract == CORE_MODEL_ONLY_LORA_LOADER_CONTRACT
    assert slot.strength_mode == "model_only"
    assert slot.default_clip_strength is None
    assert slot.editable_fields == ("model_strength",)


def test_core_class_type_alone_never_grants_core_edit_authority() -> None:
    graph, contract, activation, _ = _core_evidence(required=False)

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_unattributed_core",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "missing_core_evidence"
    assert slot.loader_contract is None
    assert slot.loader_authority_sha256 is None
    assert slot.default_model_strength is None
    assert slot.asset_binding is not None


def test_shadowed_core_class_type_is_read_only_when_origin_is_not_core() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_shadowed_core",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(replace(evidence, core_claimed=False),),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "node_not_core_claimed"
    assert slot.loader_contract is None


def test_core_schema_must_match_the_audited_loader_adapter_contract() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_wrong_core_schema",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(replace(evidence, adapter_schema_sha256="f" * 64),),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "core_loader_contract_mismatch"


def test_core_evidence_recorded_for_another_graph_is_read_only() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_core_other_graph",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(replace(evidence, api_graph_sha256="f" * 64),),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "stale_core_evidence"
    assert slot.loader_contract is None
    assert slot.editable_fields == ()


@pytest.mark.parametrize("adapter_contract_version", [True, 2])
def test_core_authority_requires_the_exact_audited_runtime_adapter_version(
    adapter_contract_version: object,
) -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    runtime = activation.bindings[0]
    identity = dict(runtime.identity)
    identity["adapter_contract_version"] = adapter_contract_version
    changed_runtime = replace(
        runtime,
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256(
            "runtime",
            identity,
        ),
    )
    changed_activation = _activation(contract, changed_runtime, activation.bindings[1])
    changed_evidence = replace(
        evidence,
        resource_identity_sha256=changed_runtime.resource_identity_sha256,
    )

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_wrong_runtime_adapter",
        api_graph=graph,
        dependency_contract=contract,
        activation=changed_activation,
        core_evidence=(changed_evidence,),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "invalid_runtime_binding"


def test_required_core_dependency_is_fully_locked() -> None:
    graph, contract, activation, evidence = _core_evidence(required=True)

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_required_core",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    ).slots[0]

    assert slot.editability == "required_locked"
    assert slot.read_only_reason == "required_workflow_dependency"
    assert slot.dependency_required is True
    assert slot.editable_fields == ()


def test_audited_rgthree_entry_requires_graph_package_and_binding_identity() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_rgthree",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    ).slots[0]

    assert slot.loader_contract == POWER_LORA_LOADER_CONTRACT
    assert slot.loader_authority_sha256 == activation.bindings[0].resource_identity_sha256
    assert slot.editability == "editable"
    assert slot.editable_fields == (
        "enabled",
        "model_strength",
        "clip_strength",
    )
    assert slot.default_enabled is True
    assert slot.default_model_strength == 0.8
    assert slot.default_clip_strength == 0.6
    assert slot.strength_mode == "separate"
    assert slot.asset_binding is not None
    assert slot.asset_binding.resource_identity_sha256 == (
        activation.bindings[1].resource_identity_sha256
    )


def test_rgthree_coupled_strength_never_invents_a_separate_clip_edit() -> None:
    graph, contract, activation, evidence = _rgthree_evidence(separate_clip=None)

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_rgthree_coupled",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    ).slots[0]

    assert slot.strength_mode == "coupled"
    assert slot.default_model_strength == 0.8
    assert slot.default_clip_strength == 0.8
    assert slot.editable_fields == ("enabled", "model_strength")


def test_rgthree_slot_identity_survives_an_exact_package_binding_repair() -> None:
    first_graph, first_contract, first_activation, first_evidence = _rgthree_evidence()
    second_graph, second_contract, second_activation, second_evidence = _rgthree_evidence(
        revision="1.0.2605082257"
    )

    first = extract_workflow_lora_slots(
        revision_scope="wfrev_package_binding",
        api_graph=first_graph,
        dependency_contract=first_contract,
        activation=first_activation,
        package_evidence=(first_evidence,),
    ).slots[0]
    second = extract_workflow_lora_slots(
        revision_scope="wfrev_package_binding",
        api_graph=second_graph,
        dependency_contract=second_contract,
        activation=second_activation,
        package_evidence=(second_evidence,),
    ).slots[0]

    assert first.loader_authority_sha256 != second.loader_authority_sha256
    assert first.slot_id == second.slot_id


def test_required_rgthree_dependency_cannot_be_disabled() -> None:
    graph, contract, activation, evidence = _rgthree_evidence(required=True)

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_required_rgthree",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    ).slots[0]

    assert slot.editability == "required_locked"
    assert slot.dependency_required is True
    assert slot.editable_fields == ()
    assert slot.read_only_reason == "required_workflow_dependency"


def test_rgthree_slot_identity_survives_missing_then_resolved_authority() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()

    unresolved = extract_workflow_lora_slots(
        revision_scope="wfrev_rgthree_authority_repair",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
    ).slots[0]
    resolved = extract_workflow_lora_slots(
        revision_scope="wfrev_rgthree_authority_repair",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    ).slots[0]

    assert unresolved.slot_id == resolved.slot_id
    assert unresolved.loader_authority_sha256 is None
    assert resolved.loader_authority_sha256 is not None


@pytest.mark.parametrize(
    ("evidence_change", "reason"),
    [
        ("missing", "missing_package_evidence"),
        ("stale_graph", "stale_package_evidence"),
        ("stale_binding", "stale_package_binding"),
    ],
)
def test_rgthree_without_exact_package_authority_is_visible_read_only(
    evidence_change: str,
    reason: str,
) -> None:
    graph, contract, activation, evidence = _rgthree_evidence()
    package_evidence: tuple[WorkflowLoraPackageEvidence, ...]
    if evidence_change == "missing":
        package_evidence = ()
    elif evidence_change == "stale_graph":
        package_evidence = (replace(evidence, api_graph_sha256="f" * 64),)
    else:
        package_evidence = (replace(evidence, resource_identity_sha256="f" * 64),)

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_no_package_authority",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=package_evidence,
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == reason
    assert slot.loader_contract is None
    assert slot.editable_fields == ()
    assert slot.default_enabled is None
    assert slot.default_model_strength is None
    assert slot.asset_binding is not None


def test_unaudited_rgthree_revision_is_detected_but_never_authorized() -> None:
    graph, contract, activation, evidence = _rgthree_evidence(revision="9.9.9")

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_unaudited",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "package_widget_revision"
    assert slot.loader_contract is None
    assert slot.editable_fields == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("package_id", "not-rgthree"),
        ("package_version", "0.0.1"),
        ("node_types", ["Some Other Loader"]),
    ],
)
def test_rgthree_binding_must_name_the_same_package_revision_and_node_type(
    field: str,
    value: object,
) -> None:
    graph, contract, activation, evidence = _rgthree_evidence()
    package = activation.bindings[0]
    wrong_identity = dict(package.identity)
    wrong_identity[field] = value
    wrong_package = replace(
        package,
        identity=wrong_identity,
        resource_identity_sha256=workflow_resource_identity_sha256(
            "registry_package",
            wrong_identity,
        ),
    )
    mismatched_contract = _contract(
        _slot("rgthree", "registry_package"),
        _slot("style", "model_asset", required=False),
    )
    mismatched_activation = _activation(
        mismatched_contract,
        wrong_package,
        activation.bindings[1],
    )
    evidence = replace(
        evidence,
        resource_identity_sha256=wrong_package.resource_identity_sha256,
    )

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_wrong_package",
        api_graph=graph,
        dependency_contract=contract,
        activation=mismatched_activation,
        package_evidence=(evidence,),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "package_binding_mismatch"


def test_rgthree_package_binding_needs_a_complete_content_identity() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()
    package = activation.bindings[0]
    incomplete_identity = dict(package.identity)
    del incomplete_identity["manifest_sha256"]
    incomplete_package = replace(
        package,
        identity=incomplete_identity,
        resource_identity_sha256=workflow_resource_identity_sha256(
            "registry_package",
            incomplete_identity,
        ),
    )
    incomplete_activation = _activation(
        contract,
        incomplete_package,
        activation.bindings[1],
    )
    evidence = replace(
        evidence,
        resource_identity_sha256=incomplete_package.resource_identity_sha256,
    )

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_incomplete_package",
        api_graph=graph,
        dependency_contract=contract,
        activation=incomplete_activation,
        package_evidence=(evidence,),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "invalid_package_binding"


def test_malformed_rgthree_entries_do_not_keep_package_edit_authority() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()
    graph["power-node-that-must-stay-private"]["inputs"]["lora_2"] = {
        "on": True,
        "lora": "styles/second.safetensors",
        "strength": True,
    }
    evidence = replace(evidence, api_graph_sha256=_graph_sha256(graph))

    slots = extract_workflow_lora_slots(
        revision_scope="wfrev_bad_rgthree",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    ).slots

    assert len(slots) == 2
    assert {slot.editability for slot in slots} == {"detected_read_only"}
    assert {slot.read_only_reason for slot in slots} == {"invalid_loader_strength"}
    assert all(slot.default_model_strength is None for slot in slots)


def test_huge_integer_strengths_refuse_as_typed_read_only_layouts() -> None:
    huge = 10**400
    core_graph, core_contract, core_activation, core_evidence = _core_evidence(required=False)
    core_graph["loader-node-that-must-stay-private"]["inputs"]["strength_model"] = huge
    core_evidence = replace(
        core_evidence,
        api_graph_sha256=_graph_sha256(core_graph),
    )

    core_slot = extract_workflow_lora_slots(
        revision_scope="wfrev_huge_core_strength",
        api_graph=core_graph,
        dependency_contract=core_contract,
        activation=core_activation,
        core_evidence=(core_evidence,),
    ).slots[0]

    rgthree_graph, rgthree_contract, rgthree_activation, package_evidence = _rgthree_evidence()
    rgthree_graph["power-node-that-must-stay-private"]["inputs"]["lora_1"]["strength"] = huge
    package_evidence = replace(
        package_evidence,
        api_graph_sha256=_graph_sha256(rgthree_graph),
    )
    rgthree_slot = extract_workflow_lora_slots(
        revision_scope="wfrev_huge_rgthree_strength",
        api_graph=rgthree_graph,
        dependency_contract=rgthree_contract,
        activation=rgthree_activation,
        package_evidence=(package_evidence,),
    ).slots[0]

    assert core_slot.read_only_reason == "invalid_loader_strength"
    assert rgthree_slot.read_only_reason == "invalid_loader_strength"
    assert core_slot.default_model_strength is None
    assert rgthree_slot.default_model_strength is None


def test_empty_audited_rgthree_loader_does_not_invent_a_lora_slot() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()
    inputs = graph["power-node-that-must-stay-private"]["inputs"]
    del inputs["lora_1"]
    evidence = replace(evidence, api_graph_sha256=_graph_sha256(graph))

    extracted = extract_workflow_lora_slots(
        revision_scope="wfrev_empty_rgthree",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    )

    assert extracted.slots == ()


def test_unknown_custom_loader_is_visible_but_no_strength_is_guessed() -> None:
    graph, contract, activation, _ = _core_evidence(required=False)
    graph["loader-node-that-must-stay-private"] = {
        "class_type": "CustomSuperLoraLoader",
        "inputs": {
            "model": ["source", 0],
            "lora_name": "styles/detail.safetensors",
            "strength_model": 0.75,
            "magic_strength": 99,
        },
        "_meta": {"title": "Custom"},
    }

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_custom",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
    ).slots[0]

    assert slot.loader_type == "CustomSuperLoraLoader"
    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "unsupported_loader_contract"
    assert slot.loader_contract is None
    assert slot.default_model_strength is None
    assert slot.default_clip_strength is None
    assert slot.editable_fields == ()
    assert slot.asset_binding is not None


def test_unknown_lora_collection_is_detected_without_parsing_its_layout() -> None:
    graph = {
        "custom": {
            "class_type": "CustomLoraCollection",
            "inputs": {
                "loras": [
                    {
                        "private_file": "do-not-read",
                        "amount": 500,
                    }
                ]
            },
            "_meta": {"title": "Custom"},
        }
    }
    contract = _contract()

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_custom_collection",
        api_graph=graph,
        dependency_contract=contract,
    ).slots[0]

    assert slot.observed_runtime_reference is None
    assert slot.default_model_strength is None
    assert slot.editability == "detected_read_only"


def test_a_class_name_alone_never_guesses_that_a_custom_node_loads_loras() -> None:
    graph = {
        "custom": {
            "class_type": "FloralPattern",
            "inputs": {"pattern": "flowers"},
            "_meta": {"title": "Not an adapter"},
        }
    }

    extracted = extract_workflow_lora_slots(
        revision_scope="wfrev_not_a_lora",
        api_graph=graph,
        dependency_contract=_contract(),
    )

    assert extracted.slots == ()


def test_a_path_shaped_custom_class_name_is_not_exposed() -> None:
    private_loader_type = "C:" + chr(92) + "private" + chr(92) + "LoraLoader"
    graph = {
        "custom": {
            "class_type": private_loader_type,
            "inputs": {"lora_name": "styles/detail.safetensors"},
            "_meta": {"title": "Custom"},
        }
    }

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_private_class",
        api_graph=graph,
        dependency_contract=_contract(),
    ).slots[0]

    assert slot.loader_type == "unrecognized"
    assert private_loader_type not in repr(slot)


def test_malformed_core_layout_loses_all_edit_authority() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    inputs = graph["loader-node-that-must-stay-private"]["inputs"]
    inputs["unknown_runtime_input"] = 1
    evidence = replace(evidence, api_graph_sha256=_graph_sha256(graph))

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_malformed_core",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "invalid_loader_layout"
    assert slot.loader_contract is None
    assert slot.default_model_strength is None
    assert slot.asset_binding is not None


def test_invalid_absolute_loader_reference_is_never_exposed() -> None:
    private_reference = "C:" + chr(92) + "Users" + chr(92) + "private.safetensors"
    graph, contract, activation, evidence = _core_evidence(required=False)
    graph["loader-node-that-must-stay-private"]["inputs"]["lora_name"] = private_reference
    evidence = replace(evidence, api_graph_sha256=_graph_sha256(graph))

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_private_reference",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "invalid_loader_reference"
    assert slot.observed_runtime_reference is None
    assert private_reference not in repr(slot)
    assert private_reference not in slot.slot_id


def test_missing_asset_binding_keeps_exact_core_defaults_but_no_edit_authority() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    activation = _activation(contract, activation.bindings[0])

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_unbound",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    ).slots[0]

    assert slot.loader_contract == CORE_LORA_LOADER_CONTRACT
    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "missing_asset_binding"
    assert slot.default_model_strength == 0.75
    assert slot.editable_fields == ()
    assert slot.asset_binding is None


def test_core_slot_identity_survives_unbound_bound_and_asset_repair_states() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    runtime = activation.bindings[0]
    unbound_activation = _activation(contract, runtime)
    repaired_asset = _asset_binding(
        "style",
        "styles/detail.safetensors",
        sha256="b" * 64,
    )
    repaired_activation = _activation(contract, runtime, repaired_asset)

    unbound = extract_workflow_lora_slots(
        revision_scope="wfrev_asset_repair",
        api_graph=graph,
        dependency_contract=contract,
        activation=unbound_activation,
        core_evidence=(evidence,),
    ).slots[0]
    bound = extract_workflow_lora_slots(
        revision_scope="wfrev_asset_repair",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    ).slots[0]
    repaired = extract_workflow_lora_slots(
        revision_scope="wfrev_asset_repair",
        api_graph=graph,
        dependency_contract=contract,
        activation=repaired_activation,
        core_evidence=(evidence,),
    ).slots[0]

    assert unbound.slot_id == bound.slot_id == repaired.slot_id
    assert unbound.asset_binding is None
    assert bound.asset_binding is not None
    assert repaired.asset_binding is not None
    assert bound.asset_binding.resource_identity_sha256 != (
        repaired.asset_binding.resource_identity_sha256
    )


def test_extraction_detaches_graph_and_mutable_binding_identity() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)

    extracted = extract_workflow_lora_slots(
        revision_scope="wfrev_detached",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )
    graph["loader-node-that-must-stay-private"]["inputs"]["strength_model"] = 4
    activation.bindings[1].identity["sha256"] = "f" * 64

    slot = extracted.slots[0]
    assert slot.default_model_strength == 0.75
    assert slot.asset_binding is not None
    assert slot.asset_binding.sha256 == "a" * 64


def test_local_binding_mount_details_are_refused_before_projection() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    private_path = "C:" + chr(92) + "Users" + chr(92) + "private" + chr(92) + "loras"
    asset = replace(
        activation.bindings[1],
        mount={"local_path": private_path},
    )
    forged_activation = replace(
        activation,
        bindings=(activation.bindings[0], asset),
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots(
            revision_scope="wfrev_no_local_mount",
            api_graph=graph,
            dependency_contract=contract,
            activation=forged_activation,
            core_evidence=(evidence,),
        )

    assert raised.value.code == "invalid_dependency_bindings"
    assert private_path not in str(raised.value)


def test_slot_identity_is_order_independent_but_revision_and_graph_scoped() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    reordered = dict(reversed(list(graph.items())))

    first = extract_workflow_lora_slots(
        revision_scope="wfrev_scope_a",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )
    same = extract_workflow_lora_slots(
        revision_scope="wfrev_scope_a",
        api_graph=reordered,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )
    other_revision = extract_workflow_lora_slots(
        revision_scope="wfrev_scope_b",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )
    changed_graph = deepcopy(graph)
    changed_graph["loader-node-that-must-stay-private"]["inputs"]["strength_model"] = 0.9
    changed_evidence = replace(evidence, api_graph_sha256=_graph_sha256(changed_graph))
    other_graph = extract_workflow_lora_slots(
        revision_scope="wfrev_scope_a",
        api_graph=changed_graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(changed_evidence,),
    )

    assert first.api_graph_sha256 == same.api_graph_sha256
    assert first.slots[0].slot_id == same.slots[0].slot_id
    assert first.slots[0].slot_id != other_revision.slots[0].slot_id
    assert first.slots[0].slot_id != other_graph.slots[0].slot_id


def test_slot_positions_are_deterministic_presentation_only() -> None:
    graph = {
        "10": _source(),
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["10", 0],
                "lora_name": "styles/first.safetensors",
                "strength_model": 1,
            },
            "_meta": {"title": "First"},
        },
        "1": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["2", 0],
                "lora_name": "styles/second.safetensors",
                "strength_model": 1,
            },
            "_meta": {"title": "Second"},
        },
    }
    contract = _contract(
        _slot(
            "styles",
            "model_asset",
            satisfaction="all_of",
            requirement_keys=("first", "second"),
        )
    )
    activation = _activation(
        contract,
        _asset_binding(
            "styles",
            "styles/first.safetensors",
            requirement_key="first",
        ),
        _asset_binding(
            "styles",
            "styles/second.safetensors",
            requirement_key="second",
            sha256="b" * 64,
        ),
    )

    extracted = extract_workflow_lora_slots(
        revision_scope="wfrev_presentation_order",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
    )

    assert extracted.ordering_authority == "presentation_only"
    assert [slot.observed_runtime_reference for slot in extracted.slots] == [
        "styles/second.safetensors",
        "styles/first.safetensors",
    ]
    assert [slot.position for slot in extracted.slots] == [0, 1]


def test_tampered_binding_identity_digest_is_refused() -> None:
    graph, contract, activation, _ = _core_evidence()
    tampered = replace(
        activation.bindings[0],
        resource_identity_sha256="f" * 64,
    )
    forged = replace(activation, bindings=(tampered, activation.bindings[1]))

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots(
            revision_scope="wfrev_tampered",
            api_graph=graph,
            dependency_contract=contract,
            activation=forged,
        )

    assert raised.value.code == "invalid_dependency_binding_digest"


def test_stale_activation_snapshot_digest_is_refused() -> None:
    graph, contract, activation, _ = _core_evidence()

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots(
            revision_scope="wfrev_stale_activation",
            api_graph=graph,
            dependency_contract=contract,
            activation=replace(activation, binding_sha256="f" * 64),
        )

    assert raised.value.code == "stale_binding_evidence"


@pytest.mark.parametrize("keep_digest", [False, True])
def test_incomplete_activation_snapshot_is_refused_instead_of_partially_used(
    keep_digest: bool,
) -> None:
    # A resolution that says it is incomplete is refused even when it still
    # carries the digest of its bindings.
    graph, contract, activation, _ = _core_evidence()

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots(
            revision_scope="wfrev_incomplete",
            api_graph=graph,
            dependency_contract=contract,
            activation=replace(
                activation,
                complete=False,
                binding_sha256=activation.binding_sha256 if keep_digest else None,
            ),
        )

    assert raised.value.code == "incomplete_binding_evidence"


def test_two_typed_assets_cannot_ambiguously_claim_one_loader_reference() -> None:
    graph = _core_graph()
    contract = _contract(
        _slot(
            "styles",
            "model_asset",
            satisfaction="all_of",
            requirement_keys=("first", "second"),
        )
    )
    first = _asset_binding(
        "styles",
        "styles/detail.safetensors",
        requirement_key="first",
        sha256="a" * 64,
    )
    second = _asset_binding(
        "styles",
        "styles/detail.safetensors",
        requirement_key="second",
        sha256="b" * 64,
    )
    activation = _activation(contract, first, second)

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots(
            revision_scope="wfrev_ambiguous_asset",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
        )

    assert raised.value.code == "ambiguous_lora_binding"


def test_duplicate_package_evidence_is_refused() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots(
            revision_scope="wfrev_duplicate_package",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            package_evidence=(evidence, evidence),
        )

    assert raised.value.code == "duplicate_package_evidence"


def test_malformed_package_evidence_fields_refuse_with_typed_errors() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()
    malformed_slot = replace(evidence, dependency_slot=7)  # type: ignore[arg-type]
    malformed_claim = replace(
        evidence,
        package_claim=PackageClaim(
            registry_id=["rgthree-comfy"],  # type: ignore[arg-type]
            repository_id="rgthree/rgthree-comfy",
            revision=_RGTHREE_REVISION,
        ),
    )

    for malformed in (malformed_slot, malformed_claim):
        with pytest.raises(WorkflowLoraSlotError) as raised:
            extract_workflow_lora_slots(
                revision_scope="wfrev_malformed_package_evidence",
                api_graph=graph,
                dependency_contract=contract,
                activation=activation,
                package_evidence=(malformed,),
            )
        assert raised.value.code == "invalid_package_evidence"


def test_core_evidence_is_typed_bounded_and_unique() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    malformed = replace(evidence, requirement_key=9)  # type: ignore[arg-type]

    with pytest.raises(WorkflowLoraSlotError) as malformed_error:
        extract_workflow_lora_slots(
            revision_scope="wfrev_malformed_core_evidence",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(malformed,),
        )
    assert malformed_error.value.code == "invalid_core_evidence"

    with pytest.raises(WorkflowLoraSlotError) as duplicate_error:
        extract_workflow_lora_slots(
            revision_scope="wfrev_duplicate_core_evidence",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(evidence, evidence),
        )
    assert duplicate_error.value.code == "duplicate_core_evidence"


def test_orphan_and_non_rgthree_package_evidence_are_refused() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()
    orphan = replace(evidence, node_id="missing")

    with pytest.raises(WorkflowLoraSlotError) as orphan_error:
        extract_workflow_lora_slots(
            revision_scope="wfrev_orphan_package",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            package_evidence=(orphan,),
        )
    assert orphan_error.value.code == "invalid_package_evidence"

    core_graph, core_contract, core_activation, _ = _core_evidence()
    wrong_node = replace(
        evidence,
        api_graph_sha256=_graph_sha256(core_graph),
        node_id="loader-node-that-must-stay-private",
    )
    with pytest.raises(WorkflowLoraSlotError) as wrong_node_error:
        extract_workflow_lora_slots(
            revision_scope="wfrev_wrong_package_node",
            api_graph=core_graph,
            dependency_contract=core_contract,
            activation=core_activation,
            package_evidence=(wrong_node,),
        )
    assert wrong_node_error.value.code == "unexpected_package_evidence"


def test_slot_count_is_bounded_without_truncating_the_workflow() -> None:
    graph: dict[str, Any] = {"source": _source()}
    for index in range(MAX_WORKFLOW_LORA_SLOTS + 1):
        graph[f"loader-{index:03d}"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["source", 0],
                "lora_name": f"styles/{index:03d}.safetensors",
                "strength_model": 1,
            },
            "_meta": {"title": "LoRA"},
        }
    original = deepcopy(graph)

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots(
            revision_scope="wfrev_too_many",
            api_graph=graph,
            dependency_contract=_contract(),
        )

    assert raised.value.code == "too_many_workflow_loras"
    assert graph == original


def test_malformed_typed_dependency_contract_refuses_with_a_typed_error() -> None:
    malformed = WorkflowDependencyContract(
        version=1,
        slots=(object(),),  # type: ignore[arg-type]
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots(
            revision_scope="wfrev_malformed_contract",
            api_graph={},
            dependency_contract=malformed,
        )

    assert raised.value.code == "invalid_dependency_contract"


@pytest.mark.parametrize(
    ("graph_change", "code"),
    [
        (
            {
                "bad": {
                    "class_type": "LoraLoader",
                    "inputs": {"strength_model": float("nan")},
                }
            },
            "non_finite_number",
        ),
        ({"bad": "not-a-node"}, "invalid_api_graph_node"),
        ({"bad": {"inputs": {}}}, "invalid_api_graph_node"),
    ],
)
def test_hostile_graphs_refuse_with_typed_errors(
    graph_change: dict[str, Any],
    code: str,
) -> None:
    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots(
            revision_scope="wfrev_hostile",
            api_graph=graph_change,
            dependency_contract=_contract(),
        )

    assert raised.value.code == code


@pytest.mark.parametrize(
    "revision_scope",
    [
        "",
        "../other-revision",
        "C:" + chr(92) + "private",
        "x" * 201,
    ],
)
def test_revision_scope_is_stable_and_never_a_path(revision_scope: str) -> None:
    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots(
            revision_scope=revision_scope,
            api_graph={},
            dependency_contract=_contract(),
        )

    assert raised.value.code == "invalid_revision_scope"


def test_package_evidence_must_be_bound_to_the_exact_graph_hash() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()
    changed = deepcopy(graph)
    changed["source"]["inputs"]["ckpt_name"] = "other.safetensors"

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_changed_graph",
        api_graph=changed,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "stale_package_evidence"


def test_package_claim_conflict_cannot_authorize_the_rgthree_adapter() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()
    conflicting = replace(
        evidence,
        package_claim=replace(
            evidence.package_claim,
            repository_id="someone/other-package",
        ),
    )

    slot = extract_workflow_lora_slots(
        revision_scope="wfrev_conflicting_claim",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(conflicting,),
    ).slots[0]

    assert slot.editability == "detected_read_only"
    assert slot.read_only_reason == "conflicting_package_claim"


def test_binding_order_does_not_change_slot_identity_or_activation_identity() -> None:
    graph, contract, activation, evidence = _rgthree_evidence()
    reversed_bindings = tuple(reversed(activation.bindings))
    reversed_activation = _activation(contract, *reversed_bindings)

    first = extract_workflow_lora_slots(
        revision_scope="wfrev_binding_order",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    )
    second = extract_workflow_lora_slots(
        revision_scope="wfrev_binding_order",
        api_graph=graph,
        dependency_contract=contract,
        activation=reversed_activation,
        package_evidence=(evidence,),
    )

    assert first.activation_binding_sha256 == second.activation_binding_sha256
    assert first.slots[0].slot_id == second.slots[0].slot_id


def test_private_core_target_is_frozen_graph_bound_and_public_compatible() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    original = deepcopy(graph)
    existing_public = extract_workflow_lora_slots(
        revision_scope="wfrev_private_core",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )

    extracted = extract_workflow_lora_slots_with_private_targets(
        revision_scope="wfrev_private_core",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )

    assert graph == original
    assert extracted.public == existing_public
    assert not hasattr(extracted.public, "edit_targets")
    assert len(extracted.edit_targets) == 1
    target = extracted.edit_targets[0]
    assert target.slot_id == extracted.public.slots[0].slot_id
    assert target.source_api_graph_sha256 == extracted.public.api_graph_sha256
    assert target.source_dependency_contract_sha256 == extracted.public.dependency_contract_sha256
    assert target.source_activation_binding_sha256 == extracted.public.activation_binding_sha256
    assert len(target.source_authority_evidence_sha256) == 64
    assert target.loader_authority_sha256 == extracted.public.slots[0].loader_authority_sha256
    assert target.asset_sha256 == "a" * 64
    assert target.node_id == "loader-node-that-must-stay-private"
    assert target.entry_locator == "lora_name"
    assert target.loader_contract == CORE_LORA_LOADER_CONTRACT
    assert target.editable_fields == ("model_strength", "clip_strength")
    assert target.node_id not in repr(extracted.public)
    for digest in (
        target.source_api_graph_sha256,
        target.source_dependency_contract_sha256,
        target.source_activation_binding_sha256,
        target.source_authority_evidence_sha256,
        target.loader_authority_sha256,
        target.asset_sha256,
    ):
        assert len(digest) == 64
        assert digest == digest.lower()
        assert set(digest) <= set("0123456789abcdef")
    with pytest.raises(FrozenInstanceError):
        target.__setattr__("node_id", "changed")


def test_private_model_only_target_has_only_the_audited_model_field() -> None:
    graph, contract, activation, evidence = _core_evidence(
        required=False,
        loader_type="LoraLoaderModelOnly",
    )

    extracted = extract_workflow_lora_slots_with_private_targets(
        revision_scope="wfrev_private_model_only",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )

    target = extracted.edit_targets[0]
    assert target.loader_contract == CORE_MODEL_ONLY_LORA_LOADER_CONTRACT
    assert target.entry_locator == "lora_name"
    assert target.editable_fields == ("model_strength",)


@pytest.mark.parametrize(
    ("separate_clip", "editable_fields"),
    [
        (0.6, ("enabled", "model_strength", "clip_strength")),
        (None, ("enabled", "model_strength")),
    ],
)
def test_private_rgthree_target_preserves_exact_separate_and_coupled_locators(
    separate_clip: float | None,
    editable_fields: tuple[str, ...],
) -> None:
    graph, contract, activation, evidence = _rgthree_evidence(separate_clip=separate_clip)

    extracted = extract_workflow_lora_slots_with_private_targets(
        revision_scope="wfrev_private_rgthree",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    )

    target = extracted.edit_targets[0]
    assert target.slot_id == extracted.public.slots[0].slot_id
    assert target.node_id == "power-node-that-must-stay-private"
    assert target.entry_locator == "lora_1"
    assert target.loader_contract == POWER_LORA_LOADER_CONTRACT
    assert target.editable_fields == editable_fields


def test_private_targets_correspond_one_to_one_in_public_slot_order() -> None:
    graph = _rgthree_graph(runtime_reference="styles/first.safetensors")
    graph["power-node-that-must-stay-private"]["inputs"]["lora_2"] = {
        "on": False,
        "lora": "styles/second.safetensors",
        "strength": 0.4,
    }
    contract = _contract(
        _slot("rgthree", "registry_package"),
        _slot(
            "styles",
            "model_asset",
            required=False,
            satisfaction="all_of",
            requirement_keys=("first", "second"),
        ),
    )
    package = _package_binding()
    activation = _activation(
        contract,
        package,
        _asset_binding(
            "styles",
            "styles/first.safetensors",
            requirement_key="first",
        ),
        _asset_binding(
            "styles",
            "styles/second.safetensors",
            requirement_key="second",
            sha256="b" * 64,
        ),
    )
    evidence = WorkflowLoraPackageEvidence(
        api_graph_sha256=_graph_sha256(graph),
        node_id="power-node-that-must-stay-private",
        package_claim=PackageClaim(
            registry_id="rgthree-comfy",
            repository_id="rgthree/rgthree-comfy",
            revision=_RGTHREE_REVISION,
        ),
        dependency_slot="rgthree",
        requirement_key="default",
        resource_identity_sha256=package.resource_identity_sha256,
    )

    extracted = extract_workflow_lora_slots_with_private_targets(
        revision_scope="wfrev_private_correspondence",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        package_evidence=(evidence,),
    )

    assert [target.slot_id for target in extracted.edit_targets] == [
        slot.slot_id for slot in extracted.public.slots if slot.editability == "editable"
    ]
    assert [target.entry_locator for target in extracted.edit_targets] == [
        "lora_1",
        "lora_2",
    ]


@pytest.mark.parametrize("loader_family", ["core", "rgthree"])
def test_required_slots_never_receive_private_targets(loader_family: str) -> None:
    if loader_family == "core":
        graph, contract, activation, core_evidence = _core_evidence(required=True)
        extracted = extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_required_core",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(core_evidence,),
        )
    else:
        graph, contract, activation, package_evidence = _rgthree_evidence(required=True)
        extracted = extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_required_rgthree",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            package_evidence=(package_evidence,),
        )

    assert extracted.public.slots[0].editability == "required_locked"
    assert extracted.edit_targets == ()


@pytest.mark.parametrize("loader_family", ["core", "rgthree"])
def test_unauthorized_slots_never_receive_private_targets(loader_family: str) -> None:
    if loader_family == "core":
        graph, contract, activation, _ = _core_evidence(required=False)
    else:
        graph, contract, activation, _ = _rgthree_evidence(required=False)

    extracted = extract_workflow_lora_slots_with_private_targets(
        revision_scope=f"wfrev_private_unauthorized_{loader_family}",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
    )

    assert extracted.public.slots[0].editability == "detected_read_only"
    assert extracted.edit_targets == ()


@pytest.mark.parametrize("loader_family", ["core", "rgthree"])
def test_malformed_slots_never_receive_private_targets(loader_family: str) -> None:
    if loader_family == "core":
        graph, contract, activation, core_evidence = _core_evidence(required=False)
        graph["loader-node-that-must-stay-private"]["inputs"]["strength_model"] = "bad"
        core_evidence = replace(core_evidence, api_graph_sha256=_graph_sha256(graph))
        extracted = extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_malformed_core",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(core_evidence,),
        )
    else:
        graph, contract, activation, package_evidence = _rgthree_evidence(required=False)
        graph["power-node-that-must-stay-private"]["inputs"]["lora_1"]["strength"] = "bad"
        package_evidence = replace(package_evidence, api_graph_sha256=_graph_sha256(graph))
        extracted = extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_malformed_rgthree",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            package_evidence=(package_evidence,),
        )

    assert {slot.editability for slot in extracted.public.slots} == {"detected_read_only"}
    assert extracted.edit_targets == ()


def test_private_target_slot_id_is_stable_across_authority_repair() -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)

    unauthorized = extract_workflow_lora_slots_with_private_targets(
        revision_scope="wfrev_private_stable_id",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
    )
    authorized = extract_workflow_lora_slots_with_private_targets(
        revision_scope="wfrev_private_stable_id",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )

    assert unauthorized.edit_targets == ()
    assert unauthorized.public.slots[0].slot_id == authorized.edit_targets[0].slot_id


def test_private_target_binds_loader_authority_and_asset_content_drift() -> None:
    first_graph, first_contract, first_activation, first_evidence = _rgthree_evidence()
    authority_graph, authority_contract, authority_activation, authority_evidence = (
        _rgthree_evidence(revision="1.0.2605082257")
    )
    changed_asset = _asset_binding(
        "style",
        "styles/power.safetensors",
        sha256="b" * 64,
    )
    asset_activation = _activation(
        first_contract,
        first_activation.bindings[0],
        changed_asset,
    )

    first = extract_workflow_lora_slots_with_private_targets(
        revision_scope="wfrev_private_drift",
        api_graph=first_graph,
        dependency_contract=first_contract,
        activation=first_activation,
        package_evidence=(first_evidence,),
    )
    authority_changed = extract_workflow_lora_slots_with_private_targets(
        revision_scope="wfrev_private_drift",
        api_graph=authority_graph,
        dependency_contract=authority_contract,
        activation=authority_activation,
        package_evidence=(authority_evidence,),
    )
    asset_changed = extract_workflow_lora_slots_with_private_targets(
        revision_scope="wfrev_private_drift",
        api_graph=first_graph,
        dependency_contract=first_contract,
        activation=asset_activation,
        package_evidence=(first_evidence,),
    )

    first_target = first.edit_targets[0]
    authority_target = authority_changed.edit_targets[0]
    asset_target = asset_changed.edit_targets[0]
    assert first_target.slot_id == authority_target.slot_id == asset_target.slot_id
    assert first_target.loader_authority_sha256 != authority_target.loader_authority_sha256
    assert first_target.asset_sha256 == authority_target.asset_sha256 == "a" * 64
    assert first_target.loader_authority_sha256 == asset_target.loader_authority_sha256
    assert first_target.asset_sha256 == "a" * 64
    assert asset_target.asset_sha256 == "b" * 64
    assert (
        first_target.source_activation_binding_sha256
        != authority_target.source_activation_binding_sha256
    )
    assert (
        first_target.source_activation_binding_sha256
        != asset_target.source_activation_binding_sha256
    )
    for digest in (
        first_target.loader_authority_sha256,
        authority_target.loader_authority_sha256,
    ):
        assert len(digest) == 64
        assert digest == digest.lower()
        assert set(digest) <= set("0123456789abcdef")


def test_private_extraction_preserves_the_existing_slot_bound() -> None:
    graph: dict[str, Any] = {"source": _source()}
    for index in range(MAX_WORKFLOW_LORA_SLOTS + 1):
        graph[f"loader-{index:03d}"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["source", 0],
                "lora_name": f"styles/{index:03d}.safetensors",
                "strength_model": 1,
            },
        }

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_too_many",
            api_graph=graph,
            dependency_contract=_contract(),
        )

    assert raised.value.code == "too_many_workflow_loras"


def test_private_target_refuses_graph_digest_drift_during_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    original_graph_sha256 = workflow_lora_slots_module._api_graph_sha256
    calls = 0

    def drifting_graph_sha256(value: Any) -> str:
        nonlocal calls
        calls += 1
        digest = original_graph_sha256(value)
        return digest if calls == 1 else "f" * 64

    monkeypatch.setattr(
        workflow_lora_slots_module,
        "_api_graph_sha256",
        drifting_graph_sha256,
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_graph_drift",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(evidence,),
        )

    assert calls == 2
    assert raised.value.code == "changed_private_edit_source"


def test_private_target_refuses_activation_digest_drift_during_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    original_binding_sha256 = workflow_activation_binding_sha256
    calls = 0

    def drifting_binding_sha256(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        digest = original_binding_sha256(*args, **kwargs)
        return digest if calls <= 2 else "f" * 64

    monkeypatch.setattr(
        workflow_lora_slots_module,
        "workflow_activation_binding_sha256",
        drifting_binding_sha256,
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_activation_drift",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(evidence,),
        )

    assert calls == 3
    assert raised.value.code == "changed_private_edit_source"


def test_private_target_refuses_same_json_graph_subclass_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SpoofedInputs(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            if key == "strength_model":
                return 9.5
            return super().get(key, default)

    graph, contract, activation, evidence = _core_evidence(required=False)
    original_evidence_sha256 = workflow_lora_slots_module._private_authority_evidence_sha256

    def drift_after_graph_snapshot(*args: Any, **kwargs: Any) -> str:
        inputs = graph["loader-node-that-must-stay-private"]["inputs"]
        graph["loader-node-that-must-stay-private"]["inputs"] = SpoofedInputs(inputs)
        return original_evidence_sha256(*args, **kwargs)

    monkeypatch.setattr(
        workflow_lora_slots_module,
        "_private_authority_evidence_sha256",
        drift_after_graph_snapshot,
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_same_json_graph_drift",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(evidence,),
        )

    drifted_inputs = graph["loader-node-that-must-stay-private"]["inputs"]
    assert drifted_inputs["strength_model"] == 0.75
    assert drifted_inputs.get("strength_model") == 9.5
    assert raised.value.code == "changed_private_edit_source"


def test_private_target_refuses_same_json_activation_subclass_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SameJsonStringSubclass(str):
        pass

    graph, contract, activation, evidence = _core_evidence(required=False)
    original_evidence_sha256 = workflow_lora_slots_module._private_authority_evidence_sha256

    def drift_after_binding_snapshot(*args: Any, **kwargs: Any) -> str:
        asset_identity = activation.bindings[1].identity
        asset_identity["sha256"] = SameJsonStringSubclass(asset_identity["sha256"])
        return original_evidence_sha256(*args, **kwargs)

    monkeypatch.setattr(
        workflow_lora_slots_module,
        "_private_authority_evidence_sha256",
        drift_after_binding_snapshot,
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_same_json_activation_drift",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(evidence,),
        )

    assert raised.value.code == "changed_private_edit_source"


def test_private_extraction_owns_a_deep_detached_dependency_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, contract, activation, evidence = _core_evidence(required=False)
    source_constraints = contract.slots[1].requirements[0].constraints
    original_private_edit_target = workflow_lora_slots_module._private_edit_target

    def mutate_source_contract_after_normalization(**kwargs: Any) -> Any:
        source_constraints["late_source_mutation"] = ["caller-owned"]
        return original_private_edit_target(**kwargs)

    monkeypatch.setattr(
        workflow_lora_slots_module,
        "_private_edit_target",
        mutate_source_contract_after_normalization,
    )

    extracted = extract_workflow_lora_slots_with_private_targets(
        revision_scope="wfrev_private_detached_contract",
        api_graph=graph,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )

    assert source_constraints == {"late_source_mutation": ["caller-owned"]}
    assert (
        extracted.edit_targets[0].source_dependency_contract_sha256
        == extracted.public.dependency_contract_sha256
    )


def test_private_target_refuses_core_evidence_drift_during_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SameJsonStringSubclass(str):
        pass

    graph, contract, activation, evidence = _core_evidence(required=False)
    original_private_edit_target = workflow_lora_slots_module._private_edit_target

    def mutate_core_evidence_after_target(**kwargs: Any) -> Any:
        target = original_private_edit_target(**kwargs)
        object.__setattr__(
            evidence,
            "loader_contract",
            SameJsonStringSubclass(evidence.loader_contract),
        )
        return target

    monkeypatch.setattr(
        workflow_lora_slots_module,
        "_private_edit_target",
        mutate_core_evidence_after_target,
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_core_evidence_drift",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(evidence,),
        )

    assert raised.value.code == "changed_private_edit_source"


def test_private_target_refuses_nested_package_claim_drift_during_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SameJsonStringSubclass(str):
        pass

    graph, contract, activation, evidence = _rgthree_evidence(required=False)
    original_private_edit_target = workflow_lora_slots_module._private_edit_target

    def mutate_package_claim_after_target(**kwargs: Any) -> Any:
        target = original_private_edit_target(**kwargs)
        object.__setattr__(
            evidence.package_claim,
            "revision",
            SameJsonStringSubclass(evidence.package_claim.revision),
        )
        return target

    monkeypatch.setattr(
        workflow_lora_slots_module,
        "_private_edit_target",
        mutate_package_claim_after_target,
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_package_evidence_drift",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            package_evidence=(evidence,),
        )

    assert raised.value.code == "changed_private_edit_source"


def test_private_target_refuses_graph_container_subclasses() -> None:
    class GraphSubclass(dict[str, Any]):
        pass

    graph, contract, activation, evidence = _core_evidence(required=False)
    subclassed_graph = GraphSubclass(graph)
    assert (
        extract_workflow_lora_slots(
            revision_scope="wfrev_private_graph_subclass",
            api_graph=subclassed_graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(evidence,),
        )
        .slots[0]
        .editability
        == "editable"
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_graph_subclass",
            api_graph=subclassed_graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(evidence,),
        )

    assert raised.value.code == "noncanonical_private_edit_source"


def test_private_target_refuses_revision_scope_subclasses() -> None:
    class RevisionScopeSubclass(str):
        pass

    graph, contract, activation, evidence = _core_evidence(required=False)

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope=RevisionScopeSubclass("wfrev_private_scope_subclass"),
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(evidence,),
        )

    assert raised.value.code == "noncanonical_private_edit_source"


def test_private_target_refuses_authority_evidence_subclasses() -> None:
    class CoreEvidenceSubclass(WorkflowLoraCoreEvidence):
        pass

    graph, contract, activation, evidence = _core_evidence(required=False)
    subclassed = CoreEvidenceSubclass(
        evidence.api_graph_sha256,
        evidence.node_id,
        evidence.loader_contract,
        evidence.adapter_schema_sha256,
        evidence.core_claimed,
        evidence.dependency_slot,
        evidence.requirement_key,
        evidence.resource_identity_sha256,
    )
    assert (
        extract_workflow_lora_slots(
            revision_scope="wfrev_private_evidence_subclass",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(subclassed,),
        )
        .slots[0]
        .editability
        == "editable"
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_evidence_subclass",
            api_graph=graph,
            dependency_contract=contract,
            activation=activation,
            core_evidence=(subclassed,),
        )

    assert raised.value.code == "noncanonical_private_edit_source"


def test_private_target_refuses_resolved_binding_subclasses() -> None:
    class BindingSubclass(ResolvedWorkflowBinding):
        pass

    graph, contract, activation, evidence = _core_evidence(required=False)
    asset = activation.bindings[1]
    subclassed_asset = BindingSubclass(
        asset.slot_name,
        asset.requirement_key,
        asset.resource_kind,
        asset.identity,
        asset.resource_identity_sha256,
        asset.mount,
    )
    subclassed_activation = _activation(
        contract,
        activation.bindings[0],
        subclassed_asset,
    )
    assert (
        extract_workflow_lora_slots(
            revision_scope="wfrev_private_binding_subclass",
            api_graph=graph,
            dependency_contract=contract,
            activation=subclassed_activation,
            core_evidence=(evidence,),
        )
        .slots[0]
        .editability
        == "editable"
    )

    with pytest.raises(WorkflowLoraSlotError) as raised:
        extract_workflow_lora_slots_with_private_targets(
            revision_scope="wfrev_private_binding_subclass",
            api_graph=graph,
            dependency_contract=contract,
            activation=subclassed_activation,
            core_evidence=(evidence,),
        )

    assert raised.value.code == "noncanonical_private_edit_source"
