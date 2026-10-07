from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any, Literal, TypedDict, cast

import pytest
from httpx2 import AsyncClient

import local_lm.workflow_lora_composition as composition_module
from local_lm.auxiliary_assets import (
    LORA_GRAPH_TRANSFORM_VERSION,
    checkpoint_lora_extension,
    detect_lora_extension,
    resolve_lora_stack,
)
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import (
    ModelAssetInstall,
    ModelInstall,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowRevision,
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
    workflow_dependency_contract_sha256,
)
from local_lm.workflow_lora_composition import (
    WORKFLOW_LORA_COMPOSITION_VERSION,
    WorkflowLoraComposition,
    WorkflowLoraCompositionError,
    compose_workflow_lora_graph,
    workflow_lora_composition_added_provenance,
    workflow_lora_composition_added_settings,
    workflow_lora_composition_graph,
    workflow_lora_composition_payload,
    workflow_lora_composition_sha256,
    workflow_lora_composition_workflow_graph,
)
from local_lm.workflow_lora_graph import (
    WorkflowLoraGraphResolution,
    workflow_lora_graph_resolution_payload,
)
from local_lm.workflow_lora_overrides import (
    WorkflowLoraOverrideCatalog,
    WorkflowLoraOverrideLayer,
    WorkflowLoraOverrideResolution,
    WorkflowLoraOverrideTargetWitness,
    parse_workflow_lora_overrides,
    resolve_workflow_lora_override_layers,
)
from local_lm.workflow_lora_slots import (
    CORE_LORA_LOADER_CONTRACT,
    CORE_LORA_LOADER_SCHEMA_SHA256,
    CORE_RUNTIME_ADAPTER_CONTRACT_VERSION,
    WorkflowLoraCoreEvidence,
    WorkflowLoraSlotExtractionWithPrivateTargets,
    extract_workflow_lora_slots_with_private_targets,
)
from local_lm.workflow_trust import canonical_graph


class _StoredGraphChanges(TypedDict, total=False):
    workflow_effective_api_graph_json: str
    effective_api_graph_json: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _graph_sha256(graph: dict[str, Any]) -> str:
    return _sha256(canonical_graph(graph))


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


def _binding(
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


def _workflow(session: Any) -> WorkflowRevision:
    base = ModelInstall(
        name="Base XL",
        role="image",
        engine="comfyui",
        local_path="C:/managed/base",
        manifest_json={"family": "sdxl"},
        active=True,
    )
    definition = WorkflowDefinition(name="Native and Added", operation="text_to_image")
    session.add_all([base, definition])
    session.flush()
    graph = {
        "source": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base.safetensors"},
        },
        "native-private-node": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["source", 0],
                "clip": ["source", 1],
                "lora_name": "styles/native.safetensors",
                "strength_model": 0.8,
                "strength_clip": 0.7,
            },
        },
        "text": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["native-private-node", 1]},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {"model": ["native-private-node", 0], "seed": 1},
        },
    }
    extension = checkpoint_lora_extension(graph)
    assert extension == {"model": ["source", 0], "clip": ["source", 1]}
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        api_graph_json=graph,
        input_schema_json={
            "type": "object",
            "properties": {
                "loras": {"type": "array", "default": [], "maxItems": 8},
            },
        },
        dependencies_json={
            "model_install_ids": [base.id],
            "extensions": {"lora": extension},
        },
        trusted=True,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    return revision


def _added_asset(session: Any) -> ModelAssetInstall:
    asset = ModelAssetInstall(
        name="Added Detail",
        kind="lora",
        family="sdxl",
        local_path="C:/managed/added-detail",
        size_bytes=1024,
        manifest_json={
            "sha256": "b" * 64,
            "comfy_name": "styles/added-detail.safetensors",
            "metadata": {"trigger_words": ["added-detail"]},
        },
        active=True,
        verified_at=utcnow(),
    )
    session.add(asset)
    session.flush()
    return asset


def _model_only_workflow(session: Any) -> WorkflowRevision:
    revision = _workflow(session)
    graph = {
        "161": {"class_type": "UNETLoader", "inputs": {}},
        "145": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["161", 0]},
        },
        "152": {"class_type": "CFGNorm", "inputs": {"model": ["145", 0]}},
        "153": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["152", 0],
                "lora_name": "lightning.safetensors",
            },
        },
        "163": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "on_true": ["153", 0],
                "on_false": ["152", 0],
            },
        },
        "169": {"class_type": "KSampler", "inputs": {"model": ["163", 0]}},
    }
    extension = detect_lora_extension(graph)
    assert extension == {"mode": "model_only", "model": ["163", 0]}
    revision.api_graph_json = graph
    revision.dependencies_json = {
        **revision.dependencies_json,
        "extensions": {"lora": extension},
    }
    session.flush()
    return revision


@dataclass(frozen=True, slots=True)
class _NativeAuthority:
    catalog: WorkflowLoraOverrideCatalog
    extraction: WorkflowLoraSlotExtractionWithPrivateTargets
    resolution: WorkflowLoraOverrideResolution


def _native_authority(
    revision: WorkflowRevision,
    *,
    model_strength: float = 1.25,
) -> _NativeAuthority:
    contract = WorkflowDependencyContract(
        version=1,
        slots=(
            _dependency_slot("comfy-runtime", "runtime"),
            _dependency_slot("native-lora", "model_asset", required=False),
        ),
    )
    revision.dependency_contract_sha256 = workflow_dependency_contract_sha256(contract)
    runtime_identity = {
        "kind": "runtime",
        "engine": "comfyui",
        "runtime_build": "ComfyUI composition test",
        "adapter_contract_version": CORE_RUNTIME_ADAPTER_CONTRACT_VERSION,
        "launch_contract_version": "v1",
    }
    runtime = _binding(
        slot_name="comfy-runtime",
        resource_kind="runtime",
        identity=runtime_identity,
    )
    asset_identity = {
        "kind": "model_asset",
        "asset_kind": "lora",
        "runtime_reference": "styles/native.safetensors",
        "sha256": "a" * 64,
    }
    asset = _binding(
        slot_name="native-lora",
        resource_kind="model_asset",
        identity=asset_identity,
    )
    bindings = (runtime, asset)
    activation = WorkflowActivationResolution(
        bindings=bindings,
        issues=(),
        missing_required_slots=(),
        complete=True,
        binding_sha256=workflow_activation_binding_sha256(contract, bindings),
    )
    graph_sha256 = _graph_sha256(revision.api_graph_json)
    evidence = WorkflowLoraCoreEvidence(
        api_graph_sha256=graph_sha256,
        node_id="native-private-node",
        loader_contract=CORE_LORA_LOADER_CONTRACT,
        adapter_schema_sha256=CORE_LORA_LOADER_SCHEMA_SHA256,
        core_claimed=True,
        dependency_slot="comfy-runtime",
        requirement_key="default",
        resource_identity_sha256=runtime.resource_identity_sha256,
    )
    extraction = extract_workflow_lora_slots_with_private_targets(
        revision_scope=revision.id,
        api_graph=revision.api_graph_json,
        dependency_contract=contract,
        activation=activation,
        core_evidence=(evidence,),
    )
    slot = extraction.public.slots[0]
    witness = WorkflowLoraOverrideTargetWitness(
        workflow_family_id="family-image",
        workflow_definition_id=revision.workflow_id,
        workflow_variant_key="text_to_image",
        workflow_revision_id=revision.id,
        slot_contract_version=1,
        revision_scope_sha256=extraction.public.revision_scope_sha256,
        api_graph_sha256=extraction.public.api_graph_sha256,
        dependency_contract_sha256=extraction.public.dependency_contract_sha256,
        activation_binding_sha256=cast(str, extraction.public.activation_binding_sha256),
        activation_witness_sha256="c" * 64,
    )
    catalog = WorkflowLoraOverrideCatalog(witness, extraction.public.slots)
    envelope = parse_workflow_lora_overrides(
        {
            "version": 1,
            "targets": [
                {
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
                            "changes": {"model_strength": model_strength},
                        }
                    ],
                }
            ],
        }
    )
    resolution = resolve_workflow_lora_override_layers(
        catalog=catalog,
        layers=(WorkflowLoraOverrideLayer("turn", envelope),),
    )
    return _NativeAuthority(catalog, extraction, resolution)


def _added_value(asset: ModelAssetInstall, *, enabled: bool = True) -> list[dict[str, object]]:
    return [
        {
            "asset_id": asset.id,
            "model_strength": 0.6,
            "clip_strength": 0.5,
            "enabled": enabled,
        }
    ]


async def test_native_then_added_composition_matches_frozen_final_graph(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        added_asset = _added_asset(session)
        native = _native_authority(revision)
        source_before = deepcopy(revision.api_graph_json)
        added_value = _added_value(added_asset)

        result = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=added_value,
            override_catalog=native.catalog,
            slot_extraction=native.extraction,
            override_resolution=native.resolution,
        )
        repeated = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=added_value,
            override_catalog=native.catalog,
            slot_extraction=native.extraction,
            override_resolution=native.resolution,
        )

    workflow_graph = workflow_lora_composition_workflow_graph(result)
    final_graph = workflow_lora_composition_graph(result)
    assert revision.api_graph_json == source_before
    assert workflow_graph["native-private-node"]["inputs"]["strength_model"] == 1.25
    assert "lma_lora_001" not in workflow_graph
    assert final_graph["native-private-node"]["inputs"]["strength_model"] == 1.25
    assert final_graph["native-private-node"]["inputs"]["model"] == ["lma_lora_001", 0]
    assert final_graph["native-private-node"]["inputs"]["clip"] == ["lma_lora_001", 1]
    assert final_graph["lma_lora_001"]["inputs"]["model"] == ["source", 0]
    assert final_graph["lma_lora_001"]["inputs"]["clip"] == ["source", 1]
    assert result.source_api_graph_sha256 == _graph_sha256(source_before)
    assert result.workflow_effective_graph_sha256 == _graph_sha256(workflow_graph)
    assert result.effective_graph_sha256 == _graph_sha256(final_graph)
    assert repeated == result
    assert workflow_lora_composition_added_settings(result) == added_value
    provenance = workflow_lora_composition_added_provenance(result)
    assert provenance[0]["sha256"] == "b" * 64
    assert provenance[0]["trigger_words"] == ["added-detail"]

    payload_text = json.dumps(workflow_lora_composition_payload(result), sort_keys=True)
    payload = workflow_lora_composition_payload(result)
    native_payload = cast(dict[str, Any], payload["workflow_native"])
    override_payload = cast(dict[str, Any], native_payload["override_resolution"])
    target = cast(dict[str, Any], override_payload["target"])
    changes = cast(dict[str, Any], override_payload["overrides"][0]["changes"])
    assert target["workflow_revision_id"] == revision.id
    assert target["workflow_definition_id"] == revision.workflow_id
    assert target["dependency_contract_sha256"] == revision.dependency_contract_sha256
    assert changes["model_strength"] == {"value": 1.25, "origin": "turn"}
    assert (
        cast(dict[str, Any], native_payload["graph_resolution"])["override_resolution_sha256"]
        == result.workflow_override_resolution_sha256
    )
    assert "native-private-node" not in payload_text
    assert "effective_api_graph_json" not in payload_text
    assert "C:/managed" not in payload_text
    assert len(workflow_lora_composition_sha256(result)) == 64

    workflow_graph["native-private-node"]["inputs"]["strength_model"] = -3.0
    final_graph["lma_lora_001"]["inputs"]["model"][0] = "mutated"
    provenance[0]["trigger_words"].append("mutated")
    assert (
        workflow_lora_composition_workflow_graph(result)["native-private-node"]["inputs"][
            "strength_model"
        ]
        == 1.25
    )
    assert workflow_lora_composition_graph(result)["lma_lora_001"]["inputs"]["model"] == [
        "source",
        0,
    ]
    assert workflow_lora_composition_added_provenance(result)[0]["trigger_words"] == [
        "added-detail"
    ]


async def test_added_only_is_byte_compatible_with_legacy_resolver(client: AsyncClient) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _added_asset(session)
        value = _added_value(asset)
        legacy = resolve_lora_stack(session, revision, value)
        result = compose_workflow_lora_graph(session, revision, added_loras=value)

    assert result.effective_graph_sha256 == legacy.graph_sha256
    assert result.source_api_graph_sha256 == result.workflow_effective_graph_sha256
    assert result.workflow_override_resolution_sha256 is None
    assert result.workflow_graph_resolution_sha256 is None
    assert result.workflow_override_resolution_json is None
    assert result.workflow_graph_resolution_json is None
    assert result.added_transform_version == LORA_GRAPH_TRANSFORM_VERSION
    assert workflow_lora_composition_added_settings(result) == legacy.settings
    assert workflow_lora_composition_added_provenance(result) == legacy.provenance


async def test_model_only_added_composition_is_byte_compatible_with_legacy_resolver(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _model_only_workflow(session)
        asset = _added_asset(session)
        value = _added_value(asset)
        legacy = resolve_lora_stack(session, revision, value)
        result = compose_workflow_lora_graph(session, revision, added_loras=value)

    final_graph = workflow_lora_composition_graph(result)
    assert result.effective_graph_sha256 == legacy.graph_sha256
    assert workflow_lora_composition_added_settings(result) == legacy.settings
    assert workflow_lora_composition_added_provenance(result) == legacy.provenance
    assert final_graph["lma_lora_001"]["class_type"] == "LoraLoaderModelOnly"
    assert "clip" not in final_graph["lma_lora_001"]["inputs"]
    assert "strength_clip" not in final_graph["lma_lora_001"]["inputs"]


async def test_model_only_clip_only_intent_changes_composition_not_graph_identity(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _model_only_workflow(session)
        asset = _added_asset(session)
        lower_value = _added_value(asset)
        higher_value = deepcopy(lower_value)
        lower_value[0]["clip_strength"] = 0.2
        higher_value[0]["clip_strength"] = 0.9
        lower = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=lower_value,
        )
        higher = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=higher_value,
        )

    assert lower.effective_graph_sha256 == higher.effective_graph_sha256
    assert workflow_lora_composition_graph(lower) == workflow_lora_composition_graph(higher)
    assert workflow_lora_composition_sha256(lower) != workflow_lora_composition_sha256(higher)
    assert workflow_lora_composition_added_settings(lower)[0]["clip_strength"] == 0.2
    assert workflow_lora_composition_added_settings(higher)[0]["clip_strength"] == 0.9


async def test_added_composition_binds_family_compatibility_to_exact_activation(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        base_id = revision.dependencies_json["model_install_ids"][0]
        base = session.get(ModelInstall, base_id)
        assert base is not None
        activation = WorkflowActivation(
            id="wfact_composition_family",
            workflow_revision_id=revision.id,
            resolver_version="workflow-bindings-v1",
            dependency_contract_sha256="a" * 64,
            binding_sha256="b" * 64,
            state="ready",
            is_active=True,
            details_json={"launch_sha256": "c" * 64},
        )
        slot = WorkflowDependencySlot(
            id="wfslot_composition_family",
            workflow_revision_id=revision.id,
            name="base_model",
            resource_kind="model_install",
            required=True,
            satisfaction="all_of",
            requirements_json=[{"key": "default", "constraints": {}}],
            contract_sha256="d" * 64,
            ordinal=0,
        )
        session.add_all([activation, slot])
        session.flush()
        session.add(
            WorkflowDependencyBinding(
                id="wfbind_composition_family",
                workflow_revision_id=revision.id,
                workflow_activation_id=activation.id,
                workflow_dependency_slot_id=slot.id,
                requirement_key="default",
                model_install_id=base.id,
                mount_json={},
                resource_identity_json={},
                resource_identity_sha256="e" * 64,
            )
        )
        asset = _added_asset(session)
        asset.family = "flux"
        revision.dependencies_json = {
            "extensions": revision.dependencies_json["extensions"],
        }
        session.flush()
        value = _added_value(asset)

        omitted = compose_workflow_lora_graph(session, revision, added_loras=value)
        assert workflow_lora_composition_added_settings(omitted) == value
        with pytest.raises(WorkflowLoraCompositionError) as raised:
            compose_workflow_lora_graph(
                session,
                revision,
                added_loras=value,
                workflow_activation_id=activation.id,
            )
        assert raised.value.code == "invalid_added_lora_composition"

        asset.family = "sdxl"
        session.flush()
        matching = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=value,
            workflow_activation_id=activation.id,
        )

    assert matching.effective_graph_sha256 == omitted.effective_graph_sha256
    assert workflow_lora_composition_added_settings(matching) == value


async def test_empty_and_disabled_added_stacks_share_graph_but_not_intent_digest(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _added_asset(session)
        empty = compose_workflow_lora_graph(session, revision, added_loras=[])
        disabled = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=_added_value(asset, enabled=False),
        )

    assert empty.effective_graph_sha256 == disabled.effective_graph_sha256
    assert workflow_lora_composition_added_settings(empty) == []
    assert workflow_lora_composition_added_settings(disabled)[0]["enabled"] is False
    assert workflow_lora_composition_sha256(empty) != workflow_lora_composition_sha256(disabled)


async def test_native_only_and_explicit_same_value_keep_distinct_intent(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        changed = _native_authority(revision, model_strength=1.25)
        same = _native_authority(revision, model_strength=0.8)
        absent = compose_workflow_lora_graph(session, revision, added_loras=[])
        changed_result = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=[],
            override_catalog=changed.catalog,
            slot_extraction=changed.extraction,
            override_resolution=changed.resolution,
        )
        same_result = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=[],
            override_catalog=same.catalog,
            slot_extraction=same.extraction,
            override_resolution=same.resolution,
        )

    assert changed_result.workflow_effective_graph_sha256 != absent.source_api_graph_sha256
    assert changed_result.workflow_effective_graph_sha256 == changed_result.effective_graph_sha256
    assert same_result.effective_graph_sha256 == absent.effective_graph_sha256
    assert same_result.workflow_graph_resolution_sha256 is not None
    assert workflow_lora_composition_sha256(same_result) != workflow_lora_composition_sha256(absent)


@pytest.mark.parametrize("mask", [1, 2, 3, 4, 5, 6])
async def test_partial_native_authority_refuses_before_added_resolution(
    client: AsyncClient,
    mask: int,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        native = _native_authority(revision)
        kwargs: dict[str, object] = {"added_loras": []}
        if mask & 1:
            kwargs["override_catalog"] = native.catalog
        if mask & 2:
            kwargs["slot_extraction"] = native.extraction
        if mask & 4:
            kwargs["override_resolution"] = native.resolution

        with pytest.raises(WorkflowLoraCompositionError) as raised:
            compose_workflow_lora_graph(session, revision, **cast(Any, kwargs))

    assert raised.value.code == "incomplete_workflow_lora_native_authority"


@pytest.mark.parametrize("dependency_digest", [None, "d" * 64])
async def test_native_authority_binds_selected_revision_dependency_contract(
    client: AsyncClient,
    dependency_digest: str | None,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        native = _native_authority(revision)
        revision.dependency_contract_sha256 = dependency_digest

        with pytest.raises(WorkflowLoraCompositionError) as raised:
            compose_workflow_lora_graph(
                session,
                revision,
                added_loras=[],
                override_catalog=native.catalog,
                slot_extraction=native.extraction,
                override_resolution=native.resolution,
            )

    assert raised.value.code == "stale_workflow_lora_composition_target"


@pytest.mark.parametrize("different_definition", [False, True])
async def test_byte_identical_revision_cannot_reuse_another_revisions_native_authority(
    client: AsyncClient,
    different_definition: bool,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        native = _native_authority(revision)
        workflow_id = revision.workflow_id
        if different_definition:
            definition = WorkflowDefinition(
                name="Byte-identical other definition",
                operation="text_to_image",
            )
            session.add(definition)
            session.flush()
            workflow_id = definition.id
        other = WorkflowRevision(
            workflow_id=workflow_id,
            version=2 if not different_definition else 1,
            engine=revision.engine,
            api_graph_json=deepcopy(revision.api_graph_json),
            input_schema_json=deepcopy(revision.input_schema_json),
            dependencies_json=deepcopy(revision.dependencies_json),
            dependency_contract_sha256=revision.dependency_contract_sha256,
            trusted=True,
        )
        session.add(other)
        session.flush()
        assert _graph_sha256(other.api_graph_json) == _graph_sha256(revision.api_graph_json)

        with pytest.raises(WorkflowLoraCompositionError) as raised:
            compose_workflow_lora_graph(
                session,
                other,
                added_loras=[],
                override_catalog=native.catalog,
                slot_extraction=native.extraction,
                override_resolution=native.resolution,
            )

    assert raised.value.code == "stale_workflow_lora_composition_target"


async def test_invalid_added_stack_is_generic_and_does_not_mutate_revision(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        original = deepcopy(revision.api_graph_json)

        with pytest.raises(WorkflowLoraCompositionError) as raised:
            compose_workflow_lora_graph(
                session,
                revision,
                added_loras=[{"asset_id": "private-missing-asset"}],
            )

        assert revision.api_graph_json == original
    assert raised.value.code == "invalid_added_lora_composition"
    assert "private-missing-asset" not in str(raised.value)


class _HostileGraph(dict[str, Any]):
    pass


class _HostileList(list[object]):
    pass


class _HostileDict(dict[str, object]):
    pass


class _HostileFloat(float):
    pass


class _HostileText(str):
    pass


async def test_hostile_source_container_is_not_composition_authority(client: AsyncClient) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        revision.api_graph_json = _HostileGraph(deepcopy(revision.api_graph_json))

        with pytest.raises(WorkflowLoraCompositionError):
            compose_workflow_lora_graph(session, revision, added_loras=[])


async def test_hostile_added_inputs_refuse_before_the_legacy_resolver(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del client
    calls = 0

    def unexpected_resolver(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("hostile Added input reached the legacy resolver")

    monkeypatch.setattr(
        composition_module,
        "resolve_lora_stack_against_graph",
        unexpected_resolver,
    )
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _added_asset(session)
        valid = _added_value(asset)[0]
        shared = dict(valid)
        cyclic: list[object] = []
        cyclic.append(cyclic)
        hostile_values: list[object] = [
            _HostileList(),
            [_HostileDict(valid)],
            [{**valid, "model_strength": _HostileFloat(0.6)}],
            [shared, shared],
            cyclic,
            [{**valid, "clip_strength": float("nan")}],
        ]
        for value in hostile_values:
            with pytest.raises(WorkflowLoraCompositionError) as raised:
                compose_workflow_lora_graph(session, revision, added_loras=value)
            assert raised.value.code == "invalid_added_lora_composition"

    assert calls == 0


async def test_nonportable_added_runtime_reference_refuses_without_echo(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _added_asset(session)
        asset.manifest_json = {
            **asset.manifest_json,
            "comfy_name": "C:/private/person/secret.safetensors",
        }
        source = deepcopy(revision.api_graph_json)

        with pytest.raises(WorkflowLoraCompositionError) as raised:
            compose_workflow_lora_graph(
                session,
                revision,
                added_loras=_added_value(asset),
            )

        assert revision.api_graph_json == source
    assert raised.value.code == "invalid_added_lora_composition"
    assert "private" not in str(raised.value).casefold()
    assert "secret" not in str(raised.value).casefold()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: replace(result, version=2),
        lambda result: replace(result, source_api_graph_sha256="d" * 64),
        lambda result: replace(result, workflow_effective_graph_sha256="d" * 64),
        lambda result: replace(result, effective_graph_sha256="d" * 64),
        lambda result: replace(result, workflow_override_resolution_sha256="d" * 64),
        lambda result: replace(result, workflow_graph_resolution_sha256="d" * 64),
        lambda result: replace(result, added_transform_version="future-transform"),
        lambda result: replace(
            result,
            added_transform_version=_HostileText(LORA_GRAPH_TRANSFORM_VERSION),
        ),
        lambda result: replace(result, workflow_override_resolution_json="{}"),
        lambda result: replace(result, workflow_graph_resolution_json="{}"),
        lambda result: replace(result, added_settings_json="[]"),
        lambda result: replace(result, added_provenance_json="[]"),
        lambda result: replace(result, workflow_effective_api_graph_json="{}"),
        lambda result: replace(result, effective_api_graph_json="{}"),
    ],
)
async def test_frozen_composition_tampering_is_refused(
    client: AsyncClient,
    mutate: Any,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _added_asset(session)
        native = _native_authority(revision)
        result = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=_added_value(asset),
            override_catalog=native.catalog,
            slot_extraction=native.extraction,
            override_resolution=native.resolution,
        )

    with pytest.raises(WorkflowLoraCompositionError) as raised:
        workflow_lora_composition_payload(mutate(result))
    assert raised.value.code in {
        "invalid_workflow_lora_composition_result",
        "invalid_workflow_lora_composition_graph",
    }


async def test_override_and_graph_receipts_cannot_be_rehashed_independently(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        native = _native_authority(revision)
        result = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=[],
            override_catalog=native.catalog,
            slot_extraction=native.extraction,
            override_resolution=native.resolution,
        )

    assert result.workflow_override_resolution_json is not None
    assert result.workflow_graph_resolution_json is not None
    override_payload = json.loads(result.workflow_override_resolution_json)
    override_payload["overrides"][0]["changes"]["model_strength"]["value"] = 1.5
    override_json = json.dumps(
        override_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    override_sha256 = _sha256(override_json)
    graph_payload = json.loads(result.workflow_graph_resolution_json)
    graph_payload["override_resolution_sha256"] = override_sha256
    graph_json = json.dumps(
        graph_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    forged = replace(
        result,
        workflow_override_resolution_sha256=override_sha256,
        workflow_graph_resolution_sha256=_sha256(graph_json),
        workflow_override_resolution_json=override_json,
        workflow_graph_resolution_json=graph_json,
    )

    with pytest.raises(WorkflowLoraCompositionError) as raised:
        workflow_lora_composition_payload(forged)
    assert raised.value.code == "invalid_workflow_lora_composition_result"


@pytest.mark.parametrize(
    "field_name",
    ["workflow_effective_api_graph_json", "effective_api_graph_json"],
)
async def test_noncanonical_stored_graph_bytes_are_refused_independently(
    client: AsyncClient,
    field_name: Literal["workflow_effective_api_graph_json", "effective_api_graph_json"],
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        result = compose_workflow_lora_graph(session, revision, added_loras=[])

    encoded = cast(str, getattr(result, field_name))
    noncanonical = json.dumps(json.loads(encoded), sort_keys=True, indent=2)
    assert noncanonical != encoded
    changes: _StoredGraphChanges = {}
    changes[field_name] = noncanonical
    forged = replace(result, **changes)

    with pytest.raises(WorkflowLoraCompositionError) as raised:
        workflow_lora_composition_payload(forged)
    assert raised.value.code == "invalid_workflow_lora_composition_result"


@pytest.mark.parametrize(
    ("field_name", "hostile_value"),
    [
        ("model_strength", 1),
        ("clip_strength", 1),
        ("enabled", 1),
    ],
)
async def test_added_provenance_mirrors_settings_with_exact_types(
    client: AsyncClient,
    field_name: str,
    hostile_value: int,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _added_asset(session)
        value = _added_value(asset)
        value[0]["model_strength"] = 1.0
        value[0]["clip_strength"] = 1.0
        result = compose_workflow_lora_graph(session, revision, added_loras=value)

    provenance = json.loads(result.added_provenance_json)
    provenance[0][field_name] = hostile_value
    forged = replace(
        result,
        added_provenance_json=json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
    )

    with pytest.raises(WorkflowLoraCompositionError) as raised:
        workflow_lora_composition_payload(forged)
    assert raised.value.code == "invalid_workflow_lora_composition_result"


async def test_composition_type_is_frozen_exact_and_has_no_prebuilt_native_input(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        result = compose_workflow_lora_graph(session, revision, added_loras=[])

    with pytest.raises(FrozenInstanceError):
        result.__setattr__("version", 2)

    rendered = repr(result)
    assert "_json" not in rendered
    assert "native-private-node" not in rendered
    assert "C:/" not in rendered

    class HostileComposition(WorkflowLoraComposition):
        pass

    hostile = HostileComposition(*(getattr(result, name) for name in result.__slots__))
    with pytest.raises(WorkflowLoraCompositionError):
        workflow_lora_composition_payload(hostile)

    signature = inspect.signature(compose_workflow_lora_graph)
    assert "native" not in signature.parameters
    assert WORKFLOW_LORA_COMPOSITION_VERSION == 1


async def test_self_consistent_forged_native_receipt_cannot_enter_composer_authority(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        forged_graph = deepcopy(revision.api_graph_json)
        forged_graph["arbitrary-executable"] = {
            "class_type": "ArbitraryExecutableNode",
            "inputs": {},
        }
        forged_json = canonical_graph(forged_graph)
        forged = WorkflowLoraGraphResolution(
            version=1,
            source_api_graph_sha256=_graph_sha256(revision.api_graph_json),
            workflow_effective_graph_sha256=_sha256(forged_json),
            override_resolution_sha256="0" * 64,
            overrides=(),
            effective_api_graph_json=forged_json,
        )
        assert workflow_lora_graph_resolution_payload(forged)[
            "workflow_effective_graph_sha256"
        ] == _sha256(forged_json)

        with pytest.raises(TypeError):
            cast(Any, compose_workflow_lora_graph)(
                session,
                revision,
                added_loras=[],
                native=forged,
            )
