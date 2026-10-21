from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from types import MappingProxyType
from typing import Any, cast

import pytest
from httpx2 import AsyncClient
from sqlalchemy.orm import Session

from local_lm.auxiliary_assets import (
    LORA_GRAPH_TRANSFORM_VERSION,
    checkpoint_lora_extension,
    derived_lora_extension,
    detect_lora_extension,
    prompt_trigger_word_provenance,
    resolve_lora_stack,
    resolve_lora_stack_against_graph,
    revision_accepts_added_loras,
    select_automatic_lora_stack,
    transform_lora_graph,
    trigger_words_to_apply,
    validate_lora_workflow_contract,
    workflow_lora_extension,
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


def _graph_hash(graph: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            graph,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class _DictSubclass(dict[str, Any]):
    def items(self) -> Any:
        raise AssertionError("hostile dict methods must not be invoked")


class _ListSubclass(list[Any]):
    pass


class _StringSubclass(str):
    pass


def _workflow(session) -> WorkflowRevision:  # type: ignore[no-untyped-def]
    base = ModelInstall(
        name="Base XL",
        role="image",
        engine="comfyui",
        local_path="C:/managed/base",
        manifest_json={"family": "sdxl"},
        active=True,
    )
    definition = WorkflowDefinition(
        name="LoRA-ready",
        operation="text_to_image",
    )
    session.add_all([base, definition])
    session.flush()
    graph = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base.safetensors"},
        },
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    extension = checkpoint_lora_extension(graph)
    assert extension
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        api_graph_json=graph,
        input_schema_json={
            "type": "object",
            "properties": {
                "loras": {
                    "type": "array",
                    "default": [],
                    "maxItems": 8,
                }
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


def _model_only_workflow(session: Session) -> WorkflowRevision:
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
        "169": {
            "class_type": "KSampler",
            "inputs": {"model": ["163", 0]},
        },
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


def _asset(session, name: str, digest: str) -> ModelAssetInstall:  # type: ignore[no-untyped-def]
    asset = ModelAssetInstall(
        name=name,
        kind="lora",
        family="sdxl",
        local_path=f"C:/managed/{name}",
        size_bytes=1024,
        manifest_json={
            "sha256": digest,
            "comfy_name": f"{name}.safetensors",
            "metadata": {
                "network_type": "networks.lora",
                "rank": 16,
                "trigger_words": [name],
            },
        },
        active=True,
        verified_at=utcnow(),
    )
    session.add(asset)
    session.flush()
    return asset


def _activation(session, revision: WorkflowRevision) -> WorkflowActivation:  # type: ignore[no-untyped-def]
    activation = WorkflowActivation(
        id="wfact_lora_test",
        workflow_revision_id=revision.id,
        resolver_version="workflow-bindings-v1",
        dependency_contract_sha256="a" * 64,
        binding_sha256="b" * 64,
        state="ready",
        is_active=True,
        details_json={"launch_sha256": "c" * 64},
    )
    session.add(activation)
    session.flush()
    return activation


def _bind_model(  # type: ignore[no-untyped-def]
    session,
    revision: WorkflowRevision,
    activation: WorkflowActivation,
    install: ModelInstall,
    ordinal: int,
) -> None:
    slot = WorkflowDependencySlot(
        id=f"wfslot_lora_{ordinal}",
        workflow_revision_id=revision.id,
        name=f"model_{ordinal}",
        resource_kind="model_install",
        required=True,
        satisfaction="all_of",
        requirements_json=[{"key": "default", "constraints": {}}],
        contract_sha256=f"{ordinal + 1:x}" * 64,
        ordinal=ordinal,
    )
    session.add(slot)
    session.flush()
    session.add(
        WorkflowDependencyBinding(
            id=f"wfbind_lora_{ordinal}",
            workflow_revision_id=revision.id,
            workflow_activation_id=activation.id,
            workflow_dependency_slot_id=slot.id,
            requirement_key="default",
            model_install_id=install.id,
            mount_json={},
            resource_identity_json={},
            resource_identity_sha256=f"{ordinal + 3:x}" * 64,
        )
    )
    session.flush()


async def test_a_workflow_accepts_added_loras_only_where_a_stack_would_be_applied(
    client: AsyncClient,
) -> None:
    """The four states the settings panel has to tell apart.

    The panel offers a LoRA control the workflow's schema does not declare,
    and it must offer one exactly where a stack would be applied. Declared and
    recorded is the ordinary workflow. Neither, over a graph a LoRA can be
    read out of, is the case the contract passes over in silence and the run
    still applies. One without the other is refused outright by the contract,
    so such a workflow accepts nothing at all, and a control offered there
    would be a control the run cannot honour.
    """

    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        declared = deepcopy(revision.input_schema_json)
        recorded = deepcopy(revision.dependencies_json)
        bare = {key: value for key, value in recorded.items() if key != "extensions"}
        undeclared = {"type": "object", "properties": {}}

        assert revision_accepts_added_loras(revision) is True

        revision.input_schema_json = undeclared
        revision.dependencies_json = bare
        assert revision_accepts_added_loras(revision) is True

        revision.input_schema_json = declared
        revision.dependencies_json = bare
        assert revision_accepts_added_loras(revision) is False

        revision.input_schema_json = undeclared
        revision.dependencies_json = recorded
        assert revision_accepts_added_loras(revision) is False

        # Nothing said and nothing to read: a graph with no checkpoint for a
        # LoRA to sit behind offers no insertion point to derive.
        revision.input_schema_json = undeclared
        revision.dependencies_json = bare
        revision.api_graph_json = {"9": {"class_type": "SaveImage", "inputs": {}}}
        assert revision_accepts_added_loras(revision) is False
        session.rollback()


def test_lora_workflow_contract_requires_a_real_typed_extension() -> None:
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    schema = {
        "type": "object",
        "properties": {
            "loras": {"type": "array", "maxItems": 8},
        },
    }
    with pytest.raises(ValueError, match="together"):
        validate_lora_workflow_contract(graph, schema, {})
    with pytest.raises(ValueError, match="both model and CLIP"):
        validate_lora_workflow_contract(
            graph,
            schema,
            {"extensions": {"lora": {"model": ["1", 0], "clip": ["1", 1]}}},
        )


async def test_lora_stack_is_validated_and_transformed_deterministically(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        first = _asset(session, "Ink", "a" * 64)
        second = _asset(session, "Paper", "b" * 64)
        stack = [
            {
                "asset_id": first.id,
                "model_strength": 0.8,
                "clip_strength": 0.7,
                "enabled": True,
            },
            {
                "asset_id": second.id,
                "model_strength": 1.1,
                "clip_strength": 1.0,
                "enabled": True,
            },
        ]
        resolved = resolve_lora_stack(session, revision, stack)
        extension = checkpoint_lora_extension(revision.api_graph_json)
        assert extension
        transformed = transform_lora_graph(
            revision.api_graph_json,
            extension,
            [
                {
                    "comfy_name": item["comfy_name"],
                    "model_strength": item["model_strength"],
                    "clip_strength": item["clip_strength"],
                }
                for item in resolved.provenance
            ],
        )
        repeated = resolve_lora_stack(session, revision, stack)
        reversed_stack = resolve_lora_stack(session, revision, list(reversed(stack)))
        detached = resolve_lora_stack_against_graph(
            session,
            revision,
            stack,
            base_api_graph=revision.api_graph_json,
        )

    assert (
        resolved.graph_sha256 == "d42defa39e11215b8b8677c4682579804d771473f1b3f71a70fc909d78c0fbaa"
    )
    assert resolved.graph_sha256 == repeated.graph_sha256
    assert resolved.graph_sha256 == _graph_hash(transformed)
    assert detached.graph_sha256 == resolved.graph_sha256
    assert detached.graph == transformed
    assert detached.settings == resolved.settings
    assert detached.provenance == resolved.provenance
    assert reversed_stack.graph_sha256 != resolved.graph_sha256
    assert [item["asset_id"] for item in resolved.provenance] == [first.id, second.id]
    assert transformed["lma_lora_001"]["class_type"] == "LoraLoader"
    assert transformed["lma_lora_002"]["inputs"]["model"] == ["lma_lora_001", 0]
    assert transformed["2"]["inputs"]["clip"] == ["lma_lora_002", 1]
    assert transformed["3"]["inputs"]["model"] == ["lma_lora_002", 0]
    assert LORA_GRAPH_TRANSFORM_VERSION == "lora-graph-v2"


async def test_detached_resolver_preserves_empty_and_disabled_stack_identity(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "Disabled", "c" * 64)
        base_graph = deepcopy(revision.api_graph_json)
        disabled_stack = [
            {
                "asset_id": asset.id,
                "model_strength": 0.25,
                "clip_strength": 0.5,
                "enabled": False,
            }
        ]

        legacy_empty = resolve_lora_stack(session, revision, [])
        legacy_disabled = resolve_lora_stack(session, revision, disabled_stack)
        empty = resolve_lora_stack_against_graph(
            session,
            revision,
            [],
            base_api_graph=base_graph,
        )
        disabled = resolve_lora_stack_against_graph(
            session,
            revision,
            disabled_stack,
            base_api_graph=base_graph,
        )

    expected_hash = "ad3f55f52d5efa5b2dfe90e135100e9fcae8310918980a8fcd44b9a3c58191a1"
    assert legacy_empty.graph_sha256 == expected_hash
    assert legacy_disabled.graph_sha256 == expected_hash
    assert empty.graph_sha256 == expected_hash == _graph_hash(empty.graph)
    assert disabled.graph_sha256 == expected_hash == _graph_hash(disabled.graph)
    assert empty.graph == base_graph and empty.graph is not base_graph
    assert disabled.graph == base_graph and disabled.graph is not base_graph
    assert empty.settings == [] and empty.provenance == []
    assert disabled.settings == disabled_stack
    assert disabled.provenance[0]["enabled"] is False
    assert "lma_lora_001" not in disabled.graph


async def test_detached_resolver_composes_after_native_scalar_patches_without_mutation(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        authored_graph = deepcopy(revision.api_graph_json)
        authored_graph["native"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": "authored.safetensors",
                "strength_model": 0.45,
                "strength_clip": 0.55,
            },
        }
        authored_graph["2"]["inputs"]["clip"] = ["native", 1]
        authored_graph["3"]["inputs"]["model"] = ["native", 0]
        revision.api_graph_json = authored_graph
        session.flush()

        asset = _asset(session, "Added", "d" * 64)
        stack = [
            {
                "asset_id": asset.id,
                "model_strength": 0.8,
                "clip_strength": 0.7,
                "enabled": True,
            }
        ]
        base_graph = deepcopy(revision.api_graph_json)
        base_graph["native"]["inputs"]["strength_model"] = 0.27
        base_graph["native"]["inputs"]["strength_clip"] = 0.31
        revision_before = deepcopy(revision.api_graph_json)
        base_before = deepcopy(base_graph)

        first = resolve_lora_stack_against_graph(
            session,
            revision,
            stack,
            base_api_graph=base_graph,
        )
        second = resolve_lora_stack_against_graph(
            session,
            revision,
            stack,
            base_api_graph=base_graph,
        )

        assert first.graph["native"]["inputs"]["strength_model"] == 0.27
        assert first.graph["native"]["inputs"]["strength_clip"] == 0.31
        assert first.graph["native"]["inputs"]["model"] == ["lma_lora_001", 0]
        assert first.graph["native"]["inputs"]["clip"] == ["lma_lora_001", 1]
        assert first.graph["2"]["inputs"]["clip"] == ["native", 1]
        assert first.graph["3"]["inputs"]["model"] == ["native", 0]
        assert first.graph["lma_lora_001"]["inputs"] == {
            "model": ["1", 0],
            "lora_name": "Added.safetensors",
            "strength_model": 0.8,
            "clip": ["1", 1],
            "strength_clip": 0.7,
        }
        assert first.graph_sha256 == _graph_hash(first.graph)
        assert second.graph_sha256 == first.graph_sha256
        assert revision.api_graph_json == revision_before
        assert base_graph == base_before

        first.graph["native"]["inputs"]["strength_model"] = 3.5
        first.graph["lma_lora_001"]["inputs"]["model"][0] = "mutated"

        assert second.graph["native"]["inputs"]["strength_model"] == 0.27
        assert second.graph["lma_lora_001"]["inputs"]["model"] == ["1", 0]
        assert revision.api_graph_json == revision_before
        assert base_graph == base_before


async def test_detached_model_only_resolution_matches_the_frozen_legacy_hash(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _model_only_workflow(session)
        first = _asset(session, "detail", "e" * 64)
        second = _asset(session, "style", "f" * 64)
        stack = [
            {
                "asset_id": first.id,
                "model_strength": 0.8,
                "clip_strength": 0.6,
            },
            {
                "asset_id": second.id,
                "model_strength": 1.1,
                "clip_strength": 0.9,
            },
        ]
        source_graph = deepcopy(revision.api_graph_json)
        legacy = resolve_lora_stack(session, revision, stack)
        resolved = resolve_lora_stack_against_graph(
            session,
            revision,
            stack,
            base_api_graph=source_graph,
        )

    expected_hash = "7e9cd73af1f088f6f4d5205cb5e02f01b516152db85ffd1352a1106c3dddd65e"
    assert legacy.graph_sha256 == expected_hash
    assert resolved.graph_sha256 == expected_hash == _graph_hash(resolved.graph)
    assert resolved.settings == legacy.settings
    assert resolved.provenance == legacy.provenance
    assert resolved.graph["lma_lora_001"]["class_type"] == "LoraLoaderModelOnly"
    assert resolved.graph["lma_lora_002"]["inputs"]["model"] == ["lma_lora_001", 0]
    assert resolved.graph["169"]["inputs"]["model"] == ["lma_lora_002", 0]
    assert "clip" not in resolved.graph["lma_lora_001"]["inputs"]
    assert "strength_clip" not in resolved.graph["lma_lora_001"]["inputs"]
    assert source_graph == revision.api_graph_json


async def test_detached_resolver_uses_exact_activation_family_without_legacy_ids(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        base_id = revision.dependencies_json["model_install_ids"][0]
        base = session.get(ModelInstall, base_id)
        assert base
        activation = _activation(session, revision)
        _bind_model(session, revision, activation, base, 0)
        revision.dependencies_json = {
            "extensions": revision.dependencies_json["extensions"],
        }
        asset = _asset(session, "ActivationFamily", "1" * 64)
        asset.family = "flux"
        session.flush()
        stack = [{"asset_id": asset.id}]

        legacy = resolve_lora_stack(session, revision, stack)
        omitted = resolve_lora_stack_against_graph(
            session,
            revision,
            stack,
            base_api_graph=revision.api_graph_json,
        )
        assert omitted.graph_sha256 == legacy.graph_sha256
        assert omitted.settings == legacy.settings
        assert omitted.provenance == legacy.provenance
        assert omitted.graph["lma_lora_001"]["inputs"]["lora_name"] == (
            "ActivationFamily.safetensors"
        )

        with pytest.raises(ValueError, match="incompatible"):
            resolve_lora_stack_against_graph(
                session,
                revision,
                stack,
                base_api_graph=revision.api_graph_json,
                workflow_activation_id=activation.id,
            )

        asset.family = "sdxl"
        session.flush()
        matching = resolve_lora_stack_against_graph(
            session,
            revision,
            stack,
            base_api_graph=revision.api_graph_json,
            workflow_activation_id=activation.id,
        )

    assert matching.graph_sha256 == omitted.graph_sha256
    assert matching.graph["lma_lora_001"]["inputs"]["lora_name"] == ("ActivationFamily.safetensors")


@pytest.mark.parametrize(
    ("failure_kind", "message"),
    [
        ("malformed_type", "identity is invalid"),
        ("malformed_text", "identity is invalid"),
        ("unavailable", "is unavailable"),
        ("foreign_revision", "selected workflow revision"),
        ("empty_family", "exactly one model family"),
        ("mixed_family", "exactly one model family"),
    ],
)
async def test_detached_resolver_fails_closed_on_invalid_activation_family_evidence(
    client: AsyncClient,
    failure_kind: str,
    message: str,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        base_id = revision.dependencies_json["model_install_ids"][0]
        base = session.get(ModelInstall, base_id)
        assert base
        activation = _activation(session, revision)
        _bind_model(session, revision, activation, base, 0)
        revision.dependencies_json = {
            "extensions": revision.dependencies_json["extensions"],
        }
        asset = _asset(session, "ActivationEvidence", "2" * 64)
        session.flush()
        stack = [{"asset_id": asset.id}]

        control = resolve_lora_stack_against_graph(
            session,
            revision,
            stack,
            base_api_graph=revision.api_graph_json,
            workflow_activation_id=activation.id,
        )
        assert control.graph["lma_lora_001"]["inputs"]["lora_name"] == (
            "ActivationEvidence.safetensors"
        )

        activation_id: str = activation.id
        selected_revision = revision
        if failure_kind == "malformed_type":
            activation_id = _StringSubclass(activation.id)
        elif failure_kind == "malformed_text":
            activation_id = f" {activation.id}"
        elif failure_kind == "unavailable":
            activation_id = "wfact_missing"
        elif failure_kind == "foreign_revision":
            selected_revision = WorkflowRevision(
                workflow_id=revision.workflow_id,
                version=revision.version + 1,
                engine=revision.engine,
                api_graph_json=deepcopy(revision.api_graph_json),
                input_schema_json=deepcopy(revision.input_schema_json),
                dependencies_json=deepcopy(revision.dependencies_json),
                trusted=True,
            )
            session.add(selected_revision)
            session.flush()
        elif failure_kind == "empty_family":
            base.manifest_json = {"family": ""}
            session.flush()
        else:
            other = ModelInstall(
                name="Foreign architecture component",
                role="image",
                engine="comfyui",
                local_path="C:/managed/foreign-architecture",
                manifest_json={"family": "flux"},
                active=True,
            )
            session.add(other)
            session.flush()
            _bind_model(session, revision, activation, other, 1)

        with pytest.raises(ValueError, match=message):
            resolve_lora_stack_against_graph(
                session,
                selected_revision,
                stack,
                base_api_graph=selected_revision.api_graph_json,
                workflow_activation_id=activation_id,
            )


@pytest.mark.parametrize(
    "hostile_kind",
    [
        "top_dict_subclass",
        "nested_dict_subclass",
        "list_subclass",
        "tuple",
        "mapping_proxy",
        "non_finite",
        "non_json_value",
        "shared_container",
    ],
)
async def test_detached_resolver_rejects_hostile_json_containers(
    client: AsyncClient,
    hostile_kind: str,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        graph: object = deepcopy(revision.api_graph_json)
        assert isinstance(graph, dict)
        if hostile_kind == "top_dict_subclass":
            graph = _DictSubclass(graph)
        elif hostile_kind == "nested_dict_subclass":
            graph["3"] = _DictSubclass(graph["3"])
        elif hostile_kind == "list_subclass":
            graph["3"]["inputs"]["model"] = _ListSubclass(["1", 0])
        elif hostile_kind == "tuple":
            graph["3"]["inputs"]["model"] = ("1", 0)
        elif hostile_kind == "mapping_proxy":
            graph["3"]["inputs"] = MappingProxyType(graph["3"]["inputs"])
        elif hostile_kind == "non_finite":
            graph["3"]["inputs"]["cfg"] = float("nan")
        elif hostile_kind == "non_json_value":
            graph["3"]["inputs"]["cfg"] = object()
        else:
            shared_link = ["1", 0]
            graph["3"]["inputs"]["model"] = shared_link
            graph["3"]["inputs"]["also_model"] = shared_link

        with pytest.raises(ValueError, match="LoRA base graph"):
            resolve_lora_stack_against_graph(
                session,
                revision,
                [],
                base_api_graph=graph,
            )


@pytest.mark.parametrize(
    "failure_kind",
    ["changed_boundary", "reserved_node", "malformed_node"],
)
async def test_detached_resolver_rejects_incompatible_or_reserved_base_graphs(
    client: AsyncClient,
    failure_kind: str,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        base_graph = deepcopy(revision.api_graph_json)
        if failure_kind == "changed_boundary":
            base_graph["3"]["inputs"]["model"] = ["missing", 0]
            message = "both model and CLIP"
        elif failure_kind == "reserved_node":
            base_graph["lma_lora_999"] = {
                "class_type": "LoraLoader",
                "inputs": {},
            }
            message = "reserves"
        else:
            base_graph["broken"] = {"class_type": "KSampler"}
            message = "valid ComfyUI API graph"
        revision_before = deepcopy(revision.api_graph_json)
        base_before = deepcopy(base_graph)

        with pytest.raises(ValueError, match=message):
            resolve_lora_stack_against_graph(
                session,
                revision,
                [],
                base_api_graph=base_graph,
            )

        assert revision.api_graph_json == revision_before
        assert base_graph == base_before


async def test_lora_trigger_word_sources_have_stable_bounded_precedence(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "Ink", "c" * 64)
        asset.manifest_json["metadata"] = {
            "trigger_words": [" ink wash ", "Soft Light"],
            "trained_words": ["soft light", "provider ink"],
        }
        stack = [{"asset_id": asset.id, "model_strength": 1, "clip_strength": 1}]

        resolved = resolve_lora_stack(session, revision, stack)

    assert resolved.provenance[0]["trigger_words"] == [
        "ink wash",
        "Soft Light",
        "provider ink",
    ]
    assert prompt_trigger_word_provenance(
        None,
        resolved.provenance,
        "An ink wash portrait",
    ) == {
        "model_trigger_words_applied": [],
        "lora_trigger_words_applied": ["Soft Light", "provider ink"],
        "trigger_words_applied": ["Soft Light", "provider ink"],
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {"trained_words": "private provider text"},
        {"trained_words": ["private provider text", 7]},
        {"trained_words": ["x" * 201]},
        {"trained_words": [f"word-{index}" for index in range(101)]},
        {
            "trigger_words": [f"manifest-{index}" for index in range(100)],
            "trained_words": ["provider-overflow"],
        },
        "private provider text",
    ],
)
async def test_lora_trigger_word_metadata_refuses_malformed_or_unbounded_values(
    client: AsyncClient,
    metadata: object,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "Ink", "c" * 64)
        asset.manifest_json["metadata"] = metadata
        stack = [{"asset_id": asset.id, "model_strength": 1, "clip_strength": 1}]

        with pytest.raises(ValueError) as raised:
            resolve_lora_stack(session, revision, stack)

    assert str(raised.value) == "A selected LoRA has invalid trigger-word metadata."
    assert "private provider text" not in str(raised.value)


async def test_lora_without_declared_trigger_words_keeps_an_empty_vocabulary(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "Ink", "c" * 64)
        asset.manifest_json["metadata"] = {"network_type": "networks.lora"}
        stack = [{"asset_id": asset.id, "model_strength": 1, "clip_strength": 1}]

        resolved = resolve_lora_stack(session, revision, stack)

    assert resolved.provenance[0]["trigger_words"] == []


async def test_typed_trigger_words_follow_the_file_s_own_and_are_applied_the_same_way(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "Ink", "c" * 64)
        asset.manifest_json["metadata"] = {"trigger_words": ["ink wash"]}
        asset.typed_trigger_words = ["Studio Glow", "INK WASH"]
        stack = [{"asset_id": asset.id, "model_strength": 1, "clip_strength": 1}]

        resolved = resolve_lora_stack(session, revision, stack)

    # The file's words first, then the typed ones, one of each in any casing.
    assert resolved.provenance[0]["trigger_words"] == ["ink wash", "Studio Glow"]
    assert prompt_trigger_word_provenance(None, resolved.provenance, "A portrait") == {
        "model_trigger_words_applied": [],
        "lora_trigger_words_applied": ["ink wash", "Studio Glow"],
        "trigger_words_applied": ["ink wash", "Studio Glow"],
    }


async def test_typed_trigger_words_survive_the_manifest_being_rewritten(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "Ink", "c" * 64)
        asset.typed_trigger_words = ["studio glow"]
        # A new record of the file's own words, as measuring it again would write.
        asset.manifest_json = {
            **asset.manifest_json,
            "metadata": {"network_type": "networks.lora"},
        }
        stack = [{"asset_id": asset.id, "model_strength": 1, "clip_strength": 1}]

        resolved = resolve_lora_stack(session, revision, stack)

    assert resolved.provenance[0]["trigger_words"] == ["studio glow"]


@pytest.mark.parametrize(
    "typed",
    [
        "studio glow",
        ["studio glow", 7],
        ["x" * 201],
        [" "],
        [f"typed-{index}" for index in range(100)],
    ],
)
async def test_stored_typed_trigger_words_are_refused_when_malformed(
    client: AsyncClient,
    typed: object,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "Ink", "c" * 64)
        cast(Any, asset).typed_trigger_words = typed
        stack = [{"asset_id": asset.id, "model_strength": 1, "clip_strength": 1}]

        with pytest.raises(ValueError) as raised:
            resolve_lora_stack(session, revision, stack)

    assert str(raised.value) == "A selected LoRA has invalid trigger-word metadata."


async def test_lora_stack_rejects_duplicates_incompatible_and_unavailable_assets(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "Mismatch", "c" * 64)
        asset.family = "flux"
        session.flush()
        try:
            resolve_lora_stack(
                session,
                revision,
                [{"asset_id": asset.id, "model_strength": 1, "clip_strength": 1}],
            )
        except ValueError as exc:
            assert "incompatible" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("family mismatch was accepted")

        asset.family = "sdxl"
        session.flush()
        duplicate = {"asset_id": asset.id, "model_strength": 1, "clip_strength": 1}
        try:
            resolve_lora_stack(session, revision, [duplicate, duplicate])
        except ValueError as exc:
            assert "same asset twice" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("duplicate LoRA was accepted")

        asset.active = False
        session.flush()
        try:
            resolve_lora_stack(session, revision, [duplicate])
        except ValueError as exc:
            assert "unavailable" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("inactive LoRA was accepted")


async def test_automatic_lora_selection_is_conservative_deterministic_and_prompt_free(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        watercolor = _asset(session, "Watercolor", "d" * 64)
        watercolor.use_case = "watercolor landscapes"
        watercolor.auto_apply = True
        watercolor.default_model_strength = 0.75
        watercolor.default_clip_strength = 0.6
        portrait = _asset(session, "Portrait", "e" * 64)
        portrait.use_case = "portrait lighting"
        portrait.auto_apply = True
        portrait.default_model_strength = 0.9
        portrait.default_clip_strength = 0.8
        incompatible = _asset(session, "Flux", "f" * 64)
        incompatible.family = "flux"
        incompatible.use_case = "watercolor landscapes"
        incompatible.auto_apply = True
        inactive = _asset(session, "Inactive", "1" * 64)
        inactive.use_case = "watercolor landscapes"
        inactive.auto_apply = True
        inactive.active = False
        unverified = _asset(session, "Unverified", "2" * 64)
        unverified.use_case = "watercolor landscapes"
        unverified.auto_apply = True
        unverified.verified_at = None
        session.flush()

        prompt = "Create watercolor landscapes with dramatic portrait lighting"
        selected = select_automatic_lora_stack(session, revision, prompt)
        repeated = select_automatic_lora_stack(session, revision, prompt)

    assert selected == repeated
    assert [item["asset_id"] for item in selected.settings] == [portrait.id, watercolor.id]
    assert selected.settings == [
        {
            "asset_id": portrait.id,
            "model_strength": 0.9,
            "clip_strength": 0.8,
            "enabled": True,
        },
        {
            "asset_id": watercolor.id,
            "model_strength": 0.75,
            "clip_strength": 0.6,
            "enabled": True,
        },
    ]
    assert selected.provenance["mode"] == "automatic"
    assert selected.provenance["selector_version"] == "lora-use-case-v1"
    assert prompt not in json.dumps(selected.provenance)
    assert selected.provenance["selected"][0]["matched_terms"] == ["lighting", "portrait"]


async def test_automatic_lora_selection_requires_a_meaningful_conservative_match(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        broad = _asset(session, "Broad", "3" * 64)
        broad.use_case = "image style"
        broad.auto_apply = True
        partial = _asset(session, "Partial", "4" * 64)
        partial.use_case = "cinematic product photography"
        partial.auto_apply = True
        substring = _asset(session, "Substring", "5" * 64)
        substring.use_case = "car"
        substring.auto_apply = True
        session.flush()

        selected = select_automatic_lora_stack(
            session,
            revision,
            "Create a cinematic cartoon portrait",
        )

    assert selected.settings == []
    assert selected.provenance["selected"] == []


def test_model_only_lora_extension_is_detected_and_transformed() -> None:
    graph = {
        "161": {"class_type": "UNETLoader", "inputs": {}},
        "145": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["161", 0]},
        },
        "152": {"class_type": "CFGNorm", "inputs": {"model": ["145", 0]}},
        "153": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["152", 0], "lora_name": "lightning.safetensors"},
        },
        "163": {
            "class_type": "ComfySwitchNode",
            "inputs": {"on_true": ["153", 0], "on_false": ["152", 0]},
        },
        "169": {"class_type": "KSampler", "inputs": {"model": ["163", 0]}},
    }
    extension = detect_lora_extension(graph)
    assert extension == {"mode": "model_only", "model": ["163", 0]}
    schema = {
        "type": "object",
        "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
    }
    dependencies = {"extensions": {"lora": extension}}
    validate_lora_workflow_contract(graph, schema, dependencies)

    transformed = transform_lora_graph(
        graph,
        extension,
        [
            {
                "comfy_name": "detail.safetensors",
                "model_strength": 0.8,
                "clip_strength": 0.6,
            },
            {
                "comfy_name": "style.safetensors",
                "model_strength": 1.1,
                "clip_strength": 0.9,
            },
        ],
    )

    assert transformed["lma_lora_001"] == {
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "LM Atelier LoRA 1"},
        "inputs": {
            "model": ["163", 0],
            "lora_name": "detail.safetensors",
            "strength_model": 0.8,
        },
    }
    assert transformed["lma_lora_002"]["inputs"]["model"] == ["lma_lora_001", 0]
    assert transformed["169"]["inputs"]["model"] == ["lma_lora_002", 0]
    assert transformed["163"] == graph["163"]


def test_model_only_lora_extension_fails_closed_for_multiple_model_paths() -> None:
    graph = {
        "base": {"class_type": "UNETLoader", "inputs": {}},
        "refiner": {"class_type": "UNETLoader", "inputs": {}},
        "first": {"class_type": "KSampler", "inputs": {"model": ["base", 0]}},
        "second": {
            "class_type": "KSamplerAdvanced",
            "inputs": {"model": ["refiner", 0]},
        },
    }
    assert detect_lora_extension(graph) is None
    schema = {
        "type": "object",
        "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
    }
    dependencies = {"extensions": {"lora": {"mode": "model_only", "model": ["base", 0]}}}
    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(graph, schema, dependencies)


def test_trigger_words_apply_once_and_never_repeat_the_prompt() -> None:
    provenance: list[dict[str, Any]] = [
        {"enabled": True, "trigger_words": ["m1ssi0nary", "soft light"]},
        {"enabled": True, "trigger_words": ["Soft Light", "film grain"]},
        {"enabled": False, "trigger_words": ["disabled-word"]},
        {"enabled": True},
    ]

    applied = trigger_words_to_apply(provenance, "A portrait in soft light by a window")

    # Deduplicated across items, case-insensitive against the prompt, and
    # words from disabled items never leak in.
    assert applied == ["m1ssi0nary", "film grain"]
    assert trigger_words_to_apply(provenance, "m1ssi0nary shot on film grain in Soft Light") == []
    assert trigger_words_to_apply([], "anything") == []


def test_model_and_lora_trigger_words_share_one_deduplicated_snapshot() -> None:
    model = {
        "manifest": {"trigger_words": ["portrait-style", "soft light"]},
        "source": {"metadata": {"trained_words": ["Portrait-Style", "film grain"]}},
    }
    loras = [
        {"enabled": True, "trigger_words": ["soft light", "atelier ink"]},
        {"enabled": False, "trigger_words": ["disabled-word"]},
    ]

    provenance = prompt_trigger_word_provenance(
        model,
        loras,
        "A candid portrait with film grain",
    )

    assert provenance == {
        "model_trigger_words_applied": ["portrait-style", "soft light"],
        "lora_trigger_words_applied": ["atelier ink"],
        "trigger_words_applied": ["portrait-style", "soft light", "atelier ink"],
    }


async def test_an_unknown_architecture_receives_no_automatic_lora(
    client: AsyncClient,
) -> None:
    """The compatibility test can only skip an asset when the workflow's family
    is known, so an unknown family used to mean no test at all - every auto-apply
    LoRA was eligible for every workflow. An adapter for another architecture
    does not refuse to load; it degrades the image while provenance reports
    success."""

    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "anything", "d" * 64)
        asset.auto_apply = True
        asset.family = "sdxl"
        asset.use_case = "portrait"
        session.flush()

        # The same asset against a workflow that does say what it runs, so the
        # assertion below cannot pass merely because nothing was ever eligible.
        eligible = select_automatic_lora_stack(session, revision, "a detailed portrait")
        assert eligible.settings, "fixture no longer selects; the test would prove nothing"

        revision.dependencies_json = {
            "extensions": {"lora": revision.dependencies_json["extensions"]["lora"]}
        }
        session.flush()

        refused = select_automatic_lora_stack(session, revision, "a detailed portrait")
        assert refused.settings == []
        assert refused.provenance == {
            "mode": "automatic",
            "selector_version": "lora-use-case-v1",
            "selected": [],
            "skipped_reason": "workflow_architecture_unknown",
        }


async def test_automatic_lora_uses_complete_activation_bindings(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        base_id = revision.dependencies_json["model_install_ids"][0]
        base = session.get(ModelInstall, base_id)
        assert base
        asset = _asset(session, "Portrait", "e" * 64)
        asset.auto_apply = True
        asset.use_case = "portrait"
        activation = _activation(session, revision)
        _bind_model(session, revision, activation, base, 0)

        # Portable workflow metadata may carry a foreign install ID. The
        # activation is the content-resolved local binding and must win.
        revision.dependencies_json = {
            "model_install_ids": ["portable-origin-id"],
            "extensions": revision.dependencies_json["extensions"],
        }
        session.flush()
        selected = select_automatic_lora_stack(
            session,
            revision,
            "a detailed portrait",
            workflow_activation_id=activation.id,
        )
        assert [item["asset_id"] for item in selected.settings] == [asset.id]

        # One known family must not conceal another unresolved model binding.
        unresolved = ModelInstall(
            id="model_without_family",
            name="Unclassified component",
            role="image",
            engine="comfyui",
            local_path="C:/managed/unclassified",
            manifest_json={},
            active=True,
        )
        session.add(unresolved)
        session.flush()
        _bind_model(session, revision, activation, unresolved, 1)

        refused = select_automatic_lora_stack(
            session,
            revision,
            "a detailed portrait",
            workflow_activation_id=activation.id,
        )
        assert refused.settings == []
        assert refused.provenance["skipped_reason"] == "workflow_architecture_unknown"


def _carried_over_workflow(session: Session) -> WorkflowRevision:
    """A revision whose graph offers one insertion point and records none.

    The shape a workflow arrives in when it was brought across rather than
    compiled here: a toggle chooses between the raw model and the end of a
    built-in adapter chain, and the sampler reads the toggle. The chain and the
    toggle belong to the workflow; the point every sampler reads is still
    single and unambiguous.
    """

    base = ModelInstall(
        name="Carried base",
        role="image",
        engine="comfyui",
        local_path="C:/managed/carried",
        manifest_json={"family": "sdxl"},
        active=True,
    )
    definition = WorkflowDefinition(name="Carried over", operation="text_to_image")
    session.add_all([base, definition])
    session.flush()
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        api_graph_json={
            "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "base.safetensors"}},
            "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": "text.safetensors"}},
            "12": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["10", 0],
                    "clip": ["11", 0],
                    "lora_name": "built-in.safetensors",
                },
            },
            "13": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
            "14": {
                "class_type": "ComfySwitchNode",
                "inputs": {"switch": ["13", 0], "on_false": ["10", 0], "on_true": ["12", 0]},
            },
            "15": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["12", 1]}},
            "16": {"class_type": "KSampler", "inputs": {"model": ["14", 0], "positive": ["15", 0]}},
        },
        input_schema_json={"type": "object", "properties": {}},
        dependencies_json={"model_install_ids": [base.id]},
        trusted=True,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    return revision


async def test_a_graph_that_offers_one_insertion_point_provides_it_unrecorded(
    client: AsyncClient,
) -> None:
    """Nothing can add the record afterwards, so its absence cannot be final."""

    del client
    with SessionLocal() as session:
        revision = _carried_over_workflow(session)

        assert revision.dependencies_json.get("extensions") is None
        assert workflow_lora_extension(revision) == {"mode": "model_only", "model": ["14", 0]}


async def test_a_recorded_insertion_point_is_read_rather_than_measured(
    client: AsyncClient,
) -> None:
    """A revision that says where its LoRAs go is believed over its graph."""

    del client
    with SessionLocal() as session:
        revision = _carried_over_workflow(session)
        revision.dependencies_json = {
            **revision.dependencies_json,
            "extensions": {"lora": {"mode": "model_only", "model": ["12", 0]}},
        }

        assert workflow_lora_extension(revision) == {"mode": "model_only", "model": ["12", 0]}


async def test_a_malformed_record_refuses_instead_of_measuring_the_graph(
    client: AsyncClient,
) -> None:
    """A wrong record is damage to report, not an absence to fill in."""

    del client
    with SessionLocal() as session:
        revision = _carried_over_workflow(session)
        revision.dependencies_json = {
            **revision.dependencies_json,
            "extensions": {"lora": {"mode": "model_only", "model": "not a link"}},
        }

        assert derived_lora_extension(revision.api_graph_json) is not None
        assert workflow_lora_extension(revision) is None


async def test_samplers_reading_different_models_offer_no_insertion_point(
    client: AsyncClient,
) -> None:
    """Injecting where only one of them reads would change half the picture."""

    del client
    with SessionLocal() as session:
        revision = _carried_over_workflow(session)
        revision.api_graph_json = {
            **revision.api_graph_json,
            "17": {"class_type": "KSampler", "inputs": {"model": ["12", 0]}},
        }

        assert workflow_lora_extension(revision) is None


_CHECKPOINT_LORAS = {"model": ["ck", 0], "clip": ["ck", 1]}


def _checkpoint_graph(**nodes: dict[str, Any]) -> dict[str, Any]:
    # One checkpoint feeding the prompt and a first pass; each case adds its own nodes.
    return {
        "ck": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ck", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ck", 1]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {}},
        "first": _sampled(["ck", 0]),
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": "second"}},
        **nodes,
    }


def _guided(guider: str) -> dict[str, Any]:
    return {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {"guider": [guider, 0], "latent_image": ["latent", 0]},
    }


# Each case names the nodes it adds; they are built per test so no case shares a node.
_GraphCases = dict[str, dict[str, Callable[[], dict[str, Any]]]]

_SAMPLERS_THE_CHECKPOINT_MISSES: _GraphCases = {
    "a second loader": {"second": lambda: _sampled(["u", 0])},
    "a patch on a second loader": {
        "p": lambda: {"class_type": "ModelSamplingSD3", "inputs": {"model": ["u", 0]}},
        "second": lambda: _sampled(["p", 0]),
    },
    "a guider on a second loader": {
        "g": lambda: {"class_type": "CFGGuider", "inputs": {"model": ["u", 0]}},
        "second": lambda: _guided("g"),
    },
    "a negative model on a second loader": {
        "g": lambda: {
            "class_type": "DualModelGuider",
            "inputs": {"model": ["ck", 0], "model_negative": ["u", 0]},
        },
        "second": lambda: _guided("g"),
    },
    "an unknown guider on a second loader": {
        "g": lambda: {"class_type": "SomeCustomGuider", "inputs": {"model": ["u", 0]}},
        "second": lambda: _guided("g"),
    },
    "an adapter on a second loader fed the first picture": {
        "decode": lambda: {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["first", 0], "vae": ["ck", 2]},
        },
        "p": lambda: {
            "class_type": "SomeImageAdapter",
            "inputs": {"model": ["u", 0], "image": ["decode", 0]},
        },
        "second": lambda: _sampled(["p", 0]),
    },
    "a LoRA loader on a second loader": {
        "l": lambda: {"class_type": "LoraLoader", "inputs": {"model": ["u", 0], "clip": ["ck", 1]}},
        "second": lambda: _sampled(["l", 0]),
    },
    "a loader the app does not know": {
        "other": lambda: {"class_type": "SomeModelLoader", "inputs": {}},
        "second": lambda: _sampled(["other", 0]),
    },
}


@pytest.mark.parametrize("case", sorted(_SAMPLERS_THE_CHECKPOINT_MISSES))
def test_checkpoint_loras_refuse_a_sampler_on_a_separately_loaded_model(case: str) -> None:
    graph = _checkpoint_graph(
        **{name: build() for name, build in _SAMPLERS_THE_CHECKPOINT_MISSES[case].items()}
    )

    assert checkpoint_lora_extension(graph) is None
    assert detect_lora_extension(graph) is None
    with pytest.raises(ValueError, match="model of every supported sampler"):
        validate_lora_workflow_contract(
            graph, _MODEL_ONLY_SCHEMA, {"extensions": {"lora": _CHECKPOINT_LORAS}}
        )


_SAMPLERS_THE_CHECKPOINT_REACHES: _GraphCases = {
    "a LoRA loader and an image adapter": {
        "l": lambda: {
            "class_type": "LoraLoader",
            "inputs": {"model": ["ck", 0], "clip": ["ck", 1]},
        },
        "image": lambda: {"class_type": "LoadImage", "inputs": {"image": "reference.png"}},
        "a": lambda: {
            "class_type": "SomeImageAdapter",
            "inputs": {"model": ["l", 0], "image": ["image", 0]},
        },
        "second": lambda: _sampled(["a", 0]),
    },
    "a second pass on the same model": {
        "second": lambda: _sampled(["ck", 0], "KSamplerAdvanced"),
    },
    "a guider on the checkpoint": {
        "g": lambda: {"class_type": "CFGGuider", "inputs": {"model": ["ck", 0]}},
        "second": lambda: _guided("g"),
    },
    # Part of what these sample carries the LoRAs, so nothing proves they miss them.
    "a merge with the checkpoint": {
        "m": lambda: {
            "class_type": "ModelMergeSimple",
            "inputs": {"model1": ["ck", 0], "model2": ["u", 0]},
        },
        "second": lambda: _sampled(["m", 0]),
    },
    "a switch with the checkpoint": {
        "s": lambda: {
            "class_type": "ComfySwitchNode",
            "inputs": {"on_true": ["ck", 0], "on_false": ["u", 0]},
        },
        "second": lambda: _sampled(["s", 0]),
    },
    # A sampler the contract cannot read has always been passed over.
    "an unrecognised sampler on a second loader": {
        "second": lambda: {**_sampled(["u", 0]), "class_type": "SomeCustomSampler"},
    },
}


@pytest.mark.parametrize("case", sorted(_SAMPLERS_THE_CHECKPOINT_REACHES))
def test_checkpoint_loras_keep_graphs_whose_samplers_can_receive_them(case: str) -> None:
    graph = _checkpoint_graph(
        **{name: build() for name, build in _SAMPLERS_THE_CHECKPOINT_REACHES[case].items()}
    )

    assert checkpoint_lora_extension(graph) == _CHECKPOINT_LORAS
    assert detect_lora_extension(graph) == _CHECKPOINT_LORAS
    validate_lora_workflow_contract(
        graph, _MODEL_ONLY_SCHEMA, {"extensions": {"lora": _CHECKPOINT_LORAS}}
    )


def test_checkpoint_loras_move_to_the_model_a_sampler_reads() -> None:
    # The checkpoint lends its text encoder; a separately loaded model does the sampling.
    graph = _checkpoint_graph(
        first=_sampled(["u", 0]),
        save={"class_type": "ModelSave", "inputs": {"model": ["ck", 0]}},
    )

    assert checkpoint_lora_extension(graph) is None
    assert detect_lora_extension(graph) == {"mode": "model_only", "model": ["u", 0]}


def test_a_recorded_checkpoint_link_refuses_a_sampler_reading_above_it() -> None:
    graph = _checkpoint_graph(
        l={"class_type": "LoraLoader", "inputs": {"model": ["ck", 0], "clip": ["ck", 1]}},
        pos={"class_type": "CLIPTextEncode", "inputs": {"clip": ["l", 1]}},
        first=_sampled(["l", 0]),
        second=_sampled(["ck", 0]),
    )
    recorded = {"extensions": {"lora": {"model": ["l", 0], "clip": ["l", 1]}}}

    with pytest.raises(ValueError, match="model of every supported sampler"):
        validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, recorded)
    validate_lora_workflow_contract(
        graph, _MODEL_ONLY_SCHEMA, {"extensions": {"lora": _CHECKPOINT_LORAS}}
    )


async def test_a_recorded_extension_that_misses_a_sampler_takes_no_loras(
    client: AsyncClient,
) -> None:
    """A stack chosen for it would be refused, even one the person never asked for."""

    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        watercolor = _asset(session, "Watercolor", "d" * 64)
        watercolor.use_case = "watercolor landscapes"
        watercolor.auto_apply = True
        session.flush()
        prompt = "Create watercolor landscapes"
        assert select_automatic_lora_stack(session, revision, prompt).settings

        revision.api_graph_json = {
            **revision.api_graph_json,
            "4": {"class_type": "UNETLoader", "inputs": {}},
            "5": {"class_type": "KSampler", "inputs": {"model": ["4", 0]}},
        }
        stack = [
            {
                "asset_id": watercolor.id,
                "model_strength": 0.8,
                "clip_strength": 0.7,
                "enabled": True,
            }
        ]

        assert revision_accepts_added_loras(revision) is False
        assert select_automatic_lora_stack(session, revision, prompt).settings == []
        with pytest.raises(ValueError, match="model of every supported sampler"):
            resolve_lora_stack(session, revision, stack)
        assert resolve_lora_stack(session, revision, []).settings == []
        session.rollback()


def test_checkpoint_reach_is_measured_in_linear_time() -> None:
    import time

    size = 40_000
    chain = {
        f"m{index}": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["u" if index == 0 else f"m{index - 1}", 0]},
        }
        for index in range(size)
    }
    graph = _checkpoint_graph(**chain, second=_sampled([f"m{size - 1}", 0]))
    reached = _checkpoint_graph(
        **{**chain, "m0": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["ck", 0]}}},
        second=_sampled([f"m{size - 1}", 0]),
    )

    started = time.perf_counter()
    assert checkpoint_lora_extension(graph) is None
    assert checkpoint_lora_extension(reached) == _CHECKPOINT_LORAS
    assert time.perf_counter() - started < 3


async def test_a_graph_holding_a_reserved_identifier_offers_nothing(
    client: AsyncClient,
) -> None:
    """The insertion would collide with a node the graph already owns."""

    del client
    with SessionLocal() as session:
        revision = _carried_over_workflow(session)
        revision.api_graph_json = {
            **revision.api_graph_json,
            "lma_lora_001": {"class_type": "LoraLoader", "inputs": {"model": ["10", 0]}},
        }

        assert workflow_lora_extension(revision) is None


async def test_a_carried_over_workflow_runs_a_lora_stack(
    client: AsyncClient,
) -> None:
    """The whole point: a stack now reaches the graph this workflow runs."""

    del client
    with SessionLocal() as session:
        revision = _carried_over_workflow(session)
        asset = _asset(session, "Ink", "a" * 64)
        stack = [
            {"asset_id": asset.id, "model_strength": 1.0, "clip_strength": 1.0, "enabled": True}
        ]

        resolved = resolve_lora_stack(session, revision, stack)
        detached = resolve_lora_stack_against_graph(
            session,
            revision,
            stack,
            base_api_graph=revision.api_graph_json,
        )

    assert [item["asset_id"] for item in resolved.provenance] == [asset.id]
    assert resolved.graph_sha256 != _graph_hash(revision.api_graph_json)
    assert detached.graph["lma_lora_001"]["class_type"] == "LoraLoaderModelOnly"
    assert detached.graph["lma_lora_001"]["inputs"]["model"] == ["14", 0]
    assert detached.graph["lma_lora_001"]["inputs"]["lora_name"] == "Ink.safetensors"
    assert detached.graph["16"]["inputs"]["model"] == ["lma_lora_001", 0]
    # Model-only insertion leaves the text encoder reading the graph's own CLIP.
    assert detached.graph["15"]["inputs"]["clip"] == ["12", 1]


@pytest.mark.parametrize("derived", [True, False])
async def test_automatic_lora_selection_records_the_saved_use_case_origin(
    client: AsyncClient, derived: bool
) -> None:
    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "Watercolor", "d" * 64)
        asset.use_case = "watercolor landscapes"
        asset.use_case_derived = derived
        asset.auto_apply = True
        session.flush()
        selection = select_automatic_lora_stack(session, revision, "Watercolor landscapes")
    assert [item["asset_id"] for item in selection.settings] == [asset.id]
    assert selection.provenance["selected"][0]["use_case"] == "watercolor landscapes"
    assert selection.provenance["selected"][0]["use_case_derived"] is derived


_MODEL_ONLY_SCHEMA = {
    "type": "object",
    "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
}


def _model_only(model: list[Any], **declared: Any) -> dict[str, Any]:
    return {"extensions": {"lora": {"mode": "model_only", "model": model, **declared}}}


def _sampled(model: list[Any], class_type: str = "KSampler") -> dict[str, Any]:
    return {
        "class_type": class_type,
        "inputs": {
            "model": model,
            "positive": ["pos", 0],
            "negative": ["neg", 0],
            "latent_image": ["latent", 0],
        },
    }


def _prompt_nodes() -> dict[str, Any]:
    return {
        "clip": {"class_type": "CLIPLoader", "inputs": {}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {}},
    }


def _two_stage_graph(stage2_model: list[Any]) -> dict[str, Any]:
    # A custom two-stage sampler whose stages are fed by different patches of one model.
    return {
        **_prompt_nodes(),
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "40": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0]}},
        "38": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["40", 0]}},
        "7": {"class_type": "UNETLoader", "inputs": {}},
        "9": {
            "class_type": "KreaTwoStageSampler",
            "inputs": {
                "stage1_model": ["40", 0],
                "stage2_model": stage2_model,
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["latent", 0],
            },
        },
    }


_TWO_STAGES = [["9", "stage1_model"], ["9", "stage2_model"]]


def test_model_only_loras_reach_both_stages_of_a_named_custom_sampler() -> None:
    graph = _two_stage_graph(["38", 0])

    for model in (["1", 0], ["40", 0]):
        dependencies = _model_only(model, sampler_inputs=_TWO_STAGES)
        validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, dependencies)
    from types import SimpleNamespace

    revision = SimpleNamespace(
        api_graph_json=graph,
        input_schema_json=_MODEL_ONLY_SCHEMA,
        dependencies_json=_model_only(["1", 0], sampler_inputs=_TWO_STAGES),
    )
    assert workflow_lora_extension(cast(Any, revision)) == {
        "mode": "model_only",
        "model": ["1", 0],
        "sampler_inputs": _TWO_STAGES,
    }
    # After the second stage's own patch is too late for the first stage.
    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(
            graph, _MODEL_ONLY_SCHEMA, _model_only(["38", 0], sampler_inputs=_TWO_STAGES)
        )

    transformed = transform_lora_graph(
        graph,
        {"mode": "model_only", "model": ["1", 0]},
        [{"comfy_name": "style.safetensors", "model_strength": 0.5, "clip_strength": 1.0}],
    )
    assert transformed["lma_lora_001"]["inputs"]["model"] == ["1", 0]
    assert transformed["40"]["inputs"]["model"] == ["lma_lora_001", 0]
    assert transformed["9"]["inputs"]["stage1_model"] == ["40", 0]
    assert transformed["38"]["inputs"]["model"] == ["40", 0]


def test_model_only_loras_refuse_a_named_stage_fed_by_another_model() -> None:
    graph = _two_stage_graph(["7", 0])

    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(
            graph, _MODEL_ONLY_SCHEMA, _model_only(["1", 0], sampler_inputs=_TWO_STAGES)
        )


def test_an_unnamed_custom_sampler_is_not_recognised_from_its_graph() -> None:
    graph = _two_stage_graph(["38", 0])

    assert detect_lora_extension(graph) is None
    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["1", 0]))


@pytest.mark.parametrize(
    "sampler_inputs",
    [[], [["9"]], [["9", ""]], [["9", "stage1_model"], ["9", "stage1_model"]], "9", [[9, "x"]]],
)
def test_a_malformed_sampler_input_list_is_refused(sampler_inputs: object) -> None:
    graph = _two_stage_graph(["38", 0])

    with pytest.raises(ValueError, match="extension is invalid"):
        validate_lora_workflow_contract(
            graph, _MODEL_ONLY_SCHEMA, _model_only(["1", 0], sampler_inputs=sampler_inputs)
        )


def test_a_named_sampler_input_must_exist_in_the_graph() -> None:
    graph = _two_stage_graph(["38", 0])

    with pytest.raises(ValueError, match="names a sampler input"):
        validate_lora_workflow_contract(
            graph,
            _MODEL_ONLY_SCHEMA,
            _model_only(["1", 0], sampler_inputs=[["9", "stage1_model"], ["9", "refiner"]]),
        )


def test_model_only_loras_insert_where_core_samplers_on_different_patches_meet() -> None:
    graph = {
        **_prompt_nodes(),
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0]}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0]}},
        "first": _sampled(["2", 0]),
        "second": _sampled(["3", 0], "KSamplerAdvanced"),
    }

    assert detect_lora_extension(graph) == {"mode": "model_only", "model": ["2", 0]}
    validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["1", 0]))
    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["3", 0]))


@pytest.mark.parametrize(
    "between",
    [
        # A switch can select the branch that does not carry the LoRAs.
        {"class_type": "ComfySwitchNode", "inputs": {"on_true": ["2", 0], "on_false": ["1", 0]}},
        # A merge with its own unpatched base would carry them only partly.
        {"class_type": "ModelMergeSimple", "inputs": {"model1": ["2", 0], "model2": ["1", 0]}},
    ],
)
def test_model_only_loras_refuse_an_insertion_point_behind_a_switch_or_merge(
    between: dict[str, Any],
) -> None:
    graph = {
        **_prompt_nodes(),
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0]}},
        "3": between,
        "4": _sampled(["3", 0]),
    }

    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["2", 0]))
    validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["3", 0]))
    assert detect_lora_extension(graph) == {"mode": "model_only", "model": ["3", 0]}


def test_model_only_loras_refuse_a_second_pass_tied_only_by_an_image() -> None:
    graph = {
        **_prompt_nodes(),
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0]}},
        "3": _sampled(["2", 0]),
        "vae": {"class_type": "VAELoader", "inputs": {}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["vae", 0]}},
        "patch": {"class_type": "ModelPatchLoader", "inputs": {}},
        "72": {
            "class_type": "QwenImageDiffsynthControlnet",
            "inputs": {"model": ["1", 0], "model_patch": ["patch", 0], "image": ["8", 0]},
        },
        "73": _sampled(["72", 0]),
    }

    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["2", 0]))
    # The control patch passes on the model it takes, so both passes meet at the loader.
    assert detect_lora_extension(graph) == {"mode": "model_only", "model": ["1", 0]}


@pytest.mark.parametrize(
    "between",
    [
        # A custom switch that can pick a model taken straight from the loader.
        {
            "class_type": "CustomModelSwitch",
            "inputs": {"model": ["2", 0], "model_alt": ["3", 0], "select": 2},
        },
        # A custom blend with a model from a loader the graph does not otherwise use.
        {
            "class_type": "CustomModelBlend",
            "inputs": {"model": ["2", 0], "model2": ["other", 0], "ratio": 0.5},
        },
        # A custom node that takes only a model, yet outputs one it loads by name.
        {
            "class_type": "CustomModelOverride",
            "inputs": {"model": ["2", 0], "unet_name": "other.safetensors"},
        },
    ],
)
def test_model_only_loras_refuse_a_node_that_may_pass_on_another_model(
    between: dict[str, Any],
) -> None:
    graph = {
        **_prompt_nodes(),
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "other": {"class_type": "SomeDiffusionModelLoader", "inputs": {}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0]}},
        "3": {"class_type": "ModelSamplingFlux", "inputs": {"model": ["1", 0]}},
        "between": between,
        "first": _sampled(["between", 0]),
        "second": _sampled(["2", 0]),
    }

    assert detect_lora_extension(graph) is None
    for model in (["1", 0], ["2", 0]):
        with pytest.raises(ValueError, match="every supported sampler"):
            validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(model))


def test_a_core_patch_with_a_linked_setting_still_passes_its_model_on() -> None:
    graph = {
        **_prompt_nodes(),
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "shift": {"class_type": "PrimitiveFloat", "inputs": {"value": 3.0}},
        "2": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["1", 0], "shift": ["shift", 0]},
        },
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0]}},
        "first": _sampled(["2", 0]),
        "second": _sampled(["3", 0]),
    }

    assert detect_lora_extension(graph) == {"mode": "model_only", "model": ["2", 0]}
    validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["1", 0]))


def test_a_model_chain_that_feeds_itself_offers_no_insertion_point() -> None:
    graph = {
        **_prompt_nodes(),
        "a": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["b", 0]}},
        "b": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["a", 0]}},
        "first": _sampled(["a", 0]),
        "second": _sampled(["b", 0]),
    }

    assert detect_lora_extension(graph) is None


def test_a_node_with_a_malformed_class_is_passed_over() -> None:
    graph = {
        **_prompt_nodes(),
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "odd": {"class_type": ["KSampler"], "inputs": {"model": ["1", 0]}},
        "first": _sampled(["1", 0]),
    }

    assert detect_lora_extension(graph) == {"mode": "model_only", "model": ["1", 0]}
    validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["1", 0]))


def test_model_only_loras_refuse_an_insertion_point_on_a_clip_output() -> None:
    graph = {
        "c": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "l": {"class_type": "LoraLoader", "inputs": {"model": ["c", 0], "clip": ["c", 1]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["l", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["l", 1]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {}},
        "k": _sampled(["l", 0]),
    }

    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["c", 1]))


def _guided_graph(guider: dict[str, Any]) -> dict[str, Any]:
    return {
        **_prompt_nodes(),
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "2": {"class_type": "UNETLoader", "inputs": {}},
        "11": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0]}},
        "guider": guider,
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"guider": ["guider", 0], "latent_image": ["latent", 0]},
        },
    }


def test_model_only_loras_reach_a_custom_advanced_sampler_through_its_guider() -> None:
    graph = _guided_graph(
        {
            "class_type": "CFGGuider",
            "inputs": {"model": ["11", 0], "positive": ["pos", 0], "negative": ["neg", 0]},
        }
    )

    validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["1", 0]))
    assert detect_lora_extension(graph) == {"mode": "model_only", "model": ["11", 0]}


def test_a_dual_model_guider_needs_the_stack_on_its_negative_model_too() -> None:
    separate = _guided_graph(
        {
            "class_type": "DualModelGuider",
            "inputs": {"model": ["11", 0], "model_negative": ["2", 0], "positive": ["pos", 0]},
        }
    )
    shared = _guided_graph(
        {
            "class_type": "DualModelGuider",
            "inputs": {"model": ["11", 0], "model_negative": ["1", 0], "positive": ["pos", 0]},
        }
    )

    assert detect_lora_extension(separate) is None
    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(separate, _MODEL_ONLY_SCHEMA, _model_only(["11", 0]))
    assert detect_lora_extension(shared) == {"mode": "model_only", "model": ["1", 0]}
    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(shared, _MODEL_ONLY_SCHEMA, _model_only(["11", 0]))


def test_an_unknown_guider_cannot_be_accounted_for() -> None:
    graph = _guided_graph({"class_type": "SomeCustomGuider", "inputs": {"model": ["11", 0]}})

    assert detect_lora_extension(graph) is None
    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["1", 0]))


def test_a_model_patch_that_takes_conditioning_keeps_its_insertion_point() -> None:
    # Video nodes such as talking-head patches take conditioning and patch the model;
    # they are not samplers, so the sampler after them still decides the point.
    graph = {
        **_prompt_nodes(),
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "11": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0]}},
        "20": {
            "class_type": "WanInfiniteTalkToVideo",
            "inputs": {"model": ["11", 0], "positive": ["pos", 0], "negative": ["neg", 0]},
        },
        "30": _sampled(["20", 0]),
    }

    assert detect_lora_extension(graph) == {"mode": "model_only", "model": ["20", 0]}
    validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["20", 0]))


def test_model_only_loras_still_refuse_a_graph_that_samples_nothing() -> None:
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "11": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0]}},
        "save": {"class_type": "ModelSave", "inputs": {"model": ["11", 0]}},
    }

    assert detect_lora_extension(graph) is None
    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["1", 0]))


def test_finding_the_insertion_point_stays_linear_in_the_graph() -> None:
    # One long model chain sampled at its far end first, then by many samplers on
    # patches along it: each step of the chain is walked once, not once per sampler.
    import time

    size = 40_000
    graph: dict[str, Any] = {**_prompt_nodes(), "m0": {"class_type": "UNETLoader", "inputs": {}}}
    for index in range(1, size):
        graph[f"m{index}"] = {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": [f"m{index - 1}", 0]},
        }
    graph["far"] = _sampled([f"m{size - 1}", 0])
    for index in range(0, size, 20):
        graph[f"p{index}"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": [f"m{index}", 0]},
        }
        graph[f"s{index}"] = _sampled([f"p{index}", 0])

    # Samplers on many outputs of one patch with many settings: only its first output
    # passes the model on, and telling so does not read the settings.
    wide: dict[str, Any] = {
        **_prompt_nodes(),
        "u": {"class_type": "UNETLoader", "inputs": {}},
        "x": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["u", 0], **{f"w{index}": 0.0 for index in range(30_000)}},
        },
    }
    for slot in range(3_000):
        wide[f"k{slot}"] = _sampled(["x", slot])

    started = time.perf_counter()
    assert detect_lora_extension(graph) == {"mode": "model_only", "model": ["m0", 0]}
    validate_lora_workflow_contract(graph, _MODEL_ONLY_SCHEMA, _model_only(["m0", 0]))
    assert detect_lora_extension(wide) is None
    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(wide, _MODEL_ONLY_SCHEMA, _model_only(["u", 0]))
    assert time.perf_counter() - started < 3
