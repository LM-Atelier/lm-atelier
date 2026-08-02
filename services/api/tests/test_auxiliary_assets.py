from __future__ import annotations

import json

import pytest
from httpx2 import AsyncClient

from local_lm.auxiliary_assets import (
    LORA_GRAPH_TRANSFORM_VERSION,
    checkpoint_lora_extension,
    detect_lora_extension,
    resolve_lora_stack,
    select_automatic_lora_stack,
    transform_lora_graph,
    trigger_words_to_apply,
    validate_lora_workflow_contract,
)
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import (
    ModelAssetInstall,
    ModelInstall,
    WorkflowDefinition,
    WorkflowRevision,
)


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

    assert resolved.graph_sha256 == repeated.graph_sha256
    assert reversed_stack.graph_sha256 != resolved.graph_sha256
    assert [item["asset_id"] for item in resolved.provenance] == [first.id, second.id]
    assert transformed["lma_lora_001"]["class_type"] == "LoraLoader"
    assert transformed["lma_lora_002"]["inputs"]["model"] == ["lma_lora_001", 0]
    assert transformed["2"]["inputs"]["clip"] == ["lma_lora_002", 1]
    assert transformed["3"]["inputs"]["model"] == ["lma_lora_002", 0]
    assert LORA_GRAPH_TRANSFORM_VERSION == "lora-graph-v2"


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
    provenance = [
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
