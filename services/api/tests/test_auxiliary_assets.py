from __future__ import annotations

import pytest
from httpx2 import AsyncClient

from local_lm.auxiliary_assets import (
    LORA_GRAPH_TRANSFORM_VERSION,
    checkpoint_lora_extension,
    resolve_lora_stack,
    transform_lora_graph,
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
    assert LORA_GRAPH_TRANSFORM_VERSION == "lora-graph-v1"


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
