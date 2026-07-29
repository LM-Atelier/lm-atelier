from __future__ import annotations

import json
import struct

import pytest

from local_lm.db import SessionLocal
from local_lm.model_manifests import (
    MAX_METADATA_BYTES,
    InspectedComponent,
    ModelManifestError,
    ModelManifestInspection,
    inspect_repository_metadata,
)
from local_lm.model_planner import persist_install_plan, resolve_install_plan


def _safetensors(tensor_names: list[str], metadata: dict[str, str] | None = None) -> bytes:
    header = {
        **{
            name: {"dtype": "F16", "shape": [1], "data_offsets": [index * 2, index * 2 + 2]}
            for index, name in enumerate(tensor_names)
        },
        "__metadata__": metadata or {},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return len(encoded).to_bytes(8, "little") + encoded


def _gguf(fields: dict[str, str], *, tensors: int = 12) -> bytes:
    payload = bytearray(b"GGUF")
    payload.extend(struct.pack("<IQQ", 3, tensors, len(fields)))
    for key, value in fields.items():
        encoded_key = key.encode()
        encoded_value = value.encode()
        payload.extend(struct.pack("<Q", len(encoded_key)))
        payload.extend(encoded_key)
        payload.extend(struct.pack("<I", 8))
        payload.extend(struct.pack("<Q", len(encoded_value)))
        payload.extend(encoded_value)
    return bytes(payload)


def _gguf_with_large_string_array(*, item_count: int, architecture: str) -> bytes:
    payload = bytearray(b"GGUF")
    payload.extend(struct.pack("<IQQ", 3, 12, 2))
    array_key = b"tokenizer.ggml.tokens"
    payload.extend(struct.pack("<Q", len(array_key)))
    payload.extend(array_key)
    payload.extend(struct.pack("<IIQ", 9, 8, item_count))
    payload.extend(struct.pack("<Q", 0) * item_count)
    architecture_key = b"general.architecture"
    encoded_architecture = architecture.encode()
    payload.extend(struct.pack("<Q", len(architecture_key)))
    payload.extend(architecture_key)
    payload.extend(struct.pack("<I", 8))
    payload.extend(struct.pack("<Q", len(encoded_architecture)))
    payload.extend(encoded_architecture)
    return bytes(payload)


def test_static_inspector_resolves_unknown_repository_by_config_and_headers() -> None:
    files = {
        "config.json": json.dumps(
            {
                "_class_name": "StableDiffusionXLPipeline",
                "architectures": ["UNet2DConditionModel"],
            }
        ).encode(),
        "weights/model.safetensors": _safetensors(
            [
                "model.diffusion_model.input_blocks.0.weight",
                "conditioner.embedders.1.model.text_projection",
                "first_stage_model.decoder.conv.weight",
            ]
        ),
    }

    inspection = inspect_repository_metadata(
        files,
        ["weights/model.safetensors"],
        role="image",
    )

    assert inspection.family == "stable-diffusion-xl"
    assert inspection.architecture == "StableDiffusionXLPipeline"
    assert inspection.components[0].kind == "checkpoint"
    assert inspection.components[0].target_folder == "checkpoints"


def test_static_inspector_distinguishes_lora_from_primary_checkpoint() -> None:
    inspection = inspect_repository_metadata(
        {
            "adapter.safetensors": _safetensors(
                ["lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight"],
                {"ss_network_module": "networks.lora"},
            )
        },
        ["adapter.safetensors"],
        role="image",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/unknown-adapter",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {
                "filename": "adapter.safetensors",
                "size": 1_024,
                "sha256": "b" * 64,
            }
        ],
        inspection=inspection,
        workflow_template_id="synthetic-template",
        workflow_template_sha256="c" * 64,
    )

    assert inspection.components[0].kind == "lora"
    assert plan.compatibility == "unsupported"
    assert plan.failure_code == "auxiliary_asset_not_primary"


def test_official_workflow_plan_accepts_a_required_lora_with_primary_weights() -> None:
    selected = ["model.safetensors", "lightning.safetensors"]
    inspection = inspect_repository_metadata(
        {
            "model.safetensors": _safetensors(["model.diffusion_model.input_blocks.0.weight"]),
            "lightning.safetensors": _safetensors(
                ["lora_unet_block.lora_down.weight"],
                {"ss_network_module": "networks.lora"},
            ),
        },
        selected,
        role="image",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/complete-edit-workflow",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {"filename": selected[0], "size": 2_048, "sha256": "b" * 64},
            {
                "filename": selected[1],
                "size": 1_024,
                "sha256": "c" * 64,
                "source_remote_id": "synthetic/lightning",
                "source_revision": "d" * 40,
                "source_filename": selected[1],
            },
        ],
        inspection=inspection,
        workflow_template_id="synthetic-template",
        workflow_template_sha256="e" * 64,
    )

    assert plan.compatibility == "supported"
    assert {artifact.kind for artifact in plan.artifacts} == {"diffusion_model", "lora"}


def test_official_workflow_plan_uses_unambiguous_declared_component_paths() -> None:
    selected = [
        "split/diffusion/model.safetensors",
        "encoders/text.safetensors",
        "vae/model.safetensors",
        "lightning.safetensors",
    ]
    inspection = ModelManifestInspection(
        architecture=None,
        family=None,
        components=tuple(
            InspectedComponent(
                path=path,
                kind="unknown_safetensors",
                target_folder="checkpoints",
            )
            for path in selected
        ),
        metadata_files=(),
    )
    plan = resolve_install_plan(
        remote_id="synthetic/complete-edit-workflow",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {"filename": path, "size": 1_024, "sha256": str(index) * 64}
            for index, path in enumerate(selected, start=1)
        ],
        inspection=inspection,
        workflow_template_id="synthetic-template",
        workflow_template_sha256="e" * 64,
        comfy_paths={
            "diffusion_models": "split/diffusion",
            "text_encoders": "encoders",
            "vae": "vae",
            "loras": ".",
        },
    )

    assert plan.compatibility == "supported"
    assert [(item.kind, item.target_folder) for item in plan.artifacts] == [
        ("diffusion_model", "diffusion_models"),
        ("text_encoder", "text_encoders"),
        ("vae", "vae"),
        ("lora", "loras"),
    ]


def test_chat_install_plan_binds_external_projector_provenance() -> None:
    model_path = "model-Q4_K_M.gguf"
    projector_path = "companions/author/model/mmproj-model-f16.gguf"
    inspection = inspect_repository_metadata(
        {
            model_path: _gguf({"general.architecture": "vision"}),
            projector_path: _gguf(
                {
                    "general.architecture": "clip",
                    "clip.projector_type": "mlp",
                }
            ),
        },
        [model_path, projector_path],
        role="chat",
    )
    plan = resolve_install_plan(
        remote_id="converter/model-gguf",
        revision="a" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[
            {
                "filename": model_path,
                "size": 10,
                "sha256": "b" * 64,
            },
            {
                "filename": projector_path,
                "size": 3,
                "sha256": "c" * 64,
                "source_remote_id": "author/model",
                "source_revision": "d" * 40,
                "source_filename": "mmproj-model-f16.gguf",
            },
        ],
        inspection=inspection,
    )

    projector = next(item for item in plan.artifacts if item.kind == "projector")
    assert projector.source_remote_id == "author/model"
    assert projector.source_revision == "d" * 40
    assert projector.source_path == "mmproj-model-f16.gguf"
    assert plan.compatibility == "supported"


def test_static_inspector_accepts_lora_only_as_a_typed_auxiliary_plan() -> None:
    inspection = inspect_repository_metadata(
        {
            "adapter.safetensors": _safetensors(
                ["lora_unet_block.lora_down.weight"],
                {
                    "ss_network_module": "networks.lora",
                    "ss_network_dim": "16",
                    "modelspec.trigger_phrase": "atelier ink, paper grain",
                },
            )
        },
        ["adapter.safetensors"],
        role="image",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/unknown-adapter",
        revision="a" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {
                "filename": "adapter.safetensors",
                "size": 1_024,
                "sha256": "b" * 64,
            }
        ],
        inspection=inspection,
        comfy_paths={"loras": "."},
        auxiliary_kind="lora",
    )

    assert plan.compatibility == "supported"
    assert plan.runtime_contract["auxiliary_kind"] == "lora"
    assert plan.artifacts[0].target_folder == "loras"
    assert inspection.components[0].metadata["network_type"] == "networks.lora"
    assert inspection.components[0].metadata["rank"] == 16
    assert inspection.components[0].metadata["trigger_words"] == [
        "atelier ink",
        "paper grain",
    ]


def test_media_plan_rejects_pickle_compatible_weight_formats() -> None:
    inspection = inspect_repository_metadata(
        {},
        ["pytorch_model.bin"],
        role="image",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/unsafe-media",
        revision="2" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {
                "filename": "pytorch_model.bin",
                "size": 1_024,
                "sha256": "3" * 64,
            }
        ],
        inspection=inspection,
        workflow_template_id="synthetic-template",
        workflow_template_sha256="4" * 64,
    )

    assert plan.compatibility == "unsupported"
    assert plan.failure_code == "unsafe_model_format"


def test_static_inspector_reads_gguf_architecture_without_filename_guessing() -> None:
    inspection = inspect_repository_metadata(
        {
            "weights.bin.gguf": _gguf(
                {
                    "general.architecture": "qwen3",
                    "general.type": "model",
                }
            )
        },
        ["weights.bin.gguf"],
        role="chat",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/future-chat-model",
        revision="d" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[
            {
                "filename": "weights.bin.gguf",
                "size": 2_048,
                "sha256": "e" * 64,
            }
        ],
        inspection=inspection,
    )

    assert inspection.architecture == "qwen3"
    assert inspection.family == "qwen"
    assert plan.compatibility == "supported"
    assert plan.activation_probe["kind"] == "chat_completion"


def test_static_inspector_skips_large_gguf_token_arrays_without_materializing_them() -> None:
    inspection = inspect_repository_metadata(
        {
            "qwen.gguf": _gguf_with_large_string_array(
                item_count=100_001,
                architecture="qwen35",
            )
        },
        ["qwen.gguf"],
        role="chat",
    )

    assert inspection.architecture == "qwen35"
    assert inspection.family == "qwen"


def test_modelopt_snapshot_produces_a_vllm_install_contract() -> None:
    selected = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "config.json",
        "hf_quant_config.json",
        "tokenizer.json",
    ]
    inspection = inspect_repository_metadata(
        {
            "config.json": json.dumps(
                {"architectures": ["Qwen3_5ForConditionalGeneration"]}
            ).encode(),
            "hf_quant_config.json": json.dumps({"quantization": {"quant_algo": "NVFP4"}}).encode(),
            "model-00001-of-00002.safetensors": _safetensors(
                ["model.layers.0.self_attn.q_proj.weight"]
            ),
            "model-00002-of-00002.safetensors": _safetensors(
                ["model.layers.1.self_attn.q_proj.weight"]
            ),
        },
        selected,
        role="chat",
    )
    plan = resolve_install_plan(
        remote_id="nvidia/Qwen3.6-27B-NVFP4",
        revision="a" * 40,
        role="chat",
        engine="vllm",
        selected_files=[
            {
                "filename": filename,
                "size": 1_024,
                "sha256": "b" * 64,
            }
            for filename in selected
        ],
        inspection=inspection,
    )

    assert plan.compatibility == "supported"
    assert plan.runtime_contract["engine"] == "vllm"
    assert plan.runtime_contract["quantization"] == "modelopt"
    assert plan.runtime_contract["model_layout"] == "transformers_snapshot"
    assert {artifact.kind for artifact in plan.artifacts} == {"weights", "metadata"}


def test_modelopt_snapshot_requires_quantization_metadata() -> None:
    inspection = inspect_repository_metadata(
        {
            "config.json": json.dumps({"model_type": "qwen3_5"}).encode(),
            "model.safetensors": _safetensors(["model.embed_tokens.weight"]),
        },
        ["model.safetensors", "config.json"],
        role="chat",
    )
    plan = resolve_install_plan(
        remote_id="synthetic/incomplete-modelopt",
        revision="a" * 40,
        role="chat",
        engine="vllm",
        selected_files=[
            {"filename": "model.safetensors", "size": 1_024, "sha256": "b" * 64},
            {"filename": "config.json", "size": 128, "sha256": "c" * 64},
        ],
        inspection=inspection,
    )

    assert plan.compatibility == "unsupported"
    assert plan.failure_code == "incomplete_modelopt_snapshot"


def test_static_inspector_rejects_oversized_and_unsafe_metadata() -> None:
    with pytest.raises(ModelManifestError, match="size limit"):
        inspect_repository_metadata(
            {"config.json": b"{" + b" " * MAX_METADATA_BYTES + b"}"},
            [],
            role="image",
        )
    with pytest.raises(ModelManifestError, match="unsafe"):
        inspect_repository_metadata(
            {"../config.json": b"{}"},
            [],
            role="image",
        )


async def test_install_plan_hash_is_stable_and_persistence_is_idempotent(client) -> None:  # type: ignore[no-untyped-def]
    inspection = inspect_repository_metadata(
        {"model.gguf": _gguf({"general.architecture": "llama"})},
        ["model.gguf"],
        role="chat",
    )
    resolved = resolve_install_plan(
        remote_id="synthetic/unknown-llama",
        revision="f" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[
            {
                "filename": "model.gguf",
                "size": 4_096,
                "sha256": "1" * 64,
            }
        ],
        inspection=inspection,
    )

    with SessionLocal() as session:
        first = persist_install_plan(session, resolved)
        session.commit()
        first_id = first.id
    with SessionLocal() as session:
        second = persist_install_plan(session, resolved)
        session.commit()
        assert second.id == first_id
        assert second.plan_hash == resolved.plan_hash


async def test_supported_plan_can_be_retried_after_a_terminal_attempt(client) -> None:  # type: ignore[no-untyped-def]
    inspection = inspect_repository_metadata(
        {"model.gguf": _gguf({"general.architecture": "future"})},
        ["model.gguf"],
        role="chat",
    )
    resolved = resolve_install_plan(
        remote_id="synthetic/retryable",
        revision="9" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[
            {
                "filename": "model.gguf",
                "size": 4_096,
                "sha256": "8" * 64,
            }
        ],
        inspection=inspection,
    )
    with SessionLocal() as session:
        plan = persist_install_plan(session, resolved)
        plan.status = "failed"
        plan.failure_code = "activation_runtime_failed"
        plan.failure_reason = "Synthetic failure"
        session.commit()
        plan_id = plan.id

    with SessionLocal() as session:
        retry = persist_install_plan(session, resolved)
        session.commit()
        assert retry.id == plan_id
        assert retry.status == "planned"
        assert retry.failure_code is None
        assert retry.failure_reason is None
