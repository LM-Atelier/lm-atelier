from __future__ import annotations

import json
import struct

import pytest

from local_lm.db import SessionLocal
from local_lm.model_manifests import (
    MAX_METADATA_BYTES,
    ModelManifestError,
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
