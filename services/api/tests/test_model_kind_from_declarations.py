"""A weight file's kind comes from its tensors and its declarations, not its prose.

An official video bundle embeds its license in every file's metadata, and the
license grants rights over "LoRA adapters". Matching "lora" anywhere in the
metadata sorted the bundle's diffusion model, VAEs and latent upscaler as LoRAs.
Its Gemma text encoders carry a vision tower, and the tower alone sorted them as
CLIP vision models. The download finished and was then refused, file by file,
for not being what the install check had planned.
"""

from __future__ import annotations

import json
import struct

import local_lm.model_manifests as manifests_module
from local_lm.downloads import DownloadManager
from local_lm.model_manifests import inspect_repository_metadata
from local_lm.models import InstallPlan

LICENSE = (
    "Community License Agreement. You may create derivatives of the model, "
    "including fine-tuned weights, LoRA adapters and distilled versions."
)
DIFFUSION = [
    "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.proj_out.weight",
]
VAE = ["vae.decoder.conv_in.weight", "vae.encoder.conv_out.weight"]
UPSCALER = ["initial_conv.weight", "upsampler.0.weight"]
GEMMA = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.norm.weight",
    "vision_model.patch_dense.weight",
    "multi_modal_projector.embedding_projection.weight",
]


def _header(tensor_names: list[str], metadata: dict[str, str]) -> bytes:
    header: dict[str, object] = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [index * 4, index * 4 + 4]}
        for index, name in enumerate(tensor_names)
    }
    header["__metadata__"] = metadata
    encoded = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def test_a_license_that_mentions_lora_adapters_does_not_make_a_model_a_lora() -> None:
    declared = {"license": LICENSE, "model_version": "2.5.0"}

    assert manifests_module._safetensors_kind(DIFFUSION, declared) == "diffusion_model"
    assert manifests_module._safetensors_kind(VAE, declared) == "vae"
    assert manifests_module._safetensors_kind(UPSCALER, declared) == "unknown_safetensors"


def test_a_file_that_declares_itself_a_lora_is_still_one() -> None:
    # One plain tensor name, so only the declaration can decide.
    for declared in (
        {"ss_network_module": "networks.lora"},
        {"modelspec.architecture": "flux-1-dev/lora"},
        {"network_type": "LoRA"},
        {"lora_rank": "16"},
    ):
        assert manifests_module._safetensors_kind(["weight"], declared) == "lora", declared


def test_a_language_model_with_a_vision_tower_is_a_text_encoder() -> None:
    assert manifests_module._safetensors_kind(GEMMA, {}) == "text_encoder"
    # A vision tower on its own is still a CLIP vision model.
    tower = ["vision_model.embeddings.patch_embedding.weight", "vision_model.post_layernorm.weight"]
    assert manifests_module._safetensors_kind(tower, {}) == "clip_vision"
    # A music diffusion model names its own submodules `embed_tokens`; it stays
    # unclassified, so the folder it is installed into decides.
    music = [
        "decoder.layers.0.self_attn.q_proj.weight",
        "encoder.lyric_encoder.embed_tokens.weight",
        "detokenizer.embed_tokens.weight",
    ]
    assert manifests_module._safetensors_kind(music, {}) == "unknown_safetensors"


def test_a_downloaded_official_bundle_with_its_license_embedded_keeps_its_planned_kinds() -> None:
    planned = {
        "diffusion_models/motion.safetensors": ("diffusion_model", "diffusion_models", DIFFUSION),
        "latent_upscale_models/motion-upscaler.safetensors": (
            "latent_upscaler",
            "latent_upscale_models",
            UPSCALER,
        ),
        "vae/motion-vae.safetensors": ("vae", "vae", VAE),
        "text_encoders/gemma-with-projection.safetensors": ("text_encoder", "text_encoders", GEMMA),
    }
    inspection = inspect_repository_metadata(
        {
            path: _header(names, {"license": LICENSE})
            for path, (_kind, _folder, names) in planned.items()
        },
        list(planned),
        role="video",
        component_folders={path: folder for path, (_kind, folder, _names) in planned.items()},
    )
    assert {item.path: (item.kind, item.target_folder) for item in inspection.components} == {
        path: (kind, folder) for path, (kind, folder, _names) in planned.items()
    }

    plan = InstallPlan(
        family=None,
        role="video",
        artifacts_json=[
            {"path": path, "kind": kind, "target_folder": folder, "required": True}
            for path, (kind, folder, _names) in planned.items()
        ],
        runtime_contract_json={
            "auxiliary_kind": None,
            "workflow_asset_kind": None,
            "workflow_template_id": "video_motion_t2v",
        },
    )
    DownloadManager._validate_staged_plan(
        plan, inspection, {path: f"{index:064x}" for index, path in enumerate(planned)}
    )
