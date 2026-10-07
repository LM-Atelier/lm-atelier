"""Construct an untrained CPU checkpoint for real runtime integration tests.

Run only with the isolated ComfyUI Python executable. No model downloads or
user assets are used. Random weights exercise mechanics, not image quality.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def construct_unet(torch: Any) -> Any:
    model = importlib.import_module("comfy.ldm.modules.diffusionmodules.openaimodel")
    ops = importlib.import_module("comfy.ops")
    return model.UNetModel(
        image_size=32,
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=1,
        channel_mult=(1,),
        num_heads=8,
        use_spatial_transformer=True,
        transformer_depth=[1],
        transformer_depth_output=[1, 1],
        transformer_depth_middle=1,
        context_dim=768,
        device=torch.device("meta"),
        dtype=torch.float32,
        operations=ops.disable_weight_init,
    )


def initialize(torch: Any, module: Any) -> None:
    module.to_empty(device="cpu")
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.ndim >= 2:
                fan_in = parameter.shape[1] * math.prod(parameter.shape[2:])
                parameter.normal_(0, math.sqrt(0.5 / fan_in))
            elif name.endswith("weight") or name == "vae_scale":
                parameter.fill_(1)
            else:
                parameter.zero_()


def prepare_text_encoder(torch: Any, destination: Path, nodes: Any) -> tuple[Any, int]:
    """Load a full seeded text encoder through the runtime's real CLIPLoader."""
    clip_module = importlib.import_module("comfy.clip_model")
    assert clip_module.__file__ is not None
    config = json.loads(Path(clip_module.__file__).with_name("sd1_clip_config.json").read_text())
    ops = importlib.import_module("comfy.ops")
    text = clip_module.CLIPTextModel(
        config, torch.float32, torch.device("meta"), ops.disable_weight_init
    )
    initialize(torch, text)
    state = {key: value.half() for key, value in text.state_dict().items()}
    target = destination / "neutral-text.safetensors"
    importlib.import_module("safetensors.torch").save_file(state, str(target))
    folders = importlib.import_module("folder_paths")
    folders.add_model_folder_path("text_encoders", str(destination), is_default=True)
    loaded = nodes.CLIPLoader().load_clip(target.name, "stable_diffusion", "cpu")[0]
    actual = loaded.cond_stage_model.clip_l.transformer.state_dict()
    assert actual.keys() == state.keys()
    assert all(torch.equal(value.cpu().half(), state[key]) for key, value in actual.items())
    return nodes.CLIPTextEncode().encode(loaded, "Neutral geometric shapes")[0], len(state)


def exercise_crop_transport(
    torch: Any,
    nodes: Any,
    model: Any,
    vae: Any,
    conditioning: Any,
    fixture: Path,
    destination: Path,
) -> dict[str, Any]:

    sd_module = importlib.import_module("comfy.sd")
    autoencoder_module = importlib.import_module("comfy.ldm.models.autoencoder")
    with torch.device("meta"):
        shape_state = {
            key: torch.empty_like(value, device="meta")
            for key, value in vae.first_stage_model.state_dict().items()
        }
        shape_vae = sd_module.VAE(sd=shape_state, device=torch.device("cpu"), dtype=torch.float32)
        four_model = autoencoder_module.AutoencoderKL(
            ddconfig={
                "double_z": True,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 2, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
            },
            embed_dim=4,
        )
        four_vae = sd_module.VAE(
            sd=four_model.state_dict(), device=torch.device("cpu"), dtype=torch.float32
        )
    for observed, multiple in ((shape_vae, 8), (four_vae, 4)):
        assert observed.spacial_compression_encode() == multiple
        assert observed.spacial_compression_decode() == multiple
        assert observed.latent_dim == 2 and observed.output_channels == 3
        assert observed.crop_input is True
        assert all(value.is_meta for value in observed.first_stage_model.state_dict().values())
    assert shape_vae.first_stage_model.state_dict().keys() == shape_state.keys()
    assert four_vae.first_stage_model.state_dict().keys() == four_model.state_dict().keys()

    image_module = importlib.import_module("PIL.Image")
    numpy = importlib.import_module("numpy")
    plan = json.loads((fixture / "plan.json").read_text())
    multiples = plan["encode_multiple"]
    actual_multiple = vae.spacial_compression_encode()
    assert multiples == {"width": actual_multiple, "height": actual_multiple}

    def pixels(name: str) -> Any:
        with image_module.open(fixture / name) as image:
            return torch.from_numpy(numpy.array(image, dtype=numpy.float32) / 255).unsqueeze(0)

    cropped = pixels("cropped.png")
    padded = pixels("transport.png")
    assert tuple(cropped.shape) == (1, 36, 64, 3)
    assert tuple(padded.shape) == (1, 40, 64, 3)
    assert torch.equal(padded[:, :36, :64], cropped)
    unprotected = vae.vae_encode_crop_pixels(cropped)
    assert tuple(unprotected.shape) == (1, 32, 64, 3)
    aligned = vae.vae_encode_crop_pixels(padded)
    assert torch.equal(aligned, padded)
    encoded = nodes.VAEEncode().encode(vae, padded)[0]
    assert tuple(encoded["samples"].shape) == (1, 4, 5, 8)
    sampled = nodes.KSampler().sample(
        model, 809, 2, 1.0, "euler", "normal", conditioning, conditioning, encoded, denoise=0.7
    )[0]
    decoded = nodes.VAEDecode().decode(vae, sampled)[0]
    assert tuple(decoded.shape) == (1, 40, 64, 3)
    crop_node = importlib.import_module("comfy_extras.nodes_images").ImageCropV2
    result = crop_node.execute(decoded, plan["output_crop"])[0]
    assert tuple(result.shape) == (1, 36, 64, 3)
    assert torch.equal(result, decoded[:, :36, :64])
    assert torch.isfinite(result).all()
    saved = nodes.SaveImage().save_images(result, filename_prefix="neutral-crop")
    record = saved["ui"]["images"][0]
    with image_module.open(destination / record["subfolder"] / record["filename"]) as image:
        assert image.size == (64, 36)
        assert image.tobytes() == bytes(
            (result.clamp(0, 1) * 255).to(torch.uint8).flatten().tolist()
        )
        assert image.tobytes() != bytes((cropped * 255).to(torch.uint8).flatten().tolist())
    return {
        "unprotected_size": [64, 32],
        "transport_size": [64, 40],
        "output_size": [64, 36],
        "crop_pixels_reach_encoder": True,
        "only_alignment_padding_removed": True,
        "generated_edit": True,
        "vae_encode_multiple": actual_multiple,
        "shape_only_factors": [8, 4],
        "shape_only_weight_bytes": 0,
    }


def exercise_checkpoint(
    torch: Any,
    unet: Any,
    destination: Path,
    text_encoder: bool = False,
    crop_transport: Path | None = None,
) -> dict[str, Any]:
    """Run core nodes with optional real text encoding; no server claim."""
    torch.manual_seed(731)
    initialize(torch, unet)
    with torch.device("meta"):
        vae = importlib.import_module("comfy.taesd.taesd").TAESD(latent_channels=4)
    initialize(torch, vae)
    # Keep neutral decoder values away from uint8 clipping.
    with torch.no_grad():
        vae.taesd_decoder[-1].bias.fill_(0.5)
    state = {
        **{"model.diffusion_model." + k: v.half() for k, v in unet.state_dict().items()},
        **{"first_stage_model." + k: v.half() for k, v in vae.state_dict().items()},
    }
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = destination / "neutral-untrained.safetensors"
    importlib.import_module("safetensors.torch").save_file(state, str(checkpoint))
    folders = importlib.import_module("folder_paths")
    folders.add_model_folder_path("checkpoints", str(destination), is_default=True)
    nodes = importlib.import_module("nodes")
    model, clip, loaded_vae = nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint.name)
    assert clip is None  # This bounded fixture intentionally has no text encoder.
    assert loaded_vae.spacial_compression_encode() == 8
    assert loaded_vae.spacial_compression_decode() == 8
    # Require every actual UNet/VAE tensor to load exactly; do not allow a
    # permissive loader to conceal missing or differently inferred weights.
    for prefix, actual in (
        ("model.diffusion_model.", model.model.diffusion_model.state_dict()),
        ("first_stage_model.", loaded_vae.first_stage_model.state_dict()),
    ):
        expected = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        assert actual.keys() == expected.keys()
        assert all(torch.equal(v.cpu().half(), expected[k]) for k, v in actual.items())
    pad_type = nodes.ImagePadForOutpaint
    composite_type = importlib.import_module("comfy_extras.nodes_mask").ImageCompositeMasked
    source = (torch.arange(16 * 24 * 3).remainder(256).float() / 255).reshape(1, 24, 16, 3)
    padded, mask = pad_type().expand_image(source, 8, 8, 8, 8, 0)
    assert tuple(padded.shape) == (1, 40, 32, 3)
    latent = nodes.VAEEncodeForInpaint().encode(loaded_vae, padded, mask, grow_mask_by=0)[0]
    assert tuple(latent["samples"].shape) == (1, 4, 5, 4)
    conditioning = [[torch.zeros((1, 77, 768)), {}]]
    text_tensors = 0
    if text_encoder:
        conditioning, text_tensors = prepare_text_encoder(torch, destination, nodes)
    folders.set_output_directory(str(destination))
    outputs = []
    for seed in (731, 732):
        sampled = nodes.KSampler().sample(
            model, seed, 2, 1.0, "euler", "normal", conditioning, conditioning, latent, denoise=1
        )[0]
        assert torch.isfinite(sampled["samples"]).all()
        decoded = nodes.VAEDecode().decode(loaded_vae, sampled)[0]
        assert tuple(decoded.shape) == (1, 40, 32, 3)
        assert torch.isfinite(decoded).all()
        result = composite_type().composite(padded, decoded, 0, 0, True, mask)[0]
        raster = (result.clamp(0, 1) * 255).to(torch.uint8)
        expected_source = (source * 255).to(torch.uint8)
        assert torch.equal(raster[:, 8:32, 8:24], expected_source)
        added = raster[mask.bool()]
        assert added.unique(dim=0).shape[0] > 1
        assert not torch.all(added == 127)
        inverted = composite_type().composite(padded, decoded, 0, 0, True, 1 - mask)[0]
        inverted_raster = (inverted.clamp(0, 1) * 255).to(torch.uint8)
        assert not torch.equal(inverted_raster[:, 8:32, 8:24], expected_source)
        assert torch.all(inverted_raster[mask.bool()] == 127)
        saved = nodes.SaveImage().save_images(result, filename_prefix=f"neutral-{seed}")
        record = saved["ui"]["images"][0]
        image_module = importlib.import_module("PIL.Image")
        with image_module.open(destination / record["subfolder"] / record["filename"]) as image:
            assert image.size == (32, 40) and image.mode == "RGB"
            assert image.crop((8, 8, 24, 32)).tobytes() == bytes(expected_source.flatten().tolist())
            outputs.append(image.tobytes())
    assert outputs[0] != outputs[1]
    crop_report = (
        exercise_crop_transport(
            torch, nodes, model, loaded_vae, conditioning, crop_transport, destination
        )
        if crop_transport is not None
        else None
    )
    return {
        "crop_transport": crop_report,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "loaded_tensors": len(state),
        "canvas": [32, 40],
        "source": [16, 24],
        "sampled_seeds": [731, 732],
        "source_equal": True,
        "extension_changes_with_seed": True,
        "conditioning": "clip" if text_encoder else "synthetic",
        "text_encoder": text_encoder,
        "text_tensors": text_tensors,
        "server_submission": False,
        "inverted_mask_control": True,
        "saved_pngs": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text-encoder", action="store_true")
    parser.add_argument("--crop-transport", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime))
    sys.argv = ["source-fit-fixture", "--cpu"]
    importlib.import_module("comfy.options").enable_args_parsing()
    torch = importlib.import_module("torch")
    torch.set_num_threads(2)
    detection = importlib.import_module("comfy.model_detection")
    unet = construct_unet(torch)
    state = {"model.diffusion_model." + key: value for key, value in unet.state_dict().items()}
    config = detection.model_config_from_unet(state, "model.diffusion_model.")
    assert config is not None and type(config).__name__ == "SD15"
    assert config.unet_config["channel_mult"] == [1]
    assert config.unet_config["num_res_blocks"] == [1]
    assert all(tensor.is_meta for tensor in state.values())
    report = {
        "model_class": type(config).__name__,
        "parameters": sum(tensor.numel() for tensor in state.values()),
        "configuration": config.unet_config,
        "allocated_weight_bytes": 0,
    }
    if args.output is not None:
        with torch.inference_mode():
            report["execution"] = exercise_checkpoint(
                torch, unet, args.output, args.text_encoder, args.crop_transport
            )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
