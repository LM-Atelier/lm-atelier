from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_lm.comfy_templates import (
    ComfyModelDependency,
    ComfyTemplate,
    ComfyTemplateRegistry,
    CompiledComfyTemplate,
    _compile_ui_graph,
    _enable_declared_four_step_edit_acceleration,
    derive_image_to_image,
)
from local_lm.config import Settings
from local_lm.schemas import CatalogFileSource


def _registry(tmp_path: Path) -> ComfyTemplateRegistry:
    comfy = tmp_path / "comfy"
    templates = (
        comfy / ".venv" / "Lib" / "site-packages" / "comfyui_workflow_templates_json" / "templates"
    )
    templates.mkdir(parents=True)
    template = {
        "nodes": [
            {
                "id": 57,
                "type": "subgraph-z",
                "inputs": [
                    {
                        "name": "text",
                        "type": "STRING",
                        "widget": {"name": "text"},
                        "link": None,
                    }
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [62]}],
                "properties": {"cnr_id": "comfy-core"},
                "widgets_values": [],
            },
            {
                "id": 9,
                "type": "SaveImage",
                "inputs": [{"name": "images", "type": "IMAGE", "link": 62}],
                "outputs": [],
                "properties": {"cnr_id": "comfy-core"},
                "widgets_values": ["z-image-turbo"],
            },
        ],
        "links": [[62, 57, 0, 9, 0, "IMAGE"]],
        "definitions": {
            "subgraphs": [
                {
                    "id": "subgraph-z",
                    "inputs": [
                        {"name": "text", "type": "STRING", "linkIds": [34]},
                        {"name": "width", "type": "INT", "linkIds": [35]},
                    ],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "linkIds": [16]}],
                    "nodes": [
                        {
                            "id": 1,
                            "type": "ImageFromText",
                            "inputs": [
                                {
                                    "name": "text",
                                    "type": "STRING",
                                    "widget": {"name": "text"},
                                    "link": 34,
                                },
                                {
                                    "name": "width",
                                    "type": "INT",
                                    "widget": {"name": "width"},
                                    "link": 35,
                                },
                            ],
                            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16]}],
                            "properties": {
                                "cnr_id": "comfy-core",
                                "models": [
                                    {
                                        "name": "z_image_turbo_int8.safetensors",
                                        "url": (
                                            "https://huggingface.co/Comfy-Org/z_image_turbo/"
                                            "resolve/main/split_files/diffusion_models/"
                                            "z_image_turbo_int8.safetensors"
                                        ),
                                        "directory": "diffusion_models",
                                    }
                                ],
                            },
                            "widgets_values": ["example prompt", 1024],
                        }
                    ],
                    "links": [
                        {
                            "id": 34,
                            "origin_id": -10,
                            "origin_slot": 0,
                            "target_id": 1,
                            "target_slot": 0,
                            "type": "STRING",
                        },
                        {
                            "id": 35,
                            "origin_id": -10,
                            "origin_slot": 1,
                            "target_id": 1,
                            "target_slot": 1,
                            "type": "INT",
                        },
                        {
                            "id": 16,
                            "origin_id": 1,
                            "origin_slot": 0,
                            "target_id": -20,
                            "target_slot": 0,
                            "type": "IMAGE",
                        },
                    ],
                }
            ]
        },
    }
    (templates / "image_z_image_turbo_int8.json").write_text(
        json.dumps(template),
        encoding="utf-8",
    )
    (templates / "sdxlturbo_example.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": 1,
                        "type": "CheckpointLoaderSimple",
                        "inputs": [],
                        "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
                        "properties": {
                            "cnr_id": "comfy-core",
                            "models": [
                                {
                                    "name": "sd_xl_turbo_1.0_fp16.safetensors",
                                    "url": (
                                        "https://huggingface.co/stabilityai/sdxl-turbo/"
                                        "resolve/main/sd_xl_turbo_1.0_fp16.safetensors"
                                    ),
                                    "directory": "checkpoints",
                                }
                            ],
                        },
                        "widgets_values": ["sd_xl_turbo_1.0_fp16.safetensors"],
                    },
                    {
                        "id": 2,
                        "type": "SaveImage",
                        "inputs": [],
                        "outputs": [],
                        "properties": {"cnr_id": "comfy-core"},
                        "widgets_values": ["sdxl-turbo"],
                    },
                    {
                        "id": 3,
                        "type": "KSamplerSelect",
                        "inputs": [],
                        "outputs": [{"name": "SAMPLER", "type": "SAMPLER", "links": []}],
                        "properties": {"cnr_id": "comfy-core"},
                        "widgets_values": ["euler"],
                    },
                ],
                "links": [],
            }
        ),
        encoding="utf-8",
    )
    (templates / "index.en.json").write_text("[]", encoding="utf-8")
    settings = Settings(data_dir=tmp_path / "data", comfy_directory=comfy)
    settings.prepare()
    return ComfyTemplateRegistry(settings)


def test_registry_requires_the_exact_backend_declared_repository(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    assert registry.matches("Tongyi-MAI/Z-Image-Turbo", "image") == []
    matches = registry.matches("Comfy-Org/z_image_turbo", "image")

    assert [item.id for item in matches] == ["image_z_image_turbo_int8"]
    assert matches[0].remote_id == "Comfy-Org/z_image_turbo"
    assert matches[0].selected_files == [
        "split_files/diffusion_models/z_image_turbo_int8.safetensors"
    ]
    assert matches[0].comfy_paths == {"diffusion_models": "split_files/diffusion_models"}


def test_registry_prefers_exact_repository_and_rejects_generic_turbo_overlap(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    matches = registry.matches("stabilityai/sdxl-turbo", "image")

    assert [item.id for item in matches] == ["sdxlturbo_example"]
    assert matches[0].remote_id == "stabilityai/sdxl-turbo"
    assert registry.matches("owner/unrelated-turbo", "image") == []


def test_registry_accepts_and_revalidates_a_pinned_multirepository_bundle(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    templates = (
        registry.settings.comfy_directory
        / ".venv"
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates"
    )
    dependencies = [
        (
            "diffusion_models",
            "model.safetensors",
            "https://huggingface.co/owner/primary/resolve/main/weights/model.safetensors",
        ),
        (
            "text_encoders",
            "encoder.safetensors",
            "https://huggingface.co/owner/text/resolve/main/encoders/encoder.safetensors",
        ),
        (
            "vae",
            "vae.safetensors",
            "https://huggingface.co/owner/vae/resolve/main/vae.safetensors",
        ),
    ]
    nodes = [
        {
            "id": index,
            "type": "ModelLoader",
            "inputs": [],
            "outputs": [],
            "properties": {
                "cnr_id": "comfy-core",
                "models": [
                    {
                        "directory": directory,
                        "name": name,
                        "url": url,
                    }
                ],
            },
            "widgets_values": [name],
        }
        for index, (directory, name, url) in enumerate(dependencies, start=1)
    ]
    nodes.append(
        {
            "id": 10,
            "type": "SaveImage",
            "inputs": [],
            "outputs": [],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": ["multi-repo"],
        }
    )
    (templates / "image_multi_repo_edit.json").write_text(
        json.dumps({"nodes": nodes, "links": []}),
        encoding="utf-8",
    )
    (templates / "index.json").write_text(
        json.dumps(
            [
                {
                    "name": "image_multi_repo_edit",
                    "title": "General image editing",
                    "date": "2026-07-10",
                }
            ]
        ),
        encoding="utf-8",
    )
    ambiguous_nodes = [
        nodes[0],
        {
            "id": 20,
            "type": "ModelLoader",
            "inputs": [],
            "outputs": [],
            "properties": {
                "cnr_id": "comfy-core",
                "models": [
                    {
                        "directory": "vae",
                        "name": "vae.safetensors",
                        "url": (
                            "https://huggingface.co/owner/vae/resolve/main/weights/vae.safetensors"
                        ),
                    }
                ],
            },
            "widgets_values": ["vae.safetensors"],
        },
        nodes[-1],
    ]
    (templates / "image_ambiguous_component_paths.json").write_text(
        json.dumps({"nodes": ambiguous_nodes, "links": []}),
        encoding="utf-8",
    )

    matches = registry.matches("owner/primary", "image")

    assert "image_ambiguous_component_paths" not in {item.id for item in matches}
    assert matches[0].id == "image_multi_repo_edit"
    assert matches[0].published_date == "2026-07-10"
    assert matches[0].selected_files == [
        "weights/model.safetensors",
        "encoders/encoder.safetensors",
        "vae.safetensors",
    ]
    assert matches[0].comfy_paths == {
        "diffusion_models": "weights",
        "text_encoders": "encoders",
        "vae": ".",
    }
    sources = {
        "encoders/encoder.safetensors": CatalogFileSource(
            remote_id="owner/text",
            revision="b" * 40,
            filename="encoders/encoder.safetensors",
        ),
        "vae.safetensors": CatalogFileSource(
            remote_id="owner/vae",
            revision="c" * 40,
            filename="vae.safetensors",
        ),
    }
    validated = registry.validate_download(
        matches[0].id,
        "image",
        "owner/primary",
        matches[0].selected_files,
        matches[0].comfy_paths,
        revision="a" * 40,
        file_sources=sources,
    )
    assert validated.id == matches[0].id

    sources["vae.safetensors"] = sources["vae.safetensors"].model_copy(
        update={"remote_id": "owner/wrong"}
    )
    with pytest.raises(ValueError, match="source binding changed"):
        registry.validate_download(
            matches[0].id,
            "image",
            "owner/primary",
            matches[0].selected_files,
            matches[0].comfy_paths,
            revision="a" * 40,
            file_sources=sources,
        )


def test_native_edit_loaders_bind_ordered_runtime_images(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    templates = (
        registry.settings.comfy_directory
        / ".venv"
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates"
    )
    revision = "a" * 40
    template = {
        "nodes": [
            {
                "id": 1,
                "type": "LoadImage",
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
                "properties": {"cnr_id": "comfy-core"},
                "widgets_values": ["sample-a.png", "image"],
            },
            {
                "id": 2,
                "type": "LoadImage",
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "properties": {"cnr_id": "comfy-core"},
                "widgets_values": ["sample-b.png", "image"],
            },
            {
                "id": 3,
                "type": "CheckpointLoaderSimple",
                "inputs": [],
                "outputs": [],
                "properties": {
                    "cnr_id": "comfy-core",
                    "models": [
                        {
                            "directory": "checkpoints",
                            "name": "model.safetensors",
                            "url": (
                                "https://huggingface.co/owner/edit/resolve/"
                                f"{revision}/model.safetensors"
                            ),
                        }
                    ],
                },
                "widgets_values": ["model.safetensors"],
            },
            {
                "id": 4,
                "type": "SaveImage",
                "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
                "outputs": [],
                "properties": {"cnr_id": "comfy-core"},
                "widgets_values": ["native-edit"],
            },
        ],
        "links": [[1, 1, 0, 4, 0, "IMAGE"]],
    }
    (templates / "image_native_edit.json").write_text(
        json.dumps(template),
        encoding="utf-8",
    )
    object_info = {
        "LoadImage": {
            "input": {
                "required": {
                    "image": [["available.png"], {"image_upload": True}],
                }
            },
            "input_order": {"required": ["image"]},
        },
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [["model.safetensors"]],
                }
            },
            "input_order": {"required": ["ckpt_name"]},
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output_node": True,
        },
    }

    compiled = registry.compile(
        "image_native_edit",
        "image",
        object_info,
        remote_id="owner/edit",
        revision=revision,
        selected_files=["model.safetensors"],
        comfy_paths={"checkpoints": "."},
    )

    assert compiled.api_graph["1"]["inputs"]["image"] == "${input_image_0}"
    assert compiled.api_graph["2"]["inputs"]["image"] == "${input_image_1}"
    assert compiled.api_graph["3"]["inputs"]["ckpt_name"] == "model.safetensors"
    assert compiled.input_schema["properties"]["input_image_0"] == {"type": "string"}
    assert compiled.input_schema["properties"]["input_image_1"] == {"type": "string"}


def test_native_image_conditioning_keeps_authored_denoise_constant() -> None:
    ui_graph = {
        "nodes": [
            {
                "id": 1,
                "type": "LoadImage",
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
                "widgets_values": ["source.png"],
            },
            {
                "id": 2,
                "type": "NativeEditConditioning",
                "inputs": [
                    {"name": "image", "type": "IMAGE", "link": 1},
                    {"name": "text", "type": "STRING", "widget": {"name": "text"}},
                ],
                "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [2]}],
                "widgets_values": ["edit prompt"],
            },
            {
                "id": 3,
                "type": "KSampler",
                "inputs": [
                    {"name": "positive", "type": "CONDITIONING", "link": 2},
                    {
                        "name": "denoise",
                        "type": "FLOAT",
                        "widget": {"name": "denoise"},
                    },
                ],
                "outputs": [],
                "widgets_values": [1.0],
            },
        ],
        "links": [
            [1, 1, 0, 2, 0, "IMAGE"],
            [2, 2, 0, 3, 0, "CONDITIONING"],
        ],
    }
    object_info = {
        "LoadImage": {
            "input": {"required": {"image": [["source.png"], {"image_upload": True}]}},
            "input_order": {"required": ["image"]},
        },
        "NativeEditConditioning": {
            "input": {
                "required": {
                    "image": ["IMAGE"],
                    "text": ["STRING", {"default": ""}],
                }
            },
            "input_order": {"required": ["image", "text"]},
        },
        "KSampler": {
            "input": {
                "required": {
                    "positive": ["CONDITIONING"],
                    "denoise": ["FLOAT", {"default": 1.0}],
                }
            },
            "input_order": {"required": ["positive", "denoise"]},
        },
    }

    graph, schema = _compile_ui_graph(
        ui_graph,
        object_info,
        operation="image_to_image",
    )

    assert graph["1"]["inputs"]["image"] == "${input_image}"
    assert graph["2"]["inputs"]["text"] == "${prompt}"
    assert graph["3"]["inputs"]["denoise"] == 1.0
    assert schema["properties"]["denoise"] == {"readOnly": True}


def test_latent_only_image_edit_still_exposes_denoise() -> None:
    ui_graph = {
        "nodes": [
            {
                "id": 1,
                "type": "LoadImage",
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
                "widgets_values": ["source.png"],
            },
            {
                "id": 2,
                "type": "VAEEncode",
                "inputs": [{"name": "pixels", "type": "IMAGE", "link": 1}],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [2]}],
                "widgets_values": [],
            },
            {
                "id": 3,
                "type": "KSampler",
                "inputs": [
                    {"name": "latent_image", "type": "LATENT", "link": 2},
                    {
                        "name": "denoise",
                        "type": "FLOAT",
                        "widget": {"name": "denoise"},
                    },
                ],
                "outputs": [],
                "widgets_values": [0.7],
            },
        ],
        "links": [
            [1, 1, 0, 2, 0, "IMAGE"],
            [2, 2, 0, 3, 0, "LATENT"],
        ],
    }
    object_info = {
        "LoadImage": {
            "input": {"required": {"image": [["source.png"], {"image_upload": True}]}},
            "input_order": {"required": ["image"]},
        },
        "VAEEncode": {
            "input": {"required": {"pixels": ["IMAGE"]}},
            "input_order": {"required": ["pixels"]},
        },
        "KSampler": {
            "input": {
                "required": {
                    "latent_image": ["LATENT"],
                    "denoise": ["FLOAT", {"default": 1.0}],
                }
            },
            "input_order": {"required": ["latent_image", "denoise"]},
        },
    }

    graph, schema = _compile_ui_graph(
        ui_graph,
        object_info,
        operation="image_to_image",
    )

    assert graph["3"]["inputs"]["denoise"] == "${denoise}"
    assert schema["properties"]["denoise"] == {"type": "number", "default": 0.7}


def _standard_checkpoint_object_info(checkpoint_names: list[str]) -> dict:  # type: ignore[type-arg]
    return {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": ["COMBO", {"options": checkpoint_names}],
                }
            },
            "input_order": {"required": ["ckpt_name"]},
        },
        "CLIPTextEncode": {
            "input": {
                "required": {
                    "clip": ["CLIP"],
                    "text": ["STRING", {"default": ""}],
                }
            },
            "input_order": {"required": ["clip", "text"]},
        },
        "EmptyLatentImage": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 512}],
                    "height": ["INT", {"default": 512}],
                    "batch_size": ["INT", {"default": 1}],
                }
            },
            "input_order": {"required": ["width", "height", "batch_size"]},
        },
        "KSampler": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "seed": [
                        "INT",
                        {"default": 0, "control_after_generate": True},
                    ],
                    "steps": ["INT", {"default": 20}],
                    "cfg": ["FLOAT", {"default": 7.0}],
                    "sampler_name": ["COMBO", {"options": ["euler"]}],
                    "scheduler": ["COMBO", {"options": ["normal"]}],
                    "positive": ["CONDITIONING"],
                    "negative": ["CONDITIONING"],
                    "latent_image": ["LATENT"],
                    "denoise": ["FLOAT", {"default": 1.0}],
                }
            },
            "input_order": {
                "required": [
                    "model",
                    "seed",
                    "steps",
                    "cfg",
                    "sampler_name",
                    "scheduler",
                    "positive",
                    "negative",
                    "latent_image",
                    "denoise",
                ]
            },
        },
        "VAEDecode": {
            "input": {
                "required": {
                    "samples": ["LATENT"],
                    "vae": ["VAE"],
                }
            },
            "input_order": {"required": ["samples", "vae"]},
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output_node": True,
        },
    }


def test_registry_builds_a_pinned_adaptive_standard_checkpoint_contract(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    adaptive = registry.adaptive_checkpoint(
        "owner/new-checkpoint",
        "a" * 40,
        ["weights/model.safetensors"],
        "image",
    )

    assert adaptive is not None
    assert adaptive.runtime_adaptive is True
    assert adaptive.id.startswith("lma_image_checkpoint_v1_")
    assert adaptive.comfy_paths == {"checkpoints": "weights"}
    assert (
        registry.adaptive_checkpoint(
            "owner/component-model",
            "a" * 40,
            ["unet/model.safetensors", "vae/model.safetensors"],
            "image",
        )
        is None
    )
    assert (
        registry.adaptive_checkpoint(
            "owner/video-model",
            "a" * 40,
            ["model.safetensors"],
            "video",
        )
        is None
    )


def test_registry_compiles_and_revalidates_an_adaptive_checkpoint(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    remote_id = "owner/new-checkpoint"
    revision = "a" * 40
    selected_files = ["weights/model.safetensors"]
    comfy_paths = {"checkpoints": "weights"}
    adaptive = registry.adaptive_checkpoint(
        remote_id,
        revision,
        selected_files,
        "image",
        comfy_paths=comfy_paths,
    )
    assert adaptive is not None

    compiled = registry.compile(
        adaptive.id,
        "image",
        _standard_checkpoint_object_info(["model.safetensors"]),
        remote_id=remote_id,
        revision=revision,
        selected_files=selected_files,
        comfy_paths=comfy_paths,
    )

    assert compiled.template.sha256 == adaptive.sha256
    assert compiled.api_graph["1"]["inputs"]["ckpt_name"] == "model.safetensors"
    assert compiled.api_graph["2"]["inputs"]["text"] == "${prompt}"
    assert compiled.api_graph["3"]["inputs"]["text"] == "${negative_prompt}"
    assert compiled.api_graph["5"]["inputs"]["model"] == ["1", 0]
    assert compiled.api_graph["7"]["inputs"]["images"] == ["6", 0]

    with pytest.raises(ValueError, match="binding does not match"):
        registry.validate_download(
            adaptive.id,
            "image",
            remote_id,
            ["weights/another.safetensors"],
            comfy_paths,
            revision=revision,
        )


def test_registry_compiles_modern_combo_widgets(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    object_info = {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [
                        "COMBO",
                        {"options": ["sd_xl_turbo_1.0_fp16.safetensors"]},
                    ]
                }
            },
            "input_order": {"required": ["ckpt_name"]},
        },
        "KSamplerSelect": {
            "input": {
                "required": {
                    "sampler_name": [
                        "COMBO",
                        {"options": ["euler", "euler_ancestral"]},
                    ]
                }
            },
            "input_order": {"required": ["sampler_name"]},
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output_node": True,
        },
    }

    compiled = registry.compile("sdxlturbo_example", "image", object_info)

    assert compiled.api_graph["1"]["inputs"]["ckpt_name"] == ("sd_xl_turbo_1.0_fp16.safetensors")
    assert compiled.api_graph["3"]["inputs"]["sampler_name"] == "${sampler}"
    assert compiled.input_schema["properties"]["sampler"] == {
        "type": "string",
        "default": "euler",
    }


def test_registry_binds_modern_random_noise_seed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert registry.settings.comfy_directory is not None
    template_path = (
        registry.settings.comfy_directory
        / ".venv"
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates"
        / "sdxlturbo_example.json"
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["nodes"].append(
        {
            "id": 4,
            "type": "RandomNoise",
            "inputs": [],
            "outputs": [{"name": "NOISE", "type": "NOISE", "links": []}],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [720512742793301],
        }
    )
    template_path.write_text(json.dumps(template), encoding="utf-8")
    object_info = {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [["sd_xl_turbo_1.0_fp16.safetensors"]],
                }
            },
            "input_order": {"required": ["ckpt_name"]},
        },
        "KSamplerSelect": {
            "input": {
                "required": {
                    "sampler_name": [["euler"]],
                }
            },
            "input_order": {"required": ["sampler_name"]},
        },
        "RandomNoise": {
            "input": {
                "required": {
                    "noise_seed": ["INT", {"default": 0}],
                }
            },
            "input_order": {"required": ["noise_seed"]},
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output_node": True,
        },
    }

    compiled = registry.compile("sdxlturbo_example", "image", object_info)

    assert compiled.api_graph["4"]["inputs"]["noise_seed"] == "${seed}"
    assert compiled.input_schema["properties"]["seed"] == {
        "type": "integer",
        "default": -1,
    }


def test_registry_normalizes_advanced_image_output_to_the_core_contract(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    assert registry.settings.comfy_directory is not None
    templates = (
        registry.settings.comfy_directory
        / ".venv"
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates"
    )
    template = {
        "nodes": [
            {
                "id": 1,
                "type": "CheckpointLoaderSimple",
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
                "properties": {
                    "cnr_id": "comfy-core",
                    "models": [
                        {
                            "name": "model.safetensors",
                            "url": (
                                "https://huggingface.co/owner/model/resolve/"
                                f"{'a' * 40}/model.safetensors"
                            ),
                            "directory": "checkpoints",
                        }
                    ],
                },
                "widgets_values": ["model.safetensors"],
            },
            {
                "id": 2,
                "type": "SaveImageAdvanced",
                "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
                "outputs": [],
                "properties": {"cnr_id": "comfy-core"},
                "widgets_values": ["Official_Template", "png", "8-bit", "sRGB"],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"]],
    }
    (templates / "image_advanced_output.json").write_text(
        json.dumps(template),
        encoding="utf-8",
    )
    object_info = {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [["model.safetensors"]],
                }
            },
            "input_order": {"required": ["ckpt_name"]},
        },
        "SaveImageAdvanced": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                    "format": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {"options": [{"key": "png"}]},
                    ],
                }
            },
            "input_order": {
                "required": ["images", "filename_prefix", "format"],
            },
            "output_node": True,
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output_node": True,
            "display_name": "Save Image",
        },
    }

    compiled = registry.compile("image_advanced_output", "image", object_info)

    assert compiled.api_graph["2"] == {
        "inputs": {
            "images": ["1", 0],
            "filename_prefix": "Official_Template",
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    }


def test_registry_rejects_a_template_model_missing_from_the_running_runtime(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    object_info = {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [
                        "COMBO",
                        {"options": ["another-model.safetensors"]},
                    ]
                }
            },
            "input_order": {"required": ["ckpt_name"]},
        },
        "KSamplerSelect": {
            "input": {
                "required": {
                    "sampler_name": [
                        "COMBO",
                        {"options": ["euler"]},
                    ]
                }
            },
            "input_order": {"required": ["sampler_name"]},
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output_node": True,
        },
    }

    with pytest.raises(ValueError, match="does not advertise"):
        registry.compile("sdxlturbo_example", "image", object_info)


def test_registry_compiles_subgraph_and_runtime_bindings(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    object_info = {
        "ImageFromText": {
            "input": {
                "required": {
                    "text": ["STRING", {"default": ""}],
                    "width": ["INT", {"default": 512}],
                }
            },
            "input_order": {"required": ["text", "width"]},
            "output_node": False,
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output_node": True,
        },
    }

    compiled = registry.compile("image_z_image_turbo_int8", "image", object_info)

    assert compiled.api_graph["57:1"]["inputs"] == {
        "text": "${prompt}",
        "width": "${width}",
    }
    assert compiled.api_graph["9"]["inputs"]["images"] == ["57:1", 0]
    assert compiled.input_schema["properties"]["prompt"] == {"type": "string"}
    assert compiled.input_schema["properties"]["width"] == {
        "type": "integer",
        "default": 1024,
    }


def test_text_to_image_graph_derives_a_standard_image_edit_workflow(tmp_path: Path) -> None:
    dependency = ComfyModelDependency(
        remote_id="Comfy-Org/z_image_turbo",
        revision="main",
        path="z_image.safetensors",
        directory="diffusion_models",
        name="z_image.safetensors",
        url="",
    )
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="image_z_image_turbo",
            path=tmp_path / "template.json",
            role="image",
            operation="text_to_image",
            score=1_000,
            sha256="a" * 64,
            dependencies=(dependency,),
        ),
        ui_graph={"nodes": []},
        api_graph={
            "empty": {"class_type": "EmptySD3LatentImage", "inputs": {}},
            "vae": {"class_type": "VAELoader", "inputs": {}},
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "latent_image": ["empty", 0],
                    "denoise": "${denoise}",
                    "steps": "${steps}",
                },
            },
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
            },
        },
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "width": {"type": "integer", "default": 1024},
                "height": {"type": "integer", "default": 1024},
                "batch_size": {"type": "integer", "default": 1},
                "denoise": {"type": "number", "default": 1.0},
                "steps": {"type": "integer", "default": 4},
            },
        },
    )

    derived = derive_image_to_image(
        compiled,
        {"LoadImage": {"input": {}}, "VAEEncode": {"input": {}}},
    )

    assert derived is not None
    assert derived.template.operation == "image_to_image"
    assert derived.api_graph["lma-load-image"]["inputs"] == {"image": "${input_image}"}
    assert derived.api_graph["lma-vae-encode"]["inputs"] == {
        "pixels": ["lma-load-image", 0],
        "vae": ["vae", 0],
    }
    assert derived.api_graph["sampler"]["inputs"]["latent_image"] == [
        "lma-vae-encode",
        0,
    ]
    assert set(derived.input_schema["properties"]) == {"prompt", "denoise", "steps"}
    assert derived.input_schema["properties"]["denoise"] == {
        "type": "number",
        "default": 0.9,
        "minimum": 0.0,
        "maximum": 1.0,
        "title": "Edit strength",
        "description": (
            "Higher values make the requested change more visible; "
            "lower values preserve more of the source."
        ),
        "x-lm-atelier-visibility": "basic",
    }
    assert derived.input_schema["x-lm-atelier-edit-calibration"] == {
        "version": 1,
        "edit_strength": {
            "parameter": "denoise",
            "minimum": 0.0,
            "maximum": 1.0,
            "recommended": {
                "minimal": 0.38,
                "localized": 0.5,
                "replacement": 0.66,
                "global": 0.82,
                "fallback": 0.56,
            },
        },
        "schedule": {
            "steps_parameter": "steps",
            "minimum_effective_steps": {
                "localized": 2,
                "replacement": 7.2,
                "global": 3,
            },
        },
    }


def test_subgraph_prompt_labels_bind_custom_conditioning_encoders() -> None:
    ui_graph = {
        "nodes": [
            {
                "id": 10,
                "type": "edit-subgraph",
                "inputs": [],
                "outputs": [],
            }
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": "edit-subgraph",
                    "inputs": [
                        {
                            "name": "prompt",
                            "label": "positive_prompt",
                            "type": "STRING",
                        },
                        {
                            "name": "prompt_1",
                            "label": "negative_prompt",
                            "type": "STRING",
                        },
                    ],
                    "nodes": [
                        {
                            "id": 1,
                            "type": "CustomInstructionTextEncode",
                            "inputs": [
                                {
                                    "name": "prompt",
                                    "type": "STRING",
                                    "widget": {"name": "prompt"},
                                }
                            ],
                            "outputs": [
                                {
                                    "name": "CONDITIONING",
                                    "type": "CONDITIONING",
                                    "links": [],
                                }
                            ],
                            "widgets_values": ["sample positive prompt"],
                        },
                        {
                            "id": 2,
                            "type": "CustomInstructionTextEncode",
                            "inputs": [
                                {
                                    "name": "prompt",
                                    "type": "STRING",
                                    "widget": {"name": "prompt"},
                                }
                            ],
                            "outputs": [
                                {
                                    "name": "CONDITIONING",
                                    "type": "CONDITIONING",
                                    "links": [],
                                }
                            ],
                            "widgets_values": [""],
                        },
                    ],
                    "links": [
                        {
                            "id": 1,
                            "origin_id": -10,
                            "origin_slot": 0,
                            "target_id": 1,
                            "target_slot": 0,
                            "type": "STRING",
                        },
                        {
                            "id": 2,
                            "origin_id": -10,
                            "origin_slot": 1,
                            "target_id": 2,
                            "target_slot": 0,
                            "type": "STRING",
                        },
                    ],
                }
            ]
        },
    }
    object_info = {
        "CustomInstructionTextEncode": {
            "input": {
                "required": {
                    "prompt": ["STRING", {"default": ""}],
                }
            },
            "input_order": {"required": ["prompt"]},
        }
    }

    graph, schema = _compile_ui_graph(
        ui_graph,
        object_info,
        operation="image_to_image",
    )

    assert graph["10:1"]["inputs"]["prompt"] == "${prompt}"
    assert graph["10:2"]["inputs"]["prompt"] == "${negative_prompt}"
    assert schema["properties"]["prompt"] == {"type": "string"}
    assert schema["properties"]["negative_prompt"] == {
        "type": "string",
        "default": "",
    }


def test_subgraph_prompt_labels_do_not_replace_conditioning_links() -> None:
    ui_graph = {
        "nodes": [
            {
                "id": 1,
                "type": "CLIPTextEncode",
                "inputs": [
                    {
                        "name": "text",
                        "type": "STRING",
                        "widget": {"name": "text"},
                    }
                ],
                "outputs": [
                    {
                        "name": "CONDITIONING",
                        "type": "CONDITIONING",
                        "links": [9],
                    }
                ],
                "widgets_values": ["sample prompt"],
            },
            {
                "id": 10,
                "type": "edit-subgraph",
                "inputs": [
                    {
                        "name": "positive",
                        "type": "CONDITIONING",
                        "link": 9,
                    }
                ],
                "outputs": [],
            },
        ],
        "links": [[9, 1, 0, 10, 0, "CONDITIONING"]],
        "definitions": {
            "subgraphs": [
                {
                    "id": "edit-subgraph",
                    "inputs": [
                        {
                            "name": "positive",
                            "label": "positive_prompt",
                            "type": "CONDITIONING",
                        }
                    ],
                    "nodes": [
                        {
                            "id": 2,
                            "type": "ReferenceLatent",
                            "inputs": [
                                {
                                    "name": "conditioning",
                                    "type": "CONDITIONING",
                                }
                            ],
                            "outputs": [
                                {
                                    "name": "CONDITIONING",
                                    "type": "CONDITIONING",
                                    "links": [],
                                }
                            ],
                            "widgets_values": [],
                        }
                    ],
                    "links": [
                        {
                            "id": 8,
                            "origin_id": -10,
                            "origin_slot": 0,
                            "target_id": 2,
                            "target_slot": 0,
                            "type": "CONDITIONING",
                        }
                    ],
                }
            ]
        },
    }
    object_info = {
        "CLIPTextEncode": {
            "input": {
                "required": {
                    "text": ["STRING", {"default": ""}],
                }
            },
            "input_order": {"required": ["text"]},
        },
        "ReferenceLatent": {
            "input": {
                "required": {
                    "conditioning": ["CONDITIONING"],
                }
            },
            "input_order": {"required": ["conditioning"]},
        },
    }

    graph, _ = _compile_ui_graph(
        ui_graph,
        object_info,
        operation="image_to_image",
    )

    assert graph["1"]["inputs"]["text"] == "${prompt}"
    assert graph["10:2"]["inputs"]["conditioning"] == ["1", 0]


def test_runtime_parameter_names_do_not_replace_graph_links() -> None:
    ui_graph = {
        "nodes": [
            {
                "id": 1,
                "type": "PrimitiveInt",
                "inputs": [],
                "outputs": [{"name": "INT", "type": "INT", "links": [1]}],
                "widgets_values": [40],
            },
            {
                "id": 2,
                "type": "KSampler",
                "inputs": [
                    {
                        "name": "steps",
                        "type": "INT",
                        "widget": {"name": "steps"},
                        "link": 1,
                    }
                ],
                "outputs": [],
                "widgets_values": [40],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "INT"]],
    }
    object_info = {
        "PrimitiveInt": {
            "input": {"required": {"value": ["INT", {"default": 0}]}},
            "input_order": {"required": ["value"]},
        },
        "KSampler": {
            "input": {"required": {"steps": ["INT", {"default": 20}]}},
            "input_order": {"required": ["steps"]},
        },
    }

    graph, schema = _compile_ui_graph(
        ui_graph,
        object_info,
        operation="image_to_image",
    )

    assert graph["2"]["inputs"]["steps"] == ["1", 0]
    assert schema["properties"]["steps"] == {"readOnly": True}
    assert schema["properties"]["cfg"] == {"readOnly": True}


def _four_step_edit_graph() -> dict[str, Any]:
    return {
        "inputs": [
            {
                "name": "value",
                "label": "enable_turbo_mode",
                "type": "BOOLEAN",
            }
        ],
        "nodes": [
            {
                "id": 1,
                "type": "PrimitiveBoolean",
                "inputs": [{"name": "value", "type": "BOOLEAN"}],
                "widgets_values": [False],
            },
            {"id": 2, "type": "PrimitiveInt", "widgets_values": [40]},
            {"id": 3, "type": "PrimitiveInt", "widgets_values": [4]},
            {"id": 4, "type": "BaseModel"},
            {
                "id": 5,
                "type": "LoraLoaderModelOnly",
                "properties": {
                    "models": [
                        {
                            "name": "edit-lightning.safetensors",
                            "url": (
                                "https://huggingface.co/example/edit-lightning/resolve/"
                                "main/edit-lightning.safetensors"
                            ),
                            "directory": "loras",
                        }
                    ]
                },
                "widgets_values": ["edit-lightning.safetensors", 1.0],
            },
            {
                "id": 6,
                "type": "ComfySwitchNode",
                "title": "Switch (Steps)",
                "inputs": [
                    {"name": "on_false", "type": "INT"},
                    {"name": "on_true", "type": "INT"},
                    {"name": "switch", "type": "BOOLEAN"},
                ],
            },
            {
                "id": 7,
                "type": "ComfySwitchNode",
                "title": "Switch (Model)",
                "inputs": [
                    {"name": "on_false", "type": "MODEL"},
                    {"name": "on_true", "type": "MODEL"},
                    {"name": "switch", "type": "BOOLEAN"},
                ],
            },
            {
                "id": 8,
                "type": "Sampler",
                "inputs": [
                    {"name": "steps", "type": "INT"},
                    {"name": "model", "type": "MODEL"},
                ],
            },
        ],
        "links": [
            [-1, -10, 0, 1, 0, "BOOLEAN"],
            [1, 1, 0, 6, 2, "BOOLEAN"],
            [2, 1, 0, 7, 2, "BOOLEAN"],
            [3, 2, 0, 6, 0, "INT"],
            [4, 3, 0, 6, 1, "INT"],
            [5, 4, 0, 7, 0, "MODEL"],
            [6, 5, 0, 7, 1, "MODEL"],
            [7, 6, 0, 8, 0, "INT"],
            [8, 7, 0, 8, 1, "MODEL"],
        ],
    }


def test_declared_four_step_edit_acceleration_enables_complete_bundled_branch() -> None:
    subgraph = _four_step_edit_graph()
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}

    accelerated = _enable_declared_four_step_edit_acceleration(
        ui_graph,
        operation="image_to_image",
    )

    assert accelerated is True
    assert subgraph["nodes"][0]["widgets_values"] == [True]


def test_declared_four_step_edit_acceleration_is_edit_only() -> None:
    subgraph = _four_step_edit_graph()
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}

    accelerated = _enable_declared_four_step_edit_acceleration(
        ui_graph,
        operation="text_to_image",
    )

    assert accelerated is False
    assert subgraph["nodes"][0]["widgets_values"] == [False]


def test_declared_four_step_edit_acceleration_requires_bundled_lora_metadata() -> None:
    subgraph = _four_step_edit_graph()
    subgraph["nodes"][4]["properties"] = {}
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}

    accelerated = _enable_declared_four_step_edit_acceleration(
        ui_graph,
        operation="image_to_image",
    )

    assert accelerated is False
    assert subgraph["nodes"][0]["widgets_values"] == [False]
