from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_lm.comfy_templates import (
    ComfyModelDependency,
    ComfyTemplate,
    ComfyTemplateRegistry,
    CompiledComfyTemplate,
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
