from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_lm.comfy_templates import ComfyTemplateRegistry
from local_lm.config import Settings


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
