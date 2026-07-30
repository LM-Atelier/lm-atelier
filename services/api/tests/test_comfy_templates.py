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
    WorkflowAccelerationRecipe,
    _apply_workflow_performance_bounds,
    _compile_ui_graph,
    _resolve_declared_workflow_acceleration,
    _template_performance_hints,
    _workflow_acceleration_provenance,
    _workflow_performance_provenance,
    derive_image_to_image,
)
from local_lm.config import Settings
from local_lm.schemas import CatalogFileSource


def _installed_templates(registry: ComfyTemplateRegistry) -> Path:
    """The directory the registry actually reads, so fixtures cannot drift from it."""
    executable = registry.settings.comfy_executable
    assert executable is not None
    return (
        executable.parent
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates"
    )


def _registry(tmp_path: Path) -> ComfyTemplateRegistry:
    # Mirror the official Windows portable release that LM Atelier provisions:
    # the interpreter lives beside the ComfyUI directory, not inside it.
    portable = tmp_path / "ComfyUI_windows_portable"
    comfy = portable / "ComfyUI"
    comfy.mkdir(parents=True)
    executable = portable / "python_embeded" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    templates = (
        executable.parent
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates"
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
    settings = Settings(
        data_dir=tmp_path / "data",
        comfy_directory=comfy,
        comfy_executable=executable,
    )
    settings.prepare()
    return ComfyTemplateRegistry(settings)


def _discovery_settings(tmp_path: Path, layout: str) -> Settings:
    """Build one of the ComfyUI layouts LM Atelier has to read templates from."""
    comfy = tmp_path / "comfy"
    executable: Path | None = None
    if layout == "portable":
        portable = tmp_path / "ComfyUI_windows_portable"
        comfy = portable / "ComfyUI"
        executable = portable / "python_embeded" / "python.exe"
        site_packages = executable.parent / "Lib" / "site-packages"
    elif layout == "portable_directory_only":
        portable = tmp_path / "ComfyUI_windows_portable"
        comfy = portable / "ComfyUI"
        site_packages = portable / "python_embeded" / "Lib" / "site-packages"
    elif layout == "windows_venv":
        executable = comfy / ".venv" / "Scripts" / "python.exe"
        site_packages = comfy / ".venv" / "Lib" / "site-packages"
    elif layout == "posix_venv":
        executable = comfy / ".venv" / "bin" / "python"
        site_packages = comfy / ".venv" / "lib" / "python3.13" / "site-packages"
    elif layout == "venv_directory_only":
        site_packages = comfy / ".venv" / "Lib" / "site-packages"
    else:  # pragma: no cover - guards a typo in the parametrisation
        raise AssertionError(f"unknown layout {layout}")
    templates = site_packages / "comfyui_workflow_templates_json" / "templates"
    templates.mkdir(parents=True)
    comfy.mkdir(parents=True, exist_ok=True)
    (templates / "example.json").write_text(
        json.dumps({"nodes": [], "links": []}), encoding="utf-8"
    )
    if executable is not None:
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
    settings = Settings(
        data_dir=tmp_path / "data",
        comfy_directory=comfy,
        comfy_executable=executable,
    )
    settings.prepare()
    return settings


@pytest.mark.parametrize(
    "layout",
    ["portable", "portable_directory_only", "windows_venv", "posix_venv", "venv_directory_only"],
)
def test_template_discovery_supports_every_runtime_layout(tmp_path: Path, layout: str) -> None:
    registry = ComfyTemplateRegistry(_discovery_settings(tmp_path, layout))

    discovered = registry._template_files()

    assert [path.name for path in discovered] == ["example.json"]


def test_template_discovery_returns_nothing_when_no_runtime_is_configured(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()

    assert ComfyTemplateRegistry(settings)._template_files() == []


def test_template_discovery_prefers_the_configured_interpreter(tmp_path: Path) -> None:
    """The executable runs the graph, so its templates win over a stale sibling."""
    portable = tmp_path / "ComfyUI_windows_portable"
    comfy = portable / "ComfyUI"
    comfy.mkdir(parents=True)
    executable = portable / "python_embeded" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    graph = json.dumps({"nodes": [], "links": []})
    for site_packages, name in (
        (executable.parent / "Lib" / "site-packages", "from_interpreter.json"),
        (comfy / ".venv" / "Lib" / "site-packages", "from_directory.json"),
    ):
        templates = site_packages / "comfyui_workflow_templates_json" / "templates"
        templates.mkdir(parents=True)
        (templates / name).write_text(graph, encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "data",
        comfy_directory=comfy,
        comfy_executable=executable,
    )
    settings.prepare()

    discovered = ComfyTemplateRegistry(settings)._template_files()

    assert [path.name for path in discovered] == ["from_interpreter.json"]


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
    templates = _installed_templates(registry)
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

    assert "image_ambiguous_component_paths" in {item.id for item in matches}
    shared_parent = next(item for item in matches if item.id == "image_ambiguous_component_paths")
    assert shared_parent.comfy_paths == {
        "diffusion_models": "weights",
        "vae": "weights",
    }
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


def test_performance_contract_uses_metadata_graph_and_runtime_evidence() -> None:
    dependency = ComfyModelDependency(
        remote_id="owner/model",
        revision="a" * 40,
        path="model.safetensors",
        directory="checkpoints",
        name="model.safetensors",
        url="https://huggingface.co/owner/model/resolve/main/model.safetensors",
    )
    hints = _template_performance_hints(
        {
            "title": "Distilled editor with KV cache",
            "description": "An authored low-step workflow with cached key values.",
        }
    )
    template = ComfyTemplate(
        id="image_native_performance_edit",
        path=Path("unused.json"),
        role="image",
        operation="image_to_image",
        score=0,
        sha256="a" * 64,
        dependencies=(dependency,),
        published_date="2026-07-20",
        performance_hints=hints,
    )
    ordinary = ComfyTemplate(
        id="image_turbo_name_only_edit",
        path=Path("unused.json"),
        role="image",
        operation="image_to_image",
        score=0,
        sha256="b" * 64,
        dependencies=(dependency,),
        published_date="2026-07-21",
    )
    ui_graph = {
        "nodes": [
            {"id": 1, "type": "Sampler", "widgets_values": [4]},
        ]
    }
    api_graph = {
        "cache": {
            "class_type": "ModelKVCache",
            "inputs": {"model": ["loader", 0]},
        },
        "sampler": {
            "class_type": "Sampler",
            "inputs": {"model": ["cache", 0], "steps": "${steps}"},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["sampler", 0]},
        },
    }
    object_info = {
        "Sampler": {
            "input": {"required": {"steps": ["INT", {"default": 20}]}},
            "input_order": {"required": ["steps"]},
        },
        "ModelKVCache": {
            "display_name": "Model KV Cache",
            "description": "Caches model key value tensors for reuse.",
            "input": {"required": {"model": ["MODEL"]}},
            "output": ["MODEL"],
        },
        "SaveImage": {"output_node": True},
    }
    input_schema = {
        "type": "object",
        "properties": {"steps": {"type": "integer", "default": 4}},
    }

    performance = _workflow_performance_provenance(
        template,
        ui_graph,
        api_graph,
        input_schema,
        object_info,
        (),
    )

    assert hints == ("distilled", "low-step", "kv-cache")
    assert template.preference_score > ordinary.preference_score
    assert performance and [item["kind"] for item in performance["signals"]] == [
        "native-low-step",
        "model-cache",
    ]
    assert performance["signals"][0]["steps"] == 4
    assert performance["signals"][1]["node_types"] == ["ModelKVCache"]
    assert performance["signals"][1]["steps"] == 4
    bounds = _apply_workflow_performance_bounds(input_schema, performance)
    assert bounds == {"steps": {"default": 4, "maximum": 8}}
    assert input_schema["properties"]["steps"]["maximum"] == 8

    name_only = _workflow_performance_provenance(
        ordinary,
        ui_graph,
        {"save": api_graph["save"]},
        {"type": "object", "properties": {"steps": {"default": 4}}},
        object_info,
        (),
    )
    assert ordinary.performance_hints == ()
    assert name_only is None


def test_declared_acceleration_receives_the_same_safe_schedule_bound() -> None:
    template = ComfyTemplate(
        id="image_generic_edit",
        path=Path("unused.json"),
        role="image",
        operation="image_to_image",
        score=0,
        sha256="a" * 64,
        dependencies=(
            ComfyModelDependency(
                remote_id="owner/model",
                revision="a" * 40,
                path="model.safetensors",
                directory="checkpoints",
                name="model.safetensors",
                url="",
            ),
        ),
    )
    recipe = WorkflowAccelerationRecipe(
        kind="lightning",
        mode="declared-low-step",
        step_schedules=((4, 40),),
        switched_inputs=("steps",),
        asset_kinds=(),
    )
    schema = {"type": "object", "properties": {"steps": {"default": 4}}}

    performance = _workflow_performance_provenance(
        template,
        {"nodes": []},
        {},
        schema,
        {},
        (recipe,),
    )

    assert performance and performance["signals"] == [
        {
            "kind": "declared-acceleration",
            "mode": "declared-low-step",
            "recipe_count": 1,
            "source": "author-declared-graph-branch",
            "steps": 4,
        }
    ]
    assert _apply_workflow_performance_bounds(schema, performance) == {
        "steps": {"default": 4, "maximum": 8}
    }


def test_native_edit_loaders_bind_ordered_runtime_images(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    templates = _installed_templates(registry)
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
    template_path = _installed_templates(registry) / "sdxlturbo_example.json"
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
    templates = _installed_templates(registry)
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


def test_declared_acceleration_enables_complete_bundled_branch() -> None:
    subgraph = _four_step_edit_graph()
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="image_to_image",
    )

    assert subgraph["nodes"][0]["widgets_values"] == [True]
    assert len(recipes) == 1
    assert recipes[0].kind == "turbo"
    assert recipes[0].mode == "bundled-four-step"
    assert recipes[0].step_schedules == ((4, 40),)
    assert recipes[0].switched_inputs == ("model", "steps")
    assert recipes[0].asset_kinds == ("bundled-lora",)
    assert _workflow_acceleration_provenance(recipes) == {
        "version": 1,
        "mode": "bundled-four-step",
        "recipe_count": 1,
        "recipes": [
            {
                "kind": "turbo",
                "mode": "bundled-four-step",
                "step_schedules": [{"steps": 4, "baseline_steps": 40}],
                "switched_inputs": ["model", "steps"],
                "asset_kinds": ["bundled-lora"],
            }
        ],
        "steps": 4,
        "baseline_steps": 40,
    }


@pytest.mark.parametrize(
    "operation",
    ["text_to_image", "image_to_image", "text_to_video", "image_to_video"],
)
def test_declared_acceleration_supports_every_media_operation(operation: str) -> None:
    subgraph = _four_step_edit_graph()
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}

    recipes = _resolve_declared_workflow_acceleration(ui_graph, operation=operation)

    assert len(recipes) == 1
    assert subgraph["nodes"][0]["widgets_values"] == [True]


def test_declared_acceleration_ignores_non_media_operations() -> None:
    subgraph = _four_step_edit_graph()
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}
    original = json.loads(json.dumps(ui_graph))

    recipes = _resolve_declared_workflow_acceleration(ui_graph, operation="chat")

    assert recipes == ()
    assert ui_graph == original


def test_declared_acceleration_requires_bundled_asset_metadata() -> None:
    subgraph = _four_step_edit_graph()
    subgraph["nodes"][4]["properties"] = {}
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}
    original = json.loads(json.dumps(ui_graph))

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="image_to_image",
    )

    assert recipes == ()
    assert ui_graph == original


def test_declared_acceleration_rejects_unrelated_fast_toggle() -> None:
    subgraph = _four_step_edit_graph()
    subgraph["inputs"][0]["label"] = "fast_preview"
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}
    original = json.loads(json.dumps(ui_graph))

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="image_to_image",
    )

    assert recipes == ()
    assert ui_graph == original


def test_declared_acceleration_requires_executed_model_branch() -> None:
    subgraph = _four_step_edit_graph()
    subgraph["links"] = [link for link in subgraph["links"] if link[0] != 8]
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}
    original = json.loads(json.dumps(ui_graph))

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="image_to_image",
    )

    assert recipes == ()
    assert ui_graph == original


def test_declared_acceleration_supports_bundled_low_step_model() -> None:
    subgraph = _four_step_edit_graph()
    subgraph["inputs"][0]["label"] = "use_lightning_mode"
    subgraph["nodes"][2]["widgets_values"] = [8]
    subgraph["nodes"][4] = {
        "id": 5,
        "type": "UNETLoader",
        "properties": {
            "models": [
                {
                    "name": "accelerated-model.safetensors",
                    "url": (
                        "https://huggingface.co/example/accelerated/resolve/"
                        "main/accelerated-model.safetensors"
                    ),
                    "directory": "diffusion_models",
                }
            ]
        },
        "widgets_values": ["accelerated-model.safetensors", "default"],
    }
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="text_to_video",
    )

    assert len(recipes) == 1
    assert recipes[0].kind == "lightning"
    assert recipes[0].mode == "declared-low-step"
    assert recipes[0].step_schedules == ((8, 40),)
    assert recipes[0].asset_kinds == ("bundled-model",)


def test_declared_acceleration_supports_step_only_distilled_workflow() -> None:
    subgraph = _four_step_edit_graph()
    subgraph["inputs"][0]["label"] = "enable_lcm_mode"
    subgraph["nodes"] = [node for node in subgraph["nodes"] if node["id"] not in {5, 7}]
    subgraph["nodes"][2]["widgets_values"] = [31]
    subgraph["links"] = [
        link for link in subgraph["links"] if link[1] not in {5, 7} and link[3] not in {5, 7}
    ]
    subgraph["links"].append([9, 4, 0, 8, 1, "MODEL"])
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="text_to_image",
    )

    assert len(recipes) == 1
    assert recipes[0].kind == "lcm"
    assert recipes[0].mode == "declared-low-step"
    assert recipes[0].step_schedules == ((31, 40),)
    assert recipes[0].switched_inputs == ("steps",)
    assert recipes[0].asset_kinds == ()
    assert subgraph["nodes"][0]["widgets_values"] == [True]


def test_declared_acceleration_discovers_nested_subgraph() -> None:
    subgraph = _four_step_edit_graph()
    nested = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [nested]}}

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="image_to_video",
    )

    assert len(recipes) == 1
    assert subgraph["nodes"][0]["widgets_values"] == [True]


def test_declared_acceleration_preserves_independent_stage_recipes() -> None:
    first = _four_step_edit_graph()
    second = _four_step_edit_graph()
    second["nodes"][2]["widgets_values"] = [6]
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [first, second]}}

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="text_to_video",
    )
    provenance = _workflow_acceleration_provenance(recipes)

    assert len(recipes) == 2
    assert first["nodes"][0]["widgets_values"] == [True]
    assert second["nodes"][0]["widgets_values"] == [True]
    assert provenance["mode"] == "declared-accelerated-branches"
    assert provenance["recipe_count"] == 2
    assert "steps" not in provenance
    assert provenance["baseline_steps"] == 40


def test_declared_acceleration_rejects_non_reducing_schedule() -> None:
    subgraph = _four_step_edit_graph()
    subgraph["nodes"][2]["widgets_values"] = [40]
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}
    original = json.loads(json.dumps(ui_graph))

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="image_to_image",
    )

    assert recipes == ()
    assert ui_graph == original


def test_declared_acceleration_leaves_ambiguous_graph_unchanged() -> None:
    first = _four_step_edit_graph()
    second = json.loads(json.dumps(first))
    for node in second["nodes"]:
        node["id"] += 100
    for link in second["links"]:
        link[0] += 100
        if link[1] != -10:
            link[1] += 100
        else:
            link[2] = 1
        if link[3] != -20:
            link[3] += 100
    combined = {
        "inputs": [first["inputs"][0], second["inputs"][0]],
        "nodes": first["nodes"] + second["nodes"],
        "links": first["links"] + second["links"],
    }
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [combined]}}
    original = json.loads(json.dumps(ui_graph))

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="image_to_image",
    )

    assert recipes == ()
    assert ui_graph == original


def _schedule_only_fast_graph() -> dict[str, Any]:
    graph = _four_step_edit_graph()
    graph["inputs"][0]["label"] = "enable_fast_mode"
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in {4, 5, 7}]
    sampler = next(node for node in graph["nodes"] if node["id"] == 8)
    sampler["inputs"] = [
        {"name": "steps", "type": "INT"},
        {"name": "cfg", "type": "FLOAT"},
    ]
    graph["nodes"].extend(
        [
            {"id": 9, "type": "PrimitiveFloat", "widgets_values": [5.0]},
            {"id": 10, "type": "PrimitiveFloat", "widgets_values": [1.0]},
            {
                "id": 11,
                "type": "ComfySwitchNode",
                "inputs": [
                    {"name": "on_false", "type": "FLOAT"},
                    {"name": "on_true", "type": "FLOAT"},
                    {"name": "switch", "type": "BOOLEAN"},
                ],
            },
        ]
    )
    graph["links"] = [
        link for link in graph["links"] if link[1] not in {4, 5, 7} and link[3] not in {7}
    ]
    graph["links"].extend(
        [
            [9, 1, 0, 11, 2, "BOOLEAN"],
            [10, 9, 0, 11, 0, "FLOAT"],
            [11, 10, 0, 11, 1, "FLOAT"],
            [12, 11, 0, 8, 1, "FLOAT"],
        ]
    )
    return graph


def test_declared_acceleration_supports_atomic_schedule_only_recipe() -> None:
    subgraph = _schedule_only_fast_graph()
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="text_to_image",
    )

    assert len(recipes) == 1
    assert recipes[0].kind == "fast"
    assert recipes[0].mode == "declared-low-step"
    assert recipes[0].step_schedules == ((4, 40),)
    assert recipes[0].switched_inputs == ("cfg", "steps")
    assert recipes[0].asset_kinds == ()
    assert subgraph["nodes"][0]["widgets_values"] == [True]


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("Enable acceleration mode", "acceleration"),
        ("Use distilled mode", "distilled"),
        ("Enable few-step mode", "few-step"),
        ("Enable Hyper-SD mode", "hyper"),
        ("Use low step mode", "low-step"),
        ("Enable Schnell mode", "schnell"),
        ("Enable TCD mode", "tcd"),
    ],
)
def test_declared_acceleration_recognizes_common_techniques(
    label: str,
    kind: str,
) -> None:
    subgraph = _four_step_edit_graph()
    subgraph["inputs"][0]["label"] = label
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="text_to_image",
    )

    assert len(recipes) == 1
    assert recipes[0].kind == kind


@pytest.mark.parametrize(("link_index", "slot_index"), [(0, 2), (0, 4)])
def test_declared_acceleration_rejects_negative_graph_slots(
    link_index: int,
    slot_index: int,
) -> None:
    subgraph = _four_step_edit_graph()
    subgraph["links"][link_index][slot_index] = -1
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}
    original = json.loads(json.dumps(ui_graph))

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="image_to_image",
    )

    assert recipes == ()
    assert ui_graph == original


def test_declared_acceleration_bounds_nested_graph_discovery() -> None:
    subgraph = _four_step_edit_graph()
    ui_graph: dict[str, Any] = subgraph
    for _ in range(256):
        ui_graph = {"nodes": [], "definitions": {"subgraphs": [ui_graph]}}

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="image_to_image",
    )

    assert recipes == ()
    assert subgraph["nodes"][0]["widgets_values"] == [False]


@pytest.mark.parametrize("schedule_input", ["start_at_step", "end_at_step"])
def test_declared_acceleration_preserves_auxiliary_schedule_switches(
    schedule_input: str,
) -> None:
    subgraph = _schedule_only_fast_graph()
    sampler = next(node for node in subgraph["nodes"] if node["id"] == 8)
    sampler["inputs"][1]["name"] = schedule_input
    ui_graph = {"nodes": [], "definitions": {"subgraphs": [subgraph]}}

    recipes = _resolve_declared_workflow_acceleration(
        ui_graph,
        operation="text_to_video",
    )

    assert len(recipes) == 1
    assert recipes[0].switched_inputs == (schedule_input, "steps")
