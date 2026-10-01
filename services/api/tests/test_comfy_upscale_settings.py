"""Expose a scale setting only when a recognized image node receives it."""

from typing import Any

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.comfy_templates import compile_authored_workflow
from local_lm.settings_registry import resolve_generation_settings, workflow_settings


def _scale_graph(class_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = {
        "nodes": [
            {"id": 1, "type": "LoadImage", "widgets_values": ["source.png"]},
            {
                "id": 2,
                "type": class_type,
                "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
                "widgets_values": [2.5],
            },
            {
                "id": 3,
                "type": "SaveImage",
                "inputs": [{"name": "images", "type": "IMAGE", "link": 2}],
                "widgets_values": ["output"],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"]],
    }
    object_info = {
        "LoadImage": {
            "input": {"required": {"image": [["source.png"], {"image_upload": True}]}},
            "output": ["IMAGE", "MASK"],
        },
        class_type: {
            "input": {
                "required": {
                    "image": ["IMAGE", {}],
                    "scale_by": ["FLOAT", {"default": 2.5, "min": 0.5, "max": 4.0}],
                }
            },
            "output": ["IMAGE"],
        },
        "SaveImage": {
            "input": {"required": {"images": ["IMAGE", {}], "filename_prefix": ["STRING"]}},
            "output": [],
            "output_node": True,
        },
    }
    return graph, object_info


def _compile_scale(class_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    graph, object_info = _scale_graph(class_type)
    return compile_authored_workflow(graph, object_info, operation="image_to_image")


def test_an_image_scale_widget_receives_the_advertised_factor() -> None:
    graph, schema = _compile_scale("ImageScaleBy")

    assert graph["2"]["inputs"]["image"] == ["1", 0]
    assert graph["2"]["inputs"]["scale_by"] == "${upscale_factor}"
    field = schema["properties"]["upscale_factor"]
    assert field["type"] == "number"
    assert field["default"] == 2.5
    assert field["maximum"] == 4.0


def test_an_unknown_scale_widget_keeps_its_authored_value() -> None:
    graph, schema = _compile_scale("NeutralImageTransform")

    assert graph["2"]["inputs"]["scale_by"] == 2.5
    assert "upscale_factor" not in schema["properties"]


def test_a_disconnected_scale_keeps_its_authored_value() -> None:
    graph, object_info = _scale_graph("ImageScaleBy")
    graph["links"][1][1] = 1

    compiled, schema = compile_authored_workflow(graph, object_info, operation="image_to_image")

    assert compiled["2"]["inputs"]["scale_by"] == 2.5
    assert "upscale_factor" not in schema["properties"]


def test_two_consecutive_scales_keep_their_authored_values() -> None:
    graph, object_info = _scale_graph("ImageScaleBy")
    graph["nodes"].append(
        {
            "id": 4,
            "type": "ImageScaleBy",
            "inputs": [{"name": "image", "type": "IMAGE", "link": 3}],
            "widgets_values": [2.5],
        }
    )
    graph["links"][1][1] = 4
    graph["links"].append([3, 2, 0, 4, 0, "IMAGE"])

    compiled, schema = compile_authored_workflow(graph, object_info, operation="image_to_image")

    assert compiled["2"]["inputs"]["scale_by"] == 2.5
    assert compiled["4"]["inputs"]["scale_by"] == 2.5
    assert "upscale_factor" not in schema["properties"]


def test_an_unscaled_output_prevents_a_shared_factor_setting() -> None:
    graph, object_info = _scale_graph("ImageScaleBy")
    graph["nodes"].append(
        {
            "id": 4,
            "type": "SaveImage",
            "inputs": [{"name": "images", "type": "IMAGE", "link": 3}],
            "widgets_values": ["original"],
        }
    )
    graph["links"].append([3, 1, 0, 4, 0, "IMAGE"])

    compiled, schema = compile_authored_workflow(graph, object_info, operation="image_to_image")

    assert compiled["2"]["inputs"]["scale_by"] == 2.5
    assert "upscale_factor" not in schema["properties"]


def test_a_fixed_factor_is_not_exposed_as_an_editable_workflow_setting() -> None:
    schema = {
        "type": "object",
        "properties": {
            "upscale_factor": {
                "type": "number",
                "readOnly": True,
                "const": 4,
                "x-lm-atelier-kind": "upscale",
            }
        },
    }

    assert workflow_settings([], schema) == []


def test_an_existing_custom_binding_keeps_its_resolved_value() -> None:
    schema = {
        "type": "object",
        "properties": {"custom_strength": {"type": "number", "readOnly": True, "default": 0.5}},
    }
    graph = {"sample": {"class_type": "KSampler", "inputs": {"denoise": "${custom_strength}"}}}

    fields = workflow_settings([], schema)
    settings = resolve_generation_settings(fields)
    dispatched = ComfyUIAdapter._compile(graph, settings)

    assert dispatched["sample"]["inputs"]["denoise"] == 0.5
