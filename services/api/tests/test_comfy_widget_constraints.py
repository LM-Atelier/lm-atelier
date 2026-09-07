from __future__ import annotations

from typing import Any

import pytest

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.comfy_templates import _compile_ui_graph
from local_lm.settings_registry import (
    IMAGE_SETTINGS,
    VIDEO_SETTINGS,
    resolve_generation_settings,
    workflow_settings,
)


def _compile(specs: dict[str, Any], values: list[Any], *, operation: str = "text_to_video"):
    node = {
        "id": 1,
        "type": "NeutralSampler",
        "inputs": [{"name": name, "type": "COMBO", "widget": {"name": name}} for name in specs],
        "widgets_values": values,
    }
    return _compile_ui_graph(
        {"nodes": [node], "links": []},
        {"NeutralSampler": {"input": {"required": specs}}},
        operation=operation,
    )


@pytest.mark.parametrize(
    ("spec", "default", "choices", "invalid"),
    [
        ([["vp9", "av1"]], "vp9", ["vp9", "av1"], "h264"),
        (["COMBO", {"options": ["vp9", "av1"]}], "vp9", ["vp9", "av1"], "h264"),
        (
            ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "auto"}, {"key": "h264"}]}],
            "auto",
            ["auto", "h264"],
            "vp9",
        ),
    ],
)
def test_bound_codec_uses_the_nodes_choices_and_default(
    spec: Any, default: str, choices: list[str], invalid: str
) -> None:
    graph, schema = _compile({"codec": spec}, [default])
    assert schema["properties"]["codec"]["enum"] == choices
    fields = workflow_settings(VIDEO_SETTINGS, schema)
    field = next(field for field in fields if field.key == "codec")
    assert field.type == "enum"
    assert field.choices == choices
    resolved = resolve_generation_settings(fields)
    assert resolved["codec"] == default
    assert ComfyUIAdapter._compile(graph, resolved)["1"]["inputs"]["codec"] == default
    for choice in choices:
        assert (
            resolve_generation_settings(fields, turn_overrides={"codec": choice})["codec"] == choice
        )
    with pytest.raises(ValueError):
        resolve_generation_settings(fields, turn_overrides={"codec": invalid})


@pytest.mark.parametrize("operation", ["text_to_image", "text_to_video"])
def test_sampler_and_scheduler_offer_declared_options_and_ignore_obsolete_saved_values(
    operation: str,
) -> None:
    graph, schema = _compile(
        {
            "sampler_name": [["euler", "heun"]],
            "scheduler": ["COMBO", {"options": ["normal", "simple"]}],
        },
        ["euler", "normal"],
        operation=operation,
    )
    fields = workflow_settings(
        IMAGE_SETTINGS if operation == "text_to_image" else VIDEO_SETTINGS, schema
    )
    sampler = next(field for field in fields if field.key == "sampler")
    assert sampler.type == "enum"
    assert sampler.choices == ["euler", "heun"]
    resolved = resolve_generation_settings(
        fields,
        profile_defaults=[{"sampler": "obsolete", "scheduler": "removed"}],
        turn_overrides={"scheduler": "simple"},
    )
    assert resolved["sampler"] == "euler"
    assert ComfyUIAdapter._compile(graph, resolved)["1"]["inputs"]["scheduler"] == "simple"
    with pytest.raises(ValueError):
        resolve_generation_settings(fields, turn_overrides={"sampler": "not_a_sampler"})


@pytest.mark.parametrize("operation", ["text_to_image", "text_to_video"])
def test_bound_numeric_widgets_keep_declared_limits_and_steps(operation: str) -> None:
    _, schema = _compile(
        {"steps": ["INT", {"min": 2, "max": 10, "step": 2}]},
        [4],
        operation=operation,
    )
    assert schema["properties"]["steps"] == {
        "type": "integer",
        "default": 4,
        "minimum": 2,
        "maximum": 10,
        "multipleOf": 2,
    }
    fields = workflow_settings(
        IMAGE_SETTINGS if operation == "text_to_image" else VIDEO_SETTINGS, schema
    )
    step = next(field for field in fields if field.key == "steps")
    assert (step.minimum, step.maximum, step.step) == (2, 10, 2)
    assert resolve_generation_settings(fields, profile_defaults=[{"steps": 50}])["steps"] == 4
    for value in (1, 3, 50):
        with pytest.raises(ValueError):
            resolve_generation_settings(fields, turn_overrides={"steps": value})


def test_bound_numeric_widgets_intersect_existing_product_limits() -> None:
    _, schema = _compile(
        {
            "cfg": ["FLOAT", {"min": 0, "max": 100, "step": 0.5}],
            "seed": ["INT", {"min": 0, "max": 18446744073709551615, "step": 1}],
            "width": ["INT", {"min": 16, "max": 8192, "step": 8}],
        },
        [8, 42, 512],
    )
    fields = workflow_settings(VIDEO_SETTINGS, schema)
    by_key = {field.key: field for field in fields}
    assert by_key["cfg"].maximum == 30
    assert by_key["cfg"].step == 0.5
    assert by_key["width"].minimum == 128
    assert by_key["width"].maximum == 4096
    assert by_key["seed"].minimum == -1
    assert resolve_generation_settings(fields)["seed"] == -1
    with pytest.raises(ValueError):
        resolve_generation_settings(fields, turn_overrides={"cfg": 8.25})


@pytest.mark.parametrize("operation", ["text_to_image", "text_to_video"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_cfg_and_guidance_preserve_separate_node_values(
    operation: str, reverse: bool
) -> None:
    nodes = [
        {"id": 1, "type": "KSampler", "inputs": [{"name": "cfg"}], "widgets_values": [8.0]},
        {
            "id": 2,
            "type": "FluxGuidance",
            "inputs": [{"name": "guidance"}],
            "widgets_values": [3.5],
        },
    ]
    if reverse:
        nodes.reverse()
    graph, schema = _compile_ui_graph(
        {"nodes": nodes, "links": []},
        {
            "KSampler": {"input": {"required": {"cfg": ["FLOAT", {"default": 8.0}]}}},
            "FluxGuidance": {"input": {"required": {"guidance": ["FLOAT", {"default": 3.5}]}}},
        },
        operation=operation,
    )
    assert schema["properties"]["cfg"]["default"] == 8.0
    fields = workflow_settings(
        IMAGE_SETTINGS if operation == "text_to_image" else VIDEO_SETTINGS, schema
    )
    resolved = resolve_generation_settings(fields, turn_overrides={"cfg": 9.0})
    dispatched = ComfyUIAdapter._compile(graph, resolved)
    assert dispatched["1"]["inputs"]["cfg"] == 9.0
    assert dispatched["2"]["inputs"]["guidance"] == 3.5


def test_dynamic_combo_uses_a_scalar_default_when_no_saved_widget_exists() -> None:
    graph, schema = _compile(
        {"codec": ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "auto"}, {"key": "h264"}]}]}, []
    )
    fields = workflow_settings(VIDEO_SETTINGS, schema)
    resolved = resolve_generation_settings(fields)
    assert ComfyUIAdapter._compile(graph, resolved)["1"]["inputs"]["codec"] == "auto"


@pytest.mark.parametrize(
    "choices", [[{"label": "unsupported"}], [], [{"key": {"nested": "value"}}]]
)
def test_bound_combo_declines_unsupported_option_shapes(choices: Any) -> None:
    with pytest.raises(ValueError):
        _compile({"codec": ["COMFY_DYNAMICCOMBO_V3", {"options": choices}]}, ["auto"])


def test_empty_unbound_model_choices_preserve_the_saved_widget() -> None:
    graph, _ = _compile({"ckpt_name": [[]]}, ["neutral-model.safetensors"])
    assert graph["1"]["inputs"]["ckpt_name"] == "neutral-model.safetensors"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("name", "specs", "saved", "expected"),
    [
        ("codec", [[["auto", "h264"]], [["h264", "h265"]]], ["auto", "h264"], "h264"),
        ("steps", [["INT", {"min": 2, "max": 10}], ["INT", {"min": 6, "max": 12}]], [4, 8], 8),
        ("steps", [["INT", {"min": 2, "max": 10}], ["INT", {"min": 6, "max": 12}]], [8, 6], 6),
        (
            "steps",
            [["INT", {"min": 2, "max": 10, "step": 2}], ["INT", {"min": 6, "max": 12, "step": 2}]],
            [4, 12],
            10,
        ),
    ],
)
def test_shared_widget_defaults_resolve_inside_the_final_intersection(
    reverse: bool, name: str, specs: list[Any], saved: list[Any], expected: Any
) -> None:
    nodes = [
        {
            "id": index + 1,
            "type": f"Neutral{index}",
            "inputs": [{"name": name}],
            "widgets_values": [value],
        }
        for index, value in enumerate(saved)
    ]
    info = {
        f"Neutral{index}": {"input": {"required": {name: spec}}} for index, spec in enumerate(specs)
    }
    if reverse:
        nodes.reverse()
    graph, schema = _compile_ui_graph(
        {"nodes": nodes, "links": []}, info, operation="text_to_video"
    )
    fields = workflow_settings(VIDEO_SETTINGS, schema)
    resolved = resolve_generation_settings(fields)
    assert resolved[name] == expected
    dispatched = ComfyUIAdapter._compile(graph, resolved)
    assert [dispatched[str(index)]["inputs"][name] for index in (1, 2)] == [expected, expected]
    with pytest.raises(ValueError):
        resolve_generation_settings(
            fields, turn_overrides={name: "unsupported" if name == "codec" else 4}
        )


@pytest.mark.parametrize("operation", ["text_to_image", "text_to_video"])
@pytest.mark.parametrize("with_pixels", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_fractional_dimensions_remain_literal_without_poisoning_pixel_settings(
    operation: str, with_pixels: bool, reverse: bool
) -> None:
    nodes = [
        {
            "id": 1,
            "type": "ConditioningSetAreaPercentage",
            "inputs": [{"name": "width"}, {"name": "height"}],
            "widgets_values": [0.5, 0.75],
        }
    ]
    info = {
        "ConditioningSetAreaPercentage": {
            "input": {
                "required": {
                    name: ["FLOAT", {"min": 0, "max": 1, "step": 0.01}]
                    for name in ("width", "height")
                }
            }
        }
    }
    if with_pixels:
        nodes.append(
            {
                "id": 2,
                "type": "EmptyLatentImage",
                "inputs": [{"name": "width"}, {"name": "height"}],
                "widgets_values": [512, 768],
            }
        )
        info["EmptyLatentImage"] = {
            "input": {
                "required": {
                    name: ["INT", {"min": 16, "max": 8192, "step": 8}]
                    for name in ("width", "height")
                }
            }
        }
    if reverse:
        nodes.reverse()
    graph, schema = _compile_ui_graph({"nodes": nodes, "links": []}, info, operation=operation)
    fields = workflow_settings(
        IMAGE_SETTINGS if operation == "text_to_image" else VIDEO_SETTINGS, schema
    )
    resolved = resolve_generation_settings(
        fields, turn_overrides={"width": 1024} if with_pixels else {}
    )
    dispatched = ComfyUIAdapter._compile(graph, resolved)
    assert dispatched["1"]["inputs"] == {"width": 0.5, "height": 0.75}
    if with_pixels:
        assert dispatched["2"]["inputs"] == {"width": 1024, "height": 768}
    else:
        assert all(field.key not in {"width", "height"} for field in fields)
