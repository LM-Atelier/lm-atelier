from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_workflow_package_import_endpoint import _object_info, _ui_graph, _wire_runtime

from local_lm.settings_registry import IMAGE_SETTINGS, workflow_settings


@pytest.mark.parametrize(
    ("name", "choices", "saved", "selected", "kind", "offered"),
    [
        ("steps", [1, 20, 30], 20, 30, "integer", [1, 20, 30]),
        ("cfg", [1, 2.5, 7.5], 2.5, 7.5, "number", [1, 2.5, 7.5]),
        ("cfg", [-1, 1, 2.5, 40], 1, 2.5, "number", [1, 2.5]),
        ("width", [16, 768, 1024, 8192], 768, 1024, "integer", [768, 1024]),
    ],
)
async def test_numeric_native_choices_reach_the_graph_as_numbers(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    choices: list[int | float],
    saved: int | float,
    selected: int | float,
    kind: str,
    offered: list[int | float],
) -> None:
    from local_lm.adapters.comfyui import ComfyUIAdapter
    from local_lm.settings_registry import resolve_generation_settings

    info, graph = _object_info(), _ui_graph()
    info["Source"]["input"]["required"][name] = [choices]
    info["Source"]["input_order"]["required"].append(name)
    graph["nodes"][0]["widgets_values"].append(saved)
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Numeric choices", "operation": "text_to_image"},
    )
    assert response.status_code == 201, response.text
    (revision,) = response.json()["revisions"]
    schema = revision["input_schema_json"]["properties"]
    assert name in schema, "Import left a native numeric choice fixed in the graph"
    assert schema[name]["type"] == kind
    assert schema[name]["enum"] == offered
    fields = workflow_settings(IMAGE_SETTINGS, revision["input_schema_json"])
    field = next(field for field in fields if field.key == name)
    assert field.type == "enum" and field.default == saved and field.choices == offered
    values = resolve_generation_settings(fields, turn_overrides={name: selected})
    dispatched = ComfyUIAdapter._compile(revision["api_graph_json"], values)["1"]["inputs"][name]
    assert type(dispatched) is type(selected) and dispatched == selected
    with pytest.raises(ValueError):
        resolve_generation_settings(fields, turn_overrides={name: str(selected)})
    with pytest.raises(ValueError):
        resolve_generation_settings(fields, turn_overrides={name: 19})
    with pytest.raises(ValueError):
        resolve_generation_settings(fields, turn_overrides={name: True})
    for excluded in set(choices) - set(offered):
        with pytest.raises(ValueError):
            resolve_generation_settings(fields, turn_overrides={name: excluded})


async def test_native_geometry_controls_keep_their_choices_and_runtime_inputs(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm.adapters.comfyui import ComfyUIAdapter
    from local_lm.settings_registry import resolve_generation_settings

    info = _object_info()
    graph = _ui_graph()
    controls = {
        "aspect_ratio": ([["1:1", "4:3", "3:4"]], "4:3"),
        "megapixel_budget": (["FLOAT", {"min": 0.25, "max": 4, "step": 0.25}], 1.25),
        "round_to_multiple": (["INT", {"min": 32, "max": 128, "step": 32}], 64),
    }
    for name, (spec, value) in controls.items():
        info["Source"]["input"]["required"][name] = spec
        info["Source"]["input_order"]["required"].append(name)
        graph["nodes"][0]["widgets_values"].append(value)
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Native geometry", "operation": "text_to_image"},
    )
    assert response.status_code == 201, response.text
    (revision,) = response.json()["revisions"]
    fields = workflow_settings(IMAGE_SETTINGS, revision["input_schema_json"])
    geometry = {
        field.key: field
        for field in fields
        if field.key in {"aspect_ratio", "megapixels", "round_to_multiple"}
    }
    assert len(geometry) == 3
    assert all(field.visibility == "basic" for field in geometry.values()), (
        "Native geometry controls are hidden behind advanced settings"
    )
    assert geometry["aspect_ratio"].choices == ["1:1", "4:3", "3:4"]
    values = resolve_generation_settings(
        fields,
        turn_overrides={
            "aspect_ratio": "3:4",
            "megapixels": 2.25,
            "round_to_multiple": 128,
        },
    )
    inputs = ComfyUIAdapter._compile(revision["api_graph_json"], values)["1"]["inputs"]
    assert inputs["aspect_ratio"] == "3:4"
    assert inputs["megapixel_budget"] == 2.25 and inputs["round_to_multiple"] == 128
    assert "width" not in inputs and "height" not in inputs
    with pytest.raises(ValueError):
        resolve_generation_settings(fields, turn_overrides={"aspect_ratio": "16:9"})


@pytest.mark.parametrize(
    ("name", "spec", "value"),
    [
        ("width", ["INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}], 768),
        ("aspect_ratio", [["1:1", "4:3", "3:4"]], "4:3"),
    ],
)
async def test_import_exposes_a_native_size_control_with_its_saved_value_and_constraints(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    spec: list[Any],
    value: int | str,
) -> None:
    info = _object_info()
    info["Source"]["input"]["required"][name] = spec
    info["Source"]["input_order"]["required"].append(name)
    graph = _ui_graph()
    graph["nodes"][0]["widgets_values"].append(value)
    original = deepcopy(graph)
    _wire_runtime(app, monkeypatch, runtime_object_info=info)

    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Native size controls", "operation": "text_to_image"},
    )

    assert response.status_code == 201, response.text
    (revision,) = response.json()["revisions"]
    assert revision["trusted"] is False
    assert revision["ui_graph_json"] == original
    inputs = revision["api_graph_json"]["1"]["inputs"]
    assert inputs["label"] == "camera"
    assert revision["api_graph_json"]["2"]["inputs"]["images"] == ["1", 0]
    assert revision["api_graph_json"]["2"]["inputs"]["filename_prefix"] == "result"
    binding = inputs[name]
    assert isinstance(binding, str) and binding.startswith("${") and binding.endswith("}"), (
        "Import left a native size control as a fixed graph value"
    )
    parameter = binding[2:-1]
    schema = revision["input_schema_json"]["properties"][parameter]
    assert schema["default"] == value
    fields = workflow_settings(IMAGE_SETTINGS, revision["input_schema_json"])
    field = next(field for field in fields if field.key == parameter)
    assert field.available and field.default == value
    if name == "width":
        assert schema["type"] == "integer"
        assert schema["minimum"] == 64 and schema["maximum"] == 4096
        assert field.minimum == 64 and field.maximum == 4096
    else:
        assert schema["enum"] == ["1:1", "4:3", "3:4"]
        assert field.choices == ["1:1", "4:3", "3:4"]


@pytest.mark.parametrize(
    ("name", "spec", "value", "minimum", "maximum", "step"),
    [
        ("seed", ["INT", {"min": 0, "max": 18446744073709551615, "step": 1}], 42, 0, 2147483647, 1),
        ("width", ["INT", {"min": 16, "max": 8192, "step": 8}], 770, 64, 4096, 8),
        ("cfg", ["FLOAT", {"min": 0, "max": 100, "step": 0.5}], 8.25, 0, 30, 0.5),
    ],
)
async def test_import_intersects_limits_without_rounding_the_saved_widget_value(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    spec: list[Any],
    value: int | float,
    minimum: int,
    maximum: int,
    step: float,
) -> None:
    from local_lm.adapters.comfyui import ComfyUIAdapter
    from local_lm.settings_registry import resolve_generation_settings

    info = _object_info()
    graph = _ui_graph()
    if name == "seed":
        info["Source"]["input"]["required"][name] = spec
        graph["nodes"][0]["widgets_values"] = ["camera", value]
    else:
        info["Source"]["input"]["required"][name] = spec
        info["Source"]["input_order"]["required"].append(name)
        graph["nodes"][0]["widgets_values"].append(value)
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Native control ranges", "operation": "text_to_image"},
    )
    assert response.status_code == 201, response.text
    (revision,) = response.json()["revisions"]
    fields = workflow_settings(IMAGE_SETTINGS, revision["input_schema_json"])
    field = next(field for field in fields if field.key == name)
    assert (field.minimum, field.maximum, field.step) == (minimum, maximum, step)
    assert field.multiple_of is None
    resolved = resolve_generation_settings(fields)
    assert resolved[name] == value
    assert (
        ComfyUIAdapter._compile(revision["api_graph_json"], resolved)["1"]["inputs"][name] == value
    )


async def test_import_refuses_an_unsupported_saved_default_without_replacing_it(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = _object_info()
    info["Source"]["input"]["required"]["seed"] = ["INT", {"min": 0, "max": 18446744073709551615}]
    graph = _ui_graph()
    graph["nodes"][0]["widgets_values"] = ["camera", 4294967296]
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": graph,
            "name": "Unsupported native default",
            "operation": "text_to_image",
        },
    )
    assert response.status_code == 422, (
        "Import accepted a default that cannot be dispatched unchanged"
    )
    assert response.json()["code"] == "workflow-setting-default-unsupported"


async def test_video_import_preserves_an_integer_frame_rate_control(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm.settings_registry import VIDEO_SETTINGS, resolve_generation_settings

    info = _object_info()
    info["Source"]["input"]["required"]["frame_rate"] = ["INT", {"min": 1, "max": 240, "step": 1}]
    info["Source"]["input_order"]["required"].append("frame_rate")
    graph = _ui_graph()
    graph["nodes"][0]["widgets_values"].append(24)
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Integer frame rate", "operation": "text_to_video"},
    )
    assert response.status_code == 201, response.text
    (revision,) = response.json()["revisions"]
    fields = workflow_settings(VIDEO_SETTINGS, revision["input_schema_json"])
    fps = next(field for field in fields if field.key == "fps")
    assert (fps.type, fps.default, fps.maximum) == ("integer", 24, 120)
    assert resolve_generation_settings(fields)["fps"] == 24
    with pytest.raises(ValueError, match="integer"):
        resolve_generation_settings(fields, turn_overrides={"fps": 23.5})


@pytest.mark.parametrize(
    ("name", "setting", "spec", "saved", "selected", "invalid"),
    [
        ("sampler_name", "sampler", [["euler", "heun"]], "euler", "heun", "unlisted"),
        ("scheduler", "scheduler", [["normal", "simple"]], "normal", "simple", "unlisted"),
        ("denoise", "denoise", ["FLOAT", {"min": 0.2, "max": 0.8, "step": 0.05}], 0.4, 0.7, 0.9),
        ("batch_size", "batch_size", ["INT", {"min": 1, "max": 4}], 2, 3, 5),
    ],
)
async def test_imported_sampling_strength_and_batch_controls_reach_their_native_inputs(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    setting: str,
    spec: list[Any],
    saved: str | int | float,
    selected: str | int | float,
    invalid: str | int | float,
) -> None:
    from local_lm.adapters.comfyui import ComfyUIAdapter
    from local_lm.settings_registry import resolve_generation_settings

    info, graph = _object_info(), _ui_graph()
    info["Source"]["input"]["required"][name] = spec
    info["Source"]["input_order"]["required"].append(name)
    graph["nodes"][0]["widgets_values"].append(saved)
    original = deepcopy(graph)
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": graph,
            "name": "Native generation controls",
            "operation": "text_to_image",
        },
    )
    assert response.status_code == 201, response.text
    (revision,) = response.json()["revisions"]
    schema = revision["input_schema_json"]
    assert schema["properties"][setting]["default"] == saved
    fields = workflow_settings(IMAGE_SETTINGS, schema)
    field = next(field for field in fields if field.key == setting)
    assert field.available and field.default == saved
    values = resolve_generation_settings(fields, turn_overrides={setting: selected})
    dispatched = ComfyUIAdapter._compile(revision["api_graph_json"], values)
    assert dispatched["1"]["inputs"][name] == selected
    assert dispatched["2"]["inputs"]["images"] == ["1", 0]
    assert revision["ui_graph_json"] == original
    with pytest.raises(ValueError):
        resolve_generation_settings(fields, turn_overrides={setting: invalid})
