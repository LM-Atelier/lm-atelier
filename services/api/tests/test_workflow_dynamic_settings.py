from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_workflow_package_import_endpoint import _object_info, _ui_graph, _wire_runtime

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.settings_registry import (
    IMAGE_SETTINGS,
    resolve_generation_settings,
    workflow_settings,
)


@pytest.mark.parametrize("nested", [False, True])
async def test_dynamic_controls_keep_native_choices_and_selected_child_constraints(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
) -> None:
    info = _object_info()
    graph = _ui_graph()
    name = "mode" if nested else "sampler"
    options: list[dict[str, Any]] = [
        {"key": "euler"},
        {"key": "heun", "inputs": {"required": {"steps": ["INT", {"min": 2, "max": 20}]}}}
        if nested
        else {"key": "heun"},
    ]
    info["Source"]["input"]["required"][name] = ["COMFY_DYNAMICCOMBO_V3", {"options": options}]
    info["Source"]["input_order"]["required"].append(name)
    graph["nodes"][0]["widgets_values"].extend(["heun", 8] if nested else ["heun"])
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": graph,
            "name": "Dynamic native controls",
            "operation": "text_to_image",
        },
    )
    assert response.status_code == 201, response.text
    (revision,) = response.json()["revisions"]
    key = "steps" if nested else "sampler"
    fields = workflow_settings(IMAGE_SETTINGS, revision["input_schema_json"])
    field = next(field for field in fields if field.key == key)
    assert field.available, "The selected dynamic widget has no generation control"
    if nested:
        assert (field.default, field.minimum, field.maximum) == (8, 2, 20)
    else:
        assert field.default == "heun" and field.choices == ["euler", "heun"]
    value = 12 if nested else "euler"
    resolved = resolve_generation_settings(fields, turn_overrides={key: value})
    dispatched = ComfyUIAdapter._compile(revision["api_graph_json"], resolved)
    assert dispatched["1"]["inputs"]["mode.steps" if nested else "sampler"] == value
    if nested:
        assert dispatched["1"]["inputs"]["mode"] == "heun"


async def test_equal_node_titles_still_offer_independent_sampling_stages(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = _object_info()
    info["Source"]["input"]["required"]["steps"] = ["INT", {"min": 1, "max": 20}]
    info["Source"]["input_order"]["required"].append("steps")
    info["Refine"] = deepcopy(info["Source"])
    info["Refine"]["input"]["required"]["images"] = ["IMAGE"]
    info["Refine"]["input_order"]["required"].insert(0, "images")
    graph = _ui_graph()
    graph["nodes"][0]["title"] = "Sampler"
    graph["nodes"][0]["widgets_values"].append(8)
    graph["nodes"][0]["outputs"][0]["links"] = [8]
    graph["nodes"].append(
        {
            "id": 3,
            "type": "Refine",
            "title": "Sampler",
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": 8}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [7]}],
            "widgets_values": ["detail", 99, "randomize", 12],
        }
    )
    graph["links"] = [[7, 3, 0, 2, 0, "IMAGE"], [8, 1, 0, 3, 0, "IMAGE"]]
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": graph,
            "name": "Two sampling stages",
            "operation": "text_to_image",
        },
    )
    assert response.status_code == 201, response.text
    (revision,) = response.json()["revisions"]
    fields = workflow_settings(IMAGE_SETTINGS, revision["input_schema_json"])
    graph = revision["api_graph_json"]
    first = graph["1"]["inputs"]["steps"][2:-1]
    second = graph["3"]["inputs"]["steps"][2:-1]
    assert first != second
    offered = [field for field in fields if field.key in {first, second}]
    assert len({field.label for field in offered}) == 2, (
        "Sampling stages have indistinguishable labels"
    )
    resolved = resolve_generation_settings(fields, turn_overrides={first: 11})
    result = ComfyUIAdapter._compile(graph, resolved)
    assert (result["1"]["inputs"]["steps"], result["3"]["inputs"]["steps"]) == (11, 12)
    assert (result["1"]["inputs"]["seed"], result["3"]["inputs"]["seed"]) == (42, 99)
