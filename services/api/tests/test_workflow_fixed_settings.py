from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_comfy_workflow_compiler import _with_supported_frontend
from test_workflow_package_import_endpoint import _object_info, _ui_graph, _wire_runtime

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.settings_registry import (
    IMAGE_SETTINGS,
    resolve_generation_settings,
    workflow_settings,
)
from local_lm.workflow_graph_settings_v1 import workflow_graph_settings


async def test_named_grown_sockets_keep_their_template_identity_and_connection(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _object_info()
    info["Source"]["input"]["required"]["seeds"] = [
        "COMFY_AUTOGROW_V3",
        {
            "template": {
                "input": {"required": {"seed": ["INT"]}},
                "names": ["first", "second"],
                "min": 1,
            }
        },
    ]
    info["Source"]["input_order"]["required"].append("seeds")
    info["SeedValue"] = {
        "input": {"required": {"value": ["INT"]}},
        "input_order": {"required": ["value"]},
        "output": ["INT"],
    }
    graph = _ui_graph()
    graph["nodes"][0]["inputs"] = [
        {"name": "seeds.first", "type": "INT", "link": 8},
        {"name": "seeds.second", "type": "INT", "link": None},
    ]
    graph["nodes"].append(
        {
            "id": 3,
            "type": "SeedValue",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "INT", "type": "INT", "links": [8]}],
            "widgets_values": [77],
        }
    )
    graph["links"].append([8, 3, 0, 1, 0, "INT"])
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Named seed inputs", "operation": "text_to_image"},
    )
    assert response.status_code == 201, response.text
    (revision,) = response.json()["revisions"]
    schema = revision["input_schema_json"]
    marker = workflow_graph_settings(schema)
    assert marker is not None
    (fixed,) = marker["fixed"]
    assert fixed["input_name"] == "seeds.first"
    assert fixed["declared_name"] == "seed" and fixed["reason"] == "linked"
    fields = workflow_settings(IMAGE_SETTINGS, schema)
    dispatched = ComfyUIAdapter._compile(
        revision["api_graph_json"], resolve_generation_settings(fields)
    )
    assert dispatched["1"]["inputs"]["seeds.first"] == ["3", 0]
    assert "seeds.second" not in dispatched["1"]["inputs"]
    assert dispatched["3"]["inputs"]["value"] == 77


@pytest.mark.parametrize("origin", ["linked", "primitive"])
async def test_fixed_graph_inputs_are_explained_and_cannot_receive_settings_overrides(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    origin: str,
) -> None:
    info = _object_info()
    graph = _with_supported_frontend(_ui_graph())
    graph["nodes"][0]["inputs"] = [
        {"name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": 8}
    ]
    primitive = origin == "primitive"
    graph["nodes"].append(
        {
            "id": 3,
            "type": "PrimitiveNode" if primitive else "SeedValue",
            "mode": 0,
            "properties": {"Run widget replace on values": False} if primitive else {},
            "inputs": [],
            "outputs": [{"name": "INT", "type": "INT", "widget": {"name": "seed"}, "links": [8]}],
            "widgets_values": [77],
        }
    )
    graph["links"].append([8, 3, 0, 1, 0, "INT"])
    info["SeedValue"] = {
        "input": {"required": {"value": ["INT", {"default": 77}]}},
        "input_order": {"required": ["value"]},
        "output": ["INT"],
    }
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Graph supplied seed", "operation": "text_to_image"},
    )
    assert response.status_code == 201, response.text
    (revision,) = response.json()["revisions"]
    schema = revision["input_schema_json"]
    fields = workflow_settings(IMAGE_SETTINGS, schema)
    seed = next(field for field in fields if field.key == "seed")
    assert seed.available is False, "A graph-supplied seed still offers an ineffective control"
    assert seed.unavailable_reason
    assert schema["x-lm-atelier-graph-settings"]["fixed"][0]["reason"] == origin
    resolved = resolve_generation_settings(fields, profile_defaults=[{"seed": 999, "width": 2048}])
    assert "seed" not in resolved and "width" not in resolved
    with pytest.raises(ValueError):
        resolve_generation_settings(fields, turn_overrides={"seed": 999})
    dispatched = ComfyUIAdapter._compile(revision["api_graph_json"], resolved)
    assert dispatched["1"]["inputs"]["seed"] == (77 if primitive else ["3", 0])


@pytest.mark.parametrize("damage", ["reason", "duplicate", "binding-conflict", "input"])
def test_invalid_fixed_control_records_are_refused(damage: str) -> None:
    fixed = {"node_id": "1", "input_name": "seed", "label": "Seed", "reason": "linked"}
    marker: dict[str, Any] = {"version": 1, "bindings": [], "fixed": [fixed]}
    schema = {
        "type": "object",
        "properties": {"seed": {"type": "integer", "default": 42}},
        "x-lm-atelier-graph-settings": marker,
    }
    if damage == "reason":
        fixed["reason"] = "unknown"
    elif damage == "duplicate":
        marker["fixed"].append(dict(fixed))
    elif damage == "binding-conflict":
        marker["bindings"].append({"node_id": "1", "input_name": "seed", "parameter": "seed"})
    else:
        fixed["input_name"] = "unrecognized"
    with pytest.raises(ValueError, match="generated setting"):
        workflow_graph_settings(schema)
