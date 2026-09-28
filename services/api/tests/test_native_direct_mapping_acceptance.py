from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_workflow_package_import_endpoint import (
    _object_info,
    _source_ui_graph,
    _ui_graph,
    _wire_runtime,
)

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.comfy_workflow_compiler import compile_comfyui_ui_graph
from local_lm.settings_registry import (
    IMAGE_SETTINGS,
    resolve_generation_settings,
    workflow_settings,
)
from local_lm.workflow_package_inputs import prepare_workflow_package_compilation


@pytest.mark.parametrize("target", ["create", "bundle", "revision"])
@pytest.mark.parametrize("graph_format", ["native", "api"])
async def test_online_native_workflow_writes_map_saved_controls(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    graph_format: str,
) -> None:
    info = _object_info()
    graph = _ui_graph()
    info["Source"]["input"]["required"]["steps"] = ["INT", {"min": 1, "max": 50}]
    info["Source"]["input_order"]["required"].append("steps")
    graph["nodes"][0]["widgets_values"].append(27)
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    compilation = compile_comfyui_ui_graph(graph, info)
    payload: dict[str, Any] = {
        "name": "Native saved controls",
        "operation": "text_to_image",
        "engine": "comfyui",
        "ui_graph": graph if graph_format == "native" else {},
        "api_graph": compilation.api_graph,
        "input_schema": {"type": "object", "properties": {}},
    }
    if target == "revision":
        original = await client.post("/api/workflows", json=payload)
        assert original.status_code == 201, original.text
        workflow_id = original.json()["id"]
        graph["nodes"][0]["widgets_values"][-1] = 31
        payload["api_graph"] = compile_comfyui_ui_graph(graph, info).api_graph
        path = f"/api/workflows/{workflow_id}/revisions"
        payload = {
            key: value
            for key, value in payload.items()
            if key in {"ui_graph", "api_graph", "input_schema"}
        }
    else:
        path = "/api/workflows/import" if target == "bundle" else "/api/workflows"
    response = await client.post(path, json=payload)
    assert response.status_code == 201, response.text
    revision = response.json() if target == "revision" else response.json()["revisions"][0]
    fields = workflow_settings(IMAGE_SETTINGS, revision["input_schema_json"])
    steps = next(field for field in fields if field.key == "steps")
    expected = 31 if target == "revision" else 27
    assert steps.default == expected, "The saved native control was not mapped by the write route"
    assert steps.minimum == 1 and steps.maximum == 50
    values = resolve_generation_settings(fields, turn_overrides={"steps": 45})
    dispatched = ComfyUIAdapter._compile(revision["api_graph_json"], values)
    assert dispatched["1"]["inputs"]["steps"] == 45


@pytest.mark.parametrize("path", ["/api/workflows", "/api/workflows/import"])
async def test_unmapped_native_writes_require_runtime_without_persisting(
    client: AsyncClient, path: str
) -> None:
    graph = _ui_graph()
    response = await client.post(
        path,
        json={
            "name": "Offline native mapping",
            "operation": "text_to_image",
            "engine": "comfyui",
            "ui_graph": graph,
            "api_graph": compile_comfyui_ui_graph(graph, _object_info()).api_graph,
        },
    )
    assert response.status_code == 503
    assert response.json()["code"] == "media-runtime-unavailable"
    assert all(
        item["name"] != "Offline native mapping"
        for item in (await client.get("/api/workflows")).json()
    )


async def test_mapped_workflows_remain_portable_offline_but_native_edits_need_mapping(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from copy import deepcopy

    graph = _ui_graph()
    _wire_runtime(app, monkeypatch)
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Portable mapped workflow",
            "operation": "text_to_image",
            "engine": "comfyui",
            "ui_graph": graph,
            "api_graph": compile_comfyui_ui_graph(graph, _object_info()).api_graph,
        },
    )
    assert response.status_code == 201, response.text
    workflow = response.json()
    revision = workflow["revisions"][0]
    monkeypatch.setattr(app.state.services.engines.media, "object_info", None)
    bundle = (await client.get(f"/api/workflows/{workflow['id']}/export")).json()
    imported = await client.post("/api/workflows/import", json=bundle)
    assert imported.status_code == 201, imported.text
    cloned = await client.post(f"/api/workflows/{workflow['id']}/clone", json={})
    assert cloned.status_code == 201, cloned.text
    payload = {
        "ui_graph": revision["ui_graph_json"],
        "api_graph": revision["api_graph_json"],
        "input_schema": revision["input_schema_json"],
    }
    unchanged = await client.post(f"/api/workflows/{workflow['id']}/revisions", json=payload)
    assert unchanged.status_code == 201, unchanged.text
    changed = deepcopy(payload)
    changed["ui_graph"]["nodes"][0]["widgets_values"][1] = 88
    refused = await client.post(f"/api/workflows/{workflow['id']}/revisions", json=changed)
    assert refused.status_code == 503
    current = next(
        item for item in (await client.get("/api/workflows")).json() if item["id"] == workflow["id"]
    )
    assert current["current_revision_id"] == unchanged.json()["id"]
    _wire_runtime(app, monkeypatch)
    remapped = await client.post(f"/api/workflows/{workflow['id']}/revisions", json=changed)
    assert remapped.status_code == 201, remapped.text
    assert remapped.json()["input_schema_json"]["properties"]["seed"]["default"] == 88
    assert revision["input_schema_json"]["properties"]["seed"]["default"] == 42


async def test_direct_native_write_preserves_the_declared_source_image_binding(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, info = _source_ui_graph(), _object_info()
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    prepared = prepare_workflow_package_compilation(graph, info, "image_to_image")
    compilation = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)
    supplied = prepared.bind(compilation.api_graph)
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Bound source image",
            "operation": "image_to_image",
            "engine": "comfyui",
            "ui_graph": graph,
            "api_graph": supplied,
            "input_schema": prepared.input_schema,
        },
    )
    assert response.status_code == 201, response.text
    revision = response.json()["revisions"][0]
    assert revision["api_graph_json"]["1"]["inputs"]["image"] == "${input_image}"
    assert (
        revision["input_schema_json"]["properties"]["input_image"]
        == prepared.input_schema["properties"]["input_image"]
    )


async def test_changed_api_revisions_rederive_controls_and_unchanged_ones_work_offline(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from copy import deepcopy

    info = _object_info()
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    created = await client.post(
        "/api/workflows",
        json={
            "name": "API revision controls",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": compile_comfyui_ui_graph(_ui_graph(), info).api_graph,
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()
    revision = workflow["revisions"][0]
    payload = {
        "api_graph": revision["api_graph_json"],
        "input_schema": revision["input_schema_json"],
    }
    path = f"/api/workflows/{workflow['id']}/revisions"
    monkeypatch.setattr(app.state.services.engines.media, "object_info", None)
    unchanged = await client.post(path, json=payload)
    assert unchanged.status_code == 201, unchanged.text
    changed = deepcopy(payload)
    changed["api_graph"]["1"]["inputs"]["steps"] = 25
    refused = await client.post(path, json=changed)
    assert refused.status_code == 503
    info["Source"]["input"]["required"]["steps"] = ["INT", {"min": 1, "max": 30}]
    info["Source"]["input_order"]["required"].append("steps")
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    remapped = await client.post(path, json=changed)
    assert remapped.status_code == 201, remapped.text
    result = remapped.json()
    assert result["input_schema_json"]["properties"]["steps"]["default"] == 25
    assert result["input_schema_json"]["properties"]["seed"]["default"] == 42
    assert "steps" not in revision["input_schema_json"]["properties"]


@pytest.mark.parametrize("online", [False, True])
async def test_executable_revision_changes_require_mapping_when_visual_graph_is_unchanged(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, online: bool
) -> None:
    from copy import deepcopy

    graph, info = _ui_graph(), _object_info()
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Consistent native revision",
            "operation": "text_to_image",
            "engine": "comfyui",
            "ui_graph": graph,
            "api_graph": compile_comfyui_ui_graph(graph, info).api_graph,
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()
    revision = workflow["revisions"][0]
    executable = deepcopy(revision["api_graph_json"])
    executable["1"]["inputs"]["steps"] = 25
    if not online:
        monkeypatch.setattr(app.state.services.engines.media, "object_info", None)
    changed = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={
            "ui_graph": revision["ui_graph_json"],
            "api_graph": executable,
            "input_schema": revision["input_schema_json"],
        },
    )
    assert changed.status_code == (422 if online else 503), changed.text
    current = next(
        item for item in (await client.get("/api/workflows")).json() if item["id"] == workflow["id"]
    )
    assert current["current_revision_id"] == revision["id"]


@pytest.mark.parametrize("invalid_inventory", [False, True])
async def test_direct_native_runtime_failure_persists_no_workflow(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, invalid_inventory: bool
) -> None:
    async def describe() -> object:
        if invalid_inventory:
            return []
        raise httpx.ConnectError("Unavailable neutral fixture")

    monkeypatch.setattr(app.state.services.engines.media, "object_info", describe, raising=False)
    graph = _ui_graph()
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Unavailable native inventory",
            "operation": "text_to_image",
            "engine": "comfyui",
            "ui_graph": graph,
            "api_graph": compile_comfyui_ui_graph(graph, _object_info()).api_graph,
        },
    )
    assert response.status_code == 503
    assert all(
        item["name"] != "Unavailable native inventory"
        for item in (await client.get("/api/workflows")).json()
    )
