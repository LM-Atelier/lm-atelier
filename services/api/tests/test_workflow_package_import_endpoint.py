"""The ready-package import hop: analyzer-gated, runtime-compiled, untrusted."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

pytestmark = pytest.mark.asyncio


def _object_info() -> dict[str, Any]:
    return {
        "Source": {
            "display_name": "Source image",
            "input": {
                "required": {
                    "label": ["STRING", {"default": "default"}],
                    "seed": ["INT", {"default": 1, "control_after_generate": True}],
                }
            },
            "input_order": {"required": ["label", "seed"]},
            "output": ["IMAGE"],
        },
        "Save": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output": [],
            "output_node": True,
        },
    }


def _ui_graph() -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "Source",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [7]}],
                "widgets_values": ["camera", 42, "randomize"],
            },
            {
                "id": 2,
                "type": "Save",
                "mode": 0,
                "inputs": [
                    {"name": "images", "type": "IMAGE", "link": 7},
                    {
                        "name": "filename_prefix",
                        "type": "STRING",
                        "widget": {"name": "filename_prefix"},
                    },
                ],
                "outputs": [],
                "widgets_values": ["result"],
            },
        ],
        "links": [[7, 1, 0, 2, 0, "IMAGE"]],
    }


def _wire_runtime(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    async def object_info() -> dict[str, Any]:
        return _object_info()

    monkeypatch.setattr(app.state.services.engines.media, "object_info", object_info, raising=False)


async def test_a_stopped_runtime_refuses_before_compiling(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_object_info() -> dict[str, Any]:
        raise RuntimeError("media runtime is not up")

    monkeypatch.setattr(
        app.state.services.engines.media, "object_info", failing_object_info, raising=False
    )

    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": _ui_graph(), "name": "Camera save", "operation": "text_to_image"},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "media-runtime-unavailable"


async def test_an_unresolved_package_refuses_with_the_analyzer_verdict(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_runtime(app, monkeypatch)
    graph = _ui_graph()
    graph["nodes"][0]["type"] = "NotInstalledNode"
    graph["nodes"][0]["properties"] = {"cnr_id": "missing-pack", "version": "1.0.0"}

    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Unresolved", "operation": "text_to_image"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "package-not-resolved"
    listed = (await client.get("/api/workflows")).json()
    assert all(workflow["name"] != "Unresolved" for workflow in listed)


async def test_a_ready_package_imports_as_an_untrusted_workflow(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_runtime(app, monkeypatch)

    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": _ui_graph(),
            "name": "Camera save",
            "operation": "image_to_image",
            "description": "From a shared export",
        },
    )

    assert response.status_code == 201
    workflow = response.json()
    assert workflow["name"] == "Camera save"
    assert workflow["operation"] == "image_to_image"
    (revision,) = workflow["revisions"]
    # Imported code is never trusted by arriving; review grants that later.
    assert revision["trusted"] is False
    assert revision["ui_graph_json"]["nodes"][0]["type"] == "Source"
    compiled = revision["api_graph_json"]
    assert compiled["1"]["class_type"] == "Source"
    assert compiled["2"]["inputs"]["images"] == ["1", 0]


async def test_compilation_refusals_keep_their_stable_codes(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_runtime(app, monkeypatch)
    graph = _ui_graph()
    # A subgraph the compiler cannot translate without the ComfyUI frontend.
    # Its inner node types are all available, so the package is still ready -
    # the refusal must come from compilation, with compilation's code.
    subgraph_id = "9bc44576-7290-4701-bda4-032ca796efbc"
    graph["nodes"].append({"id": 3, "type": subgraph_id, "mode": 0, "inputs": [], "outputs": []})
    graph["definitions"] = {
        "subgraphs": [
            {
                "id": subgraph_id,
                "nodes": [{"id": 10, "type": "Source", "mode": 0, "inputs": [], "outputs": []}],
                "links": [],
            }
        ]
    }

    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Subgraphs", "operation": "text_to_image"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "unsupported_subgraphs"
