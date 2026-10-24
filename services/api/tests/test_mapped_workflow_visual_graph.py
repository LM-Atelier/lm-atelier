"""A visual graph is kept only when it compiles to the graph that runs.

Saved settings let an unchanged workflow skip mapping, but the visual graph is
still read on its own: the native editor opens it and edit classification takes
its node output types from it. Nothing but compilation ties it to the
executable graph.
"""

from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_workflow_package_import_endpoint import _object_info, _wire_runtime

from local_lm.comfy_workflow_compiler import compile_comfyui_ui_graph
from local_lm.image_edit_kind import image_edit_kind
from local_lm.workflow_package_inputs import prepare_workflow_package_compilation


def _edit_object_info() -> dict[str, Any]:
    info = _object_info()
    info.update(
        {
            "LatentEncoder": {
                "input": {"required": {"pixels": ["IMAGE"]}},
                "input_order": {"required": ["pixels"]},
                "output": ["LATENT"],
            },
            "WordsEncoder": {
                "input": {"required": {"text": ["STRING", {"multiline": True}]}},
                "input_order": {"required": ["text"]},
                "output": ["CONDITIONING"],
            },
            "Sampler": {
                "input": {
                    "required": {
                        "positive": ["CONDITIONING"],
                        "latent_image": ["LATENT"],
                        "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}],
                    }
                },
                "input_order": {"required": ["positive", "latent_image", "denoise"]},
                "output": ["LATENT"],
            },
            "LatentDecoder": {
                "input": {"required": {"samples": ["LATENT"]}},
                "input_order": {"required": ["samples"]},
                "output": ["IMAGE"],
            },
            "SaveImage": {
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
    )
    return info


def _source_node() -> dict[str, Any]:
    return {
        "id": 1,
        "type": "LoadImage",
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [11]},
            {"name": "MASK", "type": "MASK", "links": []},
        ],
        "widgets_values": ["author-source.png", "image"],
    }


def _strength_edit_ui_graph() -> dict[str, Any]:
    """The source becomes the starting latent; the words only condition a redraw."""

    return {
        "version": 0.4,
        "nodes": [
            _source_node(),
            {
                "id": 2,
                "type": "LatentEncoder",
                "mode": 0,
                "inputs": [{"name": "pixels", "type": "IMAGE", "link": 11}],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [12]}],
                "widgets_values": [],
            },
            {
                "id": 3,
                "type": "WordsEncoder",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [13]}],
                "widgets_values": ["${prompt}"],
            },
            {
                "id": 4,
                "type": "Sampler",
                "mode": 0,
                "inputs": [
                    {"name": "positive", "type": "CONDITIONING", "link": 13},
                    {"name": "latent_image", "type": "LATENT", "link": 12},
                ],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [14]}],
                "widgets_values": [0.6],
            },
            {
                "id": 5,
                "type": "LatentDecoder",
                "mode": 0,
                "inputs": [{"name": "samples", "type": "LATENT", "link": 14}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [15]}],
                "widgets_values": [],
            },
            {
                "id": 6,
                "type": "SaveImage",
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 15}],
                "outputs": [],
                "widgets_values": ["edited"],
            },
        ],
        "links": [
            [11, 1, 0, 2, 0, "IMAGE"],
            [12, 2, 0, 4, 1, "LATENT"],
            [13, 3, 0, 4, 0, "CONDITIONING"],
            [14, 4, 0, 5, 0, "LATENT"],
            [15, 5, 0, 6, 0, "IMAGE"],
        ],
    }


def _stub_ui_graph() -> dict[str, Any]:
    """Describe only the source and its encoder, and call the encoder's output conditioning."""

    encoder = deepcopy(_strength_edit_ui_graph()["nodes"][1])
    encoder["outputs"] = [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}]
    return {
        "version": 0.4,
        "nodes": [_source_node(), encoder],
        "links": [[11, 1, 0, 2, 0, "IMAGE"]],
    }


async def _exported_strength_edit(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    graph, info = _strength_edit_ui_graph(), _edit_object_info()
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    prepared = prepare_workflow_package_compilation(graph, info, "image_to_image")
    compilation = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Mapped strength edit",
            "operation": "image_to_image",
            "engine": "comfyui",
            "ui_graph": graph,
            "api_graph": prepared.bind(compilation.api_graph),
            "input_schema": prepared.input_schema,
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()
    bundle: dict[str, Any] = (await client.get(f"/api/workflows/{workflow['id']}/export")).json()
    assert "x-lm-atelier-graph-settings" in bundle["input_schema"]
    return bundle


@pytest.mark.parametrize("path", ["/api/workflows", "/api/workflows/import"])
@pytest.mark.parametrize("forgery", ["stub", "altered"])
async def test_a_mapped_workflow_whose_visual_graph_does_not_compile_to_it_is_refused(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    forgery: str,
) -> None:
    bundle = await _exported_strength_edit(client, app, monkeypatch)
    operation, api_graph, schema = "image_to_image", bundle["api_graph"], bundle["input_schema"]
    assert image_edit_kind(operation, bundle["ui_graph"], api_graph, schema) == "strength"
    forged = deepcopy(bundle)
    forged["name"] = "Forged visual graph"
    if forgery == "stub":
        forged["ui_graph"] = _stub_ui_graph()
        # The forgery matters: kept, it would make Auto treat this as an instruction edit.
        assert image_edit_kind(operation, forged["ui_graph"], api_graph, schema) == "instruction"
    else:
        forged["ui_graph"]["nodes"][5]["widgets_values"] = ["elsewhere"]
    if path == "/api/workflows":
        for bundle_only in ("format", "version", "trusted", "source_revision"):
            forged.pop(bundle_only)
    response = await client.post(path, json=forged)
    assert response.status_code == 422, (
        "A visual graph that does not compile to its workflow was kept"
    )
    assert response.json()["code"] == "workflow-invalid", response.text
    assert all(
        item["name"] != "Forged visual graph"
        for item in (await client.get("/api/workflows")).json()
    )


async def test_an_exported_visual_workflow_imports_with_its_graphs_and_saved_settings(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _exported_strength_edit(client, app, monkeypatch)
    listed = (await client.get("/api/workflows")).json()
    workflow = next(item for item in listed if item["name"] == "Mapped strength edit")
    revision = workflow["revisions"][0]
    # A saved value that is no longer the visual graph's own is kept as saved, not derived again.
    schema = deepcopy(revision["input_schema_json"])
    assert schema["properties"]["denoise"]["default"] == 0.6
    schema["properties"]["denoise"]["default"] = 0.35
    saved = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={
            "ui_graph": revision["ui_graph_json"],
            "api_graph": revision["api_graph_json"],
            "input_schema": schema,
        },
    )
    assert saved.status_code == 201, saved.text
    bundle = (await client.get(f"/api/workflows/{workflow['id']}/export")).json()
    bundle["name"] = "Imported strength edit"

    imported = await client.post("/api/workflows/import", json=bundle)

    assert imported.status_code == 201, imported.text
    stored = imported.json()["revisions"][0]
    assert stored["ui_graph_json"] == bundle["ui_graph"]
    assert stored["api_graph_json"] == bundle["api_graph"]
    assert stored["input_schema_json"] == bundle["input_schema"]
    assert stored["input_schema_json"]["properties"]["denoise"]["default"] == 0.35
    assert (
        image_edit_kind(
            "image_to_image",
            stored["ui_graph_json"],
            stored["api_graph_json"],
            stored["input_schema_json"],
        )
        == "strength"
    )
