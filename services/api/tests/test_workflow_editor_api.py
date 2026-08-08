from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.workflow_editor_sessions import (
    WorkflowEditorSessions,
    workflow_api_graph_sha256,
    workflow_ui_graph_sha256,
)


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


def _api_graph() -> dict[str, Any]:
    return {
        "1": {
            "inputs": {"label": "camera", "seed": 42},
            "class_type": "Source",
            "_meta": {"title": "Source image"},
        },
        "2": {
            "inputs": {"images": ["1", 0], "filename_prefix": "result"},
            "class_type": "Save",
            "_meta": {"title": "Save"},
        },
    }


async def _create_workflow(
    client: AsyncClient,
    *,
    engine: str = "comfyui",
    ui_graph: dict[str, Any] | None = None,
    api_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Editable workflow",
            "operation": "text_to_image",
            "engine": engine,
            "ui_graph": _ui_graph() if ui_graph is None else ui_graph,
            "api_graph": _api_graph() if api_graph is None else api_graph,
            "trusted": True,
        },
    )
    assert response.status_code == 201
    return response.json()


def _ready_editor(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    *,
    identities: Iterator[str | None] | None = None,
) -> list[str]:
    services = app.state.services
    invalidations: list[str] = []
    identity_values = identities

    def authority() -> str | None:
        if identity_values is None:
            return "runtime_one"
        return next(identity_values)

    async def object_info() -> dict[str, Any]:
        return _object_info()

    monkeypatch.setattr(services.processes, "workflow_editor_runtime_identity", authority)
    monkeypatch.setattr(services.engines.media, "object_info", object_info, raising=False)
    monkeypatch.setattr(
        services.engines.media,
        "invalidate_object_info_cache",
        lambda: invalidations.append("invalidated"),
        raising=False,
    )
    return invalidations


async def test_start_and_cancel_bind_exact_revision_without_mutating_workflow(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalidations = _ready_editor(app, monkeypatch)
    workflow = await _create_workflow(client)
    before = next(
        item for item in (await client.get("/api/workflows")).json() if item["id"] == workflow["id"]
    )

    started = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")

    assert started.status_code == 201
    payload = started.json()
    assert payload["workflow_id"] == workflow["id"]
    assert payload["base_revision_id"] == workflow["current_revision_id"]
    assert payload["base_graph_sha256"] == workflow_ui_graph_sha256(_ui_graph())
    assert payload["base_prompt_sha256"] == workflow_api_graph_sha256(_api_graph())
    assert payload["ui_graph"] == _ui_graph()
    assert payload["nonce"]
    assert started.headers["cache-control"] == "no-store"
    assert "url" not in payload
    assert "api_graph" not in payload
    assert invalidations == ["invalidated"]

    wrong_nonce = await client.post(
        f"/api/workflows/workflow_two/editor-sessions/{payload['id']}/cancel",
        json={"nonce": "wrong"},
    )
    assert wrong_nonce.status_code == 403
    assert wrong_nonce.json()["code"] == "workflow-editor-session-authentication-failed"
    extra_field = await client.post(
        f"/api/workflows/{workflow['id']}/editor-sessions/{payload['id']}/cancel",
        json={"nonce": payload["nonce"], "query_nonce": payload["nonce"]},
    )
    assert extra_field.status_code == 422
    wrong_workflow = await client.post(
        f"/api/workflows/workflow_two/editor-sessions/{payload['id']}/cancel",
        json={"nonce": payload["nonce"]},
    )
    assert wrong_workflow.status_code == 409
    assert wrong_workflow.json()["code"] == "workflow-editor-session-mismatch"
    cancelled = await client.post(
        f"/api/workflows/{workflow['id']}/editor-sessions/{payload['id']}/cancel",
        json={"nonce": payload["nonce"]},
    )
    assert cancelled.status_code == 204
    replay = await client.post(
        f"/api/workflows/{workflow['id']}/editor-sessions/{payload['id']}/cancel",
        json={"nonce": payload["nonce"]},
    )
    assert replay.status_code == 404

    after = next(
        item for item in (await client.get("/api/workflows")).json() if item["id"] == workflow["id"]
    )
    assert after == before


@pytest.mark.parametrize(
    ("engine", "ui_graph", "api_graph", "expected_status", "expected_code"),
    [
        ("mock", _ui_graph(), _api_graph(), 422, "workflow-editor-engine-unsupported"),
        ("comfyui", {}, _api_graph(), 422, "workflow-editor-ui-graph-missing"),
        ("comfyui", _ui_graph(), {}, 409, "workflow-editor-api-graph-missing"),
    ],
)
async def test_start_refuses_ineligible_workflow(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    ui_graph: dict[str, Any],
    api_graph: dict[str, Any],
    expected_status: int,
    expected_code: str,
) -> None:
    _ready_editor(app, monkeypatch)
    workflow = await _create_workflow(
        client,
        engine=engine,
        ui_graph=ui_graph,
        api_graph=api_graph,
    )

    response = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")

    assert response.status_code == expected_status
    assert response.json()["code"] == expected_code


async def test_start_refuses_stopped_or_restarted_runtime(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = await _create_workflow(client)
    _ready_editor(app, monkeypatch, identities=iter([None]))
    stopped = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")
    assert stopped.status_code == 409
    assert stopped.json()["code"] == "workflow-editor-runtime-not-ready"

    _ready_editor(app, monkeypatch, identities=iter(["runtime_one", "runtime_two"]))
    restarted = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")
    assert restarted.status_code == 409
    assert restarted.json()["code"] == "workflow-editor-runtime-changed"
    assert app.state.services.workflow_editor_sessions.active_count == 0


async def test_start_refuses_unreachable_inventory_and_graph_prompt_mismatch(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = await _create_workflow(client)
    services = app.state.services
    monkeypatch.setattr(
        services.processes,
        "workflow_editor_runtime_identity",
        lambda: "runtime_one",
    )

    async def unavailable() -> dict[str, Any]:
        raise RuntimeError("offline")

    monkeypatch.setattr(services.engines.media, "object_info", unavailable, raising=False)
    unreachable = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")
    assert unreachable.status_code == 503
    assert unreachable.json()["code"] == "workflow-editor-node-inventory-unavailable"

    async def invalid_inventory() -> list[object]:
        return []

    monkeypatch.setattr(
        services.engines.media,
        "object_info",
        invalid_inventory,
        raising=False,
    )
    invalid = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")
    assert invalid.status_code == 503
    assert invalid.json()["code"] == "workflow-editor-node-inventory-invalid"

    _ready_editor(app, monkeypatch)
    mismatched = await _create_workflow(
        client,
        api_graph={"1": {"class_type": "Source", "inputs": {"label": "different"}}},
    )
    response = await client.post(f"/api/workflows/{mismatched['id']}/editor-sessions")
    assert response.status_code == 409
    assert response.json()["code"] == "workflow-editor-graph-prompt-mismatch"


async def test_start_reports_typed_compiler_refusal(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready_editor(app, monkeypatch)
    ui_graph = _ui_graph()
    ui_graph["nodes"][0]["type"] = "UnavailableNode"
    workflow = await _create_workflow(client, ui_graph=ui_graph)

    response = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-editor-graph-cannot-compile"
    assert response.json()["reason_code"] == "missing_node_type"


async def test_start_maps_session_capacity_without_eviction(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready_editor(app, monkeypatch)
    app.state.services.workflow_editor_sessions = WorkflowEditorSessions(max_active=1)
    workflow = await _create_workflow(client)

    first = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")
    second = await client.post(f"/api/workflows/{workflow['id']}/editor-sessions")

    assert first.status_code == 201
    assert second.status_code == 429
    assert second.json()["code"] == "workflow-editor-capacity"
