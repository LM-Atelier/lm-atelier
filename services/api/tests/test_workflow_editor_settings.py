from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_workflow_editor_api import _create_workflow
from test_workflow_package_import_endpoint import _object_info, _ui_graph, _wire_runtime

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.comfy_workflow_compiler import compile_comfyui_ui_graph
from local_lm.db import SessionLocal
from local_lm.models import WorkflowRevision
from local_lm.settings_registry import (
    IMAGE_SETTINGS,
    resolve_generation_settings,
    workflow_settings,
)
from local_lm.workflow_editor_sessions import (
    WorkflowEditorReturn,
    WorkflowEditorSessionError,
    WorkflowEditorSessions,
)
from local_lm.workflow_graph_settings import (
    bind_compiled_workflow_settings,
    generated_workflow_setting_paths,
    rebind_workflow_graph_settings,
)
from local_lm.workflow_graph_settings_v1 import GRAPH_SETTINGS_SCHEMA_KEY
from local_lm.workflow_package_inputs import WorkflowPackageInputError


@pytest.mark.parametrize("add_child", [False, True])
async def test_native_editor_rebuilds_dynamic_child_settings(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    add_child: bool,
) -> None:
    info = _object_info()
    info["Source"]["input"]["required"]["sampler"] = [
        "COMFY_DYNAMICCOMBO_V3",
        {
            "options": [
                {"key": "euler"},
                {"key": "heun", "inputs": {"required": {"steps": ["INT", {"min": 2, "max": 20}]}}},
            ]
        },
    ]
    info["Source"]["input_order"]["required"].append("sampler")
    graph = _ui_graph()
    graph["nodes"][0]["widgets_values"].extend(["euler"] if add_child else ["heun", 8])
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    monkeypatch.setattr(
        app.state.services.processes, "workflow_editor_runtime_identity", lambda: "dynamic-settings"
    )
    imported = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Editable sampler", "operation": "text_to_image"},
    )
    assert imported.status_code == 201, imported.text
    path = f"/api/workflows/{imported.json()['id']}"
    opened = await client.post(path + "/editor-sessions")
    assert opened.status_code == 201, opened.text
    started = opened.json()
    edited = _ui_graph()
    edited["nodes"][0]["widgets_values"].extend(["heun", 12] if add_child else ["euler"])
    compiled = compile_comfyui_ui_graph(edited, info)
    consumed = await client.post(
        path + f"/editor-sessions/{started['id']}/consume",
        json={
            "nonce": started["nonce"],
            "base_revision_id": started["base_revision_id"],
            "ui_graph": edited,
            "api_prompt": compiled.api_graph,
        },
    )
    assert consumed.status_code == 200, consumed.text
    saved = await client.post(
        path + "/editor-drafts",
        json={"validated_return_id": consumed.json()["validated_return_id"]},
    )
    assert saved.status_code == 200, saved.text
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, saved.json()["draft_revision_id"])
        original = session.get(WorkflowRevision, started["base_revision_id"])
        assert revision is not None and original is not None
        assert ("steps" in original.input_schema_json.get("properties", {})) is not add_child, (
            "The imported selection does not expose its native child control"
        )
        assert ("steps" in revision.input_schema_json.get("properties", {})) is add_child, (
            "The saved revision does not reflect the selected native child control"
        )
        fields = workflow_settings(IMAGE_SETTINGS, revision.input_schema_json)
        assert next(field for field in fields if field.key == "steps").available is add_child
        assert not next(field for field in fields if field.key == "sampler").available
        marker = revision.input_schema_json[GRAPH_SETTINGS_SCHEMA_KEY]
        assert any(
            item["input_name"] == "sampler" and item["reason"] == "dynamic"
            for item in marker["fixed"]
        )
        if not add_child:
            with pytest.raises(ValueError, match="not mapped to an editable input"):
                resolve_generation_settings(fields, turn_overrides={"steps": 16})
        resolved = resolve_generation_settings(
            fields, turn_overrides={"steps": 16} if add_child else {}
        )
        dispatched = ComfyUIAdapter._compile(revision.api_graph_json, resolved)
        inputs = dispatched["1"]["inputs"]
        assert inputs["sampler"] == ("heun" if add_child else "euler")
        if add_child:
            assert inputs["sampler.steps"] == 16
        else:
            assert "sampler.steps" not in inputs
            assert "steps" not in resolved


@pytest.mark.parametrize("renumber", [False, True])
@pytest.mark.parametrize("legacy", [None, "empty", "seed", "noise_seed"])
async def test_native_editor_regenerates_settings_for_the_saved_graph(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    renumber: bool,
    legacy: str | None,
) -> None:
    info = _object_info()
    graph = _ui_graph()
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    monkeypatch.setattr(
        app.state.services.processes, "workflow_editor_runtime_identity", lambda: "native-settings"
    )
    legacy_schema: dict[str, Any] = (
        {"type": "object", "properties": {legacy: {"type": "integer", "default": 14}}}
        if legacy in {"seed", "noise_seed"}
        else {}
    )
    if legacy is not None:
        workflow = await _create_workflow(
            client,
            ui_graph=graph,
            api_graph=compile_comfyui_ui_graph(graph, info).api_graph,
            input_schema=legacy_schema,
        )
    else:
        imported = await client.post(
            "/api/workflows/packages/import",
            json={
                "ui_graph": graph,
                "name": "Editable native controls",
                "operation": "text_to_image",
            },
        )
        assert imported.status_code == 201, imported.text
        workflow = imported.json()
    path = f"/api/workflows/{workflow['id']}"
    opened = await client.post(path + "/editor-sessions")
    assert opened.status_code == 201, "The native editor cannot reopen a mapped workflow"
    started = opened.json()
    edited = deepcopy(graph)
    edited["nodes"][0]["widgets_values"][1] = 99
    if renumber:
        edited["nodes"][0]["id"] = 3
        edited["links"][0][1] = 3
    compiled = compile_comfyui_ui_graph(edited, info)
    consumed = await client.post(
        path + f"/editor-sessions/{started['id']}/consume",
        json={
            "nonce": started["nonce"],
            "base_revision_id": started["base_revision_id"],
            "ui_graph": edited,
            "api_prompt": compiled.api_graph,
        },
    )
    assert consumed.status_code == 200, consumed.text
    saved = await client.post(
        path + "/editor-drafts",
        json={"validated_return_id": consumed.json()["validated_return_id"]},
    )
    assert saved.status_code == 200, saved.text
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, saved.json()["draft_revision_id"])
        assert revision is not None
        node_id = "3" if renumber else "1"
        placeholder = revision.api_graph_json[node_id]["inputs"]["seed"]
        assert isinstance(placeholder, str) and placeholder.startswith("${")
        parameter = placeholder[2:-1]
        assert revision.input_schema_json["properties"][parameter]["default"] == 99
        assert revision.input_schema_json["x-lm-atelier-graph-settings"]["bindings"] == [
            {"parameter": parameter, "node_id": node_id, "input_name": "seed"}
        ]
        fields = workflow_settings(IMAGE_SETTINGS, revision.input_schema_json)
        for key, declaration in legacy_schema.get("properties", {}).items():
            assert revision.input_schema_json["properties"][key] == declaration
            assert not any(field.key == key and field.available for field in fields), (
                "The old declaration still offers a control that cannot change this graph"
            )
            with pytest.raises(ValueError):
                resolve_generation_settings(fields, turn_overrides={key: 55})
        values = resolve_generation_settings(fields, turn_overrides={parameter: 77})
        assert (
            ComfyUIAdapter._compile(revision.api_graph_json, values)[node_id]["inputs"]["seed"]
            == 77
        )
        assert revision.trusted is False
        original = session.get(WorkflowRevision, started["base_revision_id"])
        assert original is not None
        if legacy is not None:
            assert original.input_schema_json == legacy_schema
            assert original.api_graph_json["1"]["inputs"]["seed"] == 42
        else:
            assert original.input_schema_json["properties"]["seed"]["default"] == 42


def test_regeneration_drops_removed_controls_and_preserves_explicit_contracts() -> None:
    info = _object_info()
    original = compile_comfyui_ui_graph(_ui_graph(), info)
    explicit = {"type": "object", "properties": {"quality": {"type": "string", "default": "high"}}}
    bound = bind_compiled_workflow_settings(original, original.api_graph, explicit)
    saved_schema = deepcopy(bound.input_schema)
    graph = _ui_graph()
    graph["nodes"][0]["widgets_values"] = ["camera"]
    del info["Source"]["input"]["required"]["seed"]
    info["Source"]["input_order"]["required"] = ["label"]
    revised = compile_comfyui_ui_graph(graph, info)

    result = rebind_workflow_graph_settings(
        revised, revised.api_graph, bound.api_graph, bound.input_schema
    )

    assert result.input_schema["properties"] == explicit["properties"]
    assert result.input_schema[GRAPH_SETTINGS_SCHEMA_KEY] == {"version": 1, "bindings": []}
    assert result.api_graph == revised.api_graph
    assert bound.input_schema == saved_schema


def test_mapping_preserves_an_explicit_declaration_that_the_graph_uses() -> None:
    graph = _ui_graph()
    graph["nodes"][0]["widgets_values"][0] = "${sampler}"
    compilation = compile_comfyui_ui_graph(graph, _object_info())
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "sampler": {"type": "string", "enum": ["euler", "heun"], "default": "euler"},
        },
    }
    mapped = bind_compiled_workflow_settings(compilation, compilation.api_graph, schema)
    assert mapped.input_schema["properties"]["sampler"] == schema["properties"]["sampler"]
    assert "unbound_parameters" not in mapped.input_schema[GRAPH_SETTINGS_SCHEMA_KEY]
    fields = workflow_settings(IMAGE_SETTINGS, mapped.input_schema)
    values = resolve_generation_settings(fields, turn_overrides={"sampler": "heun"})
    dispatched = ComfyUIAdapter._compile(mapped.api_graph, values)
    assert dispatched["1"]["inputs"]["label"] == "heun"
    assert dispatched["1"]["inputs"]["seed"] == 42


def test_unbound_availability_cannot_hide_a_generated_control() -> None:
    compilation = compile_comfyui_ui_graph(_ui_graph(), _object_info())
    mapped = bind_compiled_workflow_settings(compilation, compilation.api_graph, {})
    mapped.input_schema[GRAPH_SETTINGS_SCHEMA_KEY]["unbound_parameters"] = ["seed"]
    with pytest.raises(ValueError, match="availability record"):
        workflow_settings(IMAGE_SETTINGS, mapped.input_schema)


@pytest.mark.parametrize("damage", ["version", "duplicate", "input", "property"])
def test_invalid_generated_bindings_cannot_remove_declarations(damage: str) -> None:
    compilation = compile_comfyui_ui_graph(_ui_graph(), _object_info())
    bound = bind_compiled_workflow_settings(compilation, compilation.api_graph, {})
    marker = bound.input_schema[GRAPH_SETTINGS_SCHEMA_KEY]
    if damage == "version":
        marker["version"] = True
    elif damage == "duplicate":
        marker["bindings"] *= 2
    elif damage == "input":
        bound.api_graph["1"]["inputs"]["seed"] = 42
    else:
        del bound.input_schema["properties"]["seed"]

    with pytest.raises(WorkflowPackageInputError, match="generated setting"):
        generated_workflow_setting_paths(bound.input_schema, bound.api_graph)


def test_editor_replay_requires_the_same_verified_settings_schema() -> None:
    sessions = WorkflowEditorSessions()
    graph = _ui_graph()
    compiled = compile_comfyui_ui_graph(graph, _object_info())
    started = sessions.start(
        workflow_id="workflow",
        base_revision_id="revision",
        base_ui_graph=graph,
        base_api_graph=compiled.api_graph,
        runtime_identity="runtime",
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"seed": {"type": "integer", "default": 42}},
    }

    def consume() -> WorkflowEditorReturn:
        return sessions.consume(
            session_id=started.id,
            nonce=started.nonce,
            workflow_id="workflow",
            base_revision_id="revision",
            current_revision_id="revision",
            returned_ui_graph=graph,
            returned_api_graph=compiled.api_graph,
            runtime_identity="runtime",
            returned_input_schema=schema,
        )

    first = consume()
    assert consume() == first
    schema["properties"]["seed"]["default"] = 99
    with pytest.raises(WorkflowEditorSessionError) as rejected:
        consume()
    assert rejected.value.code == "workflow-editor-session-replay-mismatch"
