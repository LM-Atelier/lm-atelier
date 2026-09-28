from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_workflow_package_import_endpoint import _object_info, _ui_graph, _wire_runtime

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.comfy_workflow_compiler import compile_comfyui_ui_graph
from local_lm.db import SessionLocal
from local_lm.models import WorkflowRevision
from local_lm.settings_registry import (
    VIDEO_SETTINGS,
    resolve_generation_settings,
    workflow_settings,
)
from local_lm.video_length import resolve_video_length_settings
from local_lm.workflow_graph_settings import bind_compiled_workflow_settings


@pytest.mark.parametrize(
    ("kind", "value"), [("integer", 81), ("number", 2.5), ("boolean", True), ("string", "fixed")]
)
def test_fixed_scalar_settings_retain_their_declared_control_type(kind: str, value: Any) -> None:
    fields = workflow_settings([], {"properties": {"fixed_value": {"type": kind, "const": value}}})
    (field,) = fields
    assert field.type == kind
    assert field.default == value and field.choices == [value]
    assert resolve_generation_settings(fields) == {"fixed_value": value}
    with pytest.raises(ValueError):
        resolve_generation_settings(fields, turn_overrides={"fixed_value": "different"})


def _native_video(fps: float) -> tuple[dict[str, Any], dict[str, Any]]:
    info, graph = _object_info(), _ui_graph()
    info["Source"]["input"]["required"].update(
        {
            "frames": ["INT", {"min": 17, "max": 81, "step": 16}],
            "fps": ["FLOAT", {"min": 1, "max": 60, "step": 0.1}],
        }
    )
    info["Source"]["input_order"]["required"].extend(["frames", "fps"])
    graph["nodes"][0]["widgets_values"].extend([49, fps])
    return info, graph


@pytest.mark.parametrize("renumber", [False, True])
@pytest.mark.parametrize("edited_frames", [65, 66])
@pytest.mark.parametrize(
    ("numerator", "denominator", "expected_frames"), [(16, 1, 33), (30000, 1001, 65)]
)
async def test_native_revision_keeps_declared_rational_video_length(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    renumber: bool,
    numerator: int,
    denominator: int,
    expected_frames: int,
    edited_frames: int,
) -> None:
    fps = numerator / denominator
    info, graph = _native_video(fps)
    _wire_runtime(app, monkeypatch, runtime_object_info=info)
    monkeypatch.setattr(
        app.state.services.processes, "workflow_editor_runtime_identity", lambda: "video-settings"
    )
    compiled = compile_comfyui_ui_graph(graph, info)
    bound = bind_compiled_workflow_settings(
        compiled, compiled.api_graph, {}, operation="text_to_video"
    )
    contract = {
        "version": 1,
        "frames_parameter": "frames",
        "fps_parameter": "fps",
        "fps_numerator": numerator,
        "fps_denominator": denominator,
        "frame_alignment": 16,
        "frame_offset": 1,
    }
    bound.input_schema["x-lm-atelier-video-length"] = contract
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Declared video length",
            "operation": "text_to_video",
            "ui_graph": graph,
            "api_graph": bound.api_graph,
            "input_schema": bound.input_schema,
        },
    )
    assert response.status_code == 201, response.text
    path = f"/api/workflows/{response.json()['id']}"
    opened = await client.post(path + "/editor-sessions")
    assert opened.status_code == 201, opened.text
    started = opened.json()
    edited = deepcopy(graph)
    edited["nodes"][0]["widgets_values"][-2] = edited_frames
    if renumber:
        edited["nodes"][0]["id"] = 3
        edited["links"][0][1] = 3
    current = compile_comfyui_ui_graph(edited, info)
    consumed = await client.post(
        path + f"/editor-sessions/{started['id']}/consume",
        json={
            "nonce": started["nonce"],
            "base_revision_id": started["base_revision_id"],
            "ui_graph": edited,
            "api_prompt": current.api_graph,
        },
    )
    assert consumed.status_code == 200, consumed.text
    saved = await client.post(
        path + "/editor-drafts",
        json={
            "validated_return_id": consumed.json()["validated_return_id"],
        },
    )
    if edited_frames == 66:
        assert saved.status_code == 422, "The saved revision violated its declared frame alignment"
        stored = (await client.get(path)).json()
        assert len(stored["revisions"]) == 1
        assert stored["current_revision_id"] == started["base_revision_id"]
        return
    assert saved.status_code == 200, saved.text
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, saved.json()["draft_revision_id"])
        original = session.get(WorkflowRevision, started["base_revision_id"])
        assert revision is not None and original is not None
        assert original.input_schema_json["properties"]["frames"]["default"] == 49
        assert revision.input_schema_json["properties"]["frames"]["default"] == 65
        assert revision.input_schema_json["x-lm-atelier-video-length"] == contract
        fields = workflow_settings(VIDEO_SETTINGS, revision.input_schema_json)
        assert not {"frames", "fps"} & {field.key for field in fields}
        duration = next(field for field in fields if field.key == "duration_seconds")
        assert duration.default == pytest.approx(65 / fps)
        settings = resolve_generation_settings(fields, turn_overrides={"duration_seconds": 2})
        resolved, provenance = resolve_video_length_settings(settings, revision.input_schema_json)
        inputs = ComfyUIAdapter._compile(revision.api_graph_json, resolved)[
            "3" if renumber else "1"
        ]["inputs"]
        assert inputs["frames"] == expected_frames and inputs["fps"] == fps
        assert provenance == {
            "requested_seconds": 2.0,
            "delivered_seconds": expected_frames / fps,
            "frames": expected_frames,
            "fps": fps,
        }


def test_widget_increment_does_not_invent_a_video_alignment_contract() -> None:
    info, graph = _native_video(30000 / 1001)
    compiled = compile_comfyui_ui_graph(graph, info)
    bound = bind_compiled_workflow_settings(
        compiled, compiled.api_graph, {}, operation="text_to_video"
    )
    assert "x-lm-atelier-video-length" not in bound.input_schema
    fields = workflow_settings(VIDEO_SETTINGS, bound.input_schema)
    assert "duration_seconds" not in {field.key for field in fields}
    values = resolve_generation_settings(fields, turn_overrides={"frames": 50})
    assert ComfyUIAdapter._compile(bound.api_graph, values)["1"]["inputs"]["frames"] == 50
