"""Video frame controls preserve the runtime widget's bounded frame sequence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from workflow_fixtures import seed_workflow_trust

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.comfy_templates import (
    COMFY_TEMPLATE_COMPILER_VERSION,
    ComfyTemplate,
    ComfyTemplateRegistry,
    _compile_ui_graph,
)
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.models import ModelInstall, WorkflowRevision
from local_lm.settings_registry import VIDEO_SETTINGS, validate_settings, workflow_settings


def _frame_graph(
    *, minimum: int = 9, step: int = 8, default: int = 97, maximum: int = 4096
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = {
        "width": ["INT", {"default": 768, "min": 64, "max": 4096, "step": 32}],
        "height": ["INT", {"default": 512, "min": 64, "max": 4096, "step": 32}],
        "length": ["INT", {"default": default, "min": minimum, "max": maximum, "step": step}],
        "batch_size": ["INT", {"default": 1, "min": 1, "max": 16}],
    }
    info: dict[str, Any] = {
        "FrameFactory": {
            "input": {"required": inputs},
            "input_order": {"required": list(inputs)},
            "output": ["LATENT"],
        },
        "VideoOutput": {
            "input": {"required": {"samples": ["LATENT"]}},
            "input_order": {"required": ["samples"]},
            "output": [],
            "output_node": True,
        },
    }
    graph: dict[str, Any] = {
        "nodes": [
            {
                "id": 1,
                "type": "FrameFactory",
                "inputs": [
                    {"name": name, "type": spec[0], "widget": {"name": name}}
                    for name, spec in inputs.items()
                ],
                "outputs": [{"name": "samples", "type": "LATENT"}],
                "widgets_values": [768, 512, default, 1],
            },
            {
                "id": 2,
                "type": "VideoOutput",
                "inputs": [{"name": "samples", "type": "LATENT", "link": 1}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "LATENT"]],
    }
    return graph, info


@pytest.mark.parametrize("minimum", [1, 9])
@pytest.mark.parametrize("frames", [49, 97])
def test_video_length_binds_frames_and_preserves_the_saved_default(
    minimum: int, frames: int
) -> None:
    graph, info = _frame_graph(minimum=minimum)
    compiled, schema = _compile_ui_graph(graph, info, operation="image_to_video")
    assert compiled["1"]["inputs"]["length"] == "${frames}"
    prop = schema["properties"]["frames"]
    assert not prop.get("readOnly", False)
    assert prop["default"] == 97
    assert prop["minimum"] == minimum
    assert prop["maximum"] == 1017
    assert prop["x-lm-atelier-step"] == 8
    assert "multipleOf" not in prop
    assert prop["enum"] == list(range(minimum, 1025, 8))
    fields = workflow_settings(VIDEO_SETTINGS, schema)
    assert validate_settings({"frames": frames}, fields) == {"frames": frames}


@pytest.mark.parametrize("frames", [8, 48, 96, 1024, 1025, True, 49.5])
def test_video_frame_controls_refuse_counts_outside_the_declared_sequence(
    frames: object,
) -> None:
    graph, info = _frame_graph()
    _, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    fields = workflow_settings(VIDEO_SETTINGS, schema)
    frame_field = next(field for field in fields if field.key == "frames")
    assert frame_field.available
    with pytest.raises(ValueError):
        validate_settings({"frames": frames}, fields)


@pytest.mark.parametrize(
    "change", ["image-operation", "floating-length", "non-latent-output", "no-dimensions"]
)
def test_unrelated_length_widgets_keep_their_literal(change: str) -> None:
    graph, info = _frame_graph()
    operation = "text_to_video"
    if change == "image-operation":
        operation = "text_to_image"
    elif change == "floating-length":
        info["FrameFactory"]["input"]["required"]["length"][0] = "FLOAT"
    elif change == "non-latent-output":
        info["FrameFactory"]["output"] = ["IMAGE"]
    elif change == "no-dimensions":
        info["FrameFactory"]["input"]["required"].pop("height")
        info["FrameFactory"]["input_order"]["required"].remove("height")
        graph["nodes"][0]["inputs"].pop(1)
        graph["nodes"][0]["widgets_values"].pop(1)
    compiled, schema = _compile_ui_graph(graph, info, operation=operation)
    assert compiled["1"]["inputs"]["length"] == 97
    assert schema["properties"]["frames"].get("readOnly") is True


def test_a_linked_frame_count_stays_owned_by_its_graph() -> None:
    graph, info = _frame_graph()
    info["FrameInput"] = {
        "input": {"required": {"count": ["INT", {"default": 97}]}},
        "input_order": {"required": ["count"]},
        "output": ["INT"],
    }
    graph["nodes"].append(
        {
            "id": 3,
            "type": "FrameInput",
            "inputs": [],
            "outputs": [{"type": "INT"}],
            "widgets_values": [97],
        }
    )
    graph["nodes"][0]["inputs"][2]["link"] = 2
    graph["links"].append([2, 3, 0, 1, 2, "INT"])
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["1"]["inputs"]["length"] == ["3", 0]
    assert schema["properties"]["frames"].get("readOnly") is True


def test_an_unexecuted_length_widget_does_not_change_frame_controls() -> None:
    graph, info = _frame_graph()
    unused = copy.deepcopy(graph["nodes"][0])
    unused["id"] = 3
    unused["widgets_values"][2] = 65
    graph["nodes"].append(unused)
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["1"]["inputs"]["length"] == "${frames}"
    assert compiled["3"]["inputs"]["length"] == 65
    assert schema["properties"]["frames"]["default"] == 97


def test_zero_offset_frame_sequences_keep_a_numeric_multiple() -> None:
    graph, info = _frame_graph(minimum=8, default=96)
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["1"]["inputs"]["length"] == "${frames}"
    prop = schema["properties"]["frames"]
    assert prop["default"] == 96
    assert prop["multipleOf"] == 8
    fields = workflow_settings(VIDEO_SETTINGS, schema)
    assert validate_settings({"frames": 104}, fields) == {"frames": 104}
    with pytest.raises(ValueError):
        validate_settings({"frames": 97}, fields)


def _paired_frame_graph(
    first: tuple[int, int, int], second: tuple[int, int, int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph, info = _frame_graph(minimum=first[0], step=first[1], default=first[2])
    other_graph, other_info = _frame_graph(minimum=second[0], step=second[1], default=second[2])
    other = other_graph["nodes"][0]
    other["id"] = 3
    other["type"] = "SecondFrameFactory"
    graph["nodes"].append(other)
    info["SecondFrameFactory"] = other_info["FrameFactory"]
    info["VideoOutput"]["input"]["required"]["second_samples"] = ["LATENT"]
    info["VideoOutput"]["input_order"]["required"].append("second_samples")
    graph["nodes"][1]["inputs"].append({"name": "second_samples", "type": "LATENT", "link": 2})
    graph["links"].append([2, 3, 0, 2, 1, "LATENT"])
    return graph, info


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_frame_controls_publish_the_intersection_in_numeric_order(reverse: bool) -> None:
    specs = [(9, 8, 97), (1, 4, 97)]
    if reverse:
        specs.reverse()
    graph, info = _paired_frame_graph(*specs)
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["1"]["inputs"]["length"] == "${frames}"
    assert compiled["3"]["inputs"]["length"] == "${frames}"
    prop = schema["properties"]["frames"]
    assert prop["enum"] == list(range(9, 1025, 8))
    assert prop["minimum"] == 9
    assert prop["maximum"] == 1017
    assert prop["x-lm-atelier-step"] == 8
    assert prop["default"] == 97


@pytest.mark.parametrize("reverse", [False, True])
def test_different_offsets_share_only_counts_both_producers_accept(reverse: bool) -> None:
    specs = [(1, 6, 7), (3, 4, 7)]
    if reverse:
        specs.reverse()
    graph, info = _paired_frame_graph(*specs)
    _, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    prop = schema["properties"]["frames"]
    assert prop["enum"] == list(range(7, 1025, 12))
    assert prop["minimum"] == 7
    assert prop["maximum"] == 1015
    assert prop["x-lm-atelier-step"] == 12
    assert prop["default"] == 7
    fields = workflow_settings(VIDEO_SETTINGS, schema)
    assert validate_settings({"frames": 55}, fields) == {"frames": 55}
    with pytest.raises(ValueError):
        validate_settings({"frames": 49}, fields)


def test_incompatible_frame_producers_are_refused() -> None:
    graph, info = _paired_frame_graph((1, 8, 97), (2, 8, 97))
    with pytest.raises(ValueError, match="no common setting choice"):
        _compile_ui_graph(graph, info, operation="text_to_video")


@pytest.mark.parametrize("outputs", ["LATENT", {"LATENT": True}, None])
def test_unreadable_output_declarations_do_not_bind_a_length(outputs: object) -> None:
    graph, info = _frame_graph()
    info["FrameFactory"]["output"] = outputs
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["1"]["inputs"]["length"] == 97
    assert schema["properties"]["frames"].get("readOnly") is True


@pytest.mark.parametrize("constraint", [{"step": 0}, {"min": 9.5}, {"min": 2049}])
def test_frame_constraints_without_a_usable_integer_window_are_refused(
    constraint: dict[str, float],
) -> None:
    graph, info = _frame_graph()
    info["FrameFactory"]["input"]["required"]["length"][1].update(constraint)
    with pytest.raises(ValueError):
        _compile_ui_graph(graph, info, operation="text_to_video")


@pytest.mark.parametrize("frames", [49, 97])
def test_accepted_frame_count_reaches_the_runtime_node(frames: int) -> None:
    graph, info = _frame_graph()
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    parameters = validate_settings({"frames": frames}, workflow_settings(VIDEO_SETTINGS, schema))
    submitted = ComfyUIAdapter._compile(compiled, parameters)
    assert submitted["1"]["inputs"]["length"] == frames
    assert type(submitted["1"]["inputs"]["length"]) is int
    assert compiled["1"]["inputs"]["length"] == "${frames}"


def test_different_saved_counts_remain_independent() -> None:
    graph, info = _paired_frame_graph((9, 8, 97), (1, 4, 49))
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["1"]["inputs"]["length"] == 97
    assert compiled["3"]["inputs"]["length"] == 49
    assert schema["properties"]["frames"].get("readOnly") is True


def test_an_invalid_saved_count_falls_back_to_the_lowest_valid_integer() -> None:
    graph, info = _frame_graph(default=48)
    _, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert schema["properties"]["frames"]["default"] == 9


@pytest.mark.parametrize("reverse", [False, True])
def test_zero_and_offset_sequences_intersect_independently_of_node_order(reverse: bool) -> None:
    specs = [(2, 6, 8), (4, 4, 8)]
    if reverse:
        specs.reverse()
    graph, info = _paired_frame_graph(*specs)
    _, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    prop = schema["properties"]["frames"]
    assert prop["enum"] == list(range(8, 1025, 12))
    assert prop["minimum"] == 8
    assert prop["maximum"] == 1016
    assert prop["x-lm-atelier-step"] == 12
    fields = workflow_settings(VIDEO_SETTINGS, schema)
    assert validate_settings({"frames": 20}, fields) == {"frames": 20}
    with pytest.raises(ValueError):
        validate_settings({"frames": 14}, fields)


def test_a_spatial_latent_transformer_keeps_its_length_literal() -> None:
    graph, info = _frame_graph()
    info["FrameFactory"]["input"]["optional"] = {"latent": ["LATENT"]}
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["1"]["inputs"]["length"] == 97
    assert schema["properties"]["frames"].get("readOnly") is True


def test_another_executed_length_keeps_frame_counts_under_graph_control() -> None:
    graph, info = _frame_graph()
    info["VideoOutput"]["input"]["required"]["length"] = ["INT", {"default": 97}]
    info["VideoOutput"]["input_order"]["required"].append("length")
    graph["nodes"][1]["inputs"].append(
        {"name": "length", "type": "INT", "widget": {"name": "length"}}
    )
    graph["nodes"][1]["widgets_values"] = [97]
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["1"]["inputs"]["length"] == 97
    assert compiled["2"]["inputs"]["length"] == 97
    assert schema["properties"]["frames"].get("readOnly") is True


def test_a_linked_producer_keeps_other_producer_lengths_fixed() -> None:
    graph, info = _paired_frame_graph((9, 8, 97), (9, 8, 97))
    info["FrameInput"] = {
        "input": {"required": {"count": ["INT", {"default": 97}]}},
        "input_order": {"required": ["count"]},
        "output": ["INT"],
    }
    graph["nodes"].append(
        {
            "id": 4,
            "type": "FrameInput",
            "inputs": [],
            "outputs": [{"type": "INT"}],
            "widgets_values": [97],
        }
    )
    graph["nodes"][2]["inputs"][2]["link"] = 3
    graph["links"].append([3, 4, 0, 3, 2, "INT"])
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["1"]["inputs"]["length"] == 97
    assert compiled["3"]["inputs"]["length"] == ["4", 0]
    assert schema["properties"]["frames"].get("readOnly") is True


def _add_other_count(graph: dict[str, Any], info: dict[str, Any]) -> None:
    info["VideoOutput"]["input"]["required"]["repeat_count"] = ["INT", {"default": 97}]
    info["VideoOutput"]["input_order"]["required"].append("repeat_count")
    graph["nodes"][1]["inputs"].append(
        {"name": "repeat_count", "type": "INT", "widget": {"name": "repeat_count"}}
    )
    graph["nodes"][1]["widgets_values"] = [97]


@pytest.mark.parametrize("fan_out", [False, True])
def test_a_primitive_frame_control_preserves_other_inputs_it_drives(fan_out: bool) -> None:
    graph, info = _frame_graph()
    graph["nodes"].append(
        {
            "id": 3,
            "type": "PrimitiveNode",
            "inputs": [],
            "outputs": [{"name": "INT", "type": "INT", "widget": {"name": "length"}}],
            "widgets_values": [97, "fixed"],
        }
    )
    graph["nodes"][0]["inputs"][2]["link"] = 2
    graph["links"].append([2, 3, 0, 1, 2, "INT"])
    if fan_out:
        _add_other_count(graph, info)
        graph["nodes"][1]["inputs"][1]["link"] = 3
        graph["links"].append([3, 3, 0, 2, 1, "INT"])
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["1"]["inputs"]["length"] == (97 if fan_out else "${frames}")
    assert bool(schema["properties"]["frames"].get("readOnly")) is fan_out
    if fan_out:
        assert compiled["2"]["inputs"]["repeat_count"] == 97


def test_a_primitive_link_to_a_missing_node_does_not_break_compilation() -> None:
    graph, info = _frame_graph()
    graph["nodes"].append(
        {
            "id": 3,
            "type": "PrimitiveNode",
            "inputs": [],
            "outputs": [{"type": "INT"}],
            "widgets_values": [97, "fixed"],
        }
    )
    graph["links"].append([2, 3, 0, 999, 0, "INT"])
    compiled, _ = _compile_ui_graph(graph, info, operation="text_to_video")
    assert set(compiled) == {"1", "2"}


@pytest.mark.parametrize("fan_out", [False, True])
def test_a_subgraph_frame_control_preserves_other_inputs_it_drives(fan_out: bool) -> None:
    graph, info = _frame_graph()
    if fan_out:
        _add_other_count(graph, info)
    inner = graph["nodes"][0]
    inner["inputs"][2]["link"] = 2
    subgraph_links = [[2, -10, 0, 1, 2, "INT"], [3, 1, 0, -20, 0, "LATENT"]]
    inner_nodes = [inner]
    if fan_out:
        other = copy.deepcopy(graph["nodes"][1])
        other["inputs"][0]["link"] = 4
        other["inputs"][1]["link"] = 5
        inner_nodes.append(other)
        subgraph_links.extend([[4, 1, 0, 2, 0, "LATENT"], [5, -10, 0, 2, 1, "INT"]])
    graph["definitions"] = {
        "subgraphs": [
            {
                "id": "frame-group",
                "inputs": [{"name": "count", "type": "INT"}],
                "nodes": inner_nodes,
                "links": subgraph_links,
            }
        ]
    }
    graph["nodes"] = [
        {
            "id": 10,
            "type": "frame-group",
            "inputs": [],
            "outputs": [{"name": "samples", "type": "LATENT"}],
            "widgets_values": [],
        },
        graph["nodes"][1],
    ]
    graph["links"] = [[1, 10, 0, 2, 0, "LATENT"]]
    compiled, schema = _compile_ui_graph(graph, info, operation="text_to_video")
    assert compiled["10:1"]["inputs"]["length"] == (97 if fan_out else "${frames}")
    assert bool(schema["properties"]["frames"].get("readOnly")) is fan_out
    if fan_out:
        assert compiled["10:2"]["inputs"]["repeat_count"] == 97


@pytest.mark.parametrize("requested_frames", [None, 49])
async def test_a_turn_uses_the_compiled_templates_saved_or_requested_frame_count(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_frames: int | None,
) -> None:
    graph, info = _frame_graph()
    path = tmp_path / "video-template.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    template = ComfyTemplate(
        id="video-template",
        path=path,
        role="video",
        operation="text_to_video",
        score=0,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        dependencies=(),
    )
    registry = ComfyTemplateRegistry(settings)
    monkeypatch.setattr(registry, "get", lambda *args, **kwargs: template)
    compiled = registry.compile(template.id, "video", info)
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Video frame choices",
            "operation": "text_to_video",
            "engine": "mock",
            "ui_graph": compiled.ui_graph,
            "api_graph": compiled.api_graph,
            "input_schema": compiled.input_schema,
        },
    )
    assert created.status_code == 201, created.text
    revision_id = created.json()["current_revision_id"]
    seed_workflow_trust(revision_id)
    chat = await client.post("/api/chats", json={"title": "A blue wheel"})
    assert chat.status_code == 201, chat.text
    async with app.state.services.scheduler.lease("primary"):
        turn = await client.post(
            f"/api/chats/{chat.json()['id']}/turns",
            json={
                "text": "Create a short video of a blue wheel",
                "mode": "video",
                "workflow_revision_id": revision_id,
                "settings": {} if requested_frames is None else {"frames": requested_frames},
            },
        )
        assert turn.status_code == 202, turn.text
        run = turn.json()["run"]
        assert run["workflow_revision_id"] == revision_id
        assert run["settings_json"].get("frames") == (requested_frames or 97)
        submitted = ComfyUIAdapter._compile(compiled.api_graph, run["settings_json"])
        assert submitted["1"]["inputs"]["length"] == (requested_frames or 97)


async def test_an_installed_video_gets_editable_frame_counts_after_compiler_refresh(
    app: FastAPI,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import local_lm.downloads as downloads_module

    graph, info = _frame_graph()
    path = tmp_path / "video-template.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    template = ComfyTemplate(
        id="video-template",
        path=path,
        role="video",
        operation="text_to_video",
        score=0,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        dependencies=(),
    )
    manager = app.state.services.downloads
    monkeypatch.setattr(manager.comfy_templates, "get", lambda *args, **kwargs: template)

    async def object_info() -> dict[str, Any]:
        return info

    monkeypatch.setattr(manager, "media_adapter", app.state.services.engines.media)
    monkeypatch.setattr(manager.media_adapter, "object_info", object_info, raising=False)
    compiled = manager.comfy_templates.compile(template.id, "video", info)
    legacy = copy.deepcopy(compiled)
    legacy.api_graph["1"]["inputs"]["length"] = 97
    legacy.input_schema["properties"]["frames"] = {"readOnly": True}
    with SessionLocal() as session:
        install = ModelInstall(
            name="Video model",
            role="video",
            engine="comfyui",
            local_path=str(settings.model_dir / "video-model"),
            active=True,
            manifest_json={
                "workflow_template_id": template.id,
                "workflow_template_sha256": template.sha256,
                "remote_id": "example/video",
                "revision": "main",
                "files": [],
                "comfy_paths": {},
            },
        )
        session.add(install)
        session.flush()
        with monkeypatch.context() as older:
            older.setattr(
                downloads_module,
                "COMFY_TEMPLATE_COMPILER_VERSION",
                COMFY_TEMPLATE_COMPILER_VERSION - 1,
            )
            previous = DownloadManager._ensure_template_workflow(session, legacy, install)
        previous_id = previous.id
        definition_id = previous.workflow_id
        session.commit()
    assert await manager.refresh_installed_media_workflows() == 1
    assert await manager.refresh_installed_media_workflows() == 0
    with SessionLocal() as session:
        revisions = session.query(WorkflowRevision).filter_by(workflow_id=definition_id).all()
        assert len(revisions) == 2
        original = next(revision for revision in revisions if revision.id == previous_id)
        current = next(revision for revision in revisions if revision.id != previous_id)
        assert original.api_graph_json["1"]["inputs"]["length"] == 97
        assert original.input_schema_json["properties"]["frames"] == {"readOnly": True}
        prop = current.input_schema_json["properties"]["frames"]
        assert not prop.get("readOnly", False)
        assert prop["default"] == 97
        assert current.dependencies_json["compiler_version"] == COMFY_TEMPLATE_COMPILER_VERSION
        defaults = DownloadManager._template_defaults(compiled)
        assert defaults["frames"] == 97
        assert (
            ComfyUIAdapter._compile(current.api_graph_json, {**defaults, "frames": 49})["1"][
                "inputs"
            ]["length"]
            == 49
        )
