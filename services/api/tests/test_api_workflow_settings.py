"""Map saved API inputs using the runtime's contracts and actual graph connections."""

from copy import deepcopy
from typing import Any

import pytest
from test_workflow_package_import_endpoint import _object_info, _ui_graph

from local_lm.comfy_workflow_compiler import (
    WorkflowCompilationError,
    compile_comfyui_ui_graph,
    describe_comfyui_api_graph,
)
from local_lm.workflow_graph_settings import bind_compiled_workflow_settings
from local_lm.workflow_supplied_settings import bind_api_workflow_settings


def test_mapping_preserves_existing_runtime_inputs_without_schema_declarations() -> None:
    from local_lm.adapters.comfyui import ComfyUIAdapter
    from local_lm.settings_registry import (
        IMAGE_SETTINGS,
        resolve_generation_settings,
        workflow_settings,
    )

    graph, info = _graph()
    graph["1"]["inputs"].update({"seed": "${seed}", "label": "${input_image}"})
    original = deepcopy(graph)
    bound = bind_api_workflow_settings(graph, info, {})
    fields = workflow_settings(IMAGE_SETTINGS, bound.input_schema)
    values = resolve_generation_settings(fields, turn_overrides={"seed": 81})
    dispatched = ComfyUIAdapter._compile(bound.api_graph, {**values, "input_image": "neutral.png"})

    assert bound.api_graph == original
    assert dispatched["1"]["inputs"]["seed"] == 81
    assert dispatched["1"]["inputs"]["label"] == "neutral.png"
    assert "input_image" not in bound.input_schema["properties"]
    assert graph == original


@pytest.mark.parametrize("declared", [False, True])
def test_display_metadata_does_not_offer_an_unused_setting(declared: bool) -> None:
    from local_lm.settings_registry import IMAGE_SETTINGS, workflow_settings

    graph, info = _graph()
    graph["1"]["_meta"] = {"title": "${steps}"}
    schema = {"properties": {"steps": {"type": "integer", "default": 20}}} if declared else {}
    original = deepcopy((graph, schema))

    bound = bind_api_workflow_settings(graph, info, schema)
    fields = workflow_settings(IMAGE_SETTINGS, bound.input_schema)

    assert not next(field for field in fields if field.key == "steps").available
    assert bound.api_graph["1"]["_meta"] == graph["1"]["_meta"]
    assert (graph, schema) == original


def _graph() -> tuple[dict[str, Any], dict[str, Any]]:
    info = _object_info()
    graph = compile_comfyui_ui_graph(_ui_graph(), info).api_graph
    return graph, info


def test_api_controls_keep_explicit_bindings_and_unrelated_values() -> None:
    graph, info = _graph()
    graph["1"]["inputs"]["seed"] = "${chosen_seed}"
    info["Source"]["input"]["required"]["steps"] = ["INT", {"min": 1, "max": 40}]
    info["Source"]["input_order"]["required"].append("steps")
    graph["1"]["inputs"]["steps"] = 23
    schema = {"properties": {"chosen_seed": {"type": "integer", "default": 92}}}
    original = deepcopy((graph, info, schema))

    described = describe_comfyui_api_graph(graph, info, schema)
    bound = bind_compiled_workflow_settings(described, graph, schema)

    assert bound.api_graph["1"]["inputs"]["seed"] == "${chosen_seed}"
    assert bound.input_schema["properties"]["chosen_seed"] == schema["properties"]["chosen_seed"]
    assert bound.api_graph["1"]["inputs"]["steps"] == "${steps}"
    assert bound.input_schema["properties"]["steps"]["default"] == 23
    assert bound.api_graph["1"]["inputs"]["label"] == graph["1"]["inputs"]["label"]
    assert (graph, info, schema) == original


def test_api_connections_and_disconnected_controls_remain_fixed() -> None:
    graph, info = _graph()
    graph["unused"] = deepcopy(graph["1"])
    graph["seed"] = {"class_type": "IntegerSource", "inputs": {}}
    info["IntegerSource"] = {"input": {}, "output": ["INT"]}
    graph["1"]["inputs"]["seed"] = ["seed", 0]

    described = describe_comfyui_api_graph(graph, info, {})
    bound = bind_compiled_workflow_settings(described, graph, {})

    assert bound.api_graph["1"]["inputs"]["seed"] == ["seed", 0]
    assert bound.api_graph["unused"] == graph["unused"]
    fixed = bound.input_schema["x-lm-atelier-graph-settings"]["fixed"]
    assert any(item["node_id"] == "1" and item["reason"] == "linked" for item in fixed)
    assert any(item["node_id"] == "unused" and item["reason"] == "disconnected" for item in fixed)


def test_mapping_does_not_offer_an_alias_for_a_declared_read_only_control() -> None:
    graph, info = _graph()
    schema = {"properties": {"seed": {"type": "integer", "default": 42, "readOnly": True}}}
    described = describe_comfyui_api_graph(graph, info, schema)
    bound = bind_compiled_workflow_settings(described, graph, schema)

    assert bound.api_graph["1"]["inputs"]["seed"] == graph["1"]["inputs"]["seed"]
    assert set(bound.input_schema["properties"]) == {"seed"}
    fixed = bound.input_schema["x-lm-atelier-graph-settings"]["fixed"]
    assert any(item["input_name"] == "seed" and item["reason"] == "read_only" for item in fixed)


def test_api_dynamic_controls_use_only_the_saved_option_contract() -> None:
    graph, info = _graph()
    info["Source"]["input"]["required"]["method"] = [
        "COMFY_DYNAMICCOMBO_V3",
        {
            "options": [
                {"key": "fast", "inputs": {"required": {"steps": ["INT", {"min": 1, "max": 30}]}}},
                {
                    "key": "detailed",
                    "inputs": {"required": {"steps": ["INT", {"min": 31, "max": 50}]}},
                },
            ]
        },
    ]
    info["Source"]["input_order"]["required"].append("method")
    graph["1"]["inputs"].update({"method": "fast", "method.steps": 22})

    described = describe_comfyui_api_graph(graph, info, {})
    bound = bind_compiled_workflow_settings(described, graph, {})

    parameter = bound.api_graph["1"]["inputs"]["method.steps"][2:-1]
    assert bound.input_schema["properties"][parameter]["default"] == 22
    assert bound.input_schema["properties"][parameter]["maximum"] == 30
    assert bound.api_graph["1"]["inputs"]["method"] == "fast"


@pytest.mark.parametrize("damage", ["node", "missing", "unknown", "slot", "cycle"])
def test_api_mapping_refuses_unverifiable_inputs(damage: str) -> None:
    graph, info = _graph()
    if damage == "node":
        graph["1"]["class_type"] = "UnavailableSource"
    elif damage == "missing":
        graph["1"]["inputs"].pop("seed")
    elif damage == "unknown":
        graph["1"]["inputs"]["unknown_control"] = 12
    elif damage == "slot":
        graph["1"]["inputs"]["seed"] = ["1", 99]
    else:
        graph["1"]["inputs"]["seed"] = ["1", 0]
    with pytest.raises(WorkflowCompilationError):
        describe_comfyui_api_graph(graph, info, {})
