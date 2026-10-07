from copy import deepcopy
from typing import Any

import pytest
from test_workflow_package_import_endpoint import _object_info, _ui_graph

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.comfy_workflow_compiler import compile_comfyui_ui_graph
from local_lm.workflow_package_inputs import WorkflowPackageInputError
from local_lm.workflow_supplied_settings import bind_supplied_workflow_settings


def test_supplied_mapping_preserves_explicit_inputs_and_maps_native_values() -> None:
    info, graph = _object_info(), _ui_graph()
    info["Source"]["input"]["required"]["steps"] = ["INT", {"min": 1, "max": 50}]
    info["Source"]["input_order"]["required"].append("steps")
    graph["nodes"][0]["widgets_values"].append(27)
    compilation = compile_comfyui_ui_graph(graph, info)
    supplied: dict[str, Any] = deepcopy(compilation.api_graph)
    supplied["1"]["inputs"]["seed"] = "${seed}"
    schema = {"properties": {"seed": {"type": "integer", "default": 63}}}
    before = deepcopy(supplied)
    mapped = bind_supplied_workflow_settings(compilation, supplied, schema)
    assert mapped.input_schema["properties"]["seed"] == schema["properties"]["seed"]
    assert mapped.input_schema["properties"]["steps"]["default"] == 27
    dispatched = ComfyUIAdapter._compile(mapped.api_graph, {"seed": 99, "steps": 31})
    assert dispatched["1"]["inputs"]["seed"] == 99
    assert dispatched["1"]["inputs"]["steps"] == 31
    assert supplied == before
    assert mapped.api_graph["1"].get("_meta") == supplied["1"].get("_meta")
    assert set(schema["properties"]) == {"seed"}


@pytest.mark.parametrize("damage", ["value", "link", "undeclared", "type"])
def test_supplied_mapping_refuses_executable_disagreement(damage: str) -> None:
    compilation = compile_comfyui_ui_graph(_ui_graph(), _object_info())
    supplied: dict[str, Any] = deepcopy(compilation.api_graph)
    if damage == "value":
        supplied["1"]["inputs"]["seed"] = 99
    elif damage == "link":
        supplied["2"]["inputs"]["images"] = "${seed}"
    elif damage == "undeclared":
        supplied["1"]["inputs"]["seed"] = "${missing}"
    else:
        supplied["1"]["class_type"] = "DifferentSource"
    with pytest.raises(WorkflowPackageInputError, match="visual workflow does not match"):
        bind_supplied_workflow_settings(
            compilation, supplied, {"properties": {"seed": {"type": "integer"}}}
        )


def test_supplied_mapping_regenerates_generated_defaults() -> None:
    graph, info = _ui_graph(), _object_info()
    original = compile_comfyui_ui_graph(graph, info)
    mapped = bind_supplied_workflow_settings(original, original.api_graph, {})
    graph["nodes"][0]["widgets_values"][1] = 88
    changed = compile_comfyui_ui_graph(graph, info)
    rebuilt = bind_supplied_workflow_settings(changed, mapped.api_graph, mapped.input_schema)
    assert rebuilt.input_schema["properties"]["seed"]["default"] == 88
    assert mapped.input_schema["properties"]["seed"]["default"] == 42
