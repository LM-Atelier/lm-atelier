from __future__ import annotations

from copy import deepcopy

from test_comfy_workflow_compiler import _object_info, _with_supported_frontend, _workflow

from local_lm.comfy_workflow_compiler import compile_comfyui_ui_graph


def test_compiled_inputs_retain_native_definitions_without_aliasing_the_source() -> None:
    graph = _workflow()
    info = _object_info()
    compiled = compile_comfyui_ui_graph(graph, info)
    inputs = {(item.node_id, item.name): item for item in getattr(compiled, "inputs", ())}
    assert ("1", "seed") in inputs, "Compilation discarded native widget provenance"
    seed = inputs["1", "seed"]
    graph["nodes"][0]["widgets_values"][1] = 99
    info["Source"]["input"]["required"]["seed"][1]["default"] = 99
    assert seed.value == 42 and seed.origin == "widget" and seed.reaches_output
    assert seed.specification == ("INT", {"default": 1, "control_after_generate": True})
    assert seed.node_type == "Source" and seed.node_title == "Source image"


def test_compiled_input_provenance_distinguishes_a_link_from_a_saved_widget() -> None:
    graph = _workflow()
    graph["nodes"][1]["widgets_values"] = ["unused image", "result"]
    info = _object_info()
    info["Save"]["input"]["required"]["images"] = ["STRING"]
    compiled = compile_comfyui_ui_graph(graph, info)
    inputs = {(item.node_id, item.name): item for item in getattr(compiled, "inputs", ())}
    assert ("2", "images") in inputs, "Compilation discarded connected widget provenance"
    value = inputs["2", "images"]
    assert value.origin == "link" and value.value == ["1", 0] and value.reaches_output


def test_compiled_input_provenance_retains_a_fixed_primitive_origin() -> None:
    graph = _with_supported_frontend(_workflow())
    graph["nodes"][0]["inputs"] = [
        {"name": "label", "type": "STRING", "widget": {"name": "label"}, "link": 8}
    ]
    graph["nodes"].append(
        {
            "id": 3,
            "type": "PrimitiveNode",
            "mode": 0,
            "properties": {"Run widget replace on values": False},
            "inputs": [],
            "outputs": [
                {"name": "STRING", "type": "STRING", "widget": {"name": "label"}, "links": [8]}
            ],
            "widgets_values": ["fixed label"],
        }
    )
    graph["links"].append([8, 3, 0, 1, 0, "STRING"])
    compiled = compile_comfyui_ui_graph(graph, _object_info())
    inputs = {(item.node_id, item.name): item for item in getattr(compiled, "inputs", ())}
    assert ("1", "label") in inputs, "Compilation discarded fixed primitive provenance"
    value = inputs["1", "label"]
    assert value.origin == "primitive" and value.value == "fixed label"
    assert value.reaches_output and "3" not in compiled.api_graph


def test_compiled_inputs_identify_a_node_that_cannot_reach_an_output() -> None:
    graph = _workflow()
    isolated = deepcopy(graph["nodes"][0])
    isolated["id"] = 3
    isolated["outputs"][0]["links"] = []
    graph["nodes"].append(isolated)
    compiled = compile_comfyui_ui_graph(graph, _object_info())
    inputs = {(item.node_id, item.name): item for item in getattr(compiled, "inputs", ())}
    assert ("3", "seed") in inputs, "Compilation discarded disconnected widget provenance"
    assert not inputs["3", "seed"].reaches_output
    assert inputs["1", "seed"].reaches_output
