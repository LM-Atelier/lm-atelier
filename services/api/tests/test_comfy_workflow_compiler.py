from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from local_lm.comfy_workflow_compiler import (
    WorkflowCompilationError,
    compile_comfyui_ui_graph,
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


def _workflow() -> dict[str, Any]:
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


def _with_supported_frontend(workflow: dict[str, Any]) -> dict[str, Any]:
    workflow["extra"] = {"frontendVersion": "1.45.21"}
    return workflow


def _assert_error(code: str, workflow: dict[str, Any], object_info: dict[str, Any]) -> None:
    with pytest.raises(WorkflowCompilationError) as raised:
        compile_comfyui_ui_graph(workflow, object_info)
    assert raised.value.code == code


def test_compiles_widgets_links_metadata_and_execution_order() -> None:
    compiled = compile_comfyui_ui_graph(_workflow(), _object_info())

    assert compiled.execution_order == ("1", "2")
    assert compiled.api_graph == {
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


def test_uses_object_info_input_order_instead_of_mapping_order() -> None:
    workflow = _workflow()
    workflow["nodes"] = [
        {
            "id": "ordered",
            "type": "Ordered",
            "inputs": [],
            "outputs": [],
            "widgets_values": ["first value", "second value"],
        }
    ]
    workflow["links"] = []
    object_info = {
        "Ordered": {
            "input": {
                "required": {
                    "second": ["STRING"],
                    "first": ["STRING"],
                }
            },
            "input_order": {"required": ["first", "second"]},
        }
    }

    compiled = compile_comfyui_ui_graph(workflow, object_info)

    assert compiled.api_graph["ordered"]["inputs"] == {
        "first": "first value",
        "second": "second value",
    }


def test_link_overrides_widget_value_after_consuming_it() -> None:
    workflow = _workflow()
    workflow["nodes"][1]["widgets_values"] = ["unused image", "result"]
    object_info = _object_info()
    object_info["Save"]["input"]["required"]["images"] = ["STRING"]

    compiled = compile_comfyui_ui_graph(workflow, object_info)

    assert compiled.api_graph["2"]["inputs"] == {
        "images": ["1", 0],
        "filename_prefix": "result",
    }


def test_preserves_array_widget_values_as_arrays() -> None:
    workflow = _workflow()
    workflow["nodes"] = [
        {
            "id": 1,
            "type": "ArrayWidget",
            "inputs": [],
            "outputs": [],
            "widgets_values": [[1, 2, 3]],
        }
    ]
    workflow["links"] = []
    object_info = {
        "ArrayWidget": {
            "input": {"required": {"values": ["STRING"]}},
            "input_order": {"required": ["values"]},
        }
    }

    compiled = compile_comfyui_ui_graph(workflow, object_info)

    assert compiled.api_graph["1"]["inputs"] == {"values": [1, 2, 3]}


def test_ignores_unlinked_note_nodes() -> None:
    workflow = _workflow()
    workflow["nodes"].append(
        {"id": 3, "type": "Note", "inputs": [], "outputs": [], "widgets_values": ["note"]}
    )

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.execution_order == ("1", "2")
    assert "3" not in compiled.api_graph


@pytest.mark.parametrize("node_type", ["MarkdownNote", "Note"])
@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_ignores_frontend_modes_on_unlinked_note_nodes(node_type: str, mode: int) -> None:
    workflow = _workflow()
    workflow["nodes"].append(
        {
            "id": 3,
            "type": node_type,
            "mode": mode,
            "inputs": [],
            "outputs": [],
            "widgets_values": ["note"],
        }
    )

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.execution_order == ("1", "2")
    assert "3" not in compiled.api_graph


def test_rejects_workflows_without_executable_nodes() -> None:
    workflow = {
        "version": 0.4,
        "nodes": [
            {"id": 1, "type": "Note", "inputs": [], "outputs": [], "widgets_values": ["note"]}
        ],
        "links": [],
    }

    _assert_error("empty_executable_workflow", workflow, {})


def test_a_subgraph_that_cannot_be_expanded_exactly_still_refuses() -> None:
    workflow = _workflow()
    # An instance whose definition declares an input nothing feeds. Expansion
    # refuses rather than inventing a source, because a guess here compiles,
    # runs, and produces a picture that is not the one the author drew.
    workflow["definitions"] = {
        "subgraphs": [
            {
                "id": "group",
                "nodes": [{"id": 7, "type": "Save", "inputs": [], "outputs": []}],
                "links": [[70, "-10", 0, 7, 0, "IMAGE"]],
            }
        ]
    }
    workflow["nodes"].append({"id": 9, "type": "group", "inputs": [], "outputs": []})
    _assert_error("unconnected_subgraph_input", workflow, _object_info())


def test_compiles_a_fixed_primitive_widget_value() -> None:
    workflow = _with_supported_frontend(_workflow())
    workflow["nodes"][0]["inputs"] = [
        {
            "name": "label",
            "type": "STRING",
            "widget": {"name": "label"},
            "link": 8,
        }
    ]
    workflow["nodes"].append(
        {
            "id": 3,
            "type": "PrimitiveNode",
            "mode": 0,
            "properties": {"Run widget replace on values": False},
            "inputs": [],
            "outputs": [
                {
                    "name": "STRING",
                    "type": "STRING",
                    "widget": {"name": "label"},
                    "links": [8],
                }
            ],
            "widgets_values": ["from primitive"],
        }
    )
    workflow["links"].append([8, 3, 0, 1, 0, "STRING"])

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.execution_order == ("1", "2")
    assert compiled.api_graph["1"]["inputs"]["label"] == "from primitive"
    assert "3" not in compiled.api_graph


def test_rejects_malformed_primitive_nodes() -> None:
    workflow = _with_supported_frontend(_workflow())
    workflow["nodes"].append(
        {"id": 3, "type": "PrimitiveNode", "inputs": [], "outputs": [], "widgets_values": [1]}
    )
    _assert_error("unsupported_primitive_node", workflow, _object_info())


@pytest.mark.parametrize("node_type", ["PrimitiveNode", "Reroute"])
@pytest.mark.parametrize(
    "frontend_version",
    [None, "invalid", "1.45.20", "1.45.22", "1.45.21rc1"],
)
def test_frontend_semantics_require_a_certified_declared_version(
    node_type: str,
    frontend_version: str | None,
) -> None:
    workflow = _workflow()
    if frontend_version is not None:
        workflow["extra"] = {"frontendVersion": frontend_version}
    workflow["nodes"].append({"id": 42, "type": node_type, "mode": 0, "inputs": [], "outputs": []})

    with pytest.raises(WorkflowCompilationError) as raised:
        compile_comfyui_ui_graph(workflow, _object_info())
    assert raised.value.code == "unsupported_frontend_version"
    assert str(raised.value) == (
        "workflow uses PrimitiveNode or Reroute semantics without a certified "
        "ComfyUI frontend version"
    )


@pytest.mark.parametrize("frontend_version", [None, "invalid", "1.45.20", "1.45.22"])
def test_plain_headless_graph_does_not_require_an_editor_frontend_version(
    frontend_version: str | None,
) -> None:
    workflow = _workflow()
    if frontend_version is not None:
        workflow["extra"] = {"frontendVersion": frontend_version}

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.execution_order == ("1", "2")


@pytest.mark.parametrize(
    ("properties", "values"),
    [
        ({"Run widget replace on values": True}, ["value"]),
        ({"Run widget replace on values": False}, [1, "randomize"]),
    ],
)
def test_rejects_dynamic_primitive_widget_values(
    properties: dict[str, bool],
    values: list[object],
) -> None:
    workflow = _with_supported_frontend(_workflow())
    workflow["nodes"].append(
        {
            "id": 3,
            "type": "PrimitiveNode",
            "properties": properties,
            "inputs": [],
            "outputs": [
                {
                    "name": "STRING",
                    "type": "STRING",
                    "widget": {"name": "label"},
                    "links": [],
                }
            ],
            "widgets_values": values,
        }
    )

    _assert_error("unsupported_primitive_node", workflow, _object_info())


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_rejects_frontend_execution_modes(mode: int) -> None:
    # Mode 4 is no longer here: a bypassed node is routed around rather than
    # refused, which is what makes authored graphs that use it runnable.
    # Muted and the trigger modes still mean something the runtime cannot be
    # handed, so they still refuse.
    workflow = _workflow()
    workflow["nodes"][0]["mode"] = mode

    _assert_error("unsupported_node_mode", workflow, _object_info())


def test_rejects_dangling_links_and_missing_slots() -> None:
    workflow = _workflow()
    workflow["links"][0][1] = 99
    _assert_error("dangling_link", workflow, _object_info())

    workflow = _workflow()
    workflow["links"][0][4] = 9
    _assert_error("invalid_link_slot", workflow, _object_info())


def test_rejects_inconsistent_link_metadata() -> None:
    workflow = _workflow()
    workflow["nodes"][1]["inputs"][0]["link"] = 8

    _assert_error("inconsistent_link", workflow, _object_info())

    workflow = _workflow()
    workflow["nodes"][1]["inputs"].append({"name": "other", "type": "IMAGE", "link": 7})
    object_info = _object_info()
    object_info["Save"]["input"]["optional"] = {"other": ["IMAGE"]}
    object_info["Save"]["input_order"]["optional"] = ["other"]
    _assert_error("inconsistent_link", workflow, object_info)


def test_rejects_multiple_links_to_one_input() -> None:
    workflow = _workflow()
    workflow["nodes"].insert(
        1,
        {
            "id": 3,
            "type": "Source",
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [8]}],
            "widgets_values": ["second", 3, "fixed"],
        },
    )
    workflow["nodes"][2]["inputs"][0]["link"] = None
    workflow["links"].append([8, 3, 0, 2, 0, "IMAGE"])

    _assert_error("duplicate_input_link", workflow, _object_info())


def test_rejects_dependency_cycles() -> None:
    workflow = {
        "version": 0.4,
        "nodes": [
            {
                "id": "a",
                "type": "Pipe",
                "inputs": [{"name": "image", "type": "IMAGE", "link": 2}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
                "widgets_values": [],
            },
            {
                "id": "b",
                "type": "Pipe",
                "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]}],
                "widgets_values": [],
            },
        ],
        "links": [
            [1, "a", 0, "b", 0, "IMAGE"],
            [2, "b", 0, "a", 0, "IMAGE"],
        ],
    }
    object_info = {
        "Pipe": {
            "input": {"required": {"image": ["IMAGE"]}},
            "input_order": {"required": ["image"]},
        }
    }

    _assert_error("workflow_cycle", workflow, object_info)


def test_rejects_links_attached_to_note_nodes() -> None:
    workflow = _workflow()
    workflow["nodes"][0] = {
        "id": 1,
        "type": "Note",
        "inputs": [],
        "outputs": [{"name": "value", "type": "IMAGE", "links": [7]}],
        "widgets_values": ["not executable"],
    }

    _assert_error("frontend_node_link", workflow, _object_info())


def test_rejects_missing_required_connection_input() -> None:
    workflow = _workflow()
    workflow["nodes"] = [workflow["nodes"][1]]
    workflow["nodes"][0]["inputs"] = [workflow["nodes"][0]["inputs"][1]]
    workflow["links"] = []

    _assert_error("missing_required_input", workflow, _object_info())


def test_rejects_unknown_input_slots() -> None:
    workflow = _workflow()
    workflow["nodes"][0]["inputs"] = [{"name": "surprise", "type": "STRING"}]

    _assert_error("unknown_input_slot", workflow, _object_info())


def test_rejects_ambiguous_widget_metadata() -> None:
    workflow = _workflow()
    workflow["nodes"][1]["inputs"][1]["widget"] = {"name": "different"}

    _assert_error("ambiguous_widget", workflow, _object_info())


def test_rejects_widget_values_that_do_not_map_exactly() -> None:
    workflow = _workflow()
    workflow["nodes"][0]["widgets_values"].append("extra")

    _assert_error("unsupported_widget_values", workflow, _object_info())


def test_rejects_invalid_combo_choices() -> None:
    workflow = _workflow()
    workflow["nodes"] = [
        {
            "id": 1,
            "type": "Choice",
            "inputs": [],
            "outputs": [],
            "widgets_values": ["missing"],
        }
    ]
    workflow["links"] = []
    object_info = {
        "Choice": {
            "input": {"required": {"choice": [["available"]]}},
            "input_order": {"required": ["choice"]},
        }
    }

    _assert_error("invalid_widget_choice", workflow, object_info)


def test_rejects_custom_widget_serialization() -> None:
    workflow = _workflow()
    workflow["nodes"] = [
        {
            "id": 1,
            "type": "CustomWidget",
            "inputs": [{"name": "value", "type": "CUSTOM", "widget": {"name": "value"}}],
            "outputs": [],
            "widgets_values": ["value"],
        }
    ]
    workflow["links"] = []
    object_info = {
        "CustomWidget": {
            "input": {"required": {"value": ["CUSTOM"]}},
            "input_order": {"required": ["value"]},
        }
    }

    _assert_error("unsupported_widget", workflow, object_info)


def test_rejects_ambiguous_object_info_order() -> None:
    workflow = _workflow()
    object_info = _object_info()
    object_info["Source"]["input_order"]["required"] = ["label"]

    _assert_error("ambiguous_input_order", workflow, object_info)


def test_rejects_missing_runtime_node_type() -> None:
    workflow = deepcopy(_workflow())
    object_info = _object_info()
    del object_info["Source"]

    _assert_error("missing_node_type", workflow, object_info)


def _rerouted(count: int) -> dict[str, Any]:
    """The same workflow, with a chain of reroutes standing in the one wire."""

    workflow = _with_supported_frontend(_workflow())
    source, save = workflow["nodes"]
    chain = list(range(100, 100 + count))
    source["outputs"][0]["links"] = [chain[0]]
    workflow["links"] = []
    for index, link_id in enumerate(chain):
        origin = ("1", 0) if index == 0 else (str(chain[index - 1] + 1000), 0)
        node_id = link_id + 1000
        workflow["nodes"].append(
            {
                "id": node_id,
                "type": "Reroute",
                "mode": 0,
                "inputs": [{"name": "", "type": "*", "link": link_id}],
                "outputs": [{"name": "", "type": "IMAGE", "links": []}],
            }
        )
        workflow["links"].append([link_id, origin[0], origin[1], node_id, 0, "IMAGE"])
    last = chain[-1] + 1000
    workflow["nodes"][-1]["outputs"][0]["links"] = [7]
    workflow["links"].append([7, last, 0, 2, 0, "IMAGE"])
    del save
    return workflow


@pytest.mark.parametrize("length", [1, 2, 5])
def test_a_reroute_chain_compiles_to_the_same_graph_as_the_wire(length: int) -> None:
    """A reroute exists to make a graph readable. Dropping one would take its
    edge with it and yield a graph that runs and quietly makes a different
    picture, so it is resolved instead - and the result has to be the graph the
    author would have drawn without it."""

    direct = compile_comfyui_ui_graph(_workflow(), _object_info())
    through = compile_comfyui_ui_graph(_rerouted(length), _object_info())

    assert through.api_graph == direct.api_graph
    assert through.execution_order == direct.execution_order


def test_a_reroute_feeding_several_targets_reconnects_each_of_them() -> None:
    workflow = _with_supported_frontend(_workflow())
    workflow["nodes"][0]["outputs"][0]["links"] = [100]
    workflow["nodes"].append(
        {
            "id": 3,
            "type": "Save",
            "mode": 0,
            "inputs": [
                {"name": "images", "type": "IMAGE", "link": 8},
                {
                    "name": "filename_prefix",
                    "type": "STRING",
                    "widget": {"name": "filename_prefix"},
                },
            ],
            "outputs": [],
            "widgets_values": ["second"],
        }
    )
    workflow["nodes"].append(
        {
            "id": 9,
            "type": "Reroute",
            "mode": 0,
            "inputs": [{"name": "", "type": "*", "link": 100}],
            "outputs": [{"name": "", "type": "IMAGE", "links": [7, 8]}],
        }
    )
    workflow["links"] = [
        [100, 1, 0, 9, 0, "IMAGE"],
        [7, 9, 0, 2, 0, "IMAGE"],
        [8, 9, 0, 3, 0, "IMAGE"],
    ]

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.api_graph["2"]["inputs"]["images"] == ["1", 0]
    assert compiled.api_graph["3"]["inputs"]["images"] == ["1", 0]
    assert "9" not in compiled.api_graph


def test_a_reroute_with_consumers_and_nothing_feeding_it_is_refused() -> None:
    """The one case that must not be dropped quietly: removing it would leave a
    required input unfilled."""

    workflow = _rerouted(1)
    workflow["links"] = [link for link in workflow["links"] if link[0] != 100]
    workflow["nodes"][2]["inputs"][0]["link"] = None

    _assert_error("unconnected_pass_through", workflow, _object_info())


def test_a_reroute_attached_to_nothing_is_simply_dropped() -> None:
    workflow = _with_supported_frontend(_workflow())
    workflow["nodes"].append({"id": 42, "type": "Reroute", "mode": 0, "inputs": [], "outputs": []})

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.api_graph == compile_comfyui_ui_graph(_workflow(), _object_info()).api_graph
