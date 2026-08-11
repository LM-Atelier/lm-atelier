from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from local_lm.comfy_workflow_compiler import (
    WorkflowCompilationError,
    compile_comfyui_ui_graph,
)
from local_lm.comfy_workflow_packages import WorkflowPackageError


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


def _assert_refused_before_compiling(
    code: str, workflow: dict[str, Any], object_info: dict[str, Any]
) -> None:
    """Refused by the graph analysis the compiler runs first, not by compilation.

    Still reported as a compilation refusal, because that is what a caller asked
    for and everything this function raises means the same thing to them. The
    code carries which check it was.
    """

    with pytest.raises(WorkflowPackageError) as raised:
        compile_comfyui_ui_graph(workflow, object_info)
    assert isinstance(raised.value, WorkflowCompilationError)
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


_UNREAD_NODE_TYPE = "Fancy Stack (someone)"


def _package_drawn_widgets(
    properties: dict[str, Any] | None,
    node_type: str = _UNREAD_NODE_TYPE,
    widgets_values: list[Any] | None = None,
) -> dict[str, Any]:
    """A node whose settings live in objects only its own package understands.

    Shaped after rgthree's Power Lora Loader: the runtime declares the links and
    nothing else, because the node takes any number of further inputs, and the
    saved graph carries them as objects interleaved with the package's own
    canvas furniture. The default node type is deliberately one no layout is
    transcribed for, so these prove the refusal rather than a transcription.
    """

    workflow = _workflow()
    node: dict[str, Any] = {
        "id": 3,
        "type": node_type,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "widgets_values": (
            [
                {},
                {"type": "PowerLoraLoaderHeaderWidget"},
                {"on": True, "lora": "one.safetensors", "strength": 1, "strengthTwo": None},
                {},
                "",
            ]
            if widgets_values is None
            else widgets_values
        ),
    }
    if properties is not None:
        node["properties"] = properties
    workflow["nodes"].append(node)
    return workflow


def _drawn_object_info() -> dict[str, Any]:
    object_info = _object_info()
    definition = {
        "input": {"optional": {"model": ["MODEL"], "clip": ["CLIP"]}},
        "input_order": {"optional": ["model", "clip"]},
        "output": ["MODEL", "CLIP"],
    }
    object_info[_UNREAD_NODE_TYPE] = definition
    object_info["Power Lora Loader (rgthree)"] = definition
    return object_info


def test_widgets_only_a_package_can_read_are_refused_as_that_and_not_as_a_mapping_fault() -> None:
    """A leftover scalar is a mapping this compiler got wrong. A leftover object
    is a widget the node's own package draws, whose layout is described nowhere
    the runtime can be asked. Reporting the second as the first sends someone
    looking for a defect in the compiler."""

    workflow = _package_drawn_widgets({"cnr_id": "someone-comfy", "ver": "a" * 40})

    with pytest.raises(WorkflowCompilationError) as raised:
        compile_comfyui_ui_graph(workflow, _drawn_object_info())

    assert raised.value.code == "package_serialized_widgets"
    message = str(raised.value)
    assert _UNREAD_NODE_TYPE in message
    assert f"someone-comfy at {'a' * 40}" in message


@pytest.mark.parametrize(
    ("properties", "expected"),
    [
        ({"cnr_id": "someone-comfy"}, "someone-comfy"),
        ({"aux_id": "someone/fancy-stack"}, "someone/fancy-stack"),
        ({}, "its package"),
        (None, "its package"),
    ],
)
def test_a_package_drawn_widget_names_only_the_package_the_graph_records(
    properties: dict[str, Any] | None, expected: str
) -> None:
    """Taken from the saved graph rather than from anything installed, because
    the question is whose editor code would have to be reproduced. Naming the
    wrong package would be worse than naming none."""

    with pytest.raises(WorkflowCompilationError) as raised:
        compile_comfyui_ui_graph(_package_drawn_widgets(properties), _drawn_object_info())

    assert expected in str(raised.value)


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


def _named_wire(label: str = "image") -> dict[str, Any]:
    """The same workflow, with the one wire written and read by name."""

    workflow = _workflow()
    workflow["nodes"][0]["outputs"][0]["links"] = [100]
    workflow["nodes"].append(
        {
            "id": 10,
            "type": "SetNode",
            "mode": 0,
            "inputs": [{"name": "IMAGE", "type": "IMAGE", "link": 100}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": None}],
            "widgets_values": [label],
        }
    )
    workflow["nodes"].append(
        {
            "id": 11,
            "type": "GetNode",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [7]}],
            "widgets_values": [label],
        }
    )
    workflow["links"] = [
        [100, 1, 0, 10, 0, "IMAGE"],
        [7, 11, 0, 2, 0, "IMAGE"],
    ]
    return workflow


def test_a_named_wire_compiles_to_the_same_graph_as_the_wire() -> None:
    """A `SetNode` labels the link feeding it and a `GetNode` re-emits it. The
    pair is frontend JavaScript that no runtime can report, so counting them as
    runtime nodes made every graph using the idiom unimportable - and resolving
    them has to produce the graph the author would have drawn with one wire.

    Compiled without a declared frontend version on purpose: these are a
    package's construct rather than the editor's, so the certified-frontend gate
    does not apply to them.
    """

    direct = compile_comfyui_ui_graph(_workflow(), _object_info())
    named = compile_comfyui_ui_graph(_named_wire(), _object_info())

    assert named.api_graph == direct.api_graph
    assert named.execution_order == direct.execution_order


def test_one_written_value_feeds_every_reader_of_it() -> None:
    workflow = _named_wire()
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
            "id": 12,
            "type": "GetNode",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [8]}],
            "widgets_values": ["image"],
        }
    )
    workflow["links"].append([8, 12, 0, 3, 0, "IMAGE"])

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.api_graph["2"]["inputs"]["images"] == ["1", 0]
    assert compiled.api_graph["3"]["inputs"]["images"] == ["1", 0]
    assert "11" not in compiled.api_graph
    assert "12" not in compiled.api_graph


def test_a_value_written_through_a_reroute_resolves_through_both() -> None:
    """Carriers are resolved by one walk, so a wire may cross a reroute and a
    label in either order and still arrive at its real producer."""

    workflow = _with_supported_frontend(_named_wire())
    workflow["nodes"][0]["outputs"][0]["links"] = [200]
    workflow["nodes"].append(
        {
            "id": 20,
            "type": "Reroute",
            "mode": 0,
            "inputs": [{"name": "", "type": "*", "link": 200}],
            "outputs": [{"name": "", "type": "IMAGE", "links": [100]}],
        }
    )
    workflow["links"] = [
        [200, 1, 0, 20, 0, "IMAGE"],
        [100, 20, 0, 10, 0, "IMAGE"],
        [7, 11, 0, 2, 0, "IMAGE"],
    ]

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.api_graph["2"]["inputs"]["images"] == ["1", 0]
    assert set(compiled.api_graph) == {"1", "2"}


def test_reading_a_value_nothing_writes_is_refused() -> None:
    """Dropping it silently would leave a required input unfilled."""

    workflow = _named_wire()
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] != 10]
    workflow["links"] = [link for link in workflow["links"] if link[0] != 100]
    workflow["nodes"][0]["outputs"][0]["links"] = []

    _assert_error("undefined_named_wire", workflow, _object_info())


def test_reading_a_value_two_nodes_write_is_refused() -> None:
    """Two writers of one label have no correct reading, and choosing either
    would compile a graph the author never drew."""

    workflow = _named_wire()
    workflow["nodes"][0]["outputs"][0]["links"] = [100, 101]
    workflow["nodes"].append(
        {
            "id": 13,
            "type": "SetNode",
            "mode": 0,
            "inputs": [{"name": "IMAGE", "type": "IMAGE", "link": 101}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": None}],
            "widgets_values": ["image"],
        }
    )
    workflow["links"].append([101, 1, 0, 13, 0, "IMAGE"])

    _assert_error("duplicate_named_wire", workflow, _object_info())


def test_a_value_written_twice_but_never_read_still_compiles() -> None:
    """Ambiguity only exists where something reads it. The editor runs this
    graph, so refusing it would reject a workflow that works."""

    workflow = _named_wire()
    workflow["nodes"][0]["outputs"][0]["links"] = [100, 101]
    workflow["nodes"].append(
        {
            "id": 13,
            "type": "SetNode",
            "mode": 0,
            "inputs": [{"name": "IMAGE", "type": "IMAGE", "link": 101}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": None}],
            "widgets_values": ["unread"],
        }
    )
    workflow["links"].append([101, 1, 0, 13, 0, "IMAGE"])

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.api_graph["2"]["inputs"]["images"] == ["1", 0]
    assert set(compiled.api_graph) == {"1", "2"}


def test_reading_a_value_whose_writer_carries_nothing_names_the_writer() -> None:
    """The reader found its writer; the writer is what is empty. Reporting the
    reader would send someone to the end of the wire that is intact."""

    workflow = _named_wire()
    workflow["links"] = [link for link in workflow["links"] if link[0] != 100]
    workflow["nodes"][0]["outputs"][0]["links"] = []
    workflow["nodes"][2]["inputs"][0]["link"] = None

    with pytest.raises(WorkflowCompilationError) as raised:
        compile_comfyui_ui_graph(workflow, _object_info())

    assert raised.value.code == "unconnected_pass_through"
    assert "workflow value image" in str(raised.value)
    assert "the read of" not in str(raised.value)


@pytest.mark.parametrize("values", [[], [""], [None], "image", None])
def test_a_named_wire_without_a_readable_name_is_refused(values: Any) -> None:
    """It cannot be matched to its other end, and guessing which end it meant
    would connect two things the author did not."""

    workflow = _named_wire()
    workflow["nodes"][3]["widgets_values"] = values

    _assert_error("invalid_named_wire", workflow, _object_info())


def test_a_value_that_reads_itself_is_refused() -> None:
    workflow = _named_wire()
    workflow["nodes"][0]["outputs"][0]["links"] = []
    workflow["nodes"][3]["outputs"][0]["links"] = [7, 100]
    workflow["links"] = [
        [100, 11, 0, 10, 0, "IMAGE"],
        [7, 11, 0, 2, 0, "IMAGE"],
    ]

    _assert_error("pass_through_cycle", workflow, _object_info())


_RGTHREE = {"cnr_id": "rgthree-comfy", "ver": "6b76ee6f2c5a007710b5a16f97c94330d6ecc871"}


def _power_loras(*entries: Any, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    """The node as rgthree saves it: divider, header, the loras, spacer, button."""

    return _package_drawn_widgets(
        _RGTHREE if properties is None else properties,
        node_type="Power Lora Loader (rgthree)",
        widgets_values=[{}, {"type": "PowerLoraLoaderHeaderWidget"}, *entries, {}, ""],
    )


def _compiled_loras(*entries: Any) -> dict[str, Any]:
    compiled = compile_comfyui_ui_graph(_power_loras(*entries), _drawn_object_info())
    return dict(compiled.api_graph["3"]["inputs"])


def test_each_saved_lora_becomes_the_input_the_node_names_it() -> None:
    """rgthree names them lora_1, lora_2 and so on in the order they appear,
    counting from one, and its Python reads whatever keys it is given."""

    inputs = _compiled_loras(
        {"on": True, "lora": "first.safetensors", "strength": 1},
        {"on": True, "lora": "second.safetensors", "strength": 0.5},
    )

    assert inputs == {
        "lora_1": {"on": True, "lora": "first.safetensors", "strength": 1},
        "lora_2": {"on": True, "lora": "second.safetensors", "strength": 0.5},
    }


def test_a_lora_switched_off_is_still_sent() -> None:
    """Its own Python decides what an off entry does. Dropping it here would
    silently rewrite a graph whose author left it in place to turn back on."""

    inputs = _compiled_loras({"on": False, "lora": "held.safetensors", "strength": 1})

    assert inputs == {"lora_1": {"on": False, "lora": "held.safetensors", "strength": 1}}


@pytest.mark.parametrize("strength_two", [None, 0.8])
def test_a_separate_clip_strength_is_carried_through_exactly(strength_two: Any) -> None:
    """Absent and null mean the same thing to the node, so neither is invented
    and neither is dropped."""

    inputs = _compiled_loras(
        {"on": True, "lora": "one.safetensors", "strength": 1, "strengthTwo": strength_two}
    )

    assert inputs["lora_1"]["strengthTwo"] == strength_two


def test_a_node_with_no_loras_compiles_to_no_lora_inputs() -> None:
    compiled = compile_comfyui_ui_graph(_power_loras(), _drawn_object_info())

    assert compiled.api_graph["3"]["inputs"] == {}


def test_the_links_the_node_carries_survive_its_widgets() -> None:
    """A partial conversion would run and mean something different."""

    workflow = _power_loras({"on": True, "lora": "one.safetensors", "strength": 1})
    workflow["nodes"][0]["outputs"][0]["links"] = [7, 50]
    workflow["nodes"][2]["inputs"] = [{"name": "model", "type": "MODEL", "link": 50}]
    workflow["links"].append([50, 1, 0, 3, 0, "MODEL"])

    compiled = compile_comfyui_ui_graph(workflow, _drawn_object_info())

    assert compiled.api_graph["3"]["inputs"]["model"] == ["1", 0]
    assert compiled.api_graph["3"]["inputs"]["lora_1"]["lora"] == "one.safetensors"


@pytest.mark.parametrize(
    "widgets_values",
    [
        [{}],
        [{}, {"type": "PowerLoraLoaderHeaderWidget"}],
        [{"type": "PowerLoraLoaderHeaderWidget"}, {}, {}, ""],
        [{}, {"type": "SomethingElse"}, {}, ""],
        [{}, {"type": "PowerLoraLoaderHeaderWidget"}, {}, "add"],
        [{}, {"type": "PowerLoraLoaderHeaderWidget"}, "", {}],
    ],
)
def test_a_layout_this_build_does_not_recognise_refuses(widgets_values: list[Any]) -> None:
    """A package that moved its furniture would otherwise be read under the old
    layout and produce a graph that runs and means something else."""

    workflow = _package_drawn_widgets(
        _RGTHREE,
        node_type="Power Lora Loader (rgthree)",
        widgets_values=widgets_values,
    )

    _assert_error("package_widget_layout", workflow, _drawn_object_info())


@pytest.mark.parametrize(
    "entry",
    [
        {"lora": "one.safetensors", "strength": 1},
        {"on": True, "strength": 1},
        {"on": True, "lora": "one.safetensors"},
        {"on": True, "lora": "", "strength": 1},
        {"on": True, "lora": 7, "strength": 1},
        {"on": "yes", "lora": "one.safetensors", "strength": 1},
        {"on": True, "lora": "one.safetensors", "strength": "1"},
        {"on": True, "lora": "one.safetensors", "strength": True},
        {"on": True, "lora": "one.safetensors", "strength": 1, "strengthTwo": "half"},
        {"on": True, "lora": "one.safetensors", "strength": 1, "extra": 1},
        "not an entry at all",
    ],
)
def test_an_entry_that_is_not_the_transcribed_shape_refuses(entry: Any) -> None:
    """Read exactly or not at all. A strength of True is an int in Python and
    would quietly become 1.0, so it is reported rather than accepted."""

    workflow = _package_drawn_widgets(
        _RGTHREE,
        node_type="Power Lora Loader (rgthree)",
        widgets_values=[{}, {"type": "PowerLoraLoaderHeaderWidget"}, entry, {}, ""],
    )

    _assert_error("package_widget_layout", workflow, _drawn_object_info())


def test_the_same_node_type_from_another_package_is_not_read() -> None:
    """The layout belongs to the package that draws it, and a graph naming a
    different package is not making the same claim."""

    workflow = _power_loras(
        {"on": True, "lora": "one.safetensors", "strength": 1},
        properties={"cnr_id": "someone-else", "ver": "b" * 40},
    )

    _assert_error("package_serialized_widgets", workflow, _drawn_object_info())


def _named_widgets() -> dict[str, Any]:
    """The same workflow, with each widget saved under the input it belongs to."""

    workflow = _workflow()
    workflow["nodes"][0]["widgets_values"] = {
        "label": "camera",
        "seed": 42,
        "control_after_generate": "randomize",
    }
    workflow["nodes"][1]["widgets_values"] = {"filename_prefix": "result"}
    return workflow


def test_widgets_saved_by_name_compile_to_the_same_graph_as_widgets_saved_in_order() -> None:
    """The named shape says outright what the positional one only implies, so it
    is read by name and never counted - a node that gained or lost a widget
    shifts every position after it, which is what naming them avoids."""

    positional = compile_comfyui_ui_graph(_workflow(), _object_info())
    named = compile_comfyui_ui_graph(_named_widgets(), _object_info())

    assert named.api_graph == positional.api_graph
    assert named.execution_order == positional.execution_order


def test_a_name_the_graph_did_not_save_falls_back_to_the_declared_default() -> None:
    workflow = _named_widgets()
    del workflow["nodes"][0]["widgets_values"]["label"]

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.api_graph["1"]["inputs"]["label"] == "default"


def test_a_link_still_overrides_a_widget_saved_by_name() -> None:
    workflow = _named_widgets()
    workflow["nodes"][1]["widgets_values"] = {"images": "unused", "filename_prefix": "result"}
    object_info = _object_info()
    object_info["Save"]["input"]["required"]["images"] = ["STRING"]

    compiled = compile_comfyui_ui_graph(workflow, object_info)

    assert compiled.api_graph["2"]["inputs"] == {
        "images": ["1", 0],
        "filename_prefix": "result",
    }


def test_the_control_that_follows_a_seed_is_consumed_rather_than_sent() -> None:
    """The editor draws it and applies it before a graph is ever saved."""

    compiled = compile_comfyui_ui_graph(_named_widgets(), _object_info())

    assert compiled.api_graph["1"]["inputs"] == {"label": "camera", "seed": 42}


def test_a_name_the_node_has_no_input_for_is_refused() -> None:
    """Silently dropping it would send the node a different instruction than the
    graph recorded."""

    workflow = _named_widgets()
    workflow["nodes"][0]["widgets_values"]["invented"] = 1

    _assert_error("unknown_widget_value", workflow, _object_info())


@pytest.mark.parametrize("widgets_values", ["text", 7, None])
def test_widgets_saved_as_neither_shape_are_refused(widgets_values: Any) -> None:
    workflow = _workflow()
    workflow["nodes"][0]["widgets_values"] = widgets_values

    _assert_error("invalid_widget_values", workflow, _object_info())


_VHS_AUDITED = "8343122234b61a0f8eb3d1f3f98382b0f7aff2b9"


def _video_combine(
    extras: dict[str, Any],
    package: str = "comfyui-videohelpersuite",
    revision: str = _VHS_AUDITED,
) -> Any:
    workflow = _workflow()
    workflow["nodes"][0]["outputs"][0]["links"] = [7, 9]
    workflow["nodes"].append(
        {
            "id": 4,
            "type": "VHS_VideoCombine",
            "mode": 0,
            "properties": {"cnr_id": package, "ver": revision},
            "inputs": [{"name": "images", "type": "IMAGE", "link": 9}],
            "outputs": [],
            "widgets_values": {"frame_rate": 16, "format": "video/h264-mp4", **extras},
        }
    )
    workflow["links"].append([9, 1, 0, 4, 0, "IMAGE"])
    object_info = _object_info()
    object_info["VHS_VideoCombine"] = {
        "input": {
            "required": {
                "images": ["IMAGE"],
                "frame_rate": ["INT", {"default": 8}],
                "format": [["video/h264-mp4", "image/gif"]],
            }
        },
        "input_order": {"required": ["images", "frame_rate", "format"]},
        "output": [],
    }
    return workflow, object_info


def test_the_options_a_video_format_adds_are_carried_through_as_saved() -> None:
    """`combine_video` ends in `**kwargs`, and the widgets that land there are
    declared by the format file the `format` input selects - so their names are
    not in the runtime's declaration and the graph is what states them."""

    workflow, object_info = _video_combine(
        {"crf": 17, "pix_fmt": "yuv420p", "save_metadata": True, "trim_to_audio": False}
    )

    compiled = compile_comfyui_ui_graph(workflow, object_info)

    assert compiled.api_graph["4"]["inputs"] == {
        "images": ["1", 0],
        "frame_rate": 16,
        "format": "video/h264-mp4",
        "crf": 17,
        "pix_fmt": "yuv420p",
        "save_metadata": True,
        "trim_to_audio": False,
    }


@pytest.mark.parametrize("value", [{"nested": 1}, [1, 2]])
def test_a_format_option_that_is_not_one_widget_is_refused(value: Any) -> None:
    """Each is one editor widget, so each is one value. Anything else is not a
    format option and its own editor would never have produced it."""

    workflow, object_info = _video_combine({"crf": value})

    _assert_error("package_widget_layout", workflow, object_info)


def test_extra_names_are_only_read_for_the_package_that_declares_them() -> None:
    workflow, object_info = _video_combine({"crf": 17}, package="someone-else")

    _assert_error("unknown_widget_value", workflow, object_info)


@pytest.mark.parametrize(
    ("properties", "code"),
    [
        ({"cnr_id": "rgthree-comfy", "ver": "1.0.9999999999"}, "package_widget_revision"),
        ({"cnr_id": "rgthree-comfy"}, "package_widget_revision"),
        ({"cnr_id": "rgthree-comfy", "ver": ""}, "invalid_node_properties"),
        ({"cnr_id": "rgthree-comfy", "ver": 1}, "invalid_node_properties"),
    ],
)
def test_a_layout_is_only_read_for_a_revision_it_was_read_at(
    properties: dict[str, Any], code: str
) -> None:
    """A complete layout proves the shape, not the meaning. Two revisions can
    serialize identically and read the values differently, so matching the shape
    is not permission to skip which code drew it - and the nearest revision that
    was audited is a guess, not a fallback."""

    workflow = _power_loras(
        {"on": True, "lora": "one.safetensors", "strength": 1}, properties=properties
    )

    if code == "invalid_node_properties":
        _assert_refused_before_compiling(code, workflow, _drawn_object_info())
        return
    _assert_error(code, workflow, _drawn_object_info())


def test_the_other_audited_revision_of_the_same_layout_is_read() -> None:
    """Both audited revisions ship the file that defines this layout byte for
    byte identically, which is what makes them one layout rather than two
    assumed to match."""

    workflow = _power_loras(
        {"on": True, "lora": "one.safetensors", "strength": 1},
        properties={"cnr_id": "rgthree-comfy", "ver": "1.0.2605082257"},
    )

    compiled = compile_comfyui_ui_graph(workflow, _drawn_object_info())

    assert compiled.api_graph["3"]["inputs"]["lora_1"]["lora"] == "one.safetensors"


@pytest.mark.parametrize("divider", [{}, None])
def test_both_recorded_furniture_variants_are_read(divider: Any) -> None:
    """Real exports write a divider as an empty object or as null depending on
    how they were saved. Each is named; neither loosens the check into
    accepting whatever happens to be there."""

    workflow = _package_drawn_widgets(
        _RGTHREE,
        node_type="Power Lora Loader (rgthree)",
        widgets_values=[
            divider,
            {"type": "PowerLoraLoaderHeaderWidget"},
            {"on": True, "lora": "one.safetensors", "strength": 1},
            divider,
            "",
        ],
    )

    compiled = compile_comfyui_ui_graph(workflow, _drawn_object_info())

    assert compiled.api_graph["3"]["inputs"]["lora_1"]["lora"] == "one.safetensors"


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), float("-inf")])
def test_a_strength_that_is_not_a_finite_number_is_refused(strength: float) -> None:
    """They survive a JSON round trip in some encoders and mean nothing as a
    scaling factor. Bounded parsing already refuses them for the whole graph,
    which is a stronger guarantee than the transcription can make on its own -
    so the transcription keeps its own check and this proves the earlier one."""

    _assert_refused_before_compiling(
        "non_finite_number",
        _power_loras({"on": True, "lora": "one.safetensors", "strength": strength}),
        _drawn_object_info(),
    )


def test_a_node_claiming_two_packages_that_disagree_is_refused() -> None:
    """Choosing one would decide which package's layout to read by, on no
    evidence at all."""

    workflow = _power_loras(
        {"on": True, "lora": "one.safetensors", "strength": 1},
        properties={
            "cnr_id": "rgthree-comfy",
            "aux_id": "someone/else",
            "ver": "1.0.2605082257",
        },
    )

    _assert_error("conflicting_package_claim", workflow, _drawn_object_info())


@pytest.mark.parametrize(
    "revision",
    ["8343122234b61a0f8eb3d1f3f98382b0f7aff2b8", "1.0.0"],
)
def test_video_options_are_only_read_for_the_audited_revision(revision: str) -> None:
    """One character away from the audited revision is still a revision nobody
    read, and there is no nearest-match fallback."""

    workflow, object_info = _video_combine({"crf": 17}, revision=revision)

    _assert_error("package_widget_revision", workflow, object_info)


@pytest.mark.parametrize(
    "extras",
    [
        {"crf": 17, "reserved": "x"},
        {"crf": 17, "prompt": "leaked"},
        {"crf": 17, "unique_id": 1},
        {"pix_fmt": "yuv444p"},
        {"crf": 101},
        {"crf": -1},
        {"crf": 17.5},
        {"crf": True},
        {"save_metadata": "yes"},
        {"trim_to_audio": 1},
    ],
)
def test_a_video_option_outside_what_the_format_declares_is_refused(extras: dict[str, Any]) -> None:
    """The node ends in `**kwargs`, so a name it does not expect reaches it
    unchecked. Being a scalar is not evidence of being an option."""

    workflow, object_info = _video_combine(extras)

    _assert_error("package_widget_layout", workflow, object_info)


def test_a_video_format_this_build_has_not_read_is_refused() -> None:
    """Its options are declared by its own format file, so reading them under
    another format's set would accept names this one never had."""

    workflow, object_info = _video_combine({"crf": 17})
    workflow["nodes"][2]["widgets_values"]["format"] = "image/gif"
    object_info["VHS_VideoCombine"]["input"]["required"]["format"] = [
        ["video/h264-mp4", "image/gif"]
    ]

    _assert_error("package_widget_layout", workflow, object_info)


def test_the_players_own_state_is_recognised_and_dropped() -> None:
    """The node never reads it and the runtime has no input by that name, so it
    is neither sent nor treated as an unknown input."""

    workflow, object_info = _video_combine(
        {
            "crf": 17,
            "videopreview": {"hidden": False, "paused": False, "params": {"filename": "x.mp4"}},
        }
    )

    compiled = compile_comfyui_ui_graph(workflow, object_info)

    assert "videopreview" not in compiled.api_graph["4"]["inputs"]
    assert compiled.api_graph["4"]["inputs"]["crf"] == 17


@pytest.mark.parametrize(
    "preview",
    ["not an object", {"hidden": False, "invented": 1}, ["hidden"]],
)
def test_player_state_this_build_does_not_know_is_refused(preview: Any) -> None:
    """It is dropped either way, so this is not about what gets sent. It is
    about not calling something known furniture while it carries something never
    seen - which is how a changed layout passes unnoticed."""

    workflow, object_info = _video_combine({"crf": 17, "videopreview": preview})

    _assert_error("package_widget_layout", workflow, object_info)


@pytest.mark.parametrize("control", ["", "sometimes", "RANDOMISE"])
def test_a_named_control_value_that_is_not_a_control_is_refused(control: str) -> None:
    """The positional path only consumes a real control value; consuming any
    string here would swallow whatever the graph actually meant."""

    workflow = _named_widgets()
    workflow["nodes"][0]["widgets_values"]["control_after_generate"] = control

    _assert_error("unknown_widget_value", workflow, _object_info())


@pytest.mark.parametrize("control", ["randomize", "Fixed", "INCREMENT", "decrement"])
def test_a_named_control_value_is_consumed_however_it_is_cased(control: str) -> None:
    workflow = _named_widgets()
    workflow["nodes"][0]["widgets_values"]["control_after_generate"] = control

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.api_graph["1"]["inputs"] == {"label": "camera", "seed": 42}


# Captured from a real exported ComfyUI workflow rather than written here, with
# only the lora filename replaced. Hand-built fixtures agreed with the layout
# and still missed what a real export carries: further properties beside the
# package id, both a registry and a repository id on the same node, and slot
# fields the compiler never reads. A fixture that is tidier than reality proves
# less than it looks.
_GOLDEN_POWER_LORA_LOADER: dict[str, Any] = {
    "id": 326,
    "type": "Power Lora Loader (rgthree)",
    "mode": 0,
    "order": 12,
    "flags": {},
    "title": "High Noise LoRAs",
    "properties": {
        "cnr_id": "rgthree-comfy",
        "ver": "6b76ee6f2c5a007710b5a16f97c94330d6ecc871",
        "Show Strengths": "Single Strength",
        "Match": "",
    },
    "inputs": [
        {"dir": 3, "name": "model", "type": "MODEL", "link": 9},
        {"dir": 3, "name": "clip", "type": "CLIP", "link": None},
    ],
    "outputs": [],
    "widgets_values": [
        {},
        {"type": "PowerLoraLoaderHeaderWidget"},
        {"on": True, "lora": "captured.safetensors", "strength": 1, "strengthTwo": None},
        {},
        "",
    ],
}

_GOLDEN_VIDEO_COMBINE: dict[str, Any] = {
    "id": 80,
    "type": "VHS_VideoCombine",
    "mode": 0,
    "order": 20,
    "flags": {},
    "title": "Initial 16FPS Video",
    "properties": {
        "cnr_id": "comfyui-videohelpersuite",
        "ver": "8343122234b61a0f8eb3d1f3f98382b0f7aff2b9",
        "Node name for S&R": "VHS_VideoCombine",
        "aux_id": "Kosinkadink/ComfyUI-VideoHelperSuite",
        "widget_ue_connectable": {},
    },
    "inputs": [
        {"name": "images", "type": "IMAGE", "link": 9},
        {"name": "audio", "shape": 7, "type": "AUDIO", "link": None},
    ],
    "outputs": [],
    "widgets_values": {
        "frame_rate": 16,
        "loop_count": 0,
        "filename_prefix": "wan22/raw",
        "format": "video/h264-mp4",
        "pix_fmt": "yuv420p",
        "crf": 17,
        "save_metadata": True,
        "trim_to_audio": False,
        "pingpong": False,
        "save_output": True,
    },
}


def _golden(node: dict[str, Any], definition: dict[str, Any]) -> Any:
    workflow = _workflow()
    workflow["nodes"][0]["outputs"][0]["links"] = [7, 9]
    workflow["nodes"].append(deepcopy(node))
    workflow["links"].append([9, 1, 0, node["id"], 0, "IMAGE"])
    object_info = _object_info()
    object_info[node["type"]] = definition
    return workflow, object_info


def test_a_captured_power_lora_loader_compiles_to_its_captured_prompt() -> None:
    workflow, object_info = _golden(
        _GOLDEN_POWER_LORA_LOADER,
        {
            "input": {"optional": {"model": ["MODEL"], "clip": ["CLIP"]}},
            "input_order": {"optional": ["model", "clip"]},
            "output": ["MODEL", "CLIP"],
        },
    )

    compiled = compile_comfyui_ui_graph(workflow, object_info)

    assert compiled.api_graph["326"]["inputs"] == {
        "lora_1": {
            "on": True,
            "lora": "captured.safetensors",
            "strength": 1,
            "strengthTwo": None,
        },
        "model": ["1", 0],
    }


def test_a_captured_video_combine_compiles_to_its_captured_prompt() -> None:
    """It states a registry id and a repository id at once, which is the case
    that made difference-is-not-disagreement necessary."""

    workflow, object_info = _golden(
        _GOLDEN_VIDEO_COMBINE,
        {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "frame_rate": ["INT", {"default": 8}],
                    "loop_count": ["INT", {"default": 0}],
                    "filename_prefix": ["STRING", {"default": "AnimateDiff"}],
                    "format": [["video/h264-mp4", "image/gif"]],
                    "pingpong": ["BOOLEAN", {"default": False}],
                    "save_output": ["BOOLEAN", {"default": True}],
                },
                "optional": {"audio": ["AUDIO"], "meta_batch": ["VHS_BatchManager"]},
            },
            "input_order": {
                "required": [
                    "images",
                    "frame_rate",
                    "loop_count",
                    "filename_prefix",
                    "format",
                    "pingpong",
                    "save_output",
                ],
                "optional": ["audio", "meta_batch"],
            },
            "output": [],
        },
    )

    compiled = compile_comfyui_ui_graph(workflow, object_info)

    assert compiled.api_graph["80"]["inputs"] == {
        "images": ["1", 0],
        "frame_rate": 16,
        "loop_count": 0,
        "filename_prefix": "wan22/raw",
        "format": "video/h264-mp4",
        "pingpong": False,
        "save_output": True,
        "crf": 17,
        "pix_fmt": "yuv420p",
        "save_metadata": True,
        "trim_to_audio": False,
    }


def _scoped_named_wires(writer_scope: str, reader_scope: str, label: str = "pipe") -> Any:
    """A graph shaped the way subgraph expansion leaves one.

    Expansion flattens an instance by prefixing its inner node ids with the
    instance they came from, so nesting survives only in the ids. These are
    written directly rather than expanded from definitions, because what is
    under test is how a nesting is read, not how one is flattened.
    """

    workflow = _workflow()
    workflow["nodes"][0]["outputs"][0]["links"] = [100]
    workflow["nodes"].append(
        {
            "id": f"{writer_scope}10",
            "type": "SetNode",
            "mode": 0,
            "inputs": [{"name": "IMAGE", "type": "IMAGE", "link": 100}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": None}],
            "widgets_values": [label],
        }
    )
    workflow["nodes"].append(
        {
            "id": f"{reader_scope}11",
            "type": "GetNode",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [7]}],
            "widgets_values": [label],
        }
    )
    workflow["links"] = [
        [100, 1, 0, f"{writer_scope}10", 0, "IMAGE"],
        [7, f"{reader_scope}11", 0, 2, 0, "IMAGE"],
    ]
    return workflow


@pytest.mark.parametrize(
    ("writer_scope", "reader_scope"),
    [
        ("", ""),
        ("inner:", "inner:"),
        ("", "inner:"),
        ("", "outer:inner:"),
        ("outer:", "outer:inner:"),
    ],
)
def test_a_reader_takes_the_value_from_its_own_nesting_or_one_around_it(
    writer_scope: str, reader_scope: str
) -> None:
    """The package that draws these resolves a reader against its own graph and
    then the graphs enclosing it, so a value set outside reaches inward."""

    compiled = compile_comfyui_ui_graph(
        _scoped_named_wires(writer_scope, reader_scope), _object_info()
    )

    assert compiled.api_graph["2"]["inputs"]["images"] == ["1", 0]


@pytest.mark.parametrize(
    ("writer_scope", "reader_scope"),
    [("inner:", ""), ("one:", "two:"), ("outer:inner:", "outer:")],
)
def test_a_value_set_where_the_reader_cannot_see_it_is_not_found(
    writer_scope: str, reader_scope: str
) -> None:
    """That lookup never reaches a sibling nesting, or inward from one. Treating
    those as candidates would connect two things the editor keeps apart."""

    _assert_error(
        "undefined_named_wire", _scoped_named_wires(writer_scope, reader_scope), _object_info()
    )


def test_two_copies_of_one_nesting_each_keep_their_own_value() -> None:
    """The shape two instances of the same subgraph leave behind. Read as one
    graph they look like a label set twice, and every reuse of a subgraph that
    names a wire would be ambiguous - a graph the editor runs perfectly well."""

    workflow = _workflow()
    workflow["nodes"][0]["outputs"][0]["links"] = [100]
    workflow["nodes"].append(
        {
            "id": 5,
            "type": "Source",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [200]}],
            "widgets_values": ["camera", 42, "randomize"],
        }
    )
    workflow["nodes"].append(
        {
            "id": 6,
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
    for scope, feed, emit in (("a:", 100, 7), ("b:", 200, 8)):
        workflow["nodes"].append(
            {
                "id": f"{scope}10",
                "type": "SetNode",
                "mode": 0,
                "inputs": [{"name": "IMAGE", "type": "IMAGE", "link": feed}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": None}],
                "widgets_values": ["pipe"],
            }
        )
        workflow["nodes"].append(
            {
                "id": f"{scope}11",
                "type": "GetNode",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [emit]}],
                "widgets_values": ["pipe"],
            }
        )
    workflow["links"] = [
        [100, 1, 0, "a:10", 0, "IMAGE"],
        [7, "a:11", 0, 2, 0, "IMAGE"],
        [200, 5, 0, "b:10", 0, "IMAGE"],
        [8, "b:11", 0, 6, 0, "IMAGE"],
    ]

    compiled = compile_comfyui_ui_graph(workflow, _object_info())

    assert compiled.api_graph["2"]["inputs"]["images"] == ["1", 0]
    assert compiled.api_graph["6"]["inputs"]["images"] == ["5", 0]


def test_a_value_set_twice_in_one_nesting_is_still_ambiguous() -> None:
    """Scoping decides which copies are separate, not whether a real collision
    inside one nesting can be resolved."""

    workflow = _scoped_named_wires("inner:", "inner:")
    workflow["nodes"][0]["outputs"][0]["links"] = [100, 101]
    workflow["nodes"].append(
        {
            "id": "inner:12",
            "type": "SetNode",
            "mode": 0,
            "inputs": [{"name": "IMAGE", "type": "IMAGE", "link": 101}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": None}],
            "widgets_values": ["pipe"],
        }
    )
    workflow["links"].append([101, 1, 0, "inner:12", 0, "IMAGE"])

    _assert_error("duplicate_named_wire", workflow, _object_info())
