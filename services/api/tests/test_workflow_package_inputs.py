from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from local_lm.comfy_workflow_compiler import (
    WorkflowCompilationError,
    compile_comfyui_ui_graph,
)
from local_lm.domain import Operation
from local_lm.workflow_package_inputs import (
    WorkflowPackageInputError,
    prepare_workflow_package_compilation,
)


def _object_info() -> dict[str, Any]:
    return {
        "LoadImage": {
            "python_module": "nodes",
            "input": {
                "required": {
                    "image": [["available.png"], {"image_upload": True}],
                }
            },
            "input_order": {"required": ["image"]},
            "output": ["IMAGE", "MASK"],
        },
        "SaveImage": {
            "input": {"required": {"images": ["IMAGE"]}},
            "input_order": {"required": ["images"]},
            "output": [],
            "output_node": True,
        },
    }


def _graph(*, named: bool = False) -> dict[str, Any]:
    widgets: object = {"image": "author.png"} if named else ["author.png", "image"]
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "LoadImage",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [7]}],
                "widgets_values": widgets,
            },
            {
                "id": 2,
                "type": "SaveImage",
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 7}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [[7, 1, 0, 2, 0, "IMAGE"]],
    }


@pytest.mark.parametrize("operation", [Operation.IMAGE_TO_IMAGE, Operation.IMAGE_TO_VIDEO])
def test_one_source_is_bound_before_transient_choice_validation(
    operation: Operation,
) -> None:
    graph = _graph()
    object_info = _object_info()
    original_graph = deepcopy(graph)
    original_info = deepcopy(object_info)

    prepared = prepare_workflow_package_compilation(graph, object_info, operation)
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)
    bound = prepared.bind(compiled.api_graph)

    assert compiled.api_graph["1"]["inputs"]["image"] == "author.png"
    assert bound["1"]["inputs"]["image"] == "${input_image}"
    assert prepared.input_schema == {
        "type": "object",
        "properties": {"input_image": {"type": "string"}},
    }
    assert graph == original_graph
    assert object_info == original_info


def test_a_named_source_is_prepared_for_the_named_widget_compiler() -> None:
    graph = _graph(named=True)

    prepared = prepare_workflow_package_compilation(
        graph,
        _object_info(),
        Operation.IMAGE_TO_VIDEO,
    )

    assert prepared.source_node_id == "1"
    assert prepared.source_value == "author.png"
    assert "author.png" in prepared.object_info["LoadImage"]["input"]["required"]["image"][0]
    assert graph["nodes"][0]["widgets_values"] == {"image": "author.png"}


def test_non_source_operation_keeps_normal_widget_validation() -> None:
    graph = _graph()
    object_info = _object_info()

    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.TEXT_TO_VIDEO,
    )

    with pytest.raises(WorkflowCompilationError) as raised:
        compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)
    assert raised.value.code == "invalid_widget_choice"
    assert prepared.input_schema == {}


@pytest.mark.parametrize(
    ("nodes", "code"),
    [
        ([], "workflow_source_input_missing"),
        (
            [
                _graph()["nodes"][0],
                {**deepcopy(_graph()["nodes"][0]), "id": 3},
            ],
            "workflow_source_input_ambiguous",
        ),
    ],
)
def test_missing_or_ambiguous_source_refuses(
    nodes: list[dict[str, Any]],
    code: str,
) -> None:
    graph = _graph()
    graph["nodes"] = nodes
    graph["links"] = []

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(
            graph,
            _object_info(),
            Operation.IMAGE_TO_VIDEO,
        )

    assert raised.value.code == code


def test_a_different_packages_load_image_cannot_claim_the_source_role() -> None:
    graph = _graph()
    graph["nodes"][0]["properties"] = {
        "cnr_id": "not-comfy-core",
        "ver": "1.0.0",
    }

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(
            graph,
            _object_info(),
            Operation.IMAGE_TO_VIDEO,
        )

    assert raised.value.code == "workflow_source_input_unsupported"


@pytest.mark.parametrize("python_module", [None, "custom_nodes.shadow_loader"])
def test_graph_metadata_cannot_authorize_a_non_core_runtime_loader(
    python_module: str | None,
) -> None:
    graph = _graph()
    graph["nodes"][0]["properties"] = {"cnr_id": "comfy-core"}
    object_info = _object_info()
    if python_module is None:
        object_info["LoadImage"].pop("python_module")
    else:
        object_info["LoadImage"]["python_module"] = python_module

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(
            graph,
            object_info,
            Operation.IMAGE_TO_VIDEO,
        )

    assert raised.value.code == "workflow_source_input_unsupported"


def test_a_non_upload_load_image_contract_refuses() -> None:
    object_info = _object_info()
    object_info["LoadImage"]["input"]["required"]["image"][1]["image_upload"] = False

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(
            _graph(),
            object_info,
            Operation.IMAGE_TO_VIDEO,
        )

    assert raised.value.code == "workflow_source_input_unsupported"


def test_a_source_inside_a_subgraph_keeps_its_namespaced_identity() -> None:
    graph = _graph()
    graph["nodes"] = [
        {
            "id": 4,
            "type": "source-subgraph",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [7]}],
        },
        graph["nodes"][1],
    ]
    graph["links"] = [[7, 4, 0, 2, 0, "IMAGE"]]
    graph["definitions"] = {
        "subgraphs": [
            {
                "id": "source-subgraph",
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
                "nodes": [_graph()["nodes"][0]],
                "links": [[11, 1, 0, -20, 0, "IMAGE"]],
            }
        ]
    }

    prepared = prepare_workflow_package_compilation(
        graph,
        _object_info(),
        Operation.IMAGE_TO_VIDEO,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)
    bound = prepared.bind(compiled.api_graph)

    assert prepared.source_node_id == "4:1"
    assert compiled.api_graph["4:1"]["inputs"]["image"] == "author.png"
    assert bound["4:1"]["inputs"]["image"] == "${input_image}"


def test_binding_refuses_a_compiled_source_value_that_did_not_come_from_the_ui() -> None:
    graph = _graph()
    prepared = prepare_workflow_package_compilation(
        graph,
        _object_info(),
        Operation.IMAGE_TO_VIDEO,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)
    compiled.api_graph["1"]["inputs"]["image"] = "different.png"

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepared.bind(compiled.api_graph)

    assert raised.value.code == "workflow_source_input_binding_failed"


@pytest.mark.parametrize("tail", ["upload", "image-extra"])
def test_unknown_load_image_upload_controls_refuse(tail: str) -> None:
    graph = _graph()
    graph["nodes"][0]["widgets_values"] = ["author.png", tail]

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(
            graph,
            _object_info(),
            Operation.IMAGE_TO_VIDEO,
        )

    assert raised.value.code == "workflow_source_input_unsupported"


def test_an_unaudited_widget_after_image_refuses_instead_of_being_defaulted() -> None:
    graph = _graph()
    object_info = _object_info()
    load_image = object_info["LoadImage"]
    load_image["input"]["required"]["mode"] = ["STRING", {"default": "default"}]
    load_image["input_order"]["required"] = ["image", "mode"]

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(
            graph,
            object_info,
            Operation.IMAGE_TO_VIDEO,
        )

    assert raised.value.code == "workflow_source_input_unsupported"


def test_a_disconnected_source_cannot_claim_an_image_operation() -> None:
    graph = _graph()
    graph["nodes"][0]["outputs"][0]["links"] = []
    graph["nodes"].append(
        {
            "id": 3,
            "type": "EmptyImage",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [8]}],
            "widgets_values": [64, 64],
        }
    )
    graph["nodes"][1]["inputs"][0]["link"] = 8
    graph["links"] = [[8, 3, 0, 2, 0, "IMAGE"]]
    object_info = _object_info()
    object_info["EmptyImage"] = {
        "input": {
            "required": {
                "width": ["INT", {"default": 64}],
                "height": ["INT", {"default": 64}],
            }
        },
        "input_order": {"required": ["width", "height"]},
        "output": ["IMAGE"],
    }

    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_VIDEO,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepared.bind(compiled.api_graph)

    assert raised.value.code == "workflow_source_input_not_used"


def test_source_choice_relaxation_does_not_relax_a_model_choice() -> None:
    graph = _graph()
    graph["nodes"].append(
        {
            "id": 3,
            "type": "ModelChoice",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
            "widgets_values": ["missing.safetensors"],
        }
    )
    object_info = _object_info()
    object_info["ModelChoice"] = {
        "input": {"required": {"model": [["available.safetensors"], {}]}},
        "input_order": {"required": ["model"]},
        "output": ["MODEL"],
    }
    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_VIDEO,
    )

    with pytest.raises(WorkflowCompilationError) as raised:
        compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    assert raised.value.code == "invalid_widget_choice"


def test_source_reachability_uses_compiler_resolved_named_wires() -> None:
    graph = _graph()
    graph["nodes"][0]["outputs"][0]["links"] = [100]
    graph["nodes"].extend(
        [
            {
                "id": 10,
                "type": "SetNode",
                "mode": 0,
                "inputs": [{"name": "IMAGE", "type": "IMAGE", "link": 100}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": None}],
                "widgets_values": ["source"],
            },
            {
                "id": 11,
                "type": "GetNode",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [7]}],
                "widgets_values": ["source"],
            },
        ]
    )
    graph["links"] = [
        [100, 1, 0, 10, 0, "IMAGE"],
        [7, 11, 0, 2, 0, "IMAGE"],
    ]
    prepared = prepare_workflow_package_compilation(
        graph,
        _object_info(),
        Operation.IMAGE_TO_VIDEO,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)
    bound = prepared.bind(compiled.api_graph)

    assert bound["2"]["inputs"]["images"] == ["1", 0]


def test_an_unrelated_output_sink_refuses_without_a_verified_output_capability() -> None:
    graph = _graph()
    graph["nodes"].append(
        {
            "id": 3,
            "type": "ShowText",
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "widgets_values": ["diagnostic"],
        }
    )
    object_info = _object_info()
    object_info["ShowText"] = {
        "input": {"required": {"text": ["STRING", {"default": ""}]}},
        "input_order": {"required": ["text"]},
        "output": [],
        "output_node": True,
    }
    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_VIDEO,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepared.bind(compiled.api_graph)

    assert raised.value.code == "workflow_source_input_not_used"


def test_an_independent_media_output_refuses_even_when_one_output_uses_the_source() -> None:
    graph = _graph()
    graph["nodes"].extend(
        [
            {
                "id": 3,
                "type": "EmptyImage",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [8]}],
                "widgets_values": [64, 64],
            },
            {
                "id": 4,
                "type": "SaveImage",
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 8}],
                "outputs": [],
                "widgets_values": [],
            },
        ]
    )
    graph["links"].append([8, 3, 0, 4, 0, "IMAGE"])
    object_info = _object_info()
    object_info["EmptyImage"] = {
        "input": {
            "required": {
                "width": ["INT", {"default": 64}],
                "height": ["INT", {"default": 64}],
            }
        },
        "input_order": {"required": ["width", "height"]},
        "output": ["IMAGE"],
    }
    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_VIDEO,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepared.bind(compiled.api_graph)

    assert raised.value.code == "workflow_source_input_not_used"


def test_a_sequence_valued_widget_cannot_forge_a_source_edge() -> None:
    graph = _graph()
    graph["nodes"] = [
        graph["nodes"][0],
        {
            "id": 2,
            "type": "EmptyImage",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [7]}],
            "widgets_values": [64, 64],
        },
        {
            "id": 3,
            "type": "MediaSaver",
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": 7}],
            "outputs": [],
            "widgets_values": [["1", 0]],
        },
    ]
    graph["nodes"][0]["outputs"][0]["links"] = []
    graph["links"] = [[7, 2, 0, 3, 0, "IMAGE"]]
    object_info = _object_info()
    object_info["EmptyImage"] = {
        "input": {
            "required": {
                "width": ["INT", {"default": 64}],
                "height": ["INT", {"default": 64}],
            }
        },
        "input_order": {"required": ["width", "height"]},
        "output": ["IMAGE"],
    }
    object_info["MediaSaver"] = {
        "input": {
            "required": {
                "metadata": [[["1", 0], ["other", 0]], {}],
                "images": ["IMAGE"],
            }
        },
        "input_order": {"required": ["images", "metadata"]},
        "output": [],
        "output_node": True,
    }
    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_VIDEO,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepared.bind(compiled.api_graph)

    assert raised.value.code == "workflow_source_input_not_used"


def test_a_source_reaching_outputs_only_through_its_mask_is_refused() -> None:
    """The upload must become the picture, never the stencil cut out of it.

    A LoadImage whose IMAGE output goes nowhere and whose MASK output feeds the
    graph still "reaches" the outputs if reachability forgets which slot an edge
    left by.  Binding there hands the run's upload to a mask input and leaves the
    run with no source image, which is exactly the quiet wrong choice this module
    promises not to make.
    """
    graph = {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "LoadImage",
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": []},
                    {"name": "MASK", "type": "MASK", "links": [11]},
                ],
                "widgets_values": ["author.png", "image"],
            },
            {
                "id": 3,
                "type": "EmptyImage",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [12]}],
                "widgets_values": [64, 64],
            },
            {
                "id": 4,
                "type": "Composite",
                "mode": 0,
                "inputs": [
                    {"name": "image", "type": "IMAGE", "link": 12},
                    {"name": "mask", "type": "MASK", "link": 11},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [13]}],
                "widgets_values": [],
            },
            {
                "id": 2,
                "type": "SaveImage",
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 13}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [
            [11, 1, 1, 4, 1, "MASK"],
            [12, 3, 0, 4, 0, "IMAGE"],
            [13, 4, 0, 2, 0, "IMAGE"],
        ],
    }
    object_info = _object_info()
    object_info["EmptyImage"] = {
        "input": {"required": {"width": ["INT", {}], "height": ["INT", {}]}},
        "input_order": {"required": ["width", "height"]},
        "output": ["IMAGE"],
    }
    object_info["Composite"] = {
        "input": {"required": {"image": ["IMAGE"], "mask": ["MASK"]}},
        "input_order": {"required": ["image", "mask"]},
        "output": ["IMAGE"],
    }

    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_IMAGE,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepared.bind(compiled.api_graph)

    assert raised.value.code == "workflow_source_input_not_used"


def test_a_second_uploaded_file_of_another_class_is_ambiguous() -> None:
    """Exactly-one cannot be decided by class name.

    A second loader of a different class keeps the author's local filename in the
    compiled package - the precise non-portability this binding exists to remove -
    while the "one LoadImage" count still passes.
    """
    graph = _graph()
    graph["nodes"].append(
        {
            "id": 3,
            "type": "LoadImageOutput",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "widgets_values": ["author-mask.png", "image"],
        }
    )
    object_info = _object_info()
    object_info["LoadImageOutput"] = {
        "python_module": "nodes",
        "input": {"required": {"image": [["a.png", "author-extra.png"], {"image_upload": True}]}},
        "input_order": {"required": ["image"]},
        "output": ["IMAGE", "MASK"],
    }

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(graph, object_info, Operation.IMAGE_TO_IMAGE)

    assert raised.value.code == "workflow_source_input_ambiguous"
    assert "LoadImageOutput" in str(raised.value)


def test_an_image_producer_without_an_upload_is_not_a_second_source() -> None:
    """The counterpart to the rule above: only supplied files count.

    A node that makes an image rather than loading one carries no author
    filename, so refusing on it would reject ordinary graphs.
    """
    graph = _graph()
    graph["nodes"].append(
        {
            "id": 3,
            "type": "EmptyImage",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "widgets_values": [64, 64],
        }
    )
    object_info = _object_info()
    object_info["EmptyImage"] = {
        "input": {"required": {"width": ["INT", {}], "height": ["INT", {}]}},
        "input_order": {"required": ["width", "height"]},
        "output": ["IMAGE"],
    }

    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_IMAGE,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    assert prepared.bind(compiled.api_graph)["1"]["inputs"]["image"] == "${input_image}"


def test_a_load_image_inside_a_bypassed_subgraph_is_not_a_source() -> None:
    """Expansion inlines a container's contents without its mode.

    So the only place a bypassed subgraph is still visible as bypassed is the
    graph as the author saved it, which is where this has to be read.
    """
    graph = {
        "version": 0.4,
        "definitions": {
            "subgraphs": [
                {
                    "id": "sub",
                    "name": "Disabled source",
                    "nodes": [
                        {
                            "id": 1,
                            "type": "LoadImage",
                            "mode": 0,
                            "inputs": [],
                            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                            "widgets_values": ["author.png", "image"],
                        }
                    ],
                    "links": [],
                    "inputs": [],
                    "outputs": [],
                }
            ]
        },
        "nodes": [
            {
                "id": 4,
                "type": "sub",
                "mode": 4,
                "inputs": [],
                "outputs": [],
                "widgets_values": [],
            },
            {
                "id": 2,
                "type": "SaveImage",
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [],
    }

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(graph, _object_info(), Operation.IMAGE_TO_IMAGE)

    assert raised.value.code == "workflow_source_input_missing"


def test_a_graph_with_no_executable_output_says_so() -> None:
    """A graph that produces nothing is not a graph that ignores its source.

    Reporting the reachability failure here is vacuously true and points the
    reader at the source instead of at the missing output.
    """
    graph = _graph()
    object_info = _object_info()
    del object_info["SaveImage"]["output_node"]

    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_IMAGE,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepared.bind(compiled.api_graph)

    assert raised.value.code == "workflow_source_output_missing"


def test_the_modern_combo_options_upload_contract_binds() -> None:
    """The runtime may advertise choices under options rather than inline.

    Supported since this module was written and exercised by nothing, so a later
    edit to that branch would have gone unnoticed until an import failed.
    """
    graph = _graph()
    object_info = _object_info()
    object_info["LoadImage"]["input"]["required"]["image"] = [
        "COMBO",
        {"options": ["available.png"], "image_upload": True},
    ]

    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_IMAGE,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)
    bound = prepared.bind(compiled.api_graph)

    assert compiled.api_graph["1"]["inputs"]["image"] == "author.png"
    assert bound["1"]["inputs"]["image"] == "${input_image}"


def _subgraph_graph(*, inner_mode: int, nested: bool) -> dict[str, Any]:
    """A graph whose only LoadImage sits inside a subgraph, optionally nested.

    `inner_mode` is the mode of the instance holding it, so one builder covers
    both the disabled case and its live control.
    """
    inner = {
        "id": "inner",
        "name": "Source",
        "nodes": [
            {
                "id": 1,
                "type": "LoadImage",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": ["author.png", "image"],
            }
        ],
        "links": [],
        "inputs": [],
        "outputs": [],
    }
    outer = {
        "id": "outer",
        "name": "Wrapper",
        "nodes": [{"id": 7, "type": "inner", "mode": inner_mode, "inputs": [], "outputs": []}],
        "links": [],
        "inputs": [],
        "outputs": [],
    }
    top_type = "outer" if nested else "inner"
    top_mode = 0 if nested else inner_mode
    return {
        "version": 0.4,
        "definitions": {"subgraphs": [inner, outer] if nested else [inner]},
        "nodes": [
            {"id": 4, "type": top_type, "mode": top_mode, "inputs": [], "outputs": []},
            {
                "id": 2,
                "type": "SaveImage",
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [],
    }


def test_a_bypassed_subgraph_nested_inside_another_is_still_not_a_source() -> None:
    """Depth is the whole difficulty, so the test has to have some.

    Expansion namespaces each level separately, so a disabled instance two
    levels down is named "<outer>:<inner>:<node>" and a scope computed from
    top-level ids alone never matches it.  A test at depth one passes against
    an implementation that only reads the top level, which is why it cannot be
    the only test.
    """
    graph = _subgraph_graph(inner_mode=4, nested=True)

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(graph, _object_info(), Operation.IMAGE_TO_IMAGE)

    assert raised.value.code == "workflow_source_input_missing"


def test_a_live_subgraph_nested_inside_another_still_supplies_the_source() -> None:
    """The control for the test above, differing only in the instance's mode.

    Without it, refusing every nested subgraph would pass just as well.
    """
    graph = _subgraph_graph(inner_mode=0, nested=True)

    prepared = prepare_workflow_package_compilation(
        graph,
        _object_info(),
        Operation.IMAGE_TO_IMAGE,
    )

    assert prepared.source_node_id == "4:7:1"
    assert prepared.source_value == "author.png"


def test_a_wired_bypassed_subgraph_still_compiles() -> None:
    """A disabled side branch must compile without executing its interior.

    The author turned the branch off. Its EmptyImage and PreviewImage must not
    reach the runtime prompt, and the live source path must still bind.
    """
    graph = _graph()
    graph["definitions"] = {
        "subgraphs": [
            {
                "id": "side",
                "name": "Side branch",
                "nodes": [
                    {
                        "id": 5,
                        "type": "EmptyImage",
                        "mode": 0,
                        "inputs": [],
                        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [21]}],
                        "widgets_values": [64, 64],
                    },
                    {
                        "id": 6,
                        "type": "PreviewImage",
                        "mode": 0,
                        "inputs": [{"name": "images", "type": "IMAGE", "link": 21}],
                        "outputs": [],
                        "widgets_values": [],
                    },
                ],
                "links": [[21, 5, 0, 6, 0, "IMAGE"]],
                "inputs": [],
                "outputs": [],
            }
        ]
    }
    graph["nodes"].append({"id": 9, "type": "side", "mode": 4, "inputs": [], "outputs": []})
    object_info = _object_info()
    object_info["EmptyImage"] = {
        "input": {"required": {"width": ["INT", {}], "height": ["INT", {}]}},
        "input_order": {"required": ["width", "height"]},
        "output": ["IMAGE"],
    }
    object_info["PreviewImage"] = {
        "input": {"required": {"images": ["IMAGE"]}},
        "input_order": {"required": ["images"]},
        "output": [],
        "output_node": True,
    }

    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_IMAGE,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    assert "9:5" not in compiled.api_graph
    assert "9:6" not in compiled.api_graph
    assert prepared.output_node_ids == ("2",)
    assert prepared.bind(compiled.api_graph)["1"]["inputs"]["image"] == "${input_image}"


def test_a_downstream_non_zero_output_slot_is_still_followed() -> None:
    """The slot restriction belongs to the source alone.

    Applying it everywhere would refuse the ordinary case of a mid-graph node
    whose second output carries the picture onward, so the restriction has to be
    pinned as source-only rather than merely present.
    """
    graph = {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "LoadImage",
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [7]},
                    {"name": "MASK", "type": "MASK", "links": []},
                ],
                "widgets_values": ["author.png", "image"],
            },
            {
                "id": 3,
                "type": "Splitter",
                "mode": 0,
                "inputs": [{"name": "image", "type": "IMAGE", "link": 7}],
                "outputs": [
                    {"name": "discard", "type": "IMAGE", "links": []},
                    {"name": "keep", "type": "IMAGE", "links": [8]},
                ],
                "widgets_values": [],
            },
            {
                "id": 2,
                "type": "SaveImage",
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 8}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [[7, 1, 0, 3, 0, "IMAGE"], [8, 3, 1, 2, 0, "IMAGE"]],
    }
    object_info = _object_info()
    object_info["Splitter"] = {
        "input": {"required": {"image": ["IMAGE"]}},
        "input_order": {"required": ["image"]},
        "output": ["IMAGE", "IMAGE"],
    }

    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_IMAGE,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    assert prepared.bind(compiled.api_graph)["1"]["inputs"]["image"] == "${input_image}"


@pytest.mark.parametrize(
    "control",
    ["image_upload", "video_upload", "audio_upload", "file_upload", "document_upload"],
)
def test_any_upload_control_marks_a_second_supplied_file(control: str) -> None:
    """A package may name its own upload control.

    Listing the kinds that exist today would let exactly the unfamiliar loader
    through, and an unfamiliar loader is the one most likely to be carrying a
    path from the author's machine.
    """
    graph = _graph()
    graph["nodes"].append(
        {
            "id": 3,
            "type": "PackageLoader",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "widgets_values": ["author-reference.png", "file"],
        }
    )
    object_info = _object_info()
    object_info["PackageLoader"] = {
        "input": {"required": {"path": [["author-reference.png"], {control: True}]}},
        "input_order": {"required": ["path"]},
        "output": ["IMAGE"],
    }

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(graph, object_info, Operation.IMAGE_TO_IMAGE)

    assert raised.value.code == "workflow_source_input_ambiguous"


def test_an_optional_upload_input_counts_as_a_second_supplied_file() -> None:
    """Optional is where a second loader is most likely to be declared."""
    graph = _graph()
    graph["nodes"].append(
        {
            "id": 3,
            "type": "OptionalLoader",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "widgets_values": ["author-extra.png", "image"],
        }
    )
    object_info = _object_info()
    object_info["OptionalLoader"] = {
        "input": {"optional": {"image": [["author-extra.png"], {"image_upload": True}]}},
        "input_order": {"optional": ["image"]},
        "output": ["IMAGE"],
    }

    with pytest.raises(WorkflowPackageInputError) as raised:
        prepare_workflow_package_compilation(graph, object_info, Operation.IMAGE_TO_IMAGE)

    assert raised.value.code == "workflow_source_input_ambiguous"


def test_an_upload_control_that_is_not_enabled_is_not_a_second_source() -> None:
    """The flag has to be true, not merely present."""
    graph = _graph()
    graph["nodes"].append(
        {
            "id": 3,
            "type": "QuietLoader",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "widgets_values": ["preset.png"],
        }
    )
    object_info = _object_info()
    object_info["QuietLoader"] = {
        "input": {"required": {"image": [["preset.png"], {"image_upload": False}]}},
        "input_order": {"required": ["image"]},
        "output": ["IMAGE"],
    }

    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_IMAGE,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    assert prepared.bind(compiled.api_graph)["1"]["inputs"]["image"] == "${input_image}"


def test_a_second_uploader_inside_a_bypassed_scope_is_not_counted() -> None:
    """Selection and the second-source rule must agree on what is live.

    A disabled uploader is not a second supplied file, and refusing on one would
    reject a graph whose author had already turned it off.
    """
    graph = _graph()
    graph["definitions"] = {
        "subgraphs": [
            {
                "id": "extra",
                "name": "Disabled extra",
                "nodes": [
                    {
                        "id": 5,
                        "type": "LoadImageOutput",
                        "mode": 0,
                        "inputs": [],
                        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                        "widgets_values": ["author-extra.png", "image"],
                    }
                ],
                "links": [],
                "inputs": [],
                "outputs": [],
            }
        ]
    }
    graph["nodes"].append({"id": 9, "type": "extra", "mode": 4, "inputs": [], "outputs": []})
    object_info = _object_info()
    object_info["LoadImageOutput"] = {
        "python_module": "nodes",
        "input": {"required": {"image": [["a.png", "author-extra.png"], {"image_upload": True}]}},
        "input_order": {"required": ["image"]},
        "output": ["IMAGE", "MASK"],
    }

    prepared = prepare_workflow_package_compilation(
        graph,
        object_info,
        Operation.IMAGE_TO_IMAGE,
    )
    compiled = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)

    assert prepared.bind(compiled.api_graph)["1"]["inputs"]["image"] == "${input_image}"
