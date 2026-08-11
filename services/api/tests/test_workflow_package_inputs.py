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
