"""Use-case input support follows executable contracts instead of labels."""

from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import pytest

from local_lm.domain import Operation
from local_lm.workflow_use_cases_v1 import SelectionApplication, WorkflowUseCaseInputs


def _assess(facts: WorkflowUseCaseInputs, **changes: Any) -> Any:
    name = "local_lm.workflow_use_case_structure"
    assert find_spec(name) is not None, "Workflow use-case structural checks are missing."
    arguments = {
        "operation": facts.operation.value,
        "engine": "comfyui",
        "api_graph": {"source": {"inputs": {"image": "${input_image}"}}},
        "input_schema": {"type": "object", "properties": {}},
        **changes,
    }
    return import_module(name).assess_workflow_use_case_structure(facts, **arguments)


def _edit(**changes: Any) -> WorkflowUseCaseInputs:
    return WorkflowUseCaseInputs(Operation.IMAGE_TO_IMAGE, source_present=True, **changes)


def _schema(key: str, kind: str) -> dict[str, Any]:
    return {"type": "object", "properties": {key: {"x-lm-atelier-kind": kind}}}


def test_a_matching_image_edit_accepts_the_source_contract() -> None:
    result = _assess(_edit())
    assert result.reason is None
    assert result.upscale_kind is None


def test_an_explicit_operation_mismatch_is_not_hidden_by_graph_support() -> None:
    result = _assess(_edit(), operation="text_to_image")
    assert result.reason == "workflow-use-case-operation-mismatch"


@pytest.mark.parametrize("placeholder", ["${input_image}", "${input_images}", "${input_image_5}"])
def test_source_binding_uses_the_adapters_numbered_slot_fallback(placeholder: str) -> None:
    result = _assess(_edit(), api_graph={"nodes": [{"input": placeholder}]})
    assert result.reason is None


@pytest.mark.parametrize("value", ["prefix ${input_image}", "${input_image_64}", "image.png"])
def test_a_name_or_embedded_placeholder_cannot_supply_the_source(value: str) -> None:
    result = _assess(_edit(), api_graph={"inputs": {"image": value}})
    assert result.reason == "workflow-use-case-source-binding-missing"


def test_video_animation_also_needs_a_source_binding() -> None:
    facts = WorkflowUseCaseInputs(Operation.IMAGE_TO_VIDEO, source_present=True)
    assert (
        _assess(facts, api_graph={"node": {}}).reason == "workflow-use-case-source-binding-missing"
    )


@pytest.mark.parametrize(
    "operation", [Operation.TEXT, Operation.TEXT_TO_IMAGE, Operation.TEXT_TO_VIDEO]
)
def test_operations_without_a_source_do_not_invent_a_source_requirement(
    operation: Operation,
) -> None:
    assert _assess(WorkflowUseCaseInputs(operation), api_graph={"node": {}}).reason is None


@pytest.mark.parametrize("graph", [None, {}])
def test_a_media_graph_must_exist(graph: dict[str, Any] | None) -> None:
    facts = WorkflowUseCaseInputs(Operation.TEXT_TO_IMAGE)
    assert _assess(facts, api_graph=graph).reason == "workflow-use-case-graph-missing"


def test_chat_does_not_require_a_comfyui_graph() -> None:
    assert _assess(WorkflowUseCaseInputs(Operation.TEXT), api_graph=None).reason is None


def test_blend_selection_does_not_require_a_workflow_mask() -> None:
    assert _assess(_edit(selection=SelectionApplication.BLEND)).reason is None


def test_workflow_selection_requires_a_declared_and_bound_mask() -> None:
    facts = _edit(selection=SelectionApplication.WORKFLOW)
    assert _assess(facts).reason == "workflow-use-case-mask-unsupported"
    schema = _schema("mask", "mask")
    assert _assess(facts, input_schema=schema).reason == "workflow-use-case-mask-binding-missing"
    graph = {"source": {"image": "${input_image}", "mask": "${mask}"}}
    assert _assess(facts, api_graph=graph, input_schema=schema).reason is None


def _outpaint_graph() -> dict[str, Any]:
    return {
        "source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "pad": {
            "class_type": "ImagePadForOutpaint",
            "inputs": {
                "image": ["source", 0],
                "top": 0,
                "right": 0,
                "bottom": 0,
                "left": 0,
            },
        },
    }


def test_extension_requires_the_canonical_source_pad_contract() -> None:
    facts = _edit(extend=True)
    graph = _outpaint_graph()
    assert _assess(facts, api_graph=graph).reason == "workflow-use-case-outpaint-unsupported"
    schema = _schema("outpaint_margins", "outpaint")
    assert _assess(facts, api_graph=graph, input_schema=schema).reason is None
    graph["pad"]["inputs"]["image"] = ["source", 1]
    assert (
        _assess(facts, api_graph=graph, input_schema=schema).reason
        == "workflow-use-case-outpaint-binding-missing"
    )


def test_an_extra_padding_node_does_not_count_as_verified_extension_support() -> None:
    graph = _outpaint_graph()
    graph["other_pad"] = deepcopy(graph["pad"])
    result = _assess(
        _edit(extend=True), api_graph=graph, input_schema=_schema("outpaint_margins", "outpaint")
    )
    assert result.reason == "workflow-use-case-outpaint-binding-missing"


@pytest.mark.parametrize(
    ("node", "kind"),
    [
        ("ImageUpscaleWithModel", "model"),
        ("UltimateSDUpscale", "model"),
        ("ImageScale", "resample"),
        ("ImageScaleBy", "resample"),
    ],
)
def test_upscale_preserves_the_difference_between_detail_and_resampling(
    node: str, kind: str
) -> None:
    graph = {"node": {"class_type": node, "inputs": {"image": "${input_image}"}}}
    facts = _edit(upscale=True)
    assert _assess(facts, api_graph=graph).reason == "workflow-use-case-upscale-unsupported"
    schema = _schema("upscale_factor", "upscale")
    schema["properties"]["upscale_factor"]["readOnly"] = True
    result = _assess(facts, api_graph=graph, input_schema=schema)
    assert result.reason is None
    assert result.upscale_kind == kind


def test_an_editable_factor_must_reach_the_graph() -> None:
    graph = {"node": {"class_type": "ImageScaleBy", "inputs": {"image": "${input_image}"}}}
    schema = _schema("upscale_factor", "upscale")

    result = _assess(_edit(upscale=True), api_graph=graph, input_schema=schema)

    assert result.reason == "workflow-use-case-upscale-binding-missing"


def test_an_explicitly_bound_factor_remains_usable() -> None:
    graph = {
        "node": {
            "class_type": "ImageScaleBy",
            "inputs": {"image": "${input_image}", "scale_by": "${upscale_factor}"},
        }
    }

    result = _assess(
        _edit(upscale=True), api_graph=graph, input_schema=_schema("upscale_factor", "upscale")
    )

    assert result.reason is None
    assert result.upscale_kind == "resample"


def test_a_scale_declaration_does_not_invent_an_upscale_node() -> None:
    result = _assess(_edit(upscale=True), input_schema=_schema("upscale_factor", "upscale"))
    assert result.reason == "workflow-use-case-upscale-binding-missing"


def test_other_adapters_keep_their_own_execution_contracts() -> None:
    facts = _edit(selection=SelectionApplication.WORKFLOW)
    assert (
        _assess(facts, engine="external", api_graph=None).reason
        == "workflow-use-case-mask-unsupported"
    )
    assert (
        _assess(
            facts, engine="external", api_graph=None, input_schema=_schema("mask", "mask")
        ).reason
        is None
    )


def test_assessment_does_not_change_graph_or_schema() -> None:
    graph, schema = _outpaint_graph(), _schema("outpaint_margins", "outpaint")
    before = deepcopy((graph, schema))
    assert _assess(_edit(extend=True), api_graph=graph, input_schema=schema).reason is None
    assert (graph, schema) == before
