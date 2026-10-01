"""Offer a factor only when it changes the selected workflow's image outputs."""

import copy
from typing import Any

import pytest

from local_lm.models import WorkflowRevision
from local_lm.schemas import SettingField
from local_lm.upscale_preview import project_upscale_preview


def _graph(value: Any = "${upscale_factor}") -> dict[str, Any]:
    return {
        "source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "scale": {
            "class_type": "ImageScaleBy",
            "inputs": {"image": ["source", 0], "scale_by": value},
        },
        "output": {"class_type": "SaveImage", "inputs": {"images": ["scale", 0]}},
    }


def _project(graph: dict[str, Any], field: SettingField | None = None) -> dict[str, Any]:
    revision = WorkflowRevision(id="selected-revision", api_graph_json=graph)
    field = field or SettingField(
        key="upscale_factor",
        label="Scale",
        type="number",
        default=2,
        minimum=1,
        maximum=4,
        step=0.5,
        scope="workflow",
    )
    value = field.default if field.choices else 3
    return project_upscale_preview(revision, [field], {"upscale_factor": value}).model_dump()


def test_a_bound_factor_uses_the_effective_recipe_value_and_authored_bounds() -> None:
    graph = _graph()
    original = copy.deepcopy(graph)
    result = _project(graph)
    assert result["workflow_revision_id"] == "selected-revision"
    assert result["factor"]["default"] == 3
    assert result["factor"]["minimum"] == 1
    assert result["factor"]["maximum"] == 4
    assert result["factor"]["step"] == 0.5
    assert result["fixed_factor"] is None
    assert result["request_authorized"] is False
    assert graph == original


def test_numeric_choices_keep_their_allowed_values() -> None:
    field = SettingField(
        key="upscale_factor",
        label="Scale",
        type="enum",
        choices=[2, 4],
        default=2,
        scope="workflow",
    )
    result = _project(_graph(), field)
    assert result["factor"]["type"] == "enum"
    assert result["factor"]["choices"] == [2, 4]


def test_a_constant_factor_reports_the_composed_size_without_a_slider() -> None:
    field = SettingField(
        key="upscale_factor", label="Scale", type="number", choices=[2], default=2, scope="workflow"
    )
    graph = _graph()
    graph["fixed"] = {
        "class_type": "ImageScaleBy",
        "inputs": {"image": ["source", 0], "scale_by": 3},
    }
    graph["scale"]["inputs"]["image"] = ["fixed", 0]
    result = _project(graph, field)
    assert result["factor"] is None
    assert result["fixed_factor"] == 6


def test_a_fixed_scale_is_reported_without_an_ineffective_control() -> None:
    result = _project(_graph(4))
    assert result["factor"] is None
    assert result["fixed_factor"] == 4


@pytest.mark.parametrize(
    "change", ["disconnected", "unrelated", "unscaled-output", "unknown", "cycle"]
)
def test_unproven_factor_influence_does_not_offer_a_control(change: str) -> None:
    graph = _graph()
    if change == "disconnected":
        graph["output"]["inputs"]["images"] = ["source", 0]
    elif change == "unrelated":
        graph["scale"]["inputs"]["scale_by"] = 4
        graph["output"]["inputs"]["filename_prefix"] = "${upscale_factor}"
    elif change == "unscaled-output":
        graph["other"] = {"class_type": "SaveImage", "inputs": {"images": ["source", 0]}}
    elif change == "unknown":
        graph["scale"]["class_type"] = "UnknownImageTransform"
    else:
        graph["scale"]["inputs"]["image"] = ["scale", 0]
    assert _project(graph)["factor"] is None


def test_a_model_native_scale_is_not_guessed() -> None:
    graph = _graph()
    graph["scale"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"image": ["source", 0]}}
    result = _project(graph)
    assert result["factor"] is None
    assert result["fixed_factor"] is None


def test_an_unknown_output_prevents_a_claim_about_every_saved_picture() -> None:
    graph = _graph()
    graph["other"] = {"class_type": "CustomOutput", "inputs": {"image": ["source", 0]}}
    result = _project(graph)
    assert result["factor"] is None
    assert result["fixed_factor"] is None


def test_an_explicit_scale_after_a_model_upscaler_remains_adjustable() -> None:
    graph = _graph()
    graph["model"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"image": ["source", 0]}}
    graph["scale"]["inputs"]["image"] = ["model", 0]
    assert _project(graph)["factor"]["default"] == 3


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "4"])
def test_invalid_scale_values_do_not_create_a_fixed_size_claim(value: Any) -> None:
    result = _project(_graph(value))
    assert result["factor"] is None
    assert result["fixed_factor"] is None
