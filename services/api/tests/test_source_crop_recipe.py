"""A crop recipe: the exact canvas, the uploaded crop and the graph it runs on, as data."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError
from test_workflow_source_crop_route import graph

from local_lm.output_geometry import MAX_PIXELS
from local_lm.source_crop_recipe import SourceCropRecipe, plan_source_crop_recipe
from local_lm.source_fit_image import SourceFitImageRecord

SOURCE = SourceFitImageRecord(
    source_artifact_id="sha256:" + "a" * 64,
    prepared_artifact_id="sha256:" + "b" * 64,
    width=400,
    height=300,
)
CROP = "sha256:" + "c" * 64


def recipe(**changes: Any) -> SourceCropRecipe:
    values: dict[str, Any] = {
        "image": SOURCE,
        "cropped_artifact_id": CROP,
        "canvas_width": 1600,
        "canvas_height": 900,
        "api_graph": graph(),
        "save_node_id": "save",
    }
    values.update(changes)
    return plan_source_crop_recipe(values.pop("image"), **values)


def test_a_recipe_keeps_the_exact_rectangle_and_uploads_the_crop() -> None:
    planned = recipe()

    assert planned.mode == "crop"
    assert planned.plan().rectangle == (Fraction(0), Fraction(75, 2), Fraction(400), Fraction(225))
    assert planned.plan().scale == Fraction(4)
    # The workflow is given the crop, never the source it was cut from.
    assert planned.upload_image.prepared_artifact_id == CROP
    assert (planned.upload_image.width, planned.upload_image.height) == (1600, 900)


def test_the_graph_runs_as_it_is_at_the_strength_the_person_chose() -> None:
    api_graph = graph()
    before = deepcopy(api_graph)
    planned = recipe(api_graph=api_graph)

    bound = planned.bind_graph(api_graph)

    assert bound == before and bound is not api_graph
    assert api_graph == before
    assert planned.strength_setting(api_graph) == {}


def test_a_recipe_survives_its_own_record() -> None:
    planned = recipe()

    assert SourceCropRecipe.model_validate(planned.model_dump(mode="json")) == planned
    with pytest.raises(ValidationError):
        SourceCropRecipe.model_validate({**planned.model_dump(mode="json"), "extra": 1})


@pytest.mark.parametrize(
    "changes",
    [
        {"canvas_width": 1600.0},
        {"canvas_height": True},
        {"canvas_width": 0},
        {"canvas_width": MAX_PIXELS, "canvas_height": 2},
        {"cropped_artifact_id": "sha256:short"},
        {"save_node_id": ""},
    ],
    ids=[
        "a float width",
        "a true height",
        "no width",
        "too many pixels",
        "a malformed crop",
        "no save node",
    ],
)
def test_a_recipe_refuses_what_it_cannot_honour_exactly(changes: dict[str, Any]) -> None:
    with pytest.raises((ValueError, ValidationError)):
        recipe(**changes)


def test_a_graph_that_would_not_keep_the_crops_size_is_refused() -> None:
    api_graph = graph()
    api_graph["save"]["inputs"]["images"] = ["upscaled", 0]
    api_graph["upscaled"] = {
        "class_type": "ImageScaleBy",
        "inputs": {"image": ["decode", 0], "upscale_method": "lanczos", "scale_by": 2.0},
    }

    with pytest.raises(ValueError, match="source_fit_graph"):
        recipe(api_graph=api_graph)


def test_a_graph_changed_after_acceptance_is_refused_at_binding() -> None:
    planned = recipe()
    changed = graph()
    changed["encode"]["class_type"] = "VAEEncodeForInpaint"

    with pytest.raises(ValueError, match="source_fit_graph"):
        planned.bind_graph(changed)
