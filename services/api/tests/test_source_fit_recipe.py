"""Exact extension geometry over a retained source, before authorization."""

from __future__ import annotations

from copy import deepcopy

import pytest
from test_workflow_source_geometry import composited_graph, graph

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.source_fit_image import SourceFitImageRecord


def source(width: int = 640, height: int = 480) -> SourceFitImageRecord:
    return SourceFitImageRecord(
        source_artifact_id="sha256:" + "1" * 64,
        prepared_artifact_id="sha256:" + "2" * 64,
        width=width,
        height=height,
    )


@pytest.mark.parametrize(
    "canvas,expected",
    [
        ((864, 486), (112, 3, 112, 3)),
        ((641, 483), (0, 1, 1, 2)),
        ((640, 480), (0, 0, 0, 0)),
    ],
)
def test_recipe_keeps_exact_canvas_and_compiles_centered_pixel_margins(
    canvas: tuple[int, int],
    expected: tuple[int, int, int, int],
) -> None:
    from local_lm.source_fit_recipe import SourceExtensionRecipe, plan_source_extension

    workflow = composited_graph()
    before = deepcopy(workflow)
    recipe = plan_source_extension(
        source(),
        canvas_width=canvas[0],
        canvas_height=canvas[1],
        api_graph=workflow,
        save_node_id="save",
    )
    restored = SourceExtensionRecipe.model_validate_json(recipe.model_dump_json())
    compiled = ComfyUIAdapter._compile(restored.bind_graph(workflow), {})
    actual = tuple(compiled["pad"]["inputs"][side] for side in ("left", "top", "right", "bottom"))
    assert actual == expected
    assert (recipe.canvas_width, recipe.canvas_height) == canvas
    assert workflow == before
    assert restored.image.prepared_artifact_id == "sha256:" + "2" * 64


@pytest.mark.parametrize(
    "width,height",
    [
        (639, 480),
        (640, 479),
        (0, 480),
        (True, 480),
        (640.5, 480),
        (640, "480"),
        (3201, 480),
        (640, 2401),
        (1000001, 480),
    ],
)
def test_recipe_refuses_crop_coercion_or_unsupported_extension(
    width: object,
    height: object,
) -> None:
    from local_lm.source_fit_recipe import plan_source_extension

    with pytest.raises(ValueError):
        plan_source_extension(
            source(),
            canvas_width=width,
            canvas_height=height,
            api_graph=composited_graph(),
            save_node_id="save",
        )


def test_recipe_enforces_the_actual_pad_node_pixel_ceiling() -> None:
    from local_lm.source_fit_recipe import plan_source_extension

    image = source(10000, 100)
    recipe = plan_source_extension(
        image,
        canvas_width=42768,
        canvas_height=100,
        api_graph=composited_graph(),
        save_node_id="save",
    )
    assert recipe.margins()["right"] == 16384
    with pytest.raises(ValueError):
        plan_source_extension(
            image,
            canvas_width=42769,
            canvas_height=100,
            api_graph=composited_graph(),
            save_node_id="save",
        )


def test_recipe_plans_a_direct_decode_whose_source_is_put_back_afterwards() -> None:
    from local_lm.source_fit_recipe import plan_source_extension

    recipe = plan_source_extension(
        source(),
        canvas_width=864,
        canvas_height=486,
        api_graph=graph(),
        save_node_id="save",
    )
    assert recipe.route(graph()).composite_node_id is None
    assert recipe.margins() == {"top": 3, "right": 112, "bottom": 3, "left": 112}


def test_binding_writes_the_margins_and_full_strength_into_a_copy() -> None:
    from local_lm.source_fit_recipe import plan_source_extension

    workflow = graph()
    workflow["sample"]["inputs"]["denoise"] = "$" + "{denoise}"
    before = deepcopy(workflow)
    recipe = plan_source_extension(
        source(),
        canvas_width=864,
        canvas_height=486,
        api_graph=workflow,
        save_node_id="save",
    )
    bound = recipe.bind_graph(workflow)
    assert bound["sample"]["inputs"]["denoise"] == 1
    sides = {side: bound["pad"]["inputs"][side] for side in ("top", "right", "bottom", "left")}
    assert sides == recipe.margins()
    assert recipe.strength_setting(workflow) == {"denoise": 1}
    assert workflow == before


def test_recipe_rechecks_the_graph_that_will_be_compiled() -> None:
    from local_lm.source_fit_recipe import plan_source_extension

    workflow = composited_graph()
    recipe = plan_source_extension(
        source(),
        canvas_width=864,
        canvas_height=486,
        api_graph=workflow,
        save_node_id="save",
    )
    workflow["composite"]["inputs"]["mask"] = ["pad", 0]
    with pytest.raises(ValueError, match="source_fit_graph"):
        recipe.bind_graph(workflow)
