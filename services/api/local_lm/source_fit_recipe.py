"""Exact centered extension geometry for an accepted source image.

This serializable plan is data, not permission to run a workflow. Its caller
must bind source and prepared artifacts, selected workflow and user intent in
the accepted context, then check the actual execution graph before transport.
"""

from __future__ import annotations

from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .outpaint_workflows import pad_the_source
from .output_geometry import MAX_DIMENSION, MAX_PIXELS
from .source_fit_image import MAX_SOURCE_PIXELS, SourceFitImageRecord
from .workflow_source_geometry import (
    FULL_STRENGTH,
    MAX_OUTPAINT_PIXELS,
    SourceFitPixels,
    SourceFitRoute,
    trace_source_fit_route,
)


class SourceExtensionRecipe(BaseModel):
    """Retained identity and exact canvas; no source pixels live in this JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    mode: Literal["extend"] = "extend"
    image: SourceFitImageRecord
    canvas_width: int = Field(ge=1, le=MAX_DIMENSION)
    canvas_height: int = Field(ge=1, le=MAX_DIMENSION)
    save_node_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_extension(self) -> Self:
        self._pixels()
        return self

    def _pixels(self) -> SourceFitPixels:
        width, height = self.image.width, self.image.height
        dx, dy = self.canvas_width - width, self.canvas_height - height
        if (
            dx < 0
            or dy < 0
            or width > MAX_DIMENSION
            or height > MAX_DIMENSION
            or width * height > MAX_SOURCE_PIXELS
            or self.canvas_width * self.canvas_height > MAX_PIXELS
        ):
            raise ValueError("source_fit_dimensions")
        left, top = dx // 2, dy // 2
        right, bottom = dx - left, dy - top
        if right > min(2 * width, MAX_OUTPAINT_PIXELS) or bottom > min(
            2 * height, MAX_OUTPAINT_PIXELS
        ):
            raise ValueError("source_fit_dimensions")
        return SourceFitPixels(left, top, right, bottom, self.canvas_width, self.canvas_height)

    @property
    def upload_image(self) -> SourceFitImageRecord:
        """The picture the workflow is given: the source as prepared, which its graph pads."""
        return self.image

    @property
    def retained_artifact_ids(self) -> frozenset[str]:
        """Every picture this recipe needs kept for as long as its run can be replayed."""
        return frozenset({self.image.source_artifact_id, self.image.prepared_artifact_id})

    def route(self, api_graph: object) -> SourceFitRoute:
        """Recheck that this graph still pads, samples and saves the source this way."""
        route = trace_source_fit_route(api_graph, self.save_node_id)
        if route is None:
            raise ValueError("source_fit_graph")
        return route

    def margins(self) -> dict[str, int]:
        """Whole-pixel sides that centre the source on the accepted canvas."""
        return self._pixels().margins()

    def strength_setting(self, api_graph: object) -> dict[str, int]:
        """Pin the sampler strength where the workflow maps it to a setting."""
        parameter = self.route(api_graph).strength_parameter
        return {} if parameter is None else {parameter: FULL_STRENGTH}

    def bind_graph(self, api_graph: object) -> dict[str, Any]:
        """A copy of the graph that pads the source by these margins at full strength."""
        route = self.route(api_graph)
        bound = pad_the_source(cast(dict[str, Any], api_graph), self.margins())
        if route.strength_parameter is not None:
            bound[route.sampler_node_id]["inputs"]["denoise"] = FULL_STRENGTH
        return bound


def plan_source_extension(
    image: SourceFitImageRecord,
    *,
    canvas_width: object,
    canvas_height: object,
    api_graph: object,
    save_node_id: str,
) -> SourceExtensionRecipe:
    """Keep requested integer dimensions exactly, refusing crop or coercion."""
    if type(canvas_width) is not int or type(canvas_height) is not int:
        raise ValueError("source_fit_dimensions")
    recipe = SourceExtensionRecipe(
        image=image,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        save_node_id=save_node_id,
    )
    recipe.route(api_graph)
    return recipe
