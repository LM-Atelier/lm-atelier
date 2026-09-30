"""Exact centred crop geometry for an accepted source image.

The crop is made before the workflow runs: the source is cut to the requested
shape and resized to the canvas at one uniform scale, and those bytes are
uploaded in the source's place. The workflow then edits a picture that is
already the canvas, so its own graph is not changed. This serializable plan is
data, not permission to run a workflow; its caller binds the source, the crop
and the selected workflow in the accepted context, then checks the actual
execution graph before transport.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .output_geometry import MAX_DIMENSION, MAX_PIXELS
from .source_crop import SourceCropPlan, plan_source_crop, prepare_source_crop
from .source_fit_image import (
    PreparedSourceImage,
    SourceFitImageRecord,
    retain_prepared_source_fit_image,
)
from .workflow_source_geometry import SourceCropRoute, trace_source_crop_route


class SourceCropRecipe(BaseModel):
    """Retained identity, the uploaded crop and the exact canvas; no pixels live in this JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    mode: Literal["crop"] = "crop"
    image: SourceFitImageRecord
    #: The crop as uploaded, at the canvas size.
    cropped_artifact_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    canvas_width: int = Field(ge=1, le=MAX_DIMENSION)
    canvas_height: int = Field(ge=1, le=MAX_DIMENSION)
    save_node_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_crop(self) -> Self:
        if self.canvas_width * self.canvas_height > MAX_PIXELS:
            raise ValueError("source_fit_dimensions")
        self.plan()
        return self

    def plan(self) -> SourceCropPlan:
        """The exact rectangle and scale this canvas takes from the source."""
        return plan_source_crop(
            self.image.width, self.image.height, self.canvas_width, self.canvas_height
        )

    @property
    def upload_image(self) -> SourceFitImageRecord:
        """The picture the workflow is given in the source's place: the crop, at the canvas size."""
        return SourceFitImageRecord(
            source_artifact_id=self.image.source_artifact_id,
            prepared_artifact_id=self.cropped_artifact_id,
            width=self.canvas_width,
            height=self.canvas_height,
        )

    @property
    def retained_artifact_ids(self) -> frozenset[str]:
        """Every picture this recipe needs kept for as long as its run can be replayed."""
        return frozenset(
            {
                self.image.source_artifact_id,
                self.image.prepared_artifact_id,
                self.cropped_artifact_id,
            }
        )

    def route(self, api_graph: object) -> SourceCropRoute:
        """Recheck that this graph still encodes, samples and saves the upload as it is."""
        route = trace_source_crop_route(api_graph, self.save_node_id)
        if route is None:
            raise ValueError("source_fit_graph")
        return route

    def strength_setting(self, api_graph: object) -> dict[str, int]:
        """Nothing is pinned: a crop is an ordinary edit, at the strength the person chose."""
        self.route(api_graph)
        return {}

    def bind_graph(self, api_graph: object) -> dict[str, Any]:
        """A copy of the graph as it is; the crop is in the uploaded picture, not in the graph."""
        self.route(api_graph)
        return deepcopy(cast(dict[str, Any], api_graph))


def plan_source_crop_recipe(
    image: SourceFitImageRecord,
    *,
    cropped_artifact_id: str,
    canvas_width: object,
    canvas_height: object,
    api_graph: object,
    save_node_id: str,
) -> SourceCropRecipe:
    """Keep requested integer dimensions exactly, refusing coercion and unknown graphs."""
    if type(canvas_width) is not int or type(canvas_height) is not int:
        raise ValueError("source_fit_dimensions")
    recipe = SourceCropRecipe(
        image=image,
        cropped_artifact_id=cropped_artifact_id,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        save_node_id=save_node_id,
    )
    recipe.route(api_graph)
    return recipe


def capture_source_crop(
    session: Session,
    store: ArtifactStore,
    prepared: PreparedSourceImage,
    *,
    canvas_width: object,
    canvas_height: object,
    api_graph: object,
    save_node_id: str,
) -> SourceCropRecipe:
    """Cut the crop from verified source bytes, plan it, then keep both pictures.

    Nothing is kept unless the recipe plans and its graph routes, so a refused
    crop leaves no artifact behind. Both pictures are kept in the caller's
    acceptance transaction, and each must be the very bytes the recipe names.
    """
    crop = prepare_source_crop(prepared, canvas_width, canvas_height)
    canvas = crop.plan.canvas_size
    recipe = plan_source_crop_recipe(
        SourceFitImageRecord(
            source_artifact_id=prepared.source_artifact_id,
            prepared_artifact_id=f"sha256:{prepared.sha256}",
            width=prepared.width,
            height=prepared.height,
        ),
        cropped_artifact_id=f"sha256:{crop.sha256}",
        canvas_width=canvas[0],
        canvas_height=canvas[1],
        api_graph=api_graph,
        save_node_id=save_node_id,
    )
    kept_source = retain_prepared_source_fit_image(session, store, prepared)
    kept_crop = retain_prepared_source_fit_image(
        session,
        store,
        PreparedSourceImage(
            source_artifact_id=prepared.source_artifact_id,
            source_sha256=prepared.source_sha256,
            sha256=crop.sha256,
            width=canvas[0],
            height=canvas[1],
            content=crop.content,
        ),
    )
    if kept_source != recipe.image or kept_crop != recipe.upload_image:
        raise ValueError("source_fit_image_binding")
    return recipe
