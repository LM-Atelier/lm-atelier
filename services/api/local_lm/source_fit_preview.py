"""Read-only source canvas previews from verified artifacts and stored graphs.

A preview never retains prepared bytes, changes trust, or authorizes generation.
Admission and dispatch must still resolve their own source and workflow context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import Field, StrictInt

from .artifacts import ArtifactStore
from .model_planner import workflow_artifact_contract
from .models import Artifact, WorkflowDefinition, WorkflowRevision
from .output_geometry import MAX_DIMENSION
from .schemas import ApiModel, SourceFitRequest
from .source_crop import SourceCropPlan, plan_source_crop
from .source_crop_recipe import SourceCropRecipe
from .source_fit_image import SourceFitImageRecord, prepare_source_fit_image
from .source_fit_recipe import SourceExtensionRecipe, plan_source_extension
from .workflow_source_geometry import trace_source_crop_route, trace_source_fit_route


class SourceFitPreviewRequest(ApiModel):
    source_artifact_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_fit: SourceFitRequest


class SourceFitCapabilityOut(ApiModel):
    available: bool
    reason: Literal["source_fit_workflow_unsupported"] | None
    modes: list[Literal["extend", "crop"]]
    request_authorized: Literal[False] = False


class SourceFitDimensionsOut(ApiModel):
    width: StrictInt = Field(ge=1, le=MAX_DIMENSION)
    height: StrictInt = Field(ge=1, le=MAX_DIMENSION)


class SourceFitMarginsOut(ApiModel):
    left: StrictInt = Field(ge=0)
    top: StrictInt = Field(ge=0)
    right: StrictInt = Field(ge=0)
    bottom: StrictInt = Field(ge=0)


class SourceFitRectangleOut(SourceFitDimensionsOut):
    x: StrictInt = Field(ge=0)
    y: StrictInt = Field(ge=0)


class SourceFitKeptOut(ApiModel):
    """The part of the source a crop keeps, in the source's own pixels.

    Its edges can fall between pixels: the kept part has exactly the canvas's
    shape, and the canvas is made from it at one scale.
    """

    left: float = Field(ge=0)
    top: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class SourceFitPreviewOut(ApiModel):
    version: Literal[1] = 1
    mode: Literal["extend", "crop"] = "extend"
    workflow_revision_id: str
    workflow_artifact_sha256: str
    source_artifact_id: str
    source: SourceFitDimensionsOut
    canvas: SourceFitDimensionsOut
    margins: SourceFitMarginsOut
    source_rectangle: SourceFitRectangleOut
    #: For a crop, the part of the source that fills the canvas.
    kept: SourceFitKeptOut | None = None
    request_authorized: Literal[False] = False


def _extension_save(definition: WorkflowDefinition, revision: WorkflowRevision) -> str | None:
    return _route_save(definition, revision, trace_source_fit_route)


def _crop_save(definition: WorkflowDefinition, revision: WorkflowRevision) -> str | None:
    return _route_save(definition, revision, trace_source_crop_route)


def _route_save(
    definition: WorkflowDefinition,
    revision: WorkflowRevision,
    trace: Callable[[object, object], object | None],
) -> str | None:
    if (
        definition.id != revision.workflow_id
        or definition.operation != "image_to_image"
        or revision.engine != "comfyui"
        or revision.trusted is not True
        or not revision.artifact_sha256
        or not isinstance(revision.api_graph_json, dict)
        or not 1 <= len(revision.api_graph_json) <= 512
    ):
        return None
    try:
        calculated = workflow_artifact_contract(
            operation=definition.operation,
            engine=revision.engine,
            api_graph=revision.api_graph_json,
            input_schema=revision.input_schema_json,
            dependencies=revision.dependencies_json,
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if calculated != revision.artifact_sha256:
        return None
    saves = [
        key
        for key, node in revision.api_graph_json.items()
        if isinstance(node, dict) and node.get("class_type") == "SaveImage"
    ]
    if len(saves) != 1:
        return None
    if trace(revision.api_graph_json, saves[0]) is None:
        return None
    return saves[0]


def source_fit_capability(
    definition: WorkflowDefinition, revision: WorkflowRevision
) -> SourceFitCapabilityOut:
    modes: list[Literal["extend", "crop"]] = []
    if _extension_save(definition, revision) is not None:
        modes.append("extend")
    if _crop_save(definition, revision) is not None:
        modes.append("crop")
    return SourceFitCapabilityOut(
        available=bool(modes),
        reason=None if modes else "source_fit_workflow_unsupported",
        modes=modes,
    )


def preview_source_fit(
    definition: WorkflowDefinition,
    revision: WorkflowRevision,
    store: ArtifactStore,
    source: Artifact,
    intent: SourceFitRequest,
) -> SourceFitPreviewOut:
    """Resolve the same integer recipe as admission without ingesting an artifact."""
    if intent.mode == "crop":
        return _preview_crop(definition, revision, store, source, intent)
    save_id = _extension_save(definition, revision)
    if save_id is None:
        raise ValueError("source_fit_workflow_unsupported")
    prepared = prepare_source_fit_image(store, source)
    recipe = plan_source_extension(
        SourceFitImageRecord(
            source_artifact_id=prepared.source_artifact_id,
            prepared_artifact_id=f"sha256:{prepared.sha256}",
            width=prepared.width,
            height=prepared.height,
        ),
        canvas_width=intent.width,
        canvas_height=intent.height,
        api_graph=revision.api_graph_json,
        save_node_id=save_id,
    )
    return source_fit_preview_for_recipe(revision, recipe)


def _preview_crop(
    definition: WorkflowDefinition,
    revision: WorkflowRevision,
    store: ArtifactStore,
    source: Artifact,
    intent: SourceFitRequest,
) -> SourceFitPreviewOut:
    """Work out the crop admission would make, without cutting or keeping a picture."""
    if _crop_save(definition, revision) is None:
        raise ValueError("source_fit_workflow_unsupported")
    prepared = prepare_source_fit_image(store, source)
    plan = plan_source_crop(prepared.width, prepared.height, intent.width, intent.height)
    return _crop_preview(revision, prepared.source_artifact_id, plan)


def _crop_preview(
    revision: WorkflowRevision, source_artifact_id: str, plan: SourceCropPlan
) -> SourceFitPreviewOut:
    left, top, width, height = plan.rectangle
    canvas_width, canvas_height = plan.canvas_size
    return SourceFitPreviewOut(
        mode="crop",
        workflow_revision_id=revision.id,
        workflow_artifact_sha256=revision.artifact_sha256 or "",
        source_artifact_id=source_artifact_id,
        source=SourceFitDimensionsOut(width=plan.source_size[0], height=plan.source_size[1]),
        canvas=SourceFitDimensionsOut(width=canvas_width, height=canvas_height),
        margins=SourceFitMarginsOut(left=0, top=0, right=0, bottom=0),
        # The kept part fills the canvas, so the canvas is all source.
        source_rectangle=SourceFitRectangleOut(x=0, y=0, width=canvas_width, height=canvas_height),
        kept=SourceFitKeptOut(
            left=float(left), top=float(top), width=float(width), height=float(height)
        ),
    )


def source_crop_preview_for_image(
    revision: WorkflowRevision,
    image: SourceFitImageRecord,
    canvas_width: int,
    canvas_height: int,
    save_node_id: str,
) -> SourceFitPreviewOut:
    """The crop a retained source takes for a canvas, on the graph that would edit it."""
    if trace_source_crop_route(revision.api_graph_json, save_node_id) is None:
        raise ValueError("source_fit_graph")
    plan = plan_source_crop(image.width, image.height, canvas_width, canvas_height)
    return _crop_preview(revision, image.source_artifact_id, plan)


def source_fit_preview_for_recipe(
    revision: WorkflowRevision, recipe: SourceExtensionRecipe | SourceCropRecipe
) -> SourceFitPreviewOut:
    """Describe a validated fresh or retained recipe against its selected graph."""
    recipe.route(revision.api_graph_json)
    if isinstance(recipe, SourceCropRecipe):
        return _crop_preview(revision, recipe.image.source_artifact_id, recipe.plan())
    margins = recipe.margins()
    left, top = margins["left"], margins["top"]
    return SourceFitPreviewOut(
        workflow_revision_id=revision.id,
        workflow_artifact_sha256=revision.artifact_sha256 or "",
        source_artifact_id=recipe.image.source_artifact_id,
        source=SourceFitDimensionsOut(width=recipe.image.width, height=recipe.image.height),
        canvas=SourceFitDimensionsOut(width=recipe.canvas_width, height=recipe.canvas_height),
        margins=SourceFitMarginsOut(
            left=left,
            top=top,
            right=margins["right"],
            bottom=margins["bottom"],
        ),
        source_rectangle=SourceFitRectangleOut(
            x=left, y=top, width=recipe.image.width, height=recipe.image.height
        ),
    )
