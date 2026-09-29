"""Read-only source canvas previews from verified artifacts and stored graphs.

A preview never retains prepared bytes, changes trust, or authorizes generation.
Admission and dispatch must still resolve their own source and workflow context.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictInt

from .artifacts import ArtifactStore
from .model_planner import workflow_artifact_contract
from .models import Artifact, WorkflowDefinition, WorkflowRevision
from .output_geometry import MAX_DIMENSION
from .schemas import ApiModel, SourceFitRequest
from .source_fit_image import SourceFitImageRecord, prepare_source_fit_image
from .source_fit_recipe import SourceExtensionRecipe, plan_source_extension
from .workflow_source_geometry import trace_source_fit_route


class SourceFitPreviewRequest(ApiModel):
    source_artifact_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_fit: SourceFitRequest


class SourceFitCapabilityOut(ApiModel):
    available: bool
    reason: Literal["source_fit_workflow_unsupported"] | None
    modes: list[Literal["extend"]]
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


class SourceFitPreviewOut(ApiModel):
    version: Literal[1] = 1
    mode: Literal["extend"] = "extend"
    workflow_revision_id: str
    workflow_artifact_sha256: str
    source_artifact_id: str
    source: SourceFitDimensionsOut
    canvas: SourceFitDimensionsOut
    margins: SourceFitMarginsOut
    source_rectangle: SourceFitRectangleOut
    request_authorized: Literal[False] = False


def _extension_save(definition: WorkflowDefinition, revision: WorkflowRevision) -> str | None:
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
    if trace_source_fit_route(revision.api_graph_json, saves[0]) is None:
        return None
    return saves[0]


def source_fit_capability(
    definition: WorkflowDefinition, revision: WorkflowRevision
) -> SourceFitCapabilityOut:
    available = _extension_save(definition, revision) is not None
    return SourceFitCapabilityOut(
        available=available,
        reason=None if available else "source_fit_workflow_unsupported",
        modes=["extend"] if available else [],
    )


def preview_source_fit(
    definition: WorkflowDefinition,
    revision: WorkflowRevision,
    store: ArtifactStore,
    source: Artifact,
    intent: SourceFitRequest,
) -> SourceFitPreviewOut:
    """Resolve the same integer recipe as admission without ingesting an artifact."""
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


def source_fit_preview_for_recipe(
    revision: WorkflowRevision, recipe: SourceExtensionRecipe
) -> SourceFitPreviewOut:
    """Describe a validated fresh or retained recipe against its selected graph."""
    recipe.route(revision.api_graph_json)
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
