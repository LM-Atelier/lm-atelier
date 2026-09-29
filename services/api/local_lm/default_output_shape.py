"""The shape a person asks for when nothing else sets a new picture's or video's size.

A default is the lowest thing a person chooses, just above the workflow's own
size. Any profile, preset, project, chat, recipe or turn that names a width or
a height outranks it whole, so a size is never half one choice and half
another. It is resolved through the same proof the composer's shape buttons
use, so a workflow that cannot make the shape exactly keeps its own size
rather than producing a picture of a shape nobody asked for.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy.orm import Session

from .domain import Operation
from .models import WorkflowDefinition, WorkflowRevision
from .output_geometry import PresetId
from .workflow_output_geometry import (
    prove_workflow_output_geometry,
    resolve_workflow_output_geometry,
)

SIZE_SETTINGS = ("width", "height")

# Only work made from nothing takes a default shape: an edit keeps its
# source's shape, and a video made from a picture follows the picture.
_MODES: dict[Operation, Literal["image", "video"]] = {
    Operation.TEXT_TO_IMAGE: "image",
    Operation.TEXT_TO_VIDEO: "video",
}


@dataclass(frozen=True, slots=True)
class DefaultOutputSize:
    """A preferred shape as one revision makes it."""

    preset_id: PresetId
    width: int
    height: int

    def settings(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height}

    def provenance(self) -> dict[str, Any]:
        return {
            "source": "default",
            "preset_id": self.preset_id,
            "width": self.width,
            "height": self.height,
        }


def size_is_chosen(layers: Iterable[Mapping[str, Any] | None]) -> bool:
    """Whether any of these layers names a width or a height."""
    return any(layer is not None and any(key in layer for key in SIZE_SETTINGS) for layer in layers)


def default_output_size(
    session: Session,
    operation: Operation,
    revision: WorkflowRevision | None,
    image: PresetId | None,
    video: PresetId | None,
) -> DefaultOutputSize | None:
    """The size a preferred shape comes to on this revision, or None where it does not apply.

    None for an operation that takes no default, for a turn with no revision or
    no preference for its kind of output, and for a revision whose stored graph
    cannot be proven to make the shape exactly.
    """
    mode = _MODES.get(operation)
    preset = image if mode == "image" else video if mode == "video" else None
    if preset is None or revision is None:
        return None
    definition = session.get(WorkflowDefinition, revision.workflow_id)
    if definition is None:
        return None
    proof = prove_workflow_output_geometry(
        workflow_id=definition.id,
        revision_id=revision.id,
        operation=definition.operation,
        engine=revision.engine,
        api_graph=revision.api_graph_json,
        input_schema=revision.input_schema_json,
        dependencies=revision.dependencies_json,
        artifact_sha256=revision.artifact_sha256,
        trusted=revision.trusted,
    )
    resolution = resolve_workflow_output_geometry(
        proof, {"mode": mode, "size_mode": "preset", "preset_id": preset}
    )
    if resolution is None:
        return None
    return DefaultOutputSize(preset, resolution.geometry.width, resolution.geometry.height)
