"""Picture purposes for the explicit inputs and outputs of an ordered plan."""

from __future__ import annotations

from dataclasses import dataclass

from .media_input_roles import ImageInputRole, image_source_index, resolved_image_roles
from .schemas import OrderedWorkIntent, TurnRequest


@dataclass(frozen=True)
class OrderedImageInputs:
    artifact_ids: list[str]
    roles: list[ImageInputRole] | None
    dependency_roles: dict[str, ImageInputRole]

    def planned(self) -> tuple[list[str], list[ImageInputRole] | None]:
        """Count each planned picture in its purpose before any step is queued."""
        dependencies = {
            f"planned-image:{step}": role for step, role in self.dependency_roles.items()
        }
        return resolved_image_roles(self.artifact_ids, self.roles, list(dependencies), dependencies)


def ordered_image_inputs(
    request: TurnRequest, intent: OrderedWorkIntent, index: int
) -> OrderedImageInputs:
    """Give selected pictures to the first media step and bind later generated canvases."""
    if request.input_image_roles is None:
        return OrderedImageInputs(list(request.input_artifact_ids) if index == 0 else [], None, {})
    first_media = next((i for i, step in enumerate(intent.steps) if step.mode != "text"), None)
    ids = list(request.input_artifact_ids) if index in (0, first_media) else []
    roles = list(request.input_image_roles) if ids else []
    step = intent.steps[index]
    dependencies: dict[str, ImageInputRole] = {}
    if step.mode != "text":
        has_canvas = image_source_index(ids, roles) is not None
        modes = {item.id: item.mode for item in intent.steps}
        for binding in step.inputs:
            if binding.kind != "artifact" or modes[binding.source_step_id] != "image":
                continue
            dependencies[binding.source_step_id] = "reference" if has_canvas else "edit_source"
            has_canvas = True
    return OrderedImageInputs(ids, roles, dependencies)
