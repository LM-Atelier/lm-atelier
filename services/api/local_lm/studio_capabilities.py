"""Which studio tools this machine can actually run.

A tool is a promise: click it, draw, and the picture changes. That promise
depends on a workflow being installed, and until now nothing checked before
the click. Someone could brush a careful selection, write an instruction,
press Apply, and only then be told that no installed workflow accepts a
selection - after all the work, and with nothing to do about it.

The report answers the question early instead, per tool, with the reason and
the workflow class that would fix it, so a tool that cannot run says so while
it is still cheap to hear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .studio_masks import workflow_accepts_mask
from .upscale_workflows import workflow_declares_upscale

#: The tool kinds the surface offers, paired with what each needs installed.
#: Selection tools share one class: they are four ways to draw one mask.
TOOL_WORKFLOW_CLASSES: dict[str, str] = {
    "instruct": "image_to_image",
    "brush": "inpaint",
    "eraser": "inpaint",
    "rect": "inpaint",
    "lasso": "inpaint",
    "enhance": "upscale",
}

_CLASS_GUIDANCE = {
    "image_to_image": "Install an image editing workflow to change a picture.",
    "inpaint": "Install an inpainting workflow to edit part of a picture.",
    "upscale": "Install an upscaling workflow to enlarge a picture.",
}


@dataclass(frozen=True)
class ToolCapability:
    """One tool's answer, phrased so the surface can show it unchanged."""

    kind: str
    workflow_class: str
    available: bool
    reason: str | None


def tool_capabilities(
    *,
    edit_input_schemas: list[dict[str, Any] | None],
) -> list[ToolCapability]:
    """Judge every tool from the schemas of the installed edit workflows.

    A schema list rather than a workflow list, because what a tool needs is a
    declared input, not a name: a workflow called "inpaint" that declares no
    mask cannot honor a selection, and one called anything at all that does
    can. The declaration is the only thing that decides.
    """
    can_edit = bool(edit_input_schemas)
    can_mask = any(workflow_accepts_mask(schema) for schema in edit_input_schemas)
    can_upscale = any(workflow_declares_upscale(schema) for schema in edit_input_schemas)
    available = {
        "image_to_image": can_edit,
        "inpaint": can_mask,
        "upscale": can_upscale,
    }
    capabilities = []
    for kind, workflow_class in TOOL_WORKFLOW_CLASSES.items():
        ready = available[workflow_class]
        capabilities.append(
            ToolCapability(
                kind=kind,
                workflow_class=workflow_class,
                available=ready,
                reason=None if ready else _CLASS_GUIDANCE[workflow_class],
            )
        )
    return capabilities
