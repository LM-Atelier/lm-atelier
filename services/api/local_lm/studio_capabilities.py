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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .matting_workflows import workflow_declares_matting
from .outpaint_workflows import workflow_declares_outpaint
from .studio_masks import workflow_accepts_mask
from .upscale_workflows import workflow_declares_upscale

StudioToolKind = Literal[
    "instruct",
    "brush",
    "eraser",
    "rect",
    "lasso",
    "bucket",
    "wand",
    "enhance",
    "extend",
    "text",
    "remove",
    "relight",
    "isolate",
    "background",
    "subject",
    "transform",
    "perspective",
    "crop",
    "resize",
    "adjust",
    "blur",
    "paint",
    "caption",
    "canvas",
]


#: The tool kinds the surface offers, paired with what each needs installed.
#: Selection tools share one class: they are six ways to draw one mask. Text
#: and Remove mark a selection too, but it is placed back after a whole-picture
#: edit rather than given to the workflow, so any edit workflow but one that
#: only paints past the edge can run them.
TOOL_WORKFLOW_CLASSES: dict[StudioToolKind, str] = {
    "instruct": "image_to_image",
    "brush": "inpaint",
    "eraser": "inpaint",
    "rect": "inpaint",
    "lasso": "inpaint",
    "bucket": "inpaint",
    "wand": "inpaint",
    "enhance": "upscale",
    "extend": "outpaint",
    "text": "image_to_image",
    "remove": "image_to_image",
    "relight": "relight",
    "isolate": "matting",
    # Replacing a background cuts the subject out first, so it runs the matting
    # workflow, and then redraws around it, so it needs an edit workflow too.
    "background": "matting",
    # Replacing a subject cuts the old one out to find it, removes it with an
    # edit workflow, and places the new one cut out of the second picture, so
    # it runs the matting workflow and needs an edit workflow too.
    "subject": "matting",
    # Turning, flipping, correcting the perspective, cropping, resizing,
    # adjusting light and color, blurring or painting a marked area, adding
    # words and changing the canvas are pixel operations the studio makes
    # itself.
    "transform": "local",
    "perspective": "local",
    "crop": "local",
    "resize": "local",
    "adjust": "local",
    "blur": "local",
    "paint": "local",
    "caption": "local",
    "canvas": "local",
}

_CLASS_GUIDANCE = {
    "image_to_image": "Install an image editing workflow to change a picture.",
    "inpaint": "Install an inpainting workflow to edit part of a picture.",
    "upscale": "Install an upscaling workflow to enlarge a picture.",
    "outpaint": "Install an outpainting workflow to extend a picture past its edge.",
    "relight": (
        "Install an image editing workflow that takes a second picture and a LoRA to relight "
        "a picture."
    ),
    "matting": "Install a background removal workflow to cut a subject out of a picture.",
}
_NO_LIGHTING_ADAPTER = "Install the Qwen Multi-Angle Lighting LoRA to relight a picture."
#: For a tool whose workflow is installed but refused by the run: installing
#: another would not be the next step, finishing this one's setup would.
_WAITING_GUIDANCE = (
    "An installed workflow can do this once it has been reviewed and has the node packages "
    "it needs."
)


@dataclass(frozen=True)
class ToolCapability:
    """One tool's answer, phrased so the surface can show it unchanged."""

    kind: StudioToolKind
    workflow_class: str
    available: bool
    reason: str | None
    #: The one workflow this tool runs on, when the report can name exactly one.
    workflow_revision_id: str | None = None
    #: The installed LoRA this tool applies, when it needs one.
    adapter_asset_id: str | None = None


def tool_capabilities(
    *,
    edit_input_schemas: list[dict[str, Any] | None],
    relight_workflow_ids: Sequence[str] = (),
    lighting_adapter_ids: Sequence[str] = (),
    matting_workflow_ids: Sequence[str] = (),
    waiting_input_schemas: Sequence[dict[str, Any] | None] = (),
) -> list[ToolCapability]:
    """Judge every tool from the schemas of the installed edit workflows.

    A schema list rather than a workflow list, because what a tool needs is a
    declared input, not a name: a workflow called "inpaint" that declares no
    mask cannot honor a selection, and one called anything at all that does
    can. The declaration is the only thing that decides.

    `waiting_input_schemas` are the edit workflows installed here that the run
    would refuse until they are reviewed or given their node packages. They make
    no tool ready, but a tool one of them would serve says so, rather than
    asking for an install that has already happened.
    """
    # Only Extend names margins. Every other tool's turn names none and is never
    # given a workflow that paints past the edge, so one does not make those
    # tools ready.
    ordinary = [schema for schema in edit_input_schemas if not workflow_declares_outpaint(schema)]
    ordinary_waiting = [
        schema for schema in waiting_input_schemas if not workflow_declares_outpaint(schema)
    ]
    # A workflow that only cuts a subject out cannot carry out an instruction,
    # so on its own it does not make the instructed tools usable.
    can_edit = any(not workflow_declares_matting(schema) for schema in ordinary)
    can_mask = any(workflow_accepts_mask(schema) for schema in ordinary)
    can_upscale = any(workflow_declares_upscale(schema) for schema in ordinary)
    can_outpaint = any(workflow_declares_outpaint(schema) for schema in edit_input_schemas)
    can_matte = any(workflow_declares_matting(schema) for schema in edit_input_schemas)
    available = {
        "image_to_image": can_edit,
        "inpaint": can_mask,
        "upscale": can_upscale,
        "outpaint": can_outpaint,
        "matting": can_matte,
        # Relight needs both halves: a workflow that can take the light map and
        # a LoRA, and the one adapter whose behaviour was checked.
        "relight": bool(relight_workflow_ids) and bool(lighting_adapter_ids),
        # Nothing to install: the edit is made here, without a model.
        "local": True,
    }
    waiting = {
        "image_to_image": any(not workflow_declares_matting(schema) for schema in ordinary_waiting),
        "inpaint": any(workflow_accepts_mask(schema) for schema in ordinary_waiting),
        "upscale": any(workflow_declares_upscale(schema) for schema in ordinary_waiting),
        "outpaint": any(workflow_declares_outpaint(schema) for schema in waiting_input_schemas),
        "matting": any(workflow_declares_matting(schema) for schema in waiting_input_schemas),
    }

    def guidance(workflow_class: str) -> str:
        return _WAITING_GUIDANCE if waiting.get(workflow_class) else _CLASS_GUIDANCE[workflow_class]

    capabilities = []
    for kind, workflow_class in TOOL_WORKFLOW_CLASSES.items():
        ready = available[workflow_class]
        reason = None if ready else guidance(workflow_class)
        if kind == "background" and ready and not can_edit:
            # The cutout alone replaces nothing: an edit redraws around it.
            ready, reason = False, guidance("image_to_image")
        if kind == "subject" and ready and not can_edit:
            # The cutouts place nothing on their own: an edit removes the old subject.
            ready, reason = False, guidance("image_to_image")
        if workflow_class == "relight" and relight_workflow_ids and not lighting_adapter_ids:
            reason = _NO_LIGHTING_ADAPTER
        capabilities.append(
            ToolCapability(
                kind=kind,
                workflow_class=workflow_class,
                available=ready,
                reason=reason,
                # Relight is named only when exactly one edit workflow can do it,
                # since they differ; otherwise the studio's chosen one runs and
                # the server checks it can. Every matting workflow does the same
                # job, and the studio's chosen one never does, so one is always
                # named: the first, in the order given. A replaced subject is
                # cut out on the workflow Isolate names, both times, and the old
                # one is removed on the studio's own.
                workflow_revision_id=(
                    relight_workflow_ids[0]
                    if workflow_class == "relight" and len(relight_workflow_ids) == 1
                    else matting_workflow_ids[0]
                    if workflow_class == "matting" and matting_workflow_ids
                    else None
                ),
                adapter_asset_id=(
                    lighting_adapter_ids[0]
                    if workflow_class == "relight" and lighting_adapter_ids
                    else None
                ),
            )
        )
    return capabilities
