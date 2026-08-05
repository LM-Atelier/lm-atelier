"""Recognizing a workflow that can paint beyond the edge of a picture.

Extend is the tool with no instruction at all: drag the canvas edge outward
and the workflow invents what was never in frame. That needs a graph built for
it, which in ComfyUI means padding the image and handing the new region to an
inpaint sampler as a mask - `ImagePadForOutpaint` is what does the padding, and
its presence is what makes a graph an outpainter.

Recognized from the graph rather than the name, as with the mask and the
upscaler. The declared setting is four margins rather than one number, because
a picture can be extended on any side independently and a single "amount"
would force a symmetry nobody asked for.
"""

from __future__ import annotations

from typing import Any

OUTPAINT_NODES = frozenset({"ImagePadForOutpaint"})

OUTPAINT_SETTING_KEY = "outpaint_margins"
OUTPAINT_SCHEMA_KIND = "outpaint"
#: Beyond this the new region dwarfs the picture it was inferred from, and the
#: result stops being an extension of anything.
MAX_MARGIN_FRACTION = 2.0
_SIDES = ("top", "right", "bottom", "left")


def graph_can_outpaint(graph: dict[str, Any]) -> bool:
    """Whether this graph pads a picture before sampling into the new region."""
    return bool(_class_types(graph) & OUTPAINT_NODES)


def workflow_declares_outpaint(input_schema: dict[str, Any] | None) -> bool:
    """Whether this workflow accepts margins as a setting."""
    if not isinstance(input_schema, dict):
        return False
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return False
    declared = properties.get(OUTPAINT_SETTING_KEY)
    if not isinstance(declared, dict):
        return False
    return declared.get("x-lm-atelier-kind") == OUTPAINT_SCHEMA_KIND


def normalize_margins(value: object) -> dict[str, float]:
    """Read four side margins as fractions of the source, or refuse.

    Fractions rather than pixels: the drag happens on a view at some zoom, and
    a number of screen pixels means nothing to a workflow. A side that is
    absent is zero - not extending downward is an ordinary thing to want.
    """
    if not isinstance(value, dict):
        raise ValueError("Extension margins must be given per side.")
    margins: dict[str, float] = {}
    for side in _SIDES:
        raw = value.get(side, 0)
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"The {side} margin must be a number.")
        margin = float(raw)
        if margin < 0 or margin > MAX_MARGIN_FRACTION:
            raise ValueError(
                f"The {side} margin must be between 0 and {MAX_MARGIN_FRACTION:g} "
                "of the picture's size."
            )
        margins[side] = margin
    if not any(margins.values()):
        raise ValueError("Extending by nothing on every side would not change the picture.")
    return margins


def _class_types(value: Any) -> frozenset[str]:
    found: set[str] = set()
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            class_type = current.get("class_type")
            if isinstance(class_type, str):
                found.add(class_type)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return frozenset(found)
