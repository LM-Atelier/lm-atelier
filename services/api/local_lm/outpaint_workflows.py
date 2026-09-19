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

import copy
import math
from pathlib import Path
from typing import Any

from PIL import Image

OUTPAINT_NODES = frozenset({"ImagePadForOutpaint"})

OUTPAINT_SETTING_KEY = "outpaint_margins"
OUTPAINT_SCHEMA_KIND = "outpaint"
#: Beyond this the new region dwarfs the picture it was inferred from, and the
#: result stops being an extension of anything.
MAX_MARGIN_FRACTION = 2.0
_SIDES = ("top", "right", "bottom", "left")
#: EXIF orientations that show a stored picture turned a quarter, so its width
#: and height are seen swapped. ComfyUI's LoadImage applies the orientation too.
_QUARTER_TURNS = frozenset({5, 6, 7, 8})
_ORIENTATION_TAG = 0x0112


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


def source_pad_node(graph: object) -> str | None:
    """The one padding node that receives the source picture, or nothing.

    A declared margin has somewhere to go only when the graph pads exactly once,
    straight from the LoadImage that receives the source, and that node's four
    sides are ordinary numbers. Anything else would leave a person guessing which
    padding their drag changed, so it is not recognized.
    """
    if not isinstance(graph, dict):
        return None
    pads = [
        str(node_id)
        for node_id, node in graph.items()
        if isinstance(node, dict) and node.get("class_type") in OUTPAINT_NODES
    ]
    if len(pads) != 1:
        return None
    inputs = graph[pads[0]].get("inputs")
    if not isinstance(inputs, dict):
        return None
    link = inputs.get("image")
    if not isinstance(link, list) or len(link) != 2 or link[1] != 0:
        return None
    source = graph.get(str(link[0]))
    if not isinstance(source, dict) or source.get("class_type") != "LoadImage":
        return None
    source_inputs = source.get("inputs")
    if not isinstance(source_inputs, dict) or source_inputs.get("image") != "${input_image}":
        return None
    if any(
        isinstance(inputs.get(side), bool) or not isinstance(inputs.get(side), int)
        for side in _SIDES
    ):
        return None
    return pads[0]


def oriented_size(path: Path) -> tuple[int, int]:
    """A source picture's width and height as it is shown, not as it is stored."""
    with Image.open(path) as image:
        width, height = image.size
        orientation = image.getexif().get(_ORIENTATION_TAG)
    return (height, width) if orientation in _QUARTER_TURNS else (width, height)


def margin_pixels(margins: dict[str, float], width: int, height: int) -> dict[str, int]:
    """Whole pixels per side: left and right of the width, top and bottom of the height.

    This is the unit the Studio's edge handles produce: a horizontal drag is
    divided by the shown width and a vertical one by the shown height. A half
    pixel rounds up, so a margin that was asked for is never lost to rounding.
    """
    return {
        side: math.floor(
            margins.get(side, 0.0) * (width if side in {"left", "right"} else height) + 0.5
        )
        for side in _SIDES
    }


def pad_the_source(graph: dict[str, Any], pixels: dict[str, int]) -> dict[str, Any]:
    """A copy of the graph whose source padding is the requested number of pixels."""
    node_id = source_pad_node(graph)
    if node_id is None:
        raise ValueError("This workflow has no single padding step for the source picture.")
    padded = copy.deepcopy(graph)
    padded[node_id]["inputs"].update({side: pixels[side] for side in _SIDES})
    return padded


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
