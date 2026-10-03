"""The ComfyUI workflow a picture's own file carries, read only for the workflow review.

A workflow tool writes the editor graph it ran into the picture it saves: a PNG
text chunk named ``workflow``, or an EXIF text tag of that name in a JPEG or
WebP. Reading it here only hands it to the same review a workflow file gets,
which reports what the workflow needs and imports nothing until the person
confirms. Nothing here runs, trusts, keeps or fetches anything the graph names.
"""

from __future__ import annotations

import json
from typing import Any

from .comfy_workflow_packages import validate_bounded_workflow_json
from .exif_text_metadata import ExifText, is_jpeg, is_webp, read_jpeg_text, read_webp_text
from .png_text_metadata import PngText, read_png_text

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_picture_workflow(payload: bytes) -> dict[str, Any] | None:
    """The editor graph a PNG, JPEG or WebP carries as ``workflow``, or None.

    None as well for text the review could not take as a workflow: text that is
    not JSON, JSON naming a key twice, a graph past the review's own bounds, or
    one without the editor's list of nodes (a graph written to be run has none,
    and the review reads only what an editor saves). Two workflows in one file
    give None too, since which of them made the picture is not known.

    Raises ValueError for a file whose text is damaged, as reading its settings does.
    """

    text: PngText | ExifText
    if payload.startswith(_PNG_SIGNATURE):
        text = read_png_text(payload)
    elif is_jpeg(payload):
        text = read_jpeg_text(payload)
    elif is_webp(payload):
        text = read_webp_text(payload)
    else:
        return None
    found = [claim.text for claim in text.claims if claim.keyword == "workflow"]
    if len(found) != 1:
        return None
    try:
        graph = json.loads(found[0], object_pairs_hook=_no_repeated_keys)
        validate_bounded_workflow_json(graph)
    except (RecursionError, ValueError):
        return None
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        return None
    return graph


def _no_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # A key given twice would reach the review as only one of its values.
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("a key is repeated")
        value[key] = item
    return value
