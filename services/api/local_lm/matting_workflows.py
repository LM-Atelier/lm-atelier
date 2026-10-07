"""Recognizing a workflow that can cut a subject out of a picture.

The studio's Isolate tool makes one promise: hand it a picture and get the
subject back with the background gone. That needs a workflow built to do it,
and whether one is installed has to be answerable before the tool is offered
rather than after someone waits for a result.

Every other tool class here is recognized from what a revision DECLARES in its
input schema, because that is the app's contract with a workflow and
`tool_capabilities` is handed schemas alone. Matting is the awkward one: the
others declare a setting the tool needs somewhere to put - a factor, a margin,
a mask - and a cutout has no setting at all. It has no strength, no factor and
no margin; it either separates the subject or it does not.

So what is declared here is not a setting but a statement: this workflow
returns the subject on a transparent background. The alternative was to search
the graph for a background-removal node, which is what the mask and upscale
contracts deliberately stopped doing, because a workflow that carries a node
is not the same as one that promises the result. A workflow that can matte
without saying so is exactly the case this design refuses to guess at.

The one workflow that does say so is authored here. It can make the promise
truthfully because this module is its author and its graph is the one below,
which is not the same as reading the promise back out of somebody else's
graph. An imported workflow that removes a background still declares nothing
and is still not offered.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

#: The property a matting workflow declares, and the schema kind that marks it.
#: The property carries no value the caller sets: it is present to say what the
#: workflow returns, which is why it is read for its kind alone.
MATTING_SETTING_KEY = "matte"
MATTING_SCHEMA_KIND = "matting"

#: The version of the workflow authored below. Changing the graph is a new
#: version, so an installed copy is replaced rather than quietly kept.
BUILT_IN_MATTING_VERSION = 1

#: The node whose only widget names the background-removal model file.
_MODEL_FILE_NODE = 2

# The picture into a background-removal model, whose foreground mask becomes
# the picture's alpha. The InvertMask is not decoration: RemoveBackground's mask
# is 1 over the subject, while JoinImageWithAlpha writes alpha as 1 - mask, so
# without it the result is the background with the subject cut out of it,
# which looks deliberate and is exactly wrong. Core nodes only, and SaveImage
# keeps the alpha: it writes four channels to a PNG.
_BUILT_IN_MATTING_GRAPH: dict[str, Any] = {
    "nodes": [
        {
            "id": 1,
            "type": "LoadImage",
            "inputs": [],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": [1, 2]},
                {"name": "MASK", "type": "MASK", "links": []},
            ],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": ["source.png", "image"],
        },
        {
            "id": _MODEL_FILE_NODE,
            "type": "LoadBackgroundRemovalModel",
            "inputs": [],
            "outputs": [{"name": "bg_model", "type": "BACKGROUND_REMOVAL", "links": [3]}],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [""],
        },
        {
            "id": 3,
            "type": "RemoveBackground",
            "inputs": [
                {"name": "bg_removal_model", "type": "BACKGROUND_REMOVAL", "link": 3},
                {"name": "image", "type": "IMAGE", "link": 1},
            ],
            "outputs": [{"name": "mask", "type": "MASK", "links": [4]}],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [],
        },
        {
            "id": 4,
            "type": "InvertMask",
            "inputs": [{"name": "mask", "type": "MASK", "link": 4}],
            "outputs": [{"name": "MASK", "type": "MASK", "links": [5]}],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [],
        },
        {
            "id": 5,
            "type": "JoinImageWithAlpha",
            "inputs": [
                {"name": "image", "type": "IMAGE", "link": 2},
                {"name": "alpha", "type": "MASK", "link": 5},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [6]}],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [],
        },
        {
            "id": 6,
            "type": "SaveImage",
            "inputs": [{"name": "images", "type": "IMAGE", "link": 6}],
            "outputs": [],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": ["LM Atelier"],
        },
    ],
    "links": [
        [1, 1, 0, 3, 1, "IMAGE"],
        [2, 1, 0, 5, 0, "IMAGE"],
        [3, _MODEL_FILE_NODE, 0, 3, 0, "BACKGROUND_REMOVAL"],
        [4, 3, 0, 4, 0, "MASK"],
        [5, 4, 0, 5, 1, "MASK"],
        [6, 5, 0, 6, 0, "IMAGE"],
    ],
}


def built_in_matting_graph(model_file: str) -> dict[str, Any]:
    """The built-in matting workflow, reading the named background-removal model."""

    graph = deepcopy(_BUILT_IN_MATTING_GRAPH)
    for node in graph["nodes"]:
        if node["id"] == _MODEL_FILE_NODE:
            node["widgets_values"] = [model_file]
    return graph


def runtime_lists_model_file(object_info: dict[str, Any], model_file: str) -> bool:
    """Whether the runtime can see this background-removal model file.

    Read from the loader's own choices, in either shape a runtime reports
    them. An empty or missing list is no permission: the compiler lets that
    through for graphs whose runtime reports no choices, but the built-in names
    one exact file, and a workflow that cannot find its model is worse than
    none, since it is offered and then fails.
    """

    loader = object_info.get("LoadBackgroundRemovalModel")
    inputs = loader.get("input") if isinstance(loader, dict) else None
    required = inputs.get("required") if isinstance(inputs, dict) else None
    spec = required.get("bg_removal_name") if isinstance(required, dict) else None
    if not isinstance(spec, list) or not spec:
        return False
    if isinstance(spec[0], list):
        choices: object = spec[0]
    elif len(spec) > 1 and isinstance(spec[1], dict):
        choices = spec[1].get("options")
    else:
        return False
    return isinstance(choices, list) and model_file in choices


def built_in_matting_sha256() -> str:
    """The identity of the authored workflow, whichever model file it is given."""

    authored = {"version": BUILT_IN_MATTING_VERSION, "graph": _BUILT_IN_MATTING_GRAPH}
    return hashlib.sha256(
        json.dumps(authored, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def declare_matting(input_schema: dict[str, Any]) -> dict[str, Any]:
    """The schema with the statement that the workflow returns a cut-out subject.

    For a workflow whose author makes that promise, which today means the
    built-in above and nothing else. It carries no default, so it is never
    offered as a control: there is nothing to set.
    """

    declared = deepcopy(input_schema)
    properties = declared.setdefault("properties", {})
    properties[MATTING_SETTING_KEY] = {
        "type": "boolean",
        "readOnly": True,
        "description": "Returns the subject on a transparent background.",
        "x-lm-atelier-kind": MATTING_SCHEMA_KIND,
    }
    return declared


def workflow_declares_matting(input_schema: dict[str, Any] | None) -> bool:
    """Whether this workflow states that it returns a cut-out subject.

    The declaration is the whole answer. Nothing here reads the graph: a
    revision that separates a subject and does not say so is treated as a
    workflow that does not, so that the promise the tool makes and the promise
    the workflow made are the same promise.
    """

    if not isinstance(input_schema, dict):
        return False
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return False
    declared = properties.get(MATTING_SETTING_KEY)
    if not isinstance(declared, dict):
        return False
    return declared.get("x-lm-atelier-kind") == MATTING_SCHEMA_KIND
