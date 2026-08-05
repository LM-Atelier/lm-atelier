"""Recognizing a workflow that can enlarge a picture.

The studio's Enhance tool makes one promise: hand it a picture and get a
larger, cleaner one back. That needs a workflow built to do it, and the
question of whether one is installed has to be answerable before the tool is
offered rather than after someone waits for a result.

Recognition works from the graph rather than from a name, for the same reason
the mask contract does: a workflow called "upscale" that scales nothing cannot
honor the promise, and one called anything at all that carries an upscale node
can. ComfyUI's core offers two shapes, and both count:

- a model-driven pass, `UpscaleModelLoader` feeding `ImageUpscaleWithModel`,
  which is what actually recovers detail;
- a plain resample, `ImageScale` or `ImageScaleBy`, which enlarges without
  inventing anything.

The distinction is kept rather than flattened, because a tool that says
"Enhance" over a plain resample is promising detail it cannot deliver.
"""

from __future__ import annotations

from typing import Any

MODEL_UPSCALE_NODES = frozenset({"ImageUpscaleWithModel", "UltimateSDUpscale"})
RESAMPLE_NODES = frozenset({"ImageScale", "ImageScaleBy"})

#: The setting an upscale workflow takes, and the schema kind that declares it.
UPSCALE_SETTING_KEY = "upscale_factor"
UPSCALE_SCHEMA_KIND = "upscale"


def upscale_capability(graph: dict[str, Any]) -> str | None:
    """What kind of enlargement this graph can do, or None if it cannot.

    `model` when a trained upscaler runs, `resample` when the graph only
    resizes. A graph carrying both is `model`: the resample is a stage, and
    what the workflow delivers is the model pass.
    """
    classes = _class_types(graph)
    if classes & MODEL_UPSCALE_NODES:
        return "model"
    if classes & RESAMPLE_NODES:
        return "resample"
    return None


def workflow_declares_upscale(input_schema: dict[str, Any] | None) -> bool:
    """Whether this workflow accepts a scale factor as a setting.

    A graph that can enlarge is not the same as one that lets the user say by
    how much. The tool needs both: the ability, and somewhere to put the
    number.
    """
    if not isinstance(input_schema, dict):
        return False
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return False
    declared = properties.get(UPSCALE_SETTING_KEY)
    if not isinstance(declared, dict):
        return False
    return declared.get("x-lm-atelier-kind") == UPSCALE_SCHEMA_KIND


def _class_types(value: Any) -> frozenset[str]:
    """Every `class_type` in the graph, however deeply it nests.

    Walked rather than read off the top level, because a compiled graph can
    carry its nodes under several shapes and a missed node is a tool wrongly
    withheld.
    """
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
