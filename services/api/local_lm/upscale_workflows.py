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

import math
from typing import Any

from .graph_placeholders import binds_parameter

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
    """Whether this workflow declares fixed or adjustable enlargement."""
    if not isinstance(input_schema, dict):
        return False
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return False
    declared = properties.get(UPSCALE_SETTING_KEY)
    if not isinstance(declared, dict):
        return False
    return declared.get("x-lm-atelier-kind") == UPSCALE_SCHEMA_KIND


def upscale_setting_schema(graph: dict[str, Any], input_schema: dict[str, Any]) -> dict[str, Any]:
    """Preserve a bound factor's constraints or declare enlargement as fixed."""
    properties = input_schema.get("properties")
    field = properties.get(UPSCALE_SETTING_KEY) if isinstance(properties, dict) else None
    if (
        isinstance(field, dict)
        and field.get("type") in {"number", "integer"}
        and binds_parameter(graph, UPSCALE_SETTING_KEY)
    ):
        declared = {**field, "x-lm-atelier-kind": UPSCALE_SCHEMA_KIND}
        fixed_upscale_factor(graph, {"properties": {UPSCALE_SETTING_KEY: declared}})
        return declared
    return {
        "type": "number",
        "readOnly": True,
        "title": "Enlargement",
        "description": "The workflow determines the output size.",
        "x-lm-atelier-kind": UPSCALE_SCHEMA_KIND,
    }


def effective_upscale_schema(
    graph: dict[str, Any], input_schema: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Show an unbound enlargement declaration as fixed without changing its revision."""
    if not workflow_declares_upscale(input_schema) or binds_parameter(graph, UPSCALE_SETTING_KEY):
        return input_schema
    assert input_schema is not None
    field = input_schema["properties"][UPSCALE_SETTING_KEY]
    if (
        field.get("readOnly") is True
        or upscale_capability(graph) is None
        or field.get("type") != "number"
        or field.get("default") != 2
        or field.get("minimum") != 1
        or field.get("maximum") != 8
        or set(field)
        - {"type", "default", "minimum", "maximum", "title", "description", "x-lm-atelier-kind"}
    ):
        return input_schema
    return {
        **input_schema,
        "properties": {
            **input_schema["properties"],
            UPSCALE_SETTING_KEY: upscale_setting_schema(graph, input_schema),
        },
    }


def without_inert_upscale_setting(
    values: dict[str, Any], graph: dict[str, Any], input_schema: dict[str, Any] | None
) -> dict[str, Any]:
    """Discard a stored factor when the declared enlargement never reads it."""
    schema = effective_upscale_schema(graph, input_schema)
    if (
        UPSCALE_SETTING_KEY in values
        and workflow_declares_upscale(schema)
        and schema is not None
        and schema["properties"][UPSCALE_SETTING_KEY].get("readOnly") is True
        and not binds_parameter(graph, UPSCALE_SETTING_KEY)
        and upscale_capability(graph) is not None
    ):
        return {key: value for key, value in values.items() if key != UPSCALE_SETTING_KEY}
    return values


def fixed_upscale_factor(
    graph: dict[str, Any], input_schema: dict[str, Any] | None
) -> int | float | None:
    """Resolve an authored fixed binding without creating an editable setting."""
    if not workflow_declares_upscale(input_schema):
        return None
    assert input_schema is not None
    field = input_schema["properties"][UPSCALE_SETTING_KEY]
    if field.get("readOnly") is not True or not binds_parameter(graph, UPSCALE_SETTING_KEY):
        return None
    choices = field.get("enum")
    value = field.get("const", field.get("default"))
    if (
        "const" not in field
        and "default" not in field
        and isinstance(choices, list)
        and len(choices) == 1
    ):
        value = choices[0]
    reason = "The fixed enlargement factor has no valid declared value."
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
        or field.get("type") not in {"number", "integer"}
        or (field.get("type") == "integer" and not isinstance(value, int))
    ):
        raise ValueError(reason)
    for name in ("minimum", "maximum"):
        if name in field:
            bound = field[name]
            if (
                isinstance(bound, bool)
                or not isinstance(bound, int | float)
                or not math.isfinite(bound)
                or (name == "minimum" and value < bound)
                or (name == "maximum" and value > bound)
            ):
                raise ValueError(reason)
    if choices is not None and (
        not isinstance(choices, list)
        or not any(not isinstance(choice, bool) and choice == value for choice in choices)
    ):
        raise ValueError(reason)
    if "multipleOf" in field:
        multiple = field["multipleOf"]
        if (
            isinstance(multiple, bool)
            or not isinstance(multiple, int | float)
            or not math.isfinite(multiple)
            or multiple <= 0
        ):
            raise ValueError(reason)
        quotient = value / multiple
        if not math.isfinite(quotient) or not math.isclose(
            quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(reason)
    return value


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
