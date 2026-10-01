"""Describe enlargement controls whose values reach the selected image outputs."""

import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from .models import WorkflowRevision
from .schemas import ApiModel, SettingField
from .upscale_workflows import UPSCALE_SETTING_KEY, fixed_upscale_factor


class UpscalePreviewOut(ApiModel):
    version: Literal[1] = 1
    status: Literal["ready"] = "ready"
    workflow_revision_id: str
    factor: SettingField | None
    fixed_factor: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    request_authorized: Literal[False] = False


class UpscaleSelectionUnavailable(ValueError):
    """The exact enlargement selection must be refreshed before applying it."""


@dataclass(frozen=True)
class _Scale:
    bound: bool = False
    fixed: float | None = 1


def _image_scale(
    graph: dict[str, Any], reference: Any, factor_value: float | None = None
) -> _Scale | None:
    visited: set[str] = set()
    bound = False
    fixed: float | None = 1
    while isinstance(reference, list) and len(reference) == 2 and reference[1] == 0:
        node_id = str(reference[0])
        if node_id in visited or len(visited) >= 512:
            return None
        visited.add(node_id)
        node = graph.get(node_id)
        if not isinstance(node, dict):
            return None
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            return None
        kind = node.get("class_type")
        if kind == "LoadImage":
            return _Scale(bound, fixed) if inputs.get("image") == "${input_image}" else None
        if kind == "ImageUpscaleWithModel":
            fixed = None
        elif kind == "ImageScaleBy":
            value = inputs.get("scale_by")
            if value == "${upscale_factor}" and factor_value is None:
                bound, fixed = True, None
            else:
                value = factor_value if value == "${upscale_factor}" else value
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    return None
                if fixed is not None:
                    fixed *= value
                    if not math.isfinite(fixed):
                        return None
        else:
            return None
        reference = inputs.get("image")
    return None


def _numeric_factor_field(fields: list[SettingField]) -> SettingField | None:
    return next(
        (
            field
            for field in fields
            if field.key == UPSCALE_SETTING_KEY
            and field.type in {"number", "integer", "enum"}
            and field.available
            and field.scope == "workflow"
            and all(
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
                for value in field.choices
            )
            and (field.type != "enum" or field.choices)
        ),
        None,
    )


def project_upscale_preview(
    revision: WorkflowRevision,
    fields: list[SettingField],
    effective_settings: dict[str, Any],
) -> UpscalePreviewOut:
    graph = revision.api_graph_json
    declared_factor = fixed_upscale_factor(graph, revision.input_schema_json)
    known_nodes = {
        "LoadImage",
        "ImageScaleBy",
        "ImageUpscaleWithModel",
        "UpscaleModelLoader",
        "SaveImage",
        "PreviewImage",
    }
    if len(graph) > 512 or any(
        not isinstance(node, dict) or node.get("class_type") not in known_nodes
        for node in graph.values()
    ):
        return UpscalePreviewOut(workflow_revision_id=revision.id, factor=None)
    outputs = [
        node
        for node in graph.values()
        if isinstance(node, dict) and node.get("class_type") in {"SaveImage", "PreviewImage"}
    ]
    scales = [
        _image_scale(graph, node.get("inputs", {}).get("images"), declared_factor)
        for node in outputs
        if isinstance(node.get("inputs"), dict)
    ]
    factor = None
    fixed = None
    if scales and len(scales) == len(outputs) and all(scale is not None for scale in scales):
        if all(scale is not None and scale.bound for scale in scales):
            field = _numeric_factor_field(fields)
            if field is not None and len(field.choices) == 1:
                scales = [
                    _image_scale(graph, node["inputs"]["images"], float(field.choices[0]))
                    for node in outputs
                ]
            elif field is not None:
                factor = field.model_copy(
                    update={"default": effective_settings.get(UPSCALE_SETTING_KEY, field.default)}
                )
        values = {scale.fixed for scale in scales if scale is not None}
        if len(values) == 1:
            fixed = values.pop()
    return UpscalePreviewOut(workflow_revision_id=revision.id, factor=factor, fixed_factor=fixed)
