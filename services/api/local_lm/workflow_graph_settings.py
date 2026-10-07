"""Bind declared numeric and choice controls without replacing graph connections."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeGuard

from .comfy_workflow_compiler import ComfyWorkflowCompilation, CompiledWorkflowInput
from .domain import Operation, operation_model_role
from .settings_registry import IMAGE_SETTINGS, VIDEO_SETTINGS, validate_settings, workflow_settings
from .workflow_graph_settings_v1 import (
    GRAPH_SETTING_INPUT_NAMES as _SETTING_INPUT_NAMES,
)
from .workflow_graph_settings_v1 import (
    GRAPH_SETTINGS_SCHEMA_KEY,
    workflow_graph_settings,
)
from .workflow_package_inputs import WorkflowPackageInputError


@dataclass(frozen=True)
class BoundWorkflowSettings:
    api_graph: dict[str, dict[str, Any]]
    input_schema: dict[str, Any]


def generated_workflow_setting_paths(
    input_schema: Mapping[str, Any], api_graph: Mapping[str, Any]
) -> frozenset[tuple[tuple[str | int, ...], str]]:
    """Validate generated bindings before replacing their schema declarations."""
    try:
        marker = workflow_graph_settings(input_schema)
    except ValueError as exc:
        raise WorkflowPackageInputError("workflow-setting-bindings-invalid", str(exc)) from exc
    if marker is None:
        return frozenset()
    if set(marker.get("unbound_parameters", [])) & _runtime_parameters(api_graph):
        raise WorkflowPackageInputError(
            "workflow-setting-bindings-invalid",
            "A setting marked unused still has an executable binding.",
        )
    properties = input_schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise WorkflowPackageInputError(
            "workflow-setting-bindings-invalid", "The generated settings record is invalid."
        )
    paths: set[tuple[tuple[str | int, ...], str]] = set()
    parameters: set[str] = set()
    for binding in marker["bindings"]:
        parameter, node_id, name = (binding["parameter"], binding["node_id"], binding["input_name"])
        node = api_graph.get(node_id)
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        placeholder = "${" + parameter + "}"
        path: tuple[str | int, ...] = (node_id, "inputs", name)
        if (
            binding.get("declared_name", name) not in _SETTING_INPUT_NAMES
            or parameter in parameters
            or not isinstance(properties.get(parameter), Mapping)
            or not isinstance(inputs, Mapping)
            or inputs.get(name) != placeholder
            or (path, placeholder) in paths
        ):
            raise WorkflowPackageInputError(
                "workflow-setting-bindings-invalid",
                "A generated setting no longer matches its executable input.",
            )
        paths.add((path, placeholder))
        parameters.add(parameter)
    return frozenset(paths)


def rebind_workflow_graph_settings(
    compilation: ComfyWorkflowCompilation,
    api_graph: Mapping[str, Mapping[str, Any]],
    stored_api_graph: Mapping[str, Any],
    input_schema: Mapping[str, Any],
    *,
    operation: Operation | str = Operation.TEXT_TO_IMAGE,
    map_unmapped: bool = False,
) -> BoundWorkflowSettings:
    """Regenerate mapped controls while preserving unrelated explicit declarations."""
    schema = deepcopy(dict(input_schema))
    if GRAPH_SETTINGS_SCHEMA_KEY not in schema:
        if map_unmapped:
            return bind_compiled_workflow_settings(
                compilation, api_graph, schema, operation=operation
            )
        return BoundWorkflowSettings(
            {key: deepcopy(dict(value)) for key, value in api_graph.items()}, schema
        )
    paths = generated_workflow_setting_paths(schema, stored_api_graph)
    properties = schema.get("properties", {})
    for _, placeholder in paths:
        properties.pop(placeholder[2:-1])
    schema[GRAPH_SETTINGS_SCHEMA_KEY] = {"version": 1, "bindings": []}
    return bind_compiled_workflow_settings(compilation, api_graph, schema, operation=operation)


def _finite_number(value: object) -> TypeGuard[int | float]:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _property(item: CompiledWorkflowInput) -> dict[str, Any] | None:
    kind = item.specification[0]
    options = (
        item.specification[1]
        if len(item.specification) > 1 and isinstance(item.specification[1], Mapping)
        else {}
    )
    choices = (
        kind
        if isinstance(kind, list | tuple)
        else options.get("options")
        if kind == "COMBO"
        else None
    )
    if kind == "COMFY_DYNAMICCOMBO_V3":
        dynamic = options.get("options")
        if not isinstance(dynamic, Sequence) or isinstance(dynamic, str | bytes):
            return None
        if any(
            not isinstance(option, Mapping)
            or not isinstance(option.get("inputs", {}), Mapping)
            or any(option.get("inputs", {}).values())
            for option in dynamic
        ):
            return None
        choices = [option.get("key") for option in dynamic if isinstance(option, Mapping)]
    if isinstance(choices, Sequence) and not isinstance(choices, str | bytes):
        if not choices:
            return None
        if all(isinstance(choice, str) for choice in choices):
            choice_type = "string"
            valid_value = isinstance(item.value, str)
        elif all(_finite_number(choice) for choice in choices):
            choice_type = (
                "integer" if all(isinstance(choice, int) for choice in choices) else "number"
            )
            valid_value = _finite_number(item.value) and (
                choice_type != "integer" or isinstance(item.value, int)
            )
        else:
            return None
        if not valid_value or item.value not in choices:
            return None
        return {"type": choice_type, "default": item.value, "enum": list(choices)}
    if (kind != "INT" and kind != "FLOAT") or not _finite_number(item.value):
        return None
    if kind == "INT" and not isinstance(item.value, int):
        return None
    result: dict[str, Any] = {
        "type": "integer" if kind == "INT" else "number",
        "default": item.value,
    }
    for native, schema in (("min", "minimum"), ("max", "maximum")):
        if native in options:
            if not _finite_number(options[native]):
                return None
            result[schema] = options[native]
    if "minimum" in result and item.value < result["minimum"]:
        return None
    if "maximum" in result and item.value > result["maximum"]:
        return None
    # A widget step changes its editor increment; it does not require divisibility.
    if _finite_number(options.get("step")) and options["step"] > 0:
        result["x-lm-atelier-step"] = options["step"]
    return result


def _setting_name(item: CompiledWorkflowInput) -> str | None:
    return _SETTING_INPUT_NAMES.get(item.declared_name or item.name)


def _input_binding(item: CompiledWorkflowInput) -> dict[str, str]:
    binding = {"node_id": item.node_id, "input_name": item.name}
    if item.declared_name and item.declared_name != item.name:
        binding["declared_name"] = item.declared_name
    return binding


def _runtime_parameters(graph: Mapping[str, Any]) -> set[str]:
    parameters: set[str] = set()
    stack: list[object] = [
        node.get("inputs", {}) for node in graph.values() if isinstance(node, Mapping)
    ]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            parameters.add(value[2:-1])
    return parameters


def bind_compiled_workflow_settings(
    compilation: ComfyWorkflowCompilation,
    api_graph: Mapping[str, Mapping[str, Any]],
    input_schema: Mapping[str, Any],
    *,
    operation: Operation | str = Operation.TEXT_TO_IMAGE,
) -> BoundWorkflowSettings:
    """Return a graph and schema whose controls name the same compiled inputs."""
    graph = {key: deepcopy(dict(value)) for key, value in api_graph.items()}
    schema = deepcopy(dict(input_schema))
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise WorkflowPackageInputError(
            "workflow-setting-schema-invalid", "Workflow settings properties must be an object."
        )
    known_inputs = [item for item in compilation.inputs if _setting_name(item) is not None]
    read_only = {
        name
        for name, declaration in properties.items()
        if isinstance(declaration, Mapping) and declaration.get("readOnly") is True
    }
    candidates = [
        item
        for item in known_inputs
        if item.reaches_output and _setting_name(item) not in read_only
    ]
    counts = Counter(_setting_name(item) for item in candidates)
    titles = Counter((item.node_title, _setting_name(item)) for item in candidates)
    role = operation_model_role(Operation(operation))
    limits = {field.key: field for field in (VIDEO_SETTINGS if role == "video" else IMAGE_SETTINGS)}
    for parameter in _runtime_parameters(graph):
        if parameter in limits and parameter in _SETTING_INPUT_NAMES:
            properties.setdefault(parameter, {})
    bindings: list[dict[str, str]] = []
    fixed: list[dict[str, str]] = []
    for item in known_inputs:
        reason = (
            "disconnected"
            if not item.reaches_output
            else "read_only"
            if _setting_name(item) in read_only
            else "linked"
            if item.origin == "link"
            else "primitive"
            if item.origin == "primitive"
            else ("dynamic" if item.specification[0] == "COMFY_DYNAMICCOMBO_V3" else "unsupported")
            if _property(item) is None
            else None
        )
        if reason is not None:
            title = str(_setting_name(item)).replace("_", " ").title()
            fixed.append(
                {
                    **_input_binding(item),
                    "label": f"{title} ({item.node_title})",
                    "reason": reason,
                }
            )
    for item in candidates:
        if item.origin != "widget":
            continue
        declaration = _property(item)
        if declaration is None:
            continue
        name = str(_setting_name(item))
        if base := limits.get(name):
            for key, boundary, combine in (
                ("minimum", base.minimum, max),
                ("maximum", base.maximum, min),
            ):
                if boundary is not None:
                    declaration[key] = combine(declaration.get(key, boundary), boundary)
            try:
                fields = workflow_settings([base], {"properties": {name: declaration}})
                validate_settings({name: declaration["default"]}, fields)
            except ValueError as exc:
                raise WorkflowPackageInputError(
                    "workflow-setting-default-unsupported",
                    "A saved workflow setting is outside the supported range or type. "
                    "Change that control in the native editor before importing this workflow.",
                ) from exc
            if "enum" in declaration:
                supported = []
                for choice in declaration["enum"]:
                    try:
                        validate_settings({name: choice}, fields)
                    except ValueError:
                        continue
                    supported.append(choice)
                declaration["enum"] = supported
        node = graph.get(item.node_id)
        inputs = node.get("inputs") if node is not None else None
        if (
            node is None
            or node.get("class_type") != item.node_type
            or not isinstance(inputs, dict)
            or inputs.get(item.name) != item.value
        ):
            raise WorkflowPackageInputError(
                "workflow-setting-binding-changed",
                "The compiled workflow no longer matches its native setting input.",
            )
        if counts[name] == 1 and name not in properties:
            parameter = name
            title = name.replace("_", " ").title()
        else:
            suffix = hashlib.sha256(f"{item.node_id}\0{item.name}".encode()).hexdigest()[:12]
            parameter = f"workflow_{name}_{suffix}"
            context = item.node_title
            if titles[(item.node_title, name)] > 1:
                context += f", node {item.node_id}"
            if item.declared_name and item.declared_name != item.name:
                context += f", {item.name}"
            title = f"{name.replace('_', ' ').title()} ({context})"
        if parameter in properties:
            raise WorkflowPackageInputError(
                "workflow-setting-binding-conflict",
                "The workflow already declares a setting for this native control.",
            )
        declaration["title"] = title
        if name in {"aspect_ratio", "megapixels", "round_to_multiple"}:
            declaration["x-lm-atelier-visibility"] = "basic"
        properties[parameter] = declaration
        inputs[item.name] = "${" + parameter + "}"
        bindings.append({"parameter": parameter, **_input_binding(item)})
    schema.update({"type": "object", "properties": properties})
    schema[GRAPH_SETTINGS_SCHEMA_KEY] = {"version": 1, "bindings": bindings}
    if fixed:
        schema[GRAPH_SETTINGS_SCHEMA_KEY]["fixed"] = fixed
    parameters = _runtime_parameters(graph)
    unbound = sorted(
        name for name in properties if name in _SETTING_INPUT_NAMES and name not in parameters
    )
    if unbound:
        schema[GRAPH_SETTINGS_SCHEMA_KEY]["unbound_parameters"] = unbound
    return BoundWorkflowSettings(graph, schema)
