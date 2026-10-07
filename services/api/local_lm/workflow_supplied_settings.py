"""Map native controls while preserving explicitly declared executable inputs."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .comfy_workflow_compiler import ComfyWorkflowCompilation, describe_comfyui_api_graph
from .domain import Operation
from .workflow_graph_settings import (
    BoundWorkflowSettings,
    bind_compiled_workflow_settings,
    generated_workflow_setting_paths,
)
from .workflow_graph_settings_v1 import GRAPH_SETTINGS_SCHEMA_KEY
from .workflow_package_inputs import WorkflowPackageInputError
from .workflow_trust import canonical_graph


def bind_api_workflow_settings(
    api_graph: Mapping[str, Mapping[str, Any]],
    object_info: Mapping[str, object],
    input_schema: Mapping[str, Any],
    *,
    operation: Operation | str = Operation.TEXT_TO_IMAGE,
) -> BoundWorkflowSettings:
    """Recheck generated API controls against the current graph and runtime contracts."""
    graph = {key: deepcopy(dict(value)) for key, value in api_graph.items()}
    schema = deepcopy(dict(input_schema))
    generated = generated_workflow_setting_paths(schema, graph)
    properties = schema.get("properties", {})
    for path, placeholder in generated:
        property_schema = properties[placeholder[2:-1]]
        if "default" not in property_schema:
            raise WorkflowPackageInputError(
                "workflow-setting-default-missing",
                "A generated workflow setting has no saved value.",
            )
        graph[str(path[0])]["inputs"][str(path[2])] = deepcopy(property_schema["default"])
        properties.pop(placeholder[2:-1])
    schema.pop(GRAPH_SETTINGS_SCHEMA_KEY, None)
    compilation = describe_comfyui_api_graph(graph, object_info, schema)
    return bind_compiled_workflow_settings(compilation, graph, schema, operation=operation)


def bind_supplied_workflow_settings(
    compilation: ComfyWorkflowCompilation,
    api_graph: Mapping[str, Mapping[str, Any]],
    input_schema: Mapping[str, Any],
    *,
    operation: Operation | str = Operation.TEXT_TO_IMAGE,
) -> BoundWorkflowSettings:
    """Verify executable parity before replacing only generated declarations."""
    graph = {key: deepcopy(dict(value)) for key, value in api_graph.items()}
    schema = deepcopy(dict(input_schema))
    generated = generated_workflow_setting_paths(schema, graph)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise WorkflowPackageInputError(
            "workflow-setting-schema-invalid", "Workflow settings properties must be an object."
        )
    generated_inputs = {(path[0], path[2]) for path, _ in generated}
    inputs = {(item.node_id, item.name): item for item in compilation.inputs}
    explicit: set[tuple[str, str]] = set()
    expected = deepcopy(compilation.api_graph)
    for node_id, node in graph.items():
        native = expected.get(node_id)
        supplied = node.get("inputs") if isinstance(node, Mapping) else None
        native_inputs = native.get("inputs") if isinstance(native, Mapping) else None
        if (
            not isinstance(supplied, dict)
            or not isinstance(native, dict)
            or not isinstance(native_inputs, dict)
        ):
            continue
        for name, value in list(supplied.items()):
            if not isinstance(value, str) or not value.startswith("${") or not value.endswith("}"):
                continue
            item = inputs.get((node_id, name))
            if (
                item is None
                or item.origin != "widget"
                or not isinstance(properties.get(value[2:-1]), Mapping)
            ):
                continue
            if (node_id, name) in generated_inputs:
                supplied[name] = deepcopy(native_inputs[name])
            else:
                native_inputs[name] = value
                explicit.add((node_id, name))
        # Display metadata does not change executable inputs.
        node.pop("_meta", None)
        native.pop("_meta", None)
    if canonical_graph(graph) != canonical_graph(expected):
        raise WorkflowPackageInputError(
            "workflow-setting-graph-mismatch",
            "The visual workflow does not match its executable inputs.",
        )
    for node_id, node in graph.items():
        if "_meta" in api_graph[node_id]:
            node["_meta"] = deepcopy(api_graph[node_id]["_meta"])
    for _, placeholder in generated:
        properties.pop(placeholder[2:-1])
    schema.pop(GRAPH_SETTINGS_SCHEMA_KEY, None)
    return bind_compiled_workflow_settings(
        replace(
            compilation,
            inputs=tuple(
                item for item in compilation.inputs if (item.node_id, item.name) not in explicit
            ),
        ),
        graph,
        schema,
        operation=operation,
    )
