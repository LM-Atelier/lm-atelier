from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .comfy_workflow_packages import (
    WorkflowPackageError,
    analyze_comfyui_workflow_package,
)

_SUPPORTED_WIDGET_TYPES = frozenset({"BOOLEAN", "COMBO", "FLOAT", "INT", "STRING"})
_CONTROL_AFTER_GENERATE = frozenset({"decrement", "fixed", "increment", "randomize"})
_IGNORED_FRONTEND_NODE_TYPES = frozenset({"MarkdownNote", "Note"})


class WorkflowCompilationError(WorkflowPackageError):
    """A UI workflow cannot be translated without changing its behavior."""


@dataclass(frozen=True)
class ComfyWorkflowCompilation:
    api_graph: dict[str, dict[str, object]]
    execution_order: tuple[str, ...]


@dataclass(frozen=True)
class _Link:
    identifier: str
    origin: str
    origin_slot: int
    target: str
    target_slot: int


@dataclass(frozen=True)
class _InputDefinition:
    name: str
    spec: Sequence[object]
    required: bool


def compile_comfyui_ui_graph(
    workflow: Mapping[str, object],
    object_info: Mapping[str, object],
) -> ComfyWorkflowCompilation:
    """Compile a bounded ComfyUI v0.4 UI graph into API prompt format."""
    if not isinstance(object_info, Mapping):
        raise WorkflowCompilationError(
            "invalid_object_info", "ComfyUI node definitions must be an object"
        )
    analysis = analyze_comfyui_workflow_package(
        workflow,
        available_node_types=object_info.keys(),
    )
    if analysis.subgraph_count:
        raise WorkflowCompilationError(
            "unsupported_subgraphs",
            "workflow subgraphs require the ComfyUI frontend to compile",
        )
    unsupported_frontend = set(analysis.frontend_node_types) - _IGNORED_FRONTEND_NODE_TYPES
    if unsupported_frontend:
        node_type = sorted(unsupported_frontend, key=str.casefold)[0]
        raise WorkflowCompilationError(
            "unsupported_frontend_node",
            f"workflow node type {node_type} requires the ComfyUI frontend",
        )
    if analysis.missing_node_types:
        node_type = analysis.missing_node_types[0]
        raise WorkflowCompilationError(
            "missing_node_type", f"ComfyUI does not provide node type {node_type}"
        )

    nodes = _nodes_by_id(workflow)
    runtime_nodes = {
        node_id: node
        for node_id, node in nodes.items()
        if str(node["type"]) not in _IGNORED_FRONTEND_NODE_TYPES
    }
    if not runtime_nodes:
        raise WorkflowCompilationError(
            "empty_executable_workflow", "workflow contains no executable nodes"
        )
    slots = {node_id: _node_slots(node_id, node) for node_id, node in nodes.items()}
    links = tuple(_parse_link(value) for value in _sequence(workflow.get("links"), "links"))
    connections, successors = _validate_links(nodes, runtime_nodes, slots, links)
    order = _topological_order(tuple(runtime_nodes), successors)

    api_graph: dict[str, dict[str, object]] = {}
    for node_id in order:
        node = runtime_nodes[node_id]
        node_type = str(node["type"])
        node_info = object_info.get(node_type)
        if not isinstance(node_info, Mapping):
            raise WorkflowCompilationError(
                "invalid_node_definition", f"ComfyUI definition for {node_type} is invalid"
            )
        inputs = _compile_node_inputs(
            node_id,
            node,
            node_info,
            slots[node_id][0],
            connections,
        )
        title = node.get("title") or node_info.get("display_name") or node_type
        api_graph[node_id] = {
            "inputs": inputs,
            "class_type": node_type,
            "_meta": {"title": str(title)},
        }
    return ComfyWorkflowCompilation(api_graph, order)


def _nodes_by_id(workflow: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for value in _sequence(workflow.get("nodes"), "nodes"):
        if not isinstance(value, Mapping):
            raise WorkflowCompilationError("invalid_node", "workflow node must be an object")
        identifier = _identifier(value.get("id"), "node id")
        if identifier in result:
            raise WorkflowCompilationError("duplicate_node", "workflow has a duplicate node")
        if value.get("type") not in _IGNORED_FRONTEND_NODE_TYPES:
            mode = value.get("mode", 0)
            if isinstance(mode, bool) or not isinstance(mode, int):
                raise WorkflowCompilationError(
                    "invalid_node_mode", f"node {identifier} has invalid mode"
                )
            if mode != 0:
                raise WorkflowCompilationError(
                    "unsupported_node_mode",
                    f"node {identifier} uses a frontend-only execution mode",
                )
        result[identifier] = value
    return result


def _node_slots(
    node_id: str,
    node: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    inputs = _mapping_sequence(node.get("inputs", []), f"node {node_id} inputs")
    outputs = _mapping_sequence(node.get("outputs", []), f"node {node_id} outputs")
    names: set[str] = set()
    for value in inputs:
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise WorkflowCompilationError(
                "invalid_input_slot", f"node {node_id} has an invalid input slot"
            )
        if name in names:
            raise WorkflowCompilationError(
                "duplicate_input_slot", f"node {node_id} has duplicate input {name}"
            )
        names.add(name)
        widget = value.get("widget")
        if widget is not None and (not isinstance(widget, Mapping) or widget.get("name") != name):
            raise WorkflowCompilationError(
                "ambiguous_widget", f"node {node_id} has ambiguous widget input {name}"
            )
    return inputs, outputs


def _parse_link(value: object) -> _Link:
    if isinstance(value, Mapping):
        raw = (
            value.get("id"),
            value.get("origin_id"),
            value.get("origin_slot"),
            value.get("target_id"),
            value.get("target_slot"),
        )
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) >= 5:
        raw = (value[0], value[1], value[2], value[3], value[4])
    else:
        raise WorkflowCompilationError("invalid_link", "workflow has a malformed link")
    origin_slot = _slot_index(raw[2], "link origin slot")
    target_slot = _slot_index(raw[4], "link target slot")
    return _Link(
        _identifier(raw[0], "link id"),
        _identifier(raw[1], "link origin"),
        origin_slot,
        _identifier(raw[3], "link target"),
        target_slot,
    )


def _validate_links(
    nodes: Mapping[str, Mapping[str, object]],
    runtime_nodes: Mapping[str, Mapping[str, object]],
    slots: Mapping[
        str,
        tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]],
    ],
    links: Sequence[_Link],
) -> tuple[dict[tuple[str, str], list[object]], dict[str, set[str]]]:
    by_id: dict[str, _Link] = {}
    connections: dict[tuple[str, str], list[object]] = {}
    successors: dict[str, set[str]] = {node_id: set() for node_id in runtime_nodes}
    for link in links:
        if link.identifier in by_id:
            raise WorkflowCompilationError("duplicate_link", "workflow has a duplicate link")
        by_id[link.identifier] = link
        if link.origin not in nodes or link.target not in nodes:
            raise WorkflowCompilationError("dangling_link", "workflow has a dangling link")
        if link.origin not in runtime_nodes or link.target not in runtime_nodes:
            raise WorkflowCompilationError(
                "frontend_node_link",
                "frontend-only note nodes cannot participate in workflow links",
            )
        target_inputs, _ = slots[link.target]
        _, origin_outputs = slots[link.origin]
        if link.origin_slot >= len(origin_outputs) or link.target_slot >= len(target_inputs):
            raise WorkflowCompilationError("invalid_link_slot", "workflow link uses a missing slot")
        target_slot = target_inputs[link.target_slot]
        target_name = str(target_slot["name"])
        key = (link.target, target_name)
        if key in connections:
            raise WorkflowCompilationError(
                "duplicate_input_link", f"node {link.target} input {target_name} has multiple links"
            )
        declared_target = target_slot.get("link")
        if (
            declared_target is not None
            and _identifier(declared_target, "input link") != link.identifier
        ):
            raise WorkflowCompilationError(
                "inconsistent_link", "workflow input metadata disagrees with its link"
            )
        declared_origins = origin_outputs[link.origin_slot].get("links")
        if declared_origins is not None:
            origin_ids = {
                _identifier(value, "output link")
                for value in _sequence(declared_origins, "output links")
            }
            if link.identifier not in origin_ids:
                raise WorkflowCompilationError(
                    "inconsistent_link", "workflow output metadata disagrees with its link"
                )
        connections[key] = [link.origin, link.origin_slot]
        successors[link.origin].add(link.target)

    for node_id, (inputs, outputs) in slots.items():
        for slot_index, value in enumerate(inputs):
            declared = value.get("link")
            if declared is None:
                continue
            link_id = _identifier(declared, "input link")
            declared_link = by_id.get(link_id)
            if (
                declared_link is None
                or declared_link.target != node_id
                or declared_link.target_slot != slot_index
            ):
                raise WorkflowCompilationError(
                    "inconsistent_link", "workflow input references a missing link"
                )
        for slot_index, value in enumerate(outputs):
            declared = value.get("links")
            if declared is None:
                continue
            for raw_link_id in _sequence(declared, "output links"):
                link_id = _identifier(raw_link_id, "output link")
                declared_link = by_id.get(link_id)
                if (
                    declared_link is None
                    or declared_link.origin != node_id
                    or declared_link.origin_slot != slot_index
                ):
                    raise WorkflowCompilationError(
                        "inconsistent_link", "workflow output references a missing link"
                    )
    return connections, successors


def _topological_order(
    node_ids: tuple[str, ...],
    successors: Mapping[str, set[str]],
) -> tuple[str, ...]:
    indegree = {node_id: 0 for node_id in node_ids}
    for targets in successors.values():
        for target in targets:
            indegree[target] += 1
    position = {node_id: index for index, node_id in enumerate(node_ids)}
    pending = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    result: list[str] = []
    while pending:
        node_id = pending.popleft()
        result.append(node_id)
        for target in sorted(successors[node_id], key=position.__getitem__):
            indegree[target] -= 1
            if indegree[target] == 0:
                pending.append(target)
    if len(result) != len(node_ids):
        raise WorkflowCompilationError("workflow_cycle", "workflow contains a dependency cycle")
    return tuple(result)


def _compile_node_inputs(
    node_id: str,
    node: Mapping[str, object],
    node_info: Mapping[str, object],
    input_slots: Sequence[Mapping[str, object]],
    connections: Mapping[tuple[str, str], list[object]],
) -> dict[str, object]:
    definitions = _input_definitions(str(node["type"]), node_info)
    by_name = {definition.name: definition for definition in definitions}
    slots_by_name = {str(slot["name"]): slot for slot in input_slots}
    unknown_slots = set(slots_by_name) - set(by_name)
    if unknown_slots:
        name = sorted(unknown_slots, key=str.casefold)[0]
        raise WorkflowCompilationError(
            "unknown_input_slot", f"node {node_id} uses unknown input {name}"
        )
    raw_values = node.get("widgets_values", [])
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, str | bytes):
        raise WorkflowCompilationError(
            "invalid_widget_values", f"node {node_id} widget values must be an array"
        )
    values = list(raw_values)
    cursor = 0
    result: dict[str, object] = {}
    for definition in definitions:
        key = (node_id, definition.name)
        connected = key in connections
        if not _is_widget_spec(definition.spec):
            slot = slots_by_name.get(definition.name)
            if slot is not None and slot.get("widget") is not None:
                raise WorkflowCompilationError(
                    "unsupported_widget",
                    f"node {node_id} input {definition.name} needs frontend serialization",
                )
            if definition.required and not connected:
                raise WorkflowCompilationError(
                    "missing_required_input",
                    f"node {node_id} is missing required input {definition.name}",
                )
            continue

        selected: object | None = None
        has_value = cursor < len(values)
        if has_value:
            selected = values[cursor]
            cursor += 1
        else:
            selected, has_value = _widget_default(definition.spec)
        if has_value:
            result[definition.name] = _serialize_widget_value(
                node_id, definition.name, definition.spec, selected
            )
        elif definition.required and not connected:
            raise WorkflowCompilationError(
                "missing_required_input",
                f"node {node_id} is missing required input {definition.name}",
            )
        options = _widget_options(definition.spec)
        if (
            options.get("control_after_generate")
            and cursor < len(values)
            and isinstance(values[cursor], str)
            and values[cursor].casefold() in _CONTROL_AFTER_GENERATE
        ):
            cursor += 1
    if cursor != len(values):
        raise WorkflowCompilationError(
            "unsupported_widget_values",
            f"node {node_id} has widget values that cannot be mapped safely",
        )
    for (target, name), connection in connections.items():
        if target == node_id:
            if name not in by_name:
                raise WorkflowCompilationError(
                    "unknown_input_slot", f"node {node_id} uses unknown input {name}"
                )
            result[name] = list(connection)
    return result


def _input_definitions(
    node_type: str,
    node_info: Mapping[str, object],
) -> tuple[_InputDefinition, ...]:
    raw_input = node_info.get("input", {})
    if not isinstance(raw_input, Mapping):
        raise WorkflowCompilationError(
            "invalid_node_definition", f"ComfyUI definition for {node_type} has invalid inputs"
        )
    raw_order = node_info.get("input_order", {})
    if not isinstance(raw_order, Mapping):
        raise WorkflowCompilationError(
            "invalid_node_definition", f"ComfyUI definition for {node_type} has invalid input order"
        )
    result: list[_InputDefinition] = []
    for section, required in (("required", True), ("optional", False)):
        values = raw_input.get(section, {})
        if not isinstance(values, Mapping) or any(not isinstance(name, str) for name in values):
            raise WorkflowCompilationError(
                "invalid_node_definition",
                f"ComfyUI definition for {node_type} has invalid {section} inputs",
            )
        declared_order = raw_order.get(section)
        if declared_order is None:
            names = list(values)
        elif isinstance(declared_order, Sequence) and not isinstance(declared_order, str | bytes):
            names = list(declared_order)
            if any(not isinstance(name, str) for name in names):
                raise WorkflowCompilationError(
                    "invalid_node_definition",
                    f"ComfyUI definition for {node_type} has invalid {section} input order",
                )
            if len(set(names)) != len(names) or set(names) != set(values):
                raise WorkflowCompilationError(
                    "ambiguous_input_order",
                    f"ComfyUI definition for {node_type} has ambiguous {section} input order",
                )
        else:
            raise WorkflowCompilationError(
                "invalid_node_definition",
                f"ComfyUI definition for {node_type} has invalid {section} input order",
            )
        for name in names:
            spec = values[name]
            if not isinstance(spec, Sequence) or isinstance(spec, str | bytes) or not spec:
                raise WorkflowCompilationError(
                    "invalid_node_definition",
                    f"ComfyUI definition for {node_type} input {name} is invalid",
                )
            result.append(_InputDefinition(name, spec, required))
    return tuple(result)


def _is_widget_spec(spec: Sequence[object]) -> bool:
    kind = spec[0]
    if isinstance(kind, Sequence) and not isinstance(kind, str | bytes):
        return True
    return isinstance(kind, str) and kind in _SUPPORTED_WIDGET_TYPES


def _widget_options(spec: Sequence[object]) -> Mapping[str, object]:
    if len(spec) > 1 and isinstance(spec[1], Mapping):
        return spec[1]
    return {}


def _widget_default(spec: Sequence[object]) -> tuple[object | None, bool]:
    kind = spec[0]
    if isinstance(kind, Sequence) and not isinstance(kind, str | bytes):
        return (kind[0], True) if kind else (None, False)
    options = _widget_options(spec)
    if "default" in options:
        return options["default"], True
    choices = options.get("options")
    if kind == "COMBO" and isinstance(choices, Sequence) and not isinstance(choices, str | bytes):
        return (choices[0], True) if choices else (None, False)
    return None, False


def _serialize_widget_value(
    node_id: str,
    name: str,
    spec: Sequence[object],
    value: object,
) -> object:
    kind = spec[0]
    choices: Sequence[object] | None = None
    if isinstance(kind, Sequence) and not isinstance(kind, str | bytes):
        choices = kind
    elif kind == "COMBO":
        candidate = _widget_options(spec).get("options")
        if isinstance(candidate, Sequence) and not isinstance(candidate, str | bytes):
            choices = candidate
    if choices is not None and value not in choices:
        raise WorkflowCompilationError(
            "invalid_widget_choice", f"node {node_id} has invalid value for {name}"
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return {"__value__": list(value)}
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _mapping_sequence(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    sequence = _sequence(value, name)
    if any(not isinstance(item, Mapping) for item in sequence):
        raise WorkflowCompilationError("invalid_structure", f"{name} must contain objects")
    return tuple(item for item in sequence if isinstance(item, Mapping))


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise WorkflowCompilationError("invalid_structure", f"workflow {name} must be an array")
    return value


def _identifier(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise WorkflowCompilationError("invalid_identifier", f"workflow has an invalid {name}")
    text = str(value)
    if (
        not text
        or len(text) > 200
        or any(character < " " or character == "\x7f" for character in text)
    ):
        raise WorkflowCompilationError("invalid_identifier", f"workflow has an invalid {name}")
    return text


def _slot_index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkflowCompilationError("invalid_link_slot", f"workflow has an invalid {name}")
    return value
