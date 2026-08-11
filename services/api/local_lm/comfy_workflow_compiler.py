from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .comfy_package_widgets import (
    PackageClaim,
    PackageWidgetError,
    package_named_widget_inputs,
    package_widget_inputs,
)
from .comfy_subgraphs import SubgraphExpansionError, expand_workflow
from .comfy_version_support import COMFY_VERSION_SUPPORT
from .comfy_workflow_packages import (
    WorkflowPackageError,
    analyze_comfyui_workflow_package,
)

_SUPPORTED_WIDGET_TYPES = frozenset({"BOOLEAN", "COMBO", "FLOAT", "INT", "STRING"})
_CONTROL_AFTER_GENERATE = frozenset({"decrement", "fixed", "increment", "randomize"})
_CONTROL_WIDGET = "control_after_generate"
# Nodes that draw something and carry nothing. Dropping one loses a caption.
_IGNORED_FRONTEND_NODE_TYPES = frozenset(
    {"MarkdownNote", "Note", "PrimitiveNode", "Label (rgthree)"}
)
# Nodes that carry a wire and nothing else. They are resolved away before
# compilation rather than ignored: ignoring one drops the edge it was
# carrying, which is a different graph, not a smaller one.
#
# `Reroute` carries its wire to whatever it touches. The KJNodes pair carries one
# by name instead: `SetNode` labels the link feeding it, and any `GetNode`
# holding the same label re-emits that link from anywhere in the graph. The
# difference is only in how the far end is found, so both resolve here.
_NAMED_WIRE_SOURCE_TYPE = "SetNode"
_NAMED_WIRE_SINK_TYPE = "GetNode"
_PASS_THROUGH_TYPES = frozenset({"Reroute", _NAMED_WIRE_SOURCE_TYPE, _NAMED_WIRE_SINK_TYPE})
# Deliberately without the named-wire pair. This gate asks whether the ComfyUI
# frontend version is one whose `PrimitiveNode` and `Reroute` semantics we have
# certified. Named wires are not that frontend's construct - they come from a
# third-party package, and each node records the exact package revision that
# defined it - so asking the core frontend version about them would be asking
# the wrong thing and answering confidently.
_FRONTEND_SEMANTIC_NODE_TYPES = frozenset({"PrimitiveNode", "Reroute"})


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
    # A subgraph is a frontend construct the runtime has never seen, but the
    # file carries everything needed to rewrite it away: the definition, its
    # boundary slots, the instances, and the links. Expansion is exact or it
    # raises, and the refusal below still stands for anything it could not
    # rewrite - an approximation would compile, run, and produce a picture
    # that is quietly not the one the author drew.
    try:
        workflow = expand_workflow(workflow)
    except SubgraphExpansionError as exc:
        raise WorkflowCompilationError(exc.code, str(exc)) from exc
    # Converted rather than left to propagate. Everything this function raises is
    # "this graph cannot be compiled", and a caller that catches that should not
    # also have to catch the type the analysis happens to use - one call site
    # compiles a graph handed back by the visual editor with no analysis in
    # front of it, so a malformed graph escaped as an unhandled error rather
    # than as the refusal it is. The code is kept, so nothing loses detail.
    try:
        analysis = analyze_comfyui_workflow_package(
            workflow,
            available_node_types=object_info.keys(),
        )
    except WorkflowCompilationError:
        raise
    except WorkflowPackageError as exc:
        raise WorkflowCompilationError(exc.code, str(exc)) from exc
    if analysis.subgraph_count:
        raise WorkflowCompilationError(
            "unsupported_subgraphs",
            "workflow subgraphs require the ComfyUI frontend to compile",
        )
    unsupported_frontend = (
        set(analysis.frontend_node_types) - _IGNORED_FRONTEND_NODE_TYPES - _PASS_THROUGH_TYPES
    )
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
    frontend_semantic_nodes = set(analysis.frontend_node_types) & _FRONTEND_SEMANTIC_NODE_TYPES
    if frontend_semantic_nodes:
        frontend_support = COMFY_VERSION_SUPPORT.evaluate_frontend_semantics(
            analysis.frontend_version
        )
        if not frontend_support.supported:
            raise WorkflowCompilationError(
                "unsupported_frontend_version",
                (
                    "workflow uses PrimitiveNode or Reroute semantics without a "
                    "certified ComfyUI frontend version"
                ),
            )

    nodes = _nodes_by_id(workflow)
    links = tuple(_parse_link(value) for value in _sequence(workflow.get("links"), "links"))
    nodes, links = _elide_pass_through_nodes(nodes, links)
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
    primitive_values, primitive_link_ids = _primitive_widget_values(
        nodes,
        runtime_nodes,
        slots,
        links,
        object_info,
    )
    connections, successors = _validate_links(
        nodes,
        runtime_nodes,
        slots,
        links,
        primitive_link_ids,
    )
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
            primitive_values,
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


def _drawn_by_package(node: Mapping[str, object]) -> PackageClaim | None:
    """The package and revision a node says drew it.

    Taken from what the graph records rather than from anything installed. The
    question a layout answers is whose editor code has to be reproduced, and the
    saved graph is what states that; an installed package of the same name is a
    different claim and could disagree.

    Both ids are read because a graph may record either, but a node claiming two
    that disagree is refused rather than resolved - picking one would decide
    which package's layout to read by, on no evidence.

    The revision travels with the id and never separately. A layout is a
    transcription of specific code, so which code drew this node is half the
    question, and a package name on its own cannot answer it.
    """

    properties = node.get("properties")
    if not isinstance(properties, Mapping):
        return None
    registry = properties.get("cnr_id")
    repository = properties.get("aux_id")
    revision = properties.get("ver")
    claim = PackageClaim(
        registry if isinstance(registry, str) and registry else None,
        repository if isinstance(repository, str) and repository else None,
        revision if isinstance(revision, str) else "",
    )
    if claim.registry_id is None and claim.repository_id is None:
        return None
    return claim


def _drawn_by(node: Mapping[str, object]) -> str:
    """The same, phrased for a refusal that can be acted on.

    Empty when the graph does not say, because naming the wrong package is worse
    than naming none.
    """

    claim = _drawn_by_package(node)
    if claim is None:
        return " its package"
    if claim.revision:
        return f" {claim.package_id} at {claim.revision}"
    return f" {claim.package_id}"


def _named_wire_label(node_id: str, node: Mapping[str, object]) -> str:
    """The label a named wire is written or read under.

    Held in the node's first widget, which is also where the frontend keeps it.
    A named wire with no readable label cannot be matched to its other end, and
    guessing which end it meant would connect two things the author did not.
    """

    values = node.get("widgets_values")
    if isinstance(values, Sequence) and not isinstance(values, str | bytes):
        label = values[0] if values else None
        if isinstance(label, str) and label:
            return label
    raise WorkflowCompilationError(
        "invalid_named_wire", f"node {node_id} is a named wire with no name"
    )


def _named_wire_writers(nodes: Mapping[str, Mapping[str, object]]) -> dict[str, list[str]]:
    """Which nodes write each label.

    Every writer is kept rather than the first, because a label written twice is
    only ambiguous where something reads it. An unread duplicate is a stray node
    the author left behind, and the frontend runs that graph, so refusing it here
    would reject a workflow that works.
    """

    writers: dict[str, list[str]] = {}
    for node_id, node in nodes.items():
        if str(node.get("type")) != _NAMED_WIRE_SOURCE_TYPE:
            continue
        writers.setdefault(_named_wire_label(node_id, node), []).append(node_id)
    return writers


def _elide_pass_through_nodes(
    nodes: Mapping[str, Mapping[str, object]],
    links: Sequence[_Link],
) -> tuple[dict[str, Mapping[str, object]], tuple[_Link, ...]]:
    """Remove nodes that only carry a wire, reconnecting what they carried.

    A `Reroute` exists to make a graph readable. It registers no class with the
    runtime and holds no value, so it cannot be compiled - but it also cannot
    simply be dropped, because dropping it takes an edge with it and yields a
    graph that runs and quietly makes a different picture. It has to be resolved:
    every consumer reconnects to whatever fed the reroute, through however many
    reroutes stand between them.

    A named wire is the same problem with the far end named rather than touched.
    A `SetNode` labels the link feeding it; every `GetNode` carrying that label
    stands in for that link, wherever in the graph it sits. Resolving one is
    therefore a lookup by label rather than a step along an edge, but what
    happens next is identical, so both walk the same chain and a wire may cross
    reroutes and labels in any order.

    The refusal that matters is a carrier with consumers and nothing feeding it.
    Dropping that silently would leave a required input unfilled, so it is named
    instead - as is a label no `SetNode` defines, and a label two of them claim.
    A label whose two definitions disagree has no correct reading, and picking
    either would compile a graph the author never drew.
    """

    pass_through = {
        node_id for node_id, node in nodes.items() if str(node.get("type")) in _PASS_THROUGH_TYPES
    }
    if not pass_through:
        return dict(nodes), tuple(links)

    incoming = {link.target: link for link in links if link.target in pass_through}
    writers = _named_wire_writers(nodes)
    for node_id, node in nodes.items():
        if str(node.get("type")) != _NAMED_WIRE_SINK_TYPE:
            continue
        label = _named_wire_label(node_id, node)
        defined = writers.get(label, [])
        if not defined:
            raise WorkflowCompilationError(
                "undefined_named_wire",
                f"node {node_id} reads workflow value {label}, which nothing sets",
            )
        if len(defined) > 1:
            raise WorkflowCompilationError(
                "duplicate_named_wire",
                f"node {node_id} reads workflow value {label}, which more than one node sets",
            )
        source = defined[0]
        # Stands in for the edge a reader does not draw. Only its origin is read,
        # and the walk continues from the writer - so a writer that carries
        # nothing is reported as the writer it is, not as the reader that found
        # it.
        incoming[node_id] = _Link(f"named-wire:{label}", source, 0, node_id, 0)

    def describe(node_id: str) -> str:
        node_type = str(nodes[node_id].get("type"))
        if node_type == _NAMED_WIRE_SOURCE_TYPE:
            return f"workflow value {_named_wire_label(node_id, nodes[node_id])}"
        if node_type == _NAMED_WIRE_SINK_TYPE:
            return f"the read of workflow value {_named_wire_label(node_id, nodes[node_id])}"
        return f"reroute node {node_id}"

    def resolve(link: _Link) -> _Link:
        origin, origin_slot = link.origin, link.origin_slot
        seen: set[str] = set()
        while origin in pass_through:
            if origin in seen:
                raise WorkflowCompilationError(
                    "pass_through_cycle",
                    f"{describe(origin)} feeds itself",
                )
            seen.add(origin)
            feeding = incoming.get(origin)
            if feeding is None:
                raise WorkflowCompilationError(
                    "unconnected_pass_through",
                    f"{describe(origin)} carries a connection but nothing feeds it",
                )
            origin, origin_slot = feeding.origin, feeding.origin_slot
        if origin == link.origin and origin_slot == link.origin_slot:
            return link
        return _Link(link.identifier, origin, origin_slot, link.target, link.target_slot)

    # A link into a reroute is consumed by the resolution; only links leaving the
    # elided set survive, each keeping its own identity so the graph downstream
    # is untouched.
    rewritten = tuple(resolve(link) for link in links if link.target not in pass_through)

    # A node also declares, on each output slot, which links leave it. Rewiring
    # moves links onto a node that never listed them and away from one that did,
    # and the compiler checks that declaration against the links themselves - so
    # it is restated here from the resolved set rather than left contradicting it.
    outgoing: dict[tuple[str, int], list[str]] = {}
    for link in rewritten:
        outgoing.setdefault((link.origin, link.origin_slot), []).append(link.identifier)

    remaining: dict[str, Mapping[str, object]] = {}
    for node_id, node in nodes.items():
        if node_id in pass_through:
            continue
        outputs = node.get("outputs")
        if isinstance(outputs, Sequence) and not isinstance(outputs, str | bytes):
            restated = [
                {**slot, "links": outgoing.get((node_id, index), [])}
                if isinstance(slot, Mapping)
                else slot
                for index, slot in enumerate(outputs)
            ]
            remaining[node_id] = {**node, "outputs": restated}
        else:
            remaining[node_id] = node
    return remaining, rewritten


def _validate_links(
    nodes: Mapping[str, Mapping[str, object]],
    runtime_nodes: Mapping[str, Mapping[str, object]],
    slots: Mapping[
        str,
        tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]],
    ],
    links: Sequence[_Link],
    primitive_link_ids: frozenset[str],
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
        if link.identifier in primitive_link_ids:
            continue
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


def _primitive_widget_values(
    nodes: Mapping[str, Mapping[str, object]],
    runtime_nodes: Mapping[str, Mapping[str, object]],
    slots: Mapping[
        str,
        tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]],
    ],
    links: Sequence[_Link],
    object_info: Mapping[str, object],
) -> tuple[dict[tuple[str, str], object], frozenset[str]]:
    """Resolve PrimitiveNode propagation after the shared version preflight."""

    primitive_ids = {
        node_id for node_id, node in nodes.items() if node.get("type") == "PrimitiveNode"
    }
    if not primitive_ids:
        return {}, frozenset()

    links_by_origin: dict[str, list[_Link]] = {node_id: [] for node_id in primitive_ids}
    target_counts: dict[tuple[str, int], int] = {}
    for link in links:
        if link.origin in primitive_ids:
            links_by_origin[link.origin].append(link)
        target_key = (link.target, link.target_slot)
        target_counts[target_key] = target_counts.get(target_key, 0) + 1

    resolved: dict[tuple[str, str], object] = {}
    consumed_links: set[str] = set()
    for node_id in sorted(primitive_ids):
        node = nodes[node_id]
        mode = node.get("mode", 0)
        if isinstance(mode, bool) or not isinstance(mode, int) or mode != 0:
            raise WorkflowCompilationError(
                "unsupported_primitive_node",
                f"primitive node {node_id} uses an unsupported execution mode",
            )
        properties = node.get("properties", {})
        if not isinstance(properties, Mapping):
            raise WorkflowCompilationError(
                "unsupported_primitive_node",
                f"primitive node {node_id} has invalid properties",
            )
        replace_value = properties.get("Run widget replace on values", False)
        if not isinstance(replace_value, bool) or replace_value:
            raise WorkflowCompilationError(
                "unsupported_primitive_node",
                f"primitive node {node_id} requires frontend text replacement",
            )

        inputs, outputs = slots[node_id]
        if inputs or len(outputs) != 1:
            raise WorkflowCompilationError(
                "unsupported_primitive_node",
                f"primitive node {node_id} has an unsupported slot shape",
            )
        output = outputs[0]
        widget = output.get("widget")
        if not isinstance(widget, Mapping):
            raise WorkflowCompilationError(
                "unsupported_primitive_node",
                f"primitive node {node_id} has no widget contract",
            )
        widget_name = widget.get("name")
        if not isinstance(widget_name, str) or not widget_name:
            raise WorkflowCompilationError(
                "unsupported_primitive_node",
                f"primitive node {node_id} has an invalid widget contract",
            )

        raw_values = node.get("widgets_values", [])
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, str | bytes):
            raise WorkflowCompilationError(
                "invalid_widget_values",
                f"primitive node {node_id} widget values must be an array",
            )
        values = list(raw_values)
        if not values or len(values) > 2:
            raise WorkflowCompilationError(
                "unsupported_widget_values",
                f"primitive node {node_id} has widget values that cannot be mapped safely",
            )
        if len(values) == 2 and (not isinstance(values[1], str) or values[1].casefold() != "fixed"):
            raise WorkflowCompilationError(
                "unsupported_primitive_node",
                f"primitive node {node_id} uses a changing widget value",
            )

        for link in links_by_origin[node_id]:
            if link.origin_slot != 0 or link.target not in runtime_nodes:
                raise WorkflowCompilationError(
                    "unsupported_primitive_node",
                    f"primitive node {node_id} has an unsupported connection",
                )
            if target_counts[(link.target, link.target_slot)] != 1:
                raise WorkflowCompilationError(
                    "unsupported_primitive_node",
                    f"primitive node {node_id} feeds an ambiguous input",
                )
            target_inputs = slots[link.target][0]
            if link.target_slot >= len(target_inputs):
                raise WorkflowCompilationError(
                    "invalid_link_slot", "workflow link uses a missing slot"
                )
            target_slot = target_inputs[link.target_slot]
            target_name = str(target_slot["name"])
            target_widget = target_slot.get("widget")
            if (
                not isinstance(target_widget, Mapping)
                or target_widget.get("name") != target_name
                or widget_name != target_name
            ):
                raise WorkflowCompilationError(
                    "unsupported_primitive_node",
                    f"primitive node {node_id} does not feed a matching widget",
                )

            target_type = str(runtime_nodes[link.target]["type"])
            node_info = object_info.get(target_type)
            if not isinstance(node_info, Mapping):
                raise WorkflowCompilationError(
                    "invalid_node_definition",
                    f"ComfyUI definition for {target_type} is invalid",
                )
            definitions = {
                definition.name: definition
                for definition in _input_definitions(target_type, node_info)
            }
            definition = definitions.get(target_name)
            if definition is None or not _is_widget_spec(definition.spec):
                raise WorkflowCompilationError(
                    "unsupported_primitive_node",
                    f"primitive node {node_id} does not feed a serializable widget",
                )
            kind = definition.spec[0]
            expected_type = (
                "COMBO"
                if isinstance(kind, Sequence) and not isinstance(kind, str | bytes)
                else kind
            )
            if output.get("type") != expected_type:
                raise WorkflowCompilationError(
                    "unsupported_primitive_node",
                    f"primitive node {node_id} has incompatible widget metadata",
                )

            resolved_key = (link.target, target_name)
            resolved[resolved_key] = _serialize_widget_value(
                link.target,
                target_name,
                definition.spec,
                values[0],
            )
            consumed_links.add(link.identifier)
    return resolved, frozenset(consumed_links)


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
    primitive_values: Mapping[tuple[str, str], object],
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
    # A graph may save widgets in either of two shapes: positionally, in the
    # order the node draws them, or keyed by the input each belongs to. The
    # second says outright what the first only implies, so it is read by name
    # and never counted - a node that gained or lost a widget shifts every
    # position after it, and that is exactly what naming them avoids.
    named: dict[str, object] | None = None
    values: list[object] = []
    if isinstance(raw_values, Mapping):
        named = {str(key): value for key, value in raw_values.items()}
    elif isinstance(raw_values, Sequence) and not isinstance(raw_values, str | bytes):
        values = list(raw_values)
    else:
        raise WorkflowCompilationError(
            "invalid_widget_values",
            f"node {node_id} widget values must be an array or an object",
        )
    cursor = 0
    taken: set[str] = set()
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
        if named is not None:
            has_value = definition.name in named
            if has_value:
                selected = named[definition.name]
                taken.add(definition.name)
        else:
            has_value = cursor < len(values)
            if has_value:
                selected = values[cursor]
                cursor += 1
        if not has_value:
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
        if not options.get("control_after_generate"):
            continue
        # The control that follows a seed is drawn by the editor and applied
        # before a graph is ever saved, so it is consumed rather than sent.
        if named is not None:
            # The same rule the positional path applies. Consuming any string
            # would swallow a value that is not a control at all and quietly
            # drop whatever the graph meant by it.
            control_value = named.get(_CONTROL_WIDGET)
            if (
                isinstance(control_value, str)
                and control_value.casefold() in _CONTROL_AFTER_GENERATE
            ):
                taken.add(_CONTROL_WIDGET)
            continue
        if cursor >= len(values):
            continue
        control = values[cursor]
        if isinstance(control, str) and control.casefold() in _CONTROL_AFTER_GENERATE:
            cursor += 1
    if named is not None:
        unread = {name: named[name] for name in sorted(set(named) - taken, key=str.casefold)}
        if unread:
            try:
                extras = package_named_widget_inputs(
                    str(node["type"]), _drawn_by_package(node), result, unread
                )
            except PackageWidgetError as exc:
                raise WorkflowCompilationError(exc.code, f"node {node_id}: {exc}") from exc
            if extras is None:
                raise WorkflowCompilationError(
                    "unknown_widget_value",
                    f"node {node_id} saves a value for {next(iter(unread))},"
                    " which it has no input for",
                )
            result.update(extras)
        return _with_connected_inputs(node_id, result, by_name, connections, primitive_values)
    if cursor != len(values):
        try:
            drawn_inputs = package_widget_inputs(
                str(node["type"]), _drawn_by_package(node), values[cursor:]
            )
        except PackageWidgetError as exc:
            # The layout is transcribed and this graph does not match it. Saying
            # so beats compiling it under a layout it no longer uses, which
            # would run and mean something else.
            raise WorkflowCompilationError(exc.code, f"node {node_id}: {exc}") from exc
        if drawn_inputs is not None:
            result.update(drawn_inputs)
            cursor = len(values)
    if cursor != len(values):
        # A leftover scalar is a mapping this compiler got wrong. A leftover
        # object is something else entirely: a widget the node's own package
        # draws and serializes, whose layout lives in that package's editor code
        # and is described nowhere the runtime can be asked. Both must refuse,
        # but only one of them is about this compiler, and saying "cannot be
        # mapped safely" for the other sends someone looking for a defect here.
        drawn = next((value for value in values[cursor:] if isinstance(value, Mapping)), None)
        if drawn is not None:
            raise WorkflowCompilationError(
                "package_serialized_widgets",
                f"node {node_id} keeps its {node['type']} settings in a layout"
                f"{_drawn_by(node)} defines, which only that package can read",
            )
        raise WorkflowCompilationError(
            "unsupported_widget_values",
            f"node {node_id} has widget values that cannot be mapped safely",
        )
    return _with_connected_inputs(node_id, result, by_name, connections, primitive_values)


def _with_connected_inputs(
    node_id: str,
    result: dict[str, object],
    by_name: Mapping[str, _InputDefinition],
    connections: Mapping[tuple[str, str], list[object]],
    primitive_values: Mapping[tuple[str, str], object],
) -> dict[str, object]:
    """A link replaces whatever a widget saved for the same input.

    Shared by both saved shapes rather than written twice, because an input that
    is wired takes its value from the wire in either of them.
    """

    for (target, name), connection in connections.items():
        if target == node_id:
            if name not in by_name:
                raise WorkflowCompilationError(
                    "unknown_input_slot", f"node {node_id} uses unknown input {name}"
                )
            result[name] = list(connection)
    for (target, name), value in primitive_values.items():
        if target == node_id:
            result[name] = value
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
        return list(value)
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
