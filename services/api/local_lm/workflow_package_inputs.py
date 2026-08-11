"""Bind runtime-owned inputs into a compiled workflow package, exactly.

Shared ComfyUI graphs save whatever happened to be in the author's input
directory.  That filename is neither portable nor part of an image-edit or
image-to-video workflow's identity: LM Atelier uploads the accepted source for
each run and must substitute that exact runtime name.

Only the core LoadImage contract is handled here.  A graph with no source, more
than one possible source, or a different package's node refuses rather than
quietly choosing the wrong image.

"The wrong image" has three shapes that are easy to miss, and each is refused
here rather than trusted to the shape of an ordinary graph:

* A LoadImage can reach the outputs through its MASK output while its IMAGE
  output goes nowhere.  Substituting there would feed the run's upload in as a
  mask and leave the run with no source image at all, so reachability follows
  the source's IMAGE slot specifically rather than any edge that happens to
  leave the node.
* A second author-supplied file can sit in the graph under a different class,
  which keeps the author's local filename in a workflow whose entire purpose is
  to stop carrying it.  Any other node advertising an upload input is therefore
  counted as a second source.
* A subgraph the author bypassed still expands, and expansion does not carry the
  container's mode inward.  Disabled scopes are dropped before selection so a
  disabled LoadImage is never chosen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .comfy_subgraphs import SubgraphExpansionError, expand_workflow
from .domain import Operation

_SOURCE_IMAGE_OPERATIONS = frozenset({Operation.IMAGE_TO_IMAGE, Operation.IMAGE_TO_VIDEO})
_SOURCE_NODE_TYPE = "LoadImage"
_SOURCE_INPUT_NAME = "image"
_SOURCE_PLACEHOLDER = "${input_image}"
# The audited core contract returns ("IMAGE", "MASK"), checked before this is
# used, so the image a run supplies is slot 0 and the mask is slot 1.
_SOURCE_IMAGE_SLOT = 0
# Expansion names an inner node "<container>:<inner>"; it must match the
# compiler's separator or a bypassed container's contents look top-level.
_SCOPE_SEPARATOR = ":"
_WIDGET_TYPES = frozenset({"BOOLEAN", "COMBO", "FLOAT", "INT", "STRING"})


class WorkflowPackageInputError(ValueError):
    """A package cannot be bound to its declared runtime operation exactly."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreparedWorkflowPackageCompilation:
    ui_graph: Mapping[str, Any]
    object_info: Mapping[str, Any]
    input_schema: dict[str, Any]
    source_node_id: str | None = None
    source_value: str | None = None
    output_node_ids: tuple[str, ...] = ()
    link_input_names: Mapping[str, frozenset[str]] | None = None

    def bind(
        self,
        api_graph: Mapping[str, Mapping[str, object]],
    ) -> dict[str, dict[str, object]]:
        """Replace only the compiler-attested source with its runtime binding."""

        result = {str(key): deepcopy(dict(node)) for key, node in api_graph.items()}
        if self.source_node_id is None:
            return result
        source_nodes = [
            key for key, node in result.items() if node.get("class_type") == _SOURCE_NODE_TYPE
        ]
        if source_nodes != [self.source_node_id]:
            code = (
                "workflow_source_input_missing"
                if not source_nodes
                else "workflow_source_input_ambiguous"
            )
            raise WorkflowPackageInputError(
                code,
                "The compiled workflow no longer has exactly one core LoadImage source.",
            )
        node = result.get(self.source_node_id)
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        if (
            not isinstance(node, Mapping)
            or node.get("class_type") != _SOURCE_NODE_TYPE
            or not isinstance(inputs, Mapping)
            or not isinstance(self.source_value, str)
            or inputs.get(_SOURCE_INPUT_NAME) != self.source_value
        ):
            raise WorkflowPackageInputError(
                "workflow_source_input_binding_failed",
                "The compiled workflow did not preserve its exact source-image input.",
            )
        _require_compiled_source_reaches_outputs(
            result,
            self.source_node_id,
            self.output_node_ids,
            self.link_input_names or {},
        )
        bound_inputs = dict(inputs)
        bound_inputs[_SOURCE_INPUT_NAME] = _SOURCE_PLACEHOLDER
        bound_node = dict(node)
        bound_node["inputs"] = bound_inputs
        result[self.source_node_id] = bound_node
        return result


def prepare_workflow_package_compilation(
    ui_graph: Mapping[str, Any],
    object_info: Mapping[str, Any],
    operation: Operation | str,
) -> PreparedWorkflowPackageCompilation:
    """Prepare one portable source input without relaxing any other widget.

    The author's file is admitted only to a private copy of LoadImage's choices;
    an exact frontend-only upload control is removed from a private graph copy.
    The real graph and inventory remain untouched, and every unrelated widget
    retains normal validation.  Runtime substitution happens only after compile.
    """

    normalized_operation = _normalize_operation(operation)
    if normalized_operation not in _SOURCE_IMAGE_OPERATIONS:
        return PreparedWorkflowPackageCompilation(ui_graph, object_info, {})
    try:
        expanded = expand_workflow(ui_graph)
    except SubgraphExpansionError as exc:
        raise WorkflowPackageInputError(exc.code, str(exc)) from exc
    nodes = expanded.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, str | bytes):
        raise WorkflowPackageInputError(
            "workflow_source_input_missing",
            "The source workflow has no readable node list.",
        )
    # A node's own mode survives expansion; the mode of the container it came
    # from does not.  Without this, a LoadImage inside a subgraph the author
    # bypassed reads as live and can be chosen as the source.
    disabled_scopes = _disabled_scope_prefixes(ui_graph)
    nodes = [node for node in nodes if not _is_in_disabled_scope(node, disabled_scopes)]
    sources = [
        node
        for node in nodes
        if isinstance(node, Mapping)
        and node.get("type") == _SOURCE_NODE_TYPE
        and node.get("mode", 0) == 0
    ]
    if not sources:
        raise WorkflowPackageInputError(
            "workflow_source_input_missing",
            (
                f"{normalized_operation.value} requires one core LoadImage source, "
                "but none was found."
            ),
        )
    if len(sources) != 1:
        raise WorkflowPackageInputError(
            "workflow_source_input_ambiguous",
            (
                f"{normalized_operation.value} requires one core LoadImage source, but "
                f"{len(sources)} were found."
            ),
        )
    source = sources[0]
    _require_core_load_image(source)
    _require_no_second_uploaded_source(nodes, object_info, source)
    source_id = _node_identifier(source)
    output_node_ids = _executable_output_node_ids(nodes, object_info)
    link_input_names = _link_input_names_by_node_type(object_info)
    load_image_info = object_info.get(_SOURCE_NODE_TYPE)
    if not isinstance(load_image_info, Mapping):
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The media runtime does not advertise the core LoadImage source contract.",
        )
    _require_core_runtime_load_image(load_image_info)
    section, position, spec, widget_count = _source_widget_definition(load_image_info)
    source_value = _source_widget_value(
        source.get("widgets_values", []),
        position,
    )
    prepared_widget_values = _source_widgets_for_compilation(
        source.get("widgets_values", []),
        position,
        widget_count,
    )
    prepared_info = dict(object_info)
    prepared_load_image = deepcopy(dict(load_image_info))
    input_sections = prepared_load_image.get("input")
    if not isinstance(input_sections, dict):
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The media runtime advertises an invalid LoadImage source contract.",
        )
    definitions = input_sections.get(section)
    if not isinstance(definitions, dict):
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The media runtime advertises an invalid LoadImage source contract.",
        )
    definitions[_SOURCE_INPUT_NAME] = _with_source_choice(spec, source_value)
    prepared_info[_SOURCE_NODE_TYPE] = prepared_load_image

    prepared_nodes = list(nodes)
    source_index = next(index for index, node in enumerate(prepared_nodes) if node is source)
    prepared_source = dict(source)
    prepared_source["widgets_values"] = prepared_widget_values
    prepared_nodes[source_index] = prepared_source
    prepared_graph = dict(expanded)
    prepared_graph["nodes"] = prepared_nodes
    return PreparedWorkflowPackageCompilation(
        prepared_graph,
        prepared_info,
        {
            "type": "object",
            "properties": {_SOURCE_PLACEHOLDER[2:-1]: {"type": "string"}},
        },
        source_id,
        source_value,
        output_node_ids,
        link_input_names,
    )


def prepare_workflow_revision_compilation(
    ui_graph: Mapping[str, Any],
    object_info: Mapping[str, Any],
    operation: Operation | str,
    stored_api_graph: Mapping[str, Any],
    input_schema: Mapping[str, Any],
) -> PreparedWorkflowPackageCompilation:
    """Reapply source normalization only for an existing exact stored binding.

    Native editing also supports legacy, static, and numbered-reference image
    workflows.  Their prior raw compile/compare behavior must not be reclassified
    merely because their operation accepts an image.
    """

    if not _has_exact_source_binding(stored_api_graph, input_schema):
        return PreparedWorkflowPackageCompilation(ui_graph, object_info, {})
    return prepare_workflow_package_compilation(ui_graph, object_info, operation)


def _normalize_operation(operation: Operation | str) -> Operation:
    try:
        return operation if isinstance(operation, Operation) else Operation(operation)
    except (TypeError, ValueError) as exc:
        raise WorkflowPackageInputError(
            "workflow_source_operation_unsupported",
            "The workflow declares an unsupported runtime operation.",
        ) from exc


def _has_exact_source_binding(
    api_graph: Mapping[str, Any],
    input_schema: Mapping[str, Any],
) -> bool:
    properties = input_schema.get("properties")
    declaration = properties.get("input_image") if isinstance(properties, Mapping) else None
    if not isinstance(declaration, Mapping) or declaration.get("type") != "string":
        return False
    sources = [
        node
        for node in api_graph.values()
        if isinstance(node, Mapping) and node.get("class_type") == _SOURCE_NODE_TYPE
    ]
    if len(sources) != 1:
        return False
    inputs = sources[0].get("inputs")
    return isinstance(inputs, Mapping) and inputs.get(_SOURCE_INPUT_NAME) == _SOURCE_PLACEHOLDER


def _require_core_load_image(node: Mapping[str, object]) -> None:
    properties = node.get("properties")
    if properties is None:
        return
    if not isinstance(properties, Mapping):
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The source LoadImage node has invalid package metadata.",
        )
    claims = {
        value
        for key in ("cnr_id", "aux_id")
        if isinstance((value := properties.get(key)), str) and value
    }
    if claims and claims != {"comfy-core"}:
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The source node is not the core LoadImage contract.",
        )


def _require_core_runtime_load_image(node_info: Mapping[str, object]) -> None:
    outputs = node_info.get("output")
    if (
        node_info.get("python_module") != "nodes"
        or not isinstance(outputs, Sequence)
        or isinstance(outputs, str | bytes)
        or tuple(outputs) != ("IMAGE", "MASK")
    ):
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The runtime LoadImage class is not the audited core source contract.",
        )


def _node_identifier(node: Mapping[str, object]) -> str:
    value = node.get("id")
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The source LoadImage node has an invalid identity.",
        )
    return str(value)


def _disabled_scope_prefixes(ui_graph: Mapping[str, Any]) -> tuple[str, ...]:
    """Identity prefixes of containers the author bypassed or muted.

    Read from the graph as saved, before expansion, because expansion inlines a
    subgraph's contents without carrying the instance's mode inward.
    """
    nodes = ui_graph.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, str | bytes):
        return ()
    prefixes: list[str] = []
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("mode", 0) == 0:
            continue
        value = node.get("id")
        if not isinstance(value, bool) and isinstance(value, int | str):
            prefixes.append(f"{value}{_SCOPE_SEPARATOR}")
    return tuple(prefixes)


def _is_in_disabled_scope(node: object, prefixes: Sequence[str]) -> bool:
    if not prefixes or not isinstance(node, Mapping):
        return False
    value = node.get("id")
    if isinstance(value, bool) or not isinstance(value, int | str):
        return False
    node_id = str(value)
    return any(node_id.startswith(prefix) for prefix in prefixes)


def _require_no_second_uploaded_source(
    nodes: Sequence[object],
    object_info: Mapping[str, Any],
    source: Mapping[str, object],
) -> None:
    """Refuse a graph carrying a second author-supplied file.

    The "exactly one source" rule cannot be enforced by class name alone: a
    second loader of any other class - `LoadImageOutput`, a mask loader, a
    package's own uploader - keeps the author's local filename in the compiled
    package, which is the exact non-portability this module exists to remove.

    An upload input is the evidence, because that is what marks a widget whose
    value is a file the author put in their own input directory.  A node that
    merely produces an image without one - `EmptyImage`, a generator - is not a
    second source and is left alone.
    """
    offenders: set[str] = set()
    for node in nodes:
        if not isinstance(node, Mapping) or node is source or node.get("mode", 0) != 0:
            continue
        node_type = node.get("type")
        if not isinstance(node_type, str):
            continue
        node_info = object_info.get(node_type)
        if isinstance(node_info, Mapping) and _declares_upload_input(node_info):
            offenders.add(node_type)
    if offenders:
        listed = ", ".join(sorted(offenders))
        raise WorkflowPackageInputError(
            "workflow_source_input_ambiguous",
            (
                "This workflow loads more than one supplied file, so which one the "
                f"run should replace is ambiguous: {listed}."
            ),
        )


def _declares_upload_input(node_info: Mapping[str, object]) -> bool:
    sections = node_info.get("input")
    if not isinstance(sections, Mapping):
        return False
    for section_name in ("required", "optional"):
        definitions = sections.get(section_name)
        if not isinstance(definitions, Mapping):
            continue
        for spec in definitions.values():
            if (
                isinstance(spec, Sequence)
                and not isinstance(spec, str | bytes)
                and len(spec) > 1
                and isinstance(spec[1], Mapping)
                and any(
                    spec[1].get(control) is True
                    for control in ("image_upload", "video_upload", "audio_upload")
                )
            ):
                return True
    return False


def _executable_output_node_ids(
    nodes: Sequence[object],
    object_info: Mapping[str, Any],
) -> tuple[str, ...]:
    result: set[str] = set()
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("mode", 0) != 0:
            continue
        node_type = node.get("type")
        node_info = object_info.get(node_type) if isinstance(node_type, str) else None
        if isinstance(node_info, Mapping) and node_info.get("output_node") is True:
            result.add(_node_identifier(node))
    return tuple(sorted(result))


def _require_compiled_source_reaches_outputs(
    graph: Mapping[str, Mapping[str, object]],
    source_id: str,
    output_node_ids: Sequence[str],
    link_input_names: Mapping[str, frozenset[str]],
) -> None:
    node_ids = frozenset(graph)
    # Edges carry the origin's output slot, because leaving the source through
    # its MASK output is not the source image reaching anything.
    adjacency: dict[str, set[tuple[str, int]]] = {node_id: set() for node_id in node_ids}
    for target_id, node in graph.items():
        inputs = node.get("inputs")
        node_type = node.get("class_type")
        if not isinstance(inputs, Mapping) or not isinstance(node_type, str):
            continue
        for name in link_input_names.get(node_type, frozenset()):
            value = inputs.get(name)
            if (
                isinstance(value, Sequence)
                and not isinstance(value, str | bytes)
                and len(value) == 2
                and isinstance(value[0], int | str)
                and not isinstance(value[0], bool)
                and isinstance(value[1], int)
                and not isinstance(value[1], bool)
            ):
                origin_id = str(value[0])
                if origin_id in node_ids:
                    adjacency[origin_id].add((target_id, value[1]))
    required_outputs = frozenset(output_node_ids)
    if not required_outputs:
        # Distinct from the reachability failure below: nothing is wrong with
        # the source, the graph simply produces nothing.  Saying "the source
        # does not feed every output" here sends the reader after the wrong
        # thing, because it is vacuously true of a graph with no outputs.
        raise WorkflowPackageInputError(
            "workflow_source_output_missing",
            "This workflow has no executable output node.",
        )
    reachable: set[str] = set()
    pending = [source_id]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        for target_id, slot in adjacency.get(current, ()):
            # Only the image leaving the source counts; downstream of it every
            # slot is ordinary dataflow.
            if current == source_id and slot != _SOURCE_IMAGE_SLOT:
                continue
            pending.append(target_id)
    if not required_outputs.issubset(reachable):
        raise WorkflowPackageInputError(
            "workflow_source_input_not_used",
            "The source image does not feed every executable workflow output.",
        )


def _link_input_names_by_node_type(
    object_info: Mapping[str, Any],
) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for node_type, raw_node_info in object_info.items():
        if not isinstance(node_type, str) or not isinstance(raw_node_info, Mapping):
            continue
        sections = raw_node_info.get("input")
        if not isinstance(sections, Mapping):
            continue
        names: set[str] = set()
        for section_name in ("required", "optional"):
            definitions = sections.get(section_name)
            if not isinstance(definitions, Mapping):
                continue
            for raw_name, spec in definitions.items():
                if (
                    isinstance(raw_name, str)
                    and isinstance(spec, Sequence)
                    and not isinstance(spec, str | bytes)
                    and spec
                    and not _is_widget_spec(spec)
                ):
                    names.add(raw_name)
        if names:
            result[node_type] = frozenset(names)
    return result


def _source_widget_definition(
    node_info: Mapping[str, object],
) -> tuple[str, int, Sequence[object], int]:
    raw_input = node_info.get("input")
    raw_order = node_info.get("input_order", {})
    if not isinstance(raw_input, Mapping) or not isinstance(raw_order, Mapping):
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The media runtime advertises an invalid LoadImage source contract.",
        )
    widget_position = 0
    source_definition: tuple[str, int, Sequence[object]] | None = None
    for section in ("required", "optional"):
        definitions = raw_input.get(section, {})
        order = raw_order.get(section)
        if not isinstance(definitions, Mapping):
            continue
        names = (
            order
            if isinstance(order, Sequence) and not isinstance(order, str | bytes)
            else definitions
        )
        for raw_name in names:
            name = str(raw_name)
            spec = definitions.get(name)
            if not _is_widget_spec(spec):
                continue
            if name == _SOURCE_INPUT_NAME:
                assert isinstance(spec, Sequence)
                source_definition = section, widget_position, spec
            widget_position += 1
    if source_definition is None:
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The media runtime does not expose LoadImage.image as an upload widget.",
        )
    section, position, spec = source_definition
    if position != 0 or widget_position != 1:
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The media runtime advertises an unaudited LoadImage widget layout.",
        )
    return section, position, spec, widget_position


def _is_widget_spec(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        return False
    kind = value[0]
    return (
        isinstance(kind, Sequence)
        and not isinstance(kind, str | bytes)
        or isinstance(kind, str)
        and kind in _WIDGET_TYPES
    )


def _with_source_choice(spec: Sequence[object], source_value: str) -> list[object]:
    result = deepcopy(list(spec))
    options = result[1] if len(result) > 1 and isinstance(result[1], Mapping) else {}
    if options.get("image_upload") is not True:
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "LoadImage.image is not advertised as an upload input.",
        )
    kind = result[0]
    if isinstance(kind, Sequence) and not isinstance(kind, str | bytes):
        choices = list(kind)
        if source_value not in choices:
            choices.append(source_value)
        result[0] = choices
        return result
    option_values = options.get("options")
    if (
        kind == "COMBO"
        and isinstance(option_values, Sequence)
        and not isinstance(option_values, str | bytes)
    ):
        copied_options = dict(options)
        choices = list(option_values)
        if source_value not in choices:
            choices.append(source_value)
        copied_options["options"] = choices
        result[1] = copied_options
        return result
    raise WorkflowPackageInputError(
        "workflow_source_input_unsupported",
        "LoadImage.image does not advertise a bounded file choice.",
    )


def _source_widget_value(raw_values: object, position: int) -> str:
    if isinstance(raw_values, Mapping):
        value = raw_values.get(_SOURCE_INPUT_NAME)
        if isinstance(value, str):
            return value
    if (
        isinstance(raw_values, Sequence)
        and not isinstance(raw_values, str | bytes)
        and position < len(raw_values)
        and isinstance(raw_values[position], str)
    ):
        return str(raw_values[position])
    raise WorkflowPackageInputError(
        "workflow_source_input_unsupported",
        "The source LoadImage does not contain one saved scalar file choice.",
    )


def _source_widgets_for_compilation(
    raw_values: object,
    position: int,
    widget_count: int,
) -> object:
    if isinstance(raw_values, Mapping):
        return deepcopy(dict(raw_values))
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, str | bytes):
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The source LoadImage widgets are not saved in a supported shape.",
        )
    result = deepcopy(list(raw_values))
    if len(result) == widget_count + 1 and result[-1] == "image":
        result.pop()
    elif len(result) != widget_count:
        raise WorkflowPackageInputError(
            "workflow_source_input_unsupported",
            "The source LoadImage has unsupported saved upload controls.",
        )
    return result
