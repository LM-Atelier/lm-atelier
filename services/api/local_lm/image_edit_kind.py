"""Tell an instruction edit from a strength edit by the workflow's own graph.

An instruction edit feeds the source picture into the model's conditioning
beside the words, so the words say what to change. A strength edit encodes the
source into the starting latent and redraws it: how much changes is the denoise
strength, so a small, specific request may not take. Anything that is neither -
a utility that removes a background or resizes, a graph that takes no words, or
one that cannot be read - is "other", and is never treated as an instruction
edit.

The test is structural and never keyed on a model name. Only the part of the
executable graph that produces the saved picture counts: a muted node, a branch
that reaches no saved picture, or one that exists only in the UI graph never
makes a workflow an instruction edit.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from .comfy_subgraphs import expand_workflow
from .domain import Operation
from .graph_placeholders import binds_parameter
from .prompt_binding import binds_prompt
from .workflow_edit_calibration import safe_workflow_edit_calibration

ImageEditKind = Literal["instruction", "strength", "other"]

#: The parameter an uncalibrated strength edit binds.
STRENGTH_PARAMETER = "denoise"

#: Nodes that save the edited picture. The compiler treats the advanced saver
#: as the ordinary one.
SAVED_PICTURE_NODE_TYPES = frozenset({"SaveImage", "SaveImageAdvanced"})

#: Modes the compiler leaves out of the executable graph: muted and bypassed.
_SKIPPED_MODES = frozenset({2, 4})

#: The placeholders the adapter fills with the picture being edited: the first
#: picture, the whole list, or a numbered slot.
_SOURCE_PLACEHOLDER = re.compile(r"\$\{input_image(?:s|_[0-9]{1,2})?\}")


def image_edit_kind(
    operation: str,
    ui_graph: Mapping[str, Any] | None,
    api_graph: Mapping[str, Any] | None,
    input_schema: Mapping[str, Any] | None,
) -> ImageEditKind:
    """Classify one image edit workflow revision from its stored graphs."""

    if operation != Operation.IMAGE_TO_IMAGE.value or not api_graph:
        return "other"
    if not binds_prompt(dict(api_graph)):
        return "other"
    if _source_conditions_the_saved_picture(ui_graph, api_graph):
        return "instruction"
    calibration = safe_workflow_edit_calibration(input_schema)
    parameter = calibration.parameter if calibration else STRENGTH_PARAMETER
    if binds_parameter(api_graph, parameter):
        return "strength"
    return "other"


def _source_conditions_the_saved_picture(
    ui_graph: Mapping[str, Any] | None, api_graph: Mapping[str, Any]
) -> bool:
    """Whether the executable graph conditions its saved picture on the source.

    The path is walked in the executable API graph, because that is what runs: a
    branch that exists only in the UI graph, a node the compiler left out, or a
    result that is never saved takes no part. The walk starts at a node that
    receives the source picture, counts a node when a conditioning output of its
    own feeds that part of the graph, and needs the words to reach a saver too.
    The API graph does not say what each output is, so the UI graph supplies
    output types, and only for a node it describes: the same id, as expansion
    and the compiler both name it, and the same node type.
    """

    output_types = _described_output_types(ui_graph, api_graph)
    if not output_types:
        return False
    nodes = {
        str(node_id): node
        for node_id, node in api_graph.items()
        if isinstance(node, Mapping) and isinstance(node.get("class_type"), str)
    }
    links = _api_links(nodes)
    saved = {
        node_id for node_id, node in nodes.items() if node["class_type"] in SAVED_PICTURE_NODE_TYPES
    }
    used = _leading_to(saved, links)
    if not any(binds_prompt(dict(nodes[node_id])) for node_id in used):
        return False
    targets: dict[str, list[tuple[int, str]]] = {}
    for origin, slot, target in links:
        if origin in used and target in used:
            targets.setdefault(origin, []).append((slot, target))

    pending = [
        target
        for node_id in used
        if _receives_the_source(nodes[node_id])
        for slot, target in targets.get(node_id, [])
        if _output_type(output_types, node_id, slot) == "IMAGE"
    ]
    reached: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in reached:
            continue
        reached.add(node_id)
        outgoing = targets.get(node_id, [])
        if any(_output_type(output_types, node_id, slot) == "CONDITIONING" for slot, _ in outgoing):
            return True
        pending.extend(target for _, target in outgoing)
    return False


def _described_output_types(
    ui_graph: Mapping[str, Any] | None, api_graph: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Output types by API node id, for the nodes the UI graph describes.

    Subgraphs and bypasses are expanded first, exactly as they are before
    compilation, and a graph that cannot be expanded describes nothing. The
    compiler leaves out a node that does not run, so a muted UI node describes
    nothing either.
    """

    if not ui_graph:
        return {}
    try:
        expanded = expand_workflow(ui_graph)
    except ValueError:
        return {}
    described: dict[str, list[str]] = {}
    for node in expanded.get("nodes") or []:
        if not isinstance(node, dict) or node.get("id") is None or not _runs(node):
            continue
        node_id = str(node["id"])
        executed = api_graph.get(node_id)
        if not isinstance(executed, Mapping) or executed.get("class_type") != node.get("type"):
            continue
        outputs = node.get("outputs")
        described[node_id] = [
            str(output.get("type") or "") if isinstance(output, dict) else ""
            for output in (outputs if isinstance(outputs, list) else [])
        ]
    return described


def _api_links(nodes: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, int, str]]:
    """Every API input that takes another node's output, as origin, slot and target."""

    links: list[tuple[str, int, str]] = []
    for target, node in nodes.items():
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for value in inputs.values():
            if (
                isinstance(value, list)
                and len(value) == 2
                and str(value[0]) in nodes
                and type(value[1]) is int
            ):
                links.append((str(value[0]), value[1], target))
    return links


def _receives_the_source(node: Mapping[str, Any]) -> bool:
    """Whether the adapter hands this node the picture being edited."""

    inputs = node.get("inputs")
    return isinstance(inputs, Mapping) and any(
        isinstance(value, str) and _SOURCE_PLACEHOLDER.fullmatch(value) is not None
        for value in inputs.values()
    )


def _runs(node: Mapping[str, Any]) -> bool:
    try:
        return int(node.get("mode") or 0) not in _SKIPPED_MODES
    except (TypeError, ValueError):
        return False


def _leading_to(ends: set[str], links: list[tuple[str, int, str]]) -> set[str]:
    """The given nodes and every node with a path to one of them."""

    origins: dict[str, list[str]] = {}
    for origin, _, target in links:
        origins.setdefault(target, []).append(origin)
    found = set(ends)
    pending = list(ends)
    while pending:
        for origin in origins.get(pending.pop(), []):
            if origin not in found:
                found.add(origin)
                pending.append(origin)
    return found


def _output_type(output_types: Mapping[str, list[str]], node_id: str, slot: int) -> str:
    outputs = output_types.get(node_id, [])
    return outputs[slot] if 0 <= slot < len(outputs) else ""
