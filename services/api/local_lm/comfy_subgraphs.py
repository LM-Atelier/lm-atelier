"""Expand ComfyUI subgraphs and bypassed nodes into a plain graph.

The compiler refuses any workflow containing a subgraph, because a subgraph
is a frontend construct and the runtime has never seen one. That refusal is
correct and is also why two authored workflows this project needs cannot
run: their whole structure is subgraphs and bypasses.

This rewrites them away, before compilation and without a runtime. The
rewrite is decidable from the file alone - the definition, its boundary
slots, the instances, and the links are all present - so node types only
need checking afterwards, against the runtime that will actually execute
them.

Everything here fails closed. A graph that cannot be expanded exactly is
refused, never approximated: an approximation would compile, run, and
produce a picture that is quietly not the one the author drew.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MAX_SUBGRAPH_DEPTH = 8

# The frontend's boundary pseudo-nodes inside a subgraph: links leaving the
# input boundary carry the instance's inputs in, and links entering the
# output boundary carry its outputs back out.
INPUT_BOUNDARY = "-10"
OUTPUT_BOUNDARY = "-20"

# Only mode 4 is a bypass. Mode 2 is muted, and the others are event and
# trigger semantics that mean something entirely different; rewiring them
# as though they were pass-throughs would silently rebuild the graph.
BYPASS_MODE = 4


class SubgraphExpansionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Link:
    """One connection, in whichever of the two shapes the file used."""

    link_id: str
    origin_id: str
    origin_slot: int
    target_id: str
    target_slot: int
    kind: str | None

    def as_list(self) -> list[Any]:
        return [
            self.link_id,
            self.origin_id,
            self.origin_slot,
            self.target_id,
            self.target_slot,
            self.kind,
        ]


def expand_workflow(workflow: Mapping[str, Any]) -> dict[str, Any]:
    """Return the workflow with every subgraph and bypass rewritten away."""
    subgraphs = _subgraph_definitions(workflow)
    nodes = [dict(node) for node in _nodes_of(workflow, "workflow")]
    links = _links_of(workflow, "workflow")

    rewritten = False
    if subgraphs:
        nodes, links = _expand(nodes, links, subgraphs, depth=0, seen=())
        rewritten = True
    nodes, links, bypassed = _resolve_bypasses(nodes, links)
    # Only a rewritten graph gets its slot records rebuilt. A graph nobody
    # touched keeps whatever disagreement it shipped with, so the compiler's
    # own checks on inconsistent or doubled links still have something to
    # catch - repairing them here would hide a malformed file.
    if rewritten or bypassed:
        nodes = _reconcile_slot_metadata(nodes, links)

    expanded = {key: value for key, value in workflow.items() if key != "definitions"}
    remaining = {
        key: value
        for key, value in (workflow.get("definitions") or {}).items()
        if key != "subgraphs"
    }
    if remaining:
        expanded["definitions"] = remaining
    expanded["nodes"] = nodes
    expanded["links"] = [link.as_list() for link in links]
    return expanded


def _expand(
    nodes: list[dict[str, Any]],
    links: list[_Link],
    subgraphs: Mapping[str, Mapping[str, Any]],
    *,
    depth: int,
    seen: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[_Link]]:
    if depth > MAX_SUBGRAPH_DEPTH:
        raise SubgraphExpansionError(
            "subgraph_too_deep", "workflow nests subgraphs deeper than this can expand"
        )

    plain: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    for node in nodes:
        (instances if str(node.get("type")) in subgraphs else plain).append(node)
    if not instances:
        return nodes, links

    for instance in instances:
        identifier = str(instance.get("id"))
        definition_id = str(instance.get("type"))
        if definition_id in seen:
            raise SubgraphExpansionError(
                "recursive_subgraph", "workflow contains a subgraph that contains itself"
            )
        definition = subgraphs[definition_id]
        scope = f"{identifier}:"

        inner_nodes = [dict(node) for node in _nodes_of(definition, "subgraph")]
        inner_links = _links_of(definition, "subgraph")
        # Namespace before anything else, so an inner id can never collide
        # with a host id or with another instance of the same definition.
        for node in inner_nodes:
            node["id"] = f"{scope}{node['id']}"
        inner_links = [
            _Link(
                link_id=f"{scope}{link.link_id}",
                origin_id=link.origin_id
                if link.origin_id == INPUT_BOUNDARY
                else f"{scope}{link.origin_id}",
                origin_slot=link.origin_slot,
                target_id=link.target_id
                if link.target_id == OUTPUT_BOUNDARY
                else f"{scope}{link.target_id}",
                target_slot=link.target_slot,
                kind=link.kind,
            )
            for link in inner_links
        ]

        inner_nodes, inner_links = _expand(
            inner_nodes,
            inner_links,
            subgraphs,
            depth=depth + 1,
            seen=(*seen, definition_id),
        )
        plain.extend(inner_nodes)
        links = _splice(instance, identifier, links, inner_links)

    _refuse_duplicate_ids(plain)
    return plain, links


def _splice(
    instance: Mapping[str, Any],
    identifier: str,
    outer: list[_Link],
    inner: list[_Link],
) -> list[_Link]:
    """Join a subgraph's insides to the graph the instance sat in.

    What fed the instance's input slot now feeds whatever that slot fed
    inside; what the inside produced for an output slot now feeds whatever
    the instance's output fed. The instance itself disappears.
    """
    feeding = {link.target_slot: link for link in outer if link.target_id == identifier}
    consuming = [link for link in outer if link.origin_id == identifier]

    produced: dict[int, _Link] = {}
    for link in inner:
        if link.target_id != OUTPUT_BOUNDARY:
            continue
        if link.target_slot in produced:
            raise SubgraphExpansionError(
                "ambiguous_subgraph_output",
                "a subgraph output is fed by more than one link, so its source is not decidable",
            )
        produced[link.target_slot] = link

    spliced = [
        link for link in outer if link.target_id != identifier and link.origin_id != identifier
    ]
    for link in inner:
        if link.target_id == OUTPUT_BOUNDARY:
            continue
        if link.origin_id != INPUT_BOUNDARY:
            spliced.append(link)
            continue
        source = feeding.get(link.origin_slot)
        if source is None:
            raise SubgraphExpansionError(
                "unconnected_subgraph_input",
                "a subgraph input has nothing feeding it, so the graph is incomplete",
            )
        spliced.append(
            _Link(
                link_id=link.link_id,
                origin_id=source.origin_id,
                origin_slot=source.origin_slot,
                target_id=link.target_id,
                target_slot=link.target_slot,
                kind=link.kind or source.kind,
            )
        )
    for link in consuming:
        inside = produced.get(link.origin_slot)
        if inside is None:
            raise SubgraphExpansionError(
                "unconnected_subgraph_output",
                "a subgraph output is used but nothing inside produces it",
            )
        spliced.append(
            _Link(
                link_id=link.link_id,
                origin_id=inside.origin_id,
                origin_slot=inside.origin_slot,
                target_id=link.target_id,
                target_slot=link.target_slot,
                kind=link.kind or inside.kind,
            )
        )
    return spliced


def _resolve_bypasses(
    nodes: list[dict[str, Any]], links: list[_Link]
) -> tuple[list[dict[str, Any]], list[_Link], bool]:
    """Route around bypassed nodes, or refuse when the route is a guess.

    A bypassed node passes an input straight through to an output of the
    same type. Where the match is not unique the author's intent is not
    recoverable from the file, and picking one would rebuild the graph
    rather than expand it.
    """
    bypassed = {str(node["id"]): node for node in nodes if _mode_of(node) == BYPASS_MODE}
    if not bypassed:
        return nodes, links, False

    for identifier, node in bypassed.items():
        inputs = _slot_types(node.get("inputs"))
        outputs = _slot_types(node.get("outputs"))
        feeding = {link.target_slot: link for link in links if link.target_id == identifier}
        consuming = [link for link in links if link.origin_id == identifier]

        replacements: list[_Link] = []
        for link in consuming:
            produced = outputs.get(link.origin_slot)
            candidates = [
                slot
                for slot, kind in inputs.items()
                if kind is not None and kind == produced and slot in feeding
            ]
            if len(candidates) != 1:
                raise SubgraphExpansionError(
                    "ambiguous_bypass",
                    "a bypassed node has no single input of the type it was passing through",
                )
            source = feeding[candidates[0]]
            replacements.append(
                _Link(
                    link_id=link.link_id,
                    origin_id=source.origin_id,
                    origin_slot=source.origin_slot,
                    target_id=link.target_id,
                    target_slot=link.target_slot,
                    kind=link.kind or source.kind,
                )
            )
        links = [
            link for link in links if link.origin_id != identifier and link.target_id != identifier
        ] + replacements

    return [node for node in nodes if str(node["id"]) not in bypassed], links, True


def _reconcile_slot_metadata(
    nodes: list[dict[str, Any]], links: list[_Link]
) -> list[dict[str, Any]]:
    """Point every slot at the links that actually exist now.

    A node records the link id feeding each input and leaving each output,
    and the compiler cross-checks that record against the link list. Since
    expansion renames links and reroutes them, the record is rebuilt from
    the result rather than maintained alongside it - bookkeeping that has to
    agree with a rewrite is bookkeeping that eventually does not.
    """
    incoming: dict[tuple[str, int], str] = {}
    outgoing: dict[tuple[str, int], list[str]] = {}
    for link in links:
        seat = (link.target_id, link.target_slot)
        if seat in incoming:
            raise SubgraphExpansionError(
                "doubled_input",
                "expansion left two links feeding one input, which cannot be executed",
            )
        incoming[seat] = link.link_id
        outgoing.setdefault((link.origin_id, link.origin_slot), []).append(link.link_id)

    reconciled: list[dict[str, Any]] = []
    for node in nodes:
        updated = dict(node)
        identifier = str(updated["id"])
        inputs = updated.get("inputs")
        if isinstance(inputs, Sequence) and not isinstance(inputs, str | bytes):
            updated["inputs"] = [
                {**slot, "link": incoming.get((identifier, index))}
                if isinstance(slot, Mapping) and "link" in slot
                else slot
                for index, slot in enumerate(inputs)
            ]
        outputs = updated.get("outputs")
        if isinstance(outputs, Sequence) and not isinstance(outputs, str | bytes):
            updated["outputs"] = [
                {**slot, "links": outgoing.get((identifier, index), [])}
                if isinstance(slot, Mapping) and "links" in slot
                else slot
                for index, slot in enumerate(outputs)
            ]
        reconciled.append(updated)
    return reconciled


def _subgraph_definitions(workflow: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    definitions = workflow.get("definitions")
    if not isinstance(definitions, Mapping):
        return {}
    listed = definitions.get("subgraphs")
    if not isinstance(listed, Sequence) or isinstance(listed, str | bytes):
        return {}
    found: dict[str, Mapping[str, Any]] = {}
    for entry in listed:
        if not isinstance(entry, Mapping):
            raise SubgraphExpansionError("invalid_subgraph", "a subgraph is not an object")
        identifier = entry.get("id")
        if identifier is None:
            raise SubgraphExpansionError("invalid_subgraph", "a subgraph has no id")
        found[str(identifier)] = entry
    return found


def _nodes_of(scope: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    nodes = scope.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, str | bytes):
        raise SubgraphExpansionError("invalid_structure", f"{label} nodes must be an array")
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("id") is None:
            raise SubgraphExpansionError("invalid_node", f"{label} has a node without an id")
    return list(nodes)


def _links_of(scope: Mapping[str, Any], label: str) -> list[_Link]:
    listed = scope.get("links")
    if listed is None:
        return []
    if not isinstance(listed, Sequence) or isinstance(listed, str | bytes):
        raise SubgraphExpansionError("invalid_structure", f"{label} links must be an array")
    links: list[_Link] = []
    for value in listed:
        if isinstance(value, Mapping):
            links.append(
                _Link(
                    link_id=str(value.get("id")),
                    origin_id=str(value.get("origin_id")),
                    origin_slot=_slot(value.get("origin_slot")),
                    target_id=str(value.get("target_id")),
                    target_slot=_slot(value.get("target_slot")),
                    kind=_kind(value.get("type")),
                )
            )
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            if len(value) < 5:
                raise SubgraphExpansionError("invalid_link", f"{label} has a malformed link")
            links.append(
                _Link(
                    link_id=str(value[0]),
                    origin_id=str(value[1]),
                    origin_slot=_slot(value[2]),
                    target_id=str(value[3]),
                    target_slot=_slot(value[4]),
                    kind=_kind(value[5]) if len(value) > 5 else None,
                )
            )
        else:
            raise SubgraphExpansionError("invalid_link", f"{label} has an invalid link")
    return links


def _slot(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SubgraphExpansionError("invalid_link", "a link slot is not a whole number")
    return value


def _kind(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _mode_of(node: Mapping[str, Any]) -> int | None:
    mode = node.get("mode")
    return mode if isinstance(mode, int) and not isinstance(mode, bool) else None


def _slot_types(slots: object) -> dict[int, str | None]:
    if not isinstance(slots, Sequence) or isinstance(slots, str | bytes):
        return {}
    return {
        index: (slot.get("type") if isinstance(slot.get("type"), str) else None)
        for index, slot in enumerate(slots)
        if isinstance(slot, Mapping)
    }


def _refuse_duplicate_ids(nodes: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for node in nodes:
        identifier = str(node["id"])
        if identifier in seen:
            raise SubgraphExpansionError(
                "duplicate_node_id", "expansion produced two nodes with the same id"
            )
        seen.add(identifier)
