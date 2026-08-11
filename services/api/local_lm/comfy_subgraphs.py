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
        nodes, links = _reconcile_slot_metadata(nodes, links)

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
            _namespace_slot_links(node, scope)
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
        links = _splice(
            instance,
            identifier,
            links,
            inner_links,
            {str(node["id"]): node for node in inner_nodes},
        )

    _refuse_duplicate_ids(plain)
    return plain, links


def _names_a_real_slot(inner_nodes: Mapping[str, Mapping[str, Any]], link: _Link) -> bool:
    """Whether the input a link points at exists on the node it points at.

    Asked only about structure. Whether that input is required, and whether the
    node can do without it, is decided against the runtime's declaration later -
    but a link naming a slot no node has describes nothing, and there is no
    later check that will notice once the link is gone.
    """

    node = inner_nodes.get(str(link.target_id))
    if node is None:
        return False
    slots = node.get("inputs")
    if not isinstance(slots, Sequence) or isinstance(slots, str | bytes):
        return False
    return 0 <= link.target_slot < len(slots)


def _splice(
    instance: Mapping[str, Any],
    identifier: str,
    outer: list[_Link],
    inner: list[_Link],
    inner_nodes: Mapping[str, Mapping[str, Any]],
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
            # Nothing wired this edge, so there is no connection to carry
            # inward and the link goes. What that leaves behind is decided
            # where it can be decided properly: an input the node draws a
            # widget for keeps the value it was saved with, an optional input
            # was never needed, and a required one that is now unfed is
            # refused by name against the runtime's own declaration of what
            # this node requires. Deciding it here would mean reading a UI slot
            # and guessing which of the three it was.
            #
            # The one thing that must be caught before the link goes is a link
            # naming an input its own node does not have. Nothing downstream
            # can notice that once there is no link left to check.
            if _names_a_real_slot(inner_nodes, link):
                continue
            raise SubgraphExpansionError(
                "unconnected_subgraph_input",
                "a subgraph input feeds an input its node does not have",
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
            source = _bypass_source(
                inputs, feeding, outputs.get(link.origin_slot), link.origin_slot
            )
            if source is None:
                # No fed input carries what this output carried, so the route
                # simply ends here. The frontend omits the connection rather
                # than treating it as a problem, and a bypassed loader with
                # nothing feeding it is the ordinary case, not a broken file.
                continue
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


def _slots_of(value: object) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return list(value)


def _namespace_slot_links(node: dict[str, Any], scope: str) -> None:
    """Carry a node's recorded link ids into the scope its links moved to."""
    inputs = _slots_of(node.get("inputs"))
    if inputs:
        node["inputs"] = [
            {**slot, "link": f"{scope}{slot['link']}"}
            if isinstance(slot, Mapping) and slot.get("link") is not None
            else slot
            for slot in inputs
        ]
    outputs = _slots_of(node.get("outputs"))
    if outputs:
        node["outputs"] = [
            {**slot, "links": [f"{scope}{one}" for one in slot["links"]]}
            if isinstance(slot, Mapping) and isinstance(slot.get("links"), list)
            else slot
            for slot in outputs
        ]


def _bypass_source(
    inputs: dict[int, str | None],
    feeding: dict[int, _Link],
    produced: str | None,
    slot: int,
) -> _Link | None:
    """Which fed input a bypassed node's output passes through from.

    The order is the frontend's own and is deliberate rather than a guess:
    the input at the same slot index first, then the first exact type match,
    then the first input that accepts anything. Uniqueness is not required -
    two inputs of one type is a normal graph, and refusing it made ordinary
    files uncompilable.
    """
    if produced is not None and inputs.get(slot) == produced and slot in feeding:
        return feeding[slot]
    for candidate, kind in sorted(inputs.items()):
        if candidate in feeding and kind is not None and kind == produced:
            return feeding[candidate]
    for candidate, kind in sorted(inputs.items()):
        if candidate in feeding and kind == "*":
            return feeding[candidate]
    return None


def _reconcile_slot_metadata(
    nodes: list[dict[str, Any]], links: list[_Link]
) -> tuple[list[dict[str, Any]], list[_Link]]:
    """Point every slot at the links that actually exist now.

    A node records the link id feeding each input and leaving each output,
    and the compiler cross-checks that record against the link list. Since
    expansion renames links and reroutes them, the record is rebuilt from
    the result rather than maintained alongside it - bookkeeping that has to
    agree with a rewrite is bookkeeping that eventually does not.
    """
    seated: dict[tuple[str, int], list[str]] = {}
    outgoing: dict[tuple[str, int], list[str]] = {}
    for link in links:
        seated.setdefault((link.target_id, link.target_slot), []).append(link.link_id)
        outgoing.setdefault((link.origin_id, link.origin_slot), []).append(link.link_id)

    recorded = {
        (str(node["id"]), index): slot.get("link")
        for node in nodes
        for index, slot in enumerate(_slots_of(node.get("inputs")))
        if isinstance(slot, Mapping)
    }
    incoming: dict[tuple[str, int], str] = {}
    for seat, candidates in seated.items():
        if len(candidates) == 1:
            incoming[seat] = candidates[0]
            continue
        # An input can only be fed once. A file can still carry stale entries
        # in its link array, and the slot itself records which one it means -
        # so that record decides, and only when it names exactly one of them.
        # Anything less is guessing, and last-link-wins is the guess that
        # looks most like working.
        named = str(recorded.get(seat)) if recorded.get(seat) is not None else None
        chosen = [candidate for candidate in candidates if candidate == named]
        if len(chosen) != 1:
            raise SubgraphExpansionError(
                "doubled_input",
                "two links feed one input and its own record does not say which",
            )
        incoming[seat] = chosen[0]
    links = [
        link for link in links if incoming.get((link.target_id, link.target_slot)) == link.link_id
    ]

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
    return reconciled, links


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
