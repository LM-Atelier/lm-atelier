"""Which picture each image slot of a compiled workflow takes, as far as can be shown.

A workflow that takes several pictures gets one numbered slot per LoadImage
node, in graph order, and nothing in the slot says what the picture is for: the
one being edited, or one only read for its subject or style. This records that
role on each image property of the compiled input schema, versioned and bound
to the node the slot fills, so a turn can put each picture where it belongs
instead of by position.

A role comes from one of two places, and never from a model name:

* The graph's structure, followed from each LoadImage's IMAGE output only and
  onward only along picture, latent and conditioning outputs. A picture that
  reaches a sampler's ``latent_image`` input as a latent is the canvas being
  edited. When exactly one picture does, a picture that reaches only
  conditioning is read alongside it. Anything else stays unknown: a graph
  whose pictures all reach only conditioning says nothing about which one is
  edited, however its first one is wired, and a picture that only sets a size
  or a mask on the way says nothing about being the canvas.
* A declaration authored with a registered template, for a graph we know. It
  applies only when it names exactly as many slots as the graph has.

Every bound slot is required: a LoadImage cannot run without a file, and this
contract never says one picture may stand in for another.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

SLOT_KEY = "x-lm-atelier-image-slot"
CONTRACT_VERSION = 1

SlotRole = Literal["edit_source", "reference", "unknown"]
SlotBasis = Literal["graph", "declared", "none"]

_ROLES: frozenset[str] = frozenset(get_args(SlotRole))
_BASES: frozenset[str] = frozenset(get_args(SlotBasis))
_SOURCE_NODE_TYPE = "LoadImage"
# The core LoadImage returns (IMAGE, MASK); only the picture itself is followed.
_IMAGE_OUTPUT = 0
_LATENT_INPUT = "latent_image"
# What a picture can travel as on its way to a sampler; a size, a mask or any
# other value computed from it is not the picture.
_FOLLOWED = frozenset({"IMAGE", "LATENT", "CONDITIONING"})
_PROPERTY = re.compile(r"input_image(?:_(?P<index>0|[1-9][0-9]?))?\Z")
# The numbered slots the ComfyUI adapter fills.
_NUMBERED_SLOTS = 64
_PLACEHOLDER_PREFIX = "${input_image"

Link = tuple[str, int, str, int]


@dataclass(frozen=True)
class ImageSlot:
    """What one image property of a compiled workflow takes."""

    name: str
    node: str
    role: SlotRole
    basis: SlotBasis
    required: bool = True

    def as_schema(self) -> dict[str, object]:
        return {
            "version": CONTRACT_VERSION,
            "node": self.node,
            "role": self.role,
            "basis": self.basis,
            "required": self.required,
        }


@dataclass(frozen=True)
class _Reach:
    latent: bool = False
    conditioning: bool = False


def _output_type(node: Mapping[str, Any] | None, slot: int) -> str:
    outputs = node.get("outputs") if node else None
    if (
        isinstance(outputs, list)
        and 0 <= slot < len(outputs)
        and isinstance(outputs[slot], Mapping)
    ):
        return str(outputs[slot].get("type") or "")
    return ""


def _slot_property(name: object) -> bool:
    """Whether a name is a slot's own: input_image, or input_image_N written plainly below 64."""

    match = _PROPERTY.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        return False
    index = match.group("index")
    return index is None or int(index) < _NUMBERED_SLOTS


def _input_names(node: Mapping[str, Any] | None) -> list[str]:
    inputs = node.get("inputs") if node else None
    if not isinstance(inputs, list):
        return []
    return [str(item.get("name") or "") if isinstance(item, Mapping) else "" for item in inputs]


def _outgoing(links: Sequence[Link]) -> dict[str, list[Link]]:
    outgoing: dict[str, list[Link]] = {}
    for link in links:
        outgoing.setdefault(link[0], []).append(link)
    return outgoing


def _executes(
    nodes: Mapping[str, Mapping[str, Any]],
    outgoing: Mapping[str, list[Link]],
    sampler: str,
    output_types: frozenset[str],
) -> bool:
    """Whether a sampler's result reaches an output node, one that ComfyUI itself runs.

    A sampler whose result reaches no output never runs as part of the result,
    so what it was given proves nothing about the pictures that do. Which
    nodes are outputs is ComfyUI's own word for them; without it, none is.
    """

    pending, seen = [sampler], {sampler}
    while pending:
        current = pending.pop()
        if current != sampler and str((nodes.get(current) or {}).get("type") or "") in output_types:
            return True
        for _, _, target, _ in outgoing.get(current, []):
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return False


def _reach(
    nodes: Mapping[str, Mapping[str, Any]],
    links: Sequence[Link],
    source: str,
    output_types: frozenset[str],
) -> _Reach:
    """Where a LoadImage's picture goes, as far as the first sampler it enters.

    A sampler is any node with a ``latent_image`` input, and only one whose
    result reaches an output counts. Entering it there as a latent is being
    its canvas; entering it anywhere else as conditioning is conditioning it.
    Between the picture and the sampler, only picture, latent and conditioning
    outputs are followed, so a size or a mask computed from the picture does
    not carry it anywhere. A conditioning output that never reaches such a
    sampler counts for nothing. The walk stops at the sampler, so a picture
    that only conditions a first pass is not counted as the canvas of a second
    pass that starts from the first one's result.
    """

    outgoing = _outgoing(links)
    latent = conditioning = False
    if _output_type(nodes.get(source), _IMAGE_OUTPUT) != "IMAGE":
        return _Reach()
    pending = [link for link in outgoing.get(source, []) if link[1] == _IMAGE_OUTPUT]
    seen: set[str] = set()
    while pending:
        origin, origin_slot, target, target_slot = pending.pop()
        carried = _output_type(nodes.get(origin), origin_slot)
        names = _input_names(nodes.get(target))
        if _LATENT_INPUT in names:
            if _executes(nodes, outgoing, target, output_types):
                entered = names[target_slot] if 0 <= target_slot < len(names) else ""
                if entered == _LATENT_INPUT and carried == "LATENT":
                    latent = True
                elif entered != _LATENT_INPUT and carried == "CONDITIONING":
                    conditioning = True
            continue
        if target in seen:
            continue
        seen.add(target)
        pending.extend(
            link
            for link in outgoing.get(target, [])
            if _output_type(nodes.get(target), link[1]) in _FOLLOWED
        )
    return _Reach(latent=latent, conditioning=conditioning)


def graph_roles(
    nodes: Mapping[str, Mapping[str, Any]],
    links: Sequence[Link],
    sources: Sequence[str],
    output_types: frozenset[str],
) -> list[SlotRole]:
    """The role each source shows by structure alone, in the order given.

    ``output_types`` are the node types ComfyUI runs as outputs, from its own
    node information.
    """

    reaches = [_reach(nodes, links, source, output_types) for source in sources]
    canvases = [index for index, reach in enumerate(reaches) if reach.latent]
    if len(canvases) != 1:
        return ["unknown"] * len(sources)
    roles: list[SlotRole] = []
    for index, reach in enumerate(reaches):
        if index == canvases[0]:
            roles.append("edit_source")
        elif reach.conditioning and not reach.latent:
            roles.append("reference")
        else:
            roles.append("unknown")
    return roles


def image_slots(
    nodes: Mapping[str, Mapping[str, Any]],
    links: Sequence[Link],
    bindings: Sequence[tuple[str, str]],
    declared: Sequence[str] = (),
    output_types: frozenset[str] = frozenset(),
) -> list[ImageSlot]:
    """One slot record per bound LoadImage, given as (node id, property name) in slot order.

    An authored declaration names every slot's role in the same order and wins
    over the graph; one that names a different number of slots is for some
    other graph, and then no slot has a known role.
    """

    sources = [node for node, _ in bindings]
    if declared:
        if len(declared) != len(bindings) or not all(role in _ROLES for role in declared):
            return [ImageSlot(name, node, "unknown", "none") for node, name in bindings]
        return [
            ImageSlot(name, node, cast(SlotRole, role), "declared" if role != "unknown" else "none")
            for (node, name), role in zip(bindings, declared, strict=True)
        ]
    return [
        ImageSlot(name, node, role, "graph" if role != "unknown" else "none")
        for (node, name), role in zip(
            bindings, graph_roles(nodes, links, sources, output_types), strict=True
        )
    ]


def _bound_load_images(api_graph: Mapping[str, Any]) -> dict[str, str] | None:
    """Each image property a compiled graph's LoadImage nodes are bound to, by node id.

    None when a binding is ambiguous: two nodes on one property, which a record
    naming one node cannot speak for, or a node on a picture placeholder that
    is not a slot's own name.
    """

    bound: dict[str, str] = {}
    for node_id, node in api_graph.items():
        if not isinstance(node, Mapping) or node.get("class_type") != _SOURCE_NODE_TYPE:
            continue
        inputs = node.get("inputs")
        value = inputs.get("image") if isinstance(inputs, Mapping) else None
        if isinstance(value, str) and value.startswith(_PLACEHOLDER_PREFIX) and value.endswith("}"):
            name = value[2:-1]
            if not _slot_property(name) or name in bound.values():
                return None
            bound[str(node_id)] = name
    return bound


def verify_image_slots(
    input_schema: Mapping[str, Any], api_graph: Mapping[str, Any]
) -> dict[str, ImageSlot]:
    """The slot records of a compiled workflow, or nothing when any of them cannot be trusted.

    Every LoadImage bound to an image property must be the only one bound to
    it and carry a well-formed version 1 record naming that very node, and
    every record must name a node bound to its own property, which only a
    slot's own name can be. One record or binding that fails makes every slot
    unknown, because a role read from a partly matching set could put a
    picture in the wrong slot. Nothing malformed raises: it is only untrusted.
    """

    properties = input_schema.get("properties")
    if not isinstance(properties, Mapping):
        return {}
    bound = _bound_load_images(api_graph)
    if bound is None:
        return {}
    slots: dict[str, ImageSlot] = {}
    for name, declaration in properties.items():
        if not isinstance(declaration, Mapping):
            continue
        record = declaration.get(SLOT_KEY)
        if record is None:
            continue
        slot = _parsed(name, record) if isinstance(name, str) else None
        if slot is None or bound.get(slot.node) != name:
            return {}
        slots[name] = slot
    if set(slots) != set(bound.values()):
        return {}
    return slots


def _parsed(name: str, record: object) -> ImageSlot | None:
    if not isinstance(record, Mapping) or set(record) != {
        "version",
        "node",
        "role",
        "basis",
        "required",
    }:
        return None
    version, node, role, basis, required = (
        record["version"],
        record["node"],
        record["role"],
        record["basis"],
        record["required"],
    )
    if (
        type(version) is not int
        or version != CONTRACT_VERSION
        or not isinstance(node, str)
        or not node
        or not isinstance(role, str)
        or role not in _ROLES
        or not isinstance(basis, str)
        or basis not in _BASES
        or required is not True
        or (basis == "none") != (role == "unknown")
    ):
        return None
    return ImageSlot(name, node, cast(SlotRole, role), cast(SlotBasis, basis))
