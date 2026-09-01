"""Pure MiniMax H3 workflow node observation.

Records exact core node type names from an already-parsed list. This
module does not authorize install, runtime, execution, geometry, or
reference claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-h3-node-observation-v1"
SCHEMA_VERSION: Final = 1
INVALID_OBSERVATION: Final = "h3 node observation is invalid"
MAX_NODE_COUNT: Final = 256
MAX_NODE_TYPE_CHARS: Final = 40
MAX_NODE_TYPE_BYTES: Final = 4096
FAMILY_ID: Final = "minimax-h3"
DECLARATION_SOURCE: Final = "exact_workflow_node_types"
VariantHint = Literal["image_to_video", "reference_to_video"]
EXACT_H3_NODE_TYPES: Final = frozenset(
    {
        "EmptyMiniMaxH3LatentAV",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3SigmaShift",
    }
)
_IMAGE_TO_VIDEO: Final = "MiniMaxH3ImageToVideo"
_REFERENCE_TO_VIDEO: Final = "MiniMaxH3ReferenceToVideo"
_OBSERVATION_WITNESS = object()


class H3NodeObservationError(ValueError):
    """Fixed non-echoing refusal for invalid H3 node observation facts."""


@dataclass(frozen=True, slots=True)
class H3NodeObservationV1:
    schema: Literal["lm-atelier-h3-node-observation-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    evidence_node_types: tuple[str, ...] = field(init=False)
    variant_hints: tuple[VariantHint, ...] = field(init=False)
    family_id: Literal["minimax-h3"] | None = field(init=False)
    declaration_source: Literal["exact_workflow_node_types"] | None = field(init=False)
    runtime_verified: Literal[False] = field(init=False)
    installation_authorized: Literal[False] = field(init=False)
    execution_authorized: Literal[False] = field(init=False)
    workflow_contract_bound: Literal[False] = field(init=False)
    activation_probe_required: Literal[False] = field(init=False)
    geometry_claimed: Literal[False] = field(init=False)
    reference_slots_claimed: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise H3NodeObservationError(INVALID_OBSERVATION)


def _invalid() -> NoReturn:
    raise H3NodeObservationError(INVALID_OBSERVATION)


def _require_node_type(value: object) -> str:
    if type(value) is not str:
        _invalid()
    if not value or len(value) > MAX_NODE_TYPE_CHARS:
        _invalid()
    if value.strip() != value or any(ch.isspace() for ch in value):
        _invalid()
    if any(ord(ch) < 32 or ord(ch) == 127 or 0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        _invalid()
    return value


def _utf8_size(value: str) -> int:
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        _invalid()
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        _invalid()


def _require_node_types(raw_types: object) -> tuple[str, ...]:
    if type(raw_types) is not list and type(raw_types) is not tuple:
        _invalid()
    if len(raw_types) > MAX_NODE_COUNT:
        _invalid()
    total = 0
    for item in raw_types:
        if type(item) is not str:
            _invalid()
        total += _utf8_size(item)
        if total > MAX_NODE_TYPE_BYTES:
            _invalid()
    collected: list[str] = []
    seen: set[str] = set()
    for item in raw_types:
        token = _require_node_type(item)
        if token not in EXACT_H3_NODE_TYPES:
            continue
        if token in seen:
            continue
        seen.add(token)
        collected.append(token)
    return tuple(sorted(collected))


def _observation_from_evaluator(
    *,
    witness: object,
    evidence_node_types: tuple[str, ...],
    variant_hints: tuple[VariantHint, ...],
    family_id: Literal["minimax-h3"] | None,
    declaration_source: Literal["exact_workflow_node_types"] | None,
) -> H3NodeObservationV1:
    if witness is not _OBSERVATION_WITNESS:
        _invalid()
    observation = object.__new__(H3NodeObservationV1)
    object.__setattr__(observation, "schema", SCHEMA_ID)
    object.__setattr__(observation, "schema_version", SCHEMA_VERSION)
    object.__setattr__(observation, "evidence_node_types", evidence_node_types)
    object.__setattr__(observation, "variant_hints", variant_hints)
    object.__setattr__(observation, "family_id", family_id)
    object.__setattr__(observation, "declaration_source", declaration_source)
    object.__setattr__(observation, "runtime_verified", False)
    object.__setattr__(observation, "installation_authorized", False)
    object.__setattr__(observation, "execution_authorized", False)
    object.__setattr__(observation, "workflow_contract_bound", False)
    object.__setattr__(observation, "activation_probe_required", False)
    object.__setattr__(observation, "geometry_claimed", False)
    object.__setattr__(observation, "reference_slots_claimed", False)
    return observation


def observe_h3_node_types(node_types: object) -> H3NodeObservationV1:
    """Record exact H3 node names without granting product authority."""
    evidence = _require_node_types(node_types)
    hints: list[VariantHint] = []
    if _IMAGE_TO_VIDEO in evidence:
        hints.append("image_to_video")
    if _REFERENCE_TO_VIDEO in evidence:
        hints.append("reference_to_video")
    if evidence:
        family: Literal["minimax-h3"] | None = FAMILY_ID
        source: Literal["exact_workflow_node_types"] | None = DECLARATION_SOURCE
    else:
        family = None
        source = None
    return _observation_from_evaluator(
        witness=_OBSERVATION_WITNESS,
        evidence_node_types=evidence,
        variant_hints=tuple(hints),
        family_id=family,
        declaration_source=source,
    )
