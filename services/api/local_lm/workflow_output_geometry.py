from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Never, cast

from .model_planner import workflow_artifact_contract
from .output_geometry import (
    MAX_DIMENSION,
    MAX_PIXELS,
    PRESET_RATIOS,
    OutputGeometryCapability,
    OutputGeometryError,
    ResolvedOutputGeometry,
    declare_output_geometry,
    output_geometry_capability_payload,
    resolve_output_geometry,
)
from .settings_registry import IMAGE_SETTINGS, workflow_settings

WORKFLOW_OUTPUT_GEOMETRY_VERSION: Literal[1] = 1
WORKFLOW_OUTPUT_GEOMETRY_UNAVAILABLE = "unsupported_workflow_geometry"

_MAX_GRAPH_NODES = 512
_MAX_GRAPH_INPUTS = 128
_MAX_JSON_ITEMS = 20_000
_MAX_JSON_DEPTH = 24
_MAX_TEXT_LENGTH = 100_000
_MAX_IDENTIFIER_LENGTH = 256
_SHA256_CHARS = frozenset("0123456789abcdef")
_LATENT_ROOTS = frozenset({"EmptyLatentImage", "EmptySD3LatentImage"})
_PROOF_TOKEN = object()
_RESOLUTION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class WorkflowGeometryInputBinding:
    key: Literal["width", "height"]
    node_id: str
    input_name: Literal["width", "height"]
    default: int
    minimum: int
    maximum: int
    multiple_of: int


@dataclass(frozen=True, slots=True, init=False)
class WorkflowOutputGeometryProof:
    """Exact, revision-bound evidence; construction is reserved to this verifier."""

    version: Literal[1]
    workflow_id: str
    revision_id: str
    artifact_sha256: str
    operation: Literal["text_to_image"]
    engine: Literal["comfyui"]
    latent_node_id: str
    sampler_node_ids: tuple[str, ...]
    decode_node_ids: tuple[str, ...]
    save_node_ids: tuple[str, ...]
    width: WorkflowGeometryInputBinding
    height: WorkflowGeometryInputBinding
    capability: OutputGeometryCapability
    graph_binding_verified: Literal[True] = field(default=True, init=False)
    request_authorized: Literal[False] = field(default=False, init=False)

    def __new__(cls, token: object) -> WorkflowOutputGeometryProof:
        if token is not _PROOF_TOKEN:
            _refuse()
        return object.__new__(cls)

    def __init__(self, token: object) -> None:
        if token is not _PROOF_TOKEN:
            _refuse()


@dataclass(frozen=True, slots=True)
class WorkflowOutputGeometryResult:
    available: bool
    reason: str | None
    proof: WorkflowOutputGeometryProof | None


@dataclass(frozen=True, slots=True, init=False)
class WorkflowOutputGeometryResolution:
    """What one request resolves to for one revision; preview evidence only.

    Construction is reserved to this module for the same reason the proof's is.
    A value carrying a workflow, revision and artifact bind must never be
    assemblable by the caller who supplied the request, or the bind stops being
    evidence of anything.
    """

    version: Literal[1]
    workflow_id: str
    revision_id: str
    artifact_sha256: str
    operation: Literal["text_to_image"]
    engine: Literal["comfyui"]
    geometry: ResolvedOutputGeometry
    graph_binding_verified: Literal[True] = field(default=True, init=False)
    request_authorized: Literal[False] = field(default=False, init=False)

    def __new__(cls, token: object) -> WorkflowOutputGeometryResolution:
        if token is not _RESOLUTION_TOKEN:
            _refuse()
        return object.__new__(cls)

    def __init__(self, token: object) -> None:
        if token is not _RESOLUTION_TOKEN:
            _refuse()


class WorkflowOutputGeometryError(ValueError):
    pass


def prove_workflow_output_geometry(
    *,
    workflow_id: object,
    revision_id: object,
    operation: object,
    engine: object,
    api_graph: object,
    input_schema: object,
    dependencies: object,
    artifact_sha256: object,
    trusted: object,
) -> WorkflowOutputGeometryResult:
    """Prove exact width/height support for one stored trusted ComfyUI revision.

    Refusal is deliberately non-diagnostic. The graph may be user-authored, and
    capability discovery must not turn its contents into an error oracle.
    """

    try:
        proof = _prove(
            workflow_id=workflow_id,
            revision_id=revision_id,
            operation=operation,
            engine=engine,
            api_graph=api_graph,
            input_schema=input_schema,
            dependencies=dependencies,
            artifact_sha256=artifact_sha256,
            trusted=trusted,
        )
    except (WorkflowOutputGeometryError, TypeError, ValueError, OverflowError):
        return WorkflowOutputGeometryResult(
            available=False,
            reason=WORKFLOW_OUTPUT_GEOMETRY_UNAVAILABLE,
            proof=None,
        )
    return WorkflowOutputGeometryResult(available=True, reason=None, proof=proof)


def workflow_output_geometry_payload(result: WorkflowOutputGeometryResult) -> dict[str, object]:
    if type(result) is not WorkflowOutputGeometryResult:
        _refuse()
    proof = result.proof
    if not result.available or proof is None:
        return {
            "version": WORKFLOW_OUTPUT_GEOMETRY_VERSION,
            "available": False,
            "reason": WORKFLOW_OUTPUT_GEOMETRY_UNAVAILABLE,
            "revision_id": None,
            "workflow_id": None,
            "artifact_sha256": None,
            "operation": None,
            "engine": None,
            "size_modes": [],
            "preset_ids": [],
            "width": None,
            "height": None,
            "latent_node_id": None,
            "sampler_node_ids": [],
            "decode_node_ids": [],
            "save_node_ids": [],
            "capability": None,
            "graph_binding_verified": False,
            "request_authorized": False,
        }
    return {
        "version": proof.version,
        "available": True,
        "reason": None,
        "revision_id": proof.revision_id,
        "workflow_id": proof.workflow_id,
        "artifact_sha256": proof.artifact_sha256,
        "operation": proof.operation,
        "engine": proof.engine,
        # Read from the proven capability rather than restated, so a workflow
        # whose bounds admit no ratio never advertises presets it would then
        # refuse.
        "size_modes": [item.size_mode for item in proof.capability.combinations],
        "preset_ids": list(proof.capability.allowed_preset_ids),
        "width": _binding_payload(proof.width),
        "height": _binding_payload(proof.height),
        "latent_node_id": proof.latent_node_id,
        "sampler_node_ids": list(proof.sampler_node_ids),
        "decode_node_ids": list(proof.decode_node_ids),
        "save_node_ids": list(proof.save_node_ids),
        "capability": output_geometry_capability_payload(proof.capability),
        "graph_binding_verified": True,
        "request_authorized": False,
    }


def resolve_workflow_output_geometry(
    result: WorkflowOutputGeometryResult, request: object
) -> WorkflowOutputGeometryResolution | None:
    """Resolve one caller request against a proof this verifier produced.

    Returns None when the revision has no proof or when the proven capability
    does not admit the request. One answer covers both on purpose: telling them
    apart would let a caller who cannot see a workflow learn whether its graph
    is one this proof supports, and the request is the only thing the caller is
    entitled to reason about.

    Nothing the caller sends becomes capability, binding or authority. The only
    value that crosses into resolution is the capability the proof built from
    the stored revision.
    """

    if type(result) is not WorkflowOutputGeometryResult:
        _refuse()
    proof = result.proof
    if not result.available or type(proof) is not WorkflowOutputGeometryProof:
        return None
    try:
        geometry = resolve_output_geometry(proof.capability, request)
    except OutputGeometryError:
        # Narrow on purpose. output_geometry.py raises through exactly one site,
        # `_refuse`, and every value it touches on the way there is type-guarded
        # first, so this is the only exception a request can produce. Catching
        # more would turn a defect in the resolver into "your request is
        # invalid", which is the hardest kind of wrong answer to find.
        return None

    resolution = WorkflowOutputGeometryResolution(_RESOLUTION_TOKEN)
    object.__setattr__(resolution, "version", WORKFLOW_OUTPUT_GEOMETRY_VERSION)
    object.__setattr__(resolution, "workflow_id", proof.workflow_id)
    object.__setattr__(resolution, "revision_id", proof.revision_id)
    object.__setattr__(resolution, "artifact_sha256", proof.artifact_sha256)
    object.__setattr__(resolution, "operation", proof.operation)
    object.__setattr__(resolution, "engine", proof.engine)
    object.__setattr__(resolution, "geometry", geometry)
    object.__setattr__(resolution, "graph_binding_verified", True)
    object.__setattr__(resolution, "request_authorized", False)
    return resolution


def workflow_output_geometry_resolution_payload(
    resolution: WorkflowOutputGeometryResolution,
) -> dict[str, object]:
    if type(resolution) is not WorkflowOutputGeometryResolution:
        _refuse()
    geometry = resolution.geometry
    return {
        "version": resolution.version,
        "workflow_id": resolution.workflow_id,
        "revision_id": resolution.revision_id,
        "artifact_sha256": resolution.artifact_sha256,
        "operation": resolution.operation,
        "engine": resolution.engine,
        "mode": geometry.mode,
        "size_mode": geometry.size_mode,
        "preset_id": geometry.preset_id,
        "width": geometry.width,
        "height": geometry.height,
        "graph_binding_verified": True,
        "request_authorized": False,
    }


def _prove(**values: object) -> WorkflowOutputGeometryProof:
    workflow_id = _identifier(values["workflow_id"])
    revision_id = _identifier(values["revision_id"])
    if values["operation"] != "text_to_image" or type(values["operation"]) is not str:
        _refuse()
    if values["engine"] != "comfyui" or type(values["engine"]) is not str:
        _refuse()
    if values["trusted"] is not True:
        _refuse()
    artifact_sha256 = _sha256(values["artifact_sha256"])
    api_graph = cast(dict[str, Any], _plain_json(values["api_graph"]))
    input_schema = cast(dict[str, Any], _plain_json(values["input_schema"]))
    dependencies = cast(dict[str, Any], _plain_json(values["dependencies"]))
    if (
        type(api_graph) is not dict
        or type(input_schema) is not dict
        or type(dependencies) is not dict
    ):
        _refuse()
    calculated = workflow_artifact_contract(
        operation="text_to_image",
        engine="comfyui",
        api_graph=api_graph,
        input_schema=input_schema,
        dependencies=dependencies,
    )
    if calculated != artifact_sha256:
        _refuse()

    width, height, capability = _schema_capability(input_schema)
    (
        latent_node_id,
        sampler_node_ids,
        decode_node_ids,
        save_node_ids,
    ) = _graph_binding(api_graph)
    width = WorkflowGeometryInputBinding(
        "width",
        latent_node_id,
        "width",
        width.default,
        width.minimum,
        width.maximum,
        width.multiple_of,
    )
    height = WorkflowGeometryInputBinding(
        "height",
        latent_node_id,
        "height",
        height.default,
        height.minimum,
        height.maximum,
        height.multiple_of,
    )

    proof = WorkflowOutputGeometryProof(_PROOF_TOKEN)
    object.__setattr__(proof, "version", WORKFLOW_OUTPUT_GEOMETRY_VERSION)
    object.__setattr__(proof, "workflow_id", workflow_id)
    object.__setattr__(proof, "revision_id", revision_id)
    object.__setattr__(proof, "artifact_sha256", artifact_sha256)
    object.__setattr__(proof, "operation", "text_to_image")
    object.__setattr__(proof, "engine", "comfyui")
    object.__setattr__(proof, "latent_node_id", latent_node_id)
    object.__setattr__(proof, "sampler_node_ids", sampler_node_ids)
    object.__setattr__(proof, "decode_node_ids", decode_node_ids)
    object.__setattr__(proof, "save_node_ids", save_node_ids)
    object.__setattr__(proof, "width", width)
    object.__setattr__(proof, "height", height)
    object.__setattr__(proof, "capability", capability)
    object.__setattr__(proof, "graph_binding_verified", True)
    object.__setattr__(proof, "request_authorized", False)
    return proof


#: The only keywords this proof understands well enough to advertise a bound
#: from. A dimension schema may carry these and nothing else that asserts.
_UNDERSTOOD_DIMENSION_KEYWORDS: Final = frozenset(
    {"type", "default", "minimum", "maximum", "multipleOf"}
)
#: Keywords that describe rather than constrain, so they cannot narrow what the
#: workflow accepts and are safe to ignore.
_HARMLESS_ANNOTATION_KEYWORDS: Final = frozenset(
    {
        "title",
        "description",
        "$comment",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)
#: The only top-level keywords this proof can account for. Everything else is
#: refused, INCLUDING keywords nobody has thought of yet.
#:
#: This is an allowlist rather than a list of known-dangerous keywords because
#: the property being established is that the advertised range is a subset of
#: what the workflow accepts. A list of forbidden keywords cannot establish
#: that: it holds only for the constraints someone remembered. That is not
#: hypothetical - naming the composition keywords still let whole-object `const`
#: and `enum` through, and the capability went on advertising 64..2048 for a
#: schema that permitted exactly one pair.
_UNDERSTOOD_SCHEMA_KEYWORDS: Final = frozenset(
    {"type", "properties", "required", "additionalProperties"}
)


def _schema_capability(
    schema: dict[str, Any],
) -> tuple[WorkflowGeometryInputBinding, WorkflowGeometryInputBinding, OutputGeometryCapability]:
    if schema.get("type") != "object" or type(schema.get("properties")) is not dict:
        _refuse()
    # Anything at the top level that is not understood can narrow width or height
    # without changing their own bounds - a composition, or a whole-object const
    # or enum - so the advertised range would stop being a subset of what the
    # workflow accepts.
    if set(schema) - _UNDERSTOOD_SCHEMA_KEYWORDS - _HARMLESS_ANNOTATION_KEYWORDS:
        _refuse()
    properties = cast(dict[str, object], schema["properties"])
    for key in ("width", "height"):
        item = properties.get(key)
        if type(item) is not dict:
            _refuse()
        field_schema = cast(dict[str, object], item)
        if not set(field_schema) >= _UNDERSTOOD_DIMENSION_KEYWORDS:
            _refuse()
        if field_schema.get("type") != "integer" or field_schema.get("readOnly") is True:
            _refuse()
        # Anything beyond the keywords this proof reads, other than pure
        # annotation, could tighten the accepted range - exclusiveMaximum and a
        # nested allOf both do - and advertising the looser bound would offer a
        # size the workflow then refuses.
        unevaluated = set(field_schema) - _UNDERSTOOD_DIMENSION_KEYWORDS
        if unevaluated - _HARMLESS_ANNOTATION_KEYWORDS:
            _refuse()
    try:
        fields = workflow_settings(IMAGE_SETTINGS, schema)
    except ValueError:
        _refuse()
    by_key = {field.key: field for field in fields}
    width = _field_binding("width", by_key.get("width"))
    height = _field_binding("height", by_key.get("height"))
    max_pixels = min(MAX_PIXELS, width.maximum * height.maximum)
    if width.default * height.default > max_pixels:
        _refuse()
    bounds: dict[str, object] = {
        "mode": "image",
        "min_width": width.minimum,
        "max_width": width.maximum,
        "min_height": height.minimum,
        "max_height": height.maximum,
        "width_multiple": width.multiple_of,
        "height_multiple": height.multiple_of,
        "max_pixels": max_pixels,
        "min_aspect": [1, MAX_DIMENSION],
        "max_aspect": [MAX_DIMENSION, 1],
        "default_width": width.default,
        "default_height": height.default,
    }
    presets = _preset_dimensions(width, height, max_pixels)
    combinations: list[dict[str, object]] = [
        {
            **bounds,
            "size_mode": "exact",
            "buckets": [[width.default, height.default]],
        }
    ]
    if presets:
        # Every combination must carry the workflow's own default among its
        # buckets, so the preset list is unioned with it rather than replacing
        # it. That costs nothing: a preset always resolves to the pair whose
        # ratio matches it exactly, and no other bucket can tie.
        pairs = sorted({(width.default, height.default)} | set(presets.values()))
        combinations.append(
            {
                **bounds,
                "size_mode": "preset",
                "buckets": [[item[0], item[1]] for item in pairs],
            }
        )
    capability = declare_output_geometry(
        {
            "version": 1,
            "allowed_modes": ["image"],
            "allowed_preset_ids": sorted(presets),
            "combinations": combinations,
        }
    )
    return width, height, capability


def _preset_dimensions(
    width: WorkflowGeometryInputBinding,
    height: WorkflowGeometryInputBinding,
    max_pixels: int,
) -> dict[str, tuple[int, int]]:
    """The exact pixels each ratio preset means for this workflow, if any.

    A preset states a shape, not a pixel count, so the pair chosen for a ratio
    is the legal one whose area sits closest to the size the workflow already
    defaults to: asking for 16:9 changes what the render is, never what it
    costs. A ratio this workflow's own bounds and multiples cannot express
    exactly is not offered at all, because snapping it to a nearby shape would
    hand back an image of a ratio nobody asked for.
    """

    default_area = width.default * height.default
    resolved: dict[str, tuple[int, int]] = {}
    for preset_id, (across, down) in PRESET_RATIOS.items():
        # A reduced ratio across:down admits exactly the pairs across*count by
        # down*count, and the counts that land both sides on their own multiple
        # grids are exactly the multiples of step.
        step = math.lcm(
            width.multiple_of // math.gcd(across, width.multiple_of),
            height.multiple_of // math.gcd(down, height.multiple_of),
        )
        lowest = max(
            -(-width.minimum // (across * step)),
            -(-height.minimum // (down * step)),
            1,
        )
        area_step = across * down * step * step
        highest = min(
            width.maximum // (across * step),
            height.maximum // (down * step),
            math.isqrt(max_pixels // area_step),
        )
        if lowest > highest:
            continue
        # Area grows with the count, so the closest legal area to the default is
        # at one of the two counts around the ideal, clamped into range.
        ideal = math.isqrt(default_area // area_step)
        count = min(
            {min(max(value, lowest), highest) for value in (ideal, ideal + 1)},
            key=lambda value: (
                abs(area_step * value * value - default_area),
                0 if area_step * value * value >= default_area else 1,
                value,
            ),
        )
        resolved[preset_id] = (across * step * count, down * step * count)
    return resolved


def _field_binding(key: Literal["width", "height"], value: object) -> WorkflowGeometryInputBinding:
    if value is None:
        _refuse()
    field_value = cast(Any, value)
    if field_value.type != "integer" or field_value.available is not True:
        _refuse()
    default = _positive_integer(field_value.default)
    minimum = _positive_integer(field_value.minimum)
    maximum = _positive_integer(field_value.maximum)
    multiple = _positive_integer(field_value.multiple_of or field_value.step or 1)
    if not minimum <= default <= maximum or default % multiple or minimum > MAX_DIMENSION:
        _refuse()
    if maximum > MAX_DIMENSION:
        _refuse()
    return WorkflowGeometryInputBinding(key, "", key, default, minimum, maximum, multiple)


def _graph_binding(
    graph: dict[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not graph or len(graph) > _MAX_GRAPH_NODES:
        _refuse()
    nodes: dict[str, tuple[str, dict[str, object]]] = {}
    for raw_id, raw_node in graph.items():
        node_id = _identifier(raw_id)
        if type(raw_node) is not dict:
            _refuse()
        node = cast(dict[str, object], raw_node)
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        if (
            type(class_type) is not str
            or not class_type
            or len(class_type) > _MAX_IDENTIFIER_LENGTH
        ):
            _refuse()
        if type(inputs) is not dict or len(inputs) > _MAX_GRAPH_INPUTS:
            _refuse()
        nodes[node_id] = (class_type, cast(dict[str, object], inputs))

    save_ids = tuple(sorted(node_id for node_id, node in nodes.items() if node[0] == "SaveImage"))
    if not save_ids or len(save_ids) > 32:
        _refuse()
    latent_ids: set[str] = set()
    sampler_ids: set[str] = set()
    decode_ids: set[str] = set()
    for save_id in save_ids:
        save = nodes[save_id]
        decode_id = _required_source(nodes, save[1].get("images"), "VAEDecode", 0)
        decode_ids.add(decode_id)
        decode = nodes[decode_id]
        sampler_id = _required_source(nodes, decode[1].get("samples"), "KSampler", 0)
        sampler_ids.add(sampler_id)
        sampler = nodes[sampler_id]
        latent_id = _required_source(nodes, sampler[1].get("latent_image"), _LATENT_ROOTS, 0)
        latent_ids.add(latent_id)
    if len(latent_ids) != 1:
        _refuse()
    latent_id = next(iter(latent_ids))
    latent = nodes[latent_id]
    if latent[1].get("width") != "${width}" or latent[1].get("height") != "${height}":
        _refuse()
    return latent_id, tuple(sorted(sampler_ids)), tuple(sorted(decode_ids)), save_ids


def _required_source(
    nodes: dict[str, tuple[str, dict[str, object]]],
    value: object,
    expected_class: str | frozenset[str],
    expected_output: int,
) -> str:
    if type(value) is not list or len(value) != 2:
        _refuse()
    source_id = _identifier(value[0])
    output = value[1]
    if type(output) is not int or output != expected_output or source_id not in nodes:
        _refuse()
    class_type = nodes[source_id][0]
    allowed = expected_class if type(expected_class) is frozenset else frozenset({expected_class})
    if class_type not in allowed:
        _refuse()
    return source_id


def _binding_payload(binding: WorkflowGeometryInputBinding) -> dict[str, object]:
    return {
        "key": binding.key,
        "node_id": binding.node_id,
        "input_name": binding.input_name,
        "default": binding.default,
        "minimum": binding.minimum,
        "maximum": binding.maximum,
        "multiple_of": binding.multiple_of,
    }


def _plain_json(value: object) -> object:
    items = 0

    def clone(candidate: object, depth: int) -> object:
        nonlocal items
        items += 1
        if depth > _MAX_JSON_DEPTH or items > _MAX_JSON_ITEMS:
            _refuse()
        if candidate is None or type(candidate) in (bool, int):
            return candidate
        if type(candidate) is float:
            if not math.isfinite(candidate):
                _refuse()
            return candidate
        if type(candidate) is str:
            if len(candidate) > _MAX_TEXT_LENGTH:
                _refuse()
            return candidate
        if type(candidate) is list:
            return [clone(item, depth + 1) for item in cast(list[object], candidate)]
        if type(candidate) is dict:
            mapping = cast(dict[object, object], candidate)
            if any(type(key) is not str or len(key) > _MAX_TEXT_LENGTH for key in mapping):
                _refuse()
            return {cast(str, key): clone(item, depth + 1) for key, item in mapping.items()}
        _refuse()

    return clone(value, 0)


def _identifier(value: object) -> str:
    if type(value) is not str or not value or len(value) > _MAX_IDENTIFIER_LENGTH:
        _refuse()
    return value


def _sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in _SHA256_CHARS for char in value)
    ):
        _refuse()
    return value


def _positive_integer(value: object) -> int:
    if type(value) not in (int, float):
        _refuse()
    number = cast(int | float, value)
    if (
        not math.isfinite(float(number))
        or number != int(number)
        or not 1 <= int(number) <= MAX_DIMENSION
    ):
        _refuse()
    return int(number)


def _refuse() -> Never:
    raise WorkflowOutputGeometryError
