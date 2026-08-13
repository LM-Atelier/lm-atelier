from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, Never, cast

OUTPUT_GEOMETRY_VERSION: Literal[1] = 1
MAX_GEOMETRY_COMBINATIONS = 6
MAX_GEOMETRY_BUCKETS = 128
MAX_DIMENSION = 1_000_000
MAX_PIXELS = 1_000_000_000

OutputMode = Literal["image", "video"]
SizeMode = Literal["exact", "preset", "workflow_native"]
PresetId = Literal["1:1", "3:4", "2:3", "9:16", "4:3", "3:2", "16:9"]

_MODES = frozenset({"image", "video"})
_SIZE_MODES = frozenset({"exact", "preset", "workflow_native"})
_PRESET_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "3:4": (3, 4),
    "2:3": (2, 3),
    "9:16": (9, 16),
    "4:3": (4, 3),
    "3:2": (3, 2),
    "16:9": (16, 9),
}
_COMBINATION_KEYS = {
    "mode",
    "size_mode",
    "min_width",
    "max_width",
    "min_height",
    "max_height",
    "width_multiple",
    "height_multiple",
    "max_pixels",
    "min_aspect",
    "max_aspect",
    "default_width",
    "default_height",
    "buckets",
}
_ERROR_TEXT = "Output geometry declaration or request is invalid"
_CAPABILITY_TOKEN = object()


class OutputGeometryError(ValueError):
    def __init__(self) -> None:
        super().__init__(_ERROR_TEXT)


@dataclass(frozen=True, slots=True)
class GeometryBucket:
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class GeometryCombination:
    mode: OutputMode
    size_mode: SizeMode
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    width_multiple: int
    height_multiple: int
    max_pixels: int
    min_aspect: tuple[int, int]
    max_aspect: tuple[int, int]
    default_width: int
    default_height: int
    buckets: tuple[GeometryBucket, ...]


@dataclass(frozen=True, slots=True, init=False)
class OutputGeometryCapability:
    version: Literal[1]
    allowed_modes: tuple[OutputMode, ...]
    allowed_preset_ids: tuple[PresetId, ...]
    combinations: tuple[GeometryCombination, ...]
    repository_verified: Literal[False] = field(default=False, init=False)
    workflow_verified: Literal[False] = field(default=False, init=False)
    graph_binding_verified: Literal[False] = field(default=False, init=False)
    capability_verified: Literal[False] = field(default=False, init=False)
    request_authorized: Literal[False] = field(default=False, init=False)

    def __new__(cls, _internal_token: object) -> OutputGeometryCapability:
        """Prevent callers from constructing incomplete authority-like values."""

        if _internal_token is not _CAPABILITY_TOKEN:
            _refuse()
        return object.__new__(cls)

    def __init__(self, _internal_token: object) -> None:
        if _internal_token is not _CAPABILITY_TOKEN:
            _refuse()


@dataclass(frozen=True, slots=True)
class ResolvedOutputGeometry:
    mode: OutputMode
    size_mode: SizeMode
    width: int
    height: int
    preset_id: PresetId | None
    source_width: int | None
    source_height: int | None
    source_fit: Literal["workflow_native"] | None
    source_fit_applied: Literal[False] = field(default=False, init=False)
    source_dimensions_verified: Literal[False] = field(default=False, init=False)
    repository_verified: Literal[False] = field(default=False, init=False)
    workflow_verified: Literal[False] = field(default=False, init=False)
    graph_binding_verified: Literal[False] = field(default=False, init=False)
    capability_verified: Literal[False] = field(default=False, init=False)
    request_authorized: Literal[False] = field(default=False, init=False)


def declare_output_geometry(value: object) -> OutputGeometryCapability:
    """Validate and detach a caller declaration without proving support."""

    root = _mapping(value, {"version", "allowed_modes", "allowed_preset_ids", "combinations"})
    if type(root["version"]) is not int or root["version"] != OUTPUT_GEOMETRY_VERSION:
        _refuse()
    allowed_modes = _canonical_closed_list(root["allowed_modes"], _MODES)
    if not allowed_modes:
        _refuse()
    allowed_presets = _canonical_closed_list(root["allowed_preset_ids"], _PRESET_RATIOS)
    raw = root["combinations"]
    if type(raw) is not list or not raw or len(raw) > MAX_GEOMETRY_COMBINATIONS:
        _refuse()
    combinations = tuple(_combination(item) for item in cast(list[object], raw))
    if combinations != tuple(sorted(combinations, key=lambda item: (item.mode, item.size_mode))):
        _refuse()
    pairs = {(item.mode, item.size_mode) for item in combinations}
    if len(pairs) != len(combinations) or {item.mode for item in combinations} != set(
        allowed_modes
    ):
        _refuse()
    if any(item.size_mode == "preset" for item in combinations) != bool(allowed_presets):
        _refuse()
    return _owned_capability(
        cast(tuple[OutputMode, ...], allowed_modes),
        cast(tuple[PresetId, ...], allowed_presets),
        combinations,
    )


def resolve_output_geometry(
    capability: OutputGeometryCapability, request: object
) -> ResolvedOutputGeometry:
    """Resolve one declaration-bound request; never authorize generation."""

    if type(capability) is not OutputGeometryCapability:
        _refuse()
    base = _mapping_at_least(request, {"mode", "size_mode"})
    mode = cast(OutputMode, _closed_string(base["mode"], _MODES))
    size_mode = cast(SizeMode, _closed_string(base["size_mode"], _SIZE_MODES))
    combination = next(
        (
            item
            for item in capability.combinations
            if item.mode == mode and item.size_mode == size_mode
        ),
        None,
    )
    if mode not in capability.allowed_modes or combination is None:
        _refuse()

    if size_mode == "exact":
        exact = _mapping(request, {"mode", "size_mode", "width", "height"})
        width = _integer(exact["width"], 1, MAX_DIMENSION)
        height = _integer(exact["height"], 1, MAX_DIMENSION)
        _validate_dimensions(combination, width, height)
        return ResolvedOutputGeometry(mode, size_mode, width, height, None, None, None, None)
    if size_mode == "preset":
        preset = _mapping(request, {"mode", "size_mode", "preset_id"})
        preset_id = cast(PresetId, _closed_string(preset["preset_id"], frozenset(_PRESET_RATIOS)))
        if preset_id not in capability.allowed_preset_ids:
            _refuse()
        bucket = _snap_preset(combination, preset_id)
        return ResolvedOutputGeometry(
            mode, size_mode, bucket.width, bucket.height, preset_id, None, None, None
        )

    keys = set(cast(dict[object, object], request)) if type(request) is dict else set()
    plain = {"mode", "size_mode"}
    with_source = plain | {"source_width", "source_height"}
    if keys not in (plain, with_source):
        _refuse()
    source_width: int | None = None
    source_height: int | None = None
    if keys == with_source:
        native = cast(dict[str, object], request)
        source_width = _integer(native["source_width"], 1, MAX_DIMENSION)
        source_height = _integer(native["source_height"], 1, MAX_DIMENSION)
        if source_width * source_height > MAX_PIXELS:
            _refuse()
    return ResolvedOutputGeometry(
        mode,
        size_mode,
        combination.default_width,
        combination.default_height,
        None,
        source_width,
        source_height,
        "workflow_native",
    )


def output_geometry_capability_payload(capability: OutputGeometryCapability) -> dict[str, object]:
    if type(capability) is not OutputGeometryCapability:
        _refuse()
    return {
        "version": capability.version,
        "allowed_modes": list(capability.allowed_modes),
        "allowed_preset_ids": list(capability.allowed_preset_ids),
        "combinations": [
            {
                "mode": item.mode,
                "size_mode": item.size_mode,
                "min_width": item.min_width,
                "max_width": item.max_width,
                "min_height": item.min_height,
                "max_height": item.max_height,
                "width_multiple": item.width_multiple,
                "height_multiple": item.height_multiple,
                "max_pixels": item.max_pixels,
                "min_aspect": list(item.min_aspect),
                "max_aspect": list(item.max_aspect),
                "default_width": item.default_width,
                "default_height": item.default_height,
                "buckets": [[bucket.width, bucket.height] for bucket in item.buckets],
            }
            for item in capability.combinations
        ],
    }


def _combination(value: object) -> GeometryCombination:
    source = _mapping(value, _COMBINATION_KEYS)
    mode = cast(OutputMode, _closed_string(source["mode"], _MODES))
    size_mode = cast(SizeMode, _closed_string(source["size_mode"], _SIZE_MODES))
    min_width = _integer(source["min_width"], 1, MAX_DIMENSION)
    max_width = _integer(source["max_width"], min_width, MAX_DIMENSION)
    min_height = _integer(source["min_height"], 1, MAX_DIMENSION)
    max_height = _integer(source["max_height"], min_height, MAX_DIMENSION)
    width_multiple = _integer(source["width_multiple"], 1, MAX_DIMENSION)
    height_multiple = _integer(source["height_multiple"], 1, MAX_DIMENSION)
    max_pixels = _integer(source["max_pixels"], 1, MAX_PIXELS)
    min_aspect = _ratio(source["min_aspect"])
    max_aspect = _ratio(source["max_aspect"])
    if Fraction(*min_aspect) > Fraction(*max_aspect):
        _refuse()
    default_width = _integer(source["default_width"], 1, MAX_DIMENSION)
    default_height = _integer(source["default_height"], 1, MAX_DIMENSION)
    raw = source["buckets"]
    if type(raw) is not list or not raw or len(raw) > MAX_GEOMETRY_BUCKETS:
        _refuse()
    buckets = tuple(_bucket(item) for item in cast(list[object], raw))
    if buckets != tuple(sorted(buckets, key=lambda item: (item.width, item.height))):
        _refuse()
    if len({(item.width, item.height) for item in buckets}) != len(buckets):
        _refuse()
    combination = GeometryCombination(
        mode,
        size_mode,
        min_width,
        max_width,
        min_height,
        max_height,
        width_multiple,
        height_multiple,
        max_pixels,
        min_aspect,
        max_aspect,
        default_width,
        default_height,
        buckets,
    )
    for bucket in buckets:
        _validate_dimensions(combination, bucket.width, bucket.height)
    if (default_width, default_height) not in {(item.width, item.height) for item in buckets}:
        _refuse()
    _validate_dimensions(combination, default_width, default_height)
    return combination


def _validate_dimensions(combination: GeometryCombination, width: int, height: int) -> None:
    if not (
        combination.min_width <= width <= combination.max_width
        and combination.min_height <= height <= combination.max_height
        and width % combination.width_multiple == 0
        and height % combination.height_multiple == 0
        and width * height <= combination.max_pixels
        and Fraction(*combination.min_aspect) <= Fraction(width, height)
        and Fraction(width, height) <= Fraction(*combination.max_aspect)
    ):
        _refuse()


def _snap_preset(combination: GeometryCombination, preset_id: PresetId) -> GeometryBucket:
    numerator, denominator = _PRESET_RATIOS[preset_id]
    target = Fraction(numerator, denominator)
    default_area = combination.default_width * combination.default_height

    def key(bucket: GeometryBucket) -> tuple[Fraction, int, int, int, int, int]:
        aspect = Fraction(bucket.width, bucket.height)
        area = bucket.width * bucket.height
        return (
            abs(aspect - target),
            0 if aspect >= target else 1,
            abs(area - default_area),
            0 if area >= default_area else 1,
            bucket.width,
            bucket.height,
        )

    return min(combination.buckets, key=key)


def _bucket(value: object) -> GeometryBucket:
    if type(value) is not list or len(value) != 2:
        _refuse()
    pair = cast(list[object], value)
    return GeometryBucket(_integer(pair[0], 1, MAX_DIMENSION), _integer(pair[1], 1, MAX_DIMENSION))


def _ratio(value: object) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        _refuse()
    pair = cast(list[object], value)
    numerator = _integer(pair[0], 1, MAX_DIMENSION)
    denominator = _integer(pair[1], 1, MAX_DIMENSION)
    if math.gcd(numerator, denominator) != 1:
        _refuse()
    return numerator, denominator


def _owned_capability(
    allowed_modes: tuple[OutputMode, ...],
    allowed_preset_ids: tuple[PresetId, ...],
    combinations: tuple[GeometryCombination, ...],
) -> OutputGeometryCapability:
    capability = OutputGeometryCapability(_CAPABILITY_TOKEN)
    object.__setattr__(capability, "version", OUTPUT_GEOMETRY_VERSION)
    object.__setattr__(capability, "allowed_modes", allowed_modes)
    object.__setattr__(capability, "allowed_preset_ids", allowed_preset_ids)
    object.__setattr__(capability, "combinations", combinations)
    object.__setattr__(capability, "repository_verified", False)
    object.__setattr__(capability, "workflow_verified", False)
    object.__setattr__(capability, "graph_binding_verified", False)
    object.__setattr__(capability, "capability_verified", False)
    object.__setattr__(capability, "request_authorized", False)
    return capability


def _mapping(value: object, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict:
        _refuse()
    mapping = cast(dict[object, object], value)
    if set(mapping) != keys:
        _refuse()
    return cast(dict[str, object], mapping)


def _mapping_at_least(value: object, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict:
        _refuse()
    mapping = cast(dict[object, object], value)
    if not keys <= set(mapping):
        _refuse()
    return cast(dict[str, object], mapping)


def _canonical_closed_list(value: object, allowed: object) -> tuple[str, ...]:
    if type(value) is not list:
        _refuse()
    values = cast(list[object], value)
    choices = frozenset(cast(dict[str, object] | frozenset[str], allowed))
    result = tuple(_closed_string(item, choices) for item in values)
    if result != tuple(sorted(set(result))):
        _refuse()
    return result


def _closed_string(value: object, allowed: frozenset[str]) -> str:
    if type(value) is not str or value not in allowed:
        _refuse()
    return value


def _integer(value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _refuse()
    return value


def _refuse() -> Never:
    raise OutputGeometryError
