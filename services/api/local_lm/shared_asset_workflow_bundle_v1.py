"""Canonical neutral workflow bundles for the Shared Asset Library.

The bundle identifies exact executable workflow facts without carrying a local
display identity or lifecycle state. Callers decide whether and how to publish
the returned bytes; this module discovers no profile, database, Settings, or
shared-library root.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Final, NoReturn

from .workflow_dependencies import (
    WorkflowDependencyError,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_payload,
)

SCHEMA_ID: Final = "lm-atelier-shared-asset-workflow-bundle-v1"
SCHEMA_VERSION: Final = 1
INVALID_BUNDLE: Final = "shared asset workflow bundle is invalid"
MAX_WORKFLOW_BUNDLE_BYTES: Final = 4 * 1024 * 1024
MAX_JSON_DEPTH: Final = 32
MAX_JSON_NODES: Final = 100_000
MAX_JSON_OBJECT_MEMBERS: Final = 10_000
MAX_JSON_ARRAY_ITEMS: Final = 100_000
MAX_JSON_KEY_CHARS: Final = 256
MAX_JSON_STRING_CHARS: Final = 100_000
MAX_JSON_TEXT_CHARS: Final = 250_000
MAX_ENGINE_VERSION_CHARS: Final = 64
MAX_DECLARED_OUTPUT_COMPONENT: Final = 1_000_000

WORKFLOW_OPERATIONS: Final = frozenset(
    {
        "image_to_image",
        "image_to_video",
        "text",
        "text_to_image",
        "text_to_video",
    }
)
WORKFLOW_ENGINES: Final = frozenset({"comfyui"})
DECLARED_COMPUTE_PRECISIONS: Final = frozenset(
    {"bf16", "fp16", "fp32", "fp64", "fp8", "int4", "int8", "tf32"}
)
DECLARED_OUTPUT_FORMATS: Final = frozenset(
    {
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
        "video/h264-mp4",
        "video/mkv",
        "video/webm",
    }
)
DECLARED_COLOR_SPACES: Final = frozenset({"display-p3", "linear-srgb", "rec2020", "rec709", "srgb"})

_ENGINE_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}$")
_METADATA_KEYS: Final = frozenset({"declared_compute_precision", "declared_output"})
_OUTPUT_KEYS: Final = frozenset(
    {
        "channels",
        "color_space",
        "format",
        "frame_count",
        "frame_rate",
        "height",
        "width",
    }
)
_FRAME_RATE_KEYS: Final = frozenset({"denominator", "numerator"})
_INTEGER_OUTPUT_KEYS: Final = frozenset({"channels", "frame_count", "height", "width"})
_BUNDLE_KEYS: Final = frozenset(
    {
        "api_graph",
        "dependency_contract",
        "engine",
        "engine_version",
        "input_schema",
        "operation",
        "schema",
        "technical_metadata",
        "ui_graph",
        "version",
    }
)


class SharedAssetWorkflowBundleError(ValueError):
    """Fixed non-echoing refusal for an unusable shared workflow bundle."""


@dataclass(frozen=True)
class SharedAssetWorkflowBundle:
    operation: str
    engine: str
    engine_version: str | None
    ui_graph: dict[str, Any]
    api_graph: dict[str, Any]
    input_schema: dict[str, Any]
    dependency_contract: dict[str, object]
    technical_metadata: dict[str, Any]


def _invalid() -> NoReturn:
    raise SharedAssetWorkflowBundleError(INVALID_BUNDLE) from None


def _require_operation(value: object) -> str:
    if type(value) is not str or value not in WORKFLOW_OPERATIONS:
        _invalid()
    return value


def _require_engine(value: object) -> str:
    if type(value) is not str or value not in WORKFLOW_ENGINES:
        _invalid()
    return value


def _require_engine_version(value: object, *, budget: list[int]) -> str | None:
    if value is None:
        return None
    if type(value) is not str or len(value) > MAX_ENGINE_VERSION_CHARS:
        _invalid()
    _consume_text(value, budget=budget, key=False)
    if _ENGINE_VERSION.fullmatch(value) is None:
        _invalid()
    return value


def _consume_text(value: str, *, budget: list[int], key: bool) -> None:
    ceiling = MAX_JSON_KEY_CHARS if key else MAX_JSON_STRING_CHARS
    if (key and not value) or len(value) > ceiling:
        _invalid()
    budget[1] += len(value)
    if budget[1] > MAX_JSON_TEXT_CHARS:
        _invalid()


def _json_value(value: object, *, depth: int, budget: list[int]) -> object:
    budget[0] += 1
    if budget[0] > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
        _invalid()
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is str:
        _consume_text(value, budget=budget, key=False)
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _invalid()
        return 0.0 if value == 0 else value
    if type(value) is list:
        if len(value) > MAX_JSON_ARRAY_ITEMS:
            _invalid()
        return [_json_value(item, depth=depth + 1, budget=budget) for item in value]
    if type(value) is dict:
        if len(value) > MAX_JSON_OBJECT_MEMBERS:
            _invalid()
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                _invalid()
            _consume_text(key, budget=budget, key=True)
            result[key] = _json_value(item, depth=depth + 1, budget=budget)
        return result
    _invalid()


def _mapping(value: object, *, budget: list[int]) -> dict[str, Any]:
    if type(value) is not dict:
        _invalid()
    checked = _json_value(value, depth=0, budget=budget)
    if type(checked) is not dict:
        _invalid()
    return checked


def _nullable_positive_integer(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1 or value > MAX_DECLARED_OUTPUT_COMPONENT:
        _invalid()
    return value


def _nullable_enum(value: object, allowed: frozenset[str]) -> str | None:
    if value is None:
        return None
    if type(value) is not str or value not in allowed:
        _invalid()
    return value


def _nullable_frame_rate(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    if type(value) is not dict or set(value) != _FRAME_RATE_KEYS:
        _invalid()
    numerator = _nullable_positive_integer(value.get("numerator"))
    denominator = _nullable_positive_integer(value.get("denominator"))
    if numerator is None or denominator is None or math.gcd(numerator, denominator) != 1:
        _invalid()
    return {"denominator": denominator, "numerator": numerator}


def _require_declared_metadata(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _METADATA_KEYS:
        _invalid()
    output = value.get("declared_output")
    if type(output) is not dict or set(output) != _OUTPUT_KEYS:
        _invalid()
    normalized_output: dict[str, Any] = {}
    for key in sorted(_OUTPUT_KEYS):
        item = output.get(key)
        if key in _INTEGER_OUTPUT_KEYS:
            normalized_output[key] = _nullable_positive_integer(item)
        elif key == "frame_rate":
            normalized_output[key] = _nullable_frame_rate(item)
        elif key == "format":
            normalized_output[key] = _nullable_enum(item, DECLARED_OUTPUT_FORMATS)
        elif key == "color_space":
            normalized_output[key] = _nullable_enum(item, DECLARED_COLOR_SPACES)
        else:
            _invalid()
    return {
        "declared_compute_precision": _nullable_enum(
            value.get("declared_compute_precision"), DECLARED_COMPUTE_PRECISIONS
        ),
        "declared_output": normalized_output,
    }


def _dependency_contract(value: object) -> dict[str, object]:
    detached = _mapping(value, budget=[0, 0])
    return workflow_dependency_contract_payload(parse_workflow_dependency_contract(detached))


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, ValueError):
        _invalid()
    if len(encoded) > MAX_WORKFLOW_BUNDLE_BYTES:
        _invalid()
    return encoded


def _encode_workflow_bundle(
    *,
    operation: object,
    engine: object,
    engine_version: object,
    ui_graph: object,
    api_graph: object,
    input_schema: object,
    dependency_contract: object,
    technical_metadata: object,
) -> bytes:
    json_budget = [0, 0]
    payload = {
        "api_graph": _mapping(api_graph, budget=json_budget),
        "dependency_contract": _dependency_contract(dependency_contract),
        "engine": _require_engine(engine),
        "engine_version": _require_engine_version(engine_version, budget=json_budget),
        "input_schema": _mapping(input_schema, budget=json_budget),
        "operation": _require_operation(operation),
        "schema": SCHEMA_ID,
        "technical_metadata": _require_declared_metadata(technical_metadata),
        "ui_graph": _mapping(ui_graph, budget=json_budget),
        "version": SCHEMA_VERSION,
    }
    return _canonical_bytes(payload)


def encode_workflow_bundle(
    *,
    operation: object,
    engine: object,
    engine_version: object = None,
    ui_graph: object,
    api_graph: object,
    input_schema: object,
    dependency_contract: object,
    technical_metadata: object,
) -> bytes:
    """Return canonical bytes for immutable graph-declared workflow facts.

    The technical metadata describes only fixed literals or constraints inherent
    in the bundle. It must never contain profile settings, selected defaults,
    active runtime state, local lifecycle data, or observations from a run.
    Unknown or inapplicable declared facts are represented by the required nulls.
    Declared fields are independent facts; this codec does not infer additional
    operation/format applicability rules. Generic graph mappings use syntactic
    JSON identity: mapping order is
    normalized, while list order and JSON number type remain identity-bearing.
    Callers supply explicitly selected shareable graph content; this codec does
    not discover or sweep content from a profile-local record.
    """

    try:
        return _encode_workflow_bundle(
            operation=operation,
            engine=engine,
            engine_version=engine_version,
            ui_graph=ui_graph,
            api_graph=api_graph,
            input_schema=input_schema,
            dependency_contract=dependency_contract,
            technical_metadata=technical_metadata,
        )
    except (SharedAssetWorkflowBundleError, WorkflowDependencyError):
        pass
    _invalid()


def decode_workflow_bundle(payload: bytes) -> SharedAssetWorkflowBundle:
    """Validate canonical bundle bytes and return detached technical facts."""

    try:
        if type(payload) is not bytes or len(payload) > MAX_WORKFLOW_BUNDLE_BYTES:
            _invalid()
        value = json.loads(payload.decode("ascii"))
        if type(value) is not dict or set(value) != _BUNDLE_KEYS:
            _invalid()
        if value.get("schema") != SCHEMA_ID or value.get("version") != SCHEMA_VERSION:
            _invalid()
        canonical = _encode_workflow_bundle(
            operation=value.get("operation"),
            engine=value.get("engine"),
            engine_version=value.get("engine_version"),
            ui_graph=value.get("ui_graph"),
            api_graph=value.get("api_graph"),
            input_schema=value.get("input_schema"),
            dependency_contract=value.get("dependency_contract"),
            technical_metadata=value.get("technical_metadata"),
        )
        if canonical != payload:
            _invalid()
        decoded = json.loads(canonical.decode("ascii"))
        return SharedAssetWorkflowBundle(
            operation=decoded["operation"],
            engine=decoded["engine"],
            engine_version=decoded["engine_version"],
            ui_graph=decoded["ui_graph"],
            api_graph=decoded["api_graph"],
            input_schema=decoded["input_schema"],
            dependency_contract=decoded["dependency_contract"],
            technical_metadata=decoded["technical_metadata"],
        )
    except (OverflowError, RecursionError, TypeError, UnicodeDecodeError, ValueError):
        pass
    _invalid()


def workflow_bundle_sha256(payload: bytes) -> str:
    """Return the digest of one exact canonical bundle."""

    decode_workflow_bundle(payload)
    return hashlib.sha256(payload).hexdigest()
