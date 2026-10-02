"""The portable record of how one generated picture or video was made.

A record names the output by its content hash and says what produced it: the
operation, the prompt when the person chose to include it, the seed and settings,
the inputs by hash, and the workflow, model and LoRA files by hash. It never
names anything that only means something on this computer - no chat, message,
run, profile or install identifier, no file path - so it can be shared, and a
later import can compare it with what is installed elsewhere.

The bytes are canonical: sorted keys, compact separators, ASCII, no NaN. The
digest covers those bytes with the digest field left out, behind a domain tag,
so a record cannot be mistaken for any other kind of signed JSON. Reading a
record refuses anything that is not byte-for-byte what this module would write,
any key it does not define, and any value outside the closed vocabularies below.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping
from typing import Any, Final

SCHEMA_ID: Final = "lm-atelier-output-recipe-v1"
SCHEMA_VERSION: Final = 1
DIGEST_DOMAIN: Final = SCHEMA_ID.encode("ascii") + b"\0"

#: Generous for one output: a record carries no graph and no media.
MAX_RECORD_BYTES: Final = 256 * 1024
MAX_DEPTH: Final = 6
MAX_LIST_ITEMS: Final = 256
MAX_SETTINGS: Final = 256
MAX_STRING_LENGTH: Final = 65_536

OUTPUT_KINDS: Final = frozenset({"image", "video"})
OPERATIONS: Final = frozenset(
    {"text_to_image", "image_to_image", "text_to_video", "image_to_video"}
)
INPUT_ROLES: Final = frozenset({"source", "mask", "input"})
SEED_BINDINGS: Final = frozenset({"bound", "graph_literal", "unknown", "not_recorded"})
GRAPH_SOURCES: Final = frozenset({"frozen", "live", "unavailable"})
PROMPT_OMISSIONS: Final = frozenset(
    {"chosen", "removed_from_chat", "contains_local_reference", "too_long", "unavailable"}
)
REPRODUCIBILITY_STATES: Final = frozenset({"recorded", "incomplete"})

#: Why a record falls short of everything this application keeps for a replay.
MISSING_REASONS: Final = frozenset(
    {
        "prompt_omitted",
        "frozen_snapshot_absent",
        "frozen_snapshot_unavailable",
        "depends_on_other_outputs",
        "finished_after_generation",
        "seed_not_recorded",
        "workflow_unavailable",
        "workflow_unverified",
        "model_files_not_recorded",
        "lora_identity_missing",
        "settings_removed",
        "input_unavailable",
        "mock_engine",
    }
)

#: What no run records today, named so a record never implies it was checked.
NOT_RECORDED: Final = ("executed_graph_sha256", "runtime_version")

_HEX64 = re.compile(r"[0-9a-f]{64}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

_TOP_LEVEL: Final = frozenset(
    {
        "schema",
        "version",
        "exported_by",
        "output",
        "operation",
        "prompt",
        "seed",
        "settings",
        "inputs",
        "workflow",
        "model",
        "loras",
        "removed",
        "not_recorded",
        "reproducibility",
        "digest",
    }
)
_OUTPUT: Final = frozenset(
    {
        "sha256",
        "size_bytes",
        "media_type",
        "kind",
        "index",
        "count",
        "engine",
        "node_id",
        "collection",
        "raster",
    }
)
_RASTER: Final = frozenset({"width", "height"})
_PROMPT: Final = frozenset({"included", "positive", "negative", "omitted_reason"})
_SEED: Final = frozenset({"value", "binding"})
_SETTINGS: Final = frozenset({"bound", "unbound"})
_INPUT: Final = frozenset({"sha256", "role", "size_bytes", "media_type"})
_WORKFLOW: Final = frozenset(
    {
        "engine",
        "operation",
        "artifact_sha256",
        "verified",
        "graph_source",
        "contract_version",
        "dependency_contract_sha256",
        "binding_sha256",
    }
)
_MODEL: Final = frozenset({"files", "provider", "remote_id", "revision", "content_rating"})
_LORA: Final = frozenset({"sha256", "model_strength", "clip_strength", "enabled", "position"})
_REPRODUCIBILITY: Final = frozenset({"status", "missing"})
_EXPORTED_BY: Final = frozenset({"application", "version"})


class OutputRecipeFormatError(ValueError):
    """A record that is not exactly a version 1 record. The message never echoes input."""


def canonical_bytes(value: object) -> bytes:
    """The one encoding a record has: sorted, compact, ASCII, finite numbers only."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OutputRecipeFormatError("The record holds a value JSON cannot carry.") from exc
    return text.encode("ascii")


def record_digest(payload: Mapping[str, Any]) -> str:
    """The digest of a record, computed over everything but the digest field."""

    unsigned = {key: value for key, value in payload.items() if key != "digest"}
    return "sha256:" + hashlib.sha256(DIGEST_DOMAIN + canonical_bytes(unsigned)).hexdigest()


def seal_output_recipe(payload: Mapping[str, Any]) -> bytes:
    """Add the digest and return the record's bytes, checked as a reader would check them."""

    if "digest" in payload:
        raise OutputRecipeFormatError("A record is sealed exactly once.")
    sealed = {**payload, "digest": record_digest(payload)}
    content = canonical_bytes(sealed)
    open_output_recipe(content)
    return content


def open_output_recipe(content: bytes) -> dict[str, Any]:
    """Read a record, refusing anything this module would not have written."""

    if len(content) > MAX_RECORD_BYTES:
        raise OutputRecipeFormatError("The record is larger than a record can be.")
    try:
        text = content.decode("ascii")
        value = json.loads(text, object_pairs_hook=_refuse_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise OutputRecipeFormatError("The record is not valid canonical JSON.") from exc
    if not isinstance(value, dict):
        raise OutputRecipeFormatError("The record is not an object.")
    if canonical_bytes(value) != content:
        raise OutputRecipeFormatError("The record is not in canonical form.")
    _check_record(value)
    if not hmac.compare_digest(record_digest(value), value["digest"]):
        raise OutputRecipeFormatError("The record does not match its digest.")
    return value


def _refuse_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OutputRecipeFormatError("The record repeats a key.")
        result[key] = value
    return result


def _check_record(value: dict[str, Any]) -> None:
    _keys(value, _TOP_LEVEL, "record")
    _depth(value, 0)
    if value["schema"] != SCHEMA_ID or value["version"] != SCHEMA_VERSION:
        raise OutputRecipeFormatError("The record is not a version 1 output record.")
    if not isinstance(value["digest"], str) or not _DIGEST.fullmatch(value["digest"]):
        raise OutputRecipeFormatError("The record's digest is malformed.")
    exported_by = _object(value["exported_by"], _EXPORTED_BY, "exported_by")
    _string(exported_by["application"], "exported_by.application")
    _string(exported_by["version"], "exported_by.version")
    _check_output(value["output"])
    if value["operation"] not in OPERATIONS:
        raise OutputRecipeFormatError("The record names an unknown operation.")
    _check_prompt(value["prompt"])
    seed = _object(value["seed"], _SEED, "seed")
    if seed["binding"] not in SEED_BINDINGS:
        raise OutputRecipeFormatError("The record names an unknown seed binding.")
    if seed["value"] is not None and not _whole(seed["value"], minimum=0):
        raise OutputRecipeFormatError("The record's seed is not a whole number.")
    _check_settings(value["settings"])
    _check_inputs(value["inputs"])
    _check_workflow(value["workflow"])
    _check_model(value["model"])
    _check_loras(value["loras"])
    _sorted_names(value["removed"], None, "removed")
    _sorted_names(value["not_recorded"], frozenset(NOT_RECORDED), "not_recorded")
    reproducibility = _object(value["reproducibility"], _REPRODUCIBILITY, "reproducibility")
    if reproducibility["status"] not in REPRODUCIBILITY_STATES:
        raise OutputRecipeFormatError("The record names an unknown reproducibility state.")
    _sorted_names(reproducibility["missing"], MISSING_REASONS, "reproducibility.missing")
    if (reproducibility["status"] == "recorded") != (not reproducibility["missing"]):
        raise OutputRecipeFormatError("The record's reproducibility contradicts itself.")


def _check_output(value: object) -> None:
    output = _object(value, _OUTPUT, "output")
    _hex(output["sha256"], "output.sha256")
    if not _whole(output["size_bytes"], minimum=0):
        raise OutputRecipeFormatError("The output size is not a whole number.")
    _string(output["media_type"], "output.media_type")
    if output["kind"] not in OUTPUT_KINDS:
        raise OutputRecipeFormatError("The record names an unknown output kind.")
    if not _whole(output["index"], minimum=0) or not _whole(output["count"], minimum=1):
        raise OutputRecipeFormatError("The output position is not a whole number.")
    if output["index"] >= output["count"]:
        raise OutputRecipeFormatError("The output position is outside its count.")
    for key in ("engine", "node_id", "collection"):
        _optional_string(output[key], f"output.{key}")
    if output["raster"] is not None:
        raster = _object(output["raster"], _RASTER, "output.raster")
        if not _whole(raster["width"], minimum=1) or not _whole(raster["height"], minimum=1):
            raise OutputRecipeFormatError("The output raster size is not positive.")


def _check_prompt(value: object) -> None:
    prompt = _object(value, _PROMPT, "prompt")
    if not isinstance(prompt["included"], bool):
        raise OutputRecipeFormatError("The record does not say whether the prompt is included.")
    if prompt["included"]:
        _string(prompt["positive"], "prompt.positive")
        _optional_string(prompt["negative"], "prompt.negative")
        if prompt["omitted_reason"] is not None:
            raise OutputRecipeFormatError("An included prompt gives no reason for omission.")
        return
    if prompt["positive"] is not None or prompt["negative"] is not None:
        raise OutputRecipeFormatError("An omitted prompt carries no text.")
    if prompt["omitted_reason"] not in PROMPT_OMISSIONS:
        raise OutputRecipeFormatError("The record names an unknown reason for omission.")


def _check_settings(value: object) -> None:
    settings = _object(value, _SETTINGS, "settings")
    names: set[str] = set()
    for group in ("bound", "unbound"):
        entries = settings[group]
        if not isinstance(entries, dict) or len(entries) > MAX_SETTINGS:
            raise OutputRecipeFormatError("A settings group is not a bounded object.")
        for key, item in entries.items():
            _string(key, "settings key")
            if key in names:
                raise OutputRecipeFormatError("A setting is both bound and unbound.")
            names.add(key)
            if not _scalar(item):
                raise OutputRecipeFormatError("A setting is not a plain value.")


def _check_inputs(value: object) -> None:
    for item in _list(value, "inputs"):
        entry = _object(item, _INPUT, "input")
        _hex(entry["sha256"], "input.sha256")
        if entry["role"] not in INPUT_ROLES:
            raise OutputRecipeFormatError("The record names an unknown input role.")
        if entry["size_bytes"] is not None and not _whole(entry["size_bytes"], minimum=0):
            raise OutputRecipeFormatError("An input size is not a whole number.")
        _optional_string(entry["media_type"], "input.media_type")


def _check_workflow(value: object) -> None:
    if value is None:
        return
    workflow = _object(value, _WORKFLOW, "workflow")
    _string(workflow["engine"], "workflow.engine")
    if workflow["operation"] not in OPERATIONS:
        raise OutputRecipeFormatError("The workflow names an unknown operation.")
    for key in ("artifact_sha256", "dependency_contract_sha256", "binding_sha256"):
        if workflow[key] is not None:
            _hex(workflow[key], f"workflow.{key}")
    if not isinstance(workflow["verified"], bool):
        raise OutputRecipeFormatError("The workflow does not say whether it was verified.")
    if workflow["graph_source"] not in GRAPH_SOURCES:
        raise OutputRecipeFormatError("The workflow names an unknown graph source.")
    if not _whole(workflow["contract_version"], minimum=1):
        raise OutputRecipeFormatError("The workflow contract version is not a whole number.")


def _check_model(value: object) -> None:
    if value is None:
        return
    model = _object(value, _MODEL, "model")
    files = model["files"]
    if not isinstance(files, dict) or len(files) > MAX_LIST_ITEMS:
        raise OutputRecipeFormatError("The model files are not a bounded object.")
    for name, digest in files.items():
        _string(name, "model file name")
        _hex(digest, "model file sha256")
    for key in ("provider", "remote_id", "revision", "content_rating"):
        _optional_string(model[key], f"model.{key}")


def _check_loras(value: object) -> None:
    for item in _list(value, "loras"):
        lora = _object(item, _LORA, "lora")
        _hex(lora["sha256"], "lora.sha256")
        for key in ("model_strength", "clip_strength"):
            if not _number(lora[key]):
                raise OutputRecipeFormatError("A LoRA strength is not a number.")
        if not isinstance(lora["enabled"], bool) or not _whole(lora["position"], minimum=0):
            raise OutputRecipeFormatError("A LoRA entry is malformed.")


def _object(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OutputRecipeFormatError(f"The record's {label} is not an object.")
    _keys(value, keys, label)
    return value


def _keys(value: dict[str, Any], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise OutputRecipeFormatError(f"The record's {label} has the wrong fields.")


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list) or len(value) > MAX_LIST_ITEMS:
        raise OutputRecipeFormatError(f"The record's {label} is not a bounded list.")
    return value


def _sorted_names(value: object, allowed: frozenset[str] | None, label: str) -> None:
    names = _list(value, label)
    if any(not isinstance(name, str) or not name for name in names):
        raise OutputRecipeFormatError(f"The record's {label} holds something other than names.")
    if names != sorted(set(names)):
        raise OutputRecipeFormatError(f"The record's {label} is not sorted and distinct.")
    if allowed is not None and not set(names) <= allowed:
        raise OutputRecipeFormatError(f"The record's {label} names something unknown.")


def _depth(value: object, level: int) -> None:
    if level > MAX_DEPTH:
        raise OutputRecipeFormatError("The record is nested too deeply.")
    if isinstance(value, dict):
        for item in value.values():
            _depth(item, level + 1)
    elif isinstance(value, list):
        for item in value:
            _depth(item, level + 1)


def _string(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > MAX_STRING_LENGTH:
        raise OutputRecipeFormatError(f"The record's {label} is not a bounded string.")


def _optional_string(value: object, label: str) -> None:
    if value is not None:
        _string(value, label)


def _hex(value: object, label: str) -> None:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise OutputRecipeFormatError(f"The record's {label} is not a sha256.")


def _whole(value: object, *, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value)


def _scalar(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, str):
        return len(value) <= MAX_STRING_LENGTH
    return _number(value)
