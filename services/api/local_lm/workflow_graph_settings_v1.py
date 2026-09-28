"""Describe generated controls and graph-owned values without changing their inputs."""

from collections.abc import Mapping
from typing import Any

GRAPH_SETTINGS_SCHEMA_KEY = "x-lm-atelier-graph-settings"
GRAPH_SETTING_INPUT_NAMES = {
    "width": "width",
    "height": "height",
    "seed": "seed",
    "noise_seed": "seed",
    "steps": "steps",
    "cfg": "cfg",
    "sampler": "sampler",
    "sampler_name": "sampler",
    "scheduler": "scheduler",
    "denoise": "denoise",
    "batch_size": "batch_size",
    "frames": "frames",
    "frame_count": "frames",
    "fps": "fps",
    "frame_rate": "fps",
    "aspect_ratio": "aspect_ratio",
    "megapixels": "megapixels",
    "megapixel_budget": "megapixels",
    "round_to_multiple": "round_to_multiple",
}
GRAPH_SETTING_KEYS = frozenset(GRAPH_SETTING_INPUT_NAMES.values())
FIXED_SETTING_REASONS = {
    "read_only": "The workflow declares this control read-only.",
    "dynamic": "Changing this choice changes the node's inputs. Use the native editor.",
    "linked": "Supplied by a connected node in the workflow.",
    "primitive": "Fixed by a primitive node in the workflow.",
    "disconnected": "This node does not contribute to an output.",
    "unsupported": "This native control cannot be edited through generation settings.",
}
UNMAPPED_SETTING_REASON = "This setting is not mapped to an editable input in the workflow."


def _known_input(item: Mapping[str, Any]) -> bool:
    # Grown sockets have instance names independent of their template declaration.
    declared = item.get("declared_name", item["input_name"])
    return declared in GRAPH_SETTING_INPUT_NAMES


def workflow_graph_settings(input_schema: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Validate the generated settings record before using it to offer controls."""
    if GRAPH_SETTINGS_SCHEMA_KEY not in input_schema:
        return None
    raw = input_schema[GRAPH_SETTINGS_SCHEMA_KEY]
    properties = input_schema.get("properties", {})
    if (
        not isinstance(raw, Mapping)
        or not {"version", "bindings"}.issubset(raw)
        or set(raw) - {"version", "bindings", "fixed", "unbound_parameters"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or not isinstance(raw["bindings"], list)
        or not isinstance(raw.get("fixed", []), list)
        or not isinstance(raw.get("unbound_parameters", []), list)
        or not isinstance(properties, Mapping)
    ):
        raise ValueError("The generated settings record is invalid.")
    parameters: set[str] = set()
    paths: set[tuple[str, str]] = set()
    for binding in raw["bindings"]:
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            not in (
                {"parameter", "node_id", "input_name"},
                {"parameter", "node_id", "input_name", "declared_name"},
            )
            or not all(isinstance(value, str) and value for value in binding.values())
            or not _known_input(binding)
            or binding["parameter"] in parameters
            or not isinstance(properties.get(binding["parameter"]), Mapping)
            or (binding["node_id"], binding["input_name"]) in paths
        ):
            raise ValueError("A generated setting binding is invalid.")
        parameters.add(binding["parameter"])
        paths.add((binding["node_id"], binding["input_name"]))
    for fixed in raw.get("fixed", []):
        if (
            not isinstance(fixed, Mapping)
            or set(fixed)
            not in (
                {"node_id", "input_name", "label", "reason"},
                {"node_id", "input_name", "label", "reason", "declared_name"},
            )
            or not all(isinstance(value, str) and value for value in fixed.values())
            or not _known_input(fixed)
            or fixed["reason"] not in FIXED_SETTING_REASONS
            or (fixed["node_id"], fixed["input_name"]) in paths
        ):
            raise ValueError("A generated setting explanation is invalid.")
        paths.add((fixed["node_id"], fixed["input_name"]))
    for name in raw.get("unbound_parameters", []):
        if (
            not isinstance(name, str)
            or name not in GRAPH_SETTING_INPUT_NAMES
            or name in parameters
            or not isinstance(properties.get(name), Mapping)
        ):
            raise ValueError("A generated settings availability record is invalid.")
        parameters.add(name)
    return raw
