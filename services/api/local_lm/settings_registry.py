from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

from .schemas import SettingField

CHAT_SETTINGS = [
    SettingField(
        key="context_length",
        label="Context length",
        type="integer",
        default=8192,
        minimum=512,
        maximum=1_048_576,
        step=512,
        scope="load",
        visibility="basic",
        restart_required=True,
        help="Maximum tokens held in the model context.",
    ),
    SettingField(
        key="gpu_layers",
        label="GPU offload layers",
        type="integer",
        default=-1,
        minimum=-1,
        maximum=999,
        scope="load",
        visibility="basic",
        restart_required=True,
        help="-1 asks the runtime to offload as much as possible.",
    ),
    SettingField(
        key="threads",
        label="CPU threads",
        type="integer",
        default=0,
        minimum=0,
        maximum=512,
        scope="load",
        visibility="advanced",
        restart_required=True,
        help="0 lets the runtime choose.",
    ),
    SettingField(
        key="batch_size",
        label="Batch size",
        type="integer",
        default=512,
        minimum=1,
        maximum=8192,
        scope="load",
        visibility="advanced",
        restart_required=True,
        help="Logical prompt-processing batch size.",
    ),
    SettingField(
        key="ubatch_size",
        label="Physical batch size",
        type="integer",
        default=128,
        minimum=1,
        maximum=8192,
        scope="load",
        visibility="expert",
        restart_required=True,
        help="Physical batch size used while evaluating prompts.",
    ),
    SettingField(
        key="flash_attention",
        label="Flash attention",
        type="boolean",
        default=True,
        scope="load",
        visibility="advanced",
        restart_required=True,
        help="Use optimized attention when the backend supports it.",
    ),
    SettingField(
        key="mmap",
        label="Memory-map weights",
        type="boolean",
        default=True,
        scope="load",
        visibility="advanced",
        restart_required=True,
        help="Map model weights from disk instead of copying them eagerly.",
    ),
    SettingField(
        key="mlock",
        label="Lock model in memory",
        type="boolean",
        default=False,
        scope="load",
        visibility="expert",
        restart_required=True,
        help="Ask the operating system not to swap model pages.",
    ),
    SettingField(
        key="split_mode",
        label="GPU split mode",
        type="enum",
        default="layer",
        choices=["none", "layer", "row"],
        scope="load",
        visibility="expert",
        restart_required=True,
        help="How tensors are distributed across multiple GPUs.",
    ),
    SettingField(
        key="tensor_split",
        label="Tensor split",
        type="string",
        default="",
        scope="load",
        visibility="expert",
        restart_required=True,
        help="Comma-separated relative proportions for multiple GPUs.",
    ),
    SettingField(
        key="main_gpu",
        label="Primary GPU",
        type="integer",
        default=0,
        minimum=0,
        maximum=64,
        scope="load",
        visibility="expert",
        restart_required=True,
        help="Primary device used by split modes that require one.",
    ),
    SettingField(
        key="rope_frequency_base",
        label="RoPE frequency base",
        type="number",
        default=0,
        minimum=0,
        maximum=10_000_000,
        scope="load",
        visibility="expert",
        restart_required=True,
        help="0 preserves the value declared by the model.",
    ),
    SettingField(
        key="rope_frequency_scale",
        label="RoPE frequency scale",
        type="number",
        default=0,
        minimum=0,
        maximum=100,
        scope="load",
        visibility="expert",
        restart_required=True,
        help="0 preserves the value declared by the model.",
    ),
    SettingField(
        key="kv_cache_type_k",
        label="K cache type",
        type="enum",
        default="f16",
        choices=["f32", "f16", "bf16", "q8_0", "q4_0"],
        scope="load",
        visibility="expert",
        restart_required=True,
        help="Lower precision can reduce context memory at a quality cost.",
    ),
    SettingField(
        key="kv_cache_type_v",
        label="V cache type",
        type="enum",
        default="f16",
        choices=["f32", "f16", "bf16", "q8_0", "q4_0"],
        scope="load",
        visibility="expert",
        restart_required=True,
        help="Lower precision can reduce context memory at a quality cost.",
    ),
    SettingField(
        key="temperature",
        label="Temperature",
        type="number",
        default=0.7,
        minimum=0,
        maximum=2,
        step=0.01,
        scope="request",
        visibility="basic",
        help="Higher values make output less deterministic.",
    ),
    SettingField(
        key="top_p",
        label="Top P",
        type="number",
        default=0.95,
        minimum=0,
        maximum=1,
        step=0.01,
        scope="request",
        visibility="advanced",
        help="Nucleus sampling threshold.",
    ),
    SettingField(
        key="top_k",
        label="Top K",
        type="integer",
        default=40,
        minimum=0,
        maximum=200,
        scope="request",
        visibility="advanced",
        help="Restrict each sample to the most likely tokens; 0 disables it.",
    ),
    SettingField(
        key="min_p",
        label="Min P",
        type="number",
        default=0.05,
        minimum=0,
        maximum=1,
        step=0.01,
        scope="request",
        visibility="advanced",
        help="Exclude tokens below a probability relative to the best token.",
    ),
    SettingField(
        key="repeat_penalty",
        label="Repeat penalty",
        type="number",
        default=1.1,
        minimum=0,
        maximum=2,
        step=0.01,
        scope="request",
        visibility="advanced",
        help="Discourage recent token repetition.",
    ),
    SettingField(
        key="presence_penalty",
        label="Presence penalty",
        type="number",
        default=0,
        minimum=-2,
        maximum=2,
        step=0.01,
        scope="request",
        visibility="advanced",
        help="Penalize tokens that have appeared at least once.",
    ),
    SettingField(
        key="frequency_penalty",
        label="Frequency penalty",
        type="number",
        default=0,
        minimum=-2,
        maximum=2,
        step=0.01,
        scope="request",
        visibility="advanced",
        help="Penalize tokens according to how often they have appeared.",
    ),
    SettingField(
        key="typical_p",
        label="Typical P",
        type="number",
        default=1,
        minimum=0,
        maximum=1,
        step=0.01,
        scope="request",
        visibility="expert",
        help="Locally typical sampling threshold; 1 disables it.",
    ),
    SettingField(
        key="repeat_last_n",
        label="Repeat window",
        type="integer",
        default=64,
        minimum=-1,
        maximum=131072,
        scope="request",
        visibility="expert",
        help="Number of recent tokens considered by repetition penalties.",
    ),
    SettingField(
        key="seed",
        label="Seed",
        type="integer",
        default=-1,
        minimum=-1,
        maximum=2_147_483_647,
        scope="request",
        visibility="advanced",
        help="-1 chooses a random seed.",
    ),
    SettingField(
        key="max_tokens",
        label="Maximum output",
        type="integer",
        default=1024,
        minimum=1,
        maximum=131_072,
        scope="request",
        visibility="basic",
        help="Maximum tokens generated for one assistant run.",
    ),
    SettingField(
        key="stop",
        label="Stop sequences",
        type="array",
        default=[],
        scope="request",
        visibility="expert",
        help="Generation stops when any complete sequence appears.",
    ),
]


IMAGE_SETTINGS = [
    SettingField(
        key="negative_prompt",
        label="Negative prompt",
        type="string",
        default="",
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="seed",
        label="Seed",
        type="integer",
        default=-1,
        minimum=-1,
        maximum=2_147_483_647,
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="width",
        label="Width",
        type="integer",
        default=1024,
        minimum=64,
        maximum=4096,
        step=64,
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="height",
        label="Height",
        type="integer",
        default=1024,
        minimum=64,
        maximum=4096,
        step=64,
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="steps",
        label="Steps",
        type="integer",
        default=28,
        minimum=1,
        maximum=200,
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="cfg",
        label="CFG",
        type="number",
        default=7,
        minimum=0,
        maximum=30,
        step=0.1,
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="sampler",
        label="Sampler",
        type="string",
        default="euler",
        scope="workflow",
        visibility="advanced",
    ),
    SettingField(
        key="scheduler",
        label="Scheduler",
        type="string",
        default="normal",
        scope="workflow",
        visibility="advanced",
    ),
    SettingField(
        key="denoise",
        label="Denoise",
        type="number",
        default=1,
        minimum=0,
        maximum=1,
        step=0.01,
        scope="workflow",
        visibility="advanced",
    ),
    SettingField(
        key="batch_size",
        label="Batch size",
        type="integer",
        default=1,
        minimum=1,
        maximum=16,
        scope="workflow",
        visibility="advanced",
    ),
    SettingField(
        key="loras",
        label="LoRA stack",
        type="array",
        default=[],
        scope="workflow",
        visibility="expert",
    ),
]


VIDEO_SETTINGS = [
    SettingField(
        key="seed",
        label="Seed",
        type="integer",
        default=-1,
        minimum=-1,
        maximum=2_147_483_647,
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="width",
        label="Width",
        type="integer",
        default=768,
        minimum=128,
        maximum=4096,
        step=64,
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="height",
        label="Height",
        type="integer",
        default=432,
        minimum=128,
        maximum=4096,
        step=64,
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="frames",
        label="Frames",
        type="integer",
        default=49,
        minimum=1,
        maximum=1024,
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="fps",
        label="FPS",
        type="number",
        default=24,
        minimum=1,
        maximum=120,
        step=1,
        scope="workflow",
        visibility="basic",
    ),
    SettingField(
        key="steps",
        label="Steps",
        type="integer",
        default=30,
        minimum=1,
        maximum=200,
        scope="workflow",
        visibility="advanced",
    ),
    SettingField(
        key="guidance",
        label="Guidance",
        type="number",
        default=6,
        minimum=0,
        maximum=30,
        step=0.1,
        scope="workflow",
        visibility="advanced",
    ),
    SettingField(
        key="motion_strength",
        label="Motion strength",
        type="number",
        default=1,
        minimum=0,
        maximum=2,
        step=0.05,
        scope="workflow",
        visibility="advanced",
    ),
    SettingField(
        key="codec",
        label="Codec",
        type="enum",
        default="h264",
        choices=["h264", "vp9", "av1"],
        scope="workflow",
        visibility="expert",
    ),
]


def defaults(fields: Iterable[SettingField]) -> dict[str, Any]:
    return {field.key: field.default for field in fields if field.available}


def resolve_settings(*layers: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for layer in layers:
        if layer:
            resolved.update(layer)
    return resolved


def workflow_settings(
    fields: Iterable[SettingField],
    input_schema: Mapping[str, Any] | None,
) -> list[SettingField]:
    """Overlay a workflow's declared JSON-Schema controls on engine defaults.

    Existing role fields remain available because older workflow schemas may
    declare only the controls they customize. A schema may also add a custom
    user control when it supplies a default, const, or enum. Properties without
    one of those UI hints are treated as runtime bindings such as input_image.
    """

    base_fields = list(fields)
    if not input_schema:
        return base_fields
    properties = input_schema.get("properties")
    if not isinstance(properties, Mapping):
        return base_fields

    definitions = {field.key: field for field in base_fields}
    resolved: list[SettingField] = []
    for field in base_fields:
        property_schema = properties.get(field.key)
        resolved.append(
            _workflow_setting(field.key, property_schema, field)
            if isinstance(property_schema, Mapping)
            else field
        )

    for key, property_schema in properties.items():
        if (
            not isinstance(key, str)
            or key in definitions
            or not isinstance(property_schema, Mapping)
            or not any(name in property_schema for name in ("default", "const", "enum"))
        ):
            continue
        resolved.append(_workflow_setting(key, property_schema, None))
    return resolved


def _workflow_setting(
    key: str,
    schema: Mapping[str, Any],
    base: SettingField | None,
) -> SettingField:
    declared_type = schema.get("type")
    supported_types = {"boolean", "integer", "number", "string", "enum", "array", "object"}
    if declared_type not in supported_types:
        declared_type = (
            base.type
            if base
            else _setting_type_for_value(schema.get("const", schema.get("default")))
        )
    if declared_type not in supported_types:
        raise ValueError(f"workflow setting {key} must declare a supported type")

    choices: list[Any]
    if "const" in schema:
        choices = [schema["const"]]
    elif isinstance(schema.get("enum"), list):
        choices = list(schema["enum"])
    else:
        choices = list(base.choices) if base else []

    if "const" in schema:
        default = schema["const"]
    elif "default" in schema:
        default = schema["default"]
    elif base:
        default = base.default
    elif choices:
        default = choices[0]
    else:
        default = None

    minimum = schema.get("minimum", base.minimum if base else None)
    maximum = schema.get("maximum", base.maximum if base else None)
    # LM Atelier resolves the random-seed sentinel before dispatching to the
    # workflow, so preserve it even when a workflow declares non-negative seeds.
    if key == "seed" and base and base.default == -1 and default == -1:
        minimum = min(float(minimum), -1) if minimum is not None else -1

    help_text = str(schema.get("description") or (base.help if base else ""))
    if "const" in schema:
        fixed_note = f"Fixed by this workflow at {schema['const']}."
        help_text = f"{help_text} {fixed_note}".strip()

    return SettingField(
        key=key,
        label=str(schema.get("title") or (base.label if base else key.replace("_", " ").title())),
        type=cast(
            Literal["boolean", "integer", "number", "string", "enum", "array", "object"],
            declared_type,
        ),
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=schema.get("multipleOf", base.step if base else None),
        multiple_of=schema.get("multipleOf"),
        choices=choices,
        scope=base.scope if base else "workflow",
        visibility=base.visibility if base else "advanced",
        restart_required=base.restart_required if base else False,
        available=base.available if base else True,
        unavailable_reason=base.unavailable_reason if base else None,
        help=help_text,
    )


def _setting_type_for_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


def validate_settings(values: Mapping[str, Any], fields: Iterable[SettingField]) -> dict[str, Any]:
    definitions = {field.key: field for field in fields}
    unknown = set(values) - set(definitions)
    if unknown:
        raise ValueError(f"unsupported settings: {', '.join(sorted(unknown))}")
    validated: dict[str, Any] = {}
    for key, value in values.items():
        field = definitions[key]
        if not field.available:
            raise ValueError(field.unavailable_reason or f"{key} is unavailable")
        expected_types: dict[str, type[Any] | tuple[type[Any], ...]] = {
            "boolean": bool,
            "integer": int,
            "number": (int, float),
            "string": str,
            "enum": (str, int, float, bool),
            "array": list,
            "object": dict,
        }
        expected = expected_types[field.type]
        if not isinstance(value, expected) or (
            field.type in {"integer", "number"} and isinstance(value, bool)
        ):
            raise ValueError(f"{key} must be a {field.type}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if field.minimum is not None and value < field.minimum:
                raise ValueError(f"{key} must be at least {field.minimum}")
            if field.maximum is not None and value > field.maximum:
                raise ValueError(f"{key} must be at most {field.maximum}")
            if field.multiple_of is not None:
                quotient = value / field.multiple_of
                if not math.isclose(quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"{key} must be a multiple of {field.multiple_of}")
        if field.choices and value not in field.choices:
            raise ValueError(f"{key} must be one of {field.choices}")
        validated[key] = value
    return validated
