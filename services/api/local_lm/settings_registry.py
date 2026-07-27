from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

from .schemas import EngineCapabilities, SettingField

MAX_SETTING_FIELDS = 256
MAX_SETTING_STRING_LENGTH = 65_536
MAX_SETTING_ARRAY_ITEMS = 256
MAX_SETTING_OBJECT_ITEMS = 256
MAX_SETTING_NESTING_DEPTH = 8
MAX_SETTING_VALUE_NODES = 4_096
MAX_WORKFLOW_SCHEMA_PROPERTIES = 256
MAX_WORKFLOW_SETTING_KEY_LENGTH = 200

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
        label="Change strength",
        type="number",
        default=1,
        minimum=0,
        maximum=1,
        step=0.01,
        scope="workflow",
        visibility="advanced",
        help="For image edits, lower values preserve more of the source image.",
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

ROLE_SETTINGS: dict[str, list[SettingField]] = {
    "chat": CHAT_SETTINGS,
    "image": IMAGE_SETTINGS,
    "video": VIDEO_SETTINGS,
}

# These values are supplied by LM Atelier at dispatch time rather than by a
# user-editable settings layer. Letting a capability or workflow redefine them
# would allow persisted settings to replace the prompt or conditioning inputs.
_RESERVED_SETTING_KEYS = {
    "input_image",
    "input_images",
    "messages",
    "operation",
    "parameters",
    "prompt",
    "run_id",
    "tools",
    "workflow",
}
_NUMBERED_INPUT_IMAGE_KEY = re.compile(r"input_image_[0-9]+", re.IGNORECASE)


def builtin_settings_for_role(role: str) -> list[SettingField]:
    try:
        return list(ROLE_SETTINGS[role])
    except KeyError as exc:
        raise ValueError(f"unsupported engine role: {role}") from exc


def capability_settings_for_role(
    capabilities: EngineCapabilities,
    role: str,
) -> list[SettingField]:
    """Overlay one adapter role schema on LM Atelier's stable built-ins.

    Role mappings are preferred. The legacy flat list remains supported for
    older adapters; built-in keys known to belong only to another role are
    filtered out so a flat media schema does not make video-only controls
    appear on image profiles (or vice versa).
    """

    base_fields = builtin_settings_for_role(role)
    if role not in capabilities.roles:
        raise ValueError(f"engine does not advertise the {role} role")

    role_mapping_present = bool(capabilities.settings_by_role)
    if role_mapping_present:
        role_fields = capabilities.settings_by_role.get(role)
        if role_fields is None:
            raise ValueError(f"engine settings schema is missing the {role} role")
        adapter_fields = list(role_fields)
    else:
        exclusive_other_role_keys = {
            field.key
            for other_role, fields in ROLE_SETTINGS.items()
            if other_role != role
            for field in fields
        } - {field.key for field in base_fields}
        adapter_fields = [
            field for field in capabilities.settings if field.key not in exclusive_other_role_keys
        ]

    if len(adapter_fields) > MAX_SETTING_FIELDS:
        raise ValueError(
            f"{role} capability settings must contain at most {MAX_SETTING_FIELDS} fields"
        )

    base_by_key = {field.key: field for field in base_fields}
    advertised_by_key: dict[str, SettingField] = {}
    for field in adapter_fields:
        _validate_setting_key(field.key, source=f"{role} capability")
        prior = advertised_by_key.get(field.key)
        if prior is not None:
            if prior != field:
                base = base_by_key.get(field.key)
                if not role_mapping_present and base is not None:
                    if field == base:
                        advertised_by_key[field.key] = field
                        continue
                    if prior == base:
                        continue
                raise ValueError(
                    f"{role} capability settings contain conflicting definitions for {field.key}"
                )
            continue
        advertised_by_key[field.key] = field

    resolved: list[SettingField] = []
    for base in base_fields:
        advertised_field = advertised_by_key.pop(base.key, None)
        resolved.append(
            _merge_capability_field(base, advertised_field, role=role)
            if advertised_field is not None
            else base
        )

    for field in adapter_fields:
        if field.key not in advertised_by_key:
            continue
        advertised_by_key.pop(field.key)
        _reject_reserved_setting_key(field.key, source=f"{role} capability")
        if field.available:
            validate_settings({field.key: field.default}, [field])
        resolved.append(field)

    if len(resolved) > MAX_SETTING_FIELDS:
        raise ValueError(
            f"{role} settings must contain at most {MAX_SETTING_FIELDS} fields "
            "after built-in controls are preserved"
        )
    return resolved


def normalize_capability_settings(
    capabilities: EngineCapabilities,
) -> EngineCapabilities:
    """Return capabilities whose role schemas match server-side validation."""

    role_mapping = {
        role: capability_settings_for_role(capabilities, role) for role in capabilities.roles
    }
    legacy_settings: list[SettingField] = []
    for role in capabilities.roles:
        legacy_settings.extend(role_mapping[role])
    return capabilities.model_copy(
        update={
            "settings": legacy_settings,
            "settings_by_role": role_mapping,
        }
    )


def _merge_capability_field(
    base: SettingField,
    advertised: SettingField,
    *,
    role: str,
) -> SettingField:
    source = f"{role} capability setting {advertised.key}"
    if advertised.type != base.type:
        raise ValueError(f"{source} cannot change the built-in type")
    if advertised.scope != base.scope:
        raise ValueError(f"{source} cannot change the built-in scope")

    minimum = advertised.minimum if advertised.minimum is not None else base.minimum
    maximum = advertised.maximum if advertised.maximum is not None else base.maximum
    if (
        base.minimum is not None
        and advertised.minimum is not None
        and advertised.minimum < base.minimum
    ):
        raise ValueError(f"{source} cannot lower the built-in minimum")
    if (
        base.maximum is not None
        and advertised.maximum is not None
        and advertised.maximum > base.maximum
    ):
        raise ValueError(f"{source} cannot raise the built-in maximum")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{source} has an empty numeric range")

    choices = list(advertised.choices) if advertised.choices else list(base.choices)
    if base.choices and advertised.choices:
        broadened = [choice for choice in advertised.choices if choice not in base.choices]
        if broadened:
            raise ValueError(f"{source} cannot broaden the built-in choices")

    multiple_of = advertised.multiple_of if advertised.multiple_of is not None else base.multiple_of
    if (
        base.multiple_of is not None
        and advertised.multiple_of is not None
        and not _is_stricter_multiple(advertised.multiple_of, base.multiple_of)
    ):
        raise ValueError(f"{source} cannot weaken the built-in multiple")

    values = advertised.model_dump()
    values.update(
        {
            "minimum": minimum,
            "maximum": maximum,
            "step": advertised.step if advertised.step is not None else base.step,
            "multiple_of": multiple_of,
            "choices": choices,
        }
    )
    merged = SettingField(**values)
    if merged.available:
        validate_settings({merged.key: merged.default}, [merged])
    return merged


def _is_stricter_multiple(candidate: float, base: float) -> bool:
    quotient = candidate / base
    return candidate >= base and math.isclose(
        quotient,
        round(quotient),
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _validate_setting_key(key: str, *, source: str) -> None:
    if (
        not key
        or len(key) > MAX_WORKFLOW_SETTING_KEY_LENGTH
        or any(character < " " and character != "\t" for character in key)
    ):
        raise ValueError(
            f"{source} keys must be non-empty printable strings no longer than "
            f"{MAX_WORKFLOW_SETTING_KEY_LENGTH} characters"
        )


def _reject_reserved_setting_key(key: str, *, source: str) -> None:
    normalized = key.casefold()
    if (
        normalized.startswith("_")
        or normalized in _RESERVED_SETTING_KEYS
        or _NUMBERED_INPUT_IMAGE_KEY.fullmatch(normalized)
    ):
        raise ValueError(f"{source} cannot declare reserved setting key {key}")


def defaults(fields: Iterable[SettingField]) -> dict[str, Any]:
    return {field.key: field.default for field in fields if field.available}


def resolve_settings(*layers: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for layer in layers:
        if layer:
            resolved.update(layer)
    return resolved


def _validate_setting_value(key: str, value: Any) -> None:
    nodes = 0

    def walk(candidate: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_SETTING_VALUE_NODES:
            raise ValueError(
                f"{key} is too complex; use at most {MAX_SETTING_VALUE_NODES} nested values"
            )
        if depth > MAX_SETTING_NESTING_DEPTH:
            raise ValueError(
                f"{key} is nested too deeply; use at most {MAX_SETTING_NESTING_DEPTH} levels"
            )
        if isinstance(candidate, float):
            if not math.isfinite(candidate):
                raise ValueError(f"{key} numbers must be finite")
            return
        if isinstance(candidate, str):
            if len(candidate) > MAX_SETTING_STRING_LENGTH:
                raise ValueError(
                    f"{key} strings must be at most {MAX_SETTING_STRING_LENGTH} characters"
                )
            return
        if isinstance(candidate, list):
            if len(candidate) > MAX_SETTING_ARRAY_ITEMS:
                raise ValueError(
                    f"{key} arrays must contain at most {MAX_SETTING_ARRAY_ITEMS} items"
                )
            for item in candidate:
                walk(item, depth + 1)
            return
        if isinstance(candidate, dict):
            if len(candidate) > MAX_SETTING_OBJECT_ITEMS:
                raise ValueError(
                    f"{key} objects must contain at most {MAX_SETTING_OBJECT_ITEMS} items"
                )
            for nested_key, nested_value in candidate.items():
                if not isinstance(nested_key, str):
                    raise ValueError(f"{key} object keys must be strings")
                if len(nested_key) > MAX_WORKFLOW_SETTING_KEY_LENGTH:
                    raise ValueError(
                        f"{key} object keys must be at most "
                        f"{MAX_WORKFLOW_SETTING_KEY_LENGTH} characters"
                    )
                walk(nested_value, depth + 1)
            return
        if candidate is not None and not isinstance(candidate, (bool, int)):
            raise ValueError(f"{key} contains a value that cannot be stored as JSON")

    walk(value, 0)


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
    if isinstance(properties, Mapping) and len(properties) > MAX_WORKFLOW_SCHEMA_PROPERTIES:
        raise ValueError(
            "workflow input schema must declare at most "
            f"{MAX_WORKFLOW_SCHEMA_PROPERTIES} properties"
        )
    _validate_setting_value("workflow input schema", dict(input_schema))
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
        if len(key) > MAX_WORKFLOW_SETTING_KEY_LENGTH:
            raise ValueError(
                "workflow setting keys must be at most "
                f"{MAX_WORKFLOW_SETTING_KEY_LENGTH} characters"
            )
        _validate_setting_key(key, source="workflow setting")
        _reject_reserved_setting_key(key, source="workflow setting")
        resolved.append(_workflow_setting(key, property_schema, None))
    if len(resolved) > MAX_SETTING_FIELDS:
        raise ValueError(
            f"workflow settings must contain at most {MAX_SETTING_FIELDS} fields "
            "after engine controls are preserved"
        )
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
    if base and declared_type != base.type:
        if base.type == "enum" and declared_type in {"boolean", "integer", "number", "string"}:
            declared_type = "enum"
        else:
            raise ValueError(f"workflow setting {key} cannot change the engine setting type")

    choices: list[Any]
    if "const" in schema:
        choices = [schema["const"]]
    elif isinstance(schema.get("enum"), list):
        choices = list(schema["enum"])
    else:
        choices = list(base.choices) if base else []
    if base and base.choices and choices:
        broadened = [choice for choice in choices if choice not in base.choices]
        if broadened:
            raise ValueError(f"workflow setting {key} cannot broaden the engine choices")

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
    if base and base.minimum is not None and minimum is not None and minimum < base.minimum:
        raise ValueError(f"workflow setting {key} cannot lower the engine minimum")
    if base and base.maximum is not None and maximum is not None and maximum > base.maximum:
        raise ValueError(f"workflow setting {key} cannot raise the engine maximum")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"workflow setting {key} has an empty numeric range")
    # LM Atelier resolves the random-seed sentinel before dispatching to the
    # workflow, so preserve it even when a workflow declares non-negative seeds.
    if key == "seed" and base and base.default == -1 and default == -1:
        minimum = min(float(minimum), -1) if minimum is not None else -1

    help_text = str(schema.get("description") or (base.help if base else ""))
    if "const" in schema:
        fixed_note = f"Fixed by this workflow at {schema['const']}."
        help_text = f"{help_text} {fixed_note}".strip()

    multiple_of = schema.get("multipleOf", base.multiple_of if base else None)
    if (
        base
        and base.multiple_of is not None
        and multiple_of is not None
        and not _is_stricter_multiple(float(multiple_of), base.multiple_of)
    ):
        raise ValueError(f"workflow setting {key} cannot weaken the engine multiple")

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
        multiple_of=multiple_of,
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
    if len(values) > MAX_SETTING_FIELDS:
        raise ValueError(f"settings must contain at most {MAX_SETTING_FIELDS} fields")
    definitions: dict[str, SettingField] = {}
    for field in fields:
        if field.key in definitions:
            raise ValueError(f"settings schema contains duplicate key: {field.key}")
        definitions[field.key] = field
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
        _validate_setting_value(key, value)
        for constraint_name, constraint in (
            ("minimum", field.minimum),
            ("maximum", field.maximum),
            ("step", field.step),
            ("multiple", field.multiple_of),
        ):
            if constraint is not None and not math.isfinite(constraint):
                raise ValueError(f"{key} {constraint_name} must be finite")
        if field.multiple_of is not None and field.multiple_of <= 0:
            raise ValueError(f"{key} multiple must be greater than zero")
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


def compatible_stored_settings(
    values: Mapping[str, Any] | None,
    fields: Iterable[SettingField],
) -> dict[str, Any]:
    """Return the still-compatible subset of a persisted settings layer.

    Profiles, presets, and scoped defaults can outlive engine or workflow
    schema changes. Each value is checked independently so one obsolete
    setting does not prevent a chat or imported project from opening.
    Explicit per-turn overrides remain strict.
    """

    definitions = {field.key: field for field in fields}
    compatible: dict[str, Any] = {}
    for key, value in (values or {}).items():
        field = definitions.get(key)
        if not field:
            continue
        try:
            compatible.update(validate_settings({key: value}, [field]))
        except ValueError:
            continue
    return compatible


def resolve_generation_settings(
    fields: Iterable[SettingField],
    *,
    request_fields: Iterable[SettingField] | None = None,
    profile_defaults: Iterable[Mapping[str, Any] | None] = (),
    project_defaults: Iterable[Mapping[str, Any] | None] = (),
    chat_defaults: Iterable[Mapping[str, Any] | None] = (),
    turn_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the persisted generation hierarchy in one place.

    The ordering is engine defaults, profile-level defaults, project defaults,
    chat defaults, then the explicit per-turn override. Callers can provide
    more than one layer within a scope (for example a bound preset followed by
    directly edited defaults); later values in a scope win.
    """

    all_fields = list(fields)
    allowed_request_fields = list(request_fields) if request_fields is not None else all_fields
    resolved = defaults(all_fields)
    for layer in profile_defaults:
        resolved.update(compatible_stored_settings(layer, all_fields))
    for layers in (project_defaults, chat_defaults):
        for layer in layers:
            resolved.update(compatible_stored_settings(layer, allowed_request_fields))
    resolved.update(validate_settings(turn_overrides or {}, allowed_request_fields))
    return validate_settings(resolved, all_fields)
