"""Pure, bounded contracts for immutable user-authored prompt templates.

This module does not expand prompts with a model, resolve local resources, or
create media work. It only freezes one exact JSON-shaped contract, renders
caller-supplied slot values in one pass, and assigns a canonical content ID.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import NoReturn, cast

from .lora_constraints import MAX_LORA_STRENGTH

PROMPT_TEMPLATE_SCHEMA_VERSION = 1
PROMPT_TEMPLATE_OPERATION = "text_to_image"
PROMPT_TEMPLATE_INVALID = "Prompt template contract is invalid."
PROMPT_TEMPLATE_VALUES_INVALID = "Prompt template values are invalid."

MAX_TEMPLATE_BODY_CHARS = 16_000
MAX_TEMPLATE_SLOTS = 32
MAX_TEMPLATE_CHOICES = 64
MAX_TEMPLATE_TOTAL_CHOICES = 512
MAX_TEMPLATE_VALUE_CHARS = 2_000
MAX_TEMPLATE_GUIDANCE_CHARS = 4_000
MAX_TEMPLATE_RENDERED_CHARS = 32_000
MAX_TEMPLATE_LORAS = 16
MAX_TEMPLATE_LORA_STACKS = 32
MAX_TEMPLATE_POOL_LORAS = 64
MAX_TEMPLATE_WORKFLOW_OPTIONS = 16
MAX_TEMPLATE_DOCUMENT_DEPTH = 16
MAX_TEMPLATE_DOCUMENT_NODES = 4_096
MAX_TEMPLATE_DOCUMENT_CHARS = 65_536
MAX_TEMPLATE_DOCUMENT_BYTES = 262_144
_ROOT_KEYS = frozenset({"schema_version", "operation", "body", "slots", "resource_policy"})
_SLOT_COMMON_KEYS = frozenset({"name", "mode", "variation_scope"})
_RESOURCE_INHERITED_KEYS = frozenset({"mode"})
_RESOURCE_FIXED_KEYS = frozenset({"mode", "workflow_revision_id", "lora_policy"})
_RESOURCE_POOL_KEYS = frozenset({"mode", "strategy", "options"})
_RESOURCE_OPTION_KEYS = frozenset({"workflow_revision_id", "lora_policy"})
_LORA_POLICY_SIMPLE_KEYS = frozenset({"mode"})
_LORA_POLICY_FIXED_KEYS = frozenset({"mode", "stack"})
_LORA_POLICY_POOL_KEYS = frozenset({"mode", "strategy", "stacks"})
_LORA_KEYS = frozenset({"sha256", "model_strength", "clip_strength"})
_SLOT_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}", re.ASCII)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_TOKEN = re.compile(r"{{([a-z][a-z0-9_]{0,63})}}", re.ASCII)


class PromptTemplateError(ValueError):
    """A template contract or render request failed a closed public boundary."""

    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class PromptTemplateSlotMode(StrEnum):
    INPUT = "input"
    CHOICE = "choice"
    MODEL = "model"
    FIXED = "fixed"


class PromptTemplateChoiceStrategy(StrEnum):
    DISTINCT = "distinct"
    WITH_REPLACEMENT = "with_replacement"


class PromptTemplateVariationScope(StrEnum):
    ITEM = "item"
    BATCH = "batch"


class PromptTemplateResourceMode(StrEnum):
    INHERITED = "inherited"
    FIXED = "fixed"
    POOL = "pool"


class PromptTemplateLoraPolicyMode(StrEnum):
    INHERITED_AUTO = "inherited_auto"
    NONE = "none"
    FIXED = "fixed"
    POOL = "pool"


class PromptTemplatePoolStrategy(StrEnum):
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"


@dataclass(frozen=True, slots=True)
class PromptTemplateSlot:
    name: str
    mode: PromptTemplateSlotMode
    variation_scope: PromptTemplateVariationScope
    choices: tuple[str, ...] = ()
    choice_strategy: PromptTemplateChoiceStrategy = PromptTemplateChoiceStrategy.DISTINCT
    guidance: str | None = None
    fixed_value: str | None = None


@dataclass(frozen=True, slots=True)
class PromptTemplateLora:
    sha256: str
    model_strength: float
    clip_strength: float


@dataclass(frozen=True, slots=True)
class PromptTemplateLoraPolicy:
    mode: PromptTemplateLoraPolicyMode
    stack: tuple[PromptTemplateLora, ...] = ()
    strategy: PromptTemplatePoolStrategy | None = None
    stacks: tuple[tuple[PromptTemplateLora, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class PromptTemplateResourceOption:
    workflow_revision_id: str
    lora_policy: PromptTemplateLoraPolicy


@dataclass(frozen=True, slots=True)
class PromptTemplateResourcePolicy:
    mode: PromptTemplateResourceMode
    workflow_revision_id: str | None = None
    lora_policy: PromptTemplateLoraPolicy | None = None
    strategy: PromptTemplatePoolStrategy | None = None
    options: tuple[PromptTemplateResourceOption, ...] = ()


@dataclass(frozen=True, slots=True)
class PromptTemplateContract:
    schema_version: int
    operation: str
    body: str
    slots: tuple[PromptTemplateSlot, ...]
    resource_policy: PromptTemplateResourcePolicy


def _invalid() -> NoReturn:
    raise PromptTemplateError("prompt-template-invalid", PROMPT_TEMPLATE_INVALID)


def _invalid_values() -> NoReturn:
    raise PromptTemplateError("prompt-template-values-invalid", PROMPT_TEMPLATE_VALUES_INVALID)


def _detach_exact_json(value: object, *, values_error: bool = False) -> object:
    """Copy an exact built-in JSON tree within fixed work and memory bounds."""

    fail = _invalid_values if values_error else _invalid
    if type(value) not in {dict, list, str, int, float, bool, type(None)}:
        fail()

    seen_containers: set[int] = set()
    nodes = 0
    characters = 0
    encoded_bytes = 0

    def count_scalar(item: object) -> None:
        nonlocal nodes, characters, encoded_bytes
        nodes += 1
        if nodes > MAX_TEMPLATE_DOCUMENT_NODES:
            fail()
        if type(item) is float and not math.isfinite(item):
            fail()
        if type(item) is str:
            characters += len(item)
            if characters > MAX_TEMPLATE_DOCUMENT_CHARS:
                fail()
            encoded_bytes += len(item.encode("utf-8"))
            if encoded_bytes > MAX_TEMPLATE_DOCUMENT_BYTES:
                fail()

    if type(value) not in {dict, list}:
        count_scalar(value)
        return value

    root: dict[str, object] | list[object]
    root = {} if type(value) is dict else []
    stack: list[tuple[object, dict[str, object] | list[object], int]] = [(value, root, 0)]
    while stack:
        source, target, depth = stack.pop()
        nodes += 1
        if nodes > MAX_TEMPLATE_DOCUMENT_NODES or depth > MAX_TEMPLATE_DOCUMENT_DEPTH:
            fail()
        identity = id(source)
        if identity in seen_containers:
            fail()
        seen_containers.add(identity)

        if type(source) is dict:
            if type(target) is not dict or len(source) > MAX_TEMPLATE_DOCUMENT_NODES:
                fail()
            pending: list[tuple[object, dict[str, object] | list[object], int]] = []
            for key, child in source.items():
                if type(key) is not str or not key or len(key) > 64:
                    fail()
                count_scalar(key)
                if type(child) is dict:
                    copied: dict[str, object] = {}
                    target[key] = copied
                    pending.append((child, copied, depth + 1))
                elif type(child) is list:
                    copied_list: list[object] = []
                    target[key] = copied_list
                    pending.append((child, copied_list, depth + 1))
                elif type(child) in {str, int, float, bool, type(None)}:
                    count_scalar(child)
                    target[key] = child
                else:
                    fail()
            stack.extend(reversed(pending))
        else:
            if type(source) is not list or type(target) is not list:
                fail()
            if len(source) > MAX_TEMPLATE_DOCUMENT_NODES:
                fail()
            pending_list: list[tuple[object, dict[str, object] | list[object], int]] = []
            for child in source:
                if type(child) is dict:
                    copied_dict: dict[str, object] = {}
                    target.append(copied_dict)
                    pending_list.append((child, copied_dict, depth + 1))
                elif type(child) is list:
                    copied_list = []
                    target.append(copied_list)
                    pending_list.append((child, copied_list, depth + 1))
                elif type(child) in {str, int, float, bool, type(None)}:
                    count_scalar(child)
                    target.append(child)
                else:
                    fail()
            stack.extend(reversed(pending_list))
    return root


def _exact_keys(value: object, expected: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        _invalid()
    return value


def _has_forbidden_controls(value: str) -> bool:
    return any(
        (ord(character) < 32 and character not in {"\n", "\t"}) or 127 <= ord(character) <= 159
        for character in value
    )


def _text(value: object, *, maximum: int) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        _invalid()
    if not value.strip() or _has_forbidden_controls(value):
        _invalid()
    return value


def _slot_name(value: object) -> str:
    if type(value) is not str or _SLOT_NAME.fullmatch(value) is None:
        _invalid()
    return value


def _enum_value[EnumT: Enum](value: object, enum_type: type[EnumT]) -> EnumT:
    if type(value) is not str:
        _invalid()
    try:
        return enum_type(value)
    except ValueError:
        _invalid()


def _strength(value: object) -> float:
    if type(value) not in {int, float}:
        _invalid()
    normalized = float(cast(int | float, value))
    if not math.isfinite(normalized) or not -MAX_LORA_STRENGTH <= normalized <= MAX_LORA_STRENGTH:
        _invalid()
    return 0.0 if normalized == 0 else normalized


def _parse_slot(value: object) -> PromptTemplateSlot:
    if type(value) is not dict:
        _invalid()
    mode = _enum_value(value.get("mode"), PromptTemplateSlotMode)
    expected = _SLOT_COMMON_KEYS
    if mode is PromptTemplateSlotMode.CHOICE:
        expected = expected | {"choices"}
        if "choice_strategy" in value:
            expected = expected | {"choice_strategy"}
    elif mode is PromptTemplateSlotMode.MODEL:
        expected = expected | {"guidance"}
    elif mode is PromptTemplateSlotMode.FIXED:
        expected = expected | {"fixed_value"}
    entry = _exact_keys(value, frozenset(expected))
    name = _slot_name(entry["name"])
    scope = _enum_value(entry["variation_scope"], PromptTemplateVariationScope)
    choices: tuple[str, ...] = ()
    choice_strategy = PromptTemplateChoiceStrategy.DISTINCT
    guidance: str | None = None
    fixed_value: str | None = None
    if mode is PromptTemplateSlotMode.CHOICE:
        raw_choices = entry["choices"]
        if type(raw_choices) is not list or not raw_choices:
            _invalid()
        if len(raw_choices) > MAX_TEMPLATE_CHOICES:
            _invalid()
        parsed = tuple(_text(choice, maximum=MAX_TEMPLATE_VALUE_CHARS) for choice in raw_choices)
        if len(set(parsed)) != len(parsed):
            _invalid()
        choices = parsed
        if "choice_strategy" in entry:
            choice_strategy = _enum_value(entry["choice_strategy"], PromptTemplateChoiceStrategy)
    elif mode is PromptTemplateSlotMode.MODEL:
        guidance = _text(entry["guidance"], maximum=MAX_TEMPLATE_GUIDANCE_CHARS)
    elif mode is PromptTemplateSlotMode.FIXED:
        if scope is not PromptTemplateVariationScope.BATCH:
            _invalid()
        fixed_value = _text(entry["fixed_value"], maximum=MAX_TEMPLATE_VALUE_CHARS)
    return PromptTemplateSlot(
        name=name,
        mode=mode,
        variation_scope=scope,
        choices=choices,
        choice_strategy=choice_strategy,
        guidance=guidance,
        fixed_value=fixed_value,
    )


def _parse_lora_policy(value: object) -> PromptTemplateLoraPolicy:
    if type(value) is not dict:
        _invalid()
    mode = _enum_value(value.get("mode"), PromptTemplateLoraPolicyMode)
    expected = {
        PromptTemplateLoraPolicyMode.FIXED: _LORA_POLICY_FIXED_KEYS,
        PromptTemplateLoraPolicyMode.POOL: _LORA_POLICY_POOL_KEYS,
    }.get(mode, _LORA_POLICY_SIMPLE_KEYS)
    entry = _exact_keys(value, expected)
    if mode not in {PromptTemplateLoraPolicyMode.FIXED, PromptTemplateLoraPolicyMode.POOL}:
        return PromptTemplateLoraPolicy(mode=mode)

    def parse_stack(raw_stack: object) -> tuple[PromptTemplateLora, ...]:
        if type(raw_stack) is not list or not raw_stack or len(raw_stack) > MAX_TEMPLATE_LORAS:
            _invalid()
        stack: list[PromptTemplateLora] = []
        seen: set[str] = set()
        for raw_lora in raw_stack:
            item = _exact_keys(raw_lora, _LORA_KEYS)
            sha256 = item["sha256"]
            if type(sha256) is not str or _SHA256.fullmatch(sha256) is None:
                _invalid()
            if sha256 in seen:
                _invalid()
            seen.add(sha256)
            stack.append(
                PromptTemplateLora(
                    sha256=sha256,
                    model_strength=_strength(item["model_strength"]),
                    clip_strength=_strength(item["clip_strength"]),
                )
            )
        return tuple(stack)

    if mode is PromptTemplateLoraPolicyMode.FIXED:
        return PromptTemplateLoraPolicy(mode=mode, stack=parse_stack(entry["stack"]))
    strategy = _enum_value(entry["strategy"], PromptTemplatePoolStrategy)
    raw_stacks = entry["stacks"]
    if type(raw_stacks) is not list or not 2 <= len(raw_stacks) <= MAX_TEMPLATE_LORA_STACKS:
        _invalid()
    stacks = tuple(parse_stack(raw_stack) for raw_stack in raw_stacks)
    if sum(len(stack) for stack in stacks) > MAX_TEMPLATE_POOL_LORAS:
        _invalid()
    if len(set(stacks)) != len(stacks):
        _invalid()
    return PromptTemplateLoraPolicy(mode=mode, strategy=strategy, stacks=stacks)


def _parse_resource_option(
    value: object,
    *,
    allow_lora_pool: bool,
) -> PromptTemplateResourceOption:
    entry = _exact_keys(value, _RESOURCE_OPTION_KEYS)
    revision_id = entry["workflow_revision_id"]
    if type(revision_id) is not str or _SAFE_ID.fullmatch(revision_id) is None:
        _invalid()
    lora_policy = _parse_lora_policy(entry["lora_policy"])
    if not allow_lora_pool and lora_policy.mode is PromptTemplateLoraPolicyMode.POOL:
        _invalid()
    return PromptTemplateResourceOption(
        workflow_revision_id=revision_id,
        lora_policy=lora_policy,
    )


def _parse_resource_policy(value: object) -> PromptTemplateResourcePolicy:
    if type(value) is not dict:
        _invalid()
    mode = _enum_value(value.get("mode"), PromptTemplateResourceMode)
    expected = {
        PromptTemplateResourceMode.FIXED: _RESOURCE_FIXED_KEYS,
        PromptTemplateResourceMode.POOL: _RESOURCE_POOL_KEYS,
    }.get(mode, _RESOURCE_INHERITED_KEYS)
    entry = _exact_keys(value, expected)
    if mode is PromptTemplateResourceMode.INHERITED:
        return PromptTemplateResourcePolicy(mode=mode)
    if mode is PromptTemplateResourceMode.FIXED:
        option = _parse_resource_option(
            {
                "workflow_revision_id": entry["workflow_revision_id"],
                "lora_policy": entry["lora_policy"],
            },
            allow_lora_pool=True,
        )
        return PromptTemplateResourcePolicy(
            mode=mode,
            workflow_revision_id=option.workflow_revision_id,
            lora_policy=option.lora_policy,
        )
    strategy = _enum_value(entry["strategy"], PromptTemplatePoolStrategy)
    raw_options = entry["options"]
    if type(raw_options) is not list or not 2 <= len(raw_options) <= MAX_TEMPLATE_WORKFLOW_OPTIONS:
        _invalid()
    options = tuple(_parse_resource_option(option, allow_lora_pool=False) for option in raw_options)
    if len(set(options)) != len(options):
        _invalid()
    if (
        sum(
            len(option.lora_policy.stack)
            for option in options
            if option.lora_policy.mode is PromptTemplateLoraPolicyMode.FIXED
        )
        > MAX_TEMPLATE_POOL_LORAS
    ):
        _invalid()
    return PromptTemplateResourcePolicy(
        mode=mode,
        strategy=strategy,
        options=options,
    )


def _body_tokens(body: str) -> tuple[str, ...]:
    names: list[str] = []
    position = 0
    for match in _TOKEN.finditer(body):
        gap = body[position : match.start()]
        if "{" in gap or "}" in gap:
            _invalid()
        names.append(match.group(1))
        position = match.end()
    if "{" in body[position:] or "}" in body[position:]:
        _invalid()
    if len(names) != len(set(names)):
        _invalid()
    return tuple(names)


def parse_prompt_template_contract(value: object) -> PromptTemplateContract:
    """Parse one exact template payload into immutable normalized data."""

    detached = _detach_exact_json(value)
    root = _exact_keys(detached, _ROOT_KEYS)
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != PROMPT_TEMPLATE_SCHEMA_VERSION
    ):
        _invalid()
    if type(root["operation"]) is not str or root["operation"] != PROMPT_TEMPLATE_OPERATION:
        _invalid()
    body = _text(root["body"], maximum=MAX_TEMPLATE_BODY_CHARS)
    raw_slots = root["slots"]
    if type(raw_slots) is not list or len(raw_slots) > MAX_TEMPLATE_SLOTS:
        _invalid()
    slots = tuple(_parse_slot(item) for item in raw_slots)
    names = tuple(slot.name for slot in slots)
    if len(names) != len(set(names)):
        _invalid()
    if sum(len(slot.choices) for slot in slots) > MAX_TEMPLATE_TOTAL_CHOICES:
        _invalid()
    if _body_tokens(body) != names:
        # The declaration order is presentation order. It therefore has an
        # output identity and must match the tokens placed in the authored body.
        _invalid()
    return PromptTemplateContract(
        schema_version=PROMPT_TEMPLATE_SCHEMA_VERSION,
        operation=PROMPT_TEMPLATE_OPERATION,
        body=body,
        slots=slots,
        resource_policy=_parse_resource_policy(root["resource_policy"]),
    )


def _lora_policy_payload(
    lora_policy: PromptTemplateLoraPolicy,
    *,
    allow_pool: bool,
) -> dict[str, object]:
    if (
        type(lora_policy) is not PromptTemplateLoraPolicy
        or type(lora_policy.mode) is not PromptTemplateLoraPolicyMode
        or type(lora_policy.stack) is not tuple
        or type(lora_policy.strategy) not in {PromptTemplatePoolStrategy, type(None)}
        or type(lora_policy.stacks) is not tuple
    ):
        _invalid()
    lora_payload: dict[str, object] = {"mode": lora_policy.mode.value}
    raw_stacks: tuple[tuple[PromptTemplateLora, ...], ...]
    if lora_policy.mode is PromptTemplateLoraPolicyMode.FIXED:
        if (
            not lora_policy.stack
            or len(lora_policy.stack) > MAX_TEMPLATE_LORAS
            or lora_policy.strategy is not None
            or lora_policy.stacks
        ):
            _invalid()
        raw_stacks = (lora_policy.stack,)
    elif lora_policy.mode is PromptTemplateLoraPolicyMode.POOL:
        if (
            not allow_pool
            or lora_policy.stack
            or lora_policy.strategy is None
            or not 2 <= len(lora_policy.stacks) <= MAX_TEMPLATE_LORA_STACKS
            or any(type(stack) is not tuple for stack in lora_policy.stacks)
            or any(not stack or len(stack) > MAX_TEMPLATE_LORAS for stack in lora_policy.stacks)
            or sum(len(stack) for stack in lora_policy.stacks) > MAX_TEMPLATE_POOL_LORAS
            or len(set(lora_policy.stacks)) != len(lora_policy.stacks)
        ):
            _invalid()
        raw_stacks = lora_policy.stacks
    else:
        if lora_policy.stack or lora_policy.strategy is not None or lora_policy.stacks:
            _invalid()
        raw_stacks = ()
    serialized_stacks: list[list[dict[str, object]]] = []
    for raw_stack in raw_stacks:
        stack: list[dict[str, object]] = []
        seen: set[str] = set()
        for lora in raw_stack:
            if (
                type(lora) is not PromptTemplateLora
                or type(lora.sha256) is not str
                or _SHA256.fullmatch(lora.sha256) is None
                or lora.sha256 in seen
                or type(lora.model_strength) is not float
                or type(lora.clip_strength) is not float
            ):
                _invalid()
            _strength(lora.model_strength)
            _strength(lora.clip_strength)
            seen.add(lora.sha256)
            stack.append(
                {
                    "sha256": lora.sha256,
                    "model_strength": lora.model_strength,
                    "clip_strength": lora.clip_strength,
                }
            )
        serialized_stacks.append(stack)
    if lora_policy.mode is PromptTemplateLoraPolicyMode.FIXED:
        lora_payload["stack"] = serialized_stacks[0]
    elif lora_policy.mode is PromptTemplateLoraPolicyMode.POOL:
        if lora_policy.strategy is None:
            _invalid()
        lora_payload.update(
            {
                "strategy": lora_policy.strategy.value,
                "stacks": serialized_stacks,
            }
        )
    return lora_payload


def prompt_template_contract_payload(
    contract: PromptTemplateContract,
) -> dict[str, object]:
    """Return the unique JSON-shaped representation of a validated contract."""

    if (
        type(contract) is not PromptTemplateContract
        or type(contract.schema_version) is not int
        or type(contract.operation) is not str
        or type(contract.body) is not str
        or type(contract.slots) is not tuple
    ):
        _invalid()
    slots: list[dict[str, object]] = []
    for slot in contract.slots:
        if (
            type(slot) is not PromptTemplateSlot
            or type(slot.name) is not str
            or type(slot.mode) is not PromptTemplateSlotMode
            or type(slot.variation_scope) is not PromptTemplateVariationScope
            or type(slot.choices) is not tuple
            or type(slot.choice_strategy) is not PromptTemplateChoiceStrategy
            or any(type(value) is not str for value in slot.choices)
            or type(slot.guidance) not in {str, type(None)}
            or type(slot.fixed_value) not in {str, type(None)}
        ):
            _invalid()
        item: dict[str, object] = {
            "name": slot.name,
            "mode": slot.mode.value,
            "variation_scope": slot.variation_scope.value,
        }
        if slot.mode is PromptTemplateSlotMode.CHOICE:
            item["choices"] = list(slot.choices)
            if slot.choice_strategy is not PromptTemplateChoiceStrategy.DISTINCT:
                item["choice_strategy"] = slot.choice_strategy.value
        elif slot.mode is PromptTemplateSlotMode.MODEL:
            item["guidance"] = slot.guidance
        elif slot.mode is PromptTemplateSlotMode.FIXED:
            item["fixed_value"] = slot.fixed_value
        slots.append(item)
    resource = contract.resource_policy
    if (
        type(resource) is not PromptTemplateResourcePolicy
        or type(resource.mode) is not PromptTemplateResourceMode
        or type(resource.workflow_revision_id) not in {str, type(None)}
        or type(resource.lora_policy) not in {PromptTemplateLoraPolicy, type(None)}
        or type(resource.strategy) not in {PromptTemplatePoolStrategy, type(None)}
        or type(resource.options) is not tuple
    ):
        _invalid()
    resource_payload: dict[str, object] = {"mode": resource.mode.value}
    if resource.mode is PromptTemplateResourceMode.FIXED:
        if (
            type(resource.workflow_revision_id) is not str
            or _SAFE_ID.fullmatch(resource.workflow_revision_id) is None
            or type(resource.lora_policy) is not PromptTemplateLoraPolicy
            or resource.strategy is not None
            or resource.options
        ):
            _invalid()
        resource_payload.update(
            {
                "workflow_revision_id": resource.workflow_revision_id,
                "lora_policy": _lora_policy_payload(resource.lora_policy, allow_pool=True),
            }
        )
    elif resource.mode is PromptTemplateResourceMode.POOL:
        if (
            resource.workflow_revision_id is not None
            or resource.lora_policy is not None
            or resource.strategy is None
            or not 2 <= len(resource.options) <= MAX_TEMPLATE_WORKFLOW_OPTIONS
            or any(type(option) is not PromptTemplateResourceOption for option in resource.options)
            or len(set(resource.options)) != len(resource.options)
            or sum(
                len(option.lora_policy.stack)
                for option in resource.options
                if option.lora_policy.mode is PromptTemplateLoraPolicyMode.FIXED
            )
            > MAX_TEMPLATE_POOL_LORAS
        ):
            _invalid()
        options: list[dict[str, object]] = []
        for option in resource.options:
            if (
                type(option.workflow_revision_id) is not str
                or _SAFE_ID.fullmatch(option.workflow_revision_id) is None
                or type(option.lora_policy) is not PromptTemplateLoraPolicy
                or option.lora_policy.mode is PromptTemplateLoraPolicyMode.POOL
            ):
                _invalid()
            options.append(
                {
                    "workflow_revision_id": option.workflow_revision_id,
                    "lora_policy": _lora_policy_payload(option.lora_policy, allow_pool=False),
                }
            )
        resource_payload.update(
            {
                "strategy": resource.strategy.value,
                "options": options,
            }
        )
    elif (
        resource.workflow_revision_id is not None
        or resource.lora_policy is not None
        or resource.strategy is not None
        or resource.options
    ):
        _invalid()
    payload: dict[str, object] = {
        "schema_version": contract.schema_version,
        "operation": contract.operation,
        "body": contract.body,
        "slots": slots,
        "resource_policy": resource_payload,
    }
    # Reparse so a hand-constructed dataclass cannot bypass the wire contract.
    normalized = parse_prompt_template_contract(payload)
    if normalized != contract:
        _invalid()
    return payload


def prompt_template_contract_sha256(contract: PromptTemplateContract) -> str:
    payload = prompt_template_contract_payload(contract)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_prompt_template(contract: PromptTemplateContract, values: object) -> str:
    """Render exact declared values once; inserted values are never rescanned."""

    payload = prompt_template_contract_payload(contract)
    normalized = parse_prompt_template_contract(payload)
    detached_values = _detach_exact_json(values, values_error=True)
    if type(detached_values) is not dict:
        _invalid_values()
    expected = {
        slot.name for slot in normalized.slots if slot.mode is not PromptTemplateSlotMode.FIXED
    }
    if set(detached_values) != expected:
        _invalid_values()
    replacements: dict[str, str] = {}
    for slot in normalized.slots:
        if slot.mode is PromptTemplateSlotMode.FIXED:
            if slot.fixed_value is None:
                _invalid()
            replacements[slot.name] = slot.fixed_value
            continue
        raw = detached_values[slot.name]
        if type(raw) is not str:
            _invalid_values()
        try:
            candidate = _text(raw, maximum=MAX_TEMPLATE_VALUE_CHARS)
        except PromptTemplateError as exc:
            raise PromptTemplateError(
                "prompt-template-values-invalid",
                PROMPT_TEMPLATE_VALUES_INVALID,
            ) from exc
        if slot.mode is PromptTemplateSlotMode.CHOICE and candidate not in slot.choices:
            _invalid_values()
        replacements[slot.name] = candidate

    rendered = _TOKEN.sub(lambda match: replacements[match.group(1)], normalized.body)
    if len(rendered) > MAX_TEMPLATE_RENDERED_CHARS:
        _invalid_values()
    return rendered
