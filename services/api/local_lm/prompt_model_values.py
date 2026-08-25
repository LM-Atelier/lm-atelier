"""Closed, execution-inert values contract for Prompt Library model slots.

The local chat model is allowed to supply text for slots whose authored mode is
``model``.  This module defines that narrow boundary; it does not invoke a
model, read a database, render a prompt, or create media work.

Batch-scoped values are returned once.  Item-scoped values are returned once
per exact ordinal.  Keeping those shapes separate prevents a model response
from silently redefining authored variation scope.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import NoReturn, cast

from .prompt_templates import (
    MAX_TEMPLATE_DOCUMENT_BYTES,
    MAX_TEMPLATE_DOCUMENT_CHARS,
    MAX_TEMPLATE_DOCUMENT_DEPTH,
    MAX_TEMPLATE_DOCUMENT_NODES,
    MAX_TEMPLATE_GUIDANCE_CHARS,
    MAX_TEMPLATE_SLOTS,
    MAX_TEMPLATE_VALUE_CHARS,
    PromptTemplateContract,
    PromptTemplateError,
    PromptTemplateSlotMode,
    PromptTemplateVariationScope,
    parse_prompt_template_contract,
    prompt_template_contract_payload,
)

PROMPT_MODEL_VALUES_VERSION = 1
PROMPT_MODEL_VALUES_TOOL_NAME = "supply_prompt_model_values"
MIN_PROMPT_MODEL_ITEMS = 1
MAX_PROMPT_MODEL_ITEMS = 16
PROMPT_MODEL_VALUES_INVALID = "Prompt model values are invalid."

_RESULT_KEYS = frozenset({"version", "batch_values", "items"})
_ITEM_KEYS = frozenset({"ordinal", "values"})
_SLOT_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}", re.ASCII)


class PromptModelValuesError(ValueError):
    """A fixed, non-echoing refusal at the model-values boundary."""

    def __init__(self, message: str = PROMPT_MODEL_VALUES_INVALID) -> None:
        super().__init__(message)


def _invalid() -> NoReturn:
    raise PromptModelValuesError()


@dataclass(frozen=True, slots=True)
class PromptModelSlotSpec:
    """One exact model-authored slot, with private guidance hidden from repr."""

    name: str
    variation_scope: PromptTemplateVariationScope
    guidance: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class PromptModelSlotContract:
    """The exact slots and item count a single model response must satisfy."""

    version: int
    item_count: int
    batch_slots: tuple[PromptModelSlotSpec, ...]
    item_slots: tuple[PromptModelSlotSpec, ...]


@dataclass(frozen=True, slots=True)
class PromptModelItemValues:
    """Values for one exact item ordinal; value text is omitted from repr."""

    ordinal: int
    values: tuple[tuple[str, str], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class PromptModelValues:
    """One validated tool result; all user/model-authored text stays repr-hidden."""

    version: int
    batch_values: tuple[tuple[str, str], ...] = field(repr=False)
    items: tuple[PromptModelItemValues, ...] = field(repr=False)


def _detach_exact_json(value: object) -> object:
    """Detach an exact built-in JSON tree before any key lookup or comparison."""

    if type(value) not in {dict, list, str, int, float, bool, type(None)}:
        _invalid()

    seen: set[int] = set()
    nodes = 0
    characters = 0
    encoded_bytes = 0

    def scalar(item: object) -> None:
        nonlocal nodes, characters, encoded_bytes
        nodes += 1
        if nodes > MAX_TEMPLATE_DOCUMENT_NODES:
            _invalid()
        if type(item) is str:
            characters += len(item)
            try:
                encoded_bytes += len(item.encode("utf-8"))
            except UnicodeEncodeError:
                _invalid()
            if (
                characters > MAX_TEMPLATE_DOCUMENT_CHARS
                or encoded_bytes > MAX_TEMPLATE_DOCUMENT_BYTES
            ):
                _invalid()

    if type(value) not in {dict, list}:
        scalar(value)
        return value

    root: dict[str, object] | list[object] = {} if type(value) is dict else []
    stack: list[tuple[object, dict[str, object] | list[object], int]] = [(value, root, 0)]
    while stack:
        source, target, depth = stack.pop()
        nodes += 1
        if nodes > MAX_TEMPLATE_DOCUMENT_NODES or depth > MAX_TEMPLATE_DOCUMENT_DEPTH:
            _invalid()
        identity = id(source)
        if identity in seen:
            _invalid()
        seen.add(identity)

        pending: list[tuple[object, dict[str, object] | list[object], int]] = []
        if type(source) is dict:
            if type(target) is not dict or len(source) > MAX_TEMPLATE_DOCUMENT_NODES:
                _invalid()
            for raw_key, child in source.items():
                # Check before assigning to the detached dict. A str subclass
                # can run caller code from __hash__ or __eq__.
                if type(raw_key) is not str or not raw_key or len(raw_key) > 64:
                    _invalid()
                key = raw_key
                scalar(key)
                if type(child) is dict:
                    copied: dict[str, object] = {}
                    target[key] = copied
                    pending.append((child, copied, depth + 1))
                elif type(child) is list:
                    copied_list: list[object] = []
                    target[key] = copied_list
                    pending.append((child, copied_list, depth + 1))
                elif type(child) in {str, int, float, bool, type(None)}:
                    scalar(child)
                    target[key] = child
                else:
                    _invalid()
        else:
            if type(source) is not list or type(target) is not list:
                _invalid()
            if len(source) > MAX_TEMPLATE_DOCUMENT_NODES:
                _invalid()
            for child in source:
                if type(child) is dict:
                    copied_dict: dict[str, object] = {}
                    target.append(copied_dict)
                    pending.append((child, copied_dict, depth + 1))
                elif type(child) is list:
                    copied_list = []
                    target.append(copied_list)
                    pending.append((child, copied_list, depth + 1))
                elif type(child) in {str, int, float, bool, type(None)}:
                    scalar(child)
                    target.append(child)
                else:
                    _invalid()
        stack.extend(reversed(pending))
    return root


def _exact_keys(value: object, expected: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict:
        _invalid()
    mapping = cast(dict[str, object], value)
    if len(mapping) != len(expected) or any(type(key) is not str for key in mapping):
        _invalid()
    if frozenset(mapping) != expected:
        _invalid()
    return mapping


def _value(value: object) -> str:
    if type(value) is not str or not value or len(value) > MAX_TEMPLATE_VALUE_CHARS:
        _invalid()
    text = value
    if not text.strip():
        _invalid()
    if any(
        (ord(character) < 32 and character not in {"\n", "\t"}) or 127 <= ord(character) <= 159
        for character in text
    ):
        _invalid()
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        _invalid()
    return text


def prompt_model_slot_contract(
    template: PromptTemplateContract,
    *,
    item_count: object,
) -> PromptModelSlotContract:
    """Derive the only model-slot shape allowed by one validated template."""

    if (
        type(item_count) is not int
        or not MIN_PROMPT_MODEL_ITEMS <= item_count <= MAX_PROMPT_MODEL_ITEMS
    ):
        _invalid()
    # This validates every nested dataclass field and prevents a hand-built or
    # object.__setattr__-mutated contract from becoming authority.
    try:
        normalized_template = parse_prompt_template_contract(
            prompt_template_contract_payload(template)
        )
    except PromptTemplateError:
        _invalid()
    batch: list[PromptModelSlotSpec] = []
    items: list[PromptModelSlotSpec] = []
    for slot in normalized_template.slots:
        if slot.mode is not PromptTemplateSlotMode.MODEL:
            continue
        if slot.guidance is None:
            _invalid()
        spec = PromptModelSlotSpec(
            name=slot.name,
            variation_scope=slot.variation_scope,
            guidance=slot.guidance,
        )
        if slot.variation_scope is PromptTemplateVariationScope.BATCH:
            batch.append(spec)
        else:
            items.append(spec)
    if not batch and not items:
        _invalid()
    return PromptModelSlotContract(
        version=PROMPT_MODEL_VALUES_VERSION,
        item_count=item_count,
        batch_slots=tuple(batch),
        item_slots=tuple(items),
    )


def _valid_contract(contract: PromptModelSlotContract) -> bool:
    if (
        type(contract) is not PromptModelSlotContract
        or type(contract.version) is not int
        or contract.version != PROMPT_MODEL_VALUES_VERSION
        or type(contract.item_count) is not int
        or not MIN_PROMPT_MODEL_ITEMS <= contract.item_count <= MAX_PROMPT_MODEL_ITEMS
        or type(contract.batch_slots) is not tuple
        or type(contract.item_slots) is not tuple
        or not contract.batch_slots + contract.item_slots
        or len(contract.batch_slots) + len(contract.item_slots) > MAX_TEMPLATE_SLOTS
    ):
        return False
    seen: set[str] = set()
    guidance_characters = 0
    guidance_bytes = 0
    for expected_scope, slots in (
        (PromptTemplateVariationScope.BATCH, contract.batch_slots),
        (PromptTemplateVariationScope.ITEM, contract.item_slots),
    ):
        for slot in slots:
            if (
                type(slot) is not PromptModelSlotSpec
                or type(slot.name) is not str
                or _SLOT_NAME.fullmatch(slot.name) is None
                or type(slot.variation_scope) is not PromptTemplateVariationScope
                or slot.variation_scope is not expected_scope
                or type(slot.guidance) is not str
                or not slot.guidance
                or len(slot.guidance) > MAX_TEMPLATE_GUIDANCE_CHARS
                or any(
                    (ord(character) < 32 and character not in {"\n", "\t"})
                    or 127 <= ord(character) <= 159
                    for character in slot.guidance
                )
                or slot.name in seen
            ):
                return False
            try:
                encoded_guidance = slot.guidance.encode("utf-8")
            except UnicodeEncodeError:
                return False
            guidance_characters += len(slot.guidance)
            guidance_bytes += len(encoded_guidance)
            if (
                guidance_characters > MAX_TEMPLATE_DOCUMENT_CHARS
                or guidance_bytes > MAX_TEMPLATE_DOCUMENT_BYTES
            ):
                return False
            seen.add(slot.name)
    return True


def prompt_model_values_tool(contract: PromptModelSlotContract) -> dict[str, object]:
    """Build the closed structured-tool schema for one exact slot contract."""

    if not _valid_contract(contract):
        _invalid()

    def properties(slots: tuple[PromptModelSlotSpec, ...]) -> dict[str, object]:
        return {
            slot.name: {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TEMPLATE_VALUE_CHARS,
                "description": slot.guidance,
            }
            for slot in slots
        }

    batch_names = [slot.name for slot in contract.batch_slots]
    item_names = [slot.name for slot in contract.item_slots]
    value_object: dict[str, object] = {
        "type": "object",
        "properties": properties(contract.item_slots),
        "required": item_names,
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "function": {
            "name": PROMPT_MODEL_VALUES_TOOL_NAME,
            "description": (
                "Supply only the requested Prompt Library slot values. "
                "Do not choose workflows, models, settings, counts, or execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "version": {"type": "integer", "const": PROMPT_MODEL_VALUES_VERSION},
                    "batch_values": {
                        "type": "object",
                        "properties": properties(contract.batch_slots),
                        "required": batch_names,
                        "additionalProperties": False,
                    },
                    "items": {
                        "type": "array",
                        "minItems": contract.item_count,
                        "maxItems": contract.item_count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "ordinal": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": contract.item_count,
                                },
                                "values": value_object,
                            },
                            "required": ["ordinal", "values"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["version", "batch_values", "items"],
                "additionalProperties": False,
            },
        },
    }


def _parse_values(
    value: object, expected: tuple[PromptModelSlotSpec, ...]
) -> tuple[tuple[str, str], ...]:
    if type(value) is not dict:
        _invalid()
    mapping = cast(dict[str, object], value)
    names = tuple(slot.name for slot in expected)
    if len(mapping) != len(names) or any(type(key) is not str for key in mapping):
        _invalid()
    if frozenset(mapping) != frozenset(names):
        _invalid()
    return tuple((name, _value(mapping[name])) for name in names)


def parse_prompt_model_values(
    value: object,
    *,
    contract: PromptModelSlotContract,
) -> PromptModelValues:
    """Validate and detach one exact model tool payload."""

    if not _valid_contract(contract):
        _invalid()
    detached = _detach_exact_json(value)
    root = _exact_keys(detached, _RESULT_KEYS)
    if type(root["version"]) is not int or root["version"] != PROMPT_MODEL_VALUES_VERSION:
        _invalid()
    batch_values = _parse_values(root["batch_values"], contract.batch_slots)
    raw_items = root["items"]
    if type(raw_items) is not list or len(raw_items) != contract.item_count:
        _invalid()
    items: list[PromptModelItemValues] = []
    for expected_ordinal, raw_item in enumerate(raw_items, start=1):
        item = _exact_keys(raw_item, _ITEM_KEYS)
        if type(item["ordinal"]) is not int or item["ordinal"] != expected_ordinal:
            _invalid()
        items.append(
            PromptModelItemValues(
                ordinal=expected_ordinal,
                values=_parse_values(item["values"], contract.item_slots),
            )
        )
    return PromptModelValues(
        version=PROMPT_MODEL_VALUES_VERSION,
        batch_values=batch_values,
        items=tuple(items),
    )


def parse_prompt_model_values_json(
    value: object,
    *,
    contract: PromptModelSlotContract,
) -> PromptModelValues:
    """Parse exact raw tool arguments without losing duplicate object keys."""

    if type(value) is not str or not value:
        _invalid()
    raw = value
    if len(raw) > MAX_TEMPLATE_DOCUMENT_CHARS:
        _invalid()
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError:
        _invalid()
    if len(encoded) > MAX_TEMPLATE_DOCUMENT_BYTES:
        _invalid()

    def reject_constant(_token: str) -> NoReturn:
        _invalid()

    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if type(key) is not str or key in result:
                _invalid()
            result[key] = item
        return result

    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=exact_object,
            parse_constant=reject_constant,
        )
    except PromptModelValuesError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError, OverflowError):
        _invalid()
    return parse_prompt_model_values(decoded, contract=contract)


def prompt_model_values_payload(
    values: PromptModelValues,
    *,
    contract: PromptModelSlotContract,
) -> dict[str, object]:
    """Return the unique public JSON representation of validated model values."""

    if (
        type(values) is not PromptModelValues
        or type(values.version) is not int
        or type(values.batch_values) is not tuple
        or type(values.items) is not tuple
    ):
        _invalid()

    def pairs_payload(raw: object) -> dict[str, str]:
        if type(raw) is not tuple:
            _invalid()
        result: dict[str, str] = {}
        for pair in raw:
            if type(pair) is not tuple or len(pair) != 2:
                _invalid()
            name, text = pair
            if type(name) is not str or name in result:
                _invalid()
            result[name] = _value(text)
        return result

    item_payloads: list[dict[str, object]] = []
    for item in values.items:
        if (
            type(item) is not PromptModelItemValues
            or type(item.ordinal) is not int
            or type(item.values) is not tuple
        ):
            _invalid()
        item_payloads.append({"ordinal": item.ordinal, "values": pairs_payload(item.values)})
    candidate: dict[str, object] = {
        "version": values.version,
        "batch_values": pairs_payload(values.batch_values),
        "items": item_payloads,
    }
    normalized = parse_prompt_model_values(candidate, contract=contract)
    if normalized != values:
        _invalid()
    return candidate


def prompt_model_values_sha256(
    values: PromptModelValues,
    *,
    contract: PromptModelSlotContract,
) -> str:
    """Hash the exact accepted values without exposing their text."""

    payload = prompt_model_values_payload(values, contract=contract)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
