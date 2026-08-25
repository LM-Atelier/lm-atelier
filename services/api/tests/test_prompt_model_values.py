from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import pytest

from local_lm.prompt_model_values import (
    MAX_PROMPT_MODEL_ITEMS,
    PROMPT_MODEL_VALUES_INVALID,
    PROMPT_MODEL_VALUES_TOOL_NAME,
    PromptModelItemValues,
    PromptModelSlotContract,
    PromptModelSlotSpec,
    PromptModelValues,
    PromptModelValuesError,
    parse_prompt_model_values,
    parse_prompt_model_values_json,
    prompt_model_slot_contract,
    prompt_model_values_payload,
    prompt_model_values_sha256,
    prompt_model_values_tool,
)
from local_lm.prompt_templates import (
    MAX_TEMPLATE_DOCUMENT_CHARS,
    MAX_TEMPLATE_DOCUMENT_DEPTH,
    MAX_TEMPLATE_DOCUMENT_NODES,
    MAX_TEMPLATE_GUIDANCE_CHARS,
    MAX_TEMPLATE_VALUE_CHARS,
    PromptTemplateContract,
    PromptTemplateVariationScope,
    parse_prompt_template_contract,
)


def _template(*, slots: list[dict[str, object]] | None = None) -> PromptTemplateContract:
    selected = slots or [
        {
            "name": "style",
            "mode": "model",
            "variation_scope": "batch",
            "guidance": "one visual medium",
        },
        {
            "name": "subject",
            "mode": "input",
            "variation_scope": "batch",
        },
        {
            "name": "lighting",
            "mode": "model",
            "variation_scope": "item",
            "guidance": "one lighting treatment",
        },
    ]
    return parse_prompt_template_contract(
        {
            "schema_version": 1,
            "operation": "text_to_image",
            "body": " ".join(f"{{{{{slot['name']}}}}}" for slot in selected),
            "slots": selected,
            "resource_policy": {"mode": "inherited"},
        }
    )


def _contract(*, item_count: int = 2) -> PromptModelSlotContract:
    return prompt_model_slot_contract(_template(), item_count=item_count)


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": 1,
        "batch_values": {"style": "oil paint"},
        "items": [
            {"ordinal": 1, "values": {"lighting": "soft window light"}},
            {"ordinal": 2, "values": {"lighting": "hard rim light"}},
        ],
    }
    payload.update(overrides)
    return payload


def test_derives_only_model_slots_and_preserves_their_exact_scope() -> None:
    contract = _contract()
    assert [(slot.name, slot.variation_scope) for slot in contract.batch_slots] == [
        ("style", PromptTemplateVariationScope.BATCH)
    ]
    assert [(slot.name, slot.variation_scope) for slot in contract.item_slots] == [
        ("lighting", PromptTemplateVariationScope.ITEM)
    ]
    assert "one visual medium" not in repr(contract)
    assert "one lighting treatment" not in repr(contract)


def test_the_full_phase_one_guidance_bound_remains_usable() -> None:
    guidance = "g" * MAX_TEMPLATE_GUIDANCE_CHARS
    template = _template(
        slots=[
            {
                "name": "detail",
                "mode": "model",
                "variation_scope": "item",
                "guidance": guidance,
            }
        ]
    )
    contract = prompt_model_slot_contract(template, item_count=1)
    assert contract.item_slots[0].guidance == guidance
    assert guidance not in repr(contract)


@pytest.mark.parametrize("count", [1, MAX_PROMPT_MODEL_ITEMS])
def test_exact_item_count_bounds_are_accepted(count: int) -> None:
    assert prompt_model_slot_contract(_template(), item_count=count).item_count == count


@pytest.mark.parametrize("count", [0, MAX_PROMPT_MODEL_ITEMS + 1, -1, True, 1.0])
def test_invalid_item_counts_refuse_with_the_fixed_error(count: object) -> None:
    with pytest.raises(PromptModelValuesError) as caught:
        prompt_model_slot_contract(_template(), item_count=count)
    assert str(caught.value) == PROMPT_MODEL_VALUES_INVALID


def test_a_template_without_model_slots_does_not_create_model_authority() -> None:
    template = _template(
        slots=[
            {
                "name": "subject",
                "mode": "input",
                "variation_scope": "batch",
            }
        ]
    )
    with pytest.raises(PromptModelValuesError):
        prompt_model_slot_contract(template, item_count=1)


def test_tool_schema_is_closed_bounded_and_scope_separated() -> None:
    tool = prompt_model_values_tool(_contract())
    assert tool["type"] == "function"
    function = tool["function"]
    assert isinstance(function, dict)
    assert function["name"] == PROMPT_MODEL_VALUES_TOOL_NAME
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["additionalProperties"] is False
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {"version", "batch_values", "items"}
    batch = properties["batch_values"]
    items = properties["items"]
    assert isinstance(batch, dict) and batch["required"] == ["style"]
    assert batch["additionalProperties"] is False
    assert set(batch["properties"]) == {"style"}
    assert isinstance(items, dict)
    assert items["minItems"] == items["maxItems"] == 2
    item_schema = items["items"]
    assert isinstance(item_schema, dict)
    assert item_schema["additionalProperties"] is False
    item_properties = item_schema["properties"]
    assert isinstance(item_properties, dict)
    assert set(item_properties) == {"ordinal", "values"}
    values = item_properties["values"]
    assert isinstance(values, dict) and values["required"] == ["lighting"]
    assert values["additionalProperties"] is False
    assert set(values["properties"]) == {"lighting"}
    assert "subject" not in repr(tool)


def test_tool_schema_carries_no_resource_or_execution_authority() -> None:
    asset_sha = "a" * 64
    workflow_revision_id = "workflow_revision_private"
    template = parse_prompt_template_contract(
        {
            "schema_version": 1,
            "operation": "text_to_image",
            "body": "{{detail}}",
            "slots": [
                {
                    "name": "detail",
                    "mode": "model",
                    "variation_scope": "item",
                    "guidance": "one visible detail",
                }
            ],
            "resource_policy": {
                "mode": "fixed",
                "workflow_revision_id": workflow_revision_id,
                "lora_policy": {
                    "mode": "fixed",
                    "stack": [
                        {
                            "sha256": asset_sha,
                            "model_strength": 1.0,
                            "clip_strength": 1.0,
                        }
                    ],
                },
            },
        }
    )
    encoded = json.dumps(
        prompt_model_values_tool(prompt_model_slot_contract(template, item_count=1))
    )
    assert workflow_revision_id not in encoded
    assert asset_sha not in encoded
    assert "model_strength" not in encoded
    assert "clip_strength" not in encoded


def test_parses_the_exact_batch_and_item_values_without_mutating_source() -> None:
    source = _payload()
    before = repr(source)
    values = parse_prompt_model_values(source, contract=_contract())
    assert values.batch_values == (("style", "oil paint"),)
    assert values.items == (
        PromptModelItemValues(ordinal=1, values=(("lighting", "soft window light"),)),
        PromptModelItemValues(ordinal=2, values=(("lighting", "hard rim light"),)),
    )
    assert repr(source) == before
    assert "oil paint" not in repr(values)
    assert "soft window light" not in repr(values)


def test_batch_only_and_item_only_model_contracts_keep_the_opposite_map_empty() -> None:
    batch_template = _template(
        slots=[
            {
                "name": "style",
                "mode": "model",
                "variation_scope": "batch",
                "guidance": "one style",
            }
        ]
    )
    batch_contract = prompt_model_slot_contract(batch_template, item_count=2)
    batch = parse_prompt_model_values(
        {
            "version": 1,
            "batch_values": {"style": "ink"},
            "items": [{"ordinal": 1, "values": {}}, {"ordinal": 2, "values": {}}],
        },
        contract=batch_contract,
    )
    assert batch.batch_values == (("style", "ink"),)
    assert all(not item.values for item in batch.items)

    item_template = _template(
        slots=[
            {
                "name": "lighting",
                "mode": "model",
                "variation_scope": "item",
                "guidance": "one light",
            }
        ]
    )
    item_contract = prompt_model_slot_contract(item_template, item_count=2)
    item = parse_prompt_model_values(
        {
            "version": 1,
            "batch_values": {},
            "items": [
                {"ordinal": 1, "values": {"lighting": "dawn"}},
                {"ordinal": 2, "values": {"lighting": "dusk"}},
            ],
        },
        contract=item_contract,
    )
    assert not item.batch_values
    assert [entry.values for entry in item.items] == [
        (("lighting", "dawn"),),
        (("lighting", "dusk"),),
    ]


def test_payload_and_digest_are_canonical_stable_and_value_sensitive() -> None:
    contract = _contract()
    first = parse_prompt_model_values(_payload(), contract=contract)
    payload = prompt_model_values_payload(first, contract=contract)
    assert parse_prompt_model_values(payload, contract=contract) == first
    assert prompt_model_values_sha256(first, contract=contract) == prompt_model_values_sha256(
        parse_prompt_model_values(_payload(), contract=contract), contract=contract
    )
    changed = parse_prompt_model_values(
        _payload(
            items=[
                {"ordinal": 1, "values": {"lighting": "soft window light"}},
                {"ordinal": 2, "values": {"lighting": "blue hour"}},
            ]
        ),
        contract=contract,
    )
    assert prompt_model_values_sha256(first, contract=contract) != prompt_model_values_sha256(
        changed, contract=contract
    )


def test_raw_tool_arguments_preserve_duplicate_key_refusal() -> None:
    valid = (
        '{"version":1,"batch_values":{"style":"oil"},"items":['
        '{"ordinal":1,"values":{"lighting":"dawn"}},'
        '{"ordinal":2,"values":{"lighting":"dusk"}}]}'
    )
    assert parse_prompt_model_values_json(valid, contract=_contract()).batch_values == (
        ("style", "oil"),
    )
    hostile = [
        valid.replace('"version":1', '"version":1,"version":1'),
        valid.replace('"style":"oil"', '"style":"oil","style":"ink"'),
        valid.replace('"lighting":"dawn"', '"lighting":"dawn","lighting":"night"'),
        valid.replace('"ordinal":1', '"ordinal":1,"ordinal":1'),
        "not json",
        "NaN",
    ]
    for raw in hostile:
        with pytest.raises(PromptModelValuesError) as caught:
            parse_prompt_model_values_json(raw, contract=_contract())
        assert str(caught.value) == PROMPT_MODEL_VALUES_INVALID


def test_raw_tool_argument_text_is_exact_and_bounded() -> None:
    class Text(str):
        reads = 0

        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            type(self).reads += 1
            return super().encode(encoding, errors)

    hostile = Text("{}")
    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values_json(hostile, contract=_contract())
    assert Text.reads == 0
    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values_json(
            "x" * (MAX_TEMPLATE_DOCUMENT_CHARS + 1), contract=_contract()
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"version": True},
        {"version": 2},
        {"batch_values": {}},
        {"batch_values": {"style": "oil", "extra": "private"}},
        {"items": []},
        {"items": [{"ordinal": 1, "values": {"lighting": "one"}}]},
        {
            "items": [
                {"ordinal": 2, "values": {"lighting": "one"}},
                {"ordinal": 1, "values": {"lighting": "two"}},
            ]
        },
        {
            "items": [
                {"ordinal": 1, "values": {}},
                {"ordinal": 2, "values": {"lighting": "two"}},
            ]
        },
        {
            "items": [
                {"ordinal": 1, "values": {"lighting": "one", "extra": "private"}},
                {"ordinal": 2, "values": {"lighting": "two"}},
            ]
        },
    ],
)
def test_missing_extra_wrong_count_and_wrong_ordinal_payloads_refuse(
    payload: dict[str, object],
) -> None:
    with pytest.raises(PromptModelValuesError) as caught:
        parse_prompt_model_values(_payload(**payload), contract=_contract())
    assert str(caught.value) == PROMPT_MODEL_VALUES_INVALID


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        True,
        1,
        "x\x00y",
        "x\x7fy",
        "x" * (MAX_TEMPLATE_VALUE_CHARS + 1),
        "\ud800",
    ],
)
def test_invalid_model_value_text_refuses_without_echo(value: object) -> None:
    source = _payload(batch_values={"style": value})
    with pytest.raises(PromptModelValuesError) as caught:
        parse_prompt_model_values(source, contract=_contract())
    assert str(caught.value) == PROMPT_MODEL_VALUES_INVALID


def test_unknown_private_fields_refuse_without_echo() -> None:
    secret = "C:\\private\\model-output.txt"
    source = _payload()
    source["private"] = secret
    with pytest.raises(PromptModelValuesError) as caught:
        parse_prompt_model_values(source, contract=_contract())
    assert secret not in str(caught.value)


def test_root_and_nested_subclasses_refuse_without_invoking_them() -> None:
    class Root(dict[object, object]):
        reads = 0

        def items(self) -> Any:
            type(self).reads += 1
            return super().items()

    class Key(str):
        reads = 0

        def __hash__(self) -> int:
            type(self).reads += 1
            return super().__hash__()

    root = Root(_payload())
    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values(root, contract=_contract())
    assert Root.reads == 0

    key = Key("version")
    source = {key: 1, "batch_values": {"style": "oil"}, "items": _payload()["items"]}
    Key.reads = 0
    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values(source, contract=_contract())
    assert Key.reads == 0


def test_cycles_shared_containers_and_structural_budgets_refuse() -> None:
    cycle: dict[str, object] = {}
    cycle["again"] = cycle
    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values(cycle, contract=_contract())

    shared: dict[str, object] = {"lighting": "one"}
    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values(
            _payload(items=[{"ordinal": 1, "values": shared}, {"ordinal": 2, "values": shared}]),
            contract=_contract(),
        )
    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values(
            {
                "version": 1,
                "batch_values": {},
                "items": [],
                "x": "é" * (MAX_TEMPLATE_DOCUMENT_CHARS + 1),
            },
            contract=_contract(),
        )

    deep: object = "leaf"
    for _ in range(MAX_TEMPLATE_DOCUMENT_DEPTH + 2):
        deep = {"x": deep}
    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values(deep, contract=_contract())

    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values(
            _payload(items=[None] * (MAX_TEMPLATE_DOCUMENT_NODES + 1)), contract=_contract()
        )
    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values(
            {
                "version": 1,
                "batch_values": {},
                "items": [],
                "x": "z" * (MAX_TEMPLATE_DOCUMENT_CHARS + 1),
            },
            contract=_contract(),
        )


def test_hand_built_contracts_and_receipts_fail_closed() -> None:
    valid_contract = _contract()
    malformed_contract = replace(
        valid_contract,
        item_slots=cast(tuple[PromptModelSlotSpec, ...], (object(),)),
    )
    with pytest.raises(PromptModelValuesError):
        prompt_model_values_tool(malformed_contract)
    with pytest.raises(PromptModelValuesError):
        parse_prompt_model_values(_payload(), contract=malformed_contract)

    values = parse_prompt_model_values(_payload(), contract=valid_contract)
    object.__setattr__(values.items[0], "values", (("lighting", object()),))
    with pytest.raises(PromptModelValuesError):
        prompt_model_values_payload(values, contract=valid_contract)

    template = _template()
    object.__setattr__(template, "slots", (object(),))
    with pytest.raises(PromptModelValuesError):
        prompt_model_slot_contract(template, item_count=1)


def test_payload_revalidates_frozen_receipts_before_hashing() -> None:
    contract = _contract()
    values = parse_prompt_model_values(_payload(), contract=contract)
    object.__setattr__(values.items[0], "ordinal", 2)

    with pytest.raises(PromptModelValuesError) as caught:
        prompt_model_values_sha256(values, contract=contract)
    assert str(caught.value) == PROMPT_MODEL_VALUES_INVALID


def test_payload_requires_the_exact_root_receipt_type() -> None:
    class ValuesSubclass(PromptModelValues):
        __slots__ = ()

    contract = _contract()
    values = parse_prompt_model_values(_payload(), contract=contract)
    hostile = ValuesSubclass(
        version=values.version,
        batch_values=values.batch_values,
        items=values.items,
    )

    with pytest.raises(PromptModelValuesError) as caught:
        prompt_model_values_payload(hostile, contract=contract)
    assert str(caught.value) == PROMPT_MODEL_VALUES_INVALID


def test_payload_refuses_duplicate_names_in_frozen_pairs() -> None:
    contract = _contract()
    values = parse_prompt_model_values(_payload(), contract=contract)
    object.__setattr__(
        values,
        "batch_values",
        (("style", "oil paint"), ("style", "watercolor")),
    )

    with pytest.raises(PromptModelValuesError) as caught:
        prompt_model_values_sha256(values, contract=contract)
    assert str(caught.value) == PROMPT_MODEL_VALUES_INVALID


def test_scope_swaps_in_a_hand_built_contract_refuse() -> None:
    contract = _contract()
    swapped = PromptModelSlotContract(
        version=contract.version,
        item_count=contract.item_count,
        batch_slots=(
            PromptModelSlotSpec(
                name="style",
                variation_scope=PromptTemplateVariationScope.ITEM,
                guidance="private guidance",
            ),
        ),
        item_slots=contract.item_slots,
    )
    with pytest.raises(PromptModelValuesError):
        prompt_model_values_tool(swapped)
