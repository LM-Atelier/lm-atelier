from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from local_lm.prompt_templates import (
    MAX_LORA_STRENGTH,
    MAX_TEMPLATE_CHOICES,
    MAX_TEMPLATE_DOCUMENT_DEPTH,
    MAX_TEMPLATE_LORA_STACKS,
    MAX_TEMPLATE_LORAS,
    MAX_TEMPLATE_POOL_LORAS,
    MAX_TEMPLATE_SLOTS,
    MAX_TEMPLATE_TOTAL_CHOICES,
    PromptTemplateContract,
    PromptTemplateError,
    parse_prompt_template_contract,
    prompt_template_contract_payload,
    prompt_template_contract_sha256,
    render_prompt_template,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "text_to_image",
        "body": "{{subject}}, {{style}}, {{setting}}, {{signature}}",
        "slots": [
            {"name": "subject", "mode": "input", "variation_scope": "item"},
            {
                "name": "style",
                "mode": "choice",
                "variation_scope": "batch",
                "choices": ["editorial photo", "oil painting"],
            },
            {
                "name": "setting",
                "mode": "model",
                "variation_scope": "item",
                "guidance": "Name one concrete setting that supports the subject.",
            },
            {
                "name": "signature",
                "mode": "fixed",
                "variation_scope": "batch",
                "fixed_value": "soft natural light",
            },
        ],
        "resource_policy": {"mode": "inherited"},
    }


def _fixed_resources() -> dict[str, object]:
    payload = _payload()
    payload["resource_policy"] = {
        "mode": "fixed",
        "workflow_revision_id": "revision-123",
        "lora_policy": {
            "mode": "fixed",
            "stack": [
                {
                    "sha256": SHA_A,
                    "model_strength": 0.8,
                    "clip_strength": 0.6,
                },
                {
                    "sha256": SHA_B,
                    "model_strength": -0.0,
                    "clip_strength": 1,
                },
            ],
        },
    }
    return payload


def _pooled_resources(*, strategy: str = "round_robin") -> dict[str, object]:
    payload = _payload()
    payload["resource_policy"] = {
        "mode": "fixed",
        "workflow_revision_id": "revision-123",
        "lora_policy": {
            "mode": "pool",
            "strategy": strategy,
            "stacks": [
                [{"sha256": SHA_A, "model_strength": 0.8, "clip_strength": 0.6}],
                [{"sha256": SHA_B, "model_strength": 1.0, "clip_strength": 0.7}],
            ],
        },
    }
    return payload


def _assert_invalid(value: object) -> None:
    with pytest.raises(PromptTemplateError) as raised:
        parse_prompt_template_contract(value)
    assert raised.value.code == "prompt-template-invalid"
    assert raised.value.status_code == 422
    assert str(raised.value) == "Prompt template contract is invalid."


def _assert_values_invalid(contract: PromptTemplateContract, value: object) -> None:
    with pytest.raises(PromptTemplateError) as raised:
        render_prompt_template(contract, value)
    assert raised.value.code == "prompt-template-values-invalid"
    assert raised.value.status_code == 422
    assert str(raised.value) == "Prompt template values are invalid."


def test_all_slot_modes_parse_and_render_in_one_pass() -> None:
    source = _payload()
    before = copy.deepcopy(source)
    contract = parse_prompt_template_contract(source)
    values = {
        "subject": "a red fox",
        "style": "editorial photo",
        "setting": "a snowy city roof with {{signature}} on a sign",
    }
    values_before = copy.deepcopy(values)

    assert render_prompt_template(contract, values) == (
        "a red fox, editorial photo, "
        "a snowy city roof with {{signature}} on a sign, soft natural light"
    )
    assert source == before
    assert values == values_before
    assert tuple(slot.mode.value for slot in contract.slots) == (
        "input",
        "choice",
        "model",
        "fixed",
    )


def test_zero_slot_literal_template_is_valid() -> None:
    contract = parse_prompt_template_contract(
        {
            "schema_version": 1,
            "operation": "text_to_image",
            "body": "a complete authored prompt",
            "slots": [],
            "resource_policy": {"mode": "inherited"},
        }
    )
    assert render_prompt_template(contract, {}) == "a complete authored prompt"


def test_fixed_resources_are_portable_and_canonical() -> None:
    contract = parse_prompt_template_contract(_fixed_resources())
    payload = prompt_template_contract_payload(contract)
    policy = payload["resource_policy"]
    assert isinstance(policy, dict)
    assert policy["workflow_revision_id"] == "revision-123"
    lora_policy = policy["lora_policy"]
    assert isinstance(lora_policy, dict)
    stack = lora_policy["stack"]
    assert isinstance(stack, list)
    assert stack[1]["model_strength"] == 0.0
    assert stack[1]["clip_strength"] == 1.0
    rendered = repr(contract) + repr(payload)
    assert "node_id" not in rendered
    assert "entry_locator" not in rendered
    assert chr(92) not in rendered and chr(47) not in rendered


@pytest.mark.parametrize("mode", ["inherited_auto", "none"])
def test_fixed_workflow_supports_non_stack_lora_policies(mode: str) -> None:
    payload = _payload()
    payload["resource_policy"] = {
        "mode": "fixed",
        "workflow_revision_id": "revision_123",
        "lora_policy": {"mode": mode},
    }
    contract = parse_prompt_template_contract(payload)
    assert contract.resource_policy.lora_policy is not None
    assert contract.resource_policy.lora_policy.mode.value == mode


def test_object_key_order_does_not_change_digest() -> None:
    first = _fixed_resources()
    second = {
        "resource_policy": first["resource_policy"],
        "slots": first["slots"],
        "body": first["body"],
        "operation": first["operation"],
        "schema_version": first["schema_version"],
    }
    assert prompt_template_contract_sha256(
        parse_prompt_template_contract(first)
    ) == prompt_template_contract_sha256(parse_prompt_template_contract(second))


def _reverse_slots(payload: dict[str, object]) -> None:
    slots = payload["slots"]
    assert isinstance(slots, list)
    slots.reverse()
    payload["body"] = ", ".join("{{" + str(item["name"]) + "}}" for item in slots)


def _reverse_choices(payload: dict[str, object]) -> None:
    slots = payload["slots"]
    assert isinstance(slots, list)
    choice = slots[1]
    assert isinstance(choice, dict)
    choices = choice["choices"]
    assert isinstance(choices, list)
    choices.reverse()


def _reverse_loras(payload: dict[str, object]) -> None:
    resource = payload["resource_policy"]
    assert isinstance(resource, dict)
    lora_policy = resource["lora_policy"]
    assert isinstance(lora_policy, dict)
    stack = lora_policy["stack"]
    assert isinstance(stack, list)
    stack.reverse()


@pytest.mark.parametrize(
    "mutator",
    [
        _reverse_slots,
        _reverse_choices,
        _reverse_loras,
        lambda payload: payload.__setitem__(
            "body", "{{subject}} / {{style}} / {{setting}} / {{signature}}"
        ),
        lambda payload: payload["resource_policy"].__setitem__(
            "workflow_revision_id", "revision-456"
        ),
    ],
)
def test_every_output_affecting_order_or_value_changes_digest(
    mutator: Callable[[dict[str, object]], None],
) -> None:
    first = _fixed_resources()
    second = copy.deepcopy(first)
    mutator(second)
    assert prompt_template_contract_sha256(
        parse_prompt_template_contract(first)
    ) != prompt_template_contract_sha256(parse_prompt_template_contract(second))


def test_lora_stack_pool_round_trips_canonically_and_affects_digest() -> None:
    payload = _pooled_resources()
    contract = parse_prompt_template_contract(payload)
    assert prompt_template_contract_payload(contract) == payload

    random_pool = _pooled_resources(strategy="random")
    assert prompt_template_contract_sha256(contract) != prompt_template_contract_sha256(
        parse_prompt_template_contract(random_pool)
    )
    reversed_pool = _pooled_resources()
    resource = cast(dict[str, object], reversed_pool["resource_policy"])
    policy = cast(dict[str, object], resource["lora_policy"])
    stacks = cast(list[object], policy["stacks"])
    stacks.reverse()
    assert prompt_template_contract_sha256(contract) != prompt_template_contract_sha256(
        parse_prompt_template_contract(reversed_pool)
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda policy: policy.__setitem__("strategy", "weighted"),
        lambda policy: policy.__setitem__("stacks", []),
        lambda policy: policy.__setitem__(
            "stacks",
            [[{"sha256": SHA_A, "model_strength": 1.0, "clip_strength": 1.0}]],
        ),
        lambda policy: policy.__setitem__(
            "stacks",
            [
                [{"sha256": SHA_A, "model_strength": 1.0, "clip_strength": 1.0}],
                [{"sha256": SHA_A, "model_strength": 1.0, "clip_strength": 1.0}],
            ],
        ),
        lambda policy: policy.__setitem__("extra", True),
    ],
)
def test_lora_stack_pool_refuses_ambiguous_or_open_shapes(
    mutation: Callable[[dict[str, object]], None],
) -> None:
    payload = _pooled_resources()
    resource = cast(dict[str, object], payload["resource_policy"])
    policy = cast(dict[str, object], resource["lora_policy"])
    mutation(policy)
    _assert_invalid(payload)


def test_lora_stack_pool_enforces_stack_and_total_bounds() -> None:
    too_many_stacks = _pooled_resources()
    resource = cast(dict[str, object], too_many_stacks["resource_policy"])
    policy = cast(dict[str, object], resource["lora_policy"])
    policy["stacks"] = [
        [{"sha256": f"{index + 1:064x}", "model_strength": 1.0, "clip_strength": 1.0}]
        for index in range(MAX_TEMPLATE_LORA_STACKS + 1)
    ]
    _assert_invalid(too_many_stacks)

    total_overflow = _pooled_resources()
    resource = cast(dict[str, object], total_overflow["resource_policy"])
    policy = cast(dict[str, object], resource["lora_policy"])
    policy["stacks"] = [
        [
            {
                "sha256": f"{stack_index * MAX_TEMPLATE_LORAS + index + 1:064x}",
                "model_strength": 1.0,
                "clip_strength": 1.0,
            }
            for index in range(MAX_TEMPLATE_LORAS)
        ]
        for stack_index in range(MAX_TEMPLATE_POOL_LORAS // MAX_TEMPLATE_LORAS + 1)
    ]
    _assert_invalid(total_overflow)


@pytest.mark.parametrize(
    "body",
    [
        "{{subject}}, {{style}}, {{setting}}",
        "{{subject}}, {{style}}, {{setting}}, {{signature}}, {{unknown}}",
        "{{subject}}, {{style}}, {{setting}}, {{signature}}, {{subject}}",
        "{{Subject}}, {{style}}, {{setting}}, {{signature}}",
        "{{{subject}}}, {{style}}, {{setting}}, {{signature}}",
        "{{subject, {{style}}, {{setting}}, {{signature}}",
        "{{subject}}, {{style}}, {{setting}}, {{signature}} }",
    ],
)
def test_body_and_declared_slots_must_match_exactly(body: str) -> None:
    payload = _payload()
    payload["body"] = body
    _assert_invalid(payload)


@pytest.mark.parametrize(
    "slot",
    [
        {"name": "UPPER", "mode": "input", "variation_scope": "item"},
        {"name": "two words", "mode": "input", "variation_scope": "item"},
        {"name": "path/name", "mode": "input", "variation_scope": "item"},
        {"name": "name\x00", "mode": "input", "variation_scope": "item"},
        {"name": "subject", "mode": "unknown", "variation_scope": "item"},
        {"name": "subject", "mode": "input", "variation_scope": "unknown"},
        {
            "name": "subject",
            "mode": "input",
            "variation_scope": "item",
            "guidance": "not allowed",
        },
        {
            "name": "subject",
            "mode": "fixed",
            "variation_scope": "item",
            "fixed_value": "x",
        },
    ],
)
def test_slot_shapes_are_closed(slot: dict[str, object]) -> None:
    payload = _payload()
    payload["slots"] = [slot]
    payload["body"] = "{{subject}}"
    _assert_invalid(payload)


def test_duplicate_slots_and_choice_values_are_refused() -> None:
    payload = _payload()
    slots = payload["slots"]
    assert isinstance(slots, list)
    slots.append(copy.deepcopy(slots[0]))
    payload["body"] = "{{subject}}, {{style}}, {{setting}}, {{signature}}, {{subject}}"
    _assert_invalid(payload)

    payload = _payload()
    slots = payload["slots"]
    assert isinstance(slots, list)
    choice = slots[1]
    assert isinstance(choice, dict)
    choice["choices"] = ["same", "same"]
    _assert_invalid(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        {"extra": "x"},
        {"schema_version": True},
        {"schema_version": 2},
        {"operation": "image_to_image"},
        {"body": ""},
        {"body": "bad\x00"},
        {"slots": {}},
        {"resource_policy": []},
    ],
)
def test_root_shape_and_scalar_types_are_exact(
    mutation: dict[str, object],
) -> None:
    payload = _payload()
    payload.update(mutation)
    _assert_invalid(payload)


def test_every_nesting_rejects_unknown_keys_without_echo() -> None:
    mutations = []
    root = _payload()
    root["C:\\Users\\secret"] = "x"
    mutations.append(root)

    slot = _payload()
    slots = slot["slots"]
    assert isinstance(slots, list) and isinstance(slots[0], dict)
    slots[0]["/private/path"] = "x"
    mutations.append(slot)

    resource = _fixed_resources()
    resource_policy = resource["resource_policy"]
    assert isinstance(resource_policy, dict)
    resource_policy["..\\secret"] = "x"
    mutations.append(resource)

    policy = _fixed_resources()
    resource_policy = policy["resource_policy"]
    assert isinstance(resource_policy, dict)
    lora_policy = resource_policy["lora_policy"]
    assert isinstance(lora_policy, dict)
    lora_policy["/private"] = "x"
    mutations.append(policy)

    lora = _fixed_resources()
    resource_policy = lora["resource_policy"]
    assert isinstance(resource_policy, dict)
    lora_policy = resource_policy["lora_policy"]
    assert isinstance(lora_policy, dict)
    stack = lora_policy["stack"]
    assert isinstance(stack, list) and isinstance(stack[0], dict)
    stack[0]["secret"] = "x"
    mutations.append(lora)

    for payload in mutations:
        with pytest.raises(PromptTemplateError) as raised:
            parse_prompt_template_contract(payload)
        assert str(raised.value) == "Prompt template contract is invalid."
        assert "secret" not in str(raised.value)


class _Dict(dict[str, object]):
    pass


class _List(list[object]):
    pass


class _Text(str):
    pass


@pytest.mark.parametrize(
    "value",
    [
        _Dict(_payload()),
        {
            **_payload(),
            "slots": _List(cast(list[object], _payload()["slots"])),
        },
        {**_payload(), "body": _Text("{{subject}}")},
    ],
)
def test_subclassed_containers_and_scalars_fail_closed(value: object) -> None:
    _assert_invalid(value)


def test_cycles_and_shared_containers_fail_closed() -> None:
    cyclic = _payload()
    cyclic["extra"] = cyclic
    _assert_invalid(cyclic)

    shared: list[object] = []
    payload = _payload()
    payload["extra_a"] = shared
    payload["extra_b"] = shared
    _assert_invalid(payload)


def test_depth_and_structural_caps_fail_closed() -> None:
    deep: object = "leaf"
    for _ in range(MAX_TEMPLATE_DOCUMENT_DEPTH + 1):
        deep = [deep]
    _assert_invalid(deep)

    payload = _payload()
    payload["slots"] = [
        {
            "name": f"slot_{index}",
            "mode": "input",
            "variation_scope": "item",
        }
        for index in range(MAX_TEMPLATE_SLOTS + 1)
    ]
    payload["body"] = " ".join(f"{{{{slot_{index}}}}}" for index in range(MAX_TEMPLATE_SLOTS + 1))
    _assert_invalid(payload)

    _assert_invalid(["x"] * 4_096)
    _assert_invalid({"extra": "🦊" * 65_537})

    payload = _payload()
    slots = payload["slots"]
    assert isinstance(slots, list) and isinstance(slots[1], dict)
    slots[1]["choices"] = [f"value {index}" for index in range(MAX_TEMPLATE_CHOICES + 1)]
    _assert_invalid(payload)

    payload = _fixed_resources()
    resource = payload["resource_policy"]
    assert isinstance(resource, dict)
    lora_policy = resource["lora_policy"]
    assert isinstance(lora_policy, dict)
    lora_policy["stack"] = [
        {
            "sha256": f"{index:064x}",
            "model_strength": 1.0,
            "clip_strength": 1.0,
        }
        for index in range(MAX_TEMPLATE_LORAS + 1)
    ]
    _assert_invalid(payload)


def test_total_choice_ceiling_is_enforced_across_individually_valid_slots() -> None:
    slot_count = MAX_TEMPLATE_TOTAL_CHOICES // MAX_TEMPLATE_CHOICES
    choices = [f"choice {index}" for index in range(MAX_TEMPLATE_CHOICES)]

    def payload_for(count: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "operation": "text_to_image",
            "body": " ".join(f"{{{{choice_{index}}}}}" for index in range(count)),
            "slots": [
                {
                    "name": f"choice_{index}",
                    "mode": "choice",
                    "variation_scope": "item",
                    "choices": list(choices),
                }
                for index in range(count)
            ],
            "resource_policy": {"mode": "inherited"},
        }

    parse_prompt_template_contract(payload_for(slot_count))
    _assert_invalid(payload_for(slot_count + 1))


@pytest.mark.parametrize(
    "value",
    [
        True,
        None,
        "1",
        float("nan"),
        float("inf"),
        -float("inf"),
        4.00001,
        -4.00001,
    ],
)
def test_lora_strengths_are_finite_exact_numbers_in_shared_bounds(
    value: object,
) -> None:
    payload = _fixed_resources()
    resource = payload["resource_policy"]
    assert isinstance(resource, dict)
    policy = resource["lora_policy"]
    assert isinstance(policy, dict)
    stack = policy["stack"]
    assert isinstance(stack, list) and isinstance(stack[0], dict)
    stack[0]["model_strength"] = value
    _assert_invalid(payload)

    valid = _fixed_resources()
    resource = valid["resource_policy"]
    assert isinstance(resource, dict)
    policy = resource["lora_policy"]
    assert isinstance(policy, dict)
    stack = policy["stack"]
    assert isinstance(stack, list) and isinstance(stack[0], dict)
    stack[0]["model_strength"] = MAX_LORA_STRENGTH
    parse_prompt_template_contract(valid)


@pytest.mark.parametrize(
    "resource",
    [
        {"mode": "inherited", "workflow_revision_id": "x"},
        {
            "mode": "fixed",
            "workflow_revision_id": "C:\\private\\workflow",
            "lora_policy": {"mode": "none"},
        },
        {
            "mode": "fixed",
            "workflow_revision_id": "../workflow",
            "lora_policy": {"mode": "none"},
        },
        {
            "mode": "fixed",
            "workflow_revision_id": "revision",
            "lora_policy": {"mode": "fixed", "stack": []},
        },
        {
            "mode": "fixed",
            "workflow_revision_id": "revision",
            "lora_policy": {
                "mode": "fixed",
                "stack": [
                    {
                        "sha256": SHA_A,
                        "model_strength": 1,
                        "clip_strength": 1,
                    },
                    {
                        "sha256": SHA_A,
                        "model_strength": 1,
                        "clip_strength": 1,
                    },
                ],
            },
        },
    ],
)
def test_resource_policies_are_closed_portable_combinations(
    resource: dict[str, object],
) -> None:
    payload = _payload()
    payload["resource_policy"] = resource
    _assert_invalid(payload)


def test_render_requires_all_and_only_non_fixed_values() -> None:
    contract = parse_prompt_template_contract(_payload())
    valid = {
        "subject": "subject",
        "style": "oil painting",
        "setting": "studio",
    }
    _assert_values_invalid(contract, {**valid, "signature": "override"})
    missing = dict(valid)
    del missing["setting"]
    _assert_values_invalid(contract, missing)
    _assert_values_invalid(contract, {**valid, "unknown": "x"})
    _assert_values_invalid(contract, {**valid, "style": "not a choice"})
    _assert_values_invalid(contract, {**valid, "subject": True})
    _assert_values_invalid(contract, {**valid, "subject": "bad\x00"})


def test_hand_constructed_contracts_are_revalidated_before_use() -> None:
    contract = parse_prompt_template_contract(_payload())
    forged = replace(
        contract,
        slots=(
            replace(contract.slots[0], name="not_present"),
            *contract.slots[1:],
        ),
    )
    with pytest.raises(PromptTemplateError) as payload_error:
        prompt_template_contract_payload(forged)
    assert str(payload_error.value) == "Prompt template contract is invalid."
    with pytest.raises(PromptTemplateError):
        render_prompt_template(forged, {})

    wrong_mode = replace(contract.slots[0], mode="input")  # type: ignore[arg-type]
    with pytest.raises(PromptTemplateError):
        prompt_template_contract_payload(replace(contract, slots=(wrong_mode, *contract.slots[1:])))

    with pytest.raises(PromptTemplateError):
        prompt_template_contract_payload(
            replace(contract, slots=list(contract.slots))  # type: ignore[arg-type]
        )

    pooled = parse_prompt_template_contract(_pooled_resources())
    pooled_policy = pooled.resource_policy.lora_policy
    assert pooled_policy is not None
    for forged_policy in (
        replace(pooled_policy, stack=pooled_policy.stacks[0]),
        replace(pooled_policy, strategy=None),
        replace(pooled_policy, stacks=(pooled_policy.stacks[0],)),
    ):
        with pytest.raises(PromptTemplateError):
            prompt_template_contract_payload(
                replace(
                    pooled,
                    resource_policy=replace(
                        pooled.resource_policy,
                        lora_policy=forged_policy,
                    ),
                )
            )


def test_hostile_render_value_trees_fail_with_the_values_contract() -> None:
    contract = parse_prompt_template_contract(_payload())
    cyclic: dict[str, object] = {}
    cyclic["subject"] = cyclic
    _assert_values_invalid(contract, cyclic)
    _assert_values_invalid(contract, _Dict({"subject": "x"}))


def test_unicode_values_are_data_and_controls_are_not() -> None:
    contract = parse_prompt_template_contract(_payload())
    rendered = render_prompt_template(
        contract,
        {
            "subject": "狐 🦊",
            "style": "editorial photo",
            "setting": "東京 at dawn\nwith mist",
        },
    )
    assert "狐 🦊" in rendered
    assert "東京 at dawn\nwith mist" in rendered


def test_payload_round_trip_is_exact_and_frozen() -> None:
    contract = parse_prompt_template_contract(_fixed_resources())
    payload = prompt_template_contract_payload(contract)
    assert parse_prompt_template_contract(payload) == contract
    assert prompt_template_contract_sha256(contract) == (
        prompt_template_contract_sha256(parse_prompt_template_contract(payload))
    )
