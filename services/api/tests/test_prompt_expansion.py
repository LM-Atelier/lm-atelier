from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import pytest

from local_lm.prompt_expansion import (
    MAX_EXPANSION_INPUT_SLOTS,
    MAX_EXPANSION_ITEMS,
    MAX_SEED_MATERIAL_BYTES,
    MAX_SEED_MATERIAL_CHARS,
    MAX_SEED_MATERIAL_ENTRIES,
    PROMPT_EXPANSION_INVALID,
    SELECTION_SEED_SPACE,
    ExpansionPlan,
    ExpansionRequest,
    ExpansionValueSource,
    ModelSlotRequest,
    PromptExpansionDistinctCapacityError,
    PromptExpansionError,
    SlotEvidence,
    complete_prompt_expansion_with_model_values,
    expand_prompt_template,
    expansion_plan_digest,
    expansion_plan_payload,
    expansion_plan_payload_digest,
    expansion_request_payload,
    expansion_selection_seed,
    parse_expansion_plan_payload,
    parse_expansion_request,
    prompt_model_invocation_data,
)
from local_lm.prompt_model_values import (
    PromptModelValues,
    parse_prompt_model_values,
    prompt_model_slot_contract,
    prompt_model_values_payload,
)
from local_lm.prompt_templates import (
    MAX_TEMPLATE_CHOICES,
    PromptTemplateContract,
    PromptTemplateError,
    PromptTemplateResourceMode,
    PromptTemplateResourcePolicy,
    PromptTemplateSlot,
    PromptTemplateSlotMode,
    PromptTemplateVariationScope,
    parse_prompt_template_contract,
    prompt_template_contract_sha256,
)

_BODY = "{{style}} portrait of {{subject}} in {{mood}}."
_SLOTS: list[dict[str, object]] = [
    {"name": "style", "mode": "fixed", "variation_scope": "batch", "fixed_value": "oil"},
    {"name": "subject", "mode": "input", "variation_scope": "batch"},
    {
        "name": "mood",
        "mode": "choice",
        "variation_scope": "item",
        "choices": ["calm", "stormy", "bright"],
    },
]


def _contract(*, body: str = _BODY, slots: list[dict[str, object]] | None = None):
    return parse_prompt_template_contract(
        {
            "schema_version": 1,
            "operation": "text_to_image",
            "body": body,
            "slots": list(slots if slots is not None else _SLOTS),
            "resource_policy": {"mode": "inherited"},
        }
    )


def _request(contract=None, **overrides: object):
    """Build a request bound to the contract it will be expanded against.

    The digest is real rather than a placeholder, because the codec now refuses
    a request whose recorded digest does not match the contract handed to it.
    """

    payload: dict[str, object] = {
        "definition_id": "ptdef_expansion",
        "revision_id": "ptrev_expansion",
        "contract_sha256": prompt_template_contract_sha256(
            contract if contract is not None else _contract()
        ),
        "item_count": 3,
        "selection_seed": 12_345,
        "inputs": {"subject": "a fox"},
    }
    payload.update(overrides)
    return parse_expansion_request(payload)


def _values(plan, name: str) -> list[str | None]:
    return [
        next(entry.value for entry in item.evidence if entry.name == name) for item in plan.items
    ]


def _prompts(plan) -> list[str | None]:
    return [item.rendered_prompt for item in plan.items]


def _bridge_contract():
    slots: list[dict[str, object]] = [
        {
            "name": "style",
            "mode": "fixed",
            "variation_scope": "batch",
            "fixed_value": "oil",
        },
        {"name": "subject", "mode": "input", "variation_scope": "batch"},
        {
            "name": "medium",
            "mode": "model",
            "variation_scope": "batch",
            "guidance": "one medium",
        },
        {
            "name": "mood",
            "mode": "choice",
            "variation_scope": "item",
            "choices": ["calm", "stormy"],
        },
        {
            "name": "detail",
            "mode": "model",
            "variation_scope": "item",
            "guidance": "one visible detail",
        },
    ]
    return _contract(
        body="{{style}} {{subject}} in {{medium}}, {{mood}}, with {{detail}}",
        slots=slots,
    )


def _bridge_plan():
    contract = _bridge_contract()
    return contract, expand_prompt_template(
        contract,
        _request(contract, item_count=2, inputs={"subject": "a fox"}),
    )


def _bridge_values(contract=None):
    selected = contract if contract is not None else _bridge_contract()
    model_contract = prompt_model_slot_contract(selected, item_count=2)
    return parse_prompt_model_values(
        {
            "version": 1,
            "batch_values": {"medium": "tempera"},
            "items": [
                {"ordinal": 1, "values": {"detail": "silver leaves"}},
                {"ordinal": 2, "values": {"detail": "amber rain"}},
            ],
        },
        contract=model_contract,
    )


# --- what the codec produces -------------------------------------------------


def test_every_non_model_mode_resolves_with_its_own_recorded_source() -> None:
    plan = expand_prompt_template(_contract(), _request(item_count=1))
    sources = {entry.name: entry.source for entry in plan.items[0].evidence}
    assert sources == {
        "style": ExpansionValueSource.FIXED,
        "subject": ExpansionValueSource.INPUT,
        "mood": ExpansionValueSource.CHOICE,
    }
    assert plan.complete
    assert plan.items[0].rendered_prompt is not None


def test_an_expansion_produces_exactly_n_distinct_rendered_drafts() -> None:
    """The count is a count of DISTINCT drafts, not of ordinals."""

    plan = expand_prompt_template(_contract(), _request(item_count=3))
    prompts = _prompts(plan)
    assert [item.ordinal for item in plan.items] == [1, 2, 3]
    assert len(set(prompts)) == 3


def test_a_batch_choice_alone_cannot_produce_two_distinct_drafts() -> None:
    """Batch scope shares one value, so it adds no item diversity at all."""

    slots = [dict(slot) for slot in _SLOTS]
    slots[2]["variation_scope"] = "batch"
    contract = _contract(slots=slots)
    assert expand_prompt_template(contract, _request(contract, item_count=1)).complete
    with pytest.raises(PromptExpansionError) as caught:
        expand_prompt_template(contract, _request(contract, item_count=2))
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_two_item_choices_yield_two_drafts_and_refuse_a_third() -> None:
    slots = [dict(slot) for slot in _SLOTS]
    slots[2]["choices"] = ["calm", "stormy"]
    contract = _contract(slots=slots)
    plan = expand_prompt_template(contract, _request(contract, item_count=2))
    assert len(set(_prompts(plan))) == 2
    with pytest.raises(PromptExpansionDistinctCapacityError) as caught:
        expand_prompt_template(contract, _request(contract, item_count=3))
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_reusable_item_choice_can_repeat_through_the_global_batch_cap() -> None:
    slots = [dict(slot) for slot in _SLOTS]
    slots[2]["choices"] = ["calm"]
    slots[2]["choice_strategy"] = "with_replacement"
    contract = _contract(slots=slots)
    plan = expand_prompt_template(
        contract,
        _request(contract, item_count=MAX_EXPANSION_ITEMS),
    )
    assert len(plan.items) == MAX_EXPANSION_ITEMS
    assert _values(plan, "mood") == ["calm"] * MAX_EXPANSION_ITEMS
    assert len(set(_prompts(plan))) == 1
    assert {item.evidence[-1].choice_index for item in plan.items} == {0}
    assert expansion_plan_digest(plan) == expansion_plan_digest(
        expand_prompt_template(
            contract,
            _request(contract, item_count=MAX_EXPANSION_ITEMS),
        )
    )


def test_distinct_item_input_vectors_produce_distinct_drafts() -> None:
    """An item-scoped input is an ordered vector, one value per ordinal."""

    slots = [dict(slot) for slot in _SLOTS]
    slots[1]["variation_scope"] = "item"
    plan = expand_prompt_template(
        _contract(slots=slots),
        _request(
            _contract(slots=slots), item_count=3, inputs={"subject": ["a fox", "a hare", "an owl"]}
        ),
    )
    assert _values(plan, "subject") == ["a fox", "a hare", "an owl"]
    assert len(set(_prompts(plan))) == 3


def test_an_item_input_vector_must_be_exactly_the_requested_length() -> None:
    slots = [dict(slot) for slot in _SLOTS]
    slots[1]["variation_scope"] = "item"
    contract = _contract(slots=slots)
    with pytest.raises(PromptExpansionError):
        expand_prompt_template(
            contract,
            _request(contract, item_count=3, inputs={"subject": ["a fox", "a hare"]}),
        )


def test_a_batch_input_must_be_a_scalar_and_an_item_input_a_vector() -> None:
    with pytest.raises(PromptExpansionError):
        # `subject` is batch-scoped here, so a vector is the wrong shape.
        expand_prompt_template(_contract(), _request(item_count=1, inputs={"subject": ["a fox"]}))
    slots = [dict(slot) for slot in _SLOTS]
    slots[1]["variation_scope"] = "item"
    with pytest.raises(PromptExpansionError):
        expand_prompt_template(
            _contract(slots=slots),
            _request(_contract(slots=slots), item_count=1, inputs={"subject": "a fox"}),
        )


def test_two_value_maps_that_render_the_same_prompt_are_caught_as_one_draft() -> None:
    """Distinctness is judged on the rendered prompt, not on the value map.

    Two slots whose values are swapped can render the same literal string; the
    batch must refuse rather than present them as two drafts.
    """

    body = "{{first}}{{second}}"
    slots: list[dict[str, object]] = [
        {"name": "first", "mode": "choice", "variation_scope": "item", "choices": ["ab", "a"]},
        {"name": "second", "mode": "choice", "variation_scope": "item", "choices": ["c", "bc"]},
    ]
    contract = _contract(body=body, slots=slots)
    # Four value maps, but only three distinct strings: ("ab","c") and
    # ("a","bc") both render "abc". A codec counting value maps would think it
    # had four drafts.
    plan = expand_prompt_template(contract, _request(contract, item_count=3, inputs={}))
    assert sorted(str(prompt) for prompt in _prompts(plan)) == ["abbc", "abc", "ac"]
    with pytest.raises(PromptExpansionError) as caught:
        expand_prompt_template(contract, _request(contract, item_count=4, inputs={}))
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


# --- determinism -------------------------------------------------------------


def test_allocation_is_deterministic_for_one_seed_and_sensitive_to_another() -> None:
    contract = _contract()
    first = expand_prompt_template(
        contract,
        _request(
            contract,
        ),
    )
    again = expand_prompt_template(
        contract,
        _request(
            contract,
        ),
    )
    assert expansion_plan_digest(first) == expansion_plan_digest(again)
    assert _prompts(first) == _prompts(again)
    other = expand_prompt_template(contract, _request(contract, selection_seed=999))
    assert expansion_plan_digest(first) != expansion_plan_digest(other)


def test_every_allocated_choice_is_one_the_revision_declared() -> None:
    contract = _contract()
    plan = expand_prompt_template(contract, _request(contract, item_count=3))
    declared = next(slot for slot in contract.slots if slot.name == "mood").choices
    for item in plan.items:
        entry = next(one for one in item.evidence if one.name == "mood")
        assert entry.value in declared
        assert entry.choice_index is not None
        assert declared[entry.choice_index] == entry.value


def test_batch_scope_shares_one_value_across_every_item() -> None:
    plan = expand_prompt_template(_contract(), _request(item_count=3))
    assert len(set(_values(plan, "subject"))) == 1


# --- rendering ---------------------------------------------------------------


def test_rendering_is_one_pass_so_a_value_that_looks_like_a_token_stays_literal() -> None:
    plan = expand_prompt_template(
        _contract(), _request(item_count=1, inputs={"subject": "{{mood}}"})
    )
    assert plan.items[0].rendered_prompt is not None
    assert "{{mood}}" in plan.items[0].rendered_prompt


def test_a_model_slot_leaves_its_item_unrendered_and_is_reported_as_pending() -> None:
    slots = [
        *_SLOTS,
        {
            "name": "extra",
            "mode": "model",
            "variation_scope": "item",
            "guidance": "one concrete detail",
        },
    ]
    contract = _contract(body=_BODY[:-1] + " with {{extra}}.", slots=slots)
    plan = expand_prompt_template(contract, _request(contract, item_count=3))
    assert not plan.complete
    assert [pending.name for pending in plan.pending_model_slots] == ["extra"]
    assert all(item.rendered_prompt is None for item in plan.items)
    entry = next(one for one in plan.items[0].evidence if one.name == "extra")
    assert entry.source is ExpansionValueSource.MODEL
    assert entry.value is None


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize("count", [0, MAX_EXPANSION_ITEMS + 1, -1, True])
def test_counts_outside_the_bound_refuse(count: object) -> None:
    with pytest.raises(PromptExpansionError) as caught:
        _request(item_count=count)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


@pytest.mark.parametrize(
    "inputs",
    [{}, {"subject": "a fox", "surplus": "unexpected"}, {"wrong": "a fox"}],
)
def test_inputs_must_match_the_declared_input_slots_exactly(inputs: dict[str, object]) -> None:
    with pytest.raises(PromptExpansionError):
        expand_prompt_template(_contract(), _request(item_count=1, inputs=inputs))


@pytest.mark.parametrize(
    "payload",
    [
        {"item_count": 1.0},
        {"selection_seed": SELECTION_SEED_SPACE},
        {"selection_seed": -1},
        {"contract_sha256": "A" * 64},
        {"contract_sha256": "a" * 63},
        {"inputs": {"subject": 5}},
        {"inputs": {"subject": []}},
        {"inputs": []},
        {"definition_id": ""},
    ],
)
def test_malformed_requests_refuse_without_coercion(payload: dict[str, object]) -> None:
    with pytest.raises(PromptExpansionError) as caught:
        _request(**payload)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_a_request_with_an_unexpected_key_refuses() -> None:
    with pytest.raises(PromptExpansionError):
        parse_expansion_request(
            {
                "definition_id": "ptdef_expansion",
                "revision_id": "ptrev_expansion",
                "contract_sha256": "a" * 64,
                "item_count": 1,
                "selection_seed": 1,
                "inputs": {"subject": "a fox"},
                "unexpected": True,
            }
        )


def test_refusals_never_echo_the_template_or_the_values() -> None:
    secret = "SECRET-SUBJECT-VALUE"
    with pytest.raises(PromptExpansionError) as caught:
        expand_prompt_template(
            _contract(),
            _request(item_count=1, inputs={"subject": "a fox", "surplus": secret}),
        )
    message = str(caught.value)
    assert message == PROMPT_EXPANSION_INVALID
    assert secret not in message
    assert "portrait" not in message


# --- digest and seed domain --------------------------------------------------


def test_the_digest_is_stable_for_one_request_and_moves_with_the_inputs() -> None:
    """Renamed: the old name claimed the digest excludes the rendered text.

    That was never what the body checked, and it is no longer true - the plan
    payload recomputes every rendered digest, so the text is covered.
    """

    contract = _contract()
    plan = expand_prompt_template(
        contract,
        _request(
            contract,
        ),
    )
    same = expand_prompt_template(
        contract,
        _request(
            contract,
        ),
    )
    assert expansion_plan_digest(plan) == expansion_plan_digest(same)
    different = expand_prompt_template(contract, _request(contract, inputs={"subject": "a hare"}))
    assert expansion_plan_digest(plan) != expansion_plan_digest(different)


def test_the_selection_seed_domain_is_its_own() -> None:
    seed = expansion_selection_seed(["ptdef_expansion", "ptrev_expansion", "1"])
    assert 0 <= seed < SELECTION_SEED_SPACE
    assert seed == expansion_selection_seed(["ptdef_expansion", "ptrev_expansion", "1"])
    assert seed != expansion_selection_seed(["ptdef_expansion", "ptrev_expansion", "2"])
    with pytest.raises(PromptExpansionError):
        expansion_selection_seed([1])


def test_evidence_records_the_declared_scope_for_every_slot() -> None:
    plan = expand_prompt_template(_contract(), _request(item_count=2))
    for item in plan.items:
        scopes = {entry.name: entry.variation_scope for entry in item.evidence}
        assert scopes["subject"] is PromptTemplateVariationScope.BATCH
        assert scopes["mood"] is PromptTemplateVariationScope.ITEM
        modes = {entry.name: entry.mode for entry in item.evidence}
        assert modes["style"] is PromptTemplateSlotMode.FIXED


def test_an_item_choice_space_too_large_to_enumerate_refuses() -> None:
    """Enumeration is what makes "cannot yield N" exact; it must stay bounded.

    Four item-scoped slots of sixty-four choices is 16.7 million combinations.
    Enumerating that would hang rather than answer, so the codec refuses.
    """

    choices = [f"c{index}" for index in range(64)]
    slots: list[dict[str, object]] = [
        {
            "name": f"s{index}",
            "mode": "choice",
            "variation_scope": "item",
            "choices": list(choices),
        }
        for index in range(4)
    ]
    body = " ".join(f"{{{{s{index}}}}}" for index in range(4)) + "."
    contract = _contract(body=body, slots=slots)
    with pytest.raises(PromptExpansionError) as caught:
        expand_prompt_template(contract, _request(contract, item_count=2, inputs={}))
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


# --- exactness, binding, and the restored acceptance gates -------------------


def test_a_request_whose_digest_does_not_match_the_contract_refuses() -> None:
    """The recorded digest must bind the revision, or it is decorative."""

    contract = _contract()
    with pytest.raises(PromptExpansionError) as caught:
        expand_prompt_template(contract, _request(contract, contract_sha256="b" * 64, item_count=1))
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_a_str_subclass_root_key_is_refused_before_any_set_comparison() -> None:
    """A `str` subclass compares and hashes equal, so a set check alone passes it."""

    class SneakyKey(str):
        __slots__ = ()

    contract = _contract()
    payload: dict[object, object] = {
        SneakyKey("definition_id"): "ptdef_expansion",
        "revision_id": "ptrev_expansion",
        "contract_sha256": prompt_template_contract_sha256(contract),
        "item_count": 1,
        "selection_seed": 1,
        "inputs": {"subject": "a fox"},
    }
    with pytest.raises(PromptExpansionError):
        parse_expansion_request(payload)


def test_a_str_subclass_input_key_is_refused_too() -> None:
    class SneakyKey(str):
        __slots__ = ()

    contract = _contract()
    payload: dict[object, object] = {
        "definition_id": "ptdef_expansion",
        "revision_id": "ptrev_expansion",
        "contract_sha256": prompt_template_contract_sha256(contract),
        "item_count": 1,
        "selection_seed": 1,
        "inputs": {SneakyKey("subject"): "a fox"},
    }
    with pytest.raises(PromptExpansionError):
        parse_expansion_request(payload)


def test_seed_material_refuses_a_generator_and_stays_bounded() -> None:
    """A generator could yield differently on a second read; a pure codec cannot."""

    reads = 0

    def material():
        nonlocal reads
        reads += 1
        yield "ptdef_expansion"

    with pytest.raises(PromptExpansionError):
        expansion_selection_seed(material())
    assert reads == 0

    class Custom:
        def __iter__(self):  # pragma: no cover - must never be invoked
            raise AssertionError("caller behaviour was invoked")

    with pytest.raises(PromptExpansionError):
        expansion_selection_seed(Custom())

    with pytest.raises(PromptExpansionError):
        expansion_selection_seed([])
    with pytest.raises(PromptExpansionError):
        expansion_selection_seed(["x"] * 33)
    with pytest.raises(PromptExpansionError):
        expansion_selection_seed(["x" * 5_000])


def test_item_choices_still_vary_when_a_model_slot_is_pending() -> None:
    """N copies of one allocation is not a batch, even before the model runs."""

    slots = [
        *_SLOTS,
        {
            "name": "extra",
            "mode": "model",
            "variation_scope": "item",
            "guidance": "one concrete detail",
        },
    ]
    contract = _contract(body=_BODY[:-1] + " with {{extra}}.", slots=slots)
    plan = expand_prompt_template(contract, _request(contract, item_count=3))
    assert not plan.complete
    assert len(set(_values(plan, "mood"))) > 1


def test_n_of_one_and_n_of_sixteen_both_succeed() -> None:
    """The exact bounds of the declared range, both accepted."""

    slots = [dict(slot) for slot in _SLOTS]
    slots[1]["variation_scope"] = "item"
    contract = _contract(slots=slots)
    for count in (1, MAX_EXPANSION_ITEMS):
        subjects = [f"subject {index}" for index in range(count)]
        plan = expand_prompt_template(
            contract, _request(contract, item_count=count, inputs={"subject": subjects})
        )
        assert [item.ordinal for item in plan.items] == list(range(1, count + 1))
        assert len(set(_prompts(plan))) == count
        assert all(item.rendered_sha256 is not None for item in plan.items)


def test_declared_choice_order_changes_what_is_allocated() -> None:
    slots = [dict(slot) for slot in _SLOTS]
    slots[2]["choices"] = ["bright", "stormy", "calm"]
    reordered_contract = _contract(slots=slots)
    reordered = expand_prompt_template(
        reordered_contract, _request(reordered_contract, item_count=2)
    )
    original = expand_prompt_template(_contract(), _request(item_count=2))
    assert _values(reordered, "mood") != _values(original, "mood")


def test_a_batch_choice_stays_shared_while_an_item_source_varies() -> None:
    """Mixed scopes: batch evidence identical across items, item evidence not."""

    slots = [dict(slot) for slot in _SLOTS]
    slots[1]["variation_scope"] = "item"
    slots[2]["variation_scope"] = "batch"
    contract = _contract(slots=slots)
    plan = expand_prompt_template(
        contract,
        _request(contract, item_count=3, inputs={"subject": ["a fox", "a hare", "an owl"]}),
    )
    assert len(set(_values(plan, "mood"))) == 1
    assert len(set(_values(plan, "subject"))) == 3
    assert len(set(_prompts(plan))) == 3


def test_each_complete_item_carries_a_canonical_rendered_digest() -> None:
    plan = expand_prompt_template(_contract(), _request(item_count=3))
    digests = [item.rendered_sha256 for item in plan.items]
    assert all(digest is not None and len(digest) == 64 for digest in digests)
    assert len(set(digests)) == 3
    # The digest identifies the rendered text, so equal text means equal digest.
    again = expand_prompt_template(_contract(), _request(item_count=3))
    assert [item.rendered_sha256 for item in again.items] == digests


def test_the_plan_digest_covers_the_rendered_identity() -> None:
    """Two plans differing only in rendered text must not share a plan digest."""

    contract = _contract()
    plan = expand_prompt_template(contract, _request())
    other_contract = _contract(body="A different {{style}} {{subject}} {{mood}}.")
    other = expand_prompt_template(other_contract, _request(other_contract))
    assert expansion_plan_digest(plan) != expansion_plan_digest(other)


# --- the plan receipt cannot be forged ---------------------------------------
#
# A frozen dataclass is not tamper-proof: `object.__setattr__` rewrites any
# field on one. So a digest built by READING fields attests to whatever was
# written last, and the receipt means nothing. Every test below rewrites a
# finished plan and requires the digest call to refuse rather than agree.


def _finished_plan():
    contract = _contract()
    return contract, expand_prompt_template(contract, _request(contract))


def _pending_contract():
    slots = [
        *_SLOTS,
        {"name": "extra", "mode": "model", "variation_scope": "item", "guidance": "a detail"},
    ]
    return _contract(body=_BODY[:-1] + " with {{extra}}.", slots=slots)


def test_a_tampered_rendered_prompt_cannot_keep_its_receipt() -> None:
    """The defect this closes: the digest copied `rendered_sha256`.

    Copying meant the prompt and the digest attesting to it could disagree, and
    the plan would still hash identically. The payload now RECOMPUTES each
    digest from its prompt, so the two can never drift apart unnoticed.
    """

    _unused, plan = _finished_plan()
    before = expansion_plan_digest(plan)
    object.__setattr__(plan.items[0], "rendered_prompt", "an entirely different prompt")
    with pytest.raises(PromptExpansionError) as caught:
        expansion_plan_digest(plan)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID
    # The untampered plan still hashes, so the guard is not refusing everything.
    _unused2, fresh = _finished_plan()
    assert expansion_plan_digest(fresh) == before


def test_a_tampered_rendered_digest_cannot_stand_in_for_its_prompt() -> None:
    """The mirror direction: rewrite the digest instead of the prompt."""

    _unused, plan = _finished_plan()
    object.__setattr__(plan.items[0], "rendered_sha256", "0" * 64)
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_injected_pending_model_slots_change_the_receipt() -> None:
    """The defect this closes: `pending_model_slots` was not hashed at all.

    A plan could therefore move between complete and incomplete under one
    digest - the receipt could not distinguish a finished batch from one still
    waiting on a model.
    """

    _unused, plan = _finished_plan()
    ghost = ModelSlotRequest(
        name="ghost",
        variation_scope=PromptTemplateVariationScope.ITEM,
        guidance="one concrete detail",
    )
    object.__setattr__(plan, "pending_model_slots", (ghost,))
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_a_pending_plan_and_a_complete_one_never_share_a_receipt() -> None:
    """The same property from the legitimate direction, with no tampering."""

    contract = _pending_contract()
    pending = expand_prompt_template(contract, _request(contract))
    assert not pending.complete
    payload = expansion_plan_payload(pending)
    assert [entry["name"] for entry in payload["pending_model_slots"]] == ["extra"]
    assert all(item["rendered_sha256"] is None for item in payload["items"])
    _unused, complete = _finished_plan()
    assert expansion_plan_digest(pending) != expansion_plan_digest(complete)


def test_a_pending_plan_that_claims_a_rendered_prompt_refuses() -> None:
    """Incomplete and rendered is a state the codec never produces."""

    contract = _pending_contract()
    plan = expand_prompt_template(contract, _request(contract))
    object.__setattr__(plan.items[0], "rendered_prompt", "already finished")
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


@pytest.mark.parametrize("ordinal", [0, 2, 99, -1])
def test_a_tampered_ordinal_refuses(ordinal: object) -> None:
    _unused, plan = _finished_plan()
    object.__setattr__(plan.items[0], "ordinal", ordinal)
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_a_dropped_item_refuses_because_the_count_is_part_of_the_request() -> None:
    _unused, plan = _finished_plan()
    object.__setattr__(plan, "items", plan.items[:2])
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_evidence_whose_source_disagrees_with_its_mode_refuses() -> None:
    """Source and mode describe the same fact; they must not be able to differ."""

    _unused, plan = _finished_plan()
    entry = plan.items[0].evidence[0]
    object.__setattr__(entry, "source", ExpansionValueSource.MODEL)
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_duplicate_slot_names_within_one_item_refuse() -> None:
    _unused, plan = _finished_plan()
    first = plan.items[0].evidence[0]
    object.__setattr__(plan.items[0], "evidence", (first, first))
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_an_expansion_plan_subclass_cannot_pass_as_the_real_type() -> None:
    """`isinstance` would admit this; the codec checks the exact type."""

    sneaky = dataclasses.dataclass(frozen=True, slots=True)(
        type("SneakyPlan", (ExpansionPlan,), {})
    )
    _unused, plan = _finished_plan()
    candidate = sneaky(**{f.name: getattr(plan, f.name) for f in dataclasses.fields(plan)})
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(candidate)


def test_a_model_slot_request_subclass_cannot_pass_as_the_real_type() -> None:
    sneaky = dataclasses.dataclass(frozen=True, slots=True)(
        type("SneakySlot", (ModelSlotRequest,), {})
    )
    _unused, plan = _finished_plan()
    object.__setattr__(
        plan,
        "pending_model_slots",
        (
            sneaky(
                name="ghost",
                variation_scope=PromptTemplateVariationScope.ITEM,
                guidance="a detail",
            ),
        ),
    )
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


# --- the selection seed encodes its own boundaries ---------------------------


def test_seed_material_cannot_be_reshaped_across_a_separator() -> None:
    """The defect this closes: entries were joined with NUL.

    NUL is a legal character inside a string, so a member could impersonate the
    separator and two different lists produced one seed. Both directions are
    checked, because closing one and not the other would leave the property
    half-true.
    """

    assert expansion_selection_seed(["a", "\x00b"]) != expansion_selection_seed(["a\x00", "b"])
    assert expansion_selection_seed(["a\x00", "b"]) != expansion_selection_seed(["a", "b\x00"])
    # The same ambiguity without a NUL at all: concatenation must not equal
    # separation, or the boundaries are still not encoded.
    assert expansion_selection_seed(["ab"]) != expansion_selection_seed(["a", "b"])
    assert expansion_selection_seed(['a"', "b"]) != expansion_selection_seed(["a", '"b'])


def test_one_seed_material_is_stable_and_in_range() -> None:
    first = expansion_selection_seed(["definition", "revision", "key"])
    assert first == expansion_selection_seed(["definition", "revision", "key"])
    assert 0 <= first < SELECTION_SEED_SPACE


def test_a_lone_surrogate_refuses_rather_than_hashing_a_replacement() -> None:
    """A surrogate cannot be encoded, and substituting bytes is not determinism."""

    with pytest.raises(PromptExpansionError) as caught:
        expansion_selection_seed(["ok", "\ud800"])
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


@pytest.mark.parametrize(
    "material",
    [
        [],
        (),
        "not a sequence",
        {"a": "b"},
        [b"bytes"],
        [None],
        [1],
        ["ok", 2],
        ["x"] * (MAX_SEED_MATERIAL_ENTRIES + 1),
        ["x" * (MAX_SEED_MATERIAL_CHARS + 1)],
        ["x" * MAX_SEED_MATERIAL_CHARS, "y"],
    ],
)
def test_malformed_seed_material_refuses(material: object) -> None:
    with pytest.raises(PromptExpansionError):
        expansion_selection_seed(material)


def test_seed_material_is_bounded_in_bytes_and_not_only_in_characters() -> None:
    """A character bound alone is not a byte bound: one char can cost four."""

    assert MAX_SEED_MATERIAL_BYTES >= MAX_SEED_MATERIAL_CHARS
    # Inside both bounds, so the bound is a bound and not a wall.
    assert expansion_selection_seed(["\U0001f600" * 8]) >= 0


def test_a_list_subclass_is_not_accepted_as_seed_material() -> None:
    sneaky = type("SneakyList", (list,), {})(["a"])
    with pytest.raises(PromptExpansionError):
        expansion_selection_seed(sneaky)


# --- one error type at the template boundary ---------------------------------


def test_a_value_the_request_accepts_but_the_renderer_refuses_raises_our_type() -> None:
    """The defect this closes: `PromptTemplateError` escaped from the renderer.

    The request parser admits a NUL in an input value; the renderer refuses it.
    A caller catching this module's error would have seen the other type pass
    straight through.
    """

    contract = _contract()
    with pytest.raises(PromptExpansionError) as caught:
        expand_prompt_template(contract, _request(contract, inputs={"subject": "a\x00fox"}))
    assert str(caught.value) == PROMPT_EXPANSION_INVALID
    assert not isinstance(caught.value, PromptTemplateError)


def test_a_hand_built_malformed_contract_raises_our_type() -> None:
    """A contract that never went through the parser fails digest validation."""

    contract = copy.deepcopy(_contract())
    object.__setattr__(contract, "body", "")
    with pytest.raises(PromptExpansionError) as caught:
        expand_prompt_template(contract, _request())
    assert str(caught.value) == PROMPT_EXPANSION_INVALID
    assert not isinstance(caught.value, PromptTemplateError)


def test_the_normalised_boundary_echoes_no_template_text() -> None:
    """Normalising must not become a way to smuggle the template into a message.

    The contract here carries no model slot on purpose. A model slot leaves its
    item unrendered, so the renderer is never reached and this control would
    pass without exercising the boundary it names - which is the failure mode
    this whole review round is about. Guidance is covered separately below.
    """

    slots = [
        {
            "name": "style",
            "mode": "fixed",
            "variation_scope": "batch",
            "fixed_value": "secret-fixed-marker",
        },
        {"name": "subject", "mode": "input", "variation_scope": "batch"},
        {
            "name": "mood",
            "mode": "choice",
            "variation_scope": "item",
            "choices": ["secret-choice-marker", "another"],
        },
    ]
    contract = _contract(body="{{style}} {{subject}} {{mood}} secret-body-marker.", slots=slots)
    with pytest.raises(PromptExpansionError) as caught:
        expand_prompt_template(
            contract, _request(contract, inputs={"subject": "a\x00fox"}, item_count=2)
        )
    message = str(caught.value)
    assert message == PROMPT_EXPANSION_INVALID
    for leak in (
        "secret-body-marker",
        "secret-fixed-marker",
        "secret-choice-marker",
        "fox",
        "style",
        "subject",
        "mood",
        "ptdef_expansion",
        "ptrev_expansion",
    ):
        assert leak not in message


def test_a_refusal_over_a_pending_plan_echoes_no_guidance() -> None:
    """Guidance is template text too, and it only exists on a pending plan."""

    slots = [
        *_SLOTS,
        {
            "name": "extra",
            "mode": "model",
            "variation_scope": "item",
            "guidance": "secret-guidance-marker",
        },
    ]
    contract = _contract(body=_BODY[:-1] + " with {{extra}}.", slots=slots)
    plan = expand_prompt_template(contract, _request(contract))
    assert [slot.guidance for slot in plan.pending_model_slots] == ["secret-guidance-marker"]
    object.__setattr__(plan.items[0], "rendered_prompt", "forged")
    with pytest.raises(PromptExpansionError) as caught:
        expansion_plan_digest(plan)
    message = str(caught.value)
    assert message == PROMPT_EXPANSION_INVALID
    assert "secret-guidance-marker" not in message
    assert "forged" not in message


def test_a_model_slot_carries_guidance_as_a_string_never_none() -> None:
    """Phase 1 guarantees guidance is present, and the model-values codec on the
    other side of this seam types it `str`. Admitting None here would describe a
    state neither side can reach."""

    contract = _pending_contract()
    plan = expand_prompt_template(contract, _request(contract))
    assert [slot.guidance for slot in plan.pending_model_slots] == ["a detail"]
    field = {f.name: f for f in dataclasses.fields(ModelSlotRequest)}["guidance"]
    assert field.type == "str"


# --- guards the first mutation run found unlocked ----------------------------
#
# Each of the four below was already refusing the vector I wrote for it - but a
# DIFFERENT guard was doing the refusing, so removing the one under test changed
# nothing and the suite stayed green. These isolate them.


def _pending_plan():
    contract = _pending_contract()
    return expand_prompt_template(contract, _request(contract))


def test_an_items_model_slots_must_be_exactly_the_plans_pending_ones() -> None:
    """Isolated from completeness.

    The obvious vector - inject a pending slot into a finished plan - is caught
    by the rendered/pending consistency check instead, so it proves nothing
    about this guard. Here the plan stays incomplete and its items stay
    unrendered; only the NAMES disagree, so nothing else can fire.
    """

    plan = _pending_plan()
    renamed = ModelSlotRequest(
        name="ghost",
        variation_scope=PromptTemplateVariationScope.ITEM,
        guidance="a detail",
    )
    object.__setattr__(plan, "pending_model_slots", (renamed,))
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_a_model_slot_request_subclass_is_refused_on_an_otherwise_valid_plan() -> None:
    """The subclass carries identical field values, so only its type is wrong."""

    plan = _pending_plan()
    real = plan.pending_model_slots[0]
    sneaky = dataclasses.dataclass(frozen=True, slots=True)(
        type("SneakySlot", (ModelSlotRequest,), {})
    )
    object.__setattr__(
        plan,
        "pending_model_slots",
        (
            sneaky(
                name=real.name,
                variation_scope=real.variation_scope,
                guidance=real.guidance,
            ),
        ),
    )
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_a_slot_evidence_subclass_is_refused() -> None:
    """Same shape, wrong type - `isinstance` would let this through."""

    _unused, plan = _finished_plan()
    real = plan.items[0].evidence[0]
    sneaky = dataclasses.dataclass(frozen=True, slots=True)(
        type("SneakyEvidence", (SlotEvidence,), {})
    )
    forged = sneaky(**{f.name: getattr(real, f.name) for f in dataclasses.fields(real)})
    object.__setattr__(plan.items[0], "evidence", (forged, *plan.items[0].evidence[1:]))
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_a_tampered_guidance_on_a_pending_plan_refuses() -> None:
    """Guidance is hashed, so it has to be validated where it is hashed."""

    plan = _pending_plan()
    object.__setattr__(plan.pending_model_slots[0], "guidance", None)
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_a_contract_whose_model_slot_has_no_guidance_never_reaches_expansion() -> None:
    """Where that contract actually stops, which is not where I first assumed.

    The slot dataclass types `guidance` as `str | None` and the contract's own
    field check permits None for every mode, so I expected such a contract to
    reach the expansion and be refused there. It does not.
    `prompt_template_contract_sha256` round-trips the contract back through the
    parser, and the parser requires guidance on a model slot - so the digest
    call refuses first, and the narrowing inside the expansion is a second layer
    that nothing can reach through this entry point.

    Worth a test anyway: it is the reason `ModelSlotRequest.guidance` can be
    `str`, and if that round-trip were ever relaxed this would be the first
    thing to fail.
    """

    contract = PromptTemplateContract(
        schema_version=1,
        operation="text_to_image",
        body="{{subject}} and {{extra}}.",
        slots=(
            PromptTemplateSlot(
                name="subject",
                mode=PromptTemplateSlotMode.INPUT,
                variation_scope=PromptTemplateVariationScope.BATCH,
            ),
            PromptTemplateSlot(
                name="extra",
                mode=PromptTemplateSlotMode.MODEL,
                variation_scope=PromptTemplateVariationScope.ITEM,
                guidance=None,
            ),
        ),
        resource_policy=PromptTemplateResourcePolicy(mode=PromptTemplateResourceMode.INHERITED),
    )
    # It has no obtainable digest, because digesting it is what refuses it.
    with pytest.raises(PromptTemplateError):
        prompt_template_contract_sha256(contract)
    # And through the expansion the refusal is normalised to this module's type.
    with pytest.raises(PromptExpansionError) as caught:
        expand_prompt_template(contract, _request(contract_sha256="a" * 64))
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


# --- the request inside the receipt is validated, not merely typed -----------
#
# Checking that `plan.request` is an `ExpansionRequest` says nothing about
# what is inside it. Every field below was accepted and hashed before:
# `item_count` became `True` and serialised as JSON `true`, a seed became a
# string, and an input value became an arbitrary object that escaped
# as a raw `TypeError` out of the JSON encoder rather than a refusal.


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        # The four exact vectors from the audit.
        ("item_count", True),
        ("selection_seed", "private-seed"),
        ("contract_sha256", "C:\\private"),
        ("inputs", (("subject", object()),)),
        # bool where int is expected, in both numeric fields.
        ("selection_seed", True),
        ("item_count", False),
        # Out of bounds rather than wrong type.
        ("item_count", 0),
        ("item_count", MAX_EXPANSION_ITEMS + 1),
        ("selection_seed", -1),
        ("selection_seed", SELECTION_SEED_SPACE),
        # Wrong types on the identity fields.
        ("definition_id", 5),
        ("definition_id", ""),
        ("definition_id", None),
        ("revision_id", None),
        ("contract_sha256", "z" * 64),
        ("contract_sha256", "abc"),
        # Malformed inputs: not a tuple, wrong pair shape, wrong pair type,
        # duplicate names, a list where a tuple belongs, and an emptied map
        # whose evidence still names an input slot.
        ("inputs", "not a tuple"),
        ("inputs", [("subject", "a fox")]),
        ("inputs", (("subject",),)),
        ("inputs", (("subject", "a fox", "extra"),)),
        ("inputs", (["subject", "a fox"],)),
        ("inputs", ((5, "a fox"),)),
        ("inputs", (("subject", "a fox"), ("subject", "a hare"))),
        ("inputs", (("subject", ["a", "b"]),)),
        ("inputs", ()),
    ],
)
def test_a_tampered_request_field_cannot_be_hashed(attribute: str, value: object) -> None:
    _unused, plan = _finished_plan()
    object.__setattr__(plan.request, attribute, value)
    with pytest.raises(PromptExpansionError) as caught:
        expansion_plan_digest(plan)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_an_expansion_request_subclass_cannot_pass_as_the_real_type() -> None:
    sneaky = dataclasses.dataclass(frozen=True, slots=True)(
        type("SneakyRequest", (ExpansionRequest,), {})
    )
    _unused, plan = _finished_plan()
    real = plan.request
    object.__setattr__(
        plan,
        "request",
        sneaky(**{f.name: getattr(real, f.name) for f in dataclasses.fields(real)}),
    )
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_the_request_payload_round_trips_through_its_own_parser() -> None:
    """The property that makes the request payload worth hashing.

    It is not built by reading fields. It is rebuilt and then re-parsed, and the
    parser is the only definition of a valid request there is.
    """

    contract = _contract()
    request = _request(contract)
    payload = expansion_request_payload(request)
    assert parse_expansion_request(payload) == request
    assert payload["inputs"] == {"subject": "a fox"}
    assert payload["item_count"] == 3
    assert type(payload["item_count"]) is int


def test_the_inputs_a_receipt_declares_must_be_the_ones_its_evidence_records() -> None:
    """Emptying the inputs is not forgery - the digest changes - but the receipt
    it produces describes a plan the codec could never have built, because
    expansion requires the supplied inputs to match the declared input slots."""

    _unused, plan = _finished_plan()
    object.__setattr__(plan.request, "inputs", ())
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


# --- private text never reaches a repr ---------------------------------------


def test_no_repr_exposes_body_guidance_choices_values_or_rendered_text() -> None:
    """A repr reaches logs, tracebacks, and debugger frames.

    None of those are places a prompt body, a slot value, a choice, model
    guidance, or a rendered prompt is allowed to appear. Identity, ordinals,
    modes, scopes, and digests stay visible so a repr is still worth having.
    """

    slots = [
        {
            "name": "style",
            "mode": "fixed",
            "variation_scope": "batch",
            "fixed_value": "SECRET-FIXED",
        },
        {"name": "subject", "mode": "input", "variation_scope": "batch"},
        {
            "name": "mood",
            "mode": "choice",
            "variation_scope": "item",
            "choices": ["SECRET-CHOICE", "other-choice"],
        },
        {
            "name": "extra",
            "mode": "model",
            "variation_scope": "item",
            "guidance": "SECRET-GUIDANCE",
        },
    ]
    marked = _contract(body="{{style}} {{subject}} {{mood}} {{extra}} SECRET-BODY.", slots=slots)
    pending = expand_prompt_template(
        marked, _request(marked, inputs={"subject": "SECRET-INPUT"}, item_count=2)
    )
    plain = _contract()
    complete = expand_prompt_template(plain, _request(plain, inputs={"subject": "SECRET-INPUT"}))

    markers = (
        "SECRET-FIXED",
        "SECRET-CHOICE",
        "SECRET-GUIDANCE",
        "SECRET-BODY",
        "SECRET-INPUT",
        "oil",
        "calm",
        "stormy",
    )
    surfaces = [
        repr(pending),
        repr(complete),
        repr(complete.request),
        repr(pending.request),
        repr(complete.items[0]),
        repr(complete.items[0].evidence[0]),
        repr(pending.items[0].evidence),
        repr(pending.pending_model_slots[0]),
        repr(pending.pending_model_slots),
        str(pending),
        str(complete),
        f"{complete}",
        f"{complete!r}",
    ]
    for surface in surfaces:
        for marker in markers:
            assert marker not in surface, f"{marker} leaked into {surface[:120]}"

    # Not vacuous: the useful, non-private parts are still there.
    shown = repr(complete.items[0])
    assert "ordinal=1" in shown
    assert "style" in shown
    assert repr(complete.request).count("definition_id") == 1


# --- a template with no slots is a valid shape -------------------------------


def _slotless_contract():
    return _contract(body="A fixed authored prompt.", slots=[])


def test_a_slotless_template_produces_one_draft_with_a_valid_receipt() -> None:
    """Phase 1 permits a fixed literal template and the editor displays it.

    The canonical receipt refused this shape outright, because it required each
    item to carry at least one piece of evidence. A template with no slots has
    nothing to record, and that is not the same as a missing record.
    """

    contract = _slotless_contract()
    plan = expand_prompt_template(contract, _request(contract, inputs={}, item_count=1))
    assert plan.complete
    assert plan.items[0].evidence == ()
    assert plan.items[0].rendered_prompt == "A fixed authored prompt."
    digest = expansion_plan_digest(plan)
    assert len(digest) == 64
    payload = expansion_plan_payload(plan)
    assert payload["items"][0]["evidence"] == []


def test_a_slotless_template_still_refuses_a_second_item() -> None:
    """For the separate and correct reason: no value space, no second draft."""

    contract = _slotless_contract()
    with pytest.raises(PromptExpansionError):
        expand_prompt_template(contract, _request(contract, inputs={}, item_count=2))


# --- the reading half: a stored receipt is revalidated, not deserialised -----


def _receipt(plan) -> dict:
    """A receipt as a store would hold it: plain JSON with no dataclass behind it."""

    return json.loads(json.dumps(expansion_plan_payload(plan)))


def test_every_receipt_shape_reparses_to_itself_and_agrees_on_its_digest() -> None:
    contract = _contract()
    complete = expand_prompt_template(contract, _request(contract))
    pending_contract = _pending_contract()
    pending = expand_prompt_template(pending_contract, _request(pending_contract))
    slotless_contract = _slotless_contract()
    slotless = expand_prompt_template(
        slotless_contract, _request(slotless_contract, inputs={}, item_count=1)
    )

    for plan in (complete, pending, slotless):
        stored = _receipt(plan)
        assert parse_expansion_plan_payload(stored) == expansion_plan_payload(plan)
        assert expansion_plan_payload_digest(stored) == expansion_plan_digest(plan)


def _bend(plan, mutate):
    stored = _receipt(plan)
    mutate(stored)
    return stored


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.__setitem__("surprise", 1),
        lambda r: r.pop("template_body"),
        lambda r: r.pop("pending_model_slots"),
        lambda r: r.pop("items"),
        lambda r: r.__setitem__("codec_version", 3),
        lambda r: r.__setitem__("codec_version", "1"),
        lambda r: r.__setitem__("codec_version", True),
        lambda r: r["request"].__setitem__("item_count", 2),
        lambda r: r["request"].__setitem__("selection_seed", -1),
        lambda r: r["request"].pop("inputs"),
        lambda r: r["items"][0].__setitem__("ordinal", 7),
        lambda r: r["items"][0].__setitem__("rendered_sha256", "nope"),
        lambda r: r["items"][0].__setitem__("rendered_sha256", None),
        lambda r: r["items"][0]["evidence"][0].__setitem__("source", "model"),
        lambda r: r["items"][0]["evidence"][0].__setitem__("mode", "not-a-mode"),
        lambda r: r["items"][0]["evidence"][0].__setitem__("value", None),
        lambda r: r["items"][0]["evidence"][0].__setitem__("choice_index", -1),
        lambda r: r["items"][0]["evidence"].append(r["items"][0]["evidence"][0]),
        lambda r: r["items"].append(r["items"][0]),
        lambda r: r["items"].clear(),
        lambda r: r["pending_model_slots"].append(
            {"name": "ghost", "variation_scope": "item", "guidance": "x"}
        ),
    ],
)
def test_a_tampered_receipt_refuses_on_the_way_back_in(mutate) -> None:
    contract = _contract()
    plan = expand_prompt_template(contract, _request(contract))
    with pytest.raises(PromptExpansionError) as caught:
        expansion_plan_payload_digest(_bend(plan, mutate))
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_a_rewritten_rendered_digest_refuses_at_the_reader_boundary() -> None:
    contract = _contract()
    plan = expand_prompt_template(contract, _request(contract))
    bent = _bend(plan, lambda r: r["items"][0].__setitem__("rendered_sha256", "0" * 64))
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(bent)


def _rewrite_receipt_prompt(item: dict, prompt: str) -> None:
    item["rendered_prompt"] = prompt
    material = "\x00".join(("prompt-expansion-rendered-v1", prompt))
    item["rendered_sha256"] = hashlib.sha256(material.encode("utf-8")).hexdigest()


def test_receipt_refuses_evidence_value_not_present_in_its_own_prompt() -> None:
    contract = _contract()
    plan = expand_prompt_template(contract, _request(contract))
    bent = _receipt(plan)
    bent["items"][0]["evidence"][0]["value"] = "ink"
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(bent)

    fixed = plan.items[0].evidence[0]
    object.__setattr__(fixed, "value", "ink")
    with pytest.raises(PromptExpansionError):
        expansion_plan_digest(plan)


def test_receipt_refuses_coherent_batch_value_drift_between_items() -> None:
    contract = _contract()
    bent = _receipt(expand_prompt_template(contract, _request(contract)))
    second = bent["items"][1]
    second["evidence"][0]["value"] = "ink"
    _rewrite_receipt_prompt(second, second["rendered_prompt"].replace("oil", "ink", 1))
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(bent)


def test_receipt_refuses_reordered_evidence_and_input_substring_rewrites() -> None:
    contract = _contract()
    plan = expand_prompt_template(contract, _request(contract))

    reordered = _receipt(plan)
    reordered["items"][0]["evidence"].reverse()
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(reordered)

    rewritten = _receipt(plan)
    rewritten["items"][0]["evidence"][1]["value"] = "fox"
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(rewritten)


def test_completed_model_evidence_is_bound_to_each_rendered_prompt() -> None:
    contract, pending = _bridge_plan()
    completed = complete_prompt_expansion_with_model_values(
        contract, pending, _bridge_values(contract)
    )

    changed = _receipt(completed)
    changed["items"][0]["evidence"][2]["value"] = "charcoal"
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(changed)

    swapped = _receipt(completed)
    first = swapped["items"][0]["evidence"][4]
    second = swapped["items"][1]["evidence"][4]
    first["value"], second["value"] = second["value"], first["value"]
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(swapped)


def test_receipt_rendering_accepts_static_text_equal_to_overlapping_values() -> None:
    slots = [
        {"name": "left", "mode": "input", "variation_scope": "batch"},
        {"name": "right", "mode": "input", "variation_scope": "batch"},
    ]
    contract = _contract(body="ab {{left}} then {{right}} beside ab", slots=slots)
    plan = expand_prompt_template(
        contract,
        _request(contract, item_count=1, inputs={"left": "a", "right": "ab"}),
    )
    assert parse_expansion_plan_payload(_receipt(plan)) == expansion_plan_payload(plan)


def test_pending_receipt_binds_evidence_order_to_the_recorded_body() -> None:
    contract = _pending_contract()
    pending = expand_prompt_template(contract, _request(contract))
    bent = _receipt(pending)
    bent["items"][0]["evidence"].reverse()
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(bent)


def test_pending_receipt_binds_pending_model_slot_order() -> None:
    slots = [
        *_SLOTS,
        {"name": "first", "mode": "model", "variation_scope": "batch", "guidance": "x"},
        {"name": "second", "mode": "model", "variation_scope": "item", "guidance": "y"},
    ]
    contract = _contract(
        body=_BODY[:-1] + " with {{first}} and {{second}}.",
        slots=slots,
    )
    plan = expand_prompt_template(contract, _request(contract, item_count=2))
    bent = _receipt(plan)
    bent["pending_model_slots"].reverse()
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(bent)


def test_a_receipt_whose_key_is_a_str_subclass_refuses() -> None:
    """A `str` subclass hashes and compares equal, so a closed-key set alone
    would admit it."""

    contract = _contract()
    plan = expand_prompt_template(contract, _request(contract))
    stored = _receipt(plan)
    sneaky = type("SneakyKey", (str,), {})
    stored[sneaky("codec_version")] = stored.pop("codec_version")
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(stored)


@pytest.mark.parametrize("receipt", ["not a dict", 5, None, [], (), {"codec_version": 1}])
def test_a_malformed_receipt_refuses(receipt: object) -> None:
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(receipt)


def test_receipt_refusals_echo_nothing_from_the_template() -> None:
    slots = [
        {
            "name": "style",
            "mode": "fixed",
            "variation_scope": "batch",
            "fixed_value": "SECRET-FIXED",
        },
        {"name": "subject", "mode": "input", "variation_scope": "batch"},
        {
            "name": "mood",
            "mode": "choice",
            "variation_scope": "item",
            "choices": ["SECRET-CHOICE", "other-choice"],
        },
    ]
    contract = _contract(body="{{style}} {{subject}} {{mood}} SECRET-BODY.", slots=slots)
    plan = expand_prompt_template(
        contract, _request(contract, inputs={"subject": "SECRET-INPUT"}, item_count=2)
    )
    bent = _bend(plan, lambda r: r["items"][0].__setitem__("ordinal", 9))
    with pytest.raises(PromptExpansionError) as caught:
        expansion_plan_payload_digest(bent)
    message = str(caught.value)
    assert message == PROMPT_EXPANSION_INVALID
    for marker in ("SECRET-FIXED", "SECRET-CHOICE", "SECRET-BODY", "SECRET-INPUT"):
        assert marker not in message


# --- every caller-controlled receipt collection is bounded -----------------


def _model_only_contract(slot_count: int = MAX_EXPANSION_INPUT_SLOTS):
    slots = [
        {
            "name": f"model_{index}",
            "mode": "model",
            "variation_scope": "item",
            "guidance": f"detail {index}",
        }
        for index in range(slot_count)
    ]
    body = " ".join(f"{{{{model_{index}}}}}" for index in range(slot_count))
    return _contract(body=body, slots=slots)


def test_request_inputs_and_vectors_accept_their_caps() -> None:
    inputs = {f"slot_{index}": "value" for index in range(MAX_EXPANSION_INPUT_SLOTS)}
    request = _request(inputs=inputs, item_count=1)
    assert len(expansion_request_payload(request)["inputs"]) == MAX_EXPANSION_INPUT_SLOTS

    vector = tuple(f"value-{index}" for index in range(MAX_EXPANSION_ITEMS))
    vector_request = _request(inputs={"slot": list(vector)}, item_count=MAX_EXPANSION_ITEMS)
    assert expansion_request_payload(vector_request)["inputs"]["slot"] == list(vector)


def test_request_inputs_and_vectors_refuse_cap_plus_one_before_copying() -> None:
    request = _request(item_count=1)
    too_many = tuple((f"slot_{index}", "value") for index in range(MAX_EXPANSION_INPUT_SLOTS + 1))
    object.__setattr__(request, "inputs", too_many)
    with pytest.raises(PromptExpansionError):
        expansion_request_payload(request)

    vector_request = _request(item_count=MAX_EXPANSION_ITEMS)
    object.__setattr__(
        vector_request,
        "inputs",
        (("subject", tuple("value" for _ in range(MAX_EXPANSION_ITEMS + 1))),),
    )
    with pytest.raises(PromptExpansionError):
        expansion_request_payload(vector_request)


@pytest.mark.parametrize("kind", ["inputs", "vector"])
def test_stored_request_inputs_and_vectors_refuse_cap_plus_one(kind: str) -> None:
    _unused, plan = _finished_plan()
    stored = _receipt(plan)
    if kind == "inputs":
        stored["request"]["inputs"] = {
            f"slot_{index}": "value" for index in range(MAX_EXPANSION_INPUT_SLOTS + 1)
        }
    else:
        stored["request"]["item_count"] = MAX_EXPANSION_ITEMS
        stored["request"]["inputs"]["subject"] = ["value" for _ in range(MAX_EXPANSION_ITEMS + 1)]
    with pytest.raises(PromptExpansionError):
        expansion_plan_payload_digest(stored)


def test_pending_and_evidence_collections_accept_their_caps_on_both_boundaries() -> None:
    contract = _model_only_contract()
    plan = expand_prompt_template(contract, _request(contract, inputs={}, item_count=1))
    assert len(plan.pending_model_slots) == MAX_EXPANSION_INPUT_SLOTS
    assert len(plan.items[0].evidence) == MAX_EXPANSION_INPUT_SLOTS
    payload = expansion_plan_payload(plan)
    assert parse_expansion_plan_payload(_receipt(plan)) == payload


@pytest.mark.parametrize(
    "boundary", ["dataclass-pending", "dataclass-evidence", "json-pending", "json-evidence"]
)
def test_pending_and_evidence_collections_refuse_cap_plus_one(boundary: str) -> None:
    contract = _model_only_contract()
    plan = expand_prompt_template(contract, _request(contract, inputs={}, item_count=1))
    if boundary == "dataclass-pending":
        object.__setattr__(
            plan,
            "pending_model_slots",
            (*plan.pending_model_slots, plan.pending_model_slots[0]),
        )
        candidate: object = plan
        call = expansion_plan_digest
    elif boundary == "dataclass-evidence":
        object.__setattr__(
            plan.items[0],
            "evidence",
            (*plan.items[0].evidence, plan.items[0].evidence[0]),
        )
        candidate = plan
        call = expansion_plan_digest
    else:
        stored = _receipt(plan)
        if boundary == "json-pending":
            stored["pending_model_slots"].append(stored["pending_model_slots"][0])
        else:
            stored["items"][0]["evidence"].append(stored["items"][0]["evidence"][0])
        candidate = stored
        call = expansion_plan_payload_digest
    with pytest.raises(PromptExpansionError):
        call(candidate)


class _PrivateTrap:
    def __eq__(self, _other: object) -> bool:
        raise RuntimeError("PRIVATE-EQUALITY-TRAP")

    def __hash__(self) -> int:
        raise RuntimeError("PRIVATE-HASH-TRAP")

    def __repr__(self) -> str:
        raise RuntimeError("PRIVATE-REPR-TRAP")


class _ArmedStr(str):
    armed = False

    def __hash__(self) -> int:
        if self.armed:
            raise RuntimeError("PRIVATE-HASH-TRAP")
        return super().__hash__()

    def __repr__(self) -> str:
        raise RuntimeError("PRIVATE-REPR-TRAP")


def test_a_forged_request_is_validated_before_plan_count_equality() -> None:
    _unused, plan = _finished_plan()
    object.__setattr__(plan.request, "item_count", _PrivateTrap())
    with pytest.raises(PromptExpansionError) as caught:
        expansion_plan_digest(plan)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_expand_also_validates_a_forged_request_before_nested_equality() -> None:
    contract = _contract()
    request = _request(contract)
    object.__setattr__(request, "contract_sha256", _PrivateTrap())
    with pytest.raises(PromptExpansionError) as caught:
        expand_prompt_template(contract, request)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_request_input_keys_are_typed_before_hash_or_repr() -> None:
    request = _request(item_count=1)
    object.__setattr__(request, "inputs", ((_PrivateTrap(), "value"),))
    with pytest.raises(PromptExpansionError) as caught:
        expansion_request_payload(request)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_stored_keys_are_typed_before_hash_or_repr() -> None:
    _unused, plan = _finished_plan()
    stored = _receipt(plan)
    key = _ArmedStr("codec_version")
    stored[key] = stored.pop("codec_version")
    key.armed = True
    with pytest.raises(PromptExpansionError) as caught:
        expansion_plan_payload_digest(stored)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_large_wrong_key_records_refuse_with_the_fixed_error() -> None:
    _unused, plan = _finished_plan()
    builders = [
        lambda: {f"unknown_{index}": _PrivateTrap() for index in range(12_000)},
        lambda: _bend(
            plan,
            lambda receipt: receipt.__setitem__(
                "request", {f"unknown_{index}": _PrivateTrap() for index in range(12_000)}
            ),
        ),
        lambda: _bend(
            plan,
            lambda receipt: receipt.__setitem__(
                "pending_model_slots",
                [{f"unknown_{index}": _PrivateTrap() for index in range(12_000)}],
            ),
        ),
        lambda: _bend(
            plan,
            lambda receipt: receipt["items"].__setitem__(
                0, {f"unknown_{index}": _PrivateTrap() for index in range(12_000)}
            ),
        ),
        lambda: _bend(
            plan,
            lambda receipt: receipt["items"][0].__setitem__(
                "evidence", [{f"unknown_{index}": _PrivateTrap() for index in range(12_000)}]
            ),
        ),
    ]
    for build in builders:
        with pytest.raises(PromptExpansionError) as caught:
            expansion_plan_payload_digest(build())
        assert str(caught.value) == PROMPT_EXPANSION_INVALID


# --- a receipt describes one possible slot contract across every item -------


@pytest.mark.parametrize("boundary", ["dataclass", "json"])
@pytest.mark.parametrize("field", ["mode", "variation_scope"])
def test_one_slot_cannot_change_mode_or_scope_between_items(boundary: str, field: str) -> None:
    _unused, plan = _finished_plan()
    if boundary == "dataclass":
        if field == "mode":
            entry = plan.items[1].evidence[0]
            object.__setattr__(entry, "mode", PromptTemplateSlotMode.CHOICE)
            object.__setattr__(entry, "source", ExpansionValueSource.CHOICE)
            object.__setattr__(entry, "choice_index", 0)
        else:
            entry = plan.items[1].evidence[2]
            object.__setattr__(entry, "variation_scope", PromptTemplateVariationScope.BATCH)
        candidate: object = plan
        call = expansion_plan_digest
    else:
        stored = _receipt(plan)
        if field == "mode":
            entry = stored["items"][1]["evidence"][0]
            entry["mode"] = "choice"
            entry["source"] = "choice"
            entry["choice_index"] = 0
        else:
            stored["items"][1]["evidence"][2]["variation_scope"] = "batch"
        candidate = stored
        call = expansion_plan_payload_digest
    with pytest.raises(PromptExpansionError):
        call(candidate)


@pytest.mark.parametrize("boundary", ["dataclass", "json"])
def test_pending_model_scope_must_match_every_items_evidence(boundary: str) -> None:
    plan = _pending_plan()
    if boundary == "dataclass":
        object.__setattr__(
            plan.pending_model_slots[0],
            "variation_scope",
            PromptTemplateVariationScope.BATCH,
        )
        candidate: object = plan
        call = expansion_plan_digest
    else:
        stored = _receipt(plan)
        stored["pending_model_slots"][0]["variation_scope"] = "batch"
        candidate = stored
        call = expansion_plan_payload_digest
    with pytest.raises(PromptExpansionError):
        call(candidate)


@pytest.mark.parametrize("boundary", ["dataclass", "json"])
def test_fixed_slots_cannot_claim_item_scope_without_echoing_values(boundary: str) -> None:
    _unused, plan = _finished_plan()
    if boundary == "dataclass":
        for item in plan.items:
            entry = item.evidence[0]
            object.__setattr__(entry, "variation_scope", PromptTemplateVariationScope.ITEM)
            object.__setattr__(entry, "value", "SECRET-FIXED-VALUE")
        candidate: object = plan
        call = expansion_plan_digest
    else:
        stored = _receipt(plan)
        for item in stored["items"]:
            item["evidence"][0]["variation_scope"] = "item"
            item["evidence"][0]["value"] = "SECRET-FIXED-VALUE"
        candidate = stored
        call = expansion_plan_payload_digest
    with pytest.raises(PromptExpansionError) as caught:
        call(candidate)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID
    assert "SECRET-FIXED-VALUE" not in str(caught.value)


@pytest.mark.parametrize("boundary", ["dataclass", "json"])
def test_choice_index_accepts_the_phase_one_maximum_minus_one(boundary: str) -> None:
    _unused, plan = _finished_plan()
    if boundary == "dataclass":
        for item in plan.items:
            object.__setattr__(item.evidence[2], "choice_index", MAX_TEMPLATE_CHOICES - 1)
        assert len(expansion_plan_digest(plan)) == 64
    else:
        stored = _receipt(plan)
        for item in stored["items"]:
            item["evidence"][2]["choice_index"] = MAX_TEMPLATE_CHOICES - 1
        assert len(expansion_plan_payload_digest(stored)) == 64


@pytest.mark.parametrize("boundary", ["dataclass", "json"])
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(MAX_TEMPLATE_CHOICES, id="cap"),
        pytest.param(10**5_000, id="huge"),
    ],
)
def test_choice_index_refuses_the_cap_and_huge_integers_with_the_fixed_error(
    boundary: str, index: int
) -> None:
    _unused, plan = _finished_plan()
    if boundary == "dataclass":
        object.__setattr__(plan.items[0].evidence[2], "choice_index", index)
        candidate: object = plan
        call = expansion_plan_digest
    else:
        stored = _receipt(plan)
        stored["items"][0]["evidence"][2]["choice_index"] = index
        candidate = stored
        call = expansion_plan_payload_digest
    with pytest.raises(PromptExpansionError) as caught:
        call(candidate)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


# --- pure model-values <-> expansion bridge ----------------------------------


def test_model_values_complete_batch_and_item_slots_without_mutating_inputs() -> None:
    contract, plan = _bridge_plan()
    values = _bridge_values(contract)
    before_plan = expansion_plan_digest(plan)
    before_contract = prompt_template_contract_sha256(contract)
    model_contract = prompt_model_slot_contract(contract, item_count=2)
    before_values = prompt_model_values_payload(values, contract=model_contract)

    completed = complete_prompt_expansion_with_model_values(contract, plan, values)

    assert completed.complete
    assert plan.complete is False
    assert all(item.rendered_prompt is None for item in plan.items)
    assert expansion_plan_digest(plan) == before_plan
    assert prompt_template_contract_sha256(contract) == before_contract
    assert prompt_model_values_payload(values, contract=model_contract) == before_values
    assert _values(completed, "medium") == ["tempera", "tempera"]
    assert _values(completed, "detail") == ["silver leaves", "amber rain"]
    assert len(set(_prompts(completed))) == 2

    for original, result in zip(plan.items, completed.items, strict=True):
        assert tuple(
            entry for entry in result.evidence if entry.mode is not PromptTemplateSlotMode.MODEL
        ) == tuple(
            entry for entry in original.evidence if entry.mode is not PromptTemplateSlotMode.MODEL
        )


def test_completed_model_evidence_has_a_stable_canonical_receipt() -> None:
    contract, plan = _bridge_plan()
    values = _bridge_values(contract)
    first = complete_prompt_expansion_with_model_values(contract, plan, values)
    second = complete_prompt_expansion_with_model_values(contract, plan, values)

    payload = expansion_plan_payload(first)
    assert parse_expansion_plan_payload(copy.deepcopy(payload)) == payload
    assert expansion_plan_payload_digest(payload) == expansion_plan_digest(first)
    assert expansion_plan_digest(first) == expansion_plan_digest(second)
    assert payload["pending_model_slots"] == []


def test_invocation_data_contains_only_literal_template_and_resolved_values() -> None:
    contract, plan = _bridge_plan()
    data = prompt_model_invocation_data(contract, plan)

    assert data.template_text == contract.body
    assert data.batch_values == (("style", "oil"), ("subject", "a fox"))
    assert [item.ordinal for item in data.items] == [1, 2]
    assert all(tuple(name for name, _value in item.values) == ("mood",) for item in data.items)
    assert len({item.values[0][1] for item in data.items}) == 2
    assert "one medium" not in repr(data)
    assert "one visible detail" not in repr(data)
    assert contract.body not in repr(data)
    assert "a fox" not in repr(data)


def test_completion_refuses_a_complete_plan_and_a_contract_without_model_slots() -> None:
    contract, plan = _bridge_plan()
    values = _bridge_values(contract)
    completed = complete_prompt_expansion_with_model_values(contract, plan, values)
    with pytest.raises(PromptExpansionError):
        complete_prompt_expansion_with_model_values(contract, completed, values)

    deterministic = _contract()
    deterministic_plan = expand_prompt_template(
        deterministic,
        _request(deterministic, item_count=1),
    )
    with pytest.raises(PromptExpansionError):
        complete_prompt_expansion_with_model_values(
            deterministic,
            deterministic_plan,
            PromptModelValues(version=1, batch_values=(), items=()),
        )


@pytest.mark.parametrize("defect", ["missing", "extra", "wrong_scope", "ordinal"])
def test_completion_refuses_forged_model_value_shapes(defect: str) -> None:
    contract, plan = _bridge_plan()
    values = _bridge_values(contract)
    if defect == "missing":
        object.__setattr__(values, "batch_values", ())
    elif defect == "extra":
        object.__setattr__(
            values,
            "batch_values",
            (*values.batch_values, ("undeclared", "SECRET-EXTRA")),
        )
    elif defect == "wrong_scope":
        object.__setattr__(values, "batch_values", ())
        for item in values.items:
            object.__setattr__(item, "values", (("medium", "SECRET-WRONG-SCOPE"), *item.values))
    else:
        object.__setattr__(values.items[0], "ordinal", 2)

    with pytest.raises(PromptExpansionError) as caught:
        complete_prompt_expansion_with_model_values(contract, plan, values)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID
    assert "SECRET" not in str(caught.value)


def test_completion_binds_contract_request_pending_and_deterministic_evidence() -> None:
    contract, plan = _bridge_plan()
    values = _bridge_values(contract)

    drifted_slots = [
        {
            "name": slot.name,
            "mode": slot.mode.value,
            "variation_scope": slot.variation_scope.value,
            **(
                {"fixed_value": slot.fixed_value}
                if slot.mode is PromptTemplateSlotMode.FIXED
                else {"choices": list(slot.choices)}
                if slot.mode is PromptTemplateSlotMode.CHOICE
                else {"guidance": "different guidance"}
                if slot.mode is PromptTemplateSlotMode.MODEL
                else {}
            ),
        }
        for slot in contract.slots
    ]
    drifted_contract = _contract(body=contract.body, slots=drifted_slots)
    with pytest.raises(PromptExpansionError):
        complete_prompt_expansion_with_model_values(drifted_contract, plan, values)

    contract, plan = _bridge_plan()
    object.__setattr__(plan.request, "inputs", (("subject", "a hare"),))
    with pytest.raises(PromptExpansionError):
        complete_prompt_expansion_with_model_values(contract, plan, _bridge_values(contract))

    contract, plan = _bridge_plan()
    fixed = next(entry for entry in plan.items[0].evidence if entry.name == "style")
    object.__setattr__(fixed, "value", "SECRET-EVIDENCE-DRIFT")
    with pytest.raises(PromptExpansionError) as caught:
        complete_prompt_expansion_with_model_values(contract, plan, _bridge_values(contract))
    assert "SECRET-EVIDENCE-DRIFT" not in str(caught.value)


def test_completion_and_invocation_refuse_duplicate_pending_slots() -> None:
    contract, plan = _bridge_plan()
    object.__setattr__(
        plan,
        "pending_model_slots",
        (*plan.pending_model_slots, plan.pending_model_slots[0]),
    )
    for call in (
        lambda: complete_prompt_expansion_with_model_values(
            contract,
            plan,
            _bridge_values(contract),
        ),
        lambda: prompt_model_invocation_data(contract, plan),
    ):
        with pytest.raises(PromptExpansionError):
            call()


def test_completion_refuses_non_distinct_rendered_prompts() -> None:
    contract = _contract(
        body="{{detail}}",
        slots=[
            {
                "name": "detail",
                "mode": "model",
                "variation_scope": "item",
                "guidance": "one detail",
            }
        ],
    )
    plan = expand_prompt_template(
        contract,
        _request(contract, item_count=2, inputs={}),
    )
    model_contract = prompt_model_slot_contract(contract, item_count=2)
    values = parse_prompt_model_values(
        {
            "version": 1,
            "batch_values": {},
            "items": [
                {"ordinal": 1, "values": {"detail": "same"}},
                {"ordinal": 2, "values": {"detail": "same"}},
            ],
        },
        contract=model_contract,
    )
    with pytest.raises(PromptExpansionError):
        complete_prompt_expansion_with_model_values(contract, plan, values)


def test_completion_allows_repeated_prompts_when_a_choice_explicitly_allows_reuse() -> None:
    contract = _contract(
        body="{{choice}} {{detail}}",
        slots=[
            {
                "name": "choice",
                "mode": "choice",
                "variation_scope": "item",
                "choices": ["same"],
                "choice_strategy": "with_replacement",
            },
            {
                "name": "detail",
                "mode": "model",
                "variation_scope": "item",
                "guidance": "one detail",
            },
        ],
    )
    plan = expand_prompt_template(contract, _request(contract, item_count=2, inputs={}))
    model_contract = prompt_model_slot_contract(contract, item_count=2)
    values = parse_prompt_model_values(
        {
            "version": 1,
            "batch_values": {},
            "items": [
                {"ordinal": 1, "values": {"detail": "detail"}},
                {"ordinal": 2, "values": {"detail": "detail"}},
            ],
        },
        contract=model_contract,
    )
    completed = complete_prompt_expansion_with_model_values(contract, plan, values)
    assert _prompts(completed) == ["same detail", "same detail"]


class _ExplodingTuple(tuple):
    def __iter__(self):
        raise AssertionError("must refuse this tuple subclass before traversal")


def test_bridge_preflights_caps_and_tuple_subclasses_before_traversal() -> None:
    contract, plan = _bridge_plan()
    values = _bridge_values(contract)
    object.__setattr__(plan, "items", _ExplodingTuple(plan.items))
    with pytest.raises(PromptExpansionError):
        complete_prompt_expansion_with_model_values(contract, plan, values)

    contract, plan = _bridge_plan()
    object.__setattr__(plan, "items", plan.items * (MAX_EXPANSION_ITEMS + 1))
    with pytest.raises(PromptExpansionError):
        prompt_model_invocation_data(contract, plan)

    contract, plan = _bridge_plan()
    object.__setattr__(contract, "slots", contract.slots * (MAX_EXPANSION_INPUT_SLOTS + 1))
    with pytest.raises(PromptExpansionError):
        prompt_model_invocation_data(contract, plan)

    contract, plan = _bridge_plan()
    choice = next(slot for slot in contract.slots if slot.mode is PromptTemplateSlotMode.CHOICE)
    object.__setattr__(choice, "choices", choice.choices * (MAX_TEMPLATE_CHOICES + 1))
    with pytest.raises(PromptExpansionError):
        prompt_model_invocation_data(contract, plan)

    contract, plan = _bridge_plan()
    values = _bridge_values(contract)
    object.__setattr__(
        values,
        "items",
        values.items * (MAX_EXPANSION_ITEMS + 1),
    )
    with pytest.raises(PromptExpansionError):
        complete_prompt_expansion_with_model_values(contract, plan, values)


def test_bridge_refuses_dataclass_subclasses_with_the_fixed_error() -> None:
    contract, plan = _bridge_plan()
    values = _bridge_values(contract)

    class SneakyContract(PromptTemplateContract):
        pass

    class SneakyPlan(ExpansionPlan):
        pass

    class SneakyValues(PromptModelValues):
        pass

    sneaky_contract = SneakyContract(
        schema_version=contract.schema_version,
        operation=contract.operation,
        body=contract.body,
        slots=contract.slots,
        resource_policy=contract.resource_policy,
    )
    sneaky_plan = SneakyPlan(
        codec_version=plan.codec_version,
        request=plan.request,
        template_body=plan.template_body,
        items=plan.items,
        pending_model_slots=plan.pending_model_slots,
    )
    sneaky = SneakyValues(
        version=values.version,
        batch_values=values.batch_values,
        items=values.items,
    )
    for authority_contract, authority_plan, authority_values in (
        (sneaky_contract, plan, values),
        (contract, sneaky_plan, values),
        (contract, plan, sneaky),
    ):
        with pytest.raises(PromptExpansionError) as caught:
            complete_prompt_expansion_with_model_values(
                authority_contract,
                authority_plan,
                authority_values,
            )
        assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_bridge_normalizes_a_forged_contract_encoding_failure() -> None:
    contract, plan = _bridge_plan()
    object.__setattr__(contract, "body", chr(0xD800))
    with pytest.raises(PromptExpansionError) as caught:
        prompt_model_invocation_data(contract, plan)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID


def test_invocation_data_requires_the_exact_authored_pending_contract() -> None:
    contract, plan = _bridge_plan()
    object.__setattr__(plan.pending_model_slots[0], "guidance", "forged guidance")
    with pytest.raises(PromptExpansionError) as caught:
        prompt_model_invocation_data(contract, plan)
    assert str(caught.value) == PROMPT_EXPANSION_INVALID
    assert "forged guidance" not in str(caught.value)
