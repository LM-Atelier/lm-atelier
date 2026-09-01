"""Turning one exact Prompt Library revision into N bounded drafts, purely.

This module stops short of everything that could have an effect. It allocates
deterministic values, records where each value came from, and renders literal
prompts. It does not read a database, call a model, admit media, or touch a
graph; the persistence and service boundary is a separate slice, and the
`model` slot boundary is defined here but never invoked.

Two decisions are worth stating because they are easy to undo by accident.

**Selection randomness is its own domain.** The seed that picks a choice value
is never the seed that samples an image. They are recorded separately and
derived separately, so reading one from the other would be a category error
rather than a shortcut - a batch whose prompts differ must be able to share a
media seed, and a batch whose prompts are identical must be able to differ.

**Variation scope decides how many times a value is drawn, not what it is.** A
batch-scoped slot draws once and every item sees that value; an item-scoped slot
draws per item. Both draws are pure functions of the recorded seed, so a batch
replayed from its frozen request produces byte-identical values without storing
them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NoReturn, cast

from .prompt_model_values import (
    PromptModelItemValues,
    PromptModelSlotContract,
    PromptModelValues,
    PromptModelValuesError,
    parse_prompt_model_values,
    prompt_model_slot_contract,
    prompt_model_values_payload,
)
from .prompt_templates import (
    MAX_TEMPLATE_BODY_CHARS,
    MAX_TEMPLATE_CHOICES,
    MAX_TEMPLATE_GUIDANCE_CHARS,
    MAX_TEMPLATE_LORAS,
    MAX_TEMPLATE_RENDERED_CHARS,
    MAX_TEMPLATE_SLOTS,
    MAX_TEMPLATE_TOTAL_CHOICES,
    MAX_TEMPLATE_VALUE_CHARS,
    PromptTemplateChoiceStrategy,
    PromptTemplateContract,
    PromptTemplateError,
    PromptTemplateLoraPolicy,
    PromptTemplateResourcePolicy,
    PromptTemplateSlot,
    PromptTemplateSlotMode,
    PromptTemplateVariationScope,
    parse_prompt_template_contract,
    prompt_template_contract_payload,
    prompt_template_contract_sha256,
    render_prompt_template,
)

if TYPE_CHECKING:
    from .prompt_model_invocation import PromptModelInvocationData

PROMPT_EXPANSION_CODEC_VERSION = 2

MIN_EXPANSION_ITEMS = 1
MAX_EXPANSION_ITEMS = 16

#: The selection-seed domain. Deliberately NOT `MEDIA_SEED_SPACE`: importing that
#: constant here would tie two independent domains together at the type level and
#: invite a later change to one from silently redefining the other.
SELECTION_SEED_SPACE = 2_147_483_648

# Public compatibility name for the request codec. The template contract is
# the single authority for how many slots an expansion can describe.
MAX_EXPANSION_INPUT_SLOTS = MAX_TEMPLATE_SLOTS

#: How large an item-scoped choice space may be before the codec refuses to
#: enumerate it. Enumeration is what makes "cannot yield N distinct" an exact
#: answer rather than a guess, and an unbounded product is how that becomes a
#: hang instead of a refusal.
MAX_ITEM_CHOICE_SPACE = 4_096

#: Bounds on the material `expansion_selection_seed` will hash, so a caller
#: cannot turn seed derivation into unbounded work.
MAX_SEED_MATERIAL_ENTRIES = 32
MAX_SEED_MATERIAL_CHARS = 4_096
MAX_SEED_MATERIAL_BYTES = 16_384

PROMPT_EXPANSION_INVALID = "Prompt expansion request is invalid."

_PLAN_PAYLOAD_KEYS = frozenset(
    {"codec_version", "request", "template_body", "pending_model_slots", "items"}
)
_PLAN_ITEM_KEYS = frozenset({"ordinal", "rendered_prompt", "rendered_sha256", "evidence"})
_PENDING_SLOT_KEYS = frozenset({"name", "variation_scope", "guidance"})
_EVIDENCE_KEYS = frozenset({"name", "mode", "variation_scope", "source", "value", "choice_index"})

_REQUEST_KEYS = frozenset(
    {
        "definition_id",
        "revision_id",
        "contract_sha256",
        "item_count",
        "selection_seed",
        "inputs",
    }
)
_RECEIPT_SLOT_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}", re.ASCII)
_RECEIPT_TOKEN = re.compile(r"{{([a-z][a-z0-9_]{0,63})}}", re.ASCII)


class PromptExpansionError(ValueError):
    """A refusal that never carries template text, values, or identifiers."""

    def __init__(self, message: str = PROMPT_EXPANSION_INVALID) -> None:
        super().__init__(message)


class PromptExpansionDistinctCapacityError(PromptExpansionError):
    """The authored deterministic value space cannot yield the requested count."""


def _invalid() -> NoReturn:
    raise PromptExpansionError()


def _validate_exact_keys(payload: dict[object, object], expected: frozenset[str]) -> None:
    """Refuse a closed record cheaply before inspecting any of its keys."""

    if len(payload) != len(expected):
        _invalid()
    for key in payload:
        if type(key) is not str:
            _invalid()
    if frozenset(payload) != expected:
        _invalid()


class ExpansionValueSource(StrEnum):
    """Where one rendered slot value came from.

    Recorded per slot per item so a draft can be explained after the fact
    without re-deriving it, and so a later `model` slot cannot be mistaken for a
    deterministic one when a batch is reviewed.
    """

    FIXED = "fixed"
    INPUT = "input"
    CHOICE = "choice"
    MODEL = "model"


@dataclass(frozen=True, slots=True)
class ExpansionRequest:
    """One frozen, already-validated instruction to expand a revision."""

    definition_id: str
    revision_id: str
    contract_sha256: str
    item_count: int
    selection_seed: int
    #: Author-supplied text. `repr=False` throughout this module for every field
    #: that can carry template body, guidance, choices, values, or a rendered
    #: prompt: a repr reaches logs, tracebacks, and debugger frames, none of
    #: which are places this content is allowed to appear. Identity, ordinals,
    #: modes, scopes, and digests stay visible, so a repr is still useful.
    inputs: tuple[tuple[str, str | tuple[str, ...]], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class SlotEvidence:
    """What one slot resolved to for one item, and how."""

    name: str
    mode: PromptTemplateSlotMode
    variation_scope: PromptTemplateVariationScope
    source: ExpansionValueSource
    value: str | None = field(repr=False)
    choice_index: int | None


@dataclass(frozen=True, slots=True)
class ModelSlotRequest:
    """What a later model boundary must supply for one slot.

    Carried so the structure is frozen now and the boundary that fills it cannot
    invent its own shape later. Nothing in this module invokes a model.
    """

    name: str
    variation_scope: PromptTemplateVariationScope
    guidance: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ExpandedItem:
    """One draft: its ordinal, its per-slot evidence, and its prompt if complete.

    `rendered_sha256` is the canonical identity of the rendered text.
    Persistence needs a stable digest per item, and deriving it here means the
    store records the codec's answer instead of inventing a second
    representation of the same thing.
    """

    ordinal: int
    evidence: tuple[SlotEvidence, ...]
    rendered_prompt: str | None = field(repr=False)
    rendered_sha256: str | None


@dataclass(frozen=True, slots=True)
class ExpansionPlan:
    """The whole deterministic result for one request."""

    codec_version: int
    request: ExpansionRequest
    template_body: str = field(repr=False)
    items: tuple[ExpandedItem, ...]
    pending_model_slots: tuple[ModelSlotRequest, ...]

    @property
    def complete(self) -> bool:
        """True when every item rendered, i.e. no model slot is outstanding."""

        return not self.pending_model_slots


def _exact_str(value: object, *, maximum: int) -> str:
    if type(value) is not str:
        _invalid()
    text = value
    if not text or len(text) > maximum:
        _invalid()
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        _invalid()
    return text


def _exact_int(value: object, *, minimum: int, maximum: int) -> int:
    # `type(...) is not int` rather than isinstance: bool is an int subclass and
    # accepting True as a count is exactly the coercion this contract forbids.
    if type(value) is not int:
        _invalid()
    number = value
    if not minimum <= number <= maximum:
        _invalid()
    return number


def parse_expansion_request(value: object) -> ExpansionRequest:
    """Validate one exact expansion payload into immutable data.

    Closed keys and exact types throughout. Nothing here is normalized or
    defaulted: a caller that omits a key is refused rather than guessed at,
    because a guessed count or a guessed seed both produce a batch nobody asked
    for.
    """

    if type(value) is not dict:
        _invalid()
    payload = cast(dict[object, object], value)
    # The count check is deliberately first: a huge wrong-key map is refused
    # without walking every caller-controlled key. Exact `str` keys still come
    # before the hash comparison, because a subclass can compare equal.
    _validate_exact_keys(payload, _REQUEST_KEYS)

    definition_id = _exact_str(payload["definition_id"], maximum=64)
    revision_id = _exact_str(payload["revision_id"], maximum=64)
    digest = _exact_str(payload["contract_sha256"], maximum=64)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _invalid()
    item_count = _exact_int(
        payload["item_count"], minimum=MIN_EXPANSION_ITEMS, maximum=MAX_EXPANSION_ITEMS
    )
    selection_seed = _exact_int(
        payload["selection_seed"], minimum=0, maximum=SELECTION_SEED_SPACE - 1
    )

    raw_inputs = payload["inputs"]
    if type(raw_inputs) is not dict:
        _invalid()
    inputs = cast(dict[object, object], raw_inputs)
    if len(inputs) > MAX_EXPANSION_INPUT_SLOTS:
        _invalid()
    # No separate exact-key loop here: `_exact_str(raw_name)` below already
    # refuses a `str` subclass, so a second check would be dead code. The root
    # map does need its own loop, because its keys reach a set comparison
    # before any per-key validation runs.
    collected: list[tuple[str, str | tuple[str, ...]]] = []
    for raw_name, raw_value in inputs.items():
        name = _exact_str(raw_name, maximum=64)
        if type(raw_value) is list:
            # An item-scoped input is an ordered vector, one value per ordinal.
            # A list is the only shape that can carry per-item intent, and its
            # length is checked against the declared scope during expansion.
            vector = cast(list[object], raw_value)
            if not vector or len(vector) > MAX_EXPANSION_ITEMS:
                _invalid()
            collected.append(
                (
                    name,
                    tuple(_exact_str(entry, maximum=MAX_TEMPLATE_VALUE_CHARS) for entry in vector),
                )
            )
        else:
            collected.append((name, _exact_str(raw_value, maximum=MAX_TEMPLATE_VALUE_CHARS)))
    # No duplicate-name check here on purpose. `inputs` arrives as a dict, whose
    # keys are unique by construction, so such a check could never fire - it
    # would read as a guard while being dead code. Agreement with the revision's
    # declared input slots is enforced in `expand_prompt_template`, where it can
    # actually fail.

    return ExpansionRequest(
        definition_id=definition_id,
        revision_id=revision_id,
        contract_sha256=digest,
        item_count=item_count,
        selection_seed=selection_seed,
        inputs=tuple(sorted(collected)),
    )


def _selection_draw(seed: int, slot_name: str, ordinal: int, bound: int) -> int:
    """Pick one index deterministically from the recorded selection seed.

    Domain-separated by a literal prefix so a value drawn for a slot can never
    collide with one drawn for anything else that later hashes the same seed.
    The ordinal is part of the input for item scope and pinned to 0 for batch
    scope by the caller, which is what makes batch sharing a property of the
    derivation rather than of a copy.
    """

    if bound < 1:
        _invalid()
    material = "\x00".join(
        ("prompt-expansion-selection-v1", str(seed), slot_name, str(ordinal))
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big") % bound


def _slot_value(
    slot: PromptTemplateSlot,
    *,
    request: ExpansionRequest,
    ordinal: int,
    supplied: Mapping[str, str | tuple[str, ...]],
    item_choice_index: Mapping[str, int],
) -> SlotEvidence:
    """Resolve one slot for one item from already-decided material.

    Distinct-mode item-scoped choice indices arrive from the batch allocator.
    Reusable choices are drawn independently for each ordinal from the exact
    selection seed.
    """

    scope = slot.variation_scope

    if slot.mode is PromptTemplateSlotMode.FIXED:
        if slot.fixed_value is None:
            _invalid()
        return SlotEvidence(
            name=slot.name,
            mode=slot.mode,
            variation_scope=scope,
            source=ExpansionValueSource.FIXED,
            value=slot.fixed_value,
            choice_index=None,
        )

    if slot.mode is PromptTemplateSlotMode.INPUT:
        if slot.name not in supplied:
            _invalid()
        raw = supplied[slot.name]
        if scope is PromptTemplateVariationScope.BATCH:
            # A batch-scoped input is one scalar every item shares. A vector here
            # would mean the caller expected per-item variation the revision did
            # not declare, which is a disagreement rather than a detail.
            if type(raw) is not str:
                _invalid()
            value = raw
        else:
            if type(raw) is not tuple:
                _invalid()
            vector = raw
            if len(vector) != request.item_count:
                _invalid()
            value = vector[ordinal - 1]
        return SlotEvidence(
            name=slot.name,
            mode=slot.mode,
            variation_scope=scope,
            source=ExpansionValueSource.INPUT,
            value=value,
            choice_index=None,
        )

    if slot.mode is PromptTemplateSlotMode.CHOICE:
        # A backstop, not the primary check: `parse_prompt_template_contract`
        # already refuses a choice slot with no choices, so no test can reach
        # this line through a parsed contract. It stays because a hand-built
        # contract could reach `_slot_value` directly and dividing by an empty
        # range is worse than refusing.
        if not slot.choices:
            _invalid()
        if scope is PromptTemplateVariationScope.BATCH:
            index = _selection_draw(request.selection_seed, slot.name, 0, len(slot.choices))
        elif slot.choice_strategy is PromptTemplateChoiceStrategy.WITH_REPLACEMENT:
            index = _selection_draw(
                request.selection_seed,
                slot.name,
                ordinal,
                len(slot.choices),
            )
        else:
            if slot.name not in item_choice_index:
                _invalid()
            index = item_choice_index[slot.name]
            # Backstop for a direct caller. Indices produced by
            # `_item_choice_candidates` are drawn from range(len(choices)) and
            # so are always in range; no test can reach this through the codec.
            if not 0 <= index < len(slot.choices):
                _invalid()
        return SlotEvidence(
            name=slot.name,
            mode=slot.mode,
            variation_scope=scope,
            source=ExpansionValueSource.CHOICE,
            value=slot.choices[index],
            choice_index=index,
        )

    if slot.mode is PromptTemplateSlotMode.MODEL:
        return SlotEvidence(
            name=slot.name,
            mode=slot.mode,
            variation_scope=scope,
            source=ExpansionValueSource.MODEL,
            value=None,
            choice_index=None,
        )

    _invalid()


def _item_choice_candidates(
    contract: PromptTemplateContract,
    seed: int,
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Every distinct-mode item choice combination, in deterministic order.

    Enumerated rather than sampled so "can this template yield N distinct
    drafts" has an exact answer for the small spaces where the answer is
    actually no. Ordered by a seed-derived digest rather than by any Python
    iteration order, so the sequence does not depend on hash randomisation or
    interpreter version.
    """

    item_slots = [
        slot
        for slot in contract.slots
        if slot.mode is PromptTemplateSlotMode.CHOICE
        and slot.variation_scope is PromptTemplateVariationScope.ITEM
        and slot.choice_strategy is PromptTemplateChoiceStrategy.DISTINCT
    ]
    if not item_slots:
        return ((),)

    space = 1
    for slot in item_slots:
        space *= len(slot.choices)
        if space > MAX_ITEM_CHOICE_SPACE:
            _invalid()

    combinations: list[tuple[tuple[str, int], ...]] = []
    for index in range(space):
        remainder = index
        combination: list[tuple[str, int]] = []
        for slot in item_slots:
            size = len(slot.choices)
            combination.append((slot.name, remainder % size))
            remainder //= size
        combinations.append(tuple(combination))

    def order_key(combination: tuple[tuple[str, int], ...]) -> str:
        material = "\x00".join(
            (
                "prompt-expansion-order-v1",
                str(seed),
                *(f"{name}={index}" for name, index in combination),
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    combinations.sort(key=order_key)
    return tuple(combinations)


def _contract_digest(contract: PromptTemplateContract) -> str:
    """The contract's digest, refusing in this module's own error type.

    A hand-built contract that never went through the template parser can fail
    validation here. Letting `PromptTemplateError` escape would make one call
    raise two unrelated types depending on how its argument was constructed, and
    a caller catching only this module's error would see the other pass through.
    """

    try:
        return prompt_template_contract_sha256(contract)
    except (
        PromptTemplateError,
        UnicodeEncodeError,
        RecursionError,
        TypeError,
        OverflowError,
    ):
        _invalid()


def _render(contract: PromptTemplateContract, values: dict[str, str]) -> str:
    """Render, refusing in this module's own error type.

    The request parser and the renderer do not agree on every value: a NUL is
    accepted as an input here and refused there. Until they do, this boundary is
    where the disagreement is normalised, and the refusal carries the same fixed
    text as every other - no body, value, guidance, or identifier.
    """

    try:
        return render_prompt_template(contract, values)
    except PromptTemplateError:
        _invalid()


def expand_prompt_template(
    contract: PromptTemplateContract,
    request: ExpansionRequest,
) -> ExpansionPlan:
    """Produce N drafts from one contract, deterministically and inertly.

    Every non-model slot resolves here. A `model` slot is left unresolved and
    reported in `pending_model_slots`, and any item carrying one renders no
    prompt at all - a half-rendered prompt with a placeholder still in it is
    indistinguishable from a finished one at a glance, which is the failure this
    avoids.
    """

    if type(contract) is not PromptTemplateContract or type(request) is not ExpansionRequest:
        _invalid()
    # Rebuild and parse the request before reading any nested request field.
    # Frozen dataclasses can be rewritten with `object.__setattr__`; using the
    # original object before this boundary would let a hostile `__eq__` escape
    # instead of producing this codec's fixed refusal.
    request = parse_expansion_request(expansion_request_payload(request))
    # The request carries the digest of the revision it was built for.
    # Without this it is decorative: a request could be expanded against a
    # different contract entirely and every downstream record would still
    # name the revision the caller claimed.
    if _contract_digest(contract) != request.contract_sha256:
        _invalid()

    declared = {slot.name for slot in contract.slots}
    supplied = dict(request.inputs)
    expected_inputs = {
        slot.name for slot in contract.slots if slot.mode is PromptTemplateSlotMode.INPUT
    }
    # Both directions. A missing input cannot render; a surplus one means the
    # caller believes in a slot this revision does not have, and silently
    # dropping it would hide that disagreement until the prompt looked wrong.
    if set(supplied) != expected_inputs or not expected_inputs <= declared:
        _invalid()

    # `render_prompt_template` supplies fixed values itself and expects exactly
    # the non-fixed ones, so passing every slot would refuse.
    needed = {slot.name for slot in contract.slots if slot.mode is not PromptTemplateSlotMode.FIXED}
    candidates = _item_choice_candidates(contract, request.selection_seed)
    reusable_choices = any(
        slot.mode is PromptTemplateSlotMode.CHOICE
        and slot.variation_scope is PromptTemplateVariationScope.ITEM
        and slot.choice_strategy is PromptTemplateChoiceStrategy.WITH_REPLACEMENT
        for slot in contract.slots
    )
    pending_model_values = any(slot.mode is PromptTemplateSlotMode.MODEL for slot in contract.slots)

    items: list[ExpandedItem] = []
    seen_prompts: set[str] = set()
    for ordinal in range(1, request.item_count + 1):
        chosen: ExpandedItem | None = None
        item_candidates = (
            (candidates[(ordinal - 1) % len(candidates)],) if pending_model_values else candidates
        )
        for candidate in item_candidates:
            index_map = dict(candidate)
            evidence = tuple(
                _slot_value(
                    slot,
                    request=request,
                    ordinal=ordinal,
                    supplied=supplied,
                    item_choice_index=index_map,
                )
                for slot in contract.slots
            )
            values = {
                entry.name: entry.value
                for entry in evidence
                if entry.value is not None and entry.name in needed
            }
            if set(values) != needed:
                # A model slot is outstanding, so nothing renders yet. The
                # candidate still records this item's exact authored choices.
                chosen = ExpandedItem(
                    ordinal=ordinal,
                    evidence=evidence,
                    rendered_prompt=None,
                    rendered_sha256=None,
                )
                break
            rendered = _render(contract, values)
            if not reusable_choices and rendered in seen_prompts:
                continue
            seen_prompts.add(rendered)
            chosen = ExpandedItem(
                ordinal=ordinal,
                evidence=evidence,
                rendered_prompt=rendered,
                rendered_sha256=_rendered_digest(rendered),
            )
            break
        if chosen is None:
            # The declared value space cannot yield another distinct draft.
            # Refusing here is the whole point: duplicating a draft or looping
            # would both produce a batch the caller did not ask for.
            raise PromptExpansionDistinctCapacityError()
        items.append(chosen)

    # No post-loop distinctness assertion here. The loop above appends only a
    # prompt it has not already seen and refuses when none remains, so a
    # second check could never fire - it would read as a guarantee while
    # being dead code, which is the defect this repository keeps finding.

    pending = tuple(
        ModelSlotRequest(
            name=slot.name,
            variation_scope=slot.variation_scope,
            # The contract guarantees a model slot carries guidance, and the
            # model-values codec on the other side of this seam types it `str`.
            # Admitting None here would describe a state neither side can reach.
            #
            # Second layer, and deliberately so. `_contract_digest` above
            # round-trips the contract through the template parser, which
            # requires guidance on a model slot - so no contract missing one
            # arrives here, and no test can reach this line through
            # `expand_prompt_template`. It stays because it is what makes the
            # `str` annotation true rather than aspirational, and because the
            # coupling that makes it unreachable lives in another module.
            guidance=_exact_str(slot.guidance, maximum=MAX_TEMPLATE_GUIDANCE_CHARS),
        )
        for slot in contract.slots
        if slot.mode is PromptTemplateSlotMode.MODEL
    )

    return ExpansionPlan(
        codec_version=PROMPT_EXPANSION_CODEC_VERSION,
        request=request,
        template_body=contract.body,
        items=tuple(items),
        pending_model_slots=pending,
    )


def _rendered_digest(rendered: str) -> str:
    """Canonical identity of one rendered prompt, domain-separated."""

    material = "\x00".join(("prompt-expansion-rendered-v1", rendered))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def expansion_request_payload(request: ExpansionRequest) -> dict[str, Any]:
    """The canonical validated JSON shape of one request.

    Checking that `plan.request` is an `ExpansionRequest` says nothing about
    what is inside it: `object.__setattr__` rewrites any field on a frozen
    dataclass, and the receipt then hashed whatever was written. `item_count`
    became `True`, `selection_seed` became a string, and an input value became
    an arbitrary object that escaped as a raw `TypeError` from the JSON encoder.

    So this does not read the fields and trust them. It rebuilds the payload and
    then re-parses it, refusing unless the parser produces the same request
    back. `parse_expansion_request` is the only definition of a valid request
    there is; running it again is what makes this payload worth hashing.
    """

    if type(request) is not ExpansionRequest:
        _invalid()
    if type(request.inputs) is not tuple:
        _invalid()
    if len(request.inputs) > MAX_EXPANSION_INPUT_SLOTS:
        _invalid()

    inputs: dict[str, Any] = {}
    for pair in request.inputs:
        if type(pair) is not tuple or len(pair) != 2:
            _invalid()
        name, value = pair
        if type(name) is not str or name in inputs:
            _invalid()
        if type(value) is tuple:
            # Check the vector before copying or iterating it. The request
            # parser below will enforce the matching item scope and contents.
            if not value or len(value) > MAX_EXPANSION_ITEMS:
                _invalid()
            inputs[name] = list(value)
        else:
            inputs[name] = value

    payload: dict[str, Any] = {
        "definition_id": request.definition_id,
        "revision_id": request.revision_id,
        "contract_sha256": request.contract_sha256,
        "item_count": request.item_count,
        "selection_seed": request.selection_seed,
        "inputs": inputs,
    }
    # The round trip. A tampered field either fails the parser outright or comes
    # back different, and both refuse. It also pins the canonical ordering,
    # because the parser sorts `inputs` and an out-of-order tuple will not
    # compare equal to what it returns.
    if parse_expansion_request(payload) != request:
        _invalid()
    return payload


def _canonical_json(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError):
        _invalid()


def _receipt_body_tokens(body: str) -> tuple[str, ...]:
    names: list[str] = []
    position = 0
    for match in _RECEIPT_TOKEN.finditer(body):
        gap = body[position : match.start()]
        if "{" in gap or "}" in gap:
            _invalid()
        names.append(match.group(1))
        position = match.end()
    if "{" in body[position:] or "}" in body[position:] or len(names) != len(set(names)):
        _invalid()
    return tuple(names)


def _validate_receipt_authority(
    items: list[dict[str, Any]],
    *,
    request_payload: dict[str, Any],
    template_body: str,
    complete: bool,
) -> None:
    """Bind evidence order, scope and values to the receipt's exact body.

    Both receipt boundaries call this helper independently. Complete plans
    require exact rendering from the receipt's bounded body and recorded values,
    in addition to one batch value and one ordered slot shape.
    Pending plans have no rendered prompt and explicitly enforce only the
    conditions that remain knowable: body-token/order, batch-value, and
    ordered-shape agreement. Request-declared input values are checked in
    either state.
    """

    body_tokens = _receipt_body_tokens(template_body)
    expected_shape: tuple[tuple[str, str, str], ...] | None = None
    batch_values: dict[str, tuple[str | None, int | None]] = {}
    request_inputs = cast(dict[str, object], request_payload["inputs"])

    for position, item in enumerate(items, start=1):
        evidence = cast(list[dict[str, Any]], item["evidence"])
        shape = tuple(
            (
                cast(str, record["name"]),
                cast(str, record["mode"]),
                cast(str, record["variation_scope"]),
            )
            for record in evidence
        )
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            _invalid()
        if tuple(one[0] for one in shape) != body_tokens:
            _invalid()

        replacements: dict[str, str] = {}
        for record in evidence:
            name = cast(str, record["name"])
            scope = cast(str, record["variation_scope"])
            value = cast(str | None, record["value"])
            choice_index = cast(int | None, record["choice_index"])
            if scope == PromptTemplateVariationScope.BATCH.value:
                authority = (value, choice_index)
                previous = batch_values.setdefault(name, authority)
                if previous != authority:
                    _invalid()
            if record["mode"] == PromptTemplateSlotMode.INPUT.value:
                supplied = request_inputs.get(name)
                if scope == PromptTemplateVariationScope.BATCH.value:
                    expected = supplied if type(supplied) is str else None
                else:
                    expected = (
                        supplied[position - 1]
                        if type(supplied) is list and len(supplied) == len(items)
                        else None
                    )
                if value != expected:
                    _invalid()
            if value is not None:
                replacements[name] = value

        if complete:
            prompt = cast(str, item["rendered_prompt"])
            if len(replacements) != len(body_tokens):
                _invalid()

            def _substitute(match: re.Match[str], values: dict[str, str] = replacements) -> str:
                return values[match.group(1)]

            rendered = _RECEIPT_TOKEN.sub(_substitute, template_body)
            if rendered != prompt:
                _invalid()


def parse_expansion_plan_payload(value: object) -> dict[str, Any]:
    """Strictly validate a stored receipt, returning its canonical form.

    A store that has written a receipt needs to revalidate it on the way back
    rather than deserialize it and hope. This is the reading half of
    `expansion_plan_payload`: same rules, same refusals, applied to plain JSON
    that no longer has any dataclass to vouch for it.
    """

    if type(value) is not dict:
        _invalid()
    payload = cast(dict[object, object], value)
    _validate_exact_keys(payload, _PLAN_PAYLOAD_KEYS)

    if type(payload["codec_version"]) is not int:
        _invalid()
    if payload["codec_version"] != PROMPT_EXPANSION_CODEC_VERSION:
        _invalid()

    request = parse_expansion_request(payload["request"])
    request_payload = expansion_request_payload(request)
    template_body = _exact_str(payload["template_body"], maximum=MAX_TEMPLATE_BODY_CHARS)

    raw_pending = payload["pending_model_slots"]
    if type(raw_pending) is not list or len(raw_pending) > MAX_EXPANSION_INPUT_SLOTS:
        _invalid()
    pending: list[dict[str, Any]] = []
    pending_names: list[str] = []
    pending_scopes: dict[str, str] = {}
    for entry in cast(list[object], raw_pending):
        if type(entry) is not dict:
            _invalid()
        slot = cast(dict[object, object], entry)
        _validate_exact_keys(slot, _PENDING_SLOT_KEYS)
        name = _exact_str(slot["name"], maximum=64)
        if _RECEIPT_SLOT_NAME.fullmatch(name) is None:
            _invalid()
        scope = _exact_str(slot["variation_scope"], maximum=32)
        if scope not in {one.value for one in PromptTemplateVariationScope}:
            _invalid()
        guidance = _exact_str(slot["guidance"], maximum=MAX_TEMPLATE_GUIDANCE_CHARS)
        pending_names.append(name)
        pending_scopes[name] = scope
        pending.append({"name": name, "variation_scope": scope, "guidance": guidance})
    if len(set(pending_names)) != len(pending_names):
        _invalid()
    complete = not pending

    raw_items = payload["items"]
    if type(raw_items) is not list or len(raw_items) != request.item_count:
        _invalid()
    items: list[dict[str, Any]] = []
    expected_slot_shapes: tuple[tuple[str, str, str], ...] | None = None
    for position, entry in enumerate(cast(list[object], raw_items), start=1):
        if type(entry) is not dict:
            _invalid()
        item = cast(dict[object, object], entry)
        _validate_exact_keys(item, _PLAN_ITEM_KEYS)
        if type(item["ordinal"]) is not int or item["ordinal"] != position:
            _invalid()
        (
            evidence,
            model_scopes,
            unresolved_model_scopes,
            input_names,
            slot_shapes,
        ) = _parse_evidence_payload(item["evidence"])
        if pending_scopes:
            if model_scopes != pending_scopes or unresolved_model_scopes != pending_scopes:
                _invalid()
        elif unresolved_model_scopes:
            _invalid()
        if sorted(input_names) != sorted(request_payload["inputs"]):
            _invalid()
        if expected_slot_shapes is None:
            expected_slot_shapes = slot_shapes
        elif slot_shapes != expected_slot_shapes:
            _invalid()
        stored_prompt = item["rendered_prompt"]
        stored = item["rendered_sha256"]
        if complete:
            prompt = _exact_str(stored_prompt, maximum=MAX_TEMPLATE_RENDERED_CHARS)
            digest = _exact_str(stored, maximum=64)
            if len(digest) != 64 or any(one not in "0123456789abcdef" for one in digest):
                _invalid()
            if _rendered_digest(prompt) != digest:
                _invalid()
            items.append(
                {
                    "ordinal": position,
                    "rendered_prompt": prompt,
                    "rendered_sha256": digest,
                    "evidence": evidence,
                }
            )
        else:
            if stored_prompt is not None or stored is not None:
                _invalid()
            items.append(
                {
                    "ordinal": position,
                    "rendered_prompt": None,
                    "rendered_sha256": None,
                    "evidence": evidence,
                }
            )

    if pending:
        if expected_slot_shapes is None:
            _invalid()
        expected_pending = tuple(
            (name, scope)
            for name, mode, scope in expected_slot_shapes
            if mode == PromptTemplateSlotMode.MODEL.value
        )
        if tuple((slot["name"], slot["variation_scope"]) for slot in pending) != expected_pending:
            _invalid()

    _validate_receipt_authority(
        items,
        request_payload=request_payload,
        template_body=template_body,
        complete=complete,
    )

    return {
        "codec_version": PROMPT_EXPANSION_CODEC_VERSION,
        "request": request_payload,
        "template_body": template_body,
        "pending_model_slots": pending,
        "items": items,
    }


def _parse_evidence_payload(
    value: object,
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    dict[str, str],
    list[str],
    tuple[tuple[str, str, str], ...],
]:
    """One item's evidence, validated the same way the writing half validates it."""

    if type(value) is not list or len(value) > MAX_EXPANSION_INPUT_SLOTS:
        _invalid()
    evidence: list[dict[str, Any]] = []
    names: list[str] = []
    model_scopes: dict[str, str] = {}
    unresolved_model_scopes: dict[str, str] = {}
    input_names: list[str] = []
    slot_shapes: list[tuple[str, str, str]] = []
    for entry in cast(list[object], value):
        if type(entry) is not dict:
            _invalid()
        record = cast(dict[object, object], entry)
        _validate_exact_keys(record, _EVIDENCE_KEYS)
        name = _exact_str(record["name"], maximum=64)
        if _RECEIPT_SLOT_NAME.fullmatch(name) is None:
            _invalid()
        mode = _exact_str(record["mode"], maximum=32)
        scope = _exact_str(record["variation_scope"], maximum=32)
        source = _exact_str(record["source"], maximum=32)
        if (
            mode not in {one.value for one in PromptTemplateSlotMode}
            or scope not in {one.value for one in PromptTemplateVariationScope}
            or source != mode
            or (
                mode == PromptTemplateSlotMode.FIXED.value
                and scope != PromptTemplateVariationScope.BATCH.value
            )
        ):
            _invalid()
        raw_value = record["value"]
        raw_index = record["choice_index"]
        if mode == PromptTemplateSlotMode.MODEL.value:
            if raw_index is not None:
                _invalid()
            model_scopes[name] = scope
            if raw_value is None:
                unresolved_model_scopes[name] = scope
                stored_value: str | None = None
            else:
                stored_value = _exact_str(raw_value, maximum=MAX_TEMPLATE_VALUE_CHARS)
        else:
            stored_value = _exact_str(raw_value, maximum=MAX_TEMPLATE_VALUE_CHARS)
            if mode == PromptTemplateSlotMode.CHOICE.value:
                if type(raw_index) is not int or not 0 <= raw_index < MAX_TEMPLATE_CHOICES:
                    _invalid()
            elif raw_index is not None:
                _invalid()
            if mode == PromptTemplateSlotMode.INPUT.value:
                input_names.append(name)
        names.append(name)
        slot_shapes.append((name, mode, scope))
        evidence.append(
            {
                "name": name,
                "mode": mode,
                "variation_scope": scope,
                "source": source,
                "value": stored_value,
                "choice_index": raw_index,
            }
        )
    if len(set(names)) != len(names):
        _invalid()
    return evidence, model_scopes, unresolved_model_scopes, input_names, tuple(slot_shapes)


def expansion_plan_payload_digest(payload: object) -> str:
    """The digest of a stored receipt, computed only after revalidating it.

    A store compares this against the digest it recorded. Validating first is
    the whole point: hashing a payload straight off disk would agree with
    whatever is on disk, which is not a check.
    """

    return hashlib.sha256(
        _canonical_json(parse_expansion_plan_payload(payload)).encode("utf-8")
    ).hexdigest()


def expansion_plan_payload(plan: ExpansionPlan) -> dict[str, Any]:
    """The one canonical, fully validated JSON shape of a plan.

    Everything downstream - the digest, and any persistence that stores a
    receipt - goes through here. It does not trust the dataclasses it is handed:
    a frozen dataclass can still be rewritten with `object.__setattr__`, so a
    payload built by reading fields would attest to whatever was written last.

    Two properties matter and both were missing before. Every rendered digest is
    RECOMPUTED from its prompt rather than copied, so a tampered prompt cannot
    keep its old identity. And the pending model evidence is part of the payload,
    so a plan cannot change from incomplete to complete without changing its
    digest.
    """

    if type(plan) is not ExpansionPlan:
        _invalid()
    if type(plan.codec_version) is not int or plan.codec_version != PROMPT_EXPANSION_CODEC_VERSION:
        _invalid()
    request = plan.request
    if type(request) is not ExpansionRequest:
        _invalid()
    if type(plan.items) is not tuple or type(plan.pending_model_slots) is not tuple:
        _invalid()
    # Not just "is an ExpansionRequest". Every field inside it is revalidated
    # through the parser, because the type says nothing about the contents.
    request_payload = expansion_request_payload(request)
    template_body = _exact_str(plan.template_body, maximum=MAX_TEMPLATE_BODY_CHARS)
    request_item_count = cast(int, request_payload["item_count"])
    if len(plan.items) != request_item_count:
        _invalid()
    request_input_names = sorted(request_payload["inputs"])

    if len(plan.pending_model_slots) > MAX_EXPANSION_INPUT_SLOTS:
        _invalid()
    pending: list[dict[str, Any]] = []
    pending_names: list[str] = []
    pending_scopes: dict[str, PromptTemplateVariationScope] = {}
    for slot in plan.pending_model_slots:
        if type(slot) is not ModelSlotRequest:
            _invalid()
        name = _exact_str(slot.name, maximum=64)
        if type(slot.variation_scope) is not PromptTemplateVariationScope:
            _invalid()
        guidance = _exact_str(slot.guidance, maximum=MAX_TEMPLATE_GUIDANCE_CHARS)
        pending_names.append(name)
        pending_scopes[name] = slot.variation_scope
        pending.append(
            {
                "name": name,
                "variation_scope": str(slot.variation_scope),
                "guidance": guidance,
            }
        )
    if len(set(pending_names)) != len(pending_names):
        _invalid()
    complete = not pending

    items: list[dict[str, Any]] = []
    expected_slot_shapes: (
        tuple[tuple[str, PromptTemplateSlotMode, PromptTemplateVariationScope], ...] | None
    ) = None
    for position, item in enumerate(plan.items, start=1):
        if type(item) is not ExpandedItem:
            _invalid()
        if type(item.ordinal) is not int or item.ordinal != position:
            _invalid()
        # Not `or not item.evidence`. The contract permits a fixed literal
        # template with zero slots and the editor displays that shape, so an
        # item with no evidence is legitimate. Requiring evidence here refused a
        # valid one-item plan. A count above one still refuses, but for the
        # separate and correct reason that no value space can yield a second
        # distinct draft.
        if type(item.evidence) is not tuple or len(item.evidence) > MAX_EXPANSION_INPUT_SLOTS:
            _invalid()

        evidence: list[dict[str, Any]] = []
        names: list[str] = []
        model_slot_scopes: dict[str, PromptTemplateVariationScope] = {}
        input_slot_names: list[str] = []
        slot_shapes: list[tuple[str, PromptTemplateSlotMode, PromptTemplateVariationScope]] = []
        for entry in item.evidence:
            if type(entry) is not SlotEvidence:
                _invalid()
            entry_name = _exact_str(entry.name, maximum=64)
            if _RECEIPT_SLOT_NAME.fullmatch(entry_name) is None:
                _invalid()
            if (
                type(entry.mode) is not PromptTemplateSlotMode
                or type(entry.variation_scope) is not PromptTemplateVariationScope
                or type(entry.source) is not ExpansionValueSource
            ):
                _invalid()
            # The source must agree with the mode it claims to have come from.
            if str(entry.source) != str(entry.mode):
                _invalid()
            if (
                entry.mode is PromptTemplateSlotMode.FIXED
                and entry.variation_scope is not PromptTemplateVariationScope.BATCH
            ):
                _invalid()
            value: str | None
            if entry.mode is PromptTemplateSlotMode.MODEL:
                if entry.choice_index is not None:
                    _invalid()
                model_slot_scopes[entry_name] = entry.variation_scope
                if complete:
                    value = _exact_str(entry.value, maximum=MAX_TEMPLATE_VALUE_CHARS)
                else:
                    if entry.value is not None:
                        _invalid()
                    value = None
            else:
                value = _exact_str(entry.value, maximum=MAX_TEMPLATE_VALUE_CHARS)
                if entry.mode is PromptTemplateSlotMode.CHOICE:
                    if (
                        type(entry.choice_index) is not int
                        or not 0 <= entry.choice_index < MAX_TEMPLATE_CHOICES
                    ):
                        _invalid()
                elif entry.choice_index is not None:
                    _invalid()
                if entry.mode is PromptTemplateSlotMode.INPUT:
                    input_slot_names.append(entry_name)
            names.append(entry_name)
            slot_shapes.append((entry_name, entry.mode, entry.variation_scope))
            evidence.append(
                {
                    "name": entry_name,
                    "mode": str(entry.mode),
                    "variation_scope": str(entry.variation_scope),
                    "source": str(entry.source),
                    # The validated value, not the attribute re-read.
                    "value": value,
                    "choice_index": entry.choice_index,
                }
            )
        if len(set(names)) != len(names):
            _invalid()
        # The model slots an item carries must be exactly those the plan reports
        # as pending; otherwise "complete" means two different things at once.
        if pending_scopes and model_slot_scopes != pending_scopes:
            _invalid()
        # And the input slots it carries must be exactly the ones the request
        # supplies. Without this a request could be emptied of its inputs while
        # the evidence still recorded values for them: the digest changes, so
        # nothing is forged, but the receipt then describes a plan the codec
        # could never have produced - `expand_prompt_template` requires the
        # supplied inputs to match the declared input slots exactly.
        if sorted(input_slot_names) != sorted(request_input_names):
            _invalid()
        if expected_slot_shapes is None:
            expected_slot_shapes = tuple(slot_shapes)
        elif tuple(slot_shapes) != expected_slot_shapes:
            _invalid()

        if complete:
            rendered = _exact_str(item.rendered_prompt, maximum=MAX_TEMPLATE_RENDERED_CHARS)
            recomputed = _rendered_digest(rendered)
            stored = _exact_str(item.rendered_sha256, maximum=64)
            if len(stored) != 64 or any(one not in "0123456789abcdef" for one in stored):
                _invalid()
            # Recomputed, never copied. This is the line that makes a tampered
            # prompt change the digest.
            if stored != recomputed:
                _invalid()
            items.append(
                {
                    "ordinal": item.ordinal,
                    "rendered_prompt": rendered,
                    "rendered_sha256": recomputed,
                    "evidence": evidence,
                }
            )
        else:
            if item.rendered_prompt is not None or item.rendered_sha256 is not None:
                _invalid()
            items.append(
                {
                    "ordinal": item.ordinal,
                    "rendered_prompt": None,
                    "rendered_sha256": None,
                    "evidence": evidence,
                }
            )

    payload = {
        "codec_version": plan.codec_version,
        "request": request_payload,
        "template_body": template_body,
        "pending_model_slots": pending,
        "items": items,
    }
    _validate_receipt_authority(
        items,
        request_payload=request_payload,
        template_body=template_body,
        complete=complete,
    )
    return payload


def expansion_plan_digest(plan: ExpansionPlan) -> str:
    """A stable digest over everything that decides the drafts.

    Covers the request identity, every resolved value, the pending model
    evidence, and the digest of each rendered prompt.

    An earlier version of this docstring said the rendered prompt was
    deliberately excluded. That was the defect, not the design: excluding it let
    a prompt be replaced while its recorded digest was copied through unchanged.
    The rendered TEXT is still not hashed directly - its digest is, recomputed
    from the text rather than copied - so the receipt is bounded in size while
    still binding to what was actually rendered.
    """

    return hashlib.sha256(_canonical_json(expansion_plan_payload(plan)).encode("utf-8")).hexdigest()


def _snapshot_completion_contract(
    contract: PromptTemplateContract,
) -> PromptTemplateContract:
    """Validate and detach template authority within its public slot bound."""

    if type(contract) is not PromptTemplateContract:
        _invalid()
    try:
        slots = object.__getattribute__(contract, "slots")
    except AttributeError:
        _invalid()
    if type(slots) is not tuple or len(slots) > MAX_TEMPLATE_SLOTS:
        _invalid()
    total_choices = 0
    for slot in slots:
        if type(slot) is not PromptTemplateSlot:
            _invalid()
        try:
            choices = object.__getattribute__(slot, "choices")
        except AttributeError:
            _invalid()
        if type(choices) is not tuple or len(choices) > MAX_TEMPLATE_CHOICES:
            _invalid()
        total_choices += len(choices)
        if total_choices > MAX_TEMPLATE_TOTAL_CHOICES:
            _invalid()
    try:
        resource = object.__getattribute__(contract, "resource_policy")
    except AttributeError:
        _invalid()
    if type(resource) is not PromptTemplateResourcePolicy:
        _invalid()
    try:
        lora_policy = object.__getattribute__(resource, "lora_policy")
    except AttributeError:
        _invalid()
    if lora_policy is not None:
        if type(lora_policy) is not PromptTemplateLoraPolicy:
            _invalid()
        try:
            stack = object.__getattribute__(lora_policy, "stack")
        except AttributeError:
            _invalid()
        if type(stack) is not tuple or len(stack) > MAX_TEMPLATE_LORAS:
            _invalid()
    try:
        return parse_prompt_template_contract(prompt_template_contract_payload(contract))
    except (
        PromptTemplateError,
        UnicodeEncodeError,
        RecursionError,
        TypeError,
        OverflowError,
    ):
        _invalid()


def _snapshot_completion_plan(
    plan: ExpansionPlan,
) -> tuple[dict[str, Any], ExpansionRequest]:
    """Validate and detach an expansion authority before relating it to a template."""

    if type(plan) is not ExpansionPlan:
        _invalid()
    try:
        items = object.__getattribute__(plan, "items")
        pending = object.__getattribute__(plan, "pending_model_slots")
    except AttributeError:
        _invalid()
    if (
        type(items) is not tuple
        or len(items) > MAX_EXPANSION_ITEMS
        or type(pending) is not tuple
        or len(pending) > MAX_EXPANSION_INPUT_SLOTS
    ):
        _invalid()
    for item in items:
        if type(item) is not ExpandedItem:
            _invalid()
        try:
            evidence = object.__getattribute__(item, "evidence")
        except AttributeError:
            _invalid()
        if type(evidence) is not tuple or len(evidence) > MAX_EXPANSION_INPUT_SLOTS:
            _invalid()
    payload = parse_expansion_plan_payload(expansion_plan_payload(plan))
    return payload, parse_expansion_request(payload["request"])


def _snapshot_completion_values(
    values: PromptModelValues,
    *,
    contract: PromptModelSlotContract,
) -> PromptModelValues:
    """Validate and detach model output without traversing an unbounded container."""

    if type(values) is not PromptModelValues:
        _invalid()
    try:
        batch_values = object.__getattribute__(values, "batch_values")
        items = object.__getattribute__(values, "items")
    except AttributeError:
        _invalid()
    if (
        type(batch_values) is not tuple
        or len(batch_values) > MAX_EXPANSION_INPUT_SLOTS
        or type(items) is not tuple
        or len(items) > MAX_EXPANSION_ITEMS
    ):
        _invalid()
    for item in items:
        if type(item) is not PromptModelItemValues:
            _invalid()
        try:
            item_values = object.__getattribute__(item, "values")
        except AttributeError:
            _invalid()
        if type(item_values) is not tuple or len(item_values) > MAX_EXPANSION_INPUT_SLOTS:
            _invalid()
    try:
        payload = prompt_model_values_payload(values, contract=contract)
        return parse_prompt_model_values(payload, contract=contract)
    except PromptModelValuesError:
        _invalid()


def _bind_incomplete_plan(
    contract: PromptTemplateContract,
    payload: dict[str, Any],
    request: ExpansionRequest,
) -> ExpansionPlan:
    """Require the receipt to be exactly the expansion this contract produces."""

    expected = expand_prompt_template(contract, request)
    expected_payload = parse_expansion_plan_payload(expansion_plan_payload(expected))
    if not expected.pending_model_slots or payload != expected_payload:
        _invalid()
    return expected


def prompt_model_invocation_data(
    contract: PromptTemplateContract,
    plan: ExpansionPlan,
) -> PromptModelInvocationData:
    """Build the inert, resolved context for one exact pending model contract."""

    # Imported lazily so importing the pure expansion codec does not import the
    # adapter boundary unless a caller actually requests invocation data.
    from .prompt_model_invocation import PromptModelInvocationData, PromptModelInvocationItem

    normalized_contract = _snapshot_completion_contract(contract)
    plan_payload, request = _snapshot_completion_plan(plan)
    expected = _bind_incomplete_plan(normalized_contract, plan_payload, request)

    first = expected.items[0]
    batch_values = tuple(
        (entry.name, cast(str, entry.value))
        for entry in first.evidence
        if entry.mode is not PromptTemplateSlotMode.MODEL
        and entry.variation_scope is PromptTemplateVariationScope.BATCH
    )
    items = tuple(
        PromptModelInvocationItem(
            ordinal=item.ordinal,
            values=tuple(
                (entry.name, cast(str, entry.value))
                for entry in item.evidence
                if entry.mode is not PromptTemplateSlotMode.MODEL
                and entry.variation_scope is PromptTemplateVariationScope.ITEM
            ),
        )
        for item in expected.items
    )
    return PromptModelInvocationData(
        template_text=normalized_contract.body,
        batch_values=batch_values,
        items=items,
    )


def complete_prompt_expansion_with_model_values(
    contract: PromptTemplateContract,
    plan: ExpansionPlan,
    values: PromptModelValues,
) -> ExpansionPlan:
    """Resolve only pending model leaves and return an exact complete expansion."""

    normalized_contract = _snapshot_completion_contract(contract)
    plan_payload, request = _snapshot_completion_plan(plan)
    try:
        model_contract = prompt_model_slot_contract(
            normalized_contract,
            item_count=request.item_count,
        )
    except PromptModelValuesError:
        _invalid()
    normalized_values = _snapshot_completion_values(values, contract=model_contract)

    # Relate the three independently validated snapshots only now. Re-expanding
    # binds the request digest, pending names/scopes/guidance, deterministic
    # allocation, choice indices, and evidence order to this exact contract.
    expected = _bind_incomplete_plan(normalized_contract, plan_payload, request)

    batch_values = dict(normalized_values.batch_values)
    item_values = {item.ordinal: dict(item.values) for item in normalized_values.items}
    completed_items: list[ExpandedItem] = []
    seen_rendered: set[str] = set()
    reusable_choices = any(
        slot.mode is PromptTemplateSlotMode.CHOICE
        and slot.variation_scope is PromptTemplateVariationScope.ITEM
        and slot.choice_strategy is PromptTemplateChoiceStrategy.WITH_REPLACEMENT
        for slot in normalized_contract.slots
    )
    for item in expected.items:
        completed_evidence: list[SlotEvidence] = []
        for entry in item.evidence:
            if entry.mode is not PromptTemplateSlotMode.MODEL:
                completed_evidence.append(entry)
                continue
            if entry.variation_scope is PromptTemplateVariationScope.BATCH:
                if entry.name not in batch_values:
                    _invalid()
                model_value = batch_values[entry.name]
            else:
                ordinal_values = item_values.get(item.ordinal)
                if ordinal_values is None or entry.name not in ordinal_values:
                    _invalid()
                model_value = ordinal_values[entry.name]
            completed_evidence.append(
                SlotEvidence(
                    name=entry.name,
                    mode=entry.mode,
                    variation_scope=entry.variation_scope,
                    source=entry.source,
                    value=model_value,
                    choice_index=None,
                )
            )
        evidence = tuple(completed_evidence)
        render_values = {
            entry.name: cast(str, entry.value)
            for entry in evidence
            if entry.mode is not PromptTemplateSlotMode.FIXED
        }
        rendered = _render(normalized_contract, render_values)
        if not reusable_choices and rendered in seen_rendered:
            _invalid()
        seen_rendered.add(rendered)
        completed_items.append(
            ExpandedItem(
                ordinal=item.ordinal,
                evidence=evidence,
                rendered_prompt=rendered,
                rendered_sha256=_rendered_digest(rendered),
            )
        )

    completed = ExpansionPlan(
        codec_version=PROMPT_EXPANSION_CODEC_VERSION,
        request=request,
        template_body=normalized_contract.body,
        items=tuple(completed_items),
        pending_model_slots=(),
    )
    # Exercise both public receipt boundaries before returning. This guarantees
    # the bridge cannot manufacture a state persistence would later refuse.
    payload = expansion_plan_payload(completed)
    parse_expansion_plan_payload(payload)
    expansion_plan_digest(completed)
    return completed


def expansion_selection_seed(material: object) -> int:
    """Derive a selection seed inside its own domain from caller material.

    Takes a closed container rather than any iterable. Accepting `Iterable`
    would invoke caller-supplied `__iter__` inside a pure codec, and a
    generator could yield a different sequence on a second read - which would
    make a "deterministic" seed whatever the caller wanted it to be.
    """

    if type(material) not in (tuple, list):
        _invalid()
    entries = list(cast(list[object], material))
    if not entries or len(entries) > MAX_SEED_MATERIAL_ENTRIES:
        _invalid()
    total = 0
    checked: list[str] = []
    for item in entries:
        if type(item) is not str:
            _invalid()
        if len(item) > MAX_SEED_MATERIAL_CHARS:
            _invalid()
        total += len(item)
        if total > MAX_SEED_MATERIAL_CHARS:
            _invalid()
        try:
            encoded = item.encode("utf-8")
        except UnicodeEncodeError:
            # A lone surrogate cannot be encoded; refusing beats hashing a
            # replacement byte and calling the result deterministic.
            _invalid()
        if len(encoded) > MAX_SEED_MATERIAL_BYTES:
            _invalid()
        checked.append(item)

    # Canonical JSON rather than a delimiter join. Joining with NUL was
    # ambiguous because NUL is a legal character inside a string, so
    # ["a", "\x00b"] and ["a\x00", "b"] produced identical material and the
    # same seed. A JSON array encodes its own boundaries, so no member can
    # impersonate a separator.
    canonical = json.dumps(
        ["prompt-expansion-seed-v1", checked],
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % SELECTION_SEED_SPACE
