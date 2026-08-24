"""Durable, chat-scoped storage for complete Prompt Library expansion drafts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, NoReturn, cast

from sqlalchemy import select, text, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .domain import utcnow
from .models import (
    Chat,
    PromptExpansionBatch,
    PromptExpansionItem,
    PromptTemplateDefinition,
    PromptTemplateRevision,
)
from .prompt_expansion import (
    MAX_EXPANSION_ITEMS,
    PROMPT_EXPANSION_CODEC_VERSION,
    ExpansionPlan,
    ExpansionRequest,
    PromptExpansionError,
    complete_prompt_expansion_with_model_values,
    expand_prompt_template,
    expansion_plan_payload,
    expansion_plan_payload_digest,
    expansion_request_payload,
    parse_expansion_plan_payload,
    parse_expansion_request,
)
from .prompt_expansion_schema import (
    MAX_EXPANSION_EVIDENCE_JSON_CHARS,
    MAX_EXPANSION_MODEL_SNAPSHOT_JSON_CHARS,
    MAX_EXPANSION_REQUEST_JSON_CHARS,
)
from .prompt_model_values import (
    PROMPT_MODEL_VALUES_VERSION,
    PromptModelValues,
    PromptModelValuesError,
    parse_prompt_model_values,
    prompt_model_slot_contract,
    prompt_model_values_sha256,
)
from .prompt_templates import (
    MAX_TEMPLATE_RENDERED_CHARS,
    PromptTemplateContract,
    PromptTemplateError,
    PromptTemplateSlotMode,
    PromptTemplateVariationScope,
    parse_prompt_template_contract,
    prompt_template_contract_sha256,
    render_prompt_template,
)

PROMPT_EXPANSION_STORE_INVALID = "Prompt expansion storage is invalid."
PROMPT_EXPANSION_STORE_CONFLICT = "Prompt expansion storage changed."
MAX_IDEMPOTENCY_KEY_CHARS = 200
MAX_MODEL_ID_CHARS = 200


class PromptExpansionStoreError(ValueError):
    """A fixed refusal that never echoes prompt, model, or identity content."""

    def __init__(self, message: str = PROMPT_EXPANSION_STORE_INVALID) -> None:
        super().__init__(message)


class PromptExpansionStoreConflict(PromptExpansionStoreError):
    """An optimistic-write conflict with a fixed, non-echoing message."""

    def __init__(self) -> None:
        super().__init__(PROMPT_EXPANSION_STORE_CONFLICT)


def _invalid() -> NoReturn:
    raise PromptExpansionStoreError()


def _conflict() -> NoReturn:
    raise PromptExpansionStoreConflict()


@dataclass(frozen=True, slots=True)
class PromptExpansionModelSnapshot:
    """Closed provenance for the model-value boundary, or its absence."""

    version: int
    kind: str
    adapter_id: str | None = field(default=None, repr=False)
    model_id: str | None = field(default=None, repr=False)
    values_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class StoredExpansion:
    """A fully revalidated batch and its ordered items."""

    batch: PromptExpansionBatch
    items: tuple[PromptExpansionItem, ...]
    selection_seed: int
    replayed: bool


@dataclass(frozen=True, slots=True)
class _VerifiedStored:
    stored: StoredExpansion
    request: ExpansionRequest
    request_payload: dict[str, Any] = field(repr=False)
    model_snapshot: PromptExpansionModelSnapshot = field(repr=False)
    contract: PromptTemplateContract = field(repr=False)
    original_receipt: dict[str, Any] = field(repr=False)
    current_receipt: dict[str, Any] = field(repr=False)


def _bounded_string(value: object, *, maximum: int) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        _invalid()
    text = value
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        _invalid()
    return text


def _identifier(value: object) -> str:
    return _bounded_string(value, maximum=40)


def _key(value: object) -> str:
    return _bounded_string(value, maximum=MAX_IDEMPOTENCY_KEY_CHARS)


def _positive(value: object) -> int:
    if type(value) is not int or value < 1:
        _invalid()
    return value


def _sha256(value: object) -> str:
    if type(value) is not str or len(value) != 64:
        _invalid()
    digest = value
    if any(character not in "0123456789abcdef" for character in digest):
        _invalid()
    return digest


def _prompt(value: object) -> str:
    return _bounded_string(value, maximum=MAX_TEMPLATE_RENDERED_CHARS)


def _prompt_sha256(prompt: str) -> str:
    material = "\x00".join(("prompt-expansion-rendered-v1", prompt))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        _invalid()


def _load_canonical_json(value: object, *, maximum: int) -> object:
    if type(value) is not str or not value or len(value) > maximum:
        _invalid()
    raw = value
    try:
        if len(raw.encode("utf-8")) > maximum * 4:
            _invalid()
    except UnicodeEncodeError:
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
        decoded = json.loads(raw, object_pairs_hook=exact_object, parse_constant=reject_constant)
    except PromptExpansionStoreError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError, RecursionError):
        _invalid()
    if _canonical_json(decoded) != raw:
        _invalid()
    return decoded


def parse_expansion_model_snapshot(value: object) -> PromptExpansionModelSnapshot:
    """Detach and validate one exact model provenance union."""

    if type(value) is not dict:
        _invalid()
    payload = cast(dict[object, object], value)
    if any(type(key) is not str for key in payload):
        _invalid()
    kind = payload.get("kind")
    if type(payload.get("version")) is not int or payload["version"] != 1:
        _invalid()
    if kind == "deterministic":
        if frozenset(payload) != {"version", "kind"}:
            _invalid()
        return PromptExpansionModelSnapshot(version=1, kind="deterministic")
    if kind != "model" or frozenset(payload) != {
        "version",
        "kind",
        "adapter_id",
        "model_id",
        "values_sha256",
    }:
        _invalid()
    return PromptExpansionModelSnapshot(
        version=1,
        kind="model",
        adapter_id=_bounded_string(payload["adapter_id"], maximum=MAX_MODEL_ID_CHARS),
        model_id=_bounded_string(payload["model_id"], maximum=MAX_MODEL_ID_CHARS),
        values_sha256=_sha256(payload["values_sha256"]),
    )


def expansion_model_snapshot_payload(snapshot: PromptExpansionModelSnapshot) -> dict[str, object]:
    """Return the unique JSON payload for a revalidated snapshot."""

    if type(snapshot) is not PromptExpansionModelSnapshot:
        _invalid()
    if snapshot.kind == "deterministic":
        candidate: dict[str, object] = {"version": snapshot.version, "kind": snapshot.kind}
    else:
        candidate = {
            "version": snapshot.version,
            "kind": snapshot.kind,
            "adapter_id": snapshot.adapter_id,
            "model_id": snapshot.model_id,
            "values_sha256": snapshot.values_sha256,
        }
    if parse_expansion_model_snapshot(candidate) != snapshot:
        _invalid()
    return candidate


def _snapshot(value: object) -> tuple[PromptExpansionModelSnapshot, dict[str, object], str]:
    if type(value) is PromptExpansionModelSnapshot:
        parsed = parse_expansion_model_snapshot(expansion_model_snapshot_payload(value))
    else:
        parsed = parse_expansion_model_snapshot(value)
    payload = expansion_model_snapshot_payload(parsed)
    encoded = _canonical_json(payload)
    if len(encoded) > MAX_EXPANSION_MODEL_SNAPSHOT_JSON_CHARS:
        _invalid()
    return parsed, payload, encoded


def _complete_plan(plan: object) -> tuple[ExpansionPlan, dict[str, Any], str]:
    if type(plan) is not ExpansionPlan:
        _invalid()
    try:
        payload = expansion_plan_payload(plan)
        normalized = parse_expansion_plan_payload(payload)
        digest = expansion_plan_payload_digest(normalized)
    except PromptExpansionError:
        raise PromptExpansionStoreError() from None
    if normalized["pending_model_slots"]:
        _invalid()
    return plan, normalized, digest


def _request(value: object) -> tuple[ExpansionRequest, dict[str, Any], str]:
    if type(value) is not ExpansionRequest:
        _invalid()
    try:
        payload = expansion_request_payload(value)
        parsed = parse_expansion_request(payload)
    except PromptExpansionError:
        raise PromptExpansionStoreError() from None
    encoded = _canonical_json(payload)
    if len(encoded) > MAX_EXPANSION_REQUEST_JSON_CHARS:
        _invalid()
    return parsed, payload, encoded


def _resolved_contract(
    session: Session, request: ExpansionRequest, *, creating: bool
) -> tuple[PromptTemplateDefinition, PromptTemplateRevision, PromptTemplateContract]:
    definition = session.get(PromptTemplateDefinition, request.definition_id)
    revision = session.get(PromptTemplateRevision, request.revision_id)
    if (
        definition is None
        or revision is None
        or revision.prompt_template_id != definition.id
        or revision.schema_version != 1
        or revision.contract_sha256 != request.contract_sha256
        or (creating and (definition.archived or definition.current_revision_id != revision.id))
    ):
        _invalid()
    try:
        contract = parse_prompt_template_contract(revision.contract_json)
        digest = prompt_template_contract_sha256(contract)
    except PromptTemplateError:
        raise PromptExpansionStoreError() from None
    if digest != revision.contract_sha256:
        _invalid()
    return definition, revision, contract


def _items(session: Session, batch_id: str, expected_count: int) -> tuple[PromptExpansionItem, ...]:
    if (
        type(expected_count) is not int
        or expected_count < 1
        or expected_count > MAX_EXPANSION_ITEMS
    ):
        _invalid()
    return tuple(
        session.scalars(
            select(PromptExpansionItem)
            .where(PromptExpansionItem.batch_id == batch_id)
            .order_by(PromptExpansionItem.ordinal)
            .limit(expected_count + 1)
            .execution_options(populate_existing=True)
        ).all()
    )


def _receipt(
    request_payload: dict[str, Any],
    items: tuple[PromptExpansionItem, ...],
    *,
    contract: PromptTemplateContract,
    request: ExpansionRequest,
    original: bool,
) -> dict[str, Any]:
    entries: list[dict[str, object]] = []
    for expected, item in enumerate(items, start=1):
        if type(item.ordinal) is not int or item.ordinal != expected:
            _invalid()
        if type(item.review_version) is not int or item.review_version < 1:
            _invalid()
        if type(item.reroll_count) is not int or item.reroll_count < 0:
            _invalid()
        if type(item.selected) is not bool:
            _invalid()
        review_prompt = _prompt(item.reviewed_prompt)
        review_digest = _sha256(item.reviewed_sha256)
        if _prompt_sha256(review_prompt) != review_digest:
            _invalid()
        evidence_raw = item.original_evidence_json if original else item.current_evidence_json
        evidence = _load_canonical_json(evidence_raw, maximum=MAX_EXPANSION_EVIDENCE_JSON_CHARS)
        prompt = _render_current_evidence(contract, request, evidence, ordinal=expected)
        stored_digest = _prompt_sha256(prompt)
        if original and (
            prompt != _prompt(item.original_rendered_prompt)
            or stored_digest != _sha256(item.original_rendered_sha256)
        ):
            _invalid()
        entries.append(
            {
                "ordinal": expected,
                "rendered_prompt": prompt,
                "rendered_sha256": stored_digest,
                "evidence": evidence,
            }
        )
    candidate: dict[str, object] = {
        "codec_version": PROMPT_EXPANSION_CODEC_VERSION,
        "request": request_payload,
        "template_body": contract.body,
        "pending_model_slots": [],
        "items": entries,
    }
    try:
        return parse_expansion_plan_payload(candidate)
    except PromptExpansionError:
        raise PromptExpansionStoreError() from None


def _render_current_evidence(
    contract: PromptTemplateContract,
    request: ExpansionRequest,
    evidence: object,
    *,
    ordinal: int,
) -> str:
    """Validate one persisted evidence vector against the authoritative contract."""

    if type(evidence) is not list or len(evidence) != len(contract.slots):
        _invalid()
    supplied = dict(request.inputs)
    values: dict[str, str] = {}
    for slot, raw in zip(contract.slots, cast(list[object], evidence), strict=True):
        if type(raw) is not dict:
            _invalid()
        record = cast(dict[object, object], raw)
        if frozenset(record) != {
            "name",
            "mode",
            "variation_scope",
            "source",
            "value",
            "choice_index",
        }:
            _invalid()
        value = record["value"]
        index = record["choice_index"]
        if (
            record["name"] != slot.name
            or record["mode"] != slot.mode.value
            or record["variation_scope"] != slot.variation_scope.value
            or record["source"] != slot.mode.value
            or type(value) is not str
        ):
            _invalid()
        if slot.mode is PromptTemplateSlotMode.FIXED:
            if value != slot.fixed_value or index is not None:
                _invalid()
        elif slot.mode is PromptTemplateSlotMode.INPUT:
            raw_input = supplied.get(slot.name)
            expected = (
                raw_input
                if slot.variation_scope is PromptTemplateVariationScope.BATCH
                and type(raw_input) is str
                else raw_input[ordinal - 1]
                if slot.variation_scope is PromptTemplateVariationScope.ITEM
                and type(raw_input) is tuple
                and len(raw_input) == request.item_count
                else None
            )
            if value != expected or index is not None:
                _invalid()
        elif slot.mode is PromptTemplateSlotMode.CHOICE:
            if type(index) is not int or not 0 <= index < len(slot.choices):
                _invalid()
            if value != slot.choices[index]:
                _invalid()
        elif slot.mode is PromptTemplateSlotMode.MODEL:
            if index is not None:
                _invalid()
        else:
            _invalid()
        if slot.mode is not PromptTemplateSlotMode.FIXED:
            values[slot.name] = value
    try:
        return render_prompt_template(contract, values)
    except PromptTemplateError:
        raise PromptExpansionStoreError() from None


def _model_values_from_receipt(
    contract: PromptTemplateContract,
    request: ExpansionRequest,
    receipt: dict[str, Any],
) -> tuple[PromptModelValues, str]:
    """Reconstruct the exact model output carried by completed evidence."""

    try:
        model_contract = prompt_model_slot_contract(contract, item_count=request.item_count)
    except PromptModelValuesError:
        _invalid()
    raw_items = receipt["items"]
    if type(raw_items) is not list or len(raw_items) != request.item_count:
        _invalid()

    evidence_by_item: list[dict[str, dict[str, object]]] = []
    for raw_item in raw_items:
        if type(raw_item) is not dict or type(raw_item.get("evidence")) is not list:
            _invalid()
        indexed: dict[str, dict[str, object]] = {}
        for raw_evidence in raw_item["evidence"]:
            if type(raw_evidence) is not dict:
                _invalid()
            evidence = cast(dict[str, object], raw_evidence)
            name = evidence.get("name")
            if type(name) is not str or name in indexed:
                _invalid()
            indexed[name] = evidence
        evidence_by_item.append(indexed)

    batch_values: dict[str, str] = {}
    for slot in model_contract.batch_slots:
        observed: str | None = None
        for indexed in evidence_by_item:
            slot_evidence = indexed.get(slot.name)
            if (
                slot_evidence is None
                or slot_evidence.get("mode") != PromptTemplateSlotMode.MODEL.value
                or slot_evidence.get("variation_scope") != PromptTemplateVariationScope.BATCH.value
                or type(slot_evidence.get("value")) is not str
                or slot_evidence.get("choice_index") is not None
            ):
                _invalid()
            value = cast(str, slot_evidence["value"])
            if observed is not None and value != observed:
                _invalid()
            observed = value
        if observed is None:
            _invalid()
        batch_values[slot.name] = observed

    item_payloads: list[dict[str, object]] = []
    for ordinal, indexed in enumerate(evidence_by_item, start=1):
        item_values: dict[str, str] = {}
        for slot in model_contract.item_slots:
            slot_evidence = indexed.get(slot.name)
            if (
                slot_evidence is None
                or slot_evidence.get("mode") != PromptTemplateSlotMode.MODEL.value
                or slot_evidence.get("variation_scope") != PromptTemplateVariationScope.ITEM.value
                or type(slot_evidence.get("value")) is not str
                or slot_evidence.get("choice_index") is not None
            ):
                _invalid()
            item_values[slot.name] = cast(str, slot_evidence["value"])
        item_payloads.append({"ordinal": ordinal, "values": item_values})
    try:
        values = parse_prompt_model_values(
            {
                "version": PROMPT_MODEL_VALUES_VERSION,
                "batch_values": batch_values,
                "items": item_payloads,
            },
            contract=model_contract,
        )
        digest = prompt_model_values_sha256(values, contract=model_contract)
    except PromptModelValuesError:
        raise PromptExpansionStoreError() from None
    return values, digest


def _verify_original_plan(
    contract: PromptTemplateContract,
    request: ExpansionRequest,
    receipt: dict[str, Any],
    prompts: tuple[str, ...],
    snapshot: PromptExpansionModelSnapshot,
) -> None:
    """Reproduce deterministic or model-completed original authority exactly."""

    try:
        pending = expand_prompt_template(contract, request)
        if snapshot.kind == "deterministic":
            if not pending.complete:
                _invalid()
            expected = pending
        else:
            values, values_digest = _model_values_from_receipt(contract, request, receipt)
            if snapshot.values_sha256 != values_digest or pending.complete:
                _invalid()
            expected = complete_prompt_expansion_with_model_values(contract, pending, values)
        expected_payload = expansion_plan_payload(expected)
    except (PromptExpansionError, PromptTemplateError):
        raise PromptExpansionStoreError() from None
    if expected_payload != receipt:
        _invalid()
    if tuple(item.rendered_prompt for item in expected.items) != prompts:
        _invalid()


def _verify_stored(
    session: Session, batch: PromptExpansionBatch, *, replayed: bool
) -> _VerifiedStored:
    if (
        type(batch.plan_version) is not int
        or batch.plan_version < 1
        or batch.state not in {"draft", "queued"}
        or batch.schema_version != 1
        or batch.codec_version != PROMPT_EXPANSION_CODEC_VERSION
    ):
        _invalid()
    request_object = _load_canonical_json(
        batch.request_json, maximum=MAX_EXPANSION_REQUEST_JSON_CHARS
    )
    try:
        request = parse_expansion_request(request_object)
        request_payload = expansion_request_payload(request)
    except PromptExpansionError:
        raise PromptExpansionStoreError() from None
    if (
        request.definition_id != batch.prompt_template_id
        or request.revision_id != batch.prompt_template_revision_id
        or request.contract_sha256 != batch.contract_sha256
        or _canonical_json(request_payload) != batch.request_json
    ):
        _invalid()
    snapshot_object = _load_canonical_json(
        batch.model_snapshot_json, maximum=MAX_EXPANSION_MODEL_SNAPSHOT_JSON_CHARS
    )
    snapshot, _snapshot_payload, encoded_snapshot = _snapshot(snapshot_object)
    if encoded_snapshot != batch.model_snapshot_json:
        _invalid()
    _definition, revision, contract = _resolved_contract(session, request, creating=False)
    if revision.schema_version != batch.schema_version:
        _invalid()
    items = _items(session, batch.id, request.item_count)
    if len(items) != request.item_count:
        _invalid()
    raw_selected = session.execute(
        text(
            "SELECT id, selected, typeof(selected) "
            "FROM prompt_expansion_items "
            "WHERE batch_id = :batch_id ORDER BY ordinal "
            "LIMIT :read_limit"
        ),
        {"batch_id": batch.id, "read_limit": request.item_count + 1},
    ).all()
    if len(raw_selected) != len(items):
        _invalid()
    for item, raw in zip(items, raw_selected, strict=True):
        if (
            raw.id != item.id
            or type(raw.selected) is not int
            or raw.selected not in (0, 1)
            or raw[2] != "integer"
        ):
            _invalid()
    edits = 0
    for item in items:
        if (
            type(item.review_version) is not int
            or type(item.reroll_count) is not int
            or item.reroll_count > item.review_version - 1
        ):
            _invalid()
        edits += item.review_version - 1
    # Every item CAS advances the batch once. Queueing is the one legal
    # batch-only update and contributes exactly one final transition version.
    expected_plan_version = 1 + edits + (1 if batch.state == "queued" else 0)
    if batch.plan_version != expected_plan_version:
        _invalid()
    original = _receipt(request_payload, items, contract=contract, request=request, original=True)
    current = _receipt(request_payload, items, contract=contract, request=request, original=False)
    if expansion_plan_payload_digest(original) != _sha256(batch.original_plan_sha256):
        _invalid()
    if expansion_plan_payload_digest(current) != _sha256(batch.plan_sha256):
        _invalid()
    if snapshot.kind == "model":
        _current_values, current_values_digest = _model_values_from_receipt(
            contract, request, current
        )
        if snapshot.values_sha256 != current_values_digest:
            _invalid()
    _verify_original_plan(
        contract,
        request,
        original,
        tuple(item.original_rendered_prompt for item in items),
        snapshot,
    )
    return _VerifiedStored(
        stored=StoredExpansion(
            batch=batch,
            items=items,
            selection_seed=request.selection_seed,
            replayed=replayed,
        ),
        request=request,
        request_payload=request_payload,
        model_snapshot=snapshot,
        contract=contract,
        original_receipt=original,
        current_receipt=current,
    )


def _existing(session: Session, chat_id: str, key: str) -> PromptExpansionBatch | None:
    return session.scalar(
        select(PromptExpansionBatch)
        .where(
            PromptExpansionBatch.chat_id == chat_id,
            PromptExpansionBatch.idempotency_key == key,
        )
        .execution_options(populate_existing=True)
    )


def _exact_replay(
    verified: _VerifiedStored,
    *,
    request_payload: dict[str, Any],
    plan_payload: dict[str, Any],
    plan_digest: str,
    snapshot: PromptExpansionModelSnapshot,
) -> StoredExpansion:
    if (
        verified.request_payload != request_payload
        or verified.original_receipt != plan_payload
        or verified.stored.batch.original_plan_sha256 != plan_digest
        or verified.model_snapshot != snapshot
    ):
        _invalid()
    return verified.stored


def create_or_replay_expansion(
    session: Session,
    chat_id: object,
    idempotency_key: object,
    request: ExpansionRequest,
    plan: ExpansionPlan,
    model_snapshot: object,
) -> StoredExpansion:
    """Create exactly one complete batch or replay its exact chat-local winner."""

    chat = _identifier(chat_id)
    key = _key(idempotency_key)
    parsed_request, request_payload, request_json = _request(request)
    _plan, plan_payload, plan_digest = _complete_plan(plan)
    snapshot, _snapshot_payload, snapshot_json = _snapshot(model_snapshot)
    if plan_payload["request"] != request_payload:
        _invalid()
    if session.get(Chat, chat) is None:
        _invalid()
    definition, revision, contract = _resolved_contract(session, parsed_request, creating=True)
    _verify_original_plan(
        contract,
        parsed_request,
        plan_payload,
        tuple(item.rendered_prompt or "" for item in plan.items),
        snapshot,
    )

    existing = _existing(session, chat, key)
    if existing is not None:
        verified = _verify_stored(session, existing, replayed=True)
        return _exact_replay(
            verified,
            request_payload=request_payload,
            plan_payload=plan_payload,
            plan_digest=plan_digest,
            snapshot=snapshot,
        )

    try:
        with session.begin_nested():
            batch = PromptExpansionBatch(
                chat_id=chat,
                idempotency_key=key,
                prompt_template_id=definition.id,
                prompt_template_revision_id=revision.id,
                schema_version=revision.schema_version,
                contract_sha256=revision.contract_sha256,
                codec_version=PROMPT_EXPANSION_CODEC_VERSION,
                request_json=request_json,
                model_snapshot_json=snapshot_json,
                original_plan_sha256=plan_digest,
                plan_sha256=plan_digest,
                plan_version=1,
                state="draft",
            )
            session.add(batch)
            session.flush()
            for expanded, item_payload in zip(plan.items, plan_payload["items"], strict=True):
                if expanded.rendered_prompt is None or expanded.rendered_sha256 is None:
                    _invalid()
                evidence_json = _canonical_json(item_payload["evidence"])
                session.add(
                    PromptExpansionItem(
                        batch_id=batch.id,
                        ordinal=expanded.ordinal,
                        original_evidence_json=evidence_json,
                        current_evidence_json=evidence_json,
                        original_rendered_prompt=expanded.rendered_prompt,
                        original_rendered_sha256=expanded.rendered_sha256,
                        reviewed_prompt=expanded.rendered_prompt,
                        reviewed_sha256=expanded.rendered_sha256,
                        selected=True,
                        review_version=1,
                        reroll_count=0,
                    )
                )
            session.flush()
    except IntegrityError:
        winner = _existing(session, chat, key)
        if winner is None:
            raise PromptExpansionStoreError() from None
        verified = _verify_stored(session, winner, replayed=True)
        return _exact_replay(
            verified,
            request_payload=request_payload,
            plan_payload=plan_payload,
            plan_digest=plan_digest,
            snapshot=snapshot,
        )
    return _verify_stored(session, batch, replayed=False).stored


def read_expansion(session: Session, chat_id: object, batch_id: object) -> StoredExpansion:
    """Read and fully revalidate one batch inside its owning chat."""

    chat = _identifier(chat_id)
    identity = _identifier(batch_id)
    batch = session.scalar(
        select(PromptExpansionBatch)
        .where(PromptExpansionBatch.id == identity, PromptExpansionBatch.chat_id == chat)
        .execution_options(populate_existing=True)
    )
    if batch is None:
        _invalid()
    return _verify_stored(session, batch, replayed=True).stored


def _machine_plan_digest(receipt: dict[str, Any]) -> str:
    """Review text and selection never impersonate machine expansion evidence."""

    try:
        return expansion_plan_payload_digest(receipt)
    except PromptExpansionError:
        raise PromptExpansionStoreError() from None


def _cas_item_and_batch(
    session: Session,
    verified: _VerifiedStored,
    item: PromptExpansionItem,
    *,
    expected_item_version: int,
    expected_plan_version: int,
    prompt: str,
    digest: str,
    selected: bool,
    evidence_json: str,
    reroll_count: int,
    plan_digest: str,
) -> StoredExpansion:
    batch = verified.stored.batch
    if (
        batch.state != "draft"
        or item.review_version != expected_item_version
        or batch.plan_version != expected_plan_version
    ):
        _conflict()
    now = utcnow()
    with session.begin_nested():
        changed_item = cast(
            CursorResult[Any],
            session.execute(
                update(PromptExpansionItem)
                .execution_options(synchronize_session=False)
                .where(
                    PromptExpansionItem.id == item.id,
                    PromptExpansionItem.batch_id == batch.id,
                    PromptExpansionItem.review_version == expected_item_version,
                )
                .values(
                    current_evidence_json=evidence_json,
                    reviewed_prompt=prompt,
                    reviewed_sha256=digest,
                    selected=selected,
                    review_version=expected_item_version + 1,
                    reroll_count=reroll_count,
                    updated_at=now,
                )
            ),
        )
        if changed_item.rowcount != 1:
            _conflict()
        changed_batch = cast(
            CursorResult[Any],
            session.execute(
                update(PromptExpansionBatch)
                .execution_options(synchronize_session=False)
                .where(
                    PromptExpansionBatch.id == batch.id,
                    PromptExpansionBatch.chat_id == batch.chat_id,
                    PromptExpansionBatch.state == "draft",
                    PromptExpansionBatch.plan_version == expected_plan_version,
                )
                .values(
                    plan_sha256=plan_digest,
                    plan_version=expected_plan_version + 1,
                    updated_at=now,
                )
            ),
        )
        if changed_batch.rowcount != 1:
            _conflict()
    refreshed = session.get(PromptExpansionBatch, batch.id, populate_existing=True)
    if refreshed is None:
        _conflict()
    return _verify_stored(session, refreshed, replayed=False).stored


def update_expansion_item(
    session: Session,
    chat_id: object,
    batch_id: object,
    item_id: object,
    *,
    expected_item_version: object,
    expected_plan_version: object,
    reviewed_prompt: object,
    selected: object,
) -> StoredExpansion:
    """CAS one user edit/selection and its enclosing canonical plan receipt."""

    chat = _identifier(chat_id)
    identity = _identifier(batch_id)
    item_identity = _identifier(item_id)
    item_version = _positive(expected_item_version)
    plan_version = _positive(expected_plan_version)
    prompt = _prompt(reviewed_prompt)
    if type(selected) is not bool:
        _invalid()
    verified = _verified_by_identity(session, chat, identity)
    item = next((one for one in verified.stored.items if one.id == item_identity), None)
    if item is None:
        _invalid()
    digest = _prompt_sha256(prompt)
    plan_digest = _machine_plan_digest(verified.current_receipt)
    return _cas_item_and_batch(
        session,
        verified,
        item,
        expected_item_version=item_version,
        expected_plan_version=plan_version,
        prompt=prompt,
        digest=digest,
        selected=selected,
        evidence_json=item.current_evidence_json,
        reroll_count=item.reroll_count,
        plan_digest=plan_digest,
    )


def _verified_by_identity(session: Session, chat_id: str, batch_id: str) -> _VerifiedStored:
    batch = session.scalar(
        select(PromptExpansionBatch)
        .where(PromptExpansionBatch.id == batch_id, PromptExpansionBatch.chat_id == chat_id)
        .execution_options(populate_existing=True)
    )
    if batch is None:
        _invalid()
    return _verify_stored(session, batch, replayed=True)


def _validate_reroll_target(
    contract: PromptTemplateContract,
    request: ExpansionRequest,
    plan: ExpansionPlan,
    ordinal: int,
) -> None:
    expanded = plan.items[ordinal - 1]
    if expanded.rendered_prompt is None:
        _invalid()
    supplied = dict(request.inputs)
    try:
        deterministic = expand_prompt_template(contract, request)
    except PromptExpansionError:
        raise PromptExpansionStoreError() from None
    deterministic_evidence = deterministic.items[ordinal - 1].evidence
    values: dict[str, str] = {}
    if len(expanded.evidence) != len(contract.slots) or len(deterministic_evidence) != len(
        contract.slots
    ):
        _invalid()
    for slot, evidence, authority in zip(
        contract.slots,
        expanded.evidence,
        deterministic_evidence,
        strict=True,
    ):
        if (
            evidence.name != slot.name
            or evidence.mode is not slot.mode
            or evidence.variation_scope is not slot.variation_scope
            or evidence.value is None
        ):
            _invalid()
        if (
            slot.mode is not PromptTemplateSlotMode.MODEL
            and slot.variation_scope is PromptTemplateVariationScope.BATCH
            and evidence != authority
        ):
            _invalid()
        if slot.mode is PromptTemplateSlotMode.FIXED:
            if evidence.value != slot.fixed_value or evidence.choice_index is not None:
                _invalid()
        elif slot.mode is PromptTemplateSlotMode.INPUT:
            raw = supplied.get(slot.name)
            expected = raw if type(raw) is str else raw[ordinal - 1] if type(raw) is tuple else None
            if evidence.value != expected or evidence.choice_index is not None:
                _invalid()
        elif slot.mode is PromptTemplateSlotMode.CHOICE:
            index = evidence.choice_index
            if type(index) is not int or not 0 <= index < len(slot.choices):
                _invalid()
            if slot.choices[index] != evidence.value:
                _invalid()
        elif slot.mode is PromptTemplateSlotMode.MODEL and evidence.choice_index is not None:
            _invalid()
        if slot.mode is not PromptTemplateSlotMode.FIXED:
            values[slot.name] = evidence.value
    try:
        rendered = render_prompt_template(contract, values)
    except PromptTemplateError:
        raise PromptExpansionStoreError() from None
    if rendered != expanded.rendered_prompt:
        _invalid()


def reroll_expansion_item(
    session: Session,
    chat_id: object,
    batch_id: object,
    item_id: object,
    *,
    expected_item_version: object,
    expected_plan_version: object,
    replacement_plan: ExpansionPlan,
    model_snapshot: object,
) -> StoredExpansion:
    """Replace one pre-queue candidate from a complete, narrowly checked plan.

    ``ExpansionPlan`` is the smallest existing typed surface the codec can
    validate completely.  All non-target entries must be byte-equivalent to the
    current receipt, so the caller receives authority over only one item.
    """

    chat = _identifier(chat_id)
    identity = _identifier(batch_id)
    item_identity = _identifier(item_id)
    item_version = _positive(expected_item_version)
    plan_version = _positive(expected_plan_version)
    plan, payload, plan_digest = _complete_plan(replacement_plan)
    snapshot, _snapshot_payload, snapshot_json = _snapshot(model_snapshot)
    verified = _verified_by_identity(session, chat, identity)
    if (
        payload["request"] != verified.request_payload
        or payload["template_body"] != verified.contract.body
        or snapshot != verified.model_snapshot
        or snapshot_json != verified.stored.batch.model_snapshot_json
    ):
        _invalid()
    item = next((one for one in verified.stored.items if one.id == item_identity), None)
    if item is None:
        _invalid()
    for position, (candidate, current) in enumerate(
        zip(payload["items"], verified.current_receipt["items"], strict=True), start=1
    ):
        if position != item.ordinal and candidate != current:
            _invalid()
    if snapshot.kind == "model":
        _values, values_digest = _model_values_from_receipt(
            verified.contract, verified.request, payload
        )
        if snapshot.values_sha256 != values_digest:
            _invalid()
    else:
        try:
            prompt_model_slot_contract(verified.contract, item_count=verified.request.item_count)
        except PromptModelValuesError:
            pass
        else:
            _invalid()
    _validate_reroll_target(verified.contract, verified.request, plan, item.ordinal)
    expanded = plan.items[item.ordinal - 1]
    if expanded.rendered_prompt is None or expanded.rendered_sha256 is None:
        _invalid()
    evidence_json = _canonical_json(payload["items"][item.ordinal - 1]["evidence"])
    return _cas_item_and_batch(
        session,
        verified,
        item,
        expected_item_version=item_version,
        expected_plan_version=plan_version,
        prompt=expanded.rendered_prompt,
        digest=expanded.rendered_sha256,
        selected=item.selected,
        evidence_json=evidence_json,
        reroll_count=item.reroll_count + 1,
        plan_digest=plan_digest,
    )
