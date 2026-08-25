"""Claim one reviewed Prompt Library draft for a chat turn.

The incoming source object is only a concurrency witness. The stored batch,
item and immutable revision remain authoritative. Returned provenance is a
closed, text-free witness suitable for message metadata and run provenance.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from sqlalchemy import exists, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from .domain import utcnow
from .models import (
    ModelAssetInstall,
    PromptExpansionBatch,
    PromptExpansionItem,
    PromptTemplateRevision,
)
from .prompt_expansion_store import PromptExpansionStoreError, read_expansion
from .prompt_library import PromptLibraryError, validate_prompt_template_resources
from .prompt_templates import (
    PromptTemplateError,
    parse_prompt_template_contract,
    prompt_template_contract_payload,
)
from .schemas import PromptComposerSourceIn
from .workflow_bindings import WorkflowBindingError, materialize_model_asset

PROMPT_SOURCE_INVALID = "Prompt Library source is invalid."
PROMPT_SOURCE_CONFLICT = "Prompt Library source changed before it could be used."

_WITNESS_KEYS = frozenset(
    {
        "version",
        "kind",
        "source_chat_id",
        "batch_id",
        "review_plan_version",
        "queued_plan_version",
        "plan_sha256",
        "item_id",
        "item_ordinal",
        "item_review_version",
        "reviewed_sha256",
        "submitted_sha256",
        "relation",
        "prompt_template_id",
        "prompt_template_revision_id",
        "contract_sha256",
        "resource_policy",
    }
)


class PromptExpansionUseError(ValueError):
    pass


class PromptExpansionUseConflict(PromptExpansionUseError):
    pass


@dataclass(frozen=True, slots=True)
class PromptSourceResourceOverrides:
    workflow_revision_id: str | None
    lora_settings: tuple[dict[str, object], ...] | None


@dataclass(frozen=True, slots=True)
class PromptBatchQueueSelection:
    batch: PromptExpansionBatch
    items: tuple[PromptExpansionItem, ...]
    resource_policy: dict[str, object]
    allocated_resource_policies: tuple[dict[str, object], ...]


def _invalid() -> NoReturn:
    raise PromptExpansionUseError(PROMPT_SOURCE_INVALID)


def _conflict() -> NoReturn:
    raise PromptExpansionUseConflict(PROMPT_SOURCE_CONFLICT)


def _prompt_sha256(prompt: str) -> str:
    material = "\x00".join(("prompt-expansion-rendered-v1", prompt))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _canonical_resource_policy(value: object) -> dict[str, object]:
    """Reparse a historical policy through the public exact contract parser."""

    try:
        contract = parse_prompt_template_contract(
            {
                "schema_version": 1,
                "operation": "text_to_image",
                "body": "prompt",
                "slots": [],
                "resource_policy": value,
            }
        )
        policy = prompt_template_contract_payload(contract)["resource_policy"]
    except PromptTemplateError:
        _invalid()
    if type(policy) is not dict:
        _invalid()
    return cast(dict[str, object], policy)


def _allocate_resource_policy(
    policy: object,
    selection_seed: int,
    item_ordinal: int,
) -> dict[str, object]:
    """Freeze one deterministic, exact resource policy for one draft item."""

    canonical = _canonical_resource_policy(policy)
    if canonical["mode"] == "pool":
        strategy = canonical.get("strategy")
        options = canonical.get("options")
        if (
            type(selection_seed) is not int
            or selection_seed < 0
            or type(item_ordinal) is not int
            or item_ordinal < 1
            or strategy not in {"random", "round_robin"}
            or type(options) is not list
            or not options
        ):
            _invalid()
        if strategy == "round_robin":
            index = (selection_seed + item_ordinal - 1) % len(options)
        else:
            material = "\x00".join(
                ("prompt-template-workflow-pool-v1", str(selection_seed), str(item_ordinal))
            )
            index = int.from_bytes(
                hashlib.sha256(material.encode("ascii")).digest()[:8],
                "big",
            ) % len(options)
        selected_option = options[index]
        if type(selected_option) is not dict:
            _invalid()
        option = cast(dict[str, object], selected_option)
        canonical = _canonical_resource_policy(
            {
                "mode": "fixed",
                "workflow_revision_id": option.get("workflow_revision_id"),
                "lora_policy": option.get("lora_policy"),
            }
        )
    if canonical["mode"] != "fixed":
        return canonical
    lora_policy = canonical.get("lora_policy")
    if type(lora_policy) is not dict:
        _invalid()
    lora_policy = cast(dict[str, object], lora_policy)
    if lora_policy.get("mode") != "pool":
        return canonical
    strategy = lora_policy.get("strategy")
    stacks = lora_policy.get("stacks")
    if (
        type(selection_seed) is not int
        or selection_seed < 0
        or type(item_ordinal) is not int
        or item_ordinal < 1
        or strategy not in {"random", "round_robin"}
        or type(stacks) is not list
        or not stacks
    ):
        _invalid()
    if strategy == "round_robin":
        index = (selection_seed + item_ordinal - 1) % len(stacks)
    else:
        material = "\x00".join(
            ("prompt-template-resource-pool-v1", str(selection_seed), str(item_ordinal))
        )
        index = int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big") % len(
            stacks
        )
    selected = stacks[index]
    if type(selected) is not list:
        _invalid()
    allocated = {
        "mode": "fixed",
        "workflow_revision_id": canonical.get("workflow_revision_id"),
        "lora_policy": {
            "mode": "fixed",
            "stack": [dict(cast(dict[str, object], item)) for item in selected],
        },
    }
    return _canonical_resource_policy(allocated)


def _read_authoritative_source(
    session: Session,
    chat_id: str,
    source: PromptComposerSourceIn,
) -> tuple[PromptExpansionBatch, PromptExpansionItem, dict[str, object]]:
    try:
        stored = read_expansion(session, chat_id, source.batch_id)
    except PromptExpansionStoreError:
        _conflict()
    batch = stored.batch
    if (
        batch.state != "draft"
        or batch.plan_version != source.expected_plan_version
        or batch.plan_sha256 != source.expected_plan_sha256
        or batch.prompt_template_id != source.prompt_template_id
        or batch.prompt_template_revision_id != source.prompt_template_revision_id
        or batch.contract_sha256 != source.contract_sha256
    ):
        _conflict()
    item = next((candidate for candidate in stored.items if candidate.id == source.item_id), None)
    if (
        item is None
        or item.review_version != source.expected_review_version
        or item.reviewed_sha256 != source.expected_reviewed_sha256
        or not item.selected
    ):
        _conflict()
    revision = session.get(PromptTemplateRevision, batch.prompt_template_revision_id)
    if (
        revision is None
        or revision.prompt_template_id != batch.prompt_template_id
        or revision.contract_sha256 != batch.contract_sha256
    ):
        _conflict()
    try:
        contract = parse_prompt_template_contract(revision.contract_json)
        policy = prompt_template_contract_payload(contract)["resource_policy"]
    except PromptTemplateError:
        _conflict()
    if type(policy) is not dict:
        _conflict()
    return (
        batch,
        item,
        _allocate_resource_policy(policy, stored.selection_seed, item.ordinal),
    )


def claim_prompt_source(
    session: Session,
    chat_id: str,
    source: PromptComposerSourceIn,
    submitted_prompt: str,
) -> dict[str, Any]:
    """Revalidate and atomically queue one selected draft."""

    batch, item, resource_policy = _read_authoritative_source(session, chat_id, source)

    queued_version = batch.plan_version + 1
    changed = cast(
        CursorResult[Any],
        session.execute(
            update(PromptExpansionBatch)
            .execution_options(synchronize_session=False)
            .where(
                PromptExpansionBatch.id == batch.id,
                PromptExpansionBatch.chat_id == chat_id,
                PromptExpansionBatch.state == "draft",
                PromptExpansionBatch.plan_version == source.expected_plan_version,
                PromptExpansionBatch.plan_sha256 == source.expected_plan_sha256,
                PromptExpansionBatch.prompt_template_id == source.prompt_template_id,
                PromptExpansionBatch.prompt_template_revision_id
                == source.prompt_template_revision_id,
                PromptExpansionBatch.contract_sha256 == source.contract_sha256,
                exists(
                    select(PromptExpansionItem.id).where(
                        PromptExpansionItem.batch_id == PromptExpansionBatch.id,
                        PromptExpansionItem.id == source.item_id,
                        PromptExpansionItem.selected.is_(True),
                        PromptExpansionItem.review_version == source.expected_review_version,
                        PromptExpansionItem.reviewed_sha256 == source.expected_reviewed_sha256,
                    )
                ),
            )
            .values(state="queued", plan_version=queued_version)
        ),
    )
    if changed.rowcount != 1:
        _conflict()
    submitted_sha256 = _prompt_sha256(submitted_prompt)
    return {
        "version": 1,
        "kind": "prompt_template",
        "source_chat_id": chat_id,
        "batch_id": batch.id,
        "review_plan_version": source.expected_plan_version,
        "queued_plan_version": queued_version,
        "plan_sha256": batch.plan_sha256,
        "item_id": item.id,
        "item_ordinal": item.ordinal,
        "item_review_version": item.review_version,
        "reviewed_sha256": item.reviewed_sha256,
        "submitted_sha256": submitted_sha256,
        "relation": "exact" if submitted_sha256 == item.reviewed_sha256 else "edited",
        "prompt_template_id": batch.prompt_template_id,
        "prompt_template_revision_id": batch.prompt_template_revision_id,
        "contract_sha256": batch.contract_sha256,
        "resource_policy": resource_policy,
    }


def read_prompt_batch_queue_selection(
    session: Session,
    chat_id: str,
    batch_id: str,
    expected_plan_version: int,
    expected_plan_sha256: str,
    *,
    expected_engine: str,
) -> PromptBatchQueueSelection:
    """Read one exact draft and freeze its selected items for admission checks."""

    try:
        stored = read_expansion(session, chat_id, batch_id)
    except PromptExpansionStoreError:
        _conflict()
    batch = stored.batch
    if (
        batch.state != "draft"
        or batch.plan_version != expected_plan_version
        or batch.plan_sha256 != expected_plan_sha256
    ):
        _conflict()
    items = tuple(item for item in stored.items if item.selected)
    if not items:
        _invalid()
    revision = session.get(PromptTemplateRevision, batch.prompt_template_revision_id)
    if (
        revision is None
        or revision.prompt_template_id != batch.prompt_template_id
        or revision.contract_sha256 != batch.contract_sha256
    ):
        _conflict()
    try:
        contract = parse_prompt_template_contract(revision.contract_json)
        validate_prompt_template_resources(
            session,
            contract,
            expected_engine=expected_engine,
        )
        policy = prompt_template_contract_payload(contract)["resource_policy"]
    except (PromptLibraryError, PromptTemplateError):
        _conflict()
    if type(policy) is not dict:
        _conflict()
    canonical_policy = cast(dict[str, object], policy)
    return PromptBatchQueueSelection(
        batch=batch,
        items=items,
        resource_policy=canonical_policy,
        allocated_resource_policies=tuple(
            _allocate_resource_policy(canonical_policy, stored.selection_seed, item.ordinal)
            for item in items
        ),
    )


def claim_prompt_batch_queue(
    session: Session,
    selection: PromptBatchQueueSelection,
    queue_idempotency_key: str,
) -> tuple[dict[str, Any], ...]:
    """Atomically claim the exact reviewed selection after every admission check."""

    batch = selection.batch
    queued_version = batch.plan_version + 1
    changed = cast(
        CursorResult[Any],
        session.execute(
            update(PromptExpansionBatch)
            .execution_options(synchronize_session=False)
            .where(
                PromptExpansionBatch.id == batch.id,
                PromptExpansionBatch.chat_id == batch.chat_id,
                PromptExpansionBatch.state == "draft",
                PromptExpansionBatch.plan_version == batch.plan_version,
                PromptExpansionBatch.plan_sha256 == batch.plan_sha256,
            )
            .values(
                state="queued",
                plan_version=queued_version,
                queue_idempotency_key=queue_idempotency_key,
            )
        ),
    )
    if changed.rowcount != 1:
        _conflict()
    session.expire(batch)
    session.refresh(batch)
    return tuple(
        {
            "version": 1,
            "kind": "prompt_template",
            "source_chat_id": batch.chat_id,
            "batch_id": batch.id,
            "review_plan_version": queued_version - 1,
            "queued_plan_version": queued_version,
            "plan_sha256": batch.plan_sha256,
            "item_id": item.id,
            "item_ordinal": item.ordinal,
            "item_review_version": item.review_version,
            "reviewed_sha256": item.reviewed_sha256,
            "submitted_sha256": item.reviewed_sha256,
            "relation": "exact",
            "prompt_template_id": batch.prompt_template_id,
            "prompt_template_revision_id": batch.prompt_template_revision_id,
            "contract_sha256": batch.contract_sha256,
            "resource_policy": resource_policy,
        }
        for item, resource_policy in zip(
            selection.items,
            selection.allocated_resource_policies,
            strict=True,
        )
    )


def link_prompt_batch_execution(
    session: Session,
    selection: PromptBatchQueueSelection,
    work_plan_id: str,
    execution: tuple[tuple[str, str, int], ...],
) -> None:
    """Bind each selected item to its durable step, run, and sampled media seed."""

    if len(execution) != len(selection.items):
        _invalid()
    batch = selection.batch
    if batch.state != "queued" or batch.work_plan_id is not None or batch.queued_at is not None:
        _conflict()
    batch.work_plan_id = work_plan_id
    batch.queued_at = utcnow()
    session.flush()
    for item, (work_step_id, run_id, media_seed) in zip(selection.items, execution, strict=True):
        if (
            item.work_step_id is not None
            or item.run_id is not None
            or item.media_seed is not None
            or type(media_seed) is not int
            or not 0 <= media_seed < 2_147_483_648
        ):
            _conflict()
        item.work_step_id = work_step_id
        item.run_id = run_id
        item.media_seed = media_seed
    session.flush()


def inherit_prompt_source(value: object, submitted_prompt: str) -> dict[str, Any]:
    """Strictly copy a trusted historical witness for branch/regeneration."""

    if type(value) is not dict:
        _invalid()
    witness = cast(dict[object, object], value)
    if frozenset(witness) != _WITNESS_KEYS or any(type(key) is not str for key in witness):
        _invalid()
    if (
        witness["version"] != 1
        or witness["kind"] != "prompt_template"
        or witness["relation"] not in {"exact", "edited"}
    ):
        _invalid()
    for key in (
        "source_chat_id",
        "batch_id",
        "plan_sha256",
        "item_id",
        "reviewed_sha256",
        "submitted_sha256",
        "prompt_template_id",
        "prompt_template_revision_id",
        "contract_sha256",
    ):
        raw = witness[key]
        maximum = 64 if key.endswith("sha256") else 40
        if type(raw) is not str or not raw or len(raw) > maximum:
            _invalid()
        if key.endswith("sha256") and re.fullmatch(r"[0-9a-f]{64}", raw) is None:
            _invalid()
    for key in (
        "review_plan_version",
        "queued_plan_version",
        "item_ordinal",
        "item_review_version",
    ):
        raw = witness[key]
        if type(raw) is not int or raw < 1:
            _invalid()
    review_plan_version = cast(int, witness["review_plan_version"])
    queued_plan_version = cast(int, witness["queued_plan_version"])
    if queued_plan_version != review_plan_version + 1:
        _invalid()
    prior_relation = (
        "exact" if witness["submitted_sha256"] == witness["reviewed_sha256"] else "edited"
    )
    if witness["relation"] != prior_relation:
        _invalid()
    resource_policy = _canonical_resource_policy(witness["resource_policy"])
    copied = cast(dict[str, Any], {key: witness[key] for key in _WITNESS_KEYS})
    copied["resource_policy"] = resource_policy
    submitted_sha256 = _prompt_sha256(submitted_prompt)
    copied["submitted_sha256"] = submitted_sha256
    copied["relation"] = "exact" if submitted_sha256 == copied["reviewed_sha256"] else "edited"
    return copied


def prompt_source_resource_overrides(
    session: Session,
    chat_id: str,
    source: PromptComposerSourceIn | None,
    inherited_source: object | None,
    submitted_prompt: str,
) -> PromptSourceResourceOverrides:
    """Resolve an exact fixed policy before workflow/settings selection."""

    if source is not None and inherited_source is not None:
        _invalid()
    if source is not None:
        _, _, policy = _read_authoritative_source(session, chat_id, source)
    elif inherited_source is not None:
        inherited = inherit_prompt_source(inherited_source, submitted_prompt)
        policy = cast(dict[str, object], inherited["resource_policy"])
    else:
        return PromptSourceResourceOverrides(None, None)
    return _resource_overrides_for_policy(session, policy)


def prompt_batch_resource_overrides(
    session: Session,
    selection: PromptBatchQueueSelection,
) -> tuple[PromptSourceResourceOverrides, ...]:
    return tuple(
        _resource_overrides_for_policy(session, policy)
        for policy in selection.allocated_resource_policies
    )


def _resource_overrides_for_policy(
    session: Session,
    policy: dict[str, object],
) -> PromptSourceResourceOverrides:
    if policy["mode"] == "inherited":
        return PromptSourceResourceOverrides(None, None)
    workflow_revision_id = policy.get("workflow_revision_id")
    lora_policy = policy.get("lora_policy")
    if type(workflow_revision_id) is not str or type(lora_policy) is not dict:
        _invalid()
    lora_policy = cast(dict[str, object], lora_policy)
    mode = lora_policy.get("mode")
    if mode == "inherited_auto":
        return PromptSourceResourceOverrides(workflow_revision_id, None)
    if mode == "none":
        return PromptSourceResourceOverrides(workflow_revision_id, ())
    stack = lora_policy.get("stack")
    if mode != "fixed" or type(stack) is not list:
        _invalid()
    needed = {cast(str, item["sha256"]) for item in cast(list[dict[str, object]], stack)}
    candidates = session.scalars(
        select(ModelAssetInstall)
        .where(
            ModelAssetInstall.kind == "lora",
            ModelAssetInstall.active.is_(True),
            ModelAssetInstall.verified_at.is_not(None),
        )
        .order_by(ModelAssetInstall.id)
    ).all()
    asset_ids: dict[str, str] = {}
    for candidate in candidates:
        manifest = candidate.manifest_json
        digest = manifest.get("sha256") if type(manifest) is dict else None
        if type(digest) is not str or digest not in needed or digest in asset_ids:
            continue
        try:
            materialized = materialize_model_asset(candidate)
        except WorkflowBindingError:
            continue
        if materialized.identity.get("sha256") == digest:
            asset_ids[digest] = candidate.id
    if set(asset_ids) != needed:
        _conflict()
    settings = tuple(
        {
            "asset_id": asset_ids[cast(str, item["sha256"])],
            "model_strength": item["model_strength"],
            "clip_strength": item["clip_strength"],
            "enabled": True,
        }
        for item in cast(list[dict[str, object]], stack)
    )
    return PromptSourceResourceOverrides(workflow_revision_id, settings)
