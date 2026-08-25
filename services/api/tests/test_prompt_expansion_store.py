from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import create_engine, event, func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from local_lm import prompt_expansion_store as store_module
from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import Base
from local_lm.domain import utcnow
from local_lm.models import (
    ArtifactLibraryEntry,
    Chat,
    Message,
    PromptExpansionBatch,
    PromptExpansionItem,
    PromptTemplateDefinition,
    PromptTemplateRevision,
    Run,
    WorkPlan,
    WorkStep,
)
from local_lm.prompt_expansion import (
    ExpandedItem,
    ExpansionPlan,
    ExpansionRequest,
    SlotEvidence,
    complete_prompt_expansion_with_model_values,
    expand_prompt_template,
    expansion_plan_digest,
    expansion_plan_payload,
    expansion_plan_payload_digest,
    parse_expansion_request,
)
from local_lm.prompt_expansion_store import (
    PROMPT_EXPANSION_STORE_CONFLICT,
    PROMPT_EXPANSION_STORE_INVALID,
    PromptExpansionStoreConflict,
    PromptExpansionStoreError,
    StoredExpansion,
    create_or_replay_expansion,
    read_expansion,
    replay_expansion_request,
    reroll_expansion_item,
    update_expansion_item,
)
from local_lm.prompt_model_values import (
    PromptModelValues,
    parse_prompt_model_values,
    prompt_model_slot_contract,
    prompt_model_values_sha256,
)
from local_lm.prompt_templates import (
    PromptTemplateContract,
    parse_prompt_template_contract,
    prompt_template_contract_payload,
    prompt_template_contract_sha256,
    render_prompt_template,
)

SessionFactory = sessionmaker[Session]


@pytest.fixture()
def sessions(tmp_path: Path) -> Iterator[SessionFactory]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'expansions.sqlite3'}")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection: object, _record: object) -> None:
        connection.execute("PRAGMA foreign_keys=ON")  # type: ignore[attr-defined]

    Base.metadata.create_all(engine)
    try:
        yield sessionmaker(engine, expire_on_commit=False)
    finally:
        engine.dispose()


def _seed(
    session: Session, *, chat_title: str = "Chat"
) -> tuple[Chat, ExpansionRequest, ExpansionPlan]:
    chat = Chat(title=chat_title)
    definition = PromptTemplateDefinition(name=f"Template {chat_title}", description="")
    session.add_all((chat, definition))
    session.flush()
    contract = parse_prompt_template_contract(
        {
            "schema_version": 1,
            "operation": "text_to_image",
            "body": "{{style}} portrait of {{subject}} in {{mood}} light.",
            "slots": [
                {
                    "name": "style",
                    "mode": "fixed",
                    "variation_scope": "batch",
                    "fixed_value": "oil",
                },
                {"name": "subject", "mode": "input", "variation_scope": "batch"},
                {
                    "name": "mood",
                    "mode": "choice",
                    "variation_scope": "item",
                    "choices": ["calm", "bright", "stormy"],
                },
            ],
            "resource_policy": {"mode": "inherited"},
        }
    )
    digest = prompt_template_contract_sha256(contract)
    revision = PromptTemplateRevision(
        prompt_template_id=definition.id,
        version=1,
        schema_version=1,
        contract_json=prompt_template_contract_payload(contract),
        contract_sha256=digest,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    request = parse_expansion_request(
        {
            "definition_id": definition.id,
            "revision_id": revision.id,
            "contract_sha256": digest,
            "item_count": 2,
            "selection_seed": 412,
            "inputs": {"subject": "a fox"},
        }
    )
    return chat, request, expand_prompt_template(contract, request)


def _create(
    session: Session, *, key: str = "request-1"
) -> tuple[Chat, ExpansionRequest, ExpansionPlan, StoredExpansion]:
    chat, request, plan = _seed(session)
    stored = create_or_replay_expansion(
        session, chat.id, key, request, plan, {"version": 1, "kind": "deterministic"}
    )
    return chat, request, plan, stored


def _seed_model(
    session: Session,
) -> tuple[
    Chat,
    ExpansionRequest,
    ExpansionPlan,
    dict[str, object],
    PromptTemplateContract,
    PromptModelValues,
]:
    chat = Chat(title="Model chat")
    definition = PromptTemplateDefinition(name="Model template", description="")
    session.add_all((chat, definition))
    session.flush()
    contract = parse_prompt_template_contract(
        {
            "schema_version": 1,
            "operation": "text_to_image",
            "body": (
                "{{style}} portrait of {{subject}} in {{mood}} light, "
                "{{palette}}, {{medium}}, {{detail}}."
            ),
            "slots": [
                {
                    "name": "style",
                    "mode": "fixed",
                    "variation_scope": "batch",
                    "fixed_value": "oil",
                },
                {"name": "subject", "mode": "input", "variation_scope": "batch"},
                {
                    "name": "mood",
                    "mode": "choice",
                    "variation_scope": "item",
                    "choices": ["calm", "bright", "stormy"],
                },
                {
                    "name": "palette",
                    "mode": "choice",
                    "variation_scope": "batch",
                    "choices": ["warm", "cool"],
                },
                {
                    "name": "medium",
                    "mode": "model",
                    "variation_scope": "batch",
                    "guidance": "one medium",
                },
                {
                    "name": "detail",
                    "mode": "model",
                    "variation_scope": "item",
                    "guidance": "one visible detail",
                },
            ],
            "resource_policy": {"mode": "inherited"},
        }
    )
    digest = prompt_template_contract_sha256(contract)
    revision = PromptTemplateRevision(
        prompt_template_id=definition.id,
        version=1,
        schema_version=1,
        contract_json=prompt_template_contract_payload(contract),
        contract_sha256=digest,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    request = parse_expansion_request(
        {
            "definition_id": definition.id,
            "revision_id": revision.id,
            "contract_sha256": digest,
            "item_count": 2,
            "selection_seed": 728,
            "inputs": {"subject": "a fox"},
        }
    )
    pending = expand_prompt_template(contract, request)
    model_contract = prompt_model_slot_contract(contract, item_count=2)
    values = parse_prompt_model_values(
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
    completed = complete_prompt_expansion_with_model_values(contract, pending, values)
    snapshot: dict[str, object] = {
        "version": 1,
        "kind": "model",
        "adapter_id": "local-chat",
        "model_id": "test-model",
        "values_sha256": prompt_model_values_sha256(values, contract=model_contract),
    }
    return chat, request, completed, snapshot, contract, values


def test_create_read_and_zero_media_side_effects(sessions: SessionFactory) -> None:
    with sessions() as session:
        chat, _request, _plan, created = _create(session)
        session.commit()
        reread = read_expansion(session, chat.id, created.batch.id)
        assert created.replayed is False
        assert reread.replayed is True
        assert [item.ordinal for item in reread.items] == [1, 2]
        assert all(item.reviewed_prompt == item.original_rendered_prompt for item in reread.items)
        assert all(item.review_version == 1 and item.reroll_count == 0 for item in reread.items)
        assert session.scalar(select(func.count()).select_from(ArtifactLibraryEntry)) == 0


def test_same_chat_exact_replay_and_cross_chat_key_scope(sessions: SessionFactory) -> None:
    with sessions() as session:
        first_chat, request, plan, first = _create(session, key="same-key")
        replay = create_or_replay_expansion(
            session,
            first_chat.id,
            "same-key",
            request,
            plan,
            {"version": 1, "kind": "deterministic"},
        )
        other_chat, other_request, other_plan = _seed(session, chat_title="Other")
        other = create_or_replay_expansion(
            session,
            other_chat.id,
            "same-key",
            other_request,
            other_plan,
            {"version": 1, "kind": "deterministic"},
        )
        assert replay.replayed is True and replay.batch.id == first.batch.id
        assert other.batch.id != first.batch.id


def test_request_preflight_replays_exactly_and_refuses_changed_authority(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, request, _plan, created = _create(session, key="preflight-key")
        session.commit()
        replay = replay_expansion_request(session, chat.id, "preflight-key", request)
        assert replay is not None
        assert replay.replayed is True
        assert replay.batch.id == created.batch.id
        changed = replace(request, selection_seed=request.selection_seed + 1)
        with pytest.raises(PromptExpansionStoreConflict) as caught:
            replay_expansion_request(session, chat.id, "preflight-key", changed)
        assert str(caught.value) == PROMPT_EXPANSION_STORE_CONFLICT


def test_same_key_with_changed_exact_request_or_snapshot_refuses(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, request, plan, _stored = _create(session)
        changed = replace(request, selection_seed=request.selection_seed + 1)
        with pytest.raises(PromptExpansionStoreError, match=PROMPT_EXPANSION_STORE_INVALID):
            create_or_replay_expansion(
                session,
                chat.id,
                "request-1",
                changed,
                plan,
                {"version": 1, "kind": "deterministic"},
            )
        with pytest.raises(PromptExpansionStoreError, match=PROMPT_EXPANSION_STORE_INVALID):
            create_or_replay_expansion(
                session,
                chat.id,
                "request-1",
                request,
                plan,
                {
                    "version": 1,
                    "kind": "model",
                    "adapter_id": "local",
                    "model_id": "different",
                    "values_sha256": "a" * 64,
                },
            )


@pytest.mark.parametrize(
    "snapshot",
    [
        {},
        {"version": True, "kind": "deterministic"},
        {"version": 1, "kind": "deterministic", "model_id": "extra"},
        {
            "version": 1,
            "kind": "model",
            "adapter_id": "local",
            "model_id": "model",
            "values_sha256": "A" * 64,
        },
    ],
)
def test_model_snapshot_is_a_strict_closed_union(
    sessions: SessionFactory, snapshot: object
) -> None:
    with sessions() as session:
        chat, request, plan = _seed(session)
        with pytest.raises(PromptExpansionStoreError) as caught:
            create_or_replay_expansion(session, chat.id, "strict-snapshot", request, plan, snapshot)
        assert str(caught.value) == PROMPT_EXPANSION_STORE_INVALID


def test_model_completed_batch_creates_reads_and_replays_exactly(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, request, completed, snapshot, _contract, _values = _seed_model(session)
        created = create_or_replay_expansion(
            session, chat.id, "model-key", request, completed, snapshot
        )
        session.commit()
    with sessions() as session:
        reread = read_expansion(session, chat.id, created.batch.id)
        replay = create_or_replay_expansion(
            session, chat.id, "model-key", request, completed, snapshot
        )
        assert reread.batch.id == created.batch.id
        assert replay.replayed is True
        assert replay.batch.id == created.batch.id
        assert '"kind":"model"' in replay.batch.model_snapshot_json


def test_model_completed_batch_allows_reviewed_prompt_edits(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, request, completed, snapshot, _contract, _values = _seed_model(session)
        stored = create_or_replay_expansion(
            session, chat.id, "model-review-edit", request, completed, snapshot
        )
        machine_plan_sha256 = stored.batch.plan_sha256
        changed = update_expansion_item(
            session,
            chat.id,
            stored.batch.id,
            stored.items[0].id,
            expected_item_version=1,
            expected_plan_version=1,
            reviewed_prompt="A human-reviewed model prompt.",
            selected=True,
        )
        assert changed.items[0].reviewed_prompt == "A human-reviewed model prompt."
        assert changed.items[0].current_evidence_json == stored.items[0].current_evidence_json
        assert changed.batch.plan_sha256 == machine_plan_sha256
        assert changed.batch.plan_version == 2


@pytest.mark.parametrize("drift", ["snapshot_digest", "completed_evidence"])
def test_model_completed_create_refuses_values_snapshot_or_evidence_drift(
    sessions: SessionFactory, drift: str
) -> None:
    with sessions() as session:
        chat, request, completed, snapshot, contract, _values = _seed_model(session)
        if drift == "snapshot_digest":
            snapshot = {**snapshot, "values_sha256": "a" * 64}
        else:
            model_contract = prompt_model_slot_contract(contract, item_count=2)
            other_values = parse_prompt_model_values(
                {
                    "version": 1,
                    "batch_values": {"medium": "charcoal"},
                    "items": [
                        {"ordinal": 1, "values": {"detail": "silver leaves"}},
                        {"ordinal": 2, "values": {"detail": "amber rain"}},
                    ],
                },
                contract=model_contract,
            )
            pending = expand_prompt_template(contract, request)
            completed = complete_prompt_expansion_with_model_values(contract, pending, other_values)
        with pytest.raises(PromptExpansionStoreError) as caught:
            create_or_replay_expansion(
                session, chat.id, "model-drift", request, completed, snapshot
            )
        assert str(caught.value) == PROMPT_EXPANSION_STORE_INVALID
        assert caught.value.__cause__ is None


@pytest.mark.parametrize("drift", ["snapshot", "evidence"])
def test_model_completed_read_refuses_stored_snapshot_or_evidence_drift(
    sessions: SessionFactory, drift: str
) -> None:
    with sessions() as session:
        chat, request, completed, snapshot, _contract, _values = _seed_model(session)
        stored = create_or_replay_expansion(
            session, chat.id, "model-read-drift", request, completed, snapshot
        )
        session.commit()
        if drift == "snapshot":
            session.execute(text("DROP TRIGGER prompt_expansion_batch_update_guard"))
            corrupted = {**snapshot, "values_sha256": "b" * 64}
            session.execute(
                text("UPDATE prompt_expansion_batches SET model_snapshot_json = :snapshot"),
                {"snapshot": json.dumps(corrupted, sort_keys=True, separators=(",", ":"))},
            )
        else:
            session.execute(text("DROP TRIGGER prompt_expansion_item_update_guard"))
            session.execute(
                text(
                    "UPDATE prompt_expansion_items "
                    "SET original_evidence_json = replace("
                    "original_evidence_json, 'tempera', 'charcoal') "
                    "WHERE ordinal = 1"
                )
            )
        session.commit()
    with sessions() as session, pytest.raises(PromptExpansionStoreError) as caught:
        read_expansion(session, chat.id, stored.batch.id)
    assert str(caught.value) == PROMPT_EXPANSION_STORE_INVALID
    assert caught.value.__cause__ is None


def test_model_completed_read_binds_coherent_current_values_to_snapshot(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, request, completed, snapshot, contract, _values = _seed_model(session)
        stored = create_or_replay_expansion(
            session, chat.id, "model-current-drift", request, completed, snapshot
        )
        model_contract = prompt_model_slot_contract(contract, item_count=2)
        changed_values = parse_prompt_model_values(
            {
                "version": 1,
                "batch_values": {"medium": "charcoal"},
                "items": [
                    {"ordinal": 1, "values": {"detail": "silver leaves"}},
                    {"ordinal": 2, "values": {"detail": "amber rain"}},
                ],
            },
            contract=model_contract,
        )
        changed = complete_prompt_expansion_with_model_values(
            contract, expand_prompt_template(contract, request), changed_values
        )
        changed_payload = expansion_plan_payload(changed)
        session.commit()
        session.execute(text("DROP TRIGGER prompt_expansion_item_update_guard"))
        session.execute(text("DROP TRIGGER prompt_expansion_batch_update_guard"))
        for item, expanded, payload_item in zip(
            stored.items, changed.items, changed_payload["items"], strict=True
        ):
            assert expanded.rendered_prompt is not None
            assert expanded.rendered_sha256 is not None
            session.execute(
                text(
                    "UPDATE prompt_expansion_items "
                    "SET current_evidence_json = :evidence, reviewed_prompt = :prompt, "
                    "reviewed_sha256 = :digest, review_version = 2, reroll_count = 1 "
                    "WHERE id = :item_id"
                ),
                {
                    "evidence": json.dumps(
                        payload_item["evidence"],
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "prompt": expanded.rendered_prompt,
                    "digest": expanded.rendered_sha256,
                    "item_id": item.id,
                },
            )
        session.execute(
            text("UPDATE prompt_expansion_batches SET plan_sha256 = :digest, plan_version = 3"),
            {"digest": expansion_plan_digest(changed)},
        )
        session.commit()
    with sessions() as session, pytest.raises(PromptExpansionStoreError) as caught:
        read_expansion(session, chat.id, stored.batch.id)
    assert str(caught.value) == PROMPT_EXPANSION_STORE_INVALID
    assert caught.value.__cause__ is None


def test_unique_race_reloads_exact_winner_inside_savepoint_without_outer_rollback(
    sessions: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    with sessions() as session:
        chat, request, plan, winner = _create(session)
        marker = Chat(title="outer transaction survives")
        session.add(marker)
        session.flush()
        real_existing = store_module._existing
        calls = 0

        def miss_once(active: Session, chat_id: str, key: str) -> PromptExpansionBatch | None:
            nonlocal calls
            calls += 1
            return None if calls == 1 else real_existing(active, chat_id, key)

        monkeypatch.setattr(store_module, "_existing", miss_once)
        replay = create_or_replay_expansion(
            session,
            chat.id,
            "request-1",
            request,
            plan,
            {"version": 1, "kind": "deterministic"},
        )
        session.commit()
        assert replay.replayed is True and replay.batch.id == winner.batch.id
    with sessions() as session:
        assert session.get(Chat, marker.id) is not None
        assert session.scalar(select(func.count()).select_from(PromptExpansionBatch)) == 1


def test_review_edit_is_item_and_batch_cas_and_stale_writes_are_atomic(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, _request, _plan, stored = _create(session)
        item = stored.items[0]
        machine_plan_sha256 = stored.batch.plan_sha256
        changed = update_expansion_item(
            session,
            chat.id,
            stored.batch.id,
            item.id,
            expected_item_version=1,
            expected_plan_version=1,
            reviewed_prompt="A deliberately reviewed prompt.",
            selected=False,
        )
        assert changed.items[0].review_version == 2
        assert changed.items[0].selected is False
        assert changed.batch.plan_sha256 == machine_plan_sha256
        assert changed.batch.plan_version == 2
        with pytest.raises(PromptExpansionStoreConflict) as caught:
            update_expansion_item(
                session,
                chat.id,
                stored.batch.id,
                item.id,
                expected_item_version=1,
                expected_plan_version=1,
                reviewed_prompt="stale",
                selected=True,
            )
        assert str(caught.value) == PROMPT_EXPANSION_STORE_CONFLICT
        session.commit()
    with sessions() as session:
        reread = read_expansion(session, chat.id, stored.batch.id)
        assert reread.items[0].reviewed_prompt == "A deliberately reviewed prompt."
        assert reread.batch.plan_version == 2


def test_arbitrary_edit_then_reroll_then_edit_keeps_review_out_of_machine_receipt(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, _request, plan, stored = _create(session)
        item = stored.items[0]
        original_machine_sha256 = stored.batch.plan_sha256
        first_edit = update_expansion_item(
            session,
            chat.id,
            stored.batch.id,
            item.id,
            expected_item_version=1,
            expected_plan_version=1,
            reviewed_prompt="Human text unrelated to the template.",
            selected=True,
        )
        assert first_edit.batch.plan_sha256 == original_machine_sha256

        rerolled = reroll_expansion_item(
            session,
            chat.id,
            stored.batch.id,
            item.id,
            expected_item_version=2,
            expected_plan_version=2,
            replacement_plan=_reroll_plan(plan, 1),
            model_snapshot={"version": 1, "kind": "deterministic"},
        )
        assert rerolled.items[0].reviewed_prompt != "Human text unrelated to the template."
        rerolled_machine_sha256 = rerolled.batch.plan_sha256
        assert rerolled_machine_sha256 != original_machine_sha256

        second_edit = update_expansion_item(
            session,
            chat.id,
            stored.batch.id,
            item.id,
            expected_item_version=3,
            expected_plan_version=3,
            reviewed_prompt="Another independent human edit.",
            selected=False,
        )
        assert second_edit.batch.plan_sha256 == rerolled_machine_sha256
        assert second_edit.items[0].reviewed_prompt == "Another independent human edit."
        session.commit()

    with sessions() as session:
        reread = read_expansion(session, chat.id, stored.batch.id)
        assert reread.items[0].reviewed_prompt == "Another independent human edit."
        assert reread.items[0].selected is False


def _reroll_plan(plan: ExpansionPlan, ordinal: int) -> ExpansionPlan:
    original = plan.items[ordinal - 1]
    evidence = list(original.evidence)
    mood_position = next(index for index, one in enumerate(evidence) if one.name == "mood")
    mood = evidence[mood_position]
    next_index = (mood.choice_index + 1) % 3  # type: ignore[operator]
    choices = ("calm", "bright", "stormy")
    evidence[mood_position] = SlotEvidence(
        name=mood.name,
        mode=mood.mode,
        variation_scope=mood.variation_scope,
        source=mood.source,
        value=choices[next_index],
        choice_index=next_index,
    )
    values = {one.name: one.value for one in evidence if one.name != "style"}
    contract = parse_prompt_template_contract(
        {
            "schema_version": 1,
            "operation": "text_to_image",
            "body": "{{style}} portrait of {{subject}} in {{mood}} light.",
            "slots": [
                {
                    "name": "style",
                    "mode": "fixed",
                    "variation_scope": "batch",
                    "fixed_value": "oil",
                },
                {"name": "subject", "mode": "input", "variation_scope": "batch"},
                {
                    "name": "mood",
                    "mode": "choice",
                    "variation_scope": "item",
                    "choices": list(choices),
                },
            ],
            "resource_policy": {"mode": "inherited"},
        }
    )
    prompt = render_prompt_template(contract, values)
    digest = hashlib.sha256(
        "\x00".join(("prompt-expansion-rendered-v1", prompt)).encode()
    ).hexdigest()
    replacement = ExpandedItem(
        ordinal=ordinal,
        evidence=tuple(evidence),
        rendered_prompt=prompt,
        rendered_sha256=digest,
    )
    items = list(plan.items)
    items[ordinal - 1] = replacement
    return replace(plan, items=tuple(items))


def _reroll_model_plan(
    plan: ExpansionPlan, contract: PromptTemplateContract, ordinal: int
) -> ExpansionPlan:
    original = plan.items[ordinal - 1]
    evidence = list(original.evidence)
    position = next(index for index, one in enumerate(evidence) if one.name == "mood")
    mood = evidence[position]
    choices = ("calm", "bright", "stormy")
    next_index = (mood.choice_index + 1) % len(choices)  # type: ignore[operator]
    evidence[position] = replace(mood, value=choices[next_index], choice_index=next_index)
    prompt = render_prompt_template(
        contract,
        {one.name: one.value for one in evidence if one.mode.value != "fixed"},
    )
    digest = hashlib.sha256(
        "\x00".join(("prompt-expansion-rendered-v1", prompt)).encode()
    ).hexdigest()
    items = list(plan.items)
    items[ordinal - 1] = ExpandedItem(
        ordinal=ordinal,
        evidence=tuple(evidence),
        rendered_prompt=prompt,
        rendered_sha256=digest,
    )
    return replace(plan, items=tuple(items))


def _forge_model_batch_scope(
    plan: ExpansionPlan,
    contract: PromptTemplateContract,
    ordinal: int,
    slot_name: str,
) -> ExpansionPlan:
    original = plan.items[ordinal - 1]
    evidence = list(original.evidence)
    position = next(index for index, one in enumerate(evidence) if one.name == slot_name)
    entry = evidence[position]
    if slot_name == "palette":
        assert entry.choice_index is not None
        choices = ("warm", "cool")
        next_index = (entry.choice_index + 1) % len(choices)
        evidence[position] = replace(entry, value=choices[next_index], choice_index=next_index)
    else:
        assert slot_name == "medium"
        evidence[position] = replace(entry, value="charcoal")
    prompt = render_prompt_template(
        contract,
        {one.name: one.value for one in evidence if one.mode.value != "fixed"},
    )
    digest = hashlib.sha256(
        "\x00".join(("prompt-expansion-rendered-v1", prompt)).encode()
    ).hexdigest()
    items = list(plan.items)
    items[ordinal - 1] = ExpandedItem(
        ordinal=ordinal,
        evidence=tuple(evidence),
        rendered_prompt=prompt,
        rendered_sha256=digest,
    )
    return replace(plan, items=tuple(items))


def test_reroll_replaces_only_target_and_advances_both_versions(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, _request, plan, stored = _create(session)
        first, second = stored.items
        original_first_prompt = first.original_rendered_prompt
        previous_first_prompt = first.reviewed_prompt
        previous_second_prompt = second.reviewed_prompt
        rerolled = reroll_expansion_item(
            session,
            chat.id,
            stored.batch.id,
            first.id,
            expected_item_version=1,
            expected_plan_version=1,
            replacement_plan=_reroll_plan(plan, 1),
            model_snapshot={"version": 1, "kind": "deterministic"},
        )
        assert rerolled.items[0].reroll_count == 1
        assert rerolled.items[0].review_version == 2
        assert rerolled.items[0].original_rendered_prompt == original_first_prompt
        assert rerolled.items[0].reviewed_prompt != previous_first_prompt
        assert rerolled.items[1].reviewed_prompt == previous_second_prompt
        assert rerolled.batch.plan_version == 2


def test_reroll_refuses_a_coherent_self_declared_body_not_owned_by_the_revision(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, _request, plan, stored = _create(session)
        replacement = _reroll_plan(plan, 1)
        body = "Alternate {{style}} portrait of {{subject}} in {{mood}} light."
        items: list[ExpandedItem] = []
        for item in replacement.items:
            rendered = body
            for evidence in item.evidence:
                assert evidence.value is not None
                rendered = rendered.replace("{{" + evidence.name + "}}", evidence.value)
            digest = hashlib.sha256(
                "\x00".join(("prompt-expansion-rendered-v1", rendered)).encode()
            ).hexdigest()
            items.append(replace(item, rendered_prompt=rendered, rendered_sha256=digest))
        forged = replace(replacement, template_body=body, items=tuple(items))
        assert len(expansion_plan_digest(forged)) == 64

        with pytest.raises(PromptExpansionStoreError):
            reroll_expansion_item(
                session,
                chat.id,
                stored.batch.id,
                stored.items[0].id,
                expected_item_version=1,
                expected_plan_version=1,
                replacement_plan=forged,
                model_snapshot={"version": 1, "kind": "deterministic"},
            )


def test_read_refuses_coherent_current_choice_not_declared_by_the_revision(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, _request, plan, stored = _create(session)
        payload = expansion_plan_payload(plan)
        target = payload["items"][0]
        mood = next(entry for entry in target["evidence"] if entry["name"] == "mood")
        mood["value"] = "forged-choice"
        rendered = payload["template_body"]
        for evidence in target["evidence"]:
            rendered = rendered.replace("{{" + evidence["name"] + "}}", evidence["value"])
        digest = hashlib.sha256(
            "\x00".join(("prompt-expansion-rendered-v1", rendered)).encode()
        ).hexdigest()
        target["rendered_prompt"] = rendered
        target["rendered_sha256"] = digest
        forged_plan_sha256 = expansion_plan_payload_digest(payload)

        session.execute(
            text(
                "UPDATE prompt_expansion_items SET current_evidence_json = :evidence, "
                "reviewed_prompt = :prompt, reviewed_sha256 = :digest, "
                "review_version = 2, reroll_count = 1 WHERE id = :item_id"
            ),
            {
                "evidence": json.dumps(
                    target["evidence"],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "prompt": rendered,
                "digest": digest,
                "item_id": stored.items[0].id,
            },
        )
        session.execute(
            text(
                "UPDATE prompt_expansion_batches SET plan_sha256 = :digest, "
                "plan_version = 2 WHERE id = :batch_id"
            ),
            {"digest": forged_plan_sha256, "batch_id": stored.batch.id},
        )
        session.commit()

    with sessions() as session, pytest.raises(PromptExpansionStoreError):
        read_expansion(session, chat.id, stored.batch.id)


def test_model_completed_batch_supports_snapshot_bound_reroll(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, request, completed, snapshot, contract, _values = _seed_model(session)
        stored = create_or_replay_expansion(
            session, chat.id, "model-reroll", request, completed, snapshot
        )
        rerolled = reroll_expansion_item(
            session,
            chat.id,
            stored.batch.id,
            stored.items[0].id,
            expected_item_version=1,
            expected_plan_version=1,
            replacement_plan=_reroll_model_plan(completed, contract, 1),
            model_snapshot=snapshot,
        )
        assert rerolled.items[0].reroll_count == 1
        assert rerolled.items[0].review_version == 2
        assert rerolled.batch.plan_version == 2


@pytest.mark.parametrize("slot_name", ["palette", "medium"])
def test_model_reroll_refuses_one_item_batch_scope_changes(
    sessions: SessionFactory, slot_name: str
) -> None:
    with sessions() as session:
        chat, request, completed, snapshot, contract, _values = _seed_model(session)
        stored = create_or_replay_expansion(
            session, chat.id, "model-batch-scope-reroll", request, completed, snapshot
        )
        with pytest.raises(PromptExpansionStoreError) as caught:
            reroll_expansion_item(
                session,
                chat.id,
                stored.batch.id,
                stored.items[0].id,
                expected_item_version=1,
                expected_plan_version=1,
                replacement_plan=_forge_model_batch_scope(completed, contract, 1, slot_name),
                model_snapshot=snapshot,
            )
        assert str(caught.value) == PROMPT_EXPANSION_STORE_INVALID
        assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE prompt_expansion_items SET selected = 0, review_version = review_version + 1",
        "UPDATE prompt_expansion_batches SET plan_version = plan_version + 1",
    ],
)
def test_readback_refuses_item_only_and_batch_only_cas_partials(
    sessions: SessionFactory, statement: str
) -> None:
    with sessions() as session:
        chat, _request, _plan, stored = _create(session)
        session.commit()
        session.execute(text(statement))
        session.commit()
    with sessions() as session, pytest.raises(PromptExpansionStoreError):
        read_expansion(session, chat.id, stored.batch.id)


def test_item_reads_are_sql_bounded_before_corrupt_rows_are_materialized(
    sessions: SessionFactory,
) -> None:
    def has_limit_parameter(parameters: object) -> bool:
        if isinstance(parameters, dict):
            return 3 in parameters.values()
        if isinstance(parameters, (list, tuple)):
            return 3 in parameters
        return False

    with sessions() as session:
        chat, _request, _plan, stored = _create(session)
        session.commit()
        engine = session.get_bind()
        captured: list[tuple[str, object]] = []

        @event.listens_for(engine, "before_cursor_execute")
        def capture_item_reads(
            _connection: object,
            _cursor: object,
            statement: str,
            parameters: object,
            _context: object,
            _executemany: object,
        ) -> None:
            if "FROM prompt_expansion_items" in statement:
                captured.append((statement, parameters))

        try:
            read_expansion(session, chat.id, stored.batch.id)
        finally:
            event.remove(engine, "before_cursor_execute", capture_item_reads)

        assert len(captured) == 2
        assert all(" LIMIT " in statement.upper() for statement, _parameters in captured)
        assert all(has_limit_parameter(parameters) for _statement, parameters in captured)

        session.execute(text("DROP TRIGGER prompt_expansion_item_insert_guard"))
        session.execute(
            text(
                "WITH RECURSIVE overflow(ordinal) AS ("
                "SELECT 3 UNION ALL "
                "SELECT ordinal + 1 FROM overflow WHERE ordinal < 1002"
                ") "
                "INSERT INTO prompt_expansion_items ("
                "id, batch_id, ordinal, original_evidence_json, current_evidence_json, "
                "original_rendered_prompt, original_rendered_sha256, reviewed_prompt, "
                "reviewed_sha256, selected, review_version, reroll_count, created_at, updated_at"
                ") "
                "SELECT 'ptitem_overflow_' || overflow.ordinal, source.batch_id, "
                "overflow.ordinal, source.original_evidence_json, "
                "source.current_evidence_json, source.original_rendered_prompt, "
                "source.original_rendered_sha256, source.reviewed_prompt, "
                "source.reviewed_sha256, source.selected, source.review_version, "
                "source.reroll_count, source.created_at, source.updated_at "
                "FROM prompt_expansion_items AS source CROSS JOIN overflow "
                "WHERE source.batch_id = :batch_id AND source.ordinal = 1"
            ),
            {"batch_id": stored.batch.id},
        )
        session.commit()

    with sessions() as session:
        engine = session.get_bind()
        overflow_reads: list[tuple[str, object]] = []

        @event.listens_for(engine, "before_cursor_execute")
        def capture_overflow_read(
            _connection: object,
            _cursor: object,
            statement: str,
            parameters: object,
            _context: object,
            _executemany: object,
        ) -> None:
            if "FROM prompt_expansion_items" in statement:
                overflow_reads.append((statement, parameters))

        try:
            with pytest.raises(PromptExpansionStoreError) as caught:
                read_expansion(session, chat.id, stored.batch.id)
        finally:
            event.remove(engine, "before_cursor_execute", capture_overflow_read)

        assert str(caught.value) == PROMPT_EXPANSION_STORE_INVALID
        assert caught.value.__cause__ is None
        assert len(overflow_reads) == 1
        statement, parameters = overflow_reads[0]
        assert " LIMIT " in statement.upper()
        assert has_limit_parameter(parameters)


def test_selected_requires_exact_sqlite_zero_or_one(sessions: SessionFactory) -> None:
    with sessions() as session:
        _chat, _request, _plan, _stored = _create(session)
        session.commit()
        with pytest.raises(IntegrityError, match="item update is invalid"):
            session.execute(
                text(
                    "UPDATE prompt_expansion_items SET selected = 2, "
                    "review_version = review_version + 1"
                )
            )


def test_readback_refuses_raw_selected_two_after_guard_bypass(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, _request, _plan, stored = _create(session)
        session.commit()
        session.execute(text("DROP TRIGGER prompt_expansion_item_update_guard"))
        session.execute(
            text(
                "UPDATE prompt_expansion_items "
                "SET selected = 2, review_version = review_version + 1 "
                "WHERE ordinal = 1"
            )
        )
        session.execute(text("UPDATE prompt_expansion_batches SET plan_version = plan_version + 1"))
        session.commit()
    with sessions() as session, pytest.raises(PromptExpansionStoreError) as caught:
        read_expansion(session, chat.id, stored.batch.id)
    assert str(caught.value) == PROMPT_EXPANSION_STORE_INVALID
    assert caught.value.__cause__ is None


def test_readback_refuses_reroll_count_above_prior_review_count(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, _request, _plan, stored = _create(session)
        session.commit()
        session.execute(text("DROP TRIGGER prompt_expansion_item_update_guard"))
        session.execute(
            text(
                "UPDATE prompt_expansion_items SET reroll_count = review_version WHERE ordinal = 1"
            )
        )
        session.commit()
    with sessions() as session, pytest.raises(PromptExpansionStoreError):
        read_expansion(session, chat.id, stored.batch.id)


def test_queued_plan_has_exactly_one_transition_version(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, _request, _plan, stored = _create(session)
        changed = update_expansion_item(
            session,
            chat.id,
            stored.batch.id,
            stored.items[0].id,
            expected_item_version=1,
            expected_plan_version=1,
            reviewed_prompt="reviewed before queue",
            selected=True,
        )
        assert changed.batch.plan_version == 2
        session.commit()
        session.execute(
            text(
                "UPDATE prompt_expansion_batches SET state = 'queued', "
                "plan_version = plan_version + 1"
            )
        )
        session.commit()
    with sessions() as session:
        reread = read_expansion(session, chat.id, stored.batch.id)
        assert reread.batch.state == "queued"
        assert reread.batch.plan_version == 3


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE prompt_expansion_items SET reviewed_prompt = 'corrupt', "
        "review_version = review_version + 1",
        "UPDATE prompt_expansion_items SET current_evidence_json = '[]', "
        "review_version = review_version + 1, reroll_count = reroll_count + 1",
        "UPDATE prompt_expansion_batches SET plan_sha256 = 'a' || substr(plan_sha256, 2), "
        "plan_version = plan_version + 1",
    ],
)
def test_hostile_direct_sql_corruption_is_refused_on_read(
    sessions: SessionFactory, statement: str
) -> None:
    with sessions() as session:
        chat, _request, _plan, stored = _create(session)
        session.commit()
        session.execute(text(statement))
        session.commit()
    with sessions() as session, pytest.raises(PromptExpansionStoreError):
        read_expansion(session, chat.id, stored.batch.id)


def test_readback_refuses_noncanonical_duplicate_key_request_after_guard_bypass(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, _request, _plan, stored = _create(session)
        session.commit()
        session.execute(text("DROP TRIGGER prompt_expansion_batch_update_guard"))
        session.execute(
            text("UPDATE prompt_expansion_batches SET request_json = :request"),
            {"request": '{"definition_id":"x","definition_id":"y"}'},
        )
        session.commit()
    with sessions() as session, pytest.raises(PromptExpansionStoreError):
        read_expansion(session, chat.id, stored.batch.id)


def test_chat_delete_cascades_batches_and_items(sessions: SessionFactory) -> None:
    with sessions() as session:
        chat, _request, _plan, _stored = _create(session)
        session.commit()
        session.delete(chat)
        session.commit()
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PromptExpansionBatch)) == 0
        assert session.scalar(select(func.count()).select_from(PromptExpansionItem)) == 0


def test_queued_execution_links_follow_deleted_run_step_and_plan(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        chat, _request, _plan, stored = _create(session)
        user = Message(chat_id=chat.id, role="user", status="complete")
        session.add(user)
        session.flush()
        assistant = Message(
            chat_id=chat.id,
            parent_id=user.id,
            role="assistant",
            status="complete",
        )
        work_plan = WorkPlan(chat_id=chat.id, transcript_sequence=1)
        session.add_all((assistant, work_plan))
        session.flush()
        step = WorkStep(
            plan_id=work_plan.id,
            ordinal=1,
            operation="image",
            status="complete",
            prompt=stored.items[0].reviewed_prompt,
        )
        session.add(step)
        session.flush()
        run = Run(
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            work_plan_id=work_plan.id,
            work_step_id=step.id,
            operation="image",
            status="complete",
            standalone_prompt=stored.items[0].reviewed_prompt,
        )
        session.add(run)
        session.flush()
        step.run_id = run.id
        session.flush()
        session.execute(
            update(PromptExpansionBatch)
            .where(PromptExpansionBatch.id == stored.batch.id)
            .values(
                state="queued",
                plan_version=stored.batch.plan_version + 1,
                queue_idempotency_key="queue-delete-links",
            )
        )
        session.execute(
            update(PromptExpansionBatch)
            .where(PromptExpansionBatch.id == stored.batch.id)
            .values(work_plan_id=work_plan.id, queued_at=utcnow())
        )
        session.execute(
            update(PromptExpansionItem)
            .where(PromptExpansionItem.id == stored.items[0].id)
            .values(work_step_id=step.id, run_id=run.id, media_seed=7)
        )
        session.commit()

        session.delete(run)
        session.flush()
        session.delete(step)
        session.flush()
        session.delete(work_plan)
        session.commit()

        session.expire_all()
        item = session.get(PromptExpansionItem, stored.items[0].id)
        batch = session.get(PromptExpansionBatch, stored.batch.id)
        assert item is not None
        assert item.run_id is None
        assert item.work_step_id is None
        assert item.media_seed == 7
        assert batch is not None
        assert batch.work_plan_id is None


def test_database_guards_reject_origin_changes_version_skips_and_queued_edits(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        _chat, _request, _plan, stored = _create(session)
        session.commit()
        with pytest.raises(IntegrityError):
            session.execute(
                text("UPDATE prompt_expansion_items SET original_rendered_prompt = 'changed'")
            )
            session.flush()
        session.rollback()
        with pytest.raises(IntegrityError):
            session.execute(
                text("UPDATE prompt_expansion_batches SET plan_version = plan_version + 2")
            )
            session.flush()
        session.rollback()
        with pytest.raises(IntegrityError, match="already has all items"):
            session.execute(
                text(
                    "INSERT INTO prompt_expansion_items "
                    "(id, batch_id, ordinal, original_evidence_json, current_evidence_json, "
                    "original_rendered_prompt, original_rendered_sha256, reviewed_prompt, "
                    "reviewed_sha256, selected, review_version, reroll_count, "
                    "created_at, updated_at) "
                    "SELECT 'ptitem_extra', batch_id, 3, original_evidence_json, "
                    "current_evidence_json, original_rendered_prompt, original_rendered_sha256, "
                    "reviewed_prompt, reviewed_sha256, selected, 1, 0, created_at, updated_at "
                    "FROM prompt_expansion_items WHERE ordinal = 1"
                )
            )
            session.flush()
        session.rollback()
        session.execute(
            text(
                "UPDATE prompt_expansion_batches SET state = 'queued', "
                "plan_version = plan_version + 1"
            )
        )
        session.commit()
        with pytest.raises(IntegrityError):
            session.execute(
                text(
                    "UPDATE prompt_expansion_items SET reviewed_prompt = 'late', "
                    "reviewed_sha256 = :sha, review_version = review_version + 1"
                ),
                {"sha": hashlib.sha256(b"prompt-expansion-rendered-v1\x00late").hexdigest()},
            )
            session.flush()


def test_migration_is_self_contained_and_chat_delete_cascades(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "migration")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "head")
    database = settings.state_dir / "local-lm.sqlite3"
    migrated_engine = create_engine(f"sqlite+pysqlite:///{database}")
    try:
        # Live metadata must accept the historical trigger bodies exactly.
        Base.metadata.create_all(migrated_engine)
    finally:
        migrated_engine.dispose()
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE 'prompt_expansion_%'"
            )
        }
        assert {"prompt_expansion_batches", "prompt_expansion_items"} <= tables
        assert triggers == {
            "prompt_expansion_batch_insert_guard",
            "prompt_expansion_batch_update_guard",
            "prompt_expansion_item_insert_guard",
            "prompt_expansion_item_update_guard",
        }
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "f2a7c9d41e63",
        )
        batch_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(prompt_expansion_batches)")
        }
        item_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(prompt_expansion_items)")
        }
        assert {"queue_idempotency_key", "work_plan_id", "queued_at"} <= batch_columns
        assert {"work_step_id", "run_id", "media_seed"} <= item_columns

    command.downgrade(config, "c1e7a4b92d60")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "c1e7a4b92d60",
        )
        batch_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(prompt_expansion_batches)")
        }
        item_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(prompt_expansion_items)")
        }
        assert {"queue_idempotency_key", "work_plan_id", "queued_at"}.isdisjoint(batch_columns)
        assert {"work_step_id", "run_id", "media_seed"}.isdisjoint(item_columns)
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE 'prompt_expansion_%'"
            )
        } == {
            "prompt_expansion_batch_insert_guard",
            "prompt_expansion_batch_update_guard",
            "prompt_expansion_item_insert_guard",
            "prompt_expansion_item_update_guard",
        }

    command.upgrade(config, "head")
    command.downgrade(config, "b7c1e4a90f26")
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'prompt_expansion_%'"
            ).fetchall()
            == []
        )
