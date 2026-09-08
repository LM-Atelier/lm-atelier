from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from local_lm.db import SessionLocal
from local_lm.models import (
    ChatWorkflowSelection,
    Job,
    Message,
    ModelInstall,
    ModelProfile,
    PromptExpansionBatch,
    Run,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
)
from local_lm.prompt_expansion_use import (
    PromptExpansionUseError,
    read_prompt_batch_queue_selection,
)
from local_lm.schemas import TurnRequest


def _image_workflow(session: Session, name: str, use_case: str = "") -> tuple[str, str]:
    family = WorkflowFamily(name=name, use_case=use_case)
    definition = WorkflowDefinition(
        family=family, variant_key="create", name=name, operation="text_to_image"
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="mock",
        trusted=True,
        api_graph_json={},
        input_schema_json={},
        dependencies_json={},
    )
    session.add_all(
        [
            family,
            definition,
            revision,
            WorkflowPreference(family=family, selector_capability="image"),
        ]
    )
    session.flush()
    definition.current_revision_id = revision.id
    return family.id, revision.id


def _row_counts(session: Session) -> tuple[int, ...]:
    return tuple(
        session.scalar(select(func.count()).select_from(model)) or 0
        for model in (Message, Run, Job, WorkPlan, WorkStep)
    )


def _choice(revision_id: str, legacy: bool, capability: str = "image") -> dict[str, Any]:
    if legacy:
        return {"workflow_revision_id": revision_id}
    return {
        "workflow_selection": {
            "selector_capability": capability,
            "mode": "revision",
            "workflow_revision_id": revision_id,
        }
    }


async def test_turn_automatic_workflow_bypasses_the_saved_chat_family(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Local automatic choice"})).json()
    with SessionLocal() as session:
        saved_family, saved_revision = _image_workflow(session, "Geometric shapes", "geometry")
        _, ranked_revision = _image_workflow(
            session, "Paper boats", "watercolor paper boat landscape"
        )
        saved = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat["id"],
                ChatWorkflowSelection.selector_capability == "image",
            )
        )
        assert saved is not None
        saved.mode = "family"
        saved.workflow_family_id = saved_family
        session.commit()
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "A watercolor paper boat landscape",
                "mode": "image",
                "workflow_selection": {"selector_capability": "image", "mode": "automatic"},
            },
        )
        assert response.status_code == 202, response.text
        assert response.json()["run"]["workflow_revision_id"] == ranked_revision
        assert ranked_revision != saved_revision
        with SessionLocal() as session:
            saved = session.scalar(
                select(ChatWorkflowSelection).where(
                    ChatWorkflowSelection.chat_id == chat["id"],
                    ChatWorkflowSelection.selector_capability == "image",
                )
            )
            assert saved is not None and saved.workflow_family_id == saved_family


def _text_workflow(
    session: Session,
    condition: str,
) -> tuple[str, str, str, str]:
    install = ModelInstall(
        name="Bound chat install",
        role="chat",
        engine="mock",
        local_path="constructed-chat-model",
        active=condition != "inactive_install",
    )
    session.add(install)
    session.flush()
    profile = ModelProfile(
        name="Bound chat profile",
        role="chat",
        engine="mock",
        model_install_id=install.id,
    )
    family = WorkflowFamily(name="Exact text workflow")
    definition = WorkflowDefinition(
        family=family,
        variant_key="text",
        name="Text variant",
        operation="text",
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="llama.cpp" if condition == "wrong_engine" else "mock",
        trusted=True,
        dependency_contract_sha256="a" * 64,
    )
    newer = WorkflowRevision(
        definition=definition,
        version=2,
        engine="mock",
        trusted=True,
    )
    session.add_all([profile, family, definition, revision, newer])
    session.flush()
    definition.current_revision_id = newer.id
    activation = WorkflowActivation(
        workflow_revision_id=revision.id,
        resolver_version="resolver-v1",
        dependency_contract_sha256="a" * 64,
        binding_sha256="b" * 64,
        state="ready",
        is_active=condition != "inactive_activation",
        details_json={"launch_sha256": "c" * 64},
    )
    slot = WorkflowDependencySlot(
        workflow_revision_id=revision.id,
        name="primary",
        resource_kind="model_profile",
        required=True,
        satisfaction="all_of",
        requirements_json=[{"key": "default", "constraints": {}}],
        contract_sha256="d" * 64,
        ordinal=0,
    )
    session.add_all([activation, slot])
    session.flush()
    if condition != "missing_binding":
        session.add(
            WorkflowDependencyBinding(
                workflow_revision_id=revision.id,
                workflow_activation_id=activation.id,
                workflow_dependency_slot_id=slot.id,
                requirement_key="default",
                model_profile_id=profile.id,
                resource_identity_sha256="e" * 64,
            )
        )
    session.flush()
    return revision.id, newer.id, profile.id, activation.id


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "condition",
    [
        "ready",
        "missing_binding",
        "inactive_install",
        "inactive_activation",
        "wrong_engine",
    ],
)
async def test_exact_text_revision_keeps_its_bound_profile_or_refuses_atomically(
    app: FastAPI,
    client: AsyncClient,
    legacy: bool,
    condition: str,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Exact text choice"})).json()
    with SessionLocal() as session:
        revision_id, newer_id, profile_id, activation_id = _text_workflow(session, condition)
        session.commit()
        before = _row_counts(session)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Describe a paper boat in one sentence",
                "mode": "text",
                **_choice(revision_id, legacy, "chat"),
            },
        )
        if condition != "ready":
            assert response.status_code == 422, response.text
            assert response.json()["code"] != "request-validation-invalid"
            with SessionLocal() as session:
                assert _row_counts(session) == before
            return
        assert response.status_code == 202, response.text
        result = response.json()["run"]
        assert result["workflow_revision_id"] == revision_id != newer_id
        assert result["profile_id"] == profile_id
        with SessionLocal() as session:
            run = session.get(Run, result["id"])
            assert run is not None
            step = session.get(WorkStep, run.work_step_id)
            assert step is not None and step.workflow_revision_id == revision_id
            assert step.profile_id == profile_id
            assert run.provenance_json["model_selection"]["workflow_activation_id"] == activation_id


@pytest.mark.parametrize("selection_mode", ["default", "automatic", "family"])
async def test_ordered_workflow_without_a_matching_role_creates_no_plan(
    app: FastAPI,
    client: AsyncClient,
    selection_mode: str,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Ordered workflow mismatch"})).json()
    with SessionLocal() as session:
        family_id, _ = _image_workflow(session, "Unused family")
        session.commit()
        before = _row_counts(session)
    selection: dict[str, Any] = {"selector_capability": "video", "mode": selection_mode}
    if selection_mode == "family":
        selection["workflow_family_id"] = family_id
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Write a short story about a paper boat, then create an image based on it",
                "mode": "auto",
                "confirm_media": True,
                "workflow_selection": selection,
            },
        )
        assert response.status_code == 422, response.text
        assert "operation in this plan" in response.json()["detail"]
        with SessionLocal() as session:
            assert _row_counts(session) == before


@pytest.mark.parametrize("selector", ["legacy", "video", "image"])
async def test_ordered_exact_revision_refuses_absent_or_misdeclared_role(
    app: FastAPI,
    client: AsyncClient,
    selector: str,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Ordered exact role"})).json()
    with SessionLocal() as session:
        definition = WorkflowDefinition(
            name="Video-only variant",
            operation="text_to_video",
        )
        revision = WorkflowRevision(
            definition=definition,
            version=1,
            engine="mock",
            trusted=True,
            api_graph_json={},
            input_schema_json={},
            dependencies_json={},
        )
        session.add_all([definition, revision])
        session.flush()
        revision_id = revision.id
        definition.current_revision_id = revision_id
        session.commit()
        before = _row_counts(session)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Write a short story about a paper boat, then create an image based on it",
                "mode": "auto",
                "confirm_media": True,
                **_choice(revision_id, selector == "legacy", selector),
            },
        )
        assert response.status_code == 422, response.text
        assert response.json()["code"] != "request-validation-invalid"
        with SessionLocal() as session:
            assert _row_counts(session) == before


async def _resource_batch(
    client: AsyncClient,
    chat_id: str,
    revision_ids: list[str],
) -> dict[str, Any]:
    resource_policy: dict[str, Any] = (
        {
            "mode": "fixed",
            "workflow_revision_id": revision_ids[0],
            "lora_policy": {"mode": "none"},
        }
        if len(revision_ids) == 1
        else {
            "mode": "pool",
            "strategy": "round_robin",
            "options": [
                {"workflow_revision_id": revision_id, "lora_policy": {"mode": "none"}}
                for revision_id in revision_ids
            ],
        }
    )
    template_response = await client.post(
        "/api/prompt-templates",
        json={
            "idempotency_key": "workflow-resource-template",
            "name": "Paper boat resource selection",
            "description": "Constructed workflow selection fixture",
            "contract": {
                "schema_version": 1,
                "operation": "text_to_image",
                "body": "A {{subject}}.",
                "slots": [{"name": "subject", "mode": "input", "variation_scope": "item"}],
                "resource_policy": resource_policy,
            },
        },
    )
    assert template_response.status_code == 201, template_response.text
    revision = template_response.json()["revision"]
    response = await client.post(
        f"/api/chats/{chat_id}/prompt-batches",
        json={
            "idempotency_key": "workflow-resource-preview",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": 2,
            "selection_seed": 0,
            "inputs": {"subject": ["paper boat", "painted hillside"]},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _composer_source(batch: dict[str, Any]) -> dict[str, Any]:
    item = batch["items"][0]
    return {
        "version": 1,
        "batch_id": batch["id"],
        "expected_plan_version": batch["plan_version"],
        "expected_plan_sha256": batch["plan_sha256"],
        "item_id": item["id"],
        "expected_review_version": item["review_version"],
        "expected_reviewed_sha256": item["reviewed_sha256"],
        "prompt_template_id": batch["prompt_template_id"],
        "prompt_template_revision_id": batch["prompt_template_revision_id"],
        "contract_sha256": batch["contract_sha256"],
    }


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("agrees", [False, True])
async def test_prompt_source_checks_the_explicit_workflow_before_resource_override(
    app: FastAPI,
    client: AsyncClient,
    legacy: bool,
    agrees: bool,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Resource agreement"})).json()
    with SessionLocal() as session:
        _, resource_id = _image_workflow(session, "Template resource")
        _, conflicting_id = _image_workflow(session, "Other workflow")
        session.commit()
    batch = await _resource_batch(client, chat["id"], [resource_id])
    with SessionLocal() as session:
        before = _row_counts(session)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": batch["items"][0]["reviewed_prompt"],
                "mode": "image",
                "output_count": 1,
                "idempotency_key": "resource-turn",
                "prompt_source": _composer_source(batch),
                **_choice(resource_id if agrees else conflicting_id, legacy),
            },
        )
        if agrees:
            assert response.status_code == 202, response.text
            assert response.json()["run"]["workflow_revision_id"] == resource_id
        else:
            assert response.status_code == 422, response.text
            assert response.json()["code"] != "request-validation-invalid"
            with SessionLocal() as session:
                assert _row_counts(session) == before
                stored = session.get(PromptExpansionBatch, batch["id"])
                assert stored is not None and stored.work_plan_id is None
                assert stored.plan_version == batch["plan_version"]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("case", ["single_agrees", "single_conflicts", "first_only_agrees"])
async def test_batch_workflow_choice_must_agree_with_every_selected_resource(
    app: FastAPI,
    client: AsyncClient,
    legacy: bool,
    case: str,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Exact batch resources"})).json()
    with SessionLocal() as session:
        _, first_id = _image_workflow(session, "First resource")
        _, second_id = _image_workflow(session, "Second resource")
        session.commit()
    ids = [first_id, second_id] if case == "first_only_agrees" else [first_id]
    batch = await _resource_batch(client, chat["id"], ids)
    selected_id = second_id if case == "single_conflicts" else first_id
    request = TurnRequest.model_validate(
        {
            "text": batch["items"][0]["reviewed_prompt"],
            "mode": "image",
            "output_count": 2,
            "idempotency_key": "batch-workflow-turn",
            **_choice(selected_id, legacy),
        }
    )
    orchestrator = app.state.services.orchestrator
    # The HTTP batch queue does not expose a workflow override. Exercise the
    # shared admission boundary with its real, validated batch selection.
    async with app.state.services.scheduler.lease("primary"), orchestrator.chat_guard(chat["id"]):
        with SessionLocal() as session:
            selection = read_prompt_batch_queue_selection(
                session,
                chat["id"],
                batch["id"],
                batch["plan_version"],
                batch["plan_sha256"],
                expected_engine="mock",
            )
            before = _row_counts(session)
            if case != "single_agrees":
                with pytest.raises(PromptExpansionUseError):
                    await orchestrator._create_new_turn(
                        session,
                        chat["id"],
                        request,
                        source_action="prompt_library",
                        prompt_batch_selection=selection,
                    )
                assert _row_counts(session) == before
                session.rollback()
                assert _row_counts(session) == before
                stored = session.get(PromptExpansionBatch, batch["id"])
                assert stored is not None and stored.work_plan_id is None
                assert stored.plan_version == batch["plan_version"]
                return
            accepted = await orchestrator._create_new_turn(
                session,
                chat["id"],
                request,
                source_action="prompt_library",
                prompt_batch_selection=selection,
            )
            steps = list(
                session.scalars(
                    select(WorkStep).where(
                        WorkStep.plan_id == accepted.run.work_plan_id,
                    )
                )
            )
            assert len(steps) == 2
            assert {step.workflow_revision_id for step in steps} == {first_id}
