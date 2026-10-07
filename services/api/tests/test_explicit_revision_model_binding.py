"""A turn naming only a workflow revision runs with the one model that revision is bound to.

The studio's Isolate tool and any recipe send a revision without a model. When
that revision declares its model, the chat's default model cannot run it, and
the turn used to be refused as "not ready" even though exactly one installed
model could. Several matches, or none, still refuse: the turn never guesses.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from local_lm.db import SessionLocal
from local_lm.models import (
    Job,
    Message,
    ModelInstall,
    ModelProfile,
    Run,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
)


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 140, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


async def _source(client: AsyncClient) -> str:
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("source.png", _png(), "image/png")}
    )
    assert uploaded.status_code == 201, uploaded.text
    return str(uploaded.json()["id"])


def _row_counts(session: Session) -> tuple[int, ...]:
    return tuple(
        session.scalar(select(func.count()).select_from(model)) or 0
        for model in (Message, Run, Job, WorkPlan, WorkStep)
    )


def _profile(session: Session, name: str, install: ModelInstall) -> ModelProfile:
    profile = ModelProfile(
        name=name, role="image", engine=install.engine, model_install_id=install.id
    )
    session.add(profile)
    session.flush()
    return profile


def _bound_edit_workflow(
    session: Session,
    *,
    profiles: int = 1,
    active: bool = True,
    engine: str = "mock",
) -> tuple[str, list[str]]:
    """An edit workflow whose revision declares the one model install it runs."""
    install = ModelInstall(
        name="Cutout model",
        role="image",
        engine=engine,
        local_path="constructed-cutout-model",
        active=active,
    )
    session.add(install)
    session.flush()
    profile_ids = [
        _profile(session, f"Cutout profile {index}", install).id for index in range(profiles)
    ]
    definition = WorkflowDefinition(name="Cutout", operation="image_to_image")
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="mock",
        trusted=True,
        api_graph_json={},
        input_schema_json={},
        dependencies_json={"model_install_ids": [install.id]},
    )
    session.add_all([definition, revision])
    session.flush()
    definition.current_revision_id = revision.id
    return revision.id, profile_ids


async def _turn(client: AsyncClient, chat_id: str, source: str, **extra: Any) -> Any:
    return await client.post(
        f"/api/chats/{chat_id}/turns",
        json={
            "text": "Cut out the subject",
            "mode": "image",
            "input_artifact_ids": [source],
            **extra,
        },
    )


@pytest.mark.parametrize("legacy", [True, False])
async def test_a_revision_named_alone_runs_with_its_one_bound_model(
    app: FastAPI, client: AsyncClient, legacy: bool
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Named revision"})).json()
    source = await _source(client)
    with SessionLocal() as session:
        revision_id, (profile_id,) = _bound_edit_workflow(session)
        session.commit()
    choice: dict[str, Any] = (
        {"workflow_revision_id": revision_id}
        if legacy
        else {
            "workflow_selection": {
                "selector_capability": "image",
                "mode": "revision",
                "workflow_revision_id": revision_id,
            }
        }
    )

    async with app.state.services.scheduler.lease("primary"):
        response = await _turn(client, chat["id"], source, **choice)

    assert response.status_code == 202, response.text
    run = response.json()["run"]
    assert run["workflow_revision_id"] == revision_id
    assert run["profile_id"] == profile_id


async def test_a_newer_default_model_the_revision_cannot_run_is_not_used(
    app: FastAPI, client: AsyncClient
) -> None:
    """The live case: the chat's default model is ready, newer, and wrong for this workflow."""
    chat = (await client.post("/api/chats", json={"title": "Incompatible default"})).json()
    source = await _source(client)
    with SessionLocal() as session:
        revision_id, (profile_id,) = _bound_edit_workflow(session)
        other = ModelInstall(
            name="Default text-to-image model",
            role="image",
            engine="mock",
            local_path="constructed-default-model",
            active=True,
        )
        session.add(other)
        session.flush()
        default = _profile(session, "Default image profile", other)
        for profile in session.scalars(select(ModelProfile).where(ModelProfile.role == "image")):
            profile.is_default = profile.id == default.id
        session.commit()

    async with app.state.services.scheduler.lease("primary"):
        response = await _turn(client, chat["id"], source, workflow_revision_id=revision_id)

    assert response.status_code == 202, response.text
    assert response.json()["run"]["profile_id"] == profile_id


@pytest.mark.parametrize(
    ("condition", "setup"),
    [
        ("no ready model: its install is inactive", {"active": False}),
        ("no ready model: its install runs on another engine", {"engine": "comfyui"}),
        ("two ready models could run it", {"profiles": 2}),
    ],
)
async def test_a_revision_named_alone_refuses_without_exactly_one_ready_model(
    app: FastAPI, client: AsyncClient, condition: str, setup: dict[str, Any]
) -> None:
    chat = (await client.post("/api/chats", json={"title": condition})).json()
    source = await _source(client)
    with SessionLocal() as session:
        revision_id, _ = _bound_edit_workflow(session, **setup)
        session.commit()
        before = _row_counts(session)

    async with app.state.services.scheduler.lease("primary"):
        response = await _turn(client, chat["id"], source, workflow_revision_id=revision_id)

    assert response.status_code == 422, response.text
    with SessionLocal() as session:
        assert _row_counts(session) == before


async def test_an_explicit_model_the_revision_cannot_run_still_refuses(
    app: FastAPI, client: AsyncClient
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Explicit mismatch"})).json()
    source = await _source(client)
    with SessionLocal() as session:
        revision_id, _ = _bound_edit_workflow(session)
        other = ModelInstall(
            name="Other model",
            role="image",
            engine="mock",
            local_path="constructed-other-model",
            active=True,
        )
        session.add(other)
        session.flush()
        mismatched = _profile(session, "Other profile", other).id
        session.commit()
        before = _row_counts(session)

    async with app.state.services.scheduler.lease("primary"):
        response = await _turn(
            client,
            chat["id"],
            source,
            workflow_revision_id=revision_id,
            profile_id=mismatched,
        )

    assert response.status_code == 422, response.text
    with SessionLocal() as session:
        assert _row_counts(session) == before


async def test_an_activation_bound_model_wins_over_other_ready_models(
    app: FastAPI, client: AsyncClient
) -> None:
    """Two profiles share the install, but the activation binds one of them."""
    chat = (await client.post("/api/chats", json={"title": "Activation binding"})).json()
    source = await _source(client)
    with SessionLocal() as session:
        revision_id, (bound, _unbound) = _bound_edit_workflow(session, profiles=2)
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        family = WorkflowFamily(name="Cutout family")
        session.add(family)
        session.flush()
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        definition.family = family
        definition.variant_key = "edit"
        revision.dependency_contract_sha256 = "a" * 64
        activation = WorkflowActivation(
            workflow_revision_id=revision.id,
            resolver_version="resolver-v1",
            dependency_contract_sha256="a" * 64,
            binding_sha256="b" * 64,
            state="ready",
            is_active=True,
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
        session.add(
            WorkflowDependencyBinding(
                workflow_revision_id=revision.id,
                workflow_activation_id=activation.id,
                workflow_dependency_slot_id=slot.id,
                requirement_key="default",
                model_profile_id=bound,
                resource_identity_sha256="e" * 64,
            )
        )
        session.commit()

    async with app.state.services.scheduler.lease("primary"):
        response = await _turn(client, chat["id"], source, workflow_revision_id=revision_id)

    assert response.status_code == 202, response.text
    assert response.json()["run"]["profile_id"] == bound
