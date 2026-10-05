"""Ordered generation keeps selected purposes and the canvas produced by an earlier step."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import func, select
from test_image_role_execution import selected_images, selected_workflow

from local_lm.accepted_turn_context import accepted_context
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.media_input_roles import resolved_image_roles
from local_lm.models import (
    Message,
    Run,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
    WorkStep,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.schemas import EngineCapabilities


async def enable_purposes(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    media = app.state.services.engines.media
    original: EngineCapabilities = await media.capabilities()

    async def capabilities() -> EngineCapabilities:
        return original.model_copy(update={"image_slot_bindings": True})

    monkeypatch.setattr(media, "capabilities", capabilities)


@pytest.mark.parametrize("story_first", [False, True])
@pytest.mark.parametrize("canvas", [False, True])
async def test_the_first_media_step_receives_selected_pictures_in_their_accepted_order(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    story_first: bool,
    canvas: bool,
) -> None:
    await enable_purposes(app, monkeypatch)
    ids = await selected_images(client)
    roles = ["reference", "edit_source" if canvas else "reference", "reference"]
    revision_id = selected_workflow(
        ("edit_source", "reference", "reference") if canvas else ("reference",) * 3
    )
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        definition.operation = "image_to_image" if canvas else "text_to_image"
        session.commit()
    chat = (await client.post("/api/chats", json={"title": "Garden sequence"})).json()
    words = (
        "Write a short story about a garden, then create an image based on it"
        if story_first
        else "Create an image of a garden, then write a short story about it"
    )
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": words,
                "mode": "auto",
                "confirm_media": True,
                "input_artifact_ids": ids,
                "input_image_roles": roles,
                "role_overrides": {"image": {"workflow_revision_id": revision_id}},
            },
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run).where(Run.work_plan_id == response.json()["run"]["work_plan_id"])
                )
            )
            media = next(run for run in runs if run.operation != "text")
            frozen = accepted_context(session, media)
            assert frozen is not None
            assert frozen.input_artifact_ids == ids
            assert frozen.input_image_roles == roles
            assert frozen.operation == ("image_to_image" if canvas else "text_to_image")
            assert media.provenance_json["input_artifact_ids"] == ids
            assert media.provenance_json["input_image_roles"] == roles


@pytest.mark.parametrize("ambiguous", [False, True])
async def test_a_generated_canvas_keeps_its_frozen_purpose_after_bindings_change(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    ambiguous: bool,
) -> None:
    await enable_purposes(app, monkeypatch)
    initial_revision = selected_workflow(("reference", "reference", "reference"))
    consumer_revision = selected_workflow(("edit_source",))
    with SessionLocal() as session:
        family = WorkflowFamily(name="Garden sequence pictures")
        session.add(family)
        session.flush()
        session.add(WorkflowPreference(family=family, selector_capability="image", enabled=True))
        for revision_id, operation, variant in (
            (initial_revision, "text_to_image", "create"),
            (consumer_revision, "image_to_image", "edit"),
        ):
            revision = session.get(WorkflowRevision, revision_id)
            assert revision is not None
            definition = session.get(WorkflowDefinition, revision.workflow_id)
            assert definition is not None
            definition.family_id, definition.variant_key = family.id, variant
            definition.operation = operation
        family_id = family.id
        session.commit()
    ids = await selected_images(client)
    chat = (await client.post("/api/chats", json={"title": "Generated garden canvas"})).json()
    original_generate = app.state.services.engines.media.generate
    requests: list[MediaRequest] = []

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        requests.append(request)
        async for event in original_generate(request):
            if ambiguous and event.type == "complete" and len(requests) == 1:
                event = replace(
                    event,
                    assets=[
                        *event.assets,
                        replace(
                            event.assets[0],
                            content=event.assets[0].content + b"\n",
                            name="another-garden.svg",
                        ),
                    ],
                )
            yield event

    monkeypatch.setattr(app.state.services.engines.media, "generate", generate)
    original_resolve = ConversationOrchestrator._resolve_step_inputs
    consumer_id = ""

    def resolve(session: Any, run: Run) -> None:
        if run.id == consumer_id:
            step = session.get(WorkStep, run.work_step_id)
            assert step is not None
            step.input_bindings_json = []
            run.provenance_json = {**run.provenance_json, "input_image_roles": ["reference"]}
            session.flush()
        original_resolve(session, run)

    monkeypatch.setattr(ConversationOrchestrator, "_resolve_step_inputs", staticmethod(resolve))
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Create an image of a garden, then create an image based on it",
                "mode": "auto",
                "confirm_media": True,
                "input_artifact_ids": ids,
                "input_image_roles": ["reference"] * 3,
                "role_overrides": {
                    "image": {
                        "workflow_selection": {
                            "selector_capability": "image",
                            "mode": "family",
                            "workflow_family_id": family_id,
                        }
                    }
                },
            },
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            consumer = session.scalar(
                select(Run).where(
                    Run.work_plan_id == response.json()["run"]["work_plan_id"],
                    Run.operation == "image_to_image",
                )
            )
            assert consumer is not None
            consumer_id = consumer.id
            frozen = accepted_context(session, consumer)
            assert frozen is not None
            assert frozen.input_artifact_ids == [] and frozen.input_image_roles == []
            assert len(frozen.dependencies) == 1
            assert frozen.dependencies[0].image_role == "edit_source"

    async def read() -> dict[str, Any]:
        with SessionLocal() as session:
            consumer = session.get(Run, consumer_id)
            assert consumer is not None
            return {"status": consumer.status, "error": consumer.error}

    settled = await wait_for_terminal_status(read, what="the generated-canvas edit", expected=None)
    if ambiguous:
        assert settled["status"] == "failed", settled["error"]
        assert "dependency image purposes are ambiguous" in settled["error"]
        assert len(requests) == 1
        return
    assert settled["status"] == "complete", settled["error"]
    assert len(requests) == 2
    assert requests[0].input_image_bindings == {
        "input_image_0": (0,),
        "input_image_1": (1,),
        "input_image_2": (2,),
    }
    assert len(requests[1].input_paths) == 1
    assert requests[1].input_image_bindings == {"input_image_0": (0,)}
    assert requests[1].input_paths[0] not in requests[0].input_paths


@pytest.mark.parametrize("dependency_roles", [{}, {"later": "edit_source"}])
def test_an_unbound_or_second_dependency_canvas_is_refused(dependency_roles: Any) -> None:
    with pytest.raises(ValueError, match="dependency image purposes|one picture"):
        resolved_image_roles(["canvas"], ["edit_source"], ["later"], dependency_roles)


@pytest.mark.parametrize("defect", ["adapter", "unknown", "capacity"])
async def test_an_unusable_later_media_step_refuses_the_whole_plan_before_storage(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    if defect != "adapter":
        await enable_purposes(app, monkeypatch)
    ids = await selected_images(client)
    declared = (
        ("edit_source", "reference", "unknown")
        if defect == "unknown"
        else (
            ("edit_source", "reference")
            if defect == "capacity"
            else ("edit_source", "reference", "reference")
        )
    )
    revision_id = selected_workflow(declared)
    chat = (await client.post("/api/chats", json={"title": "Garden plan refusal"})).json()
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Write a short story about a garden, then create an image based on it",
                "mode": "auto",
                "confirm_media": True,
                "input_artifact_ids": ids,
                "input_image_roles": ["reference", "edit_source", "reference"],
                "role_overrides": {"image": {"workflow_revision_id": revision_id}},
            },
        )
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "turn-invalid"
        assert any(
            word in response.json()["detail"]
            for word in (
                "cannot bind pictures",
                "separate picture",
                "image purposes",
            )
        )
        with SessionLocal() as session:
            assert (
                session.scalar(
                    select(func.count()).select_from(Run).where(Run.chat_id == chat["id"])
                )
                == 0
            )
            assert (
                session.scalar(
                    select(func.count()).select_from(Message).where(Message.chat_id == chat["id"])
                )
                == 0
            )
