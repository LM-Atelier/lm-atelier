"""Selected picture purposes reach the exact accepted workflow inputs."""

import io
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import func, select

from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.db import SessionLocal
from local_lm.domain import Operation, RoutingMode
from local_lm.models import Artifact, Chat, Job, Run, WorkflowDefinition, WorkflowRevision
from local_lm.scheduler import JobClaim
from local_lm.schemas import EngineCapabilities, TurnRequest
from local_lm.workflow_image_slots_v1 import SLOT_KEY, ImageSlot


async def selected_images(client: AsyncClient) -> list[str]:
    result: list[str] = []
    for index in range(3):
        content = io.BytesIO()
        Image.new("RGB", (32 + index, 32), (40, 80 + index, 120)).save(content, format="PNG")
        uploaded = await client.post(
            "/api/artifacts",
            files={"file": (f"garden-{index}.png", content.getvalue(), "image/png")},
        )
        assert uploaded.status_code == 201
        result.append(uploaded.json()["id"])
    return result


def selected_workflow(roles: tuple[str, ...] = ("edit_source", "reference", "reference")) -> str:
    graph: dict[str, Any] = {}
    properties: dict[str, Any] = {}
    for index, role in enumerate(roles):
        name = f"input_image_{index}"
        node = str(index + 10)
        graph[node] = {"class_type": "LoadImage", "inputs": {"image": "${" + name + "}"}}
        properties[name] = {
            "type": "string",
            SLOT_KEY: {
                **ImageSlot(name, node, "unknown", "none").as_schema(),
                "role": role,
                "basis": "declared" if role != "unknown" else "none",
            },
        }
    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Garden pictures", operation="image_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            trusted=True,
            api_graph_json=graph,
            input_schema_json={"type": "object", "properties": properties},
            dependencies_json={},
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        revision_id = revision.id
        session.commit()
    return revision_id


@pytest.mark.parametrize(
    ("roles", "expected"),
    [
        (["reference", "reference"], Operation.TEXT_TO_IMAGE),
        (["reference", "edit_source"], Operation.IMAGE_TO_IMAGE),
        (None, Operation.IMAGE_TO_IMAGE),
    ],
)
async def test_only_a_selected_canvas_makes_attachments_an_edit(
    app: FastAPI, client: AsyncClient, roles: Any, expected: Operation
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Garden routing"})).json()
    request = TurnRequest(
        text="Paint a garden",
        mode=RoutingMode.IMAGE,
        input_artifact_ids=["layout", "canvas"],
        input_image_roles=roles,
    )
    with SessionLocal() as session:
        stored_chat = session.get(Chat, chat["id"])
        assert stored_chat is not None
        plan = await app.state.services.orchestrator._turn_routing_plan(
            stored_chat,
            request,
            RoutingMode.IMAGE,
            accepted_offer=None,
            prompt_batch_selection=None,
            has_prior_image=False,
            routing_context=[],
        )
    assert plan.operation == expected
    assert request.input_artifact_ids == ["layout", "canvas"]


@pytest.mark.parametrize("later_change", ["request", "provenance", "workflow", "capability"])
async def test_accepted_pictures_keep_their_purposes_through_media_dispatch(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, later_change: str
) -> None:
    ids = await selected_images(client)
    revision_id = selected_workflow()
    chat = (await client.post("/api/chats", json={"title": "Garden execution"})).json()
    orchestrator = app.state.services.orchestrator
    base_capabilities: EngineCapabilities = await orchestrator.engines.media.capabilities()

    async def capabilities() -> EngineCapabilities:
        return base_capabilities.model_copy(update={"image_slot_bindings": True})

    monkeypatch.setattr(orchestrator.engines.media, "capabilities", capabilities)
    seen: list[MediaRequest] = []
    compiled_images: list[str] = []
    adapter = ComfyUIAdapter("http://127.0.0.1:1")
    paths: list[Path] = []

    async def upload(request: MediaRequest) -> list[str]:
        return [f"selected-{paths.index(path)}.png" for path in request.input_paths]

    monkeypatch.setattr(adapter, "_upload_inputs", upload)

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        seen.append(request)
        parameters = await adapter._request_parameters(request)
        graph = adapter._compile(request.workflow, parameters)
        compiled_images.extend(graph[str(index + 10)]["inputs"]["image"] for index in range(3))
        yield MediaEvent(type="cancelled")

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    request = TurnRequest(
        text="Paint a garden",
        mode=RoutingMode.IMAGE,
        input_artifact_ids=ids,
        input_image_roles=["reference", "edit_source", "reference"],
        workflow_revision_id=revision_id,
    )
    try:
        async with app.state.services.scheduler.lease("primary"):
            with SessionLocal() as session:
                accepted = await orchestrator.create_turn(
                    session, chat["id"], request, activate_branch=False
                )
                run = session.get(Run, accepted.run.id)
                assert run is not None
                run_id = run.id
                job = session.scalar(select(Job).where(Job.run_id == run_id))
                assert job is not None
                claim = JobClaim(token="garden-picture-attempt", attempt=1)
                job.status = "running"
                job.claim_owner, job.attempt = claim.token, claim.attempt
                job_id = job.id
                for artifact_id in ids:
                    artifact = session.get(Artifact, artifact_id)
                    assert artifact is not None
                    paths.append(orchestrator.artifacts.verified_path(artifact))
                if later_change == "provenance":
                    run.provenance_json = {
                        **run.provenance_json,
                        "input_artifact_ids": list(reversed(ids)),
                        "input_image_roles": ["edit_source", "reference", "reference"],
                    }
                elif later_change == "workflow":
                    revision = session.get(WorkflowRevision, revision_id)
                    assert revision is not None
                    revision.api_graph_json = {"later": {"inputs": {}}}
                    revision.input_schema_json = {"properties": {}}
                session.commit()
            if later_change == "request":
                request.input_artifact_ids.reverse()
                request.input_image_roles = ["edit_source", "reference", "reference"]
            elif later_change == "capability":

                async def unavailable() -> EngineCapabilities:
                    return base_capabilities

                monkeypatch.setattr(orchestrator.engines.media, "capabilities", unavailable)
                with pytest.raises(RuntimeError, match="cannot bind pictures"):
                    await orchestrator._execute_media(job_id, run_id, claim)
                assert seen == []
                return
            await orchestrator._execute_media(job_id, run_id, claim)
        assert len(seen) == 1
        assert seen[0].input_paths == paths
        assert seen[0].input_image_bindings == {
            "input_image_0": (1,),
            "input_image_1": (0,),
            "input_image_2": (2,),
        }
        assert compiled_images == ["selected-1.png", "selected-0.png", "selected-2.png"]
    finally:
        await adapter.close()


@pytest.mark.parametrize("defect", ["adapter", "unknown", "missing_reference"])
async def test_incomplete_image_purposes_refuse_before_a_run_is_created(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    ids = await selected_images(client)
    roles = (
        ("edit_source", "reference", "unknown")
        if defect == "unknown"
        else ("edit_source", "reference", "reference")
    )
    revision_id = selected_workflow(roles)
    chat = (await client.post("/api/chats", json={"title": "Garden admission"})).json()
    orchestrator = app.state.services.orchestrator
    base_capabilities: EngineCapabilities = await orchestrator.engines.media.capabilities()

    async def capabilities() -> EngineCapabilities:
        return base_capabilities.model_copy(update={"image_slot_bindings": defect != "adapter"})

    monkeypatch.setattr(orchestrator.engines.media, "capabilities", capabilities)
    if defect == "missing_reference":
        ids = ids[:2]
    request = TurnRequest(
        text="Paint a garden",
        mode=RoutingMode.IMAGE,
        input_artifact_ids=ids,
        input_image_roles=["reference", "edit_source", *(["reference"] if len(ids) == 3 else [])],
        workflow_revision_id=revision_id,
    )
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            with pytest.raises(
                ValueError, match="cannot bind pictures|separate picture|image purposes"
            ):
                await orchestrator.create_turn(session, chat["id"], request, activate_branch=False)
            session.rollback()
        with SessionLocal() as session:
            assert (
                session.scalar(
                    select(func.count()).select_from(Run).where(Run.chat_id == chat["id"])
                )
                == 0
            )
