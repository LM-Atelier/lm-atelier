"""Accepted picture purposes survive generation records and actual replay dispatch."""

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_image_role_execution import selected_images, selected_workflow
from test_output_recipe_replay import (
    _edit_chat,
    _finished,
    _new_chat,
    _profile,
    _record_of,
    _replay,
)
from test_output_recipe_replay_plan import _plan

from local_lm.accepted_turn_context import accepted_context
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.db import SessionLocal
from local_lm.model_planner import workflow_artifact_contract
from local_lm.models import Run, WorkflowDefinition, WorkflowRevision
from local_lm.output_recipe_v1 import open_output_recipe
from local_lm.schemas import EngineCapabilities


@pytest.mark.parametrize("mode", ["image", "video"])
@pytest.mark.parametrize("selection", ["canvas", "references", "empty"])
async def test_selected_purposes_are_recorded_and_replayed_through_the_same_slots(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    selection: str,
) -> None:
    ids = [] if selection == "empty" else await selected_images(client)
    roles = (
        ["reference", "edit_source", "reference"]
        if selection == "canvas"
        else ["reference"] * len(ids)
    )
    slot_roles = (
        ("edit_source", "reference", "reference")
        if selection == "canvas"
        else ("reference",) * len(ids)
    )
    operation = ("image_to_" if selection == "canvas" else "text_to_") + mode
    revision_id = selected_workflow(slot_roles)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        definition.operation = operation
        revision.artifact_sha256 = workflow_artifact_contract(
            operation=operation,
            engine=revision.engine,
            api_graph=revision.api_graph_json,
            input_schema=revision.input_schema_json,
            dependencies=revision.dependencies_json,
        )
        session.commit()

    media = app.state.services.engines.media
    base_capabilities: EngineCapabilities = await media.capabilities()

    async def capabilities() -> EngineCapabilities:
        return base_capabilities.model_copy(update={"image_slot_bindings": True})

    monkeypatch.setattr(media, "capabilities", capabilities)
    original_generate = media.generate
    requests: list[MediaRequest] = []
    compiled: list[list[str]] = []
    adapter = ComfyUIAdapter("http://127.0.0.1:1")

    async def upload(request: MediaRequest) -> list[str]:
        return [f"selected-{index}.png" for index in range(len(request.input_paths))]

    monkeypatch.setattr(adapter, "_upload_inputs", upload)

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        requests.append(request)
        parameters = await adapter._request_parameters(request)
        graph = adapter._compile(request.workflow, parameters)
        compiled.append([graph[str(index + 10)]["inputs"]["image"] for index in range(len(ids))])
        async for event in original_generate(request):
            yield event

    monkeypatch.setattr(media, "generate", generate)
    try:
        chat_id = await _edit_chat(client)
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": "Paint a blue garden",
                "mode": mode,
                "profile_id": _profile(mode),
                "workflow_revision_id": revision_id,
                "input_artifact_ids": ids,
                "input_image_roles": roles,
                "settings": {"seed": 1234},
            },
        )
        assert response.status_code == 202, response.text
        original = await _finished(client, response.json()["run"]["id"])
        with SessionLocal() as session:
            stored = session.get(Run, original["id"])
            assert stored is not None
            frozen = accepted_context(session, stored)
            assert frozen is not None
            assert frozen.input_artifact_ids == ids
            assert frozen.input_image_roles == roles
            stored.provenance_json = {
                **stored.provenance_json,
                "input_artifact_ids": list(reversed(ids)),
                "input_image_roles": ["reference"] * len(ids),
            }
            session.commit()

        content = await _record_of(client, original)
        record = open_output_recipe(content)
        assert record["version"] == 2
        assert record["operation"] == operation
        assert [item["sha256"] for item in record["inputs"]] == [
            identity.removeprefix("sha256:") for identity in ids
        ]
        assert [item["role"] for item in record["inputs"]] == roles
        plan = await _plan(client, content)
        assert plan["ready"] is True, plan["refusals"]
        assert plan["resolved"]["input_artifact_ids"] == ids
        assert plan["resolved"]["input_image_roles"] == roles

        replayed = await _replay(client, await _new_chat(client), content)
        assert replayed.status_code == 202, replayed.text
        again = await _finished(client, replayed.json()["run"]["id"])
        repeated_record = open_output_recipe(await _record_of(client, again))
        assert repeated_record["version"] == 2
        for section in ("operation", "prompt", "seed", "settings", "inputs", "model", "loras"):
            assert repeated_record[section] == record[section], section
        with SessionLocal() as session:
            stored = session.get(Run, again["id"])
            assert stored is not None
            frozen = accepted_context(session, stored)
            assert frozen is not None
            assert frozen.input_artifact_ids == ids
            assert frozen.input_image_roles == roles
        assert len(requests) == 2
        assert requests[0].input_paths == requests[1].input_paths
        expected: dict[str, tuple[int, ...]] = {
            f"input_image_{index}": (position,)
            for index, position in enumerate(
                [1, 0, 2] if selection == "canvas" else range(len(ids))
            )
        }
        assert [request.input_image_bindings for request in requests] == [expected, expected]
        names = [
            f"selected-{position}.png"
            for position in ([1, 0, 2] if selection == "canvas" else range(len(ids)))
        ]
        assert compiled == [names, names]
    finally:
        await adapter.close()
