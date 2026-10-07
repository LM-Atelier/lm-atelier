"""Preview the actual enlargement selection without starting or queuing work."""

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_workflow_recipe_request_cases import _picture, _workflow

from local_lm.db import SessionLocal
from local_lm.models import Chat, Job, Message, Run, RunContextSnapshot, WorkflowRevision, WorkPlan


async def _selection(client: AsyncClient, *, adjustable: bool) -> tuple[str, str, dict[str, Any]]:
    family_id, revision_id = _workflow("image_upscale", "image_to_image", "image")
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        factor = (
            {"type": "number", "default": 2, "minimum": 1, "maximum": 4, "multipleOf": 0.5}
            if adjustable
            else {"type": "number", "readOnly": True}
        )
        revision.input_schema_json = {
            "type": "object",
            "properties": {"upscale_factor": {**factor, "x-lm-atelier-kind": "upscale"}},
        }
        revision.api_graph_json = {
            "source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
            "scale": {
                "class_type": "ImageScaleBy",
                "inputs": {
                    "image": ["source", 0],
                    "scale_by": "${upscale_factor}" if adjustable else 4,
                },
            },
            "output": {"class_type": "SaveImage", "inputs": {"images": ["scale", 0]}},
        }
        session.commit()
    chat = (await client.post("/api/chats", json={"title": "Enlargement preview"})).json()
    choice = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": family_id},
    )
    assert choice.status_code == 200, choice.text
    return (
        chat["id"],
        revision_id,
        {
            "text": "Enlarge this picture",
            "mode": "image",
            "upscale": True,
            "input_artifact_ids": [await _picture(client)],
        },
    )


def _writes(chat_id: str) -> tuple[Any, ...]:
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        counts = tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (Job, Run, Message, WorkPlan, RunContextSnapshot)
        )
        return *counts, chat.active_head_message_id, chat.updated_at


@pytest.mark.parametrize("adjustable", [False, True])
async def test_preview_is_read_only_and_apply_uses_its_exact_selection(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, adjustable: bool
) -> None:
    chat_id, revision_id, payload = await _selection(client, adjustable=adjustable)
    before = _writes(chat_id)

    async def no_probe(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("A preview must not probe or invoke an engine")

    with monkeypatch.context() as isolated:
        isolated.setattr(app.state.services.engines, "settings_for_role", no_probe)
        isolated.setattr(app.state.services.orchestrator, "_chat_planner_available", no_probe)
        isolated.setattr(app.state.services.orchestrator, "_compiled_visual_prompt", no_probe)
        preview = await client.post(f"/api/chats/{chat_id}/upscale/preview", json=payload)
    assert preview.status_code == 200, preview.text
    result = preview.json()
    assert result["version"] == 1 and result["status"] == "ready"
    assert result["workflow_revision_id"] == revision_id
    assert result["request_authorized"] is False
    assert _writes(chat_id) == before
    if adjustable:
        assert result["factor"]["maximum"] == 4
        assert result["factor"]["multiple_of"] == 0.5
        assert result["fixed_factor"] is None
    else:
        assert result["factor"] is None
        assert result["fixed_factor"] == 4
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                **payload,
                "workflow_revision_id": revision_id,
                "settings": {"upscale_factor": 3} if adjustable else {},
            },
        )
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["run"]["workflow_revision_id"] == revision_id
        if adjustable:
            assert accepted.json()["run"]["settings_json"]["upscale_factor"] == 3
        else:
            assert "upscale_factor" not in accepted.json()["run"]["settings_json"]


async def test_preview_uses_the_chat_recipe_for_the_slider_start(client: AsyncClient) -> None:
    chat_id, _, payload = await _selection(client, adjustable=True)
    recipe = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Enlargement",
            "use_case": "image_upscale",
            "settings_json": {"upscale_factor": 3},
        },
    )
    assert recipe.status_code == 201, recipe.text
    selected = await client.put(
        f"/api/chats/{chat_id}/workflow-use-case-presets/image_upscale",
        json={"mode": "preset", "preset_id": recipe.json()["id"]},
    )
    assert selected.status_code == 200, selected.text
    preview = await client.post(f"/api/chats/{chat_id}/upscale/preview", json=payload)
    assert preview.status_code == 200, preview.text
    assert preview.json()["factor"]["default"] == 3


@pytest.mark.parametrize("problem", ["untrusted", "dependencies", "multiple"])
async def test_unavailable_preview_returns_guidance_without_queuing_work(
    client: AsyncClient, problem: str
) -> None:
    chat_id, revision_id, payload = await _selection(client, adjustable=False)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        if problem == "untrusted":
            revision.trusted = False
        elif problem == "dependencies":
            revision.dependency_contract_sha256 = "a" * 64
        else:
            payload["output_count"] = 2
        session.commit()
    before = _writes(chat_id)
    response = await client.post(f"/api/chats/{chat_id}/upscale/preview", json=payload)
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "upscale-preview-unavailable"
    assert _writes(chat_id) == before


async def test_apply_requires_a_new_preview_when_the_selected_revision_loses_trust(
    client: AsyncClient,
) -> None:
    chat_id, revision_id, payload = await _selection(client, adjustable=True)
    preview = await client.post(f"/api/chats/{chat_id}/upscale/preview", json=payload)
    assert preview.status_code == 200, preview.text
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.trusted = False
        session.commit()
    before = _writes(chat_id)
    response = await client.post(
        f"/api/chats/{chat_id}/turns", json={**payload, "workflow_revision_id": revision_id}
    )
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "upscale-selection-unavailable"
    assert _writes(chat_id) == before
