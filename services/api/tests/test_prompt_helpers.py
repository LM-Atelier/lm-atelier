from __future__ import annotations

import asyncio

from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.adapters.base import ChatRequest
from local_lm.db import SessionLocal
from local_lm.models import Artifact, ArtifactLibraryEntry, Chat, Job, Run, WorkPlan
from local_lm.prompt_helpers import (
    PROMPT_HELPER_SCOPE,
    STANDARD_CHAT_SCOPE,
    prompt_preview_settings,
)
from local_lm.schemas import SettingField


def test_prompt_preview_settings_are_capability_bounded() -> None:
    def field(
        key: str,
        default: int,
        minimum: int | None,
        maximum: int | None,
        **overrides: object,
    ) -> SettingField:
        values: dict[str, object] = {
            "key": key,
            "label": key,
            "type": "integer",
            "default": default,
            "minimum": minimum,
            "maximum": maximum,
            "scope": "workflow",
        }
        values.update(overrides)
        return SettingField.model_validate(values)

    assert prompt_preview_settings(
        [
            field("width", 1024, 256, 2048, multiple_of=64),
            field("height", 384, 256, 2048, multiple_of=64),
            field("steps", 30, 12, 100),
            field("num_frames", 49, 1, 81),
            field("duration_seconds", 6, 1, 10),
            field("batch_size", 4, 1, 8),
            field("seed", -1, -1, 2**31),
            field("image_width", 1024, 256, 2048, scope="load"),
            field("output_height", 1024, 256, 2048, available=False),
        ]
    ) == {
        "width": 512,
        "height": 384,
        "steps": 12,
        "num_frames": 16,
        "duration_seconds": 2,
        "batch_size": 1,
    }


async def wait_for_run(client: AsyncClient, run_id: str) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            assert run["status"] == "complete", run
            return run
        await asyncio.sleep(0.03)
    raise AssertionError("helper run did not complete")


async def create_helper(client: AsyncClient, draft: str = "A blue ceramic cup") -> dict:  # type: ignore[type-arg]
    source = (await client.post("/api/chats", json={"title": "Source chat"})).json()
    response = await client.post(
        "/api/prompt-helpers",
        json={"source_chat_id": source["id"], "draft_prompt": draft},
    )
    assert response.status_code == 201
    return response.json()


async def test_prompt_helper_lifecycle_is_hidden_and_bounded(client: AsyncClient) -> None:
    helper = await create_helper(client)
    assert helper["archived"] is True
    assert helper["project_id"] is None
    assert helper["draft_prompt"] == "A blue ceramic cup"

    visible = (await client.get("/api/chats?include_archived=true")).json()
    assert all(chat["id"] != helper["id"] for chat in visible)
    assert (await client.get(f"/api/chats/{helper['id']}")).status_code == 404
    assert (
        await client.patch(f"/api/chats/{helper['id']}", json={"archived": False})
    ).status_code == 404
    assert (await client.delete(f"/api/chats/{helper['id']}")).status_code == 404

    updated = await client.patch(
        f"/api/prompt-helpers/{helper['id']}",
        json={"draft_prompt": "A blue cup on a linen table"},
    )
    assert updated.status_code == 200
    assert updated.json()["draft_prompt"] == "A blue cup on a linen table"

    with SessionLocal() as session:
        row = session.get(Chat, helper["id"])
        assert row is not None
        assert row.scope == PROMPT_HELPER_SCOPE
        assert (
            session.scalar(
                select(Chat).where(Chat.scope == STANDARD_CHAT_SCOPE, Chat.id == helper["id"])
            )
            is None
        )

    deleted = await client.delete(f"/api/prompt-helpers/{helper['id']}")
    assert deleted.status_code == 204
    assert (await client.get(f"/api/prompt-helpers/{helper['id']}")).status_code == 404


async def test_prompt_helper_uses_isolated_system_context_and_normal_chat_queue(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    source = (await client.post("/api/chats", json={"title": "Source chat"})).json()
    source_turn = await client.post(
        f"/api/chats/{source['id']}/turns",
        json={"text": "Synthetic source-only phrase", "mode": "text"},
    )
    await wait_for_run(client, source_turn.json()["run"]["id"])

    helper = (
        await client.post(
            "/api/prompt-helpers",
            json={"source_chat_id": source["id"], "draft_prompt": "A small blue cup"},
        )
    ).json()
    adapter = app.state.services.engines.chat
    original_stream = adapter.stream
    requests: list[ChatRequest] = []

    async def recording_stream(request: ChatRequest):  # type: ignore[no-untyped-def]
        requests.append(request)
        async for event in original_stream(request):
            yield event

    monkeypatch.setattr(adapter, "stream", recording_stream)
    response = await client.post(
        f"/api/chats/{helper['id']}/turns",
        json={"text": "Make the lighting cinematic", "mode": "text"},
    )
    assert response.status_code == 202
    accepted = response.json()
    await wait_for_run(client, accepted["run"]["id"])

    assert len(requests) == 1
    assert requests[0].messages[0]["role"] == "system"
    assert "LM Atelier's prompt workshop" in requests[0].messages[0]["content"]
    assert "A small blue cup" in requests[0].messages[0]["content"]
    assert "Synthetic source-only phrase" not in str(requests[0].messages)
    assert requests[0].messages[-1] == {
        "role": "user",
        "content": "Make the lighting cinematic",
    }
    with SessionLocal() as session:
        run = session.get(Run, accepted["run"]["id"])
        assert run is not None
        plan = session.get(WorkPlan, run.work_plan_id)
        job = session.scalar(select(Job).where(Job.run_id == run.id))
        assert plan is not None
        assert plan.persistence_scope == "durable"
        assert job is not None
        assert job.queue_resource == "interactive_compute"


async def test_prompt_helper_preview_uses_media_queue_and_cleanup(client: AsyncClient) -> None:
    helper = await create_helper(client, "A copper kettle")
    response = await client.post(
        f"/api/chats/{helper['id']}/turns",
        json={
            "text": "A copper kettle",
            "mode": "image",
            "settings": {"steps": 8},
        },
    )
    assert response.status_code == 202
    accepted = response.json()
    await wait_for_run(client, accepted["run"]["id"])

    with SessionLocal() as session:
        run = session.get(Run, accepted["run"]["id"])
        assert run is not None
        assert run.operation == "text_to_image"
        job = session.scalar(select(Job).where(Job.run_id == run.id))
        assert job is not None
        assert job.queue_resource == "media_compute"
        assert session.scalars(select(Artifact)).all()
        assert session.scalars(select(ArtifactLibraryEntry)).all() == []

    assert (await client.delete(f"/api/prompt-helpers/{helper['id']}")).status_code == 204
    with SessionLocal() as session:
        assert session.get(Chat, helper["id"]) is None
        assert session.scalars(select(Artifact)).all() == []
        assert session.scalars(select(Job)).all() == []
