from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.adapters.base import ChatRequest
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Run


async def wait_for_assistant(client: AsyncClient, chat_id: str) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/chats/{chat_id}")
        assert response.status_code == 200
        assistants: list[dict[str, Any]] = [
            message for message in response.json()["messages"] if message["role"] == "assistant"
        ]
        if assistants and assistants[-1]["status"] in {"complete", "failed", "cancelled"}:
            assert assistants[-1]["status"] == "complete", assistants[-1]
            return assistants[-1]
        await asyncio.sleep(0.03)
    raise AssertionError("assistant run did not complete")


async def wait_for_run(client: AsyncClient, run_id: str) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run: dict[str, Any] = response.json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            assert run["status"] == "complete", run
            return run
        await asyncio.sleep(0.03)
    raise AssertionError("run did not complete")


async def create_offer(
    client: AsyncClient, *, multiple: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Multiple offer" if multiple else "Single offer"},
        )
    ).json()
    prompt = (
        "Please offer two image prompts for blue cups."
        if multiple
        else "Please offer an image prompt for a blue cup."
    )
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": prompt, "mode": "text"},
    )
    assert accepted.status_code == 202
    await wait_for_run(client, accepted.json()["run"]["id"])
    assistant = await wait_for_assistant(client, chat["id"])
    metadata = next(
        part["metadata_json"]
        for part in assistant["parts"]
        if part["type"] == "generation_metadata"
    )
    offer = metadata["generation_offer"]
    assert offer["version"] == "generation-offer-v1"
    assert len(offer["items"]) == (2 if multiple else 1)
    return chat, assistant


async def test_primary_chat_stream_stays_tool_free_and_offer_extraction_is_isolated(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = app.state.services.engines.chat
    original_stream = adapter.stream
    requests: list[ChatRequest] = []

    async def recording_stream(request: ChatRequest):  # type: ignore[no-untyped-def]
        requests.append(request)
        async for event in original_stream(request):
            yield event

    monkeypatch.setattr(adapter, "stream", recording_stream)
    await create_offer(client, multiple=False)

    assert len(requests) == 2
    assert requests[0].tools == []
    assert requests[1].tools[0]["function"]["name"] == "offer_generation"


async def test_offer_is_inert_until_single_prompt_assent_is_confirmed(
    client: AsyncClient,
) -> None:
    chat, _assistant = await create_offer(client, multiple=False)
    with SessionLocal() as session:
        assert [run.operation for run in session.scalars(select(Run)).all()] == ["text"]
        assert session.scalars(select(Artifact)).all() == []

    payload = {
        "text": "Yes, please",
        "mode": "text",
        "settings": {"max_tokens": 64},
        "idempotency_key": "accept-single-offer",
    }
    preview = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert preview.status_code == 409
    detail = preview.json()["detail"]
    assert detail["code"] == "route_confirmation_required"
    assert detail["plan"]["operation"] == "text_to_image"
    assert detail["plan"]["standalone_prompt"] == "Mock image prompt 1."
    assert len((await client.get(f"/api/chats/{chat['id']}")).json()["messages"]) == 2

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={**payload, "confirm_media": True},
    )
    assert accepted.status_code == 202
    run = await wait_for_run(client, accepted.json()["run"]["id"])
    assert run["operation"] == "text_to_image"
    assert run["standalone_prompt"] == "Mock image prompt 1."


async def test_multiple_prompt_assent_uses_ordered_confirmation_and_preserves_prompts(
    client: AsyncClient,
) -> None:
    chat, _assistant = await create_offer(client, multiple=True)
    payload = {
        "text": "Generate them",
        "mode": "text",
        "settings": {"max_tokens": 64},
        "idempotency_key": "accept-multiple-offer",
    }
    preview = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert preview.status_code == 409
    detail = preview.json()["detail"]
    assert detail["code"] == "ordered_plan_confirmation_required"
    assert [step["prompt"] for step in detail["plan"]["steps"]] == [
        "Mock image prompt 1.",
        "Mock image prompt 2.",
    ]
    assert [step["depends_on"] for step in detail["plan"]["steps"]] == [
        [],
        ["offer_1"],
    ]
    assert len((await client.get(f"/api/chats/{chat['id']}")).json()["messages"]) == 2

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={**payload, "confirm_media": True},
    )
    assert accepted.status_code == 202
    plan_id = accepted.json()["run"]["work_plan_id"]
    plan = (await client.get(f"/api/work-plans/{plan_id}")).json()
    assert [step["operation"] for step in plan["steps"]] == [
        "text_to_image",
        "text_to_image",
    ]
    assert [step["prompt"] for step in plan["steps"]] == [
        "Mock image prompt 1.",
        "Mock image prompt 2.",
    ]
    for run_id in plan["summary_json"]["run_ids"]:
        await wait_for_run(client, run_id)


@pytest.mark.parametrize("reply", ["No thanks", "What time is it?"])
async def test_refusal_or_unrelated_reply_does_not_accept_offer(
    client: AsyncClient,
    reply: str,
) -> None:
    chat, _assistant = await create_offer(client, multiple=False)
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": reply, "mode": "auto"},
    )
    assert response.status_code == 202
    run = await wait_for_run(client, response.json()["run"]["id"])
    assert run["operation"] == "text"
    with SessionLocal() as session:
        operations = [
            item.operation
            for item in session.scalars(
                select(Run).where(Run.chat_id == chat["id"]).order_by(Run.created_at)
            ).all()
        ]
        assert operations == ["text", "text"]
        assert session.scalars(select(Artifact)).all() == []
