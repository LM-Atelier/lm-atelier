from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.db import SessionLocal
from local_lm.models import Chat, Message, Run, WorkStep
from local_lm.orchestrator import ConversationOrchestrator


@pytest.mark.parametrize("cancel_source", [False, True])
async def test_ordered_edit_consumes_its_own_pending_producer_after_source_changes(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, cancel_source: bool
) -> None:
    created = await client.post("/api/chats", json={"title": "Independent ordered edits"})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("context.txt", b"Neutral background note", "text/plain")}
    )
    assert uploaded.status_code == 201
    engine = app.state.services.engines.chat
    original_stream = engine.stream
    seen: dict[str, list[dict[str, Any]]] = {}

    async def observe(request: ChatRequest) -> AsyncIterator[ChatEvent]:
        seen[request.run_id] = deepcopy(request.messages)
        async for event in original_stream(request):
            yield event

    monkeypatch.setattr(engine, "stream", observe)
    original_resolver = ConversationOrchestrator._resolve_step_inputs
    producer_output: list[str] = []

    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": "Write a short story about a paper boat, then summarize the story",
                "mode": "auto",
                "confirm_media": True,
                "input_artifact_ids": [uploaded.json()["id"]],
            },
        )
        assert source.status_code == 202, source.text
        source = source.json()
        loaded = await client.get(f"/api/messages/{source['user_message']['id']}/edit-source")
        assert loaded.status_code == 200, loaded.text
        view = loaded.json()
        assert view["plan_kind"] == "ordered" and len(view["steps"]) == 2
        edited = await client.post(
            f"/api/messages/{view['source_user_message_id']}/edits",
            json={
                "text": "Write a short story about a wooden kite, then summarize the story",
                "source_snapshot_sha256": view["source_snapshot_sha256"],
                "confirm_media": True,
                "idempotency_key": "independent-ordered-edit",
            },
        )
        assert edited.status_code == 202, edited.text
        from local_lm.accepted_turn_context import accepted_context

        with SessionLocal() as session:
            steps = list(
                session.scalars(
                    select(WorkStep)
                    .where(WorkStep.plan_id == edited.json()["work_plan_id"])
                    .order_by(WorkStep.ordinal)
                )
            )
            assert len(steps) == 2
            producer = session.get(Run, steps[0].run_id)
            consumer = session.get(Run, steps[1].run_id)
            assert producer is not None and consumer is not None
            assert producer.status == "queued" and consumer.status == "queued"
            consumer_id, producer_id = consumer.id, producer.id
            snapshot = accepted_context(session, consumer)
            assert snapshot is not None and len(snapshot.dependencies) == 1
            dependency = snapshot.dependencies[0]
            assert dependency.plan_id == edited.json()["work_plan_id"]
            assert dependency.step_id == steps[0].id
            assert dependency.run_id == producer.id
            assert dependency.message_id == producer.assistant_message_id
            assert dependency.run_id not in {step["source_run_id"] for step in view["steps"]}
            chat = session.get(Chat, chat_id)
            assert chat is not None
            active_head = chat.active_head_message_id

        def resolve_with_changed_binding(session: Any, run: Run) -> None:
            if run.id == consumer_id:
                producer = session.get(Run, producer_id)
                assert producer is not None and producer.status == "complete"
                message = session.get(Message, producer.assistant_message_id)
                assert message is not None
                producer_output.append("\n".join(part.text for part in message.parts if part.text))
                step = session.get(WorkStep, run.work_step_id)
                assert step is not None
                step.input_bindings_json = [
                    {"type": "step_output.text", "source_step_id": view["steps"][0]["step_id"]}
                ]
                session.flush()
            original_resolver(session, run)

        monkeypatch.setattr(
            ConversationOrchestrator,
            "_resolve_step_inputs",
            staticmethod(resolve_with_changed_binding),
        )
        if cancel_source:
            cancelled = await client.post(f"/api/work-plans/{source['run']['work_plan_id']}/cancel")
            assert cancelled.status_code == 200, cancelled.text

    for _ in range(200):
        with SessionLocal() as session:
            consumer = session.get(Run, consumer_id)
            assert consumer is not None
            if consumer.status in {"complete", "failed", "cancelled"}:
                assert consumer.status == "complete", consumer.error
                chat = session.get(Chat, chat_id)
                assert chat is not None and chat.active_head_message_id == active_head
                break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("The edited ordered consumer did not settle")

    assert len(producer_output) == 1 and "wooden kite" in producer_output[0]
    messages = seen[consumer_id]
    assert {"role": "assistant", "content": producer_output[0]} in messages
    assert all("paper boat" not in str(item.get("content", "")) for item in messages)
    if cancel_source:
        assert all(step["source_run_id"] not in seen for step in view["steps"])
