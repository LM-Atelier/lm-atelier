"""Requests waiting on admission cannot revive or disclose a deleted conversation."""

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_chat_deletion import wait_for_run
from test_chat_recovery import _command, _history, _impact

from local_lm.db import SessionLocal
from local_lm.models import Chat, Run
from local_lm.schemas import TurnRequest


async def _completed(client: AsyncClient) -> tuple[str, str]:
    created = await client.post("/api/chats", json={"title": "Garden admission"})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    accepted = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={
            "text": "Keep the garden layout",
            "mode": "text",
            "idempotency_key": "completed-garden",
        },
    )
    assert accepted.status_code == 202
    run_id = accepted.json()["run"]["id"]
    await wait_for_run(client, run_id)
    return chat_id, run_id


async def _trash(client: AsyncClient, chat_id: str) -> None:
    preview = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    response = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(preview, "trash-admission-garden")
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("key", [None, "completed-garden"])
async def test_admission_rechecks_deletion_even_with_a_cached_chat_and_completed_run(
    app: FastAPI, client: AsyncClient, key: str | None
) -> None:
    chat_id, run_id = await _completed(client)
    with SessionLocal() as session:
        cached = session.get(Chat, chat_id)
        assert cached is not None
        await _trash(client, chat_id)
        before = _history(chat_id)
        with pytest.raises(LookupError, match="^chat not found$"):
            await app.state.services.orchestrator.create_turn(
                session,
                chat_id,
                TurnRequest(text="Keep the garden layout", mode="text", idempotency_key=key),
            )
        session.rollback()
        assert session.get(Chat, chat_id) is cached
        assert session.get(Run, run_id) is not None
    assert _history(chat_id) == before


async def test_a_request_that_passed_the_api_check_cannot_replay_after_trash_commits(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, _run_id = await _completed(client)
    entered, release = asyncio.Event(), asyncio.Event()
    original = app.state.services.orchestrator.create_turn

    async def wait_before_lock(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(app.state.services.orchestrator, "create_turn", wait_before_lock)
    request = asyncio.create_task(
        client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": "Keep the garden layout",
                "mode": "text",
                "idempotency_key": "completed-garden",
            },
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await _trash(client, chat_id)
        before = _history(chat_id)
    finally:
        release.set()
    response = await request
    assert response.status_code == 404, response.text
    assert response.json()["code"] == "turn-subject-not-found"
    assert _history(chat_id) == before


async def test_prompt_batch_admission_refuses_a_deleted_chat_before_resolving_any_batch(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id, _run_id = await _completed(client)
    await _trash(client, chat_id)
    before = _history(chat_id)
    with SessionLocal() as session, pytest.raises(LookupError, match="^chat not found$"):
        await app.state.services.orchestrator.create_prompt_batch_turn(
            session,
            chat_id,
            "batch-garden",
            queue_idempotency_key="queue-garden",
            expected_plan_version=1,
            expected_plan_sha256="0" * 64,
        )
    assert _history(chat_id) == before


@pytest.mark.parametrize("preview", ["source_fit", "upscale"])
async def test_preview_refuses_a_deleted_chat_before_resolving_its_source(
    app: FastAPI, client: AsyncClient, preview: str
) -> None:
    chat_id, _run_id = await _completed(client)
    await _trash(client, chat_id)
    before = _history(chat_id)
    request = TurnRequest(
        text="Fit the garden picture",
        mode="image",
        input_artifact_ids=["sha256:" + "0" * 64],
        source_fit={"mode": "extend", "width": 16, "height": 16}
        if preview == "source_fit"
        else None,
        upscale=preview == "upscale",
    )
    with SessionLocal() as session, pytest.raises(LookupError, match="^chat not found$"):
        if preview == "upscale":
            app.state.services.orchestrator.preview_turn_upscale(session, chat_id, request)
        else:
            await app.state.services.orchestrator.preview_turn_source_fit(session, chat_id, request)
    assert _history(chat_id) == before
