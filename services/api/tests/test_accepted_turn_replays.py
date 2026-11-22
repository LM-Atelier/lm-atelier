"""An accepted turn says whether it replays an earlier submission.

A turn route given an idempotency key it has already accepted in that chat
answers with the run the first submission created, and a sent turn is matched
by its key alone, so even a changed turn under a reused key gets that run.
`replayed` says which happened, so a client can tell a safe retry after a lost
response from a submission that started new work.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient, Response
from run_waits import PATIENCE_SECONDS, wait_for_terminal_status
from sqlalchemy import select
from sqlalchemy.orm import Session

from local_lm.db import SessionLocal
from local_lm.models import Run, TurnCreationClaim
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.schemas import TurnRequest


def _accepted(response: Response) -> dict[str, Any]:
    assert response.status_code == 202, response.text
    accepted: dict[str, Any] = response.json()
    return accepted


async def _complete(client: AsyncClient, run_id: str) -> None:
    async def read() -> dict[str, Any]:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        run: dict[str, Any] = response.json()
        return run

    await wait_for_terminal_status(read, what=f"run {run_id}")


async def _chat(client: AsyncClient, title: str) -> str:
    response = await client.post("/api/chats", json={"title": title})
    assert response.status_code == 201, response.text
    chat_id: str = response.json()["id"]
    return chat_id


async def _completed_turn(client: AsyncClient, chat_id: str) -> dict[str, Any]:
    source = _accepted(
        await client.post(
            f"/api/chats/{chat_id}/turns", json={"text": "Name a paper boat", "mode": "text"}
        )
    )
    await _complete(client, source["run"]["id"])
    return source


def _assert_only_the_reused_key_replays(
    first: dict[str, Any], again: dict[str, Any], other: dict[str, Any]
) -> None:
    replayed = [answer.get("replayed") for answer in (first, again, other)]
    assert replayed == [False, True, False]
    for part in ("run", "user_message", "assistant_message"):
        assert again[part]["id"] == first[part]["id"], part
    assert other["run"]["id"] != first["run"]["id"]


async def test_a_reused_turn_key_answers_with_the_first_turn_as_a_replay(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client, "Replayed turns")
    url = f"/api/chats/{chat_id}/turns"
    turn = {"text": "Name a paper boat", "mode": "text", "idempotency_key": "turn-once"}
    first = _accepted(await client.post(url, json=turn))
    # A changed request under the same key is answered with the first turn too.
    changed = {**turn, "text": "Name a paper kite"}
    again = _accepted(await client.post(url, json=changed))
    other = _accepted(await client.post(url, json={**changed, "idempotency_key": "turn-twice"}))
    unkeyed = _accepted(await client.post(url, json={"text": "Name a paper kite", "mode": "text"}))

    _assert_only_the_reused_key_replays(first, again, other)
    assert again["user_message"]["parts"][0]["text"] == "Name a paper boat"
    assert unkeyed.get("replayed") is False


async def test_a_reused_ordered_turn_key_answers_as_a_replay(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id = await _chat(client, "Replayed ordered turn")
    url = f"/api/chats/{chat_id}/turns"
    turn = {
        "text": "Write a short story about a paper boat, then create an image based on it",
        "mode": "auto",
        "confirm_media": True,
        "idempotency_key": "ordered-once",
    }
    async with app.state.services.scheduler.lease("primary"):
        first = _accepted(await client.post(url, json=turn))
        again = _accepted(await client.post(url, json=turn))
        other = _accepted(await client.post(url, json={**turn, "idempotency_key": "ordered-twice"}))

    _assert_only_the_reused_key_replays(first, again, other)
    plan = await client.get(f"/api/work-plans/{first['run']['work_plan_id']}")
    assert plan.status_code == 200, plan.text
    assert [step["operation"] for step in plan.json()["steps"]] == ["text", "text_to_image"]


async def test_a_reused_stop_and_send_key_answers_as_a_replay(client: AsyncClient) -> None:
    chat_id = await _chat(client, "Replayed stop and send")
    url = f"/api/chats/{chat_id}/stop-and-send"
    turn = {"text": "Name a paper boat", "mode": "text", "idempotency_key": "stop-once"}
    first = _accepted(await client.post(url, json=turn))
    # The first run finishes before the retry, so the retry has no current work to stop.
    await _complete(client, first["run"]["id"])
    again = _accepted(await client.post(url, json=turn))
    other = _accepted(await client.post(url, json={**turn, "idempotency_key": "stop-twice"}))

    _assert_only_the_reused_key_replays(first, again, other)


async def test_a_reused_branch_key_answers_as_a_replay(client: AsyncClient) -> None:
    chat_id = await _chat(client, "Replayed branch")
    source = await _completed_turn(client, chat_id)
    url = f"/api/messages/{source['user_message']['id']}/branch"
    turn = {"text": "Name a paper kite", "idempotency_key": "branch-once"}
    first = _accepted(await client.post(url, json=turn))
    again = _accepted(await client.post(url, json=turn))
    other = _accepted(await client.post(url, json={**turn, "idempotency_key": "branch-twice"}))

    _assert_only_the_reused_key_replays(first, again, other)


async def test_a_reused_regeneration_key_answers_as_a_replay(client: AsyncClient) -> None:
    chat_id = await _chat(client, "Replayed regeneration")
    source = await _completed_turn(client, chat_id)
    url = f"/api/messages/{source['assistant_message']['id']}/regenerate"
    request = {"settings": {}, "idempotency_key": "regenerate-once"}
    first = _accepted(await client.post(url, json=request))
    again = _accepted(await client.post(url, json=request))
    # A response takes one regeneration at a time, so a new key waits for the first.
    await _complete(client, first["run"]["id"])
    other = _accepted(
        await client.post(url, json={**request, "idempotency_key": "regenerate-twice"})
    )

    _assert_only_the_reused_key_replays(first, again, other)


async def test_a_reused_edit_key_answers_as_a_replay(client: AsyncClient) -> None:
    chat_id = await _chat(client, "Replayed edit")
    source = await _completed_turn(client, chat_id)
    url = f"/api/messages/{source['user_message']['id']}/edits"
    edit = {"text": "Name a paper kite", "idempotency_key": "edit-once"}
    first = _accepted(await client.post(url, json=edit))
    again = _accepted(await client.post(url, json=edit))
    other = _accepted(await client.post(url, json={**edit, "idempotency_key": "edit-twice"}))

    _assert_only_the_reused_key_replays(first, again, other)
    assert again["work_plan_id"] == first["work_plan_id"]
    assert other["work_plan_id"] != first["work_plan_id"]


async def test_a_turn_that_waited_for_another_instance_answers_as_a_replay(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id = await _chat(client, "Two instances")
    services = app.state.services
    creator: ConversationOrchestrator = services.orchestrator
    waiter = ConversationOrchestrator(
        services.engines,
        services.artifacts,
        services.events,
        services.scheduler,
        services.processes,
    )
    planning = asyncio.Event()
    release = asyncio.Event()
    plan = creator.router.plan_with_model

    async def held_plan(**kwargs: Any) -> Any:
        planning.set()
        await release.wait()
        return await plan(**kwargs)

    monkeypatch.setattr(creator.router, "plan_with_model", held_plan)
    monkeypatch.setattr(ConversationOrchestrator, "start", lambda *_args: None)
    request = TurnRequest(text="Name a paper boat", mode="text", idempotency_key="two-instances")
    with SessionLocal() as creating_session, SessionLocal() as waiting_session:
        creating = asyncio.create_task(creator.create_turn(creating_session, chat_id, request))
        await asyncio.wait_for(planning.wait(), timeout=PATIENCE_SECONDS)
        # The second instance finds no run yet and waits on the first one's claim.
        waiting = asyncio.create_task(waiter.create_turn(waiting_session, chat_id, request))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        release.set()
        created, replay = await asyncio.gather(creating, waiting)

    assert (created.replayed, replay.replayed) == (False, True)
    assert replay.run.id == created.run.id
    assert replay.assistant_message.id == created.assistant_message.id


async def test_a_turn_found_once_its_claim_is_taken_answers_as_a_replay(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id = await _chat(client, "Claimed replay")
    url = f"/api/chats/{chat_id}/turns"
    turn = {"text": "Name a paper boat", "mode": "text", "idempotency_key": "claimed-once"}
    first = _accepted(await client.post(url, json=turn))
    looks: list[bool] = []

    def visible_once_claimed(session: Session, chat: str, key: str) -> Run | None:
        # As when another instance commits the run after this request looked for
        # it and before this request took the claim: only the look made while
        # holding the claim finds it.
        claimed = session.scalar(
            select(TurnCreationClaim.id).where(
                TurnCreationClaim.chat_id == chat, TurnCreationClaim.idempotency_key == key
            )
        )
        looks.append(claimed is not None)
        return ConversationOrchestrator._idempotent_run(session, chat, key) if claimed else None

    monkeypatch.setattr(app.state.services.orchestrator, "_idempotent_run", visible_once_claimed)
    again = _accepted(await client.post(url, json=turn))

    assert (first.get("replayed"), again.get("replayed")) == (False, True)
    assert again["run"]["id"] == first["run"]["id"]
    assert looks[-1] and not any(looks[:-1])
    with SessionLocal() as session:
        claims = session.scalars(
            select(TurnCreationClaim).where(TurnCreationClaim.chat_id == chat_id)
        ).all()
    assert claims == []
