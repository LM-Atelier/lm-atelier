"""A text reply stopped while it waits its turn, then retried, runs to its answer.

The llama.cpp engine keeps a stop for a reply it is not writing yet and applies
it to that reply's next start, as ComfyUI does for a prompt. Stopping a reply
that was still queued told the engine all the same, so the retry, which is the
same reply, began by ending at once with nothing written about it: its job
stayed running with no owner and the turn never answered. Only a reply the
engine is writing is stopped in the engine now; one still waiting ends when its
execution is cancelled.

The engine here is the mock chat engine, keeping and applying stops the way
the llama.cpp engine does.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS, wait_for_terminal_status
from test_generation_retry import read_run
from test_media_worker_replacement import _execution_ended, _job_id

from local_lm.adapters.base import ChatEvent, ChatRequest

STOPS = [
    pytest.param(lambda _chat_id, job_id: f"/api/jobs/{job_id}/cancel", id="stop-the-reply"),
    pytest.param(lambda chat_id, _job_id: f"/api/chats/{chat_id}/cancel-all", id="cancel-all"),
]


async def _ended(client: AsyncClient, run_id: str, expected: str = "complete") -> dict[str, Any]:
    """The text reply's run once it has ended, as `expected`."""

    return dict(
        await wait_for_terminal_status(
            lambda: read_run(client, run_id), what="the text reply", expected=expected
        )
    )


async def _reply(client: AsyncClient) -> tuple[str, str]:
    """Ask for a text reply in a new chat; (chat, run)."""

    chat = (await client.post("/api/chats", json={"title": "Stops"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Name three colors.", "mode": "text"}
    )
    assert turn.status_code == 202, turn.text
    return chat["id"], turn.json()["run"]["id"]


def _keeps_stops_for_replies_it_is_not_writing(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Make the chat engine keep a stop for a reply it is not writing, as llama.cpp does.

    Returns the run ids the engine was told to stop, in order.
    """

    engine = app.state.services.engines.chat
    original = engine.stream
    writing: set[str] = set()
    kept: set[str] = set()
    told: list[str] = []

    async def cancel(run_id: str) -> None:
        told.append(run_id)
        if run_id not in writing:
            kept.add(run_id)

    async def stream(request: ChatRequest) -> AsyncIterator[ChatEvent]:
        if request.run_id in kept:
            kept.discard(request.run_id)
            yield ChatEvent(type="cancelled")
            return
        writing.add(request.run_id)
        try:
            async for event in original(request):
                yield event
        finally:
            writing.discard(request.run_id)

    monkeypatch.setattr(engine, "cancel", cancel)
    monkeypatch.setattr(engine, "stream", stream)
    return told


@pytest.mark.parametrize("stop", STOPS)
async def test_a_text_reply_stopped_while_queued_can_be_retried(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    stop: Callable[[str, str], str],
) -> None:
    told = _keeps_stops_for_replies_it_is_not_writing(app, monkeypatch)
    services = app.state.services

    # Another job holds the device, so the reply is still queued when it is stopped.
    async with services.scheduler.lease("primary"):
        chat_id, run_id = await _reply(client)
        job_id = _job_id(run_id)
        stopped = await client.post(stop(chat_id, job_id))
        assert stopped.status_code in (200, 204), stopped.text
    await _ended(client, run_id, "cancelled")
    await _execution_ended(services.orchestrator, job_id)

    retried = await client.post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200, retried.text
    run = await _ended(client, run_id)

    assert run["status"] == "complete"
    # It was never being written, so the engine had nothing to stop.
    assert told == []


async def test_a_reply_being_written_is_still_stopped_in_the_engine(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    told = _keeps_stops_for_replies_it_is_not_writing(app, monkeypatch)
    engine = app.state.services.engines.chat
    keeps_stops = engine.stream
    writing = asyncio.Event()

    async def writes_until_stopped(request: ChatRequest) -> AsyncIterator[ChatEvent]:
        async for event in keeps_stops(request):
            if event.type == "delta":
                yield event
                writing.set()
                await asyncio.Event().wait()
            yield event

    monkeypatch.setattr(engine, "stream", writes_until_stopped)
    _chat_id, run_id = await _reply(client)
    await asyncio.wait_for(writing.wait(), timeout=PATIENCE_SECONDS)

    stopped = await client.post(f"/api/jobs/{_job_id(run_id)}/cancel")
    assert stopped.status_code == 200, stopped.text
    await _ended(client, run_id, "cancelled")

    assert told == [run_id]


@pytest.mark.parametrize("stop", STOPS)
async def test_a_reply_its_engine_ends_once_asked_to_stop_reads_as_stopped(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    stop: Callable[[str, str], str],
) -> None:
    """The engine's stop takes a moment, long enough for the reply to see it end."""

    engine = app.state.services.engines.chat
    writing = asyncio.Event()
    ends: dict[str, asyncio.Event] = {}
    told: list[str] = []

    async def stream(request: ChatRequest) -> AsyncIterator[ChatEvent]:
        end = ends.setdefault(request.run_id, asyncio.Event())
        yield ChatEvent(type="delta", text="Red, ")
        writing.set()
        await end.wait()
        yield ChatEvent(type="cancelled")

    async def cancel(run_id: str) -> None:
        # As a stop sent over HTTP does, it ends the reply and then takes a moment.
        told.append(run_id)
        ends.setdefault(run_id, asyncio.Event()).set()
        await asyncio.sleep(0.05)

    monkeypatch.setattr(engine, "stream", stream)
    monkeypatch.setattr(engine, "cancel", cancel)
    chat_id, run_id = await _reply(client)
    await asyncio.wait_for(writing.wait(), timeout=PATIENCE_SECONDS)

    stopped = await client.post(stop(chat_id, _job_id(run_id)))
    assert stopped.status_code in (200, 204), stopped.text
    run = await _ended(client, run_id, "cancelled")

    assert told == [run_id]
    assert run.get("error") is None


async def test_a_reply_the_engine_ends_without_being_asked_fails_with_its_reason(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing marked it stopped, so it must not read as running with no owner."""

    async def ends_at_once(_request: ChatRequest) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(type="cancelled")

    monkeypatch.setattr(app.state.services.engines.chat, "stream", ends_at_once)
    _chat_id, run_id = await _reply(client)

    run = await _ended(client, run_id, "failed")

    assert run["error"] == (
        "The chat model stopped this reply before it finished. Retry to run it again."
    )
