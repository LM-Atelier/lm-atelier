from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status, wait_until
from test_generation_retry import choose_retries, failing_media, read_run, start_media

from local_lm.adapters.base import ChatEvent, ChatRequest, MediaEvent, MediaRequest
from local_lm.scheduler import JobClaim
from local_lm.schemas import WorkerStatus


class ChatWorker:
    """A managed chat worker that the device handoff stops and restores."""

    def __init__(self, app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
        self.running = True
        self.order: list[str] = []
        services = app.state.services
        monkeypatch.setattr(services.processes, "statuses", self.statuses)
        monkeypatch.setattr(services.processes, "stop", self.stop)
        monkeypatch.setattr(services.orchestrator, "_resume_chat_worker", self.resume)

    def statuses(self) -> list[WorkerStatus]:
        return [
            WorkerStatus(
                name="chat",
                state="ready" if self.running else "stopped",
                managed=True,
                running=self.running,
                pid=11 if self.running else None,
                profile_id="profile-chat" if self.running else None,
            ),
            WorkerStatus(name="media", state="ready", managed=True, running=True, pid=22),
        ]

    async def stop(self, name: str) -> None:
        assert name == "chat"
        self.order.append("stop")
        self.running = False

    async def resume(self, profile_id: str) -> None:
        self.order.append("restore")
        self.running = True

    async def settled_order(self) -> list[str]:
        """The order once chat is running again; it comes back after a run reads complete."""

        async def order() -> list[str]:
            return list(self.order)

        return await wait_until(order, lambda _order: self.running, what="the chat model to return")


async def finished(client: AsyncClient, run_id: str, expected: str = "complete") -> dict[str, Any]:
    return dict(
        await wait_for_terminal_status(
            lambda: read_run(client, run_id),
            what="generation with a displaced chat model",
            expected=expected,
        )
    )


@pytest.mark.parametrize("mode", ["image", "video"])
async def test_a_displaced_chat_model_comes_back_once_after_a_pictures_retries(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    chat = ChatWorker(app, monkeypatch)
    attempts = failing_media(app, monkeypatch, 3)

    result = await finished(client, await start_media(client, mode))

    assert result["status"] == "complete"
    assert len(attempts) == 4
    assert result["provenance_json"]["failure_retries"]["used"] == 3
    # Chat goes down for the first attempt and comes back after the last one,
    # not between attempts that are about to be made again.
    assert await chat.settled_order() == ["stop", "restore"]


async def test_retries_that_all_fail_give_chat_back_once_after_the_last(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = ChatWorker(app, monkeypatch)
    attempts = failing_media(app, monkeypatch, 4)

    result = await finished(client, await start_media(client), "failed")

    assert result["status"] == "failed"
    assert len(attempts) == 4
    assert chat.order == ["stop", "restore"]
    assert chat.running


async def test_a_failure_that_is_not_retried_gives_chat_back_at_once(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await choose_retries(client, 0)
    chat = ChatWorker(app, monkeypatch)
    attempts = failing_media(app, monkeypatch, 1)

    result = await finished(client, await start_media(client), "failed")

    assert result["status"] == "failed"
    assert len(attempts) == 1
    assert chat.order == ["stop", "restore"]


async def test_a_text_turn_waiting_behind_a_failing_picture_gets_chat_back_before_it_runs(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = ChatWorker(app, monkeypatch)
    services = app.state.services
    picture_started = asyncio.Event()
    release_picture = asyncio.Event()
    attempts: list[MediaRequest] = []
    original_generate = services.engines.media.generate
    original_stream = services.engines.chat.stream

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        attempts.append(request)
        if len(attempts) == 1:
            picture_started.set()
            await release_picture.wait()
            raise RuntimeError("Temporary media engine failure")
        async for event in original_generate(request):
            yield event

    async def stream(request: ChatRequest) -> AsyncIterator[ChatEvent]:
        chat.order.append("text")
        async for event in original_stream(request):
            yield event

    monkeypatch.setattr(services.engines.media, "generate", generate)
    monkeypatch.setattr(services.engines.chat, "stream", stream)

    picture = await start_media(client)
    await asyncio.wait_for(picture_started.wait(), timeout=5)
    other = await client.post("/api/chats", json={"title": "Waiting question"})
    assert other.status_code == 201
    question = await client.post(
        f"/api/chats/{other.json()['id']}/turns",
        json={"text": "Name three primary colors.", "mode": "text"},
    )
    assert question.status_code == 202, question.text
    release_picture.set()

    assert (await finished(client, question.json()["run"]["id"]))["status"] == "complete"
    assert (await finished(client, picture))["status"] == "complete"
    assert len(attempts) == 2
    # The waiting question runs on a restored model; the retry then takes the
    # device again and gives it back when it is done.
    assert await chat.settled_order() == ["stop", "restore", "text", "stop", "restore"]


async def test_an_error_after_the_picture_completed_still_gives_chat_back(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = ChatWorker(app, monkeypatch)
    orchestrator = app.state.services.orchestrator
    execute_media = orchestrator._execute_media

    async def complete_then_fail(job_id: str, run_id: str, claim: JobClaim) -> str | None:
        await execute_media(job_id, run_id, claim)
        raise RuntimeError("Temporary failure after the picture was kept")

    monkeypatch.setattr(orchestrator, "_execute_media", complete_then_fail)

    result = await finished(client, await start_media(client))

    assert result["status"] == "complete"
    # Nothing retries a finished picture, so chat is not held down for one.
    assert await chat.settled_order() == ["stop", "restore"]
