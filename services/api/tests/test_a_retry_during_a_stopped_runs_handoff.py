"""A retry pressed while a stopped render is still handing the device back runs once it is done.

Stop marks a media job cancelled at once, so the job shows Retry while its
execution is still ending: handing the device back, which stops ComfyUI, loads
the chat model the render displaced and starts ComfyUI again. A retry then
queues the job again, and its start finds the old execution still alive. It
used to do nothing at that point, and nothing started the job afterwards
either, so it stayed queued with no owner until the application restarted,
holding up the work queued behind it. Now it starts once the old execution has
ended, unless the chat's work is cancelled first.

The device handoff here is the supervisor's, recorded rather than made, and
loading the chat model again waits until the test lets it finish.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient, Response
from run_waits import PATIENCE_SECONDS, wait_until
from test_chat_handoff_across_retries import finished
from test_media_worker_replacement import ReplacementWorld, _image_turn, _job_id

from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import Job


def _job_state(job_id: str) -> tuple[str, str | None, int]:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        return job.status, job.claim_owner, job.attempt


@dataclass
class _Stopped:
    """A render stopped while it generates, whose execution is still handing the device back."""

    world: ReplacementWorld
    chat_id: str
    run_id: str
    job_id: str
    ending: asyncio.Task[None]
    stopping: asyncio.Task[Response]
    reloaded: asyncio.Event
    attempts: list[str]


async def _stopped_in_handoff(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    stop: Callable[[str, str], str],
) -> _Stopped:
    """Render until stopped through `stop`, and hold the displaced chat model's reload.

    The first attempt renders until it is stopped; any later one renders as the
    mock engine does. Stop answers once the stopped execution has ended, and
    the job shows as cancelled, with Retry, before that.
    """

    world = ReplacementWorld(app, settings, monkeypatch)
    world.scope = None
    media = world.services.engines.media
    original = media.generate
    generating = asyncio.Event()
    attempts: list[str] = []

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        attempts.append(request.run_id)
        if len(attempts) == 1:
            yield MediaEvent(type="progress", progress=0.1, phase="sampling")
            generating.set()
            await asyncio.Event().wait()
        async for event in original(request):
            yield event

    reloading = asyncio.Event()
    reloaded = asyncio.Event()

    async def reload_when_let(profile_id: str) -> None:
        reloading.set()
        await reloaded.wait()
        await world.resume(profile_id)

    monkeypatch.setattr(media, "generate", generate)
    monkeypatch.setattr(world.orchestrator, "_resume_chat_worker", reload_when_let)
    chat_id, run_id = await _image_turn(client)
    await asyncio.wait_for(generating.wait(), timeout=PATIENCE_SECONDS)
    job_id = _job_id(run_id)

    stopping = asyncio.create_task(client.post(stop(chat_id, job_id)))
    await asyncio.wait_for(reloading.wait(), timeout=PATIENCE_SECONDS)
    ending = world.orchestrator._tasks.get(job_id)
    assert ending is not None and not ending.done(), "the stopped execution has already ended"
    assert _job_state(job_id)[0] == "cancelled"
    return _Stopped(world, chat_id, run_id, job_id, ending, stopping, reloaded, attempts)


@pytest.mark.parametrize(
    "stop",
    [
        pytest.param(lambda _chat_id, job_id: f"/api/jobs/{job_id}/cancel", id="stop-the-job"),
        pytest.param(lambda chat_id, _job_id: f"/api/chats/{chat_id}/cancel", id="stop-the-chat"),
    ],
)
async def test_a_retry_pressed_while_a_stopped_render_hands_the_device_back_runs_after_it(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    stop: Callable[[str, str], str],
) -> None:
    stopped = await _stopped_in_handoff(app, client, settings, monkeypatch, stop)

    retried = await client.post(f"/api/jobs/{stopped.job_id}/retry")
    assert retried.status_code == 200, retried.text
    assert _job_state(stopped.job_id) == ("queued", None, 1)
    # Nothing starts beside the execution that is still ending.
    assert stopped.world.orchestrator._tasks.get(stopped.job_id) is stopped.ending

    stopped.reloaded.set()
    stop_answer = await stopped.stopping
    assert stop_answer.status_code == 200, stop_answer.text
    run = await finished(client, stopped.run_id)

    assert run["status"] == "complete"
    assert stopped.attempts == [stopped.run_id, stopped.run_id]

    async def settled() -> tuple[str, str | None, int]:
        return _job_state(stopped.job_id)

    state = await wait_until(settled, lambda state: state[0] == "complete", what="the job")
    assert state[:2] == ("complete", None)


async def test_cancelling_a_chats_work_withdraws_a_retry_waiting_on_a_stopped_render(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel all, as moving a chat to Recently Deleted does, leaves nothing to start later."""

    stopped = await _stopped_in_handoff(
        app, client, settings, monkeypatch, lambda _chat_id, job_id: f"/api/jobs/{job_id}/cancel"
    )
    retried = await client.post(f"/api/jobs/{stopped.job_id}/retry")
    assert retried.status_code == 200, retried.text

    cancelling = asyncio.create_task(client.post(f"/api/chats/{stopped.chat_id}/cancel-all"))
    stopped.reloaded.set()
    cancelled = await asyncio.wait_for(cancelling, timeout=PATIENCE_SECONDS)
    assert cancelled.status_code == 204, cancelled.text
    await asyncio.wait_for(stopped.stopping, timeout=PATIENCE_SECONDS)
    # Whatever runs on the event loop once the stopped execution has ended has run.
    await asyncio.sleep(0.5)

    latest = stopped.world.orchestrator._tasks.get(stopped.job_id)
    assert latest is None or latest.done(), "an execution of the cancelled job is alive"
    assert _job_state(stopped.job_id) == ("cancelled", None, 1)
    assert stopped.attempts == [stopped.run_id]
    impact = await client.get(f"/api/chats/{stopped.chat_id}/deletion-impact")
    assert impact.status_code == 200, impact.text
    assert impact.json()["counts"]["active_work"] == 0
    assert "trash" in impact.json()["available_actions"]
