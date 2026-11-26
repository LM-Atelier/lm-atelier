"""A prompt someone removes from ComfyUI's own queue before it runs ends its run at once.

ComfyUI tells nobody when a waiting prompt is removed from its queue: it
broadcasts only the queue's new length. The run used to read queued for the
whole inactivity window, fail as a silent ComfyUI and be sent again. Now a
shorter queue, or a socket gone quiet, is the cue to ask ComfyUI whether it
still holds the prompt: its job endpoint first, or its queue and the prompt's
history where that endpoint is missing. One it no longer holds anywhere ends
the run as stopped in ComfyUI, someone's choice, so it is neither retried nor
stopped again. Anything unclear counts as held, so a slow or failing ComfyUI
never ends a run this way.

ComfyUI is a constructed backend behind the adapter's own HTTP client and
progress socket.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_chat_handoff_across_retries import finished
from test_comfyui_kept_after_a_workflow_failure import (
    PICTURE,
    PROMPT_ID,
    SUCCEEDED,
    _adapter,
    _ComfyUI,
    _comfyui_engine,
    _picture,
    _ready_world,
    _recorded,
    _request,
)
from test_generation_retry import choose_retries
from test_media_worker_replacement import _image_turn

from local_lm.config import Settings
from local_lm.orchestrator import MEDIA_STOPPED_BY_ENGINE

#: ComfyUI's queue lengths as the socket connects, as the prompt joins, as it leaves.
CONNECTED, JOINED, LEFT = 0, 1, 0
#: The paths a stop for the prompt is sent to.
STOPS = {f"/api/jobs/{PROMPT_ID}/cancel", "/queue", "/interrupt"}


def _status(remaining: int) -> dict[str, Any]:
    """The broadcast ComfyUI sends as the socket connects and whenever its queue changes."""

    return {"type": "status", "data": {"status": {"exec_info": {"queue_remaining": remaining}}}}


#: The broadcasts around a prompt someone removes before it runs.
REMOVED = (_status(CONNECTED), _status(JOINED), _status(LEFT))


class _QueueingComfyUI(_ComfyUI):
    """The constructed ComfyUI, also answering for the prompt's job and its queue.

    The prompt is held until `removed_after` checks have asked about it, after
    which it is gone, as when someone removes it from ComfyUI's own page. With
    `jobs="missing"` there is no job endpoint, as in an older ComfyUI, and the
    queue answers instead: the prompt `waiting` or `running` there while held,
    or `queue` as the whole answer. `checks` counts the questions asked about
    the prompt, and `stops` every stop sent for it.
    """

    def __init__(
        self,
        *,
        removed_after: int | None = None,
        jobs: str = "endpoint",
        place: str = "waiting",
        queue: httpx.Response | None = None,
        **fields: Any,
    ) -> None:
        super().__init__(**fields)
        self.removed_after = removed_after
        self.jobs = jobs
        self.place = place
        self.queue = queue
        self.checks = 0
        self.stops: list[str] = []
        self.prompts_sent = 0

    def _held(self) -> bool:
        return self.removed_after is None or self.checks <= self.removed_after

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        route = (request.method, request.url.path)
        if route == ("POST", "/prompt"):
            self.prompts_sent += 1
        if request.method == "POST" and request.url.path in STOPS:
            self.stops.append(request.url.path)
            if self.jobs == "missing" and request.url.path.startswith("/api/jobs/"):
                return httpx.Response(404, text="404: Not Found")
            return httpx.Response(200, json={})
        if route == ("GET", f"/api/jobs/{PROMPT_ID}"):
            if self.jobs == "missing":
                return httpx.Response(404, text="404: Not Found")
            self.checks += 1
            if self._held():
                return httpx.Response(200, json={"id": PROMPT_ID, "status": "pending"})
            return httpx.Response(404, json={"error": "Job not found"})
        if route == ("GET", "/queue"):
            self.checks += 1
            if self.queue is not None:
                return self.queue
            row = [1, PROMPT_ID, {}, {}, []]
            held = self._held()
            return httpx.Response(
                200,
                json={
                    "queue_running": [row] if held and self.place == "running" else [],
                    "queue_pending": [row] if held and self.place == "waiting" else [],
                },
            )
        return await super().__call__(request)


async def _events(adapter: Any, *, patience: float = PATIENCE_SECONDS) -> list[str]:
    """The types of the events one generation of the teapot reports, in order."""

    async def generate() -> list[str]:
        return [event.type async for event in adapter.generate(_request())]

    return await asyncio.wait_for(generate(), timeout=patience)


def _queue_still_listing_the_prompt() -> httpx.Response:
    """A queue answer that still lists the prompt as waiting."""

    row = [1, PROMPT_ID, {}, {}, []]
    return httpx.Response(200, json={"queue_running": [], "queue_pending": [row]})


@pytest.mark.parametrize(
    "backend",
    [
        # The job endpoint is asked first, so a queue that still lists the
        # prompt is never read.
        pytest.param(
            {"jobs": "endpoint", "queue": _queue_still_listing_the_prompt()}, id="job-endpoint"
        ),
        pytest.param({"jobs": "missing"}, id="no-job-endpoint"),
    ],
)
async def test_a_prompt_removed_before_it_runs_ends_as_stopped_at_once_with_no_stop_sent(
    monkeypatch: pytest.MonkeyPatch, backend: dict[str, Any]
) -> None:
    """Nothing is left to stop, and a stop sent anyway could reach someone else's work."""

    comfy = _QueueingComfyUI(removed_after=0, **backend)
    adapter = await _adapter(monkeypatch, comfy, REMOVED, inactivity_seconds=60)
    try:
        # Well inside the inactivity window, which a silent wait would use up.
        events = await _events(adapter, patience=10)
    finally:
        await adapter.close()

    assert events[events.index("queued") :] == ["queued", "cancelled"]
    assert comfy.checks == 1
    assert comfy.stops == []


def _oversized_queue() -> httpx.Response:
    """A queue answer over the JSON bound, which lists nothing of the run's."""

    return httpx.Response(200, json={"queue_running": [], "queue_pending": [], "x": "y" * 256})


@pytest.mark.parametrize(
    "comfy",
    [
        pytest.param({"jobs": "endpoint"}, id="job-endpoint-holds-it"),
        pytest.param({"jobs": "missing", "place": "waiting"}, id="waiting-in-the-queue"),
        pytest.param({"jobs": "missing", "place": "running"}, id="running-without-a-record-yet"),
        pytest.param({"jobs": "missing", "queue": httpx.Response(500)}, id="queue-unreadable"),
        pytest.param(
            {
                "jobs": "missing",
                "queue": httpx.Response(200, json={"queue_running": [], "queue_pending": ["row"]}),
            },
            id="queue-row-of-another-shape",
        ),
        pytest.param({"jobs": "missing", "queue": _oversized_queue()}, id="queue-over-the-bound"),
    ],
)
async def test_a_prompt_comfyui_may_still_hold_is_waited_for_as_before(
    monkeypatch: pytest.MonkeyPatch, comfy: dict[str, Any]
) -> None:
    """Held, or not known to be gone: the run waits, and the inactivity timeout ends it."""

    monkeypatch.setattr("local_lm.adapters.comfyui._MAX_COMFY_JSON_BYTES", 128)
    backend = _QueueingComfyUI(**comfy)
    adapter = await _adapter(monkeypatch, backend, REMOVED, inactivity_seconds=0.5)
    try:
        with pytest.raises(RuntimeError, match="stopped reporting generation activity"):
            await _events(adapter)
    finally:
        await adapter.close()

    # Once at the shorter queue, once more as the socket went quiet.
    assert backend.checks == 2


async def test_a_queue_that_only_grows_is_never_asked_about(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The length read as the socket connects is only where counting starts."""

    comfy = _QueueingComfyUI(removed_after=0, record=_recorded(1), files=(_picture(PICTURE),))
    growing = (_status(CONNECTED), _status(JOINED), _status(JOINED + 1), SUCCEEDED)
    adapter = await _adapter(monkeypatch, comfy, growing, inactivity_seconds=60)
    try:
        events = await _events(adapter)
    finally:
        await adapter.close()

    assert events[-1] == "complete"
    assert comfy.checks == 0


@pytest.mark.parametrize("jobs", ["endpoint", "missing"])
async def test_a_prompt_gone_when_the_socket_goes_quiet_ends_as_stopped(
    monkeypatch: pytest.MonkeyPatch, jobs: str
) -> None:
    """A removal whose broadcast never came is found when nothing else is heard, and not stopped."""

    comfy = _QueueingComfyUI(removed_after=0, jobs=jobs)
    adapter = await _adapter(monkeypatch, comfy, (), inactivity_seconds=0.5)
    try:
        events = await _events(adapter)
    finally:
        await adapter.close()

    assert events[events.index("queued") :] == ["queued", "cancelled"]
    assert comfy.stops == []


async def test_a_prompt_whose_run_began_is_not_asked_about_when_the_socket_goes_quiet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once ComfyUI has begun running it, silence is the inactivity it always was."""

    comfy = _QueueingComfyUI(removed_after=0)
    began = (
        _status(CONNECTED),
        _status(JOINED),
        {"type": "execution_start", "data": {"prompt_id": PROMPT_ID}},
    )
    adapter = await _adapter(monkeypatch, comfy, began, inactivity_seconds=0.5)
    try:
        with pytest.raises(RuntimeError, match="stopped reporting generation activity"):
            await _events(adapter)
    finally:
        await adapter.close()

    assert comfy.checks == 0


async def test_a_prompt_that_already_ran_is_collected_as_usual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gone from the queue but in the history: it finished, and its pictures are collected."""

    comfy = _QueueingComfyUI(
        removed_after=0, jobs="missing", record=_recorded(1), files=(_picture(PICTURE),)
    )
    adapter = await _adapter(monkeypatch, comfy, (*REMOVED, SUCCEEDED), inactivity_seconds=60)
    try:
        events = await _events(adapter)
    finally:
        await adapter.close()

    assert events[-1] == "complete"
    assert "cancelled" not in events
    assert comfy.checks == 1


async def test_a_run_whose_prompt_was_removed_fails_as_stopped_in_comfyui_and_is_not_retried(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready_world(app, settings, monkeypatch)
    await choose_retries(client, 1)
    comfy = _QueueingComfyUI(removed_after=0)
    adapter = await _comfyui_engine(app, monkeypatch, comfy, REMOVED)
    try:
        _chat_id, run_id = await _image_turn(client)
        run = await finished(client, run_id, "failed")
    finally:
        await adapter.close()

    assert run["error"] == MEDIA_STOPPED_BY_ENGINE
    assert comfy.prompts_sent == 1
    assert comfy.stops == []


@pytest.mark.parametrize("jobs", ["endpoint", "missing"])
async def test_a_run_whose_prompt_was_removed_unannounced_sends_no_stop_after_it(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    jobs: str,
) -> None:
    """Found only when the socket goes quiet, the removal still leaves nothing to stop."""

    _ready_world(app, settings, monkeypatch)
    await choose_retries(client, 1)
    comfy = _QueueingComfyUI(removed_after=0, jobs=jobs)
    adapter = await _comfyui_engine(app, monkeypatch, comfy)
    adapter.inactivity_seconds = 0.5
    try:
        _chat_id, run_id = await _image_turn(client)
        run = await finished(client, run_id, "failed")
    finally:
        await adapter.close()

    assert run["error"] == MEDIA_STOPPED_BY_ENGINE
    assert comfy.prompts_sent == 1
    assert comfy.stops == []
