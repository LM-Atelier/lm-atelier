"""A stop the user asks for ends only its own prompt, and a stop that fails stays local.

Every stop names the prompt its run submitted and goes to ComfyUI's
cancel-by-ID route first; only a ComfyUI without that route gets the older
delete-then-interrupt, naming the same prompt. A stop that cannot reach
ComfyUI ends there, without retrying the older, less precise way, and the run
still stops.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest
from run_waits import PATIENCE_SECONDS

from local_lm.adapters import comfyui as comfyui_module
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter

Backend = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]


async def _adapter_over(backend: Backend, **options: Any) -> ComfyUIAdapter:
    adapter = ComfyUIAdapter("http://comfy.test", **options)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(backend)
    )
    return adapter


@pytest.mark.parametrize("atomic_status", [200, 404, 405])
@pytest.mark.parametrize("state", ["waiting", "running", "finished"])
async def test_a_user_stop_ends_only_its_own_prompt(atomic_status: int, state: str) -> None:
    submitted = "neutral-submitted"
    other = "neutral-other"
    waiting = [other, *([submitted] if state == "waiting" else [])]
    running = submitted if state == "running" else other
    sent: list[tuple[str, dict[str, object]]] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        nonlocal running
        body = json.loads(request.content or b"{}")
        sent.append((request.url.path, body))
        if request.url.path == f"/api/jobs/{submitted}/cancel":
            if atomic_status != 200:
                return httpx.Response(atomic_status)
            cancelled = submitted in waiting or running == submitted
            if submitted in waiting:
                waiting.remove(submitted)
            if running == submitted:
                running = ""
            return httpx.Response(200, json={"cancelled": cancelled})
        if request.url.path == "/queue":
            for prompt_id in body.get("delete", []):
                if prompt_id in waiting:
                    waiting.remove(prompt_id)
        if request.url.path == "/interrupt" and body.get("prompt_id") in {None, running}:
            running = ""
        return httpx.Response(200)

    adapter = await _adapter_over(backend)
    stopped = asyncio.Event()
    adapter._cancel_events["neutral-run"] = stopped
    adapter._jobs["neutral-run"] = submitted
    try:
        await adapter.cancel("neutral-run")
    finally:
        await adapter.close()

    assert stopped.is_set()
    assert waiting == [other]
    assert running == ("" if state == "running" else other)
    legacy = [("/queue", {"delete": [submitted]}), ("/interrupt", {"prompt_id": submitted})]
    assert sent == [
        (f"/api/jobs/{submitted}/cancel", {}),
        *(legacy if atomic_status != 200 else []),
    ]


@pytest.mark.parametrize("asked", [True, False], ids=["a user stop", "an abandoned prompt"])
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadError,
        httpx.ReadTimeout,
        httpx.WriteError,
        httpx.RemoteProtocolError,
    ],
)
async def test_a_stop_that_cannot_reach_comfyui_stays_local(
    asked: bool, failure: type[httpx.TransportError]
) -> None:
    sent: list[str] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.path)
        raise failure("Neutral transport failure", request=request)

    adapter = await _adapter_over(backend)
    stopped = asyncio.Event()
    adapter._cancel_events["neutral-run"] = stopped
    adapter._jobs["neutral-run"] = "neutral-submitted"
    try:
        if asked:
            await adapter.cancel("neutral-run")
        else:
            await adapter._abandon_prompt("neutral-submitted")
    finally:
        await adapter.close()

    assert sent == ["/api/jobs/neutral-submitted/cancel"], "nothing retries it the older way"
    if asked:
        assert stopped.is_set()


async def test_a_stop_before_its_prompt_is_submitted_sends_nothing() -> None:
    sent: list[str] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.path)
        return httpx.Response(200)

    adapter = await _adapter_over(backend)
    stopped = asyncio.Event()
    adapter._cancel_events["neutral-started"] = stopped
    try:
        await adapter.cancel("neutral-started")
        await adapter.cancel("neutral-not-started")
    finally:
        await adapter.close()

    assert stopped.is_set()
    assert "neutral-not-started" in adapter._cancelled, "a run not started yet is stopped later"
    assert sent == []


@pytest.mark.parametrize("still_waiting", [False, True], ids=["running", "waiting"])
async def test_a_stop_ends_the_run_before_comfyui_answers_it(
    monkeypatch: pytest.MonkeyPatch, still_waiting: bool
) -> None:
    submitted = "neutral-submitted"
    other = "neutral-other"
    waiting = [submitted] if still_waiting else []
    running = other if still_waiting else submitted
    listening = asyncio.Event()
    stopped_listening = asyncio.Event()
    stop_arrived = asyncio.Event()
    answer_stop = asyncio.Event()
    stops: list[str] = []

    class Socket:
        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def __aiter__(self) -> Socket:
            return self

        async def __anext__(self) -> str:
            listening.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped_listening.set()
            raise AssertionError("The neutral socket stays quiet until the stop")

    def connect(*_args: Any, **_kwargs: Any) -> Socket:
        return Socket()

    async def backend(request: httpx.Request) -> httpx.Response:
        nonlocal running
        if request.method == "GET":
            return httpx.Response(200, json={})
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": submitted, "node_errors": {}})
        # Every other request is part of the stop, so an interrupt shows up here.
        stops.append(request.url.path)
        if request.url.path != f"/api/jobs/{submitted}/cancel":
            return httpx.Response(200)
        if submitted in waiting:
            waiting.remove(submitted)
        if running == submitted:
            running = ""
        stop_arrived.set()
        await answer_stop.wait()
        return httpx.Response(200, json={"cancelled": True})

    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", connect)
    monkeypatch.setattr(comfyui_module, "ABANDONED_INTERRUPT_SECONDS", PATIENCE_SECONDS)
    adapter = await _adapter_over(backend, inactivity_seconds=PATIENCE_SECONDS)
    request = MediaRequest(
        run_id="neutral-run",
        operation="text_to_image",
        prompt="Neutral generation fixture",
        negative_prompt="",
        input_paths=[],
        workflow={},
        parameters={"seed": 42},
    )

    async def collect() -> list[MediaEvent]:
        return [event async for event in adapter.generate(request)]

    collecting = asyncio.create_task(collect())
    stopping: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(listening.wait(), PATIENCE_SECONDS)
        stopping = asyncio.create_task(adapter.cancel(request.run_id))
        await asyncio.wait_for(stop_arrived.wait(), PATIENCE_SECONDS)
        events = await asyncio.wait_for(collecting, PATIENCE_SECONDS)
        assert stopped_listening.is_set()
        assert not stopping.done(), "the run ended while ComfyUI had not answered the stop"
        answer_stop.set()
        await asyncio.wait_for(stopping, PATIENCE_SECONDS)
    finally:
        answer_stop.set()
        tasks = [collecting, *([stopping] if stopping is not None else [])]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await adapter.close()

    assert events[-1].type == "cancelled"
    assert request.run_id not in adapter._jobs
    assert request.run_id not in adapter._cancel_events
    assert waiting == []
    assert running == (other if still_waiting else "")
    assert stops == [f"/api/jobs/{submitted}/cancel"]
