"""Cancelling an execution stops its submitted prompt, preserving its replacement."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from local_lm.adapters.base import MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter


@pytest.mark.parametrize("replacement", [False, True])
async def test_cancelled_execution_stops_only_its_accepted_prompt(
    monkeypatch: pytest.MonkeyPatch, replacement: bool
) -> None:
    mine = "garden-attempt"
    successor = "garden-replacement"
    running: str | None = successor if replacement else mine
    pending = [mine] if replacement else []
    receiving = asyncio.Event()
    operations: list[tuple[str, dict[str, Any]]] = []

    class Socket:
        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def __aiter__(self) -> Socket:
            return self

        async def __anext__(self) -> str:
            receiving.set()
            await asyncio.Event().wait()
            raise AssertionError("The constructed socket stays silent")

    async def backend(request: httpx.Request) -> httpx.Response:
        nonlocal running
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": mine, "node_errors": {}})
        if request.url.path in {"/queue", "/interrupt"}:
            body = json.loads(request.content or b"{}")
            operations.append((request.url.path, body))
            if request.url.path == "/queue":
                for identifier in body.get("delete", []):
                    if identifier in pending:
                        pending.remove(identifier)
            elif body.get("prompt_id") in {None, running}:
                running = None
            return httpx.Response(200, json={})
        if request.url.path == f"/history/{mine}":
            return httpx.Response(200, json={})
        raise AssertionError("Unexpected constructed backend request")

    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", lambda *_a, **_k: Socket())
    adapter = ComfyUIAdapter("http://comfy.test", inactivity_seconds=60)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(backend)
    )
    request = MediaRequest(
        run_id="garden-run",
        operation="text_to_image",
        prompt="A garden path",
        negative_prompt=None,
        input_paths=[],
        workflow={},
        parameters={},
    )

    async def collect() -> None:
        async for _event in adapter.generate(request):
            pass

    execution = asyncio.create_task(collect())
    try:
        await asyncio.wait_for(receiving.wait(), timeout=2)
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution, timeout=2)
    finally:
        if not execution.done():
            execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)
        await adapter.close()

    assert operations == [("/queue", {"delete": [mine]}), ("/interrupt", {"prompt_id": mine})]
    assert mine not in pending
    assert running == (successor if replacement else None)
    assert request.run_id not in adapter._jobs
    assert request.run_id not in adapter._cancel_events


@pytest.mark.parametrize("reply_lost", [False, True])
async def test_older_execution_teardown_keeps_replacement_cancellation_state(
    monkeypatch: pytest.MonkeyPatch,
    reply_lost: bool,
) -> None:
    prompts = ["garden-older", "garden-newer"]
    submitted = 0
    connected = 0
    receiving = [asyncio.Event(), asyncio.Event()]
    operations: list[tuple[str, dict[str, Any]]] = []
    running: str | None = prompts[1]
    clients: dict[str, str] = {}
    pending = [prompts[0]]

    class Socket:
        def __init__(self, index: int) -> None:
            self.index = index

        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def __aiter__(self) -> Socket:
            return self

        async def __anext__(self) -> str:
            receiving[self.index].set()
            await asyncio.Event().wait()
            raise AssertionError("The constructed socket stays silent")

    def connect(*_args: object, **_kwargs: object) -> Socket:
        nonlocal connected
        socket = Socket(connected)
        connected += 1
        return socket

    async def backend(request: httpx.Request) -> httpx.Response:
        nonlocal submitted, running
        if request.url.path == "/prompt":
            prompt = prompts[submitted]
            submitted += 1
            clients[prompt] = json.loads(request.content)["client_id"]
            if reply_lost and prompt == prompts[0]:
                receiving[0].set()
                await asyncio.Event().wait()
            return httpx.Response(200, json={"prompt_id": prompt, "node_errors": {}})
        if request.url.path == "/queue" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "queue_running": [[1, prompts[1], {}, {"client_id": clients[prompts[1]]}, []]],
                    "queue_pending": [[0, prompts[0], {}, {"client_id": clients[prompts[0]]}, []]],
                },
            )
        if request.url.path in {"/queue", "/interrupt"}:
            body = json.loads(request.content or b"{}")
            operations.append((request.url.path, body))
            if request.url.path == "/queue":
                for prompt in body.get("delete", []):
                    if prompt in pending:
                        pending.remove(prompt)
            if request.url.path == "/interrupt" and body.get("prompt_id") in {None, running}:
                running = None
            return httpx.Response(200, json={})
        if request.url.path.startswith("/history/"):
            return httpx.Response(200, json={})
        raise AssertionError("Unexpected constructed backend request")

    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", connect)
    adapter = ComfyUIAdapter("http://comfy.test", inactivity_seconds=60)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(backend)
    )
    request = MediaRequest(
        run_id="garden-run",
        operation="text_to_image",
        prompt="A garden path",
        negative_prompt=None,
        input_paths=[],
        workflow={},
        parameters={},
    )

    async def collect() -> None:
        async for _event in adapter.generate(request):
            pass

    older = asyncio.create_task(collect())
    newer: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(receiving[0].wait(), timeout=2)
        newer = asyncio.create_task(collect())
        await asyncio.wait_for(receiving[1].wait(), timeout=2)
        newer_event = adapter._cancel_events[request.run_id]
        assert adapter._jobs[request.run_id] == prompts[1]
        older.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(older, timeout=2)

        assert adapter._jobs.get(request.run_id) == prompts[1]
        assert adapter._cancel_events.get(request.run_id) is newer_event
        assert not newer_event.is_set() and not newer.done()
        assert running == prompts[1]
        assert prompts[0] not in pending
        assert operations == [
            ("/queue", {"delete": [prompts[0]]}),
            ("/interrupt", {"prompt_id": prompts[0]}),
        ]
    finally:
        tasks = [older, *([newer] if newer is not None else [])]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await adapter.close()
