"""A stopped ComfyUI run ends, and so does one that ComfyUI stops by itself.

A stop can reach the adapter after a run has failed in ComfyUI but while the
adapter is still cleaning up after it. The run is still generating as far as
LM Atelier knows, so the stop is told to the adapter, and the adapter must not
keep it for the run's next generation: a retry reuses the run, and would end at
once without sending its prompt, leaving its job running with nothing to end it.

ComfyUI can also stop a prompt that nobody in LM Atelier asked to stop, as when
it is stopped from ComfyUI's own page. That run fails, saying why, rather than
staying running until the application restarts. Like any other failed run, it
fails once the device is handed back, so a retry asked for after that runs. A
stop LM Atelier asked for still ends its run cancelled.

ComfyUI here is a constructed backend behind the adapter's own HTTP client and
progress socket, so every request the adapter makes is answered without a
network.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from run_waits import PATIENCE_SECONDS, wait_until
from sqlalchemy import select
from test_chat_handoff_across_retries import ChatWorker
from test_generation_retry import read_run, start_media

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.db import SessionLocal
from local_lm.models import Job

#: What a progress socket reports for the prompt it was opened for.
Script = Callable[[str], list[dict[str, Any]]]


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (40, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def _failed(prompt_id: str) -> dict[str, Any]:
    return {
        "type": "execution_error",
        "data": {"prompt_id": prompt_id, "node_type": "KSampler", "exception_type": "RuntimeError"},
    }


def _finished(prompt_id: str) -> dict[str, Any]:
    return {"type": "executing", "data": {"prompt_id": prompt_id, "node": None}}


def _interrupted(prompt_id: str) -> dict[str, Any]:
    return {"type": "execution_interrupted", "data": {"prompt_id": prompt_id}}


class _Comfy:
    """A ComfyUI that takes every prompt it is sent and answers what it is asked.

    Prompts are named in the order they arrive. `slow_history` names prompts
    whose record is never answered, as when ComfyUI is slow to write one: the
    request waits until it is cancelled. `finished` names prompts with a
    picture to collect. `on_stop` runs before a stop is answered.
    """

    def __init__(
        self,
        *,
        slow_history: frozenset[str] = frozenset(),
        finished: frozenset[str] = frozenset(),
    ) -> None:
        self.slow_history = slow_history
        self.finished = finished
        self.on_stop: Callable[[], Awaitable[None]] | None = None
        self.picture = _png()
        self.prompts: list[str] = []
        self.stops: list[str] = []
        self.reading_slow_history = asyncio.Event()
        self.listening = asyncio.Event()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/prompt":
            prompt_id = f"garden-{len(self.prompts) + 1}"
            self.prompts.append(prompt_id)
            return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})
        if request.method == "POST" and path.startswith("/api/jobs/") and path.endswith("/cancel"):
            self.stops.append(path.split("/")[3])
            if self.on_stop is not None:
                await self.on_stop()
            return httpx.Response(200, json={"cancelled": True})
        if request.method == "GET" and path.startswith("/history/"):
            prompt_id = path.removeprefix("/history/")
            if prompt_id in self.slow_history:
                self.reading_slow_history.set()
                await asyncio.Event().wait()
            if prompt_id not in self.finished:
                return httpx.Response(200, json={})
            image = {"filename": f"{prompt_id}.png", "subfolder": "", "type": "output"}
            return httpx.Response(200, json={prompt_id: {"outputs": {"9": {"images": [image]}}}})
        if request.method == "GET" and path == "/view":
            return httpx.Response(200, content=self.picture, headers={"content-type": "image/png"})
        if request.method == "GET" and path == "/system_stats":
            return httpx.Response(200, json={"system": {"comfyui_version": "constructed"}})
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {path}")


class _Socket:
    """A progress socket that reports what its script says for the prompt just sent.

    The adapter opens the socket before it sends the prompt, so the prompt is
    known only once the socket is first read. After its messages the socket
    stays open and silent.
    """

    def __init__(self, comfy: _Comfy, script: Script) -> None:
        self.comfy = comfy
        self.script = script
        self.messages: list[dict[str, Any]] | None = None

    async def __aenter__(self) -> _Socket:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def __aiter__(self) -> _Socket:
        return self

    async def __anext__(self) -> str:
        if self.messages is None:
            self.messages = self.script(self.comfy.prompts[-1])
        if self.messages:
            return json.dumps(self.messages.pop(0))
        self.comfy.listening.set()
        await asyncio.Event().wait()
        raise AssertionError("The constructed socket stays silent")


async def _comfyui(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, comfy: _Comfy, script: Script
) -> ComfyUIAdapter:
    """Make the ComfyUI adapter the application's media engine, talking to `comfy`."""

    monkeypatch.setattr(
        "local_lm.adapters.comfyui.websockets.connect",
        lambda *_args, **_kwargs: _Socket(comfy, script),
    )
    adapter = ComfyUIAdapter("http://comfy.test", inactivity_seconds=60)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(comfy)
    )
    monkeypatch.setattr(app.state.services.engines, "media", adapter)
    return adapter


def _job(run_id: str) -> tuple[str, str, str | None]:
    """The run's job as the queue records it: its id, its status and its owner."""

    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        assert job is not None
        return job.id, job.status, job.claim_owner


async def _execution_ended(app: FastAPI, job_id: str) -> None:
    tasks = app.state.services.orchestrator._tasks

    async def running() -> bool:
        return job_id in tasks

    await wait_until(running, lambda still: not still, what="the job's execution to end")


class _RestoringChat(ChatWorker):
    """A chat model a picture displaces, whose reload waits until the test allows it.

    `restoring` is set once the device handoff begins to load it again, and the
    load finishes once `may_restore` is set.
    """

    def __init__(self, app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
        self.restoring = asyncio.Event()
        self.may_restore = asyncio.Event()
        super().__init__(app, monkeypatch)

    async def resume(self, profile_id: str) -> None:
        self.restoring.set()
        await self.may_restore.wait()
        await super().resume(profile_id)


def _queued_without_owner() -> list[str]:
    """The jobs waiting in the queue that no execution has claimed."""

    with SessionLocal() as session:
        return list(
            session.scalars(select(Job.id).where(Job.status == "queued", Job.claim_owner.is_(None)))
        )


async def test_a_retry_runs_after_a_stop_that_lands_while_comfyui_cleans_up(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop is not kept for the retry, so the retry's prompt reaches ComfyUI.

    The run fails in ComfyUI, and while the adapter reads the failed prompt's
    record to clean up after it, the run is stopped. Its retry then runs to
    its result.
    """

    comfy = _Comfy(slow_history=frozenset({"garden-1"}), finished=frozenset({"garden-2"}))

    def script(prompt_id: str) -> list[dict[str, Any]]:
        return [_failed(prompt_id) if prompt_id == "garden-1" else _finished(prompt_id)]

    adapter = await _comfyui(app, monkeypatch, comfy, script)
    try:
        run_id = await start_media(client)
        job_id = _job(run_id)[0]
        await asyncio.wait_for(comfy.reading_slow_history.wait(), timeout=PATIENCE_SECONDS)

        stopped = await client.post(f"/api/jobs/{job_id}/cancel")
        assert stopped.status_code == 200, stopped.text
        assert stopped.json()["status"] == "cancelled"
        # What the stopped generation left behind for the run's next one.
        left_behind = {
            "stop": run_id in adapter._cancelled,
            "event": run_id in adapter._cancel_events,
            "prompt": run_id in adapter._jobs,
        }

        retried = await client.post(f"/api/jobs/{job_id}/retry")
        assert retried.status_code == 200, retried.text
        await _execution_ended(app, job_id)
        run = await read_run(client, run_id)
    finally:
        await adapter.close()

    assert comfy.prompts == ["garden-1", "garden-2"]
    assert run["status"] == "complete", run["error"]
    assert _job(run_id)[1:] == ("complete", None)
    assert left_behind == {"stop": False, "event": False, "prompt": False}


async def test_a_run_comfyui_stops_by_itself_fails_with_its_reason(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing in LM Atelier asked for the stop, so nothing else would end the run.

    It fails rather than staying running with no owner, and it is not retried
    automatically: stopping it in ComfyUI was someone's choice.
    """

    comfy = _Comfy()
    adapter = await _comfyui(app, monkeypatch, comfy, lambda prompt_id: [_interrupted(prompt_id)])
    try:
        run_id = await start_media(client)
        job_id = _job(run_id)[0]
        await _execution_ended(app, job_id)
        run = await read_run(client, run_id)
    finally:
        await adapter.close()

    assert comfy.prompts == ["garden-1"]
    assert (run["status"], run["error"]) == (
        "failed",
        "The media engine stopped this generation before it finished. Retry to run it again.",
    )
    assert run["provenance_json"]["failure_retries"] == {"limit": 3, "used": 0, "pending": False}
    assert _job(run_id)[1:] == ("failed", None)


async def test_a_run_stopped_from_lm_atelier_ends_cancelled_when_the_adapter_reports_it_first(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop asked for here ends its run cancelled, even when the adapter reports it first.

    Stopping all of a chat's work tells ComfyUI to stop each generating run
    before it cancels the runs' executions, so a run can hear from the adapter
    that it was cancelled in between. That run still ends cancelled, not failed.
    """

    comfy = _Comfy()
    adapter = await _comfyui(app, monkeypatch, comfy, lambda _prompt_id: [])
    try:
        run_id = await start_media(client)
        job_id = _job(run_id)[0]
        chat_id = (await read_run(client, run_id))["chat_id"]
        await asyncio.wait_for(comfy.listening.wait(), timeout=PATIENCE_SECONDS)

        async def once_the_run_has_ended() -> None:
            # ComfyUI answers the stop only after the run has handled the
            # adapter's report, so that report is what ends the execution.
            await _execution_ended(app, job_id)

        comfy.on_stop = once_the_run_has_ended
        stopped = await client.post(f"/api/chats/{chat_id}/cancel-all")
        assert stopped.status_code == 204, stopped.text
        run = await read_run(client, run_id)
    finally:
        await adapter.close()

    assert comfy.stops == ["garden-1"]
    assert run["status"] == "cancelled", run["error"]
    assert _job(run_id)[1:] == ("cancelled", None)


async def test_a_retry_comfyui_stops_by_itself_fails_although_its_run_was_stopped_before(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop asked for an earlier generation is forgotten once that generation ends.

    Otherwise ComfyUI's own stop of the retry would read as that earlier stop,
    and the retry would stay running with nothing to end it.
    """

    comfy = _Comfy()

    def script(prompt_id: str) -> list[dict[str, Any]]:
        return [] if prompt_id == "garden-1" else [_interrupted(prompt_id)]

    adapter = await _comfyui(app, monkeypatch, comfy, script)
    try:
        run_id = await start_media(client)
        job_id = _job(run_id)[0]
        await asyncio.wait_for(comfy.listening.wait(), timeout=PATIENCE_SECONDS)
        stopped = await client.post(f"/api/jobs/{job_id}/cancel")
        assert stopped.status_code == 200, stopped.text

        retried = await client.post(f"/api/jobs/{job_id}/retry")
        assert retried.status_code == 200, retried.text
        await _execution_ended(app, job_id)
        run = await read_run(client, run_id)
    finally:
        await adapter.close()

    assert comfy.prompts == ["garden-1", "garden-2"]
    assert (run["status"], run["error"]) == (
        "failed",
        "The media engine stopped this generation before it finished. Retry to run it again.",
    )
    assert _job(run_id)[1:] == ("failed", None)


async def test_a_run_comfyui_stops_by_itself_fails_after_the_handoff_so_its_retry_runs(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run fails once the chat model it displaced is back, as other failed runs do.

    While that model loads again, the run still reads running and a retry is
    refused. A retry accepted then would be left queued with nothing to run it,
    and later pictures would wait behind it. Once chat is back the run reads
    failed, and a retry runs it to its result.
    """

    chat = _RestoringChat(app, monkeypatch)
    comfy = _Comfy(finished=frozenset({"garden-2"}))

    def script(prompt_id: str) -> list[dict[str, Any]]:
        return [_interrupted(prompt_id) if prompt_id == "garden-1" else _finished(prompt_id)]

    adapter = await _comfyui(app, monkeypatch, comfy, script)
    try:
        run_id = await start_media(client)
        job_id = _job(run_id)[0]
        await asyncio.wait_for(chat.restoring.wait(), timeout=PATIENCE_SECONDS)
        during = (await read_run(client, run_id))["status"]
        early = await client.post(f"/api/jobs/{job_id}/retry")
        assert (during, early.status_code) == ("running", 409), early.text

        chat.may_restore.set()
        await _execution_ended(app, job_id)
        failed = await read_run(client, run_id)
        retried = await client.post(f"/api/jobs/{job_id}/retry")
        assert retried.status_code == 200, retried.text
        await _execution_ended(app, job_id)
        run = await read_run(client, run_id)
    finally:
        chat.may_restore.set()
        await adapter.close()

    assert (failed["status"], failed["error"]) == (
        "failed",
        "The media engine stopped this generation before it finished. Retry to run it again.",
    )
    assert failed["provenance_json"]["failure_retries"] == {"limit": 3, "used": 0, "pending": False}
    assert comfy.prompts == ["garden-1", "garden-2"]
    assert run["status"] == "complete", run["error"]
    assert _job(run_id)[1:] == ("complete", None)
    assert _queued_without_owner() == []
    # Each attempt displaced the chat model and gave it back once it was done.
    assert chat.order == ["stop", "restore", "stop", "restore"]
