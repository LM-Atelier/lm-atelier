"""A prompt LM Atelier left in ComfyUI is not counted as someone else's work.

Before ComfyUI is stopped under work in progress, the caller asks how many of
the prompts it holds were queued by someone else. A prompt of LM Atelier's own
can outlive the generation that sent it, as one stuck in a node does through a
stop, and counting it would hold ComfyUI for work nobody is waiting for. Every
prompt the adapter submits carries a client id with a prefix of its own, and
the queue, which holds every prompt's graph, is read for client ids only when
one of those prompts may still be there.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from run_waits import PATIENCE_SECONDS
from test_comfyui_adapter import _BlockingSocket, media_request
from test_comfyui_identical_requests import _caching_comfy, _socket_for, _workflow

from local_lm.adapters import comfyui as comfyui_adapter
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter

#: A stand-in for the text of another application's prompt. Their queue row holds
#: it in every field but the client id, so finding it anywhere means more than the
#: client id was taken from the queue.
SENTINEL = "neutral-sentinel-4b1f"
_PNG = b"\x89PNG\r\n\x1a\nforeign-prompts"

QueueAnswer = Callable[[list[str]], Awaitable[httpx.Response]]


def _ours(client_id: str, prompt_id: str = "ours-1") -> list[Any]:
    """A queue row as ComfyUI shapes it: number, prompt id, graph, extra data, outputs."""

    return [0, prompt_id, {}, {"client_id": client_id}, ["9"]]


def _theirs() -> list[Any]:
    """Another application's queue row, carrying the sentinel in every field but the client id."""

    return [
        1,
        f"prompt-{SENTINEL}",
        {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": SENTINEL}}},
        {"client_id": "another-client", "extra_pnginfo": {"workflow": {"notes": SENTINEL}}},
        [SENTINEL],
    ]


def _nameless() -> list[Any]:
    """A queue row whose extra data names no client."""

    return [2, "prompt-without-a-client", {}, {}, []]


def _queue(running: list[list[Any]], pending: list[list[Any]]) -> httpx.Response:
    return httpx.Response(200, json={"queue_running": running, "queue_pending": pending})


class _Comfy:
    """A ComfyUI holding prompts for this adapter and for other applications.

    A submission is accepted and, unless `finishes` names its prompt, never
    ends: a stop is acknowledged and changes nothing, and no record is written,
    as with a node that hangs. `remaining` answers the count, one value per
    request, with None for a count ComfyUI cannot give. `queue` answers the
    queue from the client ids the adapter submitted under, in order.
    """

    def __init__(
        self,
        *,
        remaining: list[int | None],
        queue: QueueAnswer,
        finishes: frozenset[str] = frozenset(),
    ) -> None:
        self.remaining = remaining
        self.queue = queue
        self.finishes = finishes
        self.client_ids: list[str] = []
        self.requests: list[str] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append(f"{request.method} {path}")
        if request.method == "POST" and path == "/prompt":
            self.client_ids.append(json.loads(request.content)["client_id"])
            prompt_id = f"ours-{len(self.client_ids)}"
            return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})
        if path == "/prompt":
            count = self.remaining.pop(0)
            if count is None:
                return httpx.Response(503)
            return httpx.Response(200, json={"exec_info": {"queue_remaining": count}})
        if request.method == "GET" and path == "/queue":
            return await self.queue(self.client_ids)
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            return httpx.Response(200, json={"cancelled": False})
        if path.startswith("/history/"):
            prompt_id = path.removeprefix("/history/")
            if prompt_id not in self.finishes:
                return httpx.Response(200, json={})
            image = {"filename": f"{prompt_id}.png", "subfolder": "", "type": "output"}
            return httpx.Response(200, json={prompt_id: {"outputs": {"9": {"images": [image]}}}})
        if path == "/view":
            return httpx.Response(200, content=_PNG, headers={"content-type": "image/png"})
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {path}")


class _Listening(_BlockingSocket):
    """A silent progress socket that says when its run has started listening."""

    def __init__(self, listening: asyncio.Event) -> None:
        self.listening = listening

    async def __anext__(self) -> str:
        self.listening.set()
        return await super().__anext__()


class _Finishing(_BlockingSocket):
    """A progress socket that reports its prompt finished."""

    def __init__(self, prompt_id: str) -> None:
        self.prompt_id = prompt_id

    async def __anext__(self) -> str:
        return json.dumps(
            {"type": "executing", "data": {"prompt_id": self.prompt_id, "node": None}}
        )


async def _adapter(
    monkeypatch: pytest.MonkeyPatch,
    comfy: _Comfy,
    *,
    sockets: Callable[[], _BlockingSocket] = _BlockingSocket,
    inactivity_seconds: float = 0.01,
) -> ComfyUIAdapter:
    monkeypatch.setattr(
        "local_lm.adapters.comfyui.websockets.connect", lambda *_args, **_kwargs: sockets()
    )
    adapter = ComfyUIAdapter("http://comfy.test", inactivity_seconds=inactivity_seconds)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(comfy)
    )
    return adapter


async def _leave_a_prompt_behind(adapter: ComfyUIAdapter) -> None:
    """Run a generation that hears nothing back, so it gives up on a prompt that stays queued."""

    with pytest.raises(RuntimeError, match="stopped reporting generation activity"):
        async for _event in adapter.generate(media_request(operation="text_to_image")):
            pass


async def _generate(adapter: ComfyUIAdapter, request: MediaRequest) -> list[MediaEvent]:
    return [event async for event in adapter.generate(request)]


def _logged(caplog: pytest.LogCaptureFixture) -> str:
    """Every captured record as a handler would write it, tracebacks included."""

    formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")
    return "\n".join(formatter.format(record) for record in caplog.records)


async def test_after_a_collected_generation_the_count_is_read_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A generation that collected its outputs left nothing of its own in ComfyUI.

    ComfyUI writes a prompt's record only once the prompt has left its queue,
    so every prompt the count names belongs to someone else, and the queue,
    which holds their graphs, is not read.
    """

    output = tmp_path / "output"
    output.mkdir()
    posted: list[dict[str, Any]] = []
    caching = _caching_comfy(output, posted)
    requests: list[str] = []

    async def comfy(request: httpx.Request) -> httpx.Response:
        requests.append(f"{request.method} {request.url.path}")
        if request.method == "GET" and request.url.path == "/prompt":
            return httpx.Response(200, json={"exec_info": {"queue_remaining": 2}})
        if request.url.path == "/queue":
            return _queue([_theirs()], [])
        response: httpx.Response = await caching(request)
        return response

    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", _socket_for(posted))
    adapter = ComfyUIAdapter("http://comfy.test", managed_output_root=output)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(comfy)
    )
    try:
        events = await _generate(
            adapter, media_request(operation="text_to_image", workflow=_workflow())
        )
        generated = len(requests)
        count = await adapter.foreign_prompt_count()
    finally:
        await adapter.close()

    assert events[-1].type == "complete", events[-1]
    assert count == 2
    assert requests[generated:] == ["GET /prompt"]


async def test_a_prompt_our_stopped_generation_left_running_is_not_counted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A prompt LM Atelier gave up on is its own, whatever ComfyUI goes on doing with it.

    The generation heard nothing and asked for its prompt to be stopped, but the
    prompt runs on, as one stuck in a node does. Of the three prompts ComfyUI
    holds, that one is LM Atelier's, and the other two are someone else's,
    including one whose row names no client. Only client ids are taken from the
    queue: the other application's graph and workflow appear in no log record
    and nowhere on the adapter.
    """

    caplog.set_level(logging.DEBUG)

    async def queue(client_ids: list[str]) -> httpx.Response:
        return _queue([_ours(client_ids[0])], [_theirs(), _nameless()])

    comfy = _Comfy(remaining=[3], queue=queue)
    adapter = await _adapter(monkeypatch, comfy)
    try:
        await _leave_a_prompt_behind(adapter)
        left = len(comfy.requests)
        count = await adapter.foreign_prompt_count()
        probed = comfy.requests[left:]
        clients = await adapter._queued_client_ids()
    finally:
        await adapter.close()

    assert count == 2
    assert probed == ["GET /prompt", "GET /queue"]
    assert clients == [comfy.client_ids[0], "another-client", None]
    # The capture recorded the queue read itself, so a logged sentinel would show.
    assert any("/queue" in record.getMessage() for record in caplog.records)
    assert SENTINEL not in _logged(caplog)
    assert SENTINEL not in repr(vars(adapter))


def _readable(client_ids: list[str]) -> dict[str, Any]:
    """The queue a reading would find: LM Atelier's prompt and two others."""

    return {"queue_running": [_ours(client_ids[0])], "queue_pending": [_theirs(), _nameless()]}


async def _error_status(client_ids: list[str]) -> httpx.Response:
    return httpx.Response(503, json=_readable(client_ids))


async def _malformed_json(client_ids: list[str]) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(_readable(client_ids)).encode()[:-1])


async def _not_an_object(client_ids: list[str]) -> httpx.Response:
    return httpx.Response(200, json=list(_readable(client_ids).values()))


async def _malformed_section(client_ids: list[str]) -> httpx.Response:
    # Skipping the section would leave only LM Atelier's own prompt to see.
    return httpx.Response(
        200, json={"queue_running": {"0": _theirs()}, "queue_pending": [_ours(client_ids[0])]}
    )


async def _missing_section(client_ids: list[str]) -> httpx.Response:
    return httpx.Response(200, json={"queue_pending": [_ours(client_ids[0])]})


async def _over_the_bound(client_ids: list[str]) -> httpx.Response:
    # The body is valid JSON padded past the bound, so only the bound keeps it unread.
    body = json.dumps(_readable(client_ids)).encode()
    padding = b" " * (comfyui_adapter._MAX_COMFY_JSON_BYTES - len(body) + 1)
    return httpx.Response(200, content=body + padding)


async def _no_answer(_client_ids: list[str]) -> httpx.Response:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


async def _connection_lost(_client_ids: list[str]) -> httpx.Response:
    raise httpx.RemoteProtocolError("connection lost")


@pytest.mark.parametrize(
    ("answer", "deadline"),
    [
        pytest.param(_error_status, PATIENCE_SECONDS, id="error-status"),
        pytest.param(_malformed_json, PATIENCE_SECONDS, id="malformed-json"),
        pytest.param(_not_an_object, PATIENCE_SECONDS, id="not-an-object"),
        pytest.param(_malformed_section, PATIENCE_SECONDS, id="malformed-section"),
        pytest.param(_missing_section, PATIENCE_SECONDS, id="missing-section"),
        pytest.param(_over_the_bound, PATIENCE_SECONDS, id="over-the-bound"),
        pytest.param(_no_answer, 0.05, id="no-answer"),
        pytest.param(_connection_lost, PATIENCE_SECONDS, id="connection-lost"),
    ],
)
async def test_an_unreadable_queue_counts_every_prompt_as_someone_elses(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    answer: QueueAnswer,
    deadline: float,
) -> None:
    """A queue that cannot be read tells nothing apart, so every prompt counts.

    Counting a prompt as someone else's at worst makes a caller wait; counting
    one as LM Atelier's own could end another application's work. A failed
    reading also proves nothing gone, so the next count reads the queue again.
    Nothing of the body is logged or kept, whatever went wrong with it.
    """

    caplog.set_level(logging.DEBUG)
    # Only the case that never answers may reach the deadline, so every other
    # case is refused for its own reason.
    monkeypatch.setattr(comfyui_adapter, "RECONCILE_SECONDS", deadline)
    answers = [answer]

    async def queue(client_ids: list[str]) -> httpx.Response:
        if answers:
            return await answers.pop()(client_ids)
        return httpx.Response(200, json=_readable(client_ids))

    comfy = _Comfy(remaining=[3, 3], queue=queue)
    adapter = await _adapter(monkeypatch, comfy)
    try:
        await _leave_a_prompt_behind(adapter)
        left = len(comfy.requests)
        unread = await asyncio.wait_for(adapter.foreign_prompt_count(), timeout=PATIENCE_SECONDS)
        monkeypatch.setattr(comfyui_adapter, "RECONCILE_SECONDS", PATIENCE_SECONDS)
        read = await adapter.foreign_prompt_count()
    finally:
        await adapter.close()

    assert unread == 3
    assert read == 2
    assert comfy.requests[left:] == ["GET /prompt", "GET /queue"] * 2
    assert SENTINEL not in _logged(caplog)
    assert SENTINEL not in repr(vars(adapter))


async def test_an_empty_comfyui_settles_what_was_left_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once ComfyUI holds nothing, nothing LM Atelier left behind is still there.

    Every prompt counted after that was queued later by someone else, so the
    queue is not read for it.
    """

    async def queue(client_ids: list[str]) -> httpx.Response:
        return _queue([_ours(client_ids[0])], [])

    comfy = _Comfy(remaining=[0, 1], queue=queue)
    adapter = await _adapter(monkeypatch, comfy)
    try:
        await _leave_a_prompt_behind(adapter)
        left = len(comfy.requests)
        counts = [await adapter.foreign_prompt_count(), await adapter.foreign_prompt_count()]
    finally:
        await adapter.close()

    assert counts == [0, 1]
    assert comfy.requests[left:] == ["GET /prompt", "GET /prompt"]


async def test_prompts_left_by_several_generations_stay_ours_until_the_queue_lists_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each prompt LM Atelier left behind is its own, whichever generation sent it.

    While the queue still lists one of them, the next count reads the queue
    again. Once it lists none, they have all left, and later counts are read
    alone.
    """

    readings: list[str] = []

    async def queue(client_ids: list[str]) -> httpx.Response:
        readings.append("queue")
        if len(readings) == 1:
            return _queue(
                [_ours(client_ids[0], "ours-1")], [_ours(client_ids[1], "ours-2"), _theirs()]
            )
        return _queue([_theirs()], [])

    comfy = _Comfy(remaining=[3, 1, 1], queue=queue)
    adapter = await _adapter(monkeypatch, comfy)
    try:
        await _leave_a_prompt_behind(adapter)
        await _leave_a_prompt_behind(adapter)
        left = len(comfy.requests)
        counts = [await adapter.foreign_prompt_count() for _ in range(3)]
    finally:
        await adapter.close()

    assert len(set(comfy.client_ids)) == 2, "each submission names itself"
    assert counts == [1, 1, 1]
    assert comfy.requests[left:] == [
        "GET /prompt",
        "GET /queue",
        "GET /prompt",
        "GET /queue",
        "GET /prompt",
    ]


@pytest.mark.parametrize(
    "row",
    [
        pytest.param("not-a-row", id="not-a-list"),
        pytest.param([3, "prompt-short"], id="too-short"),
        pytest.param([3, "prompt-odd", {}, "not-extra-data", []], id="extra-data-not-an-object"),
        pytest.param([3, "prompt-odd", {}, {"client_id": 7}, []], id="client-id-not-a-string"),
    ],
)
async def test_a_row_that_names_no_client_counts_as_someone_elses(
    monkeypatch: pytest.MonkeyPatch, row: Any
) -> None:
    """A row that cannot be shown to be LM Atelier's is someone else's."""

    async def queue(client_ids: list[str]) -> httpx.Response:
        return _queue([_ours(client_ids[0])], [row])

    comfy = _Comfy(remaining=[2], queue=queue)
    adapter = await _adapter(monkeypatch, comfy)
    try:
        await _leave_a_prompt_behind(adapter)
        count = await adapter.foreign_prompt_count()
    finally:
        await adapter.close()

    assert count == 1


async def test_a_count_comfyui_cannot_give_is_not_read_from_the_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ComfyUI cannot say how many prompts it holds, neither can the adapter."""

    async def queue(client_ids: list[str]) -> httpx.Response:
        return _queue([_ours(client_ids[0])], [_theirs()])

    comfy = _Comfy(remaining=[None], queue=queue)
    adapter = await _adapter(monkeypatch, comfy)
    try:
        await _leave_a_prompt_behind(adapter)
        left = len(comfy.requests)
        count = await adapter.foreign_prompt_count()
    finally:
        await adapter.close()

    assert count is None
    assert comfy.requests[left:] == ["GET /prompt"]


async def test_a_running_generation_has_its_prompt_told_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While a generation waits its turn, its prompt is in ComfyUI and is LM Atelier's own."""

    listening = asyncio.Event()

    async def queue(client_ids: list[str]) -> httpx.Response:
        return _queue([_theirs()], [_ours(client_ids[0])])

    comfy = _Comfy(remaining=[2], queue=queue)
    adapter = await _adapter(
        monkeypatch, comfy, sockets=lambda: _Listening(listening), inactivity_seconds=60
    )
    request = media_request(operation="text_to_image")
    generating = asyncio.create_task(_generate(adapter, request))
    try:
        await asyncio.wait_for(listening.wait(), timeout=PATIENCE_SECONDS)
        submitted = len(comfy.requests)
        count = await adapter.foreign_prompt_count()
        probed = comfy.requests[submitted:]
        await adapter.cancel(request.run_id)
        events = await asyncio.wait_for(generating, timeout=PATIENCE_SECONDS)
    finally:
        generating.cancel()
        await asyncio.gather(generating, return_exceptions=True)
        await adapter.close()

    assert count == 1
    assert probed == ["GET /prompt", "GET /queue"]
    assert events[-1].type == "cancelled"


async def test_a_prompt_still_running_for_a_superseded_generation_is_told_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newer generation for the same run can finish while the older one's prompt waits.

    The newer one collected its outputs, and the older one is still running, so
    its prompt is in ComfyUI, and it is LM Atelier's own, not someone else's.
    """

    listening = asyncio.Event()
    sockets: list[_BlockingSocket] = [_Listening(listening), _Finishing("ours-2")]

    async def queue(client_ids: list[str]) -> httpx.Response:
        return _queue([_ours(client_ids[0], "ours-1")], [_theirs()])

    comfy = _Comfy(remaining=[2], queue=queue, finishes=frozenset({"ours-2"}))
    adapter = await _adapter(
        monkeypatch, comfy, sockets=lambda: sockets.pop(0), inactivity_seconds=60
    )
    request = media_request(operation="text_to_image")
    older = asyncio.create_task(_generate(adapter, request))
    try:
        await asyncio.wait_for(listening.wait(), timeout=PATIENCE_SECONDS)
        newer = await asyncio.wait_for(_generate(adapter, request), timeout=PATIENCE_SECONDS)
        finished = len(comfy.requests)
        count = await adapter.foreign_prompt_count()
        probed = comfy.requests[finished:]
        assert not older.done(), "the older generation ended before the count was read"
    finally:
        older.cancel()
        await asyncio.gather(older, return_exceptions=True)
        await adapter.close()

    assert newer[-1].type == "complete", newer[-1]
    assert count == 1
    assert probed == ["GET /prompt", "GET /queue"]
