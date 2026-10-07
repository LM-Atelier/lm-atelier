"""A linked page is read only as far, and for only as long, as a page may take."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_api import _claimed_text_job

from local_lm import orchestrator as orchestrator_module
from local_lm.db import SessionLocal
from local_lm.models import Job
from local_lm.network import outbound_client
from local_lm.web_lookup import LookupRequest
from local_lm.web_retrieval import MAX_CONTENT_BYTES, WebRetrievalError, bounded_request

PAGE = "https://page.example.test/notes"
CHUNK = 32 * 1024


class _Body(httpx.AsyncByteStream):
    """A body of many chunks that counts how many were taken, and can stall."""

    def __init__(self, chunks: int, *, stall_after: int | None = None) -> None:
        self.chunks = chunks
        self.stall_after = stall_after
        self.taken = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index in range(self.chunks):
            if index == self.stall_after:
                await asyncio.Event().wait()
            self.taken += 1
            yield b"word " * (CHUNK // 5) + b"x" * (CHUNK % 5)


class _Chunk(bytes):
    """Bytes that record each slice taken of them."""

    slices: list[int]

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        if isinstance(key, slice):
            self.slices.append(len(value))
        return value


class _OneChunk(httpx.AsyncByteStream):
    """A body sent as a single chunk far larger than a page."""

    def __init__(self, size: int) -> None:
        self.chunk = _Chunk(b"x" * size)
        self.chunk.slices = []

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.chunk


def _page(body: httpx.AsyncByteStream) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, stream=body)
    )


async def test_a_long_body_is_read_one_byte_past_what_a_page_may_be() -> None:
    body = _Body(64)
    async with httpx.AsyncClient(transport=_page(body)) as client:
        response = await bounded_request(client)(PAGE)

    assert len(response.content) == MAX_CONTENT_BYTES + 1
    assert body.taken == MAX_CONTENT_BYTES // CHUNK + 1


async def test_a_redirect_s_body_is_never_read() -> None:
    body = _Body(64)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(302, headers={"location": PAGE + "/moved"}, stream=body)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        response = await bounded_request(client)(PAGE)

    assert (response.status_code, response.headers["location"]) == (302, PAGE + "/moved")
    assert (response.content, body.taken) == (b"", 0)


@pytest.mark.parametrize("encoding", ["gzip", "br", "identity, gzip", "GZIP"])
async def test_a_compressed_body_is_refused_before_any_of_it_is_read(encoding: str) -> None:
    body = _Body(64)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/plain", "content-encoding": encoding}, stream=body
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(WebRetrievalError) as refused:
            await bounded_request(client)(PAGE)

    assert refused.value.code == "web-encoding-refused"
    assert body.taken == 0


async def test_an_oversized_chunk_is_kept_only_to_the_limit() -> None:
    oversized = _OneChunk(4 * MAX_CONTENT_BYTES)
    async with httpx.AsyncClient(transport=_page(oversized)) as client:
        response = await bounded_request(client)(PAGE)

    assert len(response.content) == MAX_CONTENT_BYTES + 1
    # Only the part a page may hold is copied out of the chunk, never all of it.
    assert oversized.chunk.slices == [MAX_CONTENT_BYTES + 1]


async def _read_through_the_page_reader(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    body: _Body,
    *,
    deadline: float | None = None,
) -> dict[str, Any] | None:
    orch = app.state.services.orchestrator
    monkeypatch.setattr(type(orch), "start", lambda self, *args: None)
    _chat_id, job_id, claim = await _claimed_text_job(client, monkeypatch)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None and job.run_id is not None
        run_id = job.run_id

    async def choose(*args: object, **kwargs: object) -> LookupRequest:
        return LookupRequest(url=PAGE, reason="the person linked it")

    def public(host: object, *args: object, **kwargs: object) -> list[tuple[Any, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    real_client = outbound_client

    def served(lease: Any, **kwargs: Any) -> httpx.AsyncClient:
        return real_client(lease, transport=_page(body), **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", public)
    monkeypatch.setattr(orchestrator_module, "outbound_client", served)
    monkeypatch.setitem(orch._read_linked_page.__globals__, "choose_from_conversation", choose)
    monkeypatch.setattr(orch.engines.settings, "web_access_enabled", True)
    if deadline is not None:
        # After the claim: taking it undoes every patch made before it.
        monkeypatch.setattr(orchestrator_module, "PAGE_DEADLINE_SECONDS", deadline, raising=False)
    messages = [{"role": "user", "content": f"Read {PAGE}"}]
    # Bounded here as well, so a read that never ends fails rather than hangs.
    return await asyncio.wait_for(
        orch._read_linked_page(messages, run_id, job_id, claim), timeout=10
    )


async def test_the_page_reader_stops_reading_a_long_page_at_its_limit(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _Body(64)

    provenance = await _read_through_the_page_reader(client, app, monkeypatch, body)

    assert provenance is not None and provenance["truncated"] is True
    assert body.taken == MAX_CONTENT_BYTES // CHUNK + 1


async def test_a_page_that_stops_sending_ends_at_the_deadline(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _Body(64, stall_after=2)

    provenance = await _read_through_the_page_reader(client, app, monkeypatch, body, deadline=0.2)

    assert provenance == {
        "url": PAGE,
        "reason": "the person linked it",
        "refused": "web-unreachable",
    }
    assert body.taken == 2
