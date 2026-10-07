"""Abandoned prompts use the backend's atomic cancel-by-ID contract."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from run_waits import PATIENCE_SECONDS

from local_lm.adapters import comfyui as comfyui_module
from local_lm.adapters.comfyui import ComfyUIAdapter


@pytest.mark.parametrize("cancelled", [False, True])
async def test_atomic_prompt_cleanup_does_not_fall_back_after_a_supported_reply(
    cancelled: bool,
) -> None:
    submitted = "neutral-submitted"
    requests: list[tuple[str, str]] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == f"/api/jobs/{submitted}/cancel":
            return httpx.Response(200, json={"cancelled": cancelled})
        return httpx.Response(200, json={})

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(backend)
    )
    try:
        await adapter._abandon_prompt(submitted)
    finally:
        await adapter.close()
    assert requests == [("POST", f"/api/jobs/{submitted}/cancel")]


@pytest.mark.parametrize("status", [404, 405])
async def test_unavailable_atomic_cancellation_retains_scoped_legacy_cleanup(status: int) -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(status if request.url.path.startswith("/api/jobs/") else 200)

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(backend)
    )
    try:
        await adapter._abandon_prompt("neutral-submitted")
    finally:
        await adapter.close()
    assert requests == [
        ("/api/jobs/neutral-submitted/cancel", {}),
        ("/queue", {"delete": ["neutral-submitted"]}),
        ("/interrupt", {"prompt_id": "neutral-submitted"}),
    ]


@pytest.mark.parametrize("status", [302, 400, 429, 500, 503])
async def test_atomic_cancellation_errors_do_not_attempt_a_less_precise_interrupt(
    status: int,
) -> None:
    requests: list[str] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(status)

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(backend)
    )
    try:
        await adapter._abandon_prompt("neutral-submitted")
    finally:
        await adapter.close()
    assert requests == ["/api/jobs/neutral-submitted/cancel"]


@pytest.mark.parametrize(
    ("identity", "path"),
    [
        ("neutral/submitted", b"/api/jobs/neutral%2Fsubmitted/cancel"),
        ("neutral?#", b"/api/jobs/neutral%3F%23/cancel"),
        ("neutral%2Fsubmitted", b"/api/jobs/neutral%252Fsubmitted/cancel"),
        (".", b"/api/jobs/%2E/cancel"),
        ("..", b"/api/jobs/%2E%2E/cancel"),
    ],
)
async def test_prompt_identity_cannot_change_the_cancellation_route(
    identity: str, path: bytes
) -> None:
    requests: list[bytes] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.raw_path)
        return httpx.Response(200, json={"cancelled": False})

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(backend)
    )
    try:
        await adapter._abandon_prompt(identity)
    finally:
        await adapter.close()
    assert requests == [path]


@pytest.mark.parametrize("legacy", [False, True])
async def test_cancellation_deadline_covers_atomic_and_legacy_requests(
    monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    monkeypatch.setattr(comfyui_module, "ABANDONED_INTERRUPT_SECONDS", 0.05)
    requests: list[str] = []
    unwound = asyncio.Event()

    async def backend(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if legacy and request.url.path.startswith("/api/jobs/"):
            return httpx.Response(404)
        try:
            await asyncio.Event().wait()
        finally:
            unwound.set()
        raise AssertionError("A blocked cancellation never returns a response")

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(backend)
    )
    try:
        await asyncio.wait_for(adapter._abandon_prompt("neutral-submitted"), PATIENCE_SECONDS)
    finally:
        await adapter.close()
    assert unwound.is_set()
    assert requests == ["/api/jobs/neutral-submitted/cancel", *(["/queue"] if legacy else [])]
