from __future__ import annotations

import asyncio
import json

import httpx

from local_lm.adapters.base import ChatRequest, estimate_chat_tokens
from local_lm.adapters.contracts import MAX_ADAPTER_EVENT_BYTES
from local_lm.adapters.llama_cpp import LlamaCppAdapter


async def test_llama_adapter_uses_chat_template_token_count_endpoint() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions/input_tokens"
        payload = json.loads(request.content)
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(200, json={"input_tokens": 7})

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test", transport=httpx.MockTransport(handler)
    )
    try:
        assert await adapter.count_tokens([{"role": "user", "content": "Hello"}]) == 7
    finally:
        await adapter.close()


async def test_llama_adapter_falls_back_when_tokenizer_routes_are_unavailable() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test", transport=httpx.MockTransport(handler)
    )
    try:
        count = await adapter.count_tokens([{"role": "user", "content": "A" * 90}])
        assert count >= 36
    finally:
        await adapter.close()


async def test_llama_adapter_advertises_vision_only_when_props_confirms_it() -> None:
    requested: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "version": "b9999"})
        assert request.url.path == "/props"
        return httpx.Response(
            200,
            json={
                "modalities": {"vision": True, "audio": False},
                "model_path": "C:/private/models/model.gguf",
            },
        )

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test", transport=httpx.MockTransport(handler)
    )
    try:
        capabilities = await adapter.capabilities()
    finally:
        await adapter.close()

    assert requested == ["/health", "/props"]
    assert capabilities.healthy is True
    assert capabilities.input_modalities == ["text", "image"]
    assert capabilities.details == {"status": "ok", "version": "b9999"}
    assert "model_path" not in capabilities.details


async def test_llama_adapter_does_not_assume_vision_when_props_is_unavailable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test", transport=httpx.MockTransport(handler)
    )
    try:
        capabilities = await adapter.capabilities()
    finally:
        await adapter.close()

    assert capabilities.healthy is True
    assert capabilities.input_modalities == ["text"]


async def test_llama_adapter_safely_ignores_untrusted_props_metadata() -> None:
    cases = ("error", "malformed", "oversized", "recursive", "wrong-modality")
    for case in cases:

        async def handler(
            request: httpx.Request,
            selected_case: str = case,
        ) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok", "version": "safe"})
            assert request.url.path == "/props"
            if selected_case == "error":
                return httpx.Response(500, text="C:/private/models/secret.gguf")
            if selected_case == "malformed":
                return httpx.Response(200, content=b"{")
            if selected_case == "oversized":
                return httpx.Response(200, content=b"x" * (MAX_ADAPTER_EVENT_BYTES + 1))
            if selected_case == "recursive":
                return httpx.Response(
                    200,
                    content=(b'{"modalities":' + (b"[" * 2_000) + b"0" + (b"]" * 2_000) + b"}"),
                )
            return httpx.Response(
                200,
                json={
                    "modalities": {"vision": "true"},
                    "model_path": "C:/private/models/secret.gguf",
                },
            )

        adapter = LlamaCppAdapter("http://llama.test")
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            base_url="http://llama.test",
            transport=httpx.MockTransport(handler),
        )
        try:
            capabilities = await adapter.capabilities()
        finally:
            await adapter.close()

        assert capabilities.healthy is True
        assert capabilities.input_modalities == ["text"]
        assert capabilities.details == {"status": "ok", "version": "safe"}
        assert "private" not in json.dumps(capabilities.model_dump())


async def test_llama_adapter_rejects_invalid_backend_token_counts() -> None:
    requested: list[str] = []
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{'A' * 10_000}"},
                },
            ],
        }
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/v1/chat/completions/input_tokens":
            return httpx.Response(200, json={"input_tokens": True})
        assert request.url.path == "/apply-template"
        return httpx.Response(200, json={"prompt": []})

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        count = await adapter.count_tokens(messages)
    finally:
        await adapter.close()

    assert requested == ["/v1/chat/completions/input_tokens", "/apply-template"]
    assert count == estimate_chat_tokens(messages)
    assert count >= 1_024


async def test_llama_adapter_sends_tools_and_streams_structured_deltas() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["function"]["name"] == "choose_route"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        stream = "".join(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":',
                '{"name":"choose_route","arguments":"{\\"mode\\":\\"image\\""}}]}}]}\n\n',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":',
                '{"arguments":",\\"confidence\\":1}"}}]},"finish_reason":"tool_calls"}]}\n\n',
                "data: [DONE]\n\n",
            ]
        )
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test", transport=httpx.MockTransport(handler)
    )
    try:
        events = [
            event
            async for event in adapter.stream(
                ChatRequest(
                    run_id="probe",
                    messages=[{"role": "user", "content": "Make an image"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "choose_route",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                )
            )
        ]
        tool_events = [event for event in events if event.type == "tool_delta"]
        assert len(tool_events) == 2
        assert events[-1].type == "complete"
        assert events[-1].data["finish_reason"] == "tool_calls"
    finally:
        await adapter.close()


async def test_llama_adapter_completes_when_terminal_choice_does_not_close_stream() -> None:
    class NeverClosingTerminalStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
            yield (
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"completion_tokens":1}}\n\n'
            )
            await asyncio.Event().wait()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=NeverClosingTerminalStream(),
            headers={"content-type": "text/event-stream"},
        )

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test", transport=httpx.MockTransport(handler)
    )
    try:
        events = await asyncio.wait_for(
            _collect_events(
                adapter,
                ChatRequest(
                    run_id="never-closing-terminal",
                    messages=[{"role": "user", "content": "Finish"}],
                ),
            ),
            timeout=0.5,
        )
        assert [event.type for event in events] == ["delta", "usage", "complete"]
        assert events[-1].data["finish_reason"] == "stop"
    finally:
        await adapter.close()


async def _collect_events(adapter: LlamaCppAdapter, request: ChatRequest):  # type: ignore[no-untyped-def]
    return [event async for event in adapter.stream(request)]


async def test_llama_adapter_names_an_http_error_after_partial_stream_output() -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=BrokenStream(),
            headers={"content-type": "text/event-stream"},
        )

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test", transport=httpx.MockTransport(handler)
    )
    try:
        events = [
            event
            async for event in adapter.stream(
                ChatRequest(
                    run_id="failed-stream",
                    messages=[{"role": "user", "content": "Keep this partial output"}],
                )
            )
        ]
        assert len(events) == 2
        assert events[0].type == "delta"
        assert events[0].text == "partial"
        assert events[1].type == "error"
        assert events[1].data["error"] == "llama.cpp connection closed before generation completed"
    finally:
        await adapter.close()


async def test_llama_adapter_recovers_when_only_length_terminal_frames_are_lost() -> None:
    class TerminalBrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
            raise httpx.ReadError("")

    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            stream=TerminalBrokenStream(),
            headers={"content-type": "text/event-stream"},
        )

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test", transport=httpx.MockTransport(handler)
    )
    try:
        events = [
            event
            async for event in adapter.stream(
                ChatRequest(
                    run_id="lost-terminal",
                    messages=[{"role": "user", "content": "Generate two tokens"}],
                    settings={"max_tokens": 2},
                )
            )
        ]
        assert calls == 1
        assert [event.text for event in events if event.type == "delta"] == ["a", "b"]
        assert events[-1].type == "complete"
        assert events[-1].data["finish_reason"] == "length"
    finally:
        await adapter.close()


async def test_llama_adapter_retries_a_fixed_seed_truncated_stream_once() -> None:
    class TruncatedStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            raise httpx.ReadError("")

    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                stream=TruncatedStream(),
                headers={"content-type": "text/event-stream"},
            )
        stream = "".join(
            [
                'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
                "data: [DONE]\n\n",
            ]
        )
        return httpx.Response(
            200,
            text=stream,
            headers={"content-type": "text/event-stream"},
        )

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test", transport=httpx.MockTransport(handler)
    )
    try:
        events = [
            event
            async for event in adapter.stream(
                ChatRequest(
                    run_id="retry-truncated",
                    messages=[{"role": "user", "content": "Generate two tokens"}],
                    settings={"seed": 42, "max_tokens": 2},
                )
            )
        ]
        assert calls == 2
        assert [event.text for event in events if event.type == "delta"] == ["a", "b"]
        assert events[-1].type == "complete"
        assert events[-1].data["finish_reason"] == "length"
    finally:
        await adapter.close()


async def test_llama_adapter_does_not_merge_a_mismatched_retry() -> None:
    class TruncatedStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            raise httpx.ReadError("")

    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                stream=TruncatedStream(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"z"}}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test", transport=httpx.MockTransport(handler)
    )
    try:
        events = [
            event
            async for event in adapter.stream(
                ChatRequest(
                    run_id="mismatched-retry",
                    messages=[{"role": "user", "content": "Generate two tokens"}],
                    settings={"seed": 42, "max_tokens": 2},
                )
            )
        ]
        assert calls == 2
        assert [event.text for event in events if event.type == "delta"] == ["a"]
        assert events[-1].type == "error"
        assert events[-1].data["error"] == "llama.cpp returned an invalid streaming response"
    finally:
        await adapter.close()


async def test_llama_adapter_treats_clean_premature_eof_as_an_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        events = [
            event async for event in adapter.stream(ChatRequest(run_id="clean-eof", messages=[]))
        ]
    finally:
        await adapter.close()

    assert [event.type for event in events] == ["delta", "error"]
    assert events[-1].data["error"] == ("llama.cpp connection closed before generation completed")


async def test_llama_adapter_rejects_malformed_and_oversized_sse_events() -> None:
    responses = iter(
        [
            "data: {not-json}\n\n",
            f"data: {'x' * (1024 * 1024)}\n\n",
        ]
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=next(responses),
            headers={"content-type": "text/event-stream"},
        )

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        malformed = [
            event async for event in adapter.stream(ChatRequest(run_id="malformed", messages=[]))
        ]
        oversized = [
            event async for event in adapter.stream(ChatRequest(run_id="oversized", messages=[]))
        ]
    finally:
        await adapter.close()

    assert [event.type for event in malformed] == ["error"]
    assert [event.type for event in oversized] == ["error"]
    assert malformed[-1].data["error"] == "llama.cpp returned an invalid streaming response"
    assert oversized[-1].data["error"] == "llama.cpp returned an invalid streaming response"


async def test_llama_adapter_cancel_wakes_a_blocked_stream() -> None:
    receiving = asyncio.Event()

    class StalledStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            receiving.set()
            await asyncio.Event().wait()
            yield b"unreachable"

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=StalledStream(),
            headers={"content-type": "text/event-stream"},
        )

    adapter = LlamaCppAdapter("http://llama.test", inactivity_seconds=60)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test",
        transport=httpx.MockTransport(handler),
    )
    request = ChatRequest(run_id="cancel-blocked", messages=[])
    collecting = asyncio.create_task(_collect_events(adapter, request))
    try:
        await asyncio.wait_for(receiving.wait(), timeout=0.5)
        await adapter.cancel(request.run_id)
        events = await asyncio.wait_for(collecting, timeout=0.5)
    finally:
        if not collecting.done():
            collecting.cancel()
        await adapter.close()

    assert [event.type for event in events] == ["cancelled"]


async def test_llama_adapter_honors_cancellation_before_stream_start() -> None:
    adapter = LlamaCppAdapter("http://llama.test")
    request = ChatRequest(run_id="cancel-before-start", messages=[])
    try:
        await adapter.cancel(request.run_id)
        events = [event async for event in adapter.stream(request)]
    finally:
        await adapter.close()

    assert [event.type for event in events] == ["cancelled"]


async def test_llama_adapter_redacts_http_error_details() -> None:
    secret = "backend-private-detail"

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=secret)

    adapter = LlamaCppAdapter("http://llama.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        events = [
            event async for event in adapter.stream(ChatRequest(run_id="http-error", messages=[]))
        ]
    finally:
        await adapter.close()

    assert [event.type for event in events] == ["error"]
    assert events[0].data["error"] == ("llama.cpp rejected the generation request (HTTP 400)")
    assert secret not in events[0].data["error"]
    assert "http://llama.test" not in events[0].data["error"]


async def test_llama_adapter_stops_a_silent_stream_after_inactivity() -> None:
    class StalledStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            await asyncio.Event().wait()
            yield b"unreachable"

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=StalledStream(),
            headers={"content-type": "text/event-stream"},
        )

    adapter = LlamaCppAdapter("http://llama.test", inactivity_seconds=0.01)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://llama.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        events = await asyncio.wait_for(
            _collect_events(
                adapter,
                ChatRequest(run_id="silent-stream", messages=[]),
            ),
            timeout=0.5,
        )
    finally:
        await adapter.close()

    assert [event.type for event in events] == ["error"]
    assert events[0].data["error"] == "llama.cpp stopped reporting generation activity"
