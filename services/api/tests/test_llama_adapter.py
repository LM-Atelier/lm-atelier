from __future__ import annotations

import asyncio
import json

import httpx

from local_lm.adapters.base import ChatRequest
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
        assert events[1].data["error"] == "llama.cpp stream failed: ReadError"
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
        assert events[-1].data["error"] == "llama.cpp stream failed: ReadError"
    finally:
        await adapter.close()
