from __future__ import annotations

import json

import httpx

from local_lm.adapters.base import ChatRequest
from local_lm.adapters.llama_cpp import LlamaCppAdapter


async def test_llama_adapter_uses_chat_template_token_count_endpoint() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions/input_tokens"
        payload = json.loads(request.content)
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
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
