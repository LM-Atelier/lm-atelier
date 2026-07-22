from __future__ import annotations

import json

import httpx

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
