from __future__ import annotations

import httpx

from local_lm.adapters.vllm import VllmAdapter


async def test_vllm_adapter_reports_modelopt_vision_contract() -> None:
    requested: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(404)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.25.0"})
        raise AssertionError(f"unexpected vLLM request: {request.url.path}")

    adapter = VllmAdapter("http://vllm.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://vllm.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        capabilities = await adapter.capabilities()
    finally:
        await adapter.close()

    assert requested == ["/health", "/props", "/version"]
    assert capabilities.engine == "vllm"
    assert capabilities.version == "0.25.0"
    assert capabilities.healthy is True
    assert capabilities.input_modalities == ["text", "image"]
    assert capabilities.formats == ["safetensors", "modelopt"]
    assert capabilities.details["quantization"] == "modelopt"


async def test_vllm_adapter_does_not_expose_untrusted_health_metadata() -> None:
    private_value = "C:/private/model/snapshot"

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(503, text=private_value)
        raise AssertionError("unhealthy vLLM must not probe more metadata")

    adapter = VllmAdapter("http://vllm.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://vllm.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        capabilities = await adapter.capabilities()
    finally:
        await adapter.close()

    assert capabilities.healthy is False
    assert capabilities.details == {
        "protocol": "openai",
        "quantization": "modelopt",
        "error": "vLLM health check failed",
    }
    assert private_value not in capabilities.model_dump_json()
