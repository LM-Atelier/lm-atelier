from __future__ import annotations

import json
from typing import Any

import httpx

from ..schemas import EngineCapabilities
from .llama_cpp import LlamaCppAdapter, _bounded_response_json


class VllmAdapter(LlamaCppAdapter):
    """OpenAI-compatible vLLM transport with explicit ModelOpt capabilities."""

    async def capabilities(self) -> EngineCapabilities:
        base = await super().capabilities()
        version = "unknown"
        details: dict[str, Any] = {
            "protocol": "openai",
            "quantization": "modelopt",
        }
        if base.healthy:
            try:
                async with self._client.stream("GET", "/version", timeout=3) as response:
                    response.raise_for_status()
                    value = await _bounded_response_json(response)
                if isinstance(value, dict):
                    json.dumps(value, allow_nan=False)
                    details.update(value)
                    version = str(value.get("version", "unknown"))[:200]
            except (
                httpx.HTTPError,
                TypeError,
                ValueError,
                OverflowError,
                RecursionError,
                UnicodeError,
            ):
                details["version_probe"] = "unavailable"
        else:
            details["error"] = "vLLM health check failed"
        return base.model_copy(
            update={
                "engine": "vllm",
                "version": version,
                "input_modalities": ["text", "image"],
                "formats": ["safetensors", "modelopt"],
                "details": details,
            }
        )
