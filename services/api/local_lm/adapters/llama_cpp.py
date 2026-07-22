from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..domain import Operation
from ..schemas import EngineCapabilities
from ..settings_registry import CHAT_SETTINGS
from .base import ChatEvent, ChatRequest


class LlamaCppAdapter:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=None)
        self._cancelled: set[str] = set()

    async def capabilities(self) -> EngineCapabilities:
        healthy = False
        details: dict[str, Any] = {}
        version = "unknown"
        try:
            response = await self._client.get("/health", timeout=3)
            healthy = response.is_success
            if response.headers.get("content-type", "").startswith("application/json"):
                details = response.json()
            version = str(details.get("version", "unknown"))
        except httpx.HTTPError as exc:
            details = {"error": str(exc)}
        return EngineCapabilities(
            engine="llama.cpp",
            version=version,
            roles=["chat"],
            operations=[Operation.TEXT.value],
            formats=["gguf"],
            devices=[],
            streaming=True,
            tool_calling=True,
            settings=CHAT_SETTINGS,
            healthy=healthy,
            details=details,
        )

    @staticmethod
    def _request_settings(settings: dict[str, Any]) -> dict[str, Any]:
        mapping = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
            "repeat_penalty": "repeat_penalty",
            "repeat_last_n": "repeat_last_n",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
            "typical_p": "typical_p",
            "seed": "seed",
            "max_tokens": "max_tokens",
            "stop": "stop",
        }
        return {
            target: settings[source] for source, target in mapping.items() if source in settings
        }

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        payload: dict[str, Any] = {
            "model": "local-model",
            "messages": request.messages,
            "stream": True,
            **self._request_settings(request.settings),
        }
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = "auto"

        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if request.run_id in self._cancelled:
                        self._cancelled.discard(request.run_id)
                        yield ChatEvent(type="cancelled")
                        return
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        yield ChatEvent(type="complete", data={"finish_reason": "stop"})
                        return
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    if content := delta.get("content"):
                        yield ChatEvent(type="delta", text=str(content))
                    if tool_calls := delta.get("tool_calls"):
                        yield ChatEvent(type="tool_delta", data={"tool_calls": tool_calls})
                    if finish_reason := choice.get("finish_reason"):
                        yield ChatEvent(type="complete", data={"finish_reason": finish_reason})
                        return
        except httpx.HTTPError as exc:
            yield ChatEvent(type="error", data={"error": str(exc)})

    async def cancel(self, run_id: str) -> None:
        self._cancelled.add(run_id)

    async def close(self) -> None:
        await self._client.aclose()
