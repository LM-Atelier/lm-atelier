from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..domain import Operation
from ..schemas import EngineCapabilities
from ..settings_registry import CHAT_SETTINGS
from .base import ChatEvent, ChatRequest, estimate_chat_tokens


class LlamaCppAdapter:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10, read=None, write=30, pool=10),
        )
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
            settings_by_role={"chat": CHAT_SETTINGS},
            healthy=healthy,
            details=details,
        )

    async def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        payload = {"model": "local-model", "messages": messages}
        try:
            response = await self._client.post(
                "/v1/chat/completions/input_tokens", json=payload, timeout=10
            )
            response.raise_for_status()
            count = int(response.json()["input_tokens"])
            if count > 0:
                return count
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            pass

        try:
            templated = await self._client.post(
                "/apply-template", json={"messages": messages}, timeout=10
            )
            templated.raise_for_status()
            prompt = str(templated.json()["prompt"])
            tokenized = await self._client.post(
                "/tokenize", json={"content": prompt, "add_special": True}, timeout=10
            )
            tokenized.raise_for_status()
            tokens = tokenized.json()["tokens"]
            if isinstance(tokens, list) and tokens:
                return len(tokens)
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            pass
        return estimate_chat_tokens(messages)

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
        self._cancelled.discard(request.run_id)
        payload: dict[str, Any] = {
            "model": "local-model",
            "messages": request.messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._request_settings(request.settings),
        }
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = "auto"

        seed = request.settings.get("seed")
        max_tokens = request.settings.get("max_tokens")
        deterministic = (
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 and not request.tools
        )
        delivered_text = ""
        retry_prefix: str | None = None

        while True:
            finish_reason: str | None = None
            completion_frames = 0
            attempt_text = ""
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
                            if retry_prefix is not None and len(attempt_text) < len(retry_prefix):
                                yield self._stream_error(httpx.ReadError(""))
                                return
                            yield ChatEvent(
                                type="complete",
                                data={"finish_reason": finish_reason or "stop"},
                            )
                            return
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        metadata = {
                            key: chunk[key] for key in ("usage", "timings") if chunk.get(key)
                        }
                        if metadata:
                            yield ChatEvent(type="usage", data=metadata)
                        choice = (chunk.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        tool_calls = delta.get("tool_calls")
                        if content is not None or tool_calls is not None:
                            completion_frames += 1
                        if content is not None:
                            text = str(content)
                            if retry_prefix is None:
                                delivered_text += text
                                if text:
                                    yield ChatEvent(type="delta", text=text)
                            else:
                                previous_length = len(attempt_text)
                                attempt_text += text
                                if not (
                                    retry_prefix.startswith(attempt_text)
                                    or attempt_text.startswith(retry_prefix)
                                ):
                                    yield self._stream_error(httpx.ReadError(""))
                                    return
                                suffix_start = max(previous_length, len(retry_prefix))
                                if suffix_start < len(attempt_text):
                                    suffix = attempt_text[suffix_start:]
                                    delivered_text += suffix
                                    yield ChatEvent(type="delta", text=suffix)
                        if tool_calls:
                            yield ChatEvent(type="tool_delta", data={"tool_calls": tool_calls})
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                    yield ChatEvent(
                        type="complete",
                        data={"finish_reason": finish_reason or "stop"},
                    )
                    return
            except httpx.ReadError as exc:
                hit_output_limit = (
                    isinstance(max_tokens, int)
                    and not isinstance(max_tokens, bool)
                    and max_tokens > 0
                    and completion_frames >= max_tokens
                )
                if finish_reason is not None or hit_output_limit:
                    yield ChatEvent(
                        type="complete",
                        data={"finish_reason": finish_reason or "length"},
                    )
                    return
                if deterministic and retry_prefix is None:
                    retry_prefix = delivered_text
                    continue
                yield self._stream_error(exc)
                return
            except httpx.HTTPError as exc:
                yield self._stream_error(exc)
                return

    @staticmethod
    def _stream_error(exc: httpx.HTTPError) -> ChatEvent:
        detail = str(exc).strip() or type(exc).__name__
        return ChatEvent(
            type="error",
            data={"error": f"llama.cpp stream failed: {detail}"},
        )

    async def cancel(self, run_id: str) -> None:
        self._cancelled.add(run_id)

    async def close(self) -> None:
        await self._client.aclose()
