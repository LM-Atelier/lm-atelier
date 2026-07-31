from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..domain import Operation
from ..schemas import EngineCapabilities
from ..settings_registry import CHAT_SETTINGS
from .base import ChatEvent, ChatRequest, estimate_chat_tokens
from .contracts import MAX_ADAPTER_EVENT_BYTES

_CANCELLED = object()


class _StreamProtocolError(RuntimeError):
    pass


class _StreamInactivityError(RuntimeError):
    pass


def _has_image_content(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(content := message.get("content"), list)
        and any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)
        for message in messages
    )


async def _bounded_response_json(response: httpx.Response) -> object:
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise ValueError("llama.cpp metadata has an invalid content length") from exc
        if declared_length < 0:
            raise ValueError("llama.cpp metadata has an invalid content length")
        if declared_length > MAX_ADAPTER_EVENT_BYTES:
            raise ValueError("llama.cpp metadata is too large")
    content = bytearray()
    async for chunk in response.aiter_bytes():
        if len(content) + len(chunk) > MAX_ADAPTER_EVENT_BYTES:
            raise ValueError("llama.cpp metadata is too large")
        content.extend(chunk)
    return json.loads(content)


class LlamaCppAdapter:
    def __init__(self, base_url: str, *, inactivity_seconds: float = 600) -> None:
        if inactivity_seconds <= 0:
            raise ValueError("llama.cpp inactivity timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.inactivity_seconds = inactivity_seconds
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10, read=inactivity_seconds, write=30, pool=10),
            trust_env=False,
        )
        self._cancelled: set[str] = set()
        self._cancel_events: dict[str, asyncio.Event] = {}

    async def capabilities(self) -> EngineCapabilities:
        healthy = False
        details: dict[str, Any] = {}
        version = "unknown"
        input_modalities = ["text"]
        try:
            async with self._client.stream("GET", "/health", timeout=3) as response:
                healthy = response.is_success
                if response.headers.get("content-type", "").startswith("application/json"):
                    value = await _bounded_response_json(response)
                    if isinstance(value, dict):
                        json.dumps(value, allow_nan=False)
                        details = value
            version = str(details.get("version", "unknown"))[:200]
        except (
            httpx.HTTPError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
            UnicodeError,
        ):
            details = {"error": "llama.cpp health check failed"}
        if healthy:
            try:
                async with self._client.stream("GET", "/props", timeout=3) as response:
                    response.raise_for_status()
                    value = await _bounded_response_json(response)
                if not isinstance(value, dict):
                    raise ValueError("llama.cpp properties metadata is invalid")
                modalities = value.get("modalities")
                if isinstance(modalities, dict) and modalities.get("vision") is True:
                    input_modalities.append("image")
            except (
                httpx.HTTPError,
                TypeError,
                ValueError,
                OverflowError,
                RecursionError,
                UnicodeError,
            ):
                # Older llama.cpp builds do not expose /props. Their chat
                # service remains usable, but vision must not be assumed.
                pass
        return EngineCapabilities(
            engine="llama.cpp",
            version=version,
            roles=["chat"],
            operations=[Operation.TEXT.value],
            input_modalities=input_modalities,
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
        payload = {
            "model": "local-model",
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = await self._client.post(
                "/v1/chat/completions/input_tokens", json=payload, timeout=10
            )
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("invalid token count response")
            count: object = value["input_tokens"]
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("invalid token count")
            if count > 0:
                return count
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
            UnicodeError,
        ):
            pass

        try:
            templated = await self._client.post(
                "/apply-template", json={"messages": messages}, timeout=10
            )
            templated.raise_for_status()
            template_value = templated.json()
            if not isinstance(template_value, dict):
                raise ValueError("invalid template response")
            prompt = template_value["prompt"]
            if not isinstance(prompt, str):
                raise TypeError("invalid template prompt")
            tokenized = await self._client.post(
                "/tokenize", json={"content": prompt, "add_special": True}, timeout=10
            )
            tokenized.raise_for_status()
            token_value = tokenized.json()
            if not isinstance(token_value, dict):
                raise ValueError("invalid tokenization response")
            tokens = token_value["tokens"]
            if isinstance(tokens, list) and tokens:
                return len(tokens)
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
            UnicodeError,
        ):
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
        if request.run_id in self._cancelled:
            self._cancelled.discard(request.run_id)
            yield ChatEvent(type="cancelled")
            return
        cancel_event = asyncio.Event()
        self._cancel_events[request.run_id] = cancel_event
        payload: dict[str, Any] = {
            "model": "local-model",
            "messages": request.messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
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

        try:
            if _has_image_content(request.messages):
                payload["stream"] = False
                payload.pop("stream_options", None)
                try:
                    events = await self._complete_without_streaming(payload, cancel_event)
                except httpx.HTTPStatusError as exc:
                    yield self._stream_error("status", status_code=exc.response.status_code)
                except httpx.TimeoutException:
                    yield self._stream_error("inactivity")
                except httpx.ConnectError:
                    yield self._stream_error("connect")
                except httpx.ReadError:
                    yield self._stream_error("read")
                except httpx.HTTPError:
                    yield self._stream_error("transport")
                except (
                    _StreamProtocolError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                    OverflowError,
                    RecursionError,
                    UnicodeError,
                ):
                    yield self._stream_error("protocol")
                else:
                    for event in events:
                        yield event
                return

            while True:
                finish_reason: str | None = None
                completion_frames = 0
                attempt_text = ""
                try:
                    async with self._client.stream(
                        "POST",
                        "/v1/chat/completions",
                        json=payload,
                        timeout=httpx.Timeout(
                            connect=10,
                            read=self.inactivity_seconds,
                            write=30,
                            pool=10,
                        ),
                    ) as response:
                        response.raise_for_status()
                        lines = response.aiter_lines()
                        while True:
                            try:
                                line = await self._next_line(lines, cancel_event)
                            except StopAsyncIteration:
                                break
                            if line is _CANCELLED:
                                yield ChatEvent(type="cancelled")
                                return
                            if not isinstance(line, str):
                                raise _StreamProtocolError
                            if len(line.encode("utf-8")) > MAX_ADAPTER_EVENT_BYTES:
                                raise _StreamProtocolError
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if raw == "[DONE]":
                                if retry_prefix is not None and len(attempt_text) < len(
                                    retry_prefix
                                ):
                                    raise _StreamProtocolError
                                yield ChatEvent(
                                    type="complete",
                                    data={"finish_reason": finish_reason or "stop"},
                                )
                                return
                            try:
                                chunk = json.loads(raw)
                            except (json.JSONDecodeError, RecursionError) as exc:
                                raise _StreamProtocolError from exc
                            if not isinstance(chunk, dict):
                                raise _StreamProtocolError
                            try:
                                json.dumps(chunk, allow_nan=False)
                            except (TypeError, ValueError, RecursionError) as exc:
                                raise _StreamProtocolError from exc
                            metadata: dict[str, Any] = {}
                            for key in ("usage", "timings"):
                                value = chunk.get(key)
                                if value is not None:
                                    if not isinstance(value, dict):
                                        raise _StreamProtocolError
                                    metadata[key] = value
                            if metadata:
                                yield ChatEvent(type="usage", data=metadata)

                            choices = chunk.get("choices", [])
                            if not isinstance(choices, list):
                                raise _StreamProtocolError
                            if not choices:
                                continue
                            choice = choices[0]
                            if not isinstance(choice, dict):
                                raise _StreamProtocolError
                            delta = choice.get("delta") or {}
                            if not isinstance(delta, dict):
                                raise _StreamProtocolError
                            content = delta.get("content")
                            tool_calls = delta.get("tool_calls")
                            if content is not None and not isinstance(content, str):
                                raise _StreamProtocolError
                            if tool_calls is not None and not isinstance(tool_calls, list):
                                raise _StreamProtocolError
                            if content is not None or tool_calls is not None:
                                completion_frames += 1
                            if content is not None:
                                if retry_prefix is None:
                                    delivered_text += content
                                    if content:
                                        yield ChatEvent(type="delta", text=content)
                                else:
                                    previous_length = len(attempt_text)
                                    attempt_text += content
                                    if not (
                                        retry_prefix.startswith(attempt_text)
                                        or attempt_text.startswith(retry_prefix)
                                    ):
                                        raise _StreamProtocolError
                                    suffix_start = max(previous_length, len(retry_prefix))
                                    if suffix_start < len(attempt_text):
                                        suffix = attempt_text[suffix_start:]
                                        delivered_text += suffix
                                        yield ChatEvent(type="delta", text=suffix)
                            if tool_calls:
                                yield ChatEvent(
                                    type="tool_delta",
                                    data={"tool_calls": tool_calls},
                                )
                            raw_finish_reason = choice.get("finish_reason")
                            if raw_finish_reason is not None:
                                if (
                                    not isinstance(raw_finish_reason, str)
                                    or not 1 <= len(raw_finish_reason) <= 64
                                    or any(
                                        not (character.isalnum() or character in {"_", "-", "."})
                                        for character in raw_finish_reason
                                    )
                                ):
                                    raise _StreamProtocolError
                                finish_reason = raw_finish_reason
                                # llama.cpp has emitted streams that publish an
                                # authoritative terminal choice but never close.
                                if retry_prefix is not None and len(attempt_text) < len(
                                    retry_prefix
                                ):
                                    raise _StreamProtocolError
                                yield ChatEvent(
                                    type="complete",
                                    data={"finish_reason": finish_reason},
                                )
                                return
                    # EOF without [DONE] or an authoritative finish reason is
                    # indistinguishable from a silently truncated response.
                    raise httpx.ReadError("stream ended before a terminal event")
                except _StreamInactivityError:
                    yield self._stream_error("inactivity")
                    return
                except _StreamProtocolError:
                    yield self._stream_error("protocol")
                    return
                except httpx.ReadError:
                    if cancel_event.is_set():
                        yield ChatEvent(type="cancelled")
                        return
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
                    yield self._stream_error("read")
                    return
                except httpx.HTTPStatusError as exc:
                    yield self._stream_error("status", status_code=exc.response.status_code)
                    return
                except httpx.TimeoutException:
                    yield self._stream_error("inactivity")
                    return
                except httpx.ConnectError:
                    yield self._stream_error("connect")
                    return
                except httpx.HTTPError:
                    yield self._stream_error("transport")
                    return
        finally:
            self._cancel_events.pop(request.run_id, None)
            self._cancelled.discard(request.run_id)

    async def _complete_without_streaming(
        self,
        payload: dict[str, Any],
        cancel_event: asyncio.Event,
    ) -> list[ChatEvent]:
        request_task = asyncio.create_task(self._request_completion(payload))
        cancel_task = asyncio.create_task(cancel_event.wait())
        tasks = {request_task, cancel_task}
        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if cancel_task in done and cancel_event.is_set():
                return [ChatEvent(type="cancelled")]
            return await request_task
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _request_completion(self, payload: dict[str, Any]) -> list[ChatEvent]:
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
            timeout=httpx.Timeout(
                connect=10,
                read=self.inactivity_seconds,
                write=30,
                pool=10,
            ),
        ) as response:
            response.raise_for_status()
            value = await _bounded_response_json(response)
        return self._completion_events(value)

    @staticmethod
    def _completion_events(value: object) -> list[ChatEvent]:
        if not isinstance(value, dict):
            raise _StreamProtocolError
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise _StreamProtocolError from exc

        choices = value.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise _StreamProtocolError
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise _StreamProtocolError
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if content is not None and not isinstance(content, str):
            raise _StreamProtocolError
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise _StreamProtocolError
        finish_reason = choice.get("finish_reason")
        if (
            not isinstance(finish_reason, str)
            or not 1 <= len(finish_reason) <= 64
            or any(
                not (character.isalnum() or character in {"_", "-", "."})
                for character in finish_reason
            )
        ):
            raise _StreamProtocolError

        events: list[ChatEvent] = []
        if content:
            events.append(ChatEvent(type="delta", text=content))
        if tool_calls:
            events.append(ChatEvent(type="tool_delta", data={"tool_calls": tool_calls}))
        metadata: dict[str, Any] = {}
        for key in ("usage", "timings"):
            item = value.get(key)
            if item is not None:
                if not isinstance(item, dict):
                    raise _StreamProtocolError
                metadata[key] = item
        if metadata:
            events.append(ChatEvent(type="usage", data=metadata))
        events.append(ChatEvent(type="complete", data={"finish_reason": finish_reason}))
        return events

    async def _next_line(
        self,
        lines: AsyncIterator[str],
        cancel_event: asyncio.Event,
    ) -> str | object:
        next_task: asyncio.Future[str] = asyncio.ensure_future(anext(lines))
        cancel_task = asyncio.create_task(cancel_event.wait())
        tasks: set[asyncio.Future[Any]] = {next_task, cancel_task}
        try:
            done, _pending = await asyncio.wait(
                tasks,
                timeout=self.inactivity_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise _StreamInactivityError
            if cancel_task in done and cancel_event.is_set():
                return _CANCELLED
            return await next_task
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _stream_error(kind: str, *, status_code: int | None = None) -> ChatEvent:
        if kind == "status":
            detail = f"llama.cpp rejected the generation request (HTTP {status_code})"
        elif kind == "inactivity":
            detail = "llama.cpp stopped reporting generation activity"
        elif kind == "connect":
            detail = "Could not connect to llama.cpp"
        elif kind == "protocol":
            detail = "llama.cpp returned an invalid streaming response"
        elif kind == "read":
            detail = "llama.cpp connection closed before generation completed"
        else:
            detail = "llama.cpp generation transport failed"
        return ChatEvent(type="error", data={"error": detail})

    async def cancel(self, run_id: str) -> None:
        if event := self._cancel_events.get(run_id):
            event.set()
        else:
            self._cancelled.add(run_id)

    async def close(self) -> None:
        for event in self._cancel_events.values():
            event.set()
        await self._client.aclose()
