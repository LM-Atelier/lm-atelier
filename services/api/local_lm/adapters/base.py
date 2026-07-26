from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..schemas import EngineCapabilities


@dataclass
class ChatRequest:
    run_id: str
    messages: list[dict[str, Any]]
    settings: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChatEvent:
    type: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def estimate_chat_tokens(messages: list[dict[str, Any]]) -> int:
    """Return a conservative fallback when an engine tokenizer is unavailable."""
    characters = 0
    images = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            characters += len(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                characters += len(part["text"])
            elif part.get("type") == "image_url":
                images += 1
    # Image tokenization varies by projector and resolution. Reserve enough
    # space to avoid treating a large base64 data URL as prose while still
    # trimming conservatively when the engine's tokenizer routes are absent.
    return max(1, math.ceil(characters / 3) + (6 * len(messages)) + (1024 * images))


@dataclass
class MediaRequest:
    run_id: str
    operation: str
    prompt: str
    negative_prompt: str | None
    input_paths: list[Path]
    workflow: dict[str, Any]
    parameters: dict[str, Any]


@dataclass
class GeneratedAsset:
    content: bytes
    media_type: str
    kind: str
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaEvent:
    type: str
    progress: float = 0
    phase: str = ""
    preview: bytes | None = None
    assets: list[GeneratedAsset] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


class ChatAdapter(Protocol):
    async def capabilities(self) -> EngineCapabilities: ...

    async def count_tokens(self, messages: list[dict[str, Any]]) -> int: ...

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]: ...

    async def cancel(self, run_id: str) -> None: ...

    async def close(self) -> None: ...


class MediaAdapter(Protocol):
    async def capabilities(self) -> EngineCapabilities: ...

    async def validate_workflow(self, workflow: dict[str, Any]) -> list[str]: ...

    def generate(self, request: MediaRequest) -> AsyncIterator[MediaEvent]: ...

    async def cancel(self, run_id: str) -> None: ...

    async def close(self) -> None: ...
