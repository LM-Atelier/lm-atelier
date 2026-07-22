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
    characters = sum(len(str(message.get("content", ""))) for message in messages)
    return max(1, math.ceil(characters / 3) + (6 * len(messages)))


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
