from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from ..schemas import EngineCapabilities


@dataclass
class ChatRequest:
    run_id: str
    messages: list[dict[str, Any]]
    settings: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    persistence_scope: str = "durable"
    scope_id: str | None = None
    tool_choice: Literal["auto", "required"] = "auto"
    parallel_tool_calls: bool | None = None


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
    persistence_scope: str = "durable"
    scope_id: str | None = None
    # Optional bytes parallel input_paths. Callers can retain an exact decoded
    # source through upload without reopening its pathname. This field is
    # transport data; accepted-source and workflow authority remain upstream.
    input_contents: tuple[bytes, ...] | None = field(default=None, repr=False)
    # The selection a workflow's mask input receives, already read the way a
    # blend reads it. Without it the selection's stored file is sent as drawn.
    mask_content: bytes | None = field(default=None, repr=False)


@dataclass
class GeneratedAsset:
    content: bytes
    media_type: str
    kind: str
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    #: What the engine said about where this file came from, while it still had
    #: the response in hand. None from an engine that does not attribute its
    #: outputs, which is a different thing from a key that arrived unusable.
    origin: dict[str, Any] | None = None


@dataclass
class MediaEvent:
    type: str
    progress: float = 0
    phase: str = ""
    preview: bytes | None = None
    assets: list[GeneratedAsset] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    #: True only for events the BACKEND originated - a queue acknowledgement,
    #: sampler progress, a preview frame, collected outputs. Adapter-local
    #: lifecycle announcements (preparing a workspace, staging inputs,
    #: submitting) keep the default: they are progress about the job, not
    #: evidence the engine is alive, and the liveness stamp downstream must
    #: not be mintable before the backend has said anything. Fail closed - an
    #: unmarked event never stamps.
    engine_sourced: bool = False


class MediaWorkflowFailure(Exception):
    """The media engine refused a run's work, or one of its steps failed in an ordinary way.

    An invalid graph, a refused input, a node's own error, a finished run whose
    outputs cannot be used: the engine itself stays usable, so a caller may keep
    its process for the next run. Any other exception may be the engine's own
    fault and is treated as one.

    The orchestrator reads the mark only when a run hands the device back, and
    acts on it only for the managed ComfyUI worker: a marked failure may keep
    that worker for the next run, as a finished run may, while after any other
    failure the worker is recycled, unless another picture or video is queued
    next or someone else is using it. The window that keeps the worker and the
    recycle both apply only while the media engine is ComfyUI, which only the
    built-in ComfyUI adapter drives. For any other engine, a third-party one
    included, the handoff neither keeps nor recycles a worker, so the mark
    changes nothing there.

    A failure is marked by a class that also derives from the type it always
    had, so whatever catches that type still catches it.
    """


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
