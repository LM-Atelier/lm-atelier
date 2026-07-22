from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from .base import ChatAdapter, ChatRequest, MediaAdapter, MediaRequest
from .discovery import ADAPTER_CONTRACT_VERSION


@dataclass(frozen=True)
class ConformanceReport:
    contract_version: int = ADAPTER_CONTRACT_VERSION
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def assert_passed(self) -> None:
        if self.errors:
            raise AssertionError("; ".join(self.errors))


async def probe_chat_adapter(adapter: ChatAdapter) -> ConformanceReport:
    errors: list[str] = []
    capabilities = await adapter.capabilities()
    if "chat" not in capabilities.roles:
        errors.append("capabilities must include the chat role")
    if "text" not in capabilities.operations:
        errors.append("capabilities must include the text operation")
    count = await adapter.count_tokens([{"role": "user", "content": "contract probe"}])
    if count < 1:
        errors.append("count_tokens must return a positive count")

    events = adapter.stream(
        ChatRequest(
            run_id="adapter-contract-probe",
            messages=[{"role": "user", "content": "Return a short test response."}],
        )
    )
    _require_async_iterator(events, errors, "stream")
    event_types: list[str] = []
    if not errors:
        async for event in events:
            event_types.append(event.type)
    if "complete" not in event_types:
        errors.append("stream must emit a terminal complete event")
    return ConformanceReport(errors=errors)


async def probe_media_adapter(
    adapter: MediaAdapter,
    *,
    workflow: dict[str, object] | None = None,
    input_paths: list[Path] | None = None,
) -> ConformanceReport:
    errors: list[str] = []
    capabilities = await adapter.capabilities()
    if not {"image", "video"}.intersection(capabilities.roles):
        errors.append("capabilities must include an image or video role")
    selected_workflow = workflow or {}
    validation = await adapter.validate_workflow(selected_workflow)
    if validation:
        errors.extend(f"workflow validation: {message}" for message in validation)
        return ConformanceReport(errors=errors)

    events = adapter.generate(
        MediaRequest(
            run_id="adapter-contract-probe",
            operation="text_to_image",
            prompt="A small blue square on a white background",
            negative_prompt=None,
            input_paths=input_paths or [],
            workflow=selected_workflow,
            parameters={"width": 64, "height": 64, "steps": 1, "seed": 1},
        )
    )
    _require_async_iterator(events, errors, "generate")
    event_types: list[str] = []
    assets = 0
    if not errors:
        async for event in events:
            event_types.append(event.type)
            assets += len(event.assets)
    if "complete" not in event_types:
        errors.append("generate must emit a terminal complete event")
    if assets < 1:
        errors.append("the terminal media stream must provide at least one asset")
    return ConformanceReport(errors=errors)


def _require_async_iterator(value: object, errors: list[str], method: str) -> None:
    if not isinstance(value, AsyncIterator):
        errors.append(f"{method} must return an async iterator")
