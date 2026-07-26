from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from .base import ChatAdapter, ChatEvent, ChatRequest, MediaAdapter, MediaEvent, MediaRequest
from .contracts import (
    ADAPTER_CONTRACT_VERSION,
    capability_contract_errors,
    chat_event_contract_errors,
    media_event_contract_errors,
)


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


async def probe_chat_adapter(
    adapter: ChatAdapter,
    *,
    timeout_seconds: float = 30,
) -> ConformanceReport:
    errors: list[str] = []
    try:
        async with asyncio.timeout(timeout_seconds):
            capabilities = await adapter.capabilities()
            errors.extend(capability_contract_errors(capabilities, "chat"))
            count = await adapter.count_tokens([{"role": "user", "content": "contract probe"}])
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                errors.append("count_tokens must return a positive integer")

            events = adapter.stream(
                ChatRequest(
                    run_id="adapter-contract-probe",
                    messages=[{"role": "user", "content": "Return a short test response."}],
                )
            )
            _require_async_iterator(events, errors, "stream")
            event_types: list[str] = []
            if isinstance(events, AsyncIterator):
                async for event in events:
                    _validate_chat_event(event, errors)
                    if isinstance(event, ChatEvent):
                        event_types.append(event.type)
            _validate_terminal_events(event_types, errors, "stream")
    except TimeoutError:
        errors.append(f"chat conformance probe exceeded {timeout_seconds:g} seconds")
    except Exception as exc:
        errors.append(f"chat conformance probe raised {type(exc).__name__}")
    return ConformanceReport(errors=errors)


async def probe_media_adapter(
    adapter: MediaAdapter,
    *,
    workflow: dict[str, object] | None = None,
    input_paths: list[Path] | None = None,
    timeout_seconds: float = 300,
) -> ConformanceReport:
    errors: list[str] = []
    try:
        async with asyncio.timeout(timeout_seconds):
            capabilities = await adapter.capabilities()
            errors.extend(capability_contract_errors(capabilities, "media"))
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
            if isinstance(events, AsyncIterator):
                async for event in events:
                    _validate_media_event(event, errors)
                    if isinstance(event, MediaEvent):
                        event_types.append(event.type)
                        assets += len(event.assets)
            _validate_terminal_events(event_types, errors, "generate")
            if assets < 1:
                errors.append("the terminal media stream must provide at least one asset")
    except TimeoutError:
        errors.append(f"media conformance probe exceeded {timeout_seconds:g} seconds")
    except Exception as exc:
        errors.append(f"media conformance probe raised {type(exc).__name__}")
    return ConformanceReport(errors=errors)


def _validate_chat_event(event: object, errors: list[str]) -> None:
    errors.extend(chat_event_contract_errors(event))


def _validate_media_event(event: object, errors: list[str]) -> None:
    errors.extend(
        media_event_contract_errors(
            event,
            max_output_bytes=512 * 1024 * 1024,
        )
    )


def _validate_terminal_events(
    event_types: list[str],
    errors: list[str],
    method: str,
) -> None:
    terminals = [
        (index, event_type)
        for index, event_type in enumerate(event_types)
        if event_type in {"complete", "cancelled", "error"}
    ]
    if len(terminals) != 1:
        errors.append(f"{method} must emit exactly one terminal event")
        return
    index, event_type = terminals[0]
    if event_type != "complete":
        errors.append(f"{method} conformance probe terminated with {event_type}")
    if index != len(event_types) - 1:
        errors.append(f"{method} emitted events after its terminal event")


def _require_async_iterator(value: object, errors: list[str], method: str) -> None:
    if not isinstance(value, AsyncIterator):
        errors.append(f"{method} must return an async iterator")
