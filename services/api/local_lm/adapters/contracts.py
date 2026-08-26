from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import replace
from typing import Any, Literal

from ..schemas import EngineCapabilities, SettingField
from ..settings_registry import capability_settings_for_role, validate_settings
from .base import (
    ChatAdapter,
    ChatEvent,
    ChatRequest,
    GeneratedAsset,
    MediaAdapter,
    MediaEvent,
    MediaRequest,
    estimate_chat_tokens,
)
from .message_projection import project_chat_messages

ADAPTER_CONTRACT_VERSION = 1
MAX_ADAPTER_EVENT_BYTES = 1024 * 1024
MAX_ADAPTER_PREVIEW_BYTES = 16 * 1024 * 1024
MAX_ADAPTER_ASSETS = 64
MAX_CAPABILITY_SETTINGS = 512
MAX_CHAT_EVENTS = 262_144
MAX_MEDIA_EVENTS = 10_000
MAX_CHAT_TEXT_BYTES = 64 * 1024 * 1024
MAX_MEDIA_PREVIEW_STREAM_BYTES = 64 * 1024 * 1024

_CHAT_EVENT_TYPES = {"delta", "tool_delta", "usage", "complete", "cancelled", "error"}
_MEDIA_EVENT_TYPES = {"queued", "progress", "preview", "complete", "cancelled", "error"}
_TERMINAL_EVENT_TYPES = {"complete", "cancelled", "error"}
_CHAT_OPERATIONS = {"text"}
_MEDIA_OPERATIONS = {
    "text_to_image",
    "image_to_image",
    "text_to_video",
    "image_to_video",
}
_INPUT_MODALITIES = {"text", "image", "video", "audio"}


class AdapterContractError(RuntimeError):
    """A selected adapter violated LM Atelier's public adapter contract."""


def capability_contract_errors(
    capabilities: object,
    kind: Literal["chat", "media"],
) -> list[str]:
    try:
        return _capability_contract_errors(capabilities, kind)
    except Exception:
        return ["capabilities contain malformed values"]


def _capability_contract_errors(
    capabilities: object,
    kind: Literal["chat", "media"],
) -> list[str]:
    if not isinstance(capabilities, EngineCapabilities):
        return ["capabilities must return EngineCapabilities"]

    errors: list[str] = []
    if len(capabilities.roles) > 3:
        errors.append("capabilities may advertise at most three roles")
    roles = set(capabilities.roles)
    if len(roles) != len(capabilities.roles):
        errors.append("capabilities must not duplicate roles")
    allowed_roles = {"chat"} if kind == "chat" else {"image", "video"}
    required_roles = {"chat"} if kind == "chat" else {"image", "video"}
    if not roles.intersection(required_roles):
        errors.append(
            "capabilities must include the chat role"
            if kind == "chat"
            else "capabilities must include an image or video role"
        )
    if roles - allowed_roles:
        errors.append("capabilities include unsupported roles")

    if len(capabilities.operations) > 4:
        errors.append("capabilities may advertise at most four operations")
    operations = set(capabilities.operations)
    if len(operations) != len(capabilities.operations):
        errors.append("capabilities must not duplicate operations")
    allowed_operations = _CHAT_OPERATIONS if kind == "chat" else _MEDIA_OPERATIONS
    if not operations.intersection(allowed_operations):
        errors.append(
            "capabilities must include the text operation"
            if kind == "chat"
            else "capabilities must include a supported media operation"
        )
    if operations - allowed_operations:
        errors.append("capabilities include unsupported operations")

    allowed_modalities = {"text", "image"} if kind == "chat" else _INPUT_MODALITIES
    modalities = set(capabilities.input_modalities)
    if len(capabilities.input_modalities) > len(allowed_modalities):
        errors.append(
            f"capabilities may advertise at most {len(allowed_modalities)} input modalities"
        )
    if len(modalities) != len(capabilities.input_modalities):
        errors.append("capabilities must not duplicate input modalities")
    if modalities - allowed_modalities:
        errors.append("capabilities include unsupported input modalities")
    if "text" not in modalities:
        errors.append("capabilities must include the text input modality")

    for label, value in (
        ("engine", capabilities.engine),
        ("version", capabilities.version),
    ):
        if not _printable_text(value, 200):
            errors.append(f"{label} must be printable text no longer than 200 characters")
    for label, values, maximum_items, maximum_bytes in (
        ("formats", capabilities.formats, 64, 100),
        ("devices", capabilities.devices, 64, 200),
    ):
        if len(values) > maximum_items or any(
            not _printable_text(value, maximum_bytes) for value in values
        ):
            errors.append(f"{label} must contain bounded printable strings")

    if len(capabilities.settings) > MAX_CAPABILITY_SETTINGS:
        errors.append(f"legacy settings must contain at most {MAX_CAPABILITY_SETTINGS} fields")
    flat_settings: dict[tuple[str, str], list[SettingField]] = defaultdict(list)
    for field in capabilities.settings:
        if isinstance(field, SettingField):
            flat_settings[(field.scope, field.key)].append(field)
        errors.extend(_setting_contract_errors(field, "legacy settings"))

    role_mapping = capabilities.settings_by_role
    if len(role_mapping) > 3:
        errors.append("settings_by_role may contain at most three roles")
    if role_mapping:
        missing_roles = roles - set(role_mapping)
        if missing_roles:
            errors.append("settings_by_role is missing advertised roles")
    for role, role_fields in role_mapping.items():
        safe_role = role if _printable_text(role, 32) else "<invalid>"
        if role not in roles:
            errors.append("settings_by_role includes an unadvertised role")
        if len(role_fields) > MAX_CAPABILITY_SETTINGS:
            errors.append(
                f"settings_by_role[{safe_role}] must contain at most "
                f"{MAX_CAPABILITY_SETTINGS} fields"
            )
        seen: set[str] = set()
        for field in role_fields:
            if not isinstance(field, SettingField):
                errors.append(f"settings_by_role[{safe_role}] must contain SettingField values")
                continue
            identity = (field.scope, field.key)
            safe_key = field.key if _printable_text(field.key, 200) else "<invalid>"
            if field.key in seen:
                errors.append(f"settings_by_role[{safe_role}] duplicates key {safe_key}")
            seen.add(field.key)
            errors.extend(_setting_contract_errors(field, f"settings_by_role[{safe_role}]"))
            if field not in flat_settings.get(identity, []):
                errors.append(
                    f"settings_by_role[{safe_role}] setting {field.scope}:{safe_key} "
                    "is missing its exact definition from legacy settings"
                )

    for role in sorted(roles & allowed_roles):
        try:
            capability_settings_for_role(capabilities, role)
        except ValueError:
            errors.append(f"settings for role {role} conflict with LM Atelier's schema")

    if not _bounded_json(capabilities.details, MAX_ADAPTER_EVENT_BYTES):
        errors.append("capability details must be finite JSON no larger than 1 MiB")
    return errors


def _setting_contract_errors(field: object, location: str) -> list[str]:
    if not isinstance(field, SettingField):
        return [f"{location} must contain SettingField values"]
    errors: list[str] = []
    safe_key = field.key if _printable_text(field.key, 200) else "<invalid>"
    if safe_key == "<invalid>":
        errors.append(f"{location} has an invalid setting key")
    if not _printable_text(field.label, 500):
        errors.append(f"{location}[{safe_key}] has an invalid label")
    if not _bounded_string(field.help, 4096):
        errors.append(f"{location}[{safe_key}] has oversized help text")
    if field.unavailable_reason is not None and not _bounded_string(
        field.unavailable_reason,
        1000,
    ):
        errors.append(f"{location}[{safe_key}] has an invalid unavailable reason")
    if len(field.choices) > 256:
        errors.append(f"{location}[{safe_key}] has too many choices")
    if not _bounded_json(field.choices, MAX_ADAPTER_EVENT_BYTES):
        errors.append(f"{location}[{safe_key}] choices must be finite, bounded JSON")
    if field.available:
        try:
            validate_settings({field.key: field.default}, [field])
        except ValueError:
            errors.append(f"{location}[{safe_key}] has an invalid default")
    for constraint in (field.minimum, field.maximum, field.step, field.multiple_of):
        if constraint is not None and not math.isfinite(constraint):
            errors.append(f"{location}[{safe_key}] has a non-finite constraint")
            break
    if field.multiple_of is not None and field.multiple_of <= 0:
        errors.append(f"{location}[{safe_key}] has an invalid multiple")
    return errors


def _printable_text(value: object, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and _bounded_string(value, maximum)
        and all(character >= " " or character == "\t" for character in value)
    )


def _bounded_string(value: object, maximum_bytes: int) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return len(value.encode("utf-8")) <= maximum_bytes
    except UnicodeEncodeError:
        return False


def _bounded_json(value: object, maximum_bytes: int) -> bool:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeEncodeError):
        return False
    return len(encoded) <= maximum_bytes


def chat_event_contract_errors(event: object) -> list[str]:
    if not isinstance(event, ChatEvent):
        return ["stream must yield ChatEvent values"]
    errors: list[str] = []
    if event.type not in _CHAT_EVENT_TYPES:
        errors.append("stream emitted an unsupported event type")
    if not _bounded_string(event.text, MAX_ADAPTER_EVENT_BYTES):
        errors.append("chat event text must be UTF-8 and no larger than 1 MiB")
    elif event.type != "delta" and event.text:
        errors.append("only delta events may contain text")
    if not _bounded_json(event.data, MAX_ADAPTER_EVENT_BYTES):
        errors.append("chat event data must be finite JSON no larger than 1 MiB")
    return errors


def media_event_contract_errors(
    event: object,
    *,
    max_output_bytes: int,
) -> list[str]:
    if not isinstance(event, MediaEvent):
        return ["generate must yield MediaEvent values"]
    errors: list[str] = []
    if event.type not in _MEDIA_EVENT_TYPES:
        errors.append("generate emitted an unsupported event type")
    if (
        isinstance(event.progress, bool)
        or not isinstance(event.progress, (int, float))
        or not math.isfinite(event.progress)
        or not 0 <= event.progress <= 1
    ):
        errors.append("media event progress must be finite and between zero and one")
    if not _printable_text(event.phase, 200):
        errors.append("media event phase must be printable and no longer than 200 bytes")
    if not _bounded_json(event.data, MAX_ADAPTER_EVENT_BYTES):
        errors.append("media event data must be finite JSON no larger than 1 MiB")
    if event.preview is not None and (
        not isinstance(event.preview, bytes) or len(event.preview) > MAX_ADAPTER_PREVIEW_BYTES
    ):
        errors.append("media event previews must be bytes no larger than 16 MiB")
    elif event.preview is not None and event.type != "preview":
        errors.append("only preview events may contain preview bytes")
    try:
        assets = event.assets
        if len(assets) > MAX_ADAPTER_ASSETS:
            errors.append(f"media events may contain at most {MAX_ADAPTER_ASSETS} assets")
        if assets and event.type != "complete":
            errors.append("only complete events may contain generated assets")
        total_bytes = 0
        for asset in assets:
            if not _valid_asset(asset):
                errors.append("media event contains an invalid generated asset")
                continue
            total_bytes += len(asset.content)
        if total_bytes > max_output_bytes:
            errors.append("media event assets exceed the configured output limit")
        if event.type == "complete" and not assets:
            errors.append("complete media events must contain at least one asset")
    except Exception:
        errors.append("media event assets are malformed")
    return errors


def _valid_asset(asset: object) -> bool:
    return (
        isinstance(asset, GeneratedAsset)
        and isinstance(asset.content, bytes)
        and asset.kind in {"image", "video"}
        and _printable_text(asset.media_type, 200)
        and "/" in asset.media_type
        and _printable_text(asset.name, 255)
        and asset.name not in {".", ".."}
        and "/" not in asset.name
        and "\\" not in asset.name
        and _bounded_json(asset.metadata, MAX_ADAPTER_EVENT_BYTES)
    )


class GuardedChatAdapter:
    """Enforce the public contract around explicitly selected third-party code."""

    def __init__(
        self,
        adapter: ChatAdapter,
        label: str,
        inactivity_seconds: float = 600,
    ) -> None:
        if inactivity_seconds <= 0:
            raise ValueError("adapter inactivity timeout must be positive")
        self._adapter = adapter
        self._label = label
        self._inactivity_seconds = inactivity_seconds

    async def capabilities(self) -> EngineCapabilities:
        try:
            async with asyncio.timeout(10):
                capabilities = await self._adapter.capabilities()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise AdapterContractError(
                f"The selected chat adapter {self._label} could not report capabilities."
            ) from None
        errors = capability_contract_errors(capabilities, "chat")
        if errors:
            raise AdapterContractError(
                f"The selected chat adapter {self._label} has invalid capabilities: "
                + "; ".join(errors)
            )
        return capabilities

    async def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        messages = project_chat_messages(messages).messages
        try:
            async with asyncio.timeout(30):
                count = await self._adapter.count_tokens(messages)
        except asyncio.CancelledError:
            raise
        except Exception:
            return estimate_chat_tokens(messages)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return estimate_chat_tokens(messages)
        return count

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        request = replace(request, messages=project_chat_messages(request.messages).messages)
        try:
            events = self._adapter.stream(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            yield self._error_event()
            return
        if not isinstance(events, AsyncIterator):
            yield self._error_event()
            return

        event_count = 0
        text_bytes = 0
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        anext(events),
                        timeout=self._inactivity_seconds,
                    )
                except StopAsyncIteration:
                    break
                event_count += 1
                if event_count > MAX_CHAT_EVENTS:
                    yield self._error_event()
                    return
                if not self._valid_chat_event(event):
                    yield self._error_event()
                    return
                text_bytes += len(event.text.encode("utf-8"))
                if text_bytes > MAX_CHAT_TEXT_BYTES:
                    yield self._error_event()
                    return
                if event.type == "error":
                    yield self._error_event()
                    return
                yield event
                if event.type in _TERMINAL_EVENT_TYPES:
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            yield self._error_event()
            return
        finally:
            await _close_iterator(events)
        yield self._error_event()

    async def cancel(self, run_id: str) -> None:
        try:
            async with asyncio.timeout(10):
                await self._adapter.cancel(run_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise AdapterContractError(
                f"The selected chat adapter {self._label} could not cancel the run."
            ) from None

    async def close(self) -> None:
        try:
            async with asyncio.timeout(10):
                await self._adapter.close()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise AdapterContractError(
                f"The selected chat adapter {self._label} could not close cleanly."
            ) from None

    def _valid_chat_event(self, event: object) -> bool:
        return not chat_event_contract_errors(event)

    def _error_event(self) -> ChatEvent:
        return ChatEvent(
            type="error",
            data={"error": f"The selected chat adapter {self._label} violated its contract."},
        )


class GuardedMediaAdapter:
    """Enforce bounded media events around explicitly selected third-party code."""

    def __init__(
        self,
        adapter: MediaAdapter,
        label: str,
        max_output_bytes: int,
        inactivity_seconds: float = 600,
    ) -> None:
        if inactivity_seconds <= 0:
            raise ValueError("adapter inactivity timeout must be positive")
        self._adapter = adapter
        self._label = label
        self._max_output_bytes = max_output_bytes
        self._inactivity_seconds = inactivity_seconds

    async def capabilities(self) -> EngineCapabilities:
        try:
            async with asyncio.timeout(10):
                capabilities = await self._adapter.capabilities()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise AdapterContractError(
                f"The selected media adapter {self._label} could not report capabilities."
            ) from None
        errors = capability_contract_errors(capabilities, "media")
        if errors:
            raise AdapterContractError(
                f"The selected media adapter {self._label} has invalid capabilities: "
                + "; ".join(errors)
            )
        return capabilities

    async def validate_workflow(self, workflow: dict[str, Any]) -> list[str]:
        try:
            async with asyncio.timeout(30):
                errors = await self._adapter.validate_workflow(workflow)
        except asyncio.CancelledError:
            raise
        except Exception:
            return ["The selected media adapter could not validate this workflow."]
        if not isinstance(errors, list):
            return ["The selected media adapter returned invalid workflow validation results."]
        if not errors:
            return []
        if len(errors) > 256 or any(not isinstance(error, str) for error in errors):
            return ["The selected media adapter returned invalid workflow validation results."]
        return [f"The selected media adapter reported {len(errors)} workflow issue(s)."]

    async def generate(self, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        try:
            events = self._adapter.generate(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise self._runtime_error() from None
        if not isinstance(events, AsyncIterator):
            raise self._runtime_error()

        event_count = 0
        preview_bytes = 0
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        anext(events),
                        timeout=self._inactivity_seconds,
                    )
                except StopAsyncIteration:
                    break
                event_count += 1
                if event_count > MAX_MEDIA_EVENTS:
                    raise self._runtime_error()
                if not self._valid_media_event(event):
                    raise self._runtime_error()
                preview_bytes += len(event.preview or b"")
                if preview_bytes > min(
                    self._max_output_bytes,
                    MAX_MEDIA_PREVIEW_STREAM_BYTES,
                ):
                    raise self._runtime_error()
                if event.type == "error":
                    raise self._runtime_error()
                yield event
                if event.type in _TERMINAL_EVENT_TYPES:
                    return
        except asyncio.CancelledError:
            raise
        except AdapterContractError:
            raise
        except Exception:
            raise self._runtime_error() from None
        finally:
            await _close_iterator(events)
        raise self._runtime_error()

    async def cancel(self, run_id: str) -> None:
        try:
            async with asyncio.timeout(10):
                await self._adapter.cancel(run_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise AdapterContractError(
                f"The selected media adapter {self._label} could not cancel the run."
            ) from None

    async def close(self) -> None:
        try:
            async with asyncio.timeout(10):
                await self._adapter.close()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise AdapterContractError(
                f"The selected media adapter {self._label} could not close cleanly."
            ) from None

    def _valid_media_event(self, event: object) -> bool:
        return not media_event_contract_errors(
            event,
            max_output_bytes=self._max_output_bytes,
        )

    def _runtime_error(self) -> AdapterContractError:
        return AdapterContractError(
            f"The selected media adapter {self._label} violated its contract."
        )


async def _close_iterator(events: AsyncIterator[Any]) -> None:
    close = getattr(events, "aclose", None)
    if not callable(close):
        return
    with suppress(Exception):
        async with asyncio.timeout(5):
            await close()
