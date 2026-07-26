from __future__ import annotations

import asyncio
from typing import Literal

from .adapters import (
    ComfyUIAdapter,
    LlamaCppAdapter,
    MockChatAdapter,
    MockMediaAdapter,
    VllmAdapter,
)
from .adapters.base import ChatAdapter, MediaAdapter
from .adapters.contracts import capability_contract_errors
from .adapters.discovery import load_external_adapter
from .config import Settings
from .schemas import EngineCapabilities, SettingField
from .settings_registry import (
    builtin_settings_for_role,
    capability_settings_for_role,
    normalize_capability_settings,
)


class EngineSchemaUnavailableError(RuntimeError):
    """The selected engine could not supply a safe capability schema."""


class EngineNotConfiguredError(EngineSchemaUnavailableError):
    """A stored profile refers to an engine that is not currently active."""


class EngineRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._chat_adapters: dict[str, ChatAdapter] = {}
        self._fallback_chat: ChatAdapter
        self.media: MediaAdapter
        if settings.chat_engine in {"llama.cpp", "vllm"}:
            self._chat_adapters["llama.cpp"] = LlamaCppAdapter(
                settings.llama_url,
                inactivity_seconds=settings.llama_inactivity_seconds,
            )
            self._chat_adapters["vllm"] = VllmAdapter(
                settings.llama_url,
                inactivity_seconds=settings.llama_inactivity_seconds,
            )
            self._fallback_chat = self._chat_adapters[settings.chat_engine]
        elif settings.chat_engine == "mock":
            self._fallback_chat = MockChatAdapter()
        else:
            self._fallback_chat = load_external_adapter("chat", settings.chat_engine, settings)
        if settings.media_engine == "comfyui":
            managed_output_root = (
                settings.comfy_output_dir
                if settings.comfy_executable and settings.comfy_directory
                else None
            )
            self.media = ComfyUIAdapter(
                settings.comfy_url,
                inactivity_seconds=settings.comfy_inactivity_seconds,
                managed_output_root=managed_output_root,
                managed_temp_root=(
                    settings.comfy_directory / "temp"
                    if managed_output_root and settings.comfy_directory
                    else None
                ),
                max_output_bytes=settings.max_generated_output_bytes,
                stale_output_seconds=settings.temporary_retention_hours * 60 * 60,
            )
        elif settings.media_engine == "mock":
            self.media = MockMediaAdapter()
        else:
            self.media = load_external_adapter("media", settings.media_engine, settings)

    @property
    def chat(self) -> ChatAdapter:
        return self._chat_adapters.get(self.settings.chat_engine, self._fallback_chat)

    async def capabilities(self) -> list[EngineCapabilities]:
        chat, media = await asyncio.gather(
            self._adapter_capabilities(self.chat, "chat"),
            self._adapter_capabilities(self.media, "media"),
        )
        return [chat, media]

    async def chat_capabilities(self) -> EngineCapabilities:
        return await self._adapter_capabilities(self.chat, "chat")

    async def settings_for_role(
        self,
        role: str,
        *,
        engine: str | None = None,
        allow_inactive: bool = False,
    ) -> list[SettingField]:
        adapter: ChatAdapter | MediaAdapter
        if role == "chat":
            adapter = self._chat_adapters.get(engine or self.settings.chat_engine, self.chat)
            kind: Literal["chat", "media"] = "chat"
            configured_engine = (
                engine if engine in self._chat_adapters else self.settings.chat_engine
            )
        elif role in {"image", "video"}:
            adapter = self.media
            kind = "media"
            configured_engine = self.settings.media_engine
        else:
            raise ValueError(f"unsupported engine role: {role}")
        if engine is not None and engine != configured_engine:
            if allow_inactive:
                return builtin_settings_for_role(role)
            raise EngineNotConfiguredError(
                f"The {engine} engine used by this {role} profile is not configured."
            )
        capabilities = await self._adapter_capabilities(adapter, kind)
        try:
            return capability_settings_for_role(capabilities, role)
        except ValueError as exc:
            raise EngineSchemaUnavailableError(
                f"The configured {role} engine has an invalid settings schema."
            ) from exc

    @staticmethod
    async def _adapter_capabilities(
        adapter: ChatAdapter | MediaAdapter,
        kind: Literal["chat", "media"],
    ) -> EngineCapabilities:
        try:
            capabilities = await adapter.capabilities()
        except Exception as exc:
            raise EngineSchemaUnavailableError(
                f"The configured {kind} engine could not report capabilities."
            ) from exc
        errors = capability_contract_errors(capabilities, kind)
        if errors:
            raise EngineSchemaUnavailableError(
                f"The configured {kind} engine reported invalid capabilities."
            )
        try:
            return normalize_capability_settings(capabilities)
        except ValueError as exc:
            raise EngineSchemaUnavailableError(
                f"The configured {kind} engine has an invalid settings schema."
            ) from exc

    async def close(self) -> None:
        chat_adapters = {id(adapter): adapter for adapter in self._chat_adapters.values()}
        chat_adapters.setdefault(id(self._fallback_chat), self._fallback_chat)
        await asyncio.gather(
            *(adapter.close() for adapter in chat_adapters.values()),
            self.media.close(),
        )
