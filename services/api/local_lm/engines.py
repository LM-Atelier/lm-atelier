from __future__ import annotations

from .adapters import ComfyUIAdapter, LlamaCppAdapter, MockChatAdapter, MockMediaAdapter
from .adapters.base import ChatAdapter, MediaAdapter
from .adapters.discovery import load_external_adapter
from .config import Settings
from .schemas import EngineCapabilities


class EngineRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chat: ChatAdapter
        self.media: MediaAdapter
        if settings.chat_engine == "llama.cpp":
            self.chat = LlamaCppAdapter(settings.llama_url)
        elif settings.chat_engine == "mock":
            self.chat = MockChatAdapter()
        else:
            self.chat = load_external_adapter("chat", settings.chat_engine, settings)
        if settings.media_engine == "comfyui":
            self.media = ComfyUIAdapter(settings.comfy_url)
        elif settings.media_engine == "mock":
            self.media = MockMediaAdapter()
        else:
            self.media = load_external_adapter("media", settings.media_engine, settings)

    async def capabilities(self) -> list[EngineCapabilities]:
        return [await self.chat.capabilities(), await self.media.capabilities()]

    async def close(self) -> None:
        await self.chat.close()
        await self.media.close()
