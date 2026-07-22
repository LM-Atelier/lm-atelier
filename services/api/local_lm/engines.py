from __future__ import annotations

from .adapters import ComfyUIAdapter, LlamaCppAdapter, MockChatAdapter, MockMediaAdapter
from .adapters.base import ChatAdapter, MediaAdapter
from .config import Settings


class EngineRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chat: ChatAdapter
        self.media: MediaAdapter
        if settings.chat_engine == "llama.cpp":
            self.chat = LlamaCppAdapter(settings.llama_url)
        else:
            self.chat = MockChatAdapter()
        if settings.media_engine == "comfyui":
            self.media = ComfyUIAdapter(settings.comfy_url)
        else:
            self.media = MockMediaAdapter()

    async def capabilities(self):  # type: ignore[no-untyped-def]
        return [await self.chat.capabilities(), await self.media.capabilities()]

    async def close(self) -> None:
        await self.chat.close()
        await self.media.close()
