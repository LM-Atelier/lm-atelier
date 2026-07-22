from __future__ import annotations

import asyncio
import html
import shutil

from ..domain import Operation
from ..schemas import EngineCapabilities
from ..settings_registry import CHAT_SETTINGS, IMAGE_SETTINGS, VIDEO_SETTINGS
from .base import ChatEvent, ChatRequest, GeneratedAsset, MediaEvent, MediaRequest


class MockChatAdapter:
    def __init__(self) -> None:
        self._cancelled: set[str] = set()

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine="mock",
            version="1",
            roles=["chat"],
            operations=[Operation.TEXT.value],
            formats=["mock"],
            devices=["cpu:0"],
            streaming=True,
            tool_calling=True,
            settings=CHAT_SETTINGS,
            healthy=True,
            details={"purpose": "offline development and contract testing"},
        )

    async def stream(self, request: ChatRequest):  # type: ignore[no-untyped-def]
        prompt = next(
            (
                str(message.get("content", ""))
                for message in reversed(request.messages)
                if message.get("role") == "user"
            ),
            "",
        )
        response = (
            "Mock local response: I received your request and completed the full "
            f"streaming path. You said: {prompt}"
        )
        for token in response.split(" "):
            if request.run_id in self._cancelled:
                self._cancelled.discard(request.run_id)
                yield ChatEvent(type="cancelled")
                return
            yield ChatEvent(type="delta", text=f"{token} ")
            await asyncio.sleep(0.01)
        yield ChatEvent(type="complete", data={"finish_reason": "stop", "mock": True})

    async def cancel(self, run_id: str) -> None:
        self._cancelled.add(run_id)

    async def close(self) -> None:
        self._cancelled.clear()


class MockMediaAdapter:
    def __init__(self) -> None:
        self._cancelled: set[str] = set()

    async def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine="mock",
            version="1",
            roles=["image", "video"],
            operations=[
                Operation.TEXT_TO_IMAGE.value,
                Operation.IMAGE_TO_IMAGE.value,
                Operation.TEXT_TO_VIDEO.value,
                Operation.IMAGE_TO_VIDEO.value,
            ],
            formats=["mock", "svg", "mp4"],
            devices=["cpu:0"],
            streaming=False,
            tool_calling=False,
            settings=[*IMAGE_SETTINGS, *VIDEO_SETTINGS],
            healthy=True,
            details={"purpose": "offline development and contract testing"},
        )

    async def validate_workflow(self, workflow: dict[str, object]) -> list[str]:
        return [] if isinstance(workflow, dict) else ["workflow must be an object"]

    async def generate(self, request: MediaRequest):  # type: ignore[no-untyped-def]
        for progress, phase in ((0.05, "loading"), (0.25, "encoding prompt"), (0.7, "sampling")):
            if request.run_id in self._cancelled:
                self._cancelled.discard(request.run_id)
                yield MediaEvent(type="cancelled", progress=progress, phase="cancelled")
                return
            yield MediaEvent(type="progress", progress=progress, phase=phase)
            await asyncio.sleep(0.05)

        if "video" in request.operation:
            asset = await self._mock_video(request)
        else:
            asset = self._mock_image(request)
        yield MediaEvent(type="complete", progress=1, phase="complete", assets=[asset])

    def _mock_image(self, request: MediaRequest) -> GeneratedAsset:
        prompt = html.escape(request.prompt[:240])
        svg = "".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="640" ',
                'viewBox="0 0 1024 640">',
                '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">',
                '<stop stop-color="#151a24"/><stop offset="1" stop-color="#6658f5"/>',
                "</linearGradient></defs>",
                '<rect width="1024" height="640" fill="url(#g)"/>',
                '<circle cx="820" cy="120" r="220" fill="#ffba6b" opacity=".28"/>',
                '<text x="72" y="270" fill="#ffffff" font-family="sans-serif" ',
                'font-size="42" font-weight="700">Local LM mock image</text>',
                '<foreignObject x="72" y="310" width="850" height="220">',
                '<div xmlns="http://www.w3.org/1999/xhtml" ',
                'style="color:#dfe5f3;font:26px sans-serif;line-height:1.4">',
                prompt,
                "</div></foreignObject></svg>",
            ]
        )
        return GeneratedAsset(
            content=svg.encode(),
            media_type="image/svg+xml",
            kind="image",
            name="mock-generation.svg",
            metadata={"mock": True, "operation": request.operation},
        )

    async def _mock_video(self, request: MediaRequest) -> GeneratedAsset:
        executable = shutil.which("ffmpeg")
        if executable:
            process = await asyncio.create_subprocess_exec(
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=0x6658f5:s=640x360:r=24:d=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "frag_keyframe+empty_moov",
                "-f",
                "mp4",
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _stderr = await process.communicate()
            if process.returncode == 0 and stdout:
                return GeneratedAsset(
                    content=stdout,
                    media_type="video/mp4",
                    kind="video",
                    name="mock-generation.mp4",
                    metadata={"mock": True, "operation": request.operation},
                )
        return GeneratedAsset(
            content=b"Mock video output unavailable because ffmpeg encoding failed.",
            media_type="text/plain",
            kind="other",
            name="mock-video-error.txt",
            metadata={"mock": True, "operation": request.operation},
        )

    async def cancel(self, run_id: str) -> None:
        self._cancelled.add(run_id)

    async def close(self) -> None:
        self._cancelled.clear()
