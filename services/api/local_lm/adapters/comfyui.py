from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse
from uuid import uuid4

import httpx
import websockets

from ..domain import Operation
from ..schemas import EngineCapabilities
from ..settings_registry import IMAGE_SETTINGS, VIDEO_SETTINGS
from .base import GeneratedAsset, MediaEvent, MediaRequest


def _is_preview_image(content: bytes) -> bool:
    return (
        content.startswith((b"\x89PNG", b"\xff\xd8", b"GIF87a", b"GIF89a"))
        or (content.startswith(b"RIFF") and content[8:12] == b"WEBP")
        or content.lstrip().startswith(b"<svg")
    )


def _preview_payload(frame: bytes) -> bytes | None:
    if _is_preview_image(frame):
        return frame
    if len(frame) < 8:
        return None
    event_type = int.from_bytes(frame[:4], "big")
    if event_type == 1:
        # PREVIEW_IMAGE: event type, image format, then encoded image.
        payload = frame[8:]
    elif event_type == 4:
        # PREVIEW_IMAGE_WITH_METADATA: event type, metadata length, JSON, image.
        metadata_length = int.from_bytes(frame[4:8], "big")
        payload_start = 8 + metadata_length
        if payload_start > len(frame):
            return None
        payload = frame[payload_start:]
    else:
        return None
    return payload if _is_preview_image(payload) else None


class ComfyUIAdapter:
    def __init__(self, base_url: str, *, inactivity_seconds: float = 600) -> None:
        if inactivity_seconds <= 0:
            raise ValueError("ComfyUI inactivity timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.inactivity_seconds = inactivity_seconds
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10, read=None, write=30, pool=10),
        )
        self._jobs: dict[str, str] = {}

    async def capabilities(self) -> EngineCapabilities:
        healthy = False
        version = "unknown"
        details: dict[str, Any] = {}
        try:
            response = await self._client.get("/system_stats", timeout=3)
            response.raise_for_status()
            details = response.json()
            healthy = True
            version = str((details.get("system") or {}).get("comfyui_version", "unknown"))
        except (httpx.HTTPError, ValueError) as exc:
            details = {"error": str(exc)}
        return EngineCapabilities(
            engine="comfyui",
            version=version,
            roles=["image", "video"],
            operations=[
                Operation.TEXT_TO_IMAGE.value,
                Operation.IMAGE_TO_IMAGE.value,
                Operation.TEXT_TO_VIDEO.value,
                Operation.IMAGE_TO_VIDEO.value,
            ],
            formats=["safetensors", "comfy-workflow"],
            devices=[],
            streaming=False,
            tool_calling=False,
            settings=[*IMAGE_SETTINGS, *VIDEO_SETTINGS],
            settings_by_role={"image": IMAGE_SETTINGS, "video": VIDEO_SETTINGS},
            healthy=healthy,
            details=details,
        )

    async def validate_workflow(self, workflow: dict[str, Any]) -> list[str]:
        if not workflow:
            return ["workflow graph is empty"]
        try:
            response = await self._client.get("/object_info", timeout=10)
            response.raise_for_status()
            node_types = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return [f"could not inspect ComfyUI nodes: {exc}"]
        errors: list[str] = []
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                errors.append(f"node {node_id} must be an object")
                continue
            class_type = node.get("class_type")
            if not class_type:
                errors.append(f"node {node_id} has no class_type")
            elif class_type not in node_types:
                errors.append(f"node {node_id} requires missing type {class_type}")
        return errors

    @staticmethod
    def _compile(value: Any, parameters: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {key: ComfyUIAdapter._compile(item, parameters) for key, item in value.items()}
        if isinstance(value, list):
            return [ComfyUIAdapter._compile(item, parameters) for item in value]
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return parameters.get(value[2:-1], value)
        return value

    def _websocket_url(self, client_id: str) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse(
            (scheme, parsed.netloc, "/ws", "", urlencode({"clientId": client_id}), "")
        )

    @staticmethod
    def _image_format(content: bytes) -> tuple[str, str]:
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png", "image/png"
        if content.startswith(b"\xff\xd8"):
            return ".jpg", "image/jpeg"
        if content.startswith((b"GIF87a", b"GIF89a")):
            return ".gif", "image/gif"
        if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
            return ".webp", "image/webp"
        if content.startswith(b"BM"):
            return ".bmp", "image/bmp"
        if content.startswith((b"II*\x00", b"MM\x00*")):
            return ".tiff", "image/tiff"
        raise ValueError("ComfyUI conditioning inputs must be supported raster images")

    async def _upload_inputs(self, request: MediaRequest) -> list[str]:
        uploaded: list[str] = []
        for index, path in enumerate(request.input_paths):
            content = await asyncio.to_thread(path.read_bytes)
            extension, media_type = self._image_format(content)
            filename = f"lm-atelier-{request.run_id}-{index}{extension}"
            response = await self._client.post(
                "/upload/image",
                data={
                    "type": "temp",
                    "subfolder": "lm-atelier",
                    "overwrite": "true",
                },
                files={"image": (filename, content, media_type)},
            )
            response.raise_for_status()
            payload = response.json()
            uploaded_name = str(payload.get("name") or "")
            upload_type = str(payload.get("type") or "")
            if not uploaded_name or upload_type not in {"input", "temp"}:
                raise ValueError("ComfyUI returned an invalid conditioning-image upload")
            subfolder = str(payload.get("subfolder") or "").strip("/\\")
            reference = f"{subfolder}/{uploaded_name}" if subfolder else uploaded_name
            if upload_type == "temp":
                reference = f"{reference} [temp]"
            uploaded.append(reference)
        return uploaded

    async def _request_parameters(self, request: MediaRequest) -> dict[str, Any]:
        if (
            request.operation
            in {
                Operation.IMAGE_TO_IMAGE.value,
                Operation.IMAGE_TO_VIDEO.value,
            }
            and not request.input_paths
        ):
            raise ValueError(f"{request.operation} requires a conditioning image")
        parameters = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            **request.parameters,
        }
        uploaded = await self._upload_inputs(request)
        if uploaded:
            parameters["input_image"] = uploaded[0]
            parameters["input_images"] = uploaded
            parameters.update(
                {f"input_image_{index}": reference for index, reference in enumerate(uploaded)}
            )
        return parameters

    async def generate(self, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        client_id = uuid4().hex
        parameters = await self._request_parameters(request)
        graph = self._compile(request.workflow, parameters)
        try:
            async with websockets.connect(
                self._websocket_url(client_id), max_size=32 * 1024 * 1024
            ) as socket:
                response = await self._client.post(
                    "/prompt", json={"prompt": graph, "client_id": client_id}
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("node_errors"):
                    raise ValueError(f"ComfyUI rejected workflow: {payload['node_errors']}")
                prompt_id = str(payload["prompt_id"])
                self._jobs[request.run_id] = prompt_id
                yield MediaEvent(
                    type="queued",
                    progress=0,
                    phase="queued",
                    data={"prompt_id": prompt_id},
                )
                messages = socket.__aiter__()
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            anext(messages),
                            timeout=self.inactivity_seconds,
                        )
                    except StopAsyncIteration:
                        break
                    except TimeoutError as exc:
                        with suppress(httpx.HTTPError):
                            await self.cancel(request.run_id)
                        raise RuntimeError(
                            "ComfyUI stopped reporting generation activity for "
                            f"{self.inactivity_seconds:g} seconds; the run was interrupted"
                        ) from exc
                    if not isinstance(raw, str):
                        if preview := _preview_payload(bytes(raw)):
                            yield MediaEvent(type="preview", phase="preview", preview=preview)
                        continue
                    message = json.loads(raw)
                    message_type = message.get("type")
                    data = message.get("data") or {}
                    if data.get("prompt_id") not in {None, prompt_id}:
                        continue
                    if message_type == "progress":
                        maximum = float(data.get("max") or 1)
                        yield MediaEvent(
                            type="progress",
                            progress=min(float(data.get("value") or 0) / maximum, 0.99),
                            phase="sampling",
                            data=data,
                        )
                    elif message_type == "executing" and data.get("node") is None:
                        break
                    elif message_type == "execution_error":
                        raise RuntimeError(str(data.get("exception_message", data)))

            assets = await self._collect_outputs(prompt_id, request.operation)
            yield MediaEvent(type="complete", progress=1, phase="complete", assets=assets)
        finally:
            self._jobs.pop(request.run_id, None)

    async def _collect_outputs(self, prompt_id: str, operation: str) -> list[GeneratedAsset]:
        response = await self._client.get(f"/history/{prompt_id}")
        response.raise_for_status()
        history = response.json().get(prompt_id, {})
        outputs = history.get("outputs") or {}
        assets: list[GeneratedAsset] = []
        for node_output in outputs.values():
            for collection, default_kind in (
                ("images", "image"),
                ("gifs", "video"),
                ("videos", "video"),
            ):
                for item in node_output.get(collection, []):
                    params = {
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                    }
                    file_response = await self._client.get("/view", params=params)
                    file_response.raise_for_status()
                    media_type = file_response.headers.get(
                        "content-type", "application/octet-stream"
                    )
                    kind = self._output_kind(
                        filename=str(item["filename"]),
                        media_type=media_type,
                        default_kind=default_kind,
                    )
                    assets.append(
                        GeneratedAsset(
                            content=file_response.content,
                            media_type=media_type,
                            kind=kind,
                            name=str(item["filename"]),
                            metadata={"prompt_id": prompt_id, "operation": operation},
                        )
                    )
        if not assets:
            raise RuntimeError("ComfyUI completed without collectible image or video outputs")
        return assets

    @staticmethod
    def _output_kind(*, filename: str, media_type: str, default_kind: str) -> str:
        if media_type.lower().startswith("video/"):
            return "video"
        if filename.lower().endswith((".mp4", ".webm", ".mov", ".mkv", ".avi")):
            return "video"
        return default_kind

    async def cancel(self, run_id: str) -> None:
        if run_id in self._jobs:
            await self._client.post("/interrupt")

    async def close(self) -> None:
        await self._client.aclose()
