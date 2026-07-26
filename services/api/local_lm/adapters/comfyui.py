from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse
from uuid import uuid4

import httpx
import websockets
from websockets.exceptions import WebSocketException

from ..domain import Operation
from ..schemas import EngineCapabilities
from ..settings_registry import IMAGE_SETTINGS, VIDEO_SETTINGS
from .base import GeneratedAsset, MediaEvent, MediaRequest
from .contracts import MAX_ADAPTER_EVENT_BYTES, MAX_ADAPTER_PREVIEW_BYTES

logger = logging.getLogger(__name__)
_CANCELLED = object()
_MAX_COMFY_JSON_BYTES = 32 * 1024 * 1024
_MAX_COMFY_OUTPUTS = 64


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
    def __init__(
        self,
        base_url: str,
        *,
        inactivity_seconds: float = 600,
        managed_output_root: Path | None = None,
        max_output_bytes: int = 512 * 1024**2,
        stale_output_seconds: float = 24 * 60 * 60,
    ) -> None:
        if inactivity_seconds <= 0:
            raise ValueError("ComfyUI inactivity timeout must be positive")
        if max_output_bytes <= 0:
            raise ValueError("ComfyUI output limit must be positive")
        if stale_output_seconds <= 0:
            raise ValueError("ComfyUI stale-output age must be positive")
        self.base_url = base_url.rstrip("/")
        self.inactivity_seconds = inactivity_seconds
        self.max_output_bytes = max_output_bytes
        self.stale_output_seconds = stale_output_seconds
        self.managed_output_root = (
            managed_output_root.expanduser().resolve() if managed_output_root else None
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            trust_env=False,
        )
        self._jobs: dict[str, str] = {}
        self._cancelled: set[str] = set()
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._last_output_sweep = 0.0

    async def capabilities(self) -> EngineCapabilities:
        healthy = False
        version = "unknown"
        details: dict[str, Any] = {}
        try:
            response = await self._client.get("/system_stats", timeout=3)
            response.raise_for_status()
            if len(response.content) > MAX_ADAPTER_EVENT_BYTES:
                raise ValueError("ComfyUI system metadata is too large")
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("invalid ComfyUI system metadata")
            json.dumps(value, allow_nan=False)
            details = value
            healthy = True
            system = details.get("system") or {}
            if not isinstance(system, dict):
                raise ValueError("invalid ComfyUI system metadata")
            version = str(system.get("comfyui_version", "unknown"))[:200]
        except (httpx.HTTPError, ValueError) as exc:
            details = {"error": f"ComfyUI health check failed ({type(exc).__name__})"}
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
            input_modalities=["text", "image"],
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
            node_types = await self.object_info()
        except (httpx.HTTPError, ValueError):
            return ["Could not inspect the available ComfyUI nodes."]
        errors: list[str] = []
        for node_id, node in workflow.items():
            safe_node_id = str(node_id)
            if len(safe_node_id) > 200 or any(
                character < " " and character != "\t" for character in safe_node_id
            ):
                safe_node_id = "<invalid>"
            if not isinstance(node, dict):
                errors.append(f"node {safe_node_id} must be an object")
                continue
            class_type = node.get("class_type")
            if not class_type:
                errors.append(f"node {safe_node_id} has no class_type")
            elif (
                not isinstance(class_type, str)
                or len(class_type) > 200
                or any(character < " " and character != "\t" for character in class_type)
            ):
                errors.append(f"node {safe_node_id} has an invalid class_type")
            elif class_type not in node_types:
                safe_class_type = class_type
                errors.append(f"node {safe_node_id} requires missing type {safe_class_type}")
        return errors

    async def object_info(self) -> dict[str, Any]:
        response = await self._client.get("/object_info", timeout=10)
        response.raise_for_status()
        if len(response.content) > _MAX_COMFY_JSON_BYTES:
            raise ValueError("ComfyUI object metadata is too large")
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("ComfyUI object metadata must be an object")
        return value

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
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("ComfyUI returned an invalid conditioning-image upload")
            uploaded_name = str(payload.get("name") or "")
            upload_type = str(payload.get("type") or "")
            if (
                not uploaded_name
                or len(uploaded_name) > 255
                or "/" in uploaded_name
                or "\\" in uploaded_name
                or uploaded_name in {".", ".."}
                or upload_type not in {"input", "temp"}
            ):
                raise ValueError("ComfyUI returned an invalid conditioning-image upload")
            raw_subfolder = str(payload.get("subfolder") or "").replace("\\", "/")
            subfolder_path = PurePosixPath(raw_subfolder)
            if subfolder_path.is_absolute() or any(
                part in {"", ".", ".."} or ":" in part for part in subfolder_path.parts
            ):
                raise ValueError("ComfyUI returned an invalid conditioning-image upload")
            subfolder = "/".join(subfolder_path.parts)
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
        if request.run_id in self._cancelled:
            self._cancelled.discard(request.run_id)
            yield MediaEvent(type="cancelled")
            return
        cancel_event = asyncio.Event()
        self._cancel_events[request.run_id] = cancel_event
        prompt_id: str | None = None
        outputs_collected = False
        try:
            await self._sweep_stale_outputs()
            client_id = uuid4().hex
            parameters = await self._request_parameters(request)
            if cancel_event.is_set():
                yield MediaEvent(type="cancelled")
                return
            graph = self._compile(request.workflow, parameters)
            async with websockets.connect(
                self._websocket_url(client_id),
                max_size=MAX_ADAPTER_PREVIEW_BYTES,
                max_queue=16,
                open_timeout=10,
                close_timeout=10,
                proxy=None,
            ) as socket:
                response = await self._client.post(
                    "/prompt",
                    json={"prompt": graph, "client_id": client_id},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RuntimeError("ComfyUI returned an invalid prompt response")
                if payload.get("node_errors"):
                    raise RuntimeError("ComfyUI rejected the selected workflow")
                raw_prompt_id = payload.get("prompt_id")
                if (
                    not isinstance(raw_prompt_id, str)
                    or not raw_prompt_id
                    or len(raw_prompt_id) > 200
                    or any(character < " " for character in raw_prompt_id)
                ):
                    raise RuntimeError("ComfyUI returned an invalid prompt identifier")
                prompt_id = raw_prompt_id
                self._jobs[request.run_id] = prompt_id
                if cancel_event.is_set():
                    await self._interrupt_prompt()
                    yield MediaEvent(type="cancelled")
                    return
                yield MediaEvent(
                    type="queued",
                    progress=0,
                    phase="queued",
                    data={"prompt_id": prompt_id},
                )
                messages = socket.__aiter__()
                while True:
                    try:
                        raw = await self._next_message(messages, cancel_event)
                    except StopAsyncIteration:
                        break
                    except TimeoutError as exc:
                        await self._interrupt_prompt()
                        raise RuntimeError(
                            "ComfyUI stopped reporting generation activity for "
                            f"{self.inactivity_seconds:g} seconds; the run was interrupted"
                        ) from exc
                    if raw is _CANCELLED:
                        yield MediaEvent(type="cancelled")
                        return
                    if not isinstance(raw, str):
                        binary = bytes(raw)
                        if len(binary) > MAX_ADAPTER_PREVIEW_BYTES:
                            raise RuntimeError("ComfyUI returned an oversized preview event")
                        if preview := _preview_payload(binary):
                            yield MediaEvent(type="preview", phase="preview", preview=preview)
                        continue
                    if len(raw.encode("utf-8")) > MAX_ADAPTER_EVENT_BYTES:
                        raise RuntimeError("ComfyUI returned an oversized progress event")
                    try:
                        message = json.loads(raw)
                    except (json.JSONDecodeError, RecursionError) as exc:
                        raise RuntimeError("ComfyUI returned a malformed progress event") from exc
                    if not isinstance(message, dict):
                        raise RuntimeError("ComfyUI returned a malformed progress event")
                    message_type = message.get("type")
                    data = message.get("data") or {}
                    if not isinstance(message_type, str) or len(message_type) > 100:
                        raise RuntimeError("ComfyUI returned a malformed progress event")
                    if not isinstance(data, dict):
                        raise RuntimeError("ComfyUI returned a malformed progress event")
                    if data.get("prompt_id") not in {None, prompt_id}:
                        continue
                    if message_type == "progress":
                        try:
                            maximum = float(data.get("max") or 1)
                            value = float(data.get("value") or 0)
                        except (TypeError, ValueError) as exc:
                            raise RuntimeError(
                                "ComfyUI returned a malformed progress event"
                            ) from exc
                        if not math.isfinite(maximum) or maximum <= 0 or not math.isfinite(value):
                            raise RuntimeError("ComfyUI returned a malformed progress event")
                        yield MediaEvent(
                            type="progress",
                            progress=min(max(value / maximum, 0), 0.99),
                            phase="sampling",
                            data={
                                "prompt_id": prompt_id,
                                "value": value,
                                "max": maximum,
                            },
                        )
                    elif (
                        message_type == "executing" and data.get("node") is None
                    ) or message_type == "execution_success":
                        break
                    elif message_type == "execution_interrupted":
                        yield MediaEvent(type="cancelled")
                        return
                    elif message_type == "execution_error":
                        raise RuntimeError("ComfyUI could not execute the selected workflow")

            if cancel_event.is_set():
                yield MediaEvent(type="cancelled")
                return
            assets = await self._collect_outputs(prompt_id, request.operation)
            outputs_collected = True
            yield MediaEvent(type="complete", progress=1, phase="complete", assets=assets)
        except asyncio.CancelledError:
            raise
        except WebSocketException:
            raise RuntimeError("ComfyUI generation connection failed") from None
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"ComfyUI rejected a generation request (HTTP {exc.response.status_code})"
            ) from None
        except httpx.TimeoutException:
            raise RuntimeError("ComfyUI generation transport timed out") from None
        except httpx.HTTPError:
            raise RuntimeError("ComfyUI generation transport failed") from None
        except OSError:
            raise RuntimeError("ComfyUI could not access local generation files") from None
        finally:
            self._jobs.pop(request.run_id, None)
            self._cancel_events.pop(request.run_id, None)
            self._cancelled.discard(request.run_id)
            if prompt_id and not outputs_collected:
                await self._cleanup_prompt_outputs(prompt_id)

    async def _next_message(
        self,
        messages: AsyncIterator[Any],
        cancel_event: asyncio.Event,
    ) -> Any:
        next_task: asyncio.Future[Any] = asyncio.ensure_future(anext(messages))
        cancel_task = asyncio.create_task(cancel_event.wait())
        tasks: set[asyncio.Future[Any]] = {next_task, cancel_task}
        try:
            done, _pending = await asyncio.wait(
                tasks,
                timeout=self.inactivity_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError
            if cancel_task in done and cancel_event.is_set():
                return _CANCELLED
            return await next_task
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def probe_workflow(
        self,
        request: MediaRequest,
        *,
        timeout_seconds: float = 300,
    ) -> None:
        """Run a bounded generation before an adaptive model is activated."""

        completed = False
        try:
            async with asyncio.timeout(timeout_seconds):
                async for event in self.generate(request):
                    if event.type == "complete" and event.assets:
                        completed = True
        except TimeoutError as exc:
            await self._interrupt_prompt()
            raise RuntimeError(
                "ComfyUI did not complete the bounded model activation probe."
            ) from exc
        if not completed:
            raise RuntimeError("ComfyUI did not produce media during the model activation probe.")

    async def _collect_outputs(self, prompt_id: str, operation: str) -> list[GeneratedAsset]:
        response = await self._client.get(f"/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        if len(response.content) > _MAX_COMFY_JSON_BYTES:
            raise RuntimeError("ComfyUI output history is too large")
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("ComfyUI returned invalid output history")
        history = payload.get(prompt_id, {})
        if not isinstance(history, dict):
            raise RuntimeError("ComfyUI returned invalid output history")
        outputs = history.get("outputs") or {}
        if not isinstance(outputs, dict):
            raise RuntimeError("ComfyUI returned invalid output history")
        assets: list[GeneratedAsset] = []
        managed_paths: set[Path] = set()
        output_items: list[tuple[dict[str, Any], str]] = []
        too_many_outputs = False
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for collection, default_kind in (
                ("images", "image"),
                ("gifs", "video"),
                ("videos", "video"),
            ):
                values = node_output.get(collection, [])
                if not isinstance(values, list):
                    continue
                for raw_item in values:
                    if not isinstance(raw_item, dict):
                        continue
                    item = dict(raw_item)
                    if len(output_items) < _MAX_COMFY_OUTPUTS:
                        output_items.append((item, default_kind))
                    else:
                        too_many_outputs = True
                    if managed_path := self._managed_output_path(item):
                        managed_paths.add(managed_path)
        try:
            if too_many_outputs:
                raise RuntimeError(f"ComfyUI returned more than {_MAX_COMFY_OUTPUTS} outputs")
            total_bytes = 0
            for item, default_kind in output_items:
                filename = str(item.get("filename") or "")
                if (
                    not filename
                    or len(filename) > 255
                    or filename in {".", ".."}
                    or "/" in filename
                    or "\\" in filename
                    or any(character < " " for character in filename)
                ):
                    raise ValueError("ComfyUI output has an invalid filename")
                params = {
                    "filename": filename,
                    "subfolder": self._output_parameter(
                        item.get("subfolder", ""),
                        "subfolder",
                        maximum=1000,
                    ),
                    "type": self._output_parameter(
                        item.get("type", "output"),
                        "type",
                        maximum=20,
                        allowed={"input", "output", "temp"},
                    ),
                }
                remaining = self.max_output_bytes - total_bytes
                if remaining <= 0:
                    raise RuntimeError(
                        f"ComfyUI outputs exceed the {self.max_output_bytes}-byte limit"
                    )
                async with self._client.stream(
                    "GET",
                    "/view",
                    params=params,
                    timeout=120,
                ) as file_response:
                    file_response.raise_for_status()
                    content = await self._read_bounded_output(file_response, remaining)
                    media_type = file_response.headers.get(
                        "content-type", "application/octet-stream"
                    )
                    if len(media_type) > 200 or any(character < " " for character in media_type):
                        media_type = "application/octet-stream"
                total_bytes += len(content)
                kind = self._output_kind(
                    filename=filename,
                    media_type=media_type,
                    default_kind=default_kind,
                )
                assets.append(
                    GeneratedAsset(
                        content=content,
                        media_type=media_type,
                        kind=kind,
                        name=filename,
                        metadata={"prompt_id": prompt_id, "operation": operation},
                    )
                )
            if not assets:
                raise RuntimeError("ComfyUI completed without collectible image or video outputs")
            return assets
        finally:
            await self._remove_managed_outputs(managed_paths)

    async def _read_bounded_output(
        self,
        response: httpx.Response,
        remaining_bytes: int,
    ) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > remaining_bytes:
                    raise RuntimeError(
                        f"ComfyUI outputs exceed the {self.max_output_bytes}-byte limit"
                    )
            except ValueError:
                pass
        content = bytearray()
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > remaining_bytes:
                raise RuntimeError(f"ComfyUI outputs exceed the {self.max_output_bytes}-byte limit")
        return bytes(content)

    @staticmethod
    def _output_parameter(
        value: Any,
        name: str,
        *,
        maximum: int,
        allowed: set[str] | None = None,
    ) -> str:
        if (
            not isinstance(value, str)
            or len(value) > maximum
            or any(character < " " for character in value)
            or (allowed is not None and value not in allowed)
        ):
            raise ValueError(f"ComfyUI output has an invalid {name}")
        return value

    async def _cleanup_prompt_outputs(self, prompt_id: str) -> None:
        try:
            response = await self._client.get(f"/history/{prompt_id}", timeout=10)
            response.raise_for_status()
            if len(response.content) > _MAX_COMFY_JSON_BYTES:
                raise ValueError("ComfyUI output history is too large")
            payload = response.json()
            history = payload.get(prompt_id, {}) if isinstance(payload, dict) else {}
            outputs = history.get("outputs", {}) if isinstance(history, dict) else {}
            if not isinstance(outputs, dict):
                return
            managed_paths = {
                path
                for node_output in outputs.values()
                if isinstance(node_output, dict)
                for collection in ("images", "gifs", "videos")
                for item in node_output.get(collection, [])
                if isinstance(item, dict)
                if (path := self._managed_output_path(item)) is not None
            }
            await self._remove_managed_outputs(managed_paths)
        except Exception as exc:
            logger.debug(
                "Could not inspect a failed ComfyUI output (%s)",
                type(exc).__name__,
            )

    async def _sweep_stale_outputs(self) -> None:
        root = self.managed_output_root
        now = time.time()
        if root is None or now - self._last_output_sweep < min(
            3600,
            self.stale_output_seconds,
        ):
            return
        self._last_output_sweep = now
        await asyncio.to_thread(
            self._sweep_stale_outputs_sync,
            root,
            now - self.stale_output_seconds,
        )

    @staticmethod
    def _sweep_stale_outputs_sync(root: Path, cutoff: float) -> None:
        if not root.is_dir():
            return
        for path in root.rglob("*"):
            try:
                if not path.is_file() or path.stat().st_mtime > cutoff:
                    continue
                resolved = path.resolve()
                resolved.relative_to(root)
                path.unlink(missing_ok=True)
            except (OSError, ValueError):
                continue
        for directory in sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            with suppress(OSError):
                directory.rmdir()

    def _managed_output_path(self, item: dict[str, Any]) -> Path | None:
        root = self.managed_output_root
        if root is None or str(item.get("type", "output")) != "output":
            return None
        filename = str(item.get("filename") or "")
        subfolder = str(item.get("subfolder") or "")
        filename_path = PurePosixPath(filename.replace("\\", "/"))
        subfolder_path = PurePosixPath(subfolder.replace("\\", "/"))
        unsafe_parts = {"", ".", ".."}
        if (
            not filename
            or filename_path.is_absolute()
            or len(filename_path.parts) != 1
            or filename_path.name in unsafe_parts
            or subfolder_path.is_absolute()
            or any(part in unsafe_parts or ":" in part for part in subfolder_path.parts)
            or ":" in filename_path.name
        ):
            return None
        candidate = root.joinpath(*subfolder_path.parts, filename_path.name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    @staticmethod
    async def _remove_managed_outputs(paths: set[Path]) -> None:
        for path in paths:
            try:
                await asyncio.to_thread(path.unlink, missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "Could not remove a collected managed ComfyUI output (%s)",
                    type(exc).__name__,
                )

    @staticmethod
    def _output_kind(*, filename: str, media_type: str, default_kind: str) -> str:
        if media_type.lower().startswith("video/"):
            return "video"
        if filename.lower().endswith((".mp4", ".webm", ".mov", ".mkv", ".avi")):
            return "video"
        return default_kind

    async def cancel(self, run_id: str) -> None:
        if event := self._cancel_events.get(run_id):
            event.set()
        else:
            self._cancelled.add(run_id)
        if run_id in self._jobs:
            await self._interrupt_prompt()

    async def _interrupt_prompt(self) -> None:
        try:
            response = await self._client.post("/interrupt", timeout=10)
            response.raise_for_status()
        except httpx.HTTPError:
            # Local cancellation remains authoritative even when the worker has
            # already exited or its interrupt endpoint is unavailable.
            return

    async def close(self) -> None:
        for event in self._cancel_events.values():
            event.set()
        await self._client.aclose()
