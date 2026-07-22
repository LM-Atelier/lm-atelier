from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse
from uuid import uuid4

import httpx
import websockets

from ..domain import Operation
from ..schemas import EngineCapabilities
from ..settings_registry import IMAGE_SETTINGS, VIDEO_SETTINGS
from .base import GeneratedAsset, MediaEvent, MediaRequest


class ComfyUIAdapter:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
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

    async def generate(self, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        client_id = uuid4().hex
        parameters = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            **request.parameters,
        }
        graph = self._compile(request.workflow, parameters)
        response = await self._client.post(
            "/prompt", json={"prompt": graph, "client_id": client_id}
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("node_errors"):
            raise ValueError(f"ComfyUI rejected workflow: {payload['node_errors']}")
        prompt_id = str(payload["prompt_id"])
        self._jobs[request.run_id] = prompt_id
        yield MediaEvent(type="queued", progress=0, phase="queued", data={"prompt_id": prompt_id})

        try:
            async with websockets.connect(
                self._websocket_url(client_id), max_size=32 * 1024 * 1024
            ) as socket:
                async for raw in socket:
                    if not isinstance(raw, str):
                        yield MediaEvent(type="preview", phase="preview", preview=bytes(raw))
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
                    assets.append(
                        GeneratedAsset(
                            content=file_response.content,
                            media_type=media_type,
                            kind=default_kind,
                            name=str(item["filename"]),
                            metadata={"prompt_id": prompt_id, "operation": operation},
                        )
                    )
        if not assets:
            raise RuntimeError("ComfyUI completed without collectible image or video outputs")
        return assets

    async def cancel(self, run_id: str) -> None:
        if run_id in self._jobs:
            await self._client.post("/interrupt")

    async def close(self) -> None:
        await self._client.aclose()
