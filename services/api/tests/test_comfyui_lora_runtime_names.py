from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest

from local_lm.adapters.base import MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter


class Socket:
    async def __aenter__(self) -> Socket:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None


@pytest.mark.parametrize("class_type", ["LoraLoader", "LoraLoaderModelOnly"])
async def test_submission_uses_exact_runtime_choice_and_preserves_selected_workflow(
    monkeypatch: pytest.MonkeyPatch, class_type: str
) -> None:
    graph = {
        "1": {
            "class_type": class_type,
            "inputs": {"lora_name": "nested/adapter.safetensors", "strength_model": 0.0},
        }
    }
    original = copy.deepcopy(graph)
    posted: list[dict[str, Any]] = []
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/object_info":
            return httpx.Response(
                200,
                json={
                    class_type: {
                        "input": {"required": {"lora_name": [["nested\\adapter.safetensors"], {}]}}
                    }
                },
            )
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {"os": "win32"}})
        assert request.url.path == "/prompt"
        posted.append(json.loads(request.content)["prompt"])
        return httpx.Response(400, json={"error": {"type": "fixture_stop_after_validation"}})

    monkeypatch.setattr(
        "local_lm.adapters.comfyui.websockets.connect", lambda *_args, **_kwargs: Socket()
    )
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    request = MediaRequest(
        run_id="run_lora",
        operation="text_to_image",
        prompt="A blue cube",
        negative_prompt="",
        input_paths=[],
        workflow=graph,
        parameters={},
    )
    try:
        with pytest.raises(RuntimeError, match="HTTP 400"):
            [event async for event in adapter.generate(request)]
    finally:
        await adapter.close()
    assert paths == ["/object_info", "/system_stats", "/prompt"]
    assert posted == [
        {
            "1": {
                "class_type": class_type,
                "inputs": {"lora_name": "nested\\adapter.safetensors", "strength_model": 0.0},
            }
        }
    ]
    assert request.workflow == original


async def test_cancel_during_metadata_lookup_prevents_prompt_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        assert request.url.path == "/object_info"
        await adapter.cancel("run_lora")
        return httpx.Response(200, json={})

    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    monkeypatch.setattr(
        "local_lm.adapters.comfyui.websockets.connect",
        lambda *_args, **_kwargs: pytest.fail("Cancelled request opened a socket"),
    )
    request = MediaRequest(
        run_id="run_lora",
        operation="text_to_image",
        prompt="A blue cube",
        negative_prompt="",
        input_paths=[],
        workflow={
            "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "nested/adapter.safetensors"}}
        },
        parameters={},
    )
    try:
        events = [event async for event in adapter.generate(request)]
    finally:
        await adapter.close()
    assert paths == ["/object_info"]
    assert events[-1].type == "cancelled"


async def test_cancel_during_platform_lookup_prevents_prompt_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/object_info":
            return httpx.Response(
                200,
                json={
                    "LoraLoader": {
                        "input": {"required": {"lora_name": [["nested\\adapter.safetensors"], {}]}}
                    }
                },
            )
        assert request.url.path == "/system_stats"
        await adapter.cancel("run_lora")
        return httpx.Response(200, json={"system": {"os": "win32"}})

    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    monkeypatch.setattr(
        "local_lm.adapters.comfyui.websockets.connect",
        lambda *_args, **_kwargs: pytest.fail("Cancelled request opened a socket"),
    )
    request = MediaRequest(
        run_id="run_lora",
        operation="text_to_image",
        prompt="A blue cube",
        negative_prompt="",
        input_paths=[],
        parameters={},
        workflow={
            "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "nested/adapter.safetensors"}}
        },
    )
    try:
        events = [event async for event in adapter.generate(request)]
    finally:
        await adapter.close()
    assert paths == ["/object_info", "/system_stats"]
    assert events[-1].type == "cancelled"


@pytest.mark.parametrize("platform", ["linux", "darwin", None])
async def test_posix_or_unknown_runtime_never_submits_a_different_literal_filename(
    monkeypatch: pytest.MonkeyPatch, platform: str | None
) -> None:
    graph = {
        "1": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": "nested/adapter.safetensors"},
        }
    }
    posted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(
                200,
                json={
                    "LoraLoaderModelOnly": {
                        "input": {"required": {"lora_name": [["nested\\adapter.safetensors"], {}]}}
                    }
                },
            )
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {"os": platform}})
        assert request.url.path == "/prompt"
        posted.append(json.loads(request.content)["prompt"])
        return httpx.Response(400, json={})

    monkeypatch.setattr(
        "local_lm.adapters.comfyui.websockets.connect", lambda *_args, **_kwargs: Socket()
    )
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    request = MediaRequest(
        run_id="run_lora",
        operation="text_to_image",
        prompt="A blue cube",
        negative_prompt="",
        input_paths=[],
        workflow=graph,
        parameters={},
    )
    try:
        with pytest.raises(RuntimeError, match="HTTP 400"):
            [event async for event in adapter.generate(request)]
    finally:
        await adapter.close()
    assert posted == [graph]


@pytest.mark.parametrize(
    "failure", ["unreadable_choices", "unreadable_platform", "unknown_platform", "exact_choice"]
)
async def test_metadata_failures_and_exact_choices_preserve_the_submission(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    graph = {
        "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "nested/adapter.safetensors"}}
    }
    posted: list[dict[str, Any]] = []
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/object_info":
            if failure == "unreadable_choices":
                return httpx.Response(503)
            choices = (
                ["nested/adapter.safetensors"]
                if failure == "exact_choice"
                else ["nested\\adapter.safetensors"]
            )
            return httpx.Response(
                200, json={"LoraLoader": {"input": {"required": {"lora_name": [choices, {}]}}}}
            )
        if request.url.path == "/system_stats":
            assert failure != "exact_choice"
            return (
                httpx.Response(503)
                if failure == "unreadable_platform"
                else httpx.Response(200, json={})
            )
        assert request.url.path == "/prompt"
        posted.append(json.loads(request.content)["prompt"])
        return httpx.Response(400, json={})

    monkeypatch.setattr(
        "local_lm.adapters.comfyui.websockets.connect", lambda *_args, **_kwargs: Socket()
    )
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    request = MediaRequest(
        run_id="run_lora",
        operation="text_to_image",
        prompt="A blue cube",
        negative_prompt="",
        input_paths=[],
        workflow=graph,
        parameters={},
    )
    try:
        with pytest.raises(RuntimeError, match="HTTP 400"):
            [event async for event in adapter.generate(request)]
    finally:
        await adapter.close()
    assert posted == [graph]
    if failure in {"exact_choice", "unreadable_choices"}:
        assert paths == ["/object_info", "/prompt"]
