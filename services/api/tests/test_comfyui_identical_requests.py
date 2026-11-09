"""An identical request to a warm ComfyUI gets files of its own.

ComfyUI answers an unchanged node from its cache without running it, and a save
node answered that way names the files an earlier request wrote. The adapter
removes the managed files it collects, so a second identical request named files
that no longer existed and failed to fetch its own result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_comfyui_adapter import media_request

from local_lm.adapters.comfyui import ComfyUIAdapter

_PNG = b"\x89PNG\r\n\x1a\nidentical"


def _workflow() -> dict[str, Any]:
    return {
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0]}},
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0], "filename_prefix": "LM Atelier"},
        },
        "linked": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0], "filename_prefix": ["name", 0]},
        },
    }


def _caching_comfy(output: Path, posted: list[dict[str, Any]]) -> Any:
    """A ComfyUI that, like the real one, answers an unchanged graph from its cache."""

    history: dict[str, Any] = {}
    last: dict[str, Any] = {}

    async def comfy(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            graph = json.loads(request.content)["prompt"]
            prompt_id = f"prompt-{len(posted) + 1}"
            if posted and graph == posted[-1]:
                # Nothing changed, so nothing runs: the previous result is reported again.
                history[prompt_id] = last["entry"]
            else:
                folder, _, name = graph["save"]["inputs"]["filename_prefix"].rpartition("/")
                target = output.joinpath(*folder.split("/")) if folder else output
                target.mkdir(parents=True, exist_ok=True)
                filename = f"{name}_00001_.png"
                (target / filename).write_bytes(_PNG)
                entry = {
                    "outputs": {
                        "save": {
                            "images": [
                                {"filename": filename, "subfolder": folder, "type": "output"}
                            ]
                        }
                    }
                }
                history[prompt_id] = last["entry"] = entry
            posted.append(graph)
            return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})
        if request.url.path.startswith("/history/"):
            prompt_id = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={prompt_id: history[prompt_id]})
        if request.url.path == "/view":
            subfolder = request.url.params.get("subfolder", "")
            path = output.joinpath(*[part for part in subfolder.split("/") if part])
            path = path / request.url.params["filename"]
            if not path.is_file():
                return httpx.Response(404)
            return httpx.Response(
                200, content=path.read_bytes(), headers={"content-type": "image/png"}
            )
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    return comfy


def _socket_for(posted: list[dict[str, Any]]) -> Any:
    class Socket:
        def __init__(self) -> None:
            self.sent = False

        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def __aiter__(self) -> Socket:
            return self

        async def __anext__(self) -> str:
            if self.sent or not posted:
                raise StopAsyncIteration
            self.sent = True
            prompt_id = f"prompt-{len(posted)}"
            return json.dumps({"type": "executing", "data": {"prompt_id": prompt_id, "node": None}})

    def connect(*_args: Any, **_kwargs: Any) -> Socket:
        return Socket()

    return connect


async def test_identical_requests_each_fetch_a_result_of_their_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    posted: list[dict[str, Any]] = []
    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", _socket_for(posted))
    adapter = ComfyUIAdapter("http://comfy.test", managed_output_root=output)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(_caching_comfy(output, posted))
    )
    workflow = _workflow()

    try:
        first = [
            event
            async for event in adapter.generate(
                media_request(operation="text_to_image", workflow=workflow)
            )
        ]
        second = [
            event
            async for event in adapter.generate(
                media_request(operation="text_to_image", workflow=workflow)
            )
        ]
    finally:
        await adapter.close()

    assert first[-1].type == "complete", first[-1]
    assert second[-1].type == "complete", second[-1]
    assert [event.assets[0].content for event in (first[-1], second[-1])] == [_PNG, _PNG]
    # Each submission saved under a folder of its own; the file name is the workflow's.
    prefixes = [graph["save"]["inputs"]["filename_prefix"] for graph in posted]
    folders = [prefix.rpartition("/")[0] for prefix in prefixes]
    assert all(prefix.endswith("/LM Atelier") for prefix in prefixes)
    assert len(set(folders)) == 2 and all(folders)
    assert second[-1].assets[0].name == "LM Atelier_00001_.png"
    # A prefix supplied by another node, and the stored workflow, are left as they were.
    assert all(graph["linked"]["inputs"]["filename_prefix"] == ["name", 0] for graph in posted)
    assert workflow["save"]["inputs"]["filename_prefix"] == "LM Atelier"


async def test_an_unmanaged_comfyui_keeps_the_workflows_own_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    posted: list[dict[str, Any]] = []
    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", _socket_for(posted))
    # Nothing it writes is removed, so a cached answer still names files that exist.
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(_caching_comfy(output, posted))
    )

    try:
        events = [
            event
            async for event in adapter.generate(
                media_request(operation="text_to_image", workflow=_workflow())
            )
        ]
    finally:
        await adapter.close()

    assert events[-1].type == "complete", events[-1]
    assert posted[0]["save"]["inputs"]["filename_prefix"] == "LM Atelier"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(200, json={"exec_info": {"queue_remaining": 2}}), 2),
        (httpx.Response(200, json={"exec_info": {"queue_remaining": 0}}), 0),
        (httpx.Response(200, json={"exec_info": {}}), None),
        (httpx.Response(200, json={"exec_info": {"queue_remaining": True}}), None),
        (httpx.Response(200, json={"exec_info": {"queue_remaining": -1}}), None),
        (httpx.Response(200, content=b"not json"), None),
        (httpx.Response(503), None),
    ],
)
async def test_the_queue_probe_reports_only_a_count_comfyui_gave(
    response: httpx.Response, expected: int | None
) -> None:
    seen: list[str] = []

    async def comfy(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return response

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(comfy)
    )
    try:
        assert await adapter.queue_remaining() == expected
    finally:
        await adapter.close()
    # The count only: nothing that could carry a prompt is read.
    assert seen == ["GET /prompt"]
