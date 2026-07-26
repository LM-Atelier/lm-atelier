from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from websockets.exceptions import WebSocketException

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter, _preview_payload


def media_request(path: Path | None = None, *, operation: str = "image_to_image") -> MediaRequest:
    return MediaRequest(
        run_id="run_conditioning",
        operation=operation,
        prompt="Restyle the source image",
        negative_prompt="blur",
        input_paths=[path] if path else [],
        workflow={},
        parameters={"seed": 42},
    )


def test_preview_binary_envelopes_are_decoded_and_validated() -> None:
    jpeg = b"\xff\xd8\xff\xe0preview"
    png = b"\x89PNG\r\n\x1a\npreview"
    metadata = b'{"node_id":"sampler"}'

    assert _preview_payload(jpeg) == jpeg
    assert _preview_payload((1).to_bytes(4, "big") + (1).to_bytes(4, "big") + jpeg) == jpeg
    assert (
        _preview_payload((4).to_bytes(4, "big") + len(metadata).to_bytes(4, "big") + metadata + png)
        == png
    )
    assert _preview_payload((3).to_bytes(4, "big") + b"text") is None
    assert _preview_payload((4).to_bytes(4, "big") + (999).to_bytes(4, "big")) is None
    assert _preview_payload((1).to_bytes(4, "big") + (1).to_bytes(4, "big") + b"invalid") is None


async def test_object_info_is_cached_by_content_hash_and_explicitly_invalidated() -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"Sampler": {"output_node": False}})

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        first = await adapter.object_info()
        second = await adapter.object_info()
        adapter.invalidate_object_info_cache()
        third = await adapter.object_info()
    finally:
        await adapter.close()

    assert first is second
    assert third == first
    assert requests == 2
    assert len(adapter._object_info_by_hash) == 1


async def test_conditioning_images_are_staged_and_exposed_as_workflow_parameters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "content-addressed-artifact"
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test-image")
    captured_body = b""

    async def upload(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = await request.aread()
        assert request.url.path == "/upload/image"
        return httpx.Response(
            200,
            json={
                "name": "lm-atelier-run_conditioning-0.png",
                "subfolder": "lm-atelier",
                "type": "temp",
            },
        )

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(upload),
    )
    try:
        parameters = await adapter._request_parameters(media_request(source))
    finally:
        await adapter.close()

    reference = "lm-atelier/lm-atelier-run_conditioning-0.png [temp]"
    assert parameters["input_image"] == reference
    assert parameters["input_image_0"] == reference
    assert parameters["input_images"] == [reference]
    assert parameters["prompt"] == "Restyle the source image"
    assert parameters["negative_prompt"] == "blur"
    assert parameters["seed"] == 42
    assert b'name="type"' in captured_body
    assert b"temp" in captured_body
    assert b'name="subfolder"' in captured_body
    assert b"lm-atelier" in captured_body
    assert b'filename="lm-atelier-run_conditioning-0.png"' in captured_body


async def test_conditioning_requires_an_input_and_rejects_non_images(tmp_path: Path) -> None:
    adapter = ComfyUIAdapter("http://comfy.test")
    try:
        with pytest.raises(ValueError, match="requires a conditioning image"):
            await adapter._request_parameters(media_request())

        invalid = tmp_path / "not-an-image"
        invalid.write_bytes(b"plain text")
        with pytest.raises(ValueError, match="supported raster images"):
            await adapter._request_parameters(media_request(invalid))
    finally:
        await adapter.close()


async def test_websocket_is_connected_before_a_warm_prompt_can_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    prompt_id = "prompt-warm"
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")

    class Socket:
        def __init__(self) -> None:
            self.messages = iter(
                [
                    json.dumps(
                        {
                            "type": "executing",
                            "data": {"prompt_id": prompt_id, "node": None},
                        }
                    )
                ]
            )

        async def __aenter__(self) -> Socket:
            events.append("connected")
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def __aiter__(self) -> Socket:
            return self

        async def __anext__(self) -> str:
            try:
                return next(self.messages)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    def connect(*_args: Any, **kwargs: Any) -> Socket:
        assert kwargs["proxy"] is None
        return Socket()

    async def comfy(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            events.append("prompted")
            assert events == ["connected", "prompted"]
            return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})
        if request.url.path == f"/history/{prompt_id}":
            return httpx.Response(
                200,
                json={
                    prompt_id: {
                        "outputs": {
                            "save": {
                                "images": [
                                    {
                                        "filename": "warm.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    }
                },
            )
        if request.url.path == "/view":
            return httpx.Response(
                200, content=b"\x89PNG\r\n\x1a\n", headers={"content-type": "image/png"}
            )
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", connect)
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    request = media_request(operation="text_to_image")
    try:
        generated = [event async for event in adapter.generate(request)]
    finally:
        await adapter.close()

    assert events == ["connected", "prompted"]
    assert [event.type for event in generated] == ["queued", "complete"]
    assert generated[-1].assets[0].name == "warm.png"


async def test_workflow_probe_requires_a_generated_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ComfyUIAdapter("http://comfy.test")
    request = media_request(operation="text_to_image")

    async def successful_probe(_request: MediaRequest):  # type: ignore[no-untyped-def]
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=b"\x89PNG\r\n\x1a\n",
                    media_type="image/png",
                    kind="image",
                    name="probe.png",
                )
            ],
        )

    async def empty_probe(_request: MediaRequest):  # type: ignore[no-untyped-def]
        yield MediaEvent(type="complete")

    try:
        monkeypatch.setattr(adapter, "generate", successful_probe)
        await adapter.probe_workflow(request, timeout_seconds=1)

        monkeypatch.setattr(adapter, "generate", empty_probe)
        with pytest.raises(RuntimeError, match="did not produce media"):
            await adapter.probe_workflow(request, timeout_seconds=1)
    finally:
        await adapter.close()


async def test_workflow_probe_interrupts_a_backend_that_exceeds_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupted = False

    async def comfy(request: httpx.Request) -> httpx.Response:
        nonlocal interrupted
        assert request.url.path == "/interrupt"
        interrupted = True
        return httpx.Response(200, json={})

    async def stalled_probe(_request: MediaRequest):  # type: ignore[no-untyped-def]
        yield MediaEvent(type="queued")
        await asyncio.Event().wait()

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    monkeypatch.setattr(adapter, "generate", stalled_probe)
    try:
        with pytest.raises(RuntimeError, match="bounded model activation probe"):
            await adapter.probe_workflow(
                media_request(operation="text_to_image"),
                timeout_seconds=0.01,
            )
    finally:
        await adapter.close()

    assert interrupted


async def test_silent_websocket_is_interrupted_after_configured_inactivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_id = "prompt-silent"
    interrupted = False

    class Socket:
        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def __aiter__(self) -> Socket:
            return self

        async def __anext__(self) -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    def connect(*_args: Any, **_kwargs: Any) -> Socket:
        return Socket()

    async def comfy(request: httpx.Request) -> httpx.Response:
        nonlocal interrupted
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})
        if request.url.path == "/interrupt":
            interrupted = True
            return httpx.Response(200, json={})
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", connect)
    adapter = ComfyUIAdapter("http://comfy.test", inactivity_seconds=0.01)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    request = media_request(operation="text_to_image")
    try:
        with pytest.raises(RuntimeError, match="stopped reporting generation activity"):
            [event async for event in adapter.generate(request)]
    finally:
        await adapter.close()

    assert interrupted
    assert request.run_id not in adapter._jobs


async def test_cancel_wakes_a_blocked_comfyui_websocket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_id = "prompt-cancelled"
    receiving = asyncio.Event()
    interrupted = False

    class Socket:
        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def __aiter__(self) -> Socket:
            return self

        async def __anext__(self) -> str:
            receiving.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    def connect(*_args: Any, **_kwargs: Any) -> Socket:
        return Socket()

    async def comfy(request: httpx.Request) -> httpx.Response:
        nonlocal interrupted
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})
        if request.url.path == "/interrupt":
            interrupted = True
            # Local cancellation must remain authoritative when the worker is
            # already stopping and its best-effort interrupt request fails.
            return httpx.Response(500)
        if request.url.path == f"/history/{prompt_id}":
            return httpx.Response(200, json={})
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", connect)
    adapter = ComfyUIAdapter("http://comfy.test", inactivity_seconds=60)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    request = media_request(operation="text_to_image")
    collecting = asyncio.create_task(_collect_media_events(adapter, request))
    try:
        await asyncio.wait_for(receiving.wait(), timeout=0.5)
        await adapter.cancel(request.run_id)
        events = await asyncio.wait_for(collecting, timeout=0.5)
    finally:
        if not collecting.done():
            collecting.cancel()
        await adapter.close()

    assert [event.type for event in events] == ["queued", "cancelled"]
    assert interrupted
    assert request.run_id not in adapter._jobs
    assert request.run_id not in adapter._cancel_events


async def _collect_media_events(
    adapter: ComfyUIAdapter,
    request: MediaRequest,
) -> list[MediaEvent]:
    return [event async for event in adapter.generate(request)]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            "{not-json",
            "ComfyUI returned a malformed progress event",
            id="malformed-json",
        ),
        pytest.param(
            "[]",
            "ComfyUI returned a malformed progress event",
            id="non-object-json",
        ),
        pytest.param(
            json.dumps(
                {
                    "type": "progress",
                    "data": {"prompt_id": "prompt-invalid", "max": float("nan"), "value": 1},
                }
            ),
            "ComfyUI returned a malformed progress event",
            id="non-finite-progress",
        ),
        pytest.param(
            json.dumps(
                {
                    "type": "execution_error",
                    "data": {
                        "prompt_id": "prompt-invalid",
                        "exception_message": "C:\\private\\model\\secret.safetensors",
                    },
                }
            ),
            "ComfyUI could not execute the selected workflow",
            id="redacted-execution-error",
        ),
        pytest.param(
            "x" * (1024 * 1024 + 1),
            "ComfyUI returned an oversized progress event",
            id="oversized-event",
        ),
    ],
)
async def test_invalid_comfyui_events_fail_with_redacted_bounded_errors(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: str,
) -> None:
    prompt_id = "prompt-invalid"

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
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return raw

    def connect(*_args: Any, **_kwargs: Any) -> Socket:
        return Socket()

    async def comfy(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})
        if request.url.path == f"/history/{prompt_id}":
            return httpx.Response(200, json={})
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", connect)
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    try:
        with pytest.raises(RuntimeError) as captured:
            [event async for event in adapter.generate(media_request(operation="text_to_image"))]
    finally:
        await adapter.close()

    assert str(captured.value) == expected
    assert "secret.safetensors" not in str(captured.value)


async def test_comfyui_transport_errors_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "C:\\private\\worker\\secret.safetensors"

    def failed_connect(*_args: Any, **_kwargs: Any) -> None:
        raise WebSocketException(secret)

    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", failed_connect)
    adapter = ComfyUIAdapter("http://comfy.test")
    try:
        with pytest.raises(RuntimeError) as captured:
            [event async for event in adapter.generate(media_request(operation="text_to_image"))]
    finally:
        await adapter.close()

    assert str(captured.value) == "ComfyUI generation connection failed"
    assert secret not in str(captured.value)


async def test_comfyui_http_error_body_and_url_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "backend-private-workflow-detail"

    class Socket:
        async def __aenter__(self) -> Socket:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    def connect(*_args: Any, **_kwargs: Any) -> Socket:
        return Socket()

    async def comfy(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=secret)

    monkeypatch.setattr("local_lm.adapters.comfyui.websockets.connect", connect)
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    try:
        with pytest.raises(RuntimeError) as captured:
            [event async for event in adapter.generate(media_request(operation="text_to_image"))]
    finally:
        await adapter.close()

    assert str(captured.value) == "ComfyUI rejected a generation request (HTTP 400)"
    assert secret not in str(captured.value)
    assert "http://comfy.test" not in str(captured.value)


async def test_native_save_video_is_classified_by_media_type_not_collection() -> None:
    prompt_id = "prompt-video"

    async def comfy(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/history/{prompt_id}":
            return httpx.Response(
                200,
                json={
                    prompt_id: {
                        "outputs": {
                            "save": {
                                "images": [
                                    {
                                        "filename": "native.mp4",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    }
                },
            )
        if request.url.path == "/view":
            return httpx.Response(
                200,
                content=b"mp4-content",
                headers={"content-type": "video/mp4"},
            )
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    try:
        outputs = await adapter._collect_outputs(prompt_id, "text_to_video")
    finally:
        await adapter.close()

    assert outputs[0].kind == "video"
    assert outputs[0].media_type == "video/mp4"


async def test_collected_managed_outputs_are_removed_after_all_downloads(tmp_path: Path) -> None:
    prompt_id = "prompt-cleanup"
    output_root = tmp_path / "output"
    first = output_root / "LMAtelier" / "first.png"
    second = output_root / "LMAtelier" / "second.mp4"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    async def comfy(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/history/{prompt_id}":
            return httpx.Response(
                200,
                json={
                    prompt_id: {
                        "outputs": {
                            "save": {
                                "images": [
                                    {
                                        "filename": first.name,
                                        "subfolder": "LMAtelier",
                                        "type": "output",
                                    },
                                    {
                                        "filename": second.name,
                                        "subfolder": "LMAtelier",
                                        "type": "output",
                                    },
                                ]
                            }
                        }
                    }
                },
            )
        if request.url.path == "/view":
            filename = request.url.params["filename"]
            content = (output_root / "LMAtelier" / filename).read_bytes()
            media_type = "video/mp4" if filename.endswith(".mp4") else "image/png"
            return httpx.Response(200, content=content, headers={"content-type": media_type})
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    adapter = ComfyUIAdapter("http://comfy.test", managed_output_root=output_root)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    try:
        outputs = await adapter._collect_outputs(prompt_id, "text_to_video")
    finally:
        await adapter.close()

    assert [output.content for output in outputs] == [b"first", b"second"]
    assert not first.exists()
    assert not second.exists()


async def test_managed_outputs_are_removed_when_collection_fails(tmp_path: Path) -> None:
    prompt_id = "prompt-partial-failure"
    output_root = tmp_path / "output"
    first = output_root / "LMAtelier" / "first.png"
    second = output_root / "LMAtelier" / "second.png"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    async def comfy(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/history/{prompt_id}":
            return httpx.Response(
                200,
                json={
                    prompt_id: {
                        "outputs": {
                            "save": {
                                "images": [
                                    {
                                        "filename": first.name,
                                        "subfolder": "LMAtelier",
                                        "type": "output",
                                    },
                                    {
                                        "filename": second.name,
                                        "subfolder": "LMAtelier",
                                        "type": "output",
                                    },
                                ]
                            }
                        }
                    }
                },
            )
        if request.url.path == "/view" and request.url.params["filename"] == first.name:
            return httpx.Response(200, content=b"first", headers={"content-type": "image/png"})
        if request.url.path == "/view":
            return httpx.Response(500, text="collection failed")
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    adapter = ComfyUIAdapter("http://comfy.test", managed_output_root=output_root)
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await adapter._collect_outputs(prompt_id, "text_to_image")
    finally:
        await adapter.close()

    assert not first.exists()
    assert not second.exists()


async def test_output_collection_enforces_total_byte_limit_and_removes_source(
    tmp_path: Path,
) -> None:
    prompt_id = "prompt-too-large"
    output_root = tmp_path / "output"
    source = output_root / "LMAtelier" / "large.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"12345")

    async def comfy(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/history/{prompt_id}":
            return httpx.Response(
                200,
                json={
                    prompt_id: {
                        "outputs": {
                            "save": {
                                "videos": [
                                    {
                                        "filename": source.name,
                                        "subfolder": "LMAtelier",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    }
                },
            )
        if request.url.path == "/view":
            return httpx.Response(
                200,
                content=b"12345",
                headers={"content-type": "video/mp4"},
            )
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    adapter = ComfyUIAdapter(
        "http://comfy.test",
        managed_output_root=output_root,
        max_output_bytes=4,
    )
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(comfy),
    )
    try:
        with pytest.raises(RuntimeError, match="4-byte limit"):
            await adapter._collect_outputs(prompt_id, "text_to_video")
    finally:
        await adapter.close()

    assert not source.exists()


def test_stale_output_sweep_keeps_fresh_files(tmp_path: Path) -> None:
    root = tmp_path / "output"
    stale = root / "LMAtelier" / "stale.png"
    fresh = root / "LMAtelier" / "fresh.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    fresh.write_bytes(b"fresh")
    now = time.time()
    os.utime(stale, (now - 7200, now - 7200))

    ComfyUIAdapter._sweep_stale_outputs_sync(root.resolve(), now - 3600)

    assert not stale.exists()
    assert fresh.exists()


def test_managed_output_cleanup_rejects_external_temporary_and_unsafe_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    managed = ComfyUIAdapter("http://comfy.test", managed_output_root=root)
    external = ComfyUIAdapter("http://comfy.test")
    try:
        safe = {"filename": "result.png", "subfolder": "LMAtelier", "type": "output"}
        assert managed._managed_output_path(safe) == (root / "LMAtelier" / "result.png").resolve()
        assert external._managed_output_path(safe) is None
        assert managed._managed_output_path({**safe, "type": "temp"}) is None
        assert managed._managed_output_path({**safe, "filename": "../outside.png"}) is None
        assert managed._managed_output_path({**safe, "subfolder": "../outside"}) is None
        assert managed._managed_output_path({**safe, "subfolder": "C:\\outside"}) is None
    finally:
        asyncio.run(managed.close())
        asyncio.run(external.close())
