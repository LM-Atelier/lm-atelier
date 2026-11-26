"""A ComfyUI run that fails on its own workflow leaves ComfyUI running for the next run.

A picture or video that fails for the last time hands the device back to the
chat model it displaced. That used to stop ComfyUI, load the chat model again
and start ComfyUI once more, so the next media run paid a cold start of both
workers even when the failure was the workflow's own and ComfyUI was healthy: a
graph ComfyUI refused, a node that raised its own error, a finished run whose
result could not be used. Such a failure now keeps the worker as a finished run
does, under the same rules: a text turn queued next still has ComfyUI stopped
before its model loads, and so does keeping ComfyUI warm turned off.

Only a failure the adapter knows to be the workflow's own keeps the worker.
Running out of memory or of room in ComfyUI's allocator, a fault of the device,
a broken connection, a silent worker and an error that cannot be placed all
still recycle ComfyUI, since a restart is what clears them. So do a graph
ComfyUI took in part before refusing the rest, and a node it blocked while it
runs the prompt's other branches: ComfyUI may still be running that prompt.
Whose the failure was matters only for the managed ComfyUI worker; with any
other media engine a run's handoff neither keeps nor recycles a worker.

ComfyUI here is a constructed backend behind the adapter's own HTTP client and
progress socket, so every request the adapter makes is answered without a
network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS, wait_until
from test_chat_handoff_across_retries import finished
from test_generation_retry import choose_retries, failing_media
from test_media_worker_replacement import ReplacementWorld, _image_turn

from local_lm.adapters.base import MediaRequest, MediaWorkflowFailure
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.config import Settings

PROMPT_ID = "prompt-teapot"
PICTURE = b"\x89PNG\r\n\x1a\nteapot"
VIDEO = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
#: A graph with nothing for a request to fill in, saving what its sampler makes.
GRAPH: dict[str, Any] = {
    "sampler": {"class_type": "KSampler", "inputs": {"seed": 7}},
    "save": {
        "class_type": "SaveImage",
        "inputs": {"images": ["sampler", 0], "filename_prefix": "teapot"},
    },
}
#: The nodes ComfyUI names when it refuses part of a graph.
REFUSED_NODES = {"sampler": {"errors": [{"type": "value_not_in_list"}]}}
#: How ComfyUI answers a graph that fails its validation.
REFUSED_GRAPH = {
    "error": {"type": "prompt_outputs_failed_validation"},
    "node_errors": REFUSED_NODES,
}
SUCCEEDED = {"type": "execution_success", "data": {"prompt_id": PROMPT_ID}}
OUT_OF_MEMORY = "Allocation on device\nThis error means you ran out of memory on your GPU."


def _accepted() -> httpx.Response:
    return httpx.Response(200, json={"prompt_id": PROMPT_ID, "node_errors": {}})


def _uploaded() -> httpx.Response:
    return httpx.Response(
        200, json={"name": "teapot.png", "subfolder": "lm-atelier", "type": "temp"}
    )


def _unreachable() -> httpx.Response:
    raise httpx.ConnectError("Nothing is listening")


def _picture(content: bytes) -> Callable[[], httpx.Response]:
    return lambda: httpx.Response(200, content=content, headers={"content-type": "image/png"})


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _streamed_picture(*chunks: bytes) -> Callable[[], httpx.Response]:
    """A picture sent in pieces, without saying its length first."""

    return lambda: httpx.Response(
        200, content=_chunks(*chunks), headers={"content-type": "image/png"}
    )


def _recorded(count: int) -> dict[str, Any]:
    """The record of a finished prompt that saved `count` pictures."""

    images = [
        {"filename": f"teapot-{index}.png", "subfolder": "", "type": "output"}
        for index in range(count)
    ]
    return {"outputs": {"save": {"images": images}}, "status": {"status_str": "success"}}


def _node_failed(
    exception_type: str, message: str | None, *, node_type: str = "KSampler"
) -> dict[str, Any]:
    """The progress report of a node that raised, as ComfyUI sends it."""

    data: dict[str, Any] = {
        "prompt_id": PROMPT_ID,
        "node_id": "sampler",
        "node_type": node_type,
        "exception_type": exception_type,
        "traceback": [],
    }
    if message is not None:
        data["exception_message"] = message
    return {"type": "execution_error", "data": data}


def _request(
    operation: str = "text_to_image",
    *,
    inputs: tuple[bytes, ...] = (),
    workflow: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    mask: bytes | None = None,
) -> MediaRequest:
    """A request for a teapot, its inputs given as the bytes a run would send."""

    return MediaRequest(
        run_id="run-teapot",
        operation=operation,
        prompt="a teapot",
        negative_prompt=None,
        input_paths=[Path(f"input-{index}") for index in range(len(inputs))],
        workflow=GRAPH if workflow is None else workflow,
        parameters=parameters or {},
        input_contents=inputs or None,
        mask_content=mask,
    )


class _ComfyUI:
    """A ComfyUI that takes the one prompt it is sent and answers as a test sets it up.

    `answer` replies to the prompt and `upload` to each input sent. `record` is
    the prompt's record once it has finished, None while it has none, and
    `files` what each download of a result sends, in order. It reports itself
    healthy, and its queue is always empty: nobody else is using it. With
    `gate`, the reply to the prompt waits for it, and `submitted` is set once
    the prompt has arrived.
    """

    def __init__(
        self,
        *,
        answer: Callable[[], httpx.Response] = _accepted,
        upload: Callable[[], httpx.Response] = _uploaded,
        record: dict[str, Any] | None = None,
        files: Sequence[Callable[[], httpx.Response]] = (),
    ) -> None:
        self.answer = answer
        self.upload = upload
        self.record = record
        self.files = list(files)
        self.gate: asyncio.Event | None = None
        self.submitted = asyncio.Event()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        route = (request.method, request.url.path)
        if route == ("GET", "/system_stats"):
            return httpx.Response(200, json={"system": {"comfyui_version": "constructed"}})
        if route == ("GET", "/prompt"):
            return httpx.Response(200, json={"exec_info": {"queue_remaining": 0}})
        if route == ("POST", "/prompt"):
            self.submitted.set()
            if self.gate is not None:
                await self.gate.wait()
            return self.answer()
        if route == ("POST", "/upload/image"):
            return self.upload()
        if route == ("POST", f"/api/jobs/{PROMPT_ID}/cancel"):
            return httpx.Response(200, json={"cancelled": True})
        if route == ("GET", f"/history/{PROMPT_ID}"):
            return httpx.Response(200, json={} if self.record is None else {PROMPT_ID: self.record})
        if route == ("GET", "/view") and self.files:
            return self.files.pop(0)()
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url.path}")


class _Socket:
    """A progress socket that reports `events` for the prompt, then stays silent."""

    def __init__(self, events: Sequence[dict[str, Any]]) -> None:
        self.messages = [json.dumps(event) for event in events]

    async def __aenter__(self) -> _Socket:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def __aiter__(self) -> _Socket:
        return self

    async def __anext__(self) -> str:
        if self.messages:
            return self.messages.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("The constructed socket stays silent")


async def _adapter(
    monkeypatch: pytest.MonkeyPatch,
    comfy: _ComfyUI,
    events: Sequence[dict[str, Any]] = (),
    *,
    max_output_bytes: int = 512 * 1024**2,
    inactivity_seconds: float = 60,
) -> ComfyUIAdapter:
    """The ComfyUI adapter talking to `comfy`, whose progress socket reports `events`."""

    monkeypatch.setattr(
        "local_lm.adapters.comfyui.websockets.connect",
        lambda *_args, **_kwargs: _Socket(events),
    )
    adapter = ComfyUIAdapter(
        "http://comfy.test",
        inactivity_seconds=inactivity_seconds,
        max_output_bytes=max_output_bytes,
    )
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(comfy)
    )
    return adapter


@dataclass(frozen=True)
class _Generation:
    """One generation of `request` against the constructed ComfyUI.

    `max_output_bytes` and `inactivity_seconds` configure the adapter under
    test. The other fields after `request` set up `_ComfyUI`, and `events` are
    what the progress socket reports once the prompt is in.
    """

    request: MediaRequest = field(default_factory=_request)
    answer: Callable[[], httpx.Response] = _accepted
    upload: Callable[[], httpx.Response] = _uploaded
    events: tuple[dict[str, Any], ...] = ()
    record: dict[str, Any] | None = None
    files: tuple[Callable[[], httpx.Response], ...] = ()
    max_output_bytes: int = 512 * 1024**2
    inactivity_seconds: float = 60


async def _failure(
    monkeypatch: pytest.MonkeyPatch, generation: _Generation
) -> RuntimeError | ValueError:
    """How the generation fails; every failure of the adapter is one of these two types."""

    # A record that never appears is waited for only briefly.
    monkeypatch.setattr("local_lm.adapters.comfyui._HISTORY_SETTLE_SECONDS", 0.05)
    monkeypatch.setattr("local_lm.adapters.comfyui._HISTORY_POLL_SECONDS", 0.01)
    comfy = _ComfyUI(
        answer=generation.answer,
        upload=generation.upload,
        record=generation.record,
        files=generation.files,
    )
    adapter = await _adapter(
        monkeypatch,
        comfy,
        generation.events,
        max_output_bytes=generation.max_output_bytes,
        inactivity_seconds=generation.inactivity_seconds,
    )
    try:
        with pytest.raises((RuntimeError, ValueError)) as captured:
            [event async for event in adapter.generate(generation.request)]
    finally:
        await adapter.close()
    failure = captured.value
    assert isinstance(failure, (RuntimeError, ValueError))
    return failure


@pytest.mark.parametrize(
    ("generation", "kind", "message"),
    [
        pytest.param(
            _Generation(answer=lambda: httpx.Response(400, json=REFUSED_GRAPH)),
            RuntimeError,
            "ComfyUI rejected a generation request (HTTP 400)",
            id="graph-refused",
        ),
        pytest.param(
            _Generation(events=(_node_failed("ValueError", "The scheduler is not one it knows"),)),
            RuntimeError,
            "ComfyUI KSampler failed (ValueError)",
            id="node-refused-its-settings",
        ),
        pytest.param(
            _Generation(
                events=(
                    _node_failed(
                        "RuntimeError",
                        "mat1 and mat2 shapes cannot be multiplied (77x768 and 1024x320)",
                    ),
                )
            ),
            RuntimeError,
            "ComfyUI KSampler failed (RuntimeError)",
            id="node-given-mismatched-shapes",
        ),
        pytest.param(
            _Generation(events=(SUCCEEDED,), record=_recorded(0)),
            RuntimeError,
            "ComfyUI completed without collectible image or video outputs",
            id="recorded-without-outputs",
        ),
        pytest.param(
            _Generation(events=(SUCCEEDED,), record=_recorded(65)),
            RuntimeError,
            "ComfyUI returned more than 64 outputs",
            id="too-many-outputs",
        ),
        pytest.param(
            _Generation(
                events=(SUCCEEDED,),
                record=_recorded(1),
                files=(_picture(b"12345"),),
                max_output_bytes=4,
            ),
            RuntimeError,
            "ComfyUI outputs exceed the 4-byte limit",
            id="output-declared-too-large",
        ),
        pytest.param(
            _Generation(
                events=(SUCCEEDED,),
                record=_recorded(1),
                files=(_streamed_picture(b"123", b"45"),),
                max_output_bytes=4,
            ),
            RuntimeError,
            "ComfyUI outputs exceed the 4-byte limit",
            id="output-sent-too-large",
        ),
        pytest.param(
            _Generation(
                events=(SUCCEEDED,),
                record=_recorded(2),
                files=(_picture(b"1234"),),
                max_output_bytes=4,
            ),
            RuntimeError,
            "ComfyUI outputs exceed the 4-byte limit",
            id="outputs-past-the-limit-together",
        ),
        pytest.param(
            _Generation(
                request=_request(
                    workflow={"load": {"class_type": "LoadImage", "inputs": {"image": "${mask}"}}}
                )
            ),
            ValueError,
            "This workflow has an input this request leaves empty, so it was not sent to "
            "ComfyUI. Choose a workflow made for this kind of request.",
            id="input-left-empty",
        ),
        pytest.param(
            _Generation(
                request=_request("image_to_image", inputs=(PICTURE,)),
                upload=lambda: httpx.Response(413),
            ),
            ValueError,
            "ComfyUI refused an input larger than its upload limit",
            id="input-over-the-upload-limit",
        ),
        pytest.param(
            _Generation(request=_request("image_to_image")),
            ValueError,
            "image_to_image requires a conditioning image",
            id="nothing-to-edit",
        ),
        pytest.param(
            _Generation(
                request=_request(
                    "image_to_image",
                    inputs=(VIDEO,),
                    workflow={**GRAPH, "clip": {"inputs": {"file": "${input_video}"}}},
                )
            ),
            ValueError,
            "image_to_image requires a conditioning image",
            id="no-picture-among-the-inputs",
        ),
        pytest.param(
            _Generation(request=_request("image_to_image", inputs=(b"plain text",))),
            ValueError,
            "ComfyUI conditioning inputs must be supported pictures or videos",
            id="input-neither-picture-nor-video",
        ),
        pytest.param(
            _Generation(request=_request(parameters={"mask": {"artifact_id": "selection"}})),
            ValueError,
            "The selection for this edit has no resolved file",
            id="selection-without-a-file",
        ),
        pytest.param(
            _Generation(
                request=_request(parameters={"mask": {"path": "selection"}}, mask=b"plain text")
            ),
            ValueError,
            "ComfyUI conditioning inputs must be supported raster images",
            id="selection-not-a-picture",
        ),
        pytest.param(
            _Generation(
                request=replace(
                    _request("image_to_image", inputs=(PICTURE,)),
                    input_contents=(PICTURE, PICTURE),
                )
            ),
            ValueError,
            "The input bytes do not match the conditioning images",
            id="bytes-for-inputs-not-sent",
        ),
        pytest.param(
            _Generation(request=_request(inputs=(VIDEO,))),
            ValueError,
            "This workflow does not read attached video 1 of 1",
            id="video-nothing-reads",
        ),
    ],
)
async def test_a_failure_of_the_workflow_or_its_inputs_is_marked_as_their_own(
    monkeypatch: pytest.MonkeyPatch,
    generation: _Generation,
    kind: type[Exception],
    message: str,
) -> None:
    """It keeps the type and the words it always had, so whatever caught it still does."""

    failure = await _failure(monkeypatch, generation)

    assert str(failure) == message
    assert isinstance(failure, kind)
    assert isinstance(failure, MediaWorkflowFailure)


@pytest.mark.parametrize(
    "exception_message",
    [
        pytest.param(
            "mat1 and mat2 shapes cannot be multiplied (77x768 and 1024x320)",
            id="matrix-shapes",
        ),
        pytest.param(
            "The size of tensor a (64) must match the size of tensor b (32) at non-singleton "
            "dimension 3",
            id="broadcast-sizes",
        ),
        pytest.param(
            "shape '[1, 4, 64, 64]' is invalid for input of size 32768",
            id="reshaped-size",
        ),
        pytest.param(
            "Sizes of tensors must match except in dimension 1. Expected size 64 but got size "
            "32 for tensor number 1 in the list.",
            id="joined-sizes",
        ),
        pytest.param(
            "Given groups=1, weight of size [320, 4, 3, 3], expected input[1, 16, 64, 64] to "
            "have 4 channels, but got 16 channels instead",
            id="convolution-channels",
        ),
        pytest.param(
            "Error(s) in loading state_dict for UNetModel:\n\tMissing key(s) in state_dict: "
            '"input_blocks.0.0.weight".',
            id="checkpoint-keys",
        ),
        pytest.param(
            "size mismatch for input_blocks.0.0.weight: copying a param with shape "
            "torch.Size([320, 4, 3, 3]) from checkpoint, the shape in current model is "
            "torch.Size([320, 8, 3, 3]).",
            id="checkpoint-shapes",
        ),
        pytest.param(
            "ERROR: Could not detect model type of: teapot.safetensors",
            id="unknown-model-type",
        ),
        pytest.param(
            "ERROR: VAE is invalid: None\n\nIf the VAE is from a checkpoint loader node your "
            "checkpoint does not contain a valid VAE.",
            id="model-file-without-its-part",
        ),
    ],
)
async def test_a_plain_runtime_error_about_parts_that_do_not_fit_is_marked(
    monkeypatch: pytest.MonkeyPatch, exception_message: str
) -> None:
    """Torch and ComfyUI say so when the tensors or model files a graph joins do not fit."""

    events = (_node_failed("RuntimeError", exception_message),)
    failure = await _failure(monkeypatch, _Generation(events=events))

    assert str(failure) == "ComfyUI KSampler failed (RuntimeError)"
    assert isinstance(failure, RuntimeError)
    assert isinstance(failure, MediaWorkflowFailure)


@pytest.mark.parametrize(
    "exception_message",
    [
        # ComfyUI's allocator, out of room in its own words.
        pytest.param("VRAM grow failed: 1073741824 bytes", id="allocator-could-not-grow"),
        pytest.param("VRAM reservation failed", id="allocator-could-not-reserve"),
        pytest.param("Fault failed: 2", id="allocator-fault"),
        pytest.param("HostBuffer allocation failed", id="allocator-host-buffer"),
        pytest.param("HostBuffer.extend failed", id="allocator-host-buffer-growth"),
        pytest.param("xfer_file_write failed", id="allocator-transfer"),
        # Torch, about the device.
        pytest.param("CUDA error: an illegal memory access was encountered", id="device-fault"),
        pytest.param("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR", id="device-library-fault"),
        # Words about parts that do not fit, beside words about the allocator.
        pytest.param("size mismatch while growing VRAM", id="parts-and-the-allocator"),
        # Words that say nothing about what failed.
        pytest.param("The node could not finish", id="unplaced"),
        pytest.param("", id="without-words"),
    ],
)
async def test_any_other_plain_runtime_error_is_not_marked(
    monkeypatch: pytest.MonkeyPatch, exception_message: str
) -> None:
    """Torch reports a fault of the device as a plain RuntimeError, and ComfyUI's allocator
    reports a lack of room the same way."""

    events = (_node_failed("RuntimeError", exception_message),)
    failure = await _failure(monkeypatch, _Generation(events=events))

    assert str(failure) == "ComfyUI KSampler failed (RuntimeError)"
    assert isinstance(failure, RuntimeError)
    assert not isinstance(failure, MediaWorkflowFailure)


@pytest.mark.parametrize(
    ("exception_type", "exception_message"),
    [
        pytest.param("torch.OutOfMemoryError", "Allocation on device", id="out-of-memory"),
        pytest.param("torch.OutOfMemoryError", OUT_OF_MEMORY, id="out-of-memory-with-advice"),
        pytest.param("MemoryError", "", id="host-out-of-memory"),
        pytest.param("MemoryError", "VBAR OOM", id="allocator-out-of-memory"),
        pytest.param(
            "torch.AcceleratorError", "an operation could not be launched", id="device-fault"
        ),
        pytest.param(
            "ValueError", "extra_vram_headroom must be non-negative", id="allocator-headroom"
        ),
        # A node's own error, in the words the device or the allocator used.
        pytest.param("ValueError", "CUDA error: unspecified launch failure", id="cuda-error"),
        pytest.param(
            "ValueError",
            "HIP error: an illegal memory access was encountered",
            id="illegal-memory-access",
        ),
        pytest.param(
            "ValueError",
            "CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx",
            id="matrix-library-fault",
        ),
        pytest.param(
            "ValueError", "cuDNN error: CUDNN_STATUS_INTERNAL_ERROR", id="network-library-fault"
        ),
        pytest.param("ValueError", "device-side assert triggered", id="device-side-assertion"),
        pytest.param(
            "OSError",
            "[WinError 1455] The paging file is too small for this operation to complete",
            id="paging-file-exhausted",
        ),
        pytest.param("ValueError", "VBAR unpin failed", id="allocator-vbar"),
        pytest.param("ValueError", "HostBuffer allocation failed", id="allocator-host-buffer"),
        pytest.param(
            "ValueError", "hostbuf_file_reader_read failed", id="allocator-host-buffer-reader"
        ),
        pytest.param("ValueError", "comfy-aimdo is not initialized", id="allocator-itself"),
        pytest.param("ValueError", "Fault failed: 2", id="allocator-fault"),
    ],
)
async def test_a_node_error_naming_memory_or_the_device_is_not_marked(
    monkeypatch: pytest.MonkeyPatch, exception_type: str, exception_message: str
) -> None:
    """ComfyUI may not be as the run found it, so a restart is the clean start."""

    events = (_node_failed(exception_type, exception_message),)
    failure = await _failure(monkeypatch, _Generation(events=events))

    assert str(failure) == f"ComfyUI KSampler failed ({exception_type})"
    assert isinstance(failure, RuntimeError)
    assert not isinstance(failure, MediaWorkflowFailure)


def _partly_accepted() -> httpx.Response:
    """ComfyUI took the prompt, queued the outputs it could, and names the nodes it refused."""

    return httpx.Response(200, json={"prompt_id": PROMPT_ID, "node_errors": REFUSED_NODES})


@pytest.mark.parametrize(
    ("generation", "kind", "message"),
    [
        pytest.param(
            _Generation(answer=_partly_accepted),
            RuntimeError,
            "ComfyUI rejected the selected workflow",
            id="graph-taken-in-part",
        ),
        pytest.param(
            _Generation(
                events=(_node_failed("ExecutionBlocked", "Execution Blocked: input missing"),)
            ),
            RuntimeError,
            "ComfyUI KSampler failed (ExecutionBlocked)",
            id="node-blocked",
        ),
        pytest.param(
            _Generation(events=(_node_failed("RuntimeError", None),)),
            RuntimeError,
            "ComfyUI KSampler failed (RuntimeError)",
            id="error-without-its-message",
        ),
        pytest.param(
            _Generation(events=(_node_failed("ValueError", "Unknown", node_type="K Sampler"),)),
            RuntimeError,
            "ComfyUI could not execute the selected workflow",
            id="error-with-an-unusable-node-label",
        ),
        pytest.param(
            _Generation(events=(_node_failed("custom error", "The node could not finish"),)),
            RuntimeError,
            "ComfyUI could not execute the selected workflow",
            id="error-with-an-unusable-exception-label",
        ),
        pytest.param(
            _Generation(answer=lambda: httpx.Response(404)),
            RuntimeError,
            "ComfyUI rejected a generation request (HTTP 404)",
            id="prompt-route-missing",
        ),
        pytest.param(
            _Generation(answer=lambda: httpx.Response(413)),
            RuntimeError,
            "ComfyUI rejected a generation request (HTTP 413)",
            id="prompt-too-large",
        ),
        pytest.param(
            _Generation(answer=lambda: httpx.Response(500)),
            RuntimeError,
            "ComfyUI rejected a generation request (HTTP 500)",
            id="prompt-met-a-server-error",
        ),
        pytest.param(
            _Generation(
                request=_request("image_to_image", inputs=(PICTURE,)),
                upload=lambda: httpx.Response(400),
            ),
            RuntimeError,
            "ComfyUI rejected a generation request (HTTP 400)",
            id="upload-refused",
        ),
        pytest.param(
            _Generation(answer=_unreachable),
            RuntimeError,
            "ComfyUI generation transport failed",
            id="unreachable",
        ),
        pytest.param(
            _Generation(inactivity_seconds=0.01),
            RuntimeError,
            "ComfyUI stopped reporting generation activity for 0.01 seconds; the run was "
            "interrupted",
            id="silent",
        ),
        pytest.param(
            _Generation(events=(SUCCEEDED,)),
            RuntimeError,
            "ComfyUI completed without collectible image or video outputs",
            id="never-recorded",
        ),
        pytest.param(
            _Generation(
                events=({"type": "progress", "data": {"prompt_id": PROMPT_ID, "max": "many"}},)
            ),
            RuntimeError,
            "ComfyUI returned a malformed progress event",
            id="malformed-progress",
        ),
        pytest.param(
            _Generation(
                request=_request("image_to_image", inputs=(PICTURE,)),
                upload=lambda: httpx.Response(200, json=[]),
            ),
            ValueError,
            "ComfyUI returned an invalid conditioning-image upload",
            id="upload-answered-wrongly",
        ),
        pytest.param(
            _Generation(
                events=(SUCCEEDED,),
                record={"outputs": {"save": {"images": [{"filename": "../teapot.png"}]}}},
            ),
            ValueError,
            "ComfyUI output has an invalid filename",
            id="output-named-wrongly",
        ),
    ],
)
async def test_any_other_failure_is_not_marked(
    monkeypatch: pytest.MonkeyPatch,
    generation: _Generation,
    kind: type[Exception],
    message: str,
) -> None:
    """The connection, the worker or its answers may be at fault, nothing says what was, or
    ComfyUI may still be running the prompt."""

    failure = await _failure(monkeypatch, generation)

    assert str(failure) == message
    assert isinstance(failure, kind)
    assert not isinstance(failure, MediaWorkflowFailure)


async def _comfyui_engine(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    comfy: _ComfyUI,
    events: Sequence[dict[str, Any]] = (),
) -> ComfyUIAdapter:
    """Make the ComfyUI adapter the application's media engine, talking to `comfy`."""

    adapter = await _adapter(monkeypatch, comfy, events)
    monkeypatch.setattr(app.state.services.engines, "media", adapter)
    return adapter


def _ready_world(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> ReplacementWorld:
    """A ready ComfyUI worker that a run uses as it is, with chat loaded beside it."""

    world = ReplacementWorld(app, settings, monkeypatch)
    world.scope = None
    return world


async def _handed_back(world: ReplacementWorld) -> list[str]:
    """What the workers did once the device went back to chat and ComfyUI came back empty.

    The run stopped chat, and its handoff stopped ComfyUI, loaded a chat model
    and started ComfyUI again: four effects, with nothing left owed.
    """

    async def settled() -> tuple[list[str], str | None]:
        return list(world.order), world.orchestrator._displaced_chat_profile_id

    order, owed = await wait_until(
        settled,
        lambda seen: len(seen[0]) >= 4 and seen[1] is None,
        what="the device to go back to chat",
    )
    assert world.orchestrator._media_warm_window is None
    return order


@pytest.mark.parametrize(
    ("answer", "events", "error"),
    [
        pytest.param(
            lambda: httpx.Response(400, json=REFUSED_GRAPH),
            (),
            "ComfyUI rejected a generation request (HTTP 400)",
            id="graph-refused",
        ),
        pytest.param(
            _accepted,
            (_node_failed("ValueError", "The label is not one it knows", node_type="Source"),),
            "ComfyUI Source failed (ValueError)",
            id="node-refused-its-settings",
        ),
    ],
)
async def test_a_run_that_failed_on_its_own_workflow_keeps_comfyui_for_the_next_one(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    answer: Callable[[], httpx.Response],
    events: tuple[dict[str, Any], ...],
    error: str,
) -> None:
    world = _ready_world(app, settings, monkeypatch)
    await choose_retries(client, 0)
    adapter = await _comfyui_engine(app, monkeypatch, _ComfyUI(answer=answer), events)
    try:
        _chat_id, run_id = await _image_turn(client)
        run = await finished(client, run_id, "failed")
    finally:
        await adapter.close()

    assert run["error"] == error
    # ComfyUI is kept as after a finished run, and the chat model it displaced
    # stays owed until the window ends or something needs chat.
    assert world.order == ["stop chat"]
    assert world.orchestrator._media_warm_window is not None
    assert world.orchestrator._displaced_chat_profile_id == "profile-chat"


@pytest.mark.parametrize(
    ("answer", "events", "error"),
    [
        pytest.param(
            _accepted,
            (_node_failed("torch.OutOfMemoryError", OUT_OF_MEMORY, node_type="Source"),),
            "ComfyUI Source failed (torch.OutOfMemoryError)",
            id="out-of-memory",
        ),
        pytest.param(
            _accepted,
            (
                _node_failed(
                    "RuntimeError", "VRAM grow failed: 1073741824 bytes", node_type="Source"
                ),
            ),
            "ComfyUI Source failed (RuntimeError)",
            id="allocator-out-of-room",
        ),
        pytest.param(
            _partly_accepted,
            (),
            "ComfyUI rejected the selected workflow",
            id="graph-taken-in-part",
        ),
        pytest.param(
            _accepted,
            (
                _node_failed(
                    "ExecutionBlocked", "Execution Blocked: input missing", node_type="Source"
                ),
            ),
            "ComfyUI Source failed (ExecutionBlocked)",
            id="node-blocked",
        ),
    ],
)
async def test_a_failure_that_may_leave_comfyui_short_of_room_or_busy_still_recycles_it(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    answer: Callable[[], httpx.Response],
    events: tuple[dict[str, Any], ...],
    error: str,
) -> None:
    """Out of memory or of room, or still running part of the prompt, ComfyUI is started again."""

    world = _ready_world(app, settings, monkeypatch)
    await choose_retries(client, 0)
    adapter = await _comfyui_engine(app, monkeypatch, _ComfyUI(answer=answer), events)
    try:
        _chat_id, run_id = await _image_turn(client)
        run = await finished(client, run_id, "failed")
        order = await _handed_back(world)
    finally:
        await adapter.close()

    assert run["error"] == error
    assert order == ["stop chat", "stop media", "resume profile-chat", "start media"]


async def test_any_other_failure_still_recycles_comfyui_as_before(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _ready_world(app, settings, monkeypatch)
    await choose_retries(client, 0)
    failing_media(app, monkeypatch, 1)

    _chat_id, run_id = await _image_turn(client)
    run = await finished(client, run_id, "failed")

    assert run["error"] == "Temporary media engine failure"
    assert await _handed_back(world) == [
        "stop chat",
        "stop media",
        "resume profile-chat",
        "start media",
    ]


async def test_a_workflow_failure_still_recycles_comfyui_when_keeping_it_warm_is_off(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    await choose_retries(client, 0)
    comfy = _ComfyUI(answer=lambda: httpx.Response(400, json=REFUSED_GRAPH))
    adapter = await _comfyui_engine(app, monkeypatch, comfy)
    try:
        _chat_id, run_id = await _image_turn(client)
        run = await finished(client, run_id, "failed")
        order = await _handed_back(world)
    finally:
        await adapter.close()

    assert run["error"] == "ComfyUI rejected a generation request (HTTP 400)"
    assert order == ["stop chat", "stop media", "resume profile-chat", "start media"]


async def test_a_text_turn_queued_behind_a_workflow_failure_still_has_comfyui_recycled_first(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The text turn's model loads at once, and ComfyUI comes back once the turn is done."""

    world = _ready_world(app, settings, monkeypatch)
    await choose_retries(client, 0)
    comfy = _ComfyUI(answer=lambda: httpx.Response(400, json=REFUSED_GRAPH))
    comfy.gate = asyncio.Event()
    adapter = await _comfyui_engine(app, monkeypatch, comfy)
    try:
        _chat_id, run_id = await _image_turn(client)
        await asyncio.wait_for(comfy.submitted.wait(), timeout=PATIENCE_SECONDS)
        other = (await client.post("/api/chats", json={"title": "Meanwhile"})).json()
        question = await client.post(
            f"/api/chats/{other['id']}/turns",
            json={"text": "Name three primary colors.", "mode": "text"},
        )
        assert question.status_code == 202, question.text
        comfy.gate.set()
        run = await finished(client, run_id, "failed")
        answered = await finished(client, question.json()["run"]["id"])
        order = await _handed_back(world)
    finally:
        await adapter.close()

    assert run["error"] == "ComfyUI rejected a generation request (HTTP 400)"
    assert isinstance(answered["profile_id"], str)
    assert order == ["stop chat", "stop media", f"resume {answered['profile_id']}", "start media"]
