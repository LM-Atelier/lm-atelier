"""A video handed to a ComfyUI workflow is uploaded as a video and bound apart from pictures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_lm.adapters.base import MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.media_references import unread_videos
from local_lm.prompt_binding import frozen_prompt, substitutable_keys
from local_lm.settings_registry import IMAGE_SETTINGS, VIDEO_SETTINGS, workflow_settings

PNG = b"\x89PNG\r\n\x1a\n" + b"picture"
MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64
MOV = b"\x00\x00\x00\x14ftypqt  \x20\x05\x03\x00" + b"\x00" * 64
WEBM = b"\x1aE\xdf\xa3\x9fB\x86\x81\x01B\xf7\x81\x01B\x82\x84webm" + b"\x00" * 64
MKV = b"\x1aE\xdf\xa3\xa3B\x86\x81\x01B\xf7\x81\x01B\x82\x88matroska" + b"\x00" * 64
AVI = b"RIFF\x00\x10\x00\x00AVI LIST" + b"\x00" * 64
AVIF = b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1" + b"\x00" * 64
HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64


@pytest.mark.parametrize(
    "head,expected",
    [
        (PNG, ("image", ".png", "image/png")),
        (b"\xff\xd8\xff\xe0jpeg", ("image", ".jpg", "image/jpeg")),
        (MP4, ("video", ".mp4", "video/mp4")),
        (MOV, ("video", ".mov", "video/quicktime")),
        (WEBM, ("video", ".webm", "video/webm")),
        (MKV, ("video", ".mkv", "video/x-matroska")),
        (AVI, ("video", ".avi", "video/x-msvideo")),
    ],
    ids=["png", "jpeg", "mp4", "mov", "webm", "mkv", "avi"],
)
def test_an_input_is_told_a_picture_or_a_video_by_its_container(
    head: bytes, expected: tuple[str, str, str]
) -> None:
    assert ComfyUIAdapter._input_format(head) == expected


@pytest.mark.parametrize("head", [AVIF, HEIC, b"plain text"], ids=["avif", "heic", "text"])
def test_a_file_comfyui_loads_as_neither_is_refused(head: bytes) -> None:
    with pytest.raises(ValueError, match="supported pictures or videos"):
        ComfyUIAdapter._input_format(head)


class _FakeComfy:
    """Answers uploads as ComfyUI does, and keeps what each one sent."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.uploads: list[bytes] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/upload/image"
        body = await request.aread()
        self.uploads.append(body)
        if self.status != 200:
            return httpx.Response(self.status)
        start = body.index(b'filename="') + len(b'filename="')
        name = body[start : body.index(b'"', start)].decode()
        return httpx.Response(200, json={"name": name, "subfolder": "lm-atelier", "type": "temp"})


async def _parameters(
    comfy: _FakeComfy,
    inputs: list[Path],
    workflow: dict[str, Any],
    operation: str = "image_to_video",
) -> dict[str, Any]:
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(comfy.handle)
    )
    try:
        return await adapter._request_parameters(
            MediaRequest(
                run_id="run_video",
                operation=operation,
                prompt="Keep the motion",
                negative_prompt=None,
                input_paths=inputs,
                workflow=workflow,
                parameters={},
            )
        )
    finally:
        await adapter.close()


def _write(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


async def test_a_picture_and_a_video_are_uploaded_as_themselves_and_bound_apart(
    tmp_path: Path,
) -> None:
    """The video comes first here, so a binding by position alone would give it to the picture."""
    comfy = _FakeComfy()
    clip = _write(tmp_path, "guide", MP4)
    picture = _write(tmp_path, "reference", PNG)
    workflow = {
        "picture": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "clip": {"class_type": "LoadVideo", "inputs": {"file": "${input_video}"}},
    }

    parameters = await _parameters(comfy, [clip, picture], workflow)

    video = "lm-atelier/lm-atelier-run_video-0.mp4 [temp]"
    image = "lm-atelier/lm-atelier-run_video-1.png [temp]"
    assert parameters["input_video"] == video
    assert parameters["input_videos"] == [video]
    assert parameters["input_video_0"] == video
    assert parameters["input_image"] == image
    assert parameters["input_images"] == [image]
    assert parameters["input_image_0"] == image
    assert b'filename="lm-atelier-run_video-0.mp4"' in comfy.uploads[0]
    assert b"Content-Type: video/mp4" in comfy.uploads[0]
    assert MP4 in comfy.uploads[0]
    assert b"Content-Type: image/png" in comfy.uploads[1]
    compiled = ComfyUIAdapter._compile(workflow, parameters)
    assert compiled["clip"]["inputs"]["file"] == video
    assert compiled["picture"]["inputs"]["image"] == image


async def test_each_video_has_its_own_number_and_a_missing_one_is_not_filled(
    tmp_path: Path,
) -> None:
    """A picture slot past the last picture reuses it; a video slot past the last stays empty."""
    comfy = _FakeComfy()
    inputs = [
        _write(tmp_path, "picture", PNG),
        _write(tmp_path, "first", MP4),
        _write(tmp_path, "second", WEBM),
    ]
    workflow = {
        "a": {"class_type": "LoadVideo", "inputs": {"file": "${input_video_0}"}},
        "b": {"class_type": "LoadVideo", "inputs": {"file": "${input_video_1}"}},
        "c": {"class_type": "LoadVideo", "inputs": {"file": "${input_video_2}"}},
    }

    parameters = await _parameters(comfy, inputs, workflow)

    assert parameters["input_video_0"].endswith("-1.mp4 [temp]")
    assert parameters["input_video_1"].endswith("-2.webm [temp]")
    assert "input_video_2" not in parameters
    assert ComfyUIAdapter._leaves_unfilled(workflow, parameters) is True


async def test_a_picture_turn_given_only_a_video_is_refused_before_anything_is_sent(
    tmp_path: Path,
) -> None:
    comfy = _FakeComfy()

    with pytest.raises(ValueError, match="image_to_video requires a conditioning image"):
        await _parameters(comfy, [_write(tmp_path, "guide", MP4)], {})

    assert comfy.uploads == []


async def test_a_video_is_streamed_from_disk_rather_than_read_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comfy = _FakeComfy()
    clip = _write(tmp_path, "guide", MP4 + b"\x01" * 200_000)
    picture = _write(tmp_path, "reference", PNG)
    read_whole = Path.read_bytes

    def refuse_the_video(path: Path) -> bytes:
        assert path != clip, "the video was read whole into memory"
        return read_whole(path)

    monkeypatch.setattr(Path, "read_bytes", refuse_the_video)

    parameters = await _parameters(comfy, [picture, clip], {"clip": "${input_video}"})

    assert parameters["input_video"].endswith(".mp4 [temp]")
    assert MP4 + b"\x01" * 200_000 in comfy.uploads[1]


async def test_a_selection_is_still_only_ever_a_picture(tmp_path: Path) -> None:
    comfy = _FakeComfy()
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(comfy.handle)
    )
    clip = _write(tmp_path, "mask", MP4)
    try:
        with pytest.raises(ValueError, match="supported raster images"):
            await adapter._request_parameters(
                MediaRequest(
                    run_id="run_video",
                    operation="image_to_image",
                    prompt="Edit the selection",
                    negative_prompt=None,
                    input_paths=[_write(tmp_path, "picture", PNG)],
                    workflow={},
                    parameters={"mask": {"path": str(clip)}},
                )
            )
    finally:
        await adapter.close()
    assert comfy.uploads == []


async def test_an_input_over_comfyuis_upload_limit_is_named_as_such(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="larger than its upload limit"):
        await _parameters(
            _FakeComfy(status=413),
            [_write(tmp_path, "picture", PNG), _write(tmp_path, "guide", MP4)],
            {"clip": "${input_video}"},
        )


@pytest.mark.parametrize("key", ["input_video", "input_videos", "input_video_3", "INPUT_VIDEO_12"])
def test_a_workflow_cannot_declare_the_video_bindings_as_its_own_settings(key: str) -> None:
    with pytest.raises(ValueError, match=f"reserved setting key {key}"):
        workflow_settings(VIDEO_SETTINGS, {"properties": {key: {"type": "string", "default": "x"}}})
    with pytest.raises(ValueError, match=f"reserved setting key {key}"):
        workflow_settings(IMAGE_SETTINGS, {"properties": {key: {"type": "string", "default": "x"}}})


@pytest.mark.parametrize("placeholder", ["${input_video}", "${input_videos}", "${input_video_12}"])
def test_a_graph_reading_a_video_binding_is_reading_something_the_run_supplies(
    placeholder: str,
) -> None:
    keys = substitutable_keys("image_to_video", None)
    assert keys is not None
    graph: dict[str, Any] = {
        "encode": {"class_type": "CLIPTextEncode", "inputs": {"text": placeholder}},
        "sampler": {"class_type": "KSampler", "inputs": {"positive": ["encode", 0]}},
    }
    assert frozen_prompt(graph, keys) is False
    graph["encode"]["inputs"]["text"] = "${input_video_12_style}"
    assert frozen_prompt(graph, keys) is True


@pytest.mark.parametrize(
    "brand,expected",
    [
        (b"M4V ", ("video", ".mp4", "video/mp4")),
        (b"3gp5", ("video", ".mp4", "video/mp4")),
        (b"heis", None),
        (b"mif2", None),
        (b"avci", None),
        (b"M4A ", None),
        (b"crx ", None),
    ],
    ids=["m4v", "3gp", "heif-sequence", "heif-2", "avc-picture", "audio", "camera-raw"],
)
def test_only_video_brands_of_the_mp4_box_are_videos(
    brand: bytes, expected: tuple[str, str, str] | None
) -> None:
    head = b"\x00\x00\x00\x18ftyp" + brand + b"\x00\x00\x00\x00" + b"\x00" * 64
    if expected is None:
        with pytest.raises(ValueError, match="supported pictures or videos"):
            ComfyUIAdapter._input_format(head)
    else:
        assert ComfyUIAdapter._input_format(head) == expected


@pytest.mark.parametrize(
    "workflow,unread",
    [
        ({"picture": {"inputs": {"image": "${input_image}"}}}, 1),
        ({"clip": {"inputs": {"file": "${input_video}"}}}, 2),
        ({"clip": {"inputs": {"file": "${input_video_1}"}}}, 1),
    ],
    ids=["no-video-slot", "one-slot-two-videos", "only-the-second-slot"],
)
async def test_a_video_the_workflow_never_reads_is_refused_before_anything_is_sent(
    tmp_path: Path, workflow: dict[str, Any], unread: int
) -> None:
    """Nothing repeats a video into a free slot, so an unread one would be dropped silently."""
    comfy = _FakeComfy()
    inputs = [
        _write(tmp_path, "picture", PNG),
        _write(tmp_path, "first", MP4),
        _write(tmp_path, "second", WEBM),
    ]

    with pytest.raises(ValueError, match=f"does not read attached video {unread} of 2"):
        await _parameters(comfy, inputs, workflow)

    assert comfy.uploads == []


async def test_a_workflow_reading_the_whole_list_takes_every_video(tmp_path: Path) -> None:
    comfy = _FakeComfy()
    inputs = [
        _write(tmp_path, "picture", PNG),
        _write(tmp_path, "first", MP4),
        _write(tmp_path, "second", WEBM),
    ]

    parameters = await _parameters(
        comfy, inputs, {"clips": {"inputs": {"files": "${input_videos}"}}}
    )

    assert len(parameters["input_videos"]) == 2


@pytest.mark.parametrize(
    "workflow,supplied,unread",
    [
        ({}, 0, []),
        ({}, 2, [0, 1]),
        ({"a": "${input_video}"}, 1, []),
        ({"a": "${input_video}", "b": "${input_video_2}"}, 3, [1]),
        ({"a": ["${input_videos}"]}, 5, []),
        ({"a": "${input_video_0_style}"}, 1, [0]),
    ],
)
def test_unread_videos_lists_every_attached_video_without_a_slot(
    workflow: dict[str, Any], supplied: int, unread: list[int]
) -> None:
    assert unread_videos(workflow, supplied) == unread


class _Socket:
    """ComfyUI's progress socket, saying only that the prompt finished."""

    def __init__(self, prompt_id: str) -> None:
        self.messages = iter(
            [json.dumps({"type": "executing", "data": {"prompt_id": prompt_id, "node": None}})]
        )

    async def __aenter__(self) -> _Socket:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def __aiter__(self) -> _Socket:
        return self

    async def __anext__(self) -> str:
        try:
            return next(self.messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


async def test_the_video_a_workflow_loaded_is_not_collected_as_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ComfyUI's LoadVideo shows the clip it loaded as an input preview; only the save is a result.

    Driven through the whole generation, so it is the run's own upload that is left out.
    """
    prompt_id = "prompt-loaded-video"
    viewed: list[str] = []
    prompted: list[dict[str, Any]] = []

    async def comfy(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/upload/image":
            body = await request.aread()
            start = body.index(b'filename="') + len(b'filename="')
            name = body[start : body.index(b'"', start)].decode()
            return httpx.Response(
                200, json={"name": name, "subfolder": "lm-atelier", "type": "temp"}
            )
        if request.url.path == "/prompt":
            prompted.append(json.loads(await request.aread()))
            return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})
        if request.url.path == f"/history/{prompt_id}":
            loaded = {
                "filename": "lm-atelier-run_video-1.mp4",
                "subfolder": "lm-atelier",
                "type": "input",
            }
            saved = {"filename": "result.mp4", "subfolder": "", "type": "output"}
            outputs = {
                "clip": {"images": [loaded], "animated": [True]},
                "save": {"images": [saved], "animated": [True]},
            }
            return httpx.Response(200, json={prompt_id: {"outputs": outputs}})
        if request.url.path == "/view":
            viewed.append(request.url.params["filename"])
            return httpx.Response(200, content=b"mp4", headers={"content-type": "video/mp4"})
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    monkeypatch.setattr(
        "local_lm.adapters.comfyui.websockets.connect", lambda *_a, **_k: _Socket(prompt_id)
    )
    adapter = ComfyUIAdapter("http://comfy.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(comfy)
    )
    workflow = {
        "picture": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "clip": {"class_type": "LoadVideo", "inputs": {"file": "${input_video}"}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["clip", 0]}},
    }
    request = MediaRequest(
        run_id="run_video",
        operation="image_to_video",
        prompt="Keep the motion",
        negative_prompt=None,
        input_paths=[_write(tmp_path, "picture", PNG), _write(tmp_path, "guide", MP4)],
        workflow=workflow,
        parameters={},
    )
    try:
        generated = [event async for event in adapter.generate(request)]
    finally:
        await adapter.close()

    loaded_name = "lm-atelier/lm-atelier-run_video-1.mp4 [temp]"
    assert prompted[0]["prompt"]["clip"]["inputs"]["file"] == loaded_name
    assert viewed == ["result.mp4"]
    complete = [event for event in generated if event.type == "complete"]
    assert len(complete) == 1 and len(complete[0].assets) == 1
