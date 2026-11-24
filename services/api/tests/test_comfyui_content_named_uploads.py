"""ComfyUI inputs are uploaded under names of their own content where a repeat stays safe.

ComfyUI reuses a node's cached outputs only while the node's inputs are unchanged.
An upload named for the run that sends it gives LoadImage a new input on every run,
so the same picture is loaded and encoded again by each regeneration that starts
from it. Named by content, the same bytes bind the same input on every run. That is
safe only while every node that writes a file is moved to a folder of the
submission's own: ComfyUI answers an unchanged saver from its cache with the files
an earlier submission wrote, and a managed backend's collected files are gone. Where
that does not hold, uploads keep the run's own names.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import pytest

from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.adapters.contracts import close_iterator

PICTURE = b"\x89PNG\r\n\x1a\n" + b"a plain grey square"
OTHER_PICTURE = b"\x89PNG\r\n\x1a\n" + b"a plain blue circle"
MASK = b"\x89PNG\r\n\x1a\n" + b"the left half"
OTHER_MASK = b"\x89PNG\r\n\x1a\n" + b"the left half, as drawn"
#: Two pictures that agree for longer than the head read to tell a picture from a video.
_SHARED_START = b"\x89PNG\r\n\x1a\n" + b"a plain grey field " * 300
LEFT_PICTURE = _SHARED_START + b"with a dot on the left"
RIGHT_PICTURE = _SHARED_START + b"with a dot on the right"
_MP4_HEAD = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
CLIP = _MP4_HEAD + bytes(64)
#: Three times the size the adapter reads at once, so a whole read stands out.
LONG_CLIP = _MP4_HEAD + bytes(range(256)) * (12 * 1024)


def _name(content: bytes, extension: str) -> str:
    """The name an input with these bytes is uploaded under when named by its content."""
    return f"lm-atelier-{hashlib.sha256(content).hexdigest()}{extension}"


def _bound(content: bytes, extension: str) -> str:
    """The value a workflow input loading that upload is given."""
    return f"lm-atelier/{_name(content, extension)} [temp]"


def _write(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


def _form(request: httpx.Request, body: bytes) -> tuple[dict[str, str], str, bytes]:
    """The plain fields of a multipart upload, and the name and bytes of its one file."""
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: "
        + request.headers["content-type"].encode("ascii")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    for part in message.walk():
        if part.is_multipart():
            continue
        payload = part.get_payload(decode=True)
        assert isinstance(payload, bytes)
        filename = part.get_filename()
        if filename:
            files.append((filename, payload))
        else:
            field = part.get_param("name", header="content-disposition")
            assert isinstance(field, str)
            fields[field] = payload.decode()
    [(filename, content)] = files
    return fields, filename, content


def _is_link(value: Any, graph: dict[str, Any]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and value[0] in graph
        and type(value[1]) is int
    )


class _Upload(NamedTuple):
    name: str
    content: bytes
    fields: dict[str, str]


#: The node classes the fake knows, and whether ComfyUI runs each as an output.
_CATALOGUE = {
    "LoadImage": False,
    "LoadImageMask": False,
    "LoadVideo": False,
    "ImageScaleBy": False,
    "ImageCompositeMasked": False,
    "ImageBatch": False,
    "EmptyImage": False,
    "PrimitiveString": False,
    "FilenameFormatter": False,
    "SaveImage": True,
    "SaveVideo": True,
    "SaveText": True,
    "Image Saver": True,
}


class _ComfyUI:
    """A ComfyUI that answers unchanged nodes from its cache, as its default cache does.

    A node is keyed by its class, its inputs, the keys of the nodes it reads and
    what its loader reads: LoadImage and LoadImageMask hash the file, and
    LoadVideo goes by when the file was last written, which an upload renews.
    Before each prompt, keys the prompt does not use are dropped. A node whose
    key is still held does not run, and the history repeats what it showed when
    it last ran. Savers write real files under the output root, uploads replace
    any file of the same name under the temp root, and a missing file is a 404.
    Its catalogue lists the classes above, less any a test leaves out.
    """

    def __init__(self, root: Path) -> None:
        self.output_root = root / "output"
        self.temp_root = root / "temp"
        self.output_root.mkdir()
        self.temp_root.mkdir()
        self.uploads: list[_Upload] = []
        self.graphs: list[dict[str, Any]] = []
        self.ran: list[list[str]] = []
        self.viewed: list[tuple[str, int]] = []
        self.shown: dict[int, list[dict[str, str]]] = {}
        self.catalogue_status = 200
        self.unknown_classes: set[str] = set()
        self.catalogue_requests = 0
        self.refuse_prompts = False
        self.fail_runs = False
        self.hold_runs = False
        self._cache: dict[Any, tuple[list[Any], dict[str, Any] | None]] = {}
        self._writes: dict[Path, int] = {}
        self._histories: dict[str, dict[str, Any]] = {}

    def _stored(self, reference: str) -> Path:
        folder, _, name = reference.removesuffix(" [temp]").rpartition("/")
        return self.temp_root / folder / name

    def _fingerprint(self, node: dict[str, Any]) -> Any:
        if node["class_type"] in {"LoadImage", "LoadImageMask"}:
            return hashlib.sha256(self._stored(node["inputs"]["image"]).read_bytes()).hexdigest()
        if node["class_type"] == "LoadVideo":
            return self._writes[self._stored(node["inputs"]["file"])]
        return None

    def _keys(self, graph: dict[str, Any]) -> dict[str, Any]:
        keys: dict[str, Any] = {}

        def key(node_id: str) -> Any:
            if node_id not in keys:
                node = graph[node_id]
                keys[node_id] = (
                    node["class_type"],
                    self._fingerprint(node),
                    tuple(
                        (name, ("link", key(value[0]), value[1]))
                        if _is_link(value, graph)
                        else (name, json.dumps(value, sort_keys=True))
                        for name, value in sorted(node["inputs"].items())
                    ),
                )
            return keys[node_id]

        for node_id in graph:
            key(node_id)
        return keys

    @staticmethod
    def _order(graph: dict[str, Any]) -> list[str]:
        order: list[str] = []

        def visit(node_id: str) -> None:
            if node_id in order:
                return
            for value in graph[node_id]["inputs"].values():
                if _is_link(value, graph):
                    visit(value[0])
            order.append(node_id)

        for node_id in graph:
            visit(node_id)
        return order

    def _save(self, folder: str, stem: str, extension: str) -> dict[str, str]:
        directory = self.output_root / folder if folder else self.output_root
        directory.mkdir(parents=True, exist_ok=True)
        counter = 1
        while (directory / f"{stem}_{counter:05}_{extension}").exists():
            counter += 1
        name = f"{stem}_{counter:05}_{extension}"
        (directory / name).write_bytes(f"{folder}/{name}".encode())
        return {"filename": name, "subfolder": folder, "type": "output"}

    def _run(
        self, graph: dict[str, Any], node_id: str, values: dict[str, list[Any]]
    ) -> tuple[list[Any], dict[str, Any] | None]:
        node = graph[node_id]
        inputs = {
            name: values[value[0]][value[1]] if _is_link(value, graph) else value
            for name, value in node["inputs"].items()
        }
        kind = node["class_type"]
        if kind in {"LoadImage", "LoadImageMask"}:
            content = self._stored(inputs["image"]).read_bytes()
            return [hashlib.sha256(content).hexdigest(), None], None
        if kind == "LoadVideo":
            folder, _, name = str(inputs["file"]).removesuffix(" [temp]").rpartition("/")
            shown = {"filename": name, "subfolder": folder, "type": "temp"}
            return [name], {"images": [shown], "animated": [True]}
        if kind == "PrimitiveString":
            return [inputs["value"]], None
        if kind in {"SaveImage", "SaveVideo"}:
            folder, _, stem = str(inputs["filename_prefix"]).rpartition("/")
            extension = ".mp4" if kind == "SaveVideo" else ".png"
            return [], {"images": [self._save(folder, stem, extension)]}
        if kind == "SaveText":
            folder, _, stem = str(inputs["filename_prefix"]).rpartition("/")
            self._save(folder, stem, ".txt")
            return [], {"text": [str(inputs["text"])]}
        if kind == "Image Saver":
            saved = self._save(str(inputs["path"]), str(inputs["filename"]), ".png")
            # It also passes on where it saved, for another node to read.
            return [f"{saved['subfolder']}/{saved['filename']}"], {"images": [saved]}
        return [repr(sorted(inputs.items()))], None

    def _execute(self, graph: dict[str, Any], prompt_id: str) -> None:
        keys = self._keys(graph)
        held = set(keys.values())
        self._cache = {key: entry for key, entry in self._cache.items() if key in held}
        values: dict[str, list[Any]] = {}
        history: dict[str, Any] = {}
        ran: list[str] = []
        for node_id in self._order(graph):
            entry = self._cache.get(keys[node_id])
            if entry is None:
                entry = self._cache[keys[node_id]] = self._run(graph, node_id, values)
                ran.append(node_id)
            values[node_id], shown = entry
            if shown is not None:
                history[node_id] = shown
        self.ran.append(ran)
        self._histories[prompt_id] = history

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/upload/image":
            fields, name, content = _form(request, await request.aread())
            stored = self.temp_root / fields["subfolder"] / name
            stored.parent.mkdir(parents=True, exist_ok=True)
            stored.write_bytes(content)
            self._writes[stored] = self._writes.get(stored, 0) + 1
            self.uploads.append(_Upload(name, content, fields))
            return httpx.Response(
                200, json={"name": name, "subfolder": fields["subfolder"], "type": fields["type"]}
            )
        if path == "/prompt":
            if self.refuse_prompts:
                return httpx.Response(500)
            self.graphs.append(json.loads(await request.aread())["prompt"])
            prompt_id = f"prompt-{len(self.graphs)}"
            self._execute(self.graphs[-1], prompt_id)
            return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})
        if path.startswith("/history/"):
            prompt_id = path.rpartition("/")[2]
            outputs = dict(self._histories.get(prompt_id, {}))
            if shown := self.shown.get(int(prompt_id.rpartition("-")[2])):
                outputs["preview"] = {"images": shown}
            return httpx.Response(200, json={prompt_id: {"outputs": outputs}})
        if path == "/view":
            params = request.url.params
            root = self.output_root if params["type"] == "output" else self.temp_root
            target = root / params["subfolder"] / params["filename"]
            status = 200 if target.is_file() else 404
            self.viewed.append((params["filename"], status))
            if status == 404:
                return httpx.Response(404)
            return httpx.Response(
                200, content=target.read_bytes(), headers={"content-type": "image/png"}
            )
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            return httpx.Response(200, json={})
        if path == "/object_info":
            self.catalogue_requests += 1
            if self.catalogue_status != 200:
                return httpx.Response(self.catalogue_status)
            return httpx.Response(
                200,
                json={
                    name: {"output_node": output}
                    for name, output in _CATALOGUE.items()
                    if name not in self.unknown_classes
                },
            )
        raise AssertionError(f"unexpected ComfyUI request: {request.method} {request.url}")

    def socket(self) -> Any:
        """ComfyUI's progress socket, reporting how the latest prompt went."""
        comfy = self

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
                if comfy.hold_runs:
                    await asyncio.Event().wait()
                if self.sent:
                    raise StopAsyncIteration
                self.sent = True
                data: dict[str, Any] = {"prompt_id": f"prompt-{len(comfy.graphs)}"}
                if comfy.fail_runs:
                    data |= {"node_type": "SaveImage", "exception_type": "RuntimeError"}
                    return json.dumps({"type": "execution_error", "data": data})
                return json.dumps({"type": "executing", "data": {**data, "node": None}})

        return Socket()


async def _adapter(
    comfy: _ComfyUI, monkeypatch: pytest.MonkeyPatch, *, managed: bool = True
) -> ComfyUIAdapter:
    """An adapter for the fake, managing its outputs as the desktop app does unless told not to."""
    monkeypatch.setattr(
        "local_lm.adapters.comfyui.websockets.connect", lambda *_args, **_kwargs: comfy.socket()
    )
    adapter = (
        ComfyUIAdapter(
            "http://comfy.test",
            managed_output_root=comfy.output_root,
            managed_temp_root=comfy.temp_root,
        )
        if managed
        else ComfyUIAdapter("http://comfy.test")
    )
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(comfy.handle)
    )
    return adapter


async def _run(adapter: ComfyUIAdapter, request: MediaRequest) -> MediaEvent:
    """Generate one request to the end and return its last event."""
    events = [event async for event in adapter.generate(request)]
    return events[-1]


async def _outcome(adapter: ComfyUIAdapter, request: MediaRequest) -> str:
    """How a generation ended: its last event, or the failure it raised."""
    try:
        return (await _run(adapter, request)).type
    except RuntimeError as exc:
        return f"failed: {exc}"


def _request(
    run_id: str,
    inputs: list[Path],
    workflow: dict[str, Any],
    *,
    operation: str = "image_to_image",
    contents: tuple[bytes, ...] | None = None,
    mask: Path | None = None,
    mask_content: bytes | None = None,
    parameters: dict[str, Any] | None = None,
) -> MediaRequest:
    return MediaRequest(
        run_id=run_id,
        operation=operation,
        prompt="Keep the shapes and soften the light",
        negative_prompt=None,
        input_paths=inputs,
        workflow=workflow,
        parameters={
            **({"mask": {"path": str(mask)}} if mask is not None else {}),
            **(parameters or {}),
        },
        input_contents=contents,
        mask_content=mask_content,
    )


def _load(placeholder: str = "${input_image}") -> dict[str, Any]:
    return {"class_type": "LoadImage", "inputs": {"image": placeholder}}


def _save(images: str, prefix: Any = "LM Atelier/result") -> dict[str, Any]:
    return {"class_type": "SaveImage", "inputs": {"images": [images, 0], "filename_prefix": prefix}}


def _picture_workflow() -> dict[str, Any]:
    return {"load": _load(), "save": _save("load")}


def _masked_workflow() -> dict[str, Any]:
    return {
        "load": _load(),
        "mask": {"class_type": "LoadImageMask", "inputs": {"image": "${mask}", "channel": "alpha"}},
        "paint": {
            "class_type": "ImageCompositeMasked",
            "inputs": {"destination": ["load", 0], "source": ["load", 0], "mask": ["mask", 0]},
        },
        "save": _save("paint"),
    }


def _video_workflow() -> dict[str, Any]:
    return {
        "load": _load(),
        "clip": {"class_type": "LoadVideo", "inputs": {"file": "${input_video}"}},
        "save": _save("load"),
        "keep": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["clip", 0], "filename_prefix": "LM Atelier/clip"},
        },
    }


async def test_runs_sending_the_same_picture_give_loadimage_the_same_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same bytes bind the same input on every run, and different bytes a different one.

    The second run's bytes are already in memory, over a stored file that differs, so the
    name is seen to come from the bytes sent rather than from the file or the run.
    """
    comfy = _ComfyUI(tmp_path)
    adapter = await _adapter(comfy, monkeypatch)
    try:
        ends = [
            await _run(
                adapter,
                _request("run-first", [_write(tmp_path, "first", PICTURE)], _picture_workflow()),
            ),
            await _run(
                adapter,
                _request(
                    "run-second",
                    [_write(tmp_path, "second", OTHER_PICTURE)],
                    _picture_workflow(),
                    contents=(PICTURE,),
                ),
            ),
            await _run(
                adapter,
                _request(
                    "run-third", [_write(tmp_path, "third", OTHER_PICTURE)], _picture_workflow()
                ),
            ),
        ]
    finally:
        await adapter.close()

    assert [end.type for end in ends] == ["complete"] * 3
    assert [(upload.name, upload.content) for upload in comfy.uploads] == [
        (_name(PICTURE, ".png"), PICTURE),
        (_name(PICTURE, ".png"), PICTURE),
        (_name(OTHER_PICTURE, ".png"), OTHER_PICTURE),
    ]
    assert [graph["load"]["inputs"]["image"] for graph in comfy.graphs] == [
        _bound(PICTURE, ".png"),
        _bound(PICTURE, ".png"),
        _bound(OTHER_PICTURE, ".png"),
    ]
    # Every upload still goes to the same temporary subfolder and replaces a file
    # already stored under its name.
    assert [upload.fields for upload in comfy.uploads] == [
        {"type": "temp", "subfolder": "lm-atelier", "overwrite": "true"}
    ] * 3


def _enlarging(saver: dict[str, Any], **helpers: dict[str, Any]) -> dict[str, Any]:
    """A seedless workflow, such as an upscale: nothing in it changes from one run to the next."""
    return {
        "load": _load(),
        "scale": {
            "class_type": "ImageScaleBy",
            "inputs": {"image": ["load", 0], "upscale_method": "lanczos", "scale_by": 2.0},
        },
        **helpers,
        "save": saver,
    }


_NAMED_BY_RUN = ["lm-atelier-run-first-0.png", "lm-atelier-run-second-0.png"]


@pytest.mark.parametrize(
    "workflow,parameters,managed,names,second_run",
    [
        (
            _enlarging(_save("scale", "LM Atelier/enlarged")),
            {},
            True,
            [_name(PICTURE, ".png")] * 2,
            ["save"],
        ),
        (
            _enlarging(
                _save("scale", ["name", 0]),
                name={"class_type": "PrimitiveString", "inputs": {"value": "LM Atelier/enlarged"}},
            ),
            {},
            True,
            _NAMED_BY_RUN,
            ["load", "scale", "save"],
        ),
        (
            _enlarging(
                {
                    "class_type": "Image Saver",
                    "inputs": {
                        "images": ["scale", 0],
                        "filename": "enlarged",
                        "path": "LM Atelier",
                    },
                }
            ),
            {},
            True,
            _NAMED_BY_RUN,
            ["load", "scale", "save"],
        ),
        (
            {
                **_enlarging(
                    {
                        "class_type": "Image Saver",
                        "inputs": {
                            "images": ["scale", 0],
                            "filename": "enlarged",
                            "path": "LM Atelier",
                        },
                    }
                ),
                "notes": {
                    "class_type": "SaveText",
                    "inputs": {"text": ["save", 0], "filename_prefix": "LM Atelier/notes"},
                },
            },
            {},
            True,
            _NAMED_BY_RUN,
            ["load", "scale", "save", "notes"],
        ),
        (
            _enlarging(_save("scale", "${filename_prefix}")),
            {"filename_prefix": ""},
            True,
            _NAMED_BY_RUN,
            ["load", "scale", "save"],
        ),
        (
            _enlarging(_save("scale", "LM Atelier/enlarged")),
            {},
            False,
            _NAMED_BY_RUN,
            ["load", "scale", "save"],
        ),
    ],
    ids=[
        "literal-prefix",
        "prefix-from-another-node",
        "saver-named-by-other-inputs",
        "saver-another-node-reads",
        "prefix-a-setting-leaves-empty",
        "unmanaged-backend",
    ],
)
async def test_the_same_picture_twice_completes_however_its_workflow_names_files(
    workflow: dict[str, Any],
    parameters: dict[str, Any],
    managed: bool,
    names: list[str],
    second_run: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeat draws a new seed this workflow never reads, so only the upload can differ.

    The adapter moves a saver's files into a folder of the submission's own only
    by a literal prefix, and only for a backend whose outputs it manages. Named
    by content, a submission with any other saver would match the one before,
    and ComfyUI would answer that saver from its cache with the file the first
    run wrote, which the adapter had collected and removed. Such a submission
    keeps the run's own names, as does one on an unmanaged backend, where the
    adapter moves nothing. A literal prefix on a managed backend keeps the
    content name, and the second run loads and scales nothing again.
    """
    comfy = _ComfyUI(tmp_path)
    adapter = await _adapter(comfy, monkeypatch, managed=managed)
    picture = _write(tmp_path, "picture", PICTURE)
    try:
        outcomes = [
            await _outcome(
                adapter,
                _request(run_id, [picture], workflow, parameters={**parameters, "seed": seed}),
            )
            for run_id, seed in (("run-first", 11), ("run-second", 22))
        ]
    finally:
        await adapter.close()

    assert outcomes == ["complete", "complete"]
    assert [status for _file, status in comfy.viewed] == [200, 200]
    assert [upload.name for upload in comfy.uploads] == names
    assert comfy.ran[1] == second_run


@pytest.mark.parametrize(
    "unreadable", [True, False], ids=["catalogue-unreadable", "class-the-catalogue-lacks"]
)
async def test_uploads_keep_run_names_where_the_catalogue_cannot_vouch_for_the_outputs(
    unreadable: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which nodes are outputs is the catalogue's to say, so without its answer none is assumed.

    The workflow's one saver takes a literal prefix, so a catalogue that knows
    every class in it would have the uploads named by their content.
    """
    comfy = _ComfyUI(tmp_path)
    if unreadable:
        comfy.catalogue_status = 500
    else:
        comfy.unknown_classes = {"ImageScaleBy"}
    adapter = await _adapter(comfy, monkeypatch)
    picture = _write(tmp_path, "picture", PICTURE)
    workflow = _enlarging(_save("scale", "LM Atelier/enlarged"))
    try:
        outcomes = [
            await _outcome(adapter, _request(run_id, [picture], workflow))
            for run_id in ("run-first", "run-second")
        ]
    finally:
        await adapter.close()

    assert outcomes == ["complete", "complete"]
    assert [upload.name for upload in comfy.uploads] == _NAMED_BY_RUN
    assert comfy.catalogue_requests > 0


def _with_formatter(prefix: Any) -> dict[str, Any]:
    """The upscale, with a node that is not an output taking a prefix, read by the save."""
    workflow = _enlarging(_save("scale", "LM Atelier/enlarged"))
    if isinstance(prefix, list):
        workflow["name"] = {"class_type": "PrimitiveString", "inputs": {"value": "LM Atelier"}}
    workflow["format"] = {"class_type": "FilenameFormatter", "inputs": {"filename_prefix": prefix}}
    workflow["save"]["inputs"]["caption"] = ["format", 0]
    return workflow


@pytest.mark.parametrize(
    "workflow",
    [
        {
            **_enlarging(_save("scale", "LM Atelier/enlarged")),
            "spare": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64}},
        },
        _with_formatter(["name", 0]),
        _with_formatter(""),
    ],
    ids=["node-nothing-reads", "prefix-from-another-node", "empty-prefix"],
)
async def test_the_shape_of_a_graph_still_keeps_run_names_the_catalogue_would_allow(
    workflow: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalogue runs none of these nodes as an output; the graph's shape decides alone.

    A node nothing reads, or one whose prefix is fed from another node or left
    empty, keeps the run's own names before the catalogue is asked.
    """
    comfy = _ComfyUI(tmp_path)
    adapter = await _adapter(comfy, monkeypatch)
    try:
        outcome = await _outcome(
            adapter, _request("run-first", [_write(tmp_path, "picture", PICTURE)], workflow)
        )
    finally:
        await adapter.close()

    assert outcome == "complete"
    assert [upload.name for upload in comfy.uploads] == _NAMED_BY_RUN[:1]
    assert comfy.catalogue_requests == 0


async def test_a_run_without_uploads_does_not_ask_for_the_catalogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an upload is named, so a run that sends nothing has no reason to ask."""
    comfy = _ComfyUI(tmp_path)
    adapter = await _adapter(comfy, monkeypatch)
    workflow = {
        "blank": {
            "class_type": "EmptyImage",
            "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0},
        },
        "save": _save("blank"),
    }
    try:
        end = await _run(adapter, _request("run-text", [], workflow, operation="text_to_image"))
    finally:
        await adapter.close()

    assert end.type == "complete"
    assert comfy.uploads == []
    assert comfy.catalogue_requests == 0


async def test_a_held_catalogue_answers_later_runs_until_it_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking again would cost a run the catalogue's whole round trip, for an answer already held.

    The first run holds none and reads it once, for its selection and its picture both.
    A run long after, when the catalogue's own short cache has lapsed, reads none, and a
    run after the catalogue is dropped reads it once more.
    """
    comfy = _ComfyUI(tmp_path)
    adapter = await _adapter(comfy, monkeypatch)
    picture = _write(tmp_path, "picture", PICTURE)
    mask = _write(tmp_path, "mask", MASK)
    reads: list[int] = []
    try:
        for run_id in ("run-first", "run-later", "run-after-dropping"):
            if run_id == "run-later":
                # Long after the first run: the catalogue's own short cache has lapsed.
                adapter._object_info_cached_until = 0.0
            elif run_id == "run-after-dropping":
                adapter.invalidate_object_info_cache()
            end = await _run(adapter, _request(run_id, [picture], _masked_workflow(), mask=mask))
            assert end.type == "complete"
            reads.append(comfy.catalogue_requests)
    finally:
        await adapter.close()

    assert reads == [1, 1, 2]
    assert [upload.name for upload in comfy.uploads] == [
        _name(MASK, ".png"),
        _name(PICTURE, ".png"),
    ] * 3


@pytest.mark.parametrize("source", ["in-memory", "from-disk"])
async def test_two_pictures_in_one_run_are_uploaded_under_names_of_their_own(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each picture is named by all of its own bytes, so neither takes the other's place.

    The two pictures agree for longer than the head read to tell a picture from
    a video. Named from that head, or both from the first picture, they would go
    up under one name, and the first LoadImage would read the second picture.
    """
    comfy = _ComfyUI(tmp_path)
    adapter = await _adapter(comfy, monkeypatch)
    pictures = (LEFT_PICTURE, RIGHT_PICTURE)
    if source == "in-memory":
        paths = [_write(tmp_path, f"stored-{index}", PICTURE) for index in range(2)]
        contents: tuple[bytes, ...] | None = pictures
    else:
        paths = [
            _write(tmp_path, f"picture-{index}", picture) for index, picture in enumerate(pictures)
        ]
        contents = None
    workflow = {
        "left": _load("${input_image_0}"),
        "right": _load("${input_image_1}"),
        "pair": {
            "class_type": "ImageBatch",
            "inputs": {"image1": ["left", 0], "image2": ["right", 0]},
        },
        "save": _save("pair"),
    }
    try:
        end = await _run(adapter, _request("run-pair", paths, workflow, contents=contents))
    finally:
        await adapter.close()

    names = [_name(picture, ".png") for picture in pictures]
    assert end.type == "complete"
    assert [upload.name for upload in comfy.uploads] == names
    graph = comfy.graphs[0]
    assert [graph[node]["inputs"]["image"] for node in ("left", "right")] == [
        f"lm-atelier/{name} [temp]" for name in names
    ]
    # The file each LoadImage reads holds its own picture.
    assert [(comfy.temp_root / "lm-atelier" / name).read_bytes() for name in names] == list(
        pictures
    )


async def test_a_selection_is_uploaded_under_a_name_of_its_own_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mask read from its file and the same mask already read by the server bind one input."""
    comfy = _ComfyUI(tmp_path)
    adapter = await _adapter(comfy, monkeypatch)
    picture = _write(tmp_path, "picture", PICTURE)
    try:
        ends = [
            await _run(
                adapter,
                _request(
                    "run-stored",
                    [picture],
                    _masked_workflow(),
                    mask=_write(tmp_path, "mask", MASK),
                ),
            ),
            await _run(
                adapter,
                _request(
                    "run-read",
                    [picture],
                    _masked_workflow(),
                    mask=_write(tmp_path, "drawn", OTHER_MASK),
                    mask_content=MASK,
                ),
            ),
        ]
    finally:
        await adapter.close()

    assert [end.type for end in ends] == ["complete"] * 2
    assert [graph["mask"]["inputs"]["image"] for graph in comfy.graphs] == [
        _bound(MASK, ".png")
    ] * 2
    # The selection goes first, then the picture, on each run.
    assert [(upload.name, upload.content) for upload in comfy.uploads] == [
        (_name(MASK, ".png"), MASK),
        (_name(PICTURE, ".png"), PICTURE),
    ] * 2


class _NotedReads(io.BufferedReader):
    """A file that notes how much each read returned and whether it ran on the event loop."""

    def __init__(self, path: Path, loop_thread: int, reads: list[tuple[bool, int]]) -> None:
        super().__init__(io.FileIO(path, "r"))
        self._loop_thread = loop_thread
        self._reads = reads

    def read(self, size: int | None = -1) -> bytes:
        data = super().read(size)
        self._reads.append((threading.get_ident() == self._loop_thread, len(data)))
        return data

    def readinto(self, buffer: Any) -> int:
        count = super().readinto(buffer)
        self._reads.append((threading.get_ident() == self._loop_thread, count))
        return count


async def test_a_streamed_video_is_named_by_reading_it_in_chunks_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest that names a video comes from chunked reads in a worker thread.

    Read on the event loop, it would hold up every other request for as long as the
    file takes to read; read whole, the video would sit in memory at once.
    """
    comfy = _ComfyUI(tmp_path)
    picture = _write(tmp_path, "picture", PICTURE)
    clip = _write(tmp_path, "clip", LONG_CLIP)
    loop_thread = threading.get_ident()
    reads: list[tuple[bool, int]] = []
    opened = Path.open

    def open_noting_clip_reads(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if path == clip and mode == "rb":
            return _NotedReads(path, loop_thread, reads)
        return opened(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_noting_clip_reads)
    adapter = await _adapter(comfy, monkeypatch)
    try:
        end = await _run(
            adapter,
            _request("run-clip", [picture, clip], _video_workflow(), operation="image_to_video"),
        )
    finally:
        await adapter.close()

    assert end.type == "complete"
    assert comfy.graphs[0]["clip"]["inputs"]["file"] == _bound(LONG_CLIP, ".mp4")
    assert (_name(LONG_CLIP, ".mp4"), LONG_CLIP) in [
        (upload.name, upload.content) for upload in comfy.uploads
    ]
    # Off the event loop, the clip was read through once: for its digest.
    assert sum(size for on_loop, size in reads if not on_loop) >= len(LONG_CLIP)
    # No read took the clip whole, neither the digest's nor the upload's.
    assert max(size for _on_loop, size in reads) < len(LONG_CLIP)


async def test_a_runs_own_upload_is_left_out_of_its_results_even_under_a_shared_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clip LoadVideo shows is the run's own input, whichever other run also sent it.

    A file in the upload subfolder that this run did not send is a result like any other,
    even when an earlier run uploaded it, and so is a file outside that subfolder that
    shares a name with one of this run's uploads.
    """
    comfy = _ComfyUI(tmp_path)
    clip = _write(tmp_path, "clip", CLIP)
    # The second run's history also shows the first run's picture, which it never sent,
    # and an output stored apart from the uploads under the name of its own clip.
    comfy.shown[2] = [
        {"filename": _name(PICTURE, ".png"), "subfolder": "lm-atelier", "type": "temp"},
        {"filename": _name(CLIP, ".mp4"), "subfolder": "", "type": "output"},
    ]
    _write(comfy.output_root, _name(CLIP, ".mp4"), b"an output named like the clip")
    adapter = await _adapter(comfy, monkeypatch)
    try:
        ends = [
            await _run(
                adapter,
                _request(
                    run_id,
                    [_write(tmp_path, run_id, picture), clip],
                    _video_workflow(),
                    operation="image_to_video",
                ),
            )
            for run_id, picture in (("run-first", PICTURE), ("run-second", OTHER_PICTURE))
        ]
    finally:
        await adapter.close()

    clips = [upload.name for upload in comfy.uploads if upload.name.endswith(".mp4")]
    assert clips == [_name(CLIP, ".mp4")] * 2
    assert [[asset.name for asset in end.assets] for end in ends] == [
        ["result_00001_.png", "clip_00001_.mp4"],
        ["result_00001_.png", "clip_00001_.mp4", _name(PICTURE, ".png"), _name(CLIP, ".mp4")],
    ]
    assert adapter._uploads == {}


@pytest.mark.parametrize("failure", ["refused", "execution-error"])
async def test_a_failed_run_leaves_none_of_its_upload_names_behind(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comfy = _ComfyUI(tmp_path)
    comfy.refuse_prompts = failure == "refused"
    comfy.fail_runs = failure == "execution-error"
    adapter = await _adapter(comfy, monkeypatch)
    request = _request("run-failed", [_write(tmp_path, "picture", PICTURE)], _picture_workflow())
    try:
        with pytest.raises(RuntimeError):
            await _run(adapter, request)
        left_behind = dict(adapter._uploads)
    finally:
        await adapter.close()

    assert [upload.name for upload in comfy.uploads] == [_name(PICTURE, ".png")]
    assert left_behind == {}


async def test_a_stopped_run_leaves_none_of_its_upload_names_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comfy = _ComfyUI(tmp_path)
    comfy.hold_runs = True
    adapter = await _adapter(comfy, monkeypatch)
    request = _request("run-stopped", [_write(tmp_path, "picture", PICTURE)], _picture_workflow())
    events = adapter.generate(request)
    try:
        while (await anext(events)).type != "queued":
            pass
        while_running = {run_id: set(names) for run_id, names in adapter._uploads.items()}
        await adapter.cancel(request.run_id)
        rest = [event.type async for event in events]
        left_behind = dict(adapter._uploads)
    finally:
        await close_iterator(events)
        await adapter.close()

    assert while_running == {"run-stopped": {_name(PICTURE, ".png")}}
    assert rest == ["cancelled"]
    assert left_behind == {}
