from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import sys
import time
from collections.abc import AsyncIterator, Collection, Iterator
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TypeGuard
from urllib.parse import quote, urlencode, urlparse, urlunparse
from uuid import uuid4

import httpx
import websockets
from websockets.exceptions import WebSocketException

from ..comfy_lora_runtime_names import (
    bind_lora_runtime_names,
    has_nested_lora_names,
    lora_runtime_names_need_binding,
)
from ..domain import Operation
from ..filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntry,
    AnchoredEntryKind,
    list_entries,
    open_child_directory,
    remove_directory_entry,
    remove_entry,
)
from ..media_references import unread_videos
from ..network import shared_tls_context
from ..output_origin import stated_origin
from ..schemas import EngineCapabilities
from ..settings_registry import IMAGE_SETTINGS, VIDEO_SETTINGS
from .base import GeneratedAsset, MediaEvent, MediaRequest, MediaWorkflowFailure
from .contracts import MAX_ADAPTER_EVENT_BYTES, MAX_ADAPTER_PREVIEW_BYTES

logger = logging.getLogger(__name__)
_CANCELLED = object()
_MAX_COMFY_JSON_BYTES = 32 * 1024 * 1024
_MAX_COMFY_OUTPUTS = 64
#: How long a finished prompt's history may take to appear, and how often to look.
_HISTORY_SETTLE_SECONDS = 30.0
_HISTORY_POLL_SECONDS = 0.25

#: The one subfolder every conditioning upload is addressed to. The backend's
#: temp directory also holds files this adapter did not write, so this is the
#: only part of it that is ours to reclaim.
_UPLOAD_SUBFOLDER = "lm-atelier"

_ERROR_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,79}\Z")
_ERRNO = re.compile(r"\[Errno (?P<number>\d{1,5})\]")
_NUMBERED_INPUT_IMAGE = re.compile(r"\$\{input_image_(?P<index>\d{1,2})\}\Z")
#: How much of an input is read to tell a picture from a video.
_INPUT_HEAD_BYTES = 4096
#: How much of a video is read at a time to name it by its content.
_DIGEST_CHUNK_BYTES = 1024 * 1024
#: MP4-family major brands of video files. The same box also holds HEIF and
#: AVIF pictures, sound and camera raw files, so any other brand is neither.
_VIDEO_BRANDS = frozenset(
    {
        b"isom",
        b"iso2",
        b"iso4",
        b"iso5",
        b"iso6",
        b"mp41",
        b"mp42",
        b"avc1",
        b"M4V ",
        b"M4VH",
        b"M4VP",
        b"mmp4",
        b"3gp4",
        b"3gp5",
        b"3gp6",
        b"3g2a",
        b"dash",
        b"f4v ",
        b"XAVC",
        b"MSNV",
    }
)


def _is_own_upload(item: dict[str, Any], uploads: Collection[str]) -> bool:
    """Whether a history entry names a file this adapter uploaded for this run.

    A content-named upload has the name every other content-named upload of
    the same bytes has; the file is still this run's own when this run sent it.
    A file in the upload subfolder that this run did not send is not.
    """
    subfolder = str(item.get("subfolder") or "").replace("\\", "/").strip("/")
    filename = str(item.get("filename") or "")
    return subfolder == _UPLOAD_SUBFOLDER and filename in uploads


def _upload_name(digest: str, extension: str) -> str:
    """The name an input is uploaded under when it is named by its content.

    Every content-named upload of the same bytes has this name. ComfyUI reuses
    a node's cached outputs only while the node's inputs are unchanged, so a
    picture uploaded under a new name for each run is loaded, and encoded,
    again by every regeneration, edit and batch that starts from it.
    """
    return f"lm-atelier-{digest}{extension}"


def _prefixes_are_literal(graph: dict[str, Any]) -> bool:
    """Whether each node that takes a prefix, or that nothing reads, takes a non-empty literal one.

    `ComfyUIAdapter._submission_owned_outputs` moves a node's files into a
    folder of the submission's own only when its `filename_prefix` is a
    non-empty literal string; a prefix fed from another node, or an empty one,
    stays where it is. A node that nothing reads runs only as an output, so it
    must take such a prefix too. The shape of a graph does not say what such a
    node writes or where: a saver named by other inputs writes files this
    adapter removes once it has collected them, while a preview's files are
    temporary and stay, and the two cannot be told apart here, so a preview is
    held to the rule as well. An empty graph, or a node that is not a set of
    named inputs, fails it.
    """
    inputs_by_node: dict[str, dict[str, Any]] = {}
    for node_id, node in graph.items():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            return False
        inputs_by_node[str(node_id)] = inputs
    read = {
        str(value[0])
        for inputs in inputs_by_node.values()
        for value in inputs.values()
        if isinstance(value, list)
        and len(value) == 2
        and str(value[0]) in inputs_by_node
        and type(value[1]) is int
    }
    for node_id, inputs in inputs_by_node.items():
        if "filename_prefix" in inputs:
            prefix = inputs["filename_prefix"]
            if not isinstance(prefix, str) or not prefix:
                return False
        elif node_id not in read:
            return False
    return bool(inputs_by_node)


def _output_nodes_take_literal_prefixes(graph: dict[str, Any], catalogue: dict[str, Any]) -> bool:
    """Whether the catalogue knows each node's class, and each output node takes a literal prefix.

    The catalogue says which classes ComfyUI runs as outputs, including one
    whose result another node reads, such as a saver named by other inputs that
    also passes a value on; the shape of a graph cannot show that. It does not
    say where an output writes, so a preview is held to the rule like a saver.
    A prefix is literal here as `_prefixes_are_literal` means it, and a class
    the catalogue does not know fails the rule.
    """
    for node in graph.values():
        if not isinstance(node, dict):
            return False
        class_type = node.get("class_type")
        info = catalogue.get(class_type) if isinstance(class_type, str) else None
        if not isinstance(info, dict):
            return False
        if info.get("output_node") is True:
            inputs = node.get("inputs")
            prefix = inputs.get("filename_prefix") if isinstance(inputs, dict) else None
            if not isinstance(prefix, str) or not prefix:
                return False
    return True


def _read_head(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(_INPUT_HEAD_BYTES)


def _file_digest(path: Path) -> str:
    """The SHA-256 of a file, read a chunk at a time rather than whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DIGEST_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _numbered_input_image_indices(workflow: dict[str, Any]) -> list[int]:
    indices: set[int] = set()
    stack: list[Any] = [workflow]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str):
            match = _NUMBERED_INPUT_IMAGE.fullmatch(value)
            if match and (index := int(match.group("index"))) < 64:
                indices.add(index)
    return sorted(indices)


def _queue_got_shorter(data: dict[str, Any], before: int | None) -> tuple[bool, int | None]:
    """Whether a status broadcast says ComfyUI's queue got shorter, and its length now.

    The first length read is only where counting starts: ComfyUI sends it as
    the socket connects, before the prompt is submitted. A broadcast without a
    readable length changes nothing.
    """

    status = data.get("status")
    info = status.get("exec_info") if isinstance(status, dict) else None
    length = info.get("queue_remaining") if isinstance(info, dict) else None
    if isinstance(length, bool) or not isinstance(length, int):
        return False, before
    return before is not None and length < before, length


def _names_no_such_job(response: httpx.Response) -> bool:
    """Whether a not-found answer is ComfyUI's job endpoint saying it holds no such job.

    A ComfyUI without that endpoint answers with its plain not-found page instead.
    """

    if "json" not in response.headers.get("content-type", ""):
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and "error" in body


def _usable_prompt_id(value: object) -> TypeGuard[str]:
    """A prompt identifier this adapter is willing to carry.

    The same test on both paths that produce one - the submission's own reply
    and the queue consulted when that reply is lost - because an identifier the
    accept path would have refused is not one to start trusting because it
    arrived by the other route.
    """

    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 200
        and all(character >= " " for character in value)
    )


class _WorkflowFailed(MediaWorkflowFailure, RuntimeError):
    """ComfyUI refused the submitted workflow, or ran it to a node's error or an unusable result."""


class _InputRefused(MediaWorkflowFailure, ValueError):
    """The request's own inputs were refused before its workflow was sent."""


#: Words that, in the exception type or message of an execution error, name a
#: fault of memory or of the device rather than one of the workflow. ComfyUI
#: unloads every model once it runs out of memory, and a device fault can leave
#: the device unusable until the process ends, so neither leaves ComfyUI as the
#: run found it. The allocator ComfyUI manages models with by default reports
#: running out of room in its own words, naming itself, its VRAM, its VBAR or
#: its host buffers, or saying that a fault failed.
_DEVICE_FAULT_WORDS = (
    "memory",
    "cuda",
    "cublas",
    "cudnn",
    "accelerator",
    "device-side",
    "paging file",
    "vram",
    "vbar",
    "hostbuf",
    "aimdo",
    "fault failed",
)

#: The only plain RuntimeErrors taken as the workflow's own: the words torch and
#: ComfyUI use when the parts a graph joins do not fit together. Torch reports
#: many faults of the device, and ComfyUI's allocator most shortages of room, as
#: a plain RuntimeError too, so any other one may be the engine's.
_WORKFLOW_RUNTIME_ERRORS = (
    # "mat1 and mat2 shapes cannot be multiplied (77x768 and 1024x320)"
    "shapes cannot be multiplied",
    # "The size of tensor a (64) must match the size of tensor b (32) at
    # non-singleton dimension 3"
    "must match the size of tensor",
    # "shape '[1, 4, 64, 64]' is invalid for input of size 32768"
    "is invalid for input of size",
    # "Sizes of tensors must match except in dimension 1. ..."
    "sizes of tensors must match",
    # "Given groups=1, weight of size [320, 4, 3, 3], expected input[1, 16, 64, 64]
    # to have 4 channels, but got 16 channels instead"
    "channels, but got",
    # A checkpoint that does not fit the model it is loaded into: "Error(s) in
    # loading state_dict for UNetModel: ...", "size mismatch for ...: copying a
    # param with shape ... from checkpoint, ..."
    "error(s) in loading state_dict",
    "size mismatch",
    # ComfyUI's own: "ERROR: Could not detect model type of: ...", and a model
    # file that "does not contain a valid" VAE, text encoder, controlnet or
    # vision model.
    "could not detect model type",
    "does not contain a valid",
)


def _execution_error_message(data: dict[str, Any]) -> str:
    node_type = str(data.get("node_type") or "")
    exception_type = str(data.get("exception_type") or "")
    if not _ERROR_LABEL.fullmatch(node_type) or not _ERROR_LABEL.fullmatch(exception_type):
        return "ComfyUI could not execute the selected workflow"
    detail = exception_type
    errno = _ERRNO.search(str(data.get("exception_message") or ""))
    if errno:
        detail = f"{detail}, errno {errno.group('number')}"
    return f"ComfyUI {node_type} failed ({detail})"


def _workflow_failed(data: dict[str, Any]) -> bool:
    """Whether an execution error is the workflow's own, leaving ComfyUI usable.

    Only an error ComfyUI describes in full can be: one whose labels are
    unusable, or that comes without its message, could be anything. One whose
    type or message names a fault of memory or of the device is not, nor is a
    plain RuntimeError in any words but `_WORKFLOW_RUNTIME_ERRORS`. Nor is a
    blocked node: ComfyUI reports it as an execution error and goes on running
    the prompt's other branches, so the prompt may still be running. The
    message is read only to decide this, and is never kept, logged or repeated:
    it can quote what a node was given.
    """
    node_type = str(data.get("node_type") or "")
    exception_type = str(data.get("exception_type") or "")
    message = data.get("exception_message")
    if (
        not _ERROR_LABEL.fullmatch(node_type)
        or not _ERROR_LABEL.fullmatch(exception_type)
        or not isinstance(message, str)
        or exception_type == "ExecutionBlocked"
    ):
        return False
    words = message.casefold()
    if any(word in f"{exception_type.casefold()}\n{words}" for word in _DEVICE_FAULT_WORDS):
        return False
    if exception_type == "RuntimeError":
        return any(phrase in words for phrase in _WORKFLOW_RUNTIME_ERRORS)
    return True


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


#: Total deadline for asking the backend to drop an abandoned prompt - not a
#: per-request timeout, which httpx measures per phase and a trickling response
#: can satisfy indefinitely. `close_iterator` gives the whole close five
#: seconds and the output cleanup runs after this, so the ask must not be able
#: to spend the entire allowance on its own.
ABANDONED_INTERRUPT_SECONDS = 2.0
#: How long a stop the run itself asks for may take in all: long enough that a
#: backend that answers slowly still gets every step of the older
#: delete-then-interrupt protocol.
STOP_SECONDS = 10.0
RECONCILE_SECONDS = 5.0
#: Emptying ComfyUI's memory: how long it is waited for in all, how often its
#: count of free device memory is read meanwhile, how long each request may
#: take, how far the count must rise to show a release, how close two readings
#: must be to show it has settled, after how long a request that changed
#: nothing is sent again, and after how long a count that has not risen at
#: all is given up on. Measured on a 16 GB card, about 10 GB came back within
#: half a second.
RELEASE_SECONDS = 10.0
RELEASE_POLL_SECONDS = 0.5
RELEASE_REQUEST_SECONDS = 3.0
RELEASE_RISE_BYTES = 256 * 1024 * 1024
RELEASE_SETTLED_BYTES = 64 * 1024 * 1024
RELEASE_ASK_AGAIN_SECONDS = 2.0
RELEASE_GIVE_UP_SECONDS = 4.0


class ComfyUIAdapter:
    def __init__(
        self,
        base_url: str,
        *,
        inactivity_seconds: float = 600,
        managed_output_root: Path | None = None,
        managed_temp_root: Path | None = None,
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
        # Cleanup must inspect the supplied ancestry, before resolve can hide a
        # linked root. Other output operations retain their existing path contract.
        self._managed_output_sweep_root = (
            managed_output_root.expanduser().absolute() if managed_output_root else None
        )
        self.managed_temp_root = (
            managed_temp_root.expanduser().absolute() if managed_temp_root else None
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            trust_env=False,
            verify=shared_tls_context(trust_environment=False),
        )
        self._jobs: dict[str, str] = {}
        self._cancelled: set[str] = set()
        self._cancel_events: dict[str, asyncio.Event] = {}
        # The names each generating run's inputs and masks were uploaded under.
        # A content-named upload does not say which run sent it: every other
        # content-named upload of the same bytes has the same name.
        self._uploads: dict[str, set[str]] = {}
        # Every prompt this adapter submits carries a client id that starts
        # with this prefix, and ComfyUI returns the client id on each queue
        # row, so a prompt LM Atelier left behind is told from other work by
        # its row alone, even one whose prompt id never arrived. The
        # application keeps one adapter, and the worker it manages never
        # outlives the process, so a prefix fixed for the adapter's life covers
        # every prompt of its own that the worker can hold.
        self._client_prefix = uuid4().hex[:16]
        # These counts say whether a prompt of this adapter's may still be in
        # ComfyUI. `_in_flight` counts generations that have sent their prompt
        # and not yet finished, and `_unsettled` those that finished without
        # collecting their outputs: ComfyUI records a prompt's outputs only once
        # the prompt has left its queue, so collected outputs are the one sign
        # it is gone. `_settled_through` is how many of the unsettled ones are
        # known to have left, because ComfyUI was later seen holding none of
        # these prompts.
        self._in_flight = 0
        #: Prompts handed to ComfyUI since this adapter started or last saw it
        #: emptied. With none, nothing this adapter sent is loaded there, and
        #: emptying it has no release to wait for.
        self._prompts_since_emptied = 0
        self._unsettled = 0
        self._settled_through = 0
        self._last_output_sweep = 0.0
        self._object_info_by_hash: dict[str, dict[str, Any]] = {}
        self._latest_object_info_hash: str | None = None
        self._object_info_cached_until = 0.0

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
        if self._latest_object_info_hash and time.monotonic() < self._object_info_cached_until:
            return self._object_info_by_hash[self._latest_object_info_hash]
        response = await self._client.get("/object_info", timeout=10)
        response.raise_for_status()
        if len(response.content) > _MAX_COMFY_JSON_BYTES:
            raise ValueError("ComfyUI object metadata is too large")
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("ComfyUI object metadata must be an object")
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        version_hash = hashlib.sha256(encoded).hexdigest()
        if version_hash not in self._object_info_by_hash:
            if len(self._object_info_by_hash) >= 4:
                self._object_info_by_hash.pop(next(iter(self._object_info_by_hash)))
            self._object_info_by_hash[version_hash] = value
        self._latest_object_info_hash = version_hash
        self._object_info_cached_until = time.monotonic() + 5
        return self._object_info_by_hash[version_hash]

    def invalidate_object_info_cache(self) -> None:
        self._latest_object_info_hash = None
        self._object_info_cached_until = 0

    async def _held_catalogue(self) -> dict[str, Any]:
        """The node catalogue this adapter last read, whatever its age, or a first read of one.

        Naming uploads trusts a held catalogue only to say which classes ComfyUI
        runs as outputs, and a class's output flag changes only with the node
        pack that defines it. A class added since the catalogue was read, by a
        pack activated or a runtime restarted with other nodes, is one it does
        not know, which keeps the run's own names. `invalidate_object_info_cache`
        drops the catalogue, and the next upload reads a fresh one.

        Reading it again would cost every run the catalogue's whole round trip,
        which can take ComfyUI a second or more on a large installation, and
        parsing and hashing it on the event loop, for an answer already held.
        """
        held = self._object_info_by_hash.get(self._latest_object_info_hash or "")
        if held is not None:
            return held
        return await self.object_info()

    @staticmethod
    def _compile(value: Any, parameters: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {key: ComfyUIAdapter._compile(item, parameters) for key, item in value.items()}
        if isinstance(value, list):
            return [ComfyUIAdapter._compile(item, parameters) for item in value]
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return parameters.get(value[2:-1], value)
        return value

    @staticmethod
    def _submission_owned_outputs(graph: dict[str, Any], folder: str) -> dict[str, Any]:
        """Save this submission's files under a folder of its own.

        ComfyUI keeps each node's result between prompts and answers a node
        whose inputs have not changed from that cache, without running it. A
        save node answered that way names the files an earlier submission
        wrote, and this adapter removes managed files as soon as it has
        collected them, so an identical request could not fetch its own
        result. A per-submission folder changes only the save nodes' inputs:
        they run again on the cached pictures and write files of their own,
        nothing upstream is computed twice, and the file names stay as the
        workflow chose them.
        """
        for node in graph.values():
            inputs = node.get("inputs") if isinstance(node, dict) else None
            prefix = inputs.get("filename_prefix") if isinstance(inputs, dict) else None
            if isinstance(inputs, dict) and isinstance(prefix, str) and prefix:
                inputs["filename_prefix"] = f"{folder}/{prefix}"
        return graph

    @staticmethod
    def _leaves_unfilled(value: Any, parameters: dict[str, Any]) -> bool:
        """Whether compiling this graph would keep a placeholder as written.

        `_compile` keeps one it has no value for, and ComfyUI would then be
        sent the placeholder itself, such as an image to load named "${mask}".
        Asked of the graph before compiling, by the same test, so a prompt that
        happens to look like a placeholder is never mistaken for one.
        """
        if isinstance(value, dict):
            return any(ComfyUIAdapter._leaves_unfilled(item, parameters) for item in value.values())
        if isinstance(value, list):
            return any(ComfyUIAdapter._leaves_unfilled(item, parameters) for item in value)
        return (
            isinstance(value, str)
            and value.startswith("${")
            and value.endswith("}")
            and value[2:-1] not in parameters
        )

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
        raise _InputRefused("ComfyUI conditioning inputs must be supported raster images")

    @classmethod
    def _input_format(cls, head: bytes) -> tuple[str, str, str]:
        """Whether an input is a picture or a video, with the extension and type to send it as.

        A video is told by its container's leading bytes, which ComfyUI's
        LoadVideo reads through its own decoder. A picture container that
        ComfyUI's LoadImage cannot read, such as AVIF or HEIC, is neither.
        """
        try:
            extension, media_type = cls._image_format(head)
        except ValueError:
            pass
        else:
            return "image", extension, media_type
        if len(head) >= 12 and head[4:8] == b"ftyp":
            if head[8:12] == b"qt  ":
                return "video", ".mov", "video/quicktime"
            if head[8:12] in _VIDEO_BRANDS:
                return "video", ".mp4", "video/mp4"
        if head.startswith(b"\x1aE\xdf\xa3"):
            if b"webm" in head:
                return "video", ".webm", "video/webm"
            return "video", ".mkv", "video/x-matroska"
        if head.startswith(b"RIFF") and head[8:12] == b"AVI ":
            return "video", ".avi", "video/x-msvideo"
        raise _InputRefused("ComfyUI conditioning inputs must be supported pictures or videos")

    async def _upload_mask(self, request: MediaRequest) -> str | None:
        """Upload the studio selection, if this turn carries one.

        A mask rides in the parameters rather than the input paths because it
        is instruction, not content: it must never be treated as another
        conditioning image or appear in the transcript.
        """
        setting = request.parameters.get("mask")
        if not isinstance(setting, dict):
            return None
        raw_path = setting.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise _InputRefused("The selection for this edit has no resolved file")
        contents = (request.mask_content,) if request.mask_content is not None else None
        return (
            await self._upload_files(request, [Path(raw_path)], label="mask", contents=contents)
        )[0]

    async def _upload_inputs(self, request: MediaRequest) -> list[tuple[str, str]]:
        """Each input's kind, picture or video, and the name it was uploaded under, in order.

        Every input is identified before any is sent, so a turn that must start
        from a picture and names none is refused without uploading anything.
        """
        paths, contents = request.input_paths, request.input_contents
        self._check_contents(paths, contents)
        heads = [
            contents[index][:_INPUT_HEAD_BYTES]
            if contents is not None
            else await asyncio.to_thread(_read_head, path)
            for index, path in enumerate(paths)
        ]
        formats = [self._input_format(head) for head in heads]
        if request.operation in {
            Operation.IMAGE_TO_IMAGE.value,
            Operation.IMAGE_TO_VIDEO.value,
        } and not any(kind == "image" for kind, _, _ in formats):
            raise _InputRefused(f"{request.operation} requires a conditioning image")
        videos = sum(kind == "video" for kind, _, _ in formats)
        if unread := unread_videos(request.workflow, videos):
            # No slot repeats a video, so one the graph does not name would be
            # attached and silently left out.
            raise _InputRefused(
                f"This workflow does not read attached video {unread[0] + 1} of {videos}"
            )
        by_content = bool(paths) and await self._names_uploads_by_content(request)
        uploaded: list[tuple[str, str]] = []
        for index, (path, (kind, extension, media_type)) in enumerate(
            zip(paths, formats, strict=True)
        ):
            filename = f"lm-atelier-{request.run_id}-{index}{extension}"
            if contents is not None:
                content = contents[index]
                if by_content:
                    filename = _upload_name(hashlib.sha256(content).hexdigest(), extension)
                reference = await self._upload_one(request, filename, content, media_type)
            elif kind == "video":
                # A video can be far larger than a picture, so it is streamed
                # from disk rather than read whole into memory, and the digest
                # that names it is read the same way, off the event loop.
                if by_content:
                    filename = _upload_name(await asyncio.to_thread(_file_digest, path), extension)
                handle = await asyncio.to_thread(path.open, "rb")
                try:
                    reference = await self._upload_one(request, filename, handle, media_type)
                finally:
                    handle.close()
            else:
                content = await asyncio.to_thread(path.read_bytes)
                if by_content:
                    filename = _upload_name(hashlib.sha256(content).hexdigest(), extension)
                reference = await self._upload_one(request, filename, content, media_type)
            uploaded.append((kind, reference))
        return uploaded

    @staticmethod
    def _check_contents(paths: list[Path], contents: tuple[bytes, ...] | None) -> None:
        if contents is not None and (
            type(contents) is not tuple
            or len(contents) != len(paths)
            or any(type(content) is not bytes for content in contents)
        ):
            raise _InputRefused("The input bytes do not match the conditioning images")

    async def _upload_files(
        self,
        request: MediaRequest,
        paths: list[Path],
        *,
        label: str,
        contents: tuple[bytes, ...] | None = None,
    ) -> list[str]:
        """Upload pictures, such as a selection's mask, that may only ever be pictures."""
        self._check_contents(paths, contents)
        by_content = await self._names_uploads_by_content(request)
        uploaded: list[str] = []
        for index, path in enumerate(paths):
            content = (
                contents[index]
                if contents is not None
                else await asyncio.to_thread(path.read_bytes)
            )
            extension, media_type = self._image_format(content)
            suffix = f"{label}-{index}" if label else str(index)
            filename = f"lm-atelier-{request.run_id}-{suffix}{extension}"
            if by_content:
                filename = _upload_name(hashlib.sha256(content).hexdigest(), extension)
            uploaded.append(await self._upload_one(request, filename, content, media_type))
        return uploaded

    async def _names_uploads_by_content(self, request: MediaRequest) -> bool:
        """Whether this request's uploads are named by their content rather than by the run.

        A per-run name changes the loader's input on every run, so the loader
        and every node downstream of it run again. Named by content, a repeated
        source leaves the loader unchanged, and with it every node that reads
        nothing else, such as an encoder fed only from it, which ComfyUI then
        answers from its cache. A repeat usually draws a new seed, so the nodes
        that read one run again either way.

        A node that writes files must run again as well, or ComfyUI answers it
        from its cache with the files an earlier submission wrote, which this
        adapter may already have removed. `generate` moves a node's files into a
        folder of the submission's own only for a backend whose outputs this
        adapter manages, and only by a literal prefix, so content names are used
        only for a submission whose every such node is moved. Which nodes are outputs
        is the catalogue's to say, as `_held_catalogue` keeps it; a catalogue
        that cannot be read, or that does not know a class in the graph, keeps
        the run's own names.

        Judged on the graph `generate` compiles, before the names this decides
        are bound into it: no upload's name changes which nodes write files or
        whether their prefixes are moved.
        """
        if self.managed_output_root is None:
            return False
        graph = self._compile(request.workflow, self._base_parameters(request))
        if not isinstance(graph, dict) or not _prefixes_are_literal(graph):
            return False
        try:
            catalogue = await self._held_catalogue()
        except (httpx.HTTPError, ValueError, RecursionError):
            return False
        return _output_nodes_take_literal_prefixes(graph, catalogue)

    async def _upload_one(
        self,
        request: MediaRequest,
        filename: str,
        content: bytes | BinaryIO,
        media_type: str,
    ) -> str:
        """Send one file to ComfyUI's temporary uploads and return the name a graph loads it by.

        The name the file was stored under is kept as one of the generating
        run's own uploads, so the run's history can tell the file from a result.
        """
        response = await self._client.post(
            "/upload/image",
            data={
                "type": "temp",
                "subfolder": _UPLOAD_SUBFOLDER,
                "overwrite": "true",
            },
            files={"image": (filename, content, media_type)},
            timeout=120 if isinstance(content, bytes) else 600,
        )
        if response.status_code == 413:
            raise _InputRefused("ComfyUI refused an input larger than its upload limit")
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
        response_subfolder = "/".join(subfolder_path.parts)
        reference = f"{response_subfolder}/{uploaded_name}" if response_subfolder else uploaded_name
        if upload_type == "temp":
            reference = f"{reference} [temp]"
        if (own := self._uploads.get(request.run_id)) is not None:
            own.add(uploaded_name)
        return reference

    @staticmethod
    def _base_parameters(request: MediaRequest) -> dict[str, Any]:
        """The values a workflow is compiled with, other than the names of uploaded inputs.

        For a selection edit, `mask` here is still the selection as the turn
        gave it, not a value any workflow is sent: `_request_parameters`
        replaces it with the name the mask is uploaded under.
        """
        return {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            **request.parameters,
        }

    async def _request_parameters(self, request: MediaRequest) -> dict[str, Any]:
        if (
            request.operation
            in {
                Operation.IMAGE_TO_IMAGE.value,
                Operation.IMAGE_TO_VIDEO.value,
            }
            and not request.input_paths
        ):
            raise _InputRefused(f"{request.operation} requires a conditioning image")
        parameters = self._base_parameters(request)
        mask_reference = await self._upload_mask(request)
        if mask_reference:
            # The workflow's declared mask input receives the uploaded name;
            # the resolved local path never leaves this adapter.
            parameters["mask"] = mask_reference
        uploaded = await self._upload_inputs(request)
        # Pictures and videos are bound under names of their own, so a graph's
        # LoadImage is never handed a video, nor its LoadVideo a picture.
        images = [reference for kind, reference in uploaded if kind == "image"]
        videos = [reference for kind, reference in uploaded if kind == "video"]
        if images:
            parameters["input_image"] = images[0]
            parameters["input_images"] = images
            parameters.update(
                {f"input_image_{index}": reference for index, reference in enumerate(images)}
            )
            for index in _numbered_input_image_indices(request.workflow):
                parameters.setdefault(
                    f"input_image_{index}",
                    images[min(index, len(images) - 1)],
                )
        if videos:
            parameters["input_video"] = videos[0]
            parameters["input_videos"] = videos
            parameters.update(
                {f"input_video_{index}": reference for index, reference in enumerate(videos)}
            )
        return parameters

    async def _runtime_lora_graph(
        self, graph: dict[str, Any], cancelled: asyncio.Event
    ) -> dict[str, Any]:
        if not has_nested_lora_names(graph):
            return graph
        try:
            info = await self.object_info()
        except (httpx.HTTPError, ValueError):
            return graph
        if cancelled.is_set() or not lora_runtime_names_need_binding(graph, info):
            return graph
        try:
            response = await self._client.get("/system_stats", timeout=10)
            response.raise_for_status()
            if len(response.content) > _MAX_COMFY_JSON_BYTES:
                return graph
            metadata = response.json()
        except (httpx.HTTPError, ValueError):
            return graph
        system = metadata.get("system") if isinstance(metadata, dict) else None
        platform = system.get("os") if isinstance(system, dict) else None
        return bind_lora_runtime_names(
            graph, info, runtime_platform=platform if isinstance(platform, str) else None
        )

    async def generate(self, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        if request.run_id in self._cancelled:
            self._cancelled.discard(request.run_id)
            yield MediaEvent(type="cancelled")
            return
        cancel_event = asyncio.Event()
        uploads: set[str] = set()
        self._cancel_events[request.run_id] = cancel_event
        self._uploads[request.run_id] = uploads
        self._jobs.pop(request.run_id, None)
        prompt_id: str | None = None
        outputs_collected = False
        submitted = False
        abandoned = False
        # Someone removed the prompt from ComfyUI before it ran: there is
        # nothing to stop, and a stop could reach someone else's work.
        removed = False
        refused_after_acceptance = False
        bound_after_submission_failure = False
        try:
            yield MediaEvent(
                type="progress",
                phase="Preparing media workspace",
                data={"indeterminate": True},
            )
            if cancel_event.is_set():
                yield MediaEvent(type="cancelled")
                return
            await self._sweep_stale_outputs()
            client_id = f"{self._client_prefix}{uuid4().hex}"
            if request.input_paths:
                yield MediaEvent(
                    type="progress",
                    phase="Staging media inputs",
                    data={"indeterminate": True},
                )
                if cancel_event.is_set():
                    yield MediaEvent(type="cancelled")
                    return
            parameters = await self._request_parameters(request)
            if cancel_event.is_set():
                yield MediaEvent(type="cancelled")
                return
            if self._leaves_unfilled(request.workflow, parameters):
                raise _InputRefused(
                    "This workflow has an input this request leaves empty, so it was not sent "
                    "to ComfyUI. Choose a workflow made for this kind of request."
                )
            graph = self._compile(request.workflow, parameters)
            graph = await self._runtime_lora_graph(graph, cancel_event)
            if self.managed_output_root is not None:
                graph = self._submission_owned_outputs(graph, client_id)
            yield MediaEvent(
                type="progress",
                phase="Submitting media workflow",
                data={"indeterminate": True},
            )
            if cancel_event.is_set():
                yield MediaEvent(type="cancelled")
                return
            async with websockets.connect(
                self._websocket_url(client_id),
                max_size=MAX_ADAPTER_PREVIEW_BYTES,
                max_queue=16,
                open_timeout=10,
                close_timeout=10,
                proxy=None,
            ) as socket:
                prompt_payload: dict[str, Any] = {
                    "prompt": graph,
                    "client_id": client_id,
                }
                # From this request on, ComfyUI may hold the prompt whatever the
                # reply turns out to be, until its outputs are collected.
                submitted = True
                self._in_flight += 1
                self._prompts_since_emptied += 1
                try:
                    response = await self._client.post(
                        "/prompt",
                        json=prompt_payload,
                        timeout=30,
                    )
                except (httpx.TimeoutException, asyncio.CancelledError):
                    # The request went out and no answer came back. Nothing
                    # here knows whether the backend took it, and the one
                    # thing that would say so is the identifier that never
                    # arrived. Ask the backend what it is holding under this
                    # submission's own client id before giving up, so a
                    # generation it did take can still be stopped rather than
                    # running on unreachable behind the caller's failure.
                    bound = await self._prompt_id_for_client(client_id)
                    if bound is not None:
                        prompt_id = bound
                        if self._cancel_events.get(request.run_id) is cancel_event:
                            self._jobs[request.run_id] = prompt_id
                        bound_after_submission_failure = True
                    raise
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as refused:
                    if refused.response.status_code != 400:
                        raise
                    # ComfyUI answers a graph that fails its validation this way,
                    # with the nodes at fault, and queues nothing, so it loads
                    # nothing either: the refusal is the workflow's own. Any
                    # other status is answered below.
                    self._prompts_since_emptied -= 1
                    raise _WorkflowFailed(
                        "ComfyUI rejected a generation request (HTTP 400)"
                    ) from None
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RuntimeError("ComfyUI returned an invalid prompt response")
                raw_prompt_id = payload.get("prompt_id")
                if _usable_prompt_id(raw_prompt_id):
                    # Recorded BEFORE either refusal below. The post already
                    # raised for status, so a body at all means the backend
                    # accepted this prompt and queued it, whatever the body goes
                    # on to say about the graph - and the repository's own
                    # fixtures model a success response as carrying prompt_id
                    # and node_errors together. This identifier is the only
                    # handle that can stop it or clean up after it.
                    prompt_id = raw_prompt_id
                    if self._cancel_events.get(request.run_id) is cancel_event:
                        self._jobs[request.run_id] = prompt_id
                if payload.get("node_errors"):
                    # Refusing a prompt the backend has ALREADY taken. The
                    # caller is told the generation failed, so the teardown has
                    # to stop it rather than only forget it. That stop is only
                    # asked for, so ComfyUI may still be running the part of
                    # the graph it took, and this failure is not marked as the
                    # workflow's own: it is handed off as any other failure is.
                    refused_after_acceptance = prompt_id is not None
                    raise RuntimeError("ComfyUI rejected the selected workflow")
                if prompt_id is None:
                    raise RuntimeError("ComfyUI returned an invalid prompt identifier")
                if cancel_event.is_set():
                    await self._abandon_prompt(prompt_id, seconds=STOP_SECONDS)
                    yield MediaEvent(type="cancelled")
                    return
                yield MediaEvent(
                    type="queued",
                    progress=0,
                    phase="Queued in media runtime",
                    data={"prompt_id": prompt_id, "indeterminate": True},
                    # The prompt id in this event CAME from the backend's
                    # response: this is the first moment the engine has spoken.
                    engine_sourced=True,
                )
                sampler_progress = 0.0
                execution_started = False
                queue_length: int | None = None
                messages = socket.__aiter__()
                while True:
                    try:
                        raw = await self._next_message(messages, cancel_event)
                    except StopAsyncIteration:
                        break
                    except TimeoutError as exc:
                        if not execution_started and not await self._prompt_still_held(prompt_id):
                            # A removal whose broadcast went unchecked: someone's
                            # choice, so it is neither waited out nor sent again.
                            removed = True
                            yield MediaEvent(type="cancelled")
                            return
                        await self._abandon_prompt(prompt_id, seconds=STOP_SECONDS)
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
                            yield MediaEvent(
                                type="preview",
                                progress=sampler_progress,
                                phase="sampling",
                                preview=preview,
                                engine_sourced=True,
                            )
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
                    if message_type == "status" and not execution_started:
                        # Removing a prompt from ComfyUI's own queue before it
                        # runs tells its client nothing: ComfyUI broadcasts only
                        # the queue's new length. Without asking, the run would
                        # read queued until the inactivity timeout and then be
                        # sent again. A shorter queue is the cue to ask whether
                        # ComfyUI still holds the prompt.
                        shorter, queue_length = _queue_got_shorter(data, queue_length)
                        if shorter and not await self._prompt_still_held(prompt_id):
                            removed = True
                            yield MediaEvent(type="cancelled")
                            return
                        continue
                    if message_type in {"execution_start", "executing"} and not execution_started:
                        execution_started = True
                        yield MediaEvent(
                            type="progress",
                            phase="Loading media model",
                            data={"prompt_id": prompt_id, "indeterminate": True},
                            engine_sourced=True,
                        )
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
                        sampler_progress = min(max(value / maximum, 0), 0.99)
                        yield MediaEvent(
                            type="progress",
                            progress=sampler_progress,
                            phase="sampling",
                            data={
                                "prompt_id": prompt_id,
                                "value": value,
                                "max": maximum,
                            },
                            engine_sourced=True,
                        )
                    elif (
                        message_type == "executing" and data.get("node") is None
                    ) or message_type == "execution_success":
                        break
                    elif message_type == "execution_interrupted":
                        yield MediaEvent(type="cancelled")
                        return
                    elif message_type == "execution_error":
                        failure = _WorkflowFailed if _workflow_failed(data) else RuntimeError
                        raise failure(_execution_error_message(data))

            if cancel_event.is_set():
                yield MediaEvent(type="cancelled")
                return
            yield MediaEvent(
                type="progress",
                phase="Collecting media output",
                data={"indeterminate": True},
            )
            if cancel_event.is_set():
                yield MediaEvent(type="cancelled")
                return
            assets = await self._collect_outputs(
                prompt_id, request.operation, cancel_event, own_uploads=uploads
            )
            # Collected outputs were removed from the backend as they were read.
            # Nothing collected means a stop ended the wait for the run's record,
            # and the teardown still has to look for its files.
            outputs_collected = bool(assets)
            if cancel_event.is_set():
                # A stop that arrives while the result is being collected wins,
                # as it does just before: the run ends cancelled, not complete.
                yield MediaEvent(type="cancelled")
                return
            yield MediaEvent(
                type="complete",
                progress=1,
                phase="complete",
                assets=assets,
                engine_sourced=True,
            )
        except GeneratorExit:
            # The consumer closed this producer rather than finishing it: a
            # superseded execution, dropping a stream whose result nobody will
            # read. Recorded here and acted on in the teardown, because that is
            # where it is known whether a prompt was ever submitted.
            abandoned = True
            raise
        except asyncio.CancelledError:
            abandoned = True
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
            if submitted:
                self._in_flight -= 1
                if not outputs_collected:
                    self._unsettled += 1
            try:
                if prompt_id and not outputs_collected:
                    if not removed and (
                        bound_after_submission_failure
                        or ((abandoned or refused_after_acceptance) and not cancel_event.is_set())
                    ):
                        # A prompt was submitted and the consumer walked away.
                        # Removing the run's entries below ends this adapter's
                        # interest in it and nothing else: the backend goes on
                        # with a result nobody will read, holding the device
                        # against the run that replaced this one.
                        #
                        # A set cancel event means `cancel` already stopped the
                        # prompt - but ONLY where `cancel` could see it. It stops
                        # the prompt `self._jobs` names for the run, and on the
                        # lost-reply recovery path there is no entry there until
                        # the reconciliation returns. A cancel arriving during that
                        # await therefore sets the event and stops nothing, so
                        # reading the set event as "already stopped" would leave
                        # the recovered prompt running with the caller told it
                        # failed - the exact state this path exists to prevent.
                        # `bound_after_submission_failure` is checked ahead of the
                        # event for that reason. Repeating a stop is harmless;
                        # skipping one is not.
                        #
                        # Still deliberately not extended to the generic error
                        # exits below. A malformed or oversized event is a different
                        # case with its own settled handling, and widening this to
                        # cover it would change behaviour six existing tests rely
                        # on.
                        #
                        # `refused_after_acceptance` is not that widening. It marks
                        # the one exit where THIS code refuses a response the backend
                        # has already accepted and queued, which is the same
                        # situation as an abandoned stream and not an error arriving
                        # from a prompt already being consumed.
                        #
                        # `bound_after_submission_failure` is the same situation reached the
                        # other way: the backend took the prompt and this code never
                        # heard so. It is only ever set when the backend's own queue
                        # named the prompt under this submission's client id, so it
                        # cannot mark a generation that was never accepted.
                        await self._abandon_prompt(prompt_id)
                    await self._cleanup_prompt_outputs(prompt_id)
            finally:
                # Last, and even when an await above is cancelled. While this
                # teardown waits on the backend, its caller can still count the
                # run as generating, so a stop can still arrive, and it has to
                # find this event to set. Without one, `cancel` keeps the stop
                # for the run's next generation, and a retry of this run would
                # end at once without sending its prompt. A newer generation of
                # the same run owns the entries once it has registered, so only
                # this generation's own are removed.
                if self._cancel_events.get(request.run_id) is cancel_event:
                    self._jobs.pop(request.run_id, None)
                    self._cancel_events.pop(request.run_id, None)
                    self._uploads.pop(request.run_id, None)
                    self._cancelled.discard(request.run_id)

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
        prompt_id: str | None = None
        try:
            async with asyncio.timeout(timeout_seconds):
                async for event in self.generate(request):
                    named = event.data.get("prompt_id")
                    if isinstance(named, str):
                        prompt_id = named
                    if event.type == "complete" and event.assets:
                        completed = True
        except TimeoutError as exc:
            # The deadline cancelled `generate`, and its teardown made a short
            # attempt to stop the probe's prompt. The probe asks again with the
            # time a requested stop gets, so a backend that answers slowly still
            # gets every step. A prompt it never saw queued has no name to give,
            # and a stop that names none would end whatever ComfyUI is running.
            if prompt_id is not None:
                await self._abandon_prompt(prompt_id, seconds=STOP_SECONDS)
            raise RuntimeError(
                "ComfyUI did not complete the bounded model activation probe."
            ) from exc
        if not completed:
            raise RuntimeError("ComfyUI did not produce media during the model activation probe.")

    async def _finished_history(
        self,
        prompt_id: str,
        cancel_event: asyncio.Event | None = None,
    ) -> dict[str, Any] | None:
        """The backend's record of a finished prompt, once it has written one.

        ComfyUI reports success from inside the run and stores the run's
        history only afterwards, once its end-of-run work is done. With large
        models that work can take long enough for a read made on the report to
        find no record of a run that did produce output. A missing record is
        therefore read again, for a bounded time, before it counts as no output;
        a record that is present is final as it stands.

        None means a stop arrived while the record was still missing; waiting
        out the rest of the bound would only delay the cancellation.
        """

        deadline = time.monotonic() + _HISTORY_SETTLE_SECONDS
        while True:
            response = await self._client.get(f"/history/{prompt_id}", timeout=30)
            response.raise_for_status()
            if len(response.content) > _MAX_COMFY_JSON_BYTES:
                raise RuntimeError("ComfyUI output history is too large")
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("ComfyUI returned invalid output history")
            if prompt_id in payload or time.monotonic() >= deadline:
                break
            if cancel_event is None:
                await asyncio.sleep(_HISTORY_POLL_SECONDS)
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(cancel_event.wait(), timeout=_HISTORY_POLL_SECONDS)
            if cancel_event.is_set():
                return None
        history = payload.get(prompt_id, {})
        if not isinstance(history, dict):
            raise RuntimeError("ComfyUI returned invalid output history")
        return history

    async def _collect_outputs(
        self,
        prompt_id: str,
        operation: str,
        cancel_event: asyncio.Event | None = None,
        *,
        own_uploads: Collection[str] = (),
    ) -> list[GeneratedAsset]:
        history = await self._finished_history(prompt_id, cancel_event)
        if history is None:
            return []
        outputs = history.get("outputs") or {}
        if not isinstance(outputs, dict):
            raise RuntimeError("ComfyUI returned invalid output history")
        assets: list[GeneratedAsset] = []
        managed_paths: set[Path] = set()
        output_items: list[tuple[dict[str, Any], str, str, object]] = []
        too_many_outputs = False
        # Keyed by the graph node that wrote each file. Dropping the key here
        # was the only place that fact existed: afterwards a file is addressed
        # by a digest of its own content, and two nodes that wrote the same
        # bytes are one row.
        for node_id, node_output in outputs.items():
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
                    if _is_own_upload(item, own_uploads):
                        # ComfyUI's LoadVideo shows the clip it loaded as a
                        # preview. That is this run's own input, not a result.
                        continue
                    if len(output_items) < _MAX_COMFY_OUTPUTS:
                        output_items.append((item, default_kind, collection, node_id))
                    else:
                        too_many_outputs = True
                    if managed_path := self._managed_output_path(item):
                        managed_paths.add(managed_path)
        try:
            if too_many_outputs:
                raise _WorkflowFailed(f"ComfyUI returned more than {_MAX_COMFY_OUTPUTS} outputs")
            total_bytes = 0
            for item, default_kind, collection, node_id in output_items:
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
                    raise _WorkflowFailed(
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
                        origin=stated_origin(node_id, params["type"], collection),
                    )
                )
            if not assets:
                # The run's record says it left nothing to collect. ComfyUI never
                # writes an empty record, so an empty one is a record that never
                # appeared, and that says nothing about how the run ended.
                failure = _WorkflowFailed if history else RuntimeError
                raise failure("ComfyUI completed without collectible image or video outputs")
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
                    raise _WorkflowFailed(
                        f"ComfyUI outputs exceed the {self.max_output_bytes}-byte limit"
                    )
            except ValueError:
                pass
        content = bytearray()
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > remaining_bytes:
                raise _WorkflowFailed(
                    f"ComfyUI outputs exceed the {self.max_output_bytes}-byte limit"
                )
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

    async def _cleanup_prompt_outputs(self, prompt_id: str, *, strict: bool = False) -> None:
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
            if strict:
                raise RuntimeError("ComfyUI could not purge private output files") from exc
            logger.debug(
                "Could not inspect a failed ComfyUI output (%s)",
                type(exc).__name__,
            )

    def _managed_upload_root(self) -> Path | None:
        """Where this adapter's uploaded conditioning input and masks land.

        `managed_temp_root` is the BACKEND's whole temp directory and holds
        files this adapter did not write, so it is not ours to sweep. Every
        upload is addressed to one subfolder of it, and that subfolder is.
        """

        root = self.managed_temp_root
        return root / _UPLOAD_SUBFOLDER if root is not None else None

    async def _sweep_stale_outputs(self) -> None:
        # Uploaded conditioning input is reclaimed on the same terms as output.
        # Nothing else removed uploads: the per-run cleanup covers outputs, and
        # this sweep walked the output root alone, so they accumulated for the
        # life of the installation. A content-named upload is rewritten by every
        # content-named upload of the same bytes, renewing its age, so one that
        # is still being reused is kept.
        roots = [
            root
            for root in (self._managed_output_sweep_root, self._managed_upload_root())
            if root is not None
        ]
        now = time.time()
        if not roots or now - self._last_output_sweep < min(
            3600,
            self.stale_output_seconds,
        ):
            return
        self._last_output_sweep = now
        cutoff = now - self.stale_output_seconds
        for root in roots:
            await asyncio.to_thread(self._sweep_stale_outputs_sync, root, cutoff)

    @staticmethod
    def _sweep_stale_outputs_sync(root: Path, cutoff: float) -> None:
        """Reclaim old entries through held directories, skipping unsafe trees."""

        try:
            with AnchoredDirectory(root) as anchor:
                # Existing cleanup directories have no entry-count ceiling.
                # A bounded-list refusal would permanently strand a large backlog;
                # retain the complete per-directory walk and its memory cost.
                pending: list[tuple[AnchoredDirectory, Iterator[AnchoredEntry], str | None]] = [
                    (anchor, iter(list_entries(anchor, limit=sys.maxsize)), None)
                ]
                try:
                    while pending:
                        current, entries, name = pending[-1]
                        entry = next(entries, None)
                        if entry is None:
                            pending.pop()
                            if name is not None:
                                current.close()
                                with suppress(AnchoredDirectoryError):
                                    remove_directory_entry(pending[-1][0], name)
                            continue
                        if entry.kind is AnchoredEntryKind.DIRECTORY:
                            child = None
                            try:
                                child = open_child_directory(current, entry.name)
                                children = iter(list_entries(child, limit=sys.maxsize))
                            except (AnchoredDirectoryError, OSError):
                                if child is not None:
                                    child.close()
                                continue
                            pending.append((child, children, entry.name))
                        elif (
                            entry.kind is AnchoredEntryKind.FILE
                            and entry.modified_at is not None
                            and entry.modified_at.timestamp() <= cutoff
                        ):
                            with suppress(AnchoredDirectoryError):
                                remove_entry(current, entry.name)
                finally:
                    # Retain every ancestor through descent and release children
                    # before their parents, including when enumeration refuses.
                    for held, _, name in reversed(pending):
                        if name is not None:
                            held.close()
        except (AnchoredDirectoryError, OSError):
            # Missing roots, linked ancestry and refused listings are skipped.
            # The next age-based sweep may retry; no pathname fallback deletes.
            return

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
        if prompt_id := self._jobs.get(run_id):
            await self._abandon_prompt(prompt_id, seconds=STOP_SECONDS)

    async def release_memory(self) -> bool:
        """Have ComfyUI unload its models and free its cache, and wait until it has.

        ComfyUI's `/free` only sets flags its worker reads between prompts, so it
        answers before anything is freed, and a request that arrives while the
        worker finishes a prompt can wait for the next one; it is sent once more
        if nothing has changed after a while. True when ComfyUI's count of free
        device memory on its first device rose by `RELEASE_RISE_BYTES` and then
        held still, or at once when ComfyUI took no prompt of this adapter's
        since it started or was last emptied: nothing of this adapter's is
        loaded and no rise would come, so anything someone else left there is
        asked to go without waiting. False when ComfyUI refused, did not
        answer, the count could not be read, it had not risen
        `RELEASE_GIVE_UP_SECONDS` after the first request, or it never settled
        within `RELEASE_SECONDS`; the caller then stops the worker as it
        otherwise would. Nothing read is logged or kept beyond the counts.
        """

        try:
            if not self._prompts_since_emptied:
                return await self._ask_to_free()
            async with asyncio.timeout(RELEASE_SECONDS):
                before = await self._free_device_memory()
                if before is None or not await self._ask_to_free():
                    return False
                asked = time.monotonic()
                asked_again = False
                previous = before
                while True:
                    await asyncio.sleep(RELEASE_POLL_SECONDS)
                    current = await self._free_device_memory()
                    if current is None:
                        return False
                    risen = current >= before + RELEASE_RISE_BYTES
                    if (
                        risen
                        and previous >= before + RELEASE_RISE_BYTES
                        and abs(current - previous) <= RELEASE_SETTLED_BYTES
                    ):
                        self._prompts_since_emptied = 0
                        return True
                    waited = time.monotonic() - asked
                    if not risen and waited >= RELEASE_GIVE_UP_SECONDS:
                        return False
                    if not risen and not asked_again and waited >= RELEASE_ASK_AGAIN_SECONDS:
                        asked_again = True
                        if not await self._ask_to_free():
                            return False
                    previous = current
        except (httpx.HTTPError, ValueError, TimeoutError) as error:
            logger.info("ComfyUI's memory was not emptied: %s", type(error).__name__)
            return False

    async def _ask_to_free(self) -> bool:
        """Ask ComfyUI to unload its models and free its cache; whether it accepted."""

        response = await self._client.post(
            "/free",
            json={"unload_models": True, "free_memory": True},
            timeout=RELEASE_REQUEST_SECONDS,
        )
        if response.status_code != 200:
            logger.info("ComfyUI refused to empty its memory: HTTP %s", response.status_code)
            return False
        return True

    async def _free_device_memory(self) -> int | None:
        """The free device memory ComfyUI counts on its first device, or None if it cannot say.

        ComfyUI frees every device, and its first one is where it loads models.
        """

        response = await self._client.get("/system_stats", timeout=RELEASE_REQUEST_SECONDS)
        if response.status_code != 200 or len(response.content) > _MAX_COMFY_JSON_BYTES:
            return None
        body = response.json()
        devices = body.get("devices") if isinstance(body, dict) else None
        first = devices[0] if isinstance(devices, list) and devices else None
        free = first.get("vram_free") if isinstance(first, dict) else None
        if isinstance(free, bool) or not isinstance(free, int | float) or not 0 <= free < 2**62:
            return None
        return int(free)

    async def queue_remaining(self) -> int | None:
        """How many prompts ComfyUI holds, running or waiting, or None if it cannot say."""

        try:
            response = await self._client.get("/prompt", timeout=5)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        info = payload.get("exec_info") if isinstance(payload, dict) else None
        remaining = info.get("queue_remaining") if isinstance(info, dict) else None
        if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
            return None
        return remaining

    async def foreign_prompt_count(self) -> int | None:
        """How many prompts ComfyUI holds that LM Atelier did not submit, or None if it cannot say.

        A prompt LM Atelier submitted can outlive the generation that sent it:
        a node can hang through a stop, and a stop can fail to reach ComfyUI.
        That prompt is LM Atelier's own work, so it is no reason to wait before
        ending what ComfyUI is doing; only work someone else queued is.

        The count is read first, and the queue only when one of this adapter's
        prompts may still be there: a generation has sent one and not finished,
        or one finished without collecting its outputs since ComfyUI was last
        seen holding none of them. The queue holds every prompt's graph, so
        only the client ids are taken from it. A queue that cannot be read
        counts every prompt as someone else's: waiting on a wrong answer costs
        time, and stopping ComfyUI on one ends work that cannot be given back.
        """

        # Only generations that finished before ComfyUI answered can be shown
        # gone by its answer; one that finishes meanwhile stays unsettled.
        seen = self._unsettled
        total = await self.queue_remaining()
        if total is None:
            return None
        if total == 0:
            self._settled_through = max(self._settled_through, seen)
            return 0
        if not self._in_flight and self._settled_through >= self._unsettled:
            return total
        clients = await self._queued_client_ids()
        if clients is None:
            return total
        ours = sum(
            1 for client in clients if client is not None and client.startswith(self._client_prefix)
        )
        if not ours:
            self._settled_through = max(self._settled_through, seen)
        return len(clients) - ours

    async def _queued_client_ids(self) -> list[str | None] | None:
        """The client id of each prompt ComfyUI holds, running or waiting.

        A queue row is `[number, prompt_id, prompt, extra_data,
        outputs_to_execute]`, with the client id in `extra_data`. The rest of a
        row is a graph and the workflow it came from, often someone else's, so
        only the client ids are taken from the body. None of it is logged or
        kept, and no error leaves this method to carry it. A row whose client
        id is missing or not a string is None, which counts as someone else's.

        None means ComfyUI could not say: no answer within the reconcile
        deadline, an error status, a body over the JSON bound, or a body or
        section of another shape. A section is never skipped, because a partial
        reading could miss another application's prompt.
        """

        with suppress(Exception):
            async with asyncio.timeout(RECONCILE_SECONDS):
                response = await self._client.get("/queue")
                response.raise_for_status()
                if len(response.content) > _MAX_COMFY_JSON_BYTES:
                    return None
                payload = response.json()
            if not isinstance(payload, dict):
                return None
            clients: list[str | None] = []
            for key in ("queue_running", "queue_pending"):
                rows = payload.get(key)
                if not isinstance(rows, list):
                    return None
                for row in rows:
                    extra = row[3] if isinstance(row, list) and len(row) >= 4 else None
                    client = extra.get("client_id") if isinstance(extra, dict) else None
                    clients.append(client if isinstance(client, str) else None)
            return clients
        return None

    async def _prompt_id_for_client(self, client_id: str) -> str | None:
        """The queued prompt this submission's client id names, if there is one.

        A submission whose reply never arrived may still have been taken. The
        request went out; only the answer was lost, and the backend has no
        reason to have refused it. Without an identifier nothing can stop that
        generation: it holds the device against every generation after it while
        the caller has already been told this one failed.

        `client_id` is minted per generation, travels in the prompt's
        `extra_data`, and is returned by `/queue`, so it names the prompt THIS
        call created and no other. A queue entry is
        `[number, prompt_id, prompt, extra_data, outputs_to_execute]`.

        Best-effort inside a deadline, because this runs on a failure path: a
        second failure here must not replace the first, and a backend that will
        not answer must not hold the unwinding open.

        Only the queue is consulted. A prompt that finished between the timeout
        and this question is in history rather than the queue, and there is
        nothing left to stop; reclaiming what it wrote is a different question
        and is not answered here.
        """

        with suppress(Exception):
            async with asyncio.timeout(RECONCILE_SECONDS):
                response = await self._client.get("/queue")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    return None
                for key in ("queue_running", "queue_pending"):
                    entries = payload.get(key)
                    if not isinstance(entries, list):
                        continue
                    for entry in entries:
                        if not isinstance(entry, list) or len(entry) < 4:
                            continue
                        extra = entry[3]
                        if not isinstance(extra, dict):
                            continue
                        if extra.get("client_id") != client_id:
                            continue
                        candidate = entry[1]
                        if _usable_prompt_id(candidate):
                            return candidate
        return None

    async def _prompt_still_held(self, prompt_id: str) -> bool:
        """Whether ComfyUI still holds the prompt: waiting, running or with a record.

        ComfyUI's job endpoint answers for this prompt alone. A ComfyUI without
        it is asked its queue and then the prompt's history: a prompt moves from
        one to the other in one step, so one in neither was removed before it
        ran, or ran and had its record cleared since. The queue carries every
        client's prompts and graphs; only prompt ids are compared, and nothing
        read is logged or kept. No answer within the deadline, an error, a body
        over the JSON bound or a row of another shape counts as held, so a slow
        or failing ComfyUI never ends a run this way.
        """

        try:
            async with asyncio.timeout(RECONCILE_SECONDS):
                identifier = quote(prompt_id, safe="")
                if identifier in {".", ".."}:
                    identifier = identifier.replace(".", "%2E")
                response = await self._client.get(f"/api/jobs/{identifier}")
                if response.status_code == 200:
                    return True
                if response.status_code == 404 and _names_no_such_job(response):
                    return False
                response = await self._client.get("/queue")
                response.raise_for_status()
                if len(response.content) > _MAX_COMFY_JSON_BYTES:
                    return True
                queue = response.json()
                if not isinstance(queue, dict):
                    return True
                for key in ("queue_running", "queue_pending"):
                    rows = queue.get(key)
                    if not isinstance(rows, list):
                        return True
                    for row in rows:
                        if not isinstance(row, list) or len(row) < 2 or row[1] == prompt_id:
                            return True
                response = await self._client.get(f"/history/{identifier}")
                response.raise_for_status()
                if len(response.content) > _MAX_COMFY_JSON_BYTES:
                    return True
                history = response.json()
                return not isinstance(history, dict) or bool(history)
        except Exception:
            return True

    async def _abandon_prompt(self, prompt_id: str, *, seconds: float | None = None) -> None:
        """Cancel the submitted prompt through the atomic job endpoint when available.

        A completed or unknown job is a successful no-op. Only an unavailable
        endpoint permits the older delete-then-interrupt protocol, which remains
        best effort because its running check and interrupt are separate steps.
        One total deadline bounds all of the requests: by default the short
        one for an attempt already unwinding, so output cleanup can still run,
        and `seconds` (STOP_SECONDS) for a stop the run asks for. Failures
        remain local.

        Every stop this adapter sends comes through here, naming the prompt it
        submitted. An interrupt that names no prompt stops whatever ComfyUI is
        running: while this run waits its turn, that can be another
        application's work or a graph queued from an open workflow editor.
        """

        with suppress(Exception):
            async with asyncio.timeout(ABANDONED_INTERRUPT_SECONDS if seconds is None else seconds):
                identifier = quote(prompt_id, safe="")
                if identifier in {".", ".."}:
                    identifier = identifier.replace(".", "%2E")
                response = await self._client.post(f"/api/jobs/{identifier}/cancel")
                if response.status_code not in (404, 405):
                    return
                await self._client.post("/queue", json={"delete": [prompt_id]})
                await self._client.post("/interrupt", json={"prompt_id": prompt_id})

    async def close(self) -> None:
        for event in self._cancel_events.values():
            event.set()
        await self._client.aclose()
