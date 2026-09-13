"""What a produced video measures, read from a frame a decoder actually made.

A picture is measured by consuming its pixel stream, because a header is not
enough: a truncated file still declares its original size. The same holds for a
video, and more so - a container states a width and height whether or not a
single frame behind it decodes. So a video is measured by DECODING its first
frame to PNG with ffmpeg, without scaling, and measuring that PNG through the
picture path under the same generation budget. The size recorded is the size of
a frame the decoder produced.

WHAT THIS DOES NOT CLAIM. One decoded frame is evidence about that frame. A
stream that changed resolution part-way through would not be caught; a video
muxed from one batch of equally sized images has no second size to change to,
and decoding every frame to rule it out would cost what the whole video costs.
Nothing here ever falls back to the size the container declares.

SUPERVISION. ffmpeg runs without a shell, reading a private temporary copy of
the bytes - an MP4 whose index sits at the end of the file cannot be read from a
pipe. Its output and its error stream are both bounded. A timeout, an oversized
answer or a cancellation kills the child, drains what it had already written,
and waits for it before the copy is removed, so no process outlives the
measurement and no file outlives the process. The time allowance belongs to the
whole generation: many videos share it rather than each taking a fresh one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Final

from .output_measurement import RECORD_VERSION, Budget, measure_output
from .subprocess_env import subprocess_environment

METHOD: Final = "video_first_frame_png"

#: The most one video may take, and never more than the generation has left.
PER_VIDEO_SECONDS: Final = 30.0
#: A decoded frame as PNG. A frame large enough to need more than this is also
#: larger than the picture path's decode budget would accept.
MAX_FRAME_BYTES: Final = 64 * 1024 * 1024
#: ffmpeg is run quiet, so anything close to this is not an error message.
MAX_ERROR_BYTES: Final = 64 * 1024

#: How long stopping a decoder may take once it has been told to stop. Killing a
#: child is immediate; draining what it already wrote into its pipes is the part
#: that needs a bound.
_STOP_SECONDS: Final = 10.0

_READ: Final = 64 * 1024

logger = logging.getLogger(__name__)


class _Oversized(Exception):
    """A stream produced more than its bound."""


def _unmeasured(about: str, reason: str) -> dict[str, Any]:
    return {"v": RECORD_VERSION, "state": "unmeasured", "about": about, "reason": reason}


async def measure_video_output(content: bytes, budget: Budget) -> dict[str, Any]:
    """Measure one produced video by its first decoded frame, or name why not.

    Like the picture path, this never sees the run, the workflow or the size
    that was asked for, so it cannot agree with a request by construction.
    """

    if budget.video_seconds <= 0:
        return _unmeasured("budget", "over_time")
    executable = shutil.which("ffmpeg")
    if not executable:
        return _unmeasured("environment", "decoder_unavailable")
    try:
        handle, name = tempfile.mkstemp(prefix="lm-atelier-measure-", suffix=".video")
    except OSError:
        return _unmeasured("environment", "scratch_unavailable")
    copy = Path(name)
    # The write is shielded rather than abandoned on cancellation: a cancelled
    # thread keeps writing, and removing the file before it finishes would
    # leave the finished copy behind with nothing left to remove it.
    writing = asyncio.ensure_future(asyncio.to_thread(_write_and_close, handle, content))
    try:
        await asyncio.shield(writing)
    except asyncio.CancelledError:
        await asyncio.gather(writing, return_exceptions=True)
        _remove(copy)
        raise
    except OSError:
        _remove(copy)
        return _unmeasured("environment", "scratch_unavailable")
    try:
        started = time.monotonic()
        try:
            frame = await _first_frame(
                executable, copy, min(PER_VIDEO_SECONDS, budget.video_seconds)
            )
        except TimeoutError:
            return _unmeasured("budget", "over_time")
        except _Oversized:
            return _unmeasured("budget", "over_output_budget")
        except OSError:
            return _unmeasured("environment", "decoder_unavailable")
        finally:
            budget.video_seconds -= time.monotonic() - started
    finally:
        _remove(copy)
    if frame is None:
        return _unmeasured("file", "decode_failed")

    record = await asyncio.to_thread(measure_output, frame, budget)
    if record.get("state") != "measured":
        # The decoder claimed success and handed back something the picture
        # path cannot measure. That is a failure to decode, not a size.
        return _unmeasured("file", "decode_failed")
    return {**record, "method": METHOD}


def _write_and_close(handle: int, content: bytes) -> None:
    """Fill the copy this measurement created, and close it before anything reads it.

    `mkstemp` creates the name exclusively, so nothing else can have opened it
    first. The handle is closed before ffmpeg starts because some platforms
    refuse to let a second process read a file that is still open for writing.
    """

    with os.fdopen(handle, "wb") as stream:
        stream.write(content)


def _remove(copy: Path) -> None:
    """Remove the copy once nothing can still be using it.

    Called only after the child has been reaped. A copy that still cannot be
    removed is left to the operating system's temporary directory rather than
    turning a run that succeeded into a failure.
    """

    with contextlib.suppress(OSError):
        copy.unlink(missing_ok=True)


async def _first_frame(executable: str, source: Path, seconds: float) -> bytes | None:
    """The first frame as PNG bytes, or None when the decoder did not produce one."""

    process = await asyncio.create_subprocess_exec(
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=subprocess_environment(),
    )
    readers: list[asyncio.Task[bytes]] = []
    try:
        async with asyncio.timeout(seconds):
            readers = [
                asyncio.create_task(_bounded(process.stdout, MAX_FRAME_BYTES)),
                asyncio.create_task(_bounded(process.stderr, MAX_ERROR_BYTES)),
            ]
            frame, _errors = await asyncio.gather(*readers)
            await process.wait()
    except BaseException:
        await _stop(process, readers)
        raise
    if process.returncode != 0 or not frame:
        return None
    return frame


async def _stop(process: asyncio.subprocess.Process, readers: list[asyncio.Task[bytes]]) -> None:
    """Stop a decoder and wait for it, however much it had already written.

    Killing the child is not enough on its own. asyncio does not report a
    process finished until both of its pipes have closed, and a pipe whose
    reader stopped early is still holding the child's output: nothing reads it,
    so the pipe never reaches its end and waiting for the process never returns.
    So the abandoned readers are finished first - a stream allows one reader at
    a time - and then both pipes are read to their end and discarded, which lets
    them close.

    The whole stop is bounded. A decoder that still has not finished once that
    time is up is logged and left; the measurement has already failed, and
    blocking the run on it would turn a refused size into a hung generation.
    """

    for reader in readers:
        reader.cancel()
    await asyncio.gather(*readers, return_exceptions=True)
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    try:
        async with asyncio.timeout(_STOP_SECONDS):
            await asyncio.gather(_discard(process.stdout), _discard(process.stderr))
            await process.wait()
    except TimeoutError:
        logger.warning("A video decoder did not finish after it was stopped")


async def _discard(stream: asyncio.StreamReader | None) -> None:
    if stream is None:
        return
    while await stream.read(_READ):
        pass


async def _bounded(stream: asyncio.StreamReader | None, limit: int) -> bytes:
    if stream is None:
        return b""
    chunks: list[bytes] = []
    size = 0
    while chunk := await stream.read(_READ):
        size += len(chunk)
        if size > limit:
            raise _Oversized
        chunks.append(chunk)
    return b"".join(chunks)
