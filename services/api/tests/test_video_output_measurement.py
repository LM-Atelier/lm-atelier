"""Measuring a produced video by a frame the decoder actually made.

The success and truncation cases drive the REAL ffmpeg, on videos this file
encodes from a flat colour, because the property is about what a decoder does
with the bytes. Hosted CI installs ffmpeg on both platforms for exactly this
kind of test, so its absence fails here rather than skipping.

The supervision cases replace the child process, because a timeout, a flood of
output or a cancellation has to be produced on demand to show the child is
killed and reaped and the temporary copy removed.
"""

from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from local_lm import video_output_measurement
from local_lm.output_measurement import Budget
from local_lm.video_output_measurement import (
    MAX_ERROR_BYTES,
    MAX_FRAME_BYTES,
    METHOD,
    measure_video_output,
)

pytestmark = pytest.mark.asyncio


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    assert executable, "these cases need a real ffmpeg on PATH, as hosted CI provides"
    return executable


def _neutral_video(tmp_path: Path, width: int, height: int, *, index_at_end: bool = True) -> bytes:
    """A one-second flat-colour H.264 MP4, encoded here from nothing but a size.

    By default the index is written at the END of the file, which is what a
    plain encode produces and what a pipe cannot read.
    """

    target = tmp_path / f"neutral-{width}x{height}.mp4"
    arguments = [
        _ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x336699:s={width}x{height}:r=24:d=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if not index_at_end:
        arguments += ["-movflags", "+faststart"]
    subprocess.run([*arguments, str(target)], check=True, capture_output=True, timeout=60)
    return target.read_bytes()


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private temporary directory, so leftover copies can be counted exactly."""

    directory = tmp_path / "scratch"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


def _copies(scratch: Path) -> list[Path]:
    return sorted(scratch.glob("lm-atelier-measure-*"))


# ---- what a real decoder produces ---------------------------------------------------


async def test_a_video_is_measured_at_the_size_of_its_decoded_frame(
    tmp_path: Path, scratch: Path
) -> None:
    content = _neutral_video(tmp_path, 96, 64)

    record = await measure_video_output(content, Budget())

    assert record["state"] == "measured"
    assert record["method"] == METHOD
    assert (record["raster_width"], record["raster_height"]) == (96, 64)
    assert _copies(scratch) == []


async def test_the_measured_size_follows_the_bytes_not_a_constant(
    tmp_path: Path, scratch: Path
) -> None:
    """Two sizes, two answers: a measurement that returned a fixed pair fails here."""

    tall = await measure_video_output(_neutral_video(tmp_path, 48, 128), Budget())
    faststart = await measure_video_output(
        _neutral_video(tmp_path, 160, 90, index_at_end=False), Budget()
    )

    assert (tall["raster_width"], tall["raster_height"]) == (48, 128)
    assert (faststart["raster_width"], faststart["raster_height"]) == (160, 90)


async def test_a_truncated_video_is_not_measured_at_the_size_it_declares(
    tmp_path: Path, scratch: Path
) -> None:
    """The failure a header read would hide.

    Cutting the file in half keeps its first bytes and loses the index at the
    end, so no frame can be decoded. The size is still written in the stream's
    parameters; reporting it would be exactly the false success this avoids.
    """

    content = _neutral_video(tmp_path, 96, 64)

    record = await measure_video_output(content[: len(content) // 2], Budget())

    assert record == {"v": 1, "state": "unmeasured", "about": "file", "reason": "decode_failed"}
    assert _copies(scratch) == []


async def test_bytes_that_are_not_a_video_are_not_measured(scratch: Path) -> None:
    record = await measure_video_output(b"not a video at all" * 64, Budget())

    assert record["state"] == "unmeasured"
    assert record["reason"] == "decode_failed"
    assert _copies(scratch) == []


# ---- the environment and the budget ---------------------------------------------------


async def test_without_a_decoder_the_video_is_unmeasured_and_nothing_is_started(
    monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> None:
    started: list[object] = []

    async def create_process(*arguments: object, **_options: object) -> object:
        started.append(arguments)
        raise AssertionError("no process may start without a decoder")

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    record = await measure_video_output(b"anything", Budget())

    assert record == {
        "v": 1,
        "state": "unmeasured",
        "about": "environment",
        "reason": "decoder_unavailable",
    }
    assert started == []
    assert _copies(scratch) == []


async def test_the_time_allowance_is_shared_by_every_video_in_a_generation(
    monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> None:
    """Spent once, gone for the rest: a second video does not get a fresh allowance."""

    launches: list[float] = []

    async def create_process(*_arguments: object, **_options: object) -> _FakeProcess:
        launches.append(1.0)
        return _FakeProcess(stdout=_Blocking(), stderr=_Blocking())

    monkeypatch.setattr(shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    budget = Budget(video_seconds=0.2)

    started = time.monotonic()
    first = await measure_video_output(b"first", budget)
    elapsed = time.monotonic() - started
    second = await measure_video_output(b"second", budget)

    assert first["reason"] == "over_time"
    # The deadline is what the generation had left, not the per-video maximum.
    # Waiting the full per-video time would still say over_time, only late.
    assert elapsed < 5, f"the first video waited {elapsed:.1f}s against a 0.2s allowance"
    assert second["reason"] == "over_time"
    assert len(launches) == 1, "the second video started a decoder with no time left"
    assert budget.video_seconds <= 0
    assert _copies(scratch) == []


# ---- supervision of the child -----------------------------------------------------------


@pytest.mark.parametrize(
    ("available", "elapsed", "remaining"),
    [(0.2, 0.19, 0.0), (60.0, 29.99, 30.0), (60.0, 32.0, 28.0)],
)
async def test_a_decoder_timeout_spends_its_allowance_even_when_the_clock_returns_early(
    monkeypatch: pytest.MonkeyPatch,
    scratch: Path,
    available: float,
    elapsed: float,
    remaining: float,
) -> None:
    clock = [100.0]
    allowances: list[float] = []

    async def timeout(_executable: str, _source: Path, seconds: float) -> bytes | None:
        allowances.append(seconds)
        clock[0] += elapsed
        raise TimeoutError

    monkeypatch.setattr(
        video_output_measurement, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(video_output_measurement, "_first_frame", timeout)
    monkeypatch.setattr(shutil, "which", lambda _name: "ffmpeg")
    budget = Budget(video_seconds=available)

    record = await measure_video_output(b"neutral", budget)

    assert record["reason"] == "over_time"
    assert allowances == [min(30.0, available)]
    assert budget.video_seconds == pytest.approx(remaining)
    if remaining == 0:
        again = await measure_video_output(b"next", budget)
        assert again["reason"] == "over_time"
        assert len(allowances) == 1
    assert _copies(scratch) == []


async def test_a_decoder_that_finishes_early_leaves_time_for_the_next_video(
    monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> None:
    clock = [100.0]
    allowances: list[float] = []

    async def finish(_executable: str, _source: Path, seconds: float) -> bytes | None:
        allowances.append(seconds)
        clock[0] += 0.05
        return None

    monkeypatch.setattr(
        video_output_measurement, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(video_output_measurement, "_first_frame", finish)
    monkeypatch.setattr(shutil, "which", lambda _name: "ffmpeg")
    budget = Budget(video_seconds=0.2)

    first = await measure_video_output(b"first", budget)
    second = await measure_video_output(b"second", budget)

    assert first["reason"] == second["reason"] == "decode_failed"
    assert allowances == pytest.approx([0.2, 0.15])
    assert budget.video_seconds == pytest.approx(0.1)
    assert _copies(scratch) == []


class _Blocking:
    """A stream that produces nothing until its process is killed, and then ends.

    Ending on kill is what a real pipe does, and what stopping a decoder relies
    on when it drains the pipe before waiting.
    """

    def __init__(self) -> None:
        self.reading = asyncio.Event()
        self.closed = asyncio.Event()

    async def read(self, _size: int) -> bytes:
        self.reading.set()
        await self.closed.wait()
        return b""


class _Flood:
    def __init__(self, size: int) -> None:
        self.remaining = size

    async def read(self, size: int) -> bytes:
        if self.remaining <= 0:
            return b""
        piece = min(size, self.remaining)
        self.remaining -= piece
        return b"x" * piece


class _Empty:
    async def read(self, _size: int) -> bytes:
        return b""


class _Once:
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def read(self, _size: int) -> bytes:
        content, self.content = self.content, b""
        return content


class _FakeProcess:
    def __init__(self, *, stdout: Any, stderr: Any, exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.returncode: int | None = None
        self.killed = False
        self.reaped = False

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        for stream in (self.stdout, self.stderr):
            if isinstance(stream, _Blocking):
                stream.closed.set()

    async def wait(self) -> int:
        if self.returncode is None:
            # Only reachable for a child that was not killed; a real one would
            # finish on its own once its streams are drained.
            self.returncode = self.exit_code
        self.reaped = True
        return self.returncode


def _replace_decoder(monkeypatch: pytest.MonkeyPatch, process: _FakeProcess) -> None:
    async def create_process(*_arguments: object, **_options: object) -> _FakeProcess:
        return process

    monkeypatch.setattr(shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)


async def test_a_frame_larger_than_its_bound_kills_and_reaps_the_decoder(
    monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> None:
    process = _FakeProcess(stdout=_Flood(MAX_FRAME_BYTES + 1), stderr=_Blocking())
    _replace_decoder(monkeypatch, process)

    record = await measure_video_output(b"neutral", Budget())

    assert record["reason"] == "over_output_budget"
    assert (process.killed, process.reaped) == (True, True)
    assert _copies(scratch) == []


async def test_an_error_stream_larger_than_its_bound_kills_and_reaps_the_decoder(
    monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> None:
    process = _FakeProcess(stdout=_Blocking(), stderr=_Flood(MAX_ERROR_BYTES + 1))
    _replace_decoder(monkeypatch, process)

    record = await measure_video_output(b"neutral", Budget())

    assert record["reason"] == "over_output_budget"
    assert (process.killed, process.reaped) == (True, True)
    assert _copies(scratch) == []


async def test_a_decoder_that_never_finishes_is_killed_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> None:
    process = _FakeProcess(stdout=_Blocking(), stderr=_Blocking())
    _replace_decoder(monkeypatch, process)

    record = await measure_video_output(b"neutral", Budget(video_seconds=0.1))

    assert record == {"v": 1, "state": "unmeasured", "about": "budget", "reason": "over_time"}
    assert (process.killed, process.reaped) == (True, True)
    assert _copies(scratch) == []


async def test_a_decoder_that_exits_with_nothing_is_a_failed_decode(
    monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> None:
    process = _FakeProcess(stdout=_Empty(), stderr=_Empty())
    _replace_decoder(monkeypatch, process)

    record = await measure_video_output(b"neutral", Budget())

    assert record["reason"] == "decode_failed"
    assert process.killed is False
    assert _copies(scratch) == []


async def test_a_frame_from_a_decoder_that_failed_is_not_measured(
    monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> None:
    """A well-formed picture on the output does not outweigh a failing exit status."""

    buffer = io.BytesIO()
    Image.new("RGB", (40, 30)).save(buffer, "PNG")
    process = _FakeProcess(stdout=_Once(buffer.getvalue()), stderr=_Empty(), exit_code=1)
    _replace_decoder(monkeypatch, process)

    record = await measure_video_output(b"neutral", Budget())

    assert record["reason"] == "decode_failed"
    assert _copies(scratch) == []


async def test_cancelling_a_measurement_reaps_the_decoder_and_removes_the_copy(
    monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> None:
    stdout = _Blocking()
    process = _FakeProcess(stdout=stdout, stderr=_Blocking())
    _replace_decoder(monkeypatch, process)

    task = asyncio.create_task(measure_video_output(b"neutral", Budget()))
    # Bounded, so a measurement that fails before ever reading ends this case
    # instead of leaving it waiting for a read that will never come.
    reading = asyncio.create_task(stdout.reading.wait())
    done, _ = await asyncio.wait({task, reading}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
    assert reading in done, "the decoder never started reading"
    assert len(_copies(scratch)) == 1, "the decoder should be reading a private copy"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (process.killed, process.reaped) == (True, True)
    assert _copies(scratch) == []
