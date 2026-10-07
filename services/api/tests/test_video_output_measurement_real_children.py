"""Stopping a decoder past its deadline, or by cancellation, with REAL subprocess pipes.

The replaced-process cases cannot show how stopping behaves on real pipes: their
wait() returns at once. The flooding decoder has its own module beside this one.
These cover the other two ways a decoder is stopped - its deadline passing and
the measurement being cancelled - with an actual child that wrote into its pipe
and then stopped responding, through the real measurement entry point, and
require cleanup to finish on its own within a bound.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from local_lm import video_output_measurement as measurement
from local_lm.output_measurement import Budget

pytestmark = pytest.mark.asyncio

#: Well above what a correct stop needs, and well below a hang.
_FINISH_WITHIN = 5


def _launch_as(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str
) -> list[asyncio.subprocess.Process]:
    """Run `script` in place of the decoder, with the real pipes the decoder would get."""

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda _name: sys.executable)
    launch = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []

    async def child(*_arguments: object, **_options: object) -> asyncio.subprocess.Process:
        process = await launch(
            sys.executable,
            "-c",
            script,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", child)
    return processes


async def _rescue(processes: list[asyncio.subprocess.Process]) -> None:
    """Only reached when the measurement failed to stop its child: clean up after the test."""

    for process in processes:
        if process.returncode is None:
            process.kill()
        await asyncio.gather(
            *(stream.read() for stream in (process.stdout, process.stderr) if stream is not None)
        )


async def test_a_real_decoder_past_its_deadline_is_stopped_and_reaped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """It wrote enough to fill its pipe and then stopped responding."""

    processes = _launch_as(
        monkeypatch,
        tmp_path,
        "import os, time; os.write(2, b'x' * 60000); time.sleep(120)",
    )

    task = asyncio.create_task(
        measurement.measure_video_output(b"neutral-video-fixture", Budget(video_seconds=0.5))
    )
    done, _ = await asyncio.wait({task}, timeout=_FINISH_WITHIN)
    if not done:
        await _rescue(processes)
        await asyncio.wait_for(task, timeout=_FINISH_WITHIN)

    assert done, "a decoder past its deadline was not stopped within the bound"
    assert task.result()["reason"] == "over_time"
    assert all(process.returncode is not None for process in processes)
    assert not list(tmp_path.glob("lm-atelier-measure-*"))


async def test_cancelling_a_real_decoder_reaps_it_and_removes_the_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    processes = _launch_as(
        monkeypatch,
        tmp_path,
        "import os, time; os.write(1, b'x' * 60000); time.sleep(120)",
    )

    task = asyncio.create_task(measurement.measure_video_output(b"neutral-video-fixture", Budget()))
    for _ in range(100):
        if processes:
            break
        await asyncio.sleep(0.05)
    assert processes, "the decoder was never started"
    await asyncio.sleep(0.3)

    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=_FINISH_WITHIN)
    if not done:
        await _rescue(processes)
        await asyncio.wait({task}, timeout=_FINISH_WITHIN)

    assert done, "a cancelled measurement did not finish stopping its decoder"
    assert task.cancelled()
    assert all(process.returncode is not None for process in processes)
    assert not list(tmp_path.glob("lm-atelier-measure-*"))
