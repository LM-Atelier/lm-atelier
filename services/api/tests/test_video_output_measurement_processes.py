"""A bounded decoder answer must also terminate with real subprocess pipes."""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from local_lm import video_output_measurement as measurement
from local_lm.output_measurement import Budget


@pytest.mark.asyncio
@pytest.mark.parametrize("descriptor", [1, 2])
async def test_a_real_flooding_decoder_finishes_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, descriptor: int
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda name: sys.executable)
    launch = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []

    async def flooding(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await launch(
            sys.executable,
            "-c",
            "import os; chunk = b'x' * 65536; "
            + f"[os.write({descriptor}, chunk) for _ in range(2048)]",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", flooding)
    task = asyncio.create_task(measurement.measure_video_output(b"neutral-video-fixture", Budget()))
    done, _ = await asyncio.wait({task}, timeout=5)
    finished_without_help = bool(done)
    if not finished_without_help:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await asyncio.gather(
                *(
                    stream.read()
                    for stream in (process.stdout, process.stderr)
                    if stream is not None
                )
            )
        await asyncio.wait_for(task, timeout=5)
    assert finished_without_help, (
        "The stopped decoder still needs its pipes drained before cleanup finishes"
    )
    assert task.result()["reason"] == "over_output_budget"
    assert all(process.returncode is not None for process in processes)
    assert not list(tmp_path.glob("lm-atelier-measure-*"))
