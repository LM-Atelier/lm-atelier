"""However a media tool's call ends, the tool and what it started have exited before it returns.

Each case starts the real ffmpeg that hosted CI installs, generating a test
pattern for as long as it is left to run, and checks what the operating system
reported once the call was over: the process has exited and both of its pipes
have reached their end.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import shutil
import sys
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anyio
import psutil
import pytest

from local_lm.media_process import ToolOutputTooLarge, run_tool

_SILENT = ["-hide_banner", "-nostdin", "-loglevel", "error"]
_PATTERN = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10"]
# Writes nothing for a minute, making a pattern in real time and throwing it
# away. Every case stops it long before then; the minute bounds a regression.
QUIET = [*_SILENT, "-re", "-t", "60", *_PATTERN, "-f", "null", "-"]
# Writes raw frames to its output as fast as it can.
CHATTY = [*_SILENT, *_PATTERN, "-f", "rawvideo", "pipe:1"]
# Like QUIET, for two minutes: far longer than any case lets a call take.
LONG = [*_SILENT, "-re", "-t", "120", *_PATTERN, "-f", "null", "-"]
LIMIT = 1024
NO_PIPES = "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0


def _ffmpeg() -> Path:
    executable = shutil.which("ffmpeg")
    assert executable, "these cases need a real ffmpeg on PATH, as hosted CI provides"
    return Path(executable)


class Started:
    """Every process the runner starts, with hooks for when it starts and when it is killed."""

    def __init__(self) -> None:
        self.processes: list[asyncio.subprocess.Process] = []
        self.on_start: Callable[[], object] | None = None
        self.on_kill: Callable[[], object] | None = None


@pytest.fixture
async def started(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Started]:
    record = Started()
    original = asyncio.create_subprocess_exec

    async def recording(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        if record.on_start is not None:
            record.on_start()
        process = await original(*args, **kwargs)
        kill = process.kill

        def observed_kill() -> None:
            kill()
            if record.on_kill is not None:
                record.on_kill()

        monkeypatch.setattr(process, "kill", observed_kill)
        record.processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", recording)
    yield record
    # A case that fails must still not leave a tool running behind it.
    record.on_kill = None
    for process in record.processes:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(TimeoutError, RuntimeError):
                async with asyncio.timeout(10):
                    await asyncio.gather(_drain(process.stdout), _drain(process.stderr))
                    await process.wait()


async def _drain(stream: asyncio.StreamReader | None) -> None:
    while stream is not None and await stream.read(65536):
        pass


async def _running(started: Started) -> None:
    """Wait until the tool has started and had time to start anything of its own."""

    async with asyncio.timeout(20):
        while not started.processes:
            await asyncio.sleep(0.01)
    await asyncio.sleep(0.5)


def _has_exited(started: Started) -> None:
    [process] = started.processes
    assert process.returncode is not None, "the tool was still running when the call returned"
    assert process.stdout is not None and process.stdout.at_eof()
    assert process.stderr is not None and process.stderr.at_eof()


class _Watched:
    """A process opened while it runs, so that whether it has exited can be asked at any moment.

    On Windows a process that is being ended is still running until its handle
    is signalled, which a handle taken beforehand can ask without waiting. On
    POSIX a process that has exited is a zombie until it is collected, and
    holds nothing open meanwhile.
    """

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._handle: Any = None
        if sys.platform == "win32":
            self._handle = _kernel().OpenProcess(_SYNCHRONIZE, False, pid)
            assert self._handle, "the program had gone before it could be watched"

    def exited(self) -> bool:
        """Whether the process has exited now, asked without waiting."""

        if sys.platform == "win32":
            return bool(_kernel().WaitForSingleObject(self._handle, 0) == _WAIT_OBJECT_0)
        try:
            return bool(psutil.Process(self.pid).status() == psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            return True

    def close(self) -> None:
        if self._handle:
            _kernel().CloseHandle(self._handle)
            self._handle = None


def _kernel() -> Any:
    windows: Any = ctypes
    kernel = windows.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    return kernel


async def _watch(record: Path) -> _Watched:
    """The program a launcher recorded, watched from as soon as its identifier is written."""

    async with asyncio.timeout(20):
        while not record.exists() or not record.read_text():
            await asyncio.sleep(0.01)
    return _Watched(int(record.read_text()))


def _launcher(record: Path, streams: str, go: Path | None = None) -> list[str]:
    """Arguments for a Python program that starts ffmpeg, records its id and exits.

    With ``go`` it exits only once that file exists, so a case can watch the
    program before the call can end.
    """

    lines = [
        "import os, subprocess, sys, time",
        f"program = subprocess.Popen([{str(_ffmpeg())!r}, *{LONG!r}], {streams})",
        f"open({str(record)!r}, 'w').write(str(program.pid))",
    ]
    if go is not None:
        lines.append(f"while not os.path.exists({str(go)!r}): time.sleep(0.01)")
    return ["-c", "\n".join(lines)]


def _call(arguments: list[str], seconds: float = 30) -> asyncio.Task[Any]:
    return asyncio.create_task(
        run_tool(_ffmpeg(), arguments, seconds=seconds, stdout_limit=LIMIT, stderr_limit=LIMIT)
    )


async def test_a_tool_that_runs_too_long_has_exited_when_the_call_times_out(
    started: Started,
) -> None:
    with pytest.raises(TimeoutError):
        await _call(QUIET, seconds=0.5)

    _has_exited(started)


async def test_a_tool_that_writes_too_much_has_exited_when_the_call_refuses_it(
    started: Started,
) -> None:
    with pytest.raises(ToolOutputTooLarge):
        await _call(CHATTY)

    _has_exited(started)


async def test_a_cancelled_call_returns_only_once_its_tool_has_exited(started: Started) -> None:
    call = _call(QUIET)
    await _running(started)

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    _has_exited(started)


@pytest.mark.parametrize("ending", ["timeout", "too-much-output", "cancelled"])
async def test_stopping_a_tool_finishes_however_often_the_call_is_cancelled_meanwhile(
    started: Started, ending: str
) -> None:
    call = _call(
        CHATTY if ending == "too-much-output" else QUIET, 0.5 if ending == "timeout" else 30
    )
    loop = asyncio.get_running_loop()
    cancelled_while_stopping = 0

    def cancel_again() -> None:
        nonlocal cancelled_while_stopping
        if not call.done():
            call.cancel()
            cancelled_while_stopping += 1
            loop.call_soon(cancel_again)

    # From the moment the tool is killed until the call returns, every turn of
    # the event loop cancels the call once more; a cancelled call is cancelled
    # again from its first cancellation on, before the kill as well as after.
    started.on_kill = lambda: loop.call_soon(cancel_again)
    if ending == "cancelled":
        await _running(started)
        call.cancel()
        loop.call_soon(cancel_again)

    with pytest.raises(asyncio.CancelledError):
        await call

    assert cancelled_while_stopping >= 2
    _has_exited(started)


async def test_a_call_cancelled_while_its_tool_starts_still_stops_that_tool(
    started: Started,
) -> None:
    call = _call(QUIET)
    started.on_start = call.cancel

    with pytest.raises(asyncio.CancelledError):
        await call

    _has_exited(started)


async def test_a_launcher_is_stopped_together_with_the_program_it_runs(
    started: Started,
) -> None:
    # Shaped like a package manager's shim: the program started runs the real
    # one as its own child, which writes to the same pipes and outlives it.
    launcher = (
        "import subprocess, sys; "
        f"subprocess.run([{str(_ffmpeg())!r}, *{LONG!r}], stdout=sys.stdout, stderr=sys.stderr)"
    )
    began = time.monotonic()

    with pytest.raises(TimeoutError):
        await run_tool(
            Path(sys.executable),
            ["-c", launcher],
            seconds=2,
            stdout_limit=LIMIT,
            stderr_limit=LIMIT,
        )

    # Left running, the launched program would hold both pipes for two minutes
    # and the call would wait for it, so the call's length shows whether it was stopped.
    assert time.monotonic() - began < 30
    _has_exited(started)


async def test_a_program_left_running_by_a_launcher_that_exited_is_stopped_too(
    started: Started, tmp_path: Path
) -> None:
    # The program keeps both of the launcher's pipes, and the launcher does not wait for it.
    record = tmp_path / "program.pid"
    began = time.monotonic()
    call = asyncio.create_task(
        run_tool(
            Path(sys.executable),
            _launcher(record, "stdout=sys.stdout, stderr=sys.stderr"),
            seconds=2,
            stdout_limit=LIMIT,
            stderr_limit=LIMIT,
        )
    )
    program = await _watch(record)
    try:
        with pytest.raises(TimeoutError):
            await call
        # Asked as the call returns: it has exited, not merely been told to.
        assert program.exited(), "the program was still running when the call returned"
    finally:
        program.close()

    # Left running, it would hold both pipes for two minutes and the call would wait for it.
    assert time.monotonic() - began < 30
    _has_exited(started)


async def test_a_program_a_tool_leaves_running_has_exited_when_the_call_returns(
    started: Started, tmp_path: Path
) -> None:
    # Started on no pipes at all, so the call ends normally while it runs on.
    record, go = tmp_path / "program.pid", tmp_path / "go"
    call = asyncio.create_task(
        run_tool(
            Path(sys.executable),
            _launcher(record, NO_PIPES, go),
            seconds=30,
            stdout_limit=LIMIT,
            stderr_limit=LIMIT,
        )
    )
    program = await _watch(record)
    try:
        go.write_text("")
        result = await call
        # Asked as the call returns: it has exited, not merely been told to.
        assert program.exited(), "the program was still running when the call returned"
    finally:
        program.close()

    assert result.returncode == 0


async def test_a_cancel_scope_around_a_call_ends_the_scope_and_not_the_task(
    started: Started,
) -> None:
    # A cancel scope cancels again on every turn until its task leaves it, and
    # knows its own cancellation only by the error it raised.
    with anyio.move_on_after(1) as scope:
        await run_tool(_ffmpeg(), QUIET, seconds=30, stdout_limit=LIMIT, stderr_limit=LIMIT)

    assert scope.cancelled_caught
    _has_exited(started)
