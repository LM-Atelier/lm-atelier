"""Run one media tool once, without a shell, with its time and both outputs bounded.

The tool gets an argument list, never a command line, an environment without
the parent's credentials, and no standard input. Everything it starts is held
together with it: a process group of its own on POSIX, a job object on
Windows. A timeout, an output past its bound or the caller's cancellation kills
that whole group, even when the tool itself has already exited, drains what was
written and waits until the tool has exited and both of its pipes have closed,
however often the call is cancelled meanwhile. When the call ends, anything the
tool left running is ended too, and the call returns only once every process
the tool started has exited, so nothing it started outlives the call.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import functools
import os
import signal
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import psutil

from .subprocess_env import subprocess_environment

_READ: Final = 64 * 1024
_CREATE_SUSPENDED: Final = 0x00000004
_PROCESS_TERMINATE: Final = 0x0001
_PROCESS_SET_QUOTA: Final = 0x0100
_PROCESS_QUERY_LIMITED: Final = 0x1000
_SYNCHRONIZE: Final = 0x00100000
_JOB_ACTIVE_PROCESS: Final = 0x00000008
_JOB_KILL_ON_CLOSE: Final = 0x00002000
_JOB_EXTENDED_LIMITS: Final = 9
_JOB_PROCESS_IDS: Final = 3
_ERROR_MORE_DATA: Final = 234
_WAIT_FOREVER: Final = 0xFFFFFFFF
#: How often a POSIX group is checked for members still running after the kill.
_GROUP_POLL_SECONDS: Final = 0.01


class ToolOutputTooLarge(Exception):
    """A media tool wrote more than its bound to one of its outputs."""


@dataclass(frozen=True)
class ToolResult:
    returncode: int
    stdout: bytes
    stderr: bytes


async def run_tool(
    executable: Path,
    arguments: Sequence[str],
    *,
    seconds: float,
    stdout_limit: int,
    stderr_limit: int,
) -> ToolResult:
    """Run the tool to its end, or stop it and raise TimeoutError or ToolOutputTooLarge."""

    held = _HeldProcesses()
    try:
        return await _run(executable, arguments, held, seconds, stdout_limit, stderr_limit)
    finally:
        # The call returns only once nothing the tool started is still running,
        # since the caller may then remove a file one of those processes reads.
        await outlast_cancellation(asyncio.ensure_future(held.release()))


async def _run(
    executable: Path,
    arguments: Sequence[str],
    held: _HeldProcesses,
    seconds: float,
    stdout_limit: int,
    stderr_limit: int,
) -> ToolResult:
    # A tool keeps starting once asked to, even when the call is cancelled
    # meanwhile, so the start finishes first and a cancelled call then stops it.
    starting = asyncio.ensure_future(
        asyncio.create_subprocess_exec(
            str(executable),
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=subprocess_environment(),
            **held.creation(),
        )
    )
    try:
        process = await outlast_cancellation(starting)
    except asyncio.CancelledError:
        if not starting.cancelled() and starting.exception() is None:
            held.track(starting.result())
            await _stop(starting.result(), [], held)
        raise
    held.track(process)
    try:
        held.hold(process)
    except (OSError, psutil.Error) as exc:
        # Not let run unheld: stopped as it is, and the call fails to start it.
        await _stop(process, [], held)
        raise OSError("the tool could not be held with what it starts") from exc
    readers: list[asyncio.Task[bytes]] = []
    try:
        async with asyncio.timeout(seconds):
            readers = [
                asyncio.create_task(_bounded(process.stdout, stdout_limit)),
                asyncio.create_task(_bounded(process.stderr, stderr_limit)),
            ]
            stdout, stderr = await asyncio.gather(*readers)
            returncode = await process.wait()
    except BaseException:
        await _stop(process, readers, held)
        raise
    return ToolResult(returncode, stdout, stderr)


class _HeldProcesses:
    """Everything one run of a tool starts, held so that all of it can be ended together.

    On POSIX the tool starts a session of its own, so it and whatever it starts
    share one process group, which a signal still reaches after the tool itself
    has exited. On Windows the tool starts suspended, is put in a job object
    that ends every process in it when it is closed, and only then runs, so
    nothing it starts is outside the job.
    """

    def __init__(self) -> None:
        self._pid: int | None = None
        self._job: Any = None
        #: Handles to the processes a stop ended, kept to wait for when the hold is let go.
        self._ended: list[Any] = []

    @staticmethod
    def creation() -> dict[str, Any]:
        if sys.platform == "win32":
            return {"creationflags": _CREATE_SUSPENDED}
        return {"start_new_session": True}

    def track(self, process: asyncio.subprocess.Process) -> None:
        self._pid = process.pid

    def hold(self, process: asyncio.subprocess.Process) -> None:
        if sys.platform == "win32":
            self._job = _windows_job(process.pid)
            psutil.Process(process.pid).resume()

    def end(self) -> None:
        """End every process the tool started, wherever it is, the tool included."""

        if sys.platform == "win32":
            if self._job is not None:
                kernel = _windows_api()
                # Taken before the job is ended, since an ending process leaves
                # its listing. Nothing here may keep the job from being ended.
                with contextlib.suppress(OSError):
                    _admit_no_more(kernel, self._job)
                    self._ended += _member_handles(kernel, self._job)
                kernel.TerminateJobObject(self._job, 1)
        elif self._pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self._pid, signal.SIGKILL)

    async def release(self) -> None:
        """End whatever the tool left running and wait until every one of them has exited.

        Ending a process only starts its exit: the call that ends it returns
        while the process may still be running and holding what it opened. So
        the hold is let go only once each process in it has finished exiting.
        As with stopping the tool, the wait has no time limit, because every
        process waited for has been killed.
        """

        if sys.platform == "win32":
            job, self._job = self._job, None
            ended, self._ended = self._ended, []
            if job is not None:
                try:
                    await asyncio.to_thread(_end_job, job, ended)
                finally:
                    for handle in ended:
                        _windows_api().CloseHandle(handle)
                    _windows_api().CloseHandle(job)
        else:
            group, self._pid = self._pid, None
            while group is not None:
                # Killed again on every pass, so nothing that joined the group
                # while it was being ended is waited for without being killed.
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(group, signal.SIGKILL)
                if not _group_runs(group):
                    break
                await asyncio.sleep(_GROUP_POLL_SECONDS)


async def outlast_cancellation[T](
    work: asyncio.Future[T], *, on_cancel: Callable[[], None] | None = None
) -> T:
    """Wait for work to finish even when the caller is cancelled meanwhile, then honour that.

    A cancellation lands at the caller's next await, so cleanup written as a
    sequence of awaits stops wherever the first one lands. The work runs as its
    own task, which cancelling the caller does not reach, and the caller keeps
    waiting for it however many times it is cancelled. ``on_cancel`` runs once,
    at the first cancellation, so work that can end early is told to. Once the
    work is done, a caller that was cancelled raises the first cancellation it
    received, so whatever cancelled it (a timeout, a task group, a cancel scope)
    still recognises its own, with any failure of the work as its cause.
    """

    cancellation: asyncio.CancelledError | None = None
    while not work.done():
        try:
            # Unlike awaiting the work itself, this leaves it running when the
            # caller is cancelled, and never raises the work's own failure.
            await asyncio.wait([work])
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
                if on_cancel is not None:
                    on_cancel()
    if cancellation is not None:
        if not work.cancelled() and (failure := work.exception()) is not None:
            cancellation.__cause__ = failure
        raise cancellation
    return work.result()


async def _stop(
    process: asyncio.subprocess.Process,
    readers: list[asyncio.Task[bytes]],
    held: _HeldProcesses,
) -> None:
    """Kill a tool and all it started, then wait until it has exited and both pipes have closed.

    The kill comes before anything is awaited, because an await is where a
    further cancellation lands. The wait has no time limit: everything holding
    the pipes has been killed, and returning before they closed would leave
    whatever holds them running.
    """

    for reader in readers:
        reader.cancel()
    _kill(process, held)
    await outlast_cancellation(asyncio.ensure_future(_join(process, readers)))


def _kill(process: asyncio.subprocess.Process, held: _HeldProcesses) -> None:
    """Kill the tool and every process it started, including those it left behind on exiting.

    A program found on the system path can be a launcher that runs the real
    tool as its own child, as package managers' shims do, and may even exit
    before that child does. The child holds the same pipes, so the whole group
    or job is ended, whether or not the launcher is still running. Children a
    still-running tool has moved out of its group are found as its descendants
    too; the tool is held still while they are listed.
    """

    started: list[psutil.Process] = []
    if process.returncode is None:
        with contextlib.suppress(psutil.Error):
            tool = psutil.Process(process.pid)
            # Only a process that is still this one's own child is the tool; its
            # identifier could otherwise have passed to an unrelated program.
            if tool.ppid() == os.getpid():
                with contextlib.suppress(psutil.Error):
                    tool.suspend()
                started = tool.children(recursive=True)
    held.end()
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    for descendant in started:
        with contextlib.suppress(psutil.Error):
            descendant.kill()


async def _join(process: asyncio.subprocess.Process, readers: list[asyncio.Task[bytes]]) -> None:
    """Wait for a killed tool, however much it had already written.

    asyncio reports a process finished only once both of its pipes have closed,
    and a pipe whose reader stopped early never reaches its end. So the
    cancelled readers are finished first, since a stream allows one reader at a
    time, and both pipes are then read to their end and discarded.
    """

    await asyncio.gather(*readers, return_exceptions=True)
    await asyncio.gather(_discard(process.stdout), _discard(process.stderr))
    await process.wait()


async def _discard(stream: asyncio.StreamReader | None) -> None:
    if stream is None:
        return
    while await stream.read(_READ):
        pass


def _group_runs(group: int) -> bool:
    """Whether any process in the POSIX process group ``group`` has yet to finish exiting.

    A process that has exited stays in its group until its parent collects it,
    holding nothing open meanwhile, so only one that has not exited counts.
    """

    if sys.platform == "win32":
        return False
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    for process in psutil.process_iter():
        with contextlib.suppress(psutil.Error, OSError):
            if os.getpgid(process.pid) == group and process.status() != psutil.STATUS_ZOMBIE:
                return True
    return False


class _BasicLimits(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    )


class _IoCounters(ctypes.Structure):
    _fields_ = tuple(
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    )


class _ExtendedLimits(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


@functools.cache
def _windows_api() -> Any:
    windows: Any = ctypes
    kernel = windows.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    kernel.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel.SetInformationJobObject.restype = ctypes.c_int
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.AssignProcessToJobObject.restype = ctypes.c_int
    kernel.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.TerminateJobObject.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    kernel.QueryInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel.QueryInformationJobObject.restype = ctypes.c_int
    kernel.IsProcessInJob.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    kernel.IsProcessInJob.restype = ctypes.c_int
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    return kernel


def _windows_job(pid: int) -> Any:
    """A job that ends its processes when closed, holding the suspended process ``pid``."""

    kernel = _windows_api()
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise OSError("a job for the tool could not be created")
    try:
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _JOB_KILL_ON_CLOSE
        if not kernel.SetInformationJobObject(
            job, _JOB_EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise OSError("the tool's job could not be set to end its processes")
        handle = kernel.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid)
        if not handle:
            raise OSError("the tool could not be opened to put it in its job")
        try:
            if not kernel.AssignProcessToJobObject(job, handle):
                raise OSError("the tool could not be put in its job")
        finally:
            kernel.CloseHandle(handle)
    except BaseException:
        kernel.CloseHandle(job)
        raise
    return job


def _end_job(job: Any, ended: list[Any]) -> None:
    """End every process in the job, then wait until each one has finished exiting.

    A process being ended leaves the job's listing as soon as its ending
    starts, well before it has exited, so each one is opened while it is still
    listed and waited for through that handle once the job is ended; ``ended``
    holds those a stop opened before ending them. The job first stops
    admitting processes, so none can join after the listing is taken and be
    ended without being waited for.
    """

    kernel = _windows_api()
    for handle in ended:
        kernel.WaitForSingleObject(handle, _WAIT_FOREVER)
    _admit_no_more(kernel, job)
    while handles := _member_handles(kernel, job):
        try:
            kernel.TerminateJobObject(job, 1)
            for handle in handles:
                kernel.WaitForSingleObject(handle, _WAIT_FOREVER)
        finally:
            for handle in handles:
                kernel.CloseHandle(handle)
    kernel.TerminateJobObject(job, 1)


def _admit_no_more(kernel: Any, job: Any) -> None:
    """Refuse every further process in the job; those already in it run on until ended."""

    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = _JOB_KILL_ON_CLOSE | _JOB_ACTIVE_PROCESS
    # Lower than the job already holds: only a process joining it is refused.
    limits.BasicLimitInformation.ActiveProcessLimit = 1
    kernel.SetInformationJobObject(
        job, _JOB_EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)
    )


def _member_handles(kernel: Any, job: Any) -> list[Any]:
    """A handle to each process in the job now, each checked to be in this job."""

    handles: list[Any] = []
    for pid in _job_members(job):
        handle = kernel.OpenProcess(_SYNCHRONIZE | _PROCESS_QUERY_LIMITED, False, pid)
        if not handle:
            # Gone between the listing and the opening.
            continue
        inside = ctypes.c_int(0)
        # An identifier can pass to an unrelated process once the one listed has gone.
        if kernel.IsProcessInJob(handle, job, ctypes.byref(inside)) and inside.value:
            handles.append(handle)
        else:
            kernel.CloseHandle(handle)
    return handles


@functools.cache
def _id_listing(capacity: int) -> type[ctypes.Structure]:
    """The job's process identifier listing, with room for ``capacity`` identifiers."""

    return type(
        "_JobProcessIds",
        (ctypes.Structure,),
        {
            "_fields_": (
                ("Assigned", ctypes.c_uint32),
                ("Listed", ctypes.c_uint32),
                ("Ids", ctypes.c_size_t * capacity),
            )
        },
    )


def _job_members(job: Any) -> list[int]:
    """The identifiers of the processes in the job now."""

    kernel = _windows_api()
    capacity = 64
    while True:
        listing: Any = _id_listing(capacity)()
        if kernel.QueryInformationJobObject(
            job, _JOB_PROCESS_IDS, ctypes.byref(listing), ctypes.sizeof(listing), None
        ):
            return [int(listing.Ids[index]) for index in range(listing.Listed)]
        windows: Any = ctypes
        if windows.get_last_error() != _ERROR_MORE_DATA:
            raise OSError("the processes in the tool's job could not be listed")
        capacity *= 4


async def _bounded(stream: asyncio.StreamReader | None, limit: int) -> bytes:
    if stream is None:
        return b""
    chunks: list[bytes] = []
    size = 0
    while chunk := await stream.read(_READ):
        size += len(chunk)
        if size > limit:
            raise ToolOutputTooLarge
        chunks.append(chunk)
    return b"".join(chunks)
