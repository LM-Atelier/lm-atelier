"""Say what a worker failure was, and what to do about it.

A worker that dies currently reports an exit code and a dozen lines of engine
output. That is enough for someone who already knows how llama.cpp and ComfyUI
fail, and useless to everyone else - which is most people, on their first
attempt, on the machine where it matters. Out of memory is the single most
common way a local model fails to start, and nothing in the backend recognised
it at all.

So the same output gets read once more, here, and turned into a code and a
sentence a person can act on. The classifier is deliberately conservative:
patterns come from strings these engines actually print, an unrecognised failure
stays `unknown` and shows the raw output as before, and no remedy ever claims to
know something it cannot see. A confident wrong remedy is worse than none - it
sends someone to change a setting that was never the problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

MAX_CLASSIFIED_CHARACTERS = 32_768
# 137 is 128+SIGKILL, which on Linux is overwhelmingly the kernel OOM killer.
# 134 is 128+SIGABRT, the usual end of an uncaught std::bad_alloc.
_HOST_OOM_EXIT_CODES = frozenset({137, 134})


class WorkerFailureCode(StrEnum):
    OOM_VRAM = "oom_vram"
    OOM_HOST = "oom_host"
    PORT_IN_USE = "port_in_use"
    MODEL_INCOMPATIBLE = "model_incompatible"
    EXECUTABLE_MISSING = "executable_missing"
    STARTUP_TIMEOUT = "startup_timeout"
    CRASHED = "crashed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorkerFailure:
    code: WorkerFailureCode
    remedy: str | None


_REMEDIES: dict[WorkerFailureCode, str] = {
    WorkerFailureCode.OOM_VRAM: (
        "The graphics card ran out of memory. Choose a smaller model or a more "
        "compressed version of this one, lower the image size or step count for "
        "image and video work, and close other programs using the GPU. If this "
        "model has an offload or layer setting, moving fewer layers onto the "
        "card also fits it in less memory."
    ),
    WorkerFailureCode.OOM_HOST: (
        "The computer ran out of system memory. Close other programs and try "
        "again, or choose a smaller model. On Windows, increasing the paging "
        "file size also helps; models are loaded through it."
    ),
    WorkerFailureCode.PORT_IN_USE: (
        "Another program is already using the port this worker needs. That is "
        "usually a copy of the engine left running from an earlier session - "
        "restart LM Atelier, or end the leftover process, then try again."
    ),
    WorkerFailureCode.MODEL_INCOMPATIBLE: (
        "The engine could not read this model file. It may be an unsupported "
        "format or architecture, built for a newer engine than the installed "
        "one, or incompletely downloaded. Reinstall the model, and if it still "
        "fails, choose one listed as supported."
    ),
    WorkerFailureCode.EXECUTABLE_MISSING: (
        "The engine program could not be found or could not start. Reinstall the "
        "runtime from setup. If you configured an engine path yourself, check "
        "that it still points at the program."
    ),
    WorkerFailureCode.STARTUP_TIMEOUT: (
        "The worker did not finish starting in the time allowed. Large models "
        "read from a slow disk can take several minutes the first time. Try "
        "again - a second start is usually much faster - and if it keeps timing "
        "out, choose a smaller model."
    ),
    WorkerFailureCode.CRASHED: (
        "The worker stopped unexpectedly. Try starting it again. If it stops "
        "every time, the output below is what the engine reported last."
    ),
}

# Order is precedence. A run that exhausts VRAM often prints a generic load
# failure afterwards, so the specific causes have to be tested before the
# general ones.
_PATTERNS: tuple[tuple[WorkerFailureCode, re.Pattern[str]], ...] = (
    (
        WorkerFailureCode.OOM_VRAM,
        re.compile(
            r"cuda out of memory"
            r"|outofmemoryerror"
            r"|out of memory error"
            r"|hip(?:error)?outofmemory"
            r"|mps backend out of memory"
            r"|allocation on device"
            r"|failed to allocate (?:cuda|device|vram|gpu)"
            r"|cuda(?:malloc| error)[^\n]{0,80}out of memory"
            r"|ggml_backend_[a-z_]*(?:cuda|vulkan|metal|sycl)[a-z_]*[^\n]{0,120}"
            r"(?:allocating|alloc)[^\n]{0,80}failed"
            r"|unable to allocate[^\n]{0,60}(?:on (?:the )?(?:gpu|device)|vram)"
            r"|insufficient (?:gpu|device|video) memory"
            r"|dxgi_error_device_removed",
            re.IGNORECASE,
        ),
    ),
    (
        WorkerFailureCode.OOM_HOST,
        re.compile(
            r"std::bad_alloc"
            r"|\bmemoryerror\b"
            r"|cannot allocate memory"
            r"|out of memory: killed process"
            r"|killed process \d+"
            r"|the paging file is too small"
            r"|not enough memory resources are available"
            r"|insufficient (?:system |host )?memory"
            r"|failed to allocate (?:host |cpu |system )?(?:buffer|memory)",
            re.IGNORECASE,
        ),
    ),
    (
        WorkerFailureCode.PORT_IN_USE,
        re.compile(
            r"address already in use"
            r"|only one usage of each socket address"
            r"|wsaeaddrinuse"
            r"|\[errno 98\]"
            r"|winerror 10048"
            r"|port \d+ is (?:already )?in use"
            r"|is already in use by another program",
            re.IGNORECASE,
        ),
    ),
    (
        WorkerFailureCode.MODEL_INCOMPATIBLE,
        re.compile(
            r"unknown model architecture"
            r"|unsupported model architecture"
            r"|unsupported architecture"
            r"|unknown architecture"
            r"|error loading model"
            r"|failed to load model"
            r"|failed to load the model"
            r"|unable to load model"
            r"|invalid magic (?:number|character)"
            r"|unsupported gguf version"
            r"|wrong number of tensors"
            r"|missing tensor"
            r"|tensor[^\n]{0,60}not found"
            r"|does not appear to have a file named"
            r"|not a valid (?:model|checkpoint|safetensors) file",
            re.IGNORECASE,
        ),
    ),
    (
        WorkerFailureCode.EXECUTABLE_MISSING,
        re.compile(
            r"no such file or directory"
            r"|is not recognized as an internal or external command"
            r"|the system cannot find the (?:file|path) specified"
            r"|winerror 2\b"
            r"|winerror 3\b"
            r"|executable (?:was )?not found"
            r"|could not find the [a-z. ]*executable"
            r"|permission denied",
            re.IGNORECASE,
        ),
    ),
    (
        WorkerFailureCode.STARTUP_TIMEOUT,
        re.compile(
            r"did not become (?:healthy|ready)"
            r"|startup timed out"
            r"|timed out (?:waiting|starting)"
            r"|worker startup timeout",
            re.IGNORECASE,
        ),
    ),
)


def classify_worker_failure(
    *,
    failure_detail: str | None,
    stderr_tail: str | None,
    exit_code: int | None,
) -> WorkerFailure:
    """Name a worker failure from what the engine and the supervisor reported.

    `failure_detail` is the supervisor's own message and is checked first: it
    describes what we were doing when it failed, which is more reliable than
    engine output for timeouts and missing executables. Engine output decides
    the rest, because only the engine knows it ran out of memory.
    """
    detail = (failure_detail or "")[:MAX_CLASSIFIED_CHARACTERS]
    tail = (stderr_tail or "")[:MAX_CLASSIFIED_CHARACTERS]
    if not detail and not tail and exit_code is None:
        return WorkerFailure(WorkerFailureCode.UNKNOWN, None)

    for code, pattern in _PATTERNS:
        if pattern.search(detail) or pattern.search(tail):
            return _failure(code)

    # An exit code alone can still name a host OOM: the kernel's OOM killer
    # leaves nothing in the worker's own output, because it never got to write
    # any. This is checked after the patterns so a process killed while already
    # reporting a VRAM failure is not relabelled.
    if exit_code in _HOST_OOM_EXIT_CODES:
        return _failure(WorkerFailureCode.OOM_HOST)
    if exit_code is not None and exit_code != 0:
        return _failure(WorkerFailureCode.CRASHED)
    return WorkerFailure(WorkerFailureCode.UNKNOWN, None)


def _failure(code: WorkerFailureCode) -> WorkerFailure:
    return WorkerFailure(code, _REMEDIES.get(code))
