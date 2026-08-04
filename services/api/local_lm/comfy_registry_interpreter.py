from __future__ import annotations

import asyncio
import json
import os
import stat
from contextlib import suppress
from pathlib import Path
from typing import Any

from .comfy_registry_runtime import (
    ComfyRegistryRuntimeDistribution,
    ComfyRegistryRuntimeError,
    canonical_comfy_registry_runtime_distributions,
)
from .comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactError,
    comfy_registry_wheel_target_sha256,
)
from .processes import WINDOWS_CREATE_NO_WINDOW
from .subprocess_env import subprocess_environment

INTERPRETER_PROBE_TIMEOUT_SECONDS = 15
MAX_INTERPRETER_PROBE_OUTPUT_BYTES = 1024 * 1024
MAX_INTERPRETER_PROBE_ERROR_BYTES = 64 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_PROBE_PROGRAM = """
import json
from importlib.metadata import distributions
from packaging.markers import default_environment
from packaging.tags import sys_tags

environment = {key: str(value) for key, value in default_environment().items()}
environment["extra"] = ""
print(json.dumps(
    {
        "marker_environment": environment,
        "supported_tags": [str(tag) for tag in sys_tags()],
        "runtime_distributions": [
            {"name": item.metadata["Name"], "version": item.version}
            for item in distributions()
            if item.metadata["Name"] and item.version
        ],
    },
    ensure_ascii=True,
    separators=(",", ":"),
))
""".strip()


class ComfyRegistryInterpreterError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _ProbeOutputTooLarge(Exception):
    pass


async def probe_comfy_registry_runtime_target(
    python_executable: Path,
) -> tuple[
    dict[str, str],
    tuple[str, ...],
    tuple[ComfyRegistryRuntimeDistribution, ...],
]:
    """Read wheel markers and tags from the exact managed ComfyUI interpreter."""

    executable = _python_executable(python_executable)
    output = await _run_probe(executable)
    try:
        payload = json.loads(output.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComfyRegistryInterpreterError(
            "interpreter_probe_invalid_output",
            "The managed runtime returned invalid wheel-target data.",
        ) from exc
    if not isinstance(payload, dict) or frozenset(payload) not in {
        frozenset({"marker_environment", "supported_tags"}),
        frozenset({"marker_environment", "supported_tags", "runtime_distributions"}),
    }:
        raise ComfyRegistryInterpreterError(
            "interpreter_probe_invalid_output",
            "The managed runtime returned invalid wheel-target data.",
        )
    environment_value = payload["marker_environment"]
    tags_value = payload["supported_tags"]
    distributions_value = payload.get("runtime_distributions", [])
    if (
        not isinstance(environment_value, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment_value.items()
        )
        or not isinstance(tags_value, list)
        or any(not isinstance(value, str) for value in tags_value)
        or not isinstance(distributions_value, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"name", "version"}
            or not isinstance(item["name"], str)
            or not isinstance(item["version"], str)
            for item in distributions_value
        )
    ):
        raise ComfyRegistryInterpreterError(
            "interpreter_probe_invalid_output",
            "The managed runtime returned invalid wheel-target data.",
        )
    environment = dict(environment_value)
    environment.setdefault("extra", "")
    tags = tuple(tags_value)
    try:
        distributions = canonical_comfy_registry_runtime_distributions(
            tuple(
                ComfyRegistryRuntimeDistribution(item["name"], item["version"])
                for item in distributions_value
            )
        )
    except ComfyRegistryRuntimeError as exc:
        raise ComfyRegistryInterpreterError(
            exc.code,
            "The managed runtime returned invalid distribution data.",
        ) from exc
    try:
        comfy_registry_wheel_target_sha256(environment, tags)
    except ComfyRegistryWheelArtifactError as exc:
        raise ComfyRegistryInterpreterError(
            "interpreter_probe_invalid_target",
            "The managed runtime returned an invalid wheel target.",
        ) from exc
    return environment, tags, distributions


async def probe_comfy_registry_wheel_target(
    python_executable: Path,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Read wheel markers and tags while preserving the original probe contract."""
    environment, tags, _ = await probe_comfy_registry_runtime_target(python_executable)
    return environment, tags


def _python_executable(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ComfyRegistryInterpreterError(
            "invalid_python_executable",
            "Managed Python executable is invalid.",
        )
    try:
        entry = value.parent.resolve(strict=True) / value.name
        resolved = entry.resolve(strict=True)
    except OSError as exc:
        raise ComfyRegistryInterpreterError(
            "invalid_python_executable",
            "Managed Python executable is missing.",
        ) from exc
    if (
        (os.name == "nt" and _is_link_or_reparse(entry))
        or not resolved.is_file()
        or _is_link_or_reparse(resolved)
    ):
        raise ComfyRegistryInterpreterError(
            "invalid_python_executable",
            "Managed Python executable is invalid.",
        )
    return entry


async def _run_probe(executable: Path) -> bytes:
    try:
        process = await asyncio.create_subprocess_exec(
            str(executable),
            "-I",
            "-c",
            _PROBE_PROGRAM,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=executable.parent,
            env=subprocess_environment(overrides={"PYTHONNOUSERSITE": "1"}),
            creationflags=WINDOWS_CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except OSError as exc:
        raise ComfyRegistryInterpreterError(
            "interpreter_probe_failed",
            "The managed runtime could not be started for package inspection.",
        ) from exc
    if process.stdout is None or process.stderr is None:
        await _terminate(process)
        raise ComfyRegistryInterpreterError(
            "interpreter_probe_failed",
            "The managed runtime could not be inspected.",
        )

    stdout_task = asyncio.create_task(
        _read_bounded(process.stdout, MAX_INTERPRETER_PROBE_OUTPUT_BYTES)
    )
    stderr_task = asyncio.create_task(
        _read_bounded(process.stderr, MAX_INTERPRETER_PROBE_ERROR_BYTES)
    )
    wait_task = asyncio.create_task(process.wait())
    tasks: tuple[asyncio.Task[Any], ...] = (stdout_task, stderr_task, wait_task)
    try:
        stdout, _, returncode = await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task, wait_task),
            timeout=INTERPRETER_PROBE_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        await _stop_tasks(process, tasks)
        raise
    except TimeoutError as exc:
        await _stop_tasks(process, tasks)
        raise ComfyRegistryInterpreterError(
            "interpreter_probe_timeout",
            "The managed runtime package inspection timed out.",
        ) from exc
    except _ProbeOutputTooLarge as exc:
        await _stop_tasks(process, tasks)
        raise ComfyRegistryInterpreterError(
            "interpreter_probe_output_too_large",
            "The managed runtime package inspection returned too much data.",
        ) from exc
    if returncode:
        raise ComfyRegistryInterpreterError(
            "interpreter_probe_failed",
            "The managed runtime package inspection failed.",
        )
    return stdout


async def _read_bounded(
    stream: asyncio.StreamReader,
    limit: int,
) -> bytes:
    result = bytearray()
    while True:
        chunk = await stream.read(min(_READ_CHUNK_BYTES, limit + 1 - len(result)))
        if not chunk:
            return bytes(result)
        result.extend(chunk)
        if len(result) > limit:
            raise _ProbeOutputTooLarge


async def _stop_tasks(
    process: asyncio.subprocess.Process,
    tasks: tuple[asyncio.Task[Any], ...],
) -> None:
    for task in tasks:
        task.cancel()
    await _terminate(process)
    await asyncio.gather(*tasks, return_exceptions=True)


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    await process.wait()


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return True
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)
