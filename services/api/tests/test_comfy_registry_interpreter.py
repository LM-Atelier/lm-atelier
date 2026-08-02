from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

import local_lm.comfy_registry_interpreter as interpreter_module
from local_lm.comfy_registry_interpreter import (
    ComfyRegistryInterpreterError,
    probe_comfy_registry_wheel_target,
)
from local_lm.comfy_registry_wheel_artifacts import (
    comfy_registry_wheel_target_sha256,
    current_comfy_registry_wheel_target,
)

pytestmark = pytest.mark.asyncio


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int | None = 0,
        block: bool = False,
    ) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stderr.feed_data(stderr)
        if not block:
            self.stdout.feed_eof()
            self.stderr.feed_eof()
        self.returncode = returncode
        self._block = block
        self.killed = False

    async def wait(self) -> int:
        while self._block and not self.killed:
            await asyncio.sleep(0)
        if self.returncode is None:
            self.returncode = -9
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self._block = False
        self.returncode = -9
        self.stdout.feed_eof()
        self.stderr.feed_eof()


def _target_payload() -> bytes:
    environment, tags = current_comfy_registry_wheel_target()
    return json.dumps(
        {
            "marker_environment": environment,
            "supported_tags": list(tags),
        }
    ).encode()


def _executable(tmp_path: Path) -> Path:
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"synthetic executable")
    return executable


async def _use_process(
    monkeypatch: pytest.MonkeyPatch,
    process: _FakeProcess,
) -> list[object]:
    captured: list[object] = []

    async def create_subprocess_exec(*args: object, **kwargs: object) -> _FakeProcess:
        captured.extend(args)
        captured.append(kwargs)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    return captured


async def test_real_interpreter_target_matches_the_current_process() -> None:
    environment, tags = await probe_comfy_registry_wheel_target(Path(sys.executable))
    expected_environment, expected_tags = current_comfy_registry_wheel_target()

    assert environment == expected_environment
    assert tags == expected_tags
    assert comfy_registry_wheel_target_sha256(
        environment,
        tags,
    ) == comfy_registry_wheel_target_sha256(expected_environment, expected_tags)


async def test_probe_uses_isolated_credential_free_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(stdout=_target_payload())
    captured = await _use_process(monkeypatch, process)
    executable = _executable(tmp_path)

    await probe_comfy_registry_wheel_target(executable)

    assert captured[:3] == [str(executable.resolve()), "-I", "-c"]
    options = captured[-1]
    assert isinstance(options, dict)
    assert options["stdin"] == asyncio.subprocess.DEVNULL
    assert options["cwd"] == executable.parent
    assert options["env"]["PYTHONNOUSERSITE"] == "1"
    assert not any("TOKEN" in key.upper() for key in options["env"])


async def test_missing_and_linked_executables_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ComfyRegistryInterpreterError) as missing:
        await probe_comfy_registry_wheel_target(tmp_path / "missing-python")
    assert missing.value.code == "invalid_python_executable"

    with pytest.raises(ComfyRegistryInterpreterError) as relative:
        await probe_comfy_registry_wheel_target(Path("python.exe"))
    assert relative.value.code == "invalid_python_executable"

    executable = _executable(tmp_path)
    monkeypatch.setattr(interpreter_module, "_is_link_or_reparse", lambda _path: True)
    with pytest.raises(ComfyRegistryInterpreterError) as linked:
        await probe_comfy_registry_wheel_target(executable)
    assert linked.value.code == "invalid_python_executable"


async def test_linked_entry_point_resolves_to_regular_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX virtual-environment entry points use file symlinks")
    executable = _executable(tmp_path)
    alias = tmp_path / "python-alias"
    try:
        alias.symlink_to(executable)
    except OSError:
        pytest.skip("file symlinks are unavailable")
    await _use_process(monkeypatch, _FakeProcess(stdout=_target_payload()))

    await probe_comfy_registry_wheel_target(alias)


@pytest.mark.parametrize(
    ("output", "code"),
    [
        (b"not-json", "interpreter_probe_invalid_output"),
        (b"{}", "interpreter_probe_invalid_output"),
        (
            json.dumps(
                {
                    "marker_environment": {"python_full_version": "not-a-version"},
                    "supported_tags": ["not-a-tag"],
                }
            ).encode(),
            "interpreter_probe_invalid_target",
        ),
    ],
)
async def test_invalid_probe_payloads_are_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: bytes,
    code: str,
) -> None:
    await _use_process(monkeypatch, _FakeProcess(stdout=output))

    with pytest.raises(ComfyRegistryInterpreterError) as captured:
        await probe_comfy_registry_wheel_target(_executable(tmp_path))

    assert captured.value.code == code


async def test_failed_probe_does_not_surface_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _use_process(
        monkeypatch,
        _FakeProcess(stderr=b"private subprocess detail", returncode=1),
    )

    with pytest.raises(ComfyRegistryInterpreterError) as captured:
        await probe_comfy_registry_wheel_target(_executable(tmp_path))

    assert captured.value.code == "interpreter_probe_failed"
    assert "private subprocess detail" not in str(captured.value)


async def test_oversized_probe_output_stops_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(
        stdout=b"x" * (interpreter_module.MAX_INTERPRETER_PROBE_OUTPUT_BYTES + 1),
        returncode=None,
        block=True,
    )
    await _use_process(monkeypatch, process)

    with pytest.raises(ComfyRegistryInterpreterError) as captured:
        await probe_comfy_registry_wheel_target(_executable(tmp_path))

    assert captured.value.code == "interpreter_probe_output_too_large"
    assert process.killed is True


async def test_probe_timeout_stops_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=None, block=True)
    await _use_process(monkeypatch, process)
    monkeypatch.setattr(interpreter_module, "INTERPRETER_PROBE_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(ComfyRegistryInterpreterError) as captured:
        await probe_comfy_registry_wheel_target(_executable(tmp_path))

    assert captured.value.code == "interpreter_probe_timeout"
    assert process.killed is True


async def test_probe_cancellation_stops_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=None, block=True)
    await _use_process(monkeypatch, process)
    task = asyncio.create_task(probe_comfy_registry_wheel_target(_executable(tmp_path)))
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed is True
