"""A model file a run has verified is not read again while it stays unchanged.

Every run of an imported workflow checks that each model file it binds still
holds the bytes it was installed with. Reading those bytes in full every time
stalled each job's start for as long as gigabytes take to read. A file whose
bytes matched, and whose identity (device, file number, mode, size, modification
and change times) has not changed since, is now taken as matching without
reading it, and the check runs in a worker thread rather than on the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from test_workflow_activations import _asset, _model

from local_lm import orchestrator as orchestrator_module
from local_lm import workflow_activations
from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.workflow_activations import WorkflowActivationError

MESSAGE = "Selected model component bytes changed"


def _counted_reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """One entry for each file the verification reads in full.

    The verification starts a digest with no data only to hash a file's bytes;
    every other digest it makes is of data it already holds.
    """

    reads: list[int] = []

    def sha256(*data: bytes) -> Any:
        if not data:
            reads.append(1)
        return hashlib.sha256(*data)

    monkeypatch.setattr(workflow_activations, "hashlib", SimpleNamespace(sha256=sha256))
    return reads


def _weights(tmp_path: Path, name: str = "weights.safetensors", size: int = 4096) -> Path:
    path = tmp_path / name
    path.write_bytes(os.urandom(size))
    return path.resolve()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _after_a_change_time_tick() -> None:
    """Some file systems keep change times in coarse ticks; let one pass."""

    time.sleep(0.05)


def test_an_unchanged_file_is_read_once_for_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _counted_reads(monkeypatch)
    path = _weights(tmp_path)
    digest = _digest(path)

    for _ in range(3):
        workflow_activations._verify_file_digest(path, digest, MESSAGE)

    assert len(reads) == 1


@pytest.mark.parametrize("change", ["content", "restored-time"])
def test_a_file_written_since_it_matched_is_read_again_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    reads = _counted_reads(monkeypatch)
    path = _weights(tmp_path)
    digest = _digest(path)
    workflow_activations._verify_file_digest(path, digest, MESSAGE)
    _after_a_change_time_tick()
    before = path.stat()
    if change == "content":
        path.write_bytes(b"other bytes")
    else:
        # The same size, and the modification time put back as it was.
        path.write_bytes(os.urandom(before.st_size))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(WorkflowActivationError) as refused:
        workflow_activations._verify_file_digest(path, digest, MESSAGE)

    assert refused.value.code == "dependency_content_drift"
    assert len(reads) == 2


def test_a_file_replaced_by_a_copy_is_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _counted_reads(monkeypatch)
    path = _weights(tmp_path)
    digest = _digest(path)
    workflow_activations._verify_file_digest(path, digest, MESSAGE)
    before = path.stat()
    copy = path.with_suffix(".copy")
    copy.write_bytes(path.read_bytes())
    os.utime(copy, ns=(before.st_atime_ns, before.st_mtime_ns))
    copy.replace(path)

    # The copy holds the same bytes, so it matches, but only after it is read.
    workflow_activations._verify_file_digest(path, digest, MESSAGE)

    assert len(reads) == 2


@pytest.mark.parametrize("change", ["touched", "same-bytes-restored-time"])
def test_a_file_that_changes_while_it_is_read_is_refused_and_not_remembered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Bytes read from a file that changed meanwhile prove nothing, even when they match."""

    path = _weights(tmp_path)
    digest = _digest(path)
    original = workflow_activations._file_digest

    def read_then_change(file: Path) -> str:
        result = original(file)
        _after_a_change_time_tick()
        before = file.stat()
        if change == "touched":
            os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        else:
            file.write_bytes(file.read_bytes())
            os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result

    monkeypatch.setattr(workflow_activations, "_file_digest", read_then_change)
    with pytest.raises(WorkflowActivationError):
        workflow_activations._verify_file_digest(path, digest, MESSAGE)

    monkeypatch.setattr(workflow_activations, "_file_digest", original)
    reads = _counted_reads(monkeypatch)
    workflow_activations._verify_file_digest(path, digest, MESSAGE)
    assert len(reads) == 1


def test_a_file_gone_since_it_matched_is_refused(tmp_path: Path) -> None:
    path = _weights(tmp_path)
    digest = _digest(path)
    workflow_activations._verify_file_digest(path, digest, MESSAGE)
    path.unlink()

    with pytest.raises(WorkflowActivationError) as refused:
        workflow_activations._verify_file_digest(path, digest, MESSAGE)

    assert refused.value.code == "dependency_content_drift"


def test_a_match_for_one_digest_says_nothing_about_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _counted_reads(monkeypatch)
    path = _weights(tmp_path)
    workflow_activations._verify_file_digest(path, _digest(path), MESSAGE)

    with pytest.raises(WorkflowActivationError):
        workflow_activations._verify_file_digest(path, "0" * 64, MESSAGE)

    assert len(reads) == 2


def test_a_file_that_failed_to_match_is_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _counted_reads(monkeypatch)
    path = _weights(tmp_path)

    for _ in range(2):
        with pytest.raises(WorkflowActivationError):
            workflow_activations._verify_file_digest(path, "0" * 64, MESSAGE)

    assert len(reads) == 2


def test_the_least_recently_checked_file_is_forgotten_past_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow_activations, "MAX_REMEMBERED_FILE_DIGESTS", 2, raising=False)
    reads = _counted_reads(monkeypatch)
    first, second, third = (_weights(tmp_path, f"part-{index}.safetensors") for index in range(3))
    digests = {path: _digest(path) for path in (first, second, third)}
    for path in (first, second, third):
        workflow_activations._verify_file_digest(path, digests[path], MESSAGE)
    assert len(reads) == 3

    read_again = []
    for path in (third, second, first, second):
        before = len(reads)
        workflow_activations._verify_file_digest(path, digests[path], MESSAGE)
        read_again.append(len(reads) > before)

    # Checking a remembered file counts as using it: second, checked just
    # before first came back, is kept, and third is the one forgotten.
    assert read_again == [False, False, True, False]


def test_a_second_launch_reads_no_unchanged_model_or_asset_file(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SessionLocal() as session:
        model = _model(session, tmp_path / "model", suffix="read-once")
        asset = _asset(session, tmp_path / "asset", suffix="read-once")
        session.commit()
        model_id, asset_id = model.id, asset.id
    reads = _counted_reads(monkeypatch)

    launches = []
    reads_per_launch = []
    for _ in range(2):
        before = len(reads)
        with SessionLocal() as session:
            launches.append(
                (
                    workflow_activations._model_launch_binding(session, model_id),
                    workflow_activations._asset_launch_binding(session, asset_id),
                )
            )
        reads_per_launch.append(len(reads) - before)

    assert launches[0] == launches[1]
    assert reads_per_launch == [2, 0]


async def test_a_media_run_revalidates_its_workflow_off_the_event_loop(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run's engine and activation are read in a worker thread.

    The mock engine binds no workflow, so this pins where the read runs; the
    ComfyUI activation it prepares is revalidated in the same call.
    """

    reads: list[tuple[str, bool]] = []
    original = accepted_context

    def observed(session: Any, run: Any) -> Any:
        caller = inspect.stack()[1].function
        if caller in {"_execute_media", "_prepare_media_run"}:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                reads.append((caller, False))
            else:
                reads.append((caller, True))
        return original(session, run)

    monkeypatch.setattr(orchestrator_module, "accepted_context", observed)
    chat = await client.post("/api/chats", json={"title": "Read the model once"})
    assert chat.status_code == 201
    chat_id = chat.json()["id"]
    accepted = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={"text": "Create an image of a quiet harbour", "mode": "image"},
    )
    assert accepted.status_code == 202

    async def read() -> dict[str, Any] | None:
        current = await client.get(f"/api/chats/{chat_id}")
        assistants = [m for m in current.json()["messages"] if m["role"] == "assistant"]
        return assistants[-1] if assistants else None

    assistant = await wait_for_terminal_status(read, what="the image run", expected=None)

    assert assistant["status"] == "complete"
    # The first read is the one that prepares the activation; the render's own
    # later reads hash nothing and stay where they were.
    assert reads[0] == ("_prepare_media_run", False)
