"""Activation cancellation waits for its temporary input write before removing it."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from run_waits import PATIENCE_SECONDS
from test_downloads import FakeProbeAdapter
from test_scheduler_claim_hold import _until

from local_lm.comfy_templates import ComfyTemplate, CompiledComfyTemplate
from local_lm.config import Settings
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker


@pytest.mark.parametrize("disposition", ["cancelled", "repeated", "owned"])
async def test_activation_input_write_finishes_before_cancellation_removes_it(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, disposition: str
) -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    paths: list[Path] = []
    real_write = Path.write_bytes

    def write(path: Path, content: bytes) -> int:
        if not path.name.startswith("activation-probe"):
            return real_write(path, content)
        paths.append(path)
        entered.set()
        assert release.wait(timeout=5), "The neutral input write was not released."
        try:
            return real_write(path, content)
        finally:
            finished.set()

    adapter = FakeProbeAdapter()
    manager = DownloadManager(
        settings, EventBroker(), media_adapter=Mock(spec_set=FakeProbeAdapter, wraps=adapter)
    )
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="neutral-input-write",
            path=settings.state_dir / "neutral-template.json",
            role="image",
            operation="image_to_image",
            score=1,
            sha256="a" * 64,
            dependencies=(),
        ),
        ui_graph={"nodes": []},
        api_graph={"source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}}},
        input_schema={"properties": {"input_image": {"type": "string"}}},
    )
    monkeypatch.setattr(Path, "write_bytes", write)
    task = asyncio.create_task(manager._probe_adaptive_checkpoint(compiled))
    try:
        await _until(entered.is_set)
        if disposition != "owned":
            task.cancel()
        await asyncio.sleep(0.04)
        if disposition == "repeated":
            task.cancel()
            await asyncio.sleep(0.04)
        assert not task.done() and not finished.is_set()
        release.set()
        outcome = await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        assert finished.is_set() and len(paths) == 1 and not paths[0].exists()
        if disposition == "owned":
            assert len(outcome) == 1 and outcome[0] is None
            assert adapter.request and adapter.request.input_paths == paths
        else:
            assert isinstance(outcome[0], asyncio.CancelledError)
            assert adapter.request is None
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if entered.is_set():
            await _until(finished.is_set)
