"""A render that fails while its prompt is in ComfyUI asks ComfyUI to stop that prompt.

The render's own failure ends its execution, and closing the engine's stream
on the way out is what stops the work it began: the ComfyUI adapter cancels
the prompt it submitted, by name, so nothing goes on rendering a result nobody
will read. The device handoff that follows recycles ComfyUI only after that,
so the cancel reaches the process that was running the prompt rather than the
one that replaces it, where it could only find nothing or another run's work.

ComfyUI is a constructed backend behind the adapter's own HTTP client and
progress socket, and the failure is a progress write the database refuses for
a reason other than being busy, which ends a render where a busy database
would only skip the write.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy.exc import OperationalError
from test_chat_handoff_across_retries import finished
from test_comfyui_kept_after_a_workflow_failure import (
    PROMPT_ID,
    _ComfyUI,
    _comfyui_engine,
    _handed_back,
    _ready_world,
)
from test_generation_retry import choose_retries
from test_media_worker_replacement import _image_turn

from local_lm import orchestrator as orchestrator_module
from local_lm.config import Settings
from local_lm.progress import apply_engine_progress

#: What the progress socket reports once the prompt is in: it starts, then samples.
STARTED_THEN_SAMPLING = (
    {"type": "executing", "data": {"prompt_id": PROMPT_ID, "node": "sampler"}},
    {"type": "progress", "data": {"prompt_id": PROMPT_ID, "value": 1, "max": 4}},
)


class _OrderedComfyUI(_ComfyUI):
    """The constructed ComfyUI, noting each cancel it is asked for among the workers' effects."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self.order = order
        self.requests: list[tuple[str, str]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if (request.method, request.url.path) == ("POST", f"/api/jobs/{PROMPT_ID}/cancel"):
            self.order.append(f"cancel {PROMPT_ID}")
        return await super().__call__(request)


def _broken_database_error(tmp_path: Path) -> OperationalError:
    """An error SQLite raises for a broken statement, not for a busy database."""

    broken_database = tmp_path / "broken.sqlite3"
    with sqlite3.connect(broken_database) as connection:
        connection.execute("CREATE TABLE present (id INTEGER PRIMARY KEY)")
        try:
            connection.execute("SELECT missing FROM present")
        except sqlite3.OperationalError as error:
            return OperationalError("statement", {}, error)
    raise AssertionError("sqlite did not fail as this test requires")


async def test_a_render_failing_while_its_prompt_runs_cancels_that_prompt_before_the_handoff(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    broken = _broken_database_error(tmp_path)
    refused: list[str] = []

    def refuse_sampling(session: Any, **values: Any) -> Any:
        if values.get("stage") == "sampling":
            refused.append("sampling")
            raise broken
        return apply_engine_progress(session, **values)

    monkeypatch.setattr(orchestrator_module, "apply_engine_progress", refuse_sampling)
    world = _ready_world(app, settings, monkeypatch)
    await choose_retries(client, 0)
    comfy = _OrderedComfyUI(world.order)
    adapter = await _comfyui_engine(app, monkeypatch, comfy, STARTED_THEN_SAMPLING)
    try:
        _chat_id, run_id = await _image_turn(client)
        run = await finished(client, run_id, "failed")
        order = await _handed_back(world)
    finally:
        await adapter.close()

    assert refused == ["sampling"], "the render never reached the write this test refuses"
    assert run["error"] == "This run stopped because of a database error."
    # The cancel names the render's own prompt, never a bare interrupt that
    # would stop whatever ComfyUI runs, and it comes before ComfyUI is
    # recycled, so it reaches the process that was running that prompt.
    assert ("POST", f"/api/jobs/{PROMPT_ID}/cancel") in comfy.requests
    assert ("POST", "/interrupt") not in comfy.requests
    assert order == [
        "stop chat",
        f"cancel {PROMPT_ID}",
        "stop media",
        "resume profile-chat",
        "start media",
    ]
