"""Reading what a picture's file says, and previewing a remix of it, stays on this computer."""

from __future__ import annotations

import json
import logging
import socket
from io import BytesIO
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from test_picture_remix import (
    NEGATIVE,
    PROMPT,
    SETTINGS,
    _png,
    _preview,
    _profile,
    _revision,
    _uploaded,
)
from test_picture_remix_edit import _edit_workflow
from test_picture_remix_edit import _preview as _edit_preview
from test_picture_remix_edit import _uploaded as _library_upload
from test_workflow_package_import_endpoint import _ui_graph
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Refuse every connection and name lookup, and record each one tried."""

    tried: list[str] = []

    def refuse(name: str) -> Any:
        def refused(*_args: object, **_kwargs: object) -> None:
            tried.append(name)
            raise OSError("this test allows no network")

        return refused

    monkeypatch.setattr(socket.socket, "connect", refuse("connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", refuse("connect_ex"))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    return tried


@pytest.fixture
def published(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every event published, as text, so what it carries can be searched."""

    seen: list[str] = []
    publish = app.state.services.events.publish

    async def observe(
        event_type: str, entity_id: str | None = None, payload: dict[str, Any] | None = None
    ) -> Any:
        seen.append(f"{event_type} {entity_id} {payload!r}")
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(app.state.services.events, "publish", observe)
    return seen


def _logged(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(f"{record.getMessage()} {record.args!r}" for record in caplog.records)


def _written_logs(app: FastAPI) -> bytes:
    log_dir = app.state.services.settings.log_dir
    if not log_dir.is_dir():
        return b""
    return b"\n".join(path.read_bytes() for path in sorted(log_dir.rglob("*")) if path.is_file())


async def test_a_picture_s_settings_and_its_remix_preview_leak_nothing(
    app: FastAPI,
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
    no_network: list[str],
    published: list[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")

    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    events_after_upload = len(published)
    settings = await client.get(f"/api/artifacts/{artifact_id}/generation-settings")
    preview = await _preview(client, artifact_id, revision_id, profile_id, ["steps", "seed"])

    # Both answers carry the words, so their absence below is not an empty reading.
    assert settings.status_code == 200, settings.text
    assert PROMPT in settings.text and NEGATIVE in settings.text
    assert preview.status_code == 200, preview.text
    assert PROMPT in preview.text
    assert no_network == []
    # Reading and previewing publish nothing; the upload's own events carry no words.
    assert len(published) == events_after_upload
    for text in (PROMPT, NEGATIVE):
        assert text not in " ".join(published)
        assert text not in _logged(caplog)
        assert text.encode("utf-8") not in _written_logs(app)


def _picture_with_its_workflow() -> bytes:
    # The words twice: in the settings line, and inside the editor graph the file carries.
    workflow = _ui_graph()
    workflow["nodes"][0]["widgets_values"][0] = PROMPT
    info = PngInfo()
    info.add_text(
        "parameters",
        f"{PROMPT}\nSteps: 20, Sampler: Euler a, Seed: 12345, Size: 512x768, "
        "Denoising strength: 0.45",
    )
    info.add_text("workflow", json.dumps(workflow))
    output = BytesIO()
    Image.new("RGB", (512, 768), (90, 120, 150)).save(output, format="PNG", pnginfo=info)
    return output.getvalue()


async def test_an_edit_preview_and_a_picture_s_workflow_leak_nothing(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    no_network: list[str],
    published: list[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    artifact_id = await _library_upload(client, _picture_with_its_workflow())
    events_before = len(published)

    edit = await _edit_preview(client, artifact_id, revision_id, profile_id)
    embedded = await client.get(f"/api/artifacts/{artifact_id}/embedded-workflow")

    assert edit.status_code == 200, edit.text
    assert PROMPT in edit.json()["resolved"]["engine_prompt"]
    assert embedded.status_code == 200, embedded.text
    assert PROMPT in embedded.text
    assert no_network == []
    assert len(published) == events_before
    assert PROMPT not in " ".join(published)
    assert PROMPT not in _logged(caplog)
    assert PROMPT.encode("utf-8") not in _written_logs(app)
