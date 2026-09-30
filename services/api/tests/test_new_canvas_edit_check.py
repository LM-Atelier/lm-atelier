"""An edit that makes new canvas is not judged as an ordinary edit of its source."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image, ImageDraw
from sqlalchemy import select
from test_outpaint_dispatch import _outpaint_graph, _outpainter, _png
from test_source_fit_acceptance import prepared_turn
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.image_edit_difference import compare_edit
from local_lm.models import Chat, Job, ModelInstall, ModelProfile, Run
from local_lm.outpaint_workflows import OUTPAINT_SETTING_KEY
from local_lm.schemas import TurnRequest
from local_lm.workflow_compatibility import mirror_legacy_chat_workflow_selections


def checking(app: FastAPI, monkeypatch: pytest.MonkeyPatch, chat_id: str) -> None:
    """Turn edit checks on for the chat, with a vision profile to make them."""
    monkeypatch.setattr(
        app.state.services.orchestrator, "_profile_has_verified_vision", lambda *_: True
    )
    with SessionLocal() as session:
        install = ModelInstall(
            id="model_canvas_checker",
            name="Constructed checker",
            role="chat",
            engine="mock",
            local_path="synthetic",
            active=True,
        )
        profile = ModelProfile(
            id="profile_canvas_checker",
            name="Canvas checker",
            role="chat",
            engine="mock",
            model_install_id=install.id,
        )
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.active_vision_profile_id = profile.id
        chat.vision_settings_json = {"verify_image_edits": True}
        session.add_all([install, profile])
        session.flush()
        mirror_legacy_chat_workflow_selections(session, chat, ["vision"])
        session.commit()


def queued_check(app: FastAPI, run_id: str, source: str) -> tuple[str | None, dict[str, Any]]:
    """Queue the edit check for a run as its completion does: the picture it would
    be judged against, or None, and what the run then records."""
    orchestrator = app.state.services.orchestrator
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        job = session.scalar(select(Job).where(Job.run_id == run.id))
        assert job is not None
        # The source stands in for the result: what is checked here is whether
        # the edit is sent to be judged at all, before any model looks.
        queued = orchestrator._queue_image_edit_verification(session, run, job.id, [source])
        session.flush()
        check = session.get(Job, queued) if queued is not None else None
        judged_against = check.payload_json["source_artifact_id"] if check is not None else None
        return judged_against, dict(run.provenance_json.get("image_edit_verification") or {})


async def test_an_extension_past_the_edge_is_not_judged_as_an_edit_of_its_source(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = app.state.services.orchestrator
    revision_id, source_id, chat_id = await _outpainter(client, _outpaint_graph(), _png(400, 300))
    checking(app, monkeypatch, chat_id)
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat_id,
                TurnRequest(
                    text="Extend the scene to the right and upward.",
                    mode="image",
                    input_artifact_ids=[source_id],
                    workflow_revision_id=revision_id,
                    settings={OUTPAINT_SETTING_KEY: {"right": 0.5, "top": 0.25}},
                ),
                freeze_context=True,
                activate_branch=False,
            )
            session.commit()
        queued, record = queued_check(app, accepted.run.id, source_id)

    assert queued is None
    assert record == {
        "version": record["version"],
        "status": "skipped",
        "reason": "new_canvas",
        "automatic_retry_executed": False,
    }


@pytest.mark.parametrize("extended", [False, True])
async def test_a_source_extended_to_a_new_canvas_in_chat_is_not_judged_as_an_edit_of_it(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, extended: bool
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    checking(app, monkeypatch, chat_id)
    source = request["input_artifact_ids"][0]
    if extended:
        request["source_fit"] = {"mode": "extend", "width": 4, "height": 5}
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(f"/api/chats/{chat_id}/turns", json=request)
        assert response.status_code == 202, response.text
        run_id = response.json()["run"]["id"]
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None
            # A turn with no fit may leave its context to be frozen at dispatch.
            if accepted_context(session, run) is None:
                app.state.services.orchestrator._freeze_turn_context(session, run)
                session.commit()
        queued, record = queued_check(app, run_id, source)

    if extended:
        assert queued is None
        assert (record["status"], record["reason"]) == ("skipped", "new_canvas")
    else:
        # The same workflow asked for an ordinary edit is judged against its source.
        assert queued == source, record


def encoded(picture: Image.Image) -> bytes:
    output = BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()


def test_the_edit_comparison_cannot_judge_new_canvas() -> None:
    """Why: a source kept exactly inside a wider canvas reads as changed nearly everywhere."""
    source = Image.new("RGB", (256, 192), (128, 128, 128))
    draw = ImageDraw.Draw(source)
    draw.rectangle((40, 40, 110, 110), fill=(40, 60, 200))
    draw.ellipse((150, 70, 220, 140), fill=(200, 50, 40))
    extended = Image.new("RGB", (384, 192), (150, 150, 150))
    extended.paste(source, (64, 0))

    unchanged = compare_edit(encoded(source), encoded(source))
    kept_but_extended = compare_edit(encoded(source), encoded(extended))

    assert (unchanged.changed, unchanged.changed_regions) == (False, 0)
    assert kept_but_extended.changed
    assert kept_but_extended.changed_regions and kept_but_extended.changed_regions >= 2
