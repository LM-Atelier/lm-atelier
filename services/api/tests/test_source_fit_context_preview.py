"""Context previews follow admission selection without admitting work."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import event
from sqlalchemy.engine import Engine
from test_source_fit_acceptance import prepared_turn
from test_source_fit_preview import counts
from test_source_fit_regeneration import completed_source
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Chat, GenerationPreset, Message, ModelProfile, Run


@contextmanager
def no_preview_writes(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def reject_sql(
        _connection: Any, _cursor: Any, _sql: str, _parameters: Any, context: Any, _many: Any
    ) -> None:
        assert not (context.isinsert or context.isupdate or context.isdelete)

    def reject_artifact(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a preview must not retain an artifact")

    with monkeypatch.context() as patch:
        patch.setattr(app.state.services.artifacts, "ingest_bytes", reject_artifact)
        event.listen(Engine, "before_cursor_execute", reject_sql)
        try:
            yield
        finally:
            event.remove(Engine, "before_cursor_execute", reject_sql)


async def test_turn_preview_uses_actual_role_settings_without_admitting_work(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, payload = await prepared_turn(app, client, monkeypatch)
    payload.update(
        source_fit={"mode": "extend", "width": 5, "height": 6},
        role_overrides={"image": {"settings": {"steps": 3}}},
    )
    orchestrator = app.state.services.orchestrator
    resolve = orchestrator.resolve_turn_setting_layers
    selected: list[int] = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = resolve(*args, **kwargs)
        selected.append(result.effective_settings["steps"])
        return result

    monkeypatch.setattr(orchestrator, "resolve_turn_setting_layers", observe)
    async with app.state.services.scheduler.lease("primary"):
        before = counts()
        with no_preview_writes(app, monkeypatch):
            response = await client.post(f"/api/chats/{chat_id}/source-fit/preview", json=payload)
        assert response.status_code == 200, response.text
        assert selected == [3]
        assert response.json()["canvas"] == {"width": 5, "height": 6}
        assert response.json()["source"] == {"width": 2, "height": 3}
        assert response.json()["request_authorized"] is False
        assert counts() == before
        accepted = await client.post(f"/api/chats/{chat_id}/turns", json=payload)
        assert accepted.status_code == 202, accepted.text
        assert selected == [3, 3]
        assert accepted.json()["run"]["settings_json"]["steps"] == 3


async def test_turn_preview_resolves_the_implicit_primary_source(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = await completed_source(app, client, monkeypatch)
    with SessionLocal() as session:
        run = session.get(Run, original["run"]["id"])
        assert run is not None
        source_id = run.provenance_json["outputs"][0]["artifact_id"]
    payload = {
        "text": "Extend the image with a blue border",
        "mode": "image",
        "profile_id": original["run"]["profile_id"],
        "workflow_revision_id": original["run"]["workflow_revision_id"],
        "source_fit": {"mode": "extend", "width": 6, "height": 7},
    }
    async with app.state.services.scheduler.lease("primary"):
        before = counts()
        with no_preview_writes(app, monkeypatch):
            response = await client.post(
                f"/api/chats/{original['run']['chat_id']}/source-fit/preview", json=payload
            )
        assert response.status_code == 200, response.text
        assert response.json()["source_artifact_id"] == source_id
        assert response.json()["source"] == {"width": 4, "height": 5}
        assert response.json()["canvas"] == {"width": 6, "height": 7}
        assert counts() == before


@pytest.mark.parametrize(
    "choice",
    [
        "inherit",
        "auto",
        "settings",
        "preset",
        "original_changed",
        "prepared_changed",
        "stale",
        "hidden",
    ],
)
async def test_prior_preview_uses_accepted_settings_and_retained_pixels(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    chat_id, original_request = await prepared_turn(app, client, monkeypatch)
    orchestrator = app.state.services.orchestrator
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={**original_request, "source_fit": {"mode": "extend", "width": 4, "height": 5}},
        )
        assert accepted.status_code == 202, accepted.text
        message_id = accepted.json()["user_message"]["id"]
        with SessionLocal() as session:
            run = session.get(Run, accepted.json()["run"]["id"])
            assert run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None and snapshot.source_fit is not None
            profile = session.get(ModelProfile, run.profile_id)
            chat = session.get(Chat, chat_id)
            assert profile is not None and chat is not None
            profile.request_settings_json = {"steps": 8}
            chat.generation_settings_json = {"image": {"steps": 9}}
            preset = GenerationPreset(
                name="Neutral preview preset", role="image", settings_json={"steps": 4}
            )
            session.add(preset)
            if choice in {"original_changed", "prepared_changed"}:
                image = snapshot.source_fit.image
                artifact = session.get(
                    Artifact,
                    image.source_artifact_id
                    if choice == "original_changed"
                    else image.prepared_artifact_id,
                )
                assert artifact is not None
                orchestrator.artifacts.resolve(artifact).write_bytes(b"changed neutral fixture")
            if choice == "hidden":
                message = session.get(Message, message_id)
                assert message is not None
                message.transcript_visible = False
            session.commit()
            preset_id = preset.id
        payload: dict[str, Any] = {
            "text": "Extend the neutral grid with a blue border",
            "mode": "auto" if choice == "auto" else "image",
            "confirm_media": True,
            "idempotency_key": "neutral-preview-edit",
            "source_fit": {"mode": "extend", "width": 6, "height": 7},
        }
        expected_steps = 2
        if choice == "settings":
            payload["settings"] = {"steps": 3}
            expected_steps = 3
        elif choice == "preset":
            payload["preset_id"] = preset_id
            expected_steps = 4
        elif choice == "stale":
            payload["source_snapshot_sha256"] = "0" * 64
        resolve = orchestrator.resolve_turn_setting_layers
        selected: list[int] = []

        def observe(*args: Any, **kwargs: Any) -> Any:
            result = resolve(*args, **kwargs)
            selected.append(result.effective_settings["steps"])
            return result

        monkeypatch.setattr(orchestrator, "resolve_turn_setting_layers", observe)
        before = counts()
        with no_preview_writes(app, monkeypatch):
            response = await client.post(
                f"/api/messages/{message_id}/edits/source-fit/preview", json=payload
            )
        assert counts() == before
        if choice in {"prepared_changed", "stale", "hidden"}:
            assert (
                response.status_code
                == {"prepared_changed": 422, "stale": 409, "hidden": 404}[choice]
            ), response.text
            return
        assert response.status_code == 200, response.text
        assert selected == [expected_steps]
        preview = response.json()
        assert preview["source"] == {"width": 2, "height": 3}
        assert preview["canvas"] == {"width": 6, "height": 7}
        assert preview["margins"] == {"left": 2, "top": 2, "right": 2, "bottom": 2}
        assert preview["request_authorized"] is False
        queued = await client.post(f"/api/messages/{message_id}/edits", json=payload)
        assert queued.status_code == 202, queued.text
        assert selected == [expected_steps, expected_steps]
        assert queued.json()["run"]["settings_json"]["steps"] == expected_steps


async def test_prior_preview_rechecks_source_after_selection_await(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, payload = await prepared_turn(app, client, monkeypatch)
    orchestrator = app.state.services.orchestrator
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={**payload, "source_fit": {"mode": "extend", "width": 4, "height": 5}},
        )
        assert accepted.status_code == 202, accepted.text
        message_id = accepted.json()["user_message"]["id"]
        settings = orchestrator.engines.settings_for_role
        calls = 0

        async def change_source(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            result = await settings(*args, **kwargs)
            calls += 1
            if calls == 2:
                # The edit preparation has completed its own recheck. Change the
                # actual source during the subsequent turn-selection await.
                with SessionLocal() as other:
                    source = other.get(Message, message_id)
                    assert source is not None
                    source.transcript_visible = False
                    other.commit()
            return result

        monkeypatch.setattr(orchestrator.engines, "settings_for_role", change_source)
        before = counts()
        response = await client.post(
            f"/api/messages/{message_id}/edits/source-fit/preview",
            json={
                "text": "Extend the neutral grid with a blue border",
                "mode": "image",
                "idempotency_key": "neutral-preview-source-change",
                "source_fit": {"mode": "extend", "width": 6, "height": 7},
            },
        )
        assert calls == 2
        assert response.status_code == 404, response.text
        assert counts() == before
