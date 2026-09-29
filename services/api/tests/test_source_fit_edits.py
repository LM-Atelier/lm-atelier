"""Prior-turn edits preserve accepted canvas intent and canonical source bytes."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_source_fit_acceptance import prepared_turn
from test_source_fit_image import png
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Run
from local_lm.source_fit_image import replay_source_fit_image


async def test_edit_source_exposes_canvas_intent(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    fit = {"mode": "extend", "width": 4, "height": 5}
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns", json={**request, "source_fit": fit}
        )
        assert response.status_code == 202, response.text
        source = await client.get(
            f"/api/messages/{response.json()['user_message']['id']}/edit-source"
        )
        assert source.status_code == 200, source.text
        assert source.json().get("source_fit") == fit


@pytest.mark.parametrize(
    "choice",
    [
        "inherit",
        "clear",
        "resize",
        "resize_original_changed",
        "new_source",
        "new_workflow",
        "original_changed",
        "prepared_changed",
        "auto",
        "auto_pinned",
    ],
)
async def test_edit_preserves_or_explicitly_changes_the_source_canvas(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    _, other = await prepared_turn(app, client, monkeypatch)
    uploaded = await client.post(
        "/api/artifacts",
        files={
            "file": ("second-grid.png", png(), "image/png"),
        },
    )
    assert uploaded.status_code == 201
    fit = {"mode": "extend", "width": 4, "height": 5}
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns", json={**request, "source_fit": fit}
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            original = session.get(Run, response.json()["run"]["id"])
            assert original is not None
            original_context = accepted_context(session, original)
            assert original_context is not None and original_context.source_fit is not None
            original_recipe = original_context.source_fit
            expected_bytes = replay_source_fit_image(
                session,
                app.state.services.orchestrator.artifacts,
                original_recipe.image,
                selected_source_id=request["input_artifact_ids"][0],
            ).content
            if choice in {"original_changed", "prepared_changed", "resize_original_changed"}:
                artifact_id = (
                    original_recipe.image.source_artifact_id
                    if choice in {"original_changed", "resize_original_changed"}
                    else original_recipe.image.prepared_artifact_id
                )
                artifact = session.get(Artifact, artifact_id)
                assert artifact is not None
                app.state.services.orchestrator.artifacts.resolve(artifact).write_bytes(
                    b"changed neutral fixture"
                )
        payload: dict[str, Any] = {
            "text": "Extend the neutral grid with a blue border",
            "mode": "auto" if choice in {"auto", "auto_pinned"} else "image",
            "confirm_media": True,
            "idempotency_key": "source-fit-edit",
        }
        if choice == "auto_pinned":
            payload.update(
                {
                    "source_fit": fit,
                    "input_artifact_ids": request["input_artifact_ids"],
                    "workflow_revision_id": request["workflow_revision_id"],
                    "profile_id": request["profile_id"],
                }
            )
        if choice == "clear":
            payload["source_fit"] = None
        elif choice in {"resize", "resize_original_changed"}:
            payload["source_fit"] = {**fit, "width": 6}
        elif choice == "new_source":
            payload["input_artifact_ids"] = [uploaded.json()["id"]]
        elif choice == "new_workflow":
            payload["workflow_revision_id"] = other["workflow_revision_id"]
            payload["profile_id"] = other["profile_id"]
            payload["source_fit"] = fit
        url = f"/api/messages/{response.json()['user_message']['id']}/edits"
        edited = await client.post(url, json=payload)
        if choice == "prepared_changed":
            assert edited.status_code == 422, edited.text
            return
        assert edited.status_code == 202, edited.text
        if choice == "auto_pinned":
            source = await client.get(
                f"/api/messages/{edited.json()['user_message']['id']}/edit-source"
            )
            assert source.status_code == 200, source.text
            assert source.json()["original_mode"] == "auto"
            assert source.json()["source_fit"] == fit
        replay = await client.post(url, json=payload)
        assert replay.status_code == 202, replay.text
        assert replay.json()["run"]["id"] == edited.json()["run"]["id"]
        with SessionLocal() as session:
            run = session.get(Run, edited.json()["run"]["id"])
            assert run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None
            if choice == "clear":
                assert snapshot.source_fit is None
                return
            assert snapshot.source_fit is not None
            recipe = snapshot.source_fit
            assert (recipe.canvas_width, recipe.canvas_height) == (
                6 if choice in {"resize", "resize_original_changed"} else 4,
                5,
            )
            assert recipe.image.source_artifact_id == (
                uploaded.json()["id"]
                if choice == "new_source"
                else request["input_artifact_ids"][0]
            )
            assert snapshot.workflow_revision_id == (
                other["workflow_revision_id"]
                if choice == "new_workflow"
                else request["workflow_revision_id"]
            )
            if choice != "new_source":
                actual = replay_source_fit_image(
                    session,
                    app.state.services.orchestrator.artifacts,
                    recipe.image,
                    selected_source_id=request["input_artifact_ids"][0],
                ).content
                assert actual == expected_bytes
