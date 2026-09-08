from __future__ import annotations

import base64
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Run, WorkPlan


@pytest.mark.parametrize("requested_count", [None, 1, 2])
async def test_auto_edit_preserves_output_count_unless_explicitly_changed(
    app: FastAPI, client: AsyncClient, requested_count: int | None
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Output count inheritance"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Draw a blue circle",
                "mode": "auto",
                "confirm_media": True,
                "output_count": 3,
            },
        )
        assert source.status_code == 202, source.text
        view = await client.get(f"/api/messages/{source.json()['user_message']['id']}/edit-source")
        assert view.status_code == 200, view.text
        assert view.json()["original_mode"] == "auto"
        assert view.json()["output_count"] == 3
        payload: dict[str, Any] = {
            "text": "Draw a green circle",
            "confirm_media": True,
            "idempotency_key": "preserve-auto-count",
        }
        if requested_count is not None:
            payload["output_count"] = requested_count
        edited = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/edits", json=payload
        )
        assert edited.status_code == 202, edited.text
        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run).where(Run.work_plan_id == edited.json()["work_plan_id"])
                )
            )
            assert len(runs) == (3 if requested_count is None else requested_count)
            original = session.get(WorkPlan, source.json()["run"]["work_plan_id"])
            assert original is not None and original.summary_json["output_count"] == 3


@pytest.mark.parametrize("mode", ["auto", "image"])
@pytest.mark.parametrize("override", ["omitted", "unrelated", "manual", "auto"])
async def test_auto_image_edit_preserves_automatic_strength_until_deliberately_changed(
    app: FastAPI, client: AsyncClient, override: str, mode: str
) -> None:
    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    upload = await client.post(
        "/api/artifacts", files={"file": ("neutral.png", image, "image/png")}
    )
    assert upload.status_code == 201, upload.text
    chat = (await client.post("/api/chats", json={"title": "Automatic edit strength"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Make the image brighter",
                "mode": mode,
                "confirm_media": True,
                "input_artifact_ids": [upload.json()["id"]],
            },
        )
        assert source.status_code == 202, source.text
        with SessionLocal() as session:
            run = session.get(Run, source.json()["run"]["id"])
            assert run is not None and run.operation == "image_to_image"
            strength = run.provenance_json["image_edit"]["strength"]
            assert strength["mode"] == "auto"
            # Model an accepted automatic value distinct from a fresh estimate.
            run.settings_json = {**run.settings_json, "denoise": 0.43}
            run.provenance_json = {
                **run.provenance_json,
                "image_edit": {
                    **run.provenance_json["image_edit"],
                    "strength": {**strength, "value": 0.43},
                },
            }
            session.commit()
        payload: dict[str, Any] = {
            "text": "Make the image brighter",
            "confirm_media": True,
            "idempotency_key": "preserve-auto-strength",
        }
        if override == "unrelated":
            payload["role_overrides"] = {"image": {"settings": {"steps": 19}}}
        elif override == "manual":
            payload["role_overrides"] = {"image": {"settings": {"denoise": 0.7}}}
        elif override == "auto":
            payload["role_overrides"] = {
                "image": {"settings": {"_image_edit_strength_mode": "auto"}}
            }
        edited = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/edits", json=payload
        )
        if override == "auto":
            # This mock workflow has no declared automatic-mode setting.
            assert edited.status_code == 422
            assert edited.json()["code"] == "turn-invalid"
            return
        assert edited.status_code == 202, edited.text
        with SessionLocal() as session:
            run = session.get(Run, edited.json()["run"]["id"])
            assert run is not None
            strength = run.provenance_json["image_edit"]["strength"]
            if override == "manual":
                assert strength["mode"] == "manual"
                assert strength["value"] == 0.7
            else:
                assert strength["mode"] == "auto"
                assert strength["value"] == 0.43
                assert strength["reason_codes"] == ["inherited_auto_value"]
                if override == "unrelated":
                    assert run.settings_json["steps"] == 19


async def _automatic_image_source(app: FastAPI, client: AsyncClient) -> dict[str, Any]:
    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    upload = await client.post(
        "/api/artifacts", files={"file": ("neutral.png", image, "image/png")}
    )
    assert upload.status_code == 201
    chat = (await client.post("/api/chats", json={"title": "Bound strength"})).json()
    source = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Make the image brighter",
            "mode": "auto",
            "confirm_media": True,
            "input_artifact_ids": [upload.json()["id"]],
        },
    )
    assert source.status_code == 202, source.text
    return source.json()


@pytest.mark.parametrize("frozen", [False, True])
async def test_strength_provenance_change_cannot_change_a_loaded_source_silently(
    app: FastAPI, client: AsyncClient, frozen: bool
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source = await _automatic_image_source(app, client)
        if frozen:
            first = await client.post(
                f"/api/messages/{source['user_message']['id']}/edits",
                json={
                    "text": "Make the image brighter",
                    "confirm_media": True,
                    "idempotency_key": "freeze-strength",
                },
            )
            assert first.status_code == 202, first.text
            source = first.json()
        before_strength = source["run"]["provenance_json"]["image_edit"]["strength"]
        loaded = await client.get(f"/api/messages/{source['user_message']['id']}/edit-source")
        assert loaded.status_code == 200, loaded.text
        with SessionLocal() as session:
            run = session.get(Run, source["run"]["id"])
            assert run is not None
            run.provenance_json = {
                **run.provenance_json,
                "image_edit": {
                    **run.provenance_json["image_edit"],
                    "strength": {**before_strength, "value": 0.83},
                },
            }
            session.commit()
        edited = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={
                "text": "Make the image brighter",
                "confirm_media": True,
                "idempotency_key": "bound-strength",
                "source_snapshot_sha256": loaded.json()["source_snapshot_sha256"],
            },
        )
        if not frozen:
            assert edited.status_code == 409, edited.text
            return
        assert edited.status_code == 202, edited.text
        strength = edited.json()["run"]["provenance_json"]["image_edit"]["strength"]
        assert strength["value"] == before_strength["value"]


@pytest.mark.parametrize("use_chat_preset", [False, True])
async def test_ordered_image_strength_ignores_an_unrelated_legacy_chat_preset(
    app: FastAPI, client: AsyncClient, use_chat_preset: bool
) -> None:
    from local_lm.models import GenerationPreset, WorkStep

    chat = (await client.post("/api/chats", json={"title": "Ordered image strength"})).json()
    text = "Create an image of a paper boat, then create an image based on it"
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": text,
                "mode": "auto",
                "confirm_media": True,
            },
        )
        assert source.status_code == 202, source.text
        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run)
                    .join(WorkStep, Run.work_step_id == WorkStep.id)
                    .where(Run.work_plan_id == source.json()["run"]["work_plan_id"])
                    .order_by(WorkStep.ordinal)
                )
            )
            assert len(runs) == 2
            assert runs[1].operation == "image_to_image"
            strength = runs[1].provenance_json["image_edit"]["strength"]
            assert strength["mode"] == "auto"
            preset = GenerationPreset(
                name="Unrelated text preset", role="chat", settings_json={"temperature": 0.3}
            )
            session.add(preset)
            session.flush()
            preset_id = preset.id
            session.commit()
        payload: dict[str, Any] = {
            "text": text,
            "confirm_media": True,
            "idempotency_key": "ordered-strength",
        }
        if use_chat_preset:
            payload["preset_id"] = preset_id
            ordinary = await client.post(
                f"/api/chats/{chat['id']}/turns", json={**payload, "mode": "auto"}
            )
            assert ordinary.status_code == 422, ordinary.text
        edited = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/edits", json=payload
        )
        assert edited.status_code == 202, edited.text
        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run)
                    .join(WorkStep, Run.work_step_id == WorkStep.id)
                    .where(Run.work_plan_id == edited.json()["work_plan_id"])
                    .order_by(WorkStep.ordinal)
                )
            )
            inherited = runs[1].provenance_json["image_edit"]["strength"]
            assert inherited["mode"] == "auto"
            assert inherited["value"] == strength["value"]
            assert inherited["reason_codes"] == ["inherited_auto_value"]
