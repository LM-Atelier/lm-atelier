from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Chat, GenerationPreset, Run, WorkPlan


async def _preset_source(client: AsyncClient) -> tuple[dict[str, Any], list[str]]:
    presets = []
    for name, role, settings in [
        ("Original preset", "chat", {"temperature": 0.23}),
        ("Chosen preset", "chat", {"temperature": 0.61}),
        ("Later chat preset", "chat", {"temperature": 0.89}),
        ("Image preset", "image", {"steps": 12}),
    ]:
        response = await client.post(
            "/api/presets", json={"name": name, "role": role, "settings": settings}
        )
        assert response.status_code == 201, response.text
        presets.append(response.json()["id"])
    chat_response = await client.post(
        "/api/chats",
        json={"title": "Edit-local preset", "generation_preset_ids_json": {"chat": presets[0]}},
    )
    assert chat_response.status_code == 201, chat_response.text
    chat = chat_response.json()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Original question", "mode": "text"}
    )
    assert original.status_code == 202, original.text
    with SessionLocal() as session:
        current = session.get(Chat, chat["id"])
        assert current is not None
        current.generation_preset_ids_json = {"chat": presets[2]}
        current.generation_settings_json = {"chat": {"temperature": 0.88}}
        session.commit()
    return original.json(), presets


@pytest.mark.parametrize(
    "choice", ["inherit", "deleted", "explicit", "override", "none", "missing", "wrong_role"]
)
async def test_edit_preset_is_local_and_source_preserving(
    app: FastAPI, client: AsyncClient, choice: str
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source, presets = await _preset_source(client)
        prior_preset = deepcopy(source["run"]["provenance_json"]["preset"])
        with SessionLocal() as session:
            original_preset = session.get(GenerationPreset, presets[0])
            assert original_preset is not None
            if choice == "deleted":
                session.delete(original_preset)
            else:
                original_preset.name = "Renamed original preset"
                original_preset.settings_json = {"temperature": 0.99}
            session.commit()
        payload: dict[str, Any] = {"text": "Edited question", "idempotency_key": "local-preset"}
        if choice in {"explicit", "override"}:
            payload["preset_id"] = presets[1]
        elif choice == "none":
            payload["preset_id"] = None
        elif choice == "missing":
            payload["preset_id"] = "missing-preset"
        elif choice == "wrong_role":
            payload["preset_id"] = presets[3]
        if choice == "override":
            payload["settings"] = {"temperature": 0.41}
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits", json=payload
        )
        expected_status = 404 if choice == "missing" else 422 if choice == "wrong_role" else 202
        assert response.status_code == expected_status, response.text
        if choice == "wrong_role":
            assert response.json()["code"] != "request-validation-invalid"
        with SessionLocal() as session:
            original = session.get(Run, source["run"]["id"])
            current = session.get(Chat, source["run"]["chat_id"])
            assert original is not None and current is not None
            assert original.settings_json == source["run"]["settings_json"]
            assert original.provenance_json["preset"] == prior_preset
            assert current.generation_preset_ids_json == {"chat": presets[2]}
            assert current.generation_settings_json == {"chat": {"temperature": 0.88}}
            if choice in {"missing", "wrong_role"}:
                assert len(list(session.scalars(select(WorkPlan)))) == 1
                return
            edited = session.get(Run, response.json()["run"]["id"])
            assert edited is not None
            expected_temperature = (
                0.41 if choice == "override" else 0.61 if choice == "explicit" else 0.23
            )
            assert edited.settings_json["temperature"] == expected_temperature
            if choice in {"inherit", "deleted"}:
                assert edited.provenance_json["preset"] == prior_preset
            elif choice == "none":
                assert edited.provenance_json["preset"] is None
                assert edited.provenance_json["preset_layers"] == []
            else:
                assert edited.provenance_json["preset"]["id"] == presets[1]
                assert edited.provenance_json["preset_layers"][-1]["scope"] == "turn"


async def test_ordered_edit_applies_selected_preset_only_to_its_role(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source, presets = await _preset_source(client)
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={
                "text": "Write a short story about a paper boat, then create an image based on it, "
                "then animate the image into a video, then summarize the video",
                "mode": "auto",
                "confirm_media": True,
                "preset_id": presets[1],
                "ordered_settings": {"chat": {"temperature": 0.41}},
                "idempotency_key": "ordered-preset",
            },
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run).where(Run.work_plan_id == response.json()["work_plan_id"])
                )
            )
            assert len(runs) == 4
            for run in runs:
                preset = run.provenance_json["preset"]
                if run.operation == "text":
                    assert run.settings_json["temperature"] == 0.41
                    assert preset["id"] == presets[1]
                else:
                    assert preset is None or preset["id"] != presets[1]


async def test_edit_initializer_uses_accepted_preset_after_provenance_changes(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source, _ = await _preset_source(client)
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={"text": "First edited version", "idempotency_key": "frozen-preset"},
        )
        assert response.status_code == 202, response.text
        edited = response.json()
        expected_preset = source["run"]["provenance_json"]["preset"]
        with SessionLocal() as session:
            run = session.get(Run, edited["run"]["id"])
            assert run is not None
            run.provenance_json = {**run.provenance_json, "preset": None, "preset_layers": []}
            session.commit()
        loaded = await client.get(f"/api/messages/{edited['user_message']['id']}/edit-source")
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["preset"] == expected_preset
        copied = await client.post(
            f"/api/messages/{edited['user_message']['id']}/edits",
            json={"text": "Second edited version", "idempotency_key": "inherited-frozen-preset"},
        )
        assert copied.status_code == 202, copied.text
        assert copied.json()["run"]["provenance_json"]["preset"] == expected_preset
