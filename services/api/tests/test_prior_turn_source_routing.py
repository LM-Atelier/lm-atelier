from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Chat, ModelProfile, Run, WorkPlan, WorkStep

ORDERED = (
    "Write a short story about a paper boat, then create an image based on it, "
    "then animate the image into a video, then summarize the video"
)


async def _turn(client: AsyncClient, **request: Any) -> dict[str, Any]:
    response = await client.post("/api/chats", json={"title": "Source routing"})
    assert response.status_code == 201, response.text
    chat_id = response.json()["id"]
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.routing_mode = "auto"
        session.commit()
    response = await client.post(f"/api/chats/{chat_id}/turns", json=request)
    assert response.status_code == 202, response.text
    return response.json()


@pytest.mark.parametrize("selected_mode", ["auto", "text", "image", "video", None])
async def test_edit_source_records_the_accepted_routing_mode_independent_of_current_chat(
    app: FastAPI, client: AsyncClient, selected_mode: str | None
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        payload: dict[str, Any] = {"text": "Describe a blue paper boat", "confirm_media": True}
        if selected_mode is not None:
            payload["mode"] = selected_mode
        source = await _turn(client, **payload)
        with SessionLocal() as session:
            chat = session.get(Chat, source["run"]["chat_id"])
            assert chat is not None
            chat.routing_mode = "video"
            session.commit()
        loaded = await client.get(f"/api/messages/{source['user_message']['id']}/edit-source")
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["original_mode"] == (selected_mode or "auto")
        assert loaded.json()["plan_kind"] == "single"
        assert loaded.json()["steps"] == []


async def test_historical_source_leaves_unrecorded_routing_unknown(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, text="Describe a paper boat", mode="text")
        with SessionLocal() as session:
            plan = session.get(WorkPlan, source["run"]["work_plan_id"])
            chat = session.get(Chat, source["run"]["chat_id"])
            assert plan is not None and chat is not None
            plan.summary_json = {
                key: value for key, value in plan.summary_json.items() if key != "routing_mode"
            }
            chat.routing_mode = "auto"
            session.commit()
        loaded = await client.get(f"/api/messages/{source['user_message']['id']}/edit-source")
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["original_mode"] is None
        assert loaded.json()["mode"] == "text"


async def test_ordered_source_keeps_distinct_repeated_role_configurations_and_binds_every_step(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, text=ORDERED, mode="auto", confirm_media=True)
        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run)
                    .join(WorkStep, Run.work_step_id == WorkStep.id)
                    .where(Run.work_plan_id == source["run"]["work_plan_id"])
                    .order_by(WorkStep.ordinal)
                )
            )
            assert len(runs) == 4
            for index, run in enumerate(runs):
                role = (
                    "chat"
                    if run.operation == "text"
                    else "video"
                    if "video" in run.operation
                    else "image"
                )
                profile = ModelProfile(
                    name=f"Step {index + 1}",
                    role=role,
                    engine="mock",
                    request_settings_json={"temperature": 0.2 + index / 10}
                    if role == "chat"
                    else {},
                )
                session.add(profile)
                session.flush()
                run.profile_id = profile.id
                run.settings_json = (
                    {**run.settings_json, "temperature": 0.2 + index / 10}
                    if role == "chat"
                    else run.settings_json
                )
            last_run_id = runs[-1].id
            session.commit()
        path = f"/api/messages/{source['user_message']['id']}/edit-source"
        loaded = await client.get(path)
        assert loaded.status_code == 200, loaded.text
        view = loaded.json()
        assert view["original_mode"] == "auto"
        assert view["plan_kind"] == "ordered"
        assert [step["ordinal"] for step in view["steps"]] == [1, 2, 3, 4]
        assert [step["settings_role"] for step in view["steps"]] == [
            "chat",
            "image",
            "video",
            "chat",
        ]
        assert view["steps"][0]["settings"]["temperature"] == 0.2
        assert view["steps"][3]["settings"]["temperature"] == 0.5
        assert len({step["profile_id"] for step in view["steps"]}) == 4
        assert all(step["depends_on"] for step in view["steps"][1:])
        with SessionLocal() as session:
            last_run = session.get(Run, last_run_id)
            assert last_run is not None
            last_run.settings_json = {**last_run.settings_json, "temperature": 0.7}
            session.commit()
        changed = await client.get(path)
        assert changed.status_code == 200, changed.text
        assert changed.json()["source_snapshot_sha256"] != view["source_snapshot_sha256"]
        refused = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={
                "text": ORDERED,
                "mode": "auto",
                "confirm_media": True,
                "idempotency_key": "stale-other-step",
                "source_snapshot_sha256": view["source_snapshot_sha256"],
            },
        )
        assert refused.status_code == 409, refused.text
        with SessionLocal() as session:
            assert len(list(session.scalars(select(WorkPlan)))) == 1


async def test_edited_ordered_source_uses_frozen_step_settings_and_recorded_mode(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, text="Describe a paper boat", mode="text")
        edited = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={
                "text": ORDERED,
                "mode": "auto",
                "confirm_media": True,
                "idempotency_key": "ordered-version",
                "role_overrides": {
                    "chat": {"settings": {"temperature": 0.31}},
                    "image": {"settings": {"steps": 7}},
                    "video": {"settings": {"steps": 8}},
                },
            },
        )
        assert edited.status_code == 202, edited.text
        edited = edited.json()
        path = f"/api/messages/{edited['user_message']['id']}/edit-source"
        loaded = await client.get(path)
        assert loaded.status_code == 200, loaded.text
        with SessionLocal() as session:
            plan = session.get(WorkPlan, edited["run"]["work_plan_id"])
            assert plan is not None
            plan.summary_json = {**plan.summary_json, "routing_mode": "video"}
            for run in session.scalars(select(Run).where(Run.work_plan_id == plan.id)):
                run.settings_json = {**run.settings_json, "temperature": 0.9, "steps": 99}
            session.commit()
        again = await client.get(path)
        assert again.status_code == 200, again.text
        assert again.json()["original_mode"] == "auto"
        assert again.json()["steps"] == loaded.json()["steps"]
        assert again.json()["source_snapshot_sha256"] == loaded.json()["source_snapshot_sha256"]


async def test_ordered_source_refuses_a_missing_step_instead_of_returning_a_partial_plan(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, text=ORDERED, mode="auto", confirm_media=True)
        with SessionLocal() as session:
            run = session.scalar(
                select(Run)
                .join(WorkStep, Run.work_step_id == WorkStep.id)
                .where(Run.work_plan_id == source["run"]["work_plan_id"])
                .order_by(WorkStep.ordinal.desc())
            )
            assert run is not None
            run.work_step_id = None
            session.commit()
        loaded = await client.get(f"/api/messages/{source['user_message']['id']}/edit-source")
        assert loaded.status_code == 409, loaded.text
