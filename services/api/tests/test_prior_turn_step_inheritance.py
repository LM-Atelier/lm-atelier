from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import (
    Chat,
    GenerationPreset,
    ModelInstall,
    ModelProfile,
    Run,
    WorkPlan,
    WorkStep,
)

ORDERED = (
    "Write a short story about a paper boat, then create an image based on it, "
    "then animate the image into a video, then summarize the video"
)


async def _source(
    client: AsyncClient, *, installed: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    created = await client.post("/api/chats", json={"title": "Step configuration"})
    assert created.status_code == 201, created.text
    accepted = await client.post(
        f"/api/chats/{created.json()['id']}/turns",
        json={
            "text": ORDERED,
            "mode": "auto",
            "confirm_media": True,
        },
    )
    assert accepted.status_code == 202, accepted.text
    accepted = accepted.json()
    with SessionLocal() as session:
        runs = list(
            session.scalars(
                select(Run)
                .join(WorkStep, Run.work_step_id == WorkStep.id)
                .where(Run.work_plan_id == accepted["run"]["work_plan_id"])
                .order_by(WorkStep.ordinal)
            )
        )
        for index, run in enumerate(runs):
            role = (
                "chat"
                if run.operation == "text"
                else "video"
                if "video" in run.operation
                else "image"
            )
            profile = ModelProfile(
                name=f"Original step {index + 1}",
                role=role,
                engine="mock",
                request_settings_json={"temperature": 0.2 + index / 10}
                if role == "chat"
                else {"steps": index + 7},
            )
            if installed and index == 0:
                install = ModelInstall(
                    name="Constructed source model",
                    role=role,
                    engine="mock",
                    local_path="constructed-source-model.bin",
                    active=True,
                )
                session.add(install)
                session.flush()
                profile.model_install_id = install.id
            session.add(profile)
            session.flush()
            run.profile_id = profile.id
            run.settings_json = {**run.settings_json, **profile.request_settings_json}
        session.commit()
    with SessionLocal() as session:
        chat = session.get(Chat, accepted["run"]["chat_id"])
        assert chat is not None
        accepted["original_active_head"] = chat.active_head_message_id
    loaded = await client.get(f"/api/messages/{accepted['user_message']['id']}/edit-source")
    assert loaded.status_code == 200, loaded.text
    return accepted, loaded.json()


def _runs(plan_id: str) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        return [
            {
                "id": run.id,
                "profile_id": run.profile_id,
                "settings": run.settings_json,
                "context": run.provenance_json.get("accepted_context_sha256"),
            }
            for run in session.scalars(
                select(Run)
                .join(WorkStep, Run.work_step_id == WorkStep.id)
                .where(Run.work_plan_id == plan_id)
                .order_by(WorkStep.ordinal)
            )
        ]


async def test_unchanged_ordered_edit_preserves_each_source_step_including_repeated_roles(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        accepted, view = await _source(client)
        before = _runs(accepted["run"]["work_plan_id"])
        edited = await client.post(
            f"/api/messages/{view['source_user_message_id']}/edits",
            json={
                "text": ORDERED,
                "confirm_media": True,
                "idempotency_key": "inherit-all-steps",
                "source_snapshot_sha256": view["source_snapshot_sha256"],
            },
        )
        assert edited.status_code == 202, edited.text
        after = _runs(edited.json()["work_plan_id"])
        assert len(after) == len(before) == 4
        assert [item["profile_id"] for item in after] == [item["profile_id"] for item in before]
        for left, right in zip(before, after, strict=True):
            for key in ("temperature", "steps"):
                if key in left["settings"]:
                    assert right["settings"][key] == left["settings"][key]
            assert right["context"]
        assert _runs(accepted["run"]["work_plan_id"]) == before
        with SessionLocal() as session:
            chat = session.get(Chat, view["chat_id"])
            assert chat is not None
            assert chat.active_head_message_id == accepted["original_active_head"]
            plan = session.get(WorkPlan, edited.json()["work_plan_id"])
            assert plan is not None and plan.summary_json["routing_mode"] == "auto"


async def test_step_choices_override_role_choices_without_changing_another_same_role_step(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        _, view = await _source(client)
        with SessionLocal() as session:
            selected_profile = ModelProfile(
                name="Selected final text model", role="chat", engine="mock"
            )
            selected_preset = GenerationPreset(
                name="Selected final text preset", role="chat", settings_json={"temperature": 0.52}
            )
            session.add_all([selected_profile, selected_preset])
            session.flush()
            profile_id, preset_id = selected_profile.id, selected_preset.id
            session.commit()
        edited = await client.post(
            f"/api/messages/{view['source_user_message_id']}/edits",
            json={
                "text": ORDERED,
                "mode": "auto",
                "confirm_media": True,
                "idempotency_key": "step-and-role",
                "role_overrides": {"chat": {"settings": {"temperature": 0.31}}},
                "step_overrides": {
                    view["steps"][-1]["step_id"]: {
                        "settings": {"temperature": 0.67},
                        "profile_id": profile_id,
                        "preset_id": preset_id,
                    }
                },
            },
        )
        assert edited.status_code == 202, edited.text
        after = _runs(edited.json()["work_plan_id"])
        assert after[0]["settings"]["temperature"] == 0.31
        assert after[3]["settings"]["temperature"] == 0.67
        assert after[0]["profile_id"] == view["steps"][0]["profile_id"]
        assert after[3]["profile_id"] == profile_id
        with SessionLocal() as session:
            last = session.get(Run, after[3]["id"])
            assert last is not None and last.provenance_json["preset"]["id"] == preset_id
        assert after[1]["settings"]["steps"] == view["steps"][1]["settings"]["steps"]
        assert after[2]["settings"]["steps"] == view["steps"][2]["settings"]["steps"]


@pytest.mark.parametrize(
    "case",
    ["unknown", "unmapped", "wrong_workflow_role", "wrong_profile_role", "wrong_preset_role"],
)
async def test_invalid_step_override_refuses_without_accepting_a_partial_plan(
    app: FastAPI, client: AsyncClient, case: str
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        _, view = await _source(client)
        step_id = "missing-step" if case == "unknown" else view["steps"][2]["step_id"]
        override = (
            {"workflow_selection": {"selector_capability": "chat", "mode": "default"}}
            if case == "wrong_workflow_role"
            else {"settings": {"steps": 7}}
        )
        if case in {"wrong_profile_role", "wrong_preset_role"}:
            with SessionLocal() as session:
                wrong = (
                    ModelProfile(name="Wrong-role choice", role="chat", engine="mock")
                    if case == "wrong_profile_role"
                    else GenerationPreset(name="Wrong-role preset", role="chat", settings_json={})
                )
                session.add(wrong)
                session.flush()
                override = {"profile_id" if case == "wrong_profile_role" else "preset_id": wrong.id}
                session.commit()
        edited = await client.post(
            f"/api/messages/{view['source_user_message_id']}/edits",
            json={
                "text": "Describe a paper boat" if case == "unmapped" else ORDERED,
                "mode": "text" if case == "unmapped" else "auto",
                "confirm_media": True,
                "idempotency_key": "invalid-step",
                "step_overrides": {step_id: override},
            },
        )
        assert edited.status_code == 422, edited.text
        with SessionLocal() as session:
            assert len(list(session.scalars(select(WorkPlan)))) == 1


@pytest.mark.parametrize("profile_kind", ["configuration", "installed", "changed_install"])
async def test_reediting_a_frozen_ordered_source_preserves_each_profile_configuration(
    app: FastAPI, client: AsyncClient, profile_kind: str
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        _, view = await _source(client, installed=profile_kind != "configuration")
        first = await client.post(
            f"/api/messages/{view['source_user_message_id']}/edits",
            json={
                "text": ORDERED,
                "mode": "auto",
                "confirm_media": True,
                "idempotency_key": "first-version",
            },
        )
        assert first.status_code == 202, first.text
        first = first.json()
        loaded = await client.get(f"/api/messages/{first['user_message']['id']}/edit-source")
        assert loaded.status_code == 200, loaded.text
        frozen = loaded.json()
        with SessionLocal() as session:
            for step in frozen["steps"]:
                profile = session.get(ModelProfile, step["profile_id"])
                assert profile is not None
                profile.request_settings_json = (
                    {"temperature": 0.9} if profile.role == "chat" else {"steps": 99}
                )
            if profile_kind == "changed_install":
                install = session.scalar(
                    select(ModelInstall).where(ModelInstall.name == "Constructed source model")
                )
                assert install is not None
                install.manifest_json = {"sha256": "c" * 64}
            session.commit()
        second = await client.post(
            f"/api/messages/{frozen['source_user_message_id']}/edits",
            json={
                "text": ORDERED,
                "confirm_media": True,
                "idempotency_key": "second-version",
                "source_snapshot_sha256": frozen["source_snapshot_sha256"],
            },
        )
        if profile_kind == "changed_install":
            assert second.status_code == 422, second.text
            with SessionLocal() as session:
                assert len(list(session.scalars(select(WorkPlan)))) == 2
            return
        assert second.status_code == 202, second.text
        again = await client.get(f"/api/messages/{second.json()['user_message']['id']}/edit-source")
        assert again.status_code == 200, again.text
        assert [step["profile_settings"] for step in again.json()["steps"]] == [
            step["profile_settings"] for step in frozen["steps"]
        ]
