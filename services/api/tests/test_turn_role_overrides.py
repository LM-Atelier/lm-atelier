from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Chat, GenerationPreset, ModelProfile, Run, WorkPlan

ORDERED = (
    "Write a short story about a paper boat, then create an image based on it, "
    "then animate the image into a video, then summarize the video"
)


async def _source(client: AsyncClient) -> dict[str, Any]:
    chat = await client.post("/api/chats", json={"title": "Role-local version"})
    assert chat.status_code == 201, chat.text
    source = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={"text": "Describe a blue paper boat", "mode": "text"},
    )
    assert source.status_code == 202, source.text
    return source.json()


def _choices() -> dict[str, dict[str, Any]]:
    result = {}
    with SessionLocal() as session:
        for role, settings in (
            ("chat", {"temperature": 0.42}),
            ("image", {"steps": 7}),
            ("video", {"steps": 6}),
        ):
            profile = ModelProfile(name=f"Chosen {role}", role=role, engine="mock")
            preset = GenerationPreset(
                name=f"Chosen {role} preset", role=role, settings_json=settings
            )
            session.add_all([profile, preset])
            session.flush()
            result[role] = {"profile_id": profile.id, "preset_id": preset.id}
        session.commit()
    return result


@pytest.mark.parametrize("endpoint", ["ordinary", "edit"])
@pytest.mark.parametrize("intent", ["text", "image", "ordered"])
async def test_auto_applies_each_roles_choices_and_settings_without_changing_defaults(
    app: FastAPI, client: AsyncClient, endpoint: str, intent: str
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source = await _source(client)
        choices = _choices()
        choices["chat"]["settings"] = {"temperature": 0.31}
        choices["image"]["settings"] = {"steps": 8}
        choices["video"]["settings"] = {"steps": 9}
        text = {
            "text": "Explain how paper boats float",
            "image": "Create an image of a blue paper boat",
            "ordered": ORDERED,
        }[intent]
        path = (
            f"/api/messages/{source['user_message']['id']}/edits"
            if endpoint == "edit"
            else f"/api/chats/{source['run']['chat_id']}/turns"
        )
        response = await client.post(
            path,
            json={
                "text": text,
                "mode": "auto",
                "confirm_media": True,
                "idempotency_key": "role-local",
                "role_overrides": choices,
            },
        )
        assert response.status_code == 202, response.text
        plan_id = response.json()["run"]["work_plan_id"]
        with SessionLocal() as session:
            runs = list(session.scalars(select(Run).where(Run.work_plan_id == plan_id)))
            assert len(runs) == (4 if intent == "ordered" else 1)
            if intent != "ordered":
                assert runs[0].operation == ("text" if intent == "text" else "text_to_image")
            for run in runs:
                role = (
                    "chat"
                    if run.operation == "text"
                    else "video"
                    if "video" in run.operation
                    else "image"
                )
                assert run.profile_id == choices[role]["profile_id"]
                assert run.provenance_json["preset"]["id"] == choices[role]["preset_id"]
                for key, value in choices[role]["settings"].items():
                    assert run.settings_json[key] == value
                if endpoint == "edit":
                    assert run.provenance_json.get("accepted_context_sha256")
            chat = session.get(Chat, source["run"]["chat_id"])
            original = session.get(Run, source["run"]["id"])
            assert chat is not None and original is not None
            assert not chat.generation_settings_json and not chat.generation_preset_ids_json
            assert original.settings_json == source["run"]["settings_json"]
            if endpoint == "edit":
                assert chat.active_head_message_id == source["assistant_message"]["id"]


@pytest.mark.parametrize("invalid", ["role", "workflow_role", "profile_role", "preset_role"])
async def test_ordered_role_overrides_refuse_wrong_roles_without_partial_acceptance(
    app: FastAPI, client: AsyncClient, invalid: str
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source = await _source(client)
        choices = _choices()
        overrides: dict[str, Any] = {}
        if invalid == "role":
            overrides["audio"] = {"settings": {}}
        elif invalid == "workflow_role":
            overrides["chat"] = {
                "workflow_selection": {"selector_capability": "image", "mode": "default"}
            }
        elif invalid == "profile_role":
            overrides["chat"] = {"profile_id": choices["image"]["profile_id"]}
        else:
            overrides["chat"] = {"preset_id": choices["image"]["preset_id"]}
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={
                "text": ORDERED,
                "mode": "auto",
                "confirm_media": True,
                "idempotency_key": "wrong-role",
                "role_overrides": overrides,
            },
        )
        assert response.status_code == 422, response.text
        with SessionLocal() as session:
            assert len(list(session.scalars(select(WorkPlan)))) == 1


async def test_fixed_edit_role_override_remains_explicit_when_it_matches_the_source(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source = await _source(client)
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={
                "text": "Describe a green paper boat",
                "mode": "text",
                "idempotency_key": "fixed-role",
                "role_overrides": {"chat": {"settings": {"temperature": 0.31}, "preset_id": None}},
            },
        )
        assert response.status_code == 202, response.text
        assert response.json()["run"]["settings_json"]["temperature"] == 0.31
        assert response.json()["run"]["provenance_json"]["preset"] is None


def _image_workflow() -> str:
    from local_lm.models import WorkflowDefinition, WorkflowFamily, WorkflowRevision

    with SessionLocal() as session:
        family = WorkflowFamily(name="Role scene")
        definition = WorkflowDefinition(
            family=family, variant_key="create", name="Role scene", operation="text_to_image"
        )
        revision = WorkflowRevision(
            definition=definition,
            version=1,
            engine="mock",
            api_graph_json={},
            input_schema_json={},
            dependencies_json={},
            trusted=True,
        )
        session.add_all([family, definition, revision])
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        return revision.id


@pytest.mark.parametrize("selection", ["selector", "revision", "wrong_role"])
async def test_ordered_role_workflow_is_exact_and_cannot_be_silently_skipped(
    app: FastAPI, client: AsyncClient, selection: str
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source = await _source(client)
        revision_id = _image_workflow()
        override = (
            {
                "workflow_selection": {
                    "selector_capability": "image",
                    "mode": "revision",
                    "workflow_revision_id": revision_id,
                }
            }
            if selection == "selector"
            else {"workflow_revision_id": revision_id}
        )
        role = "chat" if selection == "wrong_role" else "image"
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={
                "text": ORDERED,
                "mode": "auto",
                "confirm_media": True,
                "idempotency_key": "role-workflow",
                "role_overrides": {role: override},
            },
        )
        assert response.status_code == (422 if selection == "wrong_role" else 202), response.text
        with SessionLocal() as session:
            if selection == "wrong_role":
                assert len(list(session.scalars(select(WorkPlan)))) == 1
            else:
                runs = list(
                    session.scalars(
                        select(Run).where(Run.work_plan_id == response.json()["work_plan_id"])
                    )
                )
                assert len(runs) == 4
                for run in runs:
                    assert (run.workflow_revision_id == revision_id) == (
                        run.operation == "text_to_image"
                    )


async def test_role_override_edit_retry_is_bound_to_the_complete_original_request(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source = await _source(client)
        path = f"/api/messages/{source['user_message']['id']}/edits"
        payload = {
            "text": "Explain how paper boats float",
            "mode": "auto",
            "idempotency_key": "role-retry",
            "role_overrides": {"chat": {"settings": {"temperature": 0.31}}},
        }
        first = await client.post(path, json=payload)
        assert first.status_code == 202, first.text
        replay = await client.post(path, json=payload)
        assert replay.status_code == 202, replay.text
        assert replay.json()["run"]["id"] == first.json()["run"]["id"]
        different = await client.post(
            path, json={**payload, "role_overrides": {"chat": {"settings": {"temperature": 0.41}}}}
        )
        assert different.status_code == 409, different.text
        with SessionLocal() as session:
            assert len(list(session.scalars(select(WorkPlan)))) == 2
