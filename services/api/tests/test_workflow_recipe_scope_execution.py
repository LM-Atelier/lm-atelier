"""Keep Automatic recipe choices outside recipe admission and freeze their absence."""

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import Run
from local_lm.schemas import SettingField
from local_lm.workflow_use_case_execution import prepare_workflow_use_case_execution
from local_lm.workflow_use_cases_v1 import (
    SelectionApplication,
    WorkflowUseCaseInputs,
    classify_workflow_use_case,
)


@pytest.mark.parametrize("scope", ["chats", "projects"])
@pytest.mark.parametrize(
    "facts",
    [
        WorkflowUseCaseInputs(Operation.TEXT),
        WorkflowUseCaseInputs(Operation.TEXT_TO_IMAGE),
        WorkflowUseCaseInputs(Operation.IMAGE_TO_IMAGE, source_present=True),
        WorkflowUseCaseInputs(
            Operation.IMAGE_TO_IMAGE,
            source_present=True,
            selection=SelectionApplication.WORKFLOW,
        ),
        WorkflowUseCaseInputs(Operation.IMAGE_TO_IMAGE, source_present=True, extend=True),
        WorkflowUseCaseInputs(Operation.IMAGE_TO_IMAGE, source_present=True, upscale=True),
        WorkflowUseCaseInputs(Operation.TEXT_TO_VIDEO),
        WorkflowUseCaseInputs(Operation.IMAGE_TO_VIDEO, source_present=True),
    ],
)
async def test_automatic_recipe_choices_do_not_probe_or_admit_a_recipe(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    facts: WorkflowUseCaseInputs,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Recipe inheritance"})).json()
    chat = (
        await client.post(
            "/api/chats", json={"title": "Automatic recipe", "project_id": project["id"]}
        )
    ).json()
    use_case = classify_workflow_use_case(facts).use_case.value
    preset = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Workspace recipe",
            "use_case": use_case,
            "settings_json": {"unsupported_setting": 1},
            "is_default": True,
        },
    )
    assert preset.status_code == 201, preset.text
    scope_id = chat["id"] if scope == "chats" else project["id"]
    chosen = await client.put(
        f"/api/{scope}/{scope_id}/workflow-use-case-presets/{use_case}",
        json={"mode": "automatic"},
    )
    assert chosen.status_code == 200, chosen.text

    async def unavailable(role: str) -> list[SettingField]:
        raise AssertionError("Automatic recipes do not require an engine settings probe")

    monkeypatch.setattr(app.state.services.engines, "settings_for_role", unavailable)
    assert (
        await prepare_workflow_use_case_execution(
            SessionLocal, app.state.services.engines, facts, chat_id=chat["id"]
        )
        is None
    )


@pytest.mark.parametrize("scope", ["chats", "projects"])
@pytest.mark.parametrize("use_case,mode", [("chat", "text"), ("image_generation", "image")])
async def test_automatic_turn_and_its_edit_keep_recipe_absence_after_scope_changes(
    app: FastAPI,
    client: AsyncClient,
    scope: str,
    use_case: str,
    mode: str,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Accepted choices"})).json()
    chat = (
        await client.post(
            "/api/chats", json={"title": "Accepted absence", "project_id": project["id"]}
        )
    ).json()
    preset = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Incompatible workspace recipe",
            "use_case": use_case,
            "settings_json": {"unsupported_setting": 1},
            "is_default": True,
        },
    )
    assert preset.status_code == 201, preset.text
    scope_id = chat["id"] if scope == "chats" else project["id"]
    scope_url = f"/api/{scope}/{scope_id}/workflow-use-case-presets/{use_case}"
    assert (await client.put(scope_url, json={"mode": "automatic"})).status_code == 200
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat['id']}/turns", json={"text": "A blue square", "mode": mode}
        )
        assert accepted.status_code == 202, accepted.text
        with SessionLocal() as session:
            original = session.get(Run, accepted.json()["run"]["id"])
            assert original is not None
            assert original.provenance_json.get("workflow_use_case_preset") is None
        assert (await client.put(scope_url, json={"mode": "inherit"})).status_code == 200
        edited = await client.post(
            f"/api/messages/{accepted.json()['user_message']['id']}/edits",
            json={"text": "A green square", "idempotency_key": "keep-recipe-absence"},
        )
        assert edited.status_code == 202, edited.text
        with SessionLocal() as session:
            repeated = session.get(Run, edited.json()["run"]["id"])
            assert repeated is not None
            assert repeated.provenance_json.get("workflow_use_case_preset") is None
            assert "unsupported_setting" not in repeated.settings_json


@pytest.mark.parametrize(
    "project_choice,chat_choice,expected_scope,expected_steps",
    [
        ("inherit", "inherit", "workspace", 7),
        ("preset", "inherit", "project", 11),
        ("preset", "preset", "chat", 13),
    ],
)
async def test_real_turn_resolves_chat_project_workspace_recipe_precedence(
    app: FastAPI,
    client: AsyncClient,
    project_choice: str,
    chat_choice: str,
    expected_scope: str,
    expected_steps: int,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Layered recipes"})).json()
    chat = (
        await client.post(
            "/api/chats", json={"title": "Layered settings", "project_id": project["id"]}
        )
    ).json()
    ids = {}
    for scope, steps in [("workspace", 7), ("project", 11), ("chat", 13)]:
        created = await client.post(
            "/api/workflow-use-case-presets",
            json={
                "name": f"Example {scope} recipe",
                "use_case": "image_generation",
                "settings_json": {"steps": steps},
                "is_default": scope == "workspace",
            },
        )
        assert created.status_code == 201, created.text
        ids[scope] = created.json()["id"]
    for collection, target, choice, recipe_id in [
        ("projects", project["id"], project_choice, ids["project"]),
        ("chats", chat["id"], chat_choice, ids["chat"]),
    ]:
        payload = {"mode": choice, **({"preset_id": recipe_id} if choice == "preset" else {})}
        chosen = await client.put(
            f"/api/{collection}/{target}/workflow-use-case-presets/image_generation", json=payload
        )
        assert chosen.status_code == 200, chosen.text
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat['id']}/turns", json={"text": "A blue square", "mode": "image"}
        )
        assert accepted.status_code == 202, accepted.text
        with SessionLocal() as session:
            run = session.get(Run, accepted.json()["run"]["id"])
            assert run is not None and run.settings_json["steps"] == expected_steps
            receipt = run.provenance_json["workflow_use_case_preset"]
            assert receipt["scope"] == expected_scope
            assert receipt["preset_id"] == ids[expected_scope]
