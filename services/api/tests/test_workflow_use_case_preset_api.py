"""Exercise recipe management through authenticated HTTP requests."""

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import Chat, ChatWorkflowUseCaseSelection, Project, WorkflowUseCasePreset

ROOT = "/api/workflow-use-case-presets"
DEFAULT = "/api/workflow-use-case-defaults/image_generation"


async def _create(client: AsyncClient, name: str = "Example", **values: object) -> dict:
    response = await client.post(
        ROOT,
        json={"name": name, "use_case": "image_generation", "settings_json": {"seed": 9}, **values},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["prompt", "negative_prompt"])
async def test_prompt_keys_are_refused_without_echoing_or_saving_the_value(
    client: AsyncClient, key: str
) -> None:
    existing = await _create(client)
    value = "constructed recipe boundary fixture"
    payload = {"name": "Example", "use_case": "image_generation", "settings_json": {key: value}}
    for response in [
        await client.post(ROOT, json=payload),
        await client.put(f"{ROOT}/{existing['id']}", json=payload),
    ]:
        assert response.status_code == 422
        assert value not in response.text
    assert (await client.get(ROOT)).json() == [existing]
    with SessionLocal() as session:
        stored = session.get(WorkflowUseCasePreset, existing["id"])
        assert stored is not None
        stored.settings_json = {key: value}
        session.commit()
    response = await client.get(f"{ROOT}/{existing['id']}")
    assert response.status_code == 409
    assert value not in response.text


@pytest.mark.asyncio
async def test_recipe_lifecycle_and_workspace_default(client: AsyncClient) -> None:
    created = await _create(client)
    recipe_id = created["id"]
    assert not created["builtin"] and created["enabled"] and not created["is_default"]
    listed = await client.get(ROOT, params={"use_case": "image_generation", "limit": 1})
    assert listed.status_code == 200 and listed.json() == [created]
    assert (await client.get(f"{ROOT}/{recipe_id}")).json() == created
    assert (await client.get(DEFAULT)).json() == {"preset_id": None}
    saved = await client.put(DEFAULT, json={"preset_id": recipe_id})
    assert saved.status_code == 200 and saved.json() == {"preset_id": recipe_id}
    changed = await client.put(
        f"{ROOT}/{recipe_id}",
        json={
            "name": "Renamed",
            "use_case": "image_generation",
            "settings_json": {"seed": 12},
            "is_default": True,
        },
    )
    assert changed.status_code == 200 and changed.json()["settings_json"] == {"seed": 12}
    assert (await client.delete(f"{ROOT}/{recipe_id}")).status_code == 409
    assert (await client.put(DEFAULT, json={"preset_id": None})).status_code == 200
    assert (await client.delete(f"{ROOT}/{recipe_id}")).status_code == 204
    missing = await client.get(f"{ROOT}/{recipe_id}")
    assert missing.status_code == 404
    assert missing.json()["code"] == "workflow-use-case-preset-not-found"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["chats", "projects"])
async def test_scoped_choices_remain_distinct_without_legacy_mirrors(
    client: AsyncClient, scope: str
) -> None:
    with SessionLocal() as session:
        session.add_all([Chat(id="chat"), Project(id="project", name="Example")])
        session.commit()
    recipe = await _create(client)
    scope_id = "chat" if scope == "chats" else "project"
    path = f"/api/{scope}/{scope_id}/workflow-use-case-presets/image_generation"
    assert (await client.get(path)).json() == {"mode": "inherit"}
    for choice in [
        {"mode": "automatic"},
        {"mode": "preset", "preset_id": recipe["id"]},
        {"mode": "inherit"},
    ]:
        saved = await client.put(path, json=choice)
        assert saved.status_code == 200 and saved.json() == choice
        assert (await client.get(path)).json() == choice
    with SessionLocal() as session:
        chat = session.get(Chat, "chat")
        project = session.get(Project, "project")
        assert chat is not None and project is not None
        assert chat.generation_preset_ids_json == project.generation_preset_ids_json == {}


@pytest.mark.asyncio
async def test_selected_recipe_changes_fail_with_stable_errors(client: AsyncClient) -> None:
    with SessionLocal() as session:
        session.add(Chat(id="chat"))
        session.commit()
    recipe = await _create(client)
    path = "/api/chats/chat/workflow-use-case-presets/image_generation"
    assert (
        await client.put(path, json={"mode": "preset", "preset_id": recipe["id"]})
    ).status_code == 200
    disabled = await client.put(
        f"{ROOT}/{recipe['id']}",
        json={"name": "Example", "use_case": "image_generation", "enabled": False},
    )
    deleted = await client.delete(f"{ROOT}/{recipe['id']}")
    for response in [disabled, deleted]:
        assert response.status_code == 409
        assert response.json()["code"] == "workflow-use-case-preset-in-use"
    assert (await client.get(path)).json() == {"mode": "preset", "preset_id": recipe["id"]}


@pytest.mark.asyncio
async def test_built_in_default_is_selectable_but_content_is_immutable(client: AsyncClient) -> None:
    with SessionLocal() as session:
        session.add(
            WorkflowUseCasePreset(
                id="builtin", name="Example", use_case="image_generation", builtin=True
            )
        )
        session.commit()
    assert (await client.put(DEFAULT, json={"preset_id": "builtin"})).status_code == 200
    changed = await client.put(
        f"{ROOT}/builtin", json={"name": "Changed", "use_case": "image_generation"}
    )
    deleted = await client.delete(f"{ROOT}/builtin")
    for response in [changed, deleted]:
        assert response.status_code == 409
        assert response.json()["code"] == "workflow-use-case-preset-builtin"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice",
    [
        {"mode": "automatic", "preset_id": "example"},
        {"mode": "preset"},
        {"mode": "inherit", "extra": "neutral-sentinel"},
    ],
)
async def test_invalid_choices_do_not_echo_or_write_input(
    client: AsyncClient, choice: dict
) -> None:
    response = await client.put(
        "/api/chats/missing/workflow-use-case-presets/image_generation", json=choice
    )
    assert response.status_code == 422
    assert response.json() == {
        "code": "request-validation-invalid",
        "detail": "Request is invalid.",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["chats", "projects"])
async def test_missing_scopes_refuse_inheritance_and_changes(
    client: AsyncClient, scope: str
) -> None:
    path = f"/api/{scope}/missing/workflow-use-case-presets/image_generation"
    responses = [await client.get(path), await client.put(path, json={"mode": "inherit"})]
    for response in responses:
        assert response.status_code == 404
        assert response.json()["code"] == "workflow-use-case-scope-not-found"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query", [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"use_case": "missing"}]
)
async def test_recipe_listing_rejects_invalid_bounds_and_cases(
    client: AsyncClient, query: dict
) -> None:
    response = await client.get(ROOT, params=query)
    assert response.status_code == 422
    assert response.json()["code"] == "request-validation-invalid"


@pytest.mark.asyncio
async def test_recipe_mutations_require_the_session_csrf_token(client: AsyncClient) -> None:
    token = client.headers.pop("x-local-lm-csrf")
    try:
        response = await client.post(ROOT, json={"name": "Example", "use_case": "image_generation"})
        assert response.status_code == 403
    finally:
        client.headers["x-local-lm-csrf"] = token
    assert (await client.get(ROOT)).json() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_id", ["neutral-sentinel-" + "x" * 50, " neutral-sentinel "])
async def test_corrupt_stored_identifiers_refuse_without_echoing_values(
    client: AsyncClient,
    invalid_id: str,
) -> None:
    with SessionLocal() as session:
        session.add(Chat(id="chat"))
        session.add(
            WorkflowUseCasePreset(
                id=invalid_id, name="Example", use_case="image_generation", is_default=True
            )
        )
        session.commit()
        session.add(
            ChatWorkflowUseCaseSelection(
                chat_id="chat", use_case="image_generation", preset_id=invalid_id
            )
        )
        session.commit()
    for path in [
        f"{ROOT}/{invalid_id}",
        ROOT,
        DEFAULT,
        "/api/chats/chat/workflow-use-case-presets/image_generation",
    ]:
        response = await client.get(path)
        assert response.status_code == 409
        assert response.json()["code"] == "workflow-use-case-preset-invalid"
        assert "neutral-sentinel" not in response.text
