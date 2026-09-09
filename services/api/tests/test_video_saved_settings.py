"""Saved video settings remain usable across the setting rename."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient

from local_lm import db
from local_lm.models import Chat, Project

#: What a person had stored before the rename. The value is a legal video
#: guidance; only its key is retired.
LEGACY_VIDEO_DEFAULTS = {"video": {"guidance": 6}}


@pytest.mark.anyio
async def test_a_saved_video_guidance_does_not_block_chat_defaults(
    client: AsyncClient,
) -> None:
    """Editing image defaults must not fail because video holds a retired key."""

    created = await client.post("/api/chats", json={"title": "pre-upgrade chat"})
    assert created.status_code < 300, created.text
    chat_id = created.json()["id"]

    # Seeded directly because the rename also refuses to WRITE the retired key:
    # a stored value can only pre-date the change, so that is how it is created.
    with db.SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.generation_settings_json = dict(LEGACY_VIDEO_DEFAULTS)
        session.commit()

    stored = (await client.get(f"/api/chats/{chat_id}")).json()["generation_settings_json"]
    assert stored["video"].get("cfg", stored["video"].get("guidance")) == 6

    saved = await client.patch(
        f"/api/chats/{chat_id}",
        json={"generation_settings_json": {**stored, "image": {"cfg": 7}}},
    )
    assert saved.status_code < 300, (
        f"a saved video guidance blocked an unrelated image save: {saved.status_code} {saved.text}"
    )

    settled = (await client.get(f"/api/chats/{chat_id}")).json()["generation_settings_json"]
    assert settled.get("image", {}).get("cfg") == 7, settled


@pytest.mark.anyio
async def test_a_saved_video_guidance_does_not_block_project_defaults(
    client: AsyncClient,
) -> None:
    """The same, for a project - it holds the identical per-role shape."""

    created = await client.post("/api/projects", json={"name": "pre-upgrade project"})
    assert created.status_code < 300, created.text
    project_id = created.json()["id"]

    with db.SessionLocal() as session:
        project = session.get(Project, project_id)
        assert project is not None
        project.generation_settings_json = dict(LEGACY_VIDEO_DEFAULTS)
        session.commit()

    # There is no per-project GET; the browser reads projects from the list, so
    # the control does the same.
    listed = (await client.get("/api/projects")).json()
    stored = next(row["generation_settings_json"] for row in listed if row["id"] == project_id)
    assert stored["video"].get("cfg", stored["video"].get("guidance")) == 6

    saved = await client.patch(
        f"/api/projects/{project_id}",
        json={"generation_settings_json": {**stored, "image": {"cfg": 7}}},
    )
    assert saved.status_code < 300, (
        "a saved video guidance blocked an unrelated project save: "
        f"{saved.status_code} {saved.text}"
    )


@pytest.mark.anyio
async def test_an_explicit_cfg_still_wins_over_a_retired_guidance(
    client: AsyncClient,
) -> None:
    """Tolerating the retired key must not let it outrank the current one."""

    created = await client.post("/api/chats", json={"title": "both keys stored"})
    assert created.status_code < 300, created.text
    chat_id = created.json()["id"]

    with db.SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.generation_settings_json = {"video": {"guidance": 6, "cfg": 9}}
        session.commit()

    stored = (await client.get(f"/api/chats/{chat_id}")).json()["generation_settings_json"]
    saved = await client.patch(
        f"/api/chats/{chat_id}",
        json={"generation_settings_json": {**stored, "image": {"cfg": 7}}},
    )
    assert saved.status_code < 300, f"{saved.status_code} {saved.text}"

    settled = (await client.get(f"/api/chats/{chat_id}")).json()["generation_settings_json"]
    video = settled.get("video", {})
    assert video.get("cfg") == 9, (
        f"the explicit cfg stopped winning once guidance was tolerated: {video}"
    )


@pytest.mark.anyio
async def test_the_panel_and_the_dispatch_agree_about_a_stored_guidance(
    client: AsyncClient,
) -> None:
    """What the settings panel shows and what the generation uses must agree."""

    from local_lm.settings_registry import VIDEO_SETTINGS, compatible_stored_settings

    created = await client.post("/api/chats", json={"title": "stored guidance"})
    assert created.status_code < 300, created.text
    chat_id = created.json()["id"]

    with db.SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.generation_settings_json = {"video": {"guidance": 5.5}}
        session.commit()

    exposed = (await client.get(f"/api/chats/{chat_id}")).json()
    video = exposed["generation_settings_json"].get("video", {})

    cfg_field = next(field for field in VIDEO_SETTINGS if field.key == "cfg")
    request_fields = [field for field in VIDEO_SETTINGS if field.scope != "load"]

    # What the panel would render: the field's own key, or its default.
    displayed = video.get(cfg_field.key, cfg_field.default)
    # What a generation would actually use.
    dispatched = compatible_stored_settings(video, request_fields).get(
        cfg_field.key, cfg_field.default
    )

    assert displayed == dispatched, (
        f"the panel would show {displayed} while a generation uses {dispatched}"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("resource,identity", [("chats", "title"), ("projects", "name")])
async def test_old_client_can_save_video_defaults(
    client: AsyncClient, resource: str, identity: str
) -> None:
    created = await client.post(
        f"/api/{resource}",
        json={identity: "Saved defaults", "generation_settings_json": {"video": {"guidance": 5.5}}},
    )
    assert created.status_code == 201, created.text
    assert created.json()["generation_settings_json"]["video"] == {"cfg": 5.5}
    saved = await client.patch(
        f"/api/{resource}/{created.json()['id']}",
        json={"generation_settings_json": {"video": {"guidance": 4, "cfg": 9}}},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["generation_settings_json"]["video"] == {"cfg": 9}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "saved_values",
    [
        {"guidance": "invalid"},
        {"guidance": 5, "cfg": None},
        {"guidance": 5, "cfg": -1},
        {"guidance": 5, "cfg": True},
        {"guidance": 5, "unknown_setting": 2},
    ],
)
async def test_retired_video_key_does_not_relax_validation(
    client: AsyncClient, saved_values: dict
) -> None:
    response = await client.post(
        "/api/chats",
        json={"title": "Invalid defaults", "generation_settings_json": {"video": saved_values}},
    )
    assert response.status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize("role", ["chat", "image"])
async def test_other_roles_do_not_accept_the_retired_video_key(
    client: AsyncClient, role: str
) -> None:
    response = await client.post(
        "/api/chats",
        json={"title": "Other role", "generation_settings_json": {role: {"guidance": 5}}},
    )
    assert response.status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize("resource", ["profiles", "presets"])
async def test_saved_video_catalog_settings_read_clone_export_and_update(
    client: AsyncClient, resource: str
) -> None:
    from local_lm.models import GenerationPreset, ModelProfile

    profile = resource == "profiles"
    field = "request_settings" if profile else "settings"
    stored_field = field + "_json"
    payload = {"name": "Saved video", "role": "video", field: {"guidance": 5.5}}
    if profile:
        payload["engine"] = "mock"
    response = await client.post(f"/api/{resource}", json=payload)
    assert response.status_code == 201, response.text
    row_id = response.json()["id"]
    assert response.json()[stored_field] == {"cfg": 5.5}
    model = ModelProfile if profile else GenerationPreset
    with db.SessionLocal() as session:
        row = session.get(model, row_id)
        assert row is not None
        setattr(row, stored_field, {"guidance": 5.5})
        session.commit()
    listed = (await client.get(f"/api/{resource}")).json()
    assert next(row for row in listed if row["id"] == row_id)[stored_field] == {"cfg": 5.5}
    exported = await client.get(f"/api/{resource}/{row_id}/export")
    assert exported.status_code == 200, exported.text
    assert exported.json()[field] == {"cfg": 5.5}
    with db.SessionLocal() as session:
        row = session.get(model, row_id)
        assert row is not None
        assert getattr(row, stored_field) == {"guidance": 5.5}
    cloned = await client.post(f"/api/{resource}/{row_id}/clone", json={})
    assert cloned.status_code == 201, cloned.text
    assert cloned.json()[stored_field] == {"cfg": 5.5}
    updated = await client.patch(f"/api/{resource}/{row_id}", json={field: {"guidance": 7.5}})
    assert updated.status_code == 200, updated.text
    assert updated.json()[stored_field] == {"cfg": 7.5}
    invalid = await client.patch(
        f"/api/{resource}/{row_id}", json={field: {"guidance": 5, "cfg": None}}
    )
    assert invalid.status_code == 422


@pytest.mark.anyio
async def test_video_direct_defaults_win_when_preset_is_deleted_or_exported(
    client: AsyncClient,
) -> None:
    from local_lm.exports import ProjectExporter
    from local_lm.models import GenerationPreset
    from local_lm.project_dependencies import DependencySourceIndex

    created = await client.post("/api/chats", json={"title": "Layered defaults"})
    chat_id = created.json()["id"]
    with db.SessionLocal() as session:
        preset = GenerationPreset(name="Video preset", role="video", settings_json={"cfg": 4})
        session.add(preset)
        session.flush()
        preset_id = preset.id
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.generation_settings_json = {"video": {"guidance": 7.5}}
        chat.generation_preset_ids_json = {"video": preset_id}
        session.commit()
        record: dict = {}
        ProjectExporter._snapshot_generation_defaults(
            session, chat, record, DependencySourceIndex({}, {}, {}, {})
        )
        assert record["generation_settings_json"]["video"] == {"cfg": 7.5}
    deleted = await client.delete(f"/api/presets/{preset_id}")
    assert deleted.status_code == 204, deleted.text
    settled = (await client.get(f"/api/chats/{chat_id}")).json()
    assert settled["generation_settings_json"]["video"] == {"cfg": 7.5}


@pytest.mark.anyio
async def test_portable_video_dependencies_reuse_saved_legacy_settings(client: AsyncClient) -> None:
    from local_lm.models import GenerationPreset, ModelProfile
    from local_lm.project_dependencies import PortableDependencies, install_dependency_manifest

    with db.SessionLocal() as session:
        profile = ModelProfile(
            name="Portable video",
            use_case="",
            role="video",
            engine="mock",
            load_settings_json={},
            request_settings_json={"guidance": 5.5},
        )
        preset = GenerationPreset(
            name="Portable video", role="video", settings_json={"guidance": 7.5}
        )
        session.add_all([profile, preset])
        session.flush()
        dependencies = PortableDependencies.model_validate(
            {
                "profiles": [
                    {
                        "source_id": "old-profile",
                        "name": profile.name,
                        "role": "video",
                        "engine": "mock",
                        "request_settings": {"guidance": 5.5},
                    }
                ],
                "presets": [
                    {
                        "source_id": "old-preset",
                        "name": preset.name,
                        "role": "video",
                        "settings": {"guidance": 7.5},
                    }
                ],
                "workflows": [],
            }
        )
        assert dependencies.profiles[0].request_settings == {"cfg": 5.5}
        imported = install_dependency_manifest(session, dependencies)
        assert imported.profile_ids["old-profile"] == profile.id
        assert imported.preset_ids["old-preset"] == preset.id
        assert profile.request_settings_json == {"guidance": 5.5}
