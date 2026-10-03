"""Workspace cleanup preserves recoverable defaults and restores usable selections."""

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_recovery_lifecycle_boundaries import _restore, _trash

from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.models import Chat, ModelInstall, ModelProfile, Project
from local_lm.profile_service import retire_profiles_for_installs
from local_lm.workflow_compatibility import (
    ensure_legacy_profile_workflow,
    mirror_legacy_chat_workflow_selections,
)


def _bound_profile(app: FastAPI, capability: str = "chat") -> tuple[str, str, str]:
    path = app.state.services.settings.model_dir / "neutral-recovery-model.gguf"
    path.write_bytes(b"neutral model fixture")
    with SessionLocal() as session:
        install = ModelInstall(
            name="Recovery model",
            role="chat" if capability == "vision" else capability,
            engine="mock",
            local_path=str(path),
            active=True,
            manifest_json={"remote_id": "example/recovery-model"},
        )
        session.add(install)
        session.flush()
        profile = ModelProfile(
            name="Recovery profile",
            role=install.role,
            engine="mock",
            model_install_id=install.id,
        )
        session.add(profile)
        session.flush()
        ensure_legacy_profile_workflow(session, profile)
        chat = Chat(title="Recovery defaults")
        setattr(chat, f"active_{capability}_profile_id", profile.id)
        session.add(chat)
        session.flush()
        mirror_legacy_chat_workflow_selections(session, chat, [capability])
        session.commit()
        return install.id, profile.id, chat.id


@pytest.mark.parametrize("capability", ["chat", "vision", "image", "video"])
async def test_profile_deletion_explains_its_recovery_dependency_and_preserves_restore(
    app: FastAPI,
    client: AsyncClient,
    capability: str,
) -> None:
    _install_id, profile_id, chat_id = _bound_profile(app, capability)
    item = await _trash(client, f"/api/chats/{chat_id}", "trash-profile-dependent")
    response = await client.delete(f"/api/profiles/{profile_id}")
    assert response.status_code == 409
    assert response.json()["code"] == "profile-used-in-recently-deleted"
    with SessionLocal() as session:
        assert session.get(ModelProfile, profile_id) is not None
        assert getattr(session.get(Chat, chat_id), f"active_{capability}_profile_id") == profile_id
    await _restore(client, item["deletion_id"])
    response = await client.delete(f"/api/profiles/{profile_id}")
    assert response.status_code == 204, response.text
    with SessionLocal() as session:
        assert getattr(session.get(Chat, chat_id), f"active_{capability}_profile_id") == "__auto__"


async def test_model_deletion_keeps_profiles_required_by_a_recoverable_chat(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    install_id, profile_id, chat_id = _bound_profile(app)
    item = await _trash(client, f"/api/chats/{chat_id}", "trash-model-dependent")
    response = await client.delete(f"/api/models/{install_id}?delete_profiles=true")
    assert response.status_code == 409
    assert response.json()["code"] == "model-used-in-recently-deleted"
    with SessionLocal() as session:
        assert session.get(ModelInstall, install_id) is not None
        assert session.get(ModelProfile, profile_id) is not None
    await _restore(client, item["deletion_id"])
    response = await client.delete(f"/api/models/{install_id}?delete_profiles=true")
    assert response.status_code == 204, response.text


@pytest.mark.parametrize("kind", ["chat", "project"])
async def test_preset_deletion_preserves_a_hidden_binding_until_restore(
    client: AsyncClient,
    kind: str,
) -> None:
    preset = (
        await client.post(
            "/api/presets",
            json={"name": "Recovery preset", "role": "chat", "settings": {"temperature": 0.2}},
        )
    ).json()
    path = "/api/chats" if kind == "chat" else "/api/projects"
    owner = (
        await client.post(
            path, json={"title": "Preset chat"} if kind == "chat" else {"name": "Preset project"}
        )
    ).json()
    response = await client.patch(
        f"{path}/{owner['id']}", json={"generation_preset_ids_json": {"chat": preset["id"]}}
    )
    assert response.status_code == 200, response.text
    item = await _trash(client, f"{path}/{owner['id']}", "trash-preset-dependent")
    response = await client.delete(f"/api/presets/{preset['id']}")
    assert response.status_code == 409
    assert response.json()["code"] == "preset-used-in-recently-deleted"
    await _restore(client, item["deletion_id"])
    response = await client.delete(f"/api/presets/{preset['id']}")
    assert response.status_code == 204, response.text
    with SessionLocal() as session:
        resource = session.get(Chat if kind == "chat" else Project, owner["id"])
        assert resource is not None
        assert "chat" not in resource.generation_preset_ids_json
        assert resource.generation_settings_json["chat"]["temperature"] == 0.2


@pytest.mark.parametrize("replacement", [False, True])
async def test_model_retirement_skips_hidden_chats_and_restore_repairs_their_defaults(
    app: FastAPI,
    client: AsyncClient,
    replacement: bool,
) -> None:
    install_id, profile_id, chat_id = _bound_profile(app)
    item = await _trash(client, f"/api/chats/{chat_id}", "trash-retired-default")
    with SessionLocal() as session:
        if replacement:
            current = ModelInstall(
                name="Replacement",
                role="chat",
                engine="mock",
                local_path="replacement",
                active=True,
            )
            session.add(current)
            session.flush()
            profile = ModelProfile(
                name="Replacement", role="chat", engine="mock", model_install_id=current.id
            )
            session.add(profile)
            session.flush()
            retired = DownloadManager._deactivate_superseded_chat_installs(
                session, current, profile, "example/recovery-model"
            )
            assert retired == [install_id]
        else:
            session.get(ModelInstall, install_id).active = False
            retired = retire_profiles_for_installs(session, [install_id])
            assert retired == [profile_id]
        session.commit()
        assert session.get(Chat, chat_id).active_chat_profile_id == profile_id
    await _restore(client, item["deletion_id"])
    metadata: dict[str, Any] = (await client.get(f"/api/chats/{chat_id}/metadata")).json()
    assert metadata["active_chat_profile_id"] == "__auto__"


async def test_profile_deletion_does_not_delete_a_recoverable_compatibility_family(
    client: AsyncClient,
) -> None:
    from test_workflow_recovery_legacy_producers import _deleted

    profile_id, family_id = await _deleted(client, "trash")
    response = await client.delete(f"/api/profiles/{profile_id}")
    assert response.status_code == 409
    assert response.json()["code"] == "profile-used-in-recently-deleted"
    assert (await client.get(f"/api/workflow-families/{family_id}")).status_code == 404
