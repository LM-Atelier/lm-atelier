"""Saved workflow LoRA edits in chat, project, profile and preset settings."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any, cast

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select

from local_lm.db import SessionLocal
from local_lm.models import Chat, GenerationPreset, ModelProfile
from local_lm.project_dependencies import (
    refuse_workflow_lora_overrides,
    strip_workflow_lora_overrides,
)
from local_lm.settings_registry import WORKFLOW_LORA_OVERRIDES_SETTING_KEY as KEY
from local_lm.workflow_lora_overrides import parse_workflow_lora_overrides
from local_lm.workflow_lora_settings import workflow_lora_overrides_setting_value

RESET: dict[str, object] = {"version": 1, "targets": []}
MALFORMED: dict[str, object] = {"version": 1, "targets": "private marker text"}


def _digest(index: int) -> str:
    return f"{index:064x}"


def _slot(index: int, changes: dict[str, object]) -> dict[str, object]:
    return {
        "slot_id": f"wflora_{_digest(index)}",
        "loader_contract": "comfy-core-lora-loader-v1",
        "loader_authority_sha256": _digest(100 + index),
        "changes": changes,
    }


def _target(index: int, *slots: dict[str, object]) -> dict[str, object]:
    return {
        "workflow_family_id": None,
        "workflow_definition_id": f"workflow-{index}",
        "workflow_variant_key": None,
        "workflow_revision_id": f"wfrev-{index}",
        "slot_contract_version": 1,
        "revision_scope_sha256": _digest(index * 10 + 1),
        "api_graph_sha256": _digest(index * 10 + 2),
        "dependency_contract_sha256": _digest(index * 10 + 3),
        "activation_binding_sha256": _digest(index * 10 + 4),
        "activation_witness_sha256": _digest(index * 10 + 5),
        "overrides": list(slots),
    }


def _envelope(*targets: dict[str, object]) -> dict[str, object]:
    return {"version": 1, "targets": list(targets)}


def _canonical(value: object) -> dict[str, object]:
    return workflow_lora_overrides_setting_value(parse_workflow_lora_overrides(value))


# Targets and slots out of order, and an integer strength, so canonical form is visible.
RAW = _envelope(
    _target(2, _slot(3, {"model_strength": 1}), _slot(2, {"model_strength": 0.5})),
    _target(1, _slot(1, {"clip_strength": 0.4, "model_strength": 0.8})),
)


def _contains_key(value: object) -> bool:
    if isinstance(value, dict):
        return KEY in value or any(_contains_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item) for item in value)
    return False


@pytest.mark.parametrize("role", ["image", "video"])
async def test_chat_and_project_defaults_store_edits_canonically_and_keep_a_reset(
    client: AsyncClient,
    role: str,
) -> None:
    project = await client.post(
        "/api/projects",
        json={"name": f"{role} edits", "generation_settings_json": {role: {KEY: RAW}}},
    )
    assert project.status_code == 201, project.text
    assert project.json()["generation_settings_json"][role][KEY] == _canonical(RAW)

    chat = await client.post(
        "/api/chats",
        json={
            "title": f"{role} edits",
            "project_id": project.json()["id"],
            "generation_settings_json": {role: {KEY: RAW, "steps": 9}},
        },
    )
    assert chat.status_code == 201, chat.text
    assert chat.json()["generation_settings_json"][role] == {KEY: _canonical(RAW), "steps": 9}

    renamed = await client.patch(f"/api/chats/{chat.json()['id']}", json={"title": "Renamed"})
    assert renamed.json()["generation_settings_json"][role][KEY] == _canonical(RAW)

    reset = await client.patch(
        f"/api/chats/{chat.json()['id']}",
        json={"generation_settings_json": {role: {KEY: RESET}}},
    )
    assert reset.status_code == 200, reset.text
    assert reset.json()["generation_settings_json"][role] == {KEY: RESET}

    cleared = await client.patch(
        f"/api/chats/{chat.json()['id']}",
        json={"generation_settings_json": {role: {}}},
    )
    assert cleared.json()["generation_settings_json"][role] == {}


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/chats", {"title": "Bad", "generation_settings_json": {"image": {KEY: MALFORMED}}}),
        ("/api/chats", {"title": "Chat role", "generation_settings_json": {"chat": {KEY: RAW}}}),
        ("/api/projects", {"name": "Bad", "generation_settings_json": {"video": {KEY: MALFORMED}}}),
        ("/api/presets", {"name": "Bad", "role": "image", "settings": {KEY: MALFORMED}}),
        ("/api/presets", {"name": "Chat role", "role": "chat", "settings": {KEY: RAW}}),
        (
            "/api/profiles",
            {"name": "Load", "role": "chat", "engine": "mock", "load_settings": {KEY: RAW}},
        ),
    ],
)
async def test_invalid_or_misplaced_edits_are_refused_without_echo(
    client: AsyncClient,
    path: str,
    payload: dict[str, Any],
) -> None:
    response = await client.post(path, json=payload)

    assert response.status_code == 422
    assert "private marker" not in response.text
    assert "wfrev-" not in response.text


@pytest.mark.parametrize("role", ["image", "video"])
async def test_media_profiles_and_presets_keep_edits_locally_but_never_export_them(
    client: AsyncClient,
    role: str,
) -> None:
    profile = await client.post(
        "/api/profiles",
        json={
            "name": f"{role} profile",
            "role": role,
            "engine": "mock",
            "request_settings": {"steps": 12, KEY: RAW},
        },
    )
    assert profile.status_code == 201, profile.text
    assert profile.json()["request_settings_json"] == {"steps": 12, KEY: _canonical(RAW)}
    updated = await client.patch(
        f"/api/profiles/{profile.json()['id']}",
        json={"request_settings": {"steps": 13, KEY: RESET}},
    )
    assert updated.json()["request_settings_json"] == {"steps": 13, KEY: RESET}
    bundle = (await client.get(f"/api/profiles/{profile.json()['id']}/export")).json()
    assert bundle["request_settings"] == {"steps": 13}

    preset = await client.post(
        "/api/presets",
        json={"name": f"{role} preset", "role": role, "settings": {"steps": 7, KEY: RAW}},
    )
    assert preset.status_code == 201, preset.text
    assert preset.json()["settings_json"] == {"steps": 7, KEY: _canonical(RAW)}
    clone = await client.post(f"/api/presets/{preset.json()['id']}/clone", json={})
    assert clone.json()["settings_json"] == {"steps": 7, KEY: _canonical(RAW)}
    preset_bundle = (await client.get(f"/api/presets/{preset.json()['id']}/export")).json()
    assert preset_bundle["settings"] == {"steps": 7}

    with SessionLocal() as session:
        profiles_before = session.scalar(select(func.count(ModelProfile.id)))
        presets_before = session.scalar(select(func.count(GenerationPreset.id)))
    refused_profile = await client.post(
        "/api/profiles/import",
        json={**bundle, "request_settings": {"steps": 13, KEY: RAW}},
    )
    refused_preset = await client.post(
        "/api/presets/import",
        json={**preset_bundle, "settings": {"steps": 7, KEY: RAW}},
    )
    assert refused_profile.status_code == refused_preset.status_code == 422
    with SessionLocal() as session:
        assert session.scalar(select(func.count(ModelProfile.id))) == profiles_before
        assert session.scalar(select(func.count(GenerationPreset.id))) == presets_before


async def test_deleting_a_bound_preset_keeps_its_edits_below_the_chat_own_field_by_field(
    client: AsyncClient,
) -> None:
    preset_edits = _envelope(
        _target(1, _slot(1, {"model_strength": 0.3, "clip_strength": 0.2})),
        _target(2, _slot(2, {"model_strength": 0.9})),
    )
    chat_edits = _envelope(_target(1, _slot(1, {"model_strength": 0.6})))
    preset = await client.post(
        "/api/presets",
        json={"name": "Bound edits", "role": "image", "settings": {"steps": 5, KEY: preset_edits}},
    )
    chat = await client.post(
        "/api/chats",
        json={
            "title": "Folded edits",
            "generation_preset_ids_json": {"image": preset.json()["id"]},
            "generation_settings_json": {"image": {"steps": 8, KEY: chat_edits}},
        },
    )
    assert chat.status_code == 201, chat.text

    deleted = await client.delete(f"/api/presets/{preset.json()['id']}")

    assert deleted.status_code == 204
    folded = (await client.get(f"/api/chats/{chat.json()['id']}")).json()
    image = folded["generation_settings_json"]["image"]
    assert image["steps"] == 8
    assert image[KEY] == _canonical(
        _envelope(
            _target(1, _slot(1, {"model_strength": 0.6, "clip_strength": 0.2})),
            _target(2, _slot(2, {"model_strength": 0.9})),
        )
    )


async def test_a_chat_reset_survives_folding_in_a_deleted_preset(client: AsyncClient) -> None:
    preset = await client.post(
        "/api/presets",
        json={"name": "Hidden edits", "role": "image", "settings": {KEY: RAW}},
    )
    chat = await client.post(
        "/api/chats",
        json={
            "title": "Reset chat",
            "generation_preset_ids_json": {"image": preset.json()["id"]},
            "generation_settings_json": {"image": {KEY: RESET}},
        },
    )
    assert chat.status_code == 201, chat.text

    assert (await client.delete(f"/api/presets/{preset.json()['id']}")).status_code == 204

    folded = (await client.get(f"/api/chats/{chat.json()['id']}")).json()
    assert folded["generation_settings_json"]["image"][KEY] == RESET


async def test_project_archives_leave_edits_behind_and_refuse_archives_that_carry_them(
    client: AsyncClient,
) -> None:
    project = (
        await client.post(
            "/api/projects",
            json={"name": "Archived edits", "generation_settings_json": {"image": {KEY: RAW}}},
        )
    ).json()
    await client.post(
        "/api/chats",
        json={
            "title": "Archived chat",
            "project_id": project["id"],
            "generation_settings_json": {"image": {"steps": 4, KEY: RAW}},
        },
    )

    exported = (await client.post(f"/api/projects/{project['id']}/export")).json()
    archive = await client.get(exported["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        manifest = cast(dict[str, Any], json.loads(bundle.read("manifest.json")))
        members = {info.filename: bundle.read(info) for info in bundle.infolist()}

    assert not _contains_key(manifest)
    assert manifest["chats"][0]["generation_settings_json"]["image"] == {"steps": 4}
    imported = await client.post(
        "/api/projects/import",
        files={"archive": ("clean.lm-atelier.zip", archive.content, "application/zip")},
    )
    assert imported.status_code == 201, imported.text

    manifest["chats"][0]["generation_settings_json"]["image"][KEY] = RAW
    poisoned = io.BytesIO()
    with zipfile.ZipFile(poisoned, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, json.dumps(manifest) if name == "manifest.json" else data)
    with SessionLocal() as session:
        chats_before = session.scalar(select(func.count(Chat.id)))
    refused = await client.post(
        "/api/projects/import",
        files={"archive": ("edits.lm-atelier.zip", poisoned.getvalue(), "application/zip")},
    )
    assert refused.status_code == 422
    with SessionLocal() as session:
        assert session.scalar(select(func.count(Chat.id))) == chats_before


def test_archive_helpers_strip_and_refuse_edits_at_any_depth() -> None:
    nested = {"a": [{"b": {KEY: RAW, "keep": 1}}], KEY: RESET, "c": "text"}

    stripped = strip_workflow_lora_overrides(nested)

    assert stripped == {"a": [{"b": {"keep": 1}}], "c": "text"}
    assert KEY in nested
    refuse_workflow_lora_overrides(stripped)
    with pytest.raises(ValueError, match="workflow LoRA edits"):
        refuse_workflow_lora_overrides(nested)
