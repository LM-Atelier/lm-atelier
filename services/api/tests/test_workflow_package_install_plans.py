"""A preflight binds raw source and exact costs without creating installation work."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_workflow_asset_install_api import _seed_plan, _ui_graph

from local_lm import models
from local_lm.config import Settings
from local_lm.db import SessionLocal

pytestmark = pytest.mark.asyncio
URL = "/api/workflows/packages/install-plans"


@pytest.fixture(autouse=True)
def runtime_inventory(app: FastAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_runtime(app.state.services.settings, tmp_path)

    async def object_info() -> dict[str, Any]:
        return {"LoraLoader": {}, "EmptyLatentImage": {}}

    monkeypatch.setattr(app.state.services.engines.media, "object_info", object_info, raising=False)


def configure_runtime(settings: Settings, tmp_path: Path) -> None:
    directory = tmp_path / "configured-workflow-runtime"
    directory.mkdir(exist_ok=True)
    executable = directory / "python.exe"
    executable.write_bytes(b"neutral runtime executable")
    (directory / "main.py").write_bytes(b"neutral runtime source")
    settings.comfy_executable = executable
    settings.comfy_directory = directory


def _payload() -> dict[str, Any]:
    return {
        "name": "Neutral workflow installation",
        "operation": "text_to_image",
        "ui_graph": _ui_graph(),
        "dependencies": {"version": 1, "slots": []},
        "selections": [
            {
                "reference_filename": "styles/detail.safetensors",
                "install_plan_id": _seed_plan(),
                "artifact_path": "styles/detail.safetensors",
            }
        ],
    }


async def _created(client: AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
    response = await client.post(URL, json=payload)
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


@pytest.fixture(autouse=True)
def no_installation_work(client: AsyncClient) -> Iterator[None]:
    def identities() -> dict[str, set[str]]:
        with SessionLocal() as session:
            return {
                model.__tablename__: set(session.scalars(select(model.id)))
                for model in (models.WorkflowDefinition, models.WorkflowRevision, models.Job)
            }

    before = identities()
    yield
    assert identities() == before


async def test_preflight_is_durable_idempotent_and_has_no_installation_side_effects(
    client: AsyncClient,
) -> None:
    payload = _payload()
    plan = await _created(client, payload)
    assert plan["can_accept"] and plan["blockers"] == []
    assert plan["asset_download_bytes"] == plan["total_download_bytes"] == 17
    assert plan["download_requests"][0]["expected_sha256"] == {
        "styles/detail.safetensors": "a" * 64
    }
    assert plan == await _created(client, dict(reversed(list(payload.items()))))
    with SessionLocal() as session:
        assert (
            session.scalar(select(func.count()).select_from(models.WorkflowPackageInstallPlan)) == 1
        )
        record = session.get(models.WorkflowPackageInstallPlan, plan["id"])
        assert record is not None and record.request_json["ui_graph"] == payload["ui_graph"]
    restored = await client.get(f"{URL}/{plan['id']}")
    assert restored.status_code == 200 and restored.json() == plan, restored.text


@pytest.mark.parametrize("change", ["source", "operation", "name", "declaration"])
async def test_preflight_identity_binds_the_whole_source_request(
    client: AsyncClient, change: str
) -> None:
    payload = _payload()
    original = await _created(client, payload)
    if change == "source":
        payload["ui_graph"]["nodes"][0]["widgets_values"].append(0.5)
    elif change == "operation":
        payload["operation"] = "image_to_image"
    elif change == "name":
        payload["name"] = "Another neutral workflow"
    else:
        payload["dependencies"] = {}
    updated = await _created(client, payload)
    assert updated["id"] != original["id"]
    assert updated["plan_sha256"] != original["plan_sha256"]
    assert updated["asset_binding_sha256"] == original["asset_binding_sha256"]


@pytest.mark.parametrize(
    "change", ["source", "preflight", "artifact", "runtime_contract", "runtime_defaults"]
)
async def test_stored_source_or_download_drift_refuses_the_old_plan(
    client: AsyncClient, change: str
) -> None:
    plan = await _created(client, _payload())
    with SessionLocal() as session:
        if change in {"source", "preflight"}:
            record = session.get(models.WorkflowPackageInstallPlan, plan["id"])
            assert record is not None
            if change == "source":
                altered = copy.deepcopy(record.request_json)
                altered["operation"] = "image_to_image"
                record.request_json = altered
            else:
                altered = copy.deepcopy(record.preflight_json)
                altered["total_download_bytes"] = 1
                record.preflight_json = altered
        else:
            remote = session.get(models.InstallPlan, "plan_lora")
            assert remote is not None
            if change == "artifact":
                artifacts = copy.deepcopy(remote.artifacts_json)
                artifacts[0]["sha256"] = "b" * 64
                remote.artifacts_json = artifacts
            elif change == "runtime_contract":
                remote.runtime_contract_json = {
                    **remote.runtime_contract_json,
                    "comfy_paths": {"loras": "changed-styles"},
                }
            else:
                remote.runtime_contract_json = {
                    **remote.runtime_contract_json,
                    "default_settings": {"steps": 12},
                }
        session.commit()
    response = await client.get(f"{URL}/{plan['id']}")
    assert response.status_code == 409, response.text


async def test_preflight_counts_every_required_artifact_once(client: AsyncClient) -> None:
    payload = _payload()
    with SessionLocal() as session:
        remote = session.get(models.InstallPlan, "plan_lora")
        assert remote is not None
        remote.artifacts_json = [
            *remote.artifacts_json,
            {
                **remote.artifacts_json[0],
                "path": "styles/companion.safetensors",
                "size_bytes": 23,
            },
        ]
        session.commit()
    plan = await _created(client, payload)
    assert plan["total_download_bytes"] == 40
    assert len(plan["assets"]) == len(plan["download_requests"]) == 1
    assert len(plan["download_requests"][0]["allow_patterns"]) == 2


def _runtime_declaration(count: int, *, per_slot: int = 1) -> dict[str, Any]:
    return {
        "version": 1,
        "slots": [
            {
                "name": f"runtime_{index}",
                "resource_kind": "runtime",
                "required": True,
                "satisfaction": "all_of",
                "requirements": [
                    {"key": f"engine_{item}", "constraints": {"engine": "comfyui"}}
                    for item in range(min(per_slot, count - index))
                ],
            }
            for index in range(0, count, per_slot)
        ],
    }


def _installed_resource_slot(name: str, kind: str, identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "resource_kind": kind,
        "required": True,
        "satisfaction": "all_of",
        "requirements": [{"key": "installed", "constraints": identity}],
    }


def _sized_optional_declaration(size: int, *, unicode_text: bool = False) -> dict[str, Any]:
    declaration = _runtime_declaration(1)
    slot = declaration["slots"][0]
    slot["name"] = "media_runtime_1"
    slot["required"] = False
    notes = ["色" * 100 if unicode_text else ""]
    slot["requirements"][0]["constraints"] = {"notes": notes}

    def encoded_size() -> int:
        return len(json.dumps(declaration, sort_keys=True, separators=(",", ":")))

    while size - encoded_size() > 4003:
        notes.append("x" * 4000)
    remaining = size - encoded_size()
    assert remaining >= 0
    if remaining >= 3:
        notes.append("x" * (remaining - 3))
    else:
        notes[0] += "x" * remaining
    assert encoded_size() == size
    return declaration


@pytest.mark.parametrize("with_asset", [False, True])
@pytest.mark.parametrize("unicode_text", [False, True])
@pytest.mark.parametrize("fits", [False, True])
async def test_preflight_reserves_serialized_resource_identity_bytes(
    client: AsyncClient, with_asset: bool, unicode_text: bool, fits: bool
) -> None:
    from local_lm.workflow_dependencies import (
        MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES,
        parse_workflow_dependency_contract,
    )

    payload = _payload()
    if not with_asset:
        payload["selections"] = []
        payload["ui_graph"]["nodes"][0].update(
            type="EmptyLatentImage", widgets_values=[512, 512, 1]
        )
    resources = [("media_runtime_2", "runtime", {"engine": "comfyui"})]
    if with_asset:
        resources.append(
            (
                "media_asset_1",
                "model_asset",
                {
                    "kind": "model_asset",
                    "asset_kind": "lora",
                    "runtime_reference": "styles/detail.safetensors",
                    "sha256": "a" * 64,
                },
            )
        )
    additions = [
        {
            "name": name,
            "resource_kind": kind,
            "required": True,
            "satisfaction": "all_of",
            "requirements": [{"key": "installed", "constraints": constraints}],
        }
        for name, kind, constraints in resources
    ]
    added_bytes = sum(
        1 + len(json.dumps(slot, sort_keys=True, separators=(",", ":"))) for slot in additions
    )
    payload["dependencies"] = _sized_optional_declaration(
        MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES - added_bytes + (0 if fits else 1),
        unicode_text=unicode_text,
    )
    parse_workflow_dependency_contract(payload["dependencies"])
    plan = await _created(client, payload)
    assert plan["can_accept"] is fits
    assert ("dependency-contract-capacity-exceeded" in plan["blockers"]) is (not fits)
    if not fits:
        response = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
        assert response.status_code == 409, response.text


@pytest.mark.parametrize("filename", ["~neutral/detail.safetensors", "$NEUTRAL/detail.safetensors"])
async def test_preflight_refuses_unrepresentable_generated_asset_identity(
    client: AsyncClient, filename: str
) -> None:
    payload = _payload()
    with SessionLocal() as session:
        remote = session.get(models.InstallPlan, "plan_lora")
        assert remote is not None
        remote.artifacts_json = [{**remote.artifacts_json[0], "path": filename}]
        session.commit()
    payload["ui_graph"] = _ui_graph(filename)
    payload["selections"][0].update(reference_filename=filename, artifact_path=filename)
    plan = await _created(client, payload)
    assert not plan["can_accept"]
    assert "dependency-contract-unrepresentable" in plan["blockers"]
    assert "dependency-contract-capacity-exceeded" not in plan["blockers"]
    refused = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
    assert refused.status_code == 409, refused.text


@pytest.mark.parametrize("with_asset", [False, True])
@pytest.mark.parametrize("with_extension", [False, True])
@pytest.mark.parametrize("limit", ["slots", "requirements"])
@pytest.mark.parametrize("fits", [False, True])
async def test_preflight_reserves_dependencies_added_by_installation(
    client: AsyncClient, with_asset: bool, with_extension: bool, limit: str, fits: bool
) -> None:
    from local_lm.workflow_dependencies import parse_workflow_dependency_contract

    payload = _payload()
    if not with_asset:
        payload["selections"] = []
        node = payload["ui_graph"]["nodes"][0]
        node["type"] = "EmptyLatentImage"
        node["widgets_values"] = [512, 512, 1]
    if with_extension:
        payload["ui_graph"]["nodes"].append(
            {
                "id": 2,
                "type": "NeutralExtension",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [],
                "properties": {"cnr_id": "neutral-extension", "ver": "1.2.3"},
            }
        )
    added = 1 + int(with_asset) + int(with_extension)
    declared_count = (64 if limit == "slots" else 512) - added + (0 if fits else 1)
    payload["dependencies"] = _runtime_declaration(
        declared_count, per_slot=1 if limit == "slots" else 64
    )
    parse_workflow_dependency_contract(payload["dependencies"])
    plan = await _created(client, payload)
    assert plan["can_accept"] is (fits and not with_extension)
    if with_extension:
        assert plan["extensions"][0]["versions"] == ["1.2.3"]
    assert ("dependency-contract-capacity-exceeded" in plan["blockers"]) is (not fits)
    restored = await client.get(f"{URL}/{plan['id']}")
    assert restored.status_code == 200 and restored.json() == plan
    if not fits:
        response = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "workflow-package-install-plan-incomplete"


async def test_unknown_runtime_is_a_blocker_and_recovery_changes_the_plan(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload()
    original = app.state.services.engines.media.object_info

    async def unavailable() -> dict[str, Any]:
        raise RuntimeError("Neutral runtime unavailable")

    monkeypatch.setattr(app.state.services.engines.media, "object_info", unavailable)
    plan = await _created(client, payload)
    assert not plan["can_accept"] and "node-inventory-unavailable" in plan["blockers"]
    monkeypatch.setattr(app.state.services.engines.media, "object_info", original)
    stale = await client.get(f"{URL}/{plan['id']}")
    assert stale.status_code == 409, stale.text
    ready = await _created(client, payload)
    assert ready["can_accept"] and ready["id"] != plan["id"]


async def test_unresolved_extension_cost_is_unknown_instead_of_zero(client: AsyncClient) -> None:
    payload = _payload()
    payload["ui_graph"]["nodes"].append(
        {
            "id": 2,
            "type": "NeutralExtension",
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "widgets_values": [],
            "properties": {"cnr_id": "neutral-extension", "version": "1.2.3"},
        }
    )
    plan = await _created(client, payload)
    assert not plan["can_accept"] and "extension-plan-unavailable" in plan["blockers"]
    assert plan["total_download_bytes"] is None and plan["asset_download_bytes"] == 17
    assert plan["extensions"][0]["package_id"] == "neutral-extension"


async def test_an_unknown_dependency_declaration_never_becomes_an_empty_contract(
    client: AsyncClient,
) -> None:
    payload = _payload()
    del payload["dependencies"]
    plan = await _created(client, payload)
    assert plan["dependency_contract_sha256"] is None and not plan["can_accept"]
    assert "dependency-contract-unknown" in plan["blockers"]


async def test_invalid_declaration_creates_no_plan_or_work(client: AsyncClient) -> None:
    payload = _payload()
    payload["dependencies"] = {"version": 99, "slots": []}
    response = await client.post(URL, json=payload)
    assert response.status_code == 422, response.text
    with SessionLocal() as session:
        assert (
            session.scalar(select(func.count()).select_from(models.WorkflowPackageInstallPlan)) == 0
        )


async def test_core_only_plan_does_not_invent_downloads(client: AsyncClient) -> None:
    payload = _payload()
    payload["selections"] = []
    payload["ui_graph"]["nodes"][0]["type"] = "EmptyLatentImage"
    payload["ui_graph"]["nodes"][0]["widgets_values"] = [512, 512, 1]
    plan = await _created(client, payload)
    assert plan["can_accept"] and plan["total_download_bytes"] == 0
    assert plan["download_requests"] == plan["assets"] == []


async def test_retry_does_not_hide_a_damaged_persisted_plan(client: AsyncClient) -> None:
    payload = _payload()
    plan = await _created(client, payload)
    with SessionLocal() as session:
        record = session.get(models.WorkflowPackageInstallPlan, plan["id"])
        assert record is not None
        record.preflight_json = {**record.preflight_json, "total_download_bytes": 1}
        session.commit()
    response = await client.post(URL, json=payload)
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "workflow-package-install-plan-changed"


async def test_interrupted_plan_creation_rolls_back_its_asset_alias(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import api as api_module

    payload = _payload()
    await _created(client, payload)
    from local_lm.workflow_package_install_plans import create_workflow_package_install_plan

    original = create_workflow_package_install_plan

    def interrupted(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RuntimeError("Neutral preflight interruption")

    monkeypatch.setattr(api_module, "create_workflow_package_install_plan", interrupted)
    payload["ui_graph"]["nodes"][0]["widgets_values"] = ["styles/alias.safetensors"]
    payload["selections"][0]["reference_filename"] = "styles/alias.safetensors"
    with pytest.raises(RuntimeError, match="Neutral preflight interruption"):
        await client.post(URL, json=payload)
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(models.InstallPlan)) == 1
        assert (
            session.scalar(select(func.count()).select_from(models.WorkflowPackageInstallPlan)) == 1
        )
