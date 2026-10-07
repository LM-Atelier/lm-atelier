"""The saved workflow approval includes the runtime it will use and its download cost."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_package_install_plans import URL, _payload
from test_workflow_package_install_plans import runtime_inventory as runtime_inventory

from local_lm import models
from local_lm.db import SessionLocal

pytestmark = pytest.mark.asyncio


async def _plan(client: AsyncClient) -> dict[str, Any]:
    response = await client.post(URL, json=_payload())
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


async def test_runtime_operation_and_terms_survive_the_saved_approval(
    client: AsyncClient, app: FastAPI
) -> None:
    expected = app.state.services.runtimes.preflight("comfyui")
    plan = await _plan(client)
    assert plan["runtime_plan"] == asdict(expected)
    assert expected.operation == "reuse_configured" and expected.download_bytes == 0
    assert plan["total_download_bytes"] == plan["asset_download_bytes"] == 17
    with SessionLocal() as session:
        record = session.get(models.WorkflowPackageInstallPlan, plan["id"])
        assert record is not None
        assert record.preflight_json["runtime_plan"] == plan["runtime_plan"]
    restored = await client.get(f"{URL}/{plan['id']}")
    assert restored.status_code == 200 and restored.json() == plan


async def test_fresh_runtime_bytes_include_the_archive_and_overlay_before_approval(
    client: AsyncClient, app: FastAPI
) -> None:
    services = app.state.services
    services.settings.comfy_executable = None
    services.settings.comfy_directory = None
    services.runtimes._platform_keys["comfyui"] = "windows-x86_64-nvidia-cu13"
    expected = services.runtimes.preflight("comfyui")
    plan = await _plan(client)
    asset = services.runtimes._definition("comfyui")["runtime_assets"]["windows-x86_64-nvidia-cu13"]
    size = asset["size_bytes"] + sum(item["size_bytes"] for item in asset["security_overlays"])
    assert plan["runtime_plan"] == asdict(expected)
    assert expected.operation == "install_managed" and expected.download_bytes == size
    assert plan["total_download_bytes"] == size + 17
    assert plan["blockers"] == [] and plan["can_accept"]
    assert services.settings.comfy_executable is None


async def test_an_unknown_runtime_keeps_the_total_unknown_and_approval_blocked(
    client: AsyncClient, app: FastAPI
) -> None:
    services = app.state.services
    services.settings.comfy_executable = None
    services.settings.comfy_directory = None
    services.runtimes._platform_keys["comfyui"] = "unsupported-platform"
    plan = await _plan(client)
    assert plan["runtime_plan"] is None and plan["total_download_bytes"] is None
    assert "runtime-plan-unavailable" in plan["blockers"] and not plan["can_accept"]


@pytest.mark.parametrize("change", ["executable", "license", "manifest"])
async def test_runtime_changes_invalidate_a_saved_workflow_preview(
    client: AsyncClient, app: FastAPI, change: str
) -> None:
    plan = await _plan(client)
    services = app.state.services
    if change == "executable":
        services.settings.comfy_executable.write_bytes(b"changed neutral executable")
    elif change == "license":
        services.runtimes._definition("comfyui")["license"] = "Changed terms"
    else:
        services.runtimes._definition("comfyui")["pinned_release"] = "changed-release"
    response = await client.get(f"{URL}/{plan['id']}")
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "workflow-package-install-plan-changed"


async def test_changed_runtime_refuses_acceptance_before_jobs_are_created(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = await _plan(client)
    services = app.state.services
    started: list[str] = []
    monkeypatch.setattr(services.downloads, "start", started.append)
    monkeypatch.setattr(services.downloads, "start_workflow_installation", started.append)
    services.settings.comfy_executable.write_bytes(b"changed neutral executable")
    response = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "workflow-package-install-plan-changed"
    assert started == []
    with SessionLocal() as session:
        assert list(session.scalars(select(models.WorkflowInstallOffer))) == []
        assert list(session.scalars(select(models.Job))) == []


async def test_repeating_an_accepted_request_keeps_its_original_runtime_and_jobs(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = await _plan(client)
    services = app.state.services
    monkeypatch.setattr(services.downloads, "start", lambda _id: None)
    monkeypatch.setattr(services.downloads, "start_workflow_installation", lambda _id: None)
    url = f"/api/workflow-install-offers/{plan['id']}/install"
    accepted = await client.post(url)
    assert accepted.status_code == 202, accepted.text
    services.settings.comfy_executable.write_bytes(b"changed neutral executable")
    repeated = await client.post(url)
    assert repeated.status_code == 202
    assert [item["id"] for item in repeated.json()] == [item["id"] for item in accepted.json()]
    with SessionLocal() as session:
        record = session.get(models.WorkflowPackageInstallPlan, plan["id"])
        assert record is not None and record.preflight_json["runtime_plan"] == plan["runtime_plan"]
        jobs = list(session.scalars(select(models.Job).where(models.Job.kind == "download")))
        assert len(jobs) == 1 and jobs[0].payload_json == plan["download_requests"][0]


async def test_runtime_drift_during_extension_inspection_cannot_save_a_preview(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import api
    from local_lm.workflow_package_extension_preflight import preflight_workflow_extensions

    original = preflight_workflow_extensions

    async def inspect(*args: Any, **kwargs: Any) -> Any:
        result = await original(*args, **kwargs)
        app.state.services.settings.comfy_executable.write_bytes(b"changed neutral executable")
        return result

    monkeypatch.setattr(api, "preflight_workflow_extensions", inspect)
    response = await client.post(URL, json=_payload())
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "workflow-package-install-plan-changed"
    with SessionLocal() as session:
        assert list(session.scalars(select(models.WorkflowPackageInstallPlan))) == []


async def test_fresh_runtime_and_transitive_extension_downloads_share_one_exact_total(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from test_workflow_package_extension_preflight import _configure
    from test_workflow_runtime_targets import _host

    services = app.state.services
    inputs, payload = _configure(services.settings, monkeypatch, tmp_path)
    _host(monkeypatch)
    services.settings.comfy_executable = None
    services.settings.comfy_directory = None
    services.runtimes._platform_keys["comfyui"] = "windows-x86_64-nvidia-cu13"
    expected = services.runtimes.preflight("comfyui")
    try:
        response = await client.post(URL, json=payload)
        assert response.status_code == 201, response.text
        plan = response.json()
        assert plan["extension_execution"]["errors"] == {}
        assert plan["runtime_plan"] == asdict(expected)
        assert plan["total_download_bytes"] == (
            expected.download_bytes + 17 + 200 + len(inputs["state"]["content"])
        )
        assert plan["can_accept"] and plan["blockers"] == []
        assert services.settings.comfy_executable is None
        with SessionLocal() as session:
            assert list(session.scalars(select(models.ComfyRegistryInstall))) == []
    finally:
        await inputs["unused_archive"].close()
