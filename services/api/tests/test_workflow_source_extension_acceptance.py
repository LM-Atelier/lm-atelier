"""Accept exact extension previews without admitting uncovered runtime requirements."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_package_extension_preflight import _configure
from test_workflow_package_install_plans import URL
from test_workflow_package_install_plans import runtime_inventory as runtime_inventory

from local_lm import models
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.workflow_offer_packages import accepted_workflow_offer_packages
from local_lm.workflow_package_install_plans import load_stored_workflow_package_install_plan

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("commit", [False, True])
async def test_extension_plan_approval_stages_exact_work_once_without_trusting_code(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    commit: bool,
) -> None:
    inputs, payload = _configure(settings, monkeypatch, tmp_path, commit=commit)
    starts: list[str] = []
    completions: list[str] = []
    monkeypatch.setattr(app.state.services.downloads, "start", starts.append)
    monkeypatch.setattr(
        app.state.services.downloads, "start_workflow_installation", completions.append
    )
    try:
        preview = await client.post(URL, json=payload)
        assert preview.status_code == 201, preview.text
        plan = preview.json()
        assert plan["can_accept"] and plan["blockers"] == [], plan
        accepted = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
        assert accepted.status_code == 202, accepted.text
        jobs = accepted.json()
        assert sorted(job["kind"] for job in jobs) == ["download", "workflow_install"]
        with SessionLocal() as session:
            offer = session.scalar(
                select(models.WorkflowInstallOffer).where(
                    models.WorkflowInstallOffer.source_plan_id == plan["id"]
                )
            )
            assert offer is not None and offer.total_bytes == plan["total_download_bytes"]
            _, saved = load_stored_workflow_package_install_plan(session, plan["id"])
            packages = accepted_workflow_offer_packages(session, offer, saved)
            assert len(packages) == 1
            revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
            assert revision is not None and not revision.trusted and revision.api_graph_json == {}
            assert list(session.scalars(select(models.ComfyRegistryInstall))) == []
            assert list(session.scalars(select(models.WorkflowActivation))) == []
        repeated = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
        assert repeated.status_code == 202, repeated.text
        assert {job["id"] for job in repeated.json()} == {job["id"] for job in jobs}
        assert len(set(starts)) == len(set(completions)) == 1
    finally:
        await inputs["unused_archive"].close()


@pytest.mark.parametrize(
    "change",
    ["unattributed", "core", "unversioned", "conflicting-version", "unplanned", "inventory"],
)
async def test_extension_plan_does_not_cover_unknown_or_ambiguous_nodes(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: str,
) -> None:
    inputs, payload = _configure(settings, monkeypatch, tmp_path)
    node = copy.deepcopy(payload["ui_graph"]["nodes"][-1])
    node["id"] += 1
    if change == "unattributed":
        node["properties"] = {}
    elif change == "core":
        node["properties"] = {"cnr_id": "comfy-core"}
    elif change == "unversioned":
        node["properties"].pop("ver")
    elif change == "conflicting-version":
        node["properties"]["ver"] = "9.9.9"
    elif change == "unplanned":
        node["type"] = "UnavailableNeutralNode"
        node["properties"] = {}
    else:

        async def unavailable() -> dict[str, Any]:
            raise ValueError("The constructed worker is unavailable")

        monkeypatch.setattr(app.state.services.engines.media, "object_info", unavailable)
    if change != "inventory":
        payload["ui_graph"]["nodes"].append(node)
    try:
        preview = await client.post(URL, json=payload)
        assert preview.status_code == 201, preview.text
        plan = preview.json()
        assert not plan["can_accept"] and plan["blockers"], plan
        accepted = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
        assert accepted.status_code == 409, accepted.text
        with SessionLocal() as session:
            assert list(session.scalars(select(models.WorkflowInstallOffer))) == []
            assert list(session.scalars(select(models.Job))) == []
            assert list(session.scalars(select(models.ComfyRegistryInstall))) == []
    finally:
        await inputs["unused_archive"].close()
