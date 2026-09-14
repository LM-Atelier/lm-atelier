"""Approve fresh runtime setup only against its own measured built-in nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_runtime_targets import runtime as runtime
from test_workflow_source_live_comfy import _source_graph

from local_lm import models
from local_lm.db import SessionLocal
from local_lm.runtime_provisioning import RuntimeProvisioner

URL = "/api/workflows/packages/install-plans"


def _configure(
    app: FastAPI, runtime: RuntimeProvisioner, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, list[str]]:
    root = Path(__file__).resolve().parents[3]
    relative = "runtime-reviews/comfyui-v0.28.0-nodes.json"
    catalog = runtime.manifest_path.parent / relative
    catalog.write_bytes((root / "packaging" / relative).read_bytes())
    monkeypatch.setattr(app.state.services, "runtimes", runtime)
    starts: list[str] = []
    monkeypatch.setattr(app.state.services.downloads, "start_workflow_installation", starts.append)
    payload = {
        "name": "Constructed fresh runtime workflow",
        "operation": "text_to_image",
        "ui_graph": _source_graph(),
        "dependencies": {"version": 1, "slots": []},
        "selections": [],
    }
    return payload, catalog, starts


@pytest.mark.parametrize("worker_available", [False, True])
async def test_one_approval_stages_fresh_runtime_setup_without_a_live_inventory(
    app: FastAPI,
    client: AsyncClient,
    runtime: RuntimeProvisioner,
    monkeypatch: pytest.MonkeyPatch,
    worker_available: bool,
) -> None:
    payload, _catalog, starts = _configure(app, runtime, monkeypatch)

    async def inventory() -> dict[str, Any]:
        if not worker_available:
            raise ValueError("The constructed media worker is unavailable")
        return {"UnrelatedExistingNode": {}}

    monkeypatch.setattr(app.state.services.engines.media, "object_info", inventory, raising=False)
    preview = await client.post(URL, json=payload)
    assert preview.status_code == 201, preview.text
    plan = preview.json()
    assert plan["runtime_plan"]["operation"] == "install_managed"
    assert plan["can_accept"] and plan["blockers"] == [], plan
    assert plan["total_download_bytes"] == plan["runtime_plan"]["download_bytes"] > 0
    reloaded = await client.get(f"{URL}/{plan['id']}")
    assert reloaded.status_code == 200 and reloaded.json() == plan
    accepted = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
    assert accepted.status_code == 202, accepted.text
    jobs = accepted.json()
    assert [job["kind"] for job in jobs] == ["workflow_install"]
    with SessionLocal() as session:
        offer = session.scalar(select(models.WorkflowInstallOffer))
        assert offer is not None and offer.source_plan_id == plan["id"]
        draft = session.get(models.WorkflowRevision, offer.workflow_revision_id)
        assert draft is not None and not draft.trusted and draft.api_graph_json == {}
        assert list(session.scalars(select(models.WorkflowActivation))) == []
    repeated = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
    assert repeated.status_code == 202, repeated.text
    assert [job["id"] for job in repeated.json()] == [job["id"] for job in jobs]
    assert [job["payload_json"] for job in repeated.json()] == [job["payload_json"] for job in jobs]
    with SessionLocal() as session:
        assert list(session.scalars(select(models.Job.id))) == [job["id"] for job in jobs]
    assert len(set(starts)) == 1
    assert runtime.settings.comfy_executable is None
    assert list(runtime.runtime_root.iterdir()) == []


@pytest.mark.parametrize("change", ["missing-catalog", "uncovered-node", "changed-catalog"])
async def test_fresh_runtime_approval_refuses_unverified_or_changed_node_coverage(
    app: FastAPI,
    client: AsyncClient,
    runtime: RuntimeProvisioner,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    payload, catalog, starts = _configure(app, runtime, monkeypatch)

    async def unrelated_inventory() -> dict[str, Any]:
        return {"EmptyImage": {}, "SaveImage": {}, "UnavailableConstructedNode": {}}

    monkeypatch.setattr(
        app.state.services.engines.media, "object_info", unrelated_inventory, raising=False
    )
    if change == "missing-catalog":
        catalog.unlink()
    elif change == "uncovered-node":
        payload["ui_graph"]["nodes"][0]["type"] = "UnavailableConstructedNode"
        payload["ui_graph"]["nodes"][0]["properties"] = {"cnr_id": "comfy-core"}
    preview = await client.post(URL, json=payload)
    assert preview.status_code == 201, preview.text
    plan = preview.json()
    if change == "changed-catalog":
        assert plan["can_accept"], plan
        catalog.write_bytes(b"{}")
    else:
        assert not plan["can_accept"] and plan["blockers"], plan
    accepted = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
    assert accepted.status_code == 409, accepted.text
    assert starts == []
    with SessionLocal() as session:
        assert list(session.scalars(select(models.WorkflowInstallOffer))) == []
        assert list(session.scalars(select(models.Job))) == []
