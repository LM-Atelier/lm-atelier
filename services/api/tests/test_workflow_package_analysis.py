"""The analyze endpoint: report what a raw package needs, change nothing."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

pytestmark = pytest.mark.asyncio


def _node(
    identifier: int,
    node_type: str,
    *,
    package: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    if package is not None:
        properties["cnr_id"] = package
    if version is not None:
        properties["ver"] = version
    return {
        "id": identifier,
        "type": node_type,
        "properties": properties,
        "widgets_values": [],
    }


def _workflow(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": nodes,
        "links": [],
        "definitions": {"subgraphs": []},
        "extra": {"frontendVersion": "1.45.21"},
    }


async def test_analysis_names_the_gap_and_who_would_fill_it(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a live node inventory, missing types and their packages are named."""

    async def object_info() -> dict[str, Any]:
        return {"KSampler": {}, "CLIPTextEncode": {}}

    monkeypatch.setattr(
        app.state.services.engines.media,
        "object_info",
        object_info,
        raising=False,
    )
    response = await client.post(
        "/api/workflows/packages/analyze",
        json={
            "ui_graph": _workflow(
                [
                    _node(1, "KSampler", package="comfy-core", version="0.28.0"),
                    _node(2, "Power Lora Loader", package="rgthree-comfy", version="1.2.3"),
                ]
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["node_inventory_available"] is True
    assert payload["missing_node_types"] == ["Power Lora Loader"]
    assert payload["missing_nodes"] == [
        {"node_type": "Power Lora Loader", "count": 1, "package_id": "rgthree-comfy"}
    ]
    assert payload["runtime_nodes_available"] is False
    assert payload["ready"] is False
    packages = {package["package_id"]: package for package in payload["custom_packages"]}
    assert packages["rgthree-comfy"]["node_types"] == ["Power Lora Loader"]
    assert packages["rgthree-comfy"]["versions"] == ["1.2.3"]
    assert packages["rgthree-comfy"]["locally_resolved"] is False


async def test_an_offline_runtime_reports_unknown_rather_than_missing(
    client: AsyncClient,
) -> None:
    """The mock media engine cannot list nodes; the report must say so."""

    response = await client.post(
        "/api/workflows/packages/analyze",
        json={
            "ui_graph": _workflow([_node(1, "KSampler", package="comfy-core", version="0.28.0")])
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["node_inventory_available"] is False
    # Every runtime node is "missing" against an empty inventory; the flag
    # above is what tells the browser to present this as unknown, and the
    # gate stays closed rather than guessing open.
    assert payload["missing_node_types"] == ["KSampler"]
    assert payload["ready"] is False


async def test_a_resolved_package_reports_an_open_gate(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def object_info() -> dict[str, Any]:
        return {"KSampler": {}}

    monkeypatch.setattr(
        app.state.services.engines.media,
        "object_info",
        object_info,
        raising=False,
    )
    response = await client.post(
        "/api/workflows/packages/analyze",
        json={
            "ui_graph": _workflow([_node(1, "KSampler", package="comfy-core", version="0.28.0")])
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["missing_node_types"] == []
    assert payload["runtime_nodes_available"] is True
    assert payload["ready"] is True


async def test_an_installed_package_at_the_pinned_revision_counts_as_resolved(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact version evidence from custom-node installs reaches the analyzer."""

    from local_lm.db import SessionLocal
    from local_lm.models import CustomNodeInstall

    revision = "b" * 40
    with SessionLocal() as session:
        session.add(
            CustomNodeInstall(
                name="rgthree-comfy",
                source_url="https://example.invalid/rgthree-comfy",
                revision=revision,
                installed_path="C:/synthetic/custom-nodes/rgthree-comfy",
                tree_hash="c" * 64,
            )
        )
        session.commit()

    async def object_info() -> dict[str, Any]:
        return {"Power Lora Loader": {}}

    monkeypatch.setattr(
        app.state.services.engines.media,
        "object_info",
        object_info,
        raising=False,
    )
    response = await client.post(
        "/api/workflows/packages/analyze",
        json={
            "ui_graph": _workflow(
                [_node(1, "Power Lora Loader", package="rgthree-comfy", version=revision)]
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    packages = {package["package_id"]: package for package in payload["custom_packages"]}
    assert packages["rgthree-comfy"]["locally_resolved"] is True
    assert all(issue["code"] != "unresolved_custom_node_package" for issue in payload["issues"])
    assert payload["ready"] is True


async def test_a_pin_without_matching_install_evidence_stays_unresolved(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry-version pin cannot be vouched for by a different revision."""

    async def object_info() -> dict[str, Any]:
        return {"Power Lora Loader": {}}

    monkeypatch.setattr(
        app.state.services.engines.media,
        "object_info",
        object_info,
        raising=False,
    )
    response = await client.post(
        "/api/workflows/packages/analyze",
        json={
            "ui_graph": _workflow(
                [_node(1, "Power Lora Loader", package="rgthree-comfy", version="1.2.3")]
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    packages = {package["package_id"]: package for package in payload["custom_packages"]}
    assert packages["rgthree-comfy"]["locally_resolved"] is False
    assert any(
        issue["code"] == "unresolved_custom_node_package" and issue["severity"] == "blocking"
        for issue in payload["issues"]
    )
    assert payload["ready"] is False


async def test_a_referenced_model_this_machine_holds_counts_as_present(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint supplies the local model inventory to the analyzer."""

    from local_lm.db import SessionLocal
    from local_lm.models import ModelInstall

    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="model_local",
                name="Local checkpoint",
                role="image",
                engine="comfyui",
                local_path="C:/synthetic/image",
                compatibility="likely",
                manifest_json={"files": ["portrait.safetensors"]},
                active=True,
            )
        )
        session.commit()

    async def object_info() -> dict[str, Any]:
        return {"CheckpointLoaderSimple": {}}

    monkeypatch.setattr(
        app.state.services.engines.media,
        "object_info",
        object_info,
        raising=False,
    )
    graph = _workflow(
        [
            {
                "id": 1,
                "type": "CheckpointLoaderSimple",
                "properties": {"cnr_id": "comfy-core", "ver": "0.28.0"},
                "widgets_values": ["portrait.safetensors"],
            }
        ]
    )
    response = await client.post(
        "/api/workflows/packages/analyze",
        json={"ui_graph": graph},
    )

    assert response.status_code == 200
    payload = response.json()
    assets = {asset["filename"]: asset for asset in payload["asset_references"]}
    assert assets["portrait.safetensors"]["present_locally"] is True
    assert assets["portrait.safetensors"]["kind"] == "checkpoint"
    assert all(issue["code"] != "missing_asset" for issue in payload["issues"])
    assert payload["ready"] is True


async def test_a_malformed_package_is_a_typed_422(client: AsyncClient) -> None:
    response = await client.post(
        "/api/workflows/packages/analyze",
        json={"ui_graph": {"version": 0.4, "nodes": [], "links": []}},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "empty_workflow"
    assert payload["detail"]


async def test_analysis_persists_and_executes_nothing(client: AsyncClient) -> None:
    """The endpoint is read-only: no workflow rows, no jobs."""

    before_workflows = (await client.get("/api/workflows")).json()
    before_jobs = (await client.get("/api/jobs")).json()

    response = await client.post(
        "/api/workflows/packages/analyze",
        json={
            "ui_graph": _workflow([_node(1, "KSampler", package="comfy-core", version="0.28.0")])
        },
    )

    assert response.status_code == 200
    assert (await client.get("/api/workflows")).json() == before_workflows
    assert (await client.get("/api/jobs")).json() == before_jobs
