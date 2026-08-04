"""Workflow asset review and install: stateless, server-rebuilt, id-only.

The browser holds the UI graph and sends explicit selections; everything
else - the analysis, the binding, the digests, the requests - is rebuilt
here from current records. These pin that the endpoints never trust
client-supplied evidence and that queueing honors the reviewed hash.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm.model_planner import INSTALL_RESOLVER_VERSION

pytestmark = pytest.mark.asyncio

DIGEST = "a" * 64


def _ui_graph(filename: str = "styles/detail.safetensors") -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "LoraLoader",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [filename],
                "properties": {"cnr_id": "comfy-core", "version": "0.28.0"},
            }
        ],
        "links": [],
    }


def _checkpoint_graph(filename: str) -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "CheckpointLoaderSimple",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [filename],
                "properties": {"cnr_id": "comfy-core", "version": "0.28.0"},
            }
        ],
        "links": [],
    }


def _seed_plan(plan_id: str = "plan_lora", *, artifact_path: str | None = None) -> str:
    from local_lm.db import SessionLocal
    from local_lm.models import InstallPlan

    with SessionLocal() as session:
        session.add(
            InstallPlan(
                id=plan_id,
                provider="civitai",
                remote_id="101",
                revision="202",
                role="image",
                engine="comfyui",
                plan_hash=hashlib.sha256(plan_id.encode()).hexdigest(),
                resolver_version=INSTALL_RESOLVER_VERSION,
                compatibility="supported",
                artifacts_json=[
                    {
                        "path": artifact_path or "styles/detail.safetensors",
                        "kind": "lora",
                        "target_folder": "loras",
                        "size_bytes": 17,
                        "sha256": DIGEST,
                        "required": True,
                        "reuse": "download",
                        "source_version_id": "202",
                        "source_file_id": "301",
                    }
                ],
                runtime_contract_json={
                    "auxiliary_kind": "lora",
                    "comfy_paths": {"loras": "styles"},
                },
                activation_probe_json={},
                status="planned",
            )
        )
        session.commit()
    return plan_id


async def test_review_binds_selections_and_reports_only_safe_fields(
    client: AsyncClient,
) -> None:
    plan_id = _seed_plan()

    response = await client.post(
        "/api/workflows/packages/assets/review",
        json={
            "ui_graph": _ui_graph(),
            "selections": [
                {
                    "reference_filename": "styles/detail.safetensors",
                    "install_plan_id": plan_id,
                    "artifact_path": "styles/detail.safetensors",
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["binding_plan_hash"]) == 64
    assert body["download_count"] == 1
    assert body["total_bytes"] == 17
    (asset,) = body["assets"]
    assert asset["provider"] == "civitai"
    assert asset["sha256"] == DIGEST
    assert asset["target_folder"] == "loras"
    # No provider URL is ever emitted; the manager derives it server-side.
    assert "url" not in asset


async def test_an_unselected_reference_refuses_rather_than_guessing(
    client: AsyncClient,
) -> None:
    _seed_plan()

    response = await client.post(
        "/api/workflows/packages/assets/review",
        json={"ui_graph": _ui_graph(), "selections": []},
    )

    assert response.status_code == 422
    assert response.json()["code"]


async def test_installing_requires_the_reviewed_hash(client: AsyncClient) -> None:
    plan_id = _seed_plan()
    selections = [
        {
            "reference_filename": "styles/detail.safetensors",
            "install_plan_id": plan_id,
            "artifact_path": "styles/detail.safetensors",
        }
    ]
    reviewed = await client.post(
        "/api/workflows/packages/assets/review",
        json={"ui_graph": _ui_graph(), "selections": selections},
    )
    binding_hash = reviewed.json()["binding_plan_hash"]

    stale = await client.post(
        "/api/workflows/packages/assets/install",
        json={
            "ui_graph": _ui_graph(),
            "selections": selections,
            "binding_plan_hash": "f" * 64,
        },
    )
    assert stale.status_code == 422
    assert stale.json()["code"] == "binding_plan_changed"
    assert all(job["kind"] != "download" for job in (await client.get("/api/jobs")).json())

    queued = await client.post(
        "/api/workflows/packages/assets/install",
        json={
            "ui_graph": _ui_graph(),
            "selections": selections,
            "binding_plan_hash": binding_hash,
        },
    )
    assert queued.status_code == 202
    jobs = queued.json()
    # One download per distinct plan; every job is returned, not a fake unit.
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "download"


async def test_review_materializes_a_provider_filename_as_the_workflow_runtime_name(
    client: AsyncClient,
) -> None:
    from local_lm.db import SessionLocal
    from local_lm.models import InstallPlan

    source_path = "provider-checkpoint.safetensors"
    reference = "workflow-checkpoint.safetensors"
    source_plan_id = "plan_provider_checkpoint"
    with SessionLocal() as session:
        session.add(
            InstallPlan(
                id=source_plan_id,
                provider="civitai",
                remote_id="101",
                revision="202",
                role="image",
                engine="comfyui",
                plan_hash="b" * 64,
                resolver_version=INSTALL_RESOLVER_VERSION,
                compatibility="supported",
                artifacts_json=[
                    {
                        "path": source_path,
                        "kind": "checkpoint",
                        "target_folder": "checkpoints",
                        "size_bytes": 29,
                        "sha256": DIGEST,
                        "required": True,
                        "reuse": "download",
                        "source_version_id": "202",
                        "source_file_id": "301",
                    }
                ],
                runtime_contract_json={},
                activation_probe_json={},
                status="planned",
            )
        )
        session.commit()
    selections = [
        {
            "reference_filename": reference,
            "install_plan_id": source_plan_id,
            "artifact_path": source_path,
        }
    ]
    reviewed = await client.post(
        "/api/workflows/packages/assets/review",
        json={"ui_graph": _checkpoint_graph(reference), "selections": selections},
    )

    assert reviewed.status_code == 200
    body = reviewed.json()
    (asset,) = body["assets"]
    assert asset["install_plan_id"] != source_plan_id
    assert asset["artifact_path"] == reference
    assert asset["sha256"] == DIGEST
    with SessionLocal() as session:
        derived = session.get(InstallPlan, asset["install_plan_id"])
        assert derived is not None
        assert derived.artifacts_json[0]["source_file_id"] == "301"
        assert derived.runtime_contract_json["workflow_asset_kind"] == "checkpoint"
        assert derived.runtime_contract_json["workflow_asset_alias"]["source_artifact_path"] == (
            source_path
        )

    queued = await client.post(
        "/api/workflows/packages/assets/install",
        json={
            "ui_graph": _checkpoint_graph(reference),
            "selections": selections,
            "binding_plan_hash": body["binding_plan_hash"],
        },
    )

    assert queued.status_code == 202
    assert len(queued.json()) == 1


async def test_a_missing_plan_refuses_typed(client: AsyncClient) -> None:
    response = await client.post(
        "/api/workflows/packages/assets/review",
        json={
            "ui_graph": _ui_graph(),
            "selections": [
                {
                    "reference_filename": "styles/detail.safetensors",
                    "install_plan_id": "plan_absent",
                    "artifact_path": "styles/detail.safetensors",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] in {"install_plan_not_found", "unknown_install_plan"}


async def test_a_later_manager_refusal_starts_no_earlier_download(
    client: AsyncClient,
) -> None:
    """Two plans, the second refused: nothing may already be running.

    `DownloadManager.create` commits and starts each transfer, so validating
    only as it queues would leave the first download live behind a 422 that
    claims nothing happened.
    """

    from local_lm.db import SessionLocal
    from local_lm.models import InstallPlan

    first = _seed_plan("plan_first")
    second = _seed_plan("plan_second", artifact_path="styles/second.safetensors")
    with SessionLocal() as session:
        plan = session.get(InstallPlan, second)
        assert plan is not None
        # A manager-only refusal: the plan still binds, but its revision is
        # not the exact model-version id CivitAI transfers require.
        plan.revision = "not-a-version"
        session.commit()

    graph = _ui_graph()
    graph["nodes"].append(
        {
            "id": 2,
            "type": "LoraLoader",
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "widgets_values": ["styles/second.safetensors"],
            "properties": {"cnr_id": "comfy-core", "version": "0.28.0"},
        }
    )
    selections = [
        {
            "reference_filename": "styles/detail.safetensors",
            "install_plan_id": first,
            "artifact_path": "styles/detail.safetensors",
        },
        {
            "reference_filename": "styles/second.safetensors",
            "install_plan_id": second,
            "artifact_path": "styles/second.safetensors",
        },
    ]
    reviewed = await client.post(
        "/api/workflows/packages/assets/review",
        json={"ui_graph": graph, "selections": selections},
    )
    if reviewed.status_code != 200:
        # The binding contract refused first, which is also all-or-nothing.
        assert (await client.get("/api/jobs")).json() == []
        return

    queued = await client.post(
        "/api/workflows/packages/assets/install",
        json={
            "ui_graph": graph,
            "selections": selections,
            "binding_plan_hash": reviewed.json()["binding_plan_hash"],
        },
    )

    assert queued.status_code == 422
    assert queued.json()["code"] == "asset-download-refused"
    jobs = (await client.get("/api/jobs")).json()
    assert [job for job in jobs if job["kind"] == "download"] == []
