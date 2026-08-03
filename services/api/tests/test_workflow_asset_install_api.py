"""Workflow asset review and install: stateless, server-rebuilt, id-only.

The browser holds the UI graph and sends explicit selections; everything
else - the analysis, the binding, the digests, the requests - is rebuilt
here from current records. These pin that the endpoints never trust
client-supplied evidence and that queueing honors the reviewed hash.
"""

from __future__ import annotations

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


def _seed_plan(plan_id: str = "plan_lora") -> str:
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
                plan_hash="b" * 64,
                resolver_version=INSTALL_RESOLVER_VERSION,
                compatibility="supported",
                artifacts_json=[
                    {
                        "path": "styles/detail.safetensors",
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
