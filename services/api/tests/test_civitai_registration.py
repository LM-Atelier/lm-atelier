"""The registered CivitAI source: browse wiring, tokens, and plan provenance."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.civitai_catalog import CivitaiCatalog

pytestmark = pytest.mark.asyncio

SHA256 = "a" * 64


async def test_the_civitai_source_is_registered_beside_hugging_face(
    client: AsyncClient, app: FastAPI
) -> None:
    source = app.state.services.catalog_sources.get("civitai")
    assert isinstance(source, CivitaiCatalog)


async def test_a_stored_token_reaches_the_registered_source(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[str | None] = []
    source = app.state.services.catalog_sources.get("civitai")
    monkeypatch.setattr(source, "set_token", received.append)

    response = await client.put("/api/credentials/civitai", json={"token": "civitai-secret"})

    assert response.status_code == 200
    assert received == ["civitai-secret"]


async def test_the_preflight_route_refuses_unknown_sources_and_bad_ids(
    client: AsyncClient,
) -> None:
    unknown = await client.post(
        "/api/catalog/preflight",
        params={"source": "nowhere", "id": "1"},
        json={"revision": "main", "role": "image", "engine": "comfyui", "selected_files": []},
    )
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "catalog-source-unknown"

    invalid = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "../escape"},
        json={"revision": "main", "role": "image", "engine": "comfyui", "selected_files": []},
    )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "catalog-item-id-invalid"


async def test_a_civitai_preflight_plans_with_full_provenance(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The R238 contract: provider and both identities survive into the plan."""

    detail: dict[str, Any] = {
        "model": {
            "provider": "civitai",
            "remote_id": "201",
            "name": "Portrait LoRA - v1",
            "author": "creator",
            "pipeline_tag": "lora",
            "tags": ["portrait"],
            "downloads": 10,
            "likes": 2,
            "created_at": None,
            "last_modified": None,
            "gated": False,
            "private": False,
            "architecture": "SDXL 1.0",
            "formats": ["safetensors"],
            "quantizations": [],
            "license_id": None,
            "total_size_bytes": 1024,
            "compatibility": "supported",
            "compatibility_reasons": [],
            "required_runtime": "comfyui",
            "content_rating": "general",
        },
        "revision": "201",
        "files": [
            {
                "filename": "portrait.safetensors",
                "size": 1024,
                "sha256": SHA256,
                "kind": "lora",
                "metadata": {
                    "provider": "civitai",
                    "source_model_id": "101",
                    "source_version_id": "201",
                    "source_file_id": "301",
                    "trained_words": ["portrait-style"],
                },
            }
        ],
    }

    async def canned_inspect(
        self: CivitaiCatalog, item_id: str, revision: str = "main", role: str | None = None
    ) -> dict[str, Any]:
        return detail

    monkeypatch.setattr(CivitaiCatalog, "inspect", canned_inspect)

    response = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "201"},
        json={
            "revision": "201",
            "role": "image",
            "engine": "comfyui",
            "selected_files": ["portrait.safetensors"],
            "auxiliary_kind": "lora",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    plan = payload["install_plan"]
    assert plan is not None
    assert plan["provider"] == "civitai"
    (artifact,) = plan["artifacts_json"]
    assert artifact["source_version_id"] == "201"
    assert artifact["source_file_id"] == "301"
    assert artifact["sha256"] == SHA256
    source = payload["file_sources"]["portrait.safetensors"]
    assert source["source_version_id"] == "201"
    assert source["source_file_id"] == "301"


async def test_installed_asset_manifests_feed_the_workflow_inventory(
    client: AsyncClient,
) -> None:
    """A verified LoRA must not read as missing once installed (R240)."""

    from local_lm.api import _local_asset_filenames
    from local_lm.db import SessionLocal
    from local_lm.models import ModelAssetInstall

    with SessionLocal() as session:
        session.add(
            ModelAssetInstall(
                name="portrait-lora",
                kind="lora",
                family="sdxl",
                local_path="C:/data/assets/asset_123",
                size_bytes=10,
                manifest_json={
                    "comfy_name": "portrait-style-v1.safetensors",
                    "files": ["subdir/portrait-style-v1.safetensors"],
                },
                active=True,
            )
        )
        session.commit()
        names = _local_asset_filenames(session)

    assert "portrait-style-v1.safetensors" in names
    assert "subdir/portrait-style-v1.safetensors" in names
