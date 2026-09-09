"""The update check over HTTP: on-demand, per-install verdicts, no guesses."""

from __future__ import annotations

from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm.civitai_catalog import CivitaiCatalog

pytestmark = pytest.mark.asyncio


def _seed_asset(name: str, metadata: Any) -> str:
    from local_lm.db import SessionLocal
    from local_lm.models import ModelAssetInstall

    with SessionLocal() as session:
        asset = ModelAssetInstall(
            name=name,
            kind="lora",
            family="sdxl",
            local_path=f"C:/models/{name}.safetensors",
            size_bytes=10,
            manifest_json={"metadata": metadata} if metadata is not None else {},
            active=False,
        )
        session.add(asset)
        session.commit()
        return asset.id


def _civitai_metadata(model_id: str = "101", version_id: str = "201") -> dict[str, Any]:
    return {
        "provider": "civitai",
        "source_model_id": model_id,
        "source_version_id": version_id,
        "version_name": "v1",
        "published_at": "2026-06-01T00:00:00Z",
    }


async def test_an_empty_library_reports_nothing_to_compare(client: AsyncClient) -> None:
    _seed_asset("huggingface-lora", {"provider": "huggingface"})

    response = await client.get("/api/models/updates")

    assert response.status_code == 200
    assert response.json() == []


async def test_verdicts_are_per_install_and_never_guessed(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_id = _seed_asset("stale-lora", _civitai_metadata())
    current_id = _seed_asset("current-lora", _civitai_metadata(version_id="204"))
    unreachable_id = _seed_asset("unreachable-lora", _civitai_metadata(model_id="999"))
    calls: list[str] = []

    async def fake_versions(self: CivitaiCatalog, model_id: str) -> dict[str, Any]:
        calls.append(model_id)
        if model_id == "999":
            raise ValueError("CivitAI is unavailable")
        return {
            "model_id": model_id,
            "model_name": "Portrait",
            "versions": [
                {
                    "version_id": "204",
                    "version_name": "v4",
                    "published_at": "2026-07-30T00:00:00Z",
                    "base_model": "SDXL 1.0",
                    "changelog": "Sharper hands",
                },
                {"version_id": "201", "version_name": "v1", "published_at": "2026-06-01T00:00:00Z"},
            ],
        }

    monkeypatch.setattr(CivitaiCatalog, "versions", fake_versions)

    response = await client.get("/api/models/updates")

    assert response.status_code == 200
    by_id = {entry["install_id"]: entry for entry in response.json()}
    assert by_id[stale_id]["state"] == "update_available"
    assert by_id[stale_id]["update_version_id"] == "204"
    assert by_id[stale_id]["update_changelog"] == "Sharper hands"
    assert by_id[current_id]["state"] == "current"
    assert by_id[current_id]["update_version_id"] is None
    assert by_id[unreachable_id]["state"] == "unknown"
    # One provider request per distinct model, not per install.
    assert sorted(calls) == ["101", "999"]


async def test_checkpoint_versions_reach_catalog_and_update_reports(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm.db import SessionLocal
    from local_lm.models import ModelInstall, ModelSource

    with SessionLocal() as session:
        source = ModelSource(provider="civitai", remote_id="101", revision="201")
        session.add(source)
        session.flush()
        installs = [
            ModelInstall(
                source_id=source.id,
                name=name,
                role="image",
                engine="comfyui",
                local_path=f"C:/neutral-fixture/{name}",
                manifest_json={"remote_id": "101", "revision": "201"},
                active=False,
            )
            for name in ["Checkpoint A", "Checkpoint B"]
        ]
        session.add_all(installs)
        session.commit()
        install_ids = {install.id for install in installs}

    calls: list[str] = []

    async def fake_versions(self: CivitaiCatalog, model_id: str) -> dict[str, Any]:
        calls.append(model_id)
        return {
            "model_id": model_id,
            "model_name": "Landscape",
            "versions": [
                {"version_id": "201", "published_at": "2026-06-01T00:00:00Z"},
                {"version_id": "202", "published_at": "2026-07-01T00:00:00Z"},
            ],
        }

    monkeypatch.setattr(CivitaiCatalog, "versions", fake_versions)
    response = await client.get("/api/models/updates")
    assert response.status_code == 200
    assert {entry["install_id"] for entry in response.json()} == install_ids
    assert {entry["state"] for entry in response.json()} == {"update_available"}
    assert {entry["update_version_id"] for entry in response.json()} == {"202"}
    assert calls == ["101"]

    response = await client.get("/api/catalog/civitai/101/versions")
    assert response.status_code == 200
    assert [row["installed"] for row in response.json()["versions"]] == [True, False]
    assert response.json()["versions"][0]["installed_as"] in {"Checkpoint A", "Checkpoint B"}


@pytest.mark.parametrize("kind", ["checkpoint", "lora"])
async def test_an_unlisted_install_without_a_date_is_unknown_while_dated_installs_compare(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    from local_lm.db import SessionLocal
    from local_lm.models import ModelInstall, ModelSource

    dated_id = _seed_asset("Dated install", _civitai_metadata())
    current_id = _seed_asset("Listed current install", _civitai_metadata(version_id="202"))
    if kind == "checkpoint":
        with SessionLocal() as session:
            source = ModelSource(provider="civitai", remote_id="101", revision="201")
            session.add(source)
            session.flush()
            install = ModelInstall(
                source_id=source.id,
                name="Undated checkpoint",
                role="image",
                engine="comfyui",
                local_path="C:/neutral-fixture/undated-checkpoint",
                manifest_json={"remote_id": "101", "revision": "201"},
                active=False,
            )
            session.add(install)
            session.commit()
            undated_id = install.id
    else:
        metadata = _civitai_metadata()
        metadata.pop("published_at")
        undated_id = _seed_asset("A undated auxiliary install", metadata)
    calls: list[str] = []

    async def fake_versions(self: CivitaiCatalog, model_id: str) -> dict[str, Any]:
        calls.append(model_id)
        return {
            "model_id": model_id,
            "model_name": "Neutral model",
            "versions": [{"version_id": "202", "published_at": "2026-07-01T00:00:00Z"}],
        }

    monkeypatch.setattr(CivitaiCatalog, "versions", fake_versions)
    response = await client.get("/api/models/updates")
    assert response.status_code == 200
    by_id = {entry["install_id"]: entry for entry in response.json()}
    assert by_id[dated_id]["state"] == "update_available"
    assert by_id[dated_id]["update_version_id"] == "202"
    assert by_id[current_id]["state"] == "current"
    assert by_id[current_id]["update_version_id"] is None
    assert calls == ["101"]
    assert by_id[undated_id]["state"] == "unknown"
    assert by_id[undated_id]["update_version_id"] is None
