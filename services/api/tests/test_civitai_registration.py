"""The registered CivitAI source: browse wiring, tokens, and plan provenance."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.civitai_catalog import CivitaiCatalog

pytestmark = pytest.mark.asyncio

SHA256 = "a" * 64


def _lora_detail() -> dict[str, Any]:
    """One CivitAI version publishing a single LoRA file under one name."""
    return {
        "model": {
            "provider": "civitai",
            "remote_id": "201",
            "name": "Portrait",
            "author": "creator",
            "pipeline_tag": "lora",
            "tags": [],
            "downloads": 0,
            "likes": 0,
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
                "source_file_id": "301",
                "source_version_id": "201",
                "format": "SafeTensor",
                "pickle_scan_result": "Success",
                "virus_scan_result": "Success",
                "metadata": {
                    "provider": "civitai",
                    "source_model_id": "101",
                    "source_version_id": "201",
                    "model_type": "LORA",
                    "base_model": "SDXL 1.0",
                    "trained_words": ["portrait-style"],
                },
            }
        ],
    }


async def test_the_civitai_source_is_registered_beside_hugging_face(
    client: AsyncClient, app: FastAPI
) -> None:
    source = app.state.services.catalog_sources.get("civitai")
    assert isinstance(source, CivitaiCatalog)


async def test_a_stored_token_reaches_the_registered_source(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:

    from local_lm import credentials as credentials_module
    from local_lm.credentials import CredentialStore

    values: dict[str, str] = {}
    monkeypatch.setattr(
        CredentialStore,
        "state",
        lambda self, provider="huggingface": credentials_module.CredentialState(
            configured=provider in values,
            source="credential_vault" if provider in values else "none",
            vault_available=True,
        ),
    )
    monkeypatch.setattr(
        CredentialStore,
        "token",
        lambda self, provider="huggingface": values.get(provider),
    )

    def set_token(self: Any, value: str, provider: str = "huggingface") -> None:
        values[provider] = value.strip()

    monkeypatch.setattr(CredentialStore, "set_token", set_token)
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


async def test_a_workflow_owned_civitai_lora_is_not_classified_as_a_checkpoint(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect: declaring ownership correctly is what broke the plan.

    A workflow-owned asset sends `workflow_reference_kind` and must not also
    send `auxiliary_kind`, because the planner refuses that as conflicting
    ownership. The provider-declaration substitute named only the auxiliary
    field, so every workflow-owned CivitAI LoRA inspected as a checkpoint and
    was blocked as a kind mismatch.
    """
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
        # The exact shape production _normalize_files() emits: identities at
        # the file top level; shared metadata carries the version and the
        # provider's typed model_type, never the file id.
        "files": [
            {
                "filename": "portrait.safetensors",
                "size": 1024,
                "sha256": SHA256,
                "source_file_id": "301",
                "source_version_id": "201",
                "format": "SafeTensor",
                "pickle_scan_result": "Success",
                "virus_scan_result": "Success",
                "metadata": {
                    "provider": "civitai",
                    "source_model_id": "101",
                    "source_version_id": "201",
                    "version_name": "v1",
                    "published_at": None,
                    "model_type": "LORA",
                    "base_model": "SDXL 1.0",
                    "base_models": ["SDXL 1.0"],
                    "tags": ["portrait"],
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
    monkeypatch.setattr(app.state.services.settings, "civitai_token", "vaulted-token")

    response = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "201"},
        json={
            "revision": "201",
            "role": "image",
            "engine": "comfyui",
            "selected_files": ["portrait.safetensors"],
            "workflow_reference_kind": "lora",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["can_install"] is True
    plan_out = payload["install_plan"]
    assert plan_out is not None
    (artifact,) = plan_out["artifacts_json"]
    assert artifact["kind"] == "lora"
    assert artifact["target_folder"] == "loras"


async def test_a_civitai_preflight_composes_into_the_download_manager(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R238/R245: provider and both identities survive from the normalized
    detail through the plan into the manager's own source synthesis, and the
    request carries no catalog file sources - the immutable plan is the only
    identity the manager accepts. No Hugging Face call anywhere."""

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
        # The exact shape production _normalize_files() emits: identities at
        # the file top level; shared metadata carries the version and the
        # provider's typed model_type, never the file id.
        "files": [
            {
                "filename": "portrait.safetensors",
                "size": 1024,
                "sha256": SHA256,
                "source_file_id": "301",
                "source_version_id": "201",
                "format": "SafeTensor",
                "pickle_scan_result": "Success",
                "virus_scan_result": "Success",
                "metadata": {
                    "provider": "civitai",
                    "source_model_id": "101",
                    "source_version_id": "201",
                    "version_name": "v1",
                    "published_at": None,
                    "model_type": "LORA",
                    "base_model": "SDXL 1.0",
                    "base_models": ["SDXL 1.0"],
                    "tags": ["portrait"],
                    "trained_words": ["portrait-style"],
                },
            }
        ],
    }

    inspected_roles: list[str | None] = []

    async def canned_inspect(
        self: CivitaiCatalog, item_id: str, revision: str = "main", role: str | None = None
    ) -> dict[str, Any]:
        inspected_roles.append(role)
        return detail

    monkeypatch.setattr(CivitaiCatalog, "inspect", canned_inspect)

    # Without a token, even a public card blocks: the manager authenticates
    # every CivitAI transfer, so preflight must say so before the queue.
    monkeypatch.setattr(app.state.services.settings, "civitai_token", None)
    refused = await client.post(
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
    assert refused.status_code == 200
    assert refused.json()["can_install"] is False
    access = next(c for c in refused.json()["checks"] if c["id"] == "access")
    assert access["status"] == "block"

    monkeypatch.setattr(app.state.services.settings, "civitai_token", "vaulted-token")
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
    assert inspected_roles == ["lora", "lora"]
    assert payload["can_install"] is True
    plan_out = payload["install_plan"]
    assert plan_out is not None
    assert plan_out["provider"] == "civitai"
    assert plan_out["compatibility"] == "supported"
    (artifact,) = plan_out["artifacts_json"]
    assert artifact["kind"] == "lora"
    assert artifact["source_version_id"] == "201"
    assert artifact["source_file_id"] == "301"
    assert artifact["sha256"] == SHA256
    # The manager accepts identity only from the immutable plan.
    assert payload["file_sources"] == {}

    from local_lm.api import _planned_download_fields
    from local_lm.db import SessionLocal
    from local_lm.models import InstallPlan
    from local_lm.schemas import DownloadRequest

    services = app.state.services
    with SessionLocal() as session:
        plan = session.get(InstallPlan, plan_out["id"])
        assert plan is not None
        fields = _planned_download_fields(plan)
        assert fields["file_sources"] == {}
        request = DownloadRequest.model_validate({**fields, "install_plan_id": plan.id})

        def observed_start(*args: object, **kwargs: object) -> None:
            return None

        monkeypatch.setattr(services.downloads, "start", observed_start, raising=False)
        job = services.downloads.create(session, request)
        assert job.kind == "download"
        siblings, sources, revision, extra = await services.downloads._download_sources(
            request, plan
        )
    (sibling,) = siblings
    assert sibling.rfilename == "portrait.safetensors"
    assert sibling.size == 1024
    source = sources["portrait.safetensors"]
    assert source.provider == "civitai"
    assert source.remote_id == "201"
    assert source.source_file_id == "301"
    assert revision == "201"
    assert extra == {"source_version_id": "201"}


async def test_civitai_preflight_keeps_the_provider_primary_duplicate(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary_sha = "b" * 64
    alternate_sha = "c" * 64
    shared = {
        "filename": "duplicate.safetensors",
        "source_version_id": "201",
        "format": "SafeTensor",
        "pickle_scan_result": "Success",
        "virus_scan_result": "Success",
        "metadata": {
            "provider": "civitai",
            "source_model_id": "101",
            "source_version_id": "201",
            "model_type": "LORA",
            "content_rating": "general",
        },
    }
    detail: dict[str, Any] = {
        "model": {
            "provider": "civitai",
            "remote_id": "201",
            "name": "Duplicate variants",
            "compatibility": "supported",
            "required_runtime": "comfyui",
            "content_rating": "general",
        },
        "revision": "201",
        "files": [
            {
                **shared,
                "size": 24_000,
                "sha256": alternate_sha,
                "source_file_id": "302",
                "source_file_type": "Other",
                "source_file_precision": "bf16",
            },
            {
                **shared,
                "size": 12_000,
                "sha256": primary_sha,
                "source_file_id": "301",
                "source_file_type": "Model",
                "source_file_precision": "fp8",
            },
        ],
    }

    async def canned_inspect(
        self: CivitaiCatalog, item_id: str, revision: str = "main", role: str | None = None
    ) -> dict[str, Any]:
        return detail

    monkeypatch.setattr(CivitaiCatalog, "inspect", canned_inspect)
    monkeypatch.setattr(app.state.services.settings, "civitai_token", "vaulted-token")

    automatic = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "201"},
        json={
            "revision": "201",
            "role": "image",
            "engine": "comfyui",
            "selected_files": [],
            "auxiliary_kind": "lora",
        },
    )
    assert automatic.status_code == 200
    automatic_payload = automatic.json()
    assert automatic_payload["can_install"] is True
    assert automatic_payload["selected_files"] == ["duplicate.safetensors"]
    (artifact,) = automatic_payload["install_plan"]["artifacts_json"]
    assert artifact["source_file_id"] == "301"
    assert artifact["sha256"] == primary_sha
    assert artifact["size_bytes"] == 12_000

    ambiguous = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "201"},
        json={
            "revision": "201",
            "role": "image",
            "engine": "comfyui",
            "selected_files": ["duplicate.safetensors"],
            "auxiliary_kind": "lora",
        },
    )
    assert ambiguous.status_code == 200
    ambiguous_payload = ambiguous.json()
    assert ambiguous_payload["can_install"] is False
    selection = next(check for check in ambiguous_payload["checks"] if check["id"] == "selection")
    assert selection["status"] == "block"
    assert "exact file variant" in selection["detail"]


async def test_civitai_automatic_selection_skips_an_unsafe_unique_file(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe_sha = "d" * 64
    detail: dict[str, Any] = {
        "model": {
            "provider": "civitai",
            "remote_id": "201",
            "name": "Mixed files",
            "compatibility": "supported",
            "required_runtime": "comfyui",
            "content_rating": "general",
        },
        "revision": "201",
        "files": [
            {
                "filename": "model.safetensors",
                "size": 50_000,
                "sha256": None,
                "source_file_id": "302",
                "source_version_id": "201",
                "source_file_type": "Model",
                "pickle_scan_result": "Danger",
                "virus_scan_result": "Success",
                "metadata": {
                    "provider": "civitai",
                    "source_model_id": "101",
                    "source_version_id": "201",
                    "model_type": "LORA",
                    "content_rating": "general",
                },
            },
            {
                "filename": "safe.safetensors",
                "size": 10_000,
                "sha256": safe_sha,
                "source_file_id": "301",
                "source_version_id": "201",
                "source_file_type": "Other",
                "pickle_scan_result": "Success",
                "virus_scan_result": "Success",
                "metadata": {
                    "provider": "civitai",
                    "source_model_id": "101",
                    "source_version_id": "201",
                    "model_type": "LORA",
                    "content_rating": "general",
                },
            },
        ],
    }

    async def canned_inspect(
        self: CivitaiCatalog, item_id: str, revision: str = "main", role: str | None = None
    ) -> dict[str, Any]:
        return detail

    monkeypatch.setattr(CivitaiCatalog, "inspect", canned_inspect)
    monkeypatch.setattr(app.state.services.settings, "civitai_token", "vaulted-token")

    response = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "201"},
        json={
            "revision": "201",
            "role": "image",
            "engine": "comfyui",
            "selected_files": [],
            "auxiliary_kind": "lora",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["can_install"] is True
    assert payload["selected_files"] == ["safe.safetensors"]
    (artifact,) = payload["install_plan"]["artifacts_json"]
    assert artifact["source_file_id"] == "301"
    assert artifact["sha256"] == safe_sha


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


async def test_an_ambiguous_filename_comes_back_with_its_choices(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal that says "choose a variant" must say which variants.

    One CivitAI version can publish the same safetensors name several times at
    different precisions. Preflight rightly refuses to guess, and used to give
    the caller no way to answer - an instruction that could not be followed.
    """

    def _file(file_id: str, precision: str, size: int) -> dict[str, Any]:
        return {
            "filename": "lustify.safetensors",
            "size": size,
            "sha256": f"{file_id}" + "0" * (64 - len(file_id)),
            "source_file_id": file_id,
            "source_version_id": "201",
            "format": "SafeTensor",
            "pickle_scan_result": "Success",
            "virus_scan_result": "Success",
            "source_file_precision": precision,
            "metadata": {
                "provider": "civitai",
                "source_model_id": "101",
                "source_version_id": "201",
                "model_type": "Checkpoint",
            },
        }

    detail: dict[str, Any] = {
        "model": {
            "provider": "civitai",
            "remote_id": "201",
            "name": "Lustify",
            "author": "creator",
            "pipeline_tag": "checkpoint",
            "tags": [],
            "downloads": 0,
            "likes": 0,
            "created_at": None,
            "last_modified": None,
            "gated": False,
            "private": False,
            "architecture": "SDXL 1.0",
            "formats": ["safetensors"],
            "quantizations": [],
            "license_id": None,
            "total_size_bytes": 3,
            "compatibility": "supported",
            "compatibility_reasons": [],
            "required_runtime": "comfyui",
            "content_rating": "general",
        },
        "revision": "201",
        "files": [_file("301", "fp16", 2), _file("302", "fp8", 1)],
    }

    async def canned_inspect(
        self: CivitaiCatalog, item_id: str, revision: str = "main", role: str | None = None
    ) -> dict[str, Any]:
        return detail

    monkeypatch.setattr(CivitaiCatalog, "inspect", canned_inspect)
    monkeypatch.setattr(app.state.services.settings, "civitai_token", "vaulted-token")

    response = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "201"},
        json={
            "revision": "201",
            "role": "image",
            "engine": "comfyui",
            "selected_files": ["lustify.safetensors"],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["can_install"] is False
    offered = response.json()["file_variants"]["lustify.safetensors"]
    # The resolver's own order, preferred variant first, so the chooser
    # presents what selection would have picked at the top.
    assert [item["source_file_id"] for item in offered] == ["302", "301"]
    assert [item["precision"] for item in offered] == ["fp8", "fp16"]
    assert [item["size_bytes"] for item in offered] == [1, 2]
    # Naming a choice, not asserting a fact: no hash and no URL travel back.
    assert all(
        set(item) == {"source_file_id", "filename", "size_bytes", "precision"} for item in offered
    )


async def test_a_version_with_one_file_per_name_offers_no_choices(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offering a list of one would make every install a decision."""
    detail = _lora_detail()

    async def canned_inspect(
        self: CivitaiCatalog, item_id: str, revision: str = "main", role: str | None = None
    ) -> dict[str, Any]:
        return detail

    monkeypatch.setattr(CivitaiCatalog, "inspect", canned_inspect)
    monkeypatch.setattr(app.state.services.settings, "civitai_token", "vaulted-token")

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
    assert response.json()["file_variants"] == {}


async def test_choosing_an_exact_variant_plans_that_row_and_not_the_preferred_one(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two-step exchange the chooser actually performs.

    Listing the choices proves nothing about answering with one. The retry
    names an exact provider id and no filename at all - the two forms are
    mutually exclusive - and it travels the workflow-owned path, where an
    earlier guard counted filenames and refused a request that correctly
    supplied none.

    The requested id is the one ranking would not have chosen. Asking for the
    preferred row proves nothing: it comes back whether or not anything
    resolved the id, which is how an earlier version of this test passed while
    the binding did not exist.
    """

    def _file(file_id: str, precision: str, size: int) -> dict[str, Any]:
        return {
            "filename": "lustify.safetensors",
            "size": size,
            "sha256": file_id[0] * 64,
            "source_file_id": file_id,
            "source_version_id": "201",
            "format": "SafeTensor",
            "pickle_scan_result": "Success",
            "virus_scan_result": "Success",
            "source_file_precision": precision,
            "metadata": {
                "provider": "civitai",
                "source_model_id": "101",
                "source_version_id": "201",
                "model_type": "LORA",
            },
        }

    detail = _lora_detail()
    detail["files"] = [_file("301", "fp16", 2048), _file("302", "fp8", 1024)]

    async def canned_inspect(
        self: CivitaiCatalog, item_id: str, revision: str = "main", role: str | None = None
    ) -> dict[str, Any]:
        return detail

    monkeypatch.setattr(CivitaiCatalog, "inspect", canned_inspect)
    monkeypatch.setattr(app.state.services.settings, "civitai_token", "vaulted-token")

    listed = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "201"},
        json={
            "revision": "201",
            "role": "image",
            "engine": "comfyui",
            "selected_files": ["lustify.safetensors"],
            "workflow_reference_kind": "lora",
        },
    )
    assert listed.status_code == 200, listed.text
    offered = listed.json()["file_variants"]["lustify.safetensors"]
    assert sorted(item["source_file_id"] for item in offered) == ["301", "302"]

    chosen = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "201"},
        json={
            "revision": "201",
            "role": "image",
            "engine": "comfyui",
            "selected_files": [],
            "selected_file_ids": ["301"],
            "workflow_reference_kind": "lora",
        },
    )

    assert chosen.status_code == 200, chosen.text
    plan = chosen.json()["install_plan"]
    assert plan is not None
    (artifact,) = plan["artifacts_json"]
    # The requested variant, not the one ranking prefers. Asking for the
    # preferred row would pass whether or not anything resolved the id.
    assert artifact["source_file_id"] == "301"
    assert artifact["sha256"] == "3" * 64
    assert artifact["size_bytes"] == 2048
    # An answered request has nothing left to choose.
    assert chosen.json()["file_variants"] == {}


async def test_naming_a_file_both_ways_is_refused(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One identity in one form; naming it twice is a request nobody meant."""
    detail = _lora_detail()

    async def canned_inspect(
        self: CivitaiCatalog, item_id: str, revision: str = "main", role: str | None = None
    ) -> dict[str, Any]:
        return detail

    monkeypatch.setattr(CivitaiCatalog, "inspect", canned_inspect)
    monkeypatch.setattr(app.state.services.settings, "civitai_token", "vaulted-token")

    response = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "201"},
        json={
            "revision": "201",
            "role": "image",
            "engine": "comfyui",
            "selected_files": ["portrait.safetensors"],
            "selected_file_ids": ["301"],
            "workflow_reference_kind": "lora",
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-asset-file-not-exact"


async def test_an_unsafe_variant_is_never_offered_as_a_choice(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chooser that lists a row selection would refuse hands out dead ends."""

    def _file(file_id: str, size: int, **overrides: Any) -> dict[str, Any]:
        return {
            "filename": "lustify.safetensors",
            "size": size,
            "sha256": file_id[0] * 64,
            "source_file_id": file_id,
            "source_version_id": "201",
            "format": "SafeTensor",
            "pickle_scan_result": "Success",
            "virus_scan_result": "Success",
            "source_file_precision": "fp16",
            "metadata": {"provider": "civitai", "source_version_id": "201"},
            **overrides,
        }

    detail = _lora_detail()
    detail["files"] = [
        _file("301", 2048),
        _file("302", 1024),
        _file("303", 512, pickle_scan_result="Danger"),
        _file("304", 256, virus_scan_result="Infected"),
        _file("305", 128, sha256=""),
    ]

    async def canned_inspect(
        self: CivitaiCatalog, item_id: str, revision: str = "main", role: str | None = None
    ) -> dict[str, Any]:
        return detail

    monkeypatch.setattr(CivitaiCatalog, "inspect", canned_inspect)
    monkeypatch.setattr(app.state.services.settings, "civitai_token", "vaulted-token")

    response = await client.post(
        "/api/catalog/preflight",
        params={"source": "civitai", "id": "201"},
        json={
            "revision": "201",
            "role": "image",
            "engine": "comfyui",
            "selected_files": ["lustify.safetensors"],
            "workflow_reference_kind": "lora",
        },
    )

    assert response.status_code == 200, response.text
    offered = response.json()["file_variants"]["lustify.safetensors"]
    assert sorted(item["source_file_id"] for item in offered) == ["301", "302"]
