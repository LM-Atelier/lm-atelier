"""Suggested LoRAs for the model family a workflow runs."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy.orm import Session
from test_auxiliary_assets import _activation, _bind_model, _workflow

from local_lm.civitai_catalog import CivitaiCatalog
from local_lm.db import SessionLocal
from local_lm.lora_suggestions import (
    MAX_LORA_SUGGESTIONS,
    LoraSuggestionScope,
    lora_suggestion_scope,
    suggested_loras,
)
from local_lm.models import ModelAssetInstall, ModelInstall, WorkflowRevision
from local_lm.schemas import CatalogModel, CatalogPage


def _card(remote_id: str, **updates: Any) -> CatalogModel:
    values: dict[str, Any] = {
        "remote_id": remote_id,
        "name": f"LoRA {remote_id}",
        "provider": "civitai",
        "parent_model_id": f"model-{remote_id}",
        "compatibility": "advanced_import",
        "content_rating": "general",
    }
    values.update(updates)
    return CatalogModel.model_validate(values)


def _scope(**updates: Any) -> LoraSuggestionScope:
    values: dict[str, Any] = {
        "family": "sdxl",
        "base_models": ("SDXL 1.0",),
        "gap": None,
        "installed_model_ids": frozenset(),
    }
    values.update(updates)
    return LoraSuggestionScope(**values)


def _revision_with_family(
    session: Session, family: str | None, *, contract: bool
) -> WorkflowRevision:
    revision = _workflow(session)
    base = session.get(ModelInstall, revision.dependencies_json["model_install_ids"][0])
    assert base is not None
    base.manifest_json = {} if family is None else {"family": family}
    if contract:
        revision.dependency_contract_sha256 = "a" * 64
        activation = _activation(session, revision)
        _bind_model(session, revision, activation, base, 0)
    session.flush()
    return revision


@pytest.mark.parametrize("contract", [False, True])
@pytest.mark.parametrize(
    ("family", "base_models", "gap"),
    [
        ("stable-diffusion-xl", ("SDXL 1.0",), None),
        ("sdxl", ("SDXL 1.0",), None),
        ("stable-diffusion", ("SD 1.5",), None),
        ("flux", ("Flux.1 D", "Flux.1 S"), None),
        ("krea2", ("Krea 2",), None),
        ("z-image", (), "family_unsupported"),
        (None, (), "family_unknown"),
    ],
)
async def test_the_workflow_model_family_decides_which_base_models_are_asked_for(
    client: AsyncClient,
    contract: bool,
    family: str | None,
    base_models: tuple[str, ...],
    gap: str | None,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _revision_with_family(session, family, contract=contract)
        scope = lora_suggestion_scope(session, revision)

    assert scope.base_models == base_models
    assert scope.gap == gap
    assert scope.family == family


async def test_a_contract_revision_without_a_ready_activation_has_no_known_family(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        revision = _revision_with_family(session, "sdxl", contract=False)
        revision.dependency_contract_sha256 = "a" * 64
        session.flush()
        scope = lora_suggestion_scope(session, revision)

    assert scope.gap == "family_unknown"
    assert scope.base_models == ()


def test_suggestions_keep_one_installable_general_card_per_lora_in_order() -> None:
    page = CatalogPage(
        items=[
            _card("1"),
            _card("2", parent_model_id="model-1"),
            _card("3", compatibility="unsupported"),
            _card("4", content_rating="unknown"),
            _card("5", provider="huggingface"),
            _card("6"),
            _card("7"),
        ]
    )

    kept = suggested_loras(_scope(installed_model_ids=frozenset({"model-7"})), page)

    assert [card.remote_id for card in kept] == ["1", "6"]


def test_suggestions_are_bounded() -> None:
    page = CatalogPage(items=[_card(str(index)) for index in range(40)])

    assert len(suggested_loras(_scope(), page)) == MAX_LORA_SUGGESTIONS


async def test_the_route_asks_for_top_rated_loras_for_the_family_and_leaves_installed_ones_out(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        revision = _revision_with_family(session, "stable-diffusion-xl", contract=True)
        session.add(
            ModelAssetInstall(
                name="Installed style",
                kind="lora",
                family="sdxl",
                local_path="C:/managed/installed.safetensors",
                size_bytes=10,
                manifest_json={
                    "metadata": {
                        "provider": "civitai",
                        "source_model_id": "model-2",
                        "source_version_id": "2",
                    }
                },
                active=True,
            )
        )
        session.commit()
        revision_id = revision.id
    asked: list[dict[str, Any]] = []

    async def search(**kwargs: Any) -> CatalogPage:
        asked.append(kwargs)
        return CatalogPage(items=[_card("1"), _card("2"), _card("3")], next_cursor=None, stale=True)

    catalog = app.state.services.catalog_sources.get("civitai")
    assert isinstance(catalog, CivitaiCatalog)
    monkeypatch.setattr(catalog, "search", search)

    response = await client.get(f"/api/workflow-revisions/{revision_id}/lora-suggestions")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["family"] == "stable-diffusion-xl"
    assert body["gap"] is None
    assert body["stale"] is True
    assert [item["remote_id"] for item in body["items"]] == ["1", "3"]
    assert asked == [
        {
            "role": "lora",
            "sort": "likes",
            "limit": 30,
            "cursor": None,
            "base_models": ("SDXL 1.0",),
        }
    ]


async def test_the_route_says_why_there_are_no_suggestions_without_asking_civitai(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        revision = _revision_with_family(session, "z-image", contract=False)
        session.commit()
        revision_id = revision.id

    async def search(**_kwargs: Any) -> CatalogPage:
        raise AssertionError("an unsupported family must not reach CivitAI")

    monkeypatch.setattr(app.state.services.catalog_sources.get("civitai"), "search", search)

    response = await client.get(f"/api/workflow-revisions/{revision_id}/lora-suggestions")
    missing = await client.get("/api/workflow-revisions/wfrev_missing/lora-suggestions")

    assert response.status_code == 200
    assert response.json() == {
        "family": "z-image",
        "gap": "family_unsupported",
        "items": [],
        "next_cursor": None,
        "stale": False,
    }
    assert missing.status_code == 404


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (ValueError("CivitAI catalog cursor is invalid"), 422, "catalog-request-invalid"),
        (RuntimeError("connection refused"), 503, "catalog-unavailable"),
    ],
)
async def test_the_route_reports_catalog_failures_without_echoing_them(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    status: int,
    code: str,
) -> None:
    with SessionLocal() as session:
        revision = _revision_with_family(session, "flux", contract=False)
        session.commit()
        revision_id = revision.id

    async def search(**_kwargs: Any) -> CatalogPage:
        raise failure

    monkeypatch.setattr(app.state.services.catalog_sources.get("civitai"), "search", search)

    response = await client.get(
        f"/api/workflow-revisions/{revision_id}/lora-suggestions",
        params={"cursor": "https://civitai.com/api/v1/models?cursor=next"},
    )

    assert response.status_code == status
    assert response.json()["code"] == code
    assert "connection refused" not in response.text
    assert "cursor is invalid" not in response.text
