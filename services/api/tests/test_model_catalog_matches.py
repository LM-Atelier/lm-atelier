from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event

from local_lm import api
from local_lm.db import SessionLocal
from local_lm.models import ModelInstall


def _install(identity: str, *, role: str = "image", active: bool = True, **manifest: str) -> None:
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id=identity,
                name=identity,
                role=role,
                engine="mock",
                local_path=f"C:/neutral-fixture/{identity}",
                active=active,
                manifest_json=manifest,
            )
        )
        session.commit()


async def test_catalog_matches_find_remote_aliases_outside_model_pages(client: AsyncClient) -> None:
    for index in range(4):
        _install(f"catalog-install-{index}", remote_id=f"neutral/source-{index}")
    _install("catalog-aliased", remote_id="neutral/resolved", source_remote_id="neutral/requested")
    response = await client.get(
        "/api/models/catalog-matches",
        params=[
            ("role", "image"),
            ("remote_id", "neutral/source-0"),
            ("remote_id", "neutral/requested"),
            ("remote_id", "neutral/resolved"),
            ("remote_id", "neutral/missing"),
            ("remote_id", "neutral/requested"),
        ],
    )
    assert response.status_code == 200
    assert response.json() == {
        "remote_ids": ["neutral/requested", "neutral/resolved", "neutral/source-0"],
        "workflow_template_ids": [],
    }


async def test_catalog_matches_preserve_exact_workflow_variants(client: AsyncClient) -> None:
    _install("catalog-template", remote_id="neutral/shared", workflow_template_id="neutral-edit")
    response = await client.get(
        "/api/models/catalog-matches",
        params=[
            ("role", "image"),
            ("remote_id", "neutral/shared"),
            ("workflow_template_id", "neutral-create"),
            ("workflow_template_id", "neutral-edit"),
            ("workflow_template_id", "neutral-edit"),
        ],
    )
    assert response.status_code == 200
    assert response.json() == {
        "remote_ids": ["neutral/shared"],
        "workflow_template_ids": ["neutral-edit"],
    }


async def test_catalog_matches_exclude_inactive_and_other_roles(client: AsyncClient) -> None:
    _install(
        "catalog-inactive",
        active=False,
        remote_id="neutral/inactive",
        workflow_template_id="inactive",
    )
    _install(
        "catalog-other-role", role="video", remote_id="neutral/video", workflow_template_id="video"
    )
    response = await client.get(
        "/api/models/catalog-matches",
        params=[
            ("role", "image"),
            ("remote_id", "neutral/inactive"),
            ("remote_id", "neutral/video"),
            ("workflow_template_id", "inactive"),
            ("workflow_template_id", "video"),
        ],
    )
    assert response.status_code == 200
    assert response.json() == {"remote_ids": [], "workflow_template_ids": []}


async def test_catalog_matches_empty_request_returns_no_library_identities(
    client: AsyncClient,
) -> None:
    _install(
        "catalog-unrequested", remote_id="neutral/unrequested", workflow_template_id="unrequested"
    )
    response = await client.get("/api/models/catalog-matches", params={"role": "image"})
    assert response.status_code == 200
    assert response.json() == {"remote_ids": [], "workflow_template_ids": []}


@pytest.mark.parametrize("field", ["remote_id", "workflow_template_id"])
async def test_catalog_matches_bound_each_identity_batch(client: AsyncClient, field: str) -> None:
    response = await client.get(
        "/api/models/catalog-matches",
        params=[("role", "image"), *[(field, str(i)) for i in range(201)]],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("role", [None, "unknown"])
async def test_catalog_matches_require_a_model_role(client: AsyncClient, role: str | None) -> None:
    response = await client.get(
        "/api/models/catalog-matches", params={} if role is None else {"role": role}
    )
    assert response.status_code == 422


async def test_catalog_matches_do_not_hydrate_models_or_check_readiness(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install("catalog-scalar", remote_id="neutral/scalar", workflow_template_id="scalar")
    loaded: list[str] = []

    def track(row: ModelInstall, _context: object) -> None:
        loaded.append(row.id)

    def unexpected_evidence(*args: object, **kwargs: object) -> None:
        pytest.fail("Catalog identity lookup must not check model readiness")

    monkeypatch.setattr(api, "current_capability_evidence", unexpected_evidence)
    event.listen(ModelInstall, "load", track)
    try:
        response = await client.get(
            "/api/models/catalog-matches",
            params={
                "role": "image",
                "remote_id": "neutral/scalar",
                "workflow_template_id": "scalar",
            },
        )
    finally:
        event.remove(ModelInstall, "load", track)
    assert response.status_code == 200
    assert response.json() == {
        "remote_ids": ["neutral/scalar"],
        "workflow_template_ids": ["scalar"],
    }
    assert loaded == []
