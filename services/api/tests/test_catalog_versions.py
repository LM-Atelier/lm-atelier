"""Every installable version of one model, and which are already here.

A version is what installs, so a card that groups versions still has to let
someone choose one deliberately. This is the list that choice is made from.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.civitai_catalog import CivitaiCatalog

pytestmark = pytest.mark.asyncio

_SUMMARY: dict[str, Any] = {
    "model_id": "4201",
    "model_name": "Lustify",
    "versions": [
        {
            "version_id": "9002",
            "version_name": "v4.0",
            "published_at": "2026-07-01T00:00:00Z",
            "base_model": "SDXL 1.0",
            "changelog": "Sharper hands.",
            "size_bytes": 2_000_000,
        },
        {
            "version_id": "9001",
            "version_name": "v3.0",
            "published_at": "2026-05-01T00:00:00Z",
            "base_model": "SDXL 1.0",
            "changelog": None,
            "size_bytes": 1_000_000,
        },
    ],
}


def _stub(monkeypatch: pytest.MonkeyPatch, summary: dict[str, Any] | Exception) -> None:
    async def versions(self: CivitaiCatalog, model_id: str) -> dict[str, Any]:
        del self, model_id
        if isinstance(summary, Exception):
            raise summary
        return summary

    monkeypatch.setattr(CivitaiCatalog, "versions", versions)


async def test_lists_every_version_with_what_distinguishes_it(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    del app
    _stub(monkeypatch, _SUMMARY)

    response = await client.get("/api/catalog/civitai/4201/versions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "Lustify"
    assert [row["version_id"] for row in payload["versions"]] == ["9002", "9001"]
    newest = payload["versions"][0]
    assert newest["size_bytes"] == 2_000_000
    assert newest["base_model"] == "SDXL 1.0"
    assert newest["changelog"] == "Sharper hands."


async def test_says_unknown_rather_than_not_installed_when_it_cannot_tell(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinction that stops someone installing a second copy.

    Checkpoint installs record no provider version, so nothing on disk can be
    matched against these rows. Reporting `false` would be a claim the data
    cannot support; `null` says plainly that we cannot see.
    """
    del app
    _stub(monkeypatch, _SUMMARY)

    response = await client.get("/api/catalog/civitai/4201/versions")

    assert [row["installed"] for row in response.json()["versions"]] == [None, None]


async def test_refuses_an_id_that_is_not_a_model_id(client: AsyncClient, app: FastAPI) -> None:
    del app
    response = await client.get("/api/catalog/civitai/..%2Fescape/versions")

    assert response.status_code in {404, 422}


async def test_a_provider_that_cannot_answer_says_so(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    del app
    _stub(monkeypatch, ValueError("CivitAI item is not available in the general catalog"))

    response = await client.get("/api/catalog/civitai/4201/versions")

    assert response.status_code == 404
    assert response.json()["code"] == "catalog-item-not-found"
