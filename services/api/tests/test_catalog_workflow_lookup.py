from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm.catalog import HuggingFaceCatalog
from local_lm.comfy_templates import ComfyModelDependency, ComfyTemplate, ComfyTemplateRegistry
from local_lm.config import Settings
from local_lm.schemas import CatalogModel, CatalogPage


def _catalog(monkeypatch: pytest.MonkeyPatch, *, empty: bool = False):
    page = CatalogPage(
        items=[]
        if empty
        else [
            CatalogModel(remote_id="fixture/ready", name="Ready model", compatibility="likely"),
            CatalogModel(
                remote_id="fixture/adaptive",
                name="Adaptive model",
                compatibility="likely",
                formats=["safetensors"],
            ),
            CatalogModel(
                remote_id="fixture/unsupported",
                name="Other model",
                compatibility="likely",
                formats=["gguf"],
            ),
        ],
        next_cursor="next-page",
        stale=True,
    )
    matching = {"FIXTURE/READY"}
    enumerated: list[str] = []

    async def search(_source: HuggingFaceCatalog, **_params: Any) -> CatalogPage:
        return page

    def available(_registry: ComfyTemplateRegistry, role: str) -> list[ComfyTemplate]:
        enumerated.append(role)
        return [
            ComfyTemplate(
                id=remote_id,
                path=Path("neutral-workflow.json"),
                role=role,
                operation="text_to_image" if role == "image" else "text_to_video",
                score=1,
                sha256="a" * 64,
                dependencies=(
                    ComfyModelDependency(
                        remote_id=remote_id,
                        revision="main",
                        path="model.safetensors",
                        directory="checkpoints",
                        name="Neutral model",
                        url="https://example.test/model.safetensors",
                    ),
                ),
            )
            for remote_id in matching
        ]

    monkeypatch.setattr(HuggingFaceCatalog, "search", search)
    monkeypatch.setattr(ComfyTemplateRegistry, "available", available)
    return enumerated, matching


@pytest.mark.parametrize("role", ["image", "video"])
async def test_catalog_enumerates_workflows_once_per_page(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    monkeypatch.setattr(settings, "media_engine", "comfyui")
    enumerated, matching = _catalog(monkeypatch)

    response = await client.get("/api/catalog", params={"role": role})
    assert response.status_code == 200
    page = response.json()
    assert [item["remote_id"] for item in page["items"]] == [
        "fixture/ready",
        "fixture/adaptive",
        "fixture/unsupported",
    ]
    assert [item["compatibility"] for item in page["items"]] == [
        "likely",
        "advanced_import" if role == "image" else "unsupported",
        "unsupported",
    ]
    assert page["stale"] is True
    assert page["next_cursor"] == "next-page"
    assert enumerated == [role]

    # A later page request must see a newly available workflow.
    matching.add("fixture/adaptive")
    again = await client.get("/api/catalog", params={"role": role})
    assert again.status_code == 200
    assert again.json()["items"][1]["compatibility"] == "likely"
    assert enumerated == [role, role]


@pytest.mark.parametrize("role", ["image", "video"])
async def test_empty_catalog_does_not_enumerate_workflows(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    monkeypatch.setattr(settings, "media_engine", "comfyui")
    enumerated, _ = _catalog(monkeypatch, empty=True)
    response = await client.get("/api/catalog", params={"role": role})
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert enumerated == []


@pytest.mark.parametrize(
    ("role", "engine"),
    [("chat", "comfyui"), ("lora", "comfyui"), ("image", "mock"), ("video", "mock")],
)
async def test_other_catalog_paths_do_not_enumerate_workflows(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch, role: str, engine: str
) -> None:
    monkeypatch.setattr(settings, "media_engine", engine)
    enumerated, _ = _catalog(monkeypatch)
    response = await client.get("/api/catalog", params={"role": role})
    assert response.status_code == 200
    assert [item["compatibility"] for item in response.json()["items"]] == ["likely"] * 3
    assert enumerated == []
