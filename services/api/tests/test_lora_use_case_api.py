"""Preserve derived LoRA descriptions until a person edits the description itself."""

from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ModelAssetInstall


@pytest.mark.parametrize(
    "values,expected,derived",
    [
        ({"family": "flux"}, "Watercolor landscapes", True),
        ({"default_model_strength": 0.5}, "Watercolor landscapes", True),
        ({"auto_apply": True}, "Watercolor landscapes", True),
        ({"use_case": "Watercolor landscapes"}, "Watercolor landscapes", False),
        ({"use_case": "  Ink drawings  "}, "Ink drawings", False),
        ({"use_case": ""}, "", False),
    ],
)
async def test_only_explicit_use_case_edits_clear_derived_provenance(
    client: AsyncClient, values: dict[str, Any], expected: str, derived: bool
) -> None:
    with SessionLocal() as session:
        asset = ModelAssetInstall(
            name="Watercolor",
            kind="lora",
            local_path="neutral",
            family="sdxl",
            use_case="Watercolor landscapes",
            verified_at=utcnow(),
        )
        asset.use_case_derived = True
        session.add(asset)
        session.commit()
        asset_id = asset.id
    listed = await client.get("/api/model-assets")
    assert listed.status_code == 200
    assert next(item for item in listed.json() if item["id"] == asset_id)["use_case_derived"]
    updated = await client.patch(f"/api/model-assets/{asset_id}", json=values)
    assert updated.status_code == 200, updated.text
    assert updated.json()["use_case"] == expected
    assert updated.json()["use_case_derived"] is derived
    with SessionLocal() as session:
        saved = session.get(ModelAssetInstall, asset_id)
        assert saved is not None
        assert saved.use_case == expected and saved.use_case_derived is derived


@pytest.mark.parametrize("description", ["Handwritten description", ""])
async def test_existing_manual_and_cleared_descriptions_are_not_derived(
    client: AsyncClient, description: str
) -> None:
    with SessionLocal() as session:
        asset = ModelAssetInstall(
            name="Manual",
            kind="lora",
            local_path="neutral",
            use_case=description,
            manifest_json={"use_case_metadata": {"tags": ["Watercolor landscapes"]}},
        )
        session.add(asset)
        session.commit()
        asset_id = asset.id
    updated = await client.patch(f"/api/model-assets/{asset_id}", json={"family": "sdxl"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["use_case"] == description
    assert updated.json()["use_case_derived"] is False
