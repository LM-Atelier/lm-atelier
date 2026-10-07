"""Derive useful LoRA intent without mixing architecture labels into matching."""

from copy import deepcopy

import pytest
from httpx2 import AsyncClient
from test_auxiliary_assets import _asset, _workflow

from local_lm.auxiliary_assets import select_automatic_lora_stack
from local_lm.db import SessionLocal


@pytest.mark.parametrize(
    "metadata,expected",
    [
        (
            {
                "tags": ["LoRA", "Watercolor landscapes"],
                "category": "LORA",
                "trained_words": ["watercolor landscapes"],
                "base_model": "SDXL 1.0",
            },
            "Watercolor landscapes",
        ),
        ({"category": "Ink drawing", "trained_words": ["ink wash"]}, "Ink drawing; ink wash"),
        ({"tags": ["safetensors", "text-to-image"], "base_model": "SDXL 1.0"}, ""),
        ({"description": "Do not use the description", "tags": [None, 7, {}]}, ""),
        (None, ""),
    ],
)
def test_lora_use_case_contains_descriptive_metadata_only(metadata: object, expected: str) -> None:
    from local_lm.lora_use_cases import derive_lora_use_case

    original = deepcopy(metadata)
    assert derive_lora_use_case(metadata) == expected
    assert metadata == original


def test_lora_use_case_is_bounded() -> None:
    from local_lm.lora_use_cases import derive_lora_use_case

    text = derive_lora_use_case({"tags": [f"word{number} " * 80 for number in range(200)]})
    assert 0 < len(text) <= 1000
    assert all(len(part) <= 200 for part in text.split("; "))


@pytest.mark.parametrize("gap", [None, "opt_in", "disabled", "unverified", "family", "identity"])
async def test_derived_intent_feeds_automatic_selection_without_widening_eligibility(
    client: AsyncClient, gap: str | None
) -> None:
    from local_lm.lora_use_cases import derive_lora_use_case

    del client
    with SessionLocal() as session:
        revision = _workflow(session)
        asset = _asset(session, "Watercolor", "a" * 64)
        asset.use_case = derive_lora_use_case(
            {
                "tags": ["lora", "Watercolor landscapes"],
                "category": "LoRA",
                "base_model": "Stable Diffusion XL 1.0",
            }
        )
        asset.use_case_derived = True
        asset.auto_apply = gap != "opt_in"
        if gap == "disabled":
            asset.active = False
        if gap == "unverified":
            asset.verified_at = None
        if gap == "family":
            asset.family = "flux"
        if gap == "identity":
            asset.manifest_json = {}
        session.flush()
        selected = select_automatic_lora_stack(session, revision, "Paint watercolor landscapes")
        if gap is None:
            assert [item["asset_id"] for item in selected.settings] == [asset.id]
            assert selected.provenance["selected"][0]["use_case"] == "Watercolor landscapes"
            assert selected.provenance["selected"][0]["use_case_derived"] is True
        else:
            assert selected.settings == []
