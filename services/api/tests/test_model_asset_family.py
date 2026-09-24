"""A model asset's family, set by hand after it was registered.

A file that declares no family in its own header was registered without one,
and nothing could give it one afterwards: the guard against a LoRA made for
another model had nothing to compare, and automatic selection passed it over.
The update route now takes a family for any kind of asset, trims it, clears it
when given an empty one, and refuses one with nothing in it to compare by.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from local_lm.auxiliary_assets import model_only_lora_extension, resolve_lora_stack
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ModelAssetInstall, ModelInstall, WorkflowDefinition, WorkflowRevision

pytestmark = pytest.mark.asyncio


def _asset(name: str, digest: str, *, kind: str = "lora", family: str | None = None) -> str:
    with SessionLocal() as session:
        asset = ModelAssetInstall(
            name=name,
            kind=kind,
            family=family,
            local_path=f"C:/managed/{name}",
            size_bytes=1024,
            manifest_json={"sha256": digest * 64, "comfy_name": f"{name}.safetensors"},
            active=True,
            verified_at=utcnow(),
        )
        session.add(asset)
        session.commit()
        return asset.id


def _sdxl_workflow() -> str:
    graph: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0]}},
    }
    with SessionLocal() as session:
        install = ModelInstall(
            name="Base",
            role="image",
            engine="comfyui",
            local_path="C:/managed/base",
            manifest_json={"family": "sdxl"},
            active=True,
        )
        definition = WorkflowDefinition(name="Declared", operation="text_to_image")
        session.add_all([install, definition])
        session.flush()
        extension = model_only_lora_extension(graph)
        assert extension
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="comfyui",
            api_graph_json=graph,
            input_schema_json={
                "type": "object",
                "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
            },
            dependencies_json={
                "model_install_ids": [install.id],
                "extensions": {"lora": extension},
            },
            trusted=True,
        )
        session.add(revision)
        session.commit()
        return revision.id


def _admitted(revision_id: str, asset_id: str) -> bool:
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        try:
            resolve_lora_stack(session, revision, [{"asset_id": asset_id, "model_strength": 1.0}])
        except ValueError:
            return False
        return True


def _family(asset_id: str) -> str | None:
    with SessionLocal() as session:
        asset = session.get(ModelAssetInstall, asset_id)
        assert asset is not None
        return asset.family


async def test_a_family_set_by_hand_is_what_the_guard_compares(client: AsyncClient) -> None:
    lora = _asset("Undeclared", "a")
    revision = _sdxl_workflow()
    # Nothing to compare yet: the guard lets it through, as it did before.
    assert _admitted(revision, lora)

    response = await client.patch(f"/api/model-assets/{lora}", json={"family": "  krea2 "})

    assert response.status_code == 200, response.text
    assert response.json()["family"] == "krea2"
    assert _family(lora) == "krea2"
    assert not _admitted(revision, lora)


async def test_an_empty_family_clears_it(client: AsyncClient) -> None:
    lora = _asset("Declared", "b", family="krea2")

    response = await client.patch(f"/api/model-assets/{lora}", json={"family": ""})

    assert response.status_code == 200, response.text
    assert response.json()["family"] is None
    assert _family(lora) is None


@pytest.mark.parametrize("value", ["   -   ", "--", "."])
async def test_a_family_with_nothing_to_compare_by_is_refused(
    client: AsyncClient, value: str
) -> None:
    lora = _asset("Declared", "c", family="krea2")

    response = await client.patch(f"/api/model-assets/{lora}", json={"family": value})

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "asset-family-invalid"
    assert _family(lora) == "krea2"


async def test_a_family_longer_than_a_family_is_refused(client: AsyncClient) -> None:
    lora = _asset("Declared", "d", family="krea2")

    response = await client.patch(f"/api/model-assets/{lora}", json={"family": "k" * 101})

    assert response.status_code == 422, response.text
    assert _family(lora) == "krea2"


async def test_any_kind_of_asset_can_be_given_a_family(client: AsyncClient) -> None:
    model = _asset("Atelier DiT", "e", kind="diffusion_model")

    response = await client.patch(f"/api/model-assets/{model}", json={"family": "krea2"})

    assert response.status_code == 200, response.text
    assert _family(model) == "krea2"


async def test_an_update_that_names_no_family_leaves_it_alone(client: AsyncClient) -> None:
    lora = _asset("Declared", "f", family="krea2")

    response = await client.patch(f"/api/model-assets/{lora}", json={"use_case": "portrait"})

    assert response.status_code == 200, response.text
    assert _family(lora) == "krea2"
