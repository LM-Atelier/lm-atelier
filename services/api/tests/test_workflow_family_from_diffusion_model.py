"""A workflow's model family, read from the diffusion model its graph loads.

A workflow built around a diffusion-model loader, rather than a checkpoint,
declared no model and named no checkpoint, so nothing said which family it
runs: no LoRA was ever chosen for it automatically, and a LoRA made for another
family was accepted and quietly degraded the image. Its graph still names the
model file, and the registered diffusion model answering to that name records
the family. These cases hold that answer, that anything short of one exact
match stays unknown, and that one family spelled two ways is one family.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from local_lm.auxiliary_assets import (
    model_only_lora_extension,
    resolve_lora_stack,
    select_automatic_lora_stack,
    workflow_model_family,
)
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ModelAssetInstall, ModelInstall, WorkflowDefinition, WorkflowRevision

pytestmark = pytest.mark.asyncio

MODEL = "atelier-dit.safetensors"


def _graph(*models: object, checkpoint: bool = False) -> dict[str, Any]:
    graph: dict[str, Any] = {
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "text.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0]}},
    }
    for index, name in enumerate(models):
        node = "1" if index == 0 else f"1{index}"
        graph[node] = {"class_type": "UNETLoader", "inputs": {"unet_name": name}}
    if checkpoint:
        graph["9"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": MODEL}}
    return graph


def _hand_built(
    session: Session,
    graph: dict[str, Any],
    *,
    declared: dict[str, Any] | None = None,
    engine: str = "comfyui",
    operation: str = "text_to_image",
) -> WorkflowRevision:
    """A workflow as a person builds or imports it: where LoRAs go, and no model."""

    extension = model_only_lora_extension(graph)
    assert extension
    definition = WorkflowDefinition(name="Hand built", operation=operation)
    session.add(definition)
    session.flush()
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine=engine,
        api_graph_json=graph,
        input_schema_json={
            "type": "object",
            "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
        },
        dependencies_json={**(declared or {}), "extensions": {"lora": extension}},
        trusted=True,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    return revision


def _diffusion_model(
    session: Session,
    name: str,
    digest: str,
    *,
    comfy_name: str = MODEL,
    family: str | None = "krea2",
    kind: str = "diffusion_model",
    active: bool = True,
    verified: bool = True,
) -> ModelAssetInstall:
    asset = ModelAssetInstall(
        name=name,
        kind=kind,
        family=family,
        local_path=f"C:/managed/{name}",
        size_bytes=4096,
        manifest_json={"adopted": True, "sha256": digest * 64, "comfy_name": comfy_name},
        active=active,
        verified_at=utcnow() if verified else None,
    )
    session.add(asset)
    session.flush()
    return asset


def _lora(
    session: Session, name: str, family: str, digest: str, *, automatic: bool = False
) -> ModelAssetInstall:
    asset = ModelAssetInstall(
        name=name,
        kind="lora",
        family=family,
        local_path=f"C:/managed/{name}",
        size_bytes=1024,
        manifest_json={
            "sha256": digest * 64,
            "comfy_name": f"{name}.safetensors",
            "metadata": {"network_type": "networks.lora"},
        },
        active=True,
        auto_apply=automatic,
        use_case="portrait" if automatic else "",
        verified_at=utcnow(),
    )
    session.add(asset)
    session.flush()
    return asset


def _stack(asset: ModelAssetInstall) -> list[object]:
    return [{"asset_id": asset.id, "model_strength": 1.0, "clip_strength": 1.0}]


async def test_a_lora_for_another_family_is_refused_by_the_diffusion_models_family(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        _diffusion_model(session, "Atelier DiT", "1")
        revision = _hand_built(session, _graph(MODEL))
        foreign = _lora(session, "Foreign", "sdxl", "a")
        # The same family as the model's, spelled the other way files spell it.
        matching = _lora(session, "Matching", "Krea-2", "b")

        assert workflow_model_family(session, revision) == "krea2"
        with pytest.raises(ValueError, match="targets sdxl, which is incompatible"):
            resolve_lora_stack(session, revision, _stack(foreign))
        accepted = resolve_lora_stack(session, revision, _stack(matching))

    assert [item["asset_id"] for item in accepted.settings] == [matching.id]


async def test_a_turn_with_a_lora_for_another_family_is_refused_with_its_reason(
    client: AsyncClient,
) -> None:
    graph = _graph(MODEL)
    graph["3"]["inputs"]["image"] = "${input_image}"
    with SessionLocal() as session:
        _diffusion_model(session, "Atelier DiT", "2")
        _hand_built(session, graph, engine="mock", operation="image_to_image")
        foreign = _lora(session, "Foreign", "sdxl", "c")
        session.commit()
        foreign_id = foreign.id
    source = (
        await client.post(
            "/api/artifacts",
            files={"file": ("edit-source.png", b"source-image", "image/png")},
        )
    ).json()
    chat = (await client.post("/api/chats", json={"title": "Family"})).json()

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "soften the background",
            "mode": "image",
            "input_artifact_ids": [source["id"]],
            "settings": {"loras": [{"asset_id": foreign_id, "model_strength": 1.0}]},
        },
    )

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "turn-invalid"
    assert "Foreign targets sdxl, which is incompatible" in response.json()["detail"]


async def test_automatic_selection_reaches_a_workflow_built_on_a_diffusion_model(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        _diffusion_model(session, "Atelier DiT", "3")
        revision = _hand_built(session, _graph(MODEL))
        matching = _lora(session, "Portrait", "KREA-2", "d", automatic=True)
        _lora(session, "Elsewhere", "sdxl", "e", automatic=True)

        selected = select_automatic_lora_stack(session, revision, "a detailed portrait")

    assert [item["asset_id"] for item in selected.settings] == [matching.id]
    assert selected.provenance.get("skipped_reason") is None


@pytest.mark.parametrize(
    "case",
    [
        "no registered diffusion model answers to the name",
        "two registered diffusion models answer to the name",
        "only a LoRA answers to the name",
        "the graph has two model loaders",
        "the graph also has a checkpoint loader",
        "the diffusion model recorded no family",
        "the diffusion model is turned off",
        "the diffusion model was never verified",
        "the loader names no file",
    ],
)
async def test_anything_short_of_one_exact_match_leaves_the_family_unknown(
    client: AsyncClient, case: str
) -> None:
    del client
    graph = _graph(MODEL)
    with SessionLocal() as session:
        if case == "no registered diffusion model answers to the name":
            _diffusion_model(session, "Other", "4", comfy_name="other.safetensors")
        elif case == "two registered diffusion models answer to the name":
            _diffusion_model(session, "First", "5")
            _diffusion_model(session, "Second", "6", family="sdxl")
        elif case == "only a LoRA answers to the name":
            _diffusion_model(session, "Adapter", "7", kind="lora")
        elif case == "the graph has two model loaders":
            _diffusion_model(session, "Atelier DiT", "8")
            graph = _graph(MODEL, MODEL)
        elif case == "the graph also has a checkpoint loader":
            _diffusion_model(session, "Atelier DiT", "9")
            graph = _graph(MODEL, checkpoint=True)
        elif case == "the diffusion model recorded no family":
            _diffusion_model(session, "Atelier DiT", "a", family=None)
        elif case == "the diffusion model is turned off":
            _diffusion_model(session, "Atelier DiT", "b", active=False)
        elif case == "the diffusion model was never verified":
            _diffusion_model(session, "Atelier DiT", "c", verified=False)
        else:
            _diffusion_model(session, "Atelier DiT", "d")
            graph = _graph(None)
        revision = _hand_built(session, graph)
        foreign = _lora(session, "Foreign", "sdxl", "f")

        assert workflow_model_family(session, revision) is None
        # Unknown is what it was before: the guard does not fire.
        accepted = resolve_lora_stack(session, revision, _stack(foreign))

    assert [item["asset_id"] for item in accepted.settings] == [foreign.id]


async def test_installs_that_spell_one_family_two_ways_agree_on_it(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        installs = []
        for name, family in (("Base", "Krea-2"), ("Refiner", "krea2")):
            install = ModelInstall(
                name=name,
                role="image",
                engine="comfyui",
                local_path=f"C:/managed/{name}",
                manifest_json={"family": family},
                active=True,
            )
            session.add(install)
            installs.append(install)
        session.flush()
        revision = _hand_built(
            session,
            _graph(MODEL),
            declared={"model_install_ids": [install.id for install in installs]},
        )
        foreign = _lora(session, "Foreign", "sdxl", "0")
        matching = _lora(session, "Matching", "krea2", "1")

        assert workflow_model_family(session, revision) in {"krea-2", "krea2"}
        with pytest.raises(ValueError, match="targets sdxl, which is incompatible"):
            resolve_lora_stack(session, revision, _stack(foreign))
        accepted = resolve_lora_stack(session, revision, _stack(matching))

    assert [item["asset_id"] for item in accepted.settings] == [matching.id]


async def test_a_checkpoint_workflow_accepts_its_family_spelled_another_way(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                name="Atelier Base",
                role="image",
                engine="comfyui",
                local_path="C:/managed/atelier-base",
                manifest_json={"files": ["base.safetensors"], "family": "z-image-turbo"},
                active=True,
            )
        )
        graph: dict[str, Any] = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "base.safetensors"},
            },
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
            "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0]}},
        }
        revision = _hand_built(session, graph)
        spelled = _lora(session, "Spelled", "zimage-turbo", "2")

        accepted = resolve_lora_stack(session, revision, _stack(spelled))

    assert [item["asset_id"] for item in accepted.settings] == [spelled.id]
