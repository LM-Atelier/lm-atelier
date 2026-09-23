"""A hand-built workflow's model family, read from the checkpoint its graph loads.

A workflow built or imported by hand declares no model, so nothing said which
family it runs: a LoRA made for another family was accepted, and quietly
degraded the image, and no LoRA was ever chosen for it automatically. Its graph
still names the checkpoint it loads, and the family recorded for that model
when it was installed answers the question. These cases hold that answer, and
that anything short of one exact match stays unknown, which is what it was.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from local_lm.auxiliary_assets import (
    checkpoint_lora_extension,
    resolve_lora_stack,
    select_automatic_lora_stack,
    workflow_model_family,
)
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ModelAssetInstall, ModelInstall, WorkflowDefinition, WorkflowRevision

pytestmark = pytest.mark.asyncio

CHECKPOINT = "atelier-base.safetensors"


def _graph(*checkpoints: object) -> dict[str, Any]:
    graph: dict[str, Any] = {
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    for index, name in enumerate(checkpoints):
        node = "1" if index == 0 else f"1{index}"
        graph[node] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": name}}
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

    extension = checkpoint_lora_extension(_graph(CHECKPOINT))
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


def _model(
    session: Session, name: str, *, files: list[str], family: str | None = "SDXL"
) -> ModelInstall:
    manifest: dict[str, Any] = {"files": files}
    if family is not None:
        manifest["family"] = family
    install = ModelInstall(
        name=name,
        role="image",
        engine="comfyui",
        local_path=f"C:/managed/{name}",
        manifest_json=manifest,
        active=True,
    )
    session.add(install)
    session.flush()
    return install


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


async def test_a_lora_for_another_family_is_refused_by_the_checkpoints_family(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        _model(session, "Atelier Base", files=[f"weights/{CHECKPOINT}", "config.json"])
        revision = _hand_built(session, _graph(CHECKPOINT))
        foreign = _lora(session, "Foreign", "zimage-turbo", "a")
        matching = _lora(session, "Matching", "sdxl", "b")

        assert workflow_model_family(session, revision) == "sdxl"
        with pytest.raises(ValueError, match="targets zimage-turbo, which is incompatible"):
            resolve_lora_stack(session, revision, _stack(foreign))
        accepted = resolve_lora_stack(session, revision, _stack(matching))

    assert [item["asset_id"] for item in accepted.settings] == [matching.id]


async def test_a_turn_with_a_lora_for_another_family_is_refused_with_its_reason(
    client: AsyncClient,
) -> None:
    graph = _graph(CHECKPOINT)
    graph["3"]["inputs"]["image"] = "${input_image}"
    with SessionLocal() as session:
        _model(session, "Atelier Base", files=[CHECKPOINT])
        _hand_built(session, graph, engine="mock", operation="image_to_image")
        foreign = _lora(session, "Foreign", "zimage-turbo", "c")
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
    assert "Foreign targets zimage-turbo, which is incompatible" in response.json()["detail"]


@pytest.mark.parametrize(
    "case",
    [
        "no installed model holds the file",
        "two installed models hold the file",
        "the graph has two checkpoint loaders",
        "the model recorded no family",
        "the loader names no file",
    ],
)
async def test_anything_short_of_one_exact_match_leaves_the_family_unknown(
    client: AsyncClient, case: str
) -> None:
    del client
    graph = _graph(CHECKPOINT)
    with SessionLocal() as session:
        if case == "no installed model holds the file":
            _model(session, "Other", files=["other.safetensors"])
        elif case == "two installed models hold the file":
            _model(session, "First", files=[CHECKPOINT])
            _model(session, "Second", files=[f"copy/{CHECKPOINT}"], family="flux")
        elif case == "the graph has two checkpoint loaders":
            _model(session, "Atelier Base", files=[CHECKPOINT])
            graph = _graph(CHECKPOINT, CHECKPOINT)
        elif case == "the model recorded no family":
            _model(session, "Atelier Base", files=[CHECKPOINT], family=None)
        else:
            _model(session, "Atelier Base", files=[CHECKPOINT])
            graph = _graph(None)
        revision = _hand_built(session, graph)
        foreign = _lora(session, "Foreign", "zimage-turbo", "d")

        assert workflow_model_family(session, revision) is None
        # Unknown is what it was before: the guard does not fire.
        accepted = resolve_lora_stack(session, revision, _stack(foreign))

    assert [item["asset_id"] for item in accepted.settings] == [foreign.id]


async def test_a_declaration_that_is_present_and_malformed_is_not_read_past(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        _model(session, "Atelier Base", files=[CHECKPOINT])
        revision = _hand_built(
            session, _graph(CHECKPOINT), declared={"model_install_ids": "not a list"}
        )

        assert workflow_model_family(session, revision) is None


async def test_a_workflow_with_a_dependency_contract_answers_only_through_its_activation(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        _model(session, "Atelier Base", files=[CHECKPOINT])
        revision = _hand_built(session, _graph(CHECKPOINT))
        revision.dependency_contract_sha256 = "e" * 64
        foreign = _lora(session, "Foreign", "zimage-turbo", "e")
        session.flush()

        assert workflow_model_family(session, revision) is None
        accepted = resolve_lora_stack(session, revision, _stack(foreign))

    assert [item["asset_id"] for item in accepted.settings] == [foreign.id]


async def test_automatic_selection_reaches_a_hand_built_workflow_of_a_known_family(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        _model(session, "Atelier Base", files=[CHECKPOINT])
        revision = _hand_built(session, _graph(CHECKPOINT))
        matching = _lora(session, "Portrait", "sdxl", "f", automatic=True)
        _lora(session, "Elsewhere", "zimage-turbo", "0", automatic=True)

        selected = select_automatic_lora_stack(session, revision, "a detailed portrait")

    assert [item["asset_id"] for item in selected.settings] == [matching.id]
    assert selected.provenance.get("skipped_reason") is None
