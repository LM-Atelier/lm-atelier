from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.auxiliary_assets import checkpoint_lora_extension
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ModelAssetInstall, Run, WorkflowDefinition, WorkflowRevision, WorkPlan


@pytest.mark.parametrize("target_mode", ["image", "auto"])
@pytest.mark.parametrize(
    "choice", ["manifest", "metadata", "partial", "workflow", "explicit", "clear", "unchanged"]
)
@pytest.mark.parametrize("source_is_edit", [False, True])
async def test_inherited_edit_never_rebinds_an_accepted_lora_to_changed_catalog_content(
    app: FastAPI, client: AsyncClient, choice: str, source_is_edit: bool, target_mode: str
) -> None:
    graph = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "neutral.safetensors"},
        },
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    extension = checkpoint_lora_extension(graph)
    assert extension is not None
    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Constructed ink workflow", operation="text_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            trusted=True,
            api_graph_json=graph,
            input_schema_json={
                "type": "object",
                "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
            },
            dependencies_json={"extensions": {"lora": extension}},
        )
        asset = ModelAssetInstall(
            name="Constructed ink",
            kind="lora",
            family="sdxl",
            local_path="C:/managed/constructed-ink",
            size_bytes=1024,
            manifest_json={
                "sha256": "b" * 64,
                "comfy_name": "constructed-ink.safetensors",
                "metadata": {"trigger_words": ["ink wash"]},
            },
            active=True,
            verified_at=utcnow(),
        )
        session.add_all([revision, asset])
        session.flush()
        definition.current_revision_id = revision.id
        revision_id, asset_id = revision.id, asset.id
        session.commit()
    stack = [{"asset_id": asset_id, "model_strength": 0.8, "clip_strength": 0.65, "enabled": True}]
    chat = (await client.post("/api/chats", json={"title": "Inherited auxiliary settings"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Draw a blue paper boat.",
                "mode": "image",
                "workflow_revision_id": revision_id,
                "settings": {"loras": stack},
            },
        )
        assert source.status_code == 202, source.text
        first = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/edits",
            json={"text": "Draw a smaller paper boat.", "idempotency_key": "first-auxiliary-edit"},
        )
        assert first.status_code == 202, first.text
        from local_lm.accepted_turn_context import accepted_context

        with SessionLocal() as session:
            original = session.get(Run, first.json()["run"]["id"])
            assert original is not None
            frozen = accepted_context(session, original)
            assert frozen is not None
            assert frozen.auxiliary_assets["lora_stack"][0]["sha256"] == "b" * 64
            asset = session.get(ModelAssetInstall, asset_id)
            assert asset is not None
            if choice == "metadata":
                asset.manifest_json = {
                    **asset.manifest_json,
                    "metadata": {"trigger_words": ["watercolor"]},
                }
            elif choice != "unchanged":
                asset.manifest_json = {**asset.manifest_json, "sha256": "c" * 64}
            session.commit()
        payload: dict[str, Any] = {
            "text": "Create an image of a larger paper boat.",
            "mode": target_mode,
            "confirm_media": True,
            "idempotency_key": "second-auxiliary-edit",
        }
        if choice == "explicit":
            payload["settings"] = {"loras": stack}
        elif choice == "clear":
            payload["settings"] = {"loras": []}
        elif choice == "partial":
            payload["settings"] = {"steps": 12}
        elif choice == "workflow":
            payload["workflow_revision_id"] = revision_id
        target = first if source_is_edit else source
        response = await client.post(
            f"/api/messages/{target.json()['user_message']['id']}/edits", json=payload
        )
        refused = choice in {"manifest", "metadata", "partial", "workflow"}
        assert response.status_code == (409 if refused else 202), response.text
        with SessionLocal() as session:
            plans = list(session.scalars(select(WorkPlan).where(WorkPlan.chat_id == chat["id"])))
            runs = list(session.scalars(select(Run).where(Run.chat_id == chat["id"])))
            assert len(plans) == len(runs) == (2 if refused else 3)
            original = session.get(Run, first.json()["run"]["id"])
            assert original is not None and accepted_context(session, original) == frozen
            if refused:
                assert response.json()["code"] == "edit-request-conflict"
                return
            edited = session.get(Run, response.json()["run"]["id"])
            assert edited is not None
            current = accepted_context(session, edited)
            assert current is not None
            if choice == "clear":
                assert current.settings["loras"] == []
                assert not current.auxiliary_assets.get("lora_stack")
            else:
                expected = "c" * 64 if choice == "explicit" else "b" * 64
                assert current.auxiliary_assets["lora_stack"][0]["sha256"] == expected
