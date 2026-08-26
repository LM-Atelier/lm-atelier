"""Edit workflows carry the LoRA stack.

LoRAs belong on image edits too. The orchestrator already resolves
the stack for every non-text operation, and the revision builder already
adds the `loras` schema wherever `detect_lora_extension` finds an insertion
point - so what needed proving is that a checkpoint-shaped image_to_image
template actually gets both. It does; these pin it.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import AsyncClient

from local_lm.auxiliary_assets import workflow_lora_extension
from local_lm.comfy_templates import ComfyTemplate, CompiledComfyTemplate
from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.downloads import DownloadManager
from local_lm.models import ModelInstall

pytestmark = pytest.mark.asyncio


async def wait_for_run(client: AsyncClient, run_id: str) -> dict:  # type: ignore[type-arg]
    for _ in range(400):
        payload = (await client.get(f"/api/runs/{run_id}")).json()
        if payload["status"] in {"complete", "failed", "cancelled"}:
            return payload
        await asyncio.sleep(0.01)
    raise AssertionError("run did not finish in time")


async def test_a_checkpoint_edit_template_gains_the_lora_stack(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    # The standard checkpoint edit shape: loader feeding model and clip into
    # the sampler and prompt encoders - exactly what photo-edit workflows use.
    graph = {
        "loader": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["loader", 1], "text": "edit instruction"},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["loader", 1], "text": ""},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["loader", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
            },
        },
    }
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="image_checkpoint_edit_test",
            path=settings.data_dir / "checkpoint-edit-template.json",
            role="image",
            operation="image_to_image",
            score=1_000,
            sha256="8" * 64,
            dependencies=(),
        ),
        ui_graph={"nodes": []},
        api_graph=graph,
        input_schema={"type": "object", "properties": {}},
    )
    with SessionLocal() as session:
        install = ModelInstall(
            name="Checkpoint editor",
            role="image",
            engine="comfyui",
            local_path=str(settings.model_dir / "checkpoint-editor"),
            manifest_json={"family": "checkpoint-edit-test"},
            active=True,
        )
        session.add(install)
        session.flush()

        revision = DownloadManager._ensure_template_workflow(session, compiled, install)

        # The graph extension point was detected on the checkpoint shape...
        assert workflow_lora_extension(revision) == {
            "model": ["loader", 0],
            "clip": ["loader", 1],
        }
        # ...and the edit workflow's settings now offer the stack, which is
        # what surfaces the LoRA section in Image Settings for edit turns.
        loras = revision.input_schema_json["properties"]["loras"]
        assert loras["type"] == "array"
        assert loras["maxItems"] == 8
        from local_lm.models import WorkflowDefinition

        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        assert definition.operation == "image_to_image"


async def test_an_edit_turn_resolves_the_stack_and_records_its_trigger_words(
    client: AsyncClient,
) -> None:
    """No test proved an edit turn resolves the stack.

    The schema landing on the revision proves an edit workflow can carry a
    stack. It does not prove a turn that edits a picture resolves one, and
    `trigger_words_applied` appeared in no test at all - so the provenance
    that tells someone which words their LoRA added was unproven for exactly
    the operation this file covers.
    """
    from local_lm.auxiliary_assets import checkpoint_lora_extension
    from local_lm.domain import utcnow
    from local_lm.models import (
        ModelAssetInstall,
        ModelProfile,
        WorkflowDefinition,
        WorkflowRevision,
    )

    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "mock.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "image": "${input_image}"}},
    }
    extension = checkpoint_lora_extension(graph)
    assert extension

    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Edit with LoRAs", operation="image_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            api_graph_json=graph,
            input_schema_json={
                "type": "object",
                "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
            },
            dependencies_json={"extensions": {"lora": extension}},
            trusted=True,
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        lora = ModelAssetInstall(
            name="Ink",
            kind="lora",
            family="sdxl",
            local_path="C:/managed/ink",
            size_bytes=1024,
            manifest_json={
                "sha256": "b" * 64,
                "comfy_name": "ink.safetensors",
                "metadata": {"trigger_words": ["ink wash"]},
            },
            active=True,
            verified_at=utcnow(),
        )
        session.add(lora)
        profile = session.query(ModelProfile).filter_by(role="image", is_default=True).one()
        install = ModelInstall(
            name="Trigger-aware editor",
            role="image",
            engine="mock",
            local_path="C:/managed/trigger-aware-editor",
            manifest_json={"trigger_words": ["portrait-style"]},
            active=True,
        )
        session.add(install)
        session.flush()
        profile.model_install_id = install.id
        session.commit()
        lora_id = lora.id

    source = (
        await client.post(
            "/api/artifacts",
            files={"file": ("edit-source.png", b"source-image", "image/png")},
        )
    ).json()
    chat = (await client.post("/api/chats", json={"title": "Edit with LoRAs"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "soften the background",
            "mode": "image",
            "input_artifact_ids": [source["id"]],
            "settings": {
                "loras": [
                    {
                        "asset_id": lora_id,
                        "model_strength": 0.8,
                        "clip_strength": 0.65,
                        "enabled": True,
                    }
                ]
            },
        },
    )

    assert response.status_code == 202, response.text
    assert response.json()["run"]["operation"] == "image_to_image"
    run = await wait_for_run(client, response.json()["run"]["id"])
    assert run["status"] == "complete"
    auxiliary = run["provenance_json"]["auxiliary_assets"]
    assert [item["asset_id"] for item in auxiliary["lora_stack"]] == [lora_id]
    # The instruction never says "ink wash", so the word the LoRA needs was
    # added on the user's behalf - and the record says which, which is the
    # whole point of the field. Asserting the empty case would have proved
    # nothing about the mechanism.
    assert auxiliary["model_trigger_words_applied"] == ["portrait-style"]
    assert auxiliary["lora_trigger_words_applied"] == ["ink wash"]
    assert auxiliary["trigger_words_applied"] == ["portrait-style", "ink wash"]


async def test_a_turn_runs_the_workflow_a_recipe_recorded(client: AsyncClient) -> None:
    """A recipe that stores a workflow and ignores it is not a recipe.

    Two edit workflows are installed. Selection would pick the newer one; the
    turn names the other, and the run records that it used the named one.
    """
    from local_lm.models import WorkflowDefinition, WorkflowRevision

    def _install(name: str) -> str:
        with SessionLocal() as session:
            definition = WorkflowDefinition(name=name, operation="image_to_image")
            session.add(definition)
            session.flush()
            revision = WorkflowRevision(
                workflow_id=definition.id,
                version=1,
                engine="mock",
                api_graph_json={"1": {"class_type": "KSampler", "inputs": {}}},
                input_schema_json={"type": "object", "properties": {}},
                dependencies_json={},
                trusted=True,
            )
            session.add(revision)
            session.flush()
            definition.current_revision_id = revision.id
            session.commit()
            return revision.id

    recorded = _install("Recorded editor")
    _install("Newer editor")

    source = (
        await client.post(
            "/api/artifacts",
            files={"file": ("recipe.png", b"source-image", "image/png")},
        )
    ).json()
    chat = (await client.post("/api/chats", json={"title": "Recipe run"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "make it a watercolor",
            "mode": "image",
            "input_artifact_ids": [source["id"]],
            "workflow_revision_id": recorded,
        },
    )

    assert response.status_code == 202, response.text
    run = await wait_for_run(client, response.json()["run"]["id"])
    assert run["provenance_json"]["workflow"]["revision_id"] == recorded


async def test_a_turn_naming_a_workflow_for_another_operation_refuses(
    client: AsyncClient,
) -> None:
    """Quietly substituting one would be worse than saying no."""
    from local_lm.models import WorkflowDefinition, WorkflowRevision

    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Video maker", operation="text_to_video")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            api_graph_json={"1": {"class_type": "KSampler", "inputs": {}}},
            input_schema_json={},
            dependencies_json={},
            trusted=True,
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        wrong = revision.id

    source = (
        await client.post(
            "/api/artifacts",
            files={"file": ("recipe.png", b"source-image", "image/png")},
        )
    ).json()
    chat = (await client.post("/api/chats", json={"title": "Wrong workflow"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "make it a watercolor",
            "mode": "image",
            "input_artifact_ids": [source["id"]],
            "workflow_revision_id": wrong,
        },
    )

    assert response.status_code == 422, response.text
    assert "workflow" in response.json()["detail"].casefold()
