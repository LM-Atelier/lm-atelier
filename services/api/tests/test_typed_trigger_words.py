"""Trigger words a person records for a LoRA, next to the ones its file declares.

Most LoRA files declare no trigger word in their own header, so a word that
exists only on the page the file came from had nowhere to go, and the stack
that adds declared words to a prompt had nothing to add. These cases hold the
route that records a word, the separation between what was typed and what was
measured, and, through the real turn endpoint, that a typed word reaches a run
exactly as a declared one does.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from run_waits import wait_for_terminal_status

from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ModelAssetInstall

pytestmark = pytest.mark.asyncio


def _lora(*, declared: list[str] | None = None, kind: str = "lora") -> str:
    metadata: dict[str, Any] = {"network_type": "networks.lora"}
    if declared is not None:
        metadata["trigger_words"] = declared
    with SessionLocal() as session:
        asset = ModelAssetInstall(
            name="Ink",
            kind=kind,
            family="sdxl",
            local_path="C:/managed/ink",
            size_bytes=1024,
            manifest_json={
                "sha256": "b" * 64,
                "comfy_name": "ink.safetensors",
                "metadata": metadata,
            },
            active=True,
            verified_at=utcnow(),
        )
        session.add(asset)
        session.commit()
        return asset.id


async def test_a_typed_word_is_stored_apart_from_what_the_file_declares(
    client: AsyncClient,
) -> None:
    asset_id = _lora(declared=["ink wash"])

    response = await client.patch(
        f"/api/model-assets/{asset_id}",
        json={"typed_trigger_words": [" studio glow ", "", "Studio Glow", "soft edge"]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    # Trimmed, blanks dropped, a repeat in another casing collapsed to the first.
    assert body["typed_trigger_words"] == ["studio glow", "soft edge"]
    # What the file declared is untouched and still reads as the file's.
    assert body["manifest_json"]["metadata"]["trigger_words"] == ["ink wash"]
    listed = (await client.get("/api/model-assets")).json()
    assert [item["typed_trigger_words"] for item in listed if item["id"] == asset_id] == [
        ["studio glow", "soft edge"]
    ]

    cleared = await client.patch(f"/api/model-assets/{asset_id}", json={"typed_trigger_words": []})
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["typed_trigger_words"] == []


async def test_only_a_lora_takes_a_trigger_word(client: AsyncClient) -> None:
    asset_id = _lora(kind="upscale")

    response = await client.patch(
        f"/api/model-assets/{asset_id}", json={"typed_trigger_words": ["studio glow"]}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "trigger-words-lora-only"


async def test_an_unbounded_word_is_refused_without_repeating_it(
    client: AsyncClient,
) -> None:
    asset_id = _lora()
    too_long = "private-" + "x" * 200

    response = await client.patch(
        f"/api/model-assets/{asset_id}", json={"typed_trigger_words": [too_long]}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "trigger-words-invalid"
    assert "private-" not in response.text


async def test_a_list_longer_than_a_lora_can_carry_is_refused_without_repeating_it(
    client: AsyncClient,
) -> None:
    asset_id = _lora()

    response = await client.patch(
        f"/api/model-assets/{asset_id}",
        json={"typed_trigger_words": [f"private-{index}" for index in range(101)]},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "request-validation-invalid"
    assert "private-" not in response.text
    listed = (await client.get("/api/model-assets")).json()
    assert [item["typed_trigger_words"] for item in listed if item["id"] == asset_id] == [[]]


async def test_the_bound_counts_the_words_the_file_declares(client: AsyncClient) -> None:
    asset_id = _lora(declared=[f"declared-{index}" for index in range(99)])

    fits = await client.patch(
        f"/api/model-assets/{asset_id}", json={"typed_trigger_words": ["one more"]}
    )
    too_many = await client.patch(
        f"/api/model-assets/{asset_id}",
        json={"typed_trigger_words": ["one more", "and another"]},
    )

    assert fits.status_code == 200, fits.text
    assert too_many.status_code == 422
    assert too_many.json()["code"] == "trigger-words-invalid"


async def test_a_typed_word_reaches_a_run_the_way_a_declared_one_does(
    client: AsyncClient,
) -> None:
    """A word recorded through the route is applied through a real turn."""

    from local_lm.auxiliary_assets import checkpoint_lora_extension
    from local_lm.models import WorkflowDefinition, WorkflowRevision

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
        session.commit()
    # The file declares nothing, which is the case this exists for.
    lora_id = _lora()
    recorded = await client.patch(
        f"/api/model-assets/{lora_id}", json={"typed_trigger_words": ["studio glow"]}
    )
    assert recorded.status_code == 200, recorded.text

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

    async def read() -> dict[str, Any]:
        return dict((await client.get(f"/api/runs/{response.json()['run']['id']}")).json())

    run = await wait_for_terminal_status(read, what="the edit run", expected=None)
    assert run["status"] == "complete"
    auxiliary = run["provenance_json"]["auxiliary_assets"]
    assert [item["asset_id"] for item in auxiliary["lora_stack"]] == [lora_id]
    # The instruction never says it, so the typed word was added on the
    # person's behalf, and the record says so, exactly as for a declared word.
    assert auxiliary["lora_trigger_words_applied"] == ["studio glow"]
