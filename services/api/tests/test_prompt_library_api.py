from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import (
    Job,
    Message,
    ModelAssetInstall,
    PromptExpansionBatch,
    PromptExpansionItem,
    PromptTemplateDefinition,
    PromptTemplateRevision,
    Run,
    WorkPlan,
)
from local_lm.prompt_library import PromptLibraryError, create_prompt_template


def _contract(
    *,
    body_prefix: str = "A",
    resource_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "text_to_image",
        "body": f"{body_prefix} {{{{subject}}}}.",
        "slots": [
            {
                "name": "subject",
                "mode": "input",
                "variation_scope": "item",
            }
        ],
        "resource_policy": resource_policy or {"mode": "inherited"},
    }


def _create_payload(
    *,
    key: str = "prompt-create-1",
    name: str = "Portrait variants",
    contract: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "idempotency_key": key,
        "name": name,
        "description": "One controlled subject slot",
        "contract": contract or _contract(),
    }


async def _wait_for_run(client: AsyncClient, run_id: str) -> dict[str, Any]:
    for _ in range(300):
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            return run
        await asyncio.sleep(0.01)
    raise AssertionError("run did not finish")


def _composer_source(batch: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "batch_id": batch["id"],
        "expected_plan_version": batch["plan_version"],
        "expected_plan_sha256": batch["plan_sha256"],
        "item_id": item["id"],
        "expected_review_version": item["review_version"],
        "expected_reviewed_sha256": item["reviewed_sha256"],
        "prompt_template_id": batch["prompt_template_id"],
        "prompt_template_revision_id": batch["prompt_template_revision_id"],
        "contract_sha256": batch["contract_sha256"],
    }


@pytest.mark.asyncio
async def test_reviewed_prompt_source_is_claimed_once_and_inherited_without_raw_receipts(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Template composer"})).json()
    template = (await client.post("/api/prompt-templates", json=_create_payload())).json()
    revision = template["revision"]
    batch = (
        await client.post(
            f"/api/chats/{chat['id']}/prompt-batches",
            json={
                "idempotency_key": "composer-preview-1",
                "template_revision_id": revision["id"],
                "contract_sha256": revision["contract_sha256"],
                "item_count": 1,
                "selection_seed": 7,
                "inputs": {"subject": ["private castle subject"]},
            },
        )
    ).json()
    item = batch["items"][0]
    source = _composer_source(batch, item)
    submitted = f"{item['reviewed_prompt']} with a hand-painted sky"
    request_payload = {
        "text": submitted,
        "mode": "image",
        "output_count": 1,
        "idempotency_key": "composer-send-1",
        "prompt_source": source,
    }

    accepted_response = await client.post(f"/api/chats/{chat['id']}/turns", json=request_payload)
    assert accepted_response.status_code == 202, accepted_response.text
    accepted = accepted_response.json()
    witness = accepted["run"]["provenance_json"]["prompt_source"]
    message_witness = accepted["user_message"]["parts"][0]["metadata_json"]["prompt_source"]
    assert message_witness == witness
    assert set(witness) == {
        "version",
        "kind",
        "source_chat_id",
        "batch_id",
        "review_plan_version",
        "queued_plan_version",
        "plan_sha256",
        "item_id",
        "item_ordinal",
        "item_review_version",
        "reviewed_sha256",
        "submitted_sha256",
        "relation",
        "prompt_template_id",
        "prompt_template_revision_id",
        "contract_sha256",
        "resource_policy",
    }
    assert witness["relation"] == "edited"
    assert witness["review_plan_version"] == 1
    assert witness["queued_plan_version"] == 2
    assert witness["resource_policy"] == {"mode": "inherited"}
    serialized_witness = json.dumps(witness)
    assert item["reviewed_prompt"] not in serialized_witness
    assert "private castle subject" not in serialized_witness
    assert "template_body" not in serialized_witness
    assert "evidence" not in serialized_witness
    assert "inputs" not in serialized_witness

    replay = await client.post(f"/api/chats/{chat['id']}/turns", json=request_payload)
    assert replay.status_code == 202
    assert replay.json()["run"]["id"] == accepted["run"]["id"]

    stale = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={**request_payload, "idempotency_key": "composer-send-stale"},
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "prompt-source-conflict"

    reread = (await client.get(f"/api/prompt-batches/{batch['id']}")).json()
    assert reread["state"] == "queued"
    assert reread["plan_version"] == 2

    await _wait_for_run(client, accepted["run"]["id"])
    regenerated = await client.post(
        f"/api/messages/{accepted['assistant_message']['id']}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202, regenerated.text
    regenerated_witness = regenerated.json()["run"]["provenance_json"]["prompt_source"]
    assert regenerated_witness == witness

    branched = await client.post(
        f"/api/messages/{accepted['user_message']['id']}/branch",
        json={"text": "A materially different castle prompt", "mode": "image"},
    )
    assert branched.status_code == 202, branched.text
    branched_witness = branched.json()["run"]["provenance_json"]["prompt_source"]
    assert branched_witness["batch_id"] == batch["id"]
    assert branched_witness["queued_plan_version"] == 2
    assert branched_witness["reviewed_sha256"] == witness["reviewed_sha256"]
    assert branched_witness["submitted_sha256"] != witness["submitted_sha256"]
    assert branched_witness["relation"] == "edited"
    assert (
        branched.json()["user_message"]["parts"][0]["metadata_json"]["prompt_source"]
        == branched_witness
    )
    final_batch = (await client.get(f"/api/prompt-batches/{batch['id']}")).json()
    assert final_batch["state"] == "queued"
    assert final_batch["plan_version"] == 2


@pytest.mark.asyncio
async def test_prompt_source_rejects_wrong_turn_shape_without_claiming_batch(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Template refusal"})).json()
    template = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(key="prompt-create-refusal"),
        )
    ).json()
    revision = template["revision"]
    batch = (
        await client.post(
            f"/api/chats/{chat['id']}/prompt-batches",
            json={
                "idempotency_key": "composer-preview-refusal",
                "template_revision_id": revision["id"],
                "contract_sha256": revision["contract_sha256"],
                "item_count": 1,
                "selection_seed": 8,
                "inputs": {"subject": ["one subject"]},
            },
        )
    ).json()
    item = batch["items"][0]
    source = _composer_source(batch, item)

    wrong_mode = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": item["reviewed_prompt"],
            "mode": "text",
            "prompt_source": source,
        },
    )
    assert wrong_mode.status_code == 422
    assert wrong_mode.json()["code"] == "prompt-source-invalid"
    multiple = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": item["reviewed_prompt"],
            "mode": "image",
            "output_count": 2,
            "prompt_source": source,
        },
    )
    assert multiple.status_code == 422
    assert multiple.json()["code"] == "prompt-source-invalid"

    reread = (await client.get(f"/api/prompt-batches/{batch['id']}")).json()
    assert reread["state"] == "draft"
    assert reread["plan_version"] == 1
    with SessionLocal() as session:
        assert session.query(Run).count() == 0
        assert session.query(Job).count() == 0


@pytest.mark.asyncio
async def test_prompt_batch_preview_replay_and_review_create_no_media_work(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Template previews"})).json()
    created_template = (await client.post("/api/prompt-templates", json=_create_payload())).json()
    revision = created_template["revision"]
    request_payload = {
        "idempotency_key": "prompt-preview-1",
        "template_revision_id": revision["id"],
        "contract_sha256": revision["contract_sha256"],
        "item_count": 2,
        "selection_seed": 42,
        "inputs": {"subject": ["a fox", "an owl"]},
    }

    created_response = await client.post(
        f"/api/chats/{chat['id']}/prompt-batches", json=request_payload
    )
    assert created_response.status_code == 201
    created = created_response.json()
    assert created["replayed"] is False
    assert created["requested_count"] == 2
    assert created["selection_seed"] == 42
    assert created["plan_version"] == 1
    assert [item["rendered_prompt"] for item in created["items"]] == [
        "A a fox.",
        "A an owl.",
    ]

    replay = await client.post(f"/api/chats/{chat['id']}/prompt-batches", json=request_payload)
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["id"] == created["id"]

    first = created["items"][0]
    changed_response = await client.patch(
        f"/api/prompt-batches/{created['id']}/items/1",
        json={
            "expected_review_version": first["review_version"],
            "expected_plan_version": created["plan_version"],
            "reviewed_prompt": "A carefully reviewed fox.",
            "selected": False,
        },
    )
    assert changed_response.status_code == 200
    changed = changed_response.json()
    assert changed["plan_version"] == 2
    assert changed["items"][0]["review_version"] == 2
    assert changed["items"][0]["reviewed_prompt"] == "A carefully reviewed fox."
    assert changed["items"][0]["selected"] is False
    assert changed["items"][1] == created["items"][1]
    assert changed["plan_sha256"] == created["plan_sha256"]
    assert changed["items"][0]["reviewed_sha256"] != first["reviewed_sha256"]

    edited_replay = await client.post(
        f"/api/chats/{chat['id']}/prompt-batches", json=request_payload
    )
    assert edited_replay.status_code == 201
    assert edited_replay.json() == {**changed, "replayed": True}

    stale = await client.patch(
        f"/api/prompt-batches/{created['id']}/items/1",
        json={
            "expected_review_version": 1,
            "expected_plan_version": 1,
            "reviewed_prompt": "C:/private/should-not-echo.txt",
            "selected": True,
        },
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "prompt-batch-stale"
    assert "private" not in stale.text

    reread = await client.get(f"/api/prompt-batches/{created['id']}")
    assert reread.status_code == 200
    assert reread.json() == {**changed, "replayed": True}

    with SessionLocal() as session:
        assert session.query(PromptExpansionBatch).count() == 1
        assert session.query(PromptExpansionItem).count() == 2
        assert session.query(Message).count() == 0
        assert session.query(WorkPlan).count() == 0
        assert session.query(Run).count() == 0
        assert session.query(Job).count() == 0


@pytest.mark.asyncio
async def test_prompt_batch_request_is_strict_and_model_slots_fail_without_rows(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Strict previews"})).json()
    created_template = (await client.post("/api/prompt-templates", json=_create_payload())).json()
    revision = created_template["revision"]
    invalid = await client.post(
        f"/api/chats/{chat['id']}/prompt-batches",
        json={
            "idempotency_key": "prompt-preview-invalid",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": True,
            "selection_seed": 0,
            "inputs": {"subject": ["one"]},
        },
    )
    assert invalid.status_code == 422

    model_contract = _contract()
    model_contract["slots"] = [
        {
            "name": "subject",
            "mode": "model",
            "variation_scope": "item",
            "guidance": "one concrete subject",
        }
    ]
    model_template = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(
                key="prompt-model-template",
                name="Model preview template",
                contract=model_contract,
            ),
        )
    ).json()
    model_revision = model_template["revision"]
    unavailable = await client.post(
        f"/api/chats/{chat['id']}/prompt-batches",
        json={
            "idempotency_key": "prompt-model-preview",
            "template_revision_id": model_revision["id"],
            "contract_sha256": model_revision["contract_sha256"],
            "item_count": 1,
            "selection_seed": 3,
            "inputs": {},
        },
    )
    assert unavailable.status_code == 409
    assert unavailable.json()["code"] == "prompt-model-slots-unavailable"
    with SessionLocal() as session:
        assert session.query(PromptExpansionBatch).count() == 0


@pytest.mark.asyncio
async def test_prompt_template_crud_history_restore_and_archive_are_idempotent(
    client: AsyncClient,
) -> None:
    created_response = await client.post(
        "/api/prompt-templates",
        json=_create_payload(),
    )
    assert created_response.status_code == 201
    created = created_response.json()
    assert created["idempotent"] is False
    template = created["template"]
    revision_one = created["revision"]
    assert template["current_revision_id"] == revision_one["id"]
    assert revision_one["version"] == 1
    assert revision_one["contract_json"] == _contract()

    replay = await client.post("/api/prompt-templates", json=_create_payload())
    assert replay.status_code == 201
    assert replay.json()["idempotent"] is True
    assert replay.json()["template"]["id"] == template["id"]
    assert replay.json()["revision"]["id"] == revision_one["id"]

    detail = await client.get(f"/api/prompt-templates/{template['id']}")
    assert detail.status_code == 200
    assert detail.json()["current_revision"] == revision_one

    changed_contract = _contract(body_prefix="Detailed")
    update_payload = {
        "expected_current_revision_id": revision_one["id"],
        "idempotency_key": "prompt-edit-1",
        "name": "Detailed portrait variants",
        "contract": changed_contract,
    }
    updated_response = await client.patch(
        f"/api/prompt-templates/{template['id']}",
        json=update_payload,
    )
    assert updated_response.status_code == 200
    updated = updated_response.json()
    assert updated["idempotent"] is False
    revision_two = updated["revision"]
    assert revision_two["version"] == 2
    assert updated["template"]["name"] == "Detailed portrait variants"
    assert updated["template"]["current_revision_id"] == revision_two["id"]

    update_replay = await client.patch(
        f"/api/prompt-templates/{template['id']}",
        json=update_payload,
    )
    assert update_replay.status_code == 200
    assert update_replay.json()["idempotent"] is True
    assert update_replay.json()["revision"]["id"] == revision_two["id"]

    stale = await client.patch(
        f"/api/prompt-templates/{template['id']}",
        json={
            "expected_current_revision_id": revision_one["id"],
            "idempotency_key": "prompt-edit-stale",
            "contract": _contract(body_prefix="Stale private body"),
        },
    )
    assert stale.status_code == 409
    assert stale.json() == {
        "detail": "Prompt template changed. Refresh and try again.",
        "code": "prompt-template-stale",
    }
    assert "private body" not in stale.text

    history = await client.get(f"/api/prompt-templates/{template['id']}/revisions")
    assert history.status_code == 200
    assert [item["version"] for item in history.json()] == [2, 1]

    restored_response = await client.post(
        f"/api/prompt-templates/{template['id']}/revisions/{revision_one['id']}/restore",
        json={
            "expected_current_revision_id": revision_two["id"],
            "idempotency_key": "prompt-restore-1",
        },
    )
    assert restored_response.status_code == 200
    restored = restored_response.json()
    revision_three = restored["revision"]
    assert revision_three["version"] == 3
    assert revision_three["contract_sha256"] == revision_one["contract_sha256"]
    assert revision_three["id"] != revision_one["id"]

    archived_response = await client.patch(
        f"/api/prompt-templates/{template['id']}",
        json={
            "expected_current_revision_id": revision_three["id"],
            "archived": True,
        },
    )
    assert archived_response.status_code == 200
    assert archived_response.json()["template"]["archived"] is True
    assert archived_response.json()["revision"]["id"] == revision_three["id"]

    active_page = await client.get("/api/prompt-templates")
    assert active_page.status_code == 200
    assert active_page.json()["items"] == []
    archived_page = await client.get(
        "/api/prompt-templates",
        params={"include_archived": True},
    )
    assert archived_page.status_code == 200
    assert [item["id"] for item in archived_page.json()["items"]] == [template["id"]]


@pytest.mark.asyncio
async def test_prompt_template_create_and_patch_conflicts_are_fixed_and_non_echoing(
    client: AsyncClient,
) -> None:
    created = (await client.post("/api/prompt-templates", json=_create_payload())).json()
    template_id = created["template"]["id"]
    revision_id = created["revision"]["id"]

    key_reuse = await client.post(
        "/api/prompt-templates",
        json=_create_payload(name="Private collision name"),
    )
    assert key_reuse.status_code == 409
    assert key_reuse.json() == {
        "detail": "Prompt template conflicts with an existing template.",
        "code": "prompt-template-conflict",
    }
    assert "Private collision" not in key_reuse.text

    name_collision = await client.post(
        "/api/prompt-templates",
        json=_create_payload(key="prompt-create-other"),
    )
    assert name_collision.status_code == 409
    assert name_collision.json()["code"] == "prompt-template-conflict"

    for invalid_patch in (
        {"expected_current_revision_id": revision_id},
        {
            "expected_current_revision_id": revision_id,
            "name": None,
        },
        {
            "expected_current_revision_id": revision_id,
            "idempotency_key": "unused-edit-key",
            "description": "No contract",
        },
        {
            "expected_current_revision_id": revision_id,
            "contract": _contract(body_prefix="Missing key"),
        },
    ):
        response = await client.patch(
            f"/api/prompt-templates/{template_id}",
            json=invalid_patch,
        )
        assert response.status_code == 422
        assert response.json()["code"] == "prompt-template-request-invalid"

    invalid_contract = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="invalid-contract",
            name="Invalid private template",
            contract={**_contract(), "private_unknown": "C:/secret/model.safetensors"},
        ),
    )
    assert invalid_contract.status_code == 422
    assert invalid_contract.json() == {
        "detail": "Prompt template contract is invalid.",
        "code": "prompt-template-invalid",
    }
    assert "secret" not in invalid_contract.text


@pytest.mark.asyncio
async def test_archived_prompt_template_cannot_replay_as_a_live_idempotent_create(
    client: AsyncClient,
) -> None:
    payload = _create_payload(key="prompt-archived-replay")
    created = (await client.post("/api/prompt-templates", json=payload)).json()
    template = created["template"]
    archived = await client.patch(
        f"/api/prompt-templates/{template['id']}",
        json={
            "expected_current_revision_id": template["current_revision_id"],
            "archived": True,
        },
    )
    assert archived.status_code == 200

    replay = await client.post("/api/prompt-templates", json=payload)
    assert replay.status_code == 409
    assert replay.json()["code"] == "prompt-template-conflict"


def test_prompt_template_service_refuses_unbounded_or_unsafe_idempotency_keys() -> None:
    with SessionLocal() as session:
        for key in ("contains spaces", "x" * 201, "private/path"):
            with pytest.raises(PromptLibraryError) as raised:
                create_prompt_template(
                    session,
                    idempotency_key=key,
                    name="Guarded template",
                    description="",
                    contract_value=_contract(),
                    expected_engine="comfyui",
                )
            assert raised.value.code == "prompt-template-request-invalid"


@pytest.mark.asyncio
async def test_corrupt_stored_prompt_template_contract_never_crosses_the_api(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        definition = PromptTemplateDefinition(
            id="ptdef_corrupt",
            name="Corrupt stored template",
            description="",
            archived=False,
        )
        session.add(definition)
        session.flush()
        revision = PromptTemplateRevision(
            id="ptrev_corrupt",
            prompt_template_id=definition.id,
            version=1,
            schema_version=1,
            contract_json={"schema_version": 1},
            contract_sha256="d" * 64,
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()

    for path in (
        "/api/prompt-templates/ptdef_corrupt",
        "/api/prompt-templates/ptdef_corrupt/revisions",
        "/api/prompt-templates/ptdef_corrupt/revisions/ptrev_corrupt",
    ):
        response = await client.get(path)
        assert response.status_code == 409
        assert response.json() == {
            "detail": "Prompt template conflicts with an existing template.",
            "code": "prompt-template-conflict",
        }


@pytest.mark.asyncio
async def test_parseable_stored_prompt_template_sha_drift_never_crosses_the_api(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        definition = PromptTemplateDefinition(
            id="ptdef_sha_drift",
            name="Drifted stored template",
            description="",
            archived=False,
        )
        session.add(definition)
        session.flush()
        revision = PromptTemplateRevision(
            id="ptrev_sha_drift",
            prompt_template_id=definition.id,
            version=1,
            schema_version=1,
            contract_json=_contract(),
            contract_sha256="d" * 64,
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()

    for path in (
        "/api/prompt-templates/ptdef_sha_drift",
        "/api/prompt-templates/ptdef_sha_drift/revisions",
        "/api/prompt-templates/ptdef_sha_drift/revisions/ptrev_sha_drift",
    ):
        response = await client.get(path)
        assert response.status_code == 409
        assert response.json() == {
            "detail": "Prompt template conflicts with an existing template.",
            "code": "prompt-template-conflict",
        }
        assert "d" * 64 not in response.text


@pytest.mark.asyncio
async def test_prompt_template_fixed_resources_require_ready_workflow_and_verified_loras(
    client: AsyncClient,
) -> None:
    workflow_response = await client.post(
        "/api/workflows",
        json={
            "name": "Prompt template image workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
            "trusted": True,
        },
    )
    assert workflow_response.status_code == 201
    workflow_revision_id = workflow_response.json()["current_revision_id"]
    fixed = {
        "mode": "fixed",
        "workflow_revision_id": workflow_revision_id,
        "lora_policy": {"mode": "none"},
    }
    ready = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="fixed-ready",
            name="Fixed ready resources",
            contract=_contract(resource_policy=fixed),
        ),
    )
    assert ready.status_code == 201
    ready_revision = ready.json()["revision"]
    chat = (await client.post("/api/chats", json={"title": "Fixed template use"})).json()
    batch = (
        await client.post(
            f"/api/chats/{chat['id']}/prompt-batches",
            json={
                "idempotency_key": "fixed-ready-preview",
                "template_revision_id": ready_revision["id"],
                "contract_sha256": ready_revision["contract_sha256"],
                "item_count": 1,
                "selection_seed": 9,
                "inputs": {"subject": ["fixed resource subject"]},
            },
        )
    ).json()
    item = batch["items"][0]
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": item["reviewed_prompt"],
            "mode": "image",
            "prompt_source": _composer_source(batch, item),
        },
    )
    assert accepted.status_code == 202, accepted.text
    run = accepted.json()["run"]
    assert run["workflow_revision_id"] == workflow_revision_id
    assert run["settings_json"]["loras"] == []
    assert run["provenance_json"]["prompt_source"]["resource_policy"] == fixed
    assert (await _wait_for_run(client, run["id"]))["status"] == "complete"

    missing_lora_policy: dict[str, Any] = {
        "mode": "fixed",
        "workflow_revision_id": workflow_revision_id,
        "lora_policy": {
            "mode": "fixed",
            "stack": [
                {
                    "sha256": "a" * 64,
                    "model_strength": 1.0,
                    "clip_strength": 1.0,
                }
            ],
        },
    }
    missing_lora = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="fixed-missing-lora",
            name="Private missing LoRA",
            contract=_contract(resource_policy=missing_lora_policy),
        ),
    )
    assert missing_lora.status_code == 409
    assert missing_lora.json() == {
        "detail": "Prompt template resources are unavailable.",
        "code": "prompt-template-resources-unavailable",
    }
    assert "LoRA" not in missing_lora.text

    digest = "b" * 64
    with SessionLocal() as session:
        session.add(
            ModelAssetInstall(
                name="Verified prompt LoRA",
                kind="lora",
                local_path="C:/private/verified.safetensors",
                manifest_json={
                    "sha256": digest,
                    "comfy_name": "verified.safetensors",
                },
                active=True,
                verified_at=utcnow(),
            )
        )
        session.commit()
    fixed_lora_policy: dict[str, Any] = {
        "mode": "fixed",
        "workflow_revision_id": workflow_revision_id,
        "lora_policy": {
            "mode": "fixed",
            "stack": [
                {
                    "sha256": digest,
                    "model_strength": 0.8,
                    "clip_strength": 0.7,
                }
            ],
        },
    }
    verified_lora = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="fixed-verified-lora",
            name="Fixed verified LoRA",
            contract=_contract(resource_policy=fixed_lora_policy),
        ),
    )
    assert verified_lora.status_code == 201
    assert "private" not in verified_lora.text
    assert "comfy_name" not in verified_lora.text

    missing_workflow = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="fixed-missing-workflow",
            name="Private missing workflow",
            contract=_contract(
                resource_policy={
                    "mode": "fixed",
                    "workflow_revision_id": "wfrev_missing",
                    "lora_policy": {"mode": "none"},
                }
            ),
        ),
    )
    assert missing_workflow.status_code == 409
    assert missing_workflow.json()["code"] == "prompt-template-resources-unavailable"
    assert "wfrev_missing" not in missing_workflow.text

    untrusted_workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Untrusted prompt template workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
            "trusted": False,
        },
    )
    assert untrusted_workflow.status_code == 201
    untrusted = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="fixed-untrusted-workflow",
            name="Private untrusted workflow",
            contract=_contract(
                resource_policy={
                    "mode": "fixed",
                    "workflow_revision_id": untrusted_workflow.json()["current_revision_id"],
                    "lora_policy": {"mode": "none"},
                }
            ),
        ),
    )
    assert untrusted.status_code == 409
    assert untrusted.json()["code"] == "prompt-template-resources-unavailable"
    assert "untrusted" not in untrusted.text.lower()


@pytest.mark.asyncio
async def test_prompt_template_missing_and_pagination_boundaries(client: AsyncClient) -> None:
    missing = await client.get("/api/prompt-templates/ptdef_missing")
    assert missing.status_code == 404
    assert missing.json()["code"] == "prompt-template-not-found"

    missing_revision = await client.get(
        "/api/prompt-templates/ptdef_missing/revisions/ptrev_missing"
    )
    assert missing_revision.status_code == 404
    assert missing_revision.json()["code"] == "prompt-template-revision-not-found"

    assert (await client.get("/api/prompt-templates", params={"limit": 0})).status_code == 422
    assert (await client.get("/api/prompt-templates", params={"offset": 10_001})).status_code == 422
