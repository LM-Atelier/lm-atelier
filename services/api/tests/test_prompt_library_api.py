from __future__ import annotations

from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ModelAssetInstall, PromptTemplateDefinition, PromptTemplateRevision
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
