from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from workflow_fixtures import seed_workflow_trust

from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import (
    ModelAssetInstall,
    PromptTemplateDefinition,
    PromptTemplateImportWinner,
    PromptTemplateRevision,
    WorkflowRevision,
)
from local_lm.prompt_template_import import (
    PromptTemplateImportError,
    commit_prompt_template_import,
)


def _contract(resource_policy: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "text_to_image",
        "body": "A {{subject}}.",
        "slots": [{"name": "subject", "mode": "input", "variation_scope": "item"}],
        "resource_policy": resource_policy or {"mode": "inherited"},
    }


async def _portable_request(
    client: AsyncClient,
    *,
    source_key: str,
    source_name: str,
    destination_name: str,
    resource_policy: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    created = await client.post(
        "/api/prompt-templates",
        json={
            "idempotency_key": source_key,
            "name": source_name,
            "description": "Private source description",
            "contract": _contract(resource_policy),
        },
    )
    assert created.status_code == 201, created.text
    exported = await client.get(
        "/api/prompt-templates/"
        f"{created.json()['template']['id']}/revisions/"
        f"{created.json()['revision']['id']}/export"
    )
    assert exported.status_code == 200, exported.text
    serialized = json.dumps(exported.json(), separators=(",", ":"))
    preview = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized,
        headers={"content-type": "application/json"},
    )
    assert preview.status_code == 200, preview.text
    requirements = preview.json()["requirements"]
    payload = {
        "idempotency_key": f"import-{source_key}",
        "bundle_json": serialized,
        "preview_receipt": preview.json()["receipt"],
        "confirmed_bundle_sha256": exported.json()["bundle_sha256"],
        "destination_name": destination_name,
        "workflow_bindings": [
            {
                "binding_key": item["binding_key"],
                "local_ref": item["suggestions"][0]["local_ref"],
                "candidate_receipt": item["suggestions"][0]["candidate_receipt"],
            }
            for item in requirements
            if item["kind"] == "workflow"
        ],
        "lora_confirmations": [
            {
                "sha256": item["sha256"],
                "confirmation_receipt": item["confirmation_receipt"],
            }
            for item in requirements
            if item["kind"] == "lora"
        ],
    }
    return payload, preview.json()


@pytest.mark.asyncio
async def test_inherited_import_commits_once_and_replays_before_receipt_validation(
    client: AsyncClient,
) -> None:
    payload, _preview = await _portable_request(
        client,
        source_key="atomic-inherited",
        source_name="Atomic inherited source",
        destination_name="Atomic inherited destination",
    )
    first = await client.post("/api/prompt-templates/import", json=payload)
    assert first.status_code == 201, first.text
    assert first.headers["cache-control"] == "no-store"
    assert first.json()["idempotent"] is False
    assert set(first.json()) == {
        "template_id",
        "revision_id",
        "contract_sha256",
        "idempotent",
    }

    replay = await client.post(
        "/api/prompt-templates/import",
        json={
            **payload,
            "destination_name": "  Atomic inherited destination  ",
            "preview_receipt": "expired-placeholder",
        },
    )
    assert replay.status_code == 201, replay.text
    assert replay.json() == {**first.json(), "idempotent": True}

    conflict = await client.post(
        "/api/prompt-templates/import",
        json={
            **payload,
            "destination_name": "Private changed destination",
            "preview_receipt": "expired-placeholder",
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "prompt-template-import-conflict"
    assert "Private changed" not in conflict.text

    with SessionLocal() as session:
        winner = session.get(PromptTemplateImportWinner, payload["idempotency_key"])
        assert winner is not None
        assert winner.prompt_template_id == first.json()["template_id"]
        assert winner.prompt_template_revision_id == first.json()["revision_id"]

    deleted = await client.delete(
        f"/api/prompt-templates/{first.json()['template_id']}",
        params={"expected_current_revision_id": first.json()["revision_id"]},
    )
    assert deleted.status_code == 204
    deleted_replay = await client.post(
        "/api/prompt-templates/import",
        json={**payload, "preview_receipt": "expired-placeholder"},
    )
    assert deleted_replay.status_code == 409
    assert deleted_replay.json() == {
        "detail": (
            "This import already created a template that was deleted. Use a new idempotency key."
        ),
        "code": "prompt-template-import-deleted",
    }

    replacement = await client.post(
        "/api/prompt-templates/import",
        json={**payload, "idempotency_key": "import-atomic-inherited-replacement"},
    )
    assert replacement.status_code == 201, replacement.text
    assert replacement.json()["template_id"] != first.json()["template_id"]


@pytest.mark.asyncio
async def test_fixed_import_rebinds_the_exact_authorized_local_workflow(
    client: AsyncClient,
) -> None:
    workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Atomic local workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
        },
    )
    seed_workflow_trust(workflow.json()["current_revision_id"])
    assert workflow.status_code == 201, workflow.text
    local_ref = workflow.json()["current_revision_id"]
    payload, _preview = await _portable_request(
        client,
        source_key="atomic-fixed",
        source_name="Atomic fixed source",
        destination_name="Atomic fixed destination",
        resource_policy={
            "mode": "fixed",
            "workflow_revision_id": local_ref,
            "lora_policy": {"mode": "none"},
        },
    )
    committed = await client.post("/api/prompt-templates/import", json=payload)
    assert committed.status_code == 201, committed.text
    stored = await client.get(f"/api/prompt-templates/{committed.json()['template_id']}")
    assert stored.status_code == 200, stored.text
    assert (
        stored.json()["current_revision"]["contract_json"]["resource_policy"][
            "workflow_revision_id"
        ]
        == local_ref
    )

    drift_payload = {
        **payload,
        "idempotency_key": "import-atomic-fixed-drift",
        "destination_name": "Atomic fixed drift",
    }
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, local_ref)
        assert revision is not None
        revision.trusted = False
        session.commit()
    refused = await client.post("/api/prompt-templates/import", json=drift_payload)
    assert refused.status_code == 409
    replay_payload = {
        **payload,
        "preview_receipt": "expired-placeholder",
        "workflow_bindings": [
            {**item, "candidate_receipt": "stale-placeholder"}
            for item in payload["workflow_bindings"]
        ],
    }
    replay = await client.post("/api/prompt-templates/import", json=replay_payload)
    assert replay.status_code == 201, replay.text
    assert replay.json()["idempotent"] is True
    with SessionLocal() as session:
        assert (
            session.get(
                PromptTemplateImportWinner,
                drift_payload["idempotency_key"],
            )
            is None
        )


@pytest.mark.asyncio
async def test_sensitive_commit_and_candidate_validation_never_echoes_input(
    client: AsyncClient,
) -> None:
    sentinel = "private-credential-do-not-echo"
    duplicate = '{"idempotency_key":"x","bundle_json":"' + sentinel + '","bundle_json":"other"}'
    response = await client.post(
        "/api/prompt-templates/import",
        content=duplicate,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {
        "code": "prompt-template-import-request-invalid",
        "detail": "Prompt template import request is invalid.",
    }
    assert sentinel not in response.text

    extra = await client.post(
        "/api/prompt-templates/import/candidates/resolve",
        json={
            "bundle_json": sentinel,
            "preview_receipt": "receipt",
            "binding_key": "workflow_1",
            "local_ref": "local",
            "credential": sentinel,
        },
    )
    assert extra.status_code == 422
    assert sentinel not in extra.text


@pytest.mark.asyncio
async def test_service_refuses_caller_transaction_without_ending_it(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    payload, _preview = await _portable_request(
        client,
        source_key="atomic-caller-transaction",
        source_name="Atomic caller source",
        destination_name="Atomic caller destination",
    )
    with SessionLocal() as session:
        session.scalar(select(func.count()).select_from(PromptTemplateDefinition))
        assert session.in_transaction()
        with pytest.raises(PromptTemplateImportError):
            commit_prompt_template_import(
                session,
                idempotency_key=payload["idempotency_key"],
                raw_bundle=payload["bundle_json"],
                preview_receipt=payload["preview_receipt"],
                confirmed_bundle_sha256=payload["confirmed_bundle_sha256"],
                destination_name=payload["destination_name"],
                workflow_bindings=payload["workflow_bindings"],
                lora_confirmations=payload["lora_confirmations"],
                expected_engine="mock",
                signing_key=app.state.services.security.local_state_signing_key(
                    b"prompt-template-import-preview-v1"
                ),
            )
        assert session.in_transaction()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["definition", "revision", "current", "winner"])
async def test_injected_failure_rolls_back_every_import_row_and_key_is_reusable(
    app: FastAPI,
    client: AsyncClient,
    stage: str,
) -> None:
    payload, _preview = await _portable_request(
        client,
        source_key=f"atomic-rollback-{stage}",
        source_name=f"Atomic rollback source {stage}",
        destination_name=f"Atomic rollback destination {stage}",
    )

    def fail(current: str) -> None:
        if current == stage:
            raise RuntimeError("synthetic failure")

    with SessionLocal() as session:
        before_definitions = int(
            session.scalar(select(func.count()).select_from(PromptTemplateDefinition)) or 0
        )
        before_revisions = int(
            session.scalar(select(func.count()).select_from(PromptTemplateRevision)) or 0
        )
        session.rollback()
        with pytest.raises(RuntimeError, match="synthetic failure"):
            commit_prompt_template_import(
                session,
                idempotency_key=payload["idempotency_key"],
                raw_bundle=payload["bundle_json"],
                preview_receipt=payload["preview_receipt"],
                confirmed_bundle_sha256=payload["confirmed_bundle_sha256"],
                destination_name=payload["destination_name"],
                workflow_bindings=payload["workflow_bindings"],
                lora_confirmations=payload["lora_confirmations"],
                expected_engine="mock",
                signing_key=app.state.services.security.local_state_signing_key(
                    b"prompt-template-import-preview-v1"
                ),
                _stage_hook=fail,
            )
        assert session.get(PromptTemplateImportWinner, payload["idempotency_key"]) is None
        assert (
            int(session.scalar(select(func.count()).select_from(PromptTemplateDefinition)) or 0)
            == before_definitions
        )
        assert (
            int(session.scalar(select(func.count()).select_from(PromptTemplateRevision)) or 0)
            == before_revisions
        )
        session.rollback()
        result = commit_prompt_template_import(
            session,
            idempotency_key=payload["idempotency_key"],
            raw_bundle=payload["bundle_json"],
            preview_receipt=payload["preview_receipt"],
            confirmed_bundle_sha256=payload["confirmed_bundle_sha256"],
            destination_name=payload["destination_name"],
            workflow_bindings=payload["workflow_bindings"],
            lora_confirmations=payload["lora_confirmations"],
            expected_engine="mock",
            signing_key=app.state.services.security.local_state_signing_key(
                b"prompt-template-import-preview-v1"
            ),
        )
        assert result.idempotent is False


@pytest.mark.asyncio
async def test_concurrent_exact_import_has_one_creator_and_one_replay(
    client: AsyncClient,
) -> None:
    payload, _preview = await _portable_request(
        client,
        source_key="atomic-concurrent-exact",
        source_name="Atomic concurrent source",
        destination_name="Atomic concurrent destination",
    )
    first, second = await asyncio.gather(
        client.post("/api/prompt-templates/import", json=payload),
        client.post("/api/prompt-templates/import", json=payload),
    )
    assert first.status_code == second.status_code == 201
    assert {first.json()["idempotent"], second.json()["idempotent"]} == {False, True}
    assert first.json()["template_id"] == second.json()["template_id"]
    with SessionLocal() as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(PromptTemplateImportWinner)
                .where(PromptTemplateImportWinner.idempotency_key == payload["idempotency_key"])
            )
            == 1
        )


@pytest.mark.asyncio
async def test_concurrent_same_name_different_keys_has_no_orphan(
    client: AsyncClient,
) -> None:
    first_payload, _ = await _portable_request(
        client,
        source_key="atomic-name-race-a",
        source_name="Atomic name race source A",
        destination_name="Atomic shared destination",
    )
    second_payload, _ = await _portable_request(
        client,
        source_key="atomic-name-race-b",
        source_name="Atomic name race source B",
        destination_name="Atomic shared destination",
    )
    first, second = await asyncio.gather(
        client.post("/api/prompt-templates/import", json=first_payload),
        client.post("/api/prompt-templates/import", json=second_payload),
    )
    assert sorted((first.status_code, second.status_code)) == [201, 409]
    with SessionLocal() as session:
        winners = int(
            session.scalar(
                select(func.count())
                .select_from(PromptTemplateImportWinner)
                .where(
                    PromptTemplateImportWinner.idempotency_key.in_(
                        [
                            first_payload["idempotency_key"],
                            second_payload["idempotency_key"],
                        ]
                    )
                )
            )
            or 0
        )
        assert winners == 1


@pytest.mark.asyncio
async def test_missing_extra_and_duplicate_requirement_rows_fail_before_authority(
    client: AsyncClient,
) -> None:
    workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Atomic exact requirement workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
        },
    )
    seed_workflow_trust(workflow.json()["current_revision_id"])
    assert workflow.status_code == 201
    payload, _ = await _portable_request(
        client,
        source_key="atomic-exact-requirements",
        source_name="Atomic exact requirements source",
        destination_name="Atomic exact requirements destination",
        resource_policy={
            "mode": "fixed",
            "workflow_revision_id": workflow.json()["current_revision_id"],
            "lora_policy": {"mode": "none"},
        },
    )
    row = payload["workflow_bindings"][0]
    variants = [
        {**payload, "workflow_bindings": []},
        {
            **payload,
            "workflow_bindings": [
                row,
                {
                    "binding_key": "workflow_2",
                    "local_ref": row["local_ref"],
                    "candidate_receipt": row["candidate_receipt"],
                },
            ],
        },
        {**payload, "workflow_bindings": [row, row]},
    ]
    for index, variant in enumerate(variants):
        response = await client.post(
            "/api/prompt-templates/import",
            json={
                **variant,
                "idempotency_key": f"import-exact-requirements-{index}",
                "preview_receipt": "authority-must-not-run",
            },
        )
        assert response.status_code == 422
        assert response.json()["code"] == "prompt-template-import-request-invalid"


@pytest.mark.asyncio
async def test_lora_that_becomes_unavailable_after_preview_cannot_commit(
    client: AsyncClient,
) -> None:
    workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Atomic LoRA workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
        },
    )
    seed_workflow_trust(workflow.json()["current_revision_id"])
    assert workflow.status_code == 201
    digest = "9" * 64
    with SessionLocal() as session:
        session.add(
            ModelAssetInstall(
                name="Atomic private LoRA",
                kind="lora",
                local_path="C:/private/atomic-lora.safetensors",
                manifest_json={
                    "sha256": digest,
                    "comfy_name": "atomic-lora.safetensors",
                },
                active=True,
                verified_at=utcnow(),
            )
        )
        session.commit()
    payload, _ = await _portable_request(
        client,
        source_key="atomic-lora-drift",
        source_name="Atomic LoRA source",
        destination_name="Atomic LoRA destination",
        resource_policy={
            "mode": "fixed",
            "workflow_revision_id": workflow.json()["current_revision_id"],
            "lora_policy": {
                "mode": "fixed",
                "stack": [
                    {
                        "sha256": digest,
                        "model_strength": 1.0,
                        "clip_strength": 1.0,
                    }
                ],
            },
        },
    )
    with SessionLocal() as session:
        install = session.scalar(
            select(ModelAssetInstall).where(
                ModelAssetInstall.kind == "lora",
                ModelAssetInstall.manifest_json["sha256"].as_string() == digest,
            )
        )
        assert install is not None
        install.active = False
        session.commit()
    refused = await client.post("/api/prompt-templates/import", json=payload)
    assert refused.status_code == 409
    with SessionLocal() as session:
        assert session.get(PromptTemplateImportWinner, payload["idempotency_key"]) is None


@pytest.mark.asyncio
async def test_many_to_one_rebind_refuses_duplicate_options_but_keeps_distinct_policies(
    client: AsyncClient,
) -> None:
    workflow_ids: list[str] = []
    for ordinal in range(2):
        response = await client.post(
            "/api/workflows",
            json={
                "name": f"Atomic many-to-one workflow {ordinal}",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {},
            },
        )
        seed_workflow_trust(response.json()["current_revision_id"])
        assert response.status_code == 201
        workflow_ids.append(response.json()["current_revision_id"])

    duplicate_payload, _ = await _portable_request(
        client,
        source_key="atomic-many-to-one-duplicate",
        source_name="Atomic many-to-one duplicate source",
        destination_name="Atomic many-to-one duplicate destination",
        resource_policy={
            "mode": "pool",
            "strategy": "round_robin",
            "options": [
                {
                    "workflow_revision_id": workflow_ids[0],
                    "lora_policy": {"mode": "none"},
                },
                {
                    "workflow_revision_id": workflow_ids[1],
                    "lora_policy": {"mode": "none"},
                },
            ],
        },
    )
    assert (
        duplicate_payload["workflow_bindings"][0]["local_ref"]
        == duplicate_payload["workflow_bindings"][1]["local_ref"]
    )
    duplicate = await client.post(
        "/api/prompt-templates/import",
        json=duplicate_payload,
    )
    assert duplicate.status_code == 422
    assert duplicate.json()["code"] == "prompt-template-bundle-invalid"

    distinct_payload, _ = await _portable_request(
        client,
        source_key="atomic-many-to-one-distinct",
        source_name="Atomic many-to-one distinct source",
        destination_name="Atomic many-to-one distinct destination",
        resource_policy={
            "mode": "pool",
            "strategy": "round_robin",
            "options": [
                {
                    "workflow_revision_id": workflow_ids[0],
                    "lora_policy": {"mode": "none"},
                },
                {
                    "workflow_revision_id": workflow_ids[1],
                    "lora_policy": {"mode": "inherited_auto"},
                },
            ],
        },
    )
    committed = await client.post(
        "/api/prompt-templates/import",
        json={
            **distinct_payload,
            "workflow_bindings": list(reversed(distinct_payload["workflow_bindings"])),
        },
    )
    assert committed.status_code == 201, committed.text


@pytest.mark.asyncio
async def test_service_refuses_new_dirty_and_deleted_state_without_clearing_it(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    payload, _ = await _portable_request(
        client,
        source_key="atomic-pending-state",
        source_name="Atomic pending state source",
        destination_name="Atomic pending state destination",
    )

    def invoke(session: object) -> None:
        commit_prompt_template_import(
            session,
            idempotency_key=payload["idempotency_key"],
            raw_bundle=payload["bundle_json"],
            preview_receipt=payload["preview_receipt"],
            confirmed_bundle_sha256=payload["confirmed_bundle_sha256"],
            destination_name=payload["destination_name"],
            workflow_bindings=payload["workflow_bindings"],
            lora_confirmations=payload["lora_confirmations"],
            expected_engine="mock",
            signing_key=app.state.services.security.local_state_signing_key(
                b"prompt-template-import-preview-v1"
            ),
        )

    with SessionLocal() as session:
        pending = PromptTemplateDefinition(
            name="Private pending definition",
            description="",
        )
        session.add(pending)
        with pytest.raises(PromptTemplateImportError):
            invoke(session)
        assert pending in session.new

    with SessionLocal() as session:
        existing = session.scalar(
            select(PromptTemplateDefinition).where(
                PromptTemplateDefinition.name == "Atomic pending state source"
            )
        )
        assert existing is not None
        existing.description = "Private dirty description"
        with pytest.raises(PromptTemplateImportError):
            invoke(session)
        assert existing in session.dirty

    with SessionLocal() as session:
        existing = session.scalar(
            select(PromptTemplateDefinition).where(
                PromptTemplateDefinition.name == "Atomic pending state source"
            )
        )
        assert existing is not None
        session.delete(existing)
        with pytest.raises(PromptTemplateImportError):
            invoke(session)
        assert existing in session.deleted
