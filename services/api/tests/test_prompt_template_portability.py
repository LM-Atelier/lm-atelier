from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import event, func, select
from sqlalchemy.engine import Engine
from workflow_fixtures import seed_workflow_trust

from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import (
    ModelAssetInstall,
    PromptTemplateDefinition,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.prompt_template_portability import (
    PORTABLE_RECEIPT_TTL_SECONDS,
    PromptTemplatePortabilityError,
    parse_portable_prompt_template_bundle,
    verify_prompt_template_candidate_receipt,
    verify_prompt_template_import_receipt,
)

_BUNDLE_CONTEXT = b"lm-atelier-prompt-template-bundle-v1\0"


def _resign_bundle(bundle: dict[str, object]) -> dict[str, object]:
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    bundle["bundle_sha256"] = hashlib.sha256(_BUNDLE_CONTEXT + encoded).hexdigest()
    return bundle


def _contract(*, resource_policy: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "text_to_image",
        "body": "A {{subject}}.",
        "slots": [
            {
                "name": "subject",
                "mode": "input",
                "variation_scope": "item",
            }
        ],
        "resource_policy": resource_policy or {"mode": "inherited"},
    }


async def _create_template(
    client: AsyncClient,
    *,
    key: str,
    name: str,
    resource_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    response = await client.post(
        "/api/prompt-templates",
        json={
            "idempotency_key": key,
            "name": name,
            "description": "Portable private description",
            "contract": _contract(resource_policy=resource_policy),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.asyncio
async def test_inherited_bundle_export_and_preview_are_exact_and_mutation_free(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    created = await _create_template(
        client,
        key="portable-inherited",
        name="Portable inherited",
    )
    template = created["template"]
    revision = created["revision"]
    response = await client.get(
        f"/api/prompt-templates/{template['id']}/revisions/{revision['id']}/export"
    )
    assert response.status_code == 200, response.text
    bundle = response.json()
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="prompt-template-{bundle["bundle_sha256"][:12]}.json"'
    )
    assert bundle["kind"] == "lm-atelier-prompt-template"
    assert bundle["bundle_version"] == 1
    assert bundle["workflows"] == []
    assert bundle["template"]["contract"]["resource_policy"] == {"mode": "inherited"}
    serialized = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    assert template["id"] not in serialized
    assert revision["id"] not in serialized

    with SessionLocal() as session:
        before = int(
            session.scalar(select(func.count()).select_from(PromptTemplateDefinition)) or 0
        )
    bind = SessionLocal.kw["bind"]
    assert isinstance(bind, Engine)

    def refuse_write(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")):
            raise AssertionError("import preview attempted a database write")

    event.listen(bind, "before_cursor_execute", refuse_write)
    try:
        preview = await client.post(
            "/api/prompt-templates/import/preview",
            content=serialized.encode("utf-8"),
            headers={"content-type": "application/json"},
        )
    finally:
        event.remove(bind, "before_cursor_execute", refuse_write)
    assert preview.status_code == 200, preview.text
    assert preview.headers["cache-control"] == "no-store"
    result = preview.json()
    assert result["bundle"] == bundle
    assert result["requirements"] == []
    assert result["receipt"]
    with SessionLocal() as session:
        after = int(session.scalar(select(func.count()).select_from(PromptTemplateDefinition)) or 0)
    assert after == before

    wrong_media = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized,
        headers={"content-type": "text/plain"},
    )
    assert wrong_media.status_code == 415
    assert wrong_media.json()["code"] == "prompt-template-bundle-media-type-invalid"

    parsed = parse_portable_prompt_template_bundle(serialized.encode("utf-8"))
    signing_key = app.state.services.security.local_state_signing_key(
        b"prompt-template-import-preview-v1"
    )
    issued_at = result["expires_at"] - PORTABLE_RECEIPT_TTL_SECONDS
    verify_prompt_template_import_receipt(
        result["receipt"], parsed, signing_key=signing_key, now=issued_at
    )
    with pytest.raises(PromptTemplatePortabilityError):
        verify_prompt_template_import_receipt(
            result["receipt"] + "x", parsed, signing_key=signing_key, now=issued_at
        )
    with pytest.raises(PromptTemplatePortabilityError):
        verify_prompt_template_import_receipt(
            result["receipt"],
            parsed,
            signing_key=signing_key,
            now=result["expires_at"],
        )
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    payload_token, signature_token = result["receipt"].split(".", 1)
    final_index = alphabet.index(signature_token[-1])
    alias = alphabet[(final_index & 0b111100) | 1]
    alternate_receipt = f"{payload_token}.{signature_token[:-1]}{alias}"
    assert alternate_receipt != result["receipt"]
    with pytest.raises(PromptTemplatePortabilityError):
        verify_prompt_template_import_receipt(
            alternate_receipt,
            parsed,
            signing_key=signing_key,
            now=issued_at,
        )


@pytest.mark.asyncio
async def test_bundle_parser_refuses_duplicate_keys_digest_drift_and_negative_zero(
    client: AsyncClient,
) -> None:
    created = await _create_template(
        client,
        key="portable-hostile",
        name="Portable hostile",
    )
    template = created["template"]
    revision = created["revision"]
    bundle = (
        await client.get(
            f"/api/prompt-templates/{template['id']}/revisions/{revision['id']}/export"
        )
    ).json()
    raw = json.dumps(bundle, separators=(",", ":"))
    duplicate = raw.replace(
        '"kind":"lm-atelier-prompt-template"',
        '"kind":"private-forgery","kind":"lm-atelier-prompt-template"',
        1,
    )
    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=duplicate,
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "prompt-template-bundle-duplicate-key"
    assert "private-forgery" not in rejected.text

    drifted = json.loads(raw)
    drifted["template"]["contract"]["body"] = "Private tampered body {{subject}}."
    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=json.dumps(drifted),
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "prompt-template-bundle-digest-invalid"
    assert "tampered" not in rejected.text

    padded = deepcopy(bundle)
    padded["template"]["description"] = " padded description "
    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=json.dumps(_resign_bundle(padded)),
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "prompt-template-bundle-invalid"

    nested: object = None
    for _ in range(24):
        nested = [nested]
    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=json.dumps({"x": nested}),
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "prompt-template-bundle-depth-invalid"

    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content='{"x":' + "1" * 129 + "}",
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "prompt-template-bundle-number-invalid"

    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=b"\xff",
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "prompt-template-bundle-text-invalid"

    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=json.dumps({"x": "x" * 131_073}),
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "prompt-template-bundle-size-invalid"


@pytest.mark.asyncio
async def test_candidate_resolver_refuses_a_ready_workflow_with_another_descriptor(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    exported_workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Descriptor source workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
        },
    )
    seed_workflow_trust(exported_workflow.json()["current_revision_id"])
    assert exported_workflow.status_code == 201, exported_workflow.text
    other_workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Descriptor mismatch workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "SomethingElse"}},
        },
    )
    seed_workflow_trust(other_workflow.json()["current_revision_id"])
    assert other_workflow.status_code == 201, other_workflow.text
    other_ref = other_workflow.json()["current_revision_id"]
    created = await _create_template(
        client,
        key="portable-descriptor-mismatch",
        name="Portable descriptor mismatch",
        resource_policy={
            "mode": "fixed",
            "workflow_revision_id": exported_workflow.json()["current_revision_id"],
            "lora_policy": {"mode": "none"},
        },
    )
    exported = await client.get(
        "/api/prompt-templates/"
        f"{created['template']['id']}/revisions/{created['revision']['id']}/export"
    )
    assert exported.status_code == 200, exported.text
    serialized = json.dumps(exported.json(), separators=(",", ":"))
    preview = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized,
        headers={"content-type": "application/json"},
    )
    assert preview.status_code == 200, preview.text
    requirement = next(
        item for item in preview.json()["requirements"] if item["kind"] == "workflow"
    )
    assert other_ref not in {item["local_ref"] for item in requirement["suggestions"]}
    refused = await client.post(
        "/api/prompt-templates/import/candidates/resolve",
        json={
            "bundle_json": serialized,
            "preview_receipt": preview.json()["receipt"],
            "binding_key": requirement["binding_key"],
            "local_ref": other_ref,
        },
    )
    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "prompt-template-import-receipt-invalid"
    assert other_ref not in refused.text


@pytest.mark.asyncio
async def test_preview_receipt_is_refused_for_a_different_bundle(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Cross bundle workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
        },
    )
    seed_workflow_trust(workflow.json()["current_revision_id"])
    assert workflow.status_code == 201, workflow.text
    workflow_ref = workflow.json()["current_revision_id"]
    exports: list[str] = []
    for index in (0, 1):
        created = await _create_template(
            client,
            key=f"portable-cross-bundle-{index}",
            name=f"Portable cross bundle {index}",
            resource_policy={
                "mode": "fixed",
                "workflow_revision_id": workflow_ref,
                "lora_policy": {"mode": "none"},
            },
        )
        exported = await client.get(
            "/api/prompt-templates/"
            f"{created['template']['id']}/revisions/{created['revision']['id']}/export"
        )
        assert exported.status_code == 200, exported.text
        exports.append(json.dumps(exported.json(), separators=(",", ":")))
    first, second = exports
    assert first != second
    preview = await client.post(
        "/api/prompt-templates/import/preview",
        content=first,
        headers={"content-type": "application/json"},
    )
    assert preview.status_code == 200, preview.text
    requirement = next(
        item for item in preview.json()["requirements"] if item["kind"] == "workflow"
    )
    replayed = await client.post(
        "/api/prompt-templates/import/candidates/resolve",
        json={
            "bundle_json": second,
            "preview_receipt": preview.json()["receipt"],
            "binding_key": requirement["binding_key"],
            "local_ref": workflow_ref,
        },
    )
    assert replayed.status_code == 409, replayed.text
    assert replayed.json()["code"] == "prompt-template-import-receipt-invalid"


def _bundle_with_portable_strength(value: str) -> dict[str, object]:
    return _resign_bundle(
        {
            "kind": "lm-atelier-prompt-template",
            "bundle_version": 1,
            "template": {
                "name": "Portable strength guard",
                "description": "",
                "contract": {
                    "schema_version": 1,
                    "operation": "text_to_image",
                    "body": "A {{subject}}.",
                    "slots": [{"name": "subject", "mode": "input", "variation_scope": "item"}],
                    "resource_policy": {
                        "mode": "fixed",
                        "workflow_binding_key": "workflow_1",
                        "lora_policy": {
                            "mode": "fixed",
                            "stack": [
                                {
                                    "sha256": "a" * 64,
                                    "model_strength": value,
                                    "clip_strength": "0x0.0p+0",
                                }
                            ],
                        },
                    },
                },
            },
            "workflows": [
                {
                    "key": "workflow_1",
                    "descriptor": {
                        "descriptor_version": 1,
                        "operation": "text_to_image",
                        "artifact_sha256": "b" * 64,
                        "dependency_contract_sha256": None,
                    },
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_bundle_parser_refuses_regex_valid_noncanonical_hex_strength(
    client: AsyncClient,
) -> None:
    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=json.dumps(_bundle_with_portable_strength("0x0.8000000000000p+1")),
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422, rejected.text
    assert rejected.json()["code"] == "prompt-template-bundle-strength-invalid"


@pytest.mark.asyncio
async def test_bundle_parser_names_canonical_strength_above_shared_limit(
    client: AsyncClient,
) -> None:
    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=json.dumps(_bundle_with_portable_strength("0x1.0000000000000p+3")),
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422, rejected.text
    assert rejected.json()["code"] == "prompt-template-bundle-strength-invalid"


@pytest.mark.asyncio
async def test_candidate_resolver_refuses_unknown_well_formed_binding_key(
    client: AsyncClient,
) -> None:
    workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Unknown binding workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
        },
    )
    seed_workflow_trust(workflow.json()["current_revision_id"])
    assert workflow.status_code == 201, workflow.text
    workflow_ref = workflow.json()["current_revision_id"]
    created = await _create_template(
        client,
        key="portable-unknown-binding",
        name="Portable unknown binding",
        resource_policy={
            "mode": "fixed",
            "workflow_revision_id": workflow_ref,
            "lora_policy": {"mode": "none"},
        },
    )
    exported = await client.get(
        "/api/prompt-templates/"
        f"{created['template']['id']}/revisions/{created['revision']['id']}/export"
    )
    assert exported.status_code == 200, exported.text
    serialized = json.dumps(exported.json(), separators=(",", ":"))
    preview = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized,
        headers={"content-type": "application/json"},
    )
    assert preview.status_code == 200, preview.text
    refused = await client.post(
        "/api/prompt-templates/import/candidates/resolve",
        json={
            "bundle_json": serialized,
            "preview_receipt": preview.json()["receipt"],
            "binding_key": "workflow_2",
            "local_ref": workflow_ref,
        },
    )
    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "prompt-template-import-receipt-invalid"
    assert "workflow_2" not in refused.text


@pytest.mark.asyncio
async def test_fixed_export_erases_source_identity_and_preview_suggests_local_match(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Private portable workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {},
        },
    )
    seed_workflow_trust(workflow.json()["current_revision_id"])
    assert workflow.status_code == 201, workflow.text
    workflow_revision_id = workflow.json()["current_revision_id"]
    lora_digest = "a" * 64
    with SessionLocal() as session:
        for ordinal in range(2):
            session.add(
                ModelAssetInstall(
                    name=f"Private duplicate LoRA {ordinal}",
                    kind="lora",
                    local_path=f"C:/private/portable-{ordinal}.safetensors",
                    manifest_json={
                        "sha256": lora_digest,
                        "comfy_name": f"portable-{ordinal}.safetensors",
                    },
                    active=True,
                    verified_at=utcnow(),
                )
            )
        session.commit()
    created = await _create_template(
        client,
        key="portable-fixed",
        name="Portable fixed",
        resource_policy={
            "mode": "fixed",
            "workflow_revision_id": workflow_revision_id,
            "lora_policy": {
                "mode": "fixed",
                "stack": [
                    {
                        "sha256": lora_digest,
                        "model_strength": 0.75,
                        "clip_strength": 0.0,
                    }
                ],
            },
        },
    )
    template = created["template"]
    revision = created["revision"]
    exported = await client.get(
        f"/api/prompt-templates/{template['id']}/revisions/{revision['id']}/export"
    )
    assert exported.status_code == 200, exported.text
    bundle = exported.json()
    serialized = json.dumps(bundle, separators=(",", ":"))
    assert workflow_revision_id not in serialized
    assert "Private portable workflow" not in serialized
    policy = bundle["template"]["contract"]["resource_policy"]
    assert policy["workflow_binding_key"] == "workflow_1"
    assert policy["lora_policy"]["stack"][0]["model_strength"] == "0x1.8000000000000p-1"
    assert policy["lora_policy"]["stack"][0]["clip_strength"] == "0x0.0p+0"
    assert bundle["workflows"][0]["key"] == "workflow_1"

    preview = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized,
        headers={"content-type": "application/json"},
    )
    assert preview.status_code == 200, preview.text
    requirements = preview.json()["requirements"]
    workflow_requirement = requirements[0]
    assert workflow_requirement["kind"] == "workflow"
    assert workflow_requirement["binding_key"] == "workflow_1"
    assert len(workflow_requirement["suggestions"]) == 1
    suggestion = workflow_requirement["suggestions"][0]
    assert suggestion["local_ref"] == workflow_revision_id
    assert suggestion["label"] == "Private portable workflow"
    assert len(suggestion["authority_sha256"]) == 64
    assert suggestion["candidate_receipt"]
    lora_requirement = requirements[1]
    assert lora_requirement["kind"] == "lora"
    assert lora_requirement["sha256"] == lora_digest
    assert lora_requirement["available"] is True
    assert len(lora_requirement["authority_sha256"]) == 64
    assert lora_requirement["confirmation_receipt"]

    parsed = parse_portable_prompt_template_bundle(serialized)
    signing_key = app.state.services.security.local_state_signing_key(
        b"prompt-template-import-preview-v1"
    )
    now = preview.json()["expires_at"] - PORTABLE_RECEIPT_TTL_SECONDS
    candidate_payload = {
        "kind": "workflow",
        "bundle_sha256": parsed.bundle_sha256,
        "binding_key": "workflow_1",
        "local_ref": workflow_revision_id,
        "authority_sha256": suggestion["authority_sha256"],
        "expires_at": preview.json()["expires_at"],
    }
    verify_prompt_template_candidate_receipt(
        suggestion["candidate_receipt"],
        candidate_payload,
        signing_key=signing_key,
        now=now,
    )
    with pytest.raises(PromptTemplatePortabilityError):
        verify_prompt_template_candidate_receipt(
            suggestion["candidate_receipt"],
            {**candidate_payload, "authority_sha256": "f" * 64},
            signing_key=signing_key,
            now=now,
        )
    lora_payload = {
        "kind": "lora",
        "bundle_sha256": parsed.bundle_sha256,
        "sha256": lora_digest,
        "authority_sha256": lora_requirement["authority_sha256"],
        "expires_at": preview.json()["expires_at"],
    }
    verify_prompt_template_candidate_receipt(
        lora_requirement["confirmation_receipt"],
        lora_payload,
        signing_key=signing_key,
        now=now,
    )
    with SessionLocal() as session:
        for install in session.scalars(
            select(ModelAssetInstall).where(ModelAssetInstall.kind == "lora")
        ):
            install.active = False
        session.commit()
    unavailable = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized,
        headers={"content-type": "application/json"},
    )
    assert unavailable.status_code == 200, unavailable.text
    unavailable_lora = next(
        item for item in unavailable.json()["requirements"] if item["kind"] == "lora"
    )
    assert unavailable_lora["available"] is False
    assert unavailable_lora["authority_sha256"] != lora_requirement["authority_sha256"]

    noncanonical_key = deepcopy(bundle)
    noncanonical_key["workflows"][0]["key"] = "workflow_7"
    noncanonical_key["template"]["contract"]["resource_policy"]["workflow_binding_key"] = (
        "workflow_7"
    )
    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=json.dumps(_resign_bundle(noncanonical_key)),
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "prompt-template-bundle-bindings-invalid"

    with SessionLocal() as session:
        stored_workflow = session.get(WorkflowRevision, workflow_revision_id)
        assert stored_workflow is not None
        stored_workflow.artifact_sha256 = "b" * 64
        session.commit()
    drifted_export = await client.get(
        f"/api/prompt-templates/{template['id']}/revisions/{revision['id']}/export"
    )
    assert drifted_export.status_code == 409
    assert drifted_export.json()["code"] == "prompt-template-export-conflict"
    assert "b" * 64 not in drifted_export.text

    hostile_bundle = deepcopy(bundle)
    hostile_bundle["template"]["contract"]["resource_policy"]["lora_policy"]["stack"][0][
        "clip_strength"
    ] = "-0x0.0p+0"
    hostile = json.dumps(_resign_bundle(hostile_bundle))
    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=hostile,
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "prompt-template-bundle-strength-invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize("fill", ["x", "\U0001f642"])
async def test_export_envelope_has_headroom_beyond_live_contract_limits(
    client: AsyncClient,
    fill: str,
) -> None:
    slots = [
        {
            "name": f"model_{ordinal}",
            "mode": "model",
            "variation_scope": "item",
            "guidance": fill * 3_900,
        }
        for ordinal in range(12)
    ]
    tokens = " ".join(f"{{{{model_{ordinal}}}}}" for ordinal in range(12))
    body = f"{fill * (15_800 - len(tokens) - 1)} {tokens}"
    created = await client.post(
        "/api/prompt-templates",
        json={
            "idempotency_key": f"portable-envelope-{ord(fill)}",
            "name": "Portable envelope limit",
            "description": fill * 4_000,
            "contract": {
                "schema_version": 1,
                "operation": "text_to_image",
                "body": body,
                "slots": slots,
                "resource_policy": {"mode": "inherited"},
            },
        },
    )
    assert created.status_code == 201, created.text
    exported = await client.get(
        "/api/prompt-templates/"
        f"{created.json()['template']['id']}/revisions/"
        f"{created.json()['revision']['id']}/export"
    )
    assert exported.status_code == 200, exported.text
    serialized = json.dumps(exported.json(), ensure_ascii=False, separators=(",", ":"))
    if fill == "x":
        assert (
            sum(len(value) for value in [body, *(slot["guidance"] for slot in slots)]) + 4_000
            > 65_536
        )
    else:
        assert len(serialized.encode("utf-8")) > 262_144
    preview = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized.encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["bundle"] == exported.json()


@pytest.mark.asyncio
async def test_historical_export_requires_exact_definition_membership(
    client: AsyncClient,
) -> None:
    first = await _create_template(client, key="portable-owner-a", name="Portable owner A")
    second = await _create_template(client, key="portable-owner-b", name="Portable owner B")
    response = await client.get(
        "/api/prompt-templates/"
        f"{second['template']['id']}/revisions/{first['revision']['id']}/export"
    )
    assert response.status_code == 404
    assert response.json()["code"] == "prompt-template-revision-not-found"
    assert first["revision"]["id"] not in response.text


@pytest.mark.asyncio
async def test_candidate_resolver_authorizes_match_omitted_by_bounded_preview(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    workflow_ids: list[str] = []
    for ordinal in range(21):
        workflow = await client.post(
            "/api/workflows",
            json={
                "name": f"Private resolver workflow {ordinal}",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {},
            },
        )
        seed_workflow_trust(workflow.json()["current_revision_id"])
        assert workflow.status_code == 201, workflow.text
        workflow_ids.append(workflow.json()["current_revision_id"])
    created = await _create_template(
        client,
        key="portable-resolver",
        name="Portable resolver",
        resource_policy={
            "mode": "fixed",
            "workflow_revision_id": workflow_ids[0],
            "lora_policy": {"mode": "none"},
        },
    )
    exported = await client.get(
        "/api/prompt-templates/"
        f"{created['template']['id']}/revisions/{created['revision']['id']}/export"
    )
    assert exported.status_code == 200, exported.text
    serialized = json.dumps(exported.json(), separators=(",", ":"))
    preview = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized,
        headers={"content-type": "application/json"},
    )
    assert preview.status_code == 200, preview.text
    requirement = next(
        item for item in preview.json()["requirements"] if item["kind"] == "workflow"
    )
    suggested_refs = {item["local_ref"] for item in requirement["suggestions"]}
    assert len(suggested_refs) == 20
    omitted_ref = next(item for item in workflow_ids if item not in suggested_refs)
    request_json = {
        "bundle_json": serialized,
        "preview_receipt": preview.json()["receipt"],
        "binding_key": requirement["binding_key"],
        "local_ref": omitted_ref,
    }

    bind = SessionLocal.kw["bind"]
    assert isinstance(bind, Engine)

    def refuse_write(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")):
            raise AssertionError("candidate resolution attempted a database write")

    event.listen(bind, "before_cursor_execute", refuse_write)
    try:
        resolved = await client.post(
            "/api/prompt-templates/import/candidates/resolve",
            json=request_json,
        )
    finally:
        event.remove(bind, "before_cursor_execute", refuse_write)
    assert resolved.status_code == 200, resolved.text
    assert resolved.headers["cache-control"] == "no-store"
    candidate = resolved.json()
    assert candidate["local_ref"] == omitted_ref
    assert candidate["label"].startswith("Private resolver workflow ")
    assert candidate["candidate_receipt"]

    parsed = parse_portable_prompt_template_bundle(serialized)
    signing_key = app.state.services.security.local_state_signing_key(
        b"prompt-template-import-preview-v1"
    )
    verify_prompt_template_candidate_receipt(
        candidate["candidate_receipt"],
        {
            "kind": "workflow",
            "bundle_sha256": parsed.bundle_sha256,
            "binding_key": requirement["binding_key"],
            "local_ref": omitted_ref,
            "authority_sha256": candidate["authority_sha256"],
            "expires_at": preview.json()["expires_at"],
        },
        signing_key=signing_key,
        now=preview.json()["expires_at"] - PORTABLE_RECEIPT_TTL_SECONDS,
    )

    with SessionLocal() as session:
        omitted_revision = session.get(WorkflowRevision, omitted_ref)
        assert omitted_revision is not None
        omitted_revision.trusted = False
        session.commit()
    event.listen(bind, "before_cursor_execute", refuse_write)
    try:
        rejected = await client.post(
            "/api/prompt-templates/import/candidates/resolve",
            json=request_json,
        )
        missing = await client.post(
            "/api/prompt-templates/import/candidates/resolve",
            json={**request_json, "local_ref": "not-present"},
        )
    finally:
        event.remove(bind, "before_cursor_execute", refuse_write)
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "prompt-template-import-receipt-invalid"
    assert omitted_ref not in rejected.text
    assert missing.status_code == 409
    assert missing.json()["code"] == "prompt-template-import-receipt-invalid"
    assert "not-present" not in missing.text


@pytest.mark.asyncio
async def test_sensitive_candidate_validation_never_echoes_input(
    client: AsyncClient,
) -> None:
    sentinel = "private-credential-do-not-echo"
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
    assert extra.json() == {
        "code": "prompt-template-import-request-invalid",
        "detail": "Prompt template import request is invalid.",
    }
    assert sentinel not in extra.text

    duplicate = await client.post(
        "/api/prompt-templates/import/candidates/resolve",
        content=(
            '{"bundle_json":"' + sentinel + '","bundle_json":"other","preview_receipt":"receipt",'
            '"binding_key":"workflow_1","local_ref":"local"}'
        ),
        headers={"content-type": "application/json"},
    )
    assert duplicate.status_code == 422
    assert duplicate.json()["code"] == "prompt-template-import-request-invalid"
    assert sentinel not in duplicate.text


@pytest.mark.asyncio
async def test_workflow_pool_keys_deduplicate_only_exact_source_revisions(
    client: AsyncClient,
) -> None:
    workflow_ids: list[str] = []
    for ordinal in range(2):
        workflow = await client.post(
            "/api/workflows",
            json={
                "name": f"Private same-descriptor workflow {ordinal}",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {},
            },
        )
        seed_workflow_trust(workflow.json()["current_revision_id"])
        assert workflow.status_code == 201, workflow.text
        workflow_ids.append(workflow.json()["current_revision_id"])
    lora_digest = "c" * 64
    with SessionLocal() as session:
        session.add(
            ModelAssetInstall(
                name="Private pool LoRA",
                kind="lora",
                local_path="C:/private/pool.safetensors",
                manifest_json={"sha256": lora_digest, "comfy_name": "pool.safetensors"},
                active=True,
                verified_at=utcnow(),
            )
        )
        session.commit()
    created = await _create_template(
        client,
        key="portable-workflow-pool",
        name="Portable workflow pool",
        resource_policy={
            "mode": "pool",
            "strategy": "round_robin",
            "options": [
                {
                    "workflow_revision_id": workflow_ids[0],
                    "lora_policy": {"mode": "none"},
                },
                {
                    "workflow_revision_id": workflow_ids[0],
                    "lora_policy": {
                        "mode": "fixed",
                        "stack": [
                            {
                                "sha256": lora_digest,
                                "model_strength": 1.0,
                                "clip_strength": 0.5,
                            }
                        ],
                    },
                },
                {
                    "workflow_revision_id": workflow_ids[1],
                    "lora_policy": {"mode": "none"},
                },
            ],
        },
    )
    template = created["template"]
    revision = created["revision"]
    exported = await client.get(
        f"/api/prompt-templates/{template['id']}/revisions/{revision['id']}/export"
    )
    assert exported.status_code == 200, exported.text
    bundle = exported.json()
    assert [item["key"] for item in bundle["workflows"]] == ["workflow_1", "workflow_2"]
    assert bundle["workflows"][0]["descriptor"] == bundle["workflows"][1]["descriptor"]
    options = bundle["template"]["contract"]["resource_policy"]["options"]
    assert [item["workflow_binding_key"] for item in options] == [
        "workflow_1",
        "workflow_1",
        "workflow_2",
    ]
    serialized = json.dumps(bundle, separators=(",", ":"))
    assert all(workflow_id not in serialized for workflow_id in workflow_ids)
    assert "Private same-descriptor" not in serialized

    reordered_root = {key: bundle[key] for key in reversed(bundle)}
    preview = await client.post(
        "/api/prompt-templates/import/preview",
        content=json.dumps(reordered_root),
        headers={"content-type": "application/json"},
    )
    assert preview.status_code == 200, preview.text
    workflow_requirements = [
        item for item in preview.json()["requirements"] if item["kind"] == "workflow"
    ]
    assert len(workflow_requirements) == 2
    assert all(len(item["suggestions"]) == 2 for item in workflow_requirements)

    with SessionLocal() as session:
        first_revision = session.get(WorkflowRevision, workflow_ids[0])
        second_revision = session.get(WorkflowRevision, workflow_ids[1])
        assert first_revision is not None and second_revision is not None
        first_definition = session.get(WorkflowDefinition, first_revision.workflow_id)
        second_definition = session.get(WorkflowDefinition, second_revision.workflow_id)
        assert first_definition is not None and second_definition is not None
        first_definition.current_revision_id = None
        family = WorkflowFamily(
            name="Private disabled import family",
            enabled=False,
            archived=False,
        )
        session.add(family)
        session.flush()
        second_definition.family_id = family.id
        second_definition.variant_key = "image"
        session.add(
            WorkflowPreference(
                workflow_family_id=family.id,
                selector_capability="image",
                enabled=True,
                is_default=False,
                sort_order=0,
            )
        )
        session.commit()
    ineligible = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized,
        headers={"content-type": "application/json"},
    )
    assert ineligible.status_code == 200, ineligible.text
    assert all(
        item["suggestions"] == []
        for item in ineligible.json()["requirements"]
        if item["kind"] == "workflow"
    )
    with SessionLocal() as session:
        stored_family = session.get(WorkflowFamily, family.id)
        assert stored_family is not None
        stored_family.enabled = True
        stored_family.archived = True
        session.commit()
    archived = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized,
        headers={"content-type": "application/json"},
    )
    assert archived.status_code == 200, archived.text
    assert all(
        item["suggestions"] == []
        for item in archived.json()["requirements"]
        if item["kind"] == "workflow"
    )
    with SessionLocal() as session:
        stored_family = session.get(WorkflowFamily, family.id)
        preference = session.scalar(
            select(WorkflowPreference).where(
                WorkflowPreference.workflow_family_id == family.id,
                WorkflowPreference.selector_capability == "image",
            )
        )
        assert stored_family is not None and preference is not None
        stored_family.archived = False
        preference.enabled = False
        session.commit()
    preference_disabled = await client.post(
        "/api/prompt-templates/import/preview",
        content=serialized,
        headers={"content-type": "application/json"},
    )
    assert preference_disabled.status_code == 200, preference_disabled.text
    assert all(
        item["suggestions"] == []
        for item in preference_disabled.json()["requirements"]
        if item["kind"] == "workflow"
    )

    reordered_options = deepcopy(bundle)
    reordered_options["template"]["contract"]["resource_policy"]["options"].reverse()
    rejected = await client.post(
        "/api/prompt-templates/import/preview",
        content=json.dumps(reordered_options),
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "prompt-template-bundle-digest-invalid"
