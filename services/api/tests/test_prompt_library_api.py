from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm import api as api_module
from local_lm import orchestrator as orchestrator_module
from local_lm.auxiliary_assets import AutomaticLoraSelection, ResolvedLoraStack
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import (
    Chat,
    Job,
    Message,
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
    Project,
    PromptExpansionBatch,
    PromptExpansionItem,
    PromptTemplateDefinition,
    PromptTemplateRevision,
    Run,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
)
from local_lm.prompt_library import PromptLibraryError, create_prompt_template
from local_lm.prompt_model_invocation import (
    PromptModelInvocationError,
    PromptModelInvocationResult,
)
from local_lm.prompt_model_values import (
    parse_prompt_model_values,
    prompt_model_values_sha256,
)
from local_lm.schemas import WorkerStatus


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
async def test_selected_prompt_batch_queues_one_atomic_exact_media_plan(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Template batch queue"})).json()
    template = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(key="prompt-create-atomic"),
        )
    ).json()
    revision = template["revision"]
    batch = (
        await client.post(
            f"/api/chats/{chat['id']}/prompt-batches",
            json={
                "idempotency_key": "prompt-preview-atomic",
                "template_revision_id": revision["id"],
                "contract_sha256": revision["contract_sha256"],
                "item_count": 3,
                "selection_seed": 41,
                "inputs": {
                    "subject": [
                        "private first subject",
                        "private skipped subject",
                        "private third subject",
                    ]
                },
            },
        )
    ).json()
    deselected = await client.patch(
        f"/api/prompt-batches/{batch['id']}/items/2",
        json={
            "expected_review_version": batch["items"][1]["review_version"],
            "expected_plan_version": batch["plan_version"],
            "reviewed_prompt": batch["items"][1]["reviewed_prompt"],
            "selected": False,
        },
    )
    assert deselected.status_code == 200
    batch = deselected.json()
    selected = [item for item in batch["items"] if item["selected"]]
    payload = {
        "idempotency_key": "prompt-queue-atomic",
        "expected_plan_version": batch["plan_version"],
        "expected_plan_sha256": batch["plan_sha256"],
    }
    auto_selected_prompts: list[str] = []

    def select_item_lora(
        *_args: object,
        **_kwargs: object,
    ) -> AutomaticLoraSelection:
        prompt = str(_args[2])
        auto_selected_prompts.append(prompt)
        ordinal = len(auto_selected_prompts)
        return AutomaticLoraSelection(
            settings=[
                {
                    "asset_id": f"lora-item-{ordinal}",
                    "model_strength": 0.8,
                    "clip_strength": 0.7,
                    "enabled": True,
                }
            ],
            provenance={"mode": "automatic", "prompt_ordinal": ordinal},
        )

    def resolve_item_lora(
        _session: object,
        _revision: object,
        settings: list[dict[str, Any]],
    ) -> ResolvedLoraStack:
        return ResolvedLoraStack(
            settings=settings,
            provenance=[{"asset_id": settings[0]["asset_id"]}],
            graph_sha256="a" * 64,
        )

    monkeypatch.setattr(
        orchestrator_module,
        "select_automatic_lora_stack",
        select_item_lora,
    )
    monkeypatch.setattr(orchestrator_module, "resolve_lora_stack", resolve_item_lora)

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/prompt-batches/{batch['id']}/queue",
            json=payload,
        )
        assert response.status_code == 202, response.text
        queued = response.json()
        assert queued["replayed"] is False
        assert queued["state"] == "queued"
        assert queued["plan_version"] == batch["plan_version"] + 1
        assert queued["queue_idempotency_key"] == payload["idempotency_key"]
        plan_id = queued["work_plan_id"]
        assert plan_id

        plan = (await client.get(f"/api/work-plans/{plan_id}")).json()
        assert plan["planner_version"] == "prompt-template-v1"
        assert plan["source_action"] == "prompt_library"
        assert plan["summary_json"]["prompt_batch_id"] == batch["id"]
        assert plan["summary_json"]["output_count"] == 2
        assert [step["prompt"] for step in plan["steps"]] == [
            item["reviewed_prompt"] for item in selected
        ]
        assert auto_selected_prompts == [item["reviewed_prompt"] for item in selected]
        assert [step["settings_json"]["loras"][0]["asset_id"] for step in plan["steps"]] == [
            "lora-item-1",
            "lora-item-2",
        ]
        assert all(step["settings_json"].get("batch_size") == 1 for step in plan["steps"])
        seeds = [step["settings_json"]["seed"] for step in plan["steps"]]
        assert len(set(seeds)) == 2
        assert all(type(seed) is int and 0 <= seed < 2_147_483_648 for seed in seeds)

        selected_outputs = [item for item in queued["items"] if item["selected"]]
        skipped = next(item for item in queued["items"] if not item["selected"])
        assert [item["work_step_id"] for item in selected_outputs] == [
            step["id"] for step in plan["steps"]
        ]
        assert [item["run_id"] for item in selected_outputs] == plan["summary_json"]["run_ids"]
        assert [item["media_seed"] for item in selected_outputs] == seeds
        assert skipped["work_step_id"] is None
        assert skipped["run_id"] is None
        assert skipped["media_seed"] is None

        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run).where(Run.work_plan_id == plan_id).order_by(Run.work_step_id)
                ).all()
            )
            steps = list(
                session.scalars(
                    select(WorkStep).where(WorkStep.plan_id == plan_id).order_by(WorkStep.ordinal)
                ).all()
            )
            assert len(runs) == len(steps) == 2
            assert {run.standalone_prompt for run in runs} == {
                item["reviewed_prompt"] for item in selected
            }
            for run in runs:
                witness = run.provenance_json["prompt_source"]
                serialized = json.dumps(witness)
                assert witness["batch_id"] == batch["id"]
                assert witness["relation"] == "exact"
                assert "private first subject" not in serialized
                assert "private skipped subject" not in serialized
                assert "private third subject" not in serialized

        replay = await client.post(
            f"/api/prompt-batches/{batch['id']}/queue",
            json=payload,
        )
        assert replay.status_code == 202
        assert replay.json()["replayed"] is True
        assert replay.json()["work_plan_id"] == plan_id

        wrong_key = await client.post(
            f"/api/prompt-batches/{batch['id']}/queue",
            json={**payload, "idempotency_key": "prompt-queue-wrong-key"},
        )
        assert wrong_key.status_code == 409
        assert wrong_key.json()["code"] == "prompt-source-conflict"


@pytest.mark.parametrize("strategy", ["round_robin", "random"])
@pytest.mark.asyncio
async def test_prompt_batch_queue_allocates_lora_pool_per_item_deterministically(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    workflow = (
        await client.post(
            "/api/workflows",
            json={
                "name": f"Prompt pool {strategy} workflow",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {},
                "trusted": True,
            },
        )
    ).json()
    digests = ("a" * 64, "b" * 64)
    asset_ids: dict[str, str] = {}
    with SessionLocal() as session:
        for ordinal, digest in enumerate(digests, start=1):
            asset = ModelAssetInstall(
                name=f"Verified pool LoRA {ordinal}",
                kind="lora",
                local_path=f"C:/private/pool-{ordinal}.safetensors",
                manifest_json={
                    "sha256": digest,
                    "comfy_name": f"pool-{ordinal}.safetensors",
                },
                active=True,
                verified_at=utcnow(),
            )
            session.add(asset)
            session.flush()
            asset_ids[digest] = asset.id
        session.commit()
    resource_policy: dict[str, object] = {
        "mode": "fixed",
        "workflow_revision_id": workflow["current_revision_id"],
        "lora_policy": {
            "mode": "pool",
            "strategy": strategy,
            "stacks": [
                [{"sha256": digest, "model_strength": 0.8, "clip_strength": 0.7}]
                for digest in digests
            ],
        },
    }
    template_response = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key=f"prompt-pool-{strategy}",
            name=f"Prompt pool {strategy}",
            contract=_contract(resource_policy=resource_policy),
        ),
    )
    assert template_response.status_code == 201, template_response.text
    revision = template_response.json()["revision"]
    chat = (await client.post("/api/chats", json={"title": "Prompt pool queue"})).json()
    selection_seed = 7
    batch = (
        await client.post(
            f"/api/chats/{chat['id']}/prompt-batches",
            json={
                "idempotency_key": f"prompt-pool-preview-{strategy}",
                "template_revision_id": revision["id"],
                "contract_sha256": revision["contract_sha256"],
                "item_count": 3,
                "selection_seed": selection_seed,
                "inputs": {"subject": ["first", "second", "third"]},
            },
        )
    ).json()

    resolved_settings: list[list[dict[str, Any]]] = []

    def resolve_pool_lora(
        _session: object,
        _revision: object,
        settings: list[dict[str, Any]],
    ) -> ResolvedLoraStack:
        resolved_settings.append(settings)
        return ResolvedLoraStack(
            settings=settings,
            provenance=[{"asset_id": settings[0]["asset_id"]}],
            graph_sha256="c" * 64,
        )

    monkeypatch.setattr(orchestrator_module, "resolve_lora_stack", resolve_pool_lora)
    payload = {
        "idempotency_key": f"prompt-pool-queue-{strategy}",
        "expected_plan_version": batch["plan_version"],
        "expected_plan_sha256": batch["plan_sha256"],
    }
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(f"/api/prompt-batches/{batch['id']}/queue", json=payload)
        assert response.status_code == 202, response.text
        queued = response.json()
        plan = (await client.get(f"/api/work-plans/{queued['work_plan_id']}")).json()

        def allocated_index(item_ordinal: int) -> int:
            if strategy == "round_robin":
                return (selection_seed + item_ordinal - 1) % len(digests)
            material = "\x00".join(
                ("prompt-template-resource-pool-v1", str(selection_seed), str(item_ordinal))
            )
            return int.from_bytes(
                hashlib.sha256(material.encode("ascii")).digest()[:8], "big"
            ) % len(digests)

        expected_digests = tuple(digests[allocated_index(ordinal)] for ordinal in range(1, 4))
        expected_asset_ids = [asset_ids[digest] for digest in expected_digests]
        assert [settings[0]["asset_id"] for settings in resolved_settings] == expected_asset_ids
        assert [step["settings_json"]["loras"][0]["asset_id"] for step in plan["steps"]] == (
            expected_asset_ids
        )
        with SessionLocal() as session:
            runs = session.scalars(
                select(Run)
                .join(WorkStep, Run.work_step_id == WorkStep.id)
                .where(Run.work_plan_id == plan["id"])
                .order_by(WorkStep.ordinal)
            ).all()
            allocated_policies = [
                run.provenance_json["prompt_source"]["resource_policy"] for run in runs
            ]
        assert [policy["lora_policy"]["mode"] for policy in allocated_policies] == [
            "fixed",
            "fixed",
            "fixed",
        ]
        assert [policy["lora_policy"]["stack"][0]["sha256"] for policy in allocated_policies] == (
            list(expected_digests)
        )
        assert all("stacks" not in policy["lora_policy"] for policy in allocated_policies)

        replay = await client.post(f"/api/prompt-batches/{batch['id']}/queue", json=payload)
        assert replay.status_code == 202
        assert replay.json()["replayed"] is True
        assert replay.json()["work_plan_id"] == plan["id"]


@pytest.mark.asyncio
async def test_prompt_batch_queue_freezes_distinct_workflow_contexts_per_item(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    workflow_ids: list[str] = []
    for index, steps in enumerate((11, 23), start=1):
        response = await client.post(
            "/api/workflows",
            json={
                "name": f"Prompt execution workflow {index}",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"node": {"class_type": f"PromptWorkflow{index}"}},
                "input_schema": {
                    "type": "object",
                    "properties": {"steps": {"type": "integer", "default": steps}},
                },
                "trusted": True,
            },
        )
        assert response.status_code == 201
        workflow_ids.append(response.json()["current_revision_id"])

    resource_policy: dict[str, object] = {
        "mode": "pool",
        "strategy": "round_robin",
        "options": [
            {
                "workflow_revision_id": workflow_id,
                "lora_policy": {"mode": "none"},
            }
            for workflow_id in workflow_ids
        ],
    }
    template = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(
                key="prompt-workflow-pool-execution",
                name="Prompt workflow pool execution",
                contract=_contract(resource_policy=resource_policy),
            ),
        )
    ).json()
    revision = template["revision"]
    chat = (await client.post("/api/chats", json={"title": "Workflow pool queue"})).json()
    batch = (
        await client.post(
            f"/api/chats/{chat['id']}/prompt-batches",
            json={
                "idempotency_key": "prompt-workflow-pool-preview",
                "template_revision_id": revision["id"],
                "contract_sha256": revision["contract_sha256"],
                "item_count": 2,
                "selection_seed": 0,
                "inputs": {"subject": ["first", "second"]},
            },
        )
    ).json()
    payload = {
        "idempotency_key": "prompt-workflow-pool-queue",
        "expected_plan_version": batch["plan_version"],
        "expected_plan_sha256": batch["plan_sha256"],
    }

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(f"/api/prompt-batches/{batch['id']}/queue", json=payload)
        assert response.status_code == 202, response.text
        queued = response.json()
        plan = (await client.get(f"/api/work-plans/{queued['work_plan_id']}")).json()

        assert [step["workflow_revision_id"] for step in plan["steps"]] == workflow_ids
        assert [step["settings_json"]["steps"] for step in plan["steps"]] == [11, 23]
        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run)
                    .join(WorkStep, Run.work_step_id == WorkStep.id)
                    .where(Run.work_plan_id == plan["id"])
                    .order_by(WorkStep.ordinal)
                ).all()
            )
        assert [run.workflow_revision_id for run in runs] == workflow_ids
        assert [run.settings_json["steps"] for run in runs] == [11, 23]
        assert [run.provenance_json["workflow"]["revision_id"] for run in runs] == workflow_ids
        second_run_id = runs[1].id
        second_message_id = runs[1].assistant_message_id
        second_prompt_source = runs[1].provenance_json["prompt_source"]
        assert [
            run.provenance_json["prompt_source"]["resource_policy"]["workflow_revision_id"]
            for run in runs
        ] == workflow_ids
        assert all(
            run.provenance_json["prompt_source"]["resource_policy"]["mode"] == "fixed"
            for run in runs
        )
        assert [run.provenance_json["media_plan_estimate"]["work_units"] for run in runs] == [
            1024 * 1024 * 11,
            1024 * 1024 * 23,
        ]
        assert plan["summary_json"]["media_plan_estimate"]["work_units"] == (
            1024 * 1024 * (11 + 23)
        )

        replay = await client.post(f"/api/prompt-batches/{batch['id']}/queue", json=payload)
        assert replay.status_code == 202
        assert replay.json()["replayed"] is True
        assert replay.json()["work_plan_id"] == plan["id"]

    await _wait_for_run(client, second_run_id)
    regenerated = await client.post(
        f"/api/messages/{second_message_id}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202, regenerated.text
    regenerated_run = regenerated.json()["run"]
    assert regenerated_run["workflow_revision_id"] == workflow_ids[1]
    assert regenerated_run["settings_json"]["steps"] == 23
    assert regenerated_run["provenance_json"]["prompt_source"] == second_prompt_source


@pytest.mark.parametrize("scope", ["chat", "project"])
@pytest.mark.asyncio
async def test_prompt_batch_workflow_pool_refuses_partly_compatible_shared_setting_atomically(
    client: AsyncClient,
    scope: str,
) -> None:
    workflow_ids: list[str] = []
    for index, properties in enumerate(
        (
            {"render_style": {"type": "string", "default": "soft"}},
            {},
        ),
        start=1,
    ):
        response = await client.post(
            "/api/workflows",
            json={
                "name": f"Prompt settings workflow {index}",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"node": {"class_type": f"SettingsWorkflow{index}"}},
                "input_schema": {"type": "object", "properties": properties},
                "trusted": True,
            },
        )
        assert response.status_code == 201
        workflow_ids.append(response.json()["current_revision_id"])

    resource_policy: dict[str, object] = {
        "mode": "pool",
        "strategy": "round_robin",
        "options": [
            {
                "workflow_revision_id": workflow_id,
                "lora_policy": {"mode": "none"},
            }
            for workflow_id in workflow_ids
        ],
    }
    template = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(
                key="prompt-workflow-pool-settings",
                name="Prompt workflow pool settings",
                contract=_contract(resource_policy=resource_policy),
            ),
        )
    ).json()
    project = (await client.post("/api/projects", json={"name": "Workflow pool project"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Workflow pool settings", "project_id": project["id"]},
        )
    ).json()
    revision = template["revision"]
    batch = (
        await client.post(
            f"/api/chats/{chat['id']}/prompt-batches",
            json={
                "idempotency_key": "prompt-workflow-pool-settings-preview",
                "template_revision_id": revision["id"],
                "contract_sha256": revision["contract_sha256"],
                "item_count": 2,
                "selection_seed": 0,
                "inputs": {"subject": ["first", "second"]},
            },
        )
    ).json()
    with SessionLocal() as session:
        owner = (
            session.get(Chat, chat["id"])
            if scope == "chat"
            else session.get(Project, project["id"])
        )
        assert owner is not None
        owner.generation_settings_json = {"image": {"render_style": "shared"}}
        session.commit()

    queue_payload = {
        "idempotency_key": "prompt-workflow-pool-settings-queue",
        "expected_plan_version": batch["plan_version"],
        "expected_plan_sha256": batch["plan_sha256"],
    }
    refused = await client.post(
        f"/api/prompt-batches/{batch['id']}/queue",
        json=queue_payload,
    )
    assert refused.status_code == 422
    assert refused.json()["code"] == "generation-settings-invalid"
    with SessionLocal() as session:
        stored = session.get(PromptExpansionBatch, batch["id"])
        assert stored is not None
        assert stored.state == "draft"
        assert stored.work_plan_id is None
        assert session.scalar(select(WorkPlan).where(WorkPlan.chat_id == chat["id"])) is None
        assert session.scalar(select(Run).where(Run.chat_id == chat["id"])) is None
        assert (
            session.scalar(
                select(Job).join(Run, Job.run_id == Run.id).where(Run.chat_id == chat["id"])
            )
            is None
        )

        owner = (
            session.get(Chat, chat["id"])
            if scope == "chat"
            else session.get(Project, project["id"])
        )
        assert owner is not None
        owner.generation_settings_json = {"image": {"obsolete_style": "legacy"}}
        session.commit()

    # A setting stale for every option keeps the existing persisted-default
    # behavior: it is ignored rather than making an old chat unusable.
    recovered = await client.post(
        f"/api/prompt-batches/{batch['id']}/queue",
        json=queue_payload,
    )
    assert recovered.status_code == 202, recovered.text


@pytest.mark.asyncio
async def test_prompt_batch_workflow_pool_applies_setting_every_option_accepts(
    client: AsyncClient,
) -> None:
    workflow_ids: list[str] = []
    for index in (1, 2):
        response = await client.post(
            "/api/workflows",
            json={
                "name": f"Shared settings workflow {index}",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"node": {"class_type": f"SharedSettingsWorkflow{index}"}},
                "input_schema": {
                    "type": "object",
                    "properties": {"render_style": {"type": "string", "default": "soft"}},
                },
                "trusted": True,
            },
        )
        assert response.status_code == 201
        workflow_ids.append(response.json()["current_revision_id"])

    resource_policy: dict[str, object] = {
        "mode": "pool",
        "strategy": "round_robin",
        "options": [
            {
                "workflow_revision_id": workflow_id,
                "lora_policy": {"mode": "none"},
            }
            for workflow_id in workflow_ids
        ],
    }
    template = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(
                key="prompt-pool-shared-accepted",
                name="Prompt pool shared accepted",
                contract=_contract(resource_policy=resource_policy),
            ),
        )
    ).json()
    project = (await client.post("/api/projects", json={"name": "Shared pool project"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Shared pool settings", "project_id": project["id"]},
        )
    ).json()
    revision = template["revision"]
    batch = (
        await client.post(
            f"/api/chats/{chat['id']}/prompt-batches",
            json={
                "idempotency_key": "prompt-pool-shared-accepted-preview",
                "template_revision_id": revision["id"],
                "contract_sha256": revision["contract_sha256"],
                "item_count": 2,
                "selection_seed": 0,
                "inputs": {"subject": ["first", "second"]},
            },
        )
    ).json()
    with SessionLocal() as session:
        owner = session.get(Chat, chat["id"])
        assert owner is not None
        owner.generation_settings_json = {"image": {"render_style": "shared"}}
        session.commit()

    queued = await client.post(
        f"/api/prompt-batches/{batch['id']}/queue",
        json={
            "idempotency_key": "prompt-pool-shared-accepted-queue",
            "expected_plan_version": batch["plan_version"],
            "expected_plan_sha256": batch["plan_sha256"],
        },
    )
    assert queued.status_code == 202, queued.text
    plan = (await client.get(f"/api/work-plans/{queued.json()['work_plan_id']}")).json()
    assert [step["settings_json"]["render_style"] for step in plan["steps"]] == [
        "shared",
        "shared",
    ]


@pytest.mark.asyncio
async def test_prompt_batch_workflow_pool_unselected_stale_option_refuses_before_claim_or_media(
    client: AsyncClient,
) -> None:
    workflow_ids: list[str] = []
    for index in range(2):
        response = await client.post(
            "/api/workflows",
            json={
                "name": f"Prompt stale workflow {index}",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"node": {"class_type": f"StaleWorkflow{index}"}},
                "trusted": True,
            },
        )
        assert response.status_code == 201
        workflow_ids.append(response.json()["current_revision_id"])
    resource_policy: dict[str, object] = {
        "mode": "pool",
        "strategy": "round_robin",
        "options": [
            {
                "workflow_revision_id": workflow_id,
                "lora_policy": {"mode": "none"},
            }
            for workflow_id in workflow_ids
        ],
    }
    template = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(
                key="prompt-workflow-pool-stale",
                name="Prompt workflow pool stale",
                contract=_contract(resource_policy=resource_policy),
            ),
        )
    ).json()
    revision = template["revision"]
    chat = (await client.post("/api/chats", json={"title": "Workflow pool stale"})).json()
    batch = (
        await client.post(
            f"/api/chats/{chat['id']}/prompt-batches",
            json={
                "idempotency_key": "prompt-workflow-pool-stale-preview",
                "template_revision_id": revision["id"],
                "contract_sha256": revision["contract_sha256"],
                "item_count": 1,
                "selection_seed": 0,
                "inputs": {"subject": ["first"]},
            },
        )
    ).json()
    with SessionLocal() as session:
        stale = session.get(WorkflowRevision, workflow_ids[1])
        assert stale is not None
        stale.trusted = False
        before = (
            session.scalar(select(WorkPlan).where(WorkPlan.chat_id == chat["id"])),
            session.scalar(select(WorkStep).join(WorkPlan).where(WorkPlan.chat_id == chat["id"])),
            session.scalar(select(Run).where(Run.chat_id == chat["id"])),
            session.scalar(select(Job).join(Run).where(Run.chat_id == chat["id"])),
        )
        assert before == (None, None, None, None)
        session.commit()

    refused = await client.post(
        f"/api/prompt-batches/{batch['id']}/queue",
        json={
            "idempotency_key": "prompt-workflow-pool-stale-queue",
            "expected_plan_version": batch["plan_version"],
            "expected_plan_sha256": batch["plan_sha256"],
        },
    )
    assert refused.status_code in {409, 422}
    with SessionLocal() as session:
        stored = session.get(PromptExpansionBatch, batch["id"])
        assert stored is not None
        assert stored.state == "draft"
        assert stored.work_plan_id is None
        assert session.scalar(select(WorkPlan).where(WorkPlan.chat_id == chat["id"])) is None
        assert session.scalar(select(Run).where(Run.chat_id == chat["id"])) is None
        assert (
            session.scalar(
                select(Job).join(Run, Job.run_id == Run.id).where(Run.chat_id == chat["id"])
            )
            is None
        )


@pytest.mark.asyncio
async def test_prompt_batch_queue_refuses_stale_or_empty_selection_without_media_rows(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Queue refusal"})).json()
    template = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(key="prompt-create-queue-refusal"),
        )
    ).json()
    revision = template["revision"]
    batch = (
        await client.post(
            f"/api/chats/{chat['id']}/prompt-batches",
            json={
                "idempotency_key": "prompt-preview-queue-refusal",
                "template_revision_id": revision["id"],
                "contract_sha256": revision["contract_sha256"],
                "item_count": 1,
                "selection_seed": 43,
                "inputs": {"subject": ["private refusal subject"]},
            },
        )
    ).json()
    stale_payload = {
        "idempotency_key": "prompt-queue-stale",
        "expected_plan_version": batch["plan_version"],
        "expected_plan_sha256": batch["plan_sha256"],
    }
    deselected = await client.patch(
        f"/api/prompt-batches/{batch['id']}/items/1",
        json={
            "expected_review_version": batch["items"][0]["review_version"],
            "expected_plan_version": batch["plan_version"],
            "reviewed_prompt": batch["items"][0]["reviewed_prompt"],
            "selected": False,
        },
    )
    assert deselected.status_code == 200

    stale = await client.post(
        f"/api/prompt-batches/{batch['id']}/queue",
        json=stale_payload,
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "prompt-source-conflict"
    current = deselected.json()
    empty = await client.post(
        f"/api/prompt-batches/{batch['id']}/queue",
        json={
            "idempotency_key": "prompt-queue-empty",
            "expected_plan_version": current["plan_version"],
            "expected_plan_sha256": current["plan_sha256"],
        },
    )
    assert empty.status_code == 422
    assert empty.json()["code"] == "prompt-source-invalid"
    reread = (await client.get(f"/api/prompt-batches/{batch['id']}")).json()
    assert reread["state"] == "draft"
    assert reread["work_plan_id"] is None
    assert reread["items"][0]["work_step_id"] is None
    assert reread["items"][0]["run_id"] is None
    with SessionLocal() as session:
        assert not session.scalar(select(WorkPlan.id).where(WorkPlan.chat_id == chat["id"]))
        assert not session.scalar(select(Run.id).where(Run.chat_id == chat["id"]))
        assert not session.scalar(select(Job.id).join(Run).where(Run.chat_id == chat["id"]))


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
    assert unavailable.json()["code"] == "prompt-model-worker-unavailable"
    with SessionLocal() as session:
        assert session.query(PromptExpansionBatch).count() == 0


def _ready_chat_model(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    with SessionLocal() as session:
        install = ModelInstall(
            name="Prompt template chat model",
            role="chat",
            engine=app.state.services.settings.chat_engine,
            local_path="managed/prompt-template-chat-model",
            active=True,
        )
        session.add(install)
        session.flush()
        profile = ModelProfile(
            name="Prompt template chat profile",
            role="chat",
            engine=app.state.services.settings.chat_engine,
            model_install_id=install.id,
        )
        session.add(profile)
        session.commit()
        profile_id = profile.id
        install_id = install.id
    monkeypatch.setattr(
        app.state.services.processes,
        "statuses",
        lambda: [
            WorkerStatus(
                name="chat",
                state="ready",
                managed=True,
                running=True,
                profile_id=profile_id,
            )
        ],
    )
    return profile_id, install_id


@pytest.mark.asyncio
async def test_model_guided_prompt_batch_invokes_once_and_replays_before_readiness(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _profile_id, install_id = _ready_chat_model(app, monkeypatch)
    calls = 0

    async def invoke(_adapter: object, *, contract: Any, data: Any) -> PromptModelInvocationResult:
        nonlocal calls
        calls += 1
        items = data.items
        values = parse_prompt_model_values(
            {
                "version": 1,
                "batch_values": {},
                "items": [
                    {
                        "ordinal": item.ordinal,
                        "values": {"subject": f"model subject {item.ordinal}"},
                    }
                    for item in items
                ],
            },
            contract=contract,
        )
        return PromptModelInvocationResult(
            values=values,
            values_sha256=prompt_model_values_sha256(values, contract=contract),
            attempts=(),
        )

    monkeypatch.setattr(api_module, "invoke_prompt_model_values", invoke)
    chat = (await client.post("/api/chats", json={"title": "Model previews"})).json()
    model_contract = _contract()
    model_contract["slots"] = [
        {
            "name": "subject",
            "mode": "model",
            "variation_scope": "item",
            "guidance": "one concrete subject",
        }
    ]
    created = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(
                key="model-success-template",
                name="Model success template",
                contract=model_contract,
            ),
        )
    ).json()
    revision = created["revision"]
    payload = {
        "idempotency_key": "model-success-preview",
        "template_revision_id": revision["id"],
        "contract_sha256": revision["contract_sha256"],
        "item_count": 2,
        "selection_seed": 9,
        "inputs": {},
    }
    first = await client.post(f"/api/chats/{chat['id']}/prompt-batches", json=payload)
    assert first.status_code == 201
    assert first.json()["replayed"] is False
    assert len(first.json()["items"]) == 2
    assert calls == 1

    monkeypatch.setattr(app.state.services.processes, "statuses", lambda: [])
    replay = await client.post(f"/api/chats/{chat['id']}/prompt-batches", json=payload)
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["id"] == first.json()["id"]
    assert calls == 1
    with SessionLocal() as session:
        batch = session.get(PromptExpansionBatch, first.json()["id"])
        assert batch is not None
        assert json.loads(batch.model_snapshot_json) == {
            "adapter_id": app.state.services.settings.chat_engine,
            "kind": "model",
            "model_id": install_id,
            "values_sha256": json.loads(batch.model_snapshot_json)["values_sha256"],
            "version": 1,
        }


@pytest.mark.asyncio
async def test_model_guided_prompt_batch_failure_is_fixed_and_atomic(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready_chat_model(app, monkeypatch)

    async def fail(*_args: object, **_kwargs: object) -> PromptModelInvocationResult:
        raise PromptModelInvocationError()

    monkeypatch.setattr(api_module, "invoke_prompt_model_values", fail)
    chat = (await client.post("/api/chats", json={"title": "Model failure"})).json()
    model_contract = _contract()
    model_contract["slots"] = [
        {
            "name": "subject",
            "mode": "model",
            "variation_scope": "item",
            "guidance": "one concrete subject",
        }
    ]
    created = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(
                key="model-failure-template",
                name="Model failure template",
                contract=model_contract,
            ),
        )
    ).json()
    revision = created["revision"]
    failed = await client.post(
        f"/api/chats/{chat['id']}/prompt-batches",
        json={
            "idempotency_key": "model-failure-preview",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": 1,
            "selection_seed": 3,
            "inputs": {},
        },
    )
    assert failed.status_code == 503
    assert failed.json() == {
        "detail": (
            "The chat model could not fill the template slots. Retry, or use authored "
            "inputs and choices instead."
        ),
        "code": "prompt-model-invocation-failed",
    }
    with SessionLocal() as session:
        assert session.query(PromptExpansionBatch).count() == 0


@pytest.mark.asyncio
async def test_prompt_batch_distinct_capacity_has_actionable_fixed_feedback(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Choice capacity"})).json()
    choice_contract = _contract()
    choice_contract["slots"] = [
        {
            "name": "subject",
            "mode": "choice",
            "variation_scope": "item",
            "choices": ["private fox", "private hare"],
            "choice_strategy": "distinct",
        }
    ]
    created = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(
                key="choice-capacity-template",
                name="Choice capacity template",
                contract=choice_contract,
            ),
        )
    ).json()
    revision = created["revision"]
    refused = await client.post(
        f"/api/chats/{chat['id']}/prompt-batches",
        json={
            "idempotency_key": "choice-capacity-preview",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": 3,
            "selection_seed": 4,
            "inputs": {},
        },
    )
    assert refused.status_code == 422
    assert refused.json()["code"] == "prompt-batch-distinct-capacity-exceeded"
    assert "private" not in refused.text
    with SessionLocal() as session:
        assert session.query(PromptExpansionBatch).count() == 0


@pytest.mark.asyncio
async def test_prompt_batch_reusable_choice_can_repeat_without_a_choice_count_limit(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Reusable choice"})).json()
    choice_contract = _contract()
    choice_contract["slots"] = [
        {
            "name": "subject",
            "mode": "choice",
            "variation_scope": "item",
            "choices": ["repeatable subject"],
            "choice_strategy": "with_replacement",
        }
    ]
    created = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(
                key="reusable-choice-template",
                name="Reusable choice template",
                contract=choice_contract,
            ),
        )
    ).json()
    revision = created["revision"]
    response = await client.post(
        f"/api/chats/{chat['id']}/prompt-batches",
        json={
            "idempotency_key": "reusable-choice-preview",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": 3,
            "selection_seed": 4,
            "inputs": {},
        },
    )
    assert response.status_code == 201
    assert [item["rendered_prompt"] for item in response.json()["items"]] == [
        "A repeatable subject.",
        "A repeatable subject.",
        "A repeatable subject.",
    ]


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
    assert name_collision.json()["code"] == "prompt-template-name-taken"

    other = (
        await client.post(
            "/api/prompt-templates",
            json=_create_payload(key="prompt-create-unique", name="Unique template"),
        )
    ).json()
    rename_collision = await client.patch(
        f"/api/prompt-templates/{other['template']['id']}",
        json={
            "expected_current_revision_id": other["revision"]["id"],
            "name": created["template"]["name"],
        },
    )
    assert rename_collision.status_code == 409
    assert rename_collision.json()["code"] == "prompt-template-name-taken"

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
    unavailable_pool = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="pool-missing-lora",
            name="Private unavailable pool",
            contract=_contract(
                resource_policy={
                    "mode": "fixed",
                    "workflow_revision_id": workflow_revision_id,
                    "lora_policy": {
                        "mode": "pool",
                        "strategy": "round_robin",
                        "stacks": [
                            [
                                {
                                    "sha256": digest,
                                    "model_strength": 0.8,
                                    "clip_strength": 0.7,
                                }
                            ],
                            [
                                {
                                    "sha256": "c" * 64,
                                    "model_strength": 1.0,
                                    "clip_strength": 1.0,
                                }
                            ],
                        ],
                    },
                }
            ),
        ),
    )
    assert unavailable_pool.status_code == 409
    assert unavailable_pool.json()["code"] == "prompt-template-resources-unavailable"
    assert "c" * 64 not in unavailable_pool.text

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
async def test_prompt_template_workflow_pool_requires_every_option_and_lora(
    client: AsyncClient,
) -> None:
    workflow_ids: list[str] = []
    for index in range(2):
        response = await client.post(
            "/api/workflows",
            json={
                "name": f"Prompt pool workflow {index}",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {},
                "trusted": True,
            },
        )
        assert response.status_code == 201
        workflow_ids.append(response.json()["current_revision_id"])

    pool: dict[str, object] = {
        "mode": "pool",
        "strategy": "round_robin",
        "options": [
            {
                "workflow_revision_id": workflow_id,
                "lora_policy": {"mode": "none"},
            }
            for workflow_id in workflow_ids
        ],
    }
    ready = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="workflow-pool-ready",
            name="Workflow pool ready",
            contract=_contract(resource_policy=pool),
        ),
    )
    assert ready.status_code == 201, ready.text

    missing_workflow = copy.deepcopy(pool)
    missing_options = cast(list[dict[str, object]], missing_workflow["options"])
    missing_options[1]["workflow_revision_id"] = "wfrev_private_missing"
    unavailable = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="workflow-pool-missing-workflow",
            name="Workflow pool missing workflow",
            contract=_contract(resource_policy=missing_workflow),
        ),
    )
    assert unavailable.status_code == 409
    assert unavailable.json()["code"] == "prompt-template-resources-unavailable"
    assert "wfrev_private_missing" not in unavailable.text

    missing_lora = copy.deepcopy(pool)
    lora_options = cast(list[dict[str, object]], missing_lora["options"])
    lora_options[1]["lora_policy"] = {
        "mode": "fixed",
        "stack": [
            {
                "sha256": "e" * 64,
                "model_strength": 1.0,
                "clip_strength": 1.0,
            }
        ],
    }
    unavailable = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="workflow-pool-missing-lora",
            name="Workflow pool missing LoRA",
            contract=_contract(resource_policy=missing_lora),
        ),
    )
    assert unavailable.status_code == 409
    assert unavailable.json()["code"] == "prompt-template-resources-unavailable"
    assert "e" * 64 not in unavailable.text


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
