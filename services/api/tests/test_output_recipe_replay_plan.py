"""Planning an exact replay of a generation record, through the real route."""

from __future__ import annotations

import copy
from typing import Any, cast

from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import func, select

from local_lm.db import SessionLocal
from local_lm.model_planner import workflow_artifact_contract
from local_lm.models import (
    Chat,
    Job,
    Message,
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
    Run,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowRevision,
)
from local_lm.output_recipe_replay import edit_prompt_preamble
from local_lm.output_recipe_v1 import open_output_recipe, seal_output_recipe

PLAN = "/api/output-recipes/replay-plan"
PROMPT = "a ceramic cup on a wooden table"
FILES = {"model.safetensors": "1" * 64}


def _profile(files: dict[str, str] = FILES, *, name: str = "Recorded model") -> tuple[str, str]:
    """A picture profile whose install declares exactly `files`: (profile id, install id)."""

    with SessionLocal() as session:
        install = ModelInstall(
            name=name,
            role="image",
            engine="mock",
            local_path="neutral-model-folder",
            compatibility="compatible",
            manifest_json={"expected_sha256": dict(files)},
            active=True,
        )
        session.add(install)
        session.flush()
        profile = ModelProfile(name=name, role="image", engine="mock", model_install_id=install.id)
        session.add(profile)
        session.commit()
        return profile.id, install.id


async def _recorded(client: AsyncClient, profile_id: str) -> tuple[dict[str, Any], bytes]:
    """A finished picture made with `profile_id`: its run, and its record with the prompt."""

    chat = (await client.post("/api/chats", json={"title": "Recorded"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": PROMPT, "mode": "image", "profile_id": profile_id},
    )
    assert turn.status_code == 202, turn.text
    run_id = turn.json()["run"]["id"]

    async def read() -> dict[str, Any]:
        return cast(dict[str, Any], (await client.get(f"/api/runs/{run_id}")).json())

    run = dict(await wait_for_terminal_status(read, what=run_id, expected="complete"))
    artifact_id = run["provenance_json"]["outputs"][0]["artifact_id"]
    record = await client.get(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe", params={"prompts": "include"}
    )
    assert record.status_code == 200, record.text
    return run, record.content


async def _plan(client: AsyncClient, content: bytes) -> Any:
    response = await client.post(
        PLAN, content=content, headers={"content-type": "application/octet-stream"}
    )
    assert response.status_code == 200, response.text
    return response.json()


def _codes(plan: dict[str, Any]) -> list[str]:
    return [item["code"] for item in plan["refusals"]]


def _resealed(content: bytes, change: Any) -> bytes:
    """The record with `change` applied to it and sealed again, as another tool might write it."""

    payload = copy.deepcopy(open_output_recipe(content))
    del payload["digest"]
    change(payload)
    return seal_output_recipe(payload)


def _counts() -> dict[str, int]:
    with SessionLocal() as session:
        return {
            model.__name__: session.scalar(select(func.count()).select_from(model)) or 0
            for model in (Chat, Message, Run, Job)
        }


async def test_a_record_made_here_plans_back_to_what_made_it(client: AsyncClient) -> None:
    profile_id, _ = _profile()
    run, content = await _recorded(client, profile_id)
    record = open_output_recipe(content)
    assert record["model"]["files"] == FILES
    before = _counts()

    plan = await _plan(client, content)

    assert plan["ready"] is True, plan
    assert plan["refusals"] == []
    assert plan["digest"] == record["digest"]
    assert plan["resolved"] == {
        "mode": "image",
        "workflow_revision_id": run["workflow_revision_id"],
        "profile_id": profile_id,
        "lora_asset_ids": [],
        "input_artifact_ids": [],
    }
    # Nothing was started, and the answer repeats neither the prompt nor a setting.
    assert _counts() == before
    assert PROMPT not in str(plan)
    assert str(record["seed"]["value"]) not in str(plan)


async def test_a_requirement_held_twice_is_refused_rather_than_guessed(
    client: AsyncClient,
) -> None:
    profile_id, install_id = _profile()
    run, content = await _recorded(client, profile_id)
    with SessionLocal() as session:
        recorded = session.get(WorkflowRevision, run["workflow_revision_id"])
        assert recorded is not None
        definition = session.get(WorkflowDefinition, recorded.workflow_id)
        assert definition is not None
        family = WorkflowFamily(name="Same workflow again")
        twin = WorkflowDefinition(
            family=family,
            variant_key="create",
            name="Same workflow again",
            operation=definition.operation,
        )
        session.add_all(
            [
                family,
                twin,
                WorkflowRevision(
                    definition=twin,
                    version=1,
                    engine=recorded.engine,
                    api_graph_json=recorded.api_graph_json,
                    input_schema_json=recorded.input_schema_json,
                    dependencies_json=recorded.dependencies_json,
                    dependency_contract_sha256=recorded.dependency_contract_sha256,
                    artifact_sha256=recorded.artifact_sha256,
                    trusted=True,
                ),
                ModelProfile(
                    name="Same install again",
                    role="image",
                    engine="mock",
                    model_install_id=install_id,
                ),
            ]
        )
        session.commit()

    plan = await _plan(client, content)

    assert plan["ready"] is False
    assert plan["resolved"] is None
    assert _codes(plan) == ["replay-workflow-ambiguous", "replay-model-ambiguous"]


async def test_a_model_or_workflow_that_is_no_longer_exactly_the_recorded_one_is_missing(
    client: AsyncClient,
) -> None:
    profile_id, install_id = _profile()
    run, content = await _recorded(client, profile_id)
    with SessionLocal() as session:
        install = session.get(ModelInstall, install_id)
        assert install is not None
        # The recorded files and one more: a different install, not the same one.
        install.manifest_json = {"expected_sha256": {**FILES, "extra.safetensors": "2" * 64}}
        revision = session.get(WorkflowRevision, run["workflow_revision_id"])
        assert revision is not None
        # The stored identity is kept, but the graph it names is gone.
        revision.api_graph_json = {**revision.api_graph_json, "999": {"class_type": "Changed"}}
        session.commit()

    plan = await _plan(client, content)

    assert plan["ready"] is False
    assert _codes(plan) == ["replay-workflow-missing", "replay-model-missing"]
    workflow_refusal = plan["refusals"][0]
    assert workflow_refusal["sha256"] == open_output_recipe(content)["workflow"]["artifact_sha256"]


async def test_shapes_a_turn_cannot_carry_are_refused_with_their_reasons(
    client: AsyncClient,
) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)
    lora = {"sha256": "3" * 64, "model_strength": 1.0, "clip_strength": 1.0, "enabled": True}

    def gapped_and_strong(record: dict[str, Any]) -> None:
        record["loras"] = [
            {**lora, "position": 0, "model_strength": 5.0},
            {**lora, "sha256": "4" * 64, "position": 2},
        ]

    def an_input_where_none_is_taken(record: dict[str, Any]) -> None:
        record["inputs"] = [
            {"sha256": "5" * 64, "role": "source", "size_bytes": 10, "media_type": "image/png"}
        ]

    def a_mask(record: dict[str, Any]) -> None:
        record["operation"] = "image_to_image"
        record["prompt"]["positive"] = edit_prompt_preamble() + record["prompt"]["positive"]
        record["inputs"] = [
            {"sha256": "5" * 64, "role": "source", "size_bytes": 10, "media_type": "image/png"},
            {"sha256": "6" * 64, "role": "mask", "size_bytes": 10, "media_type": "image/png"},
        ]

    unsupported = {
        "gapped and strong": (gapped_and_strong, ["lora_strength", "lora_positions"]),
        "input": (an_input_where_none_is_taken, ["inputs_for_operation"]),
        "mask": (a_mask, ["mask_input"]),
    }
    for name, (change, reasons) in unsupported.items():
        plan = await _plan(client, _resealed(content, change))
        refusal = next(
            item for item in plan["refusals"] if item["code"] == "replay-record-unsupported"
        )
        assert refusal["reasons"] == reasons, name
        assert plan["ready"] is False, name


async def test_only_what_the_record_says_is_missing_and_matters_refuses(
    client: AsyncClient,
) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)
    record = open_output_recipe(content)
    # A turn freezes nothing, and the test engine made it: neither stands in the way.
    assert set(record["reproducibility"]["missing"]) == {"frozen_snapshot_absent", "mock_engine"}

    def prompt_left_out(payload: dict[str, Any]) -> None:
        payload["prompt"] = {
            "included": False,
            "positive": None,
            "negative": None,
            "omitted_reason": "chosen",
        }
        payload["reproducibility"]["missing"] = sorted(
            [*payload["reproducibility"]["missing"], "prompt_omitted"]
        )

    plan = await _plan(client, _resealed(content, prompt_left_out))

    assert plan["refusals"] == [
        {
            "code": "replay-record-incomplete",
            "kind": "record",
            "sha256": None,
            "reasons": ["prompt_omitted"],
        }
    ]


async def test_another_engine_and_absent_files_are_each_named(client: AsyncClient) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)

    def elsewhere(record: dict[str, Any]) -> None:
        record["output"]["engine"] = "comfyui"
        record["loras"] = [
            {
                "sha256": "7" * 64,
                "model_strength": 1.0,
                "clip_strength": 1.0,
                "enabled": True,
                "position": 0,
            }
        ]

    plan = await _plan(client, _resealed(content, elsewhere))

    assert _codes(plan) == ["replay-engine-differs", "replay-lora-missing"]
    assert plan["refusals"][1]["sha256"] == "7" * 64


async def test_a_lora_held_by_two_assets_is_refused(client: AsyncClient) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)
    with SessionLocal() as session:
        for name in ("First copy", "Second copy"):
            session.add(
                ModelAssetInstall(
                    name=name,
                    kind="lora",
                    local_path=f"neutral-{name}",
                    manifest_json={"sha256": "8" * 64},
                    active=True,
                )
            )
        session.commit()

    def with_lora(record: dict[str, Any]) -> None:
        record["loras"] = [
            {
                "sha256": "8" * 64,
                "model_strength": 1.0,
                "clip_strength": 1.0,
                "enabled": True,
                "position": 0,
            }
        ]

    plan = await _plan(client, _resealed(content, with_lora))

    assert _codes(plan) == ["replay-lora-ambiguous"]


async def test_the_plan_reads_the_same_files_the_check_reads(client: AsyncClient) -> None:
    response = await client.post(
        PLAN, content=b"not a record", headers={"content-type": "application/octet-stream"}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "output-recipe-unreadable"
    assert "not a record" not in response.text


def _twin(run: dict[str, Any], *, trusted: bool) -> None:
    """Another revision executing exactly the recorded graph, in a family of its own."""

    with SessionLocal() as session:
        recorded = session.get(WorkflowRevision, run["workflow_revision_id"])
        assert recorded is not None
        definition = session.get(WorkflowDefinition, recorded.workflow_id)
        assert definition is not None
        family = WorkflowFamily(name="A copy awaiting review")
        twin = WorkflowDefinition(
            family=family,
            variant_key="create",
            name="A copy awaiting review",
            operation=definition.operation,
        )
        session.add_all(
            [
                family,
                twin,
                WorkflowRevision(
                    definition=twin,
                    version=1,
                    engine=recorded.engine,
                    api_graph_json=recorded.api_graph_json,
                    input_schema_json=recorded.input_schema_json,
                    dependencies_json=recorded.dependencies_json,
                    dependency_contract_sha256=recorded.dependency_contract_sha256,
                    artifact_sha256=recorded.artifact_sha256,
                    trusted=trusted,
                ),
            ]
        )
        session.commit()


async def test_a_copy_that_cannot_run_is_not_a_second_choice(client: AsyncClient) -> None:
    profile_id, _ = _profile()
    run, content = await _recorded(client, profile_id)
    _twin(run, trusted=False)

    plan = await _plan(client, content)

    assert plan["ready"] is True, plan
    assert plan["resolved"]["workflow_revision_id"] == run["workflow_revision_id"]


async def test_an_empty_section_is_refused_even_when_the_record_does_not_say_so(
    client: AsyncClient,
) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)

    def without_model(record: dict[str, Any]) -> None:
        record["model"] = None

    def without_workflow(record: dict[str, Any]) -> None:
        record["workflow"] = None

    def unverified(record: dict[str, Any]) -> None:
        record["workflow"]["verified"] = False

    for change, reason in (
        (without_model, "model_files_not_recorded"),
        (without_workflow, "workflow_unavailable"),
        (unverified, "workflow_unverified"),
    ):
        plan = await _plan(client, _resealed(content, change))
        assert plan["ready"] is False, reason
        assert plan["refusals"][0] == {
            "code": "replay-record-incomplete",
            "kind": "record",
            "sha256": None,
            "reasons": [reason],
        }


async def test_a_repeated_or_absent_input_is_refused(client: AsyncClient) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)
    source = {"sha256": "5" * 64, "role": "source", "size_bytes": 10, "media_type": "image/png"}

    def repeated(record: dict[str, Any]) -> None:
        record["operation"] = "image_to_image"
        record["prompt"]["positive"] = edit_prompt_preamble() + record["prompt"]["positive"]
        record["inputs"] = [source, {**source, "role": "input"}]

    plan = await _plan(client, _resealed(content, repeated))

    unsupported = [item for item in plan["refusals"] if item["code"] == "replay-record-unsupported"]
    assert unsupported[0]["reasons"] == ["repeated_inputs"]
    absent = [item for item in plan["refusals"] if item["code"] == "replay-input-missing"]
    assert [item["sha256"] for item in absent] == ["5" * 64, "5" * 64]


_EMPTY_GRAPH_IDENTITY = workflow_artifact_contract(
    operation="text_to_image",
    engine="mock",
    api_graph={},
    input_schema={},
    dependencies={},
)


def _bound_revision(profile_id: str, binding: str) -> str:
    """A ready picture workflow whose activation binds it to one model."""

    with SessionLocal() as session:
        family = WorkflowFamily(name="Bound pictures")
        definition = WorkflowDefinition(
            family=family, variant_key="create", name="Bound pictures", operation="text_to_image"
        )
        revision = WorkflowRevision(
            definition=definition,
            version=1,
            engine="mock",
            api_graph_json={},
            input_schema_json={},
            dependencies_json={},
            dependency_contract_sha256="d" * 64,
            artifact_sha256=_EMPTY_GRAPH_IDENTITY,
            trusted=True,
        )
        session.add_all([family, definition, revision])
        session.flush()
        definition.current_revision_id = revision.id
        activation = WorkflowActivation(
            workflow_revision_id=revision.id,
            resolver_version="resolver-v1",
            dependency_contract_sha256="d" * 64,
            binding_sha256=binding,
            state="ready",
            is_active=True,
            details_json={"launch_sha256": "f" * 64},
        )
        slot = WorkflowDependencySlot(
            workflow_revision_id=revision.id,
            name="primary",
            resource_kind="model_profile",
            required=True,
            satisfaction="all_of",
            requirements_json=[{"key": "default", "constraints": {}}],
            contract_sha256="1" * 64,
            ordinal=0,
        )
        session.add_all([activation, slot])
        session.flush()
        session.add(
            WorkflowDependencyBinding(
                workflow_revision_id=revision.id,
                workflow_activation_id=activation.id,
                workflow_dependency_slot_id=slot.id,
                requirement_key="default",
                model_profile_id=profile_id,
                resource_identity_sha256="2" * 64,
            )
        )
        session.commit()
        return revision.id


async def test_a_bound_workflow_plans_its_own_model_only_under_the_recorded_binding(
    client: AsyncClient,
) -> None:
    profile_id, _ = _profile()
    other_profile, _ = _profile(name="Another holder of the same files")
    _, content = await _recorded(client, profile_id)
    revision_id = _bound_revision(other_profile, "e" * 64)

    def bound_to(binding: str) -> Any:
        def change(record: dict[str, Any]) -> None:
            record["workflow"] = {
                **record["workflow"],
                "artifact_sha256": _EMPTY_GRAPH_IDENTITY,
                "dependency_contract_sha256": "d" * 64,
                "binding_sha256": binding,
            }

        return change

    same = await _plan(client, _resealed(content, bound_to("e" * 64)))
    moved = await _plan(client, _resealed(content, bound_to("9" * 64)))

    # Two profiles hold the files; the activation's binding says which one runs.
    assert same["ready"] is True, same
    assert same["resolved"]["workflow_revision_id"] == revision_id
    assert same["resolved"]["profile_id"] == other_profile
    assert _codes(moved) == ["replay-workflow-binding-differs", "replay-model-ambiguous"]


async def test_a_lora_the_workflow_cannot_take_is_refused(client: AsyncClient) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)
    with SessionLocal() as session:
        session.add(
            ModelAssetInstall(
                name="Only copy",
                kind="lora",
                local_path="neutral-only-copy",
                manifest_json={"sha256": "8" * 64},
                active=True,
            )
        )
        session.commit()

    def with_lora(record: dict[str, Any]) -> None:
        record["loras"] = [
            {
                "sha256": "8" * 64,
                "model_strength": 1.0,
                "clip_strength": 1.0,
                "enabled": True,
                "position": 0,
            }
        ]

    plan = await _plan(client, _resealed(content, with_lora))

    # The test engine's workflow offers no place for an added LoRA.
    assert _codes(plan) == ["replay-lora-unusable"]
    assert plan["refusals"][0]["sha256"] is None
