"""Keep accepted recipes independent of mutable choices and exact to their revision."""

from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.workflow_use_case_preset_admission import AdmittedWorkflowUseCasePreset
from local_lm.workflow_use_case_preset_resolution import ResolvedWorkflowUseCasePreset
from local_lm.workflow_use_cases_v1 import WorkflowUseCase


def _admission() -> AdmittedWorkflowUseCasePreset:
    return AdmittedWorkflowUseCasePreset(
        "revision-original",
        ResolvedWorkflowUseCasePreset(
            WorkflowUseCase.IMAGE_EDIT,
            "preset",
            "project",
            "recipe-original",
            "Example recipe",
            {"denoise": 0.4, "options": {"sizes": [256]}},
        ),
        None,
    )


def test_recipe_receipt_roundtrip_detaches_settings_at_every_boundary() -> None:
    from local_lm.workflow_use_case_preset_provenance import (
        capture_workflow_use_case_preset,
        read_workflow_use_case_preset,
    )

    admission = _admission()
    receipt = capture_workflow_use_case_preset(admission)
    payload = receipt.model_dump(mode="json")
    restored = read_workflow_use_case_preset(payload, workflow_revision_id="revision-original")
    assert restored is not None
    assert restored.model_dump(mode="json") == payload
    original = deepcopy(payload)
    admission.preset.settings_json["options"]["sizes"].append(512)
    payload["settings_json"]["options"]["sizes"].append(1024)
    resolved = restored.resolved_preset()
    resolved.settings_json["options"]["sizes"].append(2048)
    assert receipt.model_dump(mode="json") == original
    assert restored.model_dump(mode="json") == original
    assert restored.resolved_preset().preset_id == "recipe-original"
    assert restored.resolved_preset().scope == "project"


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 2},
        {"version": True},
        {"mode": "inherit"},
        {"use_case": "unknown"},
        {"scope": None},
        {"preset_id": None},
        {"preset_id": " recipe-original"},
        {"preset_name": " "},
        {"workflow_revision_id": ""},
        {"workflow_revision_id": " revision-original"},
        {"upscale_kind": "model"},
        {"settings_json": {"denoise": float("nan")}},
        {"settings_json": {"options": {"value": float("inf")}}},
        {"settings_json": {"prompt": "Neutral fixture"}},
        {"settings_json": {"negative_prompt": "Neutral fixture"}},
        {"unexpected": "Neutral fixture"},
        {"mode": "automatic"},
        {"mode": "unconfigured"},
    ],
)
def test_malformed_recipe_receipts_fail_with_one_content_free_reason(
    changes: dict[str, Any],
) -> None:
    from local_lm.workflow_use_case_preset_provenance import (
        capture_workflow_use_case_preset,
        read_workflow_use_case_preset,
    )

    payload = capture_workflow_use_case_preset(_admission()).model_dump(mode="json")
    payload.update(changes)
    with pytest.raises(ValueError) as raised:
        read_workflow_use_case_preset(payload, workflow_revision_id="revision-original")
    assert str(raised.value) == "workflow-use-case-preset-snapshot-invalid"


@pytest.mark.parametrize("revision_id", [None, "another-revision"])
def test_a_recipe_receipt_cannot_move_to_another_selected_revision(
    revision_id: str | None,
) -> None:
    from local_lm.workflow_use_case_preset_provenance import (
        capture_workflow_use_case_preset,
        read_workflow_use_case_preset,
    )

    payload = capture_workflow_use_case_preset(_admission()).model_dump(mode="json")
    with pytest.raises(ValueError, match="^workflow-use-case-preset-snapshot-revision-mismatch$"):
        read_workflow_use_case_preset(payload, workflow_revision_id=revision_id)


def test_receipt_instances_are_revalidated_after_nested_mutation() -> None:
    from local_lm.workflow_use_case_preset_provenance import (
        capture_workflow_use_case_preset,
        read_workflow_use_case_preset,
    )

    receipt = capture_workflow_use_case_preset(_admission())
    receipt.settings_json["prompt"] = "Neutral fixture"
    with pytest.raises(ValueError, match="^workflow-use-case-preset-snapshot-invalid$"):
        read_workflow_use_case_preset(receipt, workflow_revision_id="revision-original")


@pytest.mark.parametrize("mode,scope", [("unconfigured", None), ("automatic", "chat")])
def test_empty_recipe_modes_keep_their_meaning_in_a_receipt(mode: str, scope: str | None) -> None:
    from local_lm.workflow_use_case_preset_provenance import (
        capture_workflow_use_case_preset,
        read_workflow_use_case_preset,
    )

    payload = capture_workflow_use_case_preset(_admission()).model_dump(mode="json")
    payload.update(mode=mode, scope=scope, preset_id=None, preset_name=None, settings_json={})
    restored = read_workflow_use_case_preset(payload, workflow_revision_id="revision-original")
    assert restored is not None
    assert restored.resolved_preset().mode == mode
    assert restored.resolved_preset().settings_json == {}
    assert read_workflow_use_case_preset(None, workflow_revision_id=None) is None


@pytest.mark.parametrize("kind", ["model", "resample"])
def test_upscale_receipt_preserves_the_admitted_execution_kind(kind: str) -> None:
    from local_lm.workflow_use_case_preset_provenance import (
        capture_workflow_use_case_preset,
        read_workflow_use_case_preset,
    )

    payload = capture_workflow_use_case_preset(_admission()).model_dump(mode="json")
    payload.update(use_case="image_upscale", upscale_kind=kind)
    restored = read_workflow_use_case_preset(payload, workflow_revision_id="revision-original")
    assert restored is not None
    assert restored.upscale_kind == kind


async def test_accepted_context_keeps_the_recipe_after_live_provenance_changes(
    app: FastAPI, client: AsyncClient
) -> None:
    from local_lm.accepted_turn_context import AcceptedContext, accepted_context
    from local_lm.db import SessionLocal
    from local_lm.models import Run, WorkflowDefinition, WorkflowRevision
    from local_lm.schemas import TurnRequest
    from local_lm.workflow_use_case_preset_provenance import capture_workflow_use_case_preset

    chat = (await client.post("/api/chats", json={"title": "Accepted recipe"})).json()
    orchestrator = app.state.services.orchestrator
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session, chat["id"], TurnRequest(text="A blue square", mode="image")
            )
            definition = WorkflowDefinition(name="Example recipe graph", operation="text_to_image")
            session.add(definition)
            session.flush()
            revision = WorkflowRevision(
                workflow_id=definition.id, version=1, engine="mock", trusted=True
            )
            session.add(revision)
            session.flush()
            run = session.get(Run, accepted.run.id)
            assert run is not None
            run.workflow_revision_id = revision.id
            recipe = capture_workflow_use_case_preset(
                AdmittedWorkflowUseCasePreset(
                    revision.id,
                    ResolvedWorkflowUseCasePreset(
                        WorkflowUseCase.IMAGE_GENERATION,
                        "preset",
                        "workspace",
                        "original-recipe",
                        "Original recipe",
                        {"quality": 0.7},
                    ),
                    None,
                )
            ).model_dump(mode="json")
            run.provenance_json = {**run.provenance_json, "workflow_use_case_preset": recipe}
            orchestrator._freeze_turn_context(session, run)
            session.flush()
            run.provenance_json = {
                **run.provenance_json,
                "workflow_use_case_preset": {"unexpected": "Later replacement"},
            }
            session.flush()
            snapshot = accepted_context(session, run)
            assert snapshot is not None
            assert snapshot.workflow_use_case_preset is not None
            assert snapshot.workflow_use_case_preset.model_dump(mode="json") == recipe
            payload = snapshot.model_dump(mode="json")
            payload["workflow_revision_id"] = "another-revision"
            with pytest.raises(
                ValueError, match="workflow-use-case-preset-snapshot-revision-mismatch"
            ):
                AcceptedContext.model_validate(payload)
            payload.pop("workflow_use_case_preset")
            assert AcceptedContext.model_validate(payload).workflow_use_case_preset is None
