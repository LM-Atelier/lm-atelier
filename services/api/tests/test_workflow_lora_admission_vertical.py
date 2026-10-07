from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from test_workflow_lora_execution import (
    PRIVATE_LOADER_ID,
    SeededWorkflow,
    _seed_workflow,
    _workflow_witness,
)

from local_lm import orchestrator as orchestrator_module
from local_lm import prompt_library
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.mock import MockMediaAdapter
from local_lm.auxiliary_assets import revision_accepts_added_loras
from local_lm.db import SessionLocal
from local_lm.domain import JobStatus, MessageStatus, Operation, RunStatus
from local_lm.engines import EngineRegistry
from local_lm.models import (
    Chat,
    Job,
    Message,
    Run,
    TurnCreationClaim,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.settings_registry import WORKFLOW_LORA_OVERRIDES_SETTING_KEY
from local_lm.workflow_lora_admission import (
    WORKFLOW_LORA_ADMISSION_CONFLICT_CODE,
    WORKFLOW_LORA_ADMISSION_CONFLICT_MESSAGE,
    WORKFLOW_LORA_ADMISSION_INVALID_CODE,
    WORKFLOW_LORA_ADMISSION_INVALID_MESSAGE,
    WORKFLOW_LORA_REPLAY_CONFLICT_CODE,
    WORKFLOW_LORA_REPLAY_CONFLICT_MESSAGE,
    WorkflowLoraAdmissionError,
    WorkflowLoraAdmissionLayers,
    admit_workflow_lora_composition,
    split_workflow_lora_admission_layers,
    workflow_lora_admission_provenance,
)
from local_lm.workflow_lora_overrides import workflow_lora_override_resolution_sha256
from local_lm.workflow_loras import workflow_lora_controls
from local_lm.workflow_selection import resolve_exact_workflow_revision


def _override_envelope(
    session: Session,
    seeded: SeededWorkflow,
    *,
    changes: dict[str, bool | float],
    witness_updates: dict[str, object] | None = None,
) -> dict[str, object]:
    projection = workflow_lora_controls(
        session,
        revision_id=seeded.revision_id,
    )
    assert projection.override_target is not None
    assert projection.slots
    witness = {**asdict(projection.override_target), **(witness_updates or {})}
    return {
        "version": 1,
        "targets": [
            {
                **witness,
                "overrides": [
                    {
                        "slot_id": projection.slots[0].slot_id,
                        "loader_contract": projection.slots[0].loader_contract,
                        "loader_authority_sha256": projection.slots[0].loader_authority_sha256,
                        "changes": changes,
                    }
                ],
            }
        ],
    }


ORIGINS = (
    "profile_request",
    "default_preset",
    "project_preset",
    "project",
    "chat_preset",
    "chat",
    "turn_preset",
    "turn",
)
MALFORMED_ENVELOPE: dict[str, object] = {"version": 1, "targets": "not a list"}
RESET_ENVELOPE: dict[str, object] = {"version": 1, "targets": []}


def _split_layers(**envelopes: object) -> WorkflowLoraAdmissionLayers:
    raw: dict[str, dict[str, object]] = {name: {"steps": 7} for name in ORIGINS}
    for origin, envelope in envelopes.items():
        raw[origin][WORKFLOW_LORA_OVERRIDES_SETTING_KEY] = envelope
    return split_workflow_lora_admission_layers(role="image", **raw)


def _plain_revision(session: Session, seeded: SeededWorkflow) -> SeededWorkflow:
    """Add a copy of the seeded revision that declares no dependency contract."""

    source = session.get(WorkflowRevision, seeded.revision_id)
    assert source is not None
    plain = WorkflowRevision(
        id=f"{seeded.revision_id}_plain",
        workflow_id=source.workflow_id,
        version=source.version + 1,
        engine=source.engine,
        ui_graph_json=deepcopy(source.ui_graph_json),
        api_graph_json=deepcopy(source.api_graph_json),
        input_schema_json=deepcopy(source.input_schema_json),
        dependencies_json=deepcopy(source.dependencies_json),
        dependency_contract_sha256=None,
        trusted=True,
    )
    session.add(plain)
    session.commit()
    return replace(seeded, revision_id=plain.id)


def _undeclare_lora_setting(session: Session, seeded: SeededWorkflow) -> None:
    """Say nothing about LoRAs, in a graph that still says where one goes.

    Both the declared setting and the recorded extension point go, because the
    workflow contract requires those two together and refuses a revision that
    has one without the other. What is left is the state the contract passes
    over in silence: nothing declared, nothing recorded, and a graph a LoRA
    can still be read out of. The revision is edited rather than copied so it
    keeps the activation and the dependency contract a turn checks first.
    """

    revision = session.get(WorkflowRevision, seeded.revision_id)
    assert revision is not None
    schema = deepcopy(revision.input_schema_json)
    properties = schema.get("properties")
    assert isinstance(properties, dict)
    assert properties.pop("loras", None) is not None
    revision.input_schema_json = schema
    dependencies = deepcopy(revision.dependencies_json)
    extensions = dependencies.get("extensions")
    assert isinstance(extensions, dict)
    assert extensions.pop("lora", None) is not None
    revision.dependencies_json = dependencies
    session.commit()


def _store_chat_setting(chat_id: str, envelope: object) -> None:
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.generation_settings_json = {"image": {WORKFLOW_LORA_OVERRIDES_SETTING_KEY: envelope}}
        session.commit()


def _force_seeded_workflow(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    seeded: SeededWorkflow,
) -> list[MediaRequest]:
    orchestrator: ConversationOrchestrator = app.state.services.orchestrator
    original_selector = orchestrator._profile_and_workflow_for_operation

    def forced_selector(
        session: Session,
        chat: Chat,
        operation: Operation,
        selection_text: str,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any], WorkflowRevision | None]:
        if operation == Operation.TEXT:
            return original_selector(session, chat, operation, selection_text, *args, **kwargs)
        revision = session.get(WorkflowRevision, seeded.revision_id)
        assert revision is not None
        return (
            None,
            {
                "mode": "revision",
                "workflow_definition_id": seeded.definition_id,
                "workflow_revision_id": seeded.revision_id,
            },
            revision,
        )

    monkeypatch.setattr(orchestrator, "_profile_and_workflow_for_operation", forced_selector)
    engines: EngineRegistry = app.state.services.engines
    original_settings = engines.settings_for_role

    async def forced_settings(role: str, *, engine: str | None = None) -> list[Any]:
        del engine
        return await original_settings(role, engine="mock")

    monkeypatch.setattr(app.state.services.engines, "settings_for_role", forced_settings)
    captured: list[MediaRequest] = []
    original_generate = MockMediaAdapter.generate

    async def capture_generate(
        self: MockMediaAdapter,
        request: MediaRequest,
    ) -> AsyncIterator[MediaEvent]:
        captured.append(deepcopy(request))
        async for event in original_generate(self, request):
            yield event

    monkeypatch.setattr(MockMediaAdapter, "generate", capture_generate)
    return captured


async def _wait_for_plan(client: AsyncClient, plan_id: str) -> dict[str, Any]:
    async def read_plan() -> dict[str, Any]:
        plan: dict[str, Any] = (await client.get(f"/api/work-plans/{plan_id}")).json()
        return plan

    # This plan may end in states the shared terminal set does not name, so it
    # waits on its own predicate rather than on that set.
    return await wait_until(
        read_plan,
        lambda plan: (
            plan["status"] in {"complete", "failed", "cancelled", "interrupted", "blocked"}
        ),
        what=f"work plan {plan_id}",
    )


def _assert_no_turn_rows(chat_id: str, *, title: str) -> None:
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        assert chat.active_head_message_id is None
        assert chat.title == title
        assert session.scalar(select(func.count(Message.id)).where(Message.chat_id == chat_id)) == 0
        assert (
            session.scalar(select(func.count(WorkPlan.id)).where(WorkPlan.chat_id == chat_id)) == 0
        )
        assert session.scalar(select(func.count(Run.id)).where(Run.chat_id == chat_id)) == 0
        assert (
            session.scalar(
                select(func.count(WorkStep.id))
                .join(WorkPlan, WorkPlan.id == WorkStep.plan_id)
                .where(WorkPlan.chat_id == chat_id)
            )
            == 0
        )
        assert (
            session.scalar(
                select(func.count(Job.id))
                .join(Run, Run.id == Job.run_id)
                .where(Run.chat_id == chat_id)
            )
            == 0
        )
        assert (
            session.scalar(
                select(func.count(TurnCreationClaim.id)).where(TurnCreationClaim.chat_id == chat_id)
            )
            == 0
        )


async def test_admission_resolves_each_of_the_eight_origins_with_its_origin(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_origins")
        revision = session.get(WorkflowRevision, seeded.revision_id)
        assert revision is not None
        activation = _workflow_witness(session, seeded)["activation"]
        for index, origin in enumerate(ORIGINS, start=1):
            envelope = _override_envelope(session, seeded, changes={"model_strength": 0.1 * index})
            layers = _split_layers(**{origin: envelope})
            assert layers.relevant_to(revision)
            assert layers.ordinary(origin) == {"steps": 7}
            admitted = admit_workflow_lora_composition(
                session,
                revision,
                activation_snapshot=activation,
                layers=layers,
                added_loras=[],
            )
            assert admitted.resolution.overrides[0].changes[0].origin == origin
            assert admitted.resolution.overrides[0].changes[0].value == pytest.approx(0.1 * index)
            receipt = workflow_lora_admission_provenance(admitted)
            assert receipt["override_resolution_sha256"] == (
                workflow_lora_override_resolution_sha256(admitted.resolution)
            )
            encoded = json.dumps(receipt, sort_keys=True)
            assert PRIVATE_LOADER_ID not in encoded
            assert "C:/private" not in encoded


@pytest.mark.parametrize("lower", ORIGINS[:-1])
def test_a_higher_reset_hides_corrupt_lower_state_without_reading_it(lower: str) -> None:
    higher = ORIGINS[ORIGINS.index(lower) + 1]

    layers = _split_layers(**{lower: MALFORMED_ENVELOPE, higher: RESET_ENVELOPE})

    assert layers.override_layers == ()
    assert layers.ordinary(lower) == {"steps": 7}
    assert layers.ordinary(higher) == {"steps": 7}


@pytest.mark.parametrize("origin", ORIGINS)
def test_corrupt_saved_layers_are_stale_and_a_corrupt_turn_is_invalid(origin: str) -> None:
    with pytest.raises(WorkflowLoraAdmissionError) as refused:
        _split_layers(**{origin: MALFORMED_ENVELOPE})

    assert refused.value.conflict is (origin != "turn")
    assert refused.value.code == (
        WORKFLOW_LORA_ADMISSION_INVALID_CODE
        if origin == "turn"
        else WORKFLOW_LORA_ADMISSION_CONFLICT_CODE
    )


async def test_a_reset_hides_only_layers_below_it(client: AsyncClient) -> None:
    del client
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_reset_scope")
        revision = session.get(WorkflowRevision, seeded.revision_id)
        assert revision is not None
        project = _override_envelope(session, seeded, changes={"clip_strength": 0.2})
        chat = _override_envelope(session, seeded, changes={"model_strength": 0.4})
        layers = _split_layers(project=project, chat_preset=RESET_ENVELOPE, chat=chat)
        admitted = admit_workflow_lora_composition(
            session,
            revision,
            activation_snapshot=_workflow_witness(session, seeded)["activation"],
            layers=layers,
            added_loras=[],
        )

    assert [
        (change.field, change.origin) for change in admitted.resolution.overrides[0].changes
    ] == [("model_strength", "chat")]


async def test_only_a_target_naming_the_selected_revision_needs_admission(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_relevance")
        revision = session.get(WorkflowRevision, seeded.revision_id)
        assert revision is not None
        current = _override_envelope(session, seeded, changes={"model_strength": 0.4})
        other_revision = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.4},
            witness_updates={"workflow_revision_id": "wfrev_somewhere_else"},
        )
        other_workflow = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.4},
            witness_updates={"workflow_definition_id": "workflow_somewhere_else"},
        )

        assert not _split_layers().relevant_to(revision)
        assert not _split_layers(chat=RESET_ENVELOPE).relevant_to(revision)
        assert not _split_layers(chat=other_revision).relevant_to(revision)
        assert not _split_layers(chat=other_workflow).relevant_to(revision)
        assert not _split_layers(chat=current).relevant_to(None)
        assert _split_layers(project=other_revision, chat=current).relevant_to(revision)
        with pytest.raises(WorkflowLoraAdmissionError):
            admit_workflow_lora_composition(
                session,
                revision,
                activation_snapshot=_workflow_witness(session, seeded)["activation"],
                layers=_split_layers(chat=other_revision),
                added_loras=[],
            )


async def test_a_workflow_that_takes_loras_without_declaring_them_accepts_a_stack(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The turn accepts what the settings panel offers for such a workflow.

    The panel decides whether to show a LoRA control from the insertion point
    rather than from the schema, because the insertion point is what the run
    reads. A turn's settings are checked against the fields the same function
    builds, and a value with no field behind it is refused as an unsupported
    setting. One workflow therefore cannot answer differently in the two
    places without the control becoming worse than none.
    """

    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_undeclared")
        _undeclare_lora_setting(session, seeded)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, seeded.revision_id)
        assert revision is not None
        # Both halves of the case, stated rather than assumed: nothing declares
        # the setting, and the run still finds somewhere to put a LoRA.
        assert "loras" not in revision.input_schema_json.get("properties", {})
        assert revision_accepts_added_loras(revision)

    captured = _force_seeded_workflow(app, monkeypatch, seeded)
    chat = (await client.post("/api/chats", json={"title": "Undeclared LoRA stack"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create one careful study",
            "mode": "image",
            "settings": {
                "loras": [
                    {
                        "asset_id": seeded.added_asset_id,
                        "model_strength": 0.62,
                        "clip_strength": 0.51,
                        "enabled": True,
                    }
                ],
            },
        },
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    plan = await _wait_for_plan(client, accepted["run"]["work_plan_id"])
    assert plan["status"] == "complete"
    assert len(captured) == 1

    with SessionLocal() as session:
        run = session.get(Run, accepted["run"]["id"])
        assert run is not None
        # Accepted is not the same as applied, so the stack is read back from
        # the stored settings rather than inferred from the status code.
        stored = run.settings_json["loras"]
        assert [item["asset_id"] for item in stored] == [seeded.added_asset_id]
        assert stored[0]["model_strength"] == 0.62


async def test_ordinary_admission_persists_and_dispatches_native_plus_added(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_ordinary")
        envelope = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.21, "clip_strength": 0.43},
        )
    captured = _force_seeded_workflow(app, monkeypatch, seeded)
    title = "Atomic native admission"
    chat = (await client.post("/api/chats", json={"title": title})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create two careful studies",
            "mode": "image",
            "output_count": 2,
            "settings": {
                WORKFLOW_LORA_OVERRIDES_SETTING_KEY: envelope,
                "loras": [
                    {
                        "asset_id": seeded.added_asset_id,
                        "model_strength": 0.62,
                        "clip_strength": 0.51,
                        "enabled": True,
                    }
                ],
            },
        },
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    plan = await _wait_for_plan(client, accepted["run"]["work_plan_id"])
    assert plan["status"] == "complete"
    assert len(captured) == 2

    with SessionLocal() as session:
        runs = list(
            session.scalars(
                select(Run)
                .where(Run.work_plan_id == accepted["run"]["work_plan_id"])
                .order_by(Run.created_at, Run.id)
            ).all()
        )
        steps = list(
            session.scalars(
                select(WorkStep)
                .where(WorkStep.plan_id == accepted["run"]["work_plan_id"])
                .order_by(WorkStep.ordinal)
            ).all()
        )
        assert len(runs) == len(steps) == 2
        for run, step in zip(runs, steps, strict=True):
            assert (
                run.settings_json[WORKFLOW_LORA_OVERRIDES_SETTING_KEY]
                == (step.settings_json[WORKFLOW_LORA_OVERRIDES_SETTING_KEY])
            )
            assert len(run.settings_json[WORKFLOW_LORA_OVERRIDES_SETTING_KEY]["targets"]) == 1
            receipt = run.provenance_json["workflow_lora"]
            assert set(receipt) == {
                "version",
                "override_resolution",
                "override_resolution_sha256",
                "composition",
                "composition_sha256",
            }
            canonical = json.dumps(
                receipt["composition"],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            assert receipt["composition_sha256"] == hashlib.sha256(canonical).hexdigest()
            assert run.provenance_json["workflow"]["activation"]["id"] == seeded.activation_id
            assert (
                run.provenance_json["auxiliary_assets"]["effective_graph_sha256"]
                == (receipt["composition"]["effective_graph_sha256"])
            )
            public = json.dumps(
                {
                    "settings": run.settings_json,
                    "provenance": run.provenance_json,
                },
                sort_keys=True,
            )
            assert PRIVATE_LOADER_ID not in public
            assert "C:/private" not in public

    for dispatched in captured:
        assert WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in dispatched.parameters
        assert dispatched.parameters["loras"][0]["asset_id"] == seeded.added_asset_id
        native = dispatched.workflow[PRIVATE_LOADER_ID]
        assert native["inputs"]["strength_model"] == 0.21
        assert native["inputs"]["strength_clip"] == 0.43
        assert "lma_lora_001" in dispatched.workflow
        added = dispatched.workflow["lma_lora_001"]
        assert native["inputs"]["model"] == ["lma_lora_001", 0]
        assert native["inputs"]["clip"] == ["lma_lora_001", 1]
        assert added["inputs"]["model"] == ["source", 0]
        assert added["inputs"]["clip"] == ["source", 1]


async def test_ordinary_video_admission_dispatches_the_recorded_composition(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_ordinary_video")
        envelope = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.27, "clip_strength": 0.38},
        )
    captured = _force_seeded_workflow(app, monkeypatch, seeded)
    chat = (await client.post("/api/chats", json={"title": "Native video"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create a short video study",
            "mode": "video",
            "settings": {
                WORKFLOW_LORA_OVERRIDES_SETTING_KEY: envelope,
                "loras": [
                    {
                        "asset_id": seeded.added_asset_id,
                        "model_strength": 0.49,
                        "clip_strength": 0.58,
                        "enabled": True,
                    }
                ],
            },
        },
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    plan = await _wait_for_plan(client, accepted["run"]["work_plan_id"])
    assert plan["status"] == "complete"
    assert len(captured) == 1
    dispatched = captured[0]
    assert dispatched.operation == Operation.TEXT_TO_VIDEO.value
    assert WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in dispatched.parameters
    assert dispatched.workflow[PRIVATE_LOADER_ID]["inputs"]["strength_model"] == 0.27
    assert dispatched.workflow[PRIVATE_LOADER_ID]["inputs"]["strength_clip"] == 0.38
    assert dispatched.workflow[PRIVATE_LOADER_ID]["inputs"]["model"] == ["lma_lora_001", 0]
    assert dispatched.workflow[PRIVATE_LOADER_ID]["inputs"]["clip"] == ["lma_lora_001", 1]
    assert dispatched.workflow["lma_lora_001"]["inputs"]["model"] == ["source", 0]
    assert dispatched.workflow["lma_lora_001"]["inputs"]["clip"] == ["source", 1]
    with SessionLocal() as session:
        run = session.get(Run, accepted["run"]["id"])
        assert run is not None and run.work_step_id is not None
        step = session.get(WorkStep, run.work_step_id)
        assert step is not None
        assert (
            run.settings_json[WORKFLOW_LORA_OVERRIDES_SETTING_KEY]
            == step.settings_json[WORKFLOW_LORA_OVERRIDES_SETTING_KEY]
        )
        assert run.provenance_json["workflow"]["activation"]["id"] == seeded.activation_id
        assert (
            run.provenance_json["workflow_lora"]["composition"]["effective_graph_sha256"]
            == run.provenance_json["auxiliary_assets"]["effective_graph_sha256"]
        )


async def test_ordered_image_video_admission_dispatches_each_role_before_any_added_lora(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_ordered_success")
        image_envelope = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.22, "clip_strength": 0.33},
        )
        video_envelope = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.66, "clip_strength": 0.77},
        )
    captured = _force_seeded_workflow(app, monkeypatch, seeded)
    chat = (await client.post("/api/chats", json={"title": "Ordered native roles"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of a paper boat, then animate the image into a video",
            "mode": "auto",
            "confirm_media": True,
            "ordered_settings": {
                "image": {
                    WORKFLOW_LORA_OVERRIDES_SETTING_KEY: image_envelope,
                    "loras": [
                        {
                            "asset_id": seeded.added_asset_id,
                            "model_strength": 0.41,
                            "clip_strength": 0.42,
                            "enabled": True,
                        }
                    ],
                },
                "video": {
                    WORKFLOW_LORA_OVERRIDES_SETTING_KEY: video_envelope,
                    "loras": [
                        {
                            "asset_id": seeded.added_asset_id,
                            "model_strength": 0.81,
                            "clip_strength": 0.82,
                            "enabled": True,
                        }
                    ],
                },
            },
        },
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    plan = await _wait_for_plan(client, accepted["run"]["work_plan_id"])
    assert plan["status"] == "complete"
    media = [item for item in captured if item.operation != Operation.TEXT.value]
    assert [item.operation for item in media] == [
        Operation.TEXT_TO_IMAGE.value,
        Operation.IMAGE_TO_VIDEO.value,
    ]
    expected = (
        (media[0], 0.22, 0.33, 0.41, 0.42),
        (media[1], 0.66, 0.77, 0.81, 0.82),
    )
    for dispatched, model, clip, added_model, added_clip in expected:
        assert WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in dispatched.parameters
        assert dispatched.workflow[PRIVATE_LOADER_ID]["inputs"]["strength_model"] == model
        assert dispatched.workflow[PRIVATE_LOADER_ID]["inputs"]["strength_clip"] == clip
        added = dispatched.workflow["lma_lora_001"]
        assert dispatched.workflow[PRIVATE_LOADER_ID]["inputs"]["model"] == [
            "lma_lora_001",
            0,
        ]
        assert dispatched.workflow[PRIVATE_LOADER_ID]["inputs"]["clip"] == [
            "lma_lora_001",
            1,
        ]
        assert added["inputs"]["model"] == ["source", 0]
        assert added["inputs"]["clip"] == ["source", 1]
        assert added["inputs"]["strength_model"] == added_model
        assert added["inputs"]["strength_clip"] == added_clip

    with SessionLocal() as session:
        steps = list(
            session.scalars(
                select(WorkStep)
                .where(WorkStep.plan_id == accepted["run"]["work_plan_id"])
                .order_by(WorkStep.ordinal)
            ).all()
        )
        runs = [session.get(Run, step.run_id) for step in steps]
        assert len(steps) == len(runs) == 2
        for step, run in zip(steps, runs, strict=True):
            assert run is not None
            assert (
                step.settings_json[WORKFLOW_LORA_OVERRIDES_SETTING_KEY]
                == run.settings_json[WORKFLOW_LORA_OVERRIDES_SETTING_KEY]
            )
            assert run.provenance_json["workflow"]["activation"]["id"] == seeded.activation_id
            assert set(run.provenance_json["workflow_lora"]) == {
                "version",
                "override_resolution",
                "override_resolution_sha256",
                "composition",
                "composition_sha256",
            }


async def test_ordered_final_step_stale_admission_leaves_no_rows_or_calls(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_ordered_refusal")
        stale = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.2},
            witness_updates={"api_graph_sha256": "f" * 64},
        )
    captured = _force_seeded_workflow(app, monkeypatch, seeded)
    title = "Ordered refusal stays empty"
    chat = (await client.post("/api/chats", json={"title": title})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Write a short scene, then create an image based on it",
            "mode": "auto",
            "confirm_media": True,
            "idempotency_key": "ordered-native-stale",
            "ordered_settings": {
                "image": {
                    WORKFLOW_LORA_OVERRIDES_SETTING_KEY: stale,
                    "loras": [],
                }
            },
        },
    )
    assert response.status_code == 409
    assert response.json() == {
        "code": WORKFLOW_LORA_ADMISSION_CONFLICT_CODE,
        "detail": WORKFLOW_LORA_ADMISSION_CONFLICT_MESSAGE,
    }
    assert captured == []
    _assert_no_turn_rows(chat["id"], title=title)


async def test_invalid_admission_and_retry_replay_errors_are_fixed_and_non_echoing(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_fixed_errors")
        valid = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.31},
        )
    captured = _force_seeded_workflow(app, monkeypatch, seeded)

    invalid_title = "Invalid native setting"
    invalid_chat = (await client.post("/api/chats", json={"title": invalid_title})).json()
    invalid = await client.post(
        f"/api/chats/{invalid_chat['id']}/turns",
        json={
            "text": "Create one study",
            "mode": "image",
            "idempotency_key": "invalid-native-setting",
            "settings": {
                WORKFLOW_LORA_OVERRIDES_SETTING_KEY: {
                    "version": 1,
                    "targets": "this hostile detail must never echo",
                }
            },
        },
    )
    assert invalid.status_code == 422
    assert invalid.json() == {
        "code": WORKFLOW_LORA_ADMISSION_INVALID_CODE,
        "detail": WORKFLOW_LORA_ADMISSION_INVALID_MESSAGE,
    }
    assert "hostile" not in invalid.text
    _assert_no_turn_rows(invalid_chat["id"], title=invalid_title)

    retry_chat = (await client.post("/api/chats", json={"title": "Retry preflight"})).json()
    accepted = await client.post(
        f"/api/chats/{retry_chat['id']}/turns",
        json={
            "text": "Create one retryable study",
            "mode": "image",
            "settings": {
                WORKFLOW_LORA_OVERRIDES_SETTING_KEY: valid,
                "loras": [],
            },
        },
    )
    assert accepted.status_code == 202, accepted.text
    plan = await _wait_for_plan(client, accepted.json()["run"]["work_plan_id"])
    assert plan["status"] == "complete"
    with SessionLocal() as session:
        run = session.get(Run, accepted.json()["run"]["id"])
        assert run is not None
        job = session.scalar(select(Job).where(Job.run_id == run.id))
        message = session.get(Message, run.assistant_message_id)
        assert job is not None and message is not None
        receipt = deepcopy(run.provenance_json["workflow_lora"])
        receipt["composition_sha256"] = "0" * 64
        run.provenance_json = {**run.provenance_json, "workflow_lora": receipt}
        run.status = RunStatus.FAILED.value
        run.error = "original run failure"
        job.status = JobStatus.FAILED.value
        job.error = "original job failure"
        message.status = MessageStatus.FAILED.value
        session.commit()
        job_id = job.id
        assistant_id = message.id
    calls_before_retry = len(captured)
    retried = await client.post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 409
    assert retried.json() == {
        "code": WORKFLOW_LORA_REPLAY_CONFLICT_CODE,
        "detail": WORKFLOW_LORA_REPLAY_CONFLICT_MESSAGE,
    }
    assert len(captured) == calls_before_retry
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        message = session.get(Message, assistant_id)
        assert job is not None and message is not None
        assert job.status == JobStatus.FAILED.value
        assert job.error == "original job failure"
        assert message.status == MessageStatus.FAILED.value


async def test_regenerate_and_branch_preserve_reset_or_suppress_native_settings_exactly(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_inheritance")
        envelope = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.35, "clip_strength": 0.46},
        )
    captured = _force_seeded_workflow(app, monkeypatch, seeded)
    chat = (await client.post("/api/chats", json={"title": "Native inheritance"})).json()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create the source image",
            "mode": "image",
            "settings": {
                WORKFLOW_LORA_OVERRIDES_SETTING_KEY: envelope,
                "loras": [],
            },
        },
    )
    assert original.status_code == 202, original.text
    source = original.json()
    assert (await _wait_for_plan(client, source["run"]["work_plan_id"]))["status"] == "complete"
    canonical = source["run"]["settings_json"][WORKFLOW_LORA_OVERRIDES_SETTING_KEY]

    regenerated = await client.post(
        f"/api/messages/{source['assistant_message']['id']}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202, regenerated.text
    regenerated_run = regenerated.json()["run"]
    assert regenerated_run["settings_json"][WORKFLOW_LORA_OVERRIDES_SETTING_KEY] == canonical
    assert "workflow_lora" in regenerated_run["provenance_json"]
    assert (await _wait_for_plan(client, regenerated_run["work_plan_id"]))["status"] == "complete"

    inherited_branch = await client.post(
        f"/api/messages/{source['user_message']['id']}/branch",
        json={"text": "Create an inherited branch", "mode": "image"},
    )
    assert inherited_branch.status_code == 202, inherited_branch.text
    inherited_run = inherited_branch.json()["run"]
    assert inherited_run["settings_json"][WORKFLOW_LORA_OVERRIDES_SETTING_KEY] == canonical
    assert "workflow_lora" in inherited_run["provenance_json"]
    assert (await _wait_for_plan(client, inherited_run["work_plan_id"]))["status"] == "complete"

    noninherited_branch = await client.post(
        f"/api/messages/{source['user_message']['id']}/branch",
        json={"text": "Create a clean same-role branch", "mode": "image", "settings": {}},
    )
    assert noninherited_branch.status_code == 202, noninherited_branch.text
    noninherited_run = noninherited_branch.json()["run"]
    assert WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in noninherited_run["settings_json"]
    assert "workflow_lora" not in noninherited_run["provenance_json"]
    assert (await _wait_for_plan(client, noninherited_run["work_plan_id"]))["status"] == "complete"

    changed_mode_branch = await client.post(
        f"/api/messages/{source['user_message']['id']}/branch",
        json={"text": "Create a clean video branch", "mode": "video"},
    )
    assert changed_mode_branch.status_code == 202, changed_mode_branch.text
    changed_mode_run = changed_mode_branch.json()["run"]
    assert WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in changed_mode_run["settings_json"]
    assert "workflow_lora" not in changed_mode_run["provenance_json"]
    assert (await _wait_for_plan(client, changed_mode_run["work_plan_id"]))["status"] == "complete"

    captured.clear()
    reset = await client.post(
        f"/api/messages/{source['assistant_message']['id']}/regenerate",
        json={"settings": {WORKFLOW_LORA_OVERRIDES_SETTING_KEY: RESET_ENVELOPE}},
    )
    assert reset.status_code == 202, reset.text
    reset_run = reset.json()["run"]
    assert WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in reset_run["settings_json"]
    assert "workflow_lora" not in reset_run["provenance_json"]
    assert (await _wait_for_plan(client, reset_run["work_plan_id"]))["status"] == "complete"
    assert [
        request.workflow[PRIVATE_LOADER_ID]["inputs"]["strength_model"] for request in captured
    ] == [0.75]


async def test_plan_retry_preflights_every_receipt_before_any_mutation_or_start(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_plan_retry")
        envelope = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.29},
        )
    captured = _force_seeded_workflow(app, monkeypatch, seeded)
    chat = (await client.post("/api/chats", json={"title": "Atomic plan retry"})).json()
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create two retry candidates",
            "mode": "image",
            "output_count": 2,
            "settings": {
                WORKFLOW_LORA_OVERRIDES_SETTING_KEY: envelope,
                "loras": [],
            },
        },
    )
    assert accepted.status_code == 202, accepted.text
    plan_id = accepted.json()["run"]["work_plan_id"]
    assert (await _wait_for_plan(client, plan_id))["status"] == "complete"
    calls_before_retry = len(captured)
    assert calls_before_retry == 2

    snapshots: dict[str, tuple[str, str | None, str, str | None, str, str | None]] = {}
    with SessionLocal() as session:
        plan = session.get(WorkPlan, plan_id)
        assert plan is not None
        plan.status = JobStatus.FAILED.value
        jobs = list(
            session.scalars(
                select(Job).where(Job.work_plan_id == plan_id).order_by(Job.created_at, Job.id)
            ).all()
        )
        assert len(jobs) == 2
        for index, job in enumerate(jobs):
            assert job.run_id is not None and job.work_step_id is not None
            run = session.get(Run, job.run_id)
            step = session.get(WorkStep, job.work_step_id)
            assert run is not None and step is not None
            message = session.get(Message, run.assistant_message_id)
            assert message is not None
            job.status = JobStatus.FAILED.value
            job.error = f"job failure {index}"
            run.status = RunStatus.FAILED.value
            run.error = f"run failure {index}"
            step.status = JobStatus.FAILED.value
            step.error = f"step failure {index}"
            message.status = MessageStatus.FAILED.value
            snapshots[job.id] = (
                job.status,
                job.error,
                run.status,
                run.error,
                step.status,
                step.error,
            )
        second_run = session.get(Run, jobs[1].run_id)
        assert second_run is not None
        receipt = deepcopy(second_run.provenance_json["workflow_lora"])
        receipt["composition_sha256"] = "0" * 64
        second_run.provenance_json = {
            **second_run.provenance_json,
            "workflow_lora": receipt,
        }
        session.commit()

    retried = await client.post(f"/api/work-plans/{plan_id}/retry")
    assert retried.status_code == 409
    assert retried.json() == {
        "code": WORKFLOW_LORA_REPLAY_CONFLICT_CODE,
        "detail": WORKFLOW_LORA_REPLAY_CONFLICT_MESSAGE,
    }
    assert len(captured) == calls_before_retry
    with SessionLocal() as session:
        jobs = list(
            session.scalars(
                select(Job).where(Job.work_plan_id == plan_id).order_by(Job.created_at, Job.id)
            ).all()
        )
        for job in jobs:
            assert job.run_id is not None and job.work_step_id is not None
            run = session.get(Run, job.run_id)
            step = session.get(WorkStep, job.work_step_id)
            assert run is not None and step is not None
            assert (
                job.status,
                job.error,
                run.status,
                run.error,
                step.status,
                step.error,
            ) == snapshots[job.id]


@pytest.mark.parametrize("saved", ["reset", "other_revision", "other_workflow"])
async def test_saved_edits_for_nothing_selected_leave_a_plain_workflow_turn_alone(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    saved: str,
) -> None:
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix=f"admission_inert_{saved}")
        envelope: object = (
            RESET_ENVELOPE
            if saved == "reset"
            else _override_envelope(
                session,
                seeded,
                changes={"model_strength": 0.4},
                witness_updates=(
                    {"workflow_revision_id": "wfrev_somewhere_else"}
                    if saved == "other_revision"
                    else {"workflow_definition_id": "workflow_somewhere_else"}
                ),
            )
        )
        plain = _plain_revision(session, seeded)
    captured = _force_seeded_workflow(app, monkeypatch, plain)
    chat = (await client.post("/api/chats", json={"title": "Inert saved edits"})).json()
    _store_chat_setting(chat["id"], envelope)

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create one plain study", "mode": "image"},
    )

    assert response.status_code == 202, response.text
    run = response.json()["run"]
    assert WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in run["settings_json"]
    assert "workflow_lora" not in run["provenance_json"]
    assert (await _wait_for_plan(client, run["work_plan_id"]))["status"] == "complete"
    assert len(captured) == 1
    assert captured[0].workflow[PRIVATE_LOADER_ID]["inputs"]["strength_model"] == 0.75
    assert WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in captured[0].parameters


async def test_an_edit_claiming_a_plain_selected_revision_is_refused(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="admission_plain_claim")
        plain = _plain_revision(session, seeded)
        envelope = _override_envelope(
            session,
            seeded,
            changes={"model_strength": 0.4},
            witness_updates={"workflow_revision_id": plain.revision_id},
        )
    _force_seeded_workflow(app, monkeypatch, plain)
    title = "Plain revision claim"
    chat = (await client.post("/api/chats", json={"title": title})).json()

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create one study",
            "mode": "image",
            "settings": {WORKFLOW_LORA_OVERRIDES_SETTING_KEY: envelope},
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "code": WORKFLOW_LORA_ADMISSION_CONFLICT_CODE,
        "detail": WORKFLOW_LORA_ADMISSION_CONFLICT_MESSAGE,
    }
    _assert_no_turn_rows(chat["id"], title=title)


@pytest.mark.parametrize("turn_reset", [False, True])
async def test_corrupt_saved_chat_edits_are_stale_unless_the_turn_resets_them(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    turn_reset: bool,
) -> None:
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix=f"admission_corrupt_{int(turn_reset)}")
    _force_seeded_workflow(app, monkeypatch, seeded)
    title = "Corrupt saved edits"
    chat = (await client.post("/api/chats", json={"title": title})).json()
    _store_chat_setting(chat["id"], MALFORMED_ENVELOPE)

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create one study",
            "mode": "image",
            "settings": (
                {WORKFLOW_LORA_OVERRIDES_SETTING_KEY: RESET_ENVELOPE} if turn_reset else {}
            ),
        },
    )

    if turn_reset:
        assert response.status_code == 202, response.text
        run = response.json()["run"]
        assert WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in run["settings_json"]
        assert "workflow_lora" not in run["provenance_json"]
        return
    assert response.status_code == 409
    assert response.json() == {
        "code": WORKFLOW_LORA_ADMISSION_CONFLICT_CODE,
        "detail": WORKFLOW_LORA_ADMISSION_CONFLICT_MESSAGE,
    }
    assert "not a list" not in response.text
    _assert_no_turn_rows(chat["id"], title=title)


@pytest.mark.parametrize("origin", ["chat", "turn"])
async def test_an_edit_the_slot_does_not_allow_names_the_layer_it_came_from(
    client: AsyncClient,
    origin: str,
) -> None:
    del client
    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix=f"admission_field_{origin}")
        revision = session.get(WorkflowRevision, seeded.revision_id)
        assert revision is not None
        # A core loader has no switch, so an on/off edit is not something it allows.
        envelope = _override_envelope(session, seeded, changes={"enabled": False})
        with pytest.raises(WorkflowLoraAdmissionError) as refused:
            admit_workflow_lora_composition(
                session,
                revision,
                activation_snapshot=_workflow_witness(session, seeded)["activation"],
                layers=_split_layers(**{origin: envelope}),
                added_loras=[],
            )

    assert refused.value.conflict is (origin == "chat")


def _force_media_revisions(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    default_revision_id: str,
) -> list[MediaRequest]:
    """Select the preferred revision when a batch names one, and a default otherwise."""

    orchestrator: ConversationOrchestrator = app.state.services.orchestrator
    original_selector = orchestrator._profile_and_workflow_for_operation

    def forced_selector(
        session: Session,
        chat: Chat,
        operation: Operation,
        selection_text: str,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any], WorkflowRevision | None]:
        if operation == Operation.TEXT:
            return original_selector(session, chat, operation, selection_text, *args, **kwargs)
        revision_id = kwargs.get("preferred_revision_id") or default_revision_id
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        return (
            None,
            {
                "mode": "revision",
                "workflow_definition_id": revision.workflow_id,
                "workflow_revision_id": revision.id,
            },
            revision,
        )

    monkeypatch.setattr(orchestrator, "_profile_and_workflow_for_operation", forced_selector)
    engines: EngineRegistry = app.state.services.engines
    original_settings = engines.settings_for_role

    async def forced_settings(role: str, *, engine: str | None = None) -> list[Any]:
        del engine
        return await original_settings(role, engine="mock")

    monkeypatch.setattr(app.state.services.engines, "settings_for_role", forced_settings)
    captured: list[MediaRequest] = []
    original_generate = MockMediaAdapter.generate

    async def capture_generate(
        self: MockMediaAdapter,
        request: MediaRequest,
    ) -> AsyncIterator[MediaEvent]:
        captured.append(deepcopy(request))
        async for event in original_generate(self, request):
            yield event

    monkeypatch.setattr(MockMediaAdapter, "generate", capture_generate)
    return captured


@pytest.mark.parametrize("pooled", [False, True])
async def test_prompt_batch_outputs_each_admit_and_dispatch_saved_workflow_lora_changes(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    pooled: bool,
) -> None:
    suffix = "pool" if pooled else "fixed"
    with SessionLocal() as session:
        first = _seed_workflow(session, suffix=f"admission_batch_{suffix}_a")
        seeded = [first]
        if pooled:
            seeded.append(_seed_workflow(session, suffix=f"admission_batch_{suffix}_b"))
        chat_edits = {
            "version": 1,
            "targets": [
                cast(
                    list[object],
                    _override_envelope(session, item, changes={"model_strength": 0.33})["targets"],
                )[0]
                for item in seeded
            ],
        }
    captured = _force_media_revisions(app, monkeypatch, first.revision_id)
    # Templates accept only revisions for the configured media engine, which is
    # the mock engine here, while these LoRA fixtures are ComfyUI revisions.
    ready = prompt_library.prompt_template_workflow_revision_is_ready

    def ready_for_its_own_engine(
        session: Session, revision: WorkflowRevision, *, expected_engine: str
    ) -> bool:
        del expected_engine
        return ready(session, revision, expected_engine=revision.engine)

    monkeypatch.setattr(
        prompt_library, "prompt_template_workflow_revision_is_ready", ready_for_its_own_engine
    )
    exact = resolve_exact_workflow_revision

    def exact_for_its_own_engine(
        session: Session, revision_id: str, **kwargs: Any
    ) -> tuple[Any, Any, Any]:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        return exact(session, revision_id, **{**kwargs, "engine": revision.engine})

    monkeypatch.setattr(
        orchestrator_module, "resolve_exact_workflow_revision", exact_for_its_own_engine
    )
    chat = (await client.post("/api/chats", json={"title": f"Batch edits {suffix}"})).json()
    _store_chat_setting(chat["id"], chat_edits)
    resource_policy: dict[str, Any] = (
        {
            "mode": "pool",
            "strategy": "round_robin",
            "options": [
                {"workflow_revision_id": item.revision_id, "lora_policy": {"mode": "none"}}
                for item in seeded
            ],
        }
        if pooled
        else {
            "mode": "fixed",
            "workflow_revision_id": first.revision_id,
            "lora_policy": {"mode": "none"},
        }
    )
    template = await client.post(
        "/api/prompt-templates",
        json={
            "idempotency_key": f"lora-batch-template-{suffix}",
            "name": f"Garden studies {suffix}",
            "description": "Constructed batch fixture",
            "contract": {
                "schema_version": 1,
                "operation": "text_to_image",
                "body": "A {{subject}}.",
                "slots": [{"name": "subject", "mode": "input", "variation_scope": "item"}],
                "resource_policy": resource_policy,
            },
        },
    )
    assert template.status_code == 201, template.text
    revision = template.json()["revision"]
    batch = await client.post(
        f"/api/chats/{chat['id']}/prompt-batches",
        json={
            "idempotency_key": f"lora-batch-preview-{suffix}",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": 2,
            "selection_seed": 0,
            "inputs": {"subject": ["paper boat", "painted hillside"]},
        },
    )
    assert batch.status_code == 201, batch.text

    queued = await client.post(
        f"/api/prompt-batches/{batch.json()['id']}/queue",
        json={
            "idempotency_key": f"lora-batch-queue-{suffix}",
            "expected_plan_version": batch.json()["plan_version"],
            "expected_plan_sha256": batch.json()["plan_sha256"],
        },
    )

    assert queued.status_code == 202, queued.text
    plan_id = queued.json()["work_plan_id"]
    assert (await _wait_for_plan(client, plan_id))["status"] == "complete"
    with SessionLocal() as session:
        runs = list(session.scalars(select(Run).where(Run.work_plan_id == plan_id)).all())
        assert len(runs) == 2
        assert {run.workflow_revision_id for run in runs} == {item.revision_id for item in seeded}
        for run in runs:
            setting = run.settings_json[WORKFLOW_LORA_OVERRIDES_SETTING_KEY]
            assert [target["workflow_revision_id"] for target in setting["targets"]] == [
                run.workflow_revision_id
            ]
            assert "workflow_lora" in run.provenance_json
    assert len(captured) == 2
    for dispatched in captured:
        assert dispatched.workflow[PRIVATE_LOADER_ID]["inputs"]["strength_model"] == 0.33
        assert WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in dispatched.parameters
