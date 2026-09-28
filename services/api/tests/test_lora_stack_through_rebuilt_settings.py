"""A LoRA stack chosen for a workflow that never declared one survives a rebuilt settings layer.

A workflow can take LoRAs without declaring a setting for them, and a turn on
it accepts a stack. Every later action that starts from that turn's settings,
an edit, a regenerate, a branch or an automatic retry, rebuilds them against
the fields the workflow offers, and a key with no field behind it is dropped.
Each of those is driven here through its own route, and each must keep the
stack the person chose.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy.orm import Session
from test_api import _one_pass_claim_capture, _retry_inputs
from test_workflow_lora_admission_vertical import (
    _force_seeded_workflow,
    _undeclare_lora_setting,
    _wait_for_plan,
)
from test_workflow_lora_execution import SeededWorkflow, _seed_workflow

from local_lm import orchestrator as orchestrator_module
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, Operation, utcnow
from local_lm.image_edit_verification import image_edit_verification_job_id
from local_lm.models import Job, Run, WorkflowRevision
from local_lm.workflow_selection import resolve_exact_workflow_revision


def _stack(asset_id: str) -> list[dict[str, Any]]:
    return [{"asset_id": asset_id, "model_strength": 0.62, "clip_strength": 0.51, "enabled": True}]


async def _turn_with_a_stack(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> tuple[SeededWorkflow, dict[str, Any]]:
    """One finished image turn whose settings carry a stack its workflow never declared."""

    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix=suffix)
        _undeclare_lora_setting(session, seeded)
    _force_seeded_workflow(app, monkeypatch, seeded)
    chat = (await client.post("/api/chats", json={"title": "A stack to keep"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create one careful study",
            "mode": "image",
            "settings": {"loras": _stack(seeded.added_asset_id)},
        },
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    plan = await _wait_for_plan(client, accepted["run"]["work_plan_id"])
    assert plan["status"] == "complete"
    return seeded, accepted


def _stored_stack(run_id: str) -> list[str]:
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        return [item["asset_id"] for item in run.settings_json.get("loras", [])]


def _accept_edits_without_running(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let an edit pin this ComfyUI-built revision under the mock engine, and not run it.

    An edit resolves the source's pinned revision against the configured media
    engine, so the revision is resolved for its own engine instead. Only the
    stored settings are in question, so the edited turn is accepted but never run.
    """

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
    monkeypatch.setattr(type(app.state.services.orchestrator), "start", lambda self, *args: None)


async def test_the_edit_view_offers_the_stack_back(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded, source = await _turn_with_a_stack(client, app, monkeypatch, "rebuilt_view")

    view = await client.get(f"/api/messages/{source['user_message']['id']}/edit-source")

    assert view.status_code == 200, view.text
    offered = view.json()["settings"].get("loras", [])
    assert [item["asset_id"] for item in offered] == [seeded.added_asset_id]


async def test_each_step_of_an_ordered_turn_offers_its_stack_back(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordered turn's edit view rebuilds every step's settings, not only the turn's."""

    with SessionLocal() as session:
        seeded = _seed_workflow(session, suffix="rebuilt_ordered")
        _undeclare_lora_setting(session, seeded)
    _force_seeded_workflow(app, monkeypatch, seeded)
    chat = (await client.post("/api/chats", json={"title": "Stacks on each step"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of a paper boat, then animate the image into a video",
            "mode": "auto",
            "confirm_media": True,
            "ordered_settings": {
                "image": {"loras": _stack(seeded.added_asset_id)},
                "video": {"loras": _stack(seeded.added_asset_id)},
            },
        },
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    plan = await _wait_for_plan(client, accepted["run"]["work_plan_id"])
    assert plan["status"] == "complete"

    view = await client.get(f"/api/messages/{accepted['user_message']['id']}/edit-source")

    assert view.status_code == 200, view.text
    steps = view.json()["steps"]
    assert len(steps) == 2
    for step in steps:
        offered = step["settings"].get("loras", [])
        assert [item["asset_id"] for item in offered] == [seeded.added_asset_id], step["operation"]


async def test_an_edited_turn_keeps_the_stack(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded, source = await _turn_with_a_stack(client, app, monkeypatch, "rebuilt_edit")
    _accept_edits_without_running(app, monkeypatch)

    edit = await client.post(
        f"/api/messages/{source['user_message']['id']}/edits",
        json={"text": "Create one careful study, closer", "idempotency_key": "keep-the-stack"},
    )

    assert edit.status_code == 202, edit.text
    assert _stored_stack(edit.json()["run"]["id"]) == [seeded.added_asset_id]


async def test_an_edit_routed_again_inherits_the_stack(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit sent in Auto mode is routed afresh and inherits the source's settings by role."""

    seeded, source = await _turn_with_a_stack(client, app, monkeypatch, "rebuilt_auto")
    _accept_edits_without_running(app, monkeypatch)

    edit = await client.post(
        f"/api/messages/{source['user_message']['id']}/edits",
        json={
            "text": "Generate an image of one careful study, closer",
            "mode": "auto",
            "idempotency_key": "inherit-the-stack",
        },
    )

    assert edit.status_code == 202, edit.text
    run_id = edit.json()["run"]["id"]
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        # Routed to an image again, so the inheritance by role is what applied.
        assert run.operation == Operation.TEXT_TO_IMAGE.value
    assert _stored_stack(run_id) == [seeded.added_asset_id]


async def test_a_regenerated_turn_keeps_the_stack(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded, source = await _turn_with_a_stack(client, app, monkeypatch, "rebuilt_regenerate")

    again = await client.post(
        f"/api/messages/{source['assistant_message']['id']}/regenerate", json={}
    )

    assert again.status_code == 202, again.text
    assert _stored_stack(again.json()["run"]["id"]) == [seeded.added_asset_id]


async def test_a_branch_that_inherits_the_settings_keeps_the_stack(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded, source = await _turn_with_a_stack(client, app, monkeypatch, "rebuilt_branch")

    branch = await client.post(
        f"/api/messages/{source['user_message']['id']}/branch",
        json={"text": "Create one careful study, wider"},
    )

    assert branch.status_code == 202, branch.text
    assert _stored_stack(branch.json()["run"]["id"]) == [seeded.added_asset_id]


async def test_an_automatic_image_edit_retry_keeps_the_stack(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry a verification asks for starts from the source turn's settings too."""

    seeded, source = await _turn_with_a_stack(client, app, monkeypatch, "rebuilt_retry")
    run_id = source["run"]["id"]
    verification_id = image_edit_verification_job_id(run_id)
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        # The finished turn becomes an automatic-strength edit awaiting its
        # verification, which is the state a retry is created from.
        run.operation = Operation.IMAGE_TO_IMAGE.value
        run.settings_json = {**run.settings_json, "denoise": 0.5}
        run.provenance_json = {
            **run.provenance_json,
            "image_edit": {
                "strength": {
                    "mode": "auto",
                    "parameter": "denoise",
                    "value": 0.5,
                    "scope": "localized",
                    "confidence": "high",
                }
            },
        }
        session.add(
            Job(
                id=verification_id,
                kind=JobKind.EDIT_VERIFY.value,
                status=JobStatus.QUEUED.value,
                queue_resource="media_compute",
                queue_group="primary",
                enqueued_at=utcnow(),
                payload_json={"source_run_id": run_id},
            )
        )
        session.commit()
    claim = await _one_pass_claim_capture(monkeypatch, verification_id)
    monkeypatch.undo()
    _force_seeded_workflow(app, monkeypatch, seeded)
    orchestrator = app.state.services.orchestrator
    monkeypatch.setattr(type(orchestrator), "start", lambda self, *args: None)
    payload, decision = _retry_inputs(source["run"]["chat_id"], run_id)

    with SessionLocal() as session:
        retried = await orchestrator._create_image_edit_verification_retry(
            session, payload, decision, claim=claim
        )
        retry_id = retried.run.id

    assert _stored_stack(retry_id) == [seeded.added_asset_id]
