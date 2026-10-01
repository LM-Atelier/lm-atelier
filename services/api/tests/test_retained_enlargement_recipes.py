"""Retain recipe admission and enlargement intent across older media results."""

from copy import deepcopy

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_legacy_enlargement_replay import legacy_generation

from local_lm.accepted_turn_context import _digest, accepted_context
from local_lm.db import SessionLocal
from local_lm.models import Run, RunContextSnapshot, WorkflowRevision, WorkflowUseCasePreset


@pytest.mark.parametrize("enlarge", [False, True])
async def test_a_retained_recipe_omits_the_factor_its_workflow_never_reads(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, enlarge: bool
) -> None:
    source = await legacy_generation(app, client, monkeypatch)
    created = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Recorded scale choice",
            "use_case": "image_upscale" if enlarge else "image_edit",
            "settings_json": {"upscale_factor": 2},
            "is_default": True,
        },
    )
    assert created.status_code == 201, created.text
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, source["revision"])
        assert revision is not None
        schema = deepcopy(revision.input_schema_json)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{source['chat']}/turns",
            json={
                "text": "Enlarge this picture" if enlarge else "A green geometric square",
                "mode": "image",
                "input_artifact_ids": [source["picture"]],
                "upscale": enlarge,
            },
        )
        assert response.status_code == 202, response.text
        result = response.json()["run"]
        assert "upscale_factor" not in result["settings_json"]
        assert result["provenance_json"]["upscale"] is enlarge
        receipt = result["provenance_json"]["workflow_use_case_preset"]
        assert receipt["preset_id"] == created.json()["id"]
        assert receipt["settings_json"] == {}
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, source["revision"])
        recipe = session.get(WorkflowUseCasePreset, created.json()["id"])
        assert revision is not None and revision.input_schema_json == schema
        assert recipe is not None and recipe.settings_json == {"upscale_factor": 2}


async def historical_enlargement(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frozen: bool,
    explicit_false: bool = False,
) -> dict[str, str]:
    source = await legacy_generation(app, client, monkeypatch)
    recipe = await client.post(
        "/api/workflow-use-case-presets",
        json={"name": "Recorded enlargement", "use_case": "image_upscale", "settings_json": {}},
    )
    assert recipe.status_code == 201, recipe.text
    with SessionLocal() as session:
        run = session.get(Run, source["run"])
        assert run is not None
        provenance = deepcopy(run.provenance_json)
        if not explicit_false:
            provenance.pop("upscale")
        provenance["workflow_use_case_preset"] = {
            "version": 1,
            "workflow_revision_id": source["revision"],
            "use_case": "image_upscale",
            "mode": "preset",
            "scope": "chat",
            "preset_id": recipe.json()["id"],
            "preset_name": "Recorded enlargement",
            "settings_json": {},
            "upscale_kind": "resample",
        }
        run.provenance_json = provenance
        if frozen:
            app.state.services.orchestrator._freeze_turn_context(session, run)
            row = session.get(RunContextSnapshot, run.id)
            assert row is not None
            payload = deepcopy(row.payload_json)
            if not explicit_false:
                payload.pop("upscale")
            row.payload_json = payload
            row.sha256 = _digest(payload)
            run.provenance_json = {**run.provenance_json, "accepted_context_sha256": row.sha256}
        session.commit()
    return source


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("action", ["edit", "regenerate"])
async def test_an_older_enlargement_recipe_preserves_intent_when_replayed(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    frozen: bool,
    action: str,
) -> None:
    source = await historical_enlargement(app, client, monkeypatch, frozen=frozen)
    with SessionLocal() as session:
        original = session.get(Run, source["run"])
        assert original is not None
        provenance = deepcopy(original.provenance_json)
        context = accepted_context(session, original)
    async with app.state.services.scheduler.lease("primary"):
        if action == "edit":
            response = await client.post(
                f"/api/messages/{source['user']}/edits",
                json={"text": "A green geometric square", "idempotency_key": "recorded-scale-edit"},
            )
        else:
            response = await client.post(
                f"/api/messages/{source['assistant']}/regenerate", json={"settings": {}}
            )
        assert response.status_code == 202, response.text
        result = response.json()["run"]
        assert result["provenance_json"]["upscale"] is True
        assert result["provenance_json"]["workflow_use_case_preset"]["use_case"] == "image_upscale"
        assert result["workflow_revision_id"] == source["revision"]
        assert "upscale_factor" not in result["settings_json"]
        with SessionLocal() as session:
            replay = session.get(Run, result["id"])
            original = session.get(Run, source["run"])
            assert replay is not None and original is not None
            replay_context = accepted_context(session, replay)
            assert replay_context is not None and replay_context.upscale is True
            assert original.provenance_json == provenance
            assert accepted_context(session, original) == context


@pytest.mark.parametrize(
    ("frozen", "missing_snapshot_marker"), [(False, False), (True, False), (True, True)]
)
async def test_an_explicitly_ordinary_result_cannot_inherit_an_enlargement_recipe(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    frozen: bool,
    missing_snapshot_marker: bool,
) -> None:
    source = await historical_enlargement(
        app, client, monkeypatch, frozen=frozen, explicit_false=True
    )
    if missing_snapshot_marker:
        with SessionLocal() as session:
            run = session.get(Run, source["run"])
            row = session.get(RunContextSnapshot, source["run"])
            assert run is not None and row is not None
            payload = deepcopy(row.payload_json)
            payload.pop("upscale")
            row.payload_json = payload
            row.sha256 = _digest(payload)
            run.provenance_json = {**run.provenance_json, "accepted_context_sha256": row.sha256}
            session.commit()
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/messages/{source['assistant']}/regenerate", json={"settings": {}}
        )
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "workflow-use-case-preset-mismatch"
