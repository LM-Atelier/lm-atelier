"""Resolve a detached recipe before awaiting engine fields or choosing a revision."""

import pytest
from fastapi import FastAPI

from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import Chat, ChatWorkflowUseCaseSelection, WorkflowUseCasePreset
from local_lm.schemas import SettingField
from local_lm.workflow_use_cases_v1 import WorkflowUseCaseInputs


async def test_recipe_snapshot_does_not_hold_a_transaction_during_engine_probe(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm.workflow_use_case_execution import prepare_workflow_use_case_execution

    with SessionLocal() as session:
        chat = Chat()
        preset = WorkflowUseCasePreset(
            name="Accepted recipe", use_case="image_generation", settings_json={"quality": 0.4}
        )
        session.add_all([chat, preset])
        session.flush()
        session.add(
            ChatWorkflowUseCaseSelection(
                chat_id=chat.id, use_case="image_generation", preset_id=preset.id
            )
        )
        session.commit()
        chat_id, preset_id = chat.id, preset.id

    async def fields(role: str) -> list[SettingField]:
        assert role == "image"
        with SessionLocal() as writer:
            current = writer.get(WorkflowUseCasePreset, preset_id)
            assert current is not None
            current.name = "Later recipe"
            current.settings_json = {"quality": 0.9}
            writer.commit()
        return [
            SettingField(
                key="quality", label="Quality", type="number", default=0.1, scope="workflow"
            )
        ]

    monkeypatch.setattr(app.state.services.engines, "settings_for_role", fields)
    prepared = await prepare_workflow_use_case_execution(
        SessionLocal,
        app.state.services.engines,
        WorkflowUseCaseInputs(Operation.TEXT_TO_IMAGE),
        chat_id=chat_id,
    )
    assert prepared is not None
    assert prepared.preset.preset_name == "Accepted recipe"
    assert prepared.preset.settings_json == {"quality": 0.4}
    with SessionLocal() as session:
        current = session.get(WorkflowUseCasePreset, preset_id)
        assert current is not None and current.settings_json == {"quality": 0.9}


async def test_unconfigured_recipes_do_not_probe_the_engine(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm.workflow_use_case_execution import prepare_workflow_use_case_execution

    with SessionLocal() as session:
        chat = Chat()
        session.add(chat)
        session.commit()
        chat_id = chat.id

    async def unavailable(role: str) -> list[SettingField]:
        raise AssertionError("An unconfigured recipe does not need engine settings")

    monkeypatch.setattr(app.state.services.engines, "settings_for_role", unavailable)
    assert (
        await prepare_workflow_use_case_execution(
            SessionLocal,
            app.state.services.engines,
            WorkflowUseCaseInputs(Operation.TEXT),
            chat_id=chat_id,
        )
        is None
    )
