"""New work ignores deleted project defaults while frozen work keeps its snapshot."""

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import select
from test_accepted_turn_context import _accept_context
from test_chat_deletion import _text_exchange
from test_chat_recovery import _command, _impact
from test_project_recovery_api import _project, _stored_trash
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm.db import SessionLocal
from local_lm.models import (
    ProjectWorkflowUseCaseSelection,
    Run,
    RunContextSnapshot,
    WorkflowUseCasePreset,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.workflow_use_case_preset_resolution import (
    WorkflowUseCasePresetResolutionError,
    resolve_workflow_use_case_preset,
)
from local_lm.workflow_use_cases_v1 import WorkflowUseCase


@pytest.mark.parametrize("mode", ["text", "image", "video"])
async def test_new_turns_stop_inheriting_deleted_project_settings_and_restore_them(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    project_id, chat_id = await _project(client)
    role = "chat" if mode == "text" else mode
    field, value = ("max_tokens", 37) if mode == "text" else ("seed", 17)
    updated = await client.patch(
        f"/api/projects/{project_id}",
        json={
            "instructions": "Use evenly spaced garden beds",
            "generation_settings_json": {role: {field: value}},
        },
    )
    assert updated.status_code == 200, updated.text
    seen = []
    adapter = (
        app.state.services.engines.chat if mode == "text" else app.state.services.engines.media
    )
    method = "stream" if mode == "text" else "generate"
    original = getattr(adapter, method)

    async def observe(request):
        seen.append(request)
        async for event in original(request):
            yield event

    monkeypatch.setattr(adapter, method, observe)

    async def generate():
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns", json={"text": "A garden path", "mode": mode}
        )
        assert accepted.status_code == 202, accepted.text
        run_id = accepted.json()["run"]["id"]

        async def read():
            response = await client.get(f"/api/runs/{run_id}")
            assert response.status_code == 200
            return response.json()

        await wait_for_terminal_status(read, what="garden generation")
        return seen[-1]

    first = await generate()
    assert (first.settings if mode == "text" else first.parameters)[field] == value
    item = _stored_trash(project_id, datetime.now(UTC))
    second = await generate()
    assert (second.settings if mode == "text" else second.parameters).get(field) != value
    if mode == "text":
        instruction = {"role": "system", "content": "Use evenly spaced garden beds"}
        assert instruction in first.messages and instruction not in second.messages
    preview = await _impact(client, f"/api/recovery-items/{item.deletion_id}/impact")
    restored = await client.post(
        f"/api/recovery-items/{item.deletion_id}/restore",
        json=_command(preview, "restore-defaults"),
    )
    assert restored.status_code == 200
    third = await generate()
    assert (third.settings if mode == "text" else third.parameters)[field] == value


async def test_frozen_accepted_work_keeps_its_original_project_instructions_after_trash(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        run_id, _source_id, project_id = await _accept_context(app, client)
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None
            before = ConversationOrchestrator._context_messages(session, run)
            assert {"role": "system", "content": "Use short paragraphs"} in before
        _stored_trash(project_id, datetime.now(UTC))
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None
            assert ConversationOrchestrator._context_messages(session, run) == before


async def test_deleted_project_recipe_falls_back_to_workspace_without_rewriting_selection(
    client: AsyncClient,
) -> None:
    project_id, chat_id = await _project(client)
    with SessionLocal() as session:
        session.add_all(
            [
                WorkflowUseCasePreset(
                    id="garden-project",
                    name="Garden layout",
                    use_case="image_generation",
                    settings_json={"seed": 17},
                ),
                WorkflowUseCasePreset(
                    id="garden-workspace",
                    name="Garden paths",
                    use_case="image_generation",
                    settings_json={"seed": 29},
                    is_default=True,
                ),
            ]
        )
        session.flush()
        session.add(
            ProjectWorkflowUseCaseSelection(
                project_id=project_id, use_case="image_generation", preset_id="garden-project"
            )
        )
        session.commit()
        assert (
            resolve_workflow_use_case_preset(
                session, WorkflowUseCase.IMAGE_GENERATION, chat_id=chat_id
            ).preset_id
            == "garden-project"
        )
    item = _stored_trash(project_id, datetime.now(UTC))
    with SessionLocal() as session:
        resolved = resolve_workflow_use_case_preset(
            session, WorkflowUseCase.IMAGE_GENERATION, chat_id=chat_id
        )
        assert resolved.scope == "workspace" and resolved.preset_id == "garden-workspace"
        assert (
            session.get(ProjectWorkflowUseCaseSelection, (project_id, "image_generation")).preset_id
            == "garden-project"
        )
    preview = await _impact(client, f"/api/recovery-items/{item.deletion_id}/impact")
    assert (
        await client.post(
            f"/api/recovery-items/{item.deletion_id}/restore",
            json=_command(preview, "restore-recipe"),
        )
    ).status_code == 200
    with SessionLocal() as session:
        assert (
            resolve_workflow_use_case_preset(
                session, WorkflowUseCase.IMAGE_GENERATION, chat_id=chat_id
            ).preset_id
            == "garden-project"
        )


@pytest.mark.parametrize("operation", ["read", "write", "resolve"])
async def test_deleted_project_recipe_scope_refuses_explicit_access(
    client: AsyncClient,
    operation: str,
) -> None:
    project_id, _chat_id = await _project(client)
    _stored_trash(project_id, datetime.now(UTC))
    path = f"/api/projects/{project_id}/workflow-use-case-presets/image_generation"
    if operation == "resolve":
        with (
            SessionLocal() as session,
            pytest.raises(
                WorkflowUseCasePresetResolutionError, match="^workflow-use-case-project-not-found$"
            ),
        ):
            resolve_workflow_use_case_preset(
                session, WorkflowUseCase.IMAGE_GENERATION, project_id=project_id
            )
    else:
        response = (
            await client.get(path)
            if operation == "read"
            else await client.put(path, json={"mode": "inherit"})
        )
        assert (
            response.status_code == 404
            and response.json()["code"] == "workflow-use-case-scope-not-found"
        )


async def test_deleted_project_workflow_selectors_are_inaccessible(client: AsyncClient) -> None:
    project_id, _chat_id = await _project(client)
    _stored_trash(project_id, datetime.now(UTC))
    responses = [
        await client.get(f"/api/projects/{project_id}/workflow-selections"),
        await client.put(
            f"/api/projects/{project_id}/workflow-selections/image", json={"mode": "inherit"}
        ),
    ]
    assert all(response.status_code == 404 for response in responses)


async def test_routing_context_ignores_a_deleted_projects_instructions(client: AsyncClient) -> None:
    project_id, chat_id = await _project(client)
    assert (
        await client.patch(
            f"/api/projects/{project_id}", json={"instructions": "Use evenly spaced garden beds"}
        )
    ).status_code == 200
    exchange = await _text_exchange(client, chat_id, "Keep the garden paths clear")
    _stored_trash(project_id, datetime.now(UTC))
    from local_lm.models import Chat

    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        context = ConversationOrchestrator._routing_context(
            session, chat, exchange["assistant_message"]["id"]
        )
        assert all(
            message != {"role": "system", "content": "Use evenly spaced garden beds"}
            for message in context
        )


@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_project_trash_freezes_ordinary_queued_work_before_hiding_configuration(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    project_id, chat_id = await _project(client)
    instruction = {"role": "system", "content": "Use evenly spaced garden beds"}
    assert (
        await client.patch(
            f"/api/projects/{project_id}",
            json={
                "instructions": instruction["content"],
                "generation_settings_json": {"chat": {"max_tokens": 37}},
            },
        )
    ).status_code == 200
    seen = []
    original = app.state.services.engines.chat.stream

    async def observe(request):
        seen.append(request)
        async for event in original(request):
            yield event

    monkeypatch.setattr(app.state.services.engines.chat, "stream", observe)
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={"text": "Keep the garden paths wide", "mode": "text"},
        )
        assert accepted.status_code == 202, accepted.text
        run_id = accepted.json()["run"]["id"]
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None
            before = ConversationOrchestrator._context_messages(session, run)
            assert instruction in before
            assert session.get(RunContextSnapshot, run_id) is None
        preview = await _impact(client, f"/api/projects/{project_id}/deletion-impact")
        response = await client.post(
            f"/api/projects/{project_id}/trash", json=_command(preview, "trash-queued-project")
        )
        assert response.status_code == 200, response.text
        deletion_id = response.json()["deletion_id"]
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None and session.get(RunContextSnapshot, run_id) is not None
            assert ConversationOrchestrator._context_messages(session, run) == before
        preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
        command = _command(preview, f"{action}-queued-project")
        if action == "purge":
            command["acknowledgement"] = "permanently-delete"
        transition = await client.post(f"/api/recovery-items/{deletion_id}/{action}", json=command)
        assert transition.status_code == 200, transition.text
        assert seen == []

    async def read():
        return (await client.get(f"/api/runs/{run_id}")).json()

    await wait_for_terminal_status(read, what="accepted garden work")
    assert seen and seen[-1].settings["max_tokens"] == 37
    assert instruction in seen[-1].messages


async def test_project_trash_rolls_back_new_snapshots_when_context_capture_fails(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm.models import RecoveryItem, RecoveryOperation
    from local_lm.recovery_previews import RecoveryPreviewConflict

    project_id, chat_id = await _project(client)
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={"text": "Keep the garden paths clear", "mode": "text"},
        )
        assert accepted.status_code == 202
        run_id = accepted.json()["run"]["id"]
        original = app.state.services.orchestrator._freeze_turn_context

        def fail_after_snapshot(*args, **kwargs):
            original(*args, **kwargs)
            raise RecoveryPreviewConflict("project-recovery-accepted-work-invalid")

        monkeypatch.setattr(
            app.state.services.orchestrator, "_freeze_turn_context", fail_after_snapshot
        )
        preview = await _impact(client, f"/api/projects/{project_id}/deletion-impact")
        sequence = app.state.services.events.sequence
        refused = await client.post(
            f"/api/projects/{project_id}/trash", json=_command(preview, "trash-failed-capture")
        )
        assert (
            refused.status_code == 409
            and refused.json()["code"] == "project-recovery-accepted-work-invalid"
        )
        assert app.state.services.events.sequence == sequence
        with SessionLocal() as session:
            assert session.get(RunContextSnapshot, run_id) is None
            assert session.scalar(select(RecoveryItem)) is None
            assert session.scalar(select(RecoveryOperation)) is None
