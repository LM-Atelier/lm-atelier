from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import Run, WorkflowDefinition, WorkflowRevision


@pytest.mark.parametrize("target_mode", ["image", "auto"])
@pytest.mark.parametrize(
    "choice", ["inherit_graph", "inherit_schema", "explicit_revision", "inherit_validation"]
)
async def test_edited_workflow_uses_accepted_revision_content_until_reselected(
    app: FastAPI, client: AsyncClient, choice: str, target_mode: str
) -> None:
    graph = {"nodes": [{"inputs": {"color": "blue"}}]}
    schema = {"type": "object", "properties": {"seed": {"type": "integer", "default": 10}}}
    with SessionLocal() as session:
        definition = WorkflowDefinition(
            name="Constructed color workflow", operation="text_to_image"
        )
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            trusted=True,
            api_graph_json=graph,
            input_schema_json=schema,
            dependencies_json={},
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        revision_id = revision.id
        session.commit()
    chat = (await client.post("/api/chats", json={"title": "Accepted workflow content"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Create an image of a blue paper boat.",
                "mode": "image",
                "workflow_revision_id": revision_id,
                "settings": {"seed": 10},
            },
        )
        assert source.status_code == 202, source.text
        first = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/edits",
            json={"text": "Use a smaller paper boat.", "idempotency_key": "first-workflow-content"},
        )
        assert first.status_code == 202, first.text
        from local_lm.accepted_turn_context import accepted_context, resolve_accepted_workflow

        with SessionLocal() as session:
            original = session.get(Run, first.json()["run"]["id"])
            assert original is not None
            snapshot = accepted_context(session, original)
            assert snapshot is not None and snapshot.workflow is not None
            revision = session.get(WorkflowRevision, revision_id)
            assert revision is not None
            revision.api_graph_json = {"nodes": [{"inputs": {"color": "red"}}]}
            revision.input_schema_json = {
                "type": "object",
                "properties": {"seed": {"type": "integer", "default": 20}},
            }
            if choice == "inherit_validation":
                revision.input_schema_json = {
                    "type": "object",
                    "properties": {"seed": {"type": "integer", "enum": [20], "default": 20}},
                }
            new_graph = revision.api_graph_json
            new_schema = revision.input_schema_json
            session.commit()

        payload: dict[str, Any] = {
            "text": "Create an image of a larger paper boat.",
            "mode": target_mode,
            "confirm_media": True,
            "idempotency_key": "second-workflow-content",
        }
        if choice == "explicit_revision":
            payload["workflow_revision_id"] = revision_id
        second = await client.post(
            f"/api/messages/{first.json()['user_message']['id']}/edits", json=payload
        )
        assert second.status_code == 202, second.text
        with SessionLocal() as session:
            current = session.get(Run, second.json()["run"]["id"])
            assert current is not None
            accepted = accepted_context(session, current)
            assert accepted is not None and accepted.workflow is not None
            selected = choice == "explicit_revision"
            if choice != "inherit_schema":
                assert accepted.workflow.api_graph_json == (new_graph if selected else graph)
            assert accepted.workflow.input_schema_json == (new_schema if selected else schema)
            executable = resolve_accepted_workflow(session, accepted.workflow)
            assert executable is not None
            assert executable.api_graph_json == (new_graph if selected else graph)
            assert executable.input_schema_json == (new_schema if selected else schema)
            original = session.get(Run, first.json()["run"]["id"])
            assert original is not None and accepted_context(session, original) == snapshot
