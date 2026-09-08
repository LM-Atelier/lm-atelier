from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import (
    ChatWorkflowSelection,
    Project,
    Run,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
    WorkPlan,
)


def _image_family(session: Any, name: str) -> tuple[str, str]:
    family = WorkflowFamily(name=name)
    definition = WorkflowDefinition(
        family=family, variant_key="create", name=name, operation="text_to_image"
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="mock",
        api_graph_json={},
        input_schema_json={},
        dependencies_json={},
        trusted=True,
    )
    preference = WorkflowPreference(family=family, selector_capability="image")
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    return family.id, revision.id


async def _workflow_source(
    client: AsyncClient,
) -> tuple[dict[str, Any], list[tuple[str, str]], str]:
    project = (await client.post("/api/projects", json={"name": "Version workflows"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={
                "title": "Edit-local workflow",
                "project_id": project["id"],
            },
        )
    ).json()
    with SessionLocal() as session:
        choices = [
            _image_family(session, name) for name in ("Original", "Chosen", "Project default")
        ]
        current = session.get(Project, project["id"])
        assert current is not None
        current.image_workflow_revision_id = choices[0][1]
        session.commit()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of a paper boat",
            "mode": "image",
            "workflow_revision_id": choices[0][1],
        },
    )
    assert original.status_code == 202, original.text
    with SessionLocal() as session:
        current = session.get(Project, project["id"])
        assert current is not None
        current.image_workflow_revision_id = choices[2][1]
        saved = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat["id"],
                ChatWorkflowSelection.selector_capability == "image",
            )
        )
        assert saved is not None
        saved.mode = "family"
        saved.workflow_family_id = choices[1][0]
        session.commit()
    return original.json(), choices, project["id"]


@pytest.mark.parametrize(
    "choice",
    [
        "inherit",
        "null",
        "default",
        "family",
        "revision",
        "legacy_revision",
        "missing_family",
        "missing_revision",
        "wrong_role",
        "conflict",
    ],
)
async def test_edit_workflow_choice_is_local_and_exact(
    app: FastAPI,
    client: AsyncClient,
    choice: str,
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source, choices, project_id = await _workflow_source(client)
        payload: dict[str, Any] = {
            "text": "A blue paper boat",
            "idempotency_key": "workflow-choice",
        }
        if choice == "null":
            payload["workflow_selection"] = None
        elif choice == "legacy_revision":
            payload["workflow_revision_id"] = choices[1][1]
        elif choice != "inherit":
            selection: dict[str, Any] = {"selector_capability": "image"}
            if choice in {"family", "missing_family"}:
                selection.update(
                    mode="family",
                    workflow_family_id=(choices[1][0] if choice == "family" else "absent-family"),
                )
            elif choice in {"revision", "missing_revision", "conflict"}:
                selection.update(
                    mode="revision",
                    workflow_revision_id=(
                        "absent-revision" if choice == "missing_revision" else choices[1][1]
                    ),
                )
            else:
                selection["mode"] = "default"
            if choice == "wrong_role":
                selection["selector_capability"] = "video"
            if choice == "conflict":
                payload["workflow_revision_id"] = choices[0][1]
            payload["workflow_selection"] = selection
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json=payload,
        )
        invalid = choice in {"missing_family", "missing_revision", "wrong_role", "conflict"}
        assert response.status_code == (422 if invalid else 202), response.text
        if invalid:
            assert response.json()["code"] != "request-validation-invalid" or choice == "conflict"
        with SessionLocal() as session:
            original = session.get(Run, source["run"]["id"])
            project = session.get(Project, project_id)
            saved = session.scalar(
                select(ChatWorkflowSelection).where(
                    ChatWorkflowSelection.chat_id == source["run"]["chat_id"],
                    ChatWorkflowSelection.selector_capability == "image",
                )
            )
            assert original is not None and original.workflow_revision_id == choices[0][1]
            assert project is not None and project.image_workflow_revision_id == choices[2][1]
            assert saved is not None and saved.workflow_family_id == choices[1][0]
            if invalid:
                assert len(list(session.scalars(select(WorkPlan)))) == 1
                return
            expected = (
                choices[0][1]
                if choice == "inherit"
                else (choices[2][1] if choice == "default" else choices[1][1])
            )
            assert response.json()["run"]["workflow_revision_id"] == expected


@pytest.mark.parametrize("selector", ["revision", "legacy_revision", "family"])
async def test_ordered_edit_honors_workflow_only_for_matching_steps(
    app: FastAPI,
    client: AsyncClient,
    selector: str,
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source, choices, _ = await _workflow_source(client)
        payload: dict[str, Any] = {
            "text": "Write a short story about a paper boat, then create an image based on it, "
            "then animate the image into a video, then summarize the video",
            "mode": "auto",
            "confirm_media": True,
            "idempotency_key": "ordered-workflow",
        }
        if selector == "legacy_revision":
            payload["workflow_revision_id"] = choices[0][1]
        else:
            payload["workflow_selection"] = {
                "selector_capability": "image",
                "mode": selector,
                (
                    "workflow_revision_id" if selector == "revision" else "workflow_family_id"
                ): choices[0][1] if selector == "revision" else choices[0][0],
            }
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json=payload,
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run).where(
                        Run.work_plan_id == response.json()["work_plan_id"],
                    )
                )
            )
            assert len(runs) == 4
            for run in runs:
                if run.operation == "text_to_image":
                    assert run.workflow_revision_id == choices[0][1]
                else:
                    assert run.workflow_revision_id != choices[0][1]
