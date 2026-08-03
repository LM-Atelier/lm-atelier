from __future__ import annotations

from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import (
    ChatWorkflowSelection,
    ProjectWorkflowSelection,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowRevision,
)
from local_lm.workflow_compatibility import AUTO_PROFILE_ID, compatibility_family_id


async def test_profile_and_chat_legacy_writes_are_mirrored_and_retired(
    client: AsyncClient,
) -> None:
    profile_response = await client.post(
        "/api/profiles",
        json={
            "name": "Compatibility image",
            "role": "image",
            "engine": "mock",
        },
    )
    assert profile_response.status_code == 201
    profile = profile_response.json()
    family_id = compatibility_family_id(profile["id"])
    chat = (await client.post("/api/chats", json={"title": "Compatibility"})).json()

    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_image_profile_id": profile["id"]},
    )
    assert selected.status_code == 200
    with SessionLocal() as session:
        selection = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat["id"],
                ChatWorkflowSelection.selector_capability == "image",
            )
        )
        assert selection is not None
        assert selection.mode == "family"
        assert selection.workflow_family_id == family_id
        assert session.get(WorkflowFamily, family_id) is not None

    deleted = await client.delete(f"/api/profiles/{profile['id']}")
    assert deleted.status_code == 204
    refreshed = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert refreshed["active_image_profile_id"] == AUTO_PROFILE_ID
    with SessionLocal() as session:
        selection = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat["id"],
                ChatWorkflowSelection.selector_capability == "image",
            )
        )
        assert selection is not None
        assert selection.mode == "automatic"
        assert selection.workflow_family_id is None
        assert session.get(WorkflowFamily, family_id) is None


async def test_project_api_mirrors_exact_revision_pins_and_legacy_null(
    client: AsyncClient,
) -> None:
    revision_id = "wfrev_compatibility_api"
    with SessionLocal() as session:
        definition = WorkflowDefinition(
            id="workflow_compatibility_api",
            name="Compatibility API",
            operation="text_to_image",
        )
        revision = WorkflowRevision(
            id=revision_id,
            definition=definition,
            version=1,
            trusted=True,
        )
        session.add_all([definition, revision])
        session.commit()

    created = await client.post(
        "/api/projects",
        json={
            "name": "Compatibility pin",
            "image_workflow_revision_id": revision_id,
        },
    )
    assert created.status_code == 201
    project = created.json()
    with SessionLocal() as session:
        selection = session.scalar(
            select(ProjectWorkflowSelection).where(
                ProjectWorkflowSelection.project_id == project["id"],
                ProjectWorkflowSelection.selector_capability == "image",
            )
        )
        assert selection is not None
        assert selection.mode == "revision"
        assert selection.workflow_revision_id == revision_id

    cleared = await client.patch(
        f"/api/projects/{project['id']}",
        json={"image_workflow_revision_id": None},
    )
    assert cleared.status_code == 200
    with SessionLocal() as session:
        assert (
            session.scalar(
                select(ProjectWorkflowSelection).where(
                    ProjectWorkflowSelection.project_id == project["id"],
                    ProjectWorkflowSelection.selector_capability == "image",
                )
            )
            is None
        )
