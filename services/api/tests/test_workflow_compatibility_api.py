from __future__ import annotations

from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from local_lm.db import SessionLocal
from local_lm.models import (
    ChatWorkflowSelection,
    ProjectWorkflowSelection,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.workflow_compatibility import AUTO_PROFILE_ID, compatibility_family_id


def _ready_image_family(
    session: Session,
    *,
    name: str,
    use_case: str = "",
) -> tuple[WorkflowFamily, WorkflowRevision]:
    family = WorkflowFamily(name=name, use_case=use_case)
    definition = WorkflowDefinition(
        family=family,
        variant_key="create",
        name=f"{name} create",
        operation="text_to_image",
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="mock",
        api_graph_json={"node": {"class_type": "MockImage"}},
        trusted=True,
    )
    preference = WorkflowPreference(family=family, selector_capability="image")
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return family, revision


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


async def test_explicit_chat_family_queues_its_current_operation_revision(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Workflow family"})).json()
    with SessionLocal() as session:
        family, revision = _ready_image_family(session, name="Selected family")
        selection = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat["id"],
                ChatWorkflowSelection.selector_capability == "image",
            )
        )
        assert selection is not None
        selection.mode = "family"
        selection.workflow_family_id = family.id
        session.commit()
        family_id = family.id
        revision_id = revision.id

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw a blue ceramic bowl", "mode": "image"},
    )

    assert accepted.status_code == 202, accepted.json()
    run = accepted.json()["run"]
    assert run["workflow_revision_id"] == revision_id
    assert run["profile_id"] is None
    selected = run["provenance_json"]["model_selection"]
    assert selected["mode"] == "explicit"
    assert selected["workflow_family_id"] == family_id
    assert selected["workflow_revision_id"] == revision_id


async def test_automatic_chat_selection_ranks_ready_workflow_families(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Workflow Auto"})).json()
    with SessionLocal() as session:
        selected_family, selected_revision = _ready_image_family(
            session,
            name="Diagram specialist",
            use_case="architectural diagrams and technical illustration",
        )
        _ready_image_family(
            session,
            name="Portrait specialist",
            use_case="portraits and landscapes",
        )
        selection = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat["id"],
                ChatWorkflowSelection.selector_capability == "image",
            )
        )
        assert selection is not None and selection.mode == "automatic"
        session.commit()
        family_id = selected_family.id
        revision_id = selected_revision.id

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw an architectural diagram", "mode": "image"},
    )

    assert accepted.status_code == 202, accepted.json()
    run = accepted.json()["run"]
    assert run["workflow_revision_id"] == revision_id
    selected = run["provenance_json"]["model_selection"]
    assert selected["mode"] == "auto"
    assert selected["workflow_family_id"] == family_id
    assert "architectural" in selected["matched_terms"]


async def test_project_family_overrides_the_chat_family_without_changing_its_revision(
    client: AsyncClient,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Family project"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Project family", "project_id": project["id"]},
        )
    ).json()
    with SessionLocal() as session:
        project_family, project_revision = _ready_image_family(
            session,
            name="Project workflow",
        )
        chat_family, _chat_revision = _ready_image_family(
            session,
            name="Chat workflow",
        )
        chat_selection = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat["id"],
                ChatWorkflowSelection.selector_capability == "image",
            )
        )
        assert chat_selection is not None
        chat_selection.mode = "family"
        chat_selection.workflow_family_id = chat_family.id
        session.add(
            ProjectWorkflowSelection(
                project_id=project["id"],
                selector_capability="image",
                mode="family",
                workflow_family_id=project_family.id,
            )
        )
        session.commit()
        project_family_id = project_family.id
        project_revision_id = project_revision.id

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw a project diagram", "mode": "image"},
    )

    assert accepted.status_code == 202, accepted.json()
    run = accepted.json()["run"]
    assert run["workflow_revision_id"] == project_revision_id
    assert run["provenance_json"]["model_selection"]["workflow_family_id"] == (project_family_id)
