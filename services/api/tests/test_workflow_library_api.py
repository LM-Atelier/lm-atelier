from __future__ import annotations

from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import (
    Chat,
    ChatWorkflowSelection,
    Message,
    ModelInstall,
    ModelProfile,
    Project,
    ProjectWorkflowSelection,
    Run,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
)
from local_lm.workflow_compatibility import (
    compatibility_family_id,
    reconcile_legacy_workflow_compatibility,
)


def _family(
    *,
    name: str,
    model: ModelInstall | None = None,
) -> tuple[
    WorkflowFamily,
    WorkflowDefinition,
    WorkflowRevision,
    WorkflowPreference,
    WorkflowActivation | None,
    WorkflowDependencySlot | None,
    WorkflowDependencyBinding | None,
]:
    suffix = name.casefold().replace(" ", "-")
    family = WorkflowFamily(id=f"wffamily_{suffix}", name=name)
    definition = WorkflowDefinition(
        id=f"workflow_{suffix}",
        family=family,
        variant_key="create",
        name=f"{name} create",
        operation="text_to_image",
    )
    revision = WorkflowRevision(
        id=f"wfrev_{suffix}",
        definition=definition,
        version=1,
        engine="mock",
        api_graph_json={"node": {"class_type": "MockImage"}},
        dependency_contract_sha256="a" * 64 if model is not None else None,
        trusted=True,
    )
    preference = WorkflowPreference(
        family=family,
        selector_capability="image",
    )
    if model is None:
        return family, definition, revision, preference, None, None, None
    slot = WorkflowDependencySlot(
        id=f"wfslot_{suffix}",
        revision=revision,
        name="primary",
        resource_kind="model_install",
        required=True,
        satisfaction="all_of",
        requirements_json=[{"key": "primary", "constraints": {}}],
        contract_sha256="b" * 64,
        ordinal=0,
    )
    activation = WorkflowActivation(
        id=f"wfact_{suffix}",
        revision=revision,
        resolver_version="resolver-v1",
        dependency_contract_sha256="a" * 64,
        binding_sha256=("c" if name == "Primary family" else "d") * 64,
        state="ready",
        is_active=True,
    )
    binding = WorkflowDependencyBinding(
        id=f"wfbind_{suffix}",
        workflow_revision_id=revision.id,
        workflow_activation_id=activation.id,
        workflow_dependency_slot_id=slot.id,
        requirement_key="primary",
        model_install_id=model.id,
        resource_identity_json={"kind": "model_install", "id": model.id},
        resource_identity_sha256="e" * 64,
    )
    return family, definition, revision, preference, activation, slot, binding


async def test_family_metadata_defaults_and_archive_guards(client: AsyncClient) -> None:
    with SessionLocal() as session:
        first = _family(name="First workflow")
        second = _family(name="Second workflow")
        session.add_all([*first[:4], *second[:4]])
        session.flush()
        first[1].current_revision_id = first[2].id
        second[1].current_revision_id = second[2].id
        session.commit()
        first_id = first[0].id
        second_id = second[0].id
        second_revision_id = second[2].id

    updated = await client.patch(
        f"/api/workflow-families/{first_id}",
        json={
            "name": "  Portrait maker  ",
            "description": "  Natural portraits  ",
            "use_case": "  candid portraits  ",
            "tags": [" Portrait ", "portrait", " candid "],
        },
    )
    assert updated.status_code == 200, updated.json()
    assert updated.json()["name"] == "Portrait maker"
    assert updated.json()["description"] == "Natural portraits"
    assert updated.json()["use_case"] == "candid portraits"
    assert updated.json()["tags"] == ["Portrait", "candid"]

    first_default = await client.put(
        f"/api/workflow-families/{first_id}/preferences/image",
        json={"enabled": True, "is_default": True, "sort_order": 7},
    )
    assert first_default.status_code == 200
    assert first_default.json()["is_default"] is True
    second_default = await client.put(
        f"/api/workflow-families/{second_id}/preferences/image",
        json={"enabled": True, "is_default": True, "sort_order": 3},
    )
    assert second_default.status_code == 200
    first_card = (await client.get(f"/api/workflow-families/{first_id}")).json()
    assert first_card["preferences"][0]["is_default"] is False

    chat = (await client.post("/api/chats", json={"title": "Family guard"})).json()
    selected = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": first_id},
    )
    assert selected.status_code == 200
    refused = await client.put(
        f"/api/workflow-families/{first_id}/preferences/image",
        json={"enabled": False, "is_default": False, "sort_order": 7},
    )
    assert refused.status_code == 409
    assert refused.json()["code"] == "workflow-preference-in-use"
    assert refused.json()["selector_reference_count"] == 1

    cleared = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "default"},
    )
    assert cleared.status_code == 200
    archived_default = await client.patch(
        f"/api/workflow-families/{second_id}",
        json={"archived": True},
    )
    assert archived_default.status_code == 409
    assert archived_default.json()["code"] == "workflow-family-in-use"
    removed_default = await client.put(
        f"/api/workflow-families/{second_id}/preferences/image",
        json={"enabled": True, "is_default": False, "sort_order": 3},
    )
    assert removed_default.status_code == 200
    archived = await client.patch(
        f"/api/workflow-families/{second_id}",
        json={"archived": True},
    )
    assert archived.status_code == 200
    assert archived.json()["archived"] is True
    assert archived.json()["enabled"] is False
    assert archived.json()["preferences"][0]["enabled"] is False
    assert archived.json()["variants"][0]["current_revision_id"] == second_revision_id

    typo = await client.patch(
        f"/api/workflow-families/{first_id}",
        json={"usecase": "misspelled"},
    )
    assert typo.status_code == 422


async def test_removal_impact_reports_selectors_history_pins_and_shared_dependencies(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        model = ModelInstall(
            name="Shared checkpoint",
            role="image",
            engine="mock",
            local_path="managed/shared-checkpoint",
        )
        session.add(model)
        session.flush()
        primary = _family(name="Primary family", model=model)
        sibling = _family(name="Sibling family", model=model)
        session.add_all([*primary, *sibling])
        session.flush()
        primary[1].current_revision_id = primary[2].id
        sibling[1].current_revision_id = sibling[2].id

        project = Project(
            name="Pinned project",
            image_workflow_revision_id=primary[2].id,
        )
        chat = Chat(title="Impact chat")
        session.add_all([project, chat])
        session.flush()
        session.add_all(
            [
                ChatWorkflowSelection(
                    chat_id=chat.id,
                    selector_capability="image",
                    mode="family",
                    workflow_family_id=primary[0].id,
                ),
                ProjectWorkflowSelection(
                    project_id=project.id,
                    selector_capability="image",
                    mode="revision",
                    workflow_revision_id=primary[2].id,
                ),
            ]
        )
        user = Message(chat_id=chat.id, role="user")
        assistant = Message(chat_id=chat.id, role="assistant", status="pending")
        plan = WorkPlan(chat_id=chat.id, transcript_sequence=1)
        session.add_all([user, assistant, plan])
        session.flush()
        session.add(
            WorkStep(
                plan_id=plan.id,
                ordinal=0,
                operation="text_to_image",
                workflow_revision_id=primary[2].id,
                status="queued",
            )
        )
        session.add(
            Run(
                chat_id=chat.id,
                user_message_id=user.id,
                assistant_message_id=assistant.id,
                operation="text_to_image",
                status="running",
                workflow_revision_id=primary[2].id,
            )
        )
        session.commit()
        family_id = primary[0].id
        primary_workflow_id = primary[1].id
        sibling_family_id = sibling[0].id
        sibling_workflow_id = sibling[1].id
        sibling_revision_id = sibling[2].id
        revision_id = primary[2].id
        model_id = model.id

    made_default = await client.put(
        f"/api/workflow-families/{family_id}/preferences/image",
        json={"enabled": True, "is_default": True, "sort_order": 0},
    )
    assert made_default.status_code == 200, made_default.json()

    response = await client.get(f"/api/workflow-families/{family_id}/removal-impact")

    assert response.status_code == 200, response.json()
    impact = response.json()
    assert impact["family_id"] == family_id
    assert impact["removal_strategy"] == "archive"
    assert impact["archive_blocked"] is True
    assert impact["revision_count"] == 1
    assert impact["current_revision_count"] == 1
    assert impact["chat_selection_count"] == 1
    assert impact["project_selection_count"] == 0
    assert impact["project_revision_pin_count"] == 1
    assert impact["active_run_count"] == 1
    assert impact["queued_step_count"] == 1
    assert impact["historical_run_count"] == 1
    assert impact["active_activation_count"] == 1
    assert impact["default_for"] == ["image"]
    assert impact["dependencies"] == [
        {
            "resource_kind": "model_install",
            "resource_id": model_id,
            "resource_name": "Shared checkpoint",
            "binding_count": 1,
            "revision_count": 1,
            "current_revision": True,
            "shared": True,
            "other_workflow_count": 1,
            "other_family_ids": [sibling_family_id],
        }
    ]

    consumers = await client.get(f"/api/workflow-dependencies/model_install/{model_id}/consumers")
    assert consumers.status_code == 200, consumers.json()
    assert consumers.json() == {
        "resource_kind": "model_install",
        "resource_id": model_id,
        "resource_name": "Shared checkpoint",
        "consumers": [
            {
                "workflow_id": primary_workflow_id,
                "workflow_name": "Primary family create",
                "workflow_family_id": family_id,
                "workflow_family_name": "Primary family",
                "revision_ids": [revision_id],
                "binding_count": 1,
                "current_revision": True,
            },
            {
                "workflow_id": sibling_workflow_id,
                "workflow_name": "Sibling family create",
                "workflow_family_id": sibling_family_id,
                "workflow_family_name": "Sibling family",
                "revision_ids": [sibling_revision_id],
                "binding_count": 1,
                "current_revision": True,
            },
        ],
    }

    archived = await client.patch(
        f"/api/workflow-families/{family_id}",
        json={"archived": True},
    )
    assert archived.status_code == 409
    with SessionLocal() as session:
        assert session.get(WorkflowRevision, revision_id) is not None
        assert session.get(ModelInstall, model_id) is not None


async def test_compatibility_family_writes_survive_legacy_reconciliation(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/profiles",
        json={
            "name": "Legacy image profile",
            "role": "image",
            "engine": "mock",
        },
    )
    assert created.status_code == 201, created.json()
    profile_id = created.json()["id"]
    family_id = compatibility_family_id(profile_id)

    updated = await client.patch(
        f"/api/workflow-families/{family_id}",
        json={"name": "Portrait workflow", "use_case": "natural portraits"},
    )
    assert updated.status_code == 200, updated.json()
    made_default = await client.put(
        f"/api/workflow-families/{family_id}/preferences/image",
        json={"enabled": True, "is_default": True, "sort_order": 2},
    )
    assert made_default.status_code == 200, made_default.json()

    with SessionLocal() as session:
        profile = session.get(ModelProfile, profile_id)
        assert profile is not None
        assert profile.name == "Portrait workflow"
        assert profile.use_case == "natural portraits"
        assert profile.is_default is True
        reconcile_legacy_workflow_compatibility(session)
        session.commit()

    card = (await client.get(f"/api/workflow-families/{family_id}")).json()
    assert card["name"] == "Portrait workflow"
    assert card["use_case"] == "natural portraits"
    assert card["preferences"][0]["is_default"] is True
