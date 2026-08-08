from __future__ import annotations

from collections.abc import Generator

import pytest
from httpx2 import AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.models import (
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.project_dependencies import (
    PortableDependencies,
    PortableWorkflow,
    PortableWorkflowRevision,
    install_dependency_manifest,
)
from local_lm.workflow_ownership import (
    WorkflowOwnershipConflict,
    ensure_workflow_family_ownership,
    reconcile_workflow_family_ownership,
    workflow_family_id,
)
from local_lm.workflow_package_drafts import workflow_package_draft_dependencies


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
    engine.dispose()


def _workflow(
    session: Session,
    operation: str,
    *,
    suffix: str,
    capabilities: list[str] | None = None,
    draft: bool = False,
    name: str | None = None,
) -> tuple[WorkflowDefinition, WorkflowRevision]:
    definition = WorkflowDefinition(
        id=f"workflow_{suffix}",
        name=name or f"Workflow {suffix}",
        operation=operation,
        description=f"Use case {suffix}",
    )
    revision = WorkflowRevision(
        id=f"wfrev_{suffix}",
        definition=definition,
        version=1,
        engine="mock",
        api_graph_json={"node": {"class_type": "Mock"}},
        dependencies_json=(workflow_package_draft_dependencies("a" * 64) if draft else {}),
        capabilities_json=capabilities or [],
        trusted=True,
        artifact_sha256="b" * 64,
        dependency_contract_sha256="c" * 64,
    )
    session.add_all([definition, revision])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return definition, revision


@pytest.mark.parametrize(
    ("operation", "variant", "capability"),
    [
        ("text_to_image", "create", "image"),
        ("image_to_image", "edit", "image"),
        ("text_to_video", "create", "video"),
        ("image_to_video", "animate", "video"),
    ],
)
def test_finalized_workflow_gets_a_stable_selectable_family(
    session: Session,
    operation: str,
    variant: str,
    capability: str,
) -> None:
    definition, revision = _workflow(session, operation, suffix=operation)
    original_identity = (
        definition.id,
        revision.id,
        revision.artifact_sha256,
        revision.dependency_contract_sha256,
        revision.api_graph_json,
    )

    family = ensure_workflow_family_ownership(session, definition)
    session.flush()

    assert family is not None
    assert family.id == workflow_family_id(definition.id)
    assert family.name == definition.name
    assert family.description == definition.description
    assert family.use_case == ""
    assert family.tags_json == []
    assert definition.family_id == family.id
    assert definition.variant_key == variant
    preference = session.scalar(
        select(WorkflowPreference).where(
            WorkflowPreference.workflow_family_id == family.id,
            WorkflowPreference.selector_capability == capability,
        )
    )
    assert preference is not None and preference.enabled and not preference.is_default
    assert (
        definition.id,
        revision.id,
        revision.artifact_sha256,
        revision.dependency_contract_sha256,
        revision.api_graph_json,
    ) == original_identity


def test_text_workflow_waits_for_an_executable_profile_binding_contract(
    session: Session,
) -> None:
    definition, _ = _workflow(
        session,
        "text",
        suffix="text",
        capabilities=["vision"],
    )

    assert ensure_workflow_family_ownership(session, definition) is None
    assert definition.family_id is None


def test_revision_must_be_the_definitions_current_revision(session: Session) -> None:
    definition, current = _workflow(session, "text_to_image", suffix="current")
    foreign, foreign_revision = _workflow(
        session, "text_to_image", suffix="foreign", capabilities=["vision"]
    )
    stale = WorkflowRevision(
        id="wfrev_stale",
        workflow_id=definition.id,
        version=2,
        engine="mock",
        api_graph_json={"node": {"class_type": "Stale"}},
        trusted=True,
    )
    session.add(stale)
    session.flush()

    with pytest.raises(WorkflowOwnershipConflict, match="not the current revision"):
        ensure_workflow_family_ownership(session, definition, foreign_revision)
    with pytest.raises(WorkflowOwnershipConflict, match="not the current revision"):
        ensure_workflow_family_ownership(session, definition, stale)

    assert definition.family_id is None
    assert foreign.family_id is None
    assert current.id == definition.current_revision_id


def test_reconcile_is_idempotent_and_preserves_user_managed_fields(session: Session) -> None:
    definition, _ = _workflow(session, "text_to_image", suffix="preserved")
    first = reconcile_workflow_family_ownership(session)
    family = session.get(WorkflowFamily, definition.family_id)
    assert family is not None
    preference = session.scalar(
        select(WorkflowPreference).where(WorkflowPreference.workflow_family_id == family.id)
    )
    assert preference is not None
    family.name = "My preferred label"
    family.use_case = "line drawings"
    family.tags_json = ["custom"]
    preference.enabled = False
    preference.sort_order = 9
    session.flush()

    second = reconcile_workflow_family_ownership(session)

    assert first.families_created == first.assignments_created == 1
    assert first.preferences_created == 1
    assert second.families_created == second.assignments_created == 0
    assert second.preferences_created == 0
    assert family.name == "My preferred label"
    assert family.use_case == "line drawings"
    assert family.tags_json == ["custom"]
    assert preference.enabled is False
    assert preference.sort_order == 9


def test_draft_is_hidden_until_its_current_revision_is_finalized(session: Session) -> None:
    definition, revision = _workflow(
        session,
        "text_to_image",
        suffix="draft",
        draft=True,
    )

    assert ensure_workflow_family_ownership(session, definition) is None
    assert definition.family_id is None
    revision.dependencies_json = {}
    family = ensure_workflow_family_ownership(session, definition, revision)

    assert family is not None
    assert definition.family_id == family.id


def test_same_name_workflows_remain_distinct(session: Session) -> None:
    first, _ = _workflow(session, "text_to_image", suffix="same_a", name="Same name")
    second, _ = _workflow(session, "text_to_image", suffix="same_b", name="Same name")

    first_family = ensure_workflow_family_ownership(session, first)
    second_family = ensure_workflow_family_ownership(session, second)

    assert first_family is not None and second_family is not None
    assert first_family.id != second_family.id


def test_deterministic_family_collision_fails_closed(session: Session) -> None:
    definition, _ = _workflow(session, "text_to_image", suffix="collision")
    session.add(
        WorkflowFamily(
            id=workflow_family_id(definition.id),
            name="Unrelated family",
        )
    )
    session.flush()

    with pytest.raises(WorkflowOwnershipConflict, match="family id collision"):
        ensure_workflow_family_ownership(session, definition)


def test_existing_family_assignment_is_never_regrouped(session: Session) -> None:
    family = WorkflowFamily(id="wffamily_existing", name="Existing family")
    definition, _ = _workflow(session, "image_to_image", suffix="existing")
    definition.family_id = family.id
    definition.variant_key = "custom-edit"
    preference = WorkflowPreference(
        id="wfpref_existing",
        family=family,
        selector_capability="image",
        enabled=False,
        is_default=False,
        sort_order=7,
    )
    session.add_all([family, preference])
    session.flush()

    resolved = ensure_workflow_family_ownership(session, definition)

    assert resolved is family
    assert definition.variant_key == "custom-edit"
    assert preference.enabled is False
    assert preference.sort_order == 7


@pytest.mark.parametrize(("enabled", "archived"), [(False, False), (True, True)])
def test_missing_preference_respects_unavailable_existing_family(
    session: Session,
    enabled: bool,
    archived: bool,
) -> None:
    family = WorkflowFamily(
        id=f"wffamily_unavailable_{enabled}_{archived}",
        name="Unavailable",
        enabled=enabled,
        archived=archived,
    )
    definition, _ = _workflow(
        session,
        "text_to_image",
        suffix=f"unavailable_{enabled}_{archived}",
    )
    definition.family_id = family.id
    definition.variant_key = "create"
    session.add(family)
    session.flush()

    ensure_workflow_family_ownership(session, definition)
    session.flush()
    preference = session.scalar(
        select(WorkflowPreference).where(WorkflowPreference.workflow_family_id == family.id)
    )

    assert preference is not None
    assert preference.enabled is False


def test_missing_preference_refuses_ambiguous_existing_family(session: Session) -> None:
    family = WorkflowFamily(id="wffamily_ambiguous", name="Ambiguous")
    first, _ = _workflow(session, "text_to_image", suffix="ambiguous_first")
    second, _ = _workflow(session, "text_to_image", suffix="ambiguous_second")
    first.family_id = family.id
    first.variant_key = "create-a"
    second.family_id = family.id
    second.variant_key = "create-b"
    session.add(family)
    session.flush()

    with pytest.raises(WorkflowOwnershipConflict, match="ambiguous"):
        ensure_workflow_family_ownership(session, first)


def test_existing_preference_does_not_hide_ambiguous_family(session: Session) -> None:
    family = WorkflowFamily(id="wffamily_ambiguous_existing", name="Ambiguous")
    first, _ = _workflow(session, "text_to_image", suffix="ambiguous_pref_first")
    second, _ = _workflow(session, "text_to_image", suffix="ambiguous_pref_second")
    first.family_id = family.id
    first.variant_key = "create-a"
    second.family_id = family.id
    second.variant_key = "create-b"
    preference = WorkflowPreference(
        id="wfpref_ambiguous_existing",
        family=family,
        selector_capability="image",
    )
    session.add_all([family, preference])
    session.flush()

    with pytest.raises(WorkflowOwnershipConflict, match="ambiguous"):
        ensure_workflow_family_ownership(session, first)


def test_project_import_owns_new_and_exact_matched_workflows(session: Session) -> None:
    revision = PortableWorkflowRevision(
        source_id="source_revision",
        source_version=1,
        engine="mock",
        api_graph={"node": {"class_type": "Mock"}},
        trusted=True,
    )
    dependencies = PortableDependencies(
        workflows=[
            PortableWorkflow(
                source_id="source_workflow",
                name="Portable workflow",
                operation="text_to_image",
                current_revision_source_id=revision.source_id,
                revisions=[revision],
            )
        ]
    )

    first = install_dependency_manifest(session, dependencies)
    session.flush()
    imported = session.get(WorkflowDefinition, first.workflow_ids["source_workflow"])
    assert imported is not None and imported.family_id is not None
    family_id = imported.family_id
    revision_id = first.revision_ids["source_revision"]

    second = install_dependency_manifest(session, dependencies)
    session.flush()

    assert second.workflow_ids["source_workflow"] == imported.id
    assert second.revision_ids["source_revision"] == revision_id
    assert imported.family_id == family_id
    assert session.scalar(select(WorkflowFamily).where(WorkflowFamily.id == family_id)) is not None


@pytest.mark.asyncio
async def test_authored_cloned_and_imported_workflows_are_selectable(
    client: AsyncClient,
) -> None:
    created_response = await client.post(
        "/api/workflows",
        json={
            "name": "Selectable workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "Mock"}},
            "trusted": True,
        },
    )
    assert created_response.status_code == 201
    created = created_response.json()
    bundle = (await client.get(f"/api/workflows/{created['id']}/export")).json()
    cloned = (
        await client.post(
            f"/api/workflows/{created['id']}/clone",
            json={"name": "Selectable clone"},
        )
    ).json()
    bundle["name"] = "Selectable import"
    imported_response = await client.post("/api/workflows/import", json=bundle)
    assert imported_response.status_code == 201
    imported = imported_response.json()

    families_response = await client.get("/api/workflow-families?selector_capability=image")
    assert families_response.status_code == 200
    families = families_response.json()

    def family_for(workflow_id: str) -> dict[str, object]:
        return next(
            family
            for family in families
            if any(variant["id"] == workflow_id for variant in family["variants"])
        )

    created_family = family_for(created["id"])
    cloned_family = family_for(cloned["id"])
    imported_family = family_for(imported["id"])
    assert len({created_family["id"], cloned_family["id"], imported_family["id"]}) == 3
    assert created_family["variants"][0]["readiness"] == "ready"
    assert imported_family["variants"][0]["readiness"] == "review_required"

    chat = (await client.post("/api/chats", json={"title": "Selectable"})).json()
    selected = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={
            "mode": "family",
            "workflow_family_id": created_family["id"],
        },
    )
    assert selected.status_code == 200
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw a blue cup", "mode": "image"},
    )
    assert accepted.status_code == 202
    assert accepted.json()["run"]["workflow_revision_id"] == created["current_revision_id"]
