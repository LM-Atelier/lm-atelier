from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.domain import Operation
from local_lm.models import (
    Chat,
    ChatWorkflowSelection,
    ModelProfile,
    Project,
    ProjectWorkflowSelection,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowProfileCompatibility,
    WorkflowRevision,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.workflow_compatibility import (
    AUTO_PROFILE_ID,
    WorkflowCompatibilityConflict,
    WorkflowSelectionInvalid,
    compatibility_definition_id,
    compatibility_family_id,
    copy_chat_workflow_selections,
    mirror_legacy_chat_workflow_selections,
    reconcile_legacy_workflow_compatibility,
    resolve_chat_workflow_selection,
    resolve_project_workflow_selection,
    retire_legacy_profile_workflow,
)


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
    engine.dispose()


def test_reconcile_is_deterministic_and_preserves_selection_modes(session: Session) -> None:
    chat_profile = ModelProfile(
        id="profile_chat",
        name="Writer",
        use_case="long-form writing",
        role="chat",
        engine="llama.cpp",
        is_default=True,
    )
    image_profile = ModelProfile(
        id="profile_image",
        name="Illustrator",
        use_case="editorial illustration",
        role="image",
        engine="comfyui",
        is_default=True,
    )
    chat = Chat(
        id="chat_legacy",
        title="Legacy choices",
        active_chat_profile_id=None,
        active_vision_profile_id=AUTO_PROFILE_ID,
        active_image_profile_id=image_profile.id,
        active_video_profile_id="profile_missing",
    )
    definition = WorkflowDefinition(
        id="workflow_project_pin",
        name="Pinned workflow",
        operation="text_to_image",
    )
    revision = WorkflowRevision(
        id="wfrev_project_pin",
        definition=definition,
        version=1,
        trusted=True,
    )
    project = Project(
        id="project_legacy",
        name="Pinned project",
        image_workflow_revision_id=revision.id,
    )
    session.add_all([chat_profile, image_profile, chat, definition, revision])
    session.commit()
    session.add(project)
    session.commit()

    first = reconcile_legacy_workflow_compatibility(session)
    session.commit()
    second = reconcile_legacy_workflow_compatibility(session)
    session.commit()

    assert first.profiles == 2
    assert first.families_created == 2
    assert first.definitions_created == 3
    assert first.preferences_created == 3
    assert first.chat_selections_created == 2
    assert first.project_selections_created == 1
    assert second.families_created == 0
    assert second.definitions_created == 0
    assert second.preferences_created == 0
    assert second.chat_selections_created == 0
    assert second.project_selections_created == 0

    chat_family_id = compatibility_family_id(chat_profile.id)
    image_family_id = compatibility_family_id(image_profile.id)
    assert len(chat_family_id) == 64
    assert len(compatibility_definition_id(image_profile.id, "create")) == 64
    chat_family = session.get(WorkflowFamily, chat_family_id)
    assert chat_family is not None
    assert chat_family.name == "Writer"
    assert chat_family.use_case == "long-form writing"
    assert {
        (item.variant_key, item.operation)
        for item in session.scalars(
            select(WorkflowDefinition).where(WorkflowDefinition.family_id == image_family_id)
        )
    } == {("create", "text_to_image"), ("edit", "image_to_image")}
    assert {
        item.selector_capability
        for item in session.scalars(
            select(WorkflowPreference).where(
                WorkflowPreference.workflow_family_id == chat_family_id,
                WorkflowPreference.is_default.is_(True),
            )
        )
    } == {"chat", "vision"}

    assert resolve_chat_workflow_selection(session, chat, "chat").mode == "legacy"
    vision = resolve_chat_workflow_selection(session, chat, "vision")
    assert (vision.mode, vision.profile_id) == ("automatic", AUTO_PROFILE_ID)
    image = resolve_chat_workflow_selection(session, chat, "image")
    assert (image.mode, image.profile_id, image.workflow_family_id) == (
        "family",
        image_profile.id,
        image_family_id,
    )
    assert resolve_chat_workflow_selection(session, chat, "video").mode == "legacy"
    project_image = resolve_project_workflow_selection(session, project, "image")
    assert (project_image.mode, project_image.workflow_revision_id) == (
        "revision",
        revision.id,
    )
    assert resolve_project_workflow_selection(session, project, "video").mode == "legacy"


def test_reconcile_updates_generated_metadata_after_legacy_profile_edit(session: Session) -> None:
    profile = ModelProfile(
        id="profile_edit",
        name="Original",
        use_case="portraits",
        role="image",
        engine="comfyui",
    )
    session.add(profile)
    session.commit()
    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    mapping = session.get(WorkflowProfileCompatibility, profile.id)
    assert mapping is not None
    original_fingerprint = mapping.source_fingerprint_sha256

    profile.name = "Renamed"
    profile.use_case = "product photography"
    reconcile_legacy_workflow_compatibility(session)
    session.commit()

    family = session.get(WorkflowFamily, compatibility_family_id(profile.id))
    assert family is not None
    assert (family.name, family.use_case) == ("Renamed", "product photography")
    assert mapping.source_fingerprint_sha256 != original_fingerprint


def test_real_workflow_default_takes_precedence_over_legacy_default(session: Session) -> None:
    profile = ModelProfile(
        id="profile_default",
        name="Legacy default",
        role="image",
        engine="comfyui",
        is_default=True,
    )
    real_family = WorkflowFamily(id="wffamily_real", name="Chosen workflow")
    real_default = WorkflowPreference(
        id="wfpref_real",
        workflow_family_id=real_family.id,
        selector_capability="image",
        is_default=True,
    )
    session.add_all([profile, real_family, real_default])
    session.commit()

    reconcile_legacy_workflow_compatibility(session)
    session.commit()

    generated_default_count = session.scalar(
        select(func.count(WorkflowPreference.id)).where(
            WorkflowPreference.workflow_family_id == compatibility_family_id(profile.id),
            WorkflowPreference.is_default.is_(True),
        )
    )
    assert generated_default_count == 0
    assert real_default.is_default is True


def test_duplicate_legacy_defaults_choose_a_deterministic_family(session: Session) -> None:
    updated_at = datetime(2026, 8, 3, tzinfo=UTC)
    first = ModelProfile(
        id="profile_default_a",
        name="First default",
        role="image",
        engine="comfyui",
        is_default=True,
        updated_at=updated_at,
    )
    second = ModelProfile(
        id="profile_default_z",
        name="Second default",
        role="image",
        engine="comfyui",
        is_default=True,
        updated_at=updated_at,
    )
    session.add_all([second, first])
    session.commit()

    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    reconcile_legacy_workflow_compatibility(session)
    session.commit()

    defaults = list(
        session.scalars(
            select(WorkflowPreference).where(
                WorkflowPreference.selector_capability == "image",
                WorkflowPreference.is_default.is_(True),
            )
        ).all()
    )
    assert [preference.workflow_family_id for preference in defaults] == [
        compatibility_family_id(second.id)
    ]


def test_explicit_legacy_vision_profile_maps_to_the_chat_family(session: Session) -> None:
    profile = ModelProfile(
        id="profile_vision",
        name="Vision",
        role="chat",
        engine="llama.cpp",
    )
    chat = Chat(
        id="chat_vision",
        title="Vision",
        active_vision_profile_id=profile.id,
    )
    session.add_all([profile, chat])
    session.commit()

    reconcile_legacy_workflow_compatibility(session)
    session.commit()

    selection = resolve_chat_workflow_selection(session, chat, "vision")
    assert (selection.mode, selection.profile_id, selection.workflow_family_id) == (
        "family",
        profile.id,
        compatibility_family_id(profile.id),
    )


def test_deterministic_family_collision_is_refused(session: Session) -> None:
    profile = ModelProfile(
        id="profile_collision",
        name="Collision",
        role="chat",
        engine="llama.cpp",
    )
    session.add_all(
        [
            profile,
            WorkflowFamily(
                id=compatibility_family_id(profile.id),
                name="Unrelated family",
            ),
        ]
    )
    session.commit()

    with pytest.raises(WorkflowCompatibilityConflict, match="family id collision"):
        reconcile_legacy_workflow_compatibility(session)


def test_unbound_family_selection_is_rejected_instead_of_guessing(session: Session) -> None:
    family = WorkflowFamily(id="wffamily_unbound", name="Unbound")
    chat = Chat(id="chat_unbound", title="Unbound")
    selection = ChatWorkflowSelection(
        id="wfsel_unbound",
        chat_id=chat.id,
        selector_capability="image",
        mode="family",
        workflow_family_id=family.id,
    )
    session.add_all([family, chat, selection])
    session.commit()

    with pytest.raises(WorkflowSelectionInvalid, match="family_not_bound"):
        resolve_chat_workflow_selection(session, chat, "image")


def test_new_workflow_selection_survives_reconciliation(session: Session) -> None:
    profile = ModelProfile(
        id="profile_selected",
        name="Selected",
        role="image",
        engine="comfyui",
    )
    chat = Chat(
        id="chat_selected",
        title="Selected",
        active_image_profile_id=profile.id,
    )
    session.add_all([profile, chat])
    session.commit()
    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    selection = session.scalar(
        select(ChatWorkflowSelection).where(
            ChatWorkflowSelection.chat_id == chat.id,
            ChatWorkflowSelection.selector_capability == "image",
        )
    )
    assert selection is not None
    selection.mode = "automatic"
    selection.workflow_family_id = None
    session.commit()

    reconcile_legacy_workflow_compatibility(session)
    session.commit()

    assert selection.mode == "automatic"
    assert selection.workflow_family_id is None


def test_selection_mode_constraints_are_fail_closed(session: Session) -> None:
    chat = Chat(id="chat_constraint", title="Constraint")
    session.add(chat)
    session.flush()
    session.add(
        ChatWorkflowSelection(
            id="wfsel_constraint",
            chat_id=chat.id,
            selector_capability="image",
            mode="automatic",
            workflow_family_id="wffamily_missing",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()

    session.rollback()
    project = Project(id="project_constraint", name="Constraint")
    session.add(project)
    session.flush()
    session.add(
        ProjectWorkflowSelection(
            id="wfsel_project_constraint",
            project_id=project.id,
            selector_capability="image",
            mode="revision",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_copy_prefers_workflow_selection_and_legacy_mirror_is_explicit(session: Session) -> None:
    profile = ModelProfile(
        id="profile_copy",
        name="Copy",
        role="image",
        engine="comfyui",
    )
    source = Chat(
        id="chat_copy_source",
        title="Source",
        active_image_profile_id=profile.id,
    )
    target = Chat(
        id="chat_copy_target",
        title="Target",
        active_image_profile_id=profile.id,
    )
    session.add_all([profile, source, target])
    session.commit()
    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    source_selection = session.scalar(
        select(ChatWorkflowSelection).where(
            ChatWorkflowSelection.chat_id == source.id,
            ChatWorkflowSelection.selector_capability == "image",
        )
    )
    target_selection = session.scalar(
        select(ChatWorkflowSelection).where(
            ChatWorkflowSelection.chat_id == target.id,
            ChatWorkflowSelection.selector_capability == "image",
        )
    )
    assert source_selection is not None
    assert target_selection is not None
    session.delete(target_selection)
    source_selection.mode = "automatic"
    source_selection.workflow_family_id = None
    session.commit()

    copy_chat_workflow_selections(session, source, target)
    session.commit()

    copied = resolve_chat_workflow_selection(session, target, "image")
    assert copied.mode == "automatic"
    assert target.active_image_profile_id == profile.id

    target.active_image_profile_id = None
    mirror_legacy_chat_workflow_selections(session, target, ["image"])
    session.commit()
    assert resolve_chat_workflow_selection(session, target, "image").mode == "legacy"
    target.active_image_profile_id = AUTO_PROFILE_ID
    mirror_legacy_chat_workflow_selections(session, target, ["image"])
    session.commit()
    assert resolve_chat_workflow_selection(session, target, "image").mode == "automatic"


def test_retiring_profile_resets_chats_and_removes_empty_generated_family(
    session: Session,
) -> None:
    profile = ModelProfile(
        id="profile_retire",
        name="Retire",
        role="image",
        engine="comfyui",
    )
    chat = Chat(
        id="chat_retire",
        title="Retire",
        active_image_profile_id=profile.id,
    )
    session.add_all([profile, chat])
    session.commit()
    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    family_id = compatibility_family_id(profile.id)

    retire_legacy_profile_workflow(session, profile)
    session.delete(profile)
    session.flush()
    reconcile_legacy_workflow_compatibility(session)
    session.commit()

    assert chat.active_image_profile_id == AUTO_PROFILE_ID
    assert resolve_chat_workflow_selection(session, chat, "image").mode == "automatic"
    assert session.get(WorkflowProfileCompatibility, profile.id) is None
    assert session.get(WorkflowFamily, family_id) is None
    assert (
        session.scalar(
            select(func.count(WorkflowDefinition.id)).where(
                WorkflowDefinition.family_id == family_id
            )
        )
        == 0
    )


def test_retiring_profile_preserves_family_with_executable_provenance(session: Session) -> None:
    profile = ModelProfile(
        id="profile_provenance",
        name="Provenance",
        role="image",
        engine="comfyui",
    )
    session.add(profile)
    session.commit()
    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    family_id = compatibility_family_id(profile.id)
    definition = session.get(
        WorkflowDefinition,
        compatibility_definition_id(profile.id, "create"),
    )
    assert definition is not None
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        trusted=True,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    session.commit()

    retire_legacy_profile_workflow(session, profile)
    session.delete(profile)
    session.commit()

    assert session.get(WorkflowProfileCompatibility, profile.id) is None
    assert session.get(WorkflowFamily, family_id) is not None
    assert session.get(WorkflowRevision, revision.id) is not None


def test_orchestrator_prefers_workflow_selection_over_stale_legacy_profile(
    session: Session,
) -> None:
    default = ModelProfile(
        id="profile_orchestrator_default",
        name="Default",
        role="chat",
        engine="llama.cpp",
        is_default=True,
    )
    stale = ModelProfile(
        id="profile_orchestrator_stale",
        name="Stale explicit",
        role="chat",
        engine="llama.cpp",
    )
    chat = Chat(
        id="chat_orchestrator",
        title="Orchestrator",
        active_chat_profile_id=stale.id,
    )
    session.add_all([default, stale, chat])
    session.commit()
    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    selection = session.scalar(
        select(ChatWorkflowSelection).where(
            ChatWorkflowSelection.chat_id == chat.id,
            ChatWorkflowSelection.selector_capability == "chat",
        )
    )
    assert selection is not None
    selection.mode = "automatic"
    selection.workflow_family_id = None
    session.commit()

    selected, provenance = ConversationOrchestrator._profile_for_operation(
        session,
        chat,
        Operation.TEXT,
        "ordinary request",
    )

    assert selected is not None and selected.id == default.id
    assert provenance["mode"] == "auto"


def test_orchestrator_project_pin_read_prefers_workflow_selection(session: Session) -> None:
    definition = WorkflowDefinition(
        id="workflow_stale_pin",
        name="Stale pin",
        operation="text_to_image",
    )
    revision = WorkflowRevision(
        id="wfrev_stale_pin",
        definition=definition,
        version=1,
        trusted=True,
    )
    session.add_all([definition, revision])
    session.commit()
    project = Project(
        id="project_stale_pin",
        name="Stale pin",
        image_workflow_revision_id=revision.id,
    )
    session.add(project)
    session.commit()
    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    selection = session.scalar(
        select(ProjectWorkflowSelection).where(
            ProjectWorkflowSelection.project_id == project.id,
            ProjectWorkflowSelection.selector_capability == "image",
        )
    )
    assert selection is not None
    selection.mode = "automatic"
    selection.workflow_revision_id = None
    session.commit()

    assert (
        ConversationOrchestrator._project_workflow_pin_id(
            session,
            project.id,
            Operation.TEXT_TO_IMAGE,
        )
        is None
    )
