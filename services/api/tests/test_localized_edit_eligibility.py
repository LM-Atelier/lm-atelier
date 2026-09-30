"""Automatic localized edits require source-conditioned instruction workflows."""

from unittest.mock import Mock

import pytest
from sqlalchemy.orm import Session
from test_automatic_image_edit_selection import (
    GLOBAL,
    LOCALIZED,
    _chat,
    _edit_family,
    _orchestrator,
)
from test_automatic_image_edit_selection import session as session

from local_lm.domain import Operation
from local_lm.models import (
    Chat,
    ModelInstall,
    ModelProfile,
    WorkflowProfileCompatibility,
    WorkflowRevision,
)
from local_lm.workflow_selection import WorkflowFamilySelectionError
from local_lm.workflow_use_case_errors import workflow_use_case_error


@pytest.mark.parametrize("aggregate", [False, True])
def test_localized_edit_refusal_explains_the_needed_workflow(aggregate: bool) -> None:
    code = "workflow-instruction-edit-required"
    error = WorkflowFamilySelectionError(
        capability="image",
        operation=Operation.IMAGE_TO_IMAGE,
        reason="no_ready_workflow" if aggregate else code,
        candidate_reasons=(code,) if aggregate else (),
    )
    translated = workflow_use_case_error(error)
    assert translated is not None
    actual_code, message = translated
    assert actual_code == code
    assert "instruction-edit workflow" in message
    assert "choose a workflow explicitly" in message


@pytest.mark.parametrize("branch", ["ordinary", "unused", "ui-only"])
def test_auto_refuses_strength_only_localized_edits_without_profile_fallback(
    session: Session, branch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _edit_family(
        session,
        "Color changes",
        instruction=False,
        is_default=True,
        use_case="change sweater color",
        unused_branch=branch == "unused",
        ui_only_saved_branch=branch == "ui-only",
    )
    orchestrator = _orchestrator()
    fallback = Mock(side_effect=AssertionError("Profile fallback must not run"))
    monkeypatch.setattr(orchestrator, "_profile_for_operation", fallback)
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        orchestrator._profile_and_workflow_for_operation(
            session, _chat(session, "automatic"), Operation.IMAGE_TO_IMAGE, LOCALIZED
        )
    assert raised.value.reason == "no_ready_workflow"
    assert raised.value.candidate_reasons == ("workflow-instruction-edit-required",)
    fallback.assert_not_called()


def test_localized_eligibility_removes_strength_variant_before_ambiguity(session: Session) -> None:
    strength = _edit_family(
        session, "Color changes", instruction=False, is_default=True, use_case="change color"
    )
    instruction = _edit_family(
        session, "Instruction", instruction=True, is_default=False, use_case="change color"
    )
    instruction.definition.variant_key = "instruction"
    instruction.definition.family = strength.definition.family
    session.flush()
    _, selected, revision = _orchestrator()._profile_and_workflow_for_operation(
        session, _chat(session, "automatic"), Operation.IMAGE_TO_IMAGE, LOCALIZED
    )
    assert revision == instruction
    assert selected["workflow_family_id"] == strength.definition.family_id


@pytest.mark.parametrize("declaration", ["declared", "unknown"])
@pytest.mark.parametrize("instruction", [True, False])
def test_installed_declaration_neither_vetoes_structure_nor_replaces_it(
    session: Session, declaration: str, instruction: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = _edit_family(
        session, "Installed edit", instruction=instruction, is_default=True, use_case="change color"
    )
    install = ModelInstall(
        name="Installed edit",
        role="image",
        engine="comfyui",
        local_path="neutral-model",
        active=True,
        manifest_json={"instruction_edit_capability": declaration},
    )
    session.add(install)
    session.flush()
    profile = ModelProfile(
        name="Installed edit", role="image", engine="comfyui", model_install_id=install.id
    )
    session.add(profile)
    session.flush()
    session.add(
        WorkflowProfileCompatibility(
            workflow_family_id=revision.definition.family_id,
            model_profile_id=profile.id,
            source_fingerprint_sha256="a" * 64,
        )
    )
    session.flush()
    orchestrator = _orchestrator()
    monkeypatch.setattr(orchestrator, "_workflow_for_operation", Mock(return_value=revision))
    chat = _chat(session, "automatic")
    if instruction:
        selected_profile, _, selected = orchestrator._profile_and_workflow_for_operation(
            session, chat, Operation.IMAGE_TO_IMAGE, LOCALIZED
        )
        assert selected_profile == profile
        assert selected == revision
    else:
        with pytest.raises(WorkflowFamilySelectionError) as raised:
            orchestrator._profile_and_workflow_for_operation(
                session, chat, Operation.IMAGE_TO_IMAGE, LOCALIZED
            )
        assert raised.value.candidate_reasons == ("workflow-instruction-edit-required",)


def test_accepting_recipe_check_does_not_admit_a_strength_only_localized_edit(
    session: Session,
) -> None:
    _edit_family(session, "Strength", instruction=False, is_default=True, use_case="color")
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _orchestrator()._profile_and_workflow_for_operation(
            session,
            _chat(session, "automatic"),
            Operation.IMAGE_TO_IMAGE,
            LOCALIZED,
            revision_eligibility=lambda _: None,
        )
    assert raised.value.candidate_reasons == ("workflow-instruction-edit-required",)


def test_instruction_structure_cannot_override_an_existing_eligibility_refusal(
    session: Session,
) -> None:
    rejected = _edit_family(
        session, "Instruction", instruction=True, is_default=True, use_case="color"
    )
    seen: list[WorkflowRevision | None] = []

    def eligibility(revision: WorkflowRevision | None) -> str:
        seen.append(revision)
        return "recipe-incompatible"

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _orchestrator()._profile_and_workflow_for_operation(
            session,
            _chat(session, "automatic"),
            Operation.IMAGE_TO_IMAGE,
            LOCALIZED,
            revision_eligibility=eligibility,
        )
    assert raised.value.candidate_reasons == ("recipe-incompatible",)
    assert seen == [rejected]


@pytest.mark.parametrize("mode", ["family", "legacy"])
def test_explicit_and_default_strength_workflows_remain_available(
    session: Session, mode: str
) -> None:
    revision = _edit_family(
        session, "Chosen edit", instruction=False, is_default=True, use_case="color"
    )
    if mode == "family":
        chat = _chat(session, mode, revision.definition.family_id)
    else:
        chat = Chat(title="Default edit")
        session.add(chat)
        session.flush()
    _, _, selected = _orchestrator()._profile_and_workflow_for_operation(
        session,
        chat,
        Operation.IMAGE_TO_IMAGE,
        LOCALIZED,
    )
    assert selected == revision


def test_automatic_global_transformations_still_allow_strength_workflows(session: Session) -> None:
    revision = _edit_family(
        session, "Watercolor", instruction=False, is_default=True, use_case="watercolor"
    )
    _, _, selected = _orchestrator()._profile_and_workflow_for_operation(
        session, _chat(session, "automatic"), Operation.IMAGE_TO_IMAGE, GLOBAL
    )
    assert selected == revision


@pytest.mark.parametrize("unavailable", ["absent", "different-engine", "untrusted"])
def test_localized_auto_never_uses_legacy_fallback_when_instruction_workflows_are_unavailable(
    session: Session, unavailable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if unavailable != "absent":
        revision = _edit_family(
            session, "Instruction", instruction=True, is_default=True, use_case="color"
        )
        if unavailable == "different-engine":
            revision.engine = "other"
        else:
            revision.trusted = False
        session.flush()
    orchestrator = _orchestrator()
    fallback = Mock(side_effect=AssertionError("Profile fallback must not run"))
    monkeypatch.setattr(orchestrator, "_profile_for_operation", fallback)
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        orchestrator._profile_and_workflow_for_operation(
            session, _chat(session, "automatic"), Operation.IMAGE_TO_IMAGE, LOCALIZED
        )
    assert raised.value.reason == "no_ready_workflow"
    fallback.assert_not_called()
