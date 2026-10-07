"""Apply recipe eligibility on family, exact-revision and compatibility paths."""

from typing import Any

import pytest
from sqlalchemy.orm import Session
from test_automatic_image_edit_selection import _orchestrator
from test_workflow_selection import _family_revision
from test_workflow_selection import session as session
from test_workflow_selection_eligibility import _second_variant

from local_lm.domain import Operation
from local_lm.models import Chat, ChatWorkflowSelection, ModelProfile, WorkflowRevision
from local_lm.schemas import TurnRequest
from local_lm.workflow_compatibility import ensure_legacy_profile_workflow
from local_lm.workflow_selection import WorkflowFamilySelectionError


def _chat(session: Session, mode: str, family_id: str | None = None) -> Chat:
    chat = Chat(title="Example recipe selection")
    session.add(chat)
    session.flush()
    if mode == "default":
        return chat
    session.add(
        ChatWorkflowSelection(
            chat_id=chat.id,
            selector_capability="image",
            mode=mode,
            workflow_family_id=family_id,
        )
    )
    session.flush()
    return chat


@pytest.mark.parametrize("mode", ["default", "family", "automatic"])
@pytest.mark.parametrize("ordered", [False, True])
def test_execution_does_not_fall_through_when_recipe_eligibility_refuses(
    session: Session,
    mode: str,
    ordered: bool,
) -> None:
    family, _, _ = _family_revision(session, "Selected", is_default=True)
    _family_revision(session, "Alternative")
    chat = _chat(session, mode, family.id if mode == "family" else None)
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _orchestrator()._execution_for_turn(
            session,
            chat,
            Operation.TEXT_TO_IMAGE,
            "A blue square.",
            TurnRequest(text="A blue square."),
            ordered=ordered,
            revision_eligibility=lambda revision: "recipe-incompatible",
        )
    assert raised.value.reason == (
        "no_ready_workflow" if mode == "automatic" else "recipe-incompatible"
    )


@pytest.mark.parametrize("mode", ["family", "automatic"])
def test_execution_filters_variants_before_ambiguity_and_ranking(
    session: Session,
    mode: str,
) -> None:
    family, _, first = _family_revision(session, "Selected", is_default=True)
    second = _second_variant(session, family.id)
    chat = _chat(session, mode, family.id if mode == "family" else None)
    _, selection, revision = _orchestrator()._execution_for_turn(
        session,
        chat,
        Operation.TEXT_TO_IMAGE,
        "A blue square.",
        TurnRequest(text="A blue square."),
        revision_eligibility=lambda value: "recipe-incompatible" if value == first else None,
    )
    assert revision is second
    assert selection["workflow_family_id"] == family.id


@pytest.mark.parametrize("ordered", [False, True])
def test_an_exact_turn_revision_must_pass_recipe_eligibility(
    session: Session,
    ordered: bool,
) -> None:
    _, _, revision = _family_revision(session, "Exact", is_default=True)
    chat = _chat(session, "default")
    seen: list[str | None] = []

    def refuse(value: WorkflowRevision | None) -> str:
        seen.append(value.id if value else None)
        return "recipe-incompatible"

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _orchestrator()._execution_for_turn(
            session,
            chat,
            Operation.TEXT_TO_IMAGE,
            "A blue square.",
            TurnRequest(text="A blue square.", workflow_revision_id=revision.id),
            ordered=ordered,
            revision_eligibility=refuse,
        )
    assert raised.value.reason == "recipe-incompatible"
    assert seen == [revision.id]


def test_a_compatibility_family_cannot_bypass_recipe_refusal(session: Session) -> None:
    profile = ModelProfile(name="Example profile", role="chat", engine="mock")
    session.add(profile)
    session.flush()
    family = ensure_legacy_profile_workflow(session, profile)
    assert family is not None
    chat = Chat(title="Example chat")
    session.add(chat)
    session.flush()
    session.add(
        ChatWorkflowSelection(
            chat_id=chat.id,
            selector_capability="chat",
            mode="family",
            workflow_family_id=family.id,
        )
    )
    session.flush()
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _orchestrator()._execution_for_turn(
            session,
            chat,
            Operation.TEXT,
            "A blue square.",
            TurnRequest(text="A blue square."),
            revision_eligibility=lambda revision: (
                "recipe-revision-required" if revision is None else None
            ),
        )
    assert raised.value.reason == "recipe-revision-required"


def test_an_explicit_profile_cannot_bypass_recipe_refusal(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, revision = _family_revision(session, "Exact")
    profile = ModelProfile(name="Example profile", role="image", engine="comfyui")
    session.add(profile)
    session.flush()
    chat = _chat(session, "default")
    orchestrator = _orchestrator()
    monkeypatch.setattr(orchestrator, "_workflow_for_operation", lambda *args, **kwargs: revision)
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        orchestrator._execution_for_turn(
            session,
            chat,
            Operation.TEXT_TO_IMAGE,
            "A blue square.",
            TurnRequest(text="A blue square.", profile_id=profile.id),
            revision_eligibility=lambda value: "recipe-incompatible",
        )
    assert raised.value.reason == "recipe-incompatible"


def test_exact_text_selection_checks_eligibility_after_existing_admission(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, revision = _family_revision(session, "Text", operation=Operation.TEXT, capability="chat")
    profile = ModelProfile(name="Example profile", role="chat", engine="mock")
    chat = Chat(title="Example text")
    session.add_all([profile, chat])
    session.flush()
    events: list[str] = []

    def exact(*args: Any, **kwargs: Any) -> tuple[WorkflowRevision, None, ModelProfile]:
        events.append("exact-admission")
        return revision, None, profile

    def eligibility(value: WorkflowRevision | None) -> str:
        assert value is revision
        events.append("recipe-eligibility")
        return "recipe-incompatible"

    monkeypatch.setattr("local_lm.orchestrator.resolve_exact_workflow_revision", exact)
    with pytest.raises(WorkflowFamilySelectionError):
        _orchestrator()._execution_for_turn(
            session,
            chat,
            Operation.TEXT,
            "A blue square.",
            TurnRequest(text="A blue square.", workflow_revision_id=revision.id),
            revision_eligibility=eligibility,
        )
    assert events == ["exact-admission", "recipe-eligibility"]
