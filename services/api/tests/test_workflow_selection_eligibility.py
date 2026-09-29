"""Input eligibility narrows variants before ambiguity and automatic ranking."""

from typing import Any

import pytest
from sqlalchemy.orm import Session
from test_workflow_selection import _family_revision
from test_workflow_selection import session as session

from local_lm.domain import Operation
from local_lm.models import ModelProfile, WorkflowDefinition, WorkflowRevision
from local_lm.workflow_compatibility import ensure_legacy_profile_workflow
from local_lm.workflow_selection import WorkflowFamilySelectionError, resolve_workflow_family


def _resolve(session: Session, **arguments: Any) -> Any:
    return resolve_workflow_family(
        session,
        **{
            "capability": "image",
            "operation": Operation.TEXT_TO_IMAGE,
            "engine": "comfyui",
            "mode": "automatic",
            **arguments,
        },
    )


@pytest.mark.parametrize("mode", ["explicit", "default"])
def test_an_ineligible_selected_family_never_falls_back(session: Session, mode: str) -> None:
    selected, _, rejected = _family_revision(session, "Selected", is_default=True)
    _family_revision(session, "Alternative", use_case="image")
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _resolve(
            session,
            mode=mode,
            workflow_family_id=selected.id if mode == "explicit" else None,
            revision_eligibility=lambda revision: (
                "recipe-incompatible" if revision == rejected else None
            ),
        )
    assert raised.value.reason == "recipe-incompatible"
    assert raised.value.workflow_family_id == selected.id


def test_automatic_excludes_ineligible_workflows_before_every_ranking_preference(
    session: Session,
) -> None:
    _, _, rejected = _family_revision(session, "Exact image", use_case="image", is_default=True)
    accepted, _, revision = _family_revision(session, "Other", use_case="general")
    result = _resolve(
        session,
        prompt="image",
        preferred_revision=lambda value: value == rejected,
        revision_eligibility=lambda value: "recipe-incompatible" if value == rejected else None,
    )
    assert result.workflow_family_id == accepted.id
    assert result.workflow_revision_id == revision.id


def _second_variant(session: Session, family_id: str) -> WorkflowRevision:
    definition = WorkflowDefinition(
        family_id=family_id, name="Second", variant_key="second", operation="text_to_image"
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="comfyui",
        api_graph_json={"node": {"class_type": "TestOutput"}},
        trusted=True,
    )
    session.add_all([definition, revision])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return revision


def test_eligibility_narrows_variants_inside_the_explicit_family(session: Session) -> None:
    family, _, first = _family_revision(session, "Multiple variants")
    second = _second_variant(session, family.id)
    result = _resolve(
        session,
        mode="explicit",
        workflow_family_id=family.id,
        revision_eligibility=lambda revision: "recipe-incompatible" if revision == first else None,
    )
    assert result.workflow_family_id == family.id
    assert result.workflow_revision_id == second.id


def test_two_eligible_variants_remain_ambiguous(session: Session) -> None:
    family, _, _ = _family_revision(session, "Multiple variants")
    _second_variant(session, family.id)
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _resolve(
            session,
            mode="explicit",
            workflow_family_id=family.id,
            revision_eligibility=lambda revision: None,
        )
    assert raised.value.reason == "variant_ambiguous"


def test_no_eligible_automatic_family_is_a_refusal(session: Session) -> None:
    _family_revision(session, "Default", is_default=True)
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _resolve(session, revision_eligibility=lambda revision: "recipe-incompatible")
    assert raised.value.reason == "no_ready_workflow"


@pytest.mark.parametrize(
    ("trusted", "active", "reason"),
    [(False, True, "revision_untrusted"), (True, False, "activation_not_ready")],
)
def test_eligibility_cannot_admit_an_untrusted_or_unready_revision(
    session: Session, trusted: bool, active: bool, reason: str
) -> None:
    family, _, _ = _family_revision(
        session, "Unavailable", trusted=trusted, active=active, dependency_contract_sha256="a" * 64
    )
    calls: list[str] = []

    def eligible(revision: WorkflowRevision | None) -> None:
        calls.append(revision.id if revision else "missing")

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _resolve(
            session, mode="explicit", workflow_family_id=family.id, revision_eligibility=eligible
        )
    assert raised.value.reason == reason
    assert calls == []


def test_graphless_compatibility_is_presented_to_the_eligibility_check(session: Session) -> None:
    profile = ModelProfile(name="Chat profile", role="chat", engine="mock")
    session.add(profile)
    session.flush()
    family = ensure_legacy_profile_workflow(session, profile)
    seen: list[WorkflowRevision | None] = []

    def eligibility(revision: WorkflowRevision | None) -> str:
        seen.append(revision)
        return "recipe-revision-required"

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _resolve(
            session,
            capability="chat",
            operation=Operation.TEXT,
            engine="mock",
            mode="explicit",
            workflow_family_id=family.id,
            revision_eligibility=eligibility,
        )
    assert raised.value.reason == "recipe-revision-required"
    assert seen == [None]
    result = _resolve(
        session,
        capability="chat",
        operation=Operation.TEXT,
        engine="mock",
        mode="explicit",
        workflow_family_id=family.id,
    )
    assert result.compatibility is True
    assert result.workflow_revision_id is None


def test_compatibility_revision_must_also_pass_eligibility(session: Session) -> None:
    profile = ModelProfile(name="Image profile", role="image", engine="mock")
    session.add(profile)
    session.flush()
    family = ensure_legacy_profile_workflow(session, profile)
    _, _, revision = _family_revision(session, "Legacy execution")
    revision.engine = "mock"
    session.flush()
    seen: list[str] = []

    def eligibility(value: WorkflowRevision | None) -> str:
        assert value is not None
        seen.append(value.id)
        return "recipe-incompatible"

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _resolve(
            session,
            engine="mock",
            mode="explicit",
            workflow_family_id=family.id,
            legacy_revision_resolver=lambda *_: revision,
            revision_eligibility=eligibility,
        )
    assert raised.value.reason == "recipe-incompatible"
    assert seen == [revision.id]
