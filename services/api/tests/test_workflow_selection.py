from __future__ import annotations

from collections.abc import Callable, Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.domain import Operation
from local_lm.models import (
    ModelInstall,
    ModelProfile,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.workflow_compatibility import ensure_legacy_profile_workflow
from local_lm.workflow_selection import (
    WorkflowFamilySelectionError,
    resolve_workflow_family,
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


def _family_revision(
    session: Session,
    name: str,
    *,
    operation: Operation = Operation.TEXT_TO_IMAGE,
    capability: str = "image",
    use_case: str = "",
    is_default: bool = False,
    trusted: bool = True,
    variant_key: str = "create",
    capabilities: list[str] | None = None,
    dependency_contract_sha256: str | None = None,
    active: bool = True,
) -> tuple[WorkflowFamily, WorkflowDefinition, WorkflowRevision]:
    family = WorkflowFamily(name=name, use_case=use_case)
    definition = WorkflowDefinition(
        family=family,
        variant_key=variant_key,
        name=f"{name} {variant_key}",
        operation=operation.value,
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="comfyui",
        api_graph_json={"node": {"class_type": "TestOutput"}},
        capabilities_json=capabilities or [],
        dependency_contract_sha256=dependency_contract_sha256,
        trusted=trusted,
    )
    preference = WorkflowPreference(
        family=family,
        selector_capability=capability,
        is_default=is_default,
    )
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    if dependency_contract_sha256 is not None and active:
        session.add(
            WorkflowActivation(
                workflow_revision_id=revision.id,
                resolver_version="resolver-v1",
                dependency_contract_sha256=dependency_contract_sha256,
                binding_sha256="b" * 64,
                state="ready",
                is_active=True,
                details_json={"launch_sha256": "c" * 64},
            )
        )
    session.flush()
    return family, definition, revision


def test_explicit_family_resolves_exact_current_operation_variant(session: Session) -> None:
    family, definition, revision = _family_revision(
        session,
        "Flexible image",
        operation=Operation.IMAGE_TO_IMAGE,
        variant_key="edit",
        dependency_contract_sha256="a" * 64,
    )

    result = resolve_workflow_family(
        session,
        capability="image",
        operation=Operation.IMAGE_TO_IMAGE,
        mode="explicit",
        workflow_family_id=family.id,
        engine="comfyui",
    )

    assert result.workflow_family_id == family.id
    assert result.workflow_definition_id == definition.id
    assert result.workflow_revision_id == revision.id
    assert result.workflow_activation_id is not None
    assert result.profile_id is None
    assert result.compatibility is False


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda family, _definition, _revision: setattr(family, "enabled", False),
            "family_disabled",
        ),
        (
            lambda family, _definition, _revision: setattr(family, "archived", True),
            "family_archived",
        ),
        (
            lambda _family, _definition, revision: setattr(revision, "trusted", False),
            "revision_untrusted",
        ),
    ],
)
def test_explicit_family_fails_closed(
    session: Session,
    mutation: Callable[[WorkflowFamily, WorkflowDefinition, WorkflowRevision], None],
    reason: str,
) -> None:
    family, definition, revision = _family_revision(session, "Selected")
    mutation(family, definition, revision)
    session.flush()

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        resolve_workflow_family(
            session,
            capability="image",
            operation=Operation.TEXT_TO_IMAGE,
            mode="explicit",
            workflow_family_id=family.id,
            engine="comfyui",
        )

    assert raised.value.reason == reason
    assert raised.value.workflow_family_id == family.id


def test_explicit_family_never_falls_back_to_another_ready_family(session: Session) -> None:
    selected, _definition, _revision = _family_revision(session, "Unavailable")
    selected.enabled = False
    fallback, _, _ = _family_revision(session, "Ready")
    session.flush()

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        resolve_workflow_family(
            session,
            capability="image",
            operation=Operation.TEXT_TO_IMAGE,
            mode="explicit",
            workflow_family_id=selected.id,
        )

    assert raised.value.reason == "family_disabled"
    assert raised.value.workflow_family_id != fallback.id


def test_automatic_ranks_ready_families_by_family_use_case(session: Session) -> None:
    matching, _, _ = _family_revision(
        session,
        "Technical illustrator",
        use_case="architectural diagrams and technical illustration",
    )
    _family_revision(session, "General art", use_case="portraits and landscapes")
    unavailable, _, unavailable_revision = _family_revision(
        session,
        "Exact but unavailable",
        use_case="python architectural diagrams",
    )
    unavailable_revision.trusted = False
    session.flush()

    prompt = "Draw a Python architectural diagram for a service"
    result = resolve_workflow_family(
        session,
        capability="image",
        operation=Operation.TEXT_TO_IMAGE,
        mode="automatic",
        prompt=prompt,
        engine="comfyui",
    )

    assert result.workflow_family_id == matching.id
    assert {"architectural", "image"}.issubset(result.matched_terms)
    assert result.score > 0
    assert prompt not in str(result.provenance())
    assert result.workflow_family_id != unavailable.id


def test_automatic_uses_default_then_sort_order_for_equal_scores(session: Session) -> None:
    first, _, _ = _family_revision(session, "First")
    second, _, _ = _family_revision(session, "Second")
    first_preference = first.preferences[0]
    second_preference = second.preferences[0]
    first_preference.sort_order = -10
    second_preference.is_default = True
    second_preference.sort_order = 50
    session.flush()

    default_result = resolve_workflow_family(
        session,
        capability="image",
        operation=Operation.TEXT_TO_IMAGE,
        mode="automatic",
    )
    assert default_result.workflow_family_id == second.id

    second_preference.is_default = False
    session.flush()
    ordered_result = resolve_workflow_family(
        session,
        capability="image",
        operation=Operation.TEXT_TO_IMAGE,
        mode="automatic",
    )
    assert ordered_result.workflow_family_id == first.id


def test_default_refuses_an_unready_default_instead_of_substituting(session: Session) -> None:
    selected, _, revision = _family_revision(session, "Default", is_default=True)
    revision.trusted = False
    other, _, _ = _family_revision(session, "Other")
    session.flush()

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        resolve_workflow_family(
            session,
            capability="image",
            operation=Operation.TEXT_TO_IMAGE,
            mode="default",
        )

    assert raised.value.reason == "revision_untrusted"
    assert raised.value.workflow_family_id == selected.id
    assert raised.value.workflow_family_id != other.id


def test_same_operation_variants_require_unambiguous_capability_match(session: Session) -> None:
    family, _, mask_revision = _family_revision(
        session,
        "Editor",
        operation=Operation.IMAGE_TO_IMAGE,
        variant_key="mask",
        capabilities=["mask"],
    )
    second = WorkflowDefinition(
        family=family,
        variant_key="depth",
        name="Depth edit",
        operation=Operation.IMAGE_TO_IMAGE.value,
    )
    depth_revision = WorkflowRevision(
        definition=second,
        version=1,
        engine="comfyui",
        api_graph_json={"node": {"class_type": "DepthOutput"}},
        capabilities_json=["depth"],
        trusted=True,
    )
    session.add_all([second, depth_revision])
    session.flush()
    second.current_revision_id = depth_revision.id
    session.flush()

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        resolve_workflow_family(
            session,
            capability="image",
            operation=Operation.IMAGE_TO_IMAGE,
            mode="explicit",
            workflow_family_id=family.id,
        )
    assert raised.value.reason == "variant_ambiguous"

    narrowed = resolve_workflow_family(
        session,
        capability="image",
        operation=Operation.IMAGE_TO_IMAGE,
        mode="explicit",
        workflow_family_id=family.id,
        required_capabilities=["mask"],
    )
    assert narrowed.workflow_revision_id == mask_revision.id


def test_contract_revision_requires_current_ready_activation(session: Session) -> None:
    family, _, _ = _family_revision(
        session,
        "Needs activation",
        dependency_contract_sha256="a" * 64,
        active=False,
    )

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        resolve_workflow_family(
            session,
            capability="image",
            operation=Operation.TEXT_TO_IMAGE,
            mode="explicit",
            workflow_family_id=family.id,
        )

    assert raised.value.reason == "activation_not_ready"


def test_text_family_resolves_the_unique_profile_bound_by_its_activation(
    session: Session,
) -> None:
    install = ModelInstall(
        name="Chat install",
        role="chat",
        engine="llama.cpp",
        local_path="chat-model.gguf",
        active=True,
    )
    session.add(install)
    session.flush()
    profile = ModelProfile(
        name="Chat profile",
        role="chat",
        engine="llama.cpp",
        model_install_id=install.id,
    )
    family = WorkflowFamily(name="Chat workflow")
    definition = WorkflowDefinition(
        family=family,
        variant_key="text",
        name="Text",
        operation=Operation.TEXT.value,
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="llama.cpp",
        dependency_contract_sha256="a" * 64,
        trusted=True,
    )
    preference = WorkflowPreference(family=family, selector_capability="chat")
    session.add_all([profile, family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    activation = WorkflowActivation(
        workflow_revision_id=revision.id,
        resolver_version="resolver-v1",
        dependency_contract_sha256="a" * 64,
        binding_sha256="b" * 64,
        state="ready",
        is_active=True,
        details_json={"launch_sha256": "c" * 64},
    )
    slot = WorkflowDependencySlot(
        workflow_revision_id=revision.id,
        name="primary",
        resource_kind="model_profile",
        required=True,
        satisfaction="all_of",
        requirements_json=[{"key": "default", "constraints": {}}],
        contract_sha256="d" * 64,
        ordinal=0,
    )
    session.add_all([activation, slot])
    session.flush()
    session.add(
        WorkflowDependencyBinding(
            workflow_revision_id=revision.id,
            workflow_activation_id=activation.id,
            workflow_dependency_slot_id=slot.id,
            requirement_key="default",
            model_profile_id=profile.id,
            resource_identity_sha256="e" * 64,
        )
    )
    session.flush()

    result = resolve_workflow_family(
        session,
        capability="chat",
        operation=Operation.TEXT,
        mode="explicit",
        workflow_family_id=family.id,
        engine="llama.cpp",
    )

    assert result.profile_id == profile.id
    assert result.workflow_revision_id == revision.id
    assert result.operation == Operation.TEXT


def test_compatibility_family_uses_bound_profile_and_legacy_revision(session: Session) -> None:
    install = ModelInstall(
        name="Legacy image",
        role="image",
        engine="comfyui",
        local_path="legacy-image",
        active=True,
    )
    session.add(install)
    session.flush()
    profile = ModelProfile(
        name="Legacy image",
        role="image",
        engine="comfyui",
        model_install_id=install.id,
    )
    session.add(profile)
    session.flush()
    family = ensure_legacy_profile_workflow(session, profile)
    legacy_definition = WorkflowDefinition(
        name="Legacy executable",
        operation=Operation.TEXT_TO_IMAGE.value,
    )
    revision = WorkflowRevision(
        definition=legacy_definition,
        version=1,
        engine="comfyui",
        api_graph_json={"node": {"class_type": "LegacyOutput"}},
        trusted=True,
    )
    session.add_all([legacy_definition, revision])
    session.flush()
    legacy_definition.current_revision_id = revision.id

    def legacy_resolver(
        _session: Session,
        selected_profile: ModelProfile,
        operation: Operation,
    ) -> WorkflowRevision | None:
        assert selected_profile.id == profile.id
        assert operation == Operation.TEXT_TO_IMAGE
        return revision

    result = resolve_workflow_family(
        session,
        capability="image",
        operation=Operation.TEXT_TO_IMAGE,
        mode="explicit",
        workflow_family_id=family.id,
        engine="comfyui",
        legacy_revision_resolver=legacy_resolver,
    )

    assert result.compatibility is True
    assert result.profile_id == profile.id
    assert result.workflow_family_id == family.id
    assert result.workflow_revision_id == revision.id


def test_selector_operation_mismatch_is_rejected_before_selection(session: Session) -> None:
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        resolve_workflow_family(
            session,
            capability="video",
            operation=Operation.TEXT_TO_IMAGE,
            mode="automatic",
        )

    assert raised.value.reason == "selector_operation_mismatch"
