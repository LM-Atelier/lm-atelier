from __future__ import annotations

import hashlib
from collections.abc import Callable, Generator

import pytest
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.models import (
    ModelInstall,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowRevision,
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


def _revision(session: Session, suffix: str = "one") -> WorkflowRevision:
    definition = WorkflowDefinition(
        id=f"workflow_{suffix}", name=f"Workflow {suffix}", operation="text_to_image"
    )
    session.add(definition)
    session.flush()
    revision = WorkflowRevision(
        id=f"wfrev_{suffix}",
        workflow_id=definition.id,
        version=1,
        dependency_contract_sha256="a" * 64,
    )
    session.add(revision)
    session.flush()
    return revision


def _slot(
    revision: WorkflowRevision,
    *,
    suffix: str = "primary",
    ordinal: int = 0,
) -> WorkflowDependencySlot:
    return WorkflowDependencySlot(
        id=f"wfslot_{suffix}",
        workflow_revision_id=revision.id,
        name=suffix,
        resource_kind="runtime",
        required=True,
        satisfaction="any_of",
        requirements_json=[{"key": "default", "constraints": {}}],
        contract_sha256="b" * 64,
        ordinal=ordinal,
    )


def _activation(
    revision: WorkflowRevision,
    *,
    suffix: str = "one",
    active: bool = True,
    state: str = "ready",
) -> WorkflowActivation:
    return WorkflowActivation(
        id=f"wfact_{suffix}",
        workflow_revision_id=revision.id,
        resolver_version="resolver-v1",
        dependency_contract_sha256="a" * 64,
        binding_sha256=hashlib.sha256(suffix.encode("utf-8")).hexdigest(),
        state=state,
        is_active=active,
    )


def _runtime_binding(
    activation: WorkflowActivation,
    slot: WorkflowDependencySlot,
    *,
    suffix: str = "one",
) -> WorkflowDependencyBinding:
    return WorkflowDependencyBinding(
        id=f"wfbind_{suffix}",
        workflow_revision_id=activation.workflow_revision_id,
        workflow_activation_id=activation.id,
        workflow_dependency_slot_id=slot.id,
        requirement_key="default",
        runtime_key="comfyui-v1",
        mount_json={"target_folder": "models"},
        resource_identity_json={"kind": "runtime", "engine": "comfyui"},
        resource_identity_sha256="d" * 64,
    )


def test_round_trips_ordered_slots_activation_and_runtime_binding(session: Session) -> None:
    revision = _revision(session)
    second = _slot(revision, suffix="secondary", ordinal=1)
    first = _slot(revision, ordinal=0)
    activation = _activation(revision)
    session.add_all([second, first, activation])
    session.flush()
    session.add(_runtime_binding(activation, first))
    session.commit()
    session.expire_all()

    restored = session.get(WorkflowRevision, revision.id)
    assert restored is not None
    assert [slot.name for slot in restored.dependency_slots] == ["primary", "secondary"]
    assert restored.activations[0].state == "ready"
    assert restored.activations[0].is_active is True
    assert restored.activations[0].bindings[0].runtime_key == "comfyui-v1"
    assert restored.activations[0].bindings[0].mount_json == {"target_folder": "models"}


@pytest.mark.parametrize(
    "candidate",
    [
        lambda revision: _slot(revision, suffix="primary", ordinal=1),
        lambda revision: _slot(revision, suffix="secondary", ordinal=0),
        lambda revision: WorkflowDependencySlot(
            id="wfslot_negative",
            workflow_revision_id=revision.id,
            name="negative",
            resource_kind="runtime",
            required=True,
            satisfaction="any_of",
            requirements_json=[],
            contract_sha256="b" * 64,
            ordinal=-1,
        ),
        lambda revision: WorkflowDependencySlot(
            id="wfslot_invalid_mode",
            workflow_revision_id=revision.id,
            name="invalid_mode",
            resource_kind="runtime",
            required=True,
            satisfaction="one_of",
            requirements_json=[],
            contract_sha256="b" * 64,
            ordinal=1,
        ),
    ],
)
def test_slot_constraints_reject_duplicate_or_invalid_rows(
    session: Session,
    candidate: Callable[[WorkflowRevision], WorkflowDependencySlot],
) -> None:
    revision = _revision(session)
    session.add(_slot(revision))
    session.commit()
    session.add(candidate(revision))

    with pytest.raises(IntegrityError):
        session.commit()


def test_activation_constraints_enforce_one_ready_active_binding(session: Session) -> None:
    revision = _revision(session)
    session.add(_activation(revision, suffix="alpha"))
    session.commit()
    session.add(_activation(revision, suffix="beta"))

    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize(
    ("state", "invalidated"),
    [("stale", False), ("disabled", False), ("ready", True)],
)
def test_active_activation_must_be_ready_and_not_invalidated(
    session: Session, state: str, invalidated: bool
) -> None:
    from local_lm.domain import utcnow

    revision = _revision(session)
    activation = _activation(revision, state=state)
    activation.invalidated_at = utcnow() if invalidated else None
    session.add(activation)

    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize("locator_count", [0, 2])
def test_binding_requires_exactly_one_local_locator(session: Session, locator_count: int) -> None:
    revision = _revision(session)
    slot = _slot(revision)
    activation = _activation(revision)
    session.add_all([slot, activation])
    session.flush()
    binding = _runtime_binding(activation, slot)
    if locator_count == 0:
        binding.runtime_key = None
    else:
        install = ModelInstall(
            id="model_locator",
            name="Locator",
            role="image",
            engine="comfyui",
            local_path="C:/models/locator",
        )
        session.add(install)
        session.flush()
        binding.model_install_id = install.id
    session.add(binding)

    with pytest.raises(IntegrityError):
        session.commit()


def test_binding_blocks_resource_and_slot_deletion_but_revision_cascades(
    session: Session,
) -> None:
    revision = _revision(session)
    slot = _slot(revision)
    activation = _activation(revision)
    install = ModelInstall(
        id="model_bound",
        name="Bound",
        role="image",
        engine="comfyui",
        local_path="C:/models/bound",
    )
    session.add_all([slot, activation, install])
    session.flush()
    binding = _runtime_binding(activation, slot)
    binding.runtime_key = None
    binding.model_install_id = install.id
    session.add(binding)
    session.commit()

    with pytest.raises(IntegrityError):
        session.execute(delete(ModelInstall).where(ModelInstall.id == install.id))
        session.commit()
    session.rollback()
    with pytest.raises(IntegrityError):
        session.execute(delete(WorkflowDependencySlot).where(WorkflowDependencySlot.id == slot.id))
        session.commit()
    session.rollback()

    session.expire_all()
    loaded_revision = session.get(WorkflowRevision, revision.id)
    assert loaded_revision is not None
    assert [item.id for item in loaded_revision.dependency_slots] == [slot.id]
    session.delete(loaded_revision)
    session.commit()

    assert session.scalar(select(func.count(WorkflowDependencySlot.id))) == 0
    assert session.scalar(select(func.count(WorkflowActivation.id))) == 0
    assert session.scalar(select(func.count(WorkflowDependencyBinding.id))) == 0
    assert session.get(ModelInstall, install.id) is not None


def _assert_binding_rejects_activation_and_slot_from_different_revisions(
    session: Session,
    binding_revision_source: str,
) -> None:
    first_revision = _revision(session, "first")
    second_revision = _revision(session, "second")
    first_slot = _slot(first_revision)
    second_activation = _activation(second_revision, suffix="second")
    session.add_all([first_slot, second_activation])
    session.flush()
    binding = _runtime_binding(second_activation, first_slot, suffix="cross_revision")
    binding.workflow_revision_id = (
        first_revision.id if binding_revision_source == "slot" else second_revision.id
    )
    session.add(binding)

    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize("binding_revision_source", ["slot", "activation"])
def test_binding_rejects_both_cross_revision_mismatch_directions(
    session: Session, binding_revision_source: str
) -> None:
    _assert_binding_rejects_activation_and_slot_from_different_revisions(
        session, binding_revision_source
    )


@pytest.mark.parametrize(
    ("target", "invalid_digest"),
    [
        ("revision", "A" * 64),
        ("revision", "a" * 63),
        ("slot", "g" * 64),
        ("activation_contract", "A" * 64),
        ("activation_binding", "g" * 64),
        ("binding_identity", "A" * 64),
        ("binding_identity", "a" * 65),
    ],
)
def test_sha256_columns_require_lowercase_hex(
    session: Session, target: str, invalid_digest: str
) -> None:
    revision = _revision(session)
    if target == "revision":
        revision.dependency_contract_sha256 = invalid_digest
    elif target == "slot":
        slot = _slot(revision)
        slot.contract_sha256 = invalid_digest
        session.add(slot)
    elif target in {"activation_contract", "activation_binding"}:
        activation = _activation(revision)
        if target == "activation_contract":
            activation.dependency_contract_sha256 = invalid_digest
        else:
            activation.binding_sha256 = invalid_digest
        session.add(activation)
    else:
        slot = _slot(revision)
        activation = _activation(revision)
        session.add_all([slot, activation])
        session.flush()
        binding = _runtime_binding(activation, slot)
        binding.resource_identity_sha256 = invalid_digest
        session.add(binding)

    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize(
    ("table", "column", "row_id"),
    [
        ("workflow_dependency_slots", "required", "wfslot_primary"),
        ("workflow_activations", "is_active", "wfact_one"),
    ],
)
def test_boolean_columns_reject_non_boolean_integers(
    session: Session, table: str, column: str, row_id: str
) -> None:
    revision = _revision(session)
    session.add_all([_slot(revision), _activation(revision)])
    session.commit()

    with pytest.raises(IntegrityError):
        session.connection().exec_driver_sql(
            f"UPDATE {table} SET {column} = 2 WHERE id = ?", (row_id,)
        )
