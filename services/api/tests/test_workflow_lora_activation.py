from __future__ import annotations

import hashlib
from collections.abc import Generator
from dataclasses import FrozenInstanceError

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.domain import utcnow
from local_lm.models import (
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowRevision,
)
from local_lm.workflow_activations import WorkflowActivationError
from local_lm.workflow_bindings import (
    ResolvedWorkflowBinding,
    materialize_runtime,
    workflow_activation_binding_sha256,
    workflow_resource_identity_sha256,
)
from local_lm.workflow_dependencies import (
    WorkflowDependencyContract,
    canonical_workflow_dependency_json,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_payload,
    workflow_dependency_slot_sha256,
)
from local_lm.workflow_lora_activation import (
    WORKFLOW_LORA_ACTIVATION_WITNESS_VERSION,
    load_current_workflow_lora_activation_evidence,
    load_recorded_workflow_lora_activation_evidence,
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


def _contract() -> WorkflowDependencyContract:
    return parse_workflow_dependency_contract(
        {
            "version": 1,
            "slots": [
                {
                    "name": "runtime",
                    "resource_kind": "runtime",
                    "required": True,
                    "satisfaction": "all_of",
                    "requirements": [
                        {"key": "decoder", "constraints": {}},
                        {"key": "encoder", "constraints": {}},
                    ],
                }
            ],
        }
    )


def _revision(
    session: Session,
    *,
    suffix: str = "one",
) -> tuple[WorkflowRevision, WorkflowDependencyContract, dict[str, WorkflowDependencySlot]]:
    contract = _contract()
    definition = WorkflowDefinition(
        id=f"workflow_{suffix}",
        name=f"Workflow {suffix}",
        operation="text_to_image",
    )
    revision = WorkflowRevision(
        id=f"wfrev_{suffix}",
        workflow_id=definition.id,
        version=1,
        dependency_contract_sha256=workflow_dependency_contract_sha256(contract),
    )
    session.add_all([definition, revision])
    session.flush()
    rows: dict[str, WorkflowDependencySlot] = {}
    for ordinal, slot in enumerate(contract.slots):
        payload = workflow_dependency_slot_payload(slot)
        row = WorkflowDependencySlot(
            id=f"wfslot_{suffix}_{slot.name}",
            workflow_revision_id=revision.id,
            name=slot.name,
            resource_kind=slot.resource_kind,
            required=slot.required,
            satisfaction=slot.satisfaction,
            requirements_json=payload["requirements"],
            contract_sha256=workflow_dependency_slot_sha256(slot),
            ordinal=ordinal,
        )
        session.add(row)
        rows[slot.name] = row
    session.flush()
    return revision, contract, rows


def _activation(
    session: Session,
    revision: WorkflowRevision,
    contract: WorkflowDependencyContract,
    slots: dict[str, WorkflowDependencySlot],
    *,
    suffix: str,
    active: bool,
) -> WorkflowActivation:
    resolved: list[ResolvedWorkflowBinding] = []
    rows: list[tuple[ResolvedWorkflowBinding, str]] = []
    for requirement_key in ("decoder", "encoder"):
        materialized = materialize_runtime(
            engine="comfyui",
            runtime_build=f"build-{suffix}-{requirement_key}",
            adapter_contract_version=1,
            launch_contract_version="v1",
        )
        materialized.identity["provenance"] = {
            "layers": [suffix, requirement_key],
        }
        digest = workflow_resource_identity_sha256("runtime", materialized.identity)
        binding = ResolvedWorkflowBinding(
            slot_name="runtime",
            requirement_key=requirement_key,
            resource_kind="runtime",
            identity=materialized.identity,
            resource_identity_sha256=digest,
            mount={"paths": [suffix, requirement_key]},
        )
        resolved.append(binding)
        rows.append((binding, f"runtime-{suffix}-{requirement_key}"))
    binding_sha256 = workflow_activation_binding_sha256(contract, resolved)
    launch_sha256 = hashlib.sha256(f"launch:{binding_sha256}".encode()).hexdigest()
    activation = WorkflowActivation(
        id=f"wfact_{suffix}",
        workflow_revision_id=revision.id,
        resolver_version=f"resolver-{suffix}",
        dependency_contract_sha256=workflow_dependency_contract_sha256(contract),
        binding_sha256=binding_sha256,
        state="ready",
        is_active=active,
        details_json={"launch_sha256": launch_sha256},
    )
    session.add(activation)
    session.flush()
    for index, (binding, runtime_key) in enumerate(rows):
        session.add(
            WorkflowDependencyBinding(
                id=f"wfbind_{suffix}_{index}",
                workflow_revision_id=revision.id,
                workflow_activation_id=activation.id,
                workflow_dependency_slot_id=slots[binding.slot_name].id,
                requirement_key=binding.requirement_key,
                runtime_key=runtime_key,
                mount_json=binding.mount,
                resource_identity_json=binding.identity,
                resource_identity_sha256=binding.resource_identity_sha256,
            )
        )
    session.flush()
    return activation


def _ready_pair(
    session: Session,
) -> tuple[WorkflowRevision, WorkflowActivation, WorkflowActivation]:
    revision, contract, slots = _revision(session)
    recorded = _activation(
        session,
        revision,
        contract,
        slots,
        suffix="recorded",
        active=False,
    )
    current = _activation(
        session,
        revision,
        contract,
        slots,
        suffix="current",
        active=True,
    )
    session.commit()
    return revision, recorded, current


def test_current_and_recorded_modes_load_exact_detached_frozen_evidence(
    session: Session,
) -> None:
    revision, recorded, current = _ready_pair(session)

    current_evidence = load_current_workflow_lora_activation_evidence(session, revision.id)
    recorded_evidence = load_recorded_workflow_lora_activation_evidence(
        session,
        revision.id,
        recorded.id,
    )

    assert current_evidence.activation_id == current.id
    assert recorded_evidence.activation_id == recorded.id
    assert recorded_evidence.resolution.complete is True
    assert recorded_evidence.resolution.issues == ()
    assert recorded_evidence.resolution.missing_required_slots == ()
    expected_witness = {
        "version": WORKFLOW_LORA_ACTIVATION_WITNESS_VERSION,
        "workflow_revision_id": revision.id,
        "activation_id": current.id,
        "resolver_version": current.resolver_version,
        "dependency_contract_sha256": current.dependency_contract_sha256,
        "binding_sha256": current.binding_sha256,
        "launch_sha256": current.details_json["launch_sha256"],
    }
    assert (
        current_evidence.activation_witness_sha256
        == hashlib.sha256(canonical_workflow_dependency_json(expected_witness)).hexdigest()
    )
    with pytest.raises(FrozenInstanceError):
        current_evidence.__setattr__("activation_id", "changed")

    binding_row = session.scalar(
        select(WorkflowDependencyBinding).where(
            WorkflowDependencyBinding.workflow_activation_id == current.id
        )
    )
    assert binding_row is not None
    binding_row.mount_json["paths"].append("mutated")
    assert current_evidence.resolution.bindings[0].mount["paths"] == ["current", "decoder"]


def test_resolution_reconstruction_rejects_retained_nested_identity_and_mount_mutation(
    session: Session,
) -> None:
    revision, recorded, _current = _ready_pair(session)
    evidence = load_recorded_workflow_lora_activation_evidence(
        session,
        revision.id,
        recorded.id,
    )
    original_binding_sha256 = evidence.binding_sha256
    original_witness_sha256 = evidence.activation_witness_sha256

    consumed = evidence.resolution
    provenance = consumed.bindings[0].identity["provenance"]
    paths = consumed.bindings[0].mount["paths"]
    assert isinstance(provenance, dict)
    layers = provenance["layers"]
    assert isinstance(layers, list)
    assert isinstance(paths, list)
    layers.append("hostile-identity-mutation")
    paths.append("hostile-mount-mutation")

    reconstructed = evidence.resolution
    reconstructed_provenance = reconstructed.bindings[0].identity["provenance"]
    reconstructed_paths = reconstructed.bindings[0].mount["paths"]
    assert isinstance(reconstructed_provenance, dict)
    assert reconstructed_provenance["layers"] == ["recorded", "decoder"]
    assert reconstructed_paths == ["recorded", "decoder"]
    assert reconstructed is not consumed
    assert reconstructed.bindings[0].identity is not consumed.bindings[0].identity
    assert reconstructed.bindings[0].mount is not consumed.bindings[0].mount
    assert evidence.binding_sha256 == original_binding_sha256
    assert evidence.activation_witness_sha256 == original_witness_sha256
    assert (
        workflow_activation_binding_sha256(_contract(), reconstructed.bindings)
        == evidence.binding_sha256
    )


def test_recorded_mode_never_substitutes_current_activation(session: Session) -> None:
    revision, _recorded, current = _ready_pair(session)
    other_revision, _contract_value, _slots = _revision(session, suffix="other")
    session.commit()

    with pytest.raises(WorkflowActivationError) as missing:
        load_recorded_workflow_lora_activation_evidence(
            session,
            revision.id,
            "wfact_missing",
        )
    assert missing.value.code == "workflow_activation_unavailable"

    with pytest.raises(WorkflowActivationError) as wrong_revision:
        load_recorded_workflow_lora_activation_evidence(
            session,
            other_revision.id,
            current.id,
        )
    assert wrong_revision.value.code == "workflow_activation_revision_mismatch"

    with pytest.raises(WorkflowActivationError) as no_current:
        load_current_workflow_lora_activation_evidence(session, other_revision.id)
    assert no_current.value.code == "workflow_activation_unavailable"


def test_current_mode_rejects_duplicate_active_rows_even_if_storage_is_corrupt(
    session: Session,
) -> None:
    revision, recorded, _current = _ready_pair(session)
    session.execute(text("DROP INDEX uq_workflow_activation_active_revision"))
    session.execute(
        text("UPDATE workflow_activations SET is_active = 1 WHERE id = :activation_id"),
        {"activation_id": recorded.id},
    )
    session.commit()
    session.expire_all()

    with pytest.raises(WorkflowActivationError) as raised:
        load_current_workflow_lora_activation_evidence(session, revision.id)

    assert raised.value.code == "workflow_activation_ambiguous"


@pytest.mark.parametrize(
    ("target", "expected_code"),
    [
        ("state", "workflow_activation_not_ready"),
        ("invalidated", "workflow_activation_invalidated"),
        ("resolver", "invalid_activation_snapshot"),
        ("binding", "invalid_activation_snapshot"),
        ("contract", "workflow_contract_drift"),
        ("details_type", "invalid_activation_snapshot"),
        ("details_extra", "invalid_activation_snapshot"),
        ("launch", "invalid_activation_snapshot"),
    ],
)
def test_recorded_mode_rejects_hostile_activation_identity(
    session: Session,
    target: str,
    expected_code: str,
) -> None:
    revision, recorded, _current = _ready_pair(session)
    if target == "state":
        recorded.state = "stale"
    elif target == "invalidated":
        recorded.invalidated_at = utcnow()
        recorded.invalidation_code = "stale"
        recorded.invalidation_reason = "stale"
    elif target == "resolver":
        recorded.resolver_version = "bad resolver"
    elif target == "binding":
        recorded.binding_sha256 = "not-a-digest"
    elif target == "contract":
        recorded.dependency_contract_sha256 = "f" * 64
    elif target == "details_type":
        recorded.__setattr__("details_json", [])
    elif target == "details_extra":
        recorded.details_json = {"launch_sha256": "e" * 64, "unexpected": True}
    else:
        recorded.details_json = {"launch_sha256": "not-a-digest"}

    with pytest.raises(WorkflowActivationError) as raised:
        load_recorded_workflow_lora_activation_evidence(session, revision.id, recorded.id)

    assert raised.value.code == expected_code


class _DictSubclass(dict[str, object]):
    pass


@pytest.mark.parametrize("target", ["ordinal", "slot_digest", "contract_digest", "requirements"])
def test_loader_rejects_hostile_dependency_contract_rows(
    session: Session,
    target: str,
) -> None:
    revision, recorded, _current = _ready_pair(session)
    slot = session.scalar(
        select(WorkflowDependencySlot).where(
            WorkflowDependencySlot.workflow_revision_id == revision.id
        )
    )
    assert slot is not None
    if target == "ordinal":
        slot.ordinal = 2
    elif target == "slot_digest":
        slot.contract_sha256 = "f" * 64
    elif target == "contract_digest":
        revision.dependency_contract_sha256 = "f" * 64
    else:
        requirements = list(slot.requirements_json)
        requirements[0] = {
            **requirements[0],
            "constraints": _DictSubclass(requirements[0]["constraints"]),
        }
        slot.requirements_json = requirements

    with pytest.raises(WorkflowActivationError):
        load_recorded_workflow_lora_activation_evidence(session, revision.id, recorded.id)


@pytest.mark.parametrize(
    ("target", "expected_codes"),
    [
        ("unknown_requirement", {"unknown_dependency_requirement"}),
        ("duplicate_requirement", {"duplicate_dependency_binding"}),
        ("wrong_revision", {"invalid_activation_snapshot"}),
        ("identity_digest", {"workflow_activation_incomplete"}),
        ("no_locator", {"invalid_activation_snapshot"}),
        ("two_locators", {"invalid_activation_snapshot"}),
        ("identity_subclass", {"invalid_activation_snapshot"}),
        ("mount_subclass", {"invalid_activation_snapshot"}),
        ("nested_subclass", {"invalid_activation_snapshot"}),
    ],
)
def test_loader_rejects_hostile_binding_rows(
    session: Session,
    target: str,
    expected_codes: set[str],
) -> None:
    revision, recorded, _current = _ready_pair(session)
    rows = list(
        session.scalars(
            select(WorkflowDependencyBinding)
            .where(WorkflowDependencyBinding.workflow_activation_id == recorded.id)
            .order_by(WorkflowDependencyBinding.requirement_key)
        ).all()
    )
    assert len(rows) == 2
    if target == "unknown_requirement":
        rows[0].requirement_key = "unknown"
    elif target == "duplicate_requirement":
        rows[1].requirement_key = rows[0].requirement_key
    elif target == "wrong_revision":
        rows[0].workflow_revision_id = "wfrev_wrong"
    elif target == "identity_digest":
        rows[0].resource_identity_sha256 = "f" * 64
    elif target == "no_locator":
        rows[0].runtime_key = None
    elif target == "two_locators":
        rows[0].model_install_id = "model_extra"
    elif target == "identity_subclass":
        rows[0].resource_identity_json = _DictSubclass(rows[0].resource_identity_json)
    elif target == "mount_subclass":
        rows[0].mount_json = _DictSubclass(rows[0].mount_json)
    else:
        rows[0].mount_json["nested"] = _DictSubclass({"value": True})

    with pytest.raises(WorkflowActivationError) as raised:
        load_recorded_workflow_lora_activation_evidence(session, revision.id, recorded.id)

    assert raised.value.code in expected_codes


def test_loader_rejects_missing_required_binding(
    session: Session,
) -> None:
    revision, recorded, _current = _ready_pair(session)
    row = session.scalar(
        select(WorkflowDependencyBinding).where(
            WorkflowDependencyBinding.workflow_activation_id == recorded.id,
            WorkflowDependencyBinding.requirement_key == "encoder",
        )
    )
    assert row is not None
    session.delete(row)
    session.commit()

    with pytest.raises(WorkflowActivationError) as missing:
        load_recorded_workflow_lora_activation_evidence(session, revision.id, recorded.id)
    assert missing.value.code == "workflow_activation_incomplete"


def test_loader_rejects_activation_binding_digest_drift(session: Session) -> None:
    revision, recorded, _current = _ready_pair(session)
    recorded.binding_sha256 = "f" * 64

    with pytest.raises(WorkflowActivationError) as drift:
        load_recorded_workflow_lora_activation_evidence(session, revision.id, recorded.id)

    assert drift.value.code == "dependency_binding_drift"


def test_witness_changes_when_launch_identity_changes(session: Session) -> None:
    revision, recorded, _current = _ready_pair(session)
    before = load_recorded_workflow_lora_activation_evidence(
        session,
        revision.id,
        recorded.id,
    )
    recorded.details_json = {"launch_sha256": "f" * 64}
    after = load_recorded_workflow_lora_activation_evidence(
        session,
        revision.id,
        recorded.id,
    )

    assert after.launch_sha256 == "f" * 64
    assert after.activation_witness_sha256 != before.activation_witness_sha256
