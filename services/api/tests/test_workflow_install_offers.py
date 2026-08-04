from __future__ import annotations

import hashlib
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.model_planner import INSTALL_RESOLVER_VERSION, workflow_artifact_contract
from local_lm.models import (
    InstallPlan,
    WorkflowDefinition,
    WorkflowInstallOffer,
    WorkflowRevision,
)
from local_lm.workflow_asset_bindings import WorkflowAssetPlanSelection
from local_lm.workflow_dependencies import (
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
)
from local_lm.workflow_install_offers import (
    WorkflowInstallOfferError,
    create_workflow_install_offer,
    revalidate_workflow_install_offer,
)

REFERENCE = "styles/detail.safetensors"
DIGEST = "a" * 64


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
    engine.dispose()


def _graph(filename: str = REFERENCE) -> dict[str, object]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "LoraLoader",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [filename],
                "properties": {"cnr_id": "comfy-core", "version": "0.28.0"},
            }
        ],
        "links": [],
    }


def _revision(session: Session, *, trusted: bool = True) -> WorkflowRevision:
    contract = parse_workflow_dependency_contract({"version": 1, "slots": []})
    definition = WorkflowDefinition(
        id="workflow_offer",
        name="Offer workflow",
        operation="text_to_image",
    )
    session.add(definition)
    session.flush()
    revision = WorkflowRevision(
        id="wfrev_offer",
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        ui_graph_json=_graph(),
        artifact_sha256=workflow_artifact_contract(
            operation=definition.operation,
            engine="comfyui",
            api_graph={},
            input_schema={},
            dependencies={},
        ),
        dependency_contract_sha256=workflow_dependency_contract_sha256(contract),
        trusted=trusted,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return revision


def _plan(
    session: Session,
    *,
    plan_id: str = "plan_offer",
    digest: str = DIGEST,
    size: int = 17,
) -> InstallPlan:
    plan = InstallPlan(
        id=plan_id,
        provider="civitai",
        remote_id="101",
        revision="202",
        role="image",
        engine="comfyui",
        plan_hash=hashlib.sha256(plan_id.encode()).hexdigest(),
        resolver_version=INSTALL_RESOLVER_VERSION,
        compatibility="supported",
        artifacts_json=[
            {
                "path": REFERENCE,
                "kind": "lora",
                "target_folder": "loras",
                "size_bytes": size,
                "sha256": digest,
                "required": True,
                "reuse": "download",
                "source_version_id": "202",
                "source_file_id": "301",
            }
        ],
        runtime_contract_json={
            "auxiliary_kind": "lora",
            "comfy_paths": {"loras": "styles"},
        },
        activation_probe_json={},
        status="planned",
    )
    session.add(plan)
    session.flush()
    return plan


def _selection(plan: InstallPlan) -> WorkflowAssetPlanSelection:
    return WorkflowAssetPlanSelection(REFERENCE, plan.id, REFERENCE)


def _create(
    session: Session,
    revision: WorkflowRevision,
    plan: InstallPlan,
) -> WorkflowInstallOffer:
    return create_workflow_install_offer(
        session,
        workflow_id=revision.workflow_id,
        revision_id=revision.id,
        selections=[_selection(plan)],
        available_node_types={"LoraLoader"},
        available_asset_filenames=set(),
        installed_package_versions={},
    )


def test_offer_is_idempotent_and_rebuilds_one_exact_download(session: Session) -> None:
    revision = _revision(session)
    plan = _plan(session)

    first = _create(session, revision, plan)
    second = _create(session, revision, plan)
    session.commit()

    assert second.id == first.id
    assert first.plan_count == 1
    assert first.total_bytes == 17
    assert first.binding_plan_sha256
    assert first.offer_sha256
    assert first.assets_json[0]["sha256"] == DIGEST

    current, requests = revalidate_workflow_install_offer(
        session,
        first.id,
        available_node_types={"LoraLoader"},
        available_asset_filenames=set(),
        installed_package_versions={},
    )
    assert current.id == first.id
    assert len(requests) == 1
    assert requests[0].install_plan_id == plan.id
    assert requests[0].expected_sha256 == {REFERENCE: DIGEST}


def test_plan_drift_invalidates_before_returning_downloads(session: Session) -> None:
    revision = _revision(session)
    plan = _plan(session)
    offer = _create(session, revision, plan)
    plan.status = "failed"
    session.flush()

    with pytest.raises(WorkflowInstallOfferError, match="not ready"):
        revalidate_workflow_install_offer(
            session,
            offer.id,
            available_node_types={"LoraLoader"},
            available_asset_filenames=set(),
            installed_package_versions={},
        )

    assert offer.status == "invalidated"
    assert offer.invalidated_at is not None
    assert offer.invalidation_code == "install_plan_not_pending"


def test_local_state_change_invalidates_an_obsolete_offer(session: Session) -> None:
    revision = _revision(session)
    plan = _plan(session)
    offer = _create(session, revision, plan)

    with pytest.raises(WorkflowInstallOfferError) as captured:
        revalidate_workflow_install_offer(
            session,
            offer.id,
            available_node_types={"LoraLoader"},
            available_asset_filenames={REFERENCE},
            installed_package_versions={},
        )

    assert captured.value.code == "workflow-install-not-needed"
    assert offer.status == "invalidated"


def test_offer_refuses_partial_or_non_asset_blocked_reviews(session: Session) -> None:
    revision = _revision(session)
    _plan(session)

    with pytest.raises(WorkflowInstallOfferError) as partial:
        create_workflow_install_offer(
            session,
            workflow_id=revision.workflow_id,
            revision_id=revision.id,
            selections=[],
            available_node_types={"LoraLoader"},
            available_asset_filenames=set(),
            installed_package_versions={},
        )
    assert partial.value.code == "missing_asset_selection"

    with pytest.raises(WorkflowInstallOfferError) as blocked:
        create_workflow_install_offer(
            session,
            workflow_id=revision.workflow_id,
            revision_id=revision.id,
            selections=[],
            available_node_types=set(),
            available_asset_filenames=set(),
            installed_package_versions={},
        )
    assert blocked.value.code == "workflow-install-offer-incomplete"


def test_offer_requires_current_trusted_content_bound_revision(session: Session) -> None:
    revision = _revision(session, trusted=False)
    plan = _plan(session)

    with pytest.raises(WorkflowInstallOfferError) as captured:
        _create(session, revision, plan)

    assert captured.value.code == "workflow-revision-needs-attention"

    revision.trusted = True
    revision.dependency_contract_sha256 = "c" * 64
    session.flush()
    with pytest.raises(WorkflowInstallOfferError) as drifted:
        _create(session, revision, plan)
    assert drifted.value.code == "workflow-contract-drift"


def test_new_review_supersedes_a_different_ready_offer(session: Session) -> None:
    revision = _revision(session)
    first_plan = _plan(session)
    first = _create(session, revision, first_plan)
    second_plan = _plan(session, plan_id="plan_replacement", digest="d" * 64, size=23)

    second = _create(session, revision, second_plan)

    assert second.id != first.id
    assert second.status == "ready"
    assert second.total_bytes == 23
    assert first.status == "invalidated"
    assert first.invalidation_code == "offer-superseded"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "unknown"),
        ("plan_count", 0),
        ("total_bytes", 0),
        ("offer_sha256", "A" * 64),
    ],
)
def test_offer_persistence_constraints_reject_invalid_rows(
    session: Session,
    field: str,
    value: object,
) -> None:
    revision = _revision(session)
    plan = _plan(session)
    offer = _create(session, revision, plan)
    session.commit()
    setattr(offer, field, value)

    with pytest.raises(IntegrityError):
        session.commit()
