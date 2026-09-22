"""Apply trust to the exact accepted extensions without starting their code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text

from .comfy_registry_activation import _apply_registry_policy_trust
from .comfy_registry_activation_batches import _row
from .comfy_registry_launch_verification import verify_comfy_registry_launch
from .comfy_registry_paths import registry_wheel_environment_root
from .comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from .models import ComfyRegistryInstall
from .registry_trust_policy import RegistryTrustDecision, decide_registry_trust
from .workflow_offer_packages import WorkflowOfferPackageError
from .workflow_package_execution_plan import WorkflowPackageExecutionPlan
from .workflow_package_preparation import PreparationContext, WorkflowPackagePreparationError
from .workflow_source_extensions import SessionFactory, _accepted


@dataclass(frozen=True)
class WorkflowExtensionTrust:
    package_id: str
    install_id: str
    decision: RegistryTrustDecision


@dataclass(frozen=True)
class WorkflowSourceExtensionTrust:
    state: Literal["ready", "review_required"]
    packages: tuple[WorkflowExtensionTrust, ...]


def _decision(
    row: ComfyRegistryInstall, plan: WorkflowPackageExecutionPlan
) -> RegistryTrustDecision:
    decision = decide_registry_trust(plan.resolution, row)
    review = row.review_json
    if (
        decision.outcome == "pause"
        and decision.reason
        in {"source_review_required", "unreviewed_source", "unrecognised_warning"}
        and row.trusted
        and review.get("trusted_by_local_user") is True
        and review.get("trust_authority") in (None, "local_user")
        and isinstance(review.get("reviewed_at"), str)
        and review["reviewed_at"]
    ):
        # Source identity and physical files have already been checked against
        # the accepted plan. An explicit grant for those same bytes still holds.
        return RegistryTrustDecision("already_trusted", "already_trusted", "", decision.notices)
    return decision


def trust_workflow_source_extensions(
    session_factory: SessionFactory,
    offer_id: str,
    *,
    context: PreparationContext,
    media_worker_stopped: bool,
    reviewed_inputs: ComfyRegistryReviewedInputContext | None = None,
) -> WorkflowSourceExtensionTrust:
    """Under the primary lease, verify every package before committing any new grant."""
    if media_worker_stopped is not True:
        raise WorkflowPackagePreparationError(
            "media_worker_running", "Stop the media worker before extension setup."
        )
    environment_root = registry_wheel_environment_root(context.state_root)
    with session_factory() as session:
        _offer, _saved, packages = _accepted(session, offer_id)
        identifiers = []
        for package in packages:
            if package.preparation is None:
                raise WorkflowOfferPackageError()
            identifiers.append(_row(session, package.preparation, package.plan).id)
    if not identifiers:
        return WorkflowSourceExtensionTrust("ready", ())
    verified = verify_comfy_registry_launch(
        session_factory,
        identifiers,
        include_active=True,
        custom_node_root=context.custom_node_root,
        environment_root=environment_root,
        reviewed_inputs=reviewed_inputs,
    )
    with session_factory() as session:
        session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
        session.expire_all()
        _offer, _saved, packages = _accepted(session, offer_id)
        decisions = []
        rows = []
        for package in packages:
            if package.preparation is None:
                raise WorkflowOfferPackageError()
            row = _row(session, package.preparation, package.plan)
            rows.append(row)
            decision = _decision(row, package.plan)
            if decision.outcome == "refuse":
                raise WorkflowPackagePreparationError(
                    "registry_policy_" + decision.reason, decision.explanation
                )
            decisions.append(WorkflowExtensionTrust(package.link.package_id, row.id, decision))
        if tuple(sorted(row.id for row in rows)) != verified.install_ids:
            raise WorkflowOfferPackageError()
        if any(item.decision.outcome == "pause" for item in decisions):
            return WorkflowSourceExtensionTrust("review_required", tuple(decisions))
        verified.require_current(session)
        for row, result in zip(rows, decisions, strict=True):
            if result.decision.outcome != "auto_trust":
                continue
            _apply_registry_policy_trust(row, result.decision)
        session.commit()
        return WorkflowSourceExtensionTrust("ready", tuple(decisions))
