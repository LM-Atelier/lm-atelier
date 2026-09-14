"""Complete accepted installs only after exact download results reach an activation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Never

from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import JobStatus, utcnow
from .models import (
    InstallPlan,
    Job,
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
    ModelSource,
    WorkflowActivation,
    WorkflowInstallOffer,
    WorkflowInstallOfferDownload,
)
from .schemas import DownloadRequest
from .workflow_activation_preparation import prepare_workflow_activation
from .workflow_activation_requests import (
    WorkflowActivationCreate,
    activate_reviewed_revision,
    activation_subject,
)
from .workflow_activations import WorkflowRuntimeMaterializer, revalidate_workflow_activation
from .workflow_asset_downloads import install_plan_download_request
from .workflow_bindings import materialize_model_asset, materialize_model_install
from .workflow_install_offers import assert_workflow_install_offer_identity


class WorkflowOfferCompletionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__("The workflow installation needs attention.")
        self.code = code


def _refuse(code: str) -> Never:
    raise WorkflowOfferCompletionError(code)


def _result_resources(
    session: Session, job: Job, request: DownloadRequest, plan: InstallPlan
) -> set[tuple[str, str]]:
    result = job.result_json
    asset_id = result.get("model_asset_id")
    install_id = result.get("model_install_id")
    if isinstance(asset_id, str) and install_id is None:
        asset = session.get(ModelAssetInstall, asset_id)
        if asset is None:
            _refuse("download-result-unavailable")
        materialize_model_asset(asset)
        row: ModelAssetInstall | ModelInstall = asset
        resources = {("model_asset", asset.id)}
    elif isinstance(install_id, str) and asset_id is None:
        install = session.get(ModelInstall, install_id)
        if install is None:
            _refuse("download-result-unavailable")
        materialize_model_install(session, install)
        row = install
        resources = {("model_install", install.id)}
        profile_id = result.get("profile_id")
        if isinstance(profile_id, str):
            profile = session.get(ModelProfile, profile_id)
            if profile is None or profile.model_install_id != install.id:
                _refuse("download-result-changed")
            resources.add(("model_profile", profile_id))
    else:
        _refuse("download-result-unavailable")
    source = session.get(ModelSource, row.source_id) if row.source_id else None
    if (
        source is None
        or (source.provider, source.remote_id, source.revision)
        != (plan.provider, request.remote_id, request.revision)
        or row.manifest_json.get("expected_sha256") != request.expected_sha256
        or row.manifest_json.get("remote_id") != request.remote_id
        or row.manifest_json.get("revision") != request.revision
    ):
        _refuse("download-result-changed")
    return resources


def _accepted_results(
    session: Session, offer: WorkflowInstallOffer
) -> list[set[tuple[str, str]]] | None:
    links = session.scalars(
        select(WorkflowInstallOfferDownload).where(
            WorkflowInstallOfferDownload.offer_id == offer.id,
        )
    ).all()
    if len(links) != offer.plan_count:
        _refuse("download-acceptance-unavailable")
    results: list[set[tuple[str, str]]] = []
    pending = False
    for link in links:
        request_json = link.request_json
        digest = hashlib.sha256(
            json.dumps(
                request_json,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        job = session.get(Job, link.job_id) if link.job_id else None
        if (
            link.offer_sha256 != offer.offer_sha256
            or link.request_sha256 != digest
            or job is None
            or job.kind != "download"
            or job.payload_json != request_json
        ):
            _refuse("download-acceptance-changed")
        if job.status in {JobStatus.FAILED.value, JobStatus.CANCELLED.value}:
            _refuse("workflow-download-failed")
        if job.status != JobStatus.COMPLETE.value:
            pending = True
            continue
        request = DownloadRequest.model_validate(request_json)
        plan = (
            session.get(InstallPlan, request.install_plan_id) if request.install_plan_id else None
        )
        if plan is None:
            _refuse("download-plan-changed")
        assets = [asset for asset in offer.assets_json if asset.get("install_plan_id") == plan.id]
        if (
            not assets
            or any(asset.get("install_plan_hash") != plan.plan_hash for asset in assets)
            or install_plan_download_request(plan, allow_activated=True).model_dump(mode="json")
            != request_json
        ):
            _refuse("download-plan-changed")
        results.append(_result_resources(session, job, request, plan))
    return None if pending else results


def complete_workflow_install_offer(
    session: Session,
    offer_id: str,
    *,
    runtime_materializer: WorkflowRuntimeMaterializer | None = None,
    custom_node_root: Path | None = None,
    registry_environment_root: Path | None = None,
) -> str | None:
    """Join accepted jobs, installed identities and current review in one transaction.

    A caller must own a real outer transaction and commit only after this returns.
    Pending jobs leave the offer queued. Ambiguous choices never select a resource.
    """
    offer = session.get(WorkflowInstallOffer, offer_id)
    if offer is None or offer.status != "queued":
        return None
    revision = assert_workflow_install_offer_identity(session, offer)
    subject = activation_subject(session, revision.workflow_id, revision.id)
    if (subject.workflow_artifact_sha256, subject.dependency_contract_sha256) != (
        offer.workflow_artifact_sha256,
        offer.dependency_contract_sha256,
    ):
        _refuse("workflow-install-offer-changed")
    accepted = _accepted_results(session, offer)
    if accepted is None:
        return None
    accepted_resources = {resource for result in accepted for resource in result}
    if offer.source_plan_id is not None:
        from .workflow_offer_packages import accepted_workflow_offer_packages
        from .workflow_package_acceptance import accepted_workflow_package_jobs

        _source, saved, _jobs = accepted_workflow_package_jobs(session, offer)
        accepted_resources.update(
            ("registry_package", item.preparation.install_id)
            for item in accepted_workflow_offer_packages(session, offer, saved)
            if item.preparation is not None
        )
    active = session.scalar(
        select(WorkflowActivation).where(
            WorkflowActivation.workflow_revision_id == revision.id,
            WorkflowActivation.is_active.is_(True),
        )
    )
    if active is not None:
        scope = revalidate_workflow_activation(
            session,
            active,
            runtime_materializer=runtime_materializer,
            custom_node_root=custom_node_root,
            registry_environment_root=registry_environment_root,
        )
        selected = {
            *(("model_install", identity) for identity in scope.model_install_ids),
            *(("model_asset", identity) for identity in scope.model_asset_install_ids),
        }
        if any(not (resources & selected) for resources in accepted):
            _refuse("download-results-need-binding")
        activation_id = scope.activation_id
    else:
        prepared = prepare_workflow_activation(
            session,
            revision.workflow_id,
            revision.id,
            runtime_materializer=runtime_materializer,
            accepted_resources=frozenset(accepted_resources),
        )
        if prepared.state != "prepared" or prepared.selections is None:
            _refuse("workflow-dependencies-need-selection")
        selected = {(choice.local_kind, choice.local_id) for choice in prepared.selections}
        if any(not (resources & selected) for resources in accepted):
            _refuse("download-results-need-binding")
        activation = activate_reviewed_revision(
            session,
            revision.workflow_id,
            revision.id,
            WorkflowActivationCreate(
                workflow_artifact_sha256=subject.workflow_artifact_sha256,
                dependency_contract_sha256=subject.dependency_contract_sha256,
                selections=prepared.selections,
            ),
            runtime_materializer=runtime_materializer,
            custom_node_root=custom_node_root,
            registry_environment_root=registry_environment_root,
        )
        activation_id = activation.id
    session.expire_all()
    assert_workflow_install_offer_identity(session, offer)
    if offer.status != "queued" or _accepted_results(session, offer) != accepted:
        _refuse("download-acceptance-changed")
    if activation_subject(session, revision.workflow_id, revision.id) != subject:
        _refuse("workflow-install-offer-changed")
    current = session.get(WorkflowActivation, activation_id)
    if current is None or not current.is_active or current.state != "ready":
        _refuse("workflow-install-offer-changed")
    offer.status = "completed"
    offer.completed_at = utcnow()
    return activation_id
