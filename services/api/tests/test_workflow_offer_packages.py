"""Accepted extension requests and their inert results survive as one exact transaction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_workflow_package_execution_plan import _inputs, _plan
from test_workflow_package_install_plans import _payload

from local_lm import models
from local_lm.comfy_registry_lifecycle import ComfyRegistryPreparation
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.db import SessionLocal
from local_lm.runtime_provisioning_plans import RuntimeProvisioningPlan
from local_lm.workflow_install_offers import (
    bind_workflow_offer_downloads,
    mark_workflow_install_offer_queued,
)
from local_lm.workflow_package_drafts import stage_workflow_package_draft
from local_lm.workflow_package_extension_preflight import WorkflowPackageExtensionPreflight
from local_lm.workflow_package_install_plans import (
    WorkflowPackageInstallPlanOut,
    WorkflowPackageInstallPlanRequest,
    create_workflow_package_install_plan,
)
from local_lm.workflow_package_preparation import PreparationContext, prepare_workflow_package

pytestmark = pytest.mark.asyncio


async def _setup(
    tmp_path: Path,
    *,
    package_inputs: list[dict[str, Any]] | None = None,
    source_payload: dict[str, Any] | None = None,
    preparation_context: PreparationContext | None = None,
    runtime_plan: RuntimeProvisioningPlan | None = None,
    available_nodes: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], PreparationContext, str, WorkflowPackageInstallPlanOut]:
    from local_lm.workflow_offer_packages import stage_workflow_offer_packages

    prepared_inputs = package_inputs if package_inputs is not None else [_inputs()]
    inputs = prepared_inputs[0]
    plans = {}
    payload = _payload() if source_payload is None else source_payload
    available = {"LoraLoader", "EmptyLatentImage"}
    available.update(available_nodes)
    for package_index, current in enumerate(prepared_inputs):
        try:
            plan = await _plan(current)
        finally:
            await current["archive_downloader"].close()
        selected = current["selected"]
        plans[selected.package_id] = plan
        available.update(selected.node_types)
        for index, name in enumerate(selected.node_types):
            payload["ui_graph"]["nodes"].append(
                {
                    "id": 100 + package_index * 10 + index,
                    "type": name,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": [],
                    "properties": {"cnr_id": selected.package_id, "ver": selected.declared_version},
                }
            )
    request = WorkflowPackageInstallPlanRequest.model_validate(payload)
    with SessionLocal() as session:
        saved = create_workflow_package_install_plan(
            session,
            request,
            available_node_types=available,
            available_asset_filenames=(),
            installed_package_versions={},
            extension_execution=WorkflowPackageExtensionPreflight(plans=plans),
            runtime_plan=runtime_plan
            or RuntimeProvisioningPlan(
                engine="comfyui",
                operation="reuse_configured",
                release=None,
                license="Test runtime license",
                download_bytes=0,
                required_free_bytes=0,
                inputs_sha256="1" * 64,
                plan_sha256="2" * 64,
            ),
        )
        if runtime_plan is not None and runtime_plan.operation == "install_managed":
            # Constructed runtimes have no packaged node catalog.
            assert not saved.can_accept
            assert {"node-inventory-unavailable", "runtime-installation-unavailable"} <= set(
                saved.blockers
            )
        else:
            assert saved.can_accept and saved.blockers == []
        _definition, revision = stage_workflow_package_draft(session, request)
        assert saved.dependency_contract_sha256 is not None
        offer = models.WorkflowInstallOffer(
            source_plan_id=saved.id,
            workflow_revision_id=revision.id,
            workflow_artifact_sha256=revision.artifact_sha256,
            dependency_contract_sha256=saved.dependency_contract_sha256,
            binding_plan_sha256=saved.asset_binding_sha256,
            offer_sha256=saved.plan_sha256,
            selections_json=[item.model_dump(mode="json") for item in request.selections],
            assets_json=[item.model_dump(mode="json") for item in saved.assets],
            plan_count=len(saved.download_requests),
            total_bytes=saved.total_download_bytes,
            status="ready",
        )
        session.add(offer)
        jobs = [
            models.Job(kind="download", status="queued", payload_json=item.model_dump(mode="json"))
            for item in saved.download_requests
        ]
        session.add_all(jobs)
        bind_workflow_offer_downloads(session, offer, jobs)
        stage_workflow_offer_packages(session, offer, saved)
        mark_workflow_install_offer_queued(offer)
        identifier = offer.id
        with SessionLocal() as reader:
            assert reader.get(models.WorkflowInstallOffer, identifier) is None
            assert list(reader.scalars(select(models.Job))) == []
        session.commit()
    context = preparation_context or PreparationContext(
        inputs["python_executable"], tmp_path / "nodes", tmp_path / "state"
    )
    context.custom_node_root.mkdir(parents=True, exist_ok=True)
    context.state_root.mkdir(parents=True, exist_ok=True)
    return inputs, context, identifier, saved


def _accepted(session: Session, offer_id: str, saved: WorkflowPackageInstallPlanOut) -> Any:
    from local_lm.workflow_offer_packages import accepted_workflow_offer_packages
    from local_lm.workflow_package_acceptance import accepted_workflow_package_jobs

    offer = session.get(models.WorkflowInstallOffer, offer_id)
    assert offer is not None
    _source, accepted, _jobs = accepted_workflow_package_jobs(session, offer)
    assert accepted == saved
    items = accepted_workflow_offer_packages(session, offer, saved)
    assert len(items) == 1
    return items[0]


async def _prepare(
    inputs: dict[str, Any],
    context: PreparationContext,
    saved: WorkflowPackageInstallPlanOut,
    recorder: Any,
) -> ComfyRegistryPreparation:
    from local_lm.comfy_registry_downloads import ComfyRegistryArchiveDownloader

    archive = ComfyRegistryArchiveDownloader(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=inputs["state"]["content"])
        )
    )
    wheels = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    )
    selected = inputs["selected"]
    try:
        return await prepare_workflow_package(
            SessionLocal,
            package_id=selected.package_id,
            version=selected.declared_version,
            node_types=selected.node_types,
            context=context,
            media_worker_stopped=True,
            interpreter_probe=inputs["interpreter_probe"],
            registry_client=inputs["registry_client"],
            project_client=inputs["project_client"],
            metadata_client=inputs["metadata_client"],
            archive_downloader=archive,
            wheel_downloader=wheels,
            expected_plan=saved.extension_execution.plans[selected.package_id],
            record_preparation=recorder,
        )
    finally:
        await archive.close()
        await wheels.close()


@pytest.mark.parametrize("outcome", ["complete", "rollback", "cancelled", "changed-job"])
async def test_preparation_and_accepted_result_commit_or_rollback_together(
    client: AsyncClient,
    tmp_path: Path,
    outcome: str,
) -> None:
    from local_lm.workflow_offer_packages import record_workflow_offer_package

    inputs, context, offer_id, saved = await _setup(tmp_path)
    with SessionLocal() as session:
        item = _accepted(session, offer_id, saved)
        job_id = item.job.id
        item.job.status = "cancelled" if outcome == "changed-job" else "running"
        session.commit()
    recorded = []

    def record(session: Session, preparation: ComfyRegistryPreparation) -> None:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert offer is not None
        record_workflow_offer_package(
            session,
            offer,
            saved,
            package_id=inputs["selected"].package_id,
            job_id=job_id,
            preparation=preparation,
        )
        with SessionLocal() as reader:
            assert reader.get(models.ComfyRegistryInstall, preparation.install_id) is None
            old = _accepted(reader, offer_id, saved)
            assert old.preparation is None and old.job.result_json == {}
        recorded.append(preparation)
        if outcome == "rollback":
            raise RuntimeError("Neutral result persistence failure")
        if outcome == "cancelled":
            raise asyncio.CancelledError

    if outcome == "complete":
        result = await _prepare(inputs, context, saved, record)
        assert recorded == [result]
        with SessionLocal() as session:
            item = _accepted(session, offer_id, saved)
            assert item.preparation == result
            row = session.get(models.ComfyRegistryInstall, result.install_id)
            assert row is not None and not row.trusted and not row.active
            assert item.job.status == "complete"
    else:
        with pytest.raises(
            asyncio.CancelledError if outcome == "cancelled" else (RuntimeError, ValueError)
        ):
            await _prepare(inputs, context, saved, record)
        with SessionLocal() as session:
            assert list(session.scalars(select(models.ComfyRegistryInstall))) == []
            assert _accepted(session, offer_id, saved).preparation is None
        assert list(context.custom_node_root.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    [
        "job",
        "job-payload",
        "plan",
        "offer",
        "missing-link",
        "complete-without-result",
        "stored-plan",
        "supplied-plan",
    ],
)
async def test_changed_accepted_requests_are_never_silently_recreated(
    client: AsyncClient,
    tmp_path: Path,
    change: str,
) -> None:
    from local_lm.workflow_offer_packages import accepted_workflow_offer_packages

    _inputs_value, _context, offer_id, saved = await _setup(tmp_path)
    with SessionLocal() as session:
        item = _accepted(session, offer_id, saved)
        if change == "job":
            session.delete(item.job)
        elif change == "job-payload":
            item.job.payload_json = {
                **item.job.payload_json,
                "package_id": "another-neutral-package",
            }
        elif change == "plan":
            item.link.execution_plan_sha256 = "a" * 64
        elif change == "offer":
            item.link.offer_sha256 = "b" * 64
        elif change == "complete-without-result":
            item.job.status = "complete"
        elif change == "stored-plan":
            record = session.get(models.WorkflowPackageInstallPlan, saved.id)
            assert record is not None
            record.preflight_json = {**record.preflight_json, "total_download_bytes": 1}
        elif change == "supplied-plan":
            saved = saved.model_copy(
                update={"asset_download_bytes": saved.asset_download_bytes + 1}
            )
        else:
            session.delete(item.link)
        session.commit()
    with SessionLocal() as session:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert offer is not None
        with pytest.raises(ValueError):
            accepted_workflow_offer_packages(session, offer, saved)


async def test_retry_keeps_the_prepared_identity_after_a_separate_session_reopens(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    from local_lm.workflow_offer_packages import record_workflow_offer_package

    inputs, context, offer_id, saved = await _setup(tmp_path)
    with SessionLocal() as session:
        item = _accepted(session, offer_id, saved)
        item.job.status = "running"
        job_id = item.job.id
        session.commit()

    def record(session: Session, preparation: ComfyRegistryPreparation) -> None:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert offer is not None
        record_workflow_offer_package(
            session,
            offer,
            saved,
            package_id=inputs["selected"].package_id,
            job_id=job_id,
            preparation=preparation,
        )

    result = await _prepare(inputs, context, saved, record)
    with SessionLocal() as session:
        first = _accepted(session, offer_id, saved)
        assert first.preparation == result
        first_id = first.link.id
    with SessionLocal() as session:
        again = _accepted(session, offer_id, saved)
        assert again.link.id == first_id and again.job.id == job_id and again.preparation == result
        session.delete(session.get(models.ComfyRegistryInstall, result.install_id))
        session.commit()
    with SessionLocal() as session, pytest.raises(ValueError):
        _accepted(session, offer_id, saved)
