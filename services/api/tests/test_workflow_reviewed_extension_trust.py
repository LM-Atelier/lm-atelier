from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import delete, event
from sqlalchemy.orm import Session, object_session
from test_workflow_package_install_plans import _payload
from test_workflow_package_preparation import _Registry
from test_workflow_reviewed_package_plan import _close, _inputs, _prepare

from local_lm import comfy_registry_installs as install_module
from local_lm import workflow_source_extension_trust as trust_module
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_activation import _apply_registry_policy_trust
from local_lm.comfy_registry_archives import verify_staged_comfy_registry_archive
from local_lm.comfy_registry_launch_verification import (
    VerifiedComfyRegistryLaunch,
    verify_comfy_registry_launch,
)
from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from local_lm.comfy_workflow_packages import WorkflowPackageRequirement
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ComfyRegistryInstall,
    ComfyRegistrySourceArtifactReview,
    Job,
    WorkflowInstallOffer,
)
from local_lm.registry_trust_policy import RegistryTrustDecision
from local_lm.runtime_provisioning_plans import RuntimeProvisioningPlan
from local_lm.workflow_install_offers import (
    bind_workflow_offer_downloads,
    mark_workflow_install_offer_queued,
)
from local_lm.workflow_offer_packages import (
    accepted_workflow_offer_packages,
    record_workflow_offer_package,
    stage_workflow_offer_packages,
)
from local_lm.workflow_package_drafts import stage_workflow_package_draft
from local_lm.workflow_package_execution_plan import plan_workflow_package_execution
from local_lm.workflow_package_extension_preflight import WorkflowPackageExtensionPreflight
from local_lm.workflow_package_install_plans import (
    WorkflowPackageInstallPlanRequest,
    create_workflow_package_install_plan,
)


async def _accepted_set(
    context: tuple[Session, ArtifactStore],
    root: Path,
    *,
    inactive: bool,
    empty_runtime: bool = False,
) -> dict[str, Any]:
    base = _inputs(context, root, commit=False, inactive=inactive)
    if empty_runtime:
        base["runtime"].clear()
    inputs = []
    plans = {}
    payload = _payload()
    available = {"LoraLoader", "EmptyLatentImage"}
    try:
        for index in range(2):
            selected = replace(
                base["selected"],
                package_id=f"neutral-pack-{index}",
                registry_record_id=f"neutral-record-{index}",
                node_types=(f"NeutralNode{index}",),
            )
            assert selected.declared_version is not None
            current = {
                **base,
                "selected": selected,
                "plan_arguments": {
                    **base["plan_arguments"],
                    "registry_client": _Registry(selected),
                    "requirement": WorkflowPackageRequirement(
                        selected.package_id,
                        (selected.declared_version,),
                        selected.node_types,
                        False,
                    ),
                },
            }
            plans[selected.package_id] = await plan_workflow_package_execution(
                **current["plan_arguments"]
            )
            inputs.append(current)
            available.update(selected.node_types)
            payload["ui_graph"]["nodes"].append(
                {
                    "id": 100 + index,
                    "type": selected.node_types[0],
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
                runtime_plan=RuntimeProvisioningPlan(
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
            assert saved.can_accept and saved.blockers == []
            _definition, revision = stage_workflow_package_draft(session, request)
            offer = WorkflowInstallOffer(
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
            downloads = [
                Job(kind="download", status="queued", payload_json=item.model_dump(mode="json"))
                for item in saved.download_requests
            ]
            session.add_all(downloads)
            bind_workflow_offer_downloads(session, offer, downloads)
            stage_workflow_offer_packages(session, offer, saved)
            mark_workflow_install_offer_queued(offer)
            packages = accepted_workflow_offer_packages(session, offer, saved)
            jobs = {package.link.package_id: package.job.id for package in packages}
            for package in packages:
                package.job.status = "running"
            identifier = offer.id
            session.commit()
        prepared = []
        for current in inputs:
            package_id = current["selected"].package_id
            result = await _prepare(current, plans[package_id])
            with SessionLocal() as session:
                current_offer = session.get(WorkflowInstallOffer, identifier)
                assert current_offer is not None
                record_workflow_offer_package(
                    session,
                    current_offer,
                    saved,
                    package_id=package_id,
                    job_id=jobs[package_id],
                    preparation=result,
                )
                session.commit()
            prepared.append(result)
        return dict(base=base, offer_id=identifier, prepared=prepared, jobs=jobs)
    except BaseException:
        await _close(base)
        raise


@pytest.mark.parametrize(
    "change",
    ["none", "inactive", "revoked", "declaration", "offer", "decline", "commit", "grant"],
)
async def test_reviewed_extension_grants_use_one_verified_set_and_commit_together(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        inputs = await _accepted_set((session, store), tmp_path, inactive=change == "inactive")
    base = inputs["base"]
    ids = [item.install_id for item in inputs["prepared"]]
    entered = threading.Event()
    release = threading.Event()
    file_verification_finished = False
    grants: list[str] = []
    task: asyncio.Task[trust_module.WorkflowSourceExtensionTrust] | None = None

    def verified(*args: Any, **kwargs: Any) -> VerifiedComfyRegistryLaunch:
        nonlocal file_verification_finished
        result = verify_comfy_registry_launch(*args, **kwargs)
        file_verification_finished = True
        entered.set()
        if not release.wait(30):
            raise AssertionError("Trust verification was not released")
        return result

    original_files = verify_staged_comfy_registry_archive

    def files(*args: Any, **kwargs: Any) -> Any:
        assert not file_verification_finished
        return original_files(*args, **kwargs)

    def grant(row: ComfyRegistryInstall, decision: RegistryTrustDecision) -> None:
        current = object_session(row)
        assert current is not None
        driver = current.connection().connection.driver_connection
        assert getattr(driver, "in_transaction", False)
        _apply_registry_policy_trust(row, decision)
        grants.append(row.id)
        if change == "grant" and len(grants) == 2:
            raise RuntimeError("Neutral grant failure")

    def factory() -> Session:
        session = SessionLocal()
        if change == "commit":

            def refuse_commit(_session: Session) -> None:
                with SessionLocal() as reader:
                    for identifier in ids:
                        row = reader.get(ComfyRegistryInstall, identifier)
                        assert row is not None and not row.trusted
                raise RuntimeError("Neutral commit failure")

            event.listen(session, "before_commit", refuse_commit)
        return session

    async def waiting() -> bool:
        return entered.is_set()

    try:
        monkeypatch.setattr(trust_module, "verify_comfy_registry_launch", verified)
        monkeypatch.setattr(trust_module, "_apply_registry_policy_trust", grant)
        monkeypatch.setattr(install_module, "verify_staged_comfy_registry_archive", files)
        task = asyncio.create_task(
            asyncio.to_thread(
                trust_module.trust_workflow_source_extensions,
                factory,
                inputs["offer_id"],
                context=base["context"],
                media_worker_stopped=True,
                reviewed_inputs=ComfyRegistryReviewedInputContext(
                    SessionLocal, store, base["environment"], ("py3-none-any",)
                ),
            )
        )
        await wait_until(waiting, bool, what="accepted extension verification")
        with SessionLocal() as writer:
            writer.connection().exec_driver_sql("PRAGMA busy_timeout=100")
            row = writer.get(ComfyRegistryInstall, ids[-1])
            assert row is not None
            if change == "revoked":
                writer.execute(delete(ComfyRegistrySourceArtifactReview))
            elif change == "declaration":
                row.pip_dependencies_json = ["alpha==1.0"]
            elif change == "offer":
                offer = writer.get(WorkflowInstallOffer, inputs["offer_id"])
                assert offer is not None
                offer.status = "invalidated"
            elif change == "decline":
                row.review_json = {
                    **row.review_json,
                    "reviewed_at": "2026-01-01T00:00:00+00:00",
                    "trusted_by_local_user": False,
                    "trust_authority": "local_user",
                }
            else:
                job = writer.get(Job, next(iter(inputs["jobs"].values())))
                assert job is not None
                job.phase = "Verified dependency fixture"
            writer.commit()
        release.set()
        if change in {"none", "inactive", "decline"}:
            result = await task
            assert result.state == ("review_required" if change == "decline" else "ready")
        else:
            with pytest.raises((ValueError, RuntimeError)):
                await task
        with SessionLocal() as session:
            for identifier in ids:
                row = session.get(ComfyRegistryInstall, identifier)
                assert row is not None and not row.active
                assert row.trusted is (change in {"none", "inactive"})
                assert "reviewed_wheel_closure" in row.review_json
                if row.trusted:
                    assert row.review_json["trust_authority"] == "registry-existing-review-v1"
        if change in {"none", "inactive"}:
            assert set(grants) == set(ids)
        elif change in {"revoked", "declaration", "offer", "decline"}:
            assert grants == []
    finally:
        release.set()
        if task is not None:
            with contextlib.suppress(Exception):
                await task
        await _close(base)
