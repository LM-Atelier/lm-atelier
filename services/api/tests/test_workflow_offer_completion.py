"""Real offer jobs must install their accepted bytes before activating a workflow."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_downloads import safetensors_bytes
from test_workflow_activations import _asset
from test_workflow_offer_download_acceptance import _created_offer
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm import workflow_asset_downloads
from local_lm.db import SessionLocal
from local_lm.model_manifests import inspect_repository_metadata
from local_lm.model_planner import ResolvedInstallPlan, persist_install_plan, resolve_install_plan
from local_lm.models import (
    Job,
    ModelAssetInstall,
    Run,
    WorkflowActivation,
    WorkflowDependencyBinding,
    WorkflowInstallOffer,
)
from local_lm.queue_lane_policy import change_lane_policy, read_lane_policy
from local_lm.scheduler import JobClaim
from local_lm.schemas import QueueControlCommand
from local_lm.workflow_activation_requests import WorkflowActivationOut

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "recover",
        "contract-recover",
        "contract-request",
        "replan",
        "revoke",
        "bytes",
        "request",
        "result",
        "failed",
        "cancelled",
        "late-request",
        "cancel-check",
        "install-paused",
        "install-retry",
        "legacy-recover",
        "manual",
        "manual-recover",
        "manual-other",
        "manual-request",
        "manual-bytes",
        "manual-late-bytes",
        "manual-revoke",
        "preferred",
        "preferred-optional",
        "preferred-any",
        "preferred-ambiguous",
        "batch-replaced",
        "batch-cleared",
        "batch-cancel-displaced",
        "batch-cancel-owned",
    ],
)
async def test_accepted_download_completes_offer_and_activates_exact_installed_dependency(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    legacy_contract = change.startswith("contract-")
    if legacy_contract:
        change = change.removeprefix("contract-")
    content = safetensors_bytes(["lora_unet_block.lora_down.weight"])
    digest = hashlib.sha256(content).hexdigest()
    filename, remote, revision = "detail.safetensors", "synthetic/neutral-detail", "d" * 40
    inspection = inspect_repository_metadata({filename: content}, [filename], role="image")
    planned = resolve_install_plan(
        remote_id=remote,
        revision=revision,
        role="image",
        engine="comfyui",
        selected_files=[{"filename": filename, "size": len(content), "sha256": digest}],
        inspection=inspection,
        comfy_paths={"loras": "."},
        auxiliary_kind="lora",
    )
    if legacy_contract:
        planned = replace(planned, resolver_version="install-resolver-v9")
        monkeypatch.setattr(
            workflow_asset_downloads, "INSTALL_RESOLVER_VERSION", "install-resolver-v9"
        )
    with SessionLocal() as session:
        plan = persist_install_plan(session, planned)
        session.commit()
        plan_id = plan.id
    services = app.state.services
    monkeypatch.setattr(services.engines.settings, "media_engine", "comfyui")
    manager = services.downloads
    # The network and worker boundary supply constructed bytes and node metadata.
    # Inspection, hashing, activation and every durable write remain production code.
    monkeypatch.setattr(
        manager,
        "_api",
        SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename=filename, size=len(content), lfs={"sha256": digest})
                ],
                sha=revision,
                pipeline_tag=None,
                tags=["lora"],
                gated=False,
            )
        ),
    )

    async def download_file(**kwargs: Any) -> str:
        target = kwargs["staging"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    async def start_media(*_args: object, **_kwargs: object) -> None:
        return None

    original_info = services.engines.media.object_info

    async def object_info() -> dict[str, Any]:
        return {
            **await original_info(),
            "LoraLoader": {
                "python_module": "nodes",
                "input": {"required": {"lora_name": [[filename]]}},
                "output": [],
            },
        }

    monkeypatch.setattr(manager, "_download_file", download_file)
    monkeypatch.setattr(manager, "start", lambda _job_id: None)
    monkeypatch.setattr(services.processes, "start_media", start_media)
    monkeypatch.setattr(services.engines.media, "object_info", object_info)
    monkeypatch.setattr(
        services.engines.media, "invalidate_object_info_cache", lambda: None, raising=False
    )
    monkeypatch.setattr(manager, "media_adapter", services.engines.media)
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral install completion",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": {"1": {"class_type": "LoraLoader", "inputs": {"lora_name": filename}}},
            "ui_graph": {
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
            },
            "dependencies": {
                "version": 1,
                "slots": [
                    {
                        "name": "detail",
                        "resource_kind": "model_asset",
                        "required": change != "preferred-optional",
                        "satisfaction": (
                            "any_of"
                            if change in {"preferred-any", "preferred-ambiguous"}
                            else "all_of"
                        ),
                        "requirements": [
                            {
                                "key": "detail",
                                "constraints": {"asset_kind": "lora", "sha256": digest},
                            },
                            *(
                                [
                                    {
                                        "key": "alternative",
                                        "constraints": {
                                            "asset_kind": "lora",
                                            "sha256": digest,
                                        },
                                    }
                                ]
                                if change == "preferred-ambiguous"
                                else []
                            ),
                        ],
                    },
                    *(
                        [
                            {
                                "name": "accent",
                                "resource_kind": "model_asset",
                                "required": change != "preferred-optional",
                                "satisfaction": "all_of",
                                "requirements": [
                                    {
                                        "key": "accent",
                                        "constraints": {
                                            "asset_kind": "lora",
                                            "sha256": hashlib.sha256(
                                                content + b"accent"
                                            ).hexdigest(),
                                        },
                                    }
                                ],
                            }
                        ]
                        if change.startswith("manual") or change == "preferred-optional"
                        else []
                    ),
                ],
            },
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()["id"]
    workflow_revision = created.json()["current_revision_id"]
    review_url = f"/api/workflows/{workflow}/revisions/{workflow_revision}/review"
    review = await client.get(review_url)
    assert review.status_code == 200, review.text
    approved = await client.post(
        review_url,
        json={
            "action": "approve",
            "subject_sha256": review.json()["subject_sha256"],
        },
    )
    assert approved.status_code == 200 and approved.json()["trusted"], approved.text
    offer = await client.post(
        f"/api/workflows/{workflow}/revisions/{workflow_revision}/install-offers",
        json={
            "selections": [
                {
                    "reference_filename": filename,
                    "install_plan_id": plan_id,
                    "artifact_path": filename,
                }
            ],
        },
    )
    assert offer.status_code == 201, offer.text
    offer_id = offer.json()["id"]
    if change == "install-paused":
        with SessionLocal() as session:
            policy = change_lane_policy(
                session,
                "install",
                "pause_after_current",
                QueueControlCommand(expected_revision=0, idempotency_key="pause-install"),
            )
            assert policy.dispatch_state == "paused"
    async with services.scheduler.lease("primary"):
        queued = await client.post(f"/api/workflow-install-offers/{offer_id}/install")
        assert queued.status_code == 202, queued.text
        if legacy_contract:
            monkeypatch.setattr(
                workflow_asset_downloads, "INSTALL_RESOLVER_VERSION", "install-resolver-v10"
            )
        job_id = queued.json()[0]["id"]
        with SessionLocal() as session:
            stored_offer = session.get(WorkflowInstallOffer, offer_id)
            assert stored_offer is not None and stored_offer.completion_job_id is not None
            completion_job_id = stored_offer.completion_job_id
            completion_job = session.get(Job, completion_job_id)
            assert completion_job is not None and completion_job.status == "queued"
            assert completion_job.kind == "workflow_install" and completion_job.claim_owner is None
        with SessionLocal() as session:
            assert session.scalar(select(WorkflowActivation)) is None
        if change == "legacy-recover":
            with SessionLocal() as session:
                stored_offer = session.get(WorkflowInstallOffer, offer_id)
                assert stored_offer is not None
                stored_offer.completion_job_id = None
                completion_job = session.get(Job, completion_job_id)
                assert completion_job is not None
                session.delete(completion_job)
                session.commit()
    profile = await client.post(
        "/api/profiles",
        json={
            "name": "Neutral completion profile",
            "role": "image",
            "engine": "comfyui",
        },
    )
    assert profile.status_code == 201, profile.text
    chat = await client.post("/api/chats", json={"title": "Neutral completion admission"})
    assert chat.status_code == 201, chat.text
    turn_url = f"/api/chats/{chat.json()['id']}/turns"
    payload = {
        "text": "A blue landscape",
        "mode": "image",
        "profile_id": profile.json()["id"],
        "workflow_revision_id": workflow_revision,
        "idempotency_key": "completed-offer-turn",
    }
    refused = await client.post(turn_url, json=payload)
    assert refused.status_code == 422, refused.text

    original_publish = services.events.publish
    attention: list[dict[str, object]] = []

    async def publish(
        event_type: str, entity_id: str, data: dict[str, object] | None = None
    ) -> None:
        if event_type == "download.completed" and entity_id == job_id:
            if change.startswith(("manual", "preferred")):
                with SessionLocal() as session:
                    _asset(
                        session,
                        services.settings.data_dir / "other-choice",
                        suffix="other-choice",
                        runtime_reference="other-choice.safetensors",
                        content=content,
                    )
                    for suffix in (
                        ("accent-left", "accent-right")
                        if change.startswith("manual") or change == "preferred-optional"
                        else ()
                    ):
                        _asset(
                            session,
                            services.settings.data_dir / suffix,
                            suffix=suffix,
                            runtime_reference=suffix + ".safetensors",
                            content=content + b"accent",
                        )
                    session.commit()
            if change == "revoke":
                revoked = await client.post(
                    review_url,
                    json={
                        "action": "revoke",
                        "subject_sha256": review.json()["subject_sha256"],
                    },
                )
                assert revoked.status_code == 200 and not revoked.json()["trusted"]
            elif change in {"bytes", "request", "result", "failed", "cancelled"}:
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    assert job is not None
                    if change == "bytes":
                        asset = session.get(ModelAssetInstall, job.result_json["model_asset_id"])
                        assert asset is not None
                        (Path(asset.local_path) / filename).write_bytes(content + b"changed")
                    elif change == "request":
                        job.payload_json = {**job.payload_json, "revision": "e" * 40}
                    elif change == "result":
                        job.result_json = {"model_asset_id": "missing-neutral-asset"}
                    else:
                        job.status = change
                    session.commit()
        if event_type == "workflow.install.attention":
            attention.append(data or {})
        await original_publish(event_type, entity_id, data)

    monkeypatch.setattr(services.events, "publish", publish)
    if (
        change == "late-request"
        and importlib.util.find_spec("local_lm.workflow_offer_completion") is not None
    ):
        completion_module = importlib.import_module("local_lm.workflow_offer_completion")
        original_activate = completion_module.activate_reviewed_revision

        def activate_then_change(
            session: Session, *args: Any, **kwargs: Any
        ) -> WorkflowActivationOut:
            result: WorkflowActivationOut = original_activate(session, *args, **kwargs)
            job = session.get(Job, job_id)
            assert job is not None
            job.payload_json = {**job.payload_json, "revision": "e" * 40}
            session.flush()
            return result

        monkeypatch.setattr(completion_module, "activate_reviewed_revision", activate_then_change)
    reconcile = getattr(manager, "reconcile_workflow_install_offers", None)
    if change in {"recover", "replan", "legacy-recover"} and reconcile is not None:

        async def defer_completion(_job_id: str) -> None:
            return None

        monkeypatch.setattr(manager, "reconcile_workflow_install_offers", defer_completion)
    complete = getattr(manager, "_complete_workflow_install_offer", None)
    if change.startswith("batch-"):
        await _check_completion_batch(
            app,
            client,
            monkeypatch,
            change,
            workflow,
            workflow_revision,
            offer_id,
            job_id,
            filename,
            plan_id,
            planned,
            attention,
        )
        return
    if change in {"cancel-check", "install-retry"} and complete is not None:
        entered = threading.Event()
        release = threading.Event()

        def held_completion(current_offer: str, *, claim: JobClaim) -> str | None:
            entered.set()
            assert release.wait(PATIENCE_SECONDS), "Completion was not released"
            result: str | None = complete(current_offer, claim=claim)
            return result

        monkeypatch.setattr(manager, "_complete_workflow_install_offer", held_completion)
        task = asyncio.create_task(manager._download(job_id))
        observer = None
        try:
            assert await asyncio.to_thread(entered.wait, PATIENCE_SECONDS)
            with SessionLocal() as session:
                completion_job = session.get(Job, completion_job_id)
                assert completion_job is not None and completion_job.claim_owner is not None
                assert completion_job.status == "running" and completion_job.attempt == 1
            with SessionLocal() as session:
                policy = change_lane_policy(
                    session,
                    "install",
                    "pause_after_current",
                    QueueControlCommand(expected_revision=0, idempotency_key="drain-install"),
                )
                assert policy.dispatch_state == "draining" and policy.running_jobs == 1
            if change == "install-retry":
                cancelled = await client.post(f"/api/jobs/{completion_job_id}/cancel")
                assert cancelled.status_code == 200, cancelled.text
                retried = await client.post(f"/api/jobs/{completion_job_id}/retry")
                assert retried.status_code == 200, retried.text
                with SessionLocal() as session:
                    retry_job = session.get(Job, completion_job_id)
                    assert retry_job is not None and retry_job.status == "queued"
                    assert retry_job.claim_owner is not None and retry_job.attempt == 1
                # Let the new dispatcher observe the old attempt's retained claim.
                await asyncio.wait_for(manager._offer_tasks[offer_id], timeout=PATIENCE_SECONDS)
            else:
                task.cancel()
            acquired: list[bool] = []

            async def acquire_after_completion() -> None:
                async with services.scheduler.lease("primary"):
                    acquired.append(True)

            observer = asyncio.create_task(acquire_after_completion())
            await asyncio.sleep(0)
            assert not acquired
            release.set()
            if change == "install-retry":
                await task
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
            await observer
            assert acquired == [True]
            with SessionLocal() as session:
                policy = read_lane_policy(session, "install")
                assert policy.dispatch_state == "paused" and policy.running_jobs == 0
            if change == "install-retry":
                with SessionLocal() as session:
                    stored_offer = session.get(WorkflowInstallOffer, offer_id)
                    assert stored_offer is not None and stored_offer.completion_error_code is None
                    assert stored_offer.status == "queued"
                with SessionLocal() as session:
                    change_lane_policy(
                        session,
                        "install",
                        "resume",
                        QueueControlCommand(
                            expected_revision=policy.revision, idempotency_key="retry-install"
                        ),
                    )
                await services.scheduler.queue_control_changed("install")
                async with asyncio.timeout(PATIENCE_SECONDS):
                    while True:
                        with SessionLocal() as session:
                            retry_job = session.get(Job, completion_job_id)
                            assert retry_job is not None
                            if retry_job.status == "complete":
                                assert retry_job.attempt == 2
                                break
                        await asyncio.sleep(0.01)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            if observer is not None:
                await asyncio.gather(observer, return_exceptions=True)
    elif change == "install-paused":
        task = asyncio.create_task(manager._download(job_id))
        try:
            async with asyncio.timeout(PATIENCE_SECONDS):
                while True:
                    with SessionLocal() as session:
                        download = session.get(Job, job_id)
                        completion_job = session.get(Job, completion_job_id)
                        assert completion_job is not None
                        assert completion_job.status == "queued"
                        assert completion_job.claim_owner is None and completion_job.attempt == 0
                        assert session.scalar(select(WorkflowActivation)) is None
                        if download is not None and download.status == "complete":
                            break
                    await asyncio.sleep(0.01)
            with SessionLocal() as session:
                change_lane_policy(
                    session,
                    "install",
                    "resume",
                    QueueControlCommand(expected_revision=1, idempotency_key="resume-install"),
                )
            await services.scheduler.queue_control_changed("install")
            await asyncio.wait_for(task, timeout=PATIENCE_SECONDS)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    else:
        await manager._download(job_id)
    if change.startswith("manual"):
        await _check_manual_completion(
            app,
            client,
            monkeypatch,
            change,
            workflow,
            workflow_revision,
            offer_id,
            job_id,
            filename,
            content,
            review_url,
        )
        return
    if change in {"recover", "replan", "legacy-recover"}:
        with SessionLocal() as session:
            stored = session.get(WorkflowInstallOffer, offer_id)
            assert stored is not None and stored.status == "queued"
            if change == "replan":
                repeated = persist_install_plan(session, planned)
                assert repeated.id == plan_id and repeated.status == "planned"
                session.commit()
        if reconcile is not None:
            monkeypatch.setattr(manager, "reconcile_workflow_install_offers", reconcile)
        manager.recover_interrupted()
        recovery = getattr(manager, "_offer_recovery_task", None)
        if recovery is not None:
            await recovery

    # A new request has no dependency on delivery of the live attention event.
    progress_url = f"/api/workflow-install-offers/{offer_id}/progress"
    progress = await client.get(progress_url)
    assert progress.status_code == 200, progress.text
    snapshot = progress.json()
    successful = change in {
        "none",
        "recover",
        "replan",
        "cancel-check",
        "install-paused",
        "install-retry",
        "legacy-recover",
        "preferred",
        "preferred-optional",
        "preferred-any",
    }
    assert snapshot["phase"] == ("completed" if successful else "needs_attention")
    assert snapshot["total_downloads"] == 1
    assert snapshot["completed_downloads"] == (0 if change in {"failed", "cancelled"} else 1)
    assert snapshot["failed_downloads"] == (1 if change == "failed" else 0)
    assert snapshot["cancelled_downloads"] == (1 if change == "cancelled" else 0)
    expected_codes = {
        "revoke": "workflow-review-required",
        "bytes": "workflow-completion-unavailable",
        "request": "download-acceptance-changed",
        "result": "download-result-unavailable",
        "failed": "workflow-download-failed",
        "cancelled": "workflow-download-failed",
        "late-request": "download-acceptance-changed",
        "preferred-ambiguous": "workflow-dependencies-need-selection",
    }
    assert snapshot["attention_code"] == expected_codes.get(change)
    families = await client.get("/api/workflow-families?include_archived=true")
    assert families.status_code == 200, families.text
    variant = next(
        variant
        for family in families.json()
        for variant in family["variants"]
        if variant["id"] == workflow
    )
    assert variant["install_progress"] == snapshot
    assert set(snapshot) == {
        "id",
        "workflow_revision_id",
        "status",
        "phase",
        "total_downloads",
        "completed_downloads",
        "failed_downloads",
        "cancelled_downloads",
        "paused_downloads",
        "pending_downloads",
        "unavailable_downloads",
        "attention_code",
        "retry_job_id",
    }
    # Reading progress must not retry a download, invoke verification or mutate history.
    with monkeypatch.context() as reading:

        def forbidden(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("Reading progress started work")

        reading.setattr(manager, "start", forbidden)
        reading.setattr(manager, "_complete_workflow_install_offer", forbidden)
        again = await client.get(progress_url)
        assert again.status_code == 200 and again.json() == snapshot
    with SessionLocal() as session:
        durable = session.get(WorkflowInstallOffer, offer_id)
        assert durable is not None
        assert durable.completion_error_code == snapshot["attention_code"]

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        expected_status = change if change in {"failed", "cancelled"} else "complete"
        assert job is not None and job.status == expected_status, (
            job.error if job else "missing job"
        )
        asset = (
            session.get(ModelAssetInstall, job.result_json["model_asset_id"])
            if change != "result"
            else session.scalar(select(ModelAssetInstall))
        )
        assert asset is not None and asset.active and asset.verified_at is not None
        if change != "result":
            assert job.result_json["model_asset_id"] == asset.id
        assert asset.manifest_json["sha256"] == digest
        stored = session.get(WorkflowInstallOffer, offer_id)
        if not successful:
            assert stored is not None and stored.status == "queued" and stored.completed_at is None
            assert session.scalar(select(WorkflowActivation)) is None
            assert len(attention) == 1 and attention[0].get("code")
    if not successful:
        if change == "revoke":
            review_again = await client.get(review_url)
            approved_again = await client.post(
                review_url,
                json={"action": "approve", "subject_sha256": review_again.json()["subject_sha256"]},
            )
            assert approved_again.status_code == 200 and approved_again.json()["trusted"]
            await manager.reconcile_workflow_install_offers(job_id)
            recovered = await client.get(progress_url)
            assert recovered.status_code == 200
            assert recovered.json()["phase"] == "completed"
            assert recovered.json()["attention_code"] is None
            with SessionLocal() as session:
                recovered_offer = session.get(WorkflowInstallOffer, offer_id)
                assert recovered_offer is not None
                assert recovered_offer.completion_error_code is None
                assert recovered_offer.status == "completed"
        return
    with SessionLocal() as session:
        stored = session.get(WorkflowInstallOffer, offer_id)
        assert (
            stored is not None and stored.status == "completed" and stored.completed_at is not None
        )
        activation = session.scalar(
            select(WorkflowActivation).where(
                WorkflowActivation.workflow_revision_id == workflow_revision,
                WorkflowActivation.is_active.is_(True),
            )
        )
        assert activation is not None
        activation_id = activation.id
        completion_job = session.get(Job, stored.completion_job_id)
        assert completion_job is not None and completion_job.status == "complete"
        assert completion_job.claim_owner is None
        assert completion_job.attempt == (2 if change == "install-retry" else 1)
        assert completion_job.result_json == {"version": 1, "activation_id": activation_id}
        bindings = session.scalars(
            select(WorkflowDependencyBinding).where(
                WorkflowDependencyBinding.workflow_activation_id == activation_id,
            )
        ).all()
        assert len(bindings) == 1
        assert bindings[0].model_asset_install_id == asset.id
    async with services.scheduler.lease("primary"):
        accepted = await client.post(turn_url, json=payload)
        assert accepted.status_code == 202, accepted.text
        with SessionLocal() as session:
            run = session.get(Run, accepted.json()["run"]["id"])
            assert run is not None
            assert run.provenance_json["workflow"]["activation"]["id"] == activation_id


async def _check_completion_batch(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    workflow: str,
    workflow_revision: str,
    offer_id: str,
    job_id: str,
    filename: str,
    plan_id: str,
    planned: ResolvedInstallPlan,
    attention: list[dict[str, object]],
) -> None:
    services = app.state.services
    manager = services.downloads
    reconcile = manager.reconcile_workflow_install_offers
    complete = manager._complete_workflow_install_offer
    from local_lm import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.02)

    async def deferred(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(manager, "reconcile_workflow_install_offers", deferred)
    with SessionLocal() as session:
        persist_install_plan(session, planned)
        session.commit()
    next_offer = await client.post(
        f"/api/workflows/{workflow}/revisions/{workflow_revision}/install-offers",
        json={
            "selections": [
                {
                    "reference_filename": filename,
                    "install_plan_id": plan_id,
                    "artifact_path": filename,
                }
            ]
        },
    )
    assert next_offer.status_code == 201, next_offer.text
    next_id = next_offer.json()["id"]
    assert next_id != offer_id
    async with services.scheduler.lease("primary"):
        next_download = await client.post(f"/api/workflow-install-offers/{next_id}/install")
        assert next_download.status_code == 202, next_download.text
    await manager._download(job_id)
    next_job_id = next_download.json()[0]["id"]
    if next_job_id != job_id:
        await manager._download(next_job_id)
    with SessionLocal() as session:
        order = list(
            session.scalars(
                select(WorkflowInstallOffer.id)
                .where(WorkflowInstallOffer.status == "queued")
                .distinct()
            )
        )
        assert set(order) == {offer_id, next_id}
        first_offer = session.get(WorkflowInstallOffer, order[0])
        assert first_offer is not None and first_offer.source_plan_id is None
        first_job_id = first_offer.completion_job_id
        assert first_job_id is not None
    held_entered, held_release = threading.Event(), threading.Event()
    visited: list[str] = []

    def held(current_offer: str, *, claim: JobClaim) -> str | None:
        visited.append(current_offer)
        if current_offer == order[0]:
            held_entered.set()
            assert held_release.wait(30), "Completion was not released"
        result: str | None = complete(current_offer, claim=claim)
        return result

    def first_state() -> tuple[object, ...]:
        with SessionLocal() as session:
            job = session.get(Job, first_job_id)
            stored = session.get(WorkflowInstallOffer, order[0])
            assert job is not None and stored is not None
            return (
                job.claim_owner,
                job.attempt,
                job.queue_group,
                job.claim_expires_at,
                job.heartbeat_at,
                job.status,
                job.completed_at,
                job.error,
                dict(job.result_json),
                stored.status,
                stored.completed_at,
                stored.completion_error_code,
            )

    monkeypatch.setattr(manager, "_complete_workflow_install_offer", held)
    assert reconcile is not None
    batch = asyncio.create_task(reconcile())
    displaced = change != "batch-cancel-owned"
    caller_cancelled = change.startswith("batch-cancel-")
    try:
        assert await asyncio.to_thread(held_entered.wait, 30)
        if displaced:
            with SessionLocal() as session:
                job = session.get(Job, first_job_id)
                assert job is not None and job.claim_owner is not None
                job.claim_owner = None if change == "batch-cleared" else "replacement-attempt"
                job.attempt += 1
                if job.claim_owner is not None:
                    job.queue_group = "replacement"
                session.commit()
            before = first_state()
            heartbeat = next(
                task
                for task in asyncio.all_tasks()
                if task.get_name() == f"job-heartbeat-{first_job_id}"
            )
            await asyncio.wait_for(asyncio.shield(heartbeat), timeout=PATIENCE_SECONDS)
        if caller_cancelled:
            batch.cancel()
            await asyncio.sleep(0)
            batch.cancel()
            await asyncio.sleep(0)
        assert not batch.done()
        assert visited == [order[0]]
    finally:
        held_release.set()
        await asyncio.wait_for(asyncio.gather(batch, return_exceptions=True), timeout=30)
    assert batch.cancelled() == caller_cancelled
    if not caller_cancelled:
        assert batch.result() is None
        assert visited == order
        with SessionLocal() as session:
            later = session.get(WorkflowInstallOffer, order[1])
            assert later is not None and later.status == "completed"
            later_job = session.get(Job, later.completion_job_id)
            assert later_job is not None and later_job.status == "complete"
    else:
        assert visited == [order[0]]
        with SessionLocal() as session:
            later = session.get(WorkflowInstallOffer, order[1])
            assert later is not None and later.status == "queued"
    if displaced:
        assert first_state() == before
    assert attention == []


async def _check_manual_completion(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    workflow: str,
    revision: str,
    offer_id: str,
    job_id: str,
    filename: str,
    content: bytes,
    review_url: str,
) -> None:
    manager = app.state.services.downloads
    progress_url = f"/api/workflow-install-offers/{offer_id}/progress"
    waiting = await client.get(progress_url)
    assert waiting.json()["phase"] == "needs_attention"
    assert waiting.json()["attention_code"] == "workflow-dependencies-need-selection"
    activation_url = f"/api/workflows/{workflow}/revisions/{revision}/activation"
    prepared = await client.get(activation_url + "/prepare")
    assert prepared.status_code == 200, prepared.text
    slots = {slot["name"]: slot for slot in prepared.json()["slots"]}
    choices = slots["detail"]["choices"]
    assert len(choices) == 2
    accent_choices = slots["accent"]["choices"]
    assert len(accent_choices) == 2
    accent_selection = next(
        choice["selection"]
        for choice in accent_choices
        if choice["selection"]["local_id"] == "asset_accent-left"
    )
    other_offer_id = None
    if change == "manual":
        other_offer_id = await _created_offer(app, client, monkeypatch)
        queued = await client.post(f"/api/workflow-install-offers/{other_offer_id}/install")
        assert queued.status_code == 202, queued.text
    completed_offers: list[str] = []
    complete = manager._complete_workflow_install_offer

    def record_completed_offer(current_offer_id: str, *, claim: JobClaim) -> str | None:
        completed_offers.append(current_offer_id)
        result: str | None = complete(current_offer_id, claim=claim)
        return result

    monkeypatch.setattr(manager, "_complete_workflow_install_offer", record_completed_offer)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        accepted_asset_id = job.result_json["model_asset_id"]
        asset = session.get(ModelAssetInstall, accepted_asset_id)
        assert asset is not None
        if change == "manual-bytes":
            (Path(asset.local_path) / filename).write_bytes(content + b"changed")
        elif change == "manual-request":
            job.payload_json = {**job.payload_json, "revision": "e" * 40}
            session.commit()
    if change == "manual-revoke":
        current_review = await client.get(review_url)
        revoked = await client.post(
            review_url,
            json={
                "action": "revoke",
                "subject_sha256": current_review.json()["subject_sha256"],
            },
        )
        assert revoked.status_code == 200
    selected_id = "asset_other-choice" if change == "manual-other" else accepted_asset_id
    selection = next(
        choice["selection"] for choice in choices if choice["selection"]["local_id"] == selected_id
    )
    reconcile = manager.reconcile_workflow_install_offers
    calls: list[dict[str, object]] = []

    async def recording(*args: Any, **kwargs: Any) -> None:
        calls.append(kwargs)
        if change == "manual-late-bytes":
            with SessionLocal() as session:
                asset = session.get(ModelAssetInstall, accepted_asset_id)
                assert asset is not None
                (Path(asset.local_path) / filename).write_bytes(content + b"changed")
        if change != "manual-recover":
            await reconcile(*args, **kwargs)

    monkeypatch.setattr(manager, "reconcile_workflow_install_offers", recording)
    activated = await client.post(
        activation_url,
        json={
            "workflow_artifact_sha256": prepared.json()["workflow_artifact_sha256"],
            "dependency_contract_sha256": prepared.json()["dependency_contract_sha256"],
            "selections": [selection, accent_selection],
        },
    )
    if change in {"manual-bytes", "manual-revoke"}:
        assert activated.status_code == 409, activated.text
        assert calls == []
        with SessionLocal() as session:
            assert session.scalar(select(WorkflowActivation)) is None
            offer = session.get(WorkflowInstallOffer, offer_id)
            assert offer is not None and offer.status == "queued"
        return
    assert activated.status_code == 200, activated.text
    activation_id = activated.json()["id"]
    assert calls == [{"workflow_revision_id": revision}]
    if change == "manual-recover":
        with SessionLocal() as session:
            offer = session.get(WorkflowInstallOffer, offer_id)
            assert offer is not None and offer.status == "queued"
        monkeypatch.setattr(manager, "reconcile_workflow_install_offers", reconcile)
        manager.recover_interrupted()
        await manager._offer_recovery_task
    expected_code = {
        "manual-other": "download-results-need-binding",
        "manual-request": "download-acceptance-changed",
        "manual-late-bytes": "workflow-completion-unavailable",
    }.get(change)
    result = await client.get(progress_url)
    assert result.status_code == 200
    assert result.json()["phase"] == ("needs_attention" if expected_code else "completed")
    assert result.json()["attention_code"] == expected_code
    with SessionLocal() as session:
        activations = list(session.scalars(select(WorkflowActivation)))
        assert len(activations) == 1
        assert activations[0].id == activation_id and activations[0].is_active
        bindings = list(session.scalars(select(WorkflowDependencyBinding)))
        assert {binding.model_asset_install_id for binding in bindings} == {
            selected_id,
            "asset_accent-left",
        }
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        assert (offer.completed_at is not None) == (expected_code is None)
    assert completed_offers == [offer_id]
    if other_offer_id is not None:
        with SessionLocal() as session:
            other = session.get(WorkflowInstallOffer, other_offer_id)
            assert other is not None and other.status == "queued" and other.completed_at is None
