from __future__ import annotations

import asyncio
import contextlib
import threading
from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import delete, select, text
from test_comfy_registry_reviewed_batches import _watch_files
from test_workflow_reviewed_extension_trust import _accepted_set
from test_workflow_reviewed_package_plan import _close, _prepare

from local_lm import comfy_registry_activation_batches as batches
from local_lm import workflow_source_extensions as preparation_module
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_interpreter import ComfyRegistryInterpreterError
from local_lm.comfy_registry_launch_verification import (
    VerifiedComfyRegistryLaunch,
    verify_comfy_registry_launch,
)
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ComfyRegistryInstall,
    ComfyRegistrySourceArtifactReview,
    WorkflowInstallOffer,
)
from local_lm.workflow_package_execution_plan import plan_workflow_package_execution
from local_lm.workflow_package_preparation import WorkflowPackagePreparationError
from local_lm.workflow_source_extension_trust import trust_workflow_source_extensions


@pytest.mark.parametrize(
    "change",
    [
        "active",
        "inactive",
        "empty",
        "target",
        "runtime",
        "empty-runtime-drift",
        "legacy-probe",
        "revoked",
        "offer",
        "declaration",
        "cancel",
        "active-empty",
        "missing-context",
        "probe-error",
        "recovery",
    ],
)
async def test_reused_reviewed_packages_require_a_fresh_target_and_drained_verification(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    store = ArtifactStore(settings)
    inactive = change in {"inactive", "empty", "empty-runtime-drift", "active-empty"}
    empty = change in {"empty", "empty-runtime-drift"}
    with SessionLocal() as session:
        inputs = await _accepted_set(
            (session, store), tmp_path, inactive=inactive, empty_runtime=empty
        )
    base = inputs["base"]
    context = base["context"]
    identifiers = [item.install_id for item in inputs["prepared"]]
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    held = change in {"revoked", "offer", "declaration", "cancel"}
    task: asyncio.Task[preparation_module.PreparedWorkflowExtensions] | None = None
    probes: list[Path] = []
    original_probe = base["plan_arguments"]["interpreter_probe"]
    original_worker = preparation_module._recover_and_verify

    async def probe(path: Path) -> Any:
        probes.append(path)
        if change == "probe-error":
            raise ComfyRegistryInterpreterError(
                "interpreter_probe_invalid_output", "Neutral target probe refusal"
            )
        environment, tags, runtime = await original_probe(path)
        if change == "target":
            environment = {**environment, "python_version": "0.0"}
        if change in {"runtime", "empty-runtime-drift"}:
            runtime = (ComfyRegistryRuntimeDistribution("helper", "3.0"),)
        if change == "legacy-probe":
            return environment, tags
        return environment, tags, runtime

    def verified(*args: Any, **kwargs: Any) -> VerifiedComfyRegistryLaunch:
        result = verify_comfy_registry_launch(*args, **kwargs)
        if held:
            entered.set()
            if not release.wait(30):
                raise AssertionError("Reused package verification was not released")
        return result

    def worker(*args: Any, **kwargs: Any) -> None:
        try:
            original_worker(*args, **kwargs)
        finally:
            finished.set()

    async def waiting() -> bool:
        return entered.is_set()

    async def settled() -> bool:
        return finished.is_set()

    try:
        if change == "active-empty":
            saved_runtime = list(base["runtime"])
            base["runtime"].clear()
            plan = await plan_workflow_package_execution(**base["plan_arguments"])
            other = await _prepare(base, plan)
            base["runtime"].extend(saved_runtime)
            with SessionLocal() as session:
                row = session.get(ComfyRegistryInstall, other.install_id)
                assert row is not None
                row.trusted = row.active = True
                session.commit()
        if change == "recovery":
            reviewed = ComfyRegistryReviewedInputContext(
                SessionLocal, store, base["environment"], ("py3-none-any",)
            )
            await asyncio.to_thread(
                trust_workflow_source_extensions,
                SessionLocal,
                inputs["offer_id"],
                context=context,
                media_worker_stopped=True,
                reviewed_inputs=reviewed,
            )
            with SessionLocal() as session:
                _, _, packages = preparation_module._accepted(session, inputs["offer_id"])
                plans = {
                    item.preparation.install_id: item.plan
                    for item in packages
                    if item.preparation is not None
                }
            batch, _ = await asyncio.to_thread(
                batches._begin,
                SessionLocal,
                inputs["offer_id"],
                inputs["prepared"],
                plans,
                context.custom_node_root,
                registry_wheel_environment_root(context.state_root),
                reviewed,
            )
            (batch.bindings[0].installed_path / "cache.json").write_text(
                '{"neutral":true}', encoding="utf-8"
            )
        if change == "missing-context":
            from dataclasses import replace

            context = replace(context, source_store=None)
        calls = _watch_files(monkeypatch)
        monkeypatch.setattr(preparation_module, "probe_comfy_registry_runtime_target", probe)
        monkeypatch.setattr(preparation_module, "verify_comfy_registry_launch", verified)
        monkeypatch.setattr(preparation_module, "_recover_and_verify", worker)

        async def run() -> preparation_module.PreparedWorkflowExtensions:
            try:
                return await preparation_module.prepare_workflow_source_extensions(
                    SessionLocal, inputs["offer_id"], context=context, media_worker_stopped=True
                )
            finally:
                if entered.is_set():
                    assert finished.is_set(), "Cleanup overtook reused-package verification"

        task = asyncio.create_task(run())
        if held:
            await wait_until(waiting, bool, what="reused package verification")
            with SessionLocal() as writer:
                writer.connection().exec_driver_sql("PRAGMA busy_timeout=100")
                if change == "revoked":
                    writer.execute(delete(ComfyRegistrySourceArtifactReview))
                elif change == "offer":
                    offer = writer.get(WorkflowInstallOffer, inputs["offer_id"])
                    assert offer is not None
                    offer.status = "invalidated"
                elif change == "declaration":
                    row = writer.get(ComfyRegistryInstall, identifiers[0])
                    assert row is not None
                    row.pip_dependencies_json = ["alpha==1.0"]
                else:
                    writer.execute(
                        text("UPDATE workflow_install_offers SET status = status WHERE 0")
                    )
                writer.commit()
            if change == "cancel":
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and not finished.is_set()
            release.set()
        if change in {"active", "inactive", "empty", "recovery"}:
            result = await task
            assert result.preparations == tuple(inputs["prepared"])
            assert result.reviewed_inputs is not None and calls
            assert result.reviewed_inputs.marker_environment == base["environment"]
            assert result.reviewed_inputs.supported_tags == ("py3-none-any",)
        elif change == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ValueError) as failure:
                await task
            if change == "probe-error":
                assert isinstance(failure.value, WorkflowPackagePreparationError)
                assert failure.value.code == "interpreter_probe_invalid_output"
        assert probes == ([] if change == "missing-context" else [context.python_executable])
        with SessionLocal() as session:
            rows = list(
                session.scalars(
                    select(ComfyRegistryInstall).where(ComfyRegistryInstall.id.in_(identifiers))
                )
            )
            assert len(rows) == 2
            assert all(not row.active and row.trusted == (change == "recovery") for row in rows)
            if change == "recovery":
                assert all(
                    row.review_json["activation_batch_v1"]["state"] == "failed" for row in rows
                )
    finally:
        release.set()
        if task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task
        if entered.is_set():
            await wait_until(settled, bool, what="reused package worker cleanup")
        await _close(base)
