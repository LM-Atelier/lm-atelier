from __future__ import annotations

import asyncio
import contextlib
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import delete
from test_comfy_registry_reviewed_batches import _watch_files
from test_workflow_reviewed_extension_trust import _accepted_set
from test_workflow_reviewed_package_plan import _close

from local_lm import workflow_source_extensions, workflow_source_runtime
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ComfyRegistryInstall,
    ComfyRegistrySourceArtifactReview,
    WorkflowInstallOffer,
)
from local_lm.workflow_activations import (
    WorkflowSourceLaunchScope,
    materialize_comfy_runtime_dependency,
)
from local_lm.workflow_completion_jobs import (
    begin_workflow_completion,
    stage_workflow_completion_job,
)
from local_lm.workflow_package_preparation import PreparationContext
from local_lm.workflow_source_launch import revalidate_workflow_source_launch_scope


@pytest.mark.parametrize(
    "change",
    [
        "active",
        "inactive",
        "empty",
        "revoked-before-trust",
        "revoked-at-start",
        "revoked-after-start",
        "cancel-trust",
        "cancel-scope",
        "cancel-quarantine",
    ],
)
async def test_source_runtime_keeps_reviewed_context_and_drains_its_workers(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    import test_workflow_reviewed_extension_trust as fixture_module

    monkeypatch.setattr(
        fixture_module,
        "_payload",
        lambda: {
            "name": "Neutral reviewed extension workflow",
            "operation": "text_to_image",
            "ui_graph": {"version": 0.4, "nodes": [], "links": []},
            "dependencies": {"version": 1, "slots": []},
            "selections": [],
        },
    )
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        inputs = await _accepted_set(
            (session, store),
            tmp_path,
            inactive=change in {"inactive", "empty"},
            empty_runtime=change == "empty",
        )
    base = inputs["base"]
    context = base["context"]
    reviewed = ComfyRegistryReviewedInputContext(
        SessionLocal, store, base["environment"], ("py3-none-any",)
    )
    offer_id = inputs["offer_id"]
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    cleaned = False
    starts: list[WorkflowSourceLaunchScope] = []
    stops: list[bool] = []
    running = False
    task: asyncio.Task[None] | None = None

    def revoke() -> None:
        with SessionLocal() as session:
            session.execute(delete(ComfyRegistrySourceArtifactReview))
            session.commit()

    provisioner: Any = SimpleNamespace(
        status=lambda _engine: SimpleNamespace(state="ready", release="neutral-runtime")
    )

    def materializer(requirement: Any, selection: Any) -> Any:
        return materialize_comfy_runtime_dependency(provisioner, requirement, selection)

    async def start(*, activation_scope: WorkflowSourceLaunchScope) -> None:
        nonlocal running
        if change == "revoked-at-start":
            revoke()
        await asyncio.to_thread(
            revalidate_workflow_source_launch_scope,
            SessionLocal,
            activation_scope,
            context=context,
            runtime_materializer=materializer,
            reviewed_inputs=reviewed,
        )
        starts.append(activation_scope)
        running = True
        if change == "revoked-after-start":
            revoke()

    async def stop(_name: str) -> None:
        nonlocal running
        assert not entered.is_set() or finished.is_set()
        stops.append(True)
        running = False

    async def inventory() -> frozenset[str]:
        return frozenset({"NeutralNode0", "NeutralNode1"})

    processes: Any = SimpleNamespace(
        settings=settings,
        runtimes=provisioner,
        statuses=lambda: [
            SimpleNamespace(name="media", running=running, state="ready" if running else "stopped")
        ],
        start_media=start,
        stop=stop,
        comfy_node_inventory=inventory,
        launch_scope_sha256=lambda _name: starts[-1].launch_sha256,
    )
    original_prepare = workflow_source_extensions.prepare_workflow_source_extensions

    async def prepare(*args: Any, **kwargs: Any) -> Any:
        if change == "cancel-quarantine":
            raise ValueError("Neutral preparation refusal")
        result = await original_prepare(*args, **kwargs)
        if change == "revoked-before-trust":
            revoke()
        return result

    async def run() -> None:
        nonlocal cleaned
        try:
            async with workflow_source_runtime.prepared_workflow_source_runtime(
                processes, offer_id, attempt
            ) as prepared:
                assert prepared.batch is not None
                assert len(prepared.scope.registry_packages) == 2
                with SessionLocal() as session:
                    prepared.batch.complete(session)
                    session.commit()
        finally:
            cleaned = True

    async def waiting() -> bool:
        return entered.is_set()

    async def settled() -> bool:
        return finished.is_set()

    try:
        with SessionLocal() as session:
            offer = session.get(WorkflowInstallOffer, offer_id)
            assert offer is not None
            stage_workflow_completion_job(session, offer)
            job = begin_workflow_completion(session, offer)
            assert job is not None
            attempt = job.attempt
            session.commit()
        monkeypatch.setattr(
            PreparationContext, "from_settings", classmethod(lambda _cls, _s: context)
        )
        monkeypatch.setattr(
            workflow_source_extensions,
            "probe_comfy_registry_runtime_target",
            base["plan_arguments"]["interpreter_probe"],
        )
        monkeypatch.setattr(workflow_source_runtime, "prepare_workflow_source_extensions", prepare)
        if change.startswith("cancel-"):
            name = {
                "cancel-trust": "trust_workflow_source_extensions",
                "cancel-scope": "prepare_workflow_source_launch_scope",
                "cancel-quarantine": "quarantine_workflow_source_extensions",
            }[change]
            operation = getattr(workflow_source_runtime, name)

            def held(*args: Any, **kwargs: Any) -> Any:
                try:
                    entered.set()
                    if not release.wait(30):
                        raise AssertionError("Runtime verification was not released")
                    assert not cleaned
                    return operation(*args, **kwargs)
                finally:
                    finished.set()

            monkeypatch.setattr(workflow_source_runtime, name, held)
        files = _watch_files(monkeypatch)
        task = asyncio.create_task(run())
        if change.startswith("cancel-"):
            await wait_until(waiting, bool, what="runtime worker entered")
            task.cancel()
            for _ in range(3):
                await asyncio.sleep(0)
                task.cancel()
            assert not task.done() and not cleaned
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
        elif change.startswith("revoked-"):
            with pytest.raises(ValueError):
                await task
        else:
            await task
            assert len(starts) == 1 and not stops and files
        if change.startswith(("cancel-", "revoked-")):
            assert len(starts) == (1 if change == "revoked-after-start" else 0)
            assert not running
        with SessionLocal() as session:
            for preparation in inputs["prepared"]:
                row = session.get(ComfyRegistryInstall, preparation.install_id)
                assert row is not None
                assert row.active is (change in {"active", "inactive", "empty"})
    finally:
        release.set()
        if entered.is_set():
            await wait_until(settled, bool, what="runtime worker finished")
        if task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task
        await _close(base)
