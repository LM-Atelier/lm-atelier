from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import delete, text
from test_workflow_reviewed_extension_trust import _accepted_set
from test_workflow_reviewed_package_plan import _close

from local_lm import comfy_registry_activation_batches as batches
from local_lm import comfy_registry_installs as installs
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_launch_verification import (
    VerifiedComfyRegistryLaunch,
    _verify_comfy_registry_launch,
)
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall, ComfyRegistrySourceArtifactReview, Job
from local_lm.workflow_activations import WorkflowRegistryLaunchBinding
from local_lm.workflow_source_extension_trust import trust_workflow_source_extensions
from local_lm.workflow_source_extensions import _accepted


@pytest.fixture
async def reviewed_batch(
    client: AsyncClient, settings: Settings, tmp_path: Path, request: pytest.FixtureRequest
) -> AsyncIterator[dict[str, Any]]:
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        inputs = await _accepted_set(
            (session, store), tmp_path, inactive=getattr(request, "param", False)
        )
    base = inputs["base"]
    context = base["context"]
    reviewed = ComfyRegistryReviewedInputContext(
        SessionLocal, store, base["environment"], ("py3-none-any",)
    )
    try:
        result = await asyncio.to_thread(
            trust_workflow_source_extensions,
            SessionLocal,
            inputs["offer_id"],
            context=context,
            media_worker_stopped=True,
            reviewed_inputs=reviewed,
        )
        assert result.state == "ready"
        with SessionLocal() as session:
            _, _, packages = _accepted(session, inputs["offer_id"])
            plans = {
                package.preparation.install_id: package.plan
                for package in packages
                if package.preparation is not None
            }
        yield dict(
            purpose=inputs["offer_id"],
            preparations=inputs["prepared"],
            execution_plans=plans,
            custom_node_root=context.custom_node_root,
            environment_root=registry_wheel_environment_root(context.state_root),
            media_worker_stopped=True,
            reviewed_inputs=reviewed,
        )
    finally:
        await _close(base)


def _rows(arguments: dict[str, Any]) -> dict[str, tuple[bool, dict[str, Any]]]:
    with SessionLocal() as session:
        result = {}
        for preparation in arguments["preparations"]:
            row = session.get(ComfyRegistryInstall, preparation.install_id)
            assert row is not None
            result[row.id] = (row.active, row.review_json)
        return result


def _watch_files(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    main_thread = threading.get_ident()
    calls: list[str] = []

    def wrap(function: Callable[..., Any], name: str) -> Callable[..., Any]:
        def check(*args: Any, **kwargs: Any) -> Any:
            assert threading.get_ident() != main_thread
            with SessionLocal() as writer:
                writer.connection().exec_driver_sql("PRAGMA busy_timeout=100")
                writer.execute(text("UPDATE comfy_registry_installs SET active = active WHERE 0"))
                writer.commit()
            calls.append(name)
            return function(*args, **kwargs)

        return check

    for module, name in (
        (installs, "verify_staged_comfy_registry_archive"),
        (installs, "verify_comfy_registry_wheel_environment"),
        (batches, "_registry_install_launch_binding"),
        (batches, "capture_staged_comfy_registry_runtime_files"),
        (batches, "snapshot_staged_comfy_registry_files"),
    ):
        monkeypatch.setattr(module, name, wrap(getattr(module, name), name))
    return calls


@pytest.mark.parametrize("reviewed_batch", [False, True], indirect=True)
@pytest.mark.parametrize("recovery", [False, True])
async def test_reviewed_batch_verifies_files_off_loop_without_holding_the_writer(
    reviewed_batch: dict[str, Any], monkeypatch: pytest.MonkeyPatch, recovery: bool
) -> None:
    calls = _watch_files(monkeypatch)
    arguments = reviewed_batch
    if recovery:
        batch, _ = await asyncio.to_thread(
            batches._begin,
            SessionLocal,
            arguments["purpose"],
            arguments["preparations"],
            arguments["execution_plans"],
            arguments["custom_node_root"],
            arguments["environment_root"],
            arguments["reviewed_inputs"],
        )
        (batch.bindings[0].installed_path / "cache.json").write_text(
            '{"neutral":true}', encoding="utf-8"
        )
        await asyncio.to_thread(batches.recover_registry_package_batch, SessionLocal, **arguments)
        assert all(not active for active, _ in _rows(arguments).values())
    starts = 0

    async def start(bindings: tuple[WorkflowRegistryLaunchBinding, ...]) -> object:
        nonlocal starts
        starts += 1
        assert all(active for active, _ in _rows(arguments).values())
        (bindings[0].installed_path / "runtime.json").write_text(
            '{"neutral":true}', encoding="utf-8"
        )
        return object()

    async def stop() -> bool:
        raise AssertionError("Successful activation was stopped")

    async def inventory() -> frozenset[str]:
        return frozenset({"NeutralNode0", "NeutralNode1"})

    async with batches.activate_registry_package_batch(
        SessionLocal,
        **arguments,
        start_media=start,
        stop_media=stop,
        read_node_inventory=inventory,
    ) as batch:
        count = len(calls)
        with SessionLocal() as session:
            job = Job(kind="registry_prepare", status="complete", payload_json={})
            session.add(job)
            batch.complete(session)
            with SessionLocal() as reader:
                assert reader.get(Job, job.id) is None
                assert all(
                    review["activation_batch_v1"]["state"] == "verified"
                    for _, review in _rows(arguments).values()
                )
            session.commit()
        assert len(calls) == count
    assert starts == 1 and calls
    assert all(
        active and review["activation_batch_v1"]["state"] == "complete"
        for active, review in _rows(arguments).values()
    )


@pytest.mark.parametrize(
    "phase", ["begin", "verified", "complete", "declaration", "restore", "consumer"]
)
async def test_changed_review_or_identity_prevents_the_batch_transition(
    reviewed_batch: dict[str, Any], monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    arguments = reviewed_batch
    ids = list(_rows(arguments))
    if phase in {"restore", "consumer"}:
        with SessionLocal() as session:
            previous = session.get(ComfyRegistryInstall, ids[0])
            assert previous is not None
            previous.active = True
            session.commit()
    original = _verify_comfy_registry_launch
    calls = 0
    target = {
        "begin": 1,
        "verified": 2,
        "complete": 3,
        "declaration": 3,
        "restore": 4,
        "consumer": 0,
    }[phase]

    def verify(*args: Any, **kwargs: Any) -> VerifiedComfyRegistryLaunch:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == target:
            with SessionLocal() as session:
                if phase == "declaration":
                    changed = session.get(ComfyRegistryInstall, ids[-1])
                    assert changed is not None
                    changed.pip_dependencies_json = ["alpha==1.0"]
                else:
                    session.execute(delete(ComfyRegistrySourceArtifactReview))
                session.commit()
        return result

    monkeypatch.setattr(batches, "_verify_comfy_registry_launch", verify)
    _watch_files(monkeypatch)
    starts = stops = 0

    async def start(_bindings: tuple[WorkflowRegistryLaunchBinding, ...]) -> object:
        nonlocal starts
        starts += 1
        return object()

    async def stop() -> bool:
        nonlocal stops
        stops += 1
        return True

    async def inventory() -> frozenset[str]:
        return frozenset({"NeutralNode0", "NeutralNode1"})

    with pytest.raises((ValueError, RuntimeError)):
        async with batches.activate_registry_package_batch(
            SessionLocal,
            **arguments,
            start_media=start,
            stop_media=stop,
            read_node_inventory=inventory,
        ) as batch:
            if phase in {"restore", "consumer"}:
                raise RuntimeError("Neutral consumer failure")
            with SessionLocal() as session:
                batch.complete(session)
                session.commit()
    assert starts == stops == (0 if phase == "begin" else 1)
    assert calls >= target
    assert {identifier: active for identifier, (active, _) in _rows(arguments).items()} == {
        identifier: phase == "consumer" and identifier == ids[0] for identifier in ids
    }
    assert all(
        review.get("activation_batch_v1", {}).get("state") != "complete"
        for _, review in _rows(arguments).values()
    )


@pytest.mark.parametrize("phase", [1, 2])
async def test_batch_cancellation_drains_file_verification_before_restoring_flags(
    reviewed_batch: dict[str, Any], monkeypatch: pytest.MonkeyPatch, phase: int
) -> None:
    arguments = reviewed_batch
    entered = threading.Event()
    release = threading.Event()
    original = _verify_comfy_registry_launch
    calls = starts = stops = 0

    def verify(*args: Any, **kwargs: Any) -> VerifiedComfyRegistryLaunch:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == phase:
            entered.set()
            if not release.wait(30):
                raise AssertionError("Batch verification was not released")
        return result

    monkeypatch.setattr(batches, "_verify_comfy_registry_launch", verify)
    _watch_files(monkeypatch)

    async def start(_bindings: tuple[WorkflowRegistryLaunchBinding, ...]) -> object:
        nonlocal starts
        starts += 1
        return object()

    async def stop() -> bool:
        nonlocal stops
        stops += 1
        assert release.is_set()
        return True

    async def inventory() -> frozenset[str]:
        return frozenset({"NeutralNode0", "NeutralNode1"})

    async def run() -> None:
        async with batches.activate_registry_package_batch(
            SessionLocal,
            **arguments,
            start_media=start,
            stop_media=stop,
            read_node_inventory=inventory,
        ):
            raise AssertionError("Cancelled activation reached its consumer")

    async def waiting() -> bool:
        return entered.is_set()

    task = asyncio.create_task(run())
    try:
        await wait_until(waiting, bool, what="batch file verification")
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and stops == 0 and starts == phase - 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stops == 1
        assert all(not active for active, _ in _rows(arguments).values())
    finally:
        release.set()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task
