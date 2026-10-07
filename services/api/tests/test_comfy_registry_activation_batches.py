"""Prepared extensions activate as one resumable set before workflow completion commits."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx2 import AsyncClient
from sqlalchemy import select, text
from test_workflow_package_execution_plan import _inputs, _plan
from test_workflow_package_preparation import _Registry

from local_lm.comfy_registry_activation import (
    activate_comfy_registry_install,
    record_registry_policy_trust,
)
from local_lm.comfy_registry_lifecycle import ComfyRegistryPreparation
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.comfy_workflow_packages import WorkflowPackageRequirement
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall, Job
from local_lm.workflow_activations import WorkflowRegistryLaunchBinding
from local_lm.workflow_package_execution_plan import WorkflowPackageExecutionPlan
from local_lm.workflow_package_preparation import PreparationContext, prepare_workflow_package

pytestmark = pytest.mark.asyncio
NODES = frozenset({"NeutralNode0", "NeutralNode1"})


async def _prepared(
    tmp_path: Path,
) -> tuple[
    PreparationContext,
    tuple[ComfyRegistryPreparation, ...],
    dict[str, WorkflowPackageExecutionPlan],
]:
    preparations: list[ComfyRegistryPreparation] = []
    execution_plans: dict[str, WorkflowPackageExecutionPlan] = {}
    first = _inputs()
    context = PreparationContext(first["python_executable"], tmp_path / "nodes", tmp_path / "state")
    await first["archive_downloader"].close()
    context.custom_node_root.mkdir()
    context.state_root.mkdir()
    wheels = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    )
    try:
        for number in range(2):
            inputs = _inputs()
            selected = replace(
                inputs["selected"],
                package_id=f"neutral-pack-{number}",
                registry_record_id=f"neutral-record-{number}",
                node_types=(f"NeutralNode{number}",),
            )
            assert selected.declared_version is not None
            inputs["registry_client"] = _Registry(selected)
            inputs["requirement"] = WorkflowPackageRequirement(
                selected.package_id, (selected.declared_version,), selected.node_types, False
            )
            try:
                plan = await _plan(inputs)
                preparation = await prepare_workflow_package(
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
                    archive_downloader=inputs["archive_downloader"],
                    wheel_downloader=wheels,
                    expected_plan=plan,
                )
                with SessionLocal() as session:
                    record_registry_policy_trust(
                        session,
                        install_id=preparation.install_id,
                        resolution=selected,
                        expected_archive_sha256=preparation.archive_sha256,
                        expected_manifest_sha256=preparation.manifest_sha256,
                        custom_node_root=context.custom_node_root,
                        environment_root=registry_wheel_environment_root(context.state_root),
                        media_worker_stopped=True,
                    )
                preparations.append(preparation)
                execution_plans[preparation.install_id] = plan
            finally:
                await inputs["archive_downloader"].close()
    finally:
        await wheels.close()
    return context, tuple(preparations), execution_plans


def _rows() -> dict[str, tuple[bool, bool, dict[str, Any]]]:
    with SessionLocal() as session:
        return {
            row.id: (row.active, row.trusted, row.review_json)
            for row in session.scalars(select(ComfyRegistryInstall))
        }


def _arguments(
    context: PreparationContext,
    preparations: tuple[ComfyRegistryPreparation, ...],
    execution_plans: dict[str, WorkflowPackageExecutionPlan],
) -> dict[str, Any]:
    return {
        "purpose": "neutral-source-installation",
        "preparations": preparations,
        "execution_plans": execution_plans,
        "custom_node_root": context.custom_node_root,
        "environment_root": registry_wheel_environment_root(context.state_root),
        "media_worker_stopped": True,
    }


@pytest.mark.parametrize("mode", ["ordinary", "runtime-data", "writer", "previous-active"])
async def test_batch_starts_exactly_once_after_all_packages_commit_and_finishes_with_its_consumer(
    client: AsyncClient,
    tmp_path: Path,
    mode: str,
) -> None:
    from local_lm.comfy_registry_activation_batches import activate_registry_package_batch

    context, preparations, execution_plans = await _prepared(tmp_path)
    if mode == "previous-active":

        async def earlier_start() -> object:
            return object()

        with SessionLocal() as session:
            await activate_comfy_registry_install(
                session,
                install_id=preparations[0].install_id,
                custom_node_root=context.custom_node_root,
                environment_root=registry_wheel_environment_root(context.state_root),
                media_worker_stopped=True,
                start_media=earlier_start,
            )
    starts = 0
    stops = 0

    async def start(bindings: tuple[WorkflowRegistryLaunchBinding, ...]) -> object:
        nonlocal starts
        starts += 1
        assert {item.registry_install_id for item in bindings} == {
            item.install_id for item in preparations
        }
        assert all(active and trusted for active, trusted, _review in _rows().values())
        if mode == "runtime-data":
            (bindings[0].installed_path / "cache.json").write_text(
                '{"neutral":true}', encoding="utf-8"
            )
        if mode == "writer":

            def write() -> None:
                with SessionLocal() as session:
                    session.execute(
                        text("UPDATE comfy_registry_installs SET active = active WHERE 0")
                    )
                    session.commit()

            await asyncio.to_thread(write)
        return object()

    async def stop() -> bool:
        nonlocal stops
        stops += 1
        return True

    async def inventory() -> frozenset[str]:
        return NODES

    async with activate_registry_package_batch(
        SessionLocal,
        **_arguments(context, preparations, execution_plans),
        start_media=start,
        stop_media=stop,
        read_node_inventory=inventory,
    ) as batch:
        assert all(
            review["activation_batch_v1"]["state"] == "verified"
            for _, _, review in _rows().values()
        )
        with SessionLocal() as session:
            job = Job(kind="registry_prepare", status="complete", payload_json={})
            session.add(job)
            batch.complete(session)
            with SessionLocal() as reader:
                assert reader.get(Job, job.id) is None
            session.commit()
    assert starts == 1 and stops == 0
    assert all(
        active and review["activation_batch_v1"]["state"] == "complete"
        for active, _, review in _rows().values()
    )


@pytest.mark.parametrize(
    "failure", ["start", "inventory", "code", "consumer", "uncommitted", "revoked"]
)
async def test_failed_batch_stops_before_restoring_all_prior_activation_flags(
    client: AsyncClient,
    tmp_path: Path,
    failure: str,
) -> None:
    from local_lm.comfy_registry_activation_batches import activate_registry_package_batch

    context, preparations, execution_plans = await _prepared(tmp_path)
    starts = stops = 0

    async def start(bindings: tuple[WorkflowRegistryLaunchBinding, ...]) -> object:
        nonlocal starts
        starts += 1
        if failure == "start":
            raise RuntimeError("Neutral start failure")
        if failure == "code":
            (bindings[0].installed_path / "__init__.py").write_text(
                "CHANGED = True\n", encoding="utf-8"
            )
        if failure == "revoked":
            with SessionLocal() as session:
                row = session.get(ComfyRegistryInstall, preparations[0].install_id)
                assert row is not None
                row.trusted = False
                session.commit()
        return object()

    async def stop() -> bool:
        nonlocal stops
        stops += 1
        assert all(active for active, _trusted, _review in _rows().values())
        return True

    async def inventory() -> frozenset[str]:
        return frozenset() if failure == "inventory" else NODES

    with pytest.raises((ValueError, RuntimeError)):
        async with activate_registry_package_batch(
            SessionLocal,
            **_arguments(context, preparations, execution_plans),
            start_media=start,
            stop_media=stop,
            read_node_inventory=inventory,
        ) as batch:
            if failure == "consumer":
                raise RuntimeError("Neutral workflow compilation failure")
            with SessionLocal() as session:
                batch.complete(session)
                if failure != "uncommitted":
                    session.commit()
    assert starts == 1 and stops == 1
    assert all(
        not active and review["activation_batch_v1"]["state"] == "failed"
        for active, _, review in _rows().values()
    )
    if failure == "revoked":
        assert not _rows()[preparations[0].install_id][1]


@pytest.mark.parametrize(
    "failure",
    [
        "untrusted",
        "duplicate",
        "running",
        "identity",
        "node-collision",
        "package_id",
        "package_version",
        "registry_record_id",
        "repository_url",
        "download_url",
        "warnings",
        "dependencies",
        "missing-plan",
        "swapped-plans",
        "plan-digest",
    ],
)
async def test_invalid_batch_never_starts_or_commits_partial_activation(
    client: AsyncClient,
    tmp_path: Path,
    failure: str,
) -> None:
    from local_lm.comfy_registry_activation_batches import activate_registry_package_batch

    context, preparations, execution_plans = await _prepared(tmp_path)
    arguments = _arguments(context, preparations, execution_plans)
    if failure in {"untrusted", "node-collision"}:
        with SessionLocal() as session:
            row = session.get(ComfyRegistryInstall, preparations[1].install_id)
            assert row is not None
            if failure == "untrusted":
                row.trusted = False
            else:
                row.node_types_json = ["NeutralNode0"]
            session.commit()
    elif failure == "duplicate":
        arguments["preparations"] = (preparations[0], preparations[0])
    elif failure == "identity":
        arguments["preparations"] = (
            replace(preparations[0], archive_sha256="a" * 64),
            preparations[1],
        )
    elif failure == "running":
        arguments["media_worker_stopped"] = False
    elif failure == "missing-plan":
        arguments["execution_plans"] = {}
    elif failure == "swapped-plans":
        arguments["execution_plans"] = {
            preparations[0].install_id: execution_plans[preparations[1].install_id],
            preparations[1].install_id: execution_plans[preparations[0].install_id],
        }
    elif failure == "plan-digest":
        execution_plans[preparations[0].install_id] = execution_plans[
            preparations[0].install_id
        ].model_copy(update={"plan_sha256": "b" * 64})
    else:
        with SessionLocal() as session:
            row = session.get(ComfyRegistryInstall, preparations[0].install_id)
            assert row is not None
            if failure == "warnings":
                row.review_json = {**row.review_json, "registry_warnings": ["neutral-warning"]}
            elif failure == "dependencies":
                row.pip_dependencies_json = ["neutral-wheel==1.0.0"]
            else:
                setattr(row, failure, "neutral-changed")
            session.commit()

    async def forbidden(*_args: object) -> Any:
        raise AssertionError("An invalid batch reached a worker action")

    with pytest.raises(ValueError):
        async with activate_registry_package_batch(
            SessionLocal,
            **arguments,
            start_media=forbidden,
            stop_media=forbidden,
            read_node_inventory=forbidden,
        ):
            raise AssertionError("An invalid batch reached its consumer")
    assert all(
        not active and "activation_batch_v1" not in review for active, _, review in _rows().values()
    )


@pytest.mark.parametrize("stage", ["starting", "verified", "runtime-data", "verified-runtime-data"])
async def test_interrupted_batch_reopens_its_durable_snapshot_and_can_restore_original_flags(
    client: AsyncClient,
    tmp_path: Path,
    stage: str,
) -> None:
    from local_lm import comfy_registry_activation_batches as batches

    context, preparations, execution_plans = await _prepared(tmp_path)
    environment = registry_wheel_environment_root(context.state_root)
    batch, before = batches._begin(
        SessionLocal,
        "neutral-source-installation",
        preparations,
        execution_plans,
        context.custom_node_root,
        environment,
    )
    if stage in {"runtime-data", "verified-runtime-data"}:
        (batch.bindings[0].installed_path / "cache.json").write_text(
            '{"neutral":true}', encoding="utf-8"
        )
    if stage in {"verified", "verified-runtime-data"}:
        batch = batches._verified(SessionLocal, batch, before, NODES)
    assert all(active for active, _, _ in _rows().values())

    async def start(_bindings: tuple[WorkflowRegistryLaunchBinding, ...]) -> object:
        return object()

    async def stop() -> bool:
        return True

    async def inventory() -> frozenset[str]:
        return NODES

    with pytest.raises(RuntimeError, match="Neutral consumer failure"):
        async with batches.activate_registry_package_batch(
            SessionLocal,
            **_arguments(context, preparations, execution_plans),
            start_media=start,
            stop_media=stop,
            read_node_inventory=inventory,
        ) as resumed:
            assert resumed.id == batch.id and not any(resumed.previous_active.values())
            raise RuntimeError("Neutral consumer failure")
    assert all(not active for active, _, _ in _rows().values())


@pytest.mark.parametrize("change", ["purpose", "missing-record", "review", "code", "complete"])
async def test_pending_or_completed_batch_cannot_be_adopted_with_changed_evidence(
    client: AsyncClient,
    tmp_path: Path,
    change: str,
) -> None:
    from local_lm import comfy_registry_activation_batches as batches

    context, preparations, execution_plans = await _prepared(tmp_path)
    arguments = _arguments(context, preparations, execution_plans)
    batch, before = batches._begin(
        SessionLocal,
        arguments["purpose"],
        preparations,
        execution_plans,
        context.custom_node_root,
        arguments["environment_root"],
    )
    if change == "complete":
        batch = batches._verified(SessionLocal, batch, before, NODES)
        with SessionLocal() as session:
            batch.complete(session)
            session.commit()
    elif change == "purpose":
        arguments["purpose"] = "different-neutral-installation"
    elif change == "code":
        (batch.bindings[0].installed_path / "__init__.py").write_text(
            "CHANGED = True\n", encoding="utf-8"
        )
    else:
        with SessionLocal() as session:
            row = session.get(ComfyRegistryInstall, preparations[0].install_id)
            assert row is not None
            if change == "missing-record":
                row.review_json = {
                    key: value
                    for key, value in row.review_json.items()
                    if key != "activation_batch_v1"
                }
            else:
                row.review_json = {**row.review_json, "neutral_change": True}
            session.commit()
    prior = _rows()

    async def forbidden(*_args: object) -> Any:
        raise AssertionError("Changed evidence reached a worker action")

    with pytest.raises(ValueError):
        async with batches.activate_registry_package_batch(
            SessionLocal,
            **arguments,
            start_media=forbidden,
            stop_media=forbidden,
            read_node_inventory=forbidden,
        ):
            raise AssertionError("Changed evidence reached workflow completion")
    if change == "complete":
        assert _rows() == prior
    else:
        assert all(not active for active, _, _ in _rows().values())


async def test_failed_batch_restores_an_existing_active_package_without_enabling_the_other(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    from local_lm.comfy_registry_activation_batches import activate_registry_package_batch

    context, preparations, execution_plans = await _prepared(tmp_path)

    async def earlier_start() -> object:
        return object()

    with SessionLocal() as session:
        await activate_comfy_registry_install(
            session,
            install_id=preparations[0].install_id,
            custom_node_root=context.custom_node_root,
            environment_root=registry_wheel_environment_root(context.state_root),
            media_worker_stopped=True,
            start_media=earlier_start,
        )
    original = {key: active for key, (active, _, _) in _rows().items()}
    assert sum(original.values()) == 1

    async def start(_bindings: tuple[WorkflowRegistryLaunchBinding, ...]) -> object:
        return object()

    async def stop() -> bool:
        assert all(active for active, _, _ in _rows().values())
        return True

    async def inventory() -> frozenset[str]:
        return NODES

    with pytest.raises(RuntimeError, match="Neutral consumer failure"):
        async with activate_registry_package_batch(
            SessionLocal,
            **_arguments(context, preparations, execution_plans),
            start_media=start,
            stop_media=stop,
            read_node_inventory=inventory,
        ):
            raise RuntimeError("Neutral consumer failure")
    assert {key: active for key, (active, _, _) in _rows().items()} == original


@pytest.mark.parametrize("stop_refused", [False, True])
async def test_cancellation_retains_cleanup_until_the_worker_is_confirmed_stopped(
    client: AsyncClient,
    tmp_path: Path,
    stop_refused: bool,
) -> None:
    from local_lm.comfy_registry_activation_batches import activate_registry_package_batch

    context, preparations, execution_plans = await _prepared(tmp_path)
    entered = asyncio.Event()
    stopping = asyncio.Event()
    release = asyncio.Event()

    async def start(_bindings: tuple[WorkflowRegistryLaunchBinding, ...]) -> object:
        entered.set()
        await asyncio.Event().wait()
        return object()

    async def stop() -> bool:
        stopping.set()
        await release.wait()
        return not stop_refused

    async def inventory() -> frozenset[str]:
        return NODES

    async def run() -> None:
        async with activate_registry_package_batch(
            SessionLocal,
            **_arguments(context, preparations, execution_plans),
            start_media=start,
            stop_media=stop,
            read_node_inventory=inventory,
        ):
            raise AssertionError("Cancelled startup reached its consumer")

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    await asyncio.wait_for(stopping.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and all(active for active, _, _ in _rows().values())
    release.set()
    with pytest.raises(ValueError if stop_refused else asyncio.CancelledError):
        await task
    assert all(active == stop_refused for active, _, _ in _rows().values())
