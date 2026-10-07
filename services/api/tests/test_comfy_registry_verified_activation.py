from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from run_waits import wait_until
from sqlalchemy import delete, text
from sqlalchemy.orm import Session
from test_comfy_registry_launch_verification import _add_active
from test_comfy_registry_reviewed_wheel_staging import (
    source_review_context as source_review_context,
)
from test_workflow_reviewed_package_plan import _close, _inputs, _prepare

from local_lm import comfy_registry_installs as installs
from local_lm import comfy_registry_interpreter as interpreter
from local_lm import comfy_registry_verified_activation as activation
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_activation import (
    ComfyRegistryActivationError,
    activate_comfy_registry_install,
    review_comfy_registry_install,
)
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from local_lm.comfy_registry_target_verification import ComfyRegistryVerificationTarget
from local_lm.models import ComfyRegistryInstall, ComfyRegistrySourceArtifactReview
from local_lm.source_omission_proof import record_pending_omission
from local_lm.workflow_package_execution_plan import plan_workflow_package_execution


@pytest.fixture
async def prepared(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> AsyncIterator[dict[str, Any]]:
    mode = getattr(request, "param", "active")
    inputs = _inputs(source_review_context, tmp_path, commit=False, inactive=mode != "active")
    if mode == "empty":
        inputs["runtime"].clear()
    try:
        plan = await plan_workflow_package_execution(**inputs["plan_arguments"])
        result = await _prepare(inputs, plan)
        target = ComfyRegistryVerificationTarget(
            inputs["factory"],
            tmp_path / "managed-python.exe",
            inputs["context"].custom_node_root,
            registry_wheel_environment_root(inputs["context"].state_root),
            source_review_context[1],
        )
        probe = inputs["plan_arguments"]["interpreter_probe"]
        probes: list[Path] = []

        async def actual_probe(executable: Path) -> Any:
            assert executable == target.python_executable
            probes.append(executable)
            return await probe(executable)

        monkeypatch.setattr(interpreter, "probe_comfy_registry_runtime_target", actual_probe)
        proof = await target.verify((result.install_id,))
        arguments = dict(
            install_id=result.install_id,
            custom_node_root=target.custom_node_root,
            environment_root=target.environment_root,
            media_worker_stopped=True,
        )
        with inputs["factory"]() as session:
            review_comfy_registry_install(session, **arguments, trusted=True, verified_launch=proof)
        yield dict(inputs=inputs, target=target, arguments=arguments, probes=probes)
    finally:
        await _close(inputs)


def row(prepared: dict[str, Any]) -> tuple[bool, bool, dict[str, Any]]:
    with prepared["inputs"]["factory"]() as session:
        install = session.get(ComfyRegistryInstall, prepared["arguments"]["install_id"])
        assert install is not None
        return install.trusted, install.active, install.review_json


async def run(prepared: dict[str, Any], start: Any, reader: Any = None) -> Any:
    with prepared["inputs"]["factory"]() as session:
        return await activate_comfy_registry_install(
            session,
            **prepared["arguments"],
            verification_target=prepared["target"],
            start_media=start,
            read_node_inventory=reader,
        )


def watch_files(prepared: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> list[str]:
    main_thread = threading.get_ident()
    seen: list[str] = []

    def wrap(function: Callable[..., Any], name: str) -> Callable[..., Any]:
        def check(*args: Any, **kwargs: Any) -> Any:
            assert threading.get_ident() != main_thread
            with prepared["inputs"]["factory"]() as writer:
                writer.connection().exec_driver_sql("PRAGMA busy_timeout=100")
                writer.execute(text("UPDATE comfy_registry_installs SET active = active WHERE 0"))
                writer.commit()
            seen.append(name)
            return function(*args, **kwargs)

        return check

    for module, name in (
        (installs, "verify_staged_comfy_registry_archive"),
        (installs, "verify_comfy_registry_wheel_environment"),
        (activation, "snapshot_staged_comfy_registry_files"),
        (activation, "capture_staged_comfy_registry_runtime_files"),
    ):
        monkeypatch.setattr(module, name, wrap(getattr(module, name), name))
    return seen


@pytest.mark.parametrize("prepared", ["active", "inactive", "empty"], indirect=True)
async def test_direct_activation_checks_actual_target_and_files_without_holding_the_writer(
    prepared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = watch_files(prepared, monkeypatch)
    starts: list[bool] = []

    async def start() -> None:
        starts.append(row(prepared)[1])

    result = await run(prepared, start)
    assert result.trusted and result.active and result.activated_at
    assert starts == [True] and len(prepared["probes"]) == 3
    assert len(set(seen)) == 4
    assert row(prepared)[2]["trust_authority"] == "local_user"


@pytest.mark.parametrize("change", ["review", "trust", "runtime", "target", "active_set", "files"])
async def test_direct_activation_rolls_back_when_startup_changes_verified_inputs(
    prepared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    starts: list[bool] = []

    async def start() -> None:
        starts.append(row(prepared)[1])
        if len(starts) != 1:
            return
        inputs = prepared["inputs"]
        if change == "runtime":
            inputs["runtime"][:] = [ComfyRegistryRuntimeDistribution("helper", "3.0")]
        elif change == "target":
            inputs["environment"]["python_version"] = "0.0"
        elif change == "files":
            folder = prepared["target"].custom_node_root
            files = list(folder.glob("*/__init__.py"))
            assert len(files) == 1
            files[0].write_text("raise RuntimeError('Changed neutral package')", encoding="utf-8")
        else:
            with inputs["factory"]() as session:
                install = session.get(ComfyRegistryInstall, prepared["arguments"]["install_id"])
                assert install is not None
                if change == "review":
                    session.execute(delete(ComfyRegistrySourceArtifactReview))
                elif change == "trust":
                    install.trusted = False
                else:
                    _add_active(session, install)
                session.commit()

    with pytest.raises(ComfyRegistryActivationError):
        await run(prepared, start)
    assert starts == [True, False]
    trusted, active, review = row(prepared)
    assert not active and "activated_at" not in review
    assert trusted == (change != "trust")


def pending(prepared: dict[str, Any]) -> None:
    with prepared["inputs"]["factory"]() as session:
        install = session.get(ComfyRegistryInstall, prepared["arguments"]["install_id"])
        assert install is not None
        install.review_json = record_pending_omission(
            install.review_json,
            manifest_sha256=install.manifest_sha256,
            omitted_declarations=("neutral @ git+https://example.com/neutral",),
            workflow_revision_id="neutral-revision",
            required_node_types=("NeutralNode",),
        )
        session.commit()


@pytest.mark.parametrize("present", [True, False])
async def test_direct_activation_records_only_a_current_omission_proof(
    prepared: dict[str, Any],
    present: bool,
) -> None:
    pending(prepared)
    starts: list[bool] = []

    async def start() -> None:
        starts.append(row(prepared)[1])

    async def inventory() -> frozenset[str]:
        return frozenset({"NeutralNode"} if present else {"OtherNode"})

    if present:
        await run(prepared, start, inventory)
        assert row(prepared)[2]["source_omission_digest"]
        assert starts == [True]
    else:
        with pytest.raises(ComfyRegistryActivationError, match="prior media runtime was restored"):
            await run(prepared, start, inventory)
        assert starts == [True, False] and not row(prepared)[1]


@pytest.mark.parametrize("boundary", ["snapshot", "capture", "inventory"])
async def test_direct_activation_drains_cancelled_file_work_and_restores_the_trial(
    prepared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    entered = threading.Event()
    released = threading.Event()
    finished = threading.Event()
    starts: list[bool] = []
    task = None
    if boundary == "inventory":
        pending(prepared)
    else:
        name = (
            "snapshot_staged_comfy_registry_files"
            if boundary == "snapshot"
            else "capture_staged_comfy_registry_runtime_files"
        )
        original = getattr(activation, name)

        def paused(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            try:
                assert released.wait(30)
                return original(*args, **kwargs)
            finally:
                finished.set()

        monkeypatch.setattr(activation, name, paused)

    async def start() -> None:
        if starts and boundary != "inventory":
            assert finished.is_set()
        starts.append(row(prepared)[1])

    async def inventory() -> frozenset[str]:
        entered.set()
        await asyncio.Future()
        return frozenset()

    async def reached() -> bool:
        return entered.is_set()

    try:
        task = asyncio.create_task(run(prepared, start, inventory))
        await wait_until(reached, bool, what="direct activation verification boundary")
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        if boundary != "inventory":
            assert not task.done()
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not row(prepared)[1]
        assert starts == ([] if boundary == "snapshot" else [True, False])
    finally:
        released.set()
        if boundary != "inventory" and entered.is_set():
            await asyncio.to_thread(finished.wait, 30)
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.parametrize("prepared", ["empty"], indirect=True)
async def test_direct_activation_compares_an_empty_runtime_baseline(
    prepared: dict[str, Any],
) -> None:
    starts: list[bool] = []

    async def start() -> None:
        starts.append(row(prepared)[1])
        if len(starts) == 1:
            prepared["inputs"]["runtime"].append(ComfyRegistryRuntimeDistribution("helper", "3.0"))

    with pytest.raises(ComfyRegistryActivationError):
        await run(prepared, start)
    assert starts == [True, False] and not row(prepared)[1]


async def test_direct_activation_rechecks_authority_after_file_verification(
    prepared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ComfyRegistryVerificationTarget.verify
    calls = 0
    starts: list[bool] = []

    async def verify(self: ComfyRegistryVerificationTarget, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        proof = await original(self, *args, **kwargs)
        calls += 1
        if calls == 2:
            with prepared["inputs"]["factory"]() as writer:
                writer.execute(delete(ComfyRegistrySourceArtifactReview))
                writer.commit()
        return proof

    async def start() -> None:
        starts.append(row(prepared)[1])

    monkeypatch.setattr(ComfyRegistryVerificationTarget, "verify", verify)
    with pytest.raises(ComfyRegistryActivationError):
        await run(prepared, start)
    assert calls == 2 and starts == [True, False] and not row(prepared)[1]
