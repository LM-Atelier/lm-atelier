from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import delete
from test_comfy_registry_reviewed_batches import _watch_files
from test_workflow_reviewed_extension_trust import _accepted_set
from test_workflow_reviewed_package_plan import _close, _inputs

from local_lm import comfy_registry_interpreter
from local_lm import processes as process_module
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_editor_bridge import ComfyEditorBridgeSupport
from local_lm.comfy_registry_activation_batches import _begin
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ComfyRegistryInstall,
    ComfyRegistrySourceArtifactReview,
)
from local_lm.processes import ProcessSupervisor
from local_lm.workflow_activations import (
    WorkflowActivationLaunchScope,
    materialize_comfy_runtime_dependency,
)
from local_lm.workflow_completion_jobs import (
    begin_workflow_completion,
    stage_workflow_completion_job,
)
from local_lm.workflow_source_extension_trust import trust_workflow_source_extensions
from local_lm.workflow_source_extensions import _accepted
from local_lm.workflow_source_launch import prepare_workflow_source_launch_scope


@pytest.fixture
async def launched_packages(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> AsyncIterator[dict[str, Any]]:
    import test_workflow_reviewed_extension_trust as fixture_module

    runtime = tmp_path / "managed-runtime"
    runtime.mkdir()
    (runtime / "main.py").write_text(
        "raise RuntimeError('Keep this fixture inert')\n", encoding="utf-8"
    )
    settings.comfy_directory = runtime
    settings.comfy_executable = runtime / "python.exe"
    settings.comfy_executable.write_bytes(b"Neutral managed interpreter")

    def inputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
        base = _inputs(*args, **kwargs)
        base["context"] = replace(
            base["context"],
            custom_node_root=settings.custom_node_dir,
            state_root=settings.registry_dir,
        )
        settings.custom_node_dir.mkdir(parents=True, exist_ok=True)
        settings.registry_dir.mkdir(parents=True, exist_ok=True)
        return base

    monkeypatch.setattr(fixture_module, "_inputs", inputs)
    monkeypatch.setattr(
        fixture_module,
        "_payload",
        lambda: {
            "name": "Neutral managed extension launch",
            "operation": "text_to_image",
            "ui_graph": {"version": 0.4, "nodes": [], "links": []},
            "dependencies": {"version": 1, "slots": []},
            "selections": [],
        },
    )
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        accepted = await _accepted_set(
            (session, store),
            tmp_path,
            inactive=mode in {"inactive", "empty", "empty-drift"},
            empty_runtime=mode in {"empty", "empty-drift"},
        )
    base = accepted["base"]
    context = base["context"]
    reviewed = ComfyRegistryReviewedInputContext(
        SessionLocal, store, base["environment"], ("py3-none-any",)
    )
    provisioner: Any = SimpleNamespace(
        status=lambda _engine: SimpleNamespace(state="ready", release="neutral-runtime")
    )
    probes: list[Path] = []
    original_probe = base["plan_arguments"]["interpreter_probe"]

    async def probe(executable: Path) -> Any:
        assert executable == settings.comfy_executable and executable != context.python_executable
        probes.append(executable)
        environment, tags, distributions = await original_probe(executable)
        if mode == "target":
            environment["python_version"] = "0.0"
        if mode in {"runtime", "empty-drift"}:
            distributions = (ComfyRegistryRuntimeDistribution("helper", "3.0"),)
        return environment, tags, distributions

    def prepare() -> Any:
        assert (
            trust_workflow_source_extensions(
                SessionLocal,
                accepted["offer_id"],
                context=context,
                media_worker_stopped=True,
                reviewed_inputs=reviewed,
            ).state
            == "ready"
        )
        with SessionLocal() as session:
            offer, _, packages = _accepted(session, accepted["offer_id"])
            plans = {
                p.preparation.install_id: p.plan for p in packages if p.preparation is not None
            }
            stage_workflow_completion_job(session, offer)
            assert begin_workflow_completion(session, offer) is not None
            session.commit()
        _begin(
            SessionLocal,
            accepted["offer_id"],
            accepted["prepared"],
            plans,
            context.custom_node_root,
            registry_wheel_environment_root(context.state_root),
            reviewed,
        )
        return prepare_workflow_source_launch_scope(
            SessionLocal,
            accepted["offer_id"],
            context=context,
            reviewed_inputs=reviewed,
            runtime_materializer=lambda requirement, selection: (
                materialize_comfy_runtime_dependency(provisioner, requirement, selection)
            ),
        )

    try:
        source = await asyncio.to_thread(prepare)
        supervisor = ProcessSupervisor(settings, runtimes=provisioner)
        monkeypatch.setattr(
            comfy_registry_interpreter, "probe_comfy_registry_runtime_target", probe
        )
        yield dict(
            base=base, source=source, supervisor=supervisor, probes=probes, accepted=accepted
        )
    finally:
        await _close(base)


def _revoke() -> None:
    with SessionLocal() as session:
        session.execute(delete(ComfyRegistrySourceArtifactReview))
        session.commit()


@pytest.mark.parametrize(
    "mode",
    [
        "source",
        "scoped",
        "unscoped",
        "inactive",
        "empty",
        "target",
        "runtime",
        "empty-drift",
        "revoked",
        "revoked-unscoped",
        "revoked-scoped",
        "runtime-prestart",
        "target-prestart",
        "selection-prestart",
        "executable-prestart",
        "executable-during-check",
        "unrelated",
    ],
)
async def test_managed_launch_rechecks_exact_reviewed_target_before_spawning(
    launched_packages: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    inputs = launched_packages
    supervisor: ProcessSupervisor = inputs["supervisor"]
    source = inputs["source"]
    scope = source
    if mode in {"unscoped", "revoked-unscoped", "selection-prestart"}:
        scope = None
    elif mode in {"scoped", "revoked-scoped", "unrelated"}:
        scope = WorkflowActivationLaunchScope(
            "neutral-activation",
            "neutral-revision",
            source.binding_sha256,
            source.launch_sha256,
            source.model_install_ids,
            source.model_asset_install_ids,
            source.custom_node_install_ids,
            source.registry_install_ids,
            source.runtime_keys,
            source.models,
            source.assets,
            source.custom_nodes,
            source.registry_packages,
            source.runtimes,
        )
    commands: list[tuple[str, ...]] = []
    phases: list[str] = []

    async def spawn(*command: str, **_kwargs: Any) -> Any:
        commands.append(command)
        raise RuntimeError("Neutral managed launch boundary")

    async def no_port_work(*_args: Any) -> None:
        return None

    async def phase(value: str) -> None:
        phases.append(value)
        if value != "Starting media runtime":
            return
        if mode.startswith("revoked"):
            _revoke()
        elif mode == "runtime-prestart":
            inputs["base"]["runtime"].clear()
        elif mode == "target-prestart":
            inputs["base"]["environment"]["python_version"] = "0.0"
        elif mode in {"executable-prestart", "executable-during-check"}:
            executable = supervisor.settings.comfy_executable
            assert executable is not None
            alternate = executable.with_name("other-python.exe")
            alternate.write_bytes(b"Another neutral managed interpreter")
            if mode == "executable-prestart":
                supervisor.settings.comfy_executable = alternate
            else:
                original_probe = comfy_registry_interpreter.probe_comfy_registry_runtime_target

                async def changed_probe(path: Path) -> Any:
                    result = await original_probe(path)
                    supervisor.settings.comfy_executable = alternate
                    return result

                monkeypatch.setattr(
                    comfy_registry_interpreter, "probe_comfy_registry_runtime_target", changed_probe
                )
        elif mode == "selection-prestart":
            with SessionLocal() as session:
                row = session.get(ComfyRegistryInstall, source.registry_install_ids[0])
                assert row is not None
                row.active = False
                session.commit()

    if mode == "unrelated":
        with SessionLocal() as session:
            row = session.get(ComfyRegistryInstall, source.registry_install_ids[0])
            assert row is not None
            values = {column.name: getattr(row, column.name) for column in row.__table__.columns}
            values.update(
                id="neutral-unrelated",
                package_id="neutral-unrelated",
                installed_path="absent-unrelated",
                registry_record_id="neutral-unrelated-record",
            )
            session.add(ComfyRegistryInstall(**values))
            session.commit()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(supervisor, "_reclaim_port_from_our_own_children", no_port_work)
    monkeypatch.setattr(supervisor, "_ensure_port_available", no_port_work)
    monkeypatch.setattr(
        process_module,
        "prepare_comfy_editor_bridge",
        lambda **_kwargs: SimpleNamespace(
            folder=None,
            support=ComfyEditorBridgeSupport(
                False,
                "workflow-editor-runtime-unavailable",
                "The neutral fixture has no editor bridge.",
            ),
        ),
    )
    files = _watch_files(monkeypatch)
    success = mode in {"source", "scoped", "unscoped", "inactive", "empty", "unrelated"}
    if success:
        with pytest.raises(RuntimeError, match="Neutral managed launch boundary"):
            await supervisor.start_media(activation_scope=scope, phase_callback=phase)
    else:
        with pytest.raises((ValueError, RuntimeError)):
            await supervisor.start_media(activation_scope=scope, phase_callback=phase)
    assert len(commands) == int(success)
    assert inputs["probes"] and files
    if success or "prestart" in mode or mode.startswith("revoked"):
        assert phases[-1] == "Starting media runtime"
        assert len(inputs["probes"]) >= 2
    if success:
        command = commands[0]
        assert supervisor.settings.comfy_executable is not None
        assert command[0] == str(supervisor.settings.comfy_executable.resolve())
        assert set(command[command.index("--whitelist-custom-nodes") + 1 :]) == {
            preparation.installed_path for preparation in inputs["accepted"]["prepared"]
        }


@pytest.mark.parametrize(
    "mode", ["source", "inactive", "empty", "target", "revoked", "cancel", "unselected"]
)
async def test_node_ownership_requires_current_reviewed_runtime_and_drains_inspection(
    launched_packages: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    supervisor: ProcessSupervisor = launched_packages["supervisor"]
    if mode == "unselected":
        with SessionLocal() as session:
            for identifier in launched_packages["source"].registry_install_ids:
                row = session.get(ComfyRegistryInstall, identifier)
                assert row is not None
                row.active = False
            session.commit()
        assert await supervisor.trusted_comfy_registry_package_node_types() == {}
        assert launched_packages["probes"] == []
        return
    original = supervisor._trusted_comfy_registry_package_node_types
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    cleaned = False
    task: asyncio.Task[dict[tuple[str, str], frozenset[str]]] | None = None

    def inspect(**kwargs: Any) -> dict[tuple[str, str], frozenset[str]]:
        try:
            if mode == "revoked":
                _revoke()
            if mode == "cancel":
                entered.set()
                if not release.wait(30):
                    raise AssertionError("Node ownership inspection was not released")
                assert not cleaned
            return original(**kwargs)
        finally:
            finished.set()

    async def run() -> dict[tuple[str, str], frozenset[str]]:
        nonlocal cleaned
        try:
            return await supervisor.trusted_comfy_registry_package_node_types()
        finally:
            cleaned = True

    async def waiting() -> bool:
        return entered.is_set()

    async def settled() -> bool:
        return finished.is_set()

    monkeypatch.setattr(supervisor, "_trusted_comfy_registry_package_node_types", inspect)
    _watch_files(monkeypatch)
    try:
        task = asyncio.create_task(run())
        if mode == "cancel":
            await wait_until(waiting, bool, what="node ownership inspection entered")
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done() and not cleaned
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
        elif mode in {"target", "revoked"}:
            with pytest.raises(ValueError):
                await task
            assert launched_packages["probes"]
            assert finished.is_set() is (mode == "revoked")
        else:
            assert await task == {
                (f"neutral-pack-{index}", "1.2.3"): frozenset({f"NeutralNode{index}"})
                for index in range(2)
            }
    finally:
        release.set()
        if entered.is_set():
            await wait_until(settled, bool, what="node ownership inspection finished")
        if task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task


@pytest.mark.parametrize("mode", ["cancel-contract", "cancel-scope"])
async def test_source_launch_waits_for_verification_workers_after_cancellation(
    launched_packages: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from local_lm import workflow_source_launch

    supervisor: ProcessSupervisor = launched_packages["supervisor"]
    target: Any = supervisor if mode == "cancel-contract" else workflow_source_launch
    name = (
        "_scoped_comfy_registry_contract"
        if mode == "cancel-contract"
        else "revalidate_workflow_source_launch_scope"
    )
    original = getattr(target, name)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    cleaned = False

    def verify(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        try:
            entered.set()
            if not release.wait(30):
                raise AssertionError("Source launch verification was not released")
            assert not cleaned
            return result
        finally:
            finished.set()

    async def run() -> None:
        nonlocal cleaned
        try:
            await supervisor._revalidate_source_media_scope(launched_packages["source"])
        finally:
            cleaned = True

    async def waiting() -> bool:
        return entered.is_set()

    async def settled() -> bool:
        return finished.is_set()

    monkeypatch.setattr(target, name, verify)
    _watch_files(monkeypatch)
    task = asyncio.create_task(run())
    try:
        await wait_until(waiting, bool, what="source launch verification entered")
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done() and not cleaned
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        release.set()
        if entered.is_set():
            await wait_until(settled, bool, what="source launch verification finished")
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task
