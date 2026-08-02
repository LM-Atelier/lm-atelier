"""The preparation composition: ordering, code pass-through, honest refusals."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_lm import workflow_package_preparation as composition
from local_lm.comfy_registry import ComfyNodeResolution, ComfyRegistryResolution
from local_lm.comfy_registry_closure_driver import ComfyRegistryWheelClosureDriverError
from local_lm.comfy_registry_lifecycle import ComfyRegistryLifecycleError
from local_lm.config import Settings
from local_lm.workflow_package_preparation import (
    PreparationContext,
    WorkflowPackagePreparationError,
    prepare_workflow_package,
    refuse_interpreter_probe,
)

pytestmark = pytest.mark.asyncio

_CONTEXT = PreparationContext(
    python_executable=Path("C:/synthetic/python.exe"),
    custom_node_root=Path("C:/synthetic/custom_nodes"),
    state_root=Path("C:/synthetic/registry"),
)


def _resolution(**overrides: Any) -> ComfyNodeResolution:
    values: dict[str, Any] = {
        "package_id": "example-pack",
        "declared_version": "1.2.3",
        "node_types": ("ExampleNode",),
    }
    values.update(overrides)
    return ComfyNodeResolution(**values)


class _Registry:
    def __init__(self, resolution: ComfyNodeResolution) -> None:
        self._resolution = resolution
        self.requested: list[Any] = []

    async def resolve(self, requirements: Any) -> ComfyRegistryResolution:
        self.requested.extend(requirements)
        return ComfyRegistryResolution(packages=(self._resolution,))


async def _probe(_python: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    return {"sys_platform": "win32"}, ("py3-none-any",)


def _clients() -> dict[str, Any]:
    return {
        "project_client": SimpleNamespace(fetch=None),
        "metadata_client": SimpleNamespace(fetch=None),
        "archive_downloader": SimpleNamespace(),
        "wheel_downloader": SimpleNamespace(),
    }


async def test_composes_resolve_close_prepare_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    phases: list[str] = []
    closure = SimpleNamespace(closure="closure-object")
    prepared = SimpleNamespace(install_id="install_1")

    async def fake_drive(resolution: Any, **kwargs: Any) -> Any:
        order.append("drive")
        assert resolution.package_id == "example-pack"
        assert kwargs["marker_environment"] == {"sys_platform": "win32"}
        assert kwargs["supported_tags"] == ("py3-none-any",)
        kwargs["progress"]("fetching_projects", 1, 2)
        return closure

    async def fake_prepare(session: Any, **kwargs: Any) -> Any:
        order.append("prepare")
        assert kwargs["closure"] == "closure-object"
        assert kwargs["media_worker_stopped"] is True
        assert kwargs["custom_node_root"] == _CONTEXT.custom_node_root
        kwargs["archive_progress"](10, 100)
        kwargs["wheel_progress"]("a.whl", 5, 50)
        return prepared

    monkeypatch.setattr(composition, "drive_comfy_registry_wheel_closure", fake_drive)
    monkeypatch.setattr(composition, "prepare_comfy_registry_install", fake_prepare)

    registry = _Registry(_resolution())
    result = await prepare_workflow_package(
        object(),  # session is passed through untouched
        package_id="example-pack",
        version="1.2.3",
        context=_CONTEXT,
        media_worker_stopped=True,
        interpreter_probe=_probe,
        registry_client=registry,  # type: ignore[arg-type]
        phase=lambda name, done, total: phases.append(name),
        **_clients(),
    )

    assert result is prepared
    assert order == ["drive", "prepare"]
    assert registry.requested[0].package_id == "example-pack"
    assert registry.requested[0].versions == ("1.2.3",)
    # Every stage announced itself, including the driver's round phases and
    # both download streams.
    assert phases[0] == "Resolving the package"
    assert "Probing the target runtime" in phases
    assert any("fetching projects" in name for name in phases)
    assert any("node archive" in name for name in phases)
    assert any("a.whl" in name for name in phases)


async def test_each_stage_refuses_with_its_own_code(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resolution error codes surface unchanged.
    registry = _Registry(_resolution(error_code="registry_record_missing"))
    with pytest.raises(WorkflowPackagePreparationError) as resolved:
        await prepare_workflow_package(
            object(),
            package_id="example-pack",
            version=None,
            context=_CONTEXT,
            media_worker_stopped=True,
            interpreter_probe=_probe,
            registry_client=registry,  # type: ignore[arg-type]
            **_clients(),
        )
    assert resolved.value.code == "registry_record_missing"

    # Driver refusals keep their stable codes.
    async def failing_drive(*args: Any, **kwargs: Any) -> Any:
        raise ComfyRegistryWheelClosureDriverError("closure_round_limit", "too many rounds")

    monkeypatch.setattr(composition, "drive_comfy_registry_wheel_closure", failing_drive)
    with pytest.raises(WorkflowPackagePreparationError) as driven:
        await prepare_workflow_package(
            object(),
            package_id="example-pack",
            version=None,
            context=_CONTEXT,
            media_worker_stopped=True,
            interpreter_probe=_probe,
            registry_client=_Registry(_resolution()),  # type: ignore[arg-type]
            **_clients(),
        )
    assert driven.value.code == "closure_round_limit"

    # Lifecycle refusals keep theirs.
    async def ok_drive(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(closure="closure-object")

    async def failing_prepare(*args: Any, **kwargs: Any) -> Any:
        raise ComfyRegistryLifecycleError("media_worker_running", "stop it first")

    monkeypatch.setattr(composition, "drive_comfy_registry_wheel_closure", ok_drive)
    monkeypatch.setattr(composition, "prepare_comfy_registry_install", failing_prepare)
    with pytest.raises(WorkflowPackagePreparationError) as prepared:
        await prepare_workflow_package(
            object(),
            package_id="example-pack",
            version=None,
            context=_CONTEXT,
            media_worker_stopped=False,
            interpreter_probe=_probe,
            registry_client=_Registry(_resolution()),  # type: ignore[arg-type]
            **_clients(),
        )
    assert prepared.value.code == "media_worker_running"


async def test_the_probe_placeholder_refuses_rather_than_guessing() -> None:
    with pytest.raises(WorkflowPackagePreparationError) as refused:
        await refuse_interpreter_probe(Path("C:/synthetic/python.exe"))
    assert refused.value.code == "interpreter_probe_unavailable"


async def test_context_requires_a_configured_managed_runtime(tmp_path: Path) -> None:
    unconfigured = Settings(data_dir=tmp_path, comfy_executable=None, comfy_directory=None)
    with pytest.raises(WorkflowPackagePreparationError) as refused:
        PreparationContext.from_settings(unconfigured)
    assert refused.value.code == "managed_runtime_unavailable"

    configured = Settings(
        data_dir=tmp_path,
        comfy_executable=tmp_path / "python.exe",
        comfy_directory=tmp_path / "ComfyUI",
    )
    context = PreparationContext.from_settings(configured)
    assert context.custom_node_root == tmp_path / "ComfyUI" / "custom_nodes"
    assert context.state_root == tmp_path / "registry"
