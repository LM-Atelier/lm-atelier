"""The preparation composition: ordering, code pass-through, honest refusals."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx2 import AsyncClient

from local_lm import workflow_package_preparation as composition
from local_lm.comfy_registry import ComfyNodeResolution, ComfyRegistryResolution
from local_lm.comfy_registry_archives import ComfyRegistryArchiveReport
from local_lm.comfy_registry_closure_driver import ComfyRegistryWheelClosureDriverError
from local_lm.comfy_registry_lifecycle import (
    ComfyRegistryLifecycleError,
    ComfyRegistryStagedArchive,
)
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


class _NullSessionFactory:
    def __call__(self) -> _NullSessionFactory:
        return self

    def __enter__(self) -> object:
        return object()

    def __exit__(self, *args: object) -> None:
        return None


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
        await kwargs["progress"]("fetching_projects", 1, ("a", "b"))
        return closure

    async def fake_prepare(session: Any, **kwargs: Any) -> Any:
        order.append("prepare")
        assert kwargs["closure"] == "closure-object"
        assert kwargs["media_worker_stopped"] is True
        assert kwargs["custom_node_root"] == _CONTEXT.custom_node_root
        await kwargs["archive_progress"](10, 100)
        await kwargs["wheel_progress"]("a.whl", 5, 50)
        return prepared

    monkeypatch.setattr(composition, "drive_comfy_registry_wheel_closure", fake_drive)
    monkeypatch.setattr(composition, "prepare_comfy_registry_install", fake_prepare)

    registry = _Registry(_resolution())

    result = await prepare_workflow_package(
        _NullSessionFactory(),  # opened only around the prepare step
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


async def test_commit_pin_stages_reads_closes_and_prepares_the_same_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    order: list[str] = []
    phases: list[str] = []
    destination = Path("C:/synthetic/custom_nodes/lm-atelier-registry_commit")
    report = ComfyRegistryArchiveReport(
        archive_sha256="a" * 64,
        manifest_sha256="b" * 64,
        entry_count=2,
        file_count=2,
        expanded_bytes=40,
        python_file_count=1,
        dependency_manifests=("requirements.txt",),
        install_scripts=(),
        startup_hooks=(),
        native_files=(),
        top_level_entries=("__init__.py", "requirements.txt"),
    )
    staged = ComfyRegistryStagedArchive(
        "lm-atelier-registry_commit",
        destination,
        report,
    )
    closure = SimpleNamespace(closure="closure-object")
    prepared = SimpleNamespace(install_id="install_commit")

    async def fake_stage(**kwargs: Any) -> ComfyRegistryStagedArchive:
        order.append("stage")
        await kwargs["archive_progress"](10, 10)
        return staged

    def fake_read(root: Path, manifest: str) -> tuple[str, ...]:
        order.append("read")
        assert root == destination
        assert manifest == "requirements.txt"
        return ("pillow>=12",)

    async def fake_drive(resolution: ComfyNodeResolution, **kwargs: Any) -> Any:
        order.append("drive")
        assert resolution.pip_dependencies == ("pillow>=12",)
        return closure

    async def fake_prepare(session: Any, **kwargs: Any) -> Any:
        order.append("prepare")
        assert kwargs["resolution"].pip_dependencies == ("pillow>=12",)
        assert kwargs["staged_archive"] is staged
        return prepared

    async def unexpected_discard(**kwargs: Any) -> None:
        raise AssertionError("a successfully consumed staged tree must not be discarded")

    monkeypatch.setattr(composition, "stage_comfy_registry_install_archive", fake_stage)
    monkeypatch.setattr(composition, "read_staged_requirements", fake_read)
    monkeypatch.setattr(composition, "drive_comfy_registry_wheel_closure", fake_drive)
    monkeypatch.setattr(composition, "prepare_comfy_registry_install", fake_prepare)
    monkeypatch.setattr(
        composition,
        "discard_comfy_registry_staged_archive",
        unexpected_discard,
    )

    result = await prepare_workflow_package(
        _NullSessionFactory(),
        package_id="example-pack",
        version=revision,
        context=_CONTEXT,
        media_worker_stopped=True,
        interpreter_probe=_probe,
        registry_client=_Registry(
            _resolution(
                declared_version=revision,
                install_kind="git_commit",
                repository_url="https://github.com/example/example-pack.git",
            )
        ),  # type: ignore[arg-type]
        phase=lambda name, done, total: phases.append(name),
        **_clients(),
    )

    assert result is prepared
    assert order == ["stage", "read", "drive", "prepare"]
    assert "Reading package dependencies" in phases


async def test_commit_pin_cancellation_discards_the_staged_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    order: list[str] = []
    staged = ComfyRegistryStagedArchive(
        "lm-atelier-registry_commit",
        Path("C:/synthetic/custom_nodes/lm-atelier-registry_commit"),
        ComfyRegistryArchiveReport(
            archive_sha256="a" * 64,
            manifest_sha256="b" * 64,
            entry_count=1,
            file_count=1,
            expanded_bytes=20,
            python_file_count=1,
            dependency_manifests=(),
            install_scripts=(),
            startup_hooks=(),
            native_files=(),
            top_level_entries=("__init__.py",),
        ),
    )

    async def fake_stage(**kwargs: Any) -> ComfyRegistryStagedArchive:
        order.append("stage")
        return staged

    async def cancel_drive(*args: Any, **kwargs: Any) -> Any:
        order.append("drive")
        raise asyncio.CancelledError

    async def fake_discard(**kwargs: Any) -> None:
        order.append("discard")
        assert kwargs["staged_archive"] is staged

    monkeypatch.setattr(composition, "stage_comfy_registry_install_archive", fake_stage)
    monkeypatch.setattr(composition, "drive_comfy_registry_wheel_closure", cancel_drive)
    monkeypatch.setattr(
        composition,
        "discard_comfy_registry_staged_archive",
        fake_discard,
    )

    with pytest.raises(asyncio.CancelledError):
        await prepare_workflow_package(
            _NullSessionFactory(),
            package_id="example-pack",
            version=revision,
            context=_CONTEXT,
            media_worker_stopped=True,
            interpreter_probe=_probe,
            registry_client=_Registry(
                _resolution(
                    declared_version=revision,
                    install_kind="git_commit",
                    repository_url="https://github.com/example/example-pack.git",
                )
            ),  # type: ignore[arg-type]
            **_clients(),
        )

    assert order == ["stage", "drive", "discard"]


async def test_each_stage_refuses_with_its_own_code(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resolution error codes surface unchanged.
    registry = _Registry(_resolution(error_code="registry_record_missing"))
    with pytest.raises(WorkflowPackagePreparationError) as resolved:
        await prepare_workflow_package(
            _NullSessionFactory(),
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
            _NullSessionFactory(),
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
            _NullSessionFactory(),
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


async def test_a_fresh_profile_already_has_the_registry_state_directory(tmp_path: Path) -> None:
    """Preparation validates this root rather than creating it, so startup must.

    A brand-new profile that had never prepared a package refused every
    Registry package with `invalid_managed_root`, because this one directory
    was absent from the set the application creates for itself.
    """
    settings = Settings(
        data_dir=tmp_path,
        comfy_executable=tmp_path / "python.exe",
        comfy_directory=tmp_path / "ComfyUI",
    )
    assert not settings.registry_dir.exists()
    settings.prepare()
    assert settings.registry_dir.is_dir()
    assert PreparationContext.from_settings(settings).state_root == settings.registry_dir


async def test_another_writer_progresses_while_preparation_awaits(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audited await holds no lock: a concurrent writer commits mid-prepare.

    This is the regression the async-boundary registry requires. The fake
    lifecycle blocks inside the awaited call while a second session inserts
    and commits; if the preparation session entered the await holding a
    SQLite write lock, that insert would deadlock or fail.
    """

    from local_lm.db import SessionLocal
    from local_lm.models import EditTemplate

    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_prepare(session: Any, **kwargs: Any) -> Any:
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=5)
        return SimpleNamespace(install_id="install_concurrent")

    async def ok_drive(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(closure="closure-object")

    monkeypatch.setattr(composition, "drive_comfy_registry_wheel_closure", ok_drive)
    monkeypatch.setattr(composition, "prepare_comfy_registry_install", blocking_prepare)

    preparation = asyncio.create_task(
        prepare_workflow_package(
            SessionLocal,
            package_id="example-pack",
            version="1.2.3",
            context=_CONTEXT,
            media_worker_stopped=True,
            interpreter_probe=_probe,
            registry_client=_Registry(_resolution()),  # type: ignore[arg-type]
            **_clients(),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=5)

    def concurrent_write() -> None:
        with SessionLocal() as writer:
            writer.add(
                EditTemplate(
                    name="Written mid-preparation",
                    instruction="Prove the lock is free.",
                    operation="image_to_image",
                )
            )
            writer.commit()

    await asyncio.wait_for(asyncio.to_thread(concurrent_write), timeout=5)

    release.set()
    result = await asyncio.wait_for(preparation, timeout=5)
    assert result.install_id == "install_concurrent"
    with SessionLocal() as reader:
        assert reader.query(EditTemplate).filter_by(name="Written mid-preparation").count() == 1


async def test_the_probes_typed_refusals_keep_their_codes() -> None:
    """A probe timeout must not be laundered into a generic failure."""

    from local_lm.comfy_registry_interpreter import ComfyRegistryInterpreterError

    async def timing_out_probe(_python: Path) -> tuple[dict[str, str], tuple[str, ...]]:
        raise ComfyRegistryInterpreterError("interpreter_timeout", "took too long")

    with pytest.raises(WorkflowPackagePreparationError) as refused:
        await prepare_workflow_package(
            _NullSessionFactory(),
            package_id="example-pack",
            version="1.2.3",
            context=_CONTEXT,
            media_worker_stopped=True,
            interpreter_probe=timing_out_probe,
            registry_client=_Registry(_resolution()),  # type: ignore[arg-type]
            **_clients(),
        )
    assert refused.value.code == "interpreter_timeout"
