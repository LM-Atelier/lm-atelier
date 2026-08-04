from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from packaging.markers import default_environment
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

import local_lm.comfy_registry_lifecycle as lifecycle_module
import local_lm.comfy_registry_wheel_environments as environment_module
from local_lm.comfy_registry import ComfyNodeResolution
from local_lm.comfy_registry_archives import ComfyRegistryArchiveReport
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_downloads import DownloadProgress
from local_lm.comfy_registry_lifecycle import (
    ComfyRegistryLifecycleError,
    ComfyRegistryPreparation,
    prepare_comfy_registry_install,
    stage_comfy_registry_install_archive,
)
from local_lm.comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactManifest,
    build_comfy_registry_wheel_artifact_manifest,
    comfy_registry_wheel_target_sha256,
    current_comfy_registry_wheel_target,
    resolve_comfy_registry_wheel_artifacts,
)
from local_lm.comfy_registry_wheel_closure import (
    ComfyRegistryWheelClosure,
    advance_comfy_registry_wheel_closure,
    plan_comfy_registry_wheel_closure,
)
from local_lm.comfy_registry_wheel_downloads import (
    ComfyRegistryStagedWheel,
    ComfyRegistryWheelStageReport,
    WheelDownloadProgress,
)
from local_lm.comfy_registry_wheel_selection import (
    select_comfy_registry_wheel_versions,
)
from local_lm.db import Base
from local_lm.models import ComfyRegistryInstall


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def _resolution(**changes: Any) -> ComfyNodeResolution:
    value = ComfyNodeResolution(
        package_id="comfyui-example-node",
        declared_version="1.2.3",
        node_types=("ExampleNode",),
        install_kind="registry_archive",
        repository_url="https://github.com/example/comfyui-example-node.git",
        registry_record_id="record-123",
        download_url="https://cdn.comfy.org/example/1.2.3.zip",
    )
    return replace(value, **changes)


def _closure(resolution: ComfyNodeResolution) -> ComfyRegistryWheelClosure:
    dependency_plan = plan_comfy_registry_dependencies(resolution.pip_dependencies)
    marker_environment, supported_tags = current_comfy_registry_wheel_target()
    target_sha256 = comfy_registry_wheel_target_sha256(
        marker_environment,
        supported_tags,
    )
    manifest = build_comfy_registry_wheel_artifact_manifest(
        dependency_plan.declaration_sha256,
        target_sha256,
        (),
    )
    return plan_comfy_registry_wheel_closure(
        manifest,
        {},
        marker_environment=marker_environment,
    )


class _ArchiveDownloader:
    def __init__(self, *, fail_after_create: bool = False) -> None:
        self.calls = 0
        self.fail_after_create = fail_after_create

    async def download_and_stage(
        self,
        _resolution: ComfyNodeResolution,
        destination: Path,
        *,
        progress: DownloadProgress | None = None,
    ) -> ComfyRegistryArchiveReport:
        self.calls += 1
        destination.mkdir()
        content = b"NODE_CLASS_MAPPINGS = {}\n"
        (destination / "__init__.py").write_bytes(content)
        if self.fail_after_create:
            raise RuntimeError("archive failed")
        digest = hashlib.sha256(content).hexdigest()
        manifest = f"__init__.py{chr(0)}{len(content)}{chr(0)}{digest}{chr(10)}"
        return ComfyRegistryArchiveReport(
            archive_sha256="a" * 64,
            manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
            entry_count=1,
            file_count=1,
            expanded_bytes=len(content),
            python_file_count=1,
            dependency_manifests=(),
            install_scripts=(),
            startup_hooks=(),
            native_files=(),
            top_level_entries=("__init__.py",),
        )


class _WheelDownloader:
    def __init__(self) -> None:
        self.calls = 0

    async def download_and_stage(
        self,
        _manifest: ComfyRegistryWheelArtifactManifest,
        _destination: Path,
        *,
        progress: WheelDownloadProgress | None = None,
    ) -> ComfyRegistryWheelStageReport:
        self.calls += 1
        raise AssertionError("an empty dependency closure must not download wheels")


class _PopulatedWheelDownloader:
    def __init__(self, content: bytes, *, manifest_sha256: str | None = None) -> None:
        self.content = content
        self.manifest_sha256 = manifest_sha256
        self.calls = 0

    async def download_and_stage(
        self,
        manifest: ComfyRegistryWheelArtifactManifest,
        destination: Path,
        *,
        progress: WheelDownloadProgress | None = None,
    ) -> ComfyRegistryWheelStageReport:
        self.calls += 1
        artifact = manifest.artifacts[0]
        destination.mkdir()
        (destination / artifact.filename).write_bytes(self.content)
        staged = ComfyRegistryStagedWheel(
            artifact.filename,
            artifact.sha256,
            artifact.size_bytes,
            None,
            None,
            None,
        )
        return ComfyRegistryWheelStageReport(
            self.manifest_sha256 or manifest.manifest_sha256,
            "b" * 64,
            len(self.content),
            (staged,),
        )


def _wheel_content() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("alpha/__init__.py", "VALUE = 1")
        archive.writestr(
            "alpha-1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: alpha\nVersion: 1.0\n\n",
        )
        archive.writestr(
            "alpha-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n\n",
        )
        archive.writestr("alpha-1.0.dist-info/RECORD", "")
    return output.getvalue()


def _populated_closure(content: bytes) -> ComfyRegistryWheelClosure:
    metadata = b"Metadata-Version: 2.4\nName: alpha\nVersion: 1.0\n\n"
    filename = "alpha-1.0-py3-none-any.whl"
    marker_environment = {key: str(value) for key, value in default_environment().items()}
    marker_environment["extra"] = ""
    manifest = resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies(["alpha==1.0"]),
        {
            "alpha": {
                "meta": {"api-version": "1.4"},
                "name": "alpha",
                "files": [
                    {
                        "filename": filename,
                        "url": f"https://files.pythonhosted.org/packages/aa/{filename}",
                        "hashes": {"sha256": hashlib.sha256(content).hexdigest()},
                        "requires-python": ">=3.12",
                        "yanked": False,
                        "size": len(content),
                        "core-metadata": {"sha256": hashlib.sha256(metadata).hexdigest()},
                    }
                ],
            }
        },
        marker_environment=marker_environment,
        supported_tags=("py3-none-any",),
    )
    return plan_comfy_registry_wheel_closure(
        manifest,
        {filename: metadata},
        marker_environment=marker_environment,
    )


def _transitive_closure(
    resolution: ComfyNodeResolution,
) -> ComfyRegistryWheelClosure:
    marker_environment = {key: str(value) for key, value in default_environment().items()}
    marker_environment["extra"] = ""
    alpha_metadata = b"Metadata-Version: 2.4\nName: alpha\nVersion: 1.0\nRequires-Dist: beta>=2\n\n"
    alpha_filename = "alpha-1.0-py3-none-any.whl"
    manifest = resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies(resolution.pip_dependencies),
        {
            "alpha": {
                "meta": {"api-version": "1.4"},
                "name": "alpha",
                "files": [
                    {
                        "filename": alpha_filename,
                        "url": f"https://files.pythonhosted.org/packages/aa/{alpha_filename}",
                        "hashes": {"sha256": "a" * 64},
                        "requires-python": ">=3.12",
                        "yanked": False,
                        "size": 100,
                        "core-metadata": {"sha256": hashlib.sha256(alpha_metadata).hexdigest()},
                    }
                ],
            }
        },
        marker_environment=marker_environment,
        supported_tags=("py3-none-any",),
    )
    first = plan_comfy_registry_wheel_closure(
        manifest,
        {alpha_filename: alpha_metadata},
        marker_environment=marker_environment,
    )
    beta_metadata = b"Metadata-Version: 2.4\nName: beta\nVersion: 2.0\n\n"
    beta_filename = "beta-2.0-py3-none-any.whl"
    selection = select_comfy_registry_wheel_versions(
        first.metadata_plan,
        {
            "beta": {
                "meta": {"api-version": "1.4"},
                "name": "beta",
                "files": [
                    {
                        "filename": beta_filename,
                        "url": f"https://files.pythonhosted.org/packages/bb/{beta_filename}",
                        "hashes": {"sha256": "b" * 64},
                        "requires-python": ">=3.12",
                        "yanked": False,
                        "size": 100,
                        "core-metadata": {"sha256": hashlib.sha256(beta_metadata).hexdigest()},
                    }
                ],
            }
        },
        marker_environment=marker_environment,
        supported_tags=("py3-none-any",),
    )
    return advance_comfy_registry_wheel_closure(
        first,
        selection,
        {
            alpha_filename: alpha_metadata,
            beta_filename: beta_metadata,
        },
        marker_environment=marker_environment,
    )


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    custom_nodes = tmp_path / "custom-nodes"
    state = tmp_path / "state"
    custom_nodes.mkdir(parents=True)
    state.mkdir(parents=True)
    return custom_nodes, state


async def _prepare(
    session: Session,
    tmp_path: Path,
    *,
    resolution: ComfyNodeResolution | None = None,
    closure: ComfyRegistryWheelClosure | None = None,
    archive: _ArchiveDownloader | None = None,
    media_worker_stopped: bool = True,
) -> tuple[ComfyRegistryPreparation, _ArchiveDownloader, _WheelDownloader, Path, Path]:
    selected = resolution or _resolution()
    selected_closure = closure or _closure(selected)
    archive_downloader = archive or _ArchiveDownloader()
    wheel_downloader = _WheelDownloader()
    custom_nodes, state = _roots(tmp_path)
    result = await prepare_comfy_registry_install(
        session,
        resolution=selected,
        closure=selected_closure,
        archive_downloader=archive_downloader,
        wheel_downloader=wheel_downloader,
        python_executable=Path(sys.executable),
        custom_node_root=custom_nodes,
        state_root=state,
        media_worker_stopped=media_worker_stopped,
    )
    return result, archive_downloader, wheel_downloader, custom_nodes, state


async def test_prepares_exact_package_as_committed_inert_install(
    session: Session,
    tmp_path: Path,
) -> None:
    result, archive, wheels, custom_nodes, state = await _prepare(session, tmp_path)

    assert archive.calls == 1
    assert wheels.calls == 0
    install = session.get(ComfyRegistryInstall, result.install_id)
    assert install is not None
    assert install.trusted is False
    assert install.active is False
    assert install.installed_path == result.installed_path
    assert install.wheel_closure_sha256 == result.wheel_closure_sha256
    assert install.wheel_environment_sha256 == result.wheel_environment_sha256
    assert result.reused_wheel_environment is False
    assert (custom_nodes / result.installed_path / "__init__.py").is_file()
    assert (state / "registry-wheel-environments" / result.wheel_environment_path).is_dir()
    assert not any((state / "registry-wheel-staging").iterdir())
    session.close()
    with Session(session.get_bind()) as check:
        assert check.scalar(select(func.count()).select_from(ComfyRegistryInstall)) == 1


async def test_pre_staged_commit_archive_is_reused_without_a_second_download(
    session: Session,
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    resolution = _resolution(
        declared_version=revision,
        install_kind="git_commit",
        registry_record_id=None,
        download_url=None,
        warnings=(
            "source_review_required",
            "dependency_manifest_review_required",
        ),
    )
    archive = _ArchiveDownloader()
    custom_nodes, state = _roots(tmp_path)
    staged = await stage_comfy_registry_install_archive(
        resolution=resolution,
        archive_downloader=archive,
        custom_node_root=custom_nodes,
        media_worker_stopped=True,
    )

    result = await prepare_comfy_registry_install(
        session,
        resolution=resolution,
        closure=_closure(resolution),
        archive_downloader=archive,
        wheel_downloader=_WheelDownloader(),
        python_executable=Path(sys.executable),
        custom_node_root=custom_nodes,
        state_root=state,
        media_worker_stopped=True,
        staged_archive=staged,
    )

    install = session.get(ComfyRegistryInstall, result.install_id)
    assert archive.calls == 1
    assert install is not None
    assert install.package_version == revision
    assert install.registry_record_id.startswith("github-commit:")
    assert install.archive_sha256 == staged.report.archive_sha256
    assert (custom_nodes / result.installed_path / "__init__.py").is_file()


async def test_reuses_only_database_bound_verified_environment(
    session: Session,
    tmp_path: Path,
) -> None:
    first, _, _, _, state = await _prepare(session, tmp_path / "first")
    second_resolution = _resolution(
        package_id="comfyui-second-node",
        declared_version="4.5.6",
        repository_url="https://github.com/example/comfyui-second-node.git",
        registry_record_id="record-456",
        download_url="https://cdn.comfy.org/example/4.5.6.zip",
    )
    custom_nodes = tmp_path / "second-custom-nodes"
    custom_nodes.mkdir()
    archive = _ArchiveDownloader()
    wheels = _WheelDownloader()

    second = await prepare_comfy_registry_install(
        session,
        resolution=second_resolution,
        closure=_closure(second_resolution),
        archive_downloader=archive,
        wheel_downloader=wheels,
        python_executable=Path(sys.executable),
        custom_node_root=custom_nodes,
        state_root=state,
        media_worker_stopped=True,
    )

    assert second.reused_wheel_environment is True
    assert second.wheel_environment_path == first.wheel_environment_path
    assert second.wheel_environment_sha256 == first.wheel_environment_sha256
    assert archive.calls == 1
    assert wheels.calls == 0


async def test_staged_wheel_flows_through_offline_environment_and_is_removed(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _wheel_content()
    resolution = _resolution(pip_dependencies=("alpha==1.0",))
    closure = _populated_closure(content)
    archive = _ArchiveDownloader()
    wheels = _PopulatedWheelDownloader(content)
    custom_nodes, state = _roots(tmp_path)

    async def fake_run_pip(
        _python: Path,
        staged: tuple[Path, ...],
        target: Path,
    ) -> None:
        assert staged[0].read_bytes() == content
        package = target / "alpha"
        package.mkdir()
        (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        dist_info = target / "alpha-1.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.4\nName: alpha\nVersion: 1.0\n\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(environment_module, "_run_pip", fake_run_pip)

    result = await prepare_comfy_registry_install(
        session,
        resolution=resolution,
        closure=closure,
        archive_downloader=archive,
        wheel_downloader=wheels,
        python_executable=Path(sys.executable),
        custom_node_root=custom_nodes,
        state_root=state,
        media_worker_stopped=True,
    )

    assert archive.calls == 1
    assert wheels.calls == 1
    assert not any((state / "registry-wheel-staging").iterdir())
    environment = state / "registry-wheel-environments" / result.wheel_environment_path
    assert (environment / "site-packages" / "alpha" / "__init__.py").is_file()


async def test_wheel_stage_identity_mismatch_rolls_back_every_new_artifact(
    session: Session,
    tmp_path: Path,
) -> None:
    content = _wheel_content()
    resolution = _resolution(pip_dependencies=("alpha==1.0",))
    closure = _populated_closure(content)
    archive = _ArchiveDownloader()
    wheels = _PopulatedWheelDownloader(content, manifest_sha256="f" * 64)
    custom_nodes, state = _roots(tmp_path)

    with pytest.raises(ComfyRegistryLifecycleError) as raised:
        await prepare_comfy_registry_install(
            session,
            resolution=resolution,
            closure=closure,
            archive_downloader=archive,
            wheel_downloader=wheels,
            python_executable=Path(sys.executable),
            custom_node_root=custom_nodes,
            state_root=state,
            media_worker_stopped=True,
        )

    assert raised.value.code == "wheel_stage_identity_mismatch"
    assert archive.calls == 1
    assert wheels.calls == 1
    assert not list(custom_nodes.iterdir())
    assert not any((state / "registry-wheel-staging").iterdir())
    assert not any((state / "registry-wheel-environments").iterdir())
    assert session.scalar(select(func.count()).select_from(ComfyRegistryInstall)) == 0


async def test_unbound_existing_environment_is_preserved_and_refused(
    session: Session,
    tmp_path: Path,
) -> None:
    resolution = _resolution()
    closure = _closure(resolution)
    archive = _ArchiveDownloader()
    custom_nodes, state = _roots(tmp_path)
    environments = state / "registry-wheel-environments"
    environments.mkdir()
    existing = environments / f"registry-wheels-{closure.closure_sha256}"
    existing.mkdir()
    marker = existing / "unbound.txt"
    marker.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ComfyRegistryLifecycleError) as raised:
        await prepare_comfy_registry_install(
            session,
            resolution=resolution,
            closure=closure,
            archive_downloader=archive,
            wheel_downloader=_WheelDownloader(),
            python_executable=Path(sys.executable),
            custom_node_root=custom_nodes,
            state_root=state,
            media_worker_stopped=True,
        )

    assert raised.value.code == "unbound_wheel_environment"
    assert archive.calls == 1
    assert not list(custom_nodes.iterdir())
    assert marker.read_text(encoding="utf-8") == "preserve\n"


async def test_running_worker_rejects_before_filesystem_or_download(
    session: Session,
    tmp_path: Path,
) -> None:
    archive = _ArchiveDownloader()
    with pytest.raises(ComfyRegistryLifecycleError) as raised:
        await prepare_comfy_registry_install(
            session,
            resolution=_resolution(),
            closure=_closure(_resolution()),
            archive_downloader=archive,
            wheel_downloader=_WheelDownloader(),
            python_executable=Path(sys.executable),
            custom_node_root=tmp_path / "missing-custom-nodes",
            state_root=tmp_path / "missing-state",
            media_worker_stopped=False,
        )

    assert raised.value.code == "media_worker_running"
    assert archive.calls == 0
    assert not list(tmp_path.iterdir())


async def test_dependency_mismatch_rejects_before_download_or_storage_creation(
    session: Session,
    tmp_path: Path,
) -> None:
    resolution = _resolution(pip_dependencies=("example-runtime==2.0.0",))
    archive = _ArchiveDownloader()
    custom_nodes, state = _roots(tmp_path)

    with pytest.raises(ComfyRegistryLifecycleError) as raised:
        await prepare_comfy_registry_install(
            session,
            resolution=resolution,
            closure=_closure(_resolution()),
            archive_downloader=archive,
            wheel_downloader=_WheelDownloader(),
            python_executable=Path(sys.executable),
            custom_node_root=custom_nodes,
            state_root=state,
            media_worker_stopped=True,
        )

    assert raised.value.code == "dependency_closure_mismatch"
    assert archive.calls == 0
    assert not (state / "registry-wheel-environments").exists()
    assert not (state / "registry-wheel-staging").exists()


def test_transitive_closure_keeps_root_identity_at_lifecycle_boundary() -> None:
    resolution = _resolution(pip_dependencies=("alpha==1.0",))

    artifacts = lifecycle_module._complete_closure(
        _transitive_closure(resolution),
        resolution,
    )

    assert [artifact.name for artifact in artifacts] == ["alpha", "beta"]


async def test_partial_archive_failure_is_cleaned_and_database_stays_empty(
    session: Session,
    tmp_path: Path,
) -> None:
    archive = _ArchiveDownloader(fail_after_create=True)
    custom_nodes, state = _roots(tmp_path)

    with pytest.raises(RuntimeError, match="archive failed"):
        await prepare_comfy_registry_install(
            session,
            resolution=_resolution(),
            closure=_closure(_resolution()),
            archive_downloader=archive,
            wheel_downloader=_WheelDownloader(),
            python_executable=Path(sys.executable),
            custom_node_root=custom_nodes,
            state_root=state,
            media_worker_stopped=True,
        )

    assert not list(custom_nodes.iterdir())
    assert not any((state / "registry-wheel-staging").iterdir())
    assert not any((state / "registry-wheel-environments").iterdir())
    assert session.scalar(select(func.count()).select_from(ComfyRegistryInstall)) == 0


async def test_persistence_failure_removes_new_node_and_environment(
    session: Session,
    tmp_path: Path,
) -> None:
    invalid = _resolution(repository_url="https://attacker.invalid/node.git")
    archive = _ArchiveDownloader()
    custom_nodes, state = _roots(tmp_path)

    with pytest.raises(ValueError, match="repository URL"):
        await prepare_comfy_registry_install(
            session,
            resolution=invalid,
            closure=_closure(invalid),
            archive_downloader=archive,
            wheel_downloader=_WheelDownloader(),
            python_executable=Path(sys.executable),
            custom_node_root=custom_nodes,
            state_root=state,
            media_worker_stopped=True,
        )

    assert archive.calls == 0
    assert not list(custom_nodes.iterdir())
    assert not (state / "registry-wheel-staging").exists()
    assert not (state / "registry-wheel-environments").exists()
    assert session.scalar(select(func.count()).select_from(ComfyRegistryInstall)) == 0


async def test_exact_retry_is_rejected_before_another_download(
    session: Session,
    tmp_path: Path,
) -> None:
    result, _, _, custom_nodes, state = await _prepare(session, tmp_path)
    archive = _ArchiveDownloader()

    with pytest.raises(ComfyRegistryLifecycleError) as raised:
        await prepare_comfy_registry_install(
            session,
            resolution=_resolution(),
            closure=_closure(_resolution()),
            archive_downloader=archive,
            wheel_downloader=_WheelDownloader(),
            python_executable=Path(sys.executable),
            custom_node_root=custom_nodes,
            state_root=state,
            media_worker_stopped=True,
        )

    assert raised.value.code == "registry_install_exists"
    assert archive.calls == 0
    assert (custom_nodes / result.installed_path).is_dir()


async def test_existing_install_is_not_removed_by_a_staged_argument(
    session: Session,
    tmp_path: Path,
) -> None:
    result, _, _, custom_nodes, state = await _prepare(session, tmp_path)
    content = b"NODE_CLASS_MAPPINGS = {}\n"
    digest = hashlib.sha256(content).hexdigest()
    manifest = f"__init__.py{chr(0)}{len(content)}{chr(0)}{digest}{chr(10)}"
    staged = lifecycle_module.ComfyRegistryStagedArchive(
        result.installed_path,
        custom_nodes / result.installed_path,
        ComfyRegistryArchiveReport(
            archive_sha256="a" * 64,
            manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
            entry_count=1,
            file_count=1,
            expanded_bytes=len(content),
            python_file_count=1,
            dependency_manifests=(),
            install_scripts=(),
            startup_hooks=(),
            native_files=(),
            top_level_entries=("__init__.py",),
        ),
    )

    with pytest.raises(ComfyRegistryLifecycleError) as raised:
        await prepare_comfy_registry_install(
            session,
            resolution=_resolution(),
            closure=_closure(_resolution()),
            archive_downloader=_ArchiveDownloader(),
            wheel_downloader=_WheelDownloader(),
            python_executable=Path(sys.executable),
            custom_node_root=custom_nodes,
            state_root=state,
            media_worker_stopped=True,
            staged_archive=staged,
        )

    assert raised.value.code == "registry_install_exists"
    assert (custom_nodes / result.installed_path / "__init__.py").read_bytes() == content
