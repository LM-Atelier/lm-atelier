from __future__ import annotations

import hashlib
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

import local_lm.comfy_registry_installs as install_module
from local_lm.comfy_registry import ComfyNodeResolution
from local_lm.comfy_registry_archives import ComfyRegistryArchiveReport
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_installs import (
    ComfyRegistryInstallError,
    bind_comfy_registry_wheel_environment,
    installed_comfy_registry_versions,
    persist_comfy_registry_install,
    scoped_comfy_registry_launch_contract,
    trusted_comfy_registry_launch_contract,
)
from local_lm.comfy_registry_wheel_environments import ComfyRegistryWheelEnvironmentReport
from local_lm.db import Base
from local_lm.models import ComfyRegistryInstall
from local_lm.workflow_activations import WorkflowRegistryLaunchBinding


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def _resolution(**changes: object) -> ComfyNodeResolution:
    values: dict[str, object] = {
        "package_id": "comfyui-example-node",
        "declared_version": "1.2.3",
        "node_types": ("ExampleLoader", "ExampleSampler"),
        "install_kind": "registry_archive",
        "repository_url": "https://github.com/example/comfyui-example-node.git",
        "registry_record_id": "record-123",
        "download_url": "https://cdn.comfy.org/example/1.2.3.zip",
        "pip_dependencies": ("example-runtime==2.0.0",),
        "warnings": ("deprecated_version",),
    }
    values.update(changes)
    return ComfyNodeResolution(**values)  # type: ignore[arg-type]


def _archive(**changes: object) -> ComfyRegistryArchiveReport:
    values: dict[str, object] = {
        "archive_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "entry_count": 8,
        "file_count": 6,
        "expanded_bytes": 4096,
        "python_file_count": 3,
        "dependency_manifests": ("requirements.txt",),
        "install_scripts": ("install.py",),
        "startup_hooks": ("prestartup_script.py",),
        "native_files": ("native/example.pyd",),
        "top_level_entries": ("__init__.py", "nodes"),
        "review_required": True,
    }
    values.update(changes)
    return ComfyRegistryArchiveReport(**values)  # type: ignore[arg-type]


def test_persists_exact_identity_as_inert_and_untrusted(session: Session) -> None:
    install = persist_comfy_registry_install(
        session,
        resolution=_resolution(),
        archive=_archive(),
        installed_path="lm-atelier-registry_example",
    )
    session.commit()

    stored = session.get(ComfyRegistryInstall, install.id)
    assert stored is not None
    assert stored.package_id == "comfyui-example-node"
    assert stored.package_version == "1.2.3"
    assert stored.registry_record_id == "record-123"
    assert stored.repository_url == "https://github.com/example/comfyui-example-node.git"
    assert stored.download_url == "https://cdn.comfy.org/example/1.2.3.zip"
    assert stored.archive_sha256 == "a" * 64
    assert stored.manifest_sha256 == "b" * 64
    assert stored.installed_path == "lm-atelier-registry_example"
    assert stored.node_types_json == ["ExampleLoader", "ExampleSampler"]
    assert stored.pip_dependencies_json == ["example-runtime==2.0.0"]
    assert stored.trusted is False
    assert stored.active is False
    assert stored.review_json == {
        "entry_count": 8,
        "file_count": 6,
        "expanded_bytes": 4096,
        "python_file_count": 3,
        "dependency_manifests": ["requirements.txt"],
        "install_scripts": ["install.py"],
        "startup_hooks": ["prestartup_script.py"],
        "native_files": ["native/example.pyd"],
        "top_level_entries": ["__init__.py", "nodes"],
        "review_required": True,
        "registry_warnings": ["deprecated_version"],
    }


def test_persists_commit_source_and_observed_archive_identity(session: Session) -> None:
    revision = "a" * 40
    install = persist_comfy_registry_install(
        session,
        resolution=_resolution(
            declared_version=revision,
            install_kind="git_commit",
            registry_record_id=None,
            download_url=None,
            warnings=(
                "source_review_required",
                "dependency_manifest_review_required",
            ),
        ),
        archive=_archive(),
        installed_path="lm-atelier-registry_commit",
    )

    assert install.package_version == revision
    assert install.registry_record_id.startswith("github-commit:")
    assert install.repository_url == "https://github.com/example/comfyui-example-node.git"
    assert install.download_url == (
        "https://codeload.github.com/example/comfyui-example-node/zip/" + revision
    )
    assert install.archive_sha256 == "a" * 64


def test_exact_retry_is_idempotent(session: Session) -> None:
    first = persist_comfy_registry_install(
        session,
        resolution=_resolution(),
        archive=_archive(),
        installed_path="lm-atelier-registry_example",
    )
    second = persist_comfy_registry_install(
        session,
        resolution=_resolution(),
        archive=_archive(),
        installed_path="lm-atelier-registry_example",
    )

    assert second is first
    assert session.scalar(select(func.count()).select_from(ComfyRegistryInstall)) == 1


def test_installed_versions_include_only_trusted_active_records(session: Session) -> None:
    ready = persist_comfy_registry_install(
        session,
        resolution=_resolution(),
        archive=_archive(),
        installed_path="lm-atelier-registry_ready",
    )
    ready.trusted = True
    ready.active = True
    inactive = persist_comfy_registry_install(
        session,
        resolution=_resolution(
            declared_version="2.0.0",
            registry_record_id="record-200",
            download_url="https://cdn.comfy.org/example/2.0.0.zip",
        ),
        archive=_archive(archive_sha256="c" * 64, manifest_sha256="d" * 64),
        installed_path="lm-atelier-registry_inactive",
    )
    inactive.trusted = True
    untrusted = persist_comfy_registry_install(
        session,
        resolution=_resolution(
            declared_version="3.0.0",
            registry_record_id="record-300",
            download_url="https://cdn.comfy.org/example/3.0.0.zip",
        ),
        archive=_archive(archive_sha256="e" * 64, manifest_sha256="f" * 64),
        installed_path="lm-atelier-registry_untrusted",
    )
    untrusted.active = True
    session.flush()

    assert installed_comfy_registry_versions(session) == {"comfyui-example-node": {"1.2.3"}}


def test_exact_wheel_environment_is_bound_before_trust(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install = persist_comfy_registry_install(
        session,
        resolution=_resolution(),
        archive=_archive(),
        installed_path="lm-atelier-registry_example",
    )
    closure_sha256 = "c" * 64
    environment_sha256 = "d" * 64
    dependency_plan = plan_comfy_registry_dependencies(install.pip_dependencies_json)
    closure = SimpleNamespace(
        complete=True,
        closure_sha256=closure_sha256,
        manifest=SimpleNamespace(declaration_sha256=dependency_plan.declaration_sha256),
    )
    report = ComfyRegistryWheelEnvironmentReport(
        closure_sha256,
        environment_sha256,
        1,
        2,
        100,
        (),
    )
    root = tmp_path / "environments"
    root.mkdir()
    destination = root / f"registry-wheels-{closure_sha256}"
    destination.mkdir()
    monkeypatch.setattr(
        install_module,
        "validate_comfy_registry_wheel_closure",
        lambda _closure: (object(),),
    )
    monkeypatch.setattr(
        install_module,
        "verify_comfy_registry_wheel_environment",
        lambda *_args, **_kwargs: report,
    )

    bind_comfy_registry_wheel_environment(
        install,
        closure,  # type: ignore[arg-type]
        report,
        destination,
        environment_root=root,
    )

    assert install.wheel_closure_sha256 == closure_sha256
    assert install.wheel_environment_sha256 == environment_sha256
    assert install.wheel_environment_path == destination.name


def test_trusted_launch_contract_revalidates_node_code_and_overlay(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_root = tmp_path / "custom_nodes"
    node_root.mkdir()
    folder_name = "lm-atelier-registry_example"
    folder = node_root / folder_name
    folder.mkdir()
    content = b"NODE_CLASS_MAPPINGS = {}"
    (folder / "node.py").write_bytes(content)
    manifest = (
        f"node.py{chr(0)}{len(content)}{chr(0)}{hashlib.sha256(content).hexdigest()}{chr(10)}"
    )
    install = persist_comfy_registry_install(
        session,
        resolution=_resolution(),
        archive=_archive(
            manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
            entry_count=1,
            file_count=1,
            expanded_bytes=len(content),
            python_file_count=1,
        ),
        installed_path=folder_name,
    )
    closure_sha256 = "c" * 64
    environment_sha256 = "d" * 64
    environment_root = tmp_path / "environments"
    environment_root.mkdir()
    environment = environment_root / f"registry-wheels-{closure_sha256}"
    site_packages = environment / "site-packages"
    site_packages.mkdir(parents=True)
    install.wheel_closure_sha256 = closure_sha256
    install.wheel_environment_sha256 = environment_sha256
    install.wheel_environment_path = environment.name
    install.trusted = True
    install.active = True
    runtime_content = b'{"style": "natural"}\n'
    runtime_path = folder / "styles" / "defaults.json"
    runtime_path.parent.mkdir()
    runtime_path.write_bytes(runtime_content)
    install.review_json = {
        **install.review_json,
        "runtime_files": [
            {
                "path": "styles/defaults.json",
                "size": len(runtime_content),
                "sha256": hashlib.sha256(runtime_content).hexdigest(),
            }
        ],
    }
    session.flush()
    report = ComfyRegistryWheelEnvironmentReport(
        closure_sha256,
        environment_sha256,
        1,
        0,
        0,
        (),
    )
    monkeypatch.setattr(
        install_module,
        "verify_comfy_registry_wheel_environment",
        lambda *_args, **_kwargs: report,
    )

    contract = trusted_comfy_registry_launch_contract(
        session,
        custom_node_root=node_root,
        environment_root=environment_root,
    )

    assert contract.custom_node_folders == (folder_name,)
    assert contract.site_packages == (site_packages,)
    assert contract.node_types == ("ExampleLoader", "ExampleSampler")

    unselected = persist_comfy_registry_install(
        session,
        resolution=_resolution(
            package_id="comfyui-unselected-node",
            registry_record_id="record-unselected",
            repository_url="https://github.com/example/comfyui-unselected-node.git",
            download_url="https://cdn.comfy.org/unselected/1.2.3.zip",
        ),
        archive=_archive(archive_sha256="e" * 64, manifest_sha256="f" * 64),
        installed_path="lm-atelier-registry_unselected",
    )
    unselected.wheel_closure_sha256 = "1" * 64
    unselected.wheel_environment_sha256 = "2" * 64
    unselected.wheel_environment_path = f"registry-wheels-{'1' * 64}"
    unselected.trusted = True
    unselected.active = True
    session.flush()
    binding = WorkflowRegistryLaunchBinding(
        install.id,
        folder.resolve(),
        site_packages.resolve(),
        install.package_id,
        install.package_version,
        install.archive_sha256,
        install.manifest_sha256,
        closure_sha256,
        environment_sha256,
        ("ExampleLoader", "ExampleSampler"),
    )

    scoped = scoped_comfy_registry_launch_contract(
        session,
        [binding],
        custom_node_root=node_root,
        environment_root=environment_root,
    )

    assert scoped == contract
    with pytest.raises(ComfyRegistryInstallError, match="scope identity changed"):
        scoped_comfy_registry_launch_contract(
            session,
            [replace(binding, package_version="9.9.9")],
            custom_node_root=node_root,
            environment_root=environment_root,
        )

    runtime_path.write_bytes(b'{"style": "changed"}\n')
    with pytest.raises(ComfyRegistryInstallError, match="node files failed verification"):
        trusted_comfy_registry_launch_contract(
            session,
            custom_node_root=node_root,
            environment_root=environment_root,
        )
    runtime_path.write_bytes(runtime_content)

    (folder / "node.py").write_bytes(b"changed")
    with pytest.raises(ComfyRegistryInstallError, match="node files failed verification"):
        trusted_comfy_registry_launch_contract(
            session,
            custom_node_root=node_root,
            environment_root=environment_root,
        )


def test_registry_record_cannot_change_archive_identity(session: Session) -> None:
    persist_comfy_registry_install(
        session,
        resolution=_resolution(),
        archive=_archive(),
        installed_path="lm-atelier-registry_example",
    )

    with pytest.raises(ComfyRegistryInstallError, match="record identity conflicts"):
        persist_comfy_registry_install(
            session,
            resolution=_resolution(),
            archive=_archive(archive_sha256="c" * 64),
            installed_path="lm-atelier-registry_example",
        )


def test_package_version_cannot_change_registry_record(session: Session) -> None:
    persist_comfy_registry_install(
        session,
        resolution=_resolution(),
        archive=_archive(),
        installed_path="lm-atelier-registry_example",
    )

    with pytest.raises(ComfyRegistryInstallError, match="bound to another record"):
        persist_comfy_registry_install(
            session,
            resolution=_resolution(registry_record_id="record-other"),
            archive=_archive(),
            installed_path="lm-atelier-registry_other",
        )


def test_managed_path_cannot_be_reused(session: Session) -> None:
    persist_comfy_registry_install(
        session,
        resolution=_resolution(),
        archive=_archive(),
        installed_path="lm-atelier-registry_example",
    )

    with pytest.raises(ComfyRegistryInstallError, match="path is already recorded"):
        persist_comfy_registry_install(
            session,
            resolution=_resolution(
                declared_version="2.0.0",
                registry_record_id="record-200",
                download_url="https://cdn.comfy.org/example/2.0.0.zip",
            ),
            archive=_archive(archive_sha256="c" * 64, manifest_sha256="d" * 64),
            installed_path="lm-atelier-registry_example",
        )


@pytest.mark.parametrize(
    ("resolution", "message"),
    [
        (_resolution(install_kind="git_commit"), "invalid commit revision"),
        (_resolution(error_code="registry_version_inactive"), "invalid identity"),
        (_resolution(package_id="Invalid Package"), "invalid identity"),
        (_resolution(declared_version="latest"), "invalid Registry version"),
        (_resolution(repository_url="https://example.com/node.git"), "repository URL"),
        (
            _resolution(repository_url="https://github.com/example/node"),
            "repository URL",
        ),
        (_resolution(download_url="https://example.com/node.zip"), "download URL"),
        (_resolution(node_types=()), "node type identities"),
    ],
)
def test_rejects_invalid_resolution_identity(
    session: Session,
    resolution: ComfyNodeResolution,
    message: str,
) -> None:
    with pytest.raises(ComfyRegistryInstallError, match=message):
        persist_comfy_registry_install(
            session,
            resolution=resolution,
            archive=_archive(),
            installed_path="lm-atelier-registry_example",
        )


@pytest.mark.parametrize("path", ["node", "../lm-atelier-registry_node", "folder/node"])
def test_rejects_unmanaged_install_paths(session: Session, path: str) -> None:
    with pytest.raises(ComfyRegistryInstallError, match="managed Registry install path"):
        persist_comfy_registry_install(
            session,
            resolution=_resolution(),
            archive=_archive(),
            installed_path=path,
        )


@pytest.mark.parametrize(
    "archive",
    [
        _archive(archive_sha256="invalid"),
        _archive(manifest_sha256="invalid"),
        replace(_archive(), review_required=False),
        _archive(entry_count=-1),
        _archive(file_count=9),
        _archive(dependency_manifests=("z.txt", "a.txt")),
    ],
)
def test_rejects_invalid_archive_identity(
    session: Session,
    archive: ComfyRegistryArchiveReport,
) -> None:
    with pytest.raises(ComfyRegistryInstallError):
        persist_comfy_registry_install(
            session,
            resolution=_resolution(),
            archive=archive,
            installed_path="lm-atelier-registry_example",
        )
