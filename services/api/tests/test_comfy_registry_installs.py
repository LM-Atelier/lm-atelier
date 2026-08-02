from __future__ import annotations

from collections.abc import Generator
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from local_lm.comfy_registry import ComfyNodeResolution
from local_lm.comfy_registry_archives import ComfyRegistryArchiveReport
from local_lm.comfy_registry_installs import (
    ComfyRegistryInstallError,
    installed_comfy_registry_versions,
    persist_comfy_registry_install,
)
from local_lm.db import Base
from local_lm.models import ComfyRegistryInstall


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
        (_resolution(install_kind="git_commit"), "exact Comfy Registry archive"),
        (_resolution(error_code="registry_version_inactive"), "exact Comfy Registry archive"),
        (_resolution(package_id="Invalid Package"), "invalid Registry package id"),
        (_resolution(declared_version="latest"), "invalid Registry package version"),
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
