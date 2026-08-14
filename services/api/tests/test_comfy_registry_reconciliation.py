from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from local_lm.comfy_registry_reconciliation import (
    ComfyRegistryReconciliationError,
    inspect_registry_install_disk_state,
    remove_registry_install,
)
from local_lm.db import Base
from local_lm.domain import JobKind, JobStatus
from local_lm.models import ComfyRegistryInstall, Job


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as current:
        yield current


def _install(
    session: Session,
    *,
    installed_path: str = "lm-atelier-registry_example",
    environment_path: str | None = None,
    active: bool = False,
) -> ComfyRegistryInstall:
    environment_path = environment_path or f"registry-wheels-v3-{'c' * 64}"
    install = ComfyRegistryInstall(
        package_id=f"package-{installed_path}",
        package_version="1.0.0",
        registry_record_id=f"record-{installed_path}",
        repository_url="https://example.test/repository",
        download_url="https://example.test/archive.zip",
        archive_sha256="a" * 64,
        manifest_sha256="b" * 64,
        installed_path=installed_path,
        node_types_json=["ExampleNode"],
        pip_dependencies_json=[],
        review_json={"review_required": True},
        wheel_closure_sha256="c" * 64,
        wheel_environment_sha256="d" * 64,
        wheel_environment_path=environment_path,
        trusted=False,
        active=active,
    )
    session.add(install)
    session.commit()
    return install


def _paths(tmp_path: Path, install: ComfyRegistryInstall) -> tuple[Path, Path, Path, Path]:
    node_root = tmp_path / "custom_nodes"
    environment_root = tmp_path / "registry-wheel-environments"
    node = node_root / install.installed_path
    environment = environment_root / str(install.wheel_environment_path)
    node.mkdir(parents=True)
    environment.mkdir(parents=True)
    (node / "node.py").write_bytes(b"node")
    (environment / "environment.json").write_bytes(b"environment")
    return node_root, environment_root, node, environment


def test_disk_state_reports_the_exact_missing_half(session: Session, tmp_path: Path) -> None:
    install = _install(session)
    node_root, environment_root, node, environment = _paths(tmp_path, install)

    assert (
        inspect_registry_install_disk_state(
            install,
            custom_node_root=node_root,
            environment_root=environment_root,
        ).status
        == "ready"
    )
    node.rename(node.with_name("removed-node"))
    state = inspect_registry_install_disk_state(
        install,
        custom_node_root=node_root,
        environment_root=environment_root,
    )
    assert state.status == "node_files_missing"
    assert state.node_files_present is False
    assert state.wheel_environment_present is True
    environment.rename(environment.with_name("removed-environment"))
    assert (
        inspect_registry_install_disk_state(
            install,
            custom_node_root=node_root,
            environment_root=environment_root,
        ).status
        == "files_missing"
    )


def test_removal_deletes_stale_row_and_remaining_environment(
    session: Session,
    tmp_path: Path,
) -> None:
    install = _install(session)
    node_root, environment_root, node, environment = _paths(tmp_path, install)
    node.rename(node.with_name("runtime-upgrade-removed-it"))

    remove_registry_install(
        session,
        install_id=install.id,
        custom_node_root=node_root,
        environment_root=environment_root,
    )

    assert session.get(ComfyRegistryInstall, install.id) is None
    assert not environment.exists()
    assert not list(environment_root.glob(".lm-atelier-removing-*"))


def test_removal_preserves_a_shared_wheel_environment(session: Session, tmp_path: Path) -> None:
    first = _install(session)
    second = _install(
        session,
        installed_path="lm-atelier-registry_second",
        environment_path=first.wheel_environment_path,
    )
    node_root, environment_root, first_node, environment = _paths(tmp_path, first)
    second_node = node_root / second.installed_path
    second_node.mkdir()

    remove_registry_install(
        session,
        install_id=first.id,
        custom_node_root=node_root,
        environment_root=environment_root,
    )

    assert not first_node.exists()
    assert second_node.exists()
    assert environment.exists()
    assert session.get(ComfyRegistryInstall, second.id) is not None


def test_active_and_refreshing_installs_refuse_without_mutation(
    session: Session,
    tmp_path: Path,
) -> None:
    active = _install(session, active=True)
    node_root, environment_root, node, environment = _paths(tmp_path, active)
    with pytest.raises(ComfyRegistryReconciliationError) as raised:
        remove_registry_install(
            session,
            install_id=active.id,
            custom_node_root=node_root,
            environment_root=environment_root,
        )
    assert raised.value.code == "registry_install_active"
    assert node.exists() and environment.exists()

    active.active = False
    session.add(
        Job(
            kind=JobKind.REGISTRY_PREPARE.value,
            status=JobStatus.QUEUED.value,
            payload_json={"renew_install_id": active.id},
        )
    )
    session.commit()
    with pytest.raises(ComfyRegistryReconciliationError) as raised:
        remove_registry_install(
            session,
            install_id=active.id,
            custom_node_root=node_root,
            environment_root=environment_root,
        )
    assert raised.value.code == "registry_install_busy"
    assert node.exists() and environment.exists()


def test_an_unbounded_refresh_queue_refuses_without_scanning_forever(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import local_lm.comfy_registry_reconciliation as reconciliation

    install = _install(session)
    node_root, environment_root, node, environment = _paths(tmp_path, install)
    for index in range(2):
        session.add(
            Job(
                kind=JobKind.REGISTRY_PREPARE.value,
                status=JobStatus.QUEUED.value,
                payload_json={"package_id": f"unrelated-{index}"},
            )
        )
    session.commit()
    monkeypatch.setattr(reconciliation, "_MAX_PENDING_REFRESH_JOBS", 1)

    with pytest.raises(ComfyRegistryReconciliationError) as refusal:
        remove_registry_install(
            session,
            install_id=install.id,
            custom_node_root=node_root,
            environment_root=environment_root,
        )

    assert refusal.value.code == "registry_install_busy"
    assert session.get(ComfyRegistryInstall, install.id) is not None
    assert node.is_dir()
    assert environment.is_dir()


def test_failed_commit_restores_both_managed_paths(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install = _install(session)
    node_root, environment_root, node, environment = _paths(tmp_path, install)

    def fail_commit() -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(session, "commit", fail_commit)
    with pytest.raises(ComfyRegistryReconciliationError) as raised:
        remove_registry_install(
            session,
            install_id=install.id,
            custom_node_root=node_root,
            environment_root=environment_root,
        )

    assert raised.value.code == "registry_install_remove_failed"
    assert node.exists() and environment.exists()
    assert (node / "node.py").read_bytes() == b"node"
    assert not list(node_root.glob(".lm-atelier-removing-*"))
    assert (
        session.scalar(select(ComfyRegistryInstall.id).where(ComfyRegistryInstall.id == install.id))
        == install.id
    )
