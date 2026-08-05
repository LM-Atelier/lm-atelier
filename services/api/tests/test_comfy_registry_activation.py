from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import local_lm.comfy_registry_activation as activation_module
from local_lm.comfy_registry_activation import (
    ComfyRegistryActivationError,
    activate_comfy_registry_install,
    deactivate_comfy_registry_install,
    review_comfy_registry_install,
)
from local_lm.comfy_registry_installs import ComfyRegistryLaunchContract
from local_lm.db import Base
from local_lm.models import ComfyRegistryInstall
from local_lm.source_omission_proof import PENDING_KEY, record_pending_omission


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


@pytest.fixture(autouse=True)
def registry_source_folder(tmp_path: Path) -> None:
    (tmp_path / "lm-atelier-registry_example").mkdir()


def _install(session: Session, *, trusted: bool = False, active: bool = False) -> str:
    install = ComfyRegistryInstall(
        package_id="comfyui-example-node",
        package_version="1.2.3",
        registry_record_id="record-123",
        repository_url="https://github.com/example/comfyui-example-node.git",
        download_url="https://cdn.comfy.org/example/1.2.3.zip",
        archive_sha256="a" * 64,
        manifest_sha256=hashlib.sha256(b"").hexdigest(),
        installed_path="lm-atelier-registry_example",
        node_types_json=["ExampleNode"],
        pip_dependencies_json=[],
        review_json={"review_required": True, "file_count": 0, "expanded_bytes": 0},
        wheel_closure_sha256="c" * 64,
        wheel_environment_sha256="d" * 64,
        wheel_environment_path=f"registry-wheels-{'c' * 64}",
        trusted=trusted,
        active=active,
    )
    session.add(install)
    session.commit()
    return install.id


def _verified_contract(session: Session, install_id: str) -> ComfyRegistryLaunchContract:
    install = session.get(ComfyRegistryInstall, install_id)
    assert install is not None
    assert install.trusted is True
    assert install.active is True
    return ComfyRegistryLaunchContract((install.installed_path,), (), ("ExampleNode",))


def test_explicit_trust_revalidates_with_temporary_activation_and_stays_inactive(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )

    state = review_comfy_registry_install(
        session,
        install_id=install_id,
        trusted=True,
        custom_node_root=tmp_path,
        environment_root=tmp_path,
        media_worker_stopped=True,
    )

    assert state.trusted is True
    assert state.active is False
    assert state.reviewed_at
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None
    assert stored.review_json["trusted_by_local_user"] is True


def test_failed_review_rolls_back_without_trust(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session)

    def fail(*_args: object, **_kwargs: object) -> ComfyRegistryLaunchContract:
        raise ValueError("changed")

    monkeypatch.setattr(activation_module, "trusted_comfy_registry_launch_contract", fail)

    with pytest.raises(ComfyRegistryActivationError) as raised:
        review_comfy_registry_install(
            session,
            install_id=install_id,
            trusted=True,
            custom_node_root=tmp_path,
            environment_root=tmp_path,
            media_worker_stopped=True,
        )

    assert raised.value.code == "registry_install_verification_failed"
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None
    assert stored.trusted is False
    assert stored.active is False


def test_revoking_trust_also_deactivates(session: Session, tmp_path: Path) -> None:
    install_id = _install(session, trusted=True, active=True)

    state = review_comfy_registry_install(
        session,
        install_id=install_id,
        trusted=False,
        custom_node_root=tmp_path,
        environment_root=tmp_path,
        media_worker_stopped=True,
    )

    assert state.trusted is False
    assert state.active is False
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None
    assert stored.review_json["trusted_by_local_user"] is False


async def test_activation_starts_media_only_after_verified_state_is_committed(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )
    starts = 0

    async def start_media() -> object:
        nonlocal starts
        starts += 1
        stored = session.get(ComfyRegistryInstall, install_id)
        assert stored is not None and stored.active is True
        return object()

    state = await activate_comfy_registry_install(
        session,
        install_id=install_id,
        custom_node_root=tmp_path,
        environment_root=tmp_path,
        media_worker_stopped=True,
        start_media=start_media,
    )

    assert starts == 1
    assert state.trusted is True
    assert state.active is True
    assert state.activated_at


async def test_activation_records_exact_bounded_runtime_files(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )

    async def start_media() -> object:
        styles = tmp_path / "lm-atelier-registry_example" / "styles"
        styles.mkdir(exist_ok=True)
        (styles / "defaults.json").write_bytes(b'{"style": "natural"}\n')
        return object()

    await activate_comfy_registry_install(
        session,
        install_id=install_id,
        custom_node_root=tmp_path,
        environment_root=tmp_path,
        media_worker_stopped=True,
        start_media=start_media,
    )

    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None
    assert stored.review_json["runtime_files"] == [
        {
            "path": "styles/defaults.json",
            "size": 21,
            "sha256": hashlib.sha256(b'{"style": "natural"}\n').hexdigest(),
        }
    ]


async def test_activation_restores_prior_runtime_after_unsafe_runtime_addition(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )
    observed: list[bool] = []

    async def start_media() -> object:
        stored = session.get(ComfyRegistryInstall, install_id)
        assert stored is not None
        observed.append(stored.active)
        if stored.active:
            (tmp_path / "lm-atelier-registry_example" / "added.py").write_text(
                "pass\n", encoding="utf-8"
            )
        return object()

    with pytest.raises(ComfyRegistryActivationError) as raised:
        await activate_comfy_registry_install(
            session,
            install_id=install_id,
            custom_node_root=tmp_path,
            environment_root=tmp_path,
            media_worker_stopped=True,
            start_media=start_media,
        )

    assert raised.value.code == "activation_runtime_files_failed"
    assert observed == [True, False]
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None and stored.active is False


async def test_untrusted_install_cannot_start_or_reach_runtime_verification(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session)

    def unexpected(*_args: object, **_kwargs: object) -> ComfyRegistryLaunchContract:
        raise AssertionError("untrusted activation must not reach verification")

    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        unexpected,
    )

    async def start_media() -> object:
        raise AssertionError("untrusted activation must not start media")

    with pytest.raises(ComfyRegistryActivationError) as raised:
        await activate_comfy_registry_install(
            session,
            install_id=install_id,
            custom_node_root=tmp_path,
            environment_root=tmp_path,
            media_worker_stopped=True,
            start_media=start_media,
        )

    assert raised.value.code == "registry_install_untrusted"
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None and stored.active is False


async def test_failed_activation_restores_runtime_with_package_inactive(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )
    observed: list[bool] = []

    async def start_media() -> object:
        stored = session.get(ComfyRegistryInstall, install_id)
        assert stored is not None
        observed.append(stored.active)
        if len(observed) == 1:
            raise RuntimeError("node failed")
        return object()

    with pytest.raises(ComfyRegistryActivationError) as raised:
        await activate_comfy_registry_install(
            session,
            install_id=install_id,
            custom_node_root=tmp_path,
            environment_root=tmp_path,
            media_worker_stopped=True,
            start_media=start_media,
        )

    assert raised.value.code == "activation_start_failed"
    assert observed == [True, False]
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None
    assert stored.active is False
    assert stored.review_json["activation_failure_code"] == "activation_start_failed"


async def test_failed_activation_and_failed_restore_remain_inactive(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )

    async def fail() -> object:
        raise RuntimeError("failed")

    with pytest.raises(ComfyRegistryActivationError) as raised:
        await activate_comfy_registry_install(
            session,
            install_id=install_id,
            custom_node_root=tmp_path,
            environment_root=tmp_path,
            media_worker_stopped=True,
            start_media=fail,
        )

    assert raised.value.code == "activation_restore_failed"
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None and stored.active is False


async def test_cancelled_activation_restores_runtime_then_propagates_cancel(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )
    observed: list[bool] = []

    async def start_media() -> object:
        stored = session.get(ComfyRegistryInstall, install_id)
        assert stored is not None
        observed.append(stored.active)
        if len(observed) == 1:
            raise asyncio.CancelledError
        return object()

    with pytest.raises(asyncio.CancelledError):
        await activate_comfy_registry_install(
            session,
            install_id=install_id,
            custom_node_root=tmp_path,
            environment_root=tmp_path,
            media_worker_stopped=True,
            start_media=start_media,
        )

    assert observed == [True, False]
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None and stored.active is False


async def test_deactivation_stays_safe_when_media_restart_fails(session: Session) -> None:
    install_id = _install(session, trusted=True, active=True)

    async def fail() -> object:
        raise RuntimeError("failed")

    with pytest.raises(ComfyRegistryActivationError) as raised:
        await deactivate_comfy_registry_install(
            session,
            install_id=install_id,
            media_worker_stopped=True,
            start_media=fail,
        )

    assert raised.value.code == "deactivation_restart_failed"
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None and stored.active is False


@pytest.mark.parametrize("operation", ["review", "activate", "deactivate"])
async def test_worker_must_be_stopped_before_install_lookup(
    session: Session,
    tmp_path: Path,
    operation: str,
) -> None:
    async def start_media() -> object:
        raise AssertionError("worker state refusal must not start media")

    with pytest.raises(ComfyRegistryActivationError) as raised:
        if operation == "review":
            review_comfy_registry_install(
                session,
                install_id="missing",
                trusted=True,
                custom_node_root=tmp_path,
                environment_root=tmp_path,
                media_worker_stopped=False,
            )
        elif operation == "activate":
            await activate_comfy_registry_install(
                session,
                install_id="missing",
                custom_node_root=tmp_path,
                environment_root=tmp_path,
                media_worker_stopped=False,
                start_media=start_media,
            )
        else:
            await deactivate_comfy_registry_install(
                session,
                install_id="missing",
                media_worker_stopped=False,
                start_media=start_media,
            )

    assert raised.value.code == "media_worker_running"


def _pending(session: Session, install_id: str) -> None:
    """Record the candidate preparation would have written.

    Activation reads this rather than being handed one, so the test drives the
    path a real trial takes: a caller has no way to name its own manifest hash
    or workflow.
    """
    install = session.get(ComfyRegistryInstall, install_id)
    assert install is not None
    install.review_json = record_pending_omission(
        install.review_json,
        manifest_sha256="a" * 64,
        omitted_declarations=("example @ git+https://github.com/owner/repo",),
        workflow_revision_id="revision-1",
        required_node_types=("ExampleNode",),
    )
    session.commit()


async def test_a_trial_activation_that_loads_the_required_nodes_records_its_proof(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )
    _pending(session, install_id)

    async def start_media() -> object:
        return object()

    async def read_inventory() -> frozenset[str]:
        return frozenset({"ExampleNode", "SaveImage"})

    await activate_comfy_registry_install(
        session,
        install_id=install_id,
        custom_node_root=tmp_path,
        environment_root=tmp_path,
        media_worker_stopped=True,
        start_media=start_media,
        read_node_inventory=read_inventory,
    )

    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None and stored.active is True
    assert stored.review_json["source_omission_proof"]["required_node_types"] == ["ExampleNode"]
    assert len(stored.review_json["source_omission_digest"]) == 64


async def test_a_started_runtime_missing_a_required_node_is_rolled_back(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A service that merely starts is not proof, and must not be left running."""
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )
    _pending(session, install_id)
    starts = 0

    async def start_media() -> object:
        nonlocal starts
        starts += 1
        return object()

    async def read_inventory() -> frozenset[str]:
        return frozenset({"SaveImage"})

    with pytest.raises(ComfyRegistryActivationError) as raised:
        await activate_comfy_registry_install(
            session,
            install_id=install_id,
            custom_node_root=tmp_path,
            environment_root=tmp_path,
            media_worker_stopped=True,
            start_media=start_media,
            read_node_inventory=read_inventory,
        )

    assert raised.value.code == "omission_required_nodes_missing"
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None and stored.active is False
    # Started for the trial, then started again without the package.
    assert starts == 2


async def test_a_trial_with_no_way_to_read_the_inventory_refuses_and_restores(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )
    _pending(session, install_id)

    async def start_media() -> object:
        return object()

    with pytest.raises(ComfyRegistryActivationError) as raised:
        await activate_comfy_registry_install(
            session,
            install_id=install_id,
            custom_node_root=tmp_path,
            environment_root=tmp_path,
            media_worker_stopped=True,
            start_media=start_media,
        )

    assert raised.value.code == "omission_unverifiable"
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None and stored.active is False


async def test_an_ordinary_activation_records_no_proof_at_all(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )

    async def start_media() -> object:
        return object()

    await activate_comfy_registry_install(
        session,
        install_id=install_id,
        custom_node_root=tmp_path,
        environment_root=tmp_path,
        media_worker_stopped=True,
        start_media=start_media,
    )

    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None
    # Explicitly cleared rather than merely absent: a spread that only added
    # the key would leave a previous activation's evidence standing here.
    assert stored.review_json["source_omission_proof"] is None
    assert stored.review_json["source_omission_digest"] is None


async def test_a_later_ordinary_activation_does_not_inherit_the_old_proof(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale evidence would claim a proof about a run that never happened.

    The trial proves one package against one workflow. Reactivating the same
    install afterwards, with no candidate pending, must not keep saying so.
    """
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )
    _pending(session, install_id)

    async def start_media() -> object:
        return object()

    async def read_inventory() -> frozenset[str]:
        return frozenset({"ExampleNode"})

    await activate_comfy_registry_install(
        session,
        install_id=install_id,
        custom_node_root=tmp_path,
        environment_root=tmp_path,
        media_worker_stopped=True,
        start_media=start_media,
        read_node_inventory=read_inventory,
    )
    proven = session.get(ComfyRegistryInstall, install_id)
    assert proven is not None and proven.review_json["source_omission_proof"] is not None

    # The candidate is spent; reactivating is now an ordinary activation.
    proven.review_json = {
        key: value for key, value in proven.review_json.items() if key != PENDING_KEY
    }
    session.commit()

    await activate_comfy_registry_install(
        session,
        install_id=install_id,
        custom_node_root=tmp_path,
        environment_root=tmp_path,
        media_worker_stopped=True,
        start_media=start_media,
        read_node_inventory=read_inventory,
    )

    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None
    assert stored.review_json["source_omission_proof"] is None
    assert stored.review_json["source_omission_digest"] is None


async def test_a_candidate_that_cannot_be_read_stops_the_activation_before_it_starts(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent and unreadable are different answers.

    Treating a corrupt record as "no candidate" made it an ordinary
    activation - the package would go active, the worker would start, and the
    omitted dependency would never be proven unnecessary. The record exists;
    that it cannot be read is a reason to stop, not to proceed.
    """
    install_id = _install(session, trusted=True)
    monkeypatch.setattr(
        activation_module,
        "trusted_comfy_registry_launch_contract",
        lambda current, **_kwargs: _verified_contract(current, install_id),
    )
    corrupt = session.get(ComfyRegistryInstall, install_id)
    assert corrupt is not None
    corrupt.review_json = {**corrupt.review_json, PENDING_KEY: {"omitted_declarations": "one"}}
    session.commit()

    async def start_media() -> object:
        raise AssertionError("the worker must not start for an unreadable candidate")

    with pytest.raises(ComfyRegistryActivationError) as raised:
        await activate_comfy_registry_install(
            session,
            install_id=install_id,
            custom_node_root=tmp_path,
            environment_root=tmp_path,
            media_worker_stopped=True,
            start_media=start_media,
            read_node_inventory=lambda: _never_read(),
        )

    assert raised.value.code == "omission_candidate_unreadable"
    stored = session.get(ComfyRegistryInstall, install_id)
    assert stored is not None and stored.active is False


async def _never_read() -> frozenset[str]:
    raise AssertionError("the inventory must not be read for an unreadable candidate")
