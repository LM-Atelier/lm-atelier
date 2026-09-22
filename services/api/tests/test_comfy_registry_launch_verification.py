from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from run_waits import wait_until
from sqlalchemy import delete
from sqlalchemy.orm import Session
from test_comfy_registry_reviewed_wheel_staging import (
    source_review_context as source_review_context,
)
from test_workflow_reviewed_package_plan import _close, _inputs, _prepare

from local_lm import comfy_registry_installs as install_module
from local_lm import comfy_registry_launch_verification as launch_module
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_activation import (
    ComfyRegistryActivationError,
    record_registry_policy_trust,
    review_comfy_registry_install,
)
from local_lm.comfy_registry_installs import ComfyRegistryInstallError
from local_lm.comfy_registry_launch_verification import (
    VerifiedComfyRegistryLaunch,
    verify_comfy_registry_launch,
)
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from local_lm.models import ComfyRegistryInstall, ComfyRegistrySourceArtifactReview
from local_lm.workflow_package_execution_plan import plan_workflow_package_execution


async def _prepared(
    source_context: tuple[Session, ArtifactStore], root: Path, *, inactive: bool = False
) -> dict[str, Any]:
    inputs = _inputs(source_context, root, commit=False, inactive=inactive)
    try:
        plan = await plan_workflow_package_execution(**inputs["plan_arguments"])
        inputs["prepared"] = await _prepare(inputs, plan)
    except BaseException:
        await _close(inputs)
        raise
    inputs["verify"] = dict(
        include_active=True,
        custom_node_root=inputs["context"].custom_node_root,
        environment_root=registry_wheel_environment_root(inputs["context"].state_root),
        reviewed_inputs=ComfyRegistryReviewedInputContext(
            inputs["factory"], source_context[1], inputs["environment"], ("py3-none-any",)
        ),
    )
    inputs["grant"] = dict(
        install_id=inputs["prepared"].install_id,
        custom_node_root=inputs["verify"]["custom_node_root"],
        environment_root=inputs["verify"]["environment_root"],
        media_worker_stopped=True,
    )
    return inputs


def _verify(inputs: dict[str, Any]) -> VerifiedComfyRegistryLaunch:
    return verify_comfy_registry_launch(
        inputs["factory"], (inputs["prepared"].install_id,), **inputs["verify"]
    )


def _add_active(session: Session, prepared: ComfyRegistryInstall) -> None:
    values = {
        column.name: getattr(prepared, column.name)
        for column in ComfyRegistryInstall.__table__.columns
        if column.name not in {"created_at", "updated_at"}
    }
    values.update(
        id="other-registry-install",
        package_id="other-package",
        registry_record_id="other-record",
        installed_path=prepared.installed_path + "-other",
        trusted=True,
        active=True,
    )
    session.add(ComfyRegistryInstall(**values))


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("inactive", [False, True])
async def test_trust_uses_detached_verification_without_reading_files_under_its_writer(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: bool,
    inactive: bool,
) -> None:
    inputs = await _prepared(source_review_context, tmp_path, inactive=inactive)
    try:
        proof = await asyncio.to_thread(_verify, inputs)
        reserved: list[bool] = []
        current = VerifiedComfyRegistryLaunch.require_current

        def require_current(self: VerifiedComfyRegistryLaunch, session: Session) -> None:
            current(self, session)
            driver = session.connection().connection.driver_connection
            reserved.append(bool(getattr(driver, "in_transaction", False)))

        def no_file_reads(*args: object, **kwargs: object) -> None:
            pytest.fail("Trust must use the detached file verification")

        monkeypatch.setattr(VerifiedComfyRegistryLaunch, "require_current", require_current)
        monkeypatch.setattr(install_module, "verify_staged_comfy_registry_archive", no_file_reads)
        monkeypatch.setattr(
            install_module, "verify_comfy_registry_wheel_environment", no_file_reads
        )
        monkeypatch.setattr(ComfyRegistryReviewedInputContext, "verified_authority", no_file_reads)
        with inputs["factory"]() as session:
            if policy:
                result = record_registry_policy_trust(
                    session,
                    **inputs["grant"],
                    resolution=inputs["selected"],
                    expected_archive_sha256=inputs["prepared"].archive_sha256,
                    expected_manifest_sha256=inputs["prepared"].manifest_sha256,
                    verified_launch=proof,
                )
            else:
                result = review_comfy_registry_install(
                    session, **inputs["grant"], trusted=True, verified_launch=proof
                )
            assert result.trusted and not result.active
            assert reserved == [True]
            row = session.get(ComfyRegistryInstall, result.install_id)
            assert row is not None and "reviewed_wheel_closure" in row.review_json
            with pytest.raises(ComfyRegistryActivationError):
                review_comfy_registry_install(
                    session, **inputs["grant"], trusted=True, verified_launch=proof
                )
            session.refresh(row)
            assert row.trusted and not row.active
    finally:
        await _close(inputs)


@pytest.mark.parametrize("change", ["review", "install", "active_set"])
async def test_another_writer_can_commit_during_verification_and_invalidate_the_grant(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    inputs = await _prepared(source_review_context, tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = install_module._verified_comfy_registry_launch_contract
    task: asyncio.Task[VerifiedComfyRegistryLaunch] | None = None

    def paused(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        entered.set()
        if not release.wait(30):
            raise AssertionError("Verification was not released")
        return result

    async def entered_read() -> bool:
        return entered.is_set()

    try:
        monkeypatch.setattr(launch_module, "_verified_comfy_registry_launch_contract", paused)
        with inputs["factory"]() as session:
            cached = session.get(ComfyRegistryInstall, inputs["prepared"].install_id)
            assert cached is not None
            task = asyncio.create_task(asyncio.to_thread(_verify, inputs))
            await wait_until(entered_read, bool, what="detached package verification")
            with inputs["factory"]() as writer:
                writer.connection().exec_driver_sql("PRAGMA busy_timeout=100")
                if change == "review":
                    writer.execute(delete(ComfyRegistrySourceArtifactReview))
                elif change == "install":
                    row = writer.get(ComfyRegistryInstall, cached.id)
                    assert row is not None
                    row.pip_dependencies_json = ["alpha==1.0"]
                else:
                    _add_active(writer, cached)
                writer.commit()
            release.set()
            proof = await task
            assert session.get(ComfyRegistryInstall, cached.id) is cached
            with pytest.raises(ComfyRegistryActivationError) as error:
                review_comfy_registry_install(
                    session, **inputs["grant"], trusted=True, verified_launch=proof
                )
            assert error.value.code == "registry_install_verification_failed"
            session.refresh(cached)
            assert not cached.trusted and not cached.active
    finally:
        release.set()
        if task is not None:
            await task
        await _close(inputs)


@pytest.mark.parametrize("change", ["root", "selection"])
async def test_trust_refuses_verification_for_a_different_scope(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    change: str,
) -> None:
    inputs = await _prepared(source_review_context, tmp_path)
    try:
        proof = await asyncio.to_thread(_verify, inputs)
        if change == "root":
            inputs["grant"]["custom_node_root"] = tmp_path / "other"
        else:
            proof = replace(proof, include_active=False)
        with inputs["factory"]() as session:
            with pytest.raises(ComfyRegistryActivationError):
                review_comfy_registry_install(
                    session, **inputs["grant"], trusted=True, verified_launch=proof
                )
            row = session.get(ComfyRegistryInstall, inputs["prepared"].install_id)
            assert row is not None and not row.trusted and not row.active
    finally:
        await _close(inputs)


async def test_trust_verification_includes_files_of_previously_active_packages(
    source_review_context: tuple[Session, ArtifactStore], tmp_path: Path
) -> None:
    inputs = await _prepared(source_review_context, tmp_path)
    try:
        with inputs["factory"]() as session:
            row = session.get(ComfyRegistryInstall, inputs["prepared"].install_id)
            assert row is not None
            _add_active(session, row)
            session.commit()
        with pytest.raises(ComfyRegistryInstallError, match="Registry node files are missing"):
            await asyncio.to_thread(_verify, inputs)
        with inputs["factory"]() as session:
            row = session.get(ComfyRegistryInstall, inputs["prepared"].install_id)
            assert row is not None and not row.trusted and not row.active
    finally:
        await _close(inputs)
