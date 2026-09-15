"""Only exact verified grants let an accepted extension set proceed without another prompt."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event, select
from sqlalchemy.orm import Session
from test_workflow_offer_packages import _prepare, _setup
from test_workflow_package_execution_plan import _inputs
from test_workflow_package_preparation import _Registry

from local_lm.comfy_registry_activation import review_comfy_registry_install
from local_lm.comfy_registry_lifecycle import ComfyRegistryPreparation
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_workflow_packages import WorkflowPackageRequirement
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall, WorkflowInstallOffer
from local_lm.workflow_offer_packages import (
    accepted_workflow_offer_packages,
    record_workflow_offer_package,
)

pytestmark = pytest.mark.asyncio


def _package(index: int, *, commit: bool = False, warnings: tuple[str, ...] = ()) -> dict[str, Any]:
    inputs = _inputs(commit=commit)
    selected = replace(
        inputs["selected"],
        package_id=f"example-pack-{index}",
        repository_url=f"https://github.com/example/example-pack-{index}.git",
        registry_record_id=None if commit else f"record-{index}",
        download_url=None if commit else f"https://cdn.comfy.org/example/pack-{index}/node.zip",
        node_types=(f"ExampleNode{index}",),
        warnings=warnings,
    )
    inputs.update(
        selected=selected,
        registry_client=_Registry(selected),
        requirement=WorkflowPackageRequirement(
            selected.package_id, (selected.declared_version,), selected.node_types, False
        ),
    )
    return inputs


@pytest.mark.parametrize(
    "mode",
    [
        "registry",
        "deprecated",
        "git",
        "git-approved",
        "git-approved-code-changed",
        "git-stale-refusal",
        "git-legacy",
        "git-refused",
        "git-flag-only",
        "unknown-warning",
        "warning-approved",
        "previously-refused",
        "changed-code",
        "changed-plan",
        "changed-offer",
        "missing-result",
        "running",
        "commit-failure",
    ],
)
async def test_all_accepted_extensions_need_exact_trust_before_any_new_grant_commits(
    client: AsyncClient,
    tmp_path: Path,
    mode: str,
) -> None:
    from local_lm.workflow_source_extension_trust import trust_workflow_source_extensions

    git = mode.startswith("git")
    warning = (
        ("deprecated_version",)
        if mode == "deprecated"
        else ("future_warning",)
        if mode in {"unknown-warning", "warning-approved"}
        else ()
    )
    inputs = [_package(1), _package(2, commit=git, warnings=warning)]
    _first, context, offer_id, saved = await _setup(tmp_path, package_inputs=inputs)
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        packages = accepted_workflow_offer_packages(session, offer, saved)
        jobs = {package.link.package_id: package.job.id for package in packages}
        for package in packages:
            package.job.status = "running"
        session.commit()

    prepared = []
    for current in inputs:
        package_id = current["selected"].package_id

        def record(
            session: Session, result: ComfyRegistryPreparation, *, package_id: str = package_id
        ) -> None:
            offer = session.get(WorkflowInstallOffer, offer_id)
            assert offer is not None
            record_workflow_offer_package(
                session,
                offer,
                saved,
                package_id=package_id,
                job_id=jobs[package_id],
                preparation=result,
            )

        prepared.append(await _prepare(current, context, saved, record))

    if mode in {
        "git-approved",
        "git-approved-code-changed",
        "git-stale-refusal",
        "git-legacy",
        "git-refused",
        "warning-approved",
        "previously-refused",
    }:
        with SessionLocal() as session:
            review_comfy_registry_install(
                session,
                install_id=prepared[1].install_id,
                trusted=mode not in {"git-refused", "previously-refused"},
                custom_node_root=context.custom_node_root,
                environment_root=registry_wheel_environment_root(context.state_root),
                media_worker_stopped=True,
            )
            if mode == "git-legacy":
                row = session.get(ComfyRegistryInstall, prepared[1].install_id)
                assert row is not None
                row.review_json = {
                    k: v for k, v in row.review_json.items() if k != "trust_authority"
                }
                session.commit()
    with SessionLocal() as session:
        second = session.get(ComfyRegistryInstall, prepared[1].install_id)
        assert second is not None
        if mode == "git-flag-only":
            second.trusted = True
        if mode == "changed-plan":
            second.repository_url = "https://github.com/example/different.git"
        if mode == "changed-offer":
            offer = session.get(WorkflowInstallOffer, offer_id)
            assert offer is not None
            offer.status = "invalidated"
        if mode == "missing-result":
            session.delete(second)
        session.commit()
    if mode in {"changed-code", "git-approved-code-changed"}:
        (context.custom_node_root / prepared[1].installed_path / "__init__.py").write_text(
            "raise RuntimeError('Changed inert fixture')\n", encoding="utf-8"
        )

    with SessionLocal() as session:
        before = {
            row.id: (row.trusted, row.active, dict(row.review_json))
            for row in session.scalars(select(ComfyRegistryInstall))
        }

    def factory() -> Session:
        nonlocal before
        session = SessionLocal()
        if mode == "git-stale-refusal":
            session.info["previous_grant"] = session.get(
                ComfyRegistryInstall, prepared[1].install_id
            )
            with SessionLocal() as writer:
                review_comfy_registry_install(
                    writer,
                    install_id=prepared[1].install_id,
                    trusted=False,
                    custom_node_root=context.custom_node_root,
                    environment_root=registry_wheel_environment_root(context.state_root),
                    media_worker_stopped=True,
                )
                before = {
                    row.id: (row.trusted, row.active, dict(row.review_json))
                    for row in writer.scalars(select(ComfyRegistryInstall))
                }
        if mode == "commit-failure":

            def refuse_commit(_session: Session) -> None:
                with SessionLocal() as reader:
                    assert {
                        row.id: (row.trusted, row.active, dict(row.review_json))
                        for row in reader.scalars(select(ComfyRegistryInstall))
                    } == before
                raise RuntimeError("Neutral commit failure")

            event.listen(session, "before_commit", refuse_commit)
        return session

    def run() -> Any:
        return trust_workflow_source_extensions(
            factory,
            offer_id,
            context=context,
            media_worker_stopped=mode != "running",
        )

    success = {"registry", "deprecated", "git-approved", "git-legacy", "warning-approved"}
    failure = {
        "changed-code",
        "git-approved-code-changed",
        "changed-plan",
        "changed-offer",
        "missing-result",
        "running",
        "commit-failure",
    }
    if mode in failure:
        with pytest.raises((ValueError, RuntimeError)):
            run()
    else:
        result = run()
        assert result.state == ("ready" if mode in success else "review_required")
        assert len(result.packages) == 2
        if mode == "deprecated":
            assert result.packages[1].decision.notices == ("deprecated_version",)
    with SessionLocal() as session:
        after = {
            row.id: (row.trusted, row.active, dict(row.review_json))
            for row in session.scalars(select(ComfyRegistryInstall))
        }
        if mode in success:
            assert all(trusted and not active for trusted, active, _review in after.values())
            assert (
                after[prepared[0].install_id][2]["trust_authority"] == "registry-existing-review-v1"
            )
            if mode in {"git-approved", "git-legacy", "warning-approved"}:
                assert after[prepared[1].install_id] == before[prepared[1].install_id]
        else:
            assert after == before
    if mode in success:
        assert run().state == "ready"
        with SessionLocal() as session:
            assert {
                row.id: (row.trusted, row.active, dict(row.review_json))
                for row in session.scalars(select(ComfyRegistryInstall))
            } == after
