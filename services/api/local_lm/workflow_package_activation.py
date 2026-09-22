"""Finish a verified Registry preparation with its existing policy and activation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session

from .comfy_registry import ComfyNodeResolution
from .comfy_registry_activation import (
    ComfyRegistryActivationError,
    activate_comfy_registry_install,
    record_registry_policy_trust,
)
from .comfy_registry_installs import ComfyRegistryInstallError
from .comfy_registry_lifecycle import ComfyRegistryPreparation
from .comfy_registry_paths import registry_wheel_environment_root
from .models import ComfyRegistryInstall
from .processes import ProcessSupervisor
from .registry_trust_policy import decide_registry_trust
from .workflow_package_preparation import PreparationContext, WorkflowPackagePreparationError


@dataclass(frozen=True)
class WorkflowPackageActivation:
    state: Literal["active", "review_required"]
    reason: str
    explanation: str
    notices: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason": self.reason,
            "explanation": self.explanation,
            "notices": list(self.notices),
        }


def media_worker_stopped(processes: ProcessSupervisor) -> bool:
    media = next((status for status in processes.statuses() if status.name == "media"), None)
    return media is None or (not media.running and media.state != "starting")


def _prepared_install(
    session: Session, preparation: ComfyRegistryPreparation
) -> ComfyRegistryInstall:
    install = session.get(ComfyRegistryInstall, preparation.install_id)
    if install is None or (
        install.archive_sha256,
        install.manifest_sha256,
        install.installed_path,
        install.wheel_environment_path,
        install.wheel_closure_sha256,
        install.wheel_environment_sha256,
    ) != (
        preparation.archive_sha256,
        preparation.manifest_sha256,
        preparation.installed_path,
        preparation.wheel_environment_path,
        preparation.wheel_closure_sha256,
        preparation.wheel_environment_sha256,
    ):
        raise WorkflowPackagePreparationError(
            "registry_policy_identity_mismatch",
            "The prepared extension changed before its installation could finish.",
        )
    return install


async def activate_prepared_workflow_package(
    session: Session,
    preparation: ComfyRegistryPreparation,
    resolution: ComfyNodeResolution,
    *,
    context: PreparationContext,
    processes: ProcessSupervisor,
    session_factory: Callable[[], Session],
) -> WorkflowPackageActivation:
    """Use the verified source and bytes, then apply the ordinary launch checks."""
    install = _prepared_install(session, preparation)
    decision = decide_registry_trust(resolution, install)
    if decision.outcome == "pause":
        return WorkflowPackageActivation(
            "review_required", decision.reason, decision.explanation, decision.notices
        )
    if decision.outcome == "refuse":
        raise WorkflowPackagePreparationError(
            "registry_policy_" + decision.reason, decision.explanation
        )
    environment_root = registry_wheel_environment_root(context.state_root)
    target = context.verification_target(session_factory, processes.settings)
    try:
        verified = await target.verify((preparation.install_id,))
        session.expire_all()
        _prepared_install(session, preparation)
        record_registry_policy_trust(
            session,
            install_id=preparation.install_id,
            resolution=resolution,
            expected_archive_sha256=preparation.archive_sha256,
            expected_manifest_sha256=preparation.manifest_sha256,
            custom_node_root=context.custom_node_root,
            environment_root=environment_root,
            media_worker_stopped=media_worker_stopped(processes),
            verified_launch=verified,
        )
        await activate_comfy_registry_install(
            session,
            install_id=preparation.install_id,
            custom_node_root=context.custom_node_root,
            environment_root=environment_root,
            media_worker_stopped=media_worker_stopped(processes),
            start_media=processes.start_media,
            read_node_inventory=processes.comfy_node_inventory,
            verification_target=target,
        )
    except ComfyRegistryInstallError as exc:
        session.rollback()
        raise WorkflowPackagePreparationError(
            "registry_install_verification_failed",
            "The extension could not complete its verified runtime setup.",
        ) from exc
    except ComfyRegistryActivationError as exc:
        raise WorkflowPackagePreparationError(
            exc.code, "The extension could not complete its verified runtime setup."
        ) from exc
    return WorkflowPackageActivation("active", decision.reason, "", decision.notices)
