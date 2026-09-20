from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session

from .accepted_turn_context import accepted_context
from .capability_evidence import current_capability_evidence, evidence_input_modalities
from .config import Settings
from .model_updates import installed_civitai_identities
from .models import (
    Chat,
    Job,
    ModelInstall,
    ModelProfile,
    Run,
    RunContextSnapshot,
    WorkflowActivation,
    WorkflowDependencyBinding,
)
from .workflow_compatibility import ensure_legacy_profile_workflow

if TYPE_CHECKING:
    from .runtime_provisioning import RuntimeProvisioner


class ProfileModelUpdateError(ValueError):
    def __init__(self, code: str, detail: str, status: int = 409) -> None:
        super().__init__(detail)
        self.code = code
        self.status = status


def switch_profile_model(
    session: Session,
    profile_id: str,
    *,
    expected_install_id: str,
    download_job_id: str,
    settings: Settings,
    runtimes: RuntimeProvisioner | None,
) -> ModelProfile:
    """Apply one explicit update without retiring the previous installation.

    Reserve the database writer before reading the profile and evidence so
    another switch, retirement or queue claim cannot change this decision.
    The download result supplies the target; the caller cannot substitute one.
    """
    if session.in_transaction():
        raise ProfileModelUpdateError("profile-update-conflict", "Refresh the profile and retry.")
    try:
        session.execute(text("BEGIN IMMEDIATE"))
        profile = session.get(ModelProfile, profile_id)
        if profile is None:
            raise ProfileModelUpdateError("profile-not-found", "Profile not found.", 404)
        if profile.model_install_id != expected_install_id:
            raise ProfileModelUpdateError(
                "profile-update-conflict",
                "This profile changed. Refresh it before switching models.",
            )
        job = session.get(Job, download_job_id)
        if job is None or job.kind != "download" or job.status != "complete":
            raise ProfileModelUpdateError(
                "profile-update-not-installed",
                "The update must finish installing before switching.",
            )
        target_id = job.result_json.get("model_install_id")
        previous = session.get(ModelInstall, expected_install_id)
        target = session.get(ModelInstall, target_id) if isinstance(target_id, str) else None
        if (
            previous is None
            or not previous.active
            or target is None
            or not target.active
            or target.id == previous.id
            or previous.role != profile.role
            or target.role != profile.role
            or previous.engine != profile.engine
            or target.engine != profile.engine
            or job.payload_json.get("remote_id") != target.manifest_json.get("remote_id")
            or job.payload_json.get("revision") != target.manifest_json.get("revision")
        ):
            raise ProfileModelUpdateError(
                "profile-update-install-invalid",
                "The installed update no longer matches this profile.",
            )
        identities = {
            item.install_id: item
            for item in installed_civitai_identities(session)
            if item.kind == "checkpoint"
        }
        old_identity = identities.get(previous.id)
        new_identity = identities.get(target.id)
        if (
            old_identity is None
            or new_identity is None
            or old_identity.model_id != new_identity.model_id
            or old_identity.version_id == new_identity.version_id
        ):
            raise ProfileModelUpdateError(
                "profile-update-source-mismatch",
                "Choose another verified version of the same model.",
            )
        old_family = previous.manifest_json.get("family")
        if old_family and target.manifest_json.get("family") != old_family:
            raise ProfileModelUpdateError(
                "profile-update-family-mismatch",
                "This version uses a different model family. Create a separate profile for it.",
            )
        evidence = current_capability_evidence(session, target, settings, runtimes)
        if evidence is None:
            raise ProfileModelUpdateError(
                "profile-update-unverified", "The update must pass its current runtime checks."
            )
        old_evidence = current_capability_evidence(session, previous, settings, runtimes)
        used_for_vision = (
            session.scalar(
                select(Chat.id)
                .where(
                    Chat.active_vision_profile_id == profile.id,
                )
                .limit(1)
            )
            is not None
        )
        if (
            used_for_vision or "image" in evidence_input_modalities(old_evidence)
        ) and "image" not in evidence_input_modalities(evidence):
            raise ProfileModelUpdateError(
                "profile-update-vision-unverified",
                "This profile needs verified image input support.",
            )
        pinned = session.scalar(
            select(WorkflowDependencyBinding.id)
            .join(
                WorkflowActivation,
                WorkflowActivation.id == WorkflowDependencyBinding.workflow_activation_id,
            )
            .where(
                WorkflowDependencyBinding.model_profile_id == profile.id,
                WorkflowActivation.is_active.is_(True),
            )
            .limit(1)
        )
        if pinned is not None:
            raise ProfileModelUpdateError(
                "profile-update-workflow-bound",
                "A workflow pins this profile. Review its model binding before switching versions.",
            )
        _require_frozen_runs(session, profile.id)
        profile.model_install_id = target.id
        ensure_legacy_profile_workflow(session, profile)
        session.commit()
        session.refresh(profile)
        return profile
    except Exception:
        session.rollback()
        raise


def _require_frozen_runs(session: Session, profile_id: str) -> None:
    runs = (
        session.scalars(
            select(Run)
            .join(Job, Job.run_id == Run.id)
            .outerjoin(RunContextSnapshot, RunContextSnapshot.run_id == Run.id)
            .where(
                Job.status.in_(["queued", "running", "paused"]),
                or_(
                    Run.profile_id == profile_id,
                    Run.vision_profile_id == profile_id,
                    RunContextSnapshot.payload_json["verification_profile"]["id"].as_string()
                    == profile_id,
                ),
            )
        )
        .unique()
        .all()
    )
    for run in runs:
        try:
            snapshot = accepted_context(session, run)
        except ValueError:
            snapshot = None
        if snapshot is None or not any(
            frozen is not None and frozen.id == profile_id and frozen.install is not None
            for frozen in (snapshot.profile, snapshot.vision_profile, snapshot.verification_profile)
        ):
            raise ProfileModelUpdateError(
                "profile-update-busy",
                "An earlier job still uses this profile without a saved model configuration. "
                "Wait for it to finish before switching.",
            )
