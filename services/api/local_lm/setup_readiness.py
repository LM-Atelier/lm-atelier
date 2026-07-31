from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .capability_evidence import current_capability_evidence
from .config import Settings
from .model_planner import revision_accepts_install
from .models import (
    Job,
    ModelCapabilityEvidence,
    ModelInstall,
    ModelProfile,
    WorkflowDefinition,
    WorkflowRevision,
)
from .schemas import (
    RuntimeStatus,
    SetupReadinessCheck,
    SetupReadinessReport,
    SetupRoleReadiness,
    WorkerStatus,
)
from .setup_verification import current_setup_verification

if TYPE_CHECKING:
    from .runtime_provisioning import RuntimeProvisioner

Role = Literal["chat", "image", "video"]
CheckStatus = Literal["pass", "pending", "fail"]

_ACTIVE_DOWNLOAD_STATES = {"queued", "running", "paused", "interrupted"}
MEDIA_OPERATIONS_BY_ROLE: dict[Role, set[str]] = {
    "chat": set(),
    "image": {"text_to_image", "image_to_image"},
    "video": {"text_to_video", "image_to_video"},
}


def setup_readiness_report(
    session: Session,
    settings: Settings,
    runtimes: RuntimeProvisioner,
    workers: Sequence[WorkerStatus],
) -> SetupReadinessReport:
    """Summarize prompt-free setup state without reading conversation or media data."""

    worker_by_name: dict[str, WorkerStatus] = {worker.name: worker for worker in workers}
    runtime_by_engine: dict[str, RuntimeStatus] = {
        status.engine: status for status in runtimes.statuses()
    }
    roles = [
        _role_readiness(
            session,
            settings,
            runtimes,
            runtime_by_engine,
            worker_by_name,
            role,
        )
        for role in ("chat", "image", "video")
    ]
    state: Literal["ready", "in_progress", "action_required"]
    if any(role.state == "action_required" for role in roles):
        state = "action_required"
    elif any(role.state == "in_progress" for role in roles):
        state = "in_progress"
    else:
        state = "ready"
    return SetupReadinessReport(state=state, roles=roles)


def _role_readiness(
    session: Session,
    settings: Settings,
    runtimes: RuntimeProvisioner,
    runtime_by_engine: Mapping[str, RuntimeStatus],
    worker_by_name: Mapping[str, WorkerStatus],
    role: Role,
) -> SetupRoleReadiness:
    checks: list[SetupReadinessCheck] = []
    installs = list(
        session.scalars(
            select(ModelInstall)
            .where(
                ModelInstall.role == role,
                ModelInstall.active.is_(True),
            )
            .order_by(ModelInstall.updated_at.desc(), ModelInstall.id)
        ).all()
    )
    profiles = (
        list(
            session.scalars(
                select(ModelProfile)
                .where(
                    ModelProfile.role == role,
                    ModelProfile.model_install_id.in_([install.id for install in installs]),
                )
                .order_by(
                    ModelProfile.is_default.desc(),
                    ModelProfile.updated_at.desc(),
                    ModelProfile.id,
                )
            ).all()
        )
        if installs
        else []
    )
    current_evidence = {
        install.id: current_capability_evidence(session, install, settings, runtimes)
        for install in installs
    }
    worker = worker_by_name.get("chat" if role == "chat" else "media")
    install = _select_install(installs, profiles, current_evidence, worker)

    if not install:
        # A model cannot help when its runtime can never run here, and a model is
        # tens of gigabytes. Report the runtime first so the machine is ruled in
        # or out before anything is downloaded.
        expected_runtime = runtime_by_engine.get(_expected_engine(settings, role))
        if expected_runtime:
            runtime_check = _runtime_check(expected_runtime)
            if runtime_check.status != "pass":
                checks.append(runtime_check)
                return _role_result(role, checks, engine=expected_runtime.engine)
        download = _latest_download_for_role(session, role)
        if download and download.status in _ACTIVE_DOWNLOAD_STATES:
            checks.append(
                _check(
                    "install_in_progress",
                    "pending",
                    "The model is being installed.",
                    "wait_for_install",
                )
            )
        elif download and download.status == "failed":
            checks.append(
                _check(
                    "install_failed",
                    "fail",
                    "The last model installation failed.",
                    "retry_install",
                )
            )
        else:
            checks.append(
                _check(
                    "model_missing",
                    "fail",
                    "No active model is installed for this role.",
                    "select_model",
                )
            )
        download_engine = (
            str(download.payload_json.get("engine"))
            if download and isinstance(download.payload_json.get("engine"), str)
            else None
        )
        return _role_result(
            role,
            checks,
            engine=download_engine,
            job_id=download.id if download else None,
        )

    checks.append(_check("model_ready", "pass", "An active model is installed."))
    if install.compatibility == "unsupported":
        checks.append(
            _check(
                "model_unsupported",
                "fail",
                "The active model is not compatible with this setup.",
                "select_model",
            )
        )
        return _role_result(role, checks, engine=install.engine, install_id=install.id)

    runtime = runtime_by_engine.get(install.engine)
    if runtime:
        runtime_check = _runtime_check(runtime)
        checks.append(runtime_check)
        if runtime_check.status != "pass":
            return _role_result(role, checks, engine=install.engine, install_id=install.id)
    else:
        checks.append(
            _check(
                "runtime_external",
                "pass",
                "This model uses the configured external engine.",
            )
        )

    evidence = current_evidence[install.id]
    if not evidence:
        latest = session.scalar(
            select(ModelCapabilityEvidence)
            .where(ModelCapabilityEvidence.model_install_id == install.id)
            .order_by(ModelCapabilityEvidence.probed_at.desc(), ModelCapabilityEvidence.id)
            .limit(1)
        )
        if latest and latest.result != "ready":
            checks.append(
                _check(
                    "activation_failed",
                    "fail",
                    "The model did not pass its activation probe.",
                    "activate_model",
                )
            )
        elif latest:
            checks.append(
                _check(
                    "activation_stale",
                    "fail",
                    "The model must be rechecked for the current runtime and hardware.",
                    "activate_model",
                )
            )
        else:
            checks.append(
                _check(
                    "activation_required",
                    "fail",
                    "The model has not passed an activation probe.",
                    "activate_model",
                )
            )
        return _role_result(role, checks, engine=install.engine, install_id=install.id)
    checks.append(
        _check(
            "activation_ready",
            "pass",
            "The model passed the current bounded activation probe.",
        )
    )

    profile = _select_profile(profiles, install.id, worker)
    if not profile or profile.engine != install.engine:
        checks.append(
            _check(
                "profile_missing",
                "fail",
                "No usable profile is bound to this model.",
                "create_profile",
            )
        )
        return _role_result(role, checks, engine=install.engine, install_id=install.id)
    checks.append(_check("profile_ready", "pass", "A usable model profile is available."))

    workflow: WorkflowRevision | None = None
    if role != "chat":
        workflow, workflow_check = _workflow_check(session, role, install)
        checks.append(workflow_check)
        if workflow_check.status != "pass":
            return _role_result(
                role,
                checks,
                engine=install.engine,
                install_id=install.id,
                profile_id=profile.id,
            )

    worker_check = _worker_check(
        worker,
        expected_profile_id=profile.id if role == "chat" else None,
    )
    checks.append(worker_check)
    verification = None
    if worker_check.status == "pass":
        verification = current_setup_verification(
            session,
            role,
            install,
            profile,
            workflow,
            evidence,
        )
        if verification is None:
            checks.append(
                _check(
                    "generation_verification_required",
                    "fail",
                    "Run one quick local generation test.",
                    "verify_generation",
                )
            )
        elif verification.state in {"queued", "running"}:
            checks.append(
                _check(
                    "generation_verification_running",
                    "pending",
                    "The local generation test is running.",
                    "wait_for_verification",
                )
            )
        elif verification.state == "ready":
            checks.append(
                _check(
                    "generation_verified",
                    "pass",
                    "A local generation completed with this setup.",
                )
            )
        else:
            checks.append(
                _check(
                    "generation_verification_failed",
                    "fail",
                    "The local generation test did not complete.",
                    "verify_generation",
                )
            )
    return _role_result(
        role,
        checks,
        engine=install.engine,
        job_id=verification.job_id if verification else None,
        verification_id=verification.id if verification else None,
        install_id=install.id,
        profile_id=profile.id,
        workflow_revision_id=workflow.id if workflow else None,
    )


def _select_install(
    installs: Sequence[ModelInstall],
    profiles: Sequence[ModelProfile],
    evidence: dict[str, ModelCapabilityEvidence | None],
    worker: WorkerStatus | None,
) -> ModelInstall | None:
    by_id = {install.id: install for install in installs}
    preferred_ids: list[str] = []
    if worker and worker.profile_id:
        worker_profile = next(
            (profile for profile in profiles if profile.id == worker.profile_id),
            None,
        )
        if worker_profile and worker_profile.model_install_id:
            preferred_ids.append(worker_profile.model_install_id)
    preferred_ids.extend(
        profile.model_install_id
        for profile in profiles
        if profile.is_default and profile.model_install_id
    )
    preferred_ids.extend(install.id for install in installs if evidence[install.id])
    preferred_ids.extend(
        profile.model_install_id for profile in profiles if profile.model_install_id
    )
    preferred_ids.extend(install.id for install in installs)
    for install_id in preferred_ids:
        install = by_id.get(install_id)
        if install and evidence[install.id]:
            return install
    return next((by_id[install_id] for install_id in preferred_ids if install_id in by_id), None)


def _select_profile(
    profiles: Sequence[ModelProfile],
    install_id: str,
    worker: WorkerStatus | None,
) -> ModelProfile | None:
    matching = [profile for profile in profiles if profile.model_install_id == install_id]
    if worker and worker.profile_id:
        loaded = next((profile for profile in matching if profile.id == worker.profile_id), None)
        if loaded:
            return loaded
    return matching[0] if matching else None


def _latest_download_for_role(session: Session, role: Role) -> Job | None:
    jobs = session.scalars(
        select(Job).where(Job.kind == "download").order_by(Job.created_at.desc(), Job.id)
    ).all()
    return next(
        (
            job
            for job in jobs
            if isinstance(job.payload_json, dict) and job.payload_json.get("role") == role
        ),
        None,
    )


def _expected_engine(settings: Settings, role: Role) -> str:
    """The engine a role will use once a model is installed."""

    return settings.chat_engine if role == "chat" else settings.media_engine


def _runtime_check(runtime: RuntimeStatus) -> SetupReadinessCheck:
    if runtime.state == "ready":
        return _check("runtime_ready", "pass", "The required runtime is ready.")
    if runtime.state == "installing":
        return _check(
            "runtime_installing",
            "pending",
            "The required runtime is being installed.",
            "wait_for_runtime",
        )
    if runtime.state == "failed":
        return _check(
            "runtime_failed",
            "fail",
            "The required runtime did not start or install.",
            "retry_runtime",
        )
    if runtime.state == "unsupported":
        # Terminal: no action can make an unsupported machine supported, so offer
        # none rather than sending the user back to a catalog that cannot help.
        detail = runtime.message.strip() or runtime.security_message.strip()
        message = "Automatic setup for the required runtime is unavailable on this machine."
        if detail:
            message = f"{message} {detail}"
        return _check("runtime_unsupported", "fail", message[:240])
    return _check(
        "runtime_missing",
        "fail",
        "The required runtime is not installed.",
        "install_runtime",
    )


def _workflow_check(
    session: Session,
    role: Role,
    install: ModelInstall,
) -> tuple[WorkflowRevision | None, SetupReadinessCheck]:
    definitions = session.scalars(
        select(WorkflowDefinition)
        .where(WorkflowDefinition.operation.in_(MEDIA_OPERATIONS_BY_ROLE[role]))
        .order_by(WorkflowDefinition.updated_at.desc(), WorkflowDefinition.id)
    ).all()
    candidates: list[WorkflowRevision] = []
    for definition in definitions:
        if not definition.current_revision_id:
            continue
        revision = session.get(WorkflowRevision, definition.current_revision_id)
        if not revision or revision.engine != install.engine:
            continue
        if not revision_accepts_install(session, revision.dependencies_json, install.id):
            continue
        candidates.append(revision)
    valid = next(
        (
            revision
            for revision in candidates
            if revision.trusted and (revision.engine == "mock" or bool(revision.api_graph_json))
        ),
        None,
    )
    if valid:
        return valid, _check(
            "workflow_ready",
            "pass",
            "A trusted compatible workflow is ready.",
        )
    if any(revision.trusted for revision in candidates):
        return None, _check(
            "workflow_invalid",
            "fail",
            "The compatible workflow is incomplete.",
            "repair_workflow",
        )
    if candidates:
        return None, _check(
            "workflow_untrusted",
            "fail",
            "The compatible workflow has not been trusted.",
            "review_workflow",
        )
    return None, _check(
        "workflow_missing",
        "fail",
        "No compatible workflow is installed for this model.",
        "repair_workflow",
    )


def _worker_check(
    worker: WorkerStatus | None,
    *,
    expected_profile_id: str | None = None,
) -> SetupReadinessCheck:
    if not worker:
        return _check(
            "worker_status_unavailable",
            "fail",
            "The managed worker status is unavailable.",
            "restart_worker",
        )
    if worker.state == "starting":
        return _check(
            "worker_starting",
            "pending",
            "The managed worker is starting.",
            "wait_for_worker",
        )
    if worker.state == "exited":
        return _check(
            "worker_failed",
            "fail",
            "The managed worker exited unexpectedly.",
            "restart_worker",
        )
    if (
        worker.state == "ready"
        and worker.running
        and (not expected_profile_id or worker.profile_id == expected_profile_id)
    ):
        return _check("worker_ready", "pass", "The managed worker is ready.")
    return _check(
        "worker_on_demand",
        "pass",
        "The managed worker will load this model automatically when used.",
    )


def _check(
    code: str,
    status: CheckStatus,
    message: str,
    action: str | None = None,
) -> SetupReadinessCheck:
    return SetupReadinessCheck(
        code=code,
        status=status,
        message=message,
        action=action,
    )


def _role_result(
    role: Role,
    checks: list[SetupReadinessCheck],
    *,
    engine: str | None = None,
    job_id: str | None = None,
    verification_id: str | None = None,
    install_id: str | None = None,
    profile_id: str | None = None,
    workflow_revision_id: str | None = None,
) -> SetupRoleReadiness:
    if any(check.status == "fail" for check in checks):
        state: Literal["ready", "in_progress", "action_required"] = "action_required"
    elif any(check.status == "pending" for check in checks):
        state = "in_progress"
    else:
        state = "ready"
    next_action = next(
        (check.action for check in checks if check.status != "pass" and check.action),
        None,
    )
    return SetupRoleReadiness(
        role=role,
        state=state,
        engine=engine,
        job_id=job_id,
        verification_id=verification_id,
        install_id=install_id,
        profile_id=profile_id,
        workflow_revision_id=workflow_revision_id,
        next_action=next_action,
        checks=checks,
    )
