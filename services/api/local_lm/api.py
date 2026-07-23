from __future__ import annotations

import shutil
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from .capability_probe import probe_structured_tools
from .catalog import HuggingFaceCatalog
from .config import Settings
from .credentials import CredentialVaultUnavailable
from .custom_nodes import custom_node_dependency_errors
from .db import get_session
from .domain import (
    ArtifactKind,
    CompatibilityLevel,
    JobKind,
    MessageRole,
    ModelRole,
    Operation,
    RoutingMode,
    new_id,
    utcnow,
)
from .downloads import DownloadManager
from .engines import EngineRegistry
from .hardware import collect_system_info
from .models import (
    Artifact,
    Chat,
    CustomNodeInstall,
    GenerationPreset,
    Job,
    Message,
    MessagePart,
    ModelInstall,
    ModelProfile,
    Project,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
)
from .orchestrator import ConversationOrchestrator
from .platforms import list_platform_matrix
from .preflight import assess_catalog_install
from .recipes import get_reference_recipe, list_reference_recipes, recipe_download_request
from .routing import RouteConfirmationRequired
from .schemas import (
    ArtifactCleanupRequest,
    ArtifactCleanupResult,
    ArtifactLibraryItem,
    ArtifactOut,
    ArtifactStorageInfo,
    BackupInfo,
    CatalogDetail,
    CatalogPage,
    CatalogPreflight,
    CatalogPreflightRequest,
    ChatCreate,
    ChatDetail,
    ChatOut,
    CredentialSet,
    CredentialStatus,
    CustomNodeInstallRequest,
    CustomNodeOut,
    CustomNodeTrustRequest,
    CustomNodeUpdateRequest,
    DownloadRequest,
    EngineCapabilities,
    HealthOut,
    JobOut,
    MessageOut,
    ModelImport,
    ModelInstallOut,
    ModelProfileBundle,
    ModelProfileClone,
    ModelProfileCreate,
    ModelProfileOut,
    ModelProfileUpdate,
    ModelStorageInfo,
    PlatformMatrixEntry,
    PresetBundle,
    PresetClone,
    PresetCreate,
    PresetOut,
    PresetUpdate,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
    ReferenceRecipe,
    RegenerateRequest,
    RunOut,
    StorageCleanupResult,
    SystemInfo,
    ToolCapabilityProbe,
    TurnAccepted,
    TurnRequest,
    WorkerStatus,
    WorkflowBundle,
    WorkflowClone,
    WorkflowCreate,
    WorkflowOpenTarget,
    WorkflowOut,
    WorkflowRevisionCreate,
    WorkflowRevisionOut,
    WorkflowUpdate,
)
from .security import SessionSecurity
from .settings_registry import CHAT_SETTINGS, IMAGE_SETTINGS, VIDEO_SETTINGS, validate_settings

if TYPE_CHECKING:
    from .main import Services

SessionDep = Annotated[Session, Depends(get_session)]


def _services(request: Request) -> Services:
    return cast("Services", request.app.state.services)


router = APIRouter(prefix="/api")


@router.post("/session")
async def create_session(request: Request, response: Response) -> dict[str, str | int]:
    services = _services(request)
    security: SessionSecurity = services.security
    return {
        "csrf_token": security.issue_session(response),
        "event_sequence": services.events.sequence,
    }


@router.get("/health", response_model=HealthOut)
async def health(request: Request, session: SessionDep) -> HealthOut:
    engines: EngineRegistry = _services(request).engines
    database_healthy = True
    try:
        session.execute(text("SELECT 1")).scalar_one()
    except SQLAlchemyError:
        database_healthy = False
    try:
        capabilities = await engines.capabilities()
    except Exception:
        capabilities = []
    return HealthOut(
        status=(
            "ok"
            if database_healthy and capabilities and all(item.healthy for item in capabilities)
            else "degraded"
        ),
        version="0.1.0",
        database=database_healthy,
        engines=capabilities,
    )


@router.get("/credentials/huggingface", response_model=CredentialStatus)
async def huggingface_credential_status(request: Request) -> CredentialStatus:
    state = _services(request).credentials.state()
    return CredentialStatus(
        configured=state.configured,
        source=state.source,
        vault_available=state.vault_available,
    )


@router.put("/credentials/huggingface", response_model=CredentialStatus)
async def set_huggingface_credential(payload: CredentialSet, request: Request) -> CredentialStatus:
    services = _services(request)
    try:
        services.credentials.set_token(payload.token)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except CredentialVaultUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    token = services.credentials.token()
    services.settings.hf_token = token
    services.catalog.set_token(token)
    services.downloads.set_token(token)
    return await huggingface_credential_status(request)


@router.delete("/credentials/huggingface", response_model=CredentialStatus)
async def delete_huggingface_credential(request: Request) -> CredentialStatus:
    services = _services(request)
    try:
        services.credentials.delete_token()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except CredentialVaultUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    services.settings.hf_token = None
    services.catalog.set_token(None)
    services.downloads.set_token(None)
    return await huggingface_credential_status(request)


@router.get("/system", response_model=SystemInfo)
async def system_info(request: Request) -> SystemInfo:
    settings: Settings = _services(request).settings
    return collect_system_info(settings)


@router.get("/platforms", response_model=list[PlatformMatrixEntry])
async def platform_matrix() -> list[PlatformMatrixEntry]:
    return list_platform_matrix()


@router.post("/diagnostics", response_model=ArtifactOut, status_code=201)
async def create_diagnostics(request: Request, session: SessionDep) -> ArtifactOut:
    artifact = _services(request).diagnostics.create(session)
    session.commit()
    result = ArtifactOut.model_validate(artifact)
    result.url = f"/api/artifacts/{artifact.id}/content"
    return result


@router.get("/backups", response_model=list[BackupInfo])
async def list_backups(request: Request) -> list[BackupInfo]:
    return _services(request).backups.list()


@router.post("/backups", response_model=BackupInfo, status_code=201)
async def create_backup(request: Request, include_media: bool = False) -> BackupInfo:
    return _services(request).backups.create(include_media=include_media)


@router.post("/backups/{name}/verify", response_model=BackupInfo)
async def verify_backup(name: str, request: Request) -> BackupInfo:
    try:
        return _services(request).backups.verify(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, "backup not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/backups/{name}/restore", response_model=BackupInfo)
async def restore_backup(name: str, request: Request) -> BackupInfo:
    try:
        return _services(request).backups.request_restore(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, "backup not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.delete("/backups/{name}", status_code=204)
async def delete_backup(name: str, request: Request) -> Response:
    try:
        _services(request).backups.delete(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, "backup not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(status_code=204)


@router.get("/engines", response_model=list[EngineCapabilities])
async def engine_capabilities(request: Request) -> list[EngineCapabilities]:
    return await _services(request).engines.capabilities()


@router.post("/engines/chat/tool-probe", response_model=ToolCapabilityProbe)
async def probe_chat_tool_capability(request: Request) -> ToolCapabilityProbe:
    return await probe_structured_tools(_services(request).engines.chat)


@router.get("/workers", response_model=list[WorkerStatus])
async def worker_status(request: Request, session: SessionDep) -> list[WorkerStatus]:
    statuses = _services(request).processes.statuses()
    for index, status in enumerate(statuses):
        kinds = (
            [JobKind.CHAT.value]
            if status.name == "chat"
            else [JobKind.IMAGE.value, JobKind.VIDEO.value]
        )
        active_jobs = (
            session.scalar(
                select(func.count(Job.id)).where(Job.kind.in_(kinds), Job.status == "running")
            )
            or 0
        )
        queued_jobs = (
            session.scalar(
                select(func.count(Job.id)).where(
                    Job.kind.in_(kinds), Job.status.in_(["queued", "paused"])
                )
            )
            or 0
        )
        statuses[index] = status.model_copy(
            update={"active_jobs": active_jobs, "queued_jobs": queued_jobs}
        )
    return statuses


def _ensure_worker_idle(session: Session, name: str) -> None:
    kinds = [JobKind.CHAT.value] if name == "chat" else [JobKind.IMAGE.value, JobKind.VIDEO.value]
    busy_jobs = (
        session.scalar(
            select(func.count(Job.id)).where(
                Job.kind.in_(kinds),
                Job.status.in_(["queued", "running", "paused"]),
            )
        )
        or 0
    )
    if busy_jobs:
        raise HTTPException(
            409,
            f"the {name} worker has {busy_jobs} active or queued "
            f"{'job' if busy_jobs == 1 else 'jobs'}; cancel or wait for them before "
            "changing the worker",
        )


@router.post("/workers/chat/load/{profile_id}", response_model=WorkerStatus)
async def load_chat_worker(profile_id: str, request: Request, session: SessionDep) -> WorkerStatus:
    _ensure_worker_idle(session, "chat")
    profile = session.get(ModelProfile, profile_id)
    if not profile or not profile.model_install_id:
        raise HTTPException(404, "profile with a model install not found")
    install = session.get(ModelInstall, profile.model_install_id)
    if not install:
        raise HTTPException(404, "profile model install not found")
    try:
        return await _services(request).processes.load_chat(profile, install)
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/workers/media/start", response_model=WorkerStatus)
async def start_media_worker(request: Request, session: SessionDep) -> WorkerStatus:
    _ensure_worker_idle(session, "media")
    try:
        return await _services(request).processes.start_media()
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/workers/{name}/stop", response_model=WorkerStatus)
async def stop_worker(name: str, request: Request, session: SessionDep) -> WorkerStatus:
    if name not in {"chat", "media"}:
        raise HTTPException(422, "worker must be chat or media")
    _ensure_worker_idle(session, name)
    try:
        return await _services(request).processes.stop(name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/projects", response_model=list[ProjectOut])
async def list_projects(
    session: SessionDep,
    include_archived: bool = False,
    query: str = Query(default="", max_length=500),
) -> list[Project]:
    statement = select(Project).order_by(Project.updated_at.desc())
    if not include_archived:
        statement = statement.where(Project.archived.is_(False))
    if query.strip():
        statement = statement.where(Project.name.ilike(f"%{query.strip()}%"))
    return list(session.scalars(statement).all())


def _validate_project_workflow_pins(session: Session, values: dict[str, Any]) -> None:
    expected = {
        "image_workflow_revision_id": {
            Operation.TEXT_TO_IMAGE.value,
            Operation.IMAGE_TO_IMAGE.value,
        },
        "video_workflow_revision_id": {
            Operation.TEXT_TO_VIDEO.value,
            Operation.IMAGE_TO_VIDEO.value,
        },
    }
    for field, operations in expected.items():
        revision_id = values.get(field)
        if revision_id is None:
            continue
        revision = session.get(WorkflowRevision, revision_id)
        definition = session.get(WorkflowDefinition, revision.workflow_id) if revision else None
        if not revision or not definition:
            raise HTTPException(422, f"{field} does not identify a workflow revision")
        if definition.operation not in operations:
            raise HTTPException(422, f"{field} has an incompatible workflow operation")


@router.post("/projects", response_model=ProjectOut, status_code=201)
async def create_project(payload: ProjectCreate, session: SessionDep) -> Project:
    values = payload.model_dump()
    _validate_project_workflow_pins(session, values)
    project = Project(**values)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.post("/projects/import", response_model=ProjectOut, status_code=201)
async def import_project(
    request: Request,
    session: SessionDep,
    archive: Annotated[UploadFile, File()],
) -> Project:
    archive.file.seek(0, 2)
    size = archive.file.tell()
    archive.file.seek(0)
    if size > _services(request).settings.max_project_import_bytes:
        raise HTTPException(413, "project archive exceeds the configured limit")
    try:
        project = _services(request).exports.import_archive(session, archive.file)
    except ValueError as exc:
        session.rollback()
        raise HTTPException(422, str(exc)) from exc
    session.commit()
    session.refresh(project)
    return project


@router.patch("/projects/{project_id}", response_model=ProjectOut)
async def update_project(project_id: str, payload: ProjectUpdate, session: SessionDep) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    values = payload.model_dump(exclude_unset=True)
    _validate_project_workflow_pins(session, values)
    for key, value in values.items():
        setattr(project, key, value)
    session.commit()
    session.refresh(project)
    return project


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(project_id: str, session: SessionDep) -> Response:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    for chat in project.chats:
        chat.project_id = None
    session.delete(project)
    session.commit()
    return Response(status_code=204)


@router.post("/projects/{project_id}/export", response_model=ArtifactOut, status_code=201)
async def export_project(
    project_id: str,
    request: Request,
    session: SessionDep,
    include_media: bool = True,
) -> ArtifactOut:
    try:
        artifact = _services(request).exports.export(
            session, project_id, include_media=include_media
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    session.commit()
    result = ArtifactOut.model_validate(artifact)
    result.url = f"/api/artifacts/{artifact.id}/content"
    return result


@router.get("/chats", response_model=list[ChatOut])
async def list_chats(
    session: SessionDep,
    project_id: str | None = None,
    include_archived: bool = False,
    query: str = Query(default="", max_length=500),
) -> list[Chat]:
    statement = select(Chat).order_by(Chat.updated_at.desc())
    if project_id:
        statement = statement.where(Chat.project_id == project_id)
    if not include_archived:
        statement = statement.where(Chat.archived.is_(False))
    if query.strip():
        statement = statement.where(Chat.title.ilike(f"%{query.strip()}%"))
    return list(session.scalars(statement).all())


@router.post("/chats", response_model=ChatOut, status_code=201)
async def create_chat(payload: ChatCreate, session: SessionDep) -> Chat:
    if payload.project_id and not session.get(Project, payload.project_id):
        raise HTTPException(404, "project not found")
    defaults = {
        profile.role: profile.id
        for profile in session.scalars(
            select(ModelProfile).where(ModelProfile.is_default.is_(True))
        ).all()
    }
    chat = Chat(
        title=payload.title,
        project_id=payload.project_id,
        routing_mode=payload.routing_mode.value,
        active_chat_profile_id=defaults.get(ModelRole.CHAT.value),
        active_image_profile_id=defaults.get(ModelRole.IMAGE.value),
        active_video_profile_id=defaults.get(ModelRole.VIDEO.value),
    )
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return chat


@router.get("/chats/{chat_id}", response_model=ChatDetail)
async def get_chat(chat_id: str, session: SessionDep) -> Chat:
    chat = session.scalar(
        select(Chat)
        .options(
            selectinload(Chat.messages)
            .selectinload(Message.parts)
            .selectinload(MessagePart.artifact)
        )
        .where(Chat.id == chat_id)
    )
    if not chat:
        raise HTTPException(404, "chat not found")
    return chat


@router.patch("/chats/{chat_id}", response_model=ChatOut)
async def update_chat(chat_id: str, payload: dict[str, Any], session: SessionDep) -> Chat:
    from .schemas import ChatUpdate

    validated = ChatUpdate.model_validate(payload)
    chat = session.get(Chat, chat_id)
    if not chat:
        raise HTTPException(404, "chat not found")
    values = validated.model_dump(exclude_unset=True, mode="json")
    if (
        "project_id" in values
        and values["project_id"]
        and not session.get(Project, values["project_id"])
    ):
        raise HTTPException(404, "project not found")
    for key, value in values.items():
        setattr(chat, key, value)
    session.commit()
    session.refresh(chat)
    return chat


@router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(chat_id: str, session: SessionDep) -> Response:
    chat = session.get(Chat, chat_id)
    if not chat:
        raise HTTPException(404, "chat not found")
    session.delete(chat)
    session.commit()
    return Response(status_code=204)


@router.post("/chats/{chat_id}/turns", response_model=TurnAccepted, status_code=202)
async def create_turn(
    chat_id: str, payload: TurnRequest, request: Request, session: SessionDep
) -> TurnAccepted:
    orchestrator: ConversationOrchestrator = _services(request).orchestrator
    try:
        return await orchestrator.create_turn(session, chat_id, payload)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RouteConfirmationRequired as exc:
        raise HTTPException(
            409,
            detail={
                "code": "route_confirmation_required",
                "message": str(exc),
                "plan": exc.plan.model_dump(mode="json"),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/messages/{message_id}", response_model=MessageOut)
async def get_message(message_id: str, session: SessionDep) -> Message:
    message = session.scalar(
        select(Message)
        .options(selectinload(Message.parts).selectinload(MessagePart.artifact))
        .where(Message.id == message_id)
    )
    if not message:
        raise HTTPException(404, "message not found")
    return message


@router.post("/messages/{message_id}/regenerate", response_model=TurnAccepted, status_code=202)
async def regenerate_message(
    message_id: str,
    payload: RegenerateRequest,
    request: Request,
    session: SessionDep,
) -> TurnAccepted:
    orchestrator: ConversationOrchestrator = _services(request).orchestrator
    prior_run = session.scalar(select(Run).where(Run.assistant_message_id == message_id))
    if not prior_run:
        raise HTTPException(404, "assistant run not found")
    user_message = session.scalar(
        select(Message)
        .options(selectinload(Message.parts))
        .where(Message.id == prior_run.user_message_id)
    )
    if not user_message:
        raise HTTPException(404, "source user message not found")
    text = "\n".join(part.text for part in user_message.parts if part.text).strip()
    mode = _mode_for_operation(Operation(prior_run.operation))
    prior_settings = orchestrator.request_settings_for_operation(
        Operation(prior_run.operation), prior_run.settings_json
    )
    turn = TurnRequest(
        text=text,
        mode=mode,
        parent_message_id=user_message.parent_id,
        input_artifact_ids=prior_run.provenance_json.get("input_artifact_ids", []),
        settings={**prior_settings, **payload.settings},
    )
    return await _services(request).orchestrator.create_turn(
        session, prior_run.chat_id, turn, use_explicit_parent=True
    )


@router.post("/messages/{message_id}/branch", response_model=TurnAccepted, status_code=202)
async def edit_and_branch(
    message_id: str, payload: TurnRequest, request: Request, session: SessionDep
) -> TurnAccepted:
    source = session.get(Message, message_id)
    if not source or source.role != MessageRole.USER.value:
        raise HTTPException(404, "user message not found")
    prior_run = session.scalar(select(Run).where(Run.user_message_id == source.id))
    updates: dict[str, Any] = {"parent_message_id": source.parent_id}
    if prior_run:
        if payload.mode is None:
            updates["mode"] = _mode_for_operation(Operation(prior_run.operation))
        if not payload.input_artifact_ids:
            updates["input_artifact_ids"] = prior_run.provenance_json.get("input_artifact_ids", [])
        if not payload.settings:
            updates["settings"] = _services(request).orchestrator.request_settings_for_operation(
                Operation(prior_run.operation), prior_run.settings_json
            )
    turn = payload.model_copy(update=updates)
    return await _services(request).orchestrator.create_turn(
        session, source.chat_id, turn, use_explicit_parent=True
    )


def _mode_for_operation(operation: Operation) -> RoutingMode:
    if operation == Operation.TEXT:
        return RoutingMode.TEXT
    if "video" in operation.value:
        return RoutingMode.VIDEO
    return RoutingMode.IMAGE


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(run_id: str, session: SessionDep) -> Run:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@router.get("/jobs", response_model=list[JobOut])
async def list_jobs(
    session: SessionDep,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[Job]:
    statement = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if status:
        statement = statement.where(Job.status == status)
    return list(session.scalars(statement).all())


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
async def cancel_job(job_id: str, request: Request, session: SessionDep) -> Job:
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.kind == JobKind.DOWNLOAD.value:
        changed = await _services(request).downloads.cancel(job_id)
    else:
        changed = await _services(request).orchestrator.cancel(job_id)
    if not changed:
        raise HTTPException(409, "job is already terminal or cannot be cancelled")
    session.expire_all()
    refreshed = session.get(Job, job_id)
    if not refreshed:
        raise HTTPException(404, "job not found")
    return refreshed


@router.post("/chats/{chat_id}/cancel", response_model=JobOut)
async def cancel_active_chat_run(chat_id: str, request: Request, session: SessionDep) -> Job:
    if not session.get(Chat, chat_id):
        raise HTTPException(404, "chat not found")
    job = session.scalar(
        select(Job)
        .join(Run, Job.run_id == Run.id)
        .where(
            Run.chat_id == chat_id,
            Job.status.in_(["queued", "running", "paused"]),
            Job.cancellable.is_(True),
        )
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(1)
    )
    if not job:
        raise HTTPException(409, "chat has no cancellable run")
    changed = await _services(request).orchestrator.cancel(job.id)
    if not changed:
        raise HTTPException(409, "chat run is already terminal or cannot be cancelled")
    session.expire_all()
    refreshed = session.get(Job, job.id)
    if not refreshed:
        raise HTTPException(404, "job not found")
    return refreshed


@router.post("/jobs/{job_id}/retry", response_model=JobOut)
async def retry_job(job_id: str, request: Request, session: SessionDep) -> Job:
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status not in {"failed", "cancelled", "interrupted"}:
        raise HTTPException(409, "only terminal unsuccessful jobs can be retried")
    job.status = "queued"
    job.progress = 0
    job.phase = "retry queued"
    job.error = None
    job.completed_at = None
    if job.run_id:
        run = session.get(Run, job.run_id)
        if run:
            run.status = "queued"
            run.error = None
            run.completed_at = None
    session.commit()
    if job.kind == JobKind.DOWNLOAD.value:
        _services(request).downloads.start(job.id)
    elif job.run_id:
        _services(request).orchestrator.start(job.id, job.run_id)
    else:
        raise HTTPException(422, "job has no retryable operation")
    session.refresh(job)
    return job


@router.post("/artifacts", response_model=ArtifactOut, status_code=201)
async def upload_artifact(
    request: Request,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
    kind: ArtifactKind = ArtifactKind.INPUT,
) -> ArtifactOut:
    services = _services(request)
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > services.settings.max_upload_bytes:
        raise HTTPException(413, "upload exceeds configured limit")
    artifact = services.artifacts.ingest_stream(
        session,
        file.file,
        kind=kind,
        media_type=file.content_type,
        original_name=file.filename,
        metadata={"uploaded": True},
    )
    session.commit()
    result = ArtifactOut.model_validate(artifact)
    result.url = f"/api/artifacts/{artifact.id}/content"
    return result


@router.get("/artifacts", response_model=list[ArtifactLibraryItem])
async def list_artifacts(
    session: SessionDep,
    kind: Literal["image", "video"] | None = None,
    chat_id: str | None = None,
    project_id: str | None = None,
    query: str = Query(default="", max_length=200),
) -> list[ArtifactLibraryItem]:
    reference_rows = session.execute(
        select(MessagePart.artifact_id, Message.chat_id, Chat.project_id)
        .join(Message, Message.id == MessagePart.message_id)
        .join(Chat, Chat.id == Message.chat_id)
        .where(MessagePart.artifact_id.is_not(None))
    ).all()
    references: dict[str, list[tuple[str, str | None]]] = {}
    for artifact_id, referenced_chat_id, referenced_project_id in reference_rows:
        references.setdefault(artifact_id, []).append((referenced_chat_id, referenced_project_id))
    statement = select(Artifact).where(
        Artifact.kind.in_([ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value])
    )
    if kind:
        statement = statement.where(Artifact.kind == kind)
    normalized_query = query.strip().lower()
    if normalized_query:
        statement = statement.where(
            func.lower(func.coalesce(Artifact.original_name, Artifact.sha256)).contains(
                normalized_query
            )
        )
    artifacts = session.scalars(statement.order_by(Artifact.created_at.desc())).all()
    results: list[ArtifactLibraryItem] = []
    for artifact in artifacts:
        artifact_references = references.get(artifact.id, [])
        chat_ids = sorted({item[0] for item in artifact_references})
        project_ids = sorted({item[1] for item in artifact_references if item[1]})
        if chat_id and chat_id not in chat_ids:
            continue
        if project_id and project_id not in project_ids:
            continue
        result = ArtifactLibraryItem.model_validate(artifact)
        result.reference_count = len(artifact_references)
        result.chat_ids = chat_ids
        result.project_ids = project_ids
        result.url = f"/api/artifacts/{artifact.id}/content"
        results.append(result)
    return results


@router.get("/artifacts/storage", response_model=ArtifactStorageInfo)
async def artifact_storage(request: Request, session: SessionDep) -> ArtifactStorageInfo:
    services = _services(request)
    artifacts = session.scalars(select(Artifact)).all()
    referenced = services.artifacts.referenced_artifact_ids(session)
    _marked, eligible_count, eligible_bytes = services.artifacts.cleanup_retention(
        session,
        retention_days=services.settings.artifact_retention_days,
        temporary_hours=services.settings.temporary_retention_hours,
        dry_run=True,
    )
    temporary = [
        artifact
        for artifact in artifacts
        if artifact.metadata_json.get("temporary_preview")
        or artifact.metadata_json.get("intermediate")
    ]
    referenced_artifacts = [artifact for artifact in artifacts if artifact.id in referenced]
    disk_free = shutil.disk_usage(services.settings.data_dir).free
    total_bytes = sum(artifact.size_bytes for artifact in artifacts)
    referenced_bytes = sum(artifact.size_bytes for artifact in referenced_artifacts)
    return ArtifactStorageInfo(
        total_bytes=total_bytes,
        total_count=len(artifacts),
        referenced_bytes=referenced_bytes,
        referenced_count=len(referenced_artifacts),
        unreferenced_bytes=total_bytes - referenced_bytes,
        unreferenced_count=len(artifacts) - len(referenced_artifacts),
        temporary_bytes=sum(artifact.size_bytes for artifact in temporary),
        temporary_count=len(temporary),
        eligible_bytes=eligible_bytes,
        eligible_count=eligible_count,
        disk_free_bytes=disk_free,
        warning=disk_free < services.settings.storage_warning_free_bytes,
        retention_days=services.settings.artifact_retention_days,
        temporary_retention_hours=services.settings.temporary_retention_hours,
    )


@router.post("/artifacts/cleanup", response_model=ArtifactCleanupResult)
async def cleanup_artifacts(
    payload: ArtifactCleanupRequest,
    request: Request,
    session: SessionDep,
) -> ArtifactCleanupResult:
    services = _services(request)
    marked, removed, reclaimed = services.artifacts.cleanup_retention(
        session,
        retention_days=services.settings.artifact_retention_days,
        temporary_hours=services.settings.temporary_retention_hours,
        dry_run=payload.dry_run,
    )
    if not payload.dry_run:
        session.commit()
    return ArtifactCleanupResult(
        dry_run=payload.dry_run,
        marked_count=marked,
        removed_count=removed,
        reclaimed_bytes=reclaimed,
    )


@router.get("/artifacts/{artifact_id}", response_model=ArtifactOut)
async def get_artifact(artifact_id: str, session: SessionDep) -> ArtifactOut:
    artifact = session.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(404, "artifact not found")
    result = ArtifactOut.model_validate(artifact)
    result.url = f"/api/artifacts/{artifact.id}/content"
    return result


def _byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    if not value:
        return None
    if not value.startswith("bytes=") or "," in value:
        raise HTTPException(416, "only a single byte range is supported")
    raw_start, separator, raw_end = value[6:].partition("-")
    if not separator:
        raise HTTPException(416, "invalid byte range")
    try:
        if raw_start:
            start = int(raw_start)
            end = min(int(raw_end), size - 1) if raw_end else size - 1
        else:
            suffix = int(raw_end)
            if suffix <= 0:
                raise ValueError
            start = max(size - suffix, 0)
            end = size - 1
    except ValueError as exc:
        raise HTTPException(416, "invalid byte range") from exc
    if start < 0 or start >= size or end < start:
        raise HTTPException(416, "byte range is outside the artifact")
    return start, end


@router.get("/artifacts/{artifact_id}/content")
async def artifact_content(artifact_id: str, request: Request, session: SessionDep) -> Response:
    artifact = session.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(404, "artifact not found")
    path: Path = _services(request).artifacts.resolve(artifact)
    if not path.is_file():
        raise HTTPException(410, "artifact file is missing")
    size = path.stat().st_size
    selected_range = _byte_range(request.headers.get("range"), size)
    start, end = selected_range or (0, max(size - 1, 0))
    length = end - start + 1 if size else 0
    filename = quote(artifact.original_name or "artifact")
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Disposition": f"inline; filename*=UTF-8''{filename}",
    }
    if selected_range:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    with path.open("rb") as handle:
        handle.seek(start)
        content = handle.read(length)
    return Response(
        content,
        media_type=artifact.media_type,
        status_code=206 if selected_range else 200,
        headers=headers,
    )


@router.get("/catalog", response_model=CatalogPage)
async def catalog_search(
    request: Request,
    query: str = "",
    role: str | None = None,
    sort: str = "trending",
    limit: int = Query(default=30, ge=1, le=100),
    cursor: str | None = None,
    compatibility: str | None = None,
    file_format: str | None = None,
    quantization: str | None = None,
    license_id: str | None = None,
    gated: str | None = None,
    architecture: str | None = None,
    min_parameters: int | None = Query(default=None, ge=0),
    max_parameters: int | None = Query(default=None, ge=0),
    max_size_bytes: int | None = Query(default=None, ge=0),
) -> CatalogPage:
    catalog: HuggingFaceCatalog = _services(request).catalog
    try:
        return await catalog.search(
            query=query,
            role=role,
            sort=sort,
            limit=limit,
            cursor=cursor,
            compatibility=compatibility,
            file_format=file_format,
            quantization=quantization,
            license_id=license_id,
            gated=gated,
            architecture=architecture,
            min_parameters=min_parameters,
            max_parameters=max_parameters,
            max_size_bytes=max_size_bytes,
        )
    except ValueError as exc:
        raise HTTPException(422, f"invalid catalog request: {exc}") from exc
    except Exception as exc:
        raise HTTPException(502, f"catalog request failed: {exc}") from exc


@router.get("/catalog/{owner}/{name}", response_model=CatalogDetail)
async def catalog_detail(
    owner: str,
    name: str,
    request: Request,
    revision: str = "main",
    role: str | None = None,
) -> CatalogDetail:
    try:
        detail = await _services(request).catalog.inspect(f"{owner}/{name}", revision, role)
        return CatalogDetail.model_validate(detail)
    except Exception as exc:
        raise HTTPException(502, f"catalog request failed: {exc}") from exc


@router.post("/catalog/{owner}/{name}/preflight", response_model=CatalogPreflight)
async def catalog_preflight(
    owner: str,
    name: str,
    payload: CatalogPreflightRequest,
    request: Request,
) -> CatalogPreflight:
    services = _services(request)
    try:
        raw_detail = await services.catalog.inspect(
            f"{owner}/{name}", payload.revision, payload.role
        )
        detail = CatalogDetail.model_validate(raw_detail)
    except Exception as exc:
        raise HTTPException(502, f"catalog request failed: {exc}") from exc
    return assess_catalog_install(
        detail,
        payload,
        services.settings,
        collect_system_info(services.settings),
    )


@router.post("/downloads", response_model=JobOut, status_code=202)
async def create_download(payload: DownloadRequest, request: Request, session: SessionDep) -> Job:
    manager: DownloadManager = _services(request).downloads
    try:
        return manager.create(session, payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/downloads/cleanup", response_model=StorageCleanupResult)
async def cleanup_partial_downloads(request: Request, session: SessionDep) -> StorageCleanupResult:
    removed_count, reclaimed_bytes = _services(request).downloads.cleanup_partials(session)
    return StorageCleanupResult(removed_count=removed_count, reclaimed_bytes=reclaimed_bytes)


@router.post("/downloads/{job_id}/pause", response_model=JobOut)
async def pause_download(job_id: str, request: Request, session: SessionDep) -> Job:
    if not await _services(request).downloads.pause(job_id):
        raise HTTPException(409, "download is not running or cannot be paused")
    session.expire_all()
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "download job not found")
    return job


@router.post("/downloads/{job_id}/resume", response_model=JobOut)
async def resume_download(job_id: str, request: Request, session: SessionDep) -> Job:
    if not _services(request).downloads.resume(job_id):
        raise HTTPException(409, "download is not paused, failed, or interrupted")
    session.expire_all()
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "download job not found")
    return job


@router.get("/recipes", response_model=list[ReferenceRecipe])
async def list_recipes() -> list[ReferenceRecipe]:
    return list_reference_recipes()


@router.get("/recipes/{recipe_id}", response_model=ReferenceRecipe)
async def get_recipe(recipe_id: str) -> ReferenceRecipe:
    recipe = get_reference_recipe(recipe_id)
    if not recipe:
        raise HTTPException(404, "reference recipe not found")
    return recipe


@router.post("/recipes/{recipe_id}/install", response_model=JobOut, status_code=202)
async def install_recipe(recipe_id: str, request: Request, session: SessionDep) -> Job:
    recipe = get_reference_recipe(recipe_id)
    if not recipe:
        raise HTTPException(404, "reference recipe not found")
    manager: DownloadManager = _services(request).downloads
    try:
        return manager.create(session, recipe_download_request(recipe))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/models", response_model=list[ModelInstallOut])
async def list_models(session: SessionDep) -> list[ModelInstall]:
    return list(
        session.scalars(select(ModelInstall).order_by(ModelInstall.updated_at.desc())).all()
    )


@router.get("/models/storage", response_model=ModelStorageInfo)
async def model_storage(request: Request, session: SessionDep) -> ModelStorageInfo:
    settings = _services(request).settings
    partials = list(settings.download_dir.glob("*.partial"))
    return ModelStorageInfo(
        installed_bytes=_path_size(settings.model_dir),
        partial_download_bytes=sum(_path_size(path) for path in partials),
        catalog_cache_bytes=_path_size(settings.catalog_cache_dir),
        installed_count=session.scalar(select(func.count(ModelInstall.id))) or 0,
        partial_download_count=len(partials),
    )


@router.post("/models/import", response_model=ModelInstallOut, status_code=201)
async def import_model(payload: ModelImport, session: SessionDep) -> ModelInstall:
    path = Path(payload.local_path).expanduser().resolve(strict=True)
    blocked = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
    files = [child for child in path.rglob("*") if child.is_file()] if path.is_dir() else [path]
    unsafe = [child for child in files if child.suffix.lower() in blocked]
    if unsafe:
        raise HTTPException(422, "pickle-compatible model files are blocked by default")
    size = sum(child.stat().st_size for child in files)
    install = ModelInstall(
        id=new_id("model"),
        name=payload.name,
        role=payload.role,
        engine=payload.engine,
        local_path=str(path),
        size_bytes=size,
        compatibility=CompatibilityLevel.ADVANCED.value,
        manifest_json={
            "imported": True,
            "path_type": "directory" if path.is_dir() else "file",
            "file_count": len(files),
            "pickle_compatible_weights": False,
        },
    )
    session.add(install)
    session.commit()
    session.refresh(install)
    return install


@router.delete("/models/{model_id}", status_code=204)
async def delete_model(model_id: str, request: Request, session: SessionDep) -> Response:
    install = session.get(ModelInstall, model_id)
    if not install:
        raise HTTPException(404, "model not found")
    if session.scalar(
        select(func.count(ModelProfile.id)).where(ModelProfile.model_install_id == install.id)
    ):
        raise HTTPException(409, "delete profiles that use this model before deleting it")
    model_root = _services(request).settings.model_dir.resolve()
    path = Path(install.local_path).resolve()
    siblings = session.scalars(
        select(ModelInstall).where(
            ModelInstall.id != install.id,
            ModelInstall.local_path == install.local_path,
        )
    ).all()
    if model_root in path.parents and path != model_root:
        if siblings and path.is_dir():
            retained_files = {
                str(filename)
                for sibling in siblings
                for filename in sibling.manifest_json.get("files", [])
            }
            for filename in install.manifest_json.get("files", []):
                if str(filename) in retained_files:
                    continue
                candidate = (path / str(filename)).resolve()
                if path in candidate.parents and candidate.is_file():
                    candidate.unlink()
            for directory in sorted(
                (item for item in path.rglob("*") if item.is_dir()), reverse=True
            ):
                with suppress(OSError):
                    directory.rmdir()
        elif not siblings:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    session.delete(install)
    session.commit()
    return Response(status_code=204)


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@router.get("/profiles", response_model=list[ModelProfileOut])
async def list_profiles(session: SessionDep, role: str | None = None) -> list[ModelProfile]:
    statement = select(ModelProfile).order_by(ModelProfile.role, ModelProfile.name)
    if role:
        statement = statement.where(ModelProfile.role == role)
    return list(session.scalars(statement).all())


@router.post("/profiles", response_model=ModelProfileOut, status_code=201)
async def create_profile(payload: ModelProfileCreate, session: SessionDep) -> ModelProfile:
    if payload.model_install_id and not session.get(ModelInstall, payload.model_install_id):
        raise HTTPException(404, "model install not found")
    if payload.is_default:
        for profile in session.scalars(
            select(ModelProfile).where(ModelProfile.role == payload.role)
        ).all():
            profile.is_default = False
    fields = _role_fields(payload.role)
    try:
        load_settings = validate_settings(
            payload.load_settings, [field for field in fields if field.scope == "load"]
        )
        request_settings = validate_settings(
            payload.request_settings, [field for field in fields if field.scope != "load"]
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    profile = ModelProfile(
        name=payload.name,
        role=payload.role,
        engine=payload.engine,
        model_install_id=payload.model_install_id,
        load_settings_json=load_settings,
        request_settings_json=request_settings,
        is_default=payload.is_default,
    )
    session.add(profile)
    session.commit()
    session.refresh(profile)
    return profile


@router.patch("/profiles/{profile_id}", response_model=ModelProfileOut)
async def update_profile(
    profile_id: str, payload: ModelProfileUpdate, session: SessionDep
) -> ModelProfile:
    profile = session.get(ModelProfile, profile_id)
    if not profile:
        raise HTTPException(404, "profile not found")
    values = payload.model_dump(exclude_unset=True)
    if "is_default" in values:
        is_default = bool(values.pop("is_default"))
        if is_default:
            for sibling in session.scalars(
                select(ModelProfile).where(ModelProfile.role == profile.role)
            ).all():
                sibling.is_default = False
        profile.is_default = is_default
    if "load_settings" in values:
        try:
            profile.load_settings_json = validate_settings(
                values.pop("load_settings") or {},
                [field for field in _role_fields(profile.role) if field.scope == "load"],
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if "request_settings" in values:
        try:
            profile.request_settings_json = validate_settings(
                values.pop("request_settings") or {},
                [field for field in _role_fields(profile.role) if field.scope != "load"],
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    for key, value in values.items():
        setattr(profile, key, value)
    session.commit()
    session.refresh(profile)
    return profile


@router.delete("/profiles/{profile_id}", status_code=204)
async def delete_profile(profile_id: str, request: Request, session: SessionDep) -> Response:
    profile = session.get(ModelProfile, profile_id)
    if not profile:
        raise HTTPException(404, "profile not found")
    if any(
        worker.running and worker.profile_id == profile.id
        for worker in _services(request).processes.statuses()
    ):
        raise HTTPException(409, "unload the active worker before deleting its profile")
    session.delete(profile)
    session.commit()
    return Response(status_code=204)


@router.post("/profiles/{profile_id}/clone", response_model=ModelProfileOut, status_code=201)
async def clone_profile(
    profile_id: str, payload: ModelProfileClone, session: SessionDep
) -> ModelProfile:
    source = session.get(ModelProfile, profile_id)
    if not source:
        raise HTTPException(404, "profile not found")
    return await create_profile(
        ModelProfileCreate(
            name=payload.name or f"{source.name} copy",
            role=cast(Literal["chat", "image", "video"], source.role),
            engine=source.engine,
            model_install_id=source.model_install_id,
            load_settings=source.load_settings_json,
            request_settings=source.request_settings_json,
        ),
        session,
    )


@router.post("/profiles/{profile_id}/reset", response_model=ModelProfileOut)
async def reset_profile(profile_id: str, session: SessionDep) -> ModelProfile:
    profile = session.get(ModelProfile, profile_id)
    if not profile:
        raise HTTPException(404, "profile not found")
    profile.load_settings_json = {}
    profile.request_settings_json = {}
    session.commit()
    session.refresh(profile)
    return profile


@router.get("/profiles/{profile_id}/export", response_model=ModelProfileBundle)
async def export_profile(profile_id: str, session: SessionDep) -> ModelProfileBundle:
    profile = session.get(ModelProfile, profile_id)
    if not profile:
        raise HTTPException(404, "profile not found")
    return ModelProfileBundle(
        name=profile.name,
        role=cast(Literal["chat", "image", "video"], profile.role),
        engine=profile.engine,
        model_install_id=profile.model_install_id,
        load_settings=profile.load_settings_json,
        request_settings=profile.request_settings_json,
    )


@router.post("/profiles/import", response_model=ModelProfileOut, status_code=201)
async def import_profile(payload: ModelProfileBundle, session: SessionDep) -> ModelProfile:
    model_install_id = payload.model_install_id
    if model_install_id and not session.get(ModelInstall, model_install_id):
        model_install_id = None
    return await create_profile(
        ModelProfileCreate(
            name=payload.name,
            role=payload.role,
            engine=payload.engine,
            model_install_id=model_install_id,
            load_settings=payload.load_settings,
            request_settings=payload.request_settings,
        ),
        session,
    )


def _role_fields(role: str):  # type: ignore[no-untyped-def]
    return {"chat": CHAT_SETTINGS, "image": IMAGE_SETTINGS, "video": VIDEO_SETTINGS}[role]


def _preset_fields(role: str):  # type: ignore[no-untyped-def]
    return [field for field in _role_fields(role) if field.scope != "load"]


@router.get("/presets", response_model=list[PresetOut])
async def list_presets(session: SessionDep, role: str | None = None) -> list[GenerationPreset]:
    statement = select(GenerationPreset).order_by(GenerationPreset.role, GenerationPreset.name)
    if role:
        statement = statement.where(GenerationPreset.role == role)
    return list(session.scalars(statement).all())


@router.post("/presets", response_model=PresetOut, status_code=201)
async def create_preset(payload: PresetCreate, session: SessionDep) -> GenerationPreset:
    try:
        values = validate_settings(payload.settings, _preset_fields(payload.role))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if payload.is_default:
        for sibling in session.scalars(
            select(GenerationPreset).where(GenerationPreset.role == payload.role)
        ).all():
            sibling.is_default = False
    preset = GenerationPreset(
        name=payload.name,
        role=payload.role,
        settings_json=values,
        is_default=payload.is_default,
    )
    session.add(preset)
    session.commit()
    session.refresh(preset)
    return preset


@router.patch("/presets/{preset_id}", response_model=PresetOut)
async def update_preset(
    preset_id: str, payload: PresetUpdate, session: SessionDep
) -> GenerationPreset:
    preset = session.get(GenerationPreset, preset_id)
    if not preset:
        raise HTTPException(404, "preset not found")
    values = payload.model_dump(exclude_unset=True)
    if "settings" in values:
        try:
            preset.settings_json = validate_settings(
                values.pop("settings") or {}, _preset_fields(preset.role)
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if "is_default" in values:
        is_default = bool(values.pop("is_default"))
        if is_default:
            for sibling in session.scalars(
                select(GenerationPreset).where(GenerationPreset.role == preset.role)
            ).all():
                sibling.is_default = False
        preset.is_default = is_default
    for key, value in values.items():
        setattr(preset, key, value)
    session.commit()
    session.refresh(preset)
    return preset


@router.delete("/presets/{preset_id}", status_code=204)
async def delete_preset(preset_id: str, session: SessionDep) -> Response:
    preset = session.get(GenerationPreset, preset_id)
    if not preset:
        raise HTTPException(404, "preset not found")
    session.delete(preset)
    session.commit()
    return Response(status_code=204)


@router.post("/presets/{preset_id}/clone", response_model=PresetOut, status_code=201)
async def clone_preset(
    preset_id: str, payload: PresetClone, session: SessionDep
) -> GenerationPreset:
    source = session.get(GenerationPreset, preset_id)
    if not source:
        raise HTTPException(404, "preset not found")
    return await create_preset(
        PresetCreate(
            name=payload.name or f"{source.name} copy",
            role=cast(Literal["chat", "image", "video"], source.role),
            settings=source.settings_json,
        ),
        session,
    )


@router.post("/presets/{preset_id}/reset", response_model=PresetOut)
async def reset_preset(preset_id: str, session: SessionDep) -> GenerationPreset:
    preset = session.get(GenerationPreset, preset_id)
    if not preset:
        raise HTTPException(404, "preset not found")
    preset.settings_json = {}
    session.commit()
    session.refresh(preset)
    return preset


@router.get("/presets/{preset_id}/export", response_model=PresetBundle)
async def export_preset(preset_id: str, session: SessionDep) -> PresetBundle:
    preset = session.get(GenerationPreset, preset_id)
    if not preset:
        raise HTTPException(404, "preset not found")
    return PresetBundle(
        name=preset.name,
        role=cast(Literal["chat", "image", "video"], preset.role),
        settings=preset.settings_json,
    )


@router.post("/presets/import", response_model=PresetOut, status_code=201)
async def import_preset(payload: PresetBundle, session: SessionDep) -> GenerationPreset:
    return await create_preset(
        PresetCreate(name=payload.name, role=payload.role, settings=payload.settings), session
    )


def _require_media_worker_stopped(request: Request) -> None:
    if any(
        worker.name == "media" and worker.running
        for worker in _services(request).processes.statuses()
    ):
        raise HTTPException(409, "stop the media worker before changing custom nodes")


@router.get("/custom-nodes", response_model=list[CustomNodeOut])
async def list_custom_nodes(session: SessionDep) -> list[CustomNodeInstall]:
    return list(session.scalars(select(CustomNodeInstall).order_by(CustomNodeInstall.name)).all())


@router.post("/custom-nodes", response_model=CustomNodeOut, status_code=201)
async def install_custom_node(
    payload: CustomNodeInstallRequest, request: Request, session: SessionDep
) -> CustomNodeInstall:
    _require_media_worker_stopped(request)
    try:
        source_url = _services(request).custom_nodes.normalize_source(payload.source_url)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    existing = session.scalar(
        select(CustomNodeInstall).where(CustomNodeInstall.source_url == source_url)
    )
    if existing:
        raise HTTPException(409, "this custom node source is already managed")
    try:
        install = await _services(request).custom_nodes.install(
            session,
            name=payload.name,
            source_url=source_url,
            revision=payload.revision,
        )
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        session.rollback()
        raise HTTPException(422, str(exc)) from exc
    session.commit()
    session.refresh(install)
    return install


@router.patch("/custom-nodes/{node_id}", response_model=CustomNodeOut)
async def update_custom_node(
    node_id: str, payload: CustomNodeUpdateRequest, request: Request, session: SessionDep
) -> CustomNodeInstall:
    _require_media_worker_stopped(request)
    install = session.get(CustomNodeInstall, node_id)
    if not install:
        raise HTTPException(404, "custom node install not found")
    try:
        await _services(request).custom_nodes.update(install, payload.revision)
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        session.rollback()
        raise HTTPException(422, str(exc)) from exc
    session.commit()
    session.refresh(install)
    return install


@router.post("/custom-nodes/{node_id}/trust", response_model=CustomNodeOut)
async def trust_custom_node(
    node_id: str, payload: CustomNodeTrustRequest, request: Request, session: SessionDep
) -> CustomNodeInstall:
    _require_media_worker_stopped(request)
    install = session.get(CustomNodeInstall, node_id)
    if not install:
        raise HTTPException(404, "custom node install not found")
    if payload.trusted:
        try:
            await _services(request).custom_nodes.verify(install)
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            raise HTTPException(422, str(exc)) from exc
    install.trusted = payload.trusted
    install.security_json = {
        **install.security_json,
        "reviewed_at": utcnow().isoformat(),
        "trusted_by_local_user": payload.trusted,
    }
    session.commit()
    session.refresh(install)
    return install


@router.post("/custom-nodes/{node_id}/rollback", response_model=CustomNodeOut)
async def rollback_custom_node(
    node_id: str, request: Request, session: SessionDep
) -> CustomNodeInstall:
    _require_media_worker_stopped(request)
    install = session.get(CustomNodeInstall, node_id)
    if not install:
        raise HTTPException(404, "custom node install not found")
    try:
        await _services(request).custom_nodes.rollback(install)
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        session.rollback()
        raise HTTPException(422, str(exc)) from exc
    session.commit()
    session.refresh(install)
    return install


@router.delete("/custom-nodes/{node_id}", status_code=204)
async def remove_custom_node(node_id: str, request: Request, session: SessionDep) -> Response:
    _require_media_worker_stopped(request)
    install = session.get(CustomNodeInstall, node_id)
    if not install:
        raise HTTPException(404, "custom node install not found")
    _services(request).custom_nodes.remove(install)
    session.delete(install)
    session.commit()
    return Response(status_code=204)


@router.get("/workflows", response_model=list[WorkflowOut])
async def list_workflows(session: SessionDep) -> list[WorkflowDefinition]:
    return list(
        session.scalars(
            select(WorkflowDefinition)
            .options(selectinload(WorkflowDefinition.revisions))
            .order_by(WorkflowDefinition.name)
        ).all()
    )


@router.post("/workflows", response_model=WorkflowOut, status_code=201)
async def create_workflow(payload: WorkflowCreate, session: SessionDep) -> WorkflowDefinition:
    definition = WorkflowDefinition(
        name=payload.name,
        operation=payload.operation.value,
        description=payload.description,
    )
    session.add(definition)
    session.flush()
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine=payload.engine,
        engine_version=payload.engine_version,
        ui_graph_json=payload.ui_graph,
        api_graph_json=payload.api_graph,
        input_schema_json=payload.input_schema,
        dependencies_json=payload.dependencies,
        trusted=payload.trusted,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    session.commit()
    created = session.scalar(
        select(WorkflowDefinition)
        .options(selectinload(WorkflowDefinition.revisions))
        .where(WorkflowDefinition.id == definition.id)
    )
    if not created:
        raise HTTPException(500, "workflow could not be reloaded")
    return created


@router.patch("/workflows/{workflow_id}", response_model=WorkflowOut)
async def update_workflow(
    workflow_id: str, payload: WorkflowUpdate, session: SessionDep
) -> WorkflowDefinition:
    definition = session.get(WorkflowDefinition, workflow_id)
    if not definition:
        raise HTTPException(404, "workflow not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(definition, key, value)
    session.commit()
    updated = session.scalar(
        select(WorkflowDefinition)
        .options(selectinload(WorkflowDefinition.revisions))
        .where(WorkflowDefinition.id == workflow_id)
    )
    if not updated:
        raise HTTPException(500, "workflow could not be reloaded")
    return updated


@router.get("/workflows/{workflow_id}/export", response_model=WorkflowBundle)
async def export_workflow(workflow_id: str, session: SessionDep) -> WorkflowBundle:
    definition, revision = _workflow_and_revision(session, workflow_id)
    return _workflow_bundle(definition, revision)


@router.get("/workflows/{workflow_id}/open-target", response_model=WorkflowOpenTarget)
async def workflow_open_target(
    workflow_id: str, request: Request, session: SessionDep
) -> WorkflowOpenTarget:
    definition, revision = _workflow_and_revision(session, workflow_id)
    if not revision.ui_graph_json:
        raise HTTPException(422, "this workflow has no ComfyUI user-interface graph")
    filename = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in definition.name
    ).strip("-")
    return WorkflowOpenTarget(
        url=_services(request).settings.comfy_url,
        filename=f"{filename or 'workflow'}.comfyui.json",
        ui_graph=revision.ui_graph_json,
    )


@router.post("/workflows/import", response_model=WorkflowOut, status_code=201)
async def import_workflow(payload: WorkflowBundle, session: SessionDep) -> WorkflowDefinition:
    return await create_workflow(
        WorkflowCreate(
            name=payload.name,
            operation=payload.operation,
            description=payload.description,
            engine=payload.engine,
            engine_version=payload.engine_version,
            ui_graph=payload.ui_graph,
            api_graph=payload.api_graph,
            input_schema=payload.input_schema,
            dependencies=payload.dependencies,
            trusted=payload.trusted,
        ),
        session,
    )


@router.post("/workflows/{workflow_id}/clone", response_model=WorkflowOut, status_code=201)
async def clone_workflow(
    workflow_id: str, payload: WorkflowClone, session: SessionDep
) -> WorkflowDefinition:
    definition, revision = _workflow_and_revision(session, workflow_id)
    bundle = _workflow_bundle(definition, revision)
    bundle.name = payload.name or f"{definition.name} copy"
    return await import_workflow(bundle, session)


@router.post(
    "/workflows/{workflow_id}/revisions",
    response_model=WorkflowRevisionOut,
    status_code=201,
)
async def create_workflow_revision(
    workflow_id: str, payload: WorkflowRevisionCreate, session: SessionDep
) -> WorkflowRevision:
    definition = session.get(WorkflowDefinition, workflow_id)
    if not definition:
        raise HTTPException(404, "workflow not found")
    version = (
        session.scalar(
            select(func.max(WorkflowRevision.version)).where(
                WorkflowRevision.workflow_id == workflow_id
            )
        )
        or 0
    )
    current = session.get(WorkflowRevision, definition.current_revision_id)
    revision = WorkflowRevision(
        workflow_id=workflow_id,
        version=version + 1,
        engine=current.engine if current else "comfyui",
        engine_version=payload.engine_version,
        ui_graph_json=payload.ui_graph,
        api_graph_json=payload.api_graph,
        input_schema_json=payload.input_schema,
        dependencies_json=payload.dependencies,
        trusted=payload.trusted,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    session.commit()
    session.refresh(revision)
    return revision


@router.post(
    "/workflows/{workflow_id}/revisions/{revision_id}/restore",
    response_model=WorkflowRevisionOut,
    status_code=201,
)
async def restore_workflow_revision(
    workflow_id: str, revision_id: str, session: SessionDep
) -> WorkflowRevision:
    source = session.get(WorkflowRevision, revision_id)
    if not source or source.workflow_id != workflow_id:
        raise HTTPException(404, "workflow revision not found")
    return await create_workflow_revision(
        workflow_id,
        WorkflowRevisionCreate(
            engine_version=source.engine_version,
            ui_graph=source.ui_graph_json,
            api_graph=source.api_graph_json,
            input_schema=source.input_schema_json,
            dependencies=source.dependencies_json,
            trusted=source.trusted,
        ),
        session,
    )


@router.post("/workflows/{workflow_id}/validate")
async def validate_workflow(
    workflow_id: str, request: Request, session: SessionDep
) -> dict[str, Any]:
    definition = session.get(WorkflowDefinition, workflow_id)
    if not definition or not definition.current_revision_id:
        raise HTTPException(404, "workflow not found")
    revision = session.get(WorkflowRevision, definition.current_revision_id)
    if not revision:
        raise HTTPException(404, "workflow revision not found")
    errors = await _services(request).engines.media.validate_workflow(revision.api_graph_json)
    warnings: list[str] = []
    dependencies = revision.dependencies_json
    errors.extend(custom_node_dependency_errors(session, dependencies.get("custom_nodes")))
    required_models = dependencies.get("models", [])
    installed = session.scalars(select(ModelInstall)).all()
    installed_values = {
        value for model in installed for value in (model.id, model.name, model.local_path)
    }
    for dependency in required_models if isinstance(required_models, list) else []:
        value = (
            dependency.get("id") or dependency.get("name") or dependency.get("path")
            if isinstance(dependency, dict)
            else dependency
        )
        if value and str(value) not in installed_values:
            errors.append(f"missing model dependency: {value}")
    required_engine_version = dependencies.get("engine_version")
    if required_engine_version:
        capabilities = await _services(request).engines.media.capabilities()
        if str(required_engine_version) != capabilities.version:
            errors.append(
                "media engine version does not match the workflow requirement: "
                f"expected {required_engine_version}, found {capabilities.version}"
            )
    minimum_vram = dependencies.get("minimum_vram_bytes")
    if isinstance(minimum_vram, int) and minimum_vram > 0:
        system = collect_system_info(_services(request).settings)
        available = max(
            (
                device.available_memory_bytes or 0
                for device in system.devices
                if device.kind != "cpu"
            ),
            default=0,
        )
        if not available:
            warnings.append("no accelerator memory was detected for this workflow")
        elif available < minimum_vram:
            errors.append("available accelerator memory is below the workflow requirement")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "revision_id": revision.id,
    }


def _workflow_and_revision(
    session: Session, workflow_id: str
) -> tuple[WorkflowDefinition, WorkflowRevision]:
    definition = session.get(WorkflowDefinition, workflow_id)
    if not definition or not definition.current_revision_id:
        raise HTTPException(404, "workflow not found")
    revision = session.get(WorkflowRevision, definition.current_revision_id)
    if not revision:
        raise HTTPException(404, "workflow revision not found")
    return definition, revision


def _workflow_bundle(definition: WorkflowDefinition, revision: WorkflowRevision) -> WorkflowBundle:
    return WorkflowBundle(
        name=definition.name,
        operation=Operation(definition.operation),
        description=definition.description,
        engine=revision.engine,
        engine_version=revision.engine_version,
        ui_graph=revision.ui_graph_json,
        api_graph=revision.api_graph_json,
        input_schema=revision.input_schema_json,
        dependencies=revision.dependencies_json,
        trusted=revision.trusted,
        source_revision=revision.version,
    )
