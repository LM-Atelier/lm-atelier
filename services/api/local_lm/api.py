from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import stat
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload
from starlette.responses import FileResponse

from . import __version__
from .auxiliary_assets import COMFY_AUXILIARY_FOLDERS, validate_lora_workflow_contract
from .capability_evidence import current_capability_evidence, evidence_input_modalities
from .capability_probe import probe_structured_tools
from .catalog import HuggingFaceCatalog
from .comfy_templates import (
    COMFY_TEMPLATE_COMPILER_VERSION,
    ComfyTemplate,
    ComfyTemplateRegistry,
)
from .config import Settings
from .credentials import CredentialVaultUnavailable
from .custom_nodes import custom_node_dependency_errors
from .db import SessionLocal, get_session
from .domain import (
    ArtifactKind,
    CompatibilityLevel,
    JobKind,
    MessageRole,
    MessageStatus,
    ModelRole,
    Operation,
    RoutingMode,
    new_id,
    utcnow,
)
from .downloads import DownloadManager
from .engines import (
    EngineNotConfiguredError,
    EngineRegistry,
    EngineSchemaUnavailableError,
)
from .gguf import (
    GGUFSelectionError,
    automatic_gguf_selection,
    automatic_mmproj_selection,
)
from .hardware import collect_system_info
from .image_edit_strength import STRENGTH_MODE_PARAMETER
from .model_manifests import (
    MAX_METADATA_BYTES,
    MAX_WEIGHT_HEADER_BYTES,
    ModelManifestError,
    inspect_repository_metadata,
)
from .model_planner import persist_install_plan, resolve_install_plan
from .models import (
    AppSetting,
    Artifact,
    Chat,
    CustomNodeInstall,
    GenerationPreset,
    InstallPlan,
    Job,
    Message,
    MessagePart,
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
    Project,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
)
from .orchestrator import ConversationOrchestrator, ResponseRevisionConflict
from .ordered_planning import OrderedPlanConfirmationRequired
from .platforms import list_platform_matrix
from .preflight import assess_catalog_install
from .profile_service import (
    AUTO_PROFILE_ID,
    LAST_CHAT_PROFILE_KEY,
    ensure_profile_for_install,
    validate_profile_binding,
    validate_profile_install,
)
from .progress import update_job_progress
from .recipes import get_reference_recipe, list_reference_recipes, recipe_download_request
from .routing import RouteConfirmationRequired
from .schemas import (
    ApplicationInfo,
    ArtifactCleanupRequest,
    ArtifactCleanupResult,
    ArtifactDeleteResult,
    ArtifactLibraryItem,
    ArtifactOut,
    ArtifactStorageInfo,
    BackupInfo,
    CatalogDetail,
    CatalogModel,
    CatalogPage,
    CatalogPreflight,
    CatalogPreflightCheck,
    CatalogPreflightRequest,
    ChatCreate,
    ChatDetail,
    ChatOut,
    ChatUpdate,
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
    ModelAssetOut,
    ModelAssetUpdate,
    ModelCapabilityEvidenceOut,
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
    RuntimeStatus,
    SettingField,
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
    WorkPlanOut,
    WorkStepOut,
)
from .security import SessionSecurity
from .settings_registry import (
    defaults,
    validate_settings,
    workflow_settings,
)
from .workflow_edit_calibration import validate_workflow_edit_calibration

if TYPE_CHECKING:
    from .main import Services

SessionDep = Annotated[Session, Depends(get_session)]


async def get_conversation_session(request: Request) -> AsyncGenerator[Session, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


ConversationSessionDep = Annotated[Session, Depends(get_conversation_session)]


def _services(request: Request) -> Services:
    return cast("Services", request.app.state.services)


router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


async def _engine_role_fields(
    request: Request,
    role: str,
    *,
    engine: str | None = None,
    allow_inactive: bool = False,
) -> list[SettingField]:
    try:
        return await _services(request).engines.settings_for_role(
            role,
            engine=engine,
            allow_inactive=allow_inactive,
        )
    except EngineNotConfiguredError as exc:
        raise HTTPException(409, str(exc)) from exc
    except EngineSchemaUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/session")
async def create_session(request: Request, response: Response) -> dict[str, str | int]:
    services = _services(request)
    security: SessionSecurity = services.security
    return {
        "csrf_token": security.issue_session(response),
        "event_epoch": services.events.epoch,
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
        version=__version__,
        database=database_healthy,
        engines=capabilities,
    )


@router.get("/ready")
async def ready() -> dict[str, str]:
    return {"version": __version__}


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


@router.get("/about", response_model=ApplicationInfo)
async def application_info(request: Request) -> ApplicationInfo:
    settings: Settings = _services(request).settings
    return ApplicationInfo(
        version=__version__,
        data_directory=str(settings.data_dir.resolve()),
        log_directory=str(settings.log_dir.resolve()),
    )


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
    return await asyncio.to_thread(_services(request).backups.list)


@router.post("/backups", response_model=BackupInfo, status_code=201)
async def create_backup(request: Request, include_media: bool = False) -> BackupInfo:
    return await asyncio.to_thread(
        _services(request).backups.create,
        include_media=include_media,
    )


@router.post("/backups/{name}/verify", response_model=BackupInfo)
async def verify_backup(name: str, request: Request) -> BackupInfo:
    try:
        return await asyncio.to_thread(_services(request).backups.verify, name)
    except FileNotFoundError as exc:
        raise HTTPException(404, "backup not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/backups/{name}/restore", response_model=BackupInfo)
async def restore_backup(name: str, request: Request) -> BackupInfo:
    try:
        return await asyncio.to_thread(_services(request).backups.request_restore, name)
    except FileNotFoundError as exc:
        raise HTTPException(404, "backup not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.delete("/backups/{name}", status_code=204)
async def delete_backup(name: str, request: Request) -> Response:
    try:
        await asyncio.to_thread(_services(request).backups.delete, name)
    except FileNotFoundError as exc:
        raise HTTPException(404, "backup not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(status_code=204)


@router.get("/engines", response_model=list[EngineCapabilities])
async def engine_capabilities(request: Request) -> list[EngineCapabilities]:
    try:
        return await _services(request).engines.capabilities()
    except EngineSchemaUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc


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


@router.get("/runtimes", response_model=list[RuntimeStatus])
async def runtime_status(request: Request) -> list[RuntimeStatus]:
    return _services(request).runtimes.statuses()


@router.post("/runtimes/{engine}/install", response_model=RuntimeStatus, status_code=202)
async def install_runtime(engine: str, request: Request) -> RuntimeStatus:
    if engine not in {"llama.cpp", "vllm", "comfyui"}:
        raise HTTPException(422, "runtime must be llama.cpp, vllm, or comfyui")
    status = _services(request).runtimes.start(
        cast(Literal["llama.cpp", "vllm", "comfyui"], engine)
    )
    if status.state == "unsupported":
        raise HTTPException(422, status.message)
    return status


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


def _validated_profile_install(
    session: Session,
    *,
    model_install_id: str | None,
    role: str,
    engine: str,
) -> ModelInstall | None:
    try:
        return validate_profile_install(
            session,
            model_install_id=model_install_id,
            role=role,
            engine=engine,
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/workers/chat/load/{profile_id}", response_model=WorkerStatus)
async def load_chat_worker(profile_id: str, request: Request, session: SessionDep) -> WorkerStatus:
    services = _services(request)
    async with services.scheduler.lease("primary"):
        _ensure_worker_idle(session, "chat")
        profile = session.get(ModelProfile, profile_id)
        if not profile or not profile.model_install_id:
            raise HTTPException(404, "profile with a model install not found")
        if profile.role != ModelRole.CHAT.value:
            raise HTTPException(422, "chat worker requires a chat profile")
        install = _validated_profile_install(
            session,
            model_install_id=profile.model_install_id,
            role=profile.role,
            engine=profile.engine,
        )
        if not install:
            raise HTTPException(404, "profile with a model install not found")
        try:
            status = await services.processes.load_chat(profile, install)
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            raise HTTPException(422, str(exc)) from exc
        setting = session.get(AppSetting, LAST_CHAT_PROFILE_KEY)
        if setting:
            setting.value_json = profile.id
        else:
            session.add(AppSetting(key=LAST_CHAT_PROFILE_KEY, value_json=profile.id))
        session.commit()
        return status


@router.post("/workers/media/start", response_model=WorkerStatus)
async def start_media_worker(request: Request, session: SessionDep) -> WorkerStatus:
    services = _services(request)
    async with services.scheduler.lease("primary"):
        _ensure_worker_idle(session, "media")
        if services.settings.media_engine != "comfyui":
            raise HTTPException(422, "The ComfyUI media engine is not active.")
        try:
            return await services.processes.start_media()
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            raise HTTPException(422, str(exc)) from exc


@router.post("/workers/{name}/stop", response_model=WorkerStatus)
async def stop_worker(name: str, request: Request, session: SessionDep) -> WorkerStatus:
    if name not in {"chat", "media"}:
        raise HTTPException(422, "worker must be chat or media")
    services = _services(request)
    async with services.scheduler.lease("primary"):
        _ensure_worker_idle(session, name)
        try:
            return await services.processes.stop(name)
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


async def _validate_generation_defaults(
    request: Request,
    session: Session,
    values: dict[str, Any],
) -> None:
    scoped = values.get("generation_settings_json")
    if scoped is None and "generation_settings_json" in values:
        values["generation_settings_json"] = {}
        scoped = {}
    if scoped:
        for role, settings in scoped.items():
            if len(settings) > 256 or any(len(key) > 200 for key in settings):
                raise HTTPException(422, f"{role} generation defaults are too large")
            request_settings = settings
            if STRENGTH_MODE_PARAMETER in settings:
                mode = settings[STRENGTH_MODE_PARAMETER]
                if role != ModelRole.IMAGE.value or mode not in {"auto", "manual"}:
                    raise HTTPException(
                        422,
                        "image edit strength mode must be auto or manual for image defaults",
                    )
                request_settings = {
                    key: value for key, value in settings.items() if key != STRENGTH_MODE_PARAMETER
                }
            fields = await _engine_role_fields(request, role)
            request_fields = [field for field in fields if field.scope != "load"]
            load_keys = {field.key for field in fields if field.scope == "load"}
            disallowed = sorted(load_keys & set(request_settings))
            if disallowed:
                raise HTTPException(
                    422,
                    f"{role} generation defaults cannot include load settings: "
                    f"{', '.join(disallowed)}",
                )
            try:
                validate_settings(request_settings, request_fields)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    bindings = values.get("generation_preset_ids_json")
    if bindings is None and "generation_preset_ids_json" in values:
        values["generation_preset_ids_json"] = {}
        bindings = {}
    for role, preset_id in (bindings or {}).items():
        if preset_id is None:
            continue
        if len(preset_id) > 40:
            raise HTTPException(422, f"{role} generation preset id is too long")
        preset = session.get(GenerationPreset, preset_id)
        if not preset:
            raise HTTPException(404, f"{role} generation preset not found")
        if preset.role != role:
            raise HTTPException(422, f"{role} generation preset has an incompatible role")


@router.post("/projects", response_model=ProjectOut, status_code=201)
async def create_project(
    payload: ProjectCreate,
    request: Request,
    session: SessionDep,
) -> Project:
    values = payload.model_dump()
    _validate_project_workflow_pins(session, values)
    await _validate_generation_defaults(request, session, values)
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
async def update_project(
    project_id: str,
    payload: ProjectUpdate,
    request: Request,
    session: SessionDep,
) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    values = payload.model_dump(exclude_unset=True)
    _validate_project_workflow_pins(session, values)
    await _validate_generation_defaults(request, session, values)
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
    session: ConversationSessionDep,
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
async def create_chat(
    payload: ChatCreate,
    request: Request,
    session: ConversationSessionDep,
) -> Chat:
    if payload.project_id and not session.get(Project, payload.project_id):
        raise HTTPException(404, "project not found")
    values = payload.model_dump(mode="json")
    await _validate_generation_defaults(request, session, values)
    chat = Chat(
        title=payload.title,
        project_id=payload.project_id,
        routing_mode=payload.routing_mode.value,
        generation_settings_json=values["generation_settings_json"],
        generation_preset_ids_json=values["generation_preset_ids_json"],
        vision_settings_json=values["vision_settings_json"],
        active_chat_profile_id=AUTO_PROFILE_ID,
        active_vision_profile_id=AUTO_PROFILE_ID,
        active_image_profile_id=AUTO_PROFILE_ID,
        active_video_profile_id=AUTO_PROFILE_ID,
    )
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return chat


@router.get("/chats/{chat_id}", response_model=ChatDetail)
async def get_chat(chat_id: str, session: ConversationSessionDep) -> Chat:
    chat = session.scalar(
        select(Chat)
        .options(
            selectinload(Chat.messages)
            .selectinload(Message.parts)
            .selectinload(MessagePart.artifact),
            selectinload(Chat.messages)
            .selectinload(Message.response_revisions)
            .selectinload(ResponseRevision.parts)
            .selectinload(ResponseRevisionPart.artifact),
        )
        .where(Chat.id == chat_id)
    )
    if not chat:
        raise HTTPException(404, "chat not found")
    return chat


@router.patch("/chats/{chat_id}", response_model=ChatOut)
async def update_chat(
    chat_id: str,
    payload: ChatUpdate,
    request: Request,
    session: ConversationSessionDep,
) -> Chat:
    chat = session.get(Chat, chat_id)
    if not chat:
        raise HTTPException(404, "chat not found")
    values = payload.model_dump(exclude_unset=True, mode="json")
    await _validate_generation_defaults(request, session, values)
    if (
        "project_id" in values
        and values["project_id"]
        and not session.get(Project, values["project_id"])
    ):
        raise HTTPException(404, "project not found")
    profile_fields = {
        "active_chat_profile_id": ModelRole.CHAT.value,
        "active_vision_profile_id": ModelRole.CHAT.value,
        "active_image_profile_id": ModelRole.IMAGE.value,
        "active_video_profile_id": ModelRole.VIDEO.value,
    }
    for field, role in profile_fields.items():
        profile_id = values.get(field)
        if profile_id in (None, AUTO_PROFILE_ID):
            continue
        profile = session.get(ModelProfile, profile_id)
        if not profile:
            raise HTTPException(404, f"{role} profile not found")
        if profile.role != role:
            raise HTTPException(422, f"{field} requires a {role} profile")
        _validated_profile_install(
            session,
            model_install_id=profile.model_install_id,
            role=profile.role,
            engine=profile.engine,
        )
        if field == "active_vision_profile_id":
            install = session.get(ModelInstall, profile.model_install_id)
            evidence = (
                current_capability_evidence(
                    session,
                    install,
                    _services(request).settings,
                    _services(request).runtimes,
                )
                if install
                else None
            )
            if "image" not in evidence_input_modalities(evidence):
                raise HTTPException(
                    422,
                    "active_vision_profile_id requires a runtime-verified vision profile",
                )
    for key, value in values.items():
        setattr(chat, key, value)
    session.commit()
    session.refresh(chat)
    return chat


@router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(
    chat_id: str,
    request: Request,
    session: ConversationSessionDep,
    delete_generated_media: bool = Query(False),
) -> Response:
    chat = session.get(Chat, chat_id)
    if not chat:
        raise HTTPException(404, "chat not found")
    services = _services(request)
    async with services.orchestrator.prepare_chat_deletion(chat_id):
        session.expire_all()
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(404, "chat not found")
        if delete_generated_media:
            services.artifacts.delete_chat_generated_media(session, chat_id)
        session.delete(chat)
        session.commit()
    return Response(status_code=204)


@router.post("/chats/{chat_id}/turns", response_model=TurnAccepted, status_code=202)
async def create_turn(
    chat_id: str, payload: TurnRequest, request: Request, session: ConversationSessionDep
) -> TurnAccepted:
    orchestrator: ConversationOrchestrator = _services(request).orchestrator
    return await _accept_turn(orchestrator, session, chat_id, payload)


async def _accept_turn(
    orchestrator: ConversationOrchestrator,
    session: Session,
    chat_id: str,
    payload: TurnRequest,
    *,
    use_explicit_parent: bool = False,
    replacement_message_id: str | None = None,
    source_action: str = "send",
    inherited_image_edit_strength: dict[str, Any] | None = None,
) -> TurnAccepted:
    try:
        return await orchestrator.create_turn(
            session,
            chat_id,
            payload,
            use_explicit_parent=use_explicit_parent,
            replacement_message_id=replacement_message_id,
            source_action=source_action,
            inherited_image_edit_strength=inherited_image_edit_strength,
        )
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
    except OrderedPlanConfirmationRequired as exc:
        raise HTTPException(
            409,
            detail={
                "code": "ordered_plan_confirmation_required",
                "message": str(exc),
                "plan": exc.intent.model_dump(mode="json"),
                "estimate": exc.estimate,
            },
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(
            409,
            detail={
                "code": "idempotency_request_in_progress",
                "message": str(exc),
            },
        ) from exc
    except ResponseRevisionConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except EngineNotConfiguredError as exc:
        raise HTTPException(409, str(exc)) from exc
    except EngineSchemaUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/messages/{message_id}", response_model=MessageOut)
async def get_message(message_id: str, session: ConversationSessionDep) -> Message:
    message = session.scalar(
        select(Message)
        .options(
            selectinload(Message.parts).selectinload(MessagePart.artifact),
            selectinload(Message.response_revisions)
            .selectinload(ResponseRevision.parts)
            .selectinload(ResponseRevisionPart.artifact),
        )
        .where(Message.id == message_id)
    )
    if not message:
        raise HTTPException(404, "message not found")
    return message


def _inherited_auto_image_edit_strength(run: Run) -> dict[str, Any] | None:
    image_edit = run.provenance_json.get("image_edit")
    if not isinstance(image_edit, dict):
        return None
    strength = image_edit.get("strength")
    if not isinstance(strength, dict) or strength.get("mode") != "auto":
        return None
    value = strength.get("value")
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return strength


@router.post("/messages/{message_id}/regenerate", response_model=TurnAccepted, status_code=202)
async def regenerate_message(
    message_id: str,
    payload: RegenerateRequest,
    request: Request,
    session: ConversationSessionDep,
) -> TurnAccepted:
    orchestrator: ConversationOrchestrator = _services(request).orchestrator
    source_assistant = session.get(Message, message_id)
    if (
        not source_assistant
        or not source_assistant.transcript_visible
        or source_assistant.status != MessageStatus.COMPLETE.value
    ):
        raise HTTPException(409, "only a completed visible response can be regenerated")
    pending_revision = session.scalar(
        select(ResponseRevision.id).where(
            ResponseRevision.message_id == message_id,
            ResponseRevision.status == MessageStatus.PENDING.value,
        )
    )
    if pending_revision:
        raise HTTPException(409, "this response is already being regenerated")
    active_revision = (
        session.get(ResponseRevision, source_assistant.active_response_revision_id)
        if source_assistant.active_response_revision_id
        else None
    )
    prior_run = (
        session.get(Run, active_revision.run_id)
        if active_revision and active_revision.run_id
        else None
    )
    if not prior_run:
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
    prior_revision = (
        session.get(WorkflowRevision, prior_run.workflow_revision_id)
        if prior_run.workflow_revision_id
        else None
    )
    prior_profile = (
        session.get(ModelProfile, prior_run.profile_id) if prior_run.profile_id else None
    )
    try:
        prior_settings = await orchestrator.request_settings_for_operation(
            Operation(prior_run.operation),
            prior_run.settings_json,
            input_schema=prior_revision.input_schema_json if prior_revision else None,
            engine=prior_profile.engine if prior_profile else None,
        )
    except EngineNotConfiguredError as exc:
        raise HTTPException(409, str(exc)) from exc
    except EngineSchemaUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    turn = TurnRequest(
        text=text,
        mode=mode,
        parent_message_id=user_message.parent_id,
        input_artifact_ids=orchestrator.input_artifact_ids_for_run(session, prior_run),
        settings={**prior_settings, **payload.settings},
    )
    prior_strength = _inherited_auto_image_edit_strength(prior_run)
    inherited_parameter = (
        prior_strength.get("parameter")
        if prior_strength and isinstance(prior_strength.get("parameter"), str)
        else "denoise"
    )
    inherited_image_edit_strength = (
        None if inherited_parameter in payload.settings else prior_strength
    )
    return await _accept_turn(
        orchestrator,
        session,
        prior_run.chat_id,
        turn,
        use_explicit_parent=True,
        replacement_message_id=message_id,
        source_action="regenerate",
        inherited_image_edit_strength=inherited_image_edit_strength,
    )


@router.post(
    "/messages/{message_id}/revisions/{revision_id}/select",
    response_model=MessageOut,
)
async def select_response_revision(
    message_id: str,
    revision_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Message:
    try:
        return _services(request).orchestrator.select_response_revision(
            session,
            message_id,
            revision_id,
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/messages/{message_id}/branch", response_model=TurnAccepted, status_code=202)
async def edit_and_branch(
    message_id: str,
    payload: TurnRequest,
    request: Request,
    session: ConversationSessionDep,
) -> TurnAccepted:
    source = session.get(Message, message_id)
    if not source or source.role != MessageRole.USER.value:
        raise HTTPException(404, "user message not found")
    prior_run = session.scalar(select(Run).where(Run.user_message_id == source.id))
    updates: dict[str, Any] = {"parent_message_id": source.parent_id}
    inherited_image_edit_strength: dict[str, Any] | None = None
    if prior_run:
        prior_mode = _mode_for_operation(Operation(prior_run.operation))
        mode_was_supplied = "mode" in payload.model_fields_set and payload.mode is not None
        target_mode = payload.mode if mode_was_supplied else prior_mode
        updates["mode"] = target_mode
        same_explicit_role = mode_was_supplied and target_mode == prior_mode
        legacy_inheritance = not mode_was_supplied
        inherit_inputs = (legacy_inheritance and not payload.input_artifact_ids) or (
            same_explicit_role and "input_artifact_ids" not in payload.model_fields_set
        )
        inherit_settings = (legacy_inheritance and not payload.settings) or (
            same_explicit_role and "settings" not in payload.model_fields_set
        )
        if inherit_inputs:
            updates["input_artifact_ids"] = _services(
                request
            ).orchestrator.input_artifact_ids_for_run(session, prior_run)
        if inherit_settings:
            prior_revision = (
                session.get(WorkflowRevision, prior_run.workflow_revision_id)
                if prior_run.workflow_revision_id
                else None
            )
            prior_profile = (
                session.get(ModelProfile, prior_run.profile_id) if prior_run.profile_id else None
            )
            try:
                updates["settings"] = await _services(
                    request
                ).orchestrator.request_settings_for_operation(
                    Operation(prior_run.operation),
                    prior_run.settings_json,
                    input_schema=(prior_revision.input_schema_json if prior_revision else None),
                    engine=prior_profile.engine if prior_profile else None,
                )
                inherited_image_edit_strength = _inherited_auto_image_edit_strength(prior_run)
            except EngineNotConfiguredError as exc:
                raise HTTPException(409, str(exc)) from exc
            except EngineSchemaUnavailableError as exc:
                raise HTTPException(503, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
    turn = payload.model_copy(update=updates)
    return await _accept_turn(
        _services(request).orchestrator,
        session,
        source.chat_id,
        turn,
        use_explicit_parent=True,
        source_action="edit_and_branch",
        inherited_image_edit_strength=inherited_image_edit_strength,
    )


def _mode_for_operation(operation: Operation) -> RoutingMode:
    if operation == Operation.TEXT:
        return RoutingMode.TEXT
    if "video" in operation.value:
        return RoutingMode.VIDEO
    return RoutingMode.IMAGE


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(run_id: str, session: ConversationSessionDep) -> Run:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@router.get("/work-plans", response_model=list[WorkPlanOut])
async def list_work_plans(
    session: ConversationSessionDep,
    chat_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[WorkPlan]:
    statement = (
        select(WorkPlan)
        .options(selectinload(WorkPlan.steps))
        .order_by(WorkPlan.created_at.desc(), WorkPlan.id.desc())
        .limit(limit)
    )
    if chat_id:
        statement = statement.where(WorkPlan.chat_id == chat_id)
    return list(session.scalars(statement).unique().all())


@router.get("/work-plans/{plan_id}", response_model=WorkPlanOut)
async def get_work_plan(plan_id: str, session: ConversationSessionDep) -> WorkPlan:
    plan = session.scalar(
        select(WorkPlan).options(selectinload(WorkPlan.steps)).where(WorkPlan.id == plan_id)
    )
    if not plan:
        raise HTTPException(404, "work plan not found")
    return plan


@router.get("/work-steps/{step_id}", response_model=WorkStepOut)
async def get_work_step(step_id: str, session: ConversationSessionDep) -> WorkStep:
    step = session.get(WorkStep, step_id)
    if not step:
        raise HTTPException(404, "work step not found")
    return step


@router.post("/work-plans/{plan_id}/cancel", response_model=WorkPlanOut)
async def cancel_work_plan(
    plan_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> WorkPlan:
    plan = session.scalar(
        select(WorkPlan).options(selectinload(WorkPlan.steps)).where(WorkPlan.id == plan_id)
    )
    if not plan:
        raise HTTPException(404, "work plan not found")
    jobs = list(
        session.scalars(
            select(Job)
            .where(
                Job.work_plan_id == plan_id,
                Job.status.in_(["queued", "running", "paused"]),
            )
            .order_by(Job.created_at, Job.id)
        ).all()
    )
    if not jobs:
        raise HTTPException(409, "work plan has no cancellable steps")
    for job in jobs:
        await _services(request).orchestrator.cancel(job.id)
    session.expire_all()
    refreshed = session.scalar(
        select(WorkPlan).options(selectinload(WorkPlan.steps)).where(WorkPlan.id == plan_id)
    )
    if not refreshed:
        raise HTTPException(404, "work plan not found")
    return refreshed


@router.post("/work-steps/{step_id}/cancel", response_model=JobOut)
async def cancel_work_step(
    step_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Job:
    job = session.scalar(select(Job).where(Job.work_step_id == step_id))
    if not job:
        raise HTTPException(404, "work step job not found")
    return await cancel_job(job.id, request, session)


@router.post("/work-plans/{plan_id}/retry", response_model=WorkPlanOut)
async def retry_work_plan(
    plan_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> WorkPlan:
    plan = session.scalar(
        select(WorkPlan).options(selectinload(WorkPlan.steps)).where(WorkPlan.id == plan_id)
    )
    if not plan:
        raise HTTPException(404, "work plan not found")
    jobs = list(
        session.scalars(
            select(Job)
            .where(
                Job.work_plan_id == plan_id,
                Job.status.in_(["failed", "cancelled", "interrupted"]),
            )
            .order_by(Job.created_at, Job.id)
        ).all()
    )
    if not jobs:
        raise HTTPException(409, "work plan has no retryable steps")
    for job in jobs:
        await retry_job(job.id, request, session)
    session.expire_all()
    refreshed = session.scalar(
        select(WorkPlan).options(selectinload(WorkPlan.steps)).where(WorkPlan.id == plan_id)
    )
    if not refreshed:
        raise HTTPException(404, "work plan not found")
    return refreshed


@router.post("/work-steps/{step_id}/retry", response_model=JobOut)
async def retry_work_step(
    step_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Job:
    job = session.scalar(select(Job).where(Job.work_step_id == step_id))
    if not job:
        raise HTTPException(404, "work step job not found")
    return await retry_job(job.id, request, session)


@router.get("/jobs", response_model=list[JobOut])
async def list_jobs(
    session: ConversationSessionDep,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[Job]:
    statement = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if status:
        statement = statement.where(Job.status == status)
    return list(session.scalars(statement).all())


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
async def cancel_job(
    job_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Job:
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
async def cancel_active_chat_run(
    chat_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Job:
    if not session.get(Chat, chat_id):
        raise HTTPException(404, "chat not found")
    job = _current_chat_job(session, chat_id)
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


@router.post(
    "/chats/{chat_id}/stop-and-send",
    response_model=TurnAccepted,
    status_code=202,
)
async def stop_and_send_turn(
    chat_id: str,
    payload: TurnRequest,
    request: Request,
    session: ConversationSessionDep,
) -> TurnAccepted:
    if not session.get(Chat, chat_id):
        raise HTTPException(404, "chat not found")
    job = _current_chat_job(session, chat_id)
    if job:
        await _services(request).orchestrator.cancel(job.id)
        session.expire_all()
    return await _accept_turn(
        _services(request).orchestrator,
        session,
        chat_id,
        payload,
        source_action="stop_and_send",
    )


def _current_chat_job(session: Session, chat_id: str) -> Job | None:
    running = session.scalar(
        select(Job)
        .join(Run, Job.run_id == Run.id)
        .where(
            Run.chat_id == chat_id,
            Job.status == "running",
            Job.cancellable.is_(True),
        )
        .order_by(Job.started_at, Job.created_at, Job.id)
        .limit(1)
    )
    if running:
        return running
    return session.scalar(
        select(Job)
        .join(Run, Job.run_id == Run.id)
        .where(
            Run.chat_id == chat_id,
            Job.status.in_(["queued", "paused"]),
            Job.cancellable.is_(True),
        )
        .order_by(Job.created_at, Job.id)
        .limit(1)
    )


@router.post("/jobs/{job_id}/retry", response_model=JobOut)
async def retry_job(
    job_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Job:
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status not in {"failed", "cancelled", "interrupted"}:
        raise HTTPException(409, "only terminal unsuccessful jobs can be retried")
    if job.kind == JobKind.DOWNLOAD.value:
        job.status = "queued"
        job.progress = 0
        job.error = None
        job.started_at = None
        job.completed_at = None
        job.enqueued_at = utcnow()
        job.claim_owner = None
        job.claim_expires_at = None
        job.heartbeat_at = None
        update_job_progress(
            job,
            stage="retry queued",
            queue_resource=job.queue_resource or "network",
            indeterminate=True,
        )
        session.commit()
        _services(request).downloads.start(job.id)
        session.refresh(job)
        return job
    if not job.run_id:
        raise HTTPException(422, "job has no retryable operation")
    run = session.get(Run, job.run_id)
    if not run:
        raise HTTPException(422, "job has no retryable operation")

    orchestrator = _services(request).orchestrator
    async with orchestrator.chat_guard(run.chat_id):
        session.expire_all()
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(404, "job not found")
        if job.status not in {"failed", "cancelled", "interrupted"}:
            raise HTTPException(409, "only terminal unsuccessful jobs can be retried")
        if not job.run_id:
            raise HTTPException(422, "job has no retryable operation")
        run = session.get(Run, job.run_id)
        if not run:
            raise HTTPException(422, "job has no retryable operation")
        job.status = "queued"
        job.progress = 0
        job.error = None
        job.started_at = None
        job.completed_at = None
        job.enqueued_at = utcnow()
        job.claim_owner = None
        job.claim_expires_at = None
        job.heartbeat_at = None
        update_job_progress(
            job,
            stage="retry queued",
            queue_resource=job.queue_resource or "interactive_compute",
            indeterminate=True,
        )
        run.status = "queued"
        run.error = None
        run.completed_at = None
        try:
            orchestrator.prepare_retry(session, run)
        except LookupError as exc:
            raise HTTPException(422, str(exc)) from exc
        session.commit()
        orchestrator.start(job.id, run.id)
        session.refresh(job)
        return job


@router.post("/artifacts", response_model=ArtifactOut, status_code=201)
async def upload_artifact(
    request: Request,
    session: ConversationSessionDep,
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
    session: ConversationSessionDep,
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
async def artifact_storage(
    request: Request,
    session: ConversationSessionDep,
) -> ArtifactStorageInfo:
    services = _services(request)
    artifacts = session.scalars(select(Artifact)).all()
    referenced = services.artifacts.referenced_artifact_ids(session)
    retention = services.artifacts.cleanup_retention(
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
        eligible_bytes=retention.reclaimed_bytes,
        eligible_count=retention.removed_count,
        retention_pending_count=retention.pending_count,
        disk_free_bytes=disk_free,
        warning=disk_free < services.settings.storage_warning_free_bytes,
        retention_days=services.settings.artifact_retention_days,
        temporary_retention_hours=services.settings.temporary_retention_hours,
    )


@router.post("/artifacts/cleanup", response_model=ArtifactCleanupResult)
async def cleanup_artifacts(
    payload: ArtifactCleanupRequest,
    request: Request,
    session: ConversationSessionDep,
) -> ArtifactCleanupResult:
    services = _services(request)
    cleanup = services.artifacts.cleanup_retention(
        session,
        retention_days=services.settings.artifact_retention_days,
        temporary_hours=services.settings.temporary_retention_hours,
        dry_run=payload.dry_run,
    )
    if not payload.dry_run:
        session.commit()
    return ArtifactCleanupResult(
        dry_run=payload.dry_run,
        marked_count=cleanup.marked_count,
        retention_pending_count=cleanup.pending_count,
        removed_count=cleanup.removed_count,
        reclaimed_bytes=cleanup.reclaimed_bytes,
    )


@router.delete("/artifacts/{artifact_id}", response_model=ArtifactDeleteResult)
async def delete_artifact(
    artifact_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> ArtifactDeleteResult:
    artifact = session.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(404, "artifact not found")
    try:
        references, removed, reclaimed = _services(request).artifacts.delete_library_artifact(
            session, artifact
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    session.commit()
    return ArtifactDeleteResult(
        artifact_id=artifact_id,
        reference_count=references,
        removed_count=removed,
        reclaimed_bytes=reclaimed,
    )


@router.get("/artifacts/{artifact_id}", response_model=ArtifactOut)
async def get_artifact(
    artifact_id: str,
    session: ConversationSessionDep,
) -> ArtifactOut:
    artifact = session.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(404, "artifact not found")
    result = ArtifactOut.model_validate(artifact)
    result.url = f"/api/artifacts/{artifact.id}/content"
    return result


@router.get("/artifacts/{artifact_id}/content")
async def artifact_content(
    artifact_id: str,
    request: Request,
    session: ConversationSessionDep,
) -> Response:
    artifact = session.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(404, "artifact not found")
    try:
        path, media_type, disposition = _services(request).artifacts.delivery_metadata(artifact)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(410, "artifact file is missing or corrupt") from exc
    return FileResponse(
        path,
        media_type=media_type,
        filename=Path(artifact.original_name or "artifact").name,
        content_disposition_type=disposition,
        stat_result=path.stat(),
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "Cross-Origin-Resource-Policy": "same-origin",
            "ETag": f'"{artifact.sha256}"',
            "X-Content-Type-Options": "nosniff",
        },
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
    updated_within_days: int | None = Query(default=None, ge=1, le=3650),
) -> CatalogPage:
    services = _services(request)
    catalog: HuggingFaceCatalog = services.catalog
    try:
        media_catalog = role in {"image", "video"} and services.settings.media_engine == "comfyui"
        page = await catalog.search(
            query=query,
            role=role,
            sort=sort,
            limit=limit,
            cursor=cursor,
            compatibility=None if media_catalog else compatibility,
            file_format=file_format,
            quantization=quantization,
            license_id=license_id,
            gated=gated,
            architecture=architecture,
            min_parameters=min_parameters,
            max_parameters=max_parameters,
            max_size_bytes=max_size_bytes,
            updated_within_days=updated_within_days,
        )
        if not media_catalog or role is None:
            return page
        registry = ComfyTemplateRegistry(services.settings)
        items = []
        for item in page.items:
            ready = bool(registry.matches(item.remote_id, role))
            adaptive_candidate = role == "image" and "safetensors" in {
                item_format.casefold() for item_format in item.formats
            }
            items.append(
                item.model_copy(
                    update={
                        "compatibility": (
                            "likely"
                            if ready
                            else ("advanced_import" if adaptive_candidate else "unsupported")
                        ),
                        "compatibility_reasons": (
                            ["Official ComfyUI workflow available"]
                            if ready
                            else (
                                ["One-click checkpoint compatibility is checked before download"]
                                if adaptive_candidate
                                else ["No safe automatic workflow contract is available"]
                            )
                        ),
                    }
                )
            )
        if compatibility:
            items = [item for item in items if item.compatibility == compatibility]
        if sort == "compatible":
            items.sort(
                key=lambda item: (
                    item.compatibility != "likely",
                    -(item.downloads or 0),
                )
            )
        return page.model_copy(update={"items": items})
    except ValueError as exc:
        raise HTTPException(422, f"invalid catalog request: {exc}") from exc
    except Exception as exc:
        raise HTTPException(
            503,
            "Hugging Face is temporarily unavailable. Check your connection and retry.",
        ) from exc


@router.get("/catalog/workflow-models", response_model=list[CatalogModel])
async def workflow_catalog_models(request: Request, role: str) -> list[CatalogModel]:
    if role not in {"image", "video"}:
        return []
    registry = ComfyTemplateRegistry(_services(request).settings)
    repositories: dict[str, CatalogModel] = {}
    for template in registry.available(role):
        key = template.remote_id.casefold()
        if key in repositories:
            continue
        remote_id = template.remote_id
        repositories[key] = CatalogModel(
            remote_id=remote_id,
            name=remote_id.rsplit("/", 1)[-1],
            author=remote_id.split("/", 1)[0],
            pipeline_tag="text-to-image" if role == "image" else "text-to-video",
            formats=sorted(
                {
                    Path(dependency.path).suffix.lower().lstrip(".")
                    for dependency in template.dependencies
                    if Path(dependency.path).suffix
                }
            ),
            compatibility="likely",
            compatibility_reasons=["Official ComfyUI workflow available"],
        )
    return sorted(repositories.values(), key=lambda item: item.remote_id.casefold())


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
        raise HTTPException(
            503,
            "Hugging Face is temporarily unavailable. Check your connection and retry.",
        ) from exc


@router.post("/catalog/{owner}/{name}/preflight", response_model=CatalogPreflight)
async def catalog_preflight(
    owner: str,
    name: str,
    payload: CatalogPreflightRequest,
    request: Request,
    session: SessionDep,
) -> CatalogPreflight:
    services = _services(request)
    try:
        raw_detail = await services.catalog.inspect(
            f"{owner}/{name}", payload.revision, payload.role
        )
        detail = CatalogDetail.model_validate(raw_detail)
        if payload.role == "chat" and payload.engine == "llama.cpp":
            try:
                initial_selection = (
                    list(payload.selected_files)
                    if payload.selected_files
                    else automatic_gguf_selection(
                        detail.files,
                        collect_system_info(services.settings).memory_total_bytes,
                    )
                )
                if not automatic_mmproj_selection(detail.files, initial_selection):
                    companion = await services.catalog.discover_vision_projector(
                        detail.model.remote_id,
                        initial_selection,
                    )
                    if companion:
                        detail = detail.model_copy(update={"files": [*detail.files, companion]})
            except (GGUFSelectionError, httpx.HTTPError, ValueError):
                pass
    except Exception as exc:
        raise HTTPException(
            503,
            "Hugging Face is temporarily unavailable. Check your connection and retry.",
        ) from exc
    system = collect_system_info(services.settings)

    async def finalize(
        result: CatalogPreflight,
        resolved_detail: CatalogDetail,
    ) -> CatalogPreflight:
        metadata: dict[str, bytes] = {}
        inspect_prefix = getattr(services.catalog, "inspect_file_prefix", None)
        repo_files = {
            str(item.get("filename") or "")
            for item in resolved_detail.files
            if item.get("filename")
        }
        metadata_paths = sorted(
            path
            for path in repo_files
            if PurePosixPath(path).name.casefold()
            in {"config.json", "model_index.json", "generation_config.json"}
        )[:16]
        selected_binary_paths = [
            path
            for path in result.selected_files
            if path.casefold().endswith((".gguf", ".safetensors"))
        ][:8]

        file_metadata = {str(item.get("filename") or ""): item for item in resolved_detail.files}

        async def fetch_prefix(path: str) -> tuple[str, bytes] | None:
            if not callable(inspect_prefix):
                return None
            try:
                source = file_metadata.get(path) or {}
                source_remote_id = str(
                    source.get("source_remote_id") or resolved_detail.model.remote_id
                )
                source_revision = str(source.get("source_revision") or resolved_detail.revision)
                source_filename = str(source.get("source_filename") or path)
                if path.casefold().endswith(".safetensors"):
                    prefix = await inspect_prefix(
                        source_remote_id,
                        source_revision,
                        source_filename,
                        max_bytes=8,
                    )
                    if len(prefix) < 8:
                        return path, prefix
                    header_size = int.from_bytes(prefix[:8], "little")
                    limit = min(8 + header_size, MAX_WEIGHT_HEADER_BYTES + 8)
                elif path.casefold().endswith(".gguf"):
                    limit = MAX_WEIGHT_HEADER_BYTES
                else:
                    limit = MAX_METADATA_BYTES
                return (
                    path,
                    await inspect_prefix(
                        source_remote_id,
                        source_revision,
                        source_filename,
                        max_bytes=limit,
                    ),
                )
            except Exception:
                # Staged bytes are inspected again before activation. Remote
                # prefix inspection is an optimization, never a trust bypass.
                return None

        inspected_prefixes = await asyncio.gather(
            *(fetch_prefix(path) for path in [*metadata_paths, *selected_binary_paths])
        )
        metadata.update(item for item in inspected_prefixes if item is not None)
        if not metadata and resolved_detail.model.architecture:
            metadata["catalog-config.json"] = json.dumps(
                {"model_type": resolved_detail.model.architecture},
                separators=(",", ":"),
            ).encode()
        inspection_error: str | None = None
        try:
            inspection = inspect_repository_metadata(
                metadata,
                result.selected_files,
                role=payload.role,
            )
        except ModelManifestError as exc:
            inspection_error = str(exc)
            inspection = inspect_repository_metadata(
                {},
                result.selected_files,
                role=payload.role,
            )
        files = {str(item.get("filename") or ""): item for item in resolved_detail.files}
        selected_metadata = [
            files.get(
                filename,
                {
                    "filename": filename,
                    "size": None,
                    "sha256": result.expected_sha256.get(filename),
                },
            )
            for filename in result.selected_files
        ]
        resolved = resolve_install_plan(
            remote_id=result.remote_id,
            revision=result.revision,
            role=payload.role,
            engine=payload.engine,
            selected_files=selected_metadata,
            inspection=inspection,
            workflow_template_id=result.workflow_template_id,
            workflow_template_sha256=result.workflow_template_sha256,
            comfy_paths=result.comfy_paths,
            source_remote_id=result.source_remote_id,
            auxiliary_kind=payload.auxiliary_kind,
        )
        if inspection_error:
            resolved = resolved.blocked(
                "metadata_inspection_failed",
                inspection_error,
            )
        if not result.can_install:
            resolved = resolved.blocked(
                "preflight_blocked",
                next(
                    (check.detail for check in result.checks if check.status == "block"),
                    "The install check did not pass.",
                ),
            )
        plan = persist_install_plan(session, resolved)
        session.commit()
        return result.model_copy(update={"install_plan": plan})

    if payload.auxiliary_kind:
        if payload.role != "image":
            result = assess_catalog_install(detail, payload, services.settings, system)
            return await finalize(
                result.model_copy(
                    update={
                        "can_install": False,
                        "checks": [
                            *result.checks,
                            CatalogPreflightCheck(
                                id="auxiliary-role",
                                label="Asset role",
                                status="block",
                                detail="LoRA assets currently extend image workflows.",
                            ),
                        ],
                    }
                ),
                detail,
            )
        return await finalize(
            assess_catalog_install(detail, payload, services.settings, system).model_copy(
                update={
                    "comfy_paths": {COMFY_AUXILIARY_FOLDERS[payload.auxiliary_kind]: "."},
                }
            ),
            detail,
        )

    if (
        payload.role == "chat"
        or payload.engine != "comfyui"
        or services.settings.media_engine != "comfyui"
    ):
        return await finalize(
            assess_catalog_install(detail, payload, services.settings, system),
            detail,
        )

    runtime_status = services.runtimes.status("comfyui")
    if runtime_status.security_status == "blocked" and runtime_status.state != "ready":
        result = assess_catalog_install(detail, payload, services.settings, system)
        checks = [
            *[check for check in result.checks if check.id != "runtime"],
            CatalogPreflightCheck(
                id="runtime",
                label="Runtime security",
                status="block",
                detail=runtime_status.security_message or runtime_status.message,
            ),
        ]
        return await finalize(
            result.model_copy(update={"can_install": False, "checks": checks}),
            detail,
        )

    registry = ComfyTemplateRegistry(services.settings)
    candidates = registry.matches(detail.model.remote_id, payload.role)
    if not candidates:
        resolved_payload = payload.model_copy(update={"revision": detail.revision})
        result = assess_catalog_install(
            detail,
            resolved_payload,
            services.settings,
            system,
        )
        adaptive = registry.adaptive_checkpoint(
            detail.model.remote_id,
            detail.revision,
            result.selected_files,
            payload.role,
        )
        if adaptive:
            checks = [
                *[
                    (
                        CatalogPreflightCheck(
                            id="runtime",
                            label="Runtime compatibility",
                            status="warn",
                            detail=(
                                "This is a safe single-file checkpoint layout. ComfyUI "
                                "must accept the checkpoint before LM Atelier activates it."
                            ),
                        )
                        if check.id == "runtime"
                        else check
                    )
                    for check in result.checks
                ],
                CatalogPreflightCheck(
                    id="workflow-template",
                    label="Automatic runtime setup",
                    status="warn",
                    detail=(
                        "LM Atelier will use ComfyUI's standard checkpoint workflow and "
                        "keep the model inactive if live runtime validation fails."
                    ),
                ),
            ]
            return await finalize(
                result.model_copy(
                    update={
                        "source_remote_id": detail.model.remote_id,
                        "comfy_paths": adaptive.comfy_paths,
                        "workflow_template_id": adaptive.id,
                        "workflow_template_sha256": adaptive.sha256,
                        "checks": checks,
                    }
                ),
                detail,
            )
        checks = [
            *[check for check in result.checks if check.id != "selection"],
            CatalogPreflightCheck(
                id="workflow-template",
                label="Automatic runtime setup",
                status="block",
                detail=(
                    "No official workflow or safe adaptive standard-checkpoint contract "
                    "is available for this model."
                ),
            ),
        ]
        return await finalize(
            result.model_copy(
                update={
                    "source_remote_id": detail.model.remote_id,
                    "selected_files": [],
                    "expected_sha256": {},
                    "download_bytes": 0,
                    "can_install": False,
                    "checks": checks,
                }
            ),
            detail,
        )

    inspected: dict[tuple[str, str], CatalogDetail] = {}
    viable: list[tuple[int, ComfyTemplate, CatalogDetail]] = []
    best_score = candidates[0].score
    for candidate in candidates:
        if candidate.score != best_score:
            break
        key = (candidate.remote_id, candidate.revision)
        resolved = inspected.get(key)
        if not resolved:
            try:
                raw_resolved = await services.catalog.inspect(
                    candidate.remote_id,
                    candidate.revision,
                    payload.role,
                )
                resolved = CatalogDetail.model_validate(raw_resolved)
            except Exception:
                continue
            inspected[key] = resolved
        files = {str(item.get("filename") or ""): item for item in resolved.files}
        if not all(filename in files for filename in candidate.selected_files):
            continue
        size = sum(int(files[filename].get("size") or 0) for filename in candidate.selected_files)
        viable.append((size, candidate, resolved))
    if not viable:
        result = assess_catalog_install(detail, payload, services.settings, system)
        checks = [
            *[check for check in result.checks if check.id != "selection"],
            CatalogPreflightCheck(
                id="workflow-template",
                label="Automatic runtime setup",
                status="block",
                detail=(
                    "ComfyUI advertises a matching workflow, but its complete safe model "
                    "bundle is not available from the declared repository."
                ),
            ),
        ]
        return await finalize(
            result.model_copy(
                update={
                    "source_remote_id": detail.model.remote_id,
                    "selected_files": [],
                    "expected_sha256": {},
                    "download_bytes": 0,
                    "can_install": False,
                    "checks": checks,
                }
            ),
            detail,
        )

    _, template, resolved_detail = min(viable, key=lambda item: (item[0], item[1].id))
    resolved_payload = payload.model_copy(
        update={
            "revision": template.revision,
            "selected_files": template.selected_files,
        }
    )
    result = assess_catalog_install(
        resolved_detail,
        resolved_payload,
        services.settings,
        system,
    )
    checks = [
        *[
            (
                CatalogPreflightCheck(
                    id="runtime",
                    label="Runtime compatibility",
                    status="pass",
                    detail=(
                        "The selected files match the official workflow shipped with "
                        "the active ComfyUI runtime."
                    ),
                )
                if check.id == "runtime"
                else check
            )
            for check in result.checks
        ],
        CatalogPreflightCheck(
            id="workflow-template",
            label="Automatic runtime setup",
            status="pass",
            detail=(
                "LM Atelier found a complete official ComfyUI model bundle and workflow; "
                "both will be configured automatically."
            ),
        ),
    ]
    return await finalize(
        result.model_copy(
            update={
                "source_remote_id": detail.model.remote_id,
                "comfy_paths": template.comfy_paths,
                "workflow_template_id": template.id,
                "workflow_template_sha256": template.sha256,
                "checks": checks,
            }
        ),
        resolved_detail,
    )


@router.post("/downloads", response_model=JobOut, status_code=202)
async def create_download(payload: DownloadRequest, request: Request, session: SessionDep) -> Job:
    manager: DownloadManager = _services(request).downloads
    try:
        if payload.install_plan_id:
            plan = session.get(InstallPlan, payload.install_plan_id)
            if not plan:
                raise ValueError("install plan not found; run the install check again")
            if plan.status != "planned":
                raise ValueError("install plan is no longer active; run the install check again")
            if plan.compatibility != "supported":
                raise ValueError(plan.failure_reason or "this model layout is unsupported")
            planned_files = [
                str(item.get("path") or "")
                for item in plan.artifacts_json
                if item.get("required", True)
            ]
            planned_hashes = {
                str(item["path"]): str(item["sha256"])
                for item in plan.artifacts_json
                if item.get("sha256")
            }
            runtime = plan.runtime_contract_json
            if (
                runtime.get("workflow_template_id")
                and runtime.get("workflow_compiler_version") != COMFY_TEMPLATE_COMPILER_VERSION
            ):
                raise ValueError("workflow contract changed; run the install check again")
            expected = {
                "remote_id": plan.remote_id,
                "revision": plan.revision,
                "role": plan.role,
                "engine": plan.engine,
                "allow_patterns": planned_files,
                "expected_sha256": planned_hashes,
                "file_sources": {
                    str(item["path"]): {
                        "remote_id": str(item["source_remote_id"]),
                        "revision": str(item["source_revision"]),
                        "filename": str(item["source_path"]),
                        "size_bytes": item.get("size_bytes"),
                        "sha256": item.get("sha256"),
                    }
                    for item in plan.artifacts_json
                    if item.get("source_remote_id")
                    and item.get("source_revision")
                    and item.get("source_path")
                },
                "source_remote_id": runtime.get("source_remote_id"),
                "comfy_paths": runtime.get("comfy_paths") or {},
                "workflow_template_id": runtime.get("workflow_template_id"),
                "workflow_template_sha256": runtime.get("workflow_template_sha256"),
                "auxiliary_kind": runtime.get("auxiliary_kind"),
            }
            supplied = payload.model_dump()
            mismatched = [key for key, value in expected.items() if supplied.get(key) != value]
            if mismatched:
                raise ValueError(
                    "install request no longer matches its immutable plan; "
                    "run the install check again"
                )
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
async def list_models(request: Request, session: SessionDep) -> list[ModelInstallOut]:
    installs = list(
        session.scalars(
            select(ModelInstall)
            .where(ModelInstall.active.is_(True))
            .order_by(ModelInstall.updated_at.desc())
        ).all()
    )
    services = _services(request)
    evidence_by_install = {
        install.id: evidence
        for install in installs
        if (
            evidence := current_capability_evidence(
                session,
                install,
                services.settings,
                services.runtimes,
            )
        )
    }
    return [
        ModelInstallOut.model_validate(install).model_copy(
            update={
                "readiness": (
                    "ready"
                    if install.id in evidence_by_install
                    else (
                        "unsupported"
                        if install.compatibility == CompatibilityLevel.UNSUPPORTED.value
                        else "unverified"
                    )
                ),
                "capability_evidence": (
                    ModelCapabilityEvidenceOut.model_validate(evidence_by_install[install.id])
                    if install.id in evidence_by_install
                    else None
                ),
            }
        )
        for install in installs
    ]


@router.get("/model-assets", response_model=list[ModelAssetOut])
async def list_model_assets(
    session: SessionDep,
    kind: str | None = None,
) -> list[ModelAssetInstall]:
    statement = select(ModelAssetInstall)
    if kind:
        if kind not in COMFY_AUXILIARY_FOLDERS:
            raise HTTPException(422, "unsupported model asset kind")
        statement = statement.where(ModelAssetInstall.kind == kind)
    return list(
        session.scalars(statement.order_by(ModelAssetInstall.name, ModelAssetInstall.id)).all()
    )


@router.patch("/model-assets/{asset_id}", response_model=ModelAssetOut)
async def update_model_asset(
    asset_id: str,
    payload: ModelAssetUpdate,
    request: Request,
    session: SessionDep,
) -> ModelAssetInstall:
    services = _services(request)
    asset = session.get(ModelAssetInstall, asset_id)
    if not asset:
        raise HTTPException(404, "model asset not found")
    if payload.active and not asset.verified_at:
        raise HTTPException(409, "only a verified model asset can be enabled")
    async with services.scheduler.lease("primary"):
        previous_active = asset.active
        was_running = next(
            worker.running for worker in services.processes.statuses() if worker.name == "media"
        )
        asset.active = payload.active
        session.commit()
        if was_running:
            try:
                await services.processes.start_media()
            except Exception:
                asset.active = previous_active
                session.commit()
                with suppress(Exception):
                    await services.processes.start_media()
                raise
    session.refresh(asset)
    return asset


@router.delete("/model-assets/{asset_id}", status_code=204)
async def delete_model_asset(
    asset_id: str,
    request: Request,
    session: SessionDep,
) -> Response:
    services = _services(request)
    async with services.scheduler.lease("primary"):
        asset = session.get(ModelAssetInstall, asset_id)
        if not asset:
            raise HTTPException(404, "model asset not found")
        was_running = next(
            worker.running for worker in services.processes.statuses() if worker.name == "media"
        )
        deletion_error: BaseException | None = None
        try:
            if was_running:
                await services.processes.stop("media")
            quarantine = _delete_model_asset_locked(asset, services.settings.model_dir, session)
            try:
                await asyncio.to_thread(_finalize_model_quarantine, quarantine)
            except Exception:
                logger.warning(
                    "Deleted auxiliary model files remain safely quarantined at %s",
                    quarantine,
                    exc_info=True,
                )
        except BaseException as exc:
            deletion_error = exc
            raise
        finally:
            if was_running and not next(
                worker.running for worker in services.processes.statuses() if worker.name == "media"
            ):
                try:
                    await services.processes.start_media()
                except Exception:
                    if deletion_error is None:
                        raise
                    logger.exception(
                        "Could not restore the media worker after auxiliary deletion failed"
                    )
    return Response(status_code=204)


def _delete_model_asset_locked(
    asset: ModelAssetInstall,
    model_dir: Path,
    session: Session,
) -> Path | None:
    model_root = model_dir.resolve()
    try:
        path = _managed_model_path(model_root, asset.local_path)
        recover_model_delete_quarantines(session, model_root, strict=True)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    moves: list[tuple[Path, Path]] = []
    quarantine: Path | None = None
    commit_started = False
    try:
        session.flush()
        if (
            path is not None
            and path.exists()
            and not _model_asset_path_is_shared(
                session,
                asset,
                model_root,
                path,
            )
        ):
            _ensure_model_tree_link_free(path)
            quarantine = _new_model_quarantine(model_root, asset.id)
            staged = quarantine / "payload"
            os.replace(path, staged)
            moves.append((staged, path))
        session.delete(asset)
        session.flush()
        commit_started = True
        session.commit()
    except Exception:
        with suppress(Exception):
            session.rollback()
        committed = _model_asset_delete_was_committed(asset.id) if commit_started else False
        if committed is True:
            return quarantine
        if committed is None:
            logger.error(
                "Could not determine auxiliary deletion outcome; leaving files in quarantine %s",
                quarantine,
                exc_info=True,
            )
            raise
        try:
            _restore_model_moves(moves)
        except Exception:
            logger.exception(
                "Auxiliary deletion rollback left recoverable files in quarantine %s",
                quarantine,
            )
        else:
            with suppress(Exception):
                _finalize_model_quarantine(quarantine)
        raise
    return quarantine


def _model_asset_path_is_shared(
    session: Session,
    asset: ModelAssetInstall,
    model_root: Path,
    path: Path,
) -> bool:
    candidates: list[ModelInstall | ModelAssetInstall] = [
        *session.scalars(select(ModelInstall)).all(),
        *session.scalars(select(ModelAssetInstall).where(ModelAssetInstall.id != asset.id)).all(),
    ]
    for candidate in candidates:
        try:
            sibling = _managed_model_path(model_root, candidate.local_path)
        except ValueError:
            return True
        if sibling is not None and (
            sibling == path or sibling in path.parents or path in sibling.parents
        ):
            return True
    return False


def _model_asset_delete_was_committed(asset_id: str) -> bool | None:
    try:
        with SessionLocal() as verification:
            return verification.get(ModelAssetInstall, asset_id) is None
    except Exception:
        logger.exception("Could not verify the auxiliary model deletion database outcome")
        return None


@router.get("/models/storage", response_model=ModelStorageInfo)
async def model_storage(request: Request, session: SessionDep) -> ModelStorageInfo:
    settings = _services(request).settings
    partials = list(settings.download_dir.glob("*.partial"))
    return ModelStorageInfo(
        installed_bytes=_path_size(settings.model_dir),
        partial_download_bytes=sum(_path_size(path) for path in partials),
        catalog_cache_bytes=_path_size(settings.catalog_cache_dir),
        installed_count=(
            session.scalar(select(func.count(ModelInstall.id)).where(ModelInstall.active.is_(True)))
            or 0
        )
        + (
            session.scalar(
                select(func.count(ModelAssetInstall.id)).where(
                    ModelAssetInstall.active.is_(True),
                    ModelAssetInstall.verified_at.is_not(None),
                )
            )
            or 0
        ),
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
    session.flush()
    ensure_profile_for_install(session, install)
    session.commit()
    session.refresh(install)
    return install


@router.delete("/models/{model_id}", status_code=204)
async def delete_model(
    model_id: str,
    request: Request,
    session: SessionDep,
    delete_profiles: bool = False,
) -> Response:
    services = _services(request)
    async with services.scheduler.lease("primary"):
        install = session.get(ModelInstall, model_id)
        if not install:
            raise HTTPException(404, "model not found")
        worker_name = "chat" if install.role == ModelRole.CHAT.value else "media"
        _ensure_worker_idle(session, worker_name)
        if worker_name == "media" and any(
            worker.name == "media" and worker.running for worker in services.processes.statuses()
        ):
            raise HTTPException(409, "stop the media worker before deleting this model")
        quarantine = _delete_model_locked(
            model_id,
            request,
            session,
            delete_profiles=delete_profiles,
        )
        try:
            await asyncio.to_thread(_finalize_model_quarantine, quarantine)
        except Exception:
            # The database commit is authoritative. Startup recovery will
            # finish this same-volume deletion without another user action.
            logger.warning(
                "Deleted model files remain safely quarantined at %s",
                quarantine,
                exc_info=True,
            )
        return Response(status_code=204)


def _delete_model_locked(
    model_id: str,
    request: Request,
    session: Session,
    *,
    delete_profiles: bool,
) -> Path | None:
    install = session.get(ModelInstall, model_id)
    if not install:
        raise HTTPException(404, "model not found")
    model_root = _services(request).settings.model_dir.resolve()
    try:
        path = _managed_model_path(model_root, install.local_path)
        recover_model_delete_quarantines(session, model_root, strict=True)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    profiles = list(
        session.scalars(
            select(ModelProfile).where(ModelProfile.model_install_id == install.id)
        ).all()
    )
    if profiles and not delete_profiles:
        raise HTTPException(409, "delete profiles that use this model before deleting it")
    profile_ids = {profile.id for profile in profiles}
    if profile_ids and any(
        worker.running and worker.profile_id in profile_ids
        for worker in _services(request).processes.statuses()
    ):
        raise HTTPException(409, "unload the active worker before deleting this model")
    if profile_ids:
        affected_chats = session.scalars(
            select(Chat).where(
                or_(
                    Chat.active_chat_profile_id.in_(profile_ids),
                    Chat.active_vision_profile_id.in_(profile_ids),
                    Chat.active_image_profile_id.in_(profile_ids),
                    Chat.active_video_profile_id.in_(profile_ids),
                )
            )
        ).all()
        for chat in affected_chats:
            if chat.active_chat_profile_id in profile_ids:
                chat.active_chat_profile_id = AUTO_PROFILE_ID
            if chat.active_vision_profile_id in profile_ids:
                chat.active_vision_profile_id = AUTO_PROFILE_ID
            if chat.active_image_profile_id in profile_ids:
                chat.active_image_profile_id = AUTO_PROFILE_ID
            if chat.active_video_profile_id in profile_ids:
                chat.active_video_profile_id = AUTO_PROFILE_ID
        for profile in profiles:
            session.delete(profile)
    moves: list[tuple[Path, Path]] = []
    quarantine: Path | None = None
    commit_started = False
    try:
        # Flush relationship and chat changes before touching the filesystem.
        # A constraint failure here therefore cannot move any model content.
        session.flush()
        if path is not None:
            moves, quarantine = _quarantine_model_files(
                session,
                install,
                model_root,
                path,
            )
        session.delete(install)
        # Force database errors while the quarantined files are still
        # restorable, rather than deferring every check into commit().
        session.flush()
        commit_started = True
        session.commit()
    except Exception as exc:
        try:
            session.rollback()
        except Exception:
            logger.exception("Database rollback failed during model deletion")
        if commit_started:
            committed = _model_delete_was_committed(model_id)
            if committed is True:
                logger.warning(
                    "Model deletion commit completed despite a reported database error",
                    exc_info=True,
                )
                return quarantine
            if committed is None:
                logger.error(
                    "Could not determine model deletion outcome; leaving files in quarantine %s",
                    quarantine,
                    exc_info=True,
                )
                raise
        try:
            _restore_model_moves(moves)
        except Exception:
            logger.exception(
                "Model deletion rollback left recoverable files in quarantine %s",
                quarantine,
            )
        else:
            try:
                _finalize_model_quarantine(quarantine)
            except Exception:
                logger.warning(
                    "Restored model files but could not prune quarantine %s",
                    quarantine,
                    exc_info=True,
                )
        if isinstance(exc, ValueError):
            raise HTTPException(422, str(exc)) from exc
        raise
    return quarantine


_MODEL_DELETE_QUARANTINE = ".delete-pending"
_MODEL_DELETE_MARKER = ".model-id"


def _model_delete_was_committed(model_id: str) -> bool | None:
    """Resolve an ambiguous commit error from a fresh database transaction."""

    try:
        with SessionLocal() as verification:
            return verification.get(ModelInstall, model_id) is None
    except Exception:
        logger.exception("Could not verify the model deletion database outcome")
        return None


def _managed_model_path(model_root: Path, value: str) -> Path | None:
    """Return a confined, link-free managed path or None for external imports."""

    raw = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    try:
        relative = raw.relative_to(model_root)
    except ValueError:
        return None
    if not relative.parts or relative.parts[0] == _MODEL_DELETE_QUARANTINE:
        return None
    cursor = model_root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and _model_path_is_link(cursor):
            raise ValueError("managed model paths cannot use filesystem links")
    resolved = raw.resolve(strict=False)
    if model_root not in resolved.parents or resolved == model_root:
        raise ValueError("managed model path escapes model storage")
    return resolved


def _model_path_is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _ensure_model_tree_link_free(path: Path) -> None:
    if _model_path_is_link(path):
        raise ValueError("managed model paths cannot use filesystem links")
    if not path.is_dir():
        return
    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                candidate = Path(entry.path)
                if _model_path_is_link(candidate):
                    raise ValueError("managed model directories cannot contain filesystem links")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(candidate)


def _quarantine_model_files(
    session: Session,
    install: ModelInstall,
    model_root: Path,
    path: Path,
) -> tuple[list[tuple[Path, Path]], Path | None]:
    if not path.exists():
        return [], None
    siblings = _related_model_installs(session, install, model_root, path)
    quarantine: Path | None = None
    moves: list[tuple[Path, Path]] = []
    try:
        if not siblings:
            _ensure_model_tree_link_free(path)
            quarantine = _new_model_quarantine(model_root, install.id)
            staged = quarantine / "payload"
            os.replace(path, staged)
            return [(staged, path)], quarantine

        if not path.is_dir():
            # Another install owns the same managed file.
            return [], None

        retained_paths, retained_roots = _retained_model_paths(
            siblings,
            model_root,
        )
        for relative in _manifest_model_files(install):
            candidate = _confined_manifest_file(path, relative)
            if not candidate.exists():
                continue
            if _model_path_is_link(candidate):
                raise ValueError("managed model files cannot use filesystem links")
            if not candidate.is_file():
                raise ValueError("managed model manifests may only reference files")
            if candidate in retained_paths or any(
                candidate == root or root in candidate.parents for root in retained_roots
            ):
                continue
            if quarantine is None:
                quarantine = _new_model_quarantine(model_root, install.id)
            staged = _safe_quarantine_file_path(quarantine, relative)
            os.replace(candidate, staged)
            moves.append((staged, candidate))
        return moves, quarantine
    except Exception:
        try:
            _restore_model_moves(moves)
        except Exception:
            logger.exception(
                "Model staging failure left recoverable files in quarantine %s",
                quarantine,
            )
        else:
            try:
                _finalize_model_quarantine(quarantine)
            except Exception:
                logger.warning(
                    "Could not prune a failed model deletion quarantine %s",
                    quarantine,
                    exc_info=True,
                )
        raise


def _related_model_installs(
    session: Session,
    install: ModelInstall,
    model_root: Path,
    path: Path,
) -> list[tuple[ModelInstall, Path]]:
    related: list[tuple[ModelInstall, Path]] = []
    for sibling in session.scalars(select(ModelInstall).where(ModelInstall.id != install.id)).all():
        try:
            sibling_path = _managed_model_path(model_root, sibling.local_path)
        except ValueError:
            # A linked sibling is never evidence that it is safe to remove
            # content through the current install.
            continue
        if sibling_path is None:
            continue
        if sibling_path == path or sibling_path in path.parents or path in sibling_path.parents:
            related.append((sibling, sibling_path))
    return related


def _retained_model_paths(
    siblings: list[tuple[ModelInstall, Path]],
    model_root: Path,
) -> tuple[set[Path], set[Path]]:
    retained_paths: set[Path] = set()
    retained_roots: set[Path] = set()
    for sibling, sibling_path in siblings:
        if sibling_path.is_file():
            retained_paths.add(sibling_path)
            continue
        try:
            files = _manifest_model_files(sibling)
        except ValueError:
            retained_roots.add(sibling_path)
            continue
        if not files:
            retained_roots.add(sibling_path)
            continue
        for relative in files:
            try:
                candidate = _confined_manifest_file(sibling_path, relative)
            except ValueError:
                retained_roots.add(sibling_path)
                break
            if candidate == model_root or model_root not in candidate.parents:
                retained_roots.add(sibling_path)
                break
            retained_paths.add(candidate)
    return retained_paths, retained_roots


def _manifest_model_files(install: ModelInstall) -> list[PurePosixPath]:
    raw_files = install.manifest_json.get("files", [])
    if not isinstance(raw_files, list):
        raise ValueError("managed model manifest files must be a list")
    files: list[PurePosixPath] = []
    seen: set[str] = set()
    for value in raw_files:
        if not isinstance(value, str) or not value:
            raise ValueError("managed model manifest contains an invalid file path")
        relative = PurePosixPath(value.replace("\\", "/"))
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} or ":" in part for part in relative.parts)
        ):
            raise ValueError("managed model manifest contains an unsafe file path")
        identity = relative.as_posix().casefold()
        if identity in seen:
            continue
        seen.add(identity)
        files.append(relative)
    return files


def _confined_manifest_file(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and _model_path_is_link(cursor):
            raise ValueError("managed model files cannot use filesystem links")
    resolved = candidate.resolve(strict=False)
    if root not in resolved.parents:
        raise ValueError("managed model manifest path escapes its install directory")
    return resolved


def _new_model_quarantine(model_root: Path, model_id: str) -> Path:
    parent = model_root / _MODEL_DELETE_QUARANTINE
    if parent.exists() and _model_path_is_link(parent):
        raise ValueError("model deletion quarantine cannot use a filesystem link")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or parent.resolve().parent != model_root:
        raise ValueError("model deletion quarantine escapes model storage")
    quarantine = parent / new_id("delete")
    quarantine.mkdir(mode=0o700)
    try:
        marker = quarantine / _MODEL_DELETE_MARKER
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(model_id)
            handle.flush()
            os.fsync(handle.fileno())
        marker.chmod(0o600)
    except Exception:
        with suppress(OSError):
            quarantine.rmdir()
        raise
    return quarantine


def _safe_quarantine_file_path(
    quarantine: Path,
    relative: PurePosixPath,
) -> Path:
    if _model_path_is_link(quarantine) or not quarantine.is_dir():
        raise ValueError("model deletion quarantine contains a filesystem link")
    files = quarantine / "files"
    if _model_path_is_link(files):
        raise ValueError("model deletion quarantine contains a filesystem link")
    if not files.exists():
        files.mkdir()
    if _model_path_is_link(files) or not files.is_dir():
        raise ValueError("model deletion quarantine contains an unsafe files directory")
    files_root = files.resolve()
    if files_root.parent != quarantine.resolve():
        raise ValueError("model deletion quarantine escapes model storage")
    cursor = files
    for part in relative.parts[:-1]:
        cursor /= part
        if _model_path_is_link(cursor):
            raise ValueError("model deletion quarantine contains a filesystem link")
        if cursor.exists():
            if not cursor.is_dir():
                raise ValueError("model deletion quarantine contains a filesystem link")
        else:
            cursor.mkdir()
            if _model_path_is_link(cursor) or not cursor.is_dir():
                raise ValueError("model deletion quarantine contains a filesystem link")
    staged = cursor / relative.name
    if staged.exists() or _model_path_is_link(staged):
        raise ValueError("model deletion quarantine contains an unexpected file")
    resolved_parent = staged.parent.resolve()
    if resolved_parent != files_root and files_root not in resolved_parent.parents:
        raise ValueError("model deletion quarantine escapes model storage")
    return staged


def _restore_model_moves(moves: list[tuple[Path, Path]]) -> None:
    for staged, original in reversed(moves):
        if not staged.exists():
            continue
        if original.exists():
            raise RuntimeError("cannot restore a quarantined model over an existing path")
        original.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, original)


def _finalize_model_quarantine(quarantine: Path | None) -> None:
    if quarantine is None or not quarantine.exists():
        return
    if _model_path_is_link(quarantine) or not quarantine.is_dir():
        raise OSError("refusing to follow an unsafe model deletion quarantine")
    marker = quarantine / _MODEL_DELETE_MARKER
    if _model_path_is_link(marker) or not marker.is_file():
        raise OSError("model deletion quarantine has no safe ownership marker")
    for child in quarantine.iterdir():
        if child == marker:
            continue
        if _model_path_is_link(child):
            raise OSError("refusing to follow a link in model deletion quarantine")
        if child.is_dir():
            try:
                _ensure_model_tree_link_free(child)
            except ValueError as exc:
                raise OSError("refusing to follow a link in model deletion quarantine") from exc
            shutil.rmtree(child)
        elif child.is_file():
            child.unlink()
        else:
            raise OSError("model deletion quarantine contains an unsupported entry")
    # Remove the ownership marker last. If payload cleanup fails partway
    # through, the next recovery pass can still determine whether to restore
    # the remaining files or finish deleting them.
    marker.unlink(missing_ok=True)
    quarantine.rmdir()
    with suppress(OSError):
        quarantine.parent.rmdir()


def recover_model_delete_quarantines(
    session: Session,
    model_root: Path,
    *,
    strict: bool = False,
) -> None:
    parent = model_root / _MODEL_DELETE_QUARANTINE
    if not parent.exists():
        return
    if _model_path_is_link(parent) or not parent.is_dir():
        raise ValueError("model deletion quarantine is not a safe directory")
    for quarantine in list(parent.iterdir()):
        if not quarantine.name.startswith("delete_"):
            continue
        try:
            if not quarantine.is_dir() or _model_path_is_link(quarantine):
                raise ValueError("model deletion quarantine contains a filesystem link")
            marker = quarantine / _MODEL_DELETE_MARKER
            if not marker.exists():
                if not any(quarantine.iterdir()):
                    quarantine.rmdir()
                    continue
                raise ValueError("model deletion quarantine has no ownership marker")
            if _model_path_is_link(marker) or not marker.is_file():
                raise ValueError("model deletion quarantine contains an unsafe marker")
            marker_model_id = marker.read_text(encoding="utf-8")
            if (
                not marker_model_id
                or marker_model_id != marker_model_id.strip()
                or len(marker_model_id) > 200
            ):
                raise ValueError("model deletion quarantine has an invalid owner")
            install: ModelInstall | ModelAssetInstall | None = session.get(
                ModelInstall, marker_model_id
            )
            if install is None:
                install = session.get(ModelAssetInstall, marker_model_id)
            if install is None:
                _finalize_model_quarantine(quarantine)
                continue
            path = _managed_model_path(model_root, install.local_path)
            if path is None:
                raise ValueError("model deletion quarantine belongs to an external model")
            _restore_model_quarantine(quarantine, path)
        except (OSError, UnicodeError, ValueError):
            if strict:
                raise
            logger.warning(
                "Could not reconcile model deletion quarantine %s",
                quarantine,
                exc_info=True,
            )
    with suppress(OSError):
        parent.rmdir()


def _restore_model_quarantine(quarantine: Path, path: Path) -> None:
    moves: list[tuple[Path, Path]] = []
    payload = quarantine / "payload"
    if payload.exists():
        _ensure_model_tree_link_free(payload)
        moves.append((payload, path))
    files = quarantine / "files"
    if files.exists():
        if _model_path_is_link(files) or not files.is_dir():
            raise ValueError("model deletion quarantine contains an unsafe files directory")
        _ensure_model_tree_link_free(files)
        for staged in files.rglob("*"):
            if _model_path_is_link(staged):
                raise ValueError("model deletion quarantine contains a filesystem link")
            if not staged.is_file():
                continue
            relative = staged.relative_to(files)
            original = _confined_manifest_file(
                path,
                PurePosixPath(*relative.parts),
            )
            moves.append((staged, original))
    try:
        _restore_model_moves(moves)
    except RuntimeError as exc:
        raise ValueError("model deletion recovery needs manual conflict resolution") from exc
    _finalize_model_quarantine(quarantine)


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@router.get("/profiles", response_model=list[ModelProfileOut])
async def list_profiles(
    request: Request,
    session: SessionDep,
    role: str | None = None,
) -> list[ModelProfileOut]:
    statement = (
        select(ModelProfile)
        .outerjoin(ModelInstall, ModelInstall.id == ModelProfile.model_install_id)
        .where(
            or_(
                ModelProfile.model_install_id.is_(None),
                and_(
                    ModelInstall.active.is_(True),
                    ModelInstall.role == ModelProfile.role,
                    ModelInstall.engine == ModelProfile.engine,
                ),
            )
        )
        .order_by(ModelProfile.role, ModelProfile.name)
    )
    if role:
        statement = statement.where(ModelProfile.role == role)
    services = _services(request)
    results: list[ModelProfileOut] = []
    for profile in session.scalars(statement).all():
        evidence = None
        if profile.model_install_id:
            install = session.get(ModelInstall, profile.model_install_id)
            if install:
                evidence = current_capability_evidence(
                    session,
                    install,
                    services.settings,
                    services.runtimes,
                )
        results.append(
            ModelProfileOut.model_validate(profile).model_copy(
                update={"input_modalities": evidence_input_modalities(evidence)}
            )
        )
    return results


@router.post("/profiles", response_model=ModelProfileOut, status_code=201)
async def create_profile(
    payload: ModelProfileCreate,
    request: Request,
    session: SessionDep,
) -> ModelProfile:
    _validated_profile_install(
        session,
        model_install_id=payload.model_install_id,
        role=payload.role,
        engine=payload.engine,
    )
    fields = await _engine_role_fields(
        request,
        payload.role,
        engine=payload.engine,
        allow_inactive=True,
    )
    if payload.is_default:
        for profile in session.scalars(
            select(ModelProfile).where(ModelProfile.role == payload.role)
        ).all():
            profile.is_default = False
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
        use_case=payload.use_case,
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
    profile_id: str,
    payload: ModelProfileUpdate,
    request: Request,
    session: SessionDep,
) -> ModelProfile:
    profile = session.get(ModelProfile, profile_id)
    if not profile:
        raise HTTPException(404, "profile not found")
    try:
        validate_profile_binding(session, profile)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    values = payload.model_dump(exclude_unset=True)
    fields = (
        await _engine_role_fields(
            request,
            profile.role,
            engine=profile.engine,
            allow_inactive=True,
        )
        if {"load_settings", "request_settings"} & values.keys()
        else []
    )
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
                [field for field in fields if field.scope == "load"],
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if "request_settings" in values:
        try:
            profile.request_settings_json = validate_settings(
                values.pop("request_settings") or {},
                [field for field in fields if field.scope != "load"],
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
    services = _services(request)
    async with services.scheduler.lease("primary"):
        profile = session.get(ModelProfile, profile_id)
        if not profile:
            raise HTTPException(404, "profile not found")
        worker_name = "chat" if profile.role == ModelRole.CHAT.value else "media"
        _ensure_worker_idle(session, worker_name)
        if any(
            worker.running and worker.profile_id == profile.id
            for worker in services.processes.statuses()
        ):
            raise HTTPException(409, "unload the active worker before deleting its profile")
        session.delete(profile)
        session.commit()
        return Response(status_code=204)


@router.post("/profiles/{profile_id}/clone", response_model=ModelProfileOut, status_code=201)
async def clone_profile(
    profile_id: str,
    payload: ModelProfileClone,
    request: Request,
    session: SessionDep,
) -> ModelProfile:
    source = session.get(ModelProfile, profile_id)
    if not source:
        raise HTTPException(404, "profile not found")
    return await create_profile(
        ModelProfileCreate(
            name=payload.name or f"{source.name} copy",
            use_case=source.use_case,
            role=cast(Literal["chat", "image", "video"], source.role),
            engine=source.engine,
            model_install_id=source.model_install_id,
            load_settings=source.load_settings_json,
            request_settings=source.request_settings_json,
        ),
        request,
        session,
    )


@router.post("/profiles/{profile_id}/reset", response_model=ModelProfileOut)
async def reset_profile(profile_id: str, session: SessionDep) -> ModelProfile:
    profile = session.get(ModelProfile, profile_id)
    if not profile:
        raise HTTPException(404, "profile not found")
    try:
        validate_profile_binding(session, profile)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
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
        use_case=profile.use_case,
        role=cast(Literal["chat", "image", "video"], profile.role),
        engine=profile.engine,
        model_install_id=profile.model_install_id,
        load_settings=profile.load_settings_json,
        request_settings=profile.request_settings_json,
    )


@router.post("/profiles/import", response_model=ModelProfileOut, status_code=201)
async def import_profile(
    payload: ModelProfileBundle,
    request: Request,
    session: SessionDep,
) -> ModelProfile:
    return await create_profile(
        ModelProfileCreate(
            name=payload.name,
            use_case=payload.use_case,
            role=payload.role,
            engine=payload.engine,
            model_install_id=payload.model_install_id,
            load_settings=payload.load_settings,
            request_settings=payload.request_settings,
        ),
        request,
        session,
    )


@router.get("/presets", response_model=list[PresetOut])
async def list_presets(session: SessionDep, role: str | None = None) -> list[GenerationPreset]:
    statement = select(GenerationPreset).order_by(GenerationPreset.role, GenerationPreset.name)
    if role:
        statement = statement.where(GenerationPreset.role == role)
    return list(session.scalars(statement).all())


@router.post("/presets", response_model=PresetOut, status_code=201)
async def create_preset(
    payload: PresetCreate,
    request: Request,
    session: SessionDep,
) -> GenerationPreset:
    fields = await _engine_role_fields(request, payload.role)
    try:
        values = validate_settings(
            payload.settings,
            [field for field in fields if field.scope != "load"],
        )
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
    preset_id: str,
    payload: PresetUpdate,
    request: Request,
    session: SessionDep,
) -> GenerationPreset:
    preset = session.get(GenerationPreset, preset_id)
    if not preset:
        raise HTTPException(404, "preset not found")
    values = payload.model_dump(exclude_unset=True)
    if "settings" in values:
        fields = await _engine_role_fields(request, preset.role)
        try:
            preset.settings_json = validate_settings(
                values.pop("settings") or {},
                [field for field in fields if field.scope != "load"],
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
    owners: list[Project | Chat] = []
    owners.extend(session.scalars(select(Project)).all())
    owners.extend(session.scalars(select(Chat)).all())
    for owner in owners:
        bindings = (
            dict(owner.generation_preset_ids_json)
            if isinstance(owner.generation_preset_ids_json, dict)
            else {}
        )
        if bindings.get(preset.role) != preset.id:
            continue
        bindings.pop(preset.role, None)
        scoped = (
            dict(owner.generation_settings_json)
            if isinstance(owner.generation_settings_json, dict)
            else {}
        )
        direct = scoped.get(preset.role)
        scoped[preset.role] = {
            **preset.settings_json,
            **(direct if isinstance(direct, dict) else {}),
        }
        owner.generation_preset_ids_json = bindings
        owner.generation_settings_json = scoped
    session.delete(preset)
    session.commit()
    return Response(status_code=204)


@router.post("/presets/{preset_id}/clone", response_model=PresetOut, status_code=201)
async def clone_preset(
    preset_id: str,
    payload: PresetClone,
    request: Request,
    session: SessionDep,
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
        request,
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
async def import_preset(
    payload: PresetBundle,
    request: Request,
    session: SessionDep,
) -> GenerationPreset:
    return await create_preset(
        PresetCreate(name=payload.name, role=payload.role, settings=payload.settings),
        request,
        session,
    )


def _require_media_worker_stopped(request: Request) -> None:
    if any(
        worker.name == "media" and worker.running
        for worker in _services(request).processes.statuses()
    ):
        raise HTTPException(409, "stop the media worker before changing custom nodes")


async def _custom_node_lifecycle(
    request: Request,
    session: SessionDep,
) -> AsyncIterator[None]:
    services = _services(request)
    async with services.scheduler.lease("primary"):
        _ensure_worker_idle(session, "media")
        _require_media_worker_stopped(request)
        yield


CustomNodeLifecycleDep = Annotated[None, Depends(_custom_node_lifecycle)]


@router.get("/custom-nodes", response_model=list[CustomNodeOut])
async def list_custom_nodes(session: SessionDep) -> list[CustomNodeInstall]:
    return list(session.scalars(select(CustomNodeInstall).order_by(CustomNodeInstall.name)).all())


@router.post("/custom-nodes", response_model=CustomNodeOut, status_code=201)
async def install_custom_node(
    payload: CustomNodeInstallRequest,
    request: Request,
    session: SessionDep,
    _lifecycle: CustomNodeLifecycleDep,
) -> CustomNodeInstall:
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
    node_id: str,
    payload: CustomNodeUpdateRequest,
    request: Request,
    session: SessionDep,
    _lifecycle: CustomNodeLifecycleDep,
) -> CustomNodeInstall:
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
    node_id: str,
    payload: CustomNodeTrustRequest,
    request: Request,
    session: SessionDep,
    _lifecycle: CustomNodeLifecycleDep,
) -> CustomNodeInstall:
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
    node_id: str,
    request: Request,
    session: SessionDep,
    _lifecycle: CustomNodeLifecycleDep,
) -> CustomNodeInstall:
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
async def remove_custom_node(
    node_id: str,
    request: Request,
    session: SessionDep,
    _lifecycle: CustomNodeLifecycleDep,
) -> Response:
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
    try:
        validate_lora_workflow_contract(
            payload.api_graph,
            payload.input_schema,
            payload.dependencies,
        )
        validate_workflow_edit_calibration(payload.input_schema)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
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
            trusted=False,
        ),
        session,
    )


@router.post("/workflows/{workflow_id}/clone", response_model=WorkflowOut, status_code=201)
async def clone_workflow(
    workflow_id: str, payload: WorkflowClone, session: SessionDep
) -> WorkflowDefinition:
    definition, revision = _workflow_and_revision(session, workflow_id)
    return await create_workflow(
        WorkflowCreate(
            name=payload.name or f"{definition.name} copy",
            operation=Operation(definition.operation),
            description=definition.description,
            engine=revision.engine,
            engine_version=revision.engine_version,
            ui_graph=revision.ui_graph_json,
            api_graph=revision.api_graph_json,
            input_schema=revision.input_schema_json,
            dependencies=revision.dependencies_json,
            trusted=revision.trusted,
        ),
        session,
    )


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
    try:
        validate_lora_workflow_contract(
            payload.api_graph,
            payload.input_schema,
            payload.dependencies,
        )
        validate_workflow_edit_calibration(payload.input_schema)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
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
    if revision.engine == "comfyui" and not revision.trusted:
        errors.append(
            "workflow revision is not trusted; review it and create a trusted revision "
            "before execution"
        )
    try:
        role = "video" if "video" in definition.operation else "image"
        base_fields = await _engine_role_fields(
            request,
            role,
            engine=revision.engine,
            allow_inactive=True,
        )
        declared_fields = workflow_settings(base_fields, revision.input_schema_json)
        validate_settings(defaults(declared_fields), declared_fields)
        validate_workflow_edit_calibration(revision.input_schema_json)
    except ValueError as exc:
        errors.append(f"invalid workflow input schema: {exc}")
    dependencies = revision.dependencies_json
    errors.extend(custom_node_dependency_errors(session, dependencies.get("custom_nodes")))
    required_models = dependencies.get("models", [])
    installed = session.scalars(select(ModelInstall).where(ModelInstall.active.is_(True))).all()
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
        accelerators = [device for device in system.devices if device.kind != "cpu"]
        capacity = max(
            (device.total_memory_bytes or 0 for device in accelerators),
            default=0,
        )
        available_values = [
            device.available_memory_bytes
            for device in accelerators
            if device.available_memory_bytes is not None
        ]
        available = max(available_values, default=None)
        if not capacity:
            warnings.append("no accelerator memory capacity was detected for this workflow")
        elif capacity < minimum_vram:
            errors.append("accelerator memory capacity is below the workflow requirement")
        elif available is not None and available < minimum_vram:
            warnings.append(
                "currently available accelerator memory is below the workflow requirement"
            )
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
