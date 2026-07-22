from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from .catalog import HuggingFaceCatalog
from .config import Settings
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
)
from .downloads import DownloadManager
from .engines import EngineRegistry
from .hardware import collect_system_info
from .models import (
    Artifact,
    Chat,
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
from .recipes import get_reference_recipe, list_reference_recipes, recipe_download_request
from .schemas import (
    ArtifactOut,
    BackupInfo,
    CatalogPage,
    ChatCreate,
    ChatDetail,
    ChatOut,
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
    SystemInfo,
    TurnAccepted,
    TurnRequest,
    WorkerStatus,
    WorkflowCreate,
    WorkflowOut,
    WorkflowRevisionCreate,
    WorkflowRevisionOut,
)
from .security import SessionSecurity
from .settings_registry import CHAT_SETTINGS, IMAGE_SETTINGS, VIDEO_SETTINGS, validate_settings

SessionDep = Annotated[Session, Depends(get_session)]


def _services(request: Request) -> Any:
    return request.app.state.services


router = APIRouter(prefix="/api")


@router.post("/session")
async def create_session(request: Request, response: Response) -> dict[str, str]:
    security: SessionSecurity = _services(request).security
    return {"csrf_token": security.issue_session(response)}


@router.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    engines: EngineRegistry = _services(request).engines
    try:
        capabilities = await engines.capabilities()
    except Exception:
        capabilities = []
    return HealthOut(
        status="ok" if all(item.healthy for item in capabilities) else "degraded",
        version="0.1.0",
        database=True,
        engines=capabilities,
    )


@router.get("/system", response_model=SystemInfo)
async def system_info(request: Request) -> SystemInfo:
    settings: Settings = _services(request).settings
    return collect_system_info(settings)


@router.get("/platforms", response_model=list[PlatformMatrixEntry])
async def platform_matrix() -> list[PlatformMatrixEntry]:
    return list_platform_matrix()


@router.get("/backups", response_model=list[BackupInfo])
async def list_backups(request: Request) -> list[BackupInfo]:
    return _services(request).backups.list()


@router.post("/backups", response_model=BackupInfo, status_code=201)
async def create_backup(request: Request) -> BackupInfo:
    return _services(request).backups.create()


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


@router.get("/workers", response_model=list[WorkerStatus])
async def worker_status(request: Request) -> list[WorkerStatus]:
    return _services(request).processes.statuses()


@router.post("/workers/chat/load/{profile_id}", response_model=WorkerStatus)
async def load_chat_worker(profile_id: str, request: Request, session: SessionDep) -> WorkerStatus:
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
async def start_media_worker(request: Request) -> WorkerStatus:
    try:
        return await _services(request).processes.start_media()
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/workers/{name}/stop", response_model=WorkerStatus)
async def stop_worker(name: str, request: Request) -> WorkerStatus:
    try:
        return await _services(request).processes.stop(name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/projects", response_model=list[ProjectOut])
async def list_projects(session: SessionDep, include_archived: bool = False) -> list[Project]:
    statement = select(Project).order_by(Project.updated_at.desc())
    if not include_archived:
        statement = statement.where(Project.archived.is_(False))
    return list(session.scalars(statement).all())


@router.post("/projects", response_model=ProjectOut, status_code=201)
async def create_project(payload: ProjectCreate, session: SessionDep) -> Project:
    project = Project(**payload.model_dump())
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.patch("/projects/{project_id}", response_model=ProjectOut)
async def update_project(project_id: str, payload: ProjectUpdate, session: SessionDep) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
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
async def export_project(project_id: str, request: Request, session: SessionDep) -> ArtifactOut:
    try:
        artifact = _services(request).exports.export(session, project_id)
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
) -> list[Chat]:
    statement = select(Chat).order_by(Chat.updated_at.desc())
    if project_id:
        statement = statement.where(Chat.project_id == project_id)
    if not include_archived:
        statement = statement.where(Chat.archived.is_(False))
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
        return orchestrator.create_turn(session, chat_id, payload)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
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
    operation = Operation(prior_run.operation)
    mode = (
        RoutingMode.TEXT
        if operation == Operation.TEXT
        else RoutingMode.VIDEO
        if "video" in operation.value
        else RoutingMode.IMAGE
    )
    turn = TurnRequest(
        text=text,
        mode=mode,
        parent_message_id=user_message.parent_id,
        input_artifact_ids=prior_run.provenance_json.get("input_artifact_ids", []),
        settings={**prior_run.settings_json, **payload.settings},
    )
    return _services(request).orchestrator.create_turn(session, prior_run.chat_id, turn)


@router.post("/messages/{message_id}/branch", response_model=TurnAccepted, status_code=202)
async def edit_and_branch(
    message_id: str, payload: TurnRequest, request: Request, session: SessionDep
) -> TurnAccepted:
    source = session.get(Message, message_id)
    if not source or source.role != MessageRole.USER.value:
        raise HTTPException(404, "user message not found")
    turn = payload.model_copy(update={"parent_message_id": source.parent_id})
    return _services(request).orchestrator.create_turn(session, source.chat_id, turn)


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
) -> CatalogPage:
    catalog: HuggingFaceCatalog = _services(request).catalog
    try:
        return await catalog.search(query=query, role=role, sort=sort, limit=limit, cursor=cursor)
    except Exception as exc:
        raise HTTPException(502, f"catalog request failed: {exc}") from exc


@router.get("/catalog/{owner}/{name}")
async def catalog_detail(
    owner: str, name: str, request: Request, revision: str = "main"
) -> dict[str, Any]:
    try:
        return await _services(request).catalog.inspect(f"{owner}/{name}", revision)
    except Exception as exc:
        raise HTTPException(502, f"catalog request failed: {exc}") from exc


@router.post("/downloads", response_model=JobOut, status_code=202)
async def create_download(payload: DownloadRequest, request: Request, session: SessionDep) -> Job:
    manager: DownloadManager = _services(request).downloads
    try:
        return manager.create(session, payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


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


@router.post("/models/import", response_model=ModelInstallOut, status_code=201)
async def import_model(payload: ModelImport, session: SessionDep) -> ModelInstall:
    path = Path(payload.local_path).expanduser().resolve(strict=True)
    blocked = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
    if path.is_file() and path.suffix.lower() in blocked:
        raise HTTPException(422, "pickle-compatible model files are blocked by default")
    if path.is_dir():
        size = sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    else:
        size = path.stat().st_size
    install = ModelInstall(
        id=new_id("model"),
        name=payload.name,
        role=payload.role,
        engine=payload.engine,
        local_path=str(path),
        size_bytes=size,
        compatibility=CompatibilityLevel.ADVANCED.value,
        manifest_json={"imported": True},
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
    model_root = _services(request).settings.model_dir.resolve()
    path = Path(install.local_path).resolve()
    if model_root in path.parents:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    session.delete(install)
    session.commit()
    return Response(status_code=204)


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
    fields = _preset_fields(payload.role)
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
                [field for field in _preset_fields(profile.role) if field.scope == "load"],
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if "request_settings" in values:
        try:
            profile.request_settings_json = validate_settings(
                values.pop("request_settings") or {},
                [field for field in _preset_fields(profile.role) if field.scope != "load"],
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    for key, value in values.items():
        setattr(profile, key, value)
    session.commit()
    session.refresh(profile)
    return profile


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


def _preset_fields(role: str):  # type: ignore[no-untyped-def]
    return {"chat": CHAT_SETTINGS, "image": IMAGE_SETTINGS, "video": VIDEO_SETTINGS}[role]


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
    return {"valid": not errors, "errors": errors, "revision_id": revision.id}
