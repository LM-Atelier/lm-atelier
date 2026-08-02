from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .adapters.comfyui import ComfyUIAdapter
from .api import recover_model_delete_quarantines, router
from .artifacts import ArtifactStore
from .backups import BackupManager
from .catalog import HuggingFaceCatalog
from .config import Settings, get_settings
from .credentials import CredentialStore
from .custom_nodes import CustomNodeManager
from .database_migrations import upgrade_database
from .db import SessionLocal, configure_database
from .diagnostics import DiagnosticBundleBuilder
from .downloads import DownloadManager
from .engines import EngineRegistry
from .events import EventBroker
from .exports import ProjectExporter
from .instance_identity import INSTANCE_ID_HEADER, load_or_create_instance_identity
from .orchestrator import ConversationOrchestrator
from .processes import ProcessSupervisor
from .runtime_provisioning import RuntimeProvisioner
from .scheduler import ResourceScheduler
from .security import (
    JsonBodyLimitMiddleware,
    SecurityHeadersMiddleware,
    SessionSecurity,
    UploadBodyLimitMiddleware,
)
from .seed import seed_defaults
from .worker_startup import restore_configured_workers

logger = logging.getLogger("local_lm")
AUTOMATIC_BACKUP_CHECK_INTERVAL_SECONDS = 60 * 60
API_LOG_MAX_BYTES = 2 * 1024 * 1024
API_LOG_BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _request_hostname(authority: str) -> str | None:
    if not authority or authority != authority.strip() or authority.endswith(":"):
        return None
    try:
        parsed = urlsplit(f"//{authority}")
        _ = parsed.port
    except ValueError:
        return None
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return None
    return parsed.hostname.casefold()


class LocalHostMiddleware:
    def __init__(self, app: ASGIApp, *, allow_test_hosts: bool = False) -> None:
        self.app = app
        self.allowed_hosts = {"127.0.0.1", "localhost", "::1"}
        if allow_test_hosts:
            self.allowed_hosts.update({"testserver", "testclient"})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        authority = Headers(scope=scope).get("host", "")
        if _request_hostname(authority) not in self.allowed_hosts:
            response = PlainTextResponse("Invalid host header", status_code=400)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _configure_console_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


@contextmanager
def _api_file_logging(settings: Settings) -> Iterator[None]:
    handler = RotatingFileHandler(
        settings.log_dir / "api.log",
        maxBytes=API_LOG_MAX_BYTES,
        backupCount=API_LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield
    except Exception:
        logger.exception("LM Atelier application lifecycle failed")
        raise
    finally:
        root_logger.removeHandler(handler)
        handler.close()


async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def _wait_for_websocket_disconnect(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def _stream_events(websocket: WebSocket, broker: EventBroker, *, after: int) -> None:
    disconnect_task = asyncio.create_task(
        _wait_for_websocket_disconnect(websocket),
        name="event-websocket-disconnect",
    )
    event_task: asyncio.Task[Any] | None = None
    send_task: asyncio.Task[Any] | None = None
    try:
        async with broker.subscribe(after) as queue:
            while True:
                event_task = asyncio.create_task(queue.get(), name="event-websocket-next-event")
                done, _ = await asyncio.wait(
                    {disconnect_task, event_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task in done:
                    await _cancel_task(event_task)
                    event_task = None
                    await disconnect_task
                    return

                event = event_task.result()
                event_task = None
                send_task = asyncio.create_task(
                    websocket.send_json(event.model_dump(mode="json")),
                    name="event-websocket-send",
                )
                done, _ = await asyncio.wait(
                    {disconnect_task, send_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task in done:
                    await _cancel_task(send_task)
                    send_task = None
                    await disconnect_task
                    return
                await send_task
                send_task = None
    finally:
        await _cancel_task(event_task)
        await _cancel_task(send_task)
        await _cancel_task(disconnect_task)


@dataclass
class Services:
    settings: Settings
    security: SessionSecurity
    events: EventBroker
    artifacts: ArtifactStore
    engines: EngineRegistry
    catalog: HuggingFaceCatalog
    downloads: DownloadManager
    scheduler: ResourceScheduler
    orchestrator: ConversationOrchestrator
    processes: ProcessSupervisor
    runtimes: RuntimeProvisioner
    backups: BackupManager
    exports: ProjectExporter
    diagnostics: DiagnosticBundleBuilder
    custom_nodes: CustomNodeManager
    credentials: CredentialStore


def build_services(settings: Settings) -> Services:
    credentials = CredentialStore(
        environment_tokens={
            "huggingface": settings.hf_token,
            "civitai": settings.civitai_token,
        }
    )
    settings.hf_token = credentials.token("huggingface")
    settings.civitai_token = credentials.token("civitai")
    events = EventBroker(settings.event_history_size)
    artifacts = ArtifactStore(settings)
    runtimes = RuntimeProvisioner(settings)
    engines = EngineRegistry(settings)
    scheduler = ResourceScheduler(events)
    processes = ProcessSupervisor(settings, runtimes, events)
    orchestrator = ConversationOrchestrator(engines, artifacts, events, scheduler, processes)
    services = Services(
        settings=settings,
        security=SessionSecurity(settings),
        events=events,
        artifacts=artifacts,
        engines=engines,
        catalog=HuggingFaceCatalog(settings),
        downloads=DownloadManager(
            settings,
            events,
            chat_adapter=lambda: engines.chat,
            media_adapter=engines.media if isinstance(engines.media, ComfyUIAdapter) else None,
            processes=processes,
            scheduler=scheduler,
        ),
        scheduler=scheduler,
        orchestrator=orchestrator,
        processes=processes,
        runtimes=runtimes,
        backups=BackupManager(settings),
        exports=ProjectExporter(settings, artifacts),
        diagnostics=DiagnosticBundleBuilder(settings, artifacts, processes),
        custom_nodes=CustomNodeManager(settings),
        credentials=credentials,
    )
    return services


async def ensure_automatic_recovery_backup(backups: BackupManager) -> None:
    """Create today's recovery point without blocking the API event loop."""

    operation = asyncio.create_task(
        asyncio.to_thread(backups.ensure_daily_backup),
        name="automatic-recovery-backup-check",
    )
    try:
        await asyncio.shield(operation)
    except asyncio.CancelledError:
        # A filesystem/SQLite transaction cannot be cancelled safely midway.
        # Let the bounded operation finish before completing app shutdown.
        with suppress(Exception):
            await operation
        raise
    except Exception:
        logger.exception("Could not maintain the automatic LM Atelier recovery backup")


async def maintain_automatic_recovery_backups(
    backups: BackupManager,
    *,
    interval_seconds: float = AUTOMATIC_BACKUP_CHECK_INTERVAL_SECONDS,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("automatic backup interval must be positive")
    while True:
        await asyncio.sleep(interval_seconds)
        await ensure_automatic_recovery_backup(backups)


def create_app(settings: Settings | None = None) -> FastAPI:
    _configure_console_logging()
    active_settings = settings or get_settings()
    active_settings.prepare()
    instance_identity = load_or_create_instance_identity(active_settings.data_dir)
    BackupManager(active_settings).apply_pending_restore()
    configure_database(active_settings)
    services = build_services(active_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        with _api_file_logging(active_settings):
            upgrade_database(active_settings)
            with SessionLocal() as session:
                recover_model_delete_quarantines(
                    session,
                    active_settings.model_dir.resolve(),
                )
                seed_defaults(session, active_settings)
                services.artifacts.cleanup_retention(
                    session,
                    retention_days=active_settings.artifact_retention_days,
                    temporary_hours=active_settings.temporary_retention_hours,
                    dry_run=False,
                )
                session.commit()
            await ensure_automatic_recovery_backup(services.backups)
            services.orchestrator.recover_interrupted()
            services.downloads.recover_interrupted()
            logger.info(
                "LM Atelier %s started on %s:%s",
                __version__,
                active_settings.host,
                active_settings.port,
            )
            worker_restore = asyncio.create_task(
                restore_configured_workers(services),
                name="restore-configured-workers",
            )
            backup_maintenance = asyncio.create_task(
                maintain_automatic_recovery_backups(services.backups),
                name="maintain-automatic-recovery-backups",
            )
            try:
                yield
            finally:
                for task in (worker_restore, backup_maintenance):
                    if not task.done():
                        task.cancel()
                for task in (worker_restore, backup_maintenance):
                    with suppress(asyncio.CancelledError):
                        await task
                await services.downloads.close()
                await services.orchestrator.close()
                await services.runtimes.close()
                await services.catalog.close()
                await services.engines.close()
                await services.processes.close()

    app = FastAPI(
        title="LM Atelier API",
        version=__version__,
        docs_url="/api/docs" if active_settings.dev else None,
        redoc_url=None,
        openapi_url="/openapi.json" if active_settings.dev else None,
        lifespan=lifespan,
    )
    app.state.services = services
    app.add_middleware(JsonBodyLimitMiddleware)
    app.add_middleware(
        UploadBodyLimitMiddleware,
        artifact_max_bytes=active_settings.max_upload_bytes,
        project_max_bytes=active_settings.max_project_import_bytes,
    )
    app.add_middleware(LocalHostMiddleware, allow_test_hosts=active_settings.dev)
    if active_settings.dev:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def local_session(request: Request, call_next):  # type: ignore[no-untyped-def]
        public = {"/api/session", "/api/health", "/api/ready"}
        if request.url.path.startswith("/api"):
            try:
                services.security.validate_origin(request.headers.get("origin"))
            except Exception as exc:
                status_code = getattr(exc, "status_code", 403)
                detail = getattr(exc, "detail", "untrusted browser origin")
                return JSONResponse({"detail": detail}, status_code=status_code)
        if request.url.path.startswith("/api") and request.url.path not in public:
            try:
                services.security.validate_request(request)
            except Exception as exc:
                status_code = getattr(exc, "status_code", 401)
                detail = getattr(exc, "detail", "authentication failed")
                return JSONResponse({"detail": detail}, status_code=status_code)
        response = await call_next(request)
        if request.url.path == "/api/ready":
            response.headers[INSTANCE_ID_HEADER] = instance_identity
        return response

    # Register this last so it wraps host, session, and body-limit rejections too.
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(router)

    @app.websocket("/api/events")
    async def events_socket(websocket: WebSocket, after: int = 0) -> None:
        if not await services.security.validate_websocket(websocket):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        try:
            await _stream_events(websocket, services.events, after=after)
        except WebSocketDisconnect:
            return

    web_dist = (
        active_settings.web_dist_dir.expanduser().resolve()
        if active_settings.web_dist_dir
        else Path.cwd() / "apps" / "web" / "dist"
    )
    if not web_dist.is_dir():
        web_dist = Path(__file__).resolve().parents[3] / "apps" / "web" / "dist"
    if not web_dist.is_dir():
        web_dist = Path(__file__).resolve().parent / "web"
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "local_lm.main:app" if settings.dev else app,
        host=settings.host,
        port=settings.port,
        reload=settings.dev,
    )
