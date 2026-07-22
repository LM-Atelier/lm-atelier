from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .api import router
from .artifacts import ArtifactStore
from .backups import BackupManager
from .catalog import HuggingFaceCatalog
from .config import Settings, get_settings
from .database_migrations import upgrade_database
from .db import SessionLocal, configure_database
from .downloads import DownloadManager
from .engines import EngineRegistry
from .events import EventBroker
from .exports import ProjectExporter
from .orchestrator import ConversationOrchestrator
from .processes import ProcessSupervisor
from .scheduler import ResourceScheduler
from .security import SessionSecurity
from .seed import seed_defaults

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("local_lm")


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
    backups: BackupManager
    exports: ProjectExporter


def build_services(settings: Settings) -> Services:
    events = EventBroker(settings.event_history_size)
    artifacts = ArtifactStore(settings)
    engines = EngineRegistry(settings)
    scheduler = ResourceScheduler()
    processes = ProcessSupervisor(settings)
    orchestrator = ConversationOrchestrator(engines, artifacts, events, scheduler, processes)
    return Services(
        settings=settings,
        security=SessionSecurity(settings),
        events=events,
        artifacts=artifacts,
        engines=engines,
        catalog=HuggingFaceCatalog(settings),
        downloads=DownloadManager(settings, events),
        scheduler=scheduler,
        orchestrator=orchestrator,
        processes=processes,
        backups=BackupManager(settings),
        exports=ProjectExporter(settings, artifacts),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    active_settings.prepare()
    BackupManager(active_settings).apply_pending_restore()
    configure_database(active_settings)
    services = build_services(active_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        upgrade_database(active_settings)
        with SessionLocal() as session:
            seed_defaults(session, active_settings)
        services.orchestrator.recover_interrupted()
        services.downloads.recover_interrupted()
        logger.info(
            "Local LM %s started on %s:%s", __version__, active_settings.host, active_settings.port
        )
        yield
        await services.catalog.close()
        await services.engines.close()
        await services.processes.close()

    app = FastAPI(
        title="Local LM API",
        version=__version__,
        docs_url="/api/docs" if active_settings.dev else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.services = services
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
        if active_settings.allow_lan
        else ["127.0.0.1", "localhost", "[::1]", "testserver", "testclient"],
    )
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
        public = {"/api/session", "/api/health"}
        if request.url.path.startswith("/api") and request.url.path not in public:
            try:
                services.security.validate_request(request)
            except Exception as exc:
                status_code = getattr(exc, "status_code", 401)
                detail = getattr(exc, "detail", "authentication failed")
                return JSONResponse({"detail": detail}, status_code=status_code)
        return await call_next(request)

    app.include_router(router)

    @app.websocket("/api/events")
    async def events_socket(websocket: WebSocket, after: int = 0) -> None:
        if not await services.security.validate_websocket(websocket):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        try:
            async with services.events.subscribe(after) as queue:
                while True:
                    event = await queue.get()
                    await websocket.send_json(event.model_dump(mode="json"))
        except WebSocketDisconnect:
            return

    web_dist = (
        active_settings.web_dist_dir.expanduser().resolve()
        if active_settings.web_dist_dir
        else Path.cwd() / "apps" / "web" / "dist"
    )
    if not web_dist.is_dir():
        web_dist = Path(__file__).resolve().parents[3] / "apps" / "web" / "dist"
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "local_lm.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.dev,
    )
