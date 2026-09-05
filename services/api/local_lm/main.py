from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
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
from sqlalchemy.exc import OperationalError
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .adapters.comfyui import ComfyUIAdapter
from .api import (
    recover_model_delete_quarantines,
    router,
    shutdown_registry_preparations,
)
from .api_errors import register_api_error_handler
from .artifacts import (
    RETENTION_BATCH_DELETIONS,
    RETENTION_BATCH_SECONDS,
    ArtifactStore,
    RetentionCleanupSummary,
)
from .backups import BackupManager
from .catalog import HuggingFaceCatalog
from .catalog_sources import CatalogSources
from .civitai_catalog import CivitaiCatalog
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
from .workflow_editor_sessions import WorkflowEditorSessions

logger = logging.getLogger("local_lm")
AUTOMATIC_BACKUP_CHECK_INTERVAL_SECONDS = 60 * 60
API_LOG_MAX_BYTES = 2 * 1024 * 1024
API_LOG_BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
STARTUP_STAGE_WARN_SECONDS = 30.0


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


def _emit_startup_notice(message: str) -> None:
    """Say it on both channels, because either one can be the missing one.

    The api.log handler does not exist until the lifespan installs it, and the
    stages that run before that are exactly the ones a silent hang hides. A
    flushed stderr line survives both that gap and a wedged file handler.
    """
    logger.warning(message)
    print(f"local_lm: {message}", file=sys.stderr, flush=True)


@contextmanager
def _startup_stage(name: str, *, warn_after: float = STARTUP_STAGE_WARN_SECONDS) -> Iterator[None]:
    """Name a slow startup stage while it is still running, not after it ends.

    Timing a stage and reporting the duration afterwards proves nothing about a
    stage that never returns, which is the case worth diagnosing: a start that
    hangs for seventy minutes reports no duration at all, because nothing
    reaches the line that would log it. A watchdog thread reports from outside
    the stage instead, so the last name emitted is the stage still running.

    Bounded on purpose. A healthy start emits nothing, because a stage that
    beats warn_after says nothing on entry or exit; only a stage that overruns
    speaks, and then it repeats so the log shows the wait growing rather than
    one line that could be mistaken for a completed step.

    Emission is serialized with finalization because clearing the event is not
    enough on its own. `finished.set()` stops another wait cycle, but a watchdog
    that has already returned from `wait` is past that check and will still
    write its line - after the stage has exited and recorded that it finished.
    That inverts the one guarantee here, which is that the last stage named is
    the stage still running. Holding the lock across the check and the write
    means a watchdog either speaks while the stage is genuinely running or,
    finding the stage finished ahead of it, does not speak at all.
    """
    finished = threading.Event()
    speaking = threading.Lock()
    started = time.perf_counter()

    def watch() -> None:
        while not finished.wait(warn_after):
            with speaking:
                if finished.is_set():
                    return
                waited = time.perf_counter() - started
                _emit_startup_notice(f"startup stage {name} still running after {waited:.0f}s")

    watcher = threading.Thread(target=watch, name=f"startup-stage-{name}", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        with speaking:
            elapsed = time.perf_counter() - started
            if elapsed >= warn_after:
                _emit_startup_notice(f"startup stage {name} finished after {elapsed:.0f}s")


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
    catalog_sources: CatalogSources
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
    workflow_editor_sessions: WorkflowEditorSessions

    @property
    def catalog(self) -> HuggingFaceCatalog:
        source = self.catalog_sources.get("huggingface")
        if not isinstance(source, HuggingFaceCatalog):
            raise RuntimeError("the default Hugging Face catalog source is unavailable")
        return source


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
    catalog = HuggingFaceCatalog(settings)
    civitai = CivitaiCatalog(settings, token=settings.civitai_token)
    services = Services(
        settings=settings,
        security=SessionSecurity(settings),
        events=events,
        artifacts=artifacts,
        engines=engines,
        catalog_sources=CatalogSources([catalog, civitai]),
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
        workflow_editor_sessions=WorkflowEditorSessions(),
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
        # Check first, then sleep. The startup path used to await the first
        # check before the lifespan yield, which held the port closed for as
        # long as verification took - an integrity check, a foreign-key check
        # and a whole-file digest, none of which needs the port shut.
        await ensure_automatic_recovery_backup(backups)
        await asyncio.sleep(interval_seconds)


RETENTION_BATCH_PAUSE_SECONDS = 1.0
# Contention with a user transaction is waited out, a bounded number of times
# in a row; a whole store is drained in successive passes, since a poster is
# freed only by the pass that removes the artifact naming it, up to a bound
# that keeps a pathological store from sweeping forever.
RETENTION_LOCK_RETRIES = 5
RETENTION_MAX_PASSES = 20


RETENTION_PROGRESS_WARN_SECONDS = 30.0


class _RetentionBatchProgress:
    """Observe a batch without changing its deletion clock or transaction."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.phase_started = self.started
        self.current_phase = "starting"
        self.writer_started: float | None = None
        self.writer_seconds = 0.0
        self.finished = threading.Event()
        self.speaking = threading.Lock()

    def phase(self, name: str) -> None:
        with self.speaking:
            now = time.monotonic()
            if name == "writer-acquired":
                self.writer_started = now
            self.current_phase = name
            self.phase_started = now

    def transaction_finished(self) -> None:
        with self.speaking:
            if self.writer_started is not None:
                self.writer_seconds = time.monotonic() - self.writer_started
                self.writer_started = None

    def watch(self) -> None:
        while not self.finished.wait(RETENTION_PROGRESS_WARN_SECONDS):
            with self.speaking:
                if self.finished.is_set():
                    return
                now = time.monotonic()
                held = self.writer_seconds
                if self.writer_started is not None:
                    held = now - self.writer_started
                logger.warning(
                    "Artifact retention batch still running: phase %s for %.3fs; "
                    "elapsed %.3fs; writer reservation %.3fs",
                    self.current_phase,
                    now - self.phase_started,
                    now - self.started,
                    held,
                )


@contextmanager
def _retention_batch_progress() -> Iterator[_RetentionBatchProgress]:
    progress = _RetentionBatchProgress()
    watcher = threading.Thread(
        target=progress.watch, name="artifact-retention-progress", daemon=True
    )
    watcher.start()
    try:
        yield progress
    finally:
        progress.finished.set()
        # Serialize completion with an in-flight notice. No notice may appear
        # after the final summary for the batch it describes.
        watcher.join()


async def sweep_artifact_retention(
    artifacts: ArtifactStore,
    settings: Settings,
    *,
    batch_deletions: int | None = None,
    pause_seconds: float | None = None,
    batch_seconds: float | None = None,
) -> None:
    """Run retention after the port opens, committing one bounded batch at a time.

    Each batch holds its writer reservation through one complete reference
    snapshot and a deletion phase bounded by time and count. Snapshot cost is
    fixed work before the deletion clock starts. A stop request still ends the
    batch at the next deletion boundary, and completed deletions are committed
    before the next batch. Failures are logged while the application keeps
    running; the next start resumes the sweep.
    """

    # Resolved at call time, not at definition time, so the module constants
    # are the single place to tune - and to patch under test.
    if batch_deletions is None:
        batch_deletions = RETENTION_BATCH_DELETIONS
    if pause_seconds is None:
        pause_seconds = RETENTION_BATCH_PAUSE_SECONDS
    if batch_seconds is None:
        batch_seconds = RETENTION_BATCH_SECONDS
    if batch_deletions < 1:
        raise ValueError("retention batch size must be positive")
    if pause_seconds < 0:
        raise ValueError("retention batch pause must not be negative")
    if batch_seconds < 0:
        raise ValueError("retention batch budget must not be negative")
    stop = threading.Event()
    # The session factory is rebound in place whenever the database is
    # reconfigured; a sweep binds to the engine it started with, so a later
    # rebinding cannot redirect a batch mid-sweep.
    bind = SessionLocal.kw.get("bind")

    def run_batch(*, deletions: int, seconds: float | None) -> RetentionCleanupSummary:
        deadline: float | None = None

        def should_stop() -> bool:
            nonlocal deadline
            if stop.is_set():
                return True
            if seconds is None:
                return False
            if deadline is None:
                # The complete graph and validity scan are fixed work. Charge
                # the budget from the first deletion boundary after that mint.
                deadline = time.monotonic() + seconds
            return time.monotonic() >= deadline

        with _retention_batch_progress() as progress, SessionLocal(bind=bind) as session:
            try:
                summary = artifacts.cleanup_retention(
                    session,
                    retention_days=settings.artifact_retention_days,
                    temporary_hours=settings.temporary_retention_hours,
                    dry_run=False,
                    max_deletions=deletions,
                    should_stop=should_stop,
                    report_phase=progress.phase,
                )
                progress.phase("commit")
                session.commit()
                progress.transaction_finished()
            except BaseException:
                # Roll back here, not implicitly at close, so the rollback
                # listeners restore staged files before the store is observed.
                progress.phase("rollback")
                session.rollback()
                progress.transaction_finished()
                raise
        logger.info(
            "Artifact retention batch committed: %s row(s) examined, %s item(s) removed; "
            "elapsed %.3fs; writer reservation %.3fs",
            summary.examined_count,
            summary.removed_count,
            time.monotonic() - progress.started,
            progress.writer_seconds,
        )
        return summary

    started = time.monotonic()
    logger.info("Artifact retention sweep started")
    batches = 0
    examined = 0
    removed = 0
    passes = 0
    locked_out = 0
    single = False
    try:
        while True:
            deletions = batch_deletions
            seconds: float | None = batch_seconds
            if single:
                # A zero or exhausted deletion budget cannot make progress.
                # Fall back to one deletion while still honoring shutdown.
                deletions = 1
                seconds = None
            operation = asyncio.create_task(
                asyncio.to_thread(run_batch, deletions=deletions, seconds=seconds),
                name="artifact-retention-batch",
            )
            try:
                summary = await asyncio.shield(operation)
            except asyncio.CancelledError:
                # A batch in flight cannot be cancelled inside SQLite. The stop
                # flag ends it at its next deletion boundary and it commits what
                # it has; wait for that rather than tearing the session down.
                stop.set()
                with suppress(Exception):
                    summary = await operation
                    batches += 1
                    examined += summary.examined_count
                    removed += summary.removed_count
                raise
            except OperationalError as exc:
                # A user transaction held the writer for longer than
                # busy_timeout. That is contention, not corruption: wait and
                # try again, a bounded number of times in a row.
                locked_out += 1
                if locked_out > RETENTION_LOCK_RETRIES:
                    logger.error(
                        "Artifact retention sweep gave up after %s consecutive waits on "
                        "the database writer; it runs again at the next start (%s)",
                        locked_out - 1,
                        exc,
                    )
                    return
                logger.warning(
                    "Artifact retention batch waited on the database writer; retrying (%s)",
                    exc,
                )
                await asyncio.sleep(max(pause_seconds, 1.0))
                continue
            locked_out = 0
            batches += 1
            examined += summary.examined_count
            removed += summary.removed_count
            if summary.truncated:
                if summary.removed_count == 0 and not single:
                    single = True
                    logger.info(
                        "Artifact retention batch %s made no progress within %.1fs; "
                        "continuing one deletion per batch",
                        batches,
                        batch_seconds,
                    )
                else:
                    logger.info(
                        "Artifact retention batch %s removed %s artifact(s); more remain",
                        batches,
                        summary.removed_count,
                    )
                await asyncio.sleep(pause_seconds)
                continue
            passes += 1
            if summary.removed_count == 0 or passes >= RETENTION_MAX_PASSES:
                break
            # A completed pass that removed something may have freed more: a
            # poster goes only once the artifact naming it is gone. Sweep again
            # until a whole pass finds nothing, rather than once per start.
            await asyncio.sleep(pause_seconds)
    except asyncio.CancelledError:
        logger.info(
            "Artifact retention sweep stopped after %s batch(es), %s removed; "
            "elapsed %.3fs; it resumes at the next start",
            batches,
            removed,
            time.monotonic() - started,
        )
        raise
    except Exception:
        logger.exception(
            "Artifact retention sweep failed after %s batch(es); elapsed %.3fs; "
            "it runs again at the next start",
            batches,
            time.monotonic() - started,
        )
        return
    logger.info(
        "Artifact retention sweep complete: %s batch(es), %s artifact(s) removed; "
        "%s row examination(s); elapsed %.3fs",
        batches,
        removed,
        examined,
        time.monotonic() - started,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    _configure_console_logging()
    with _startup_stage("load-settings"):
        active_settings = settings or get_settings()
    with _startup_stage("settings-prepare"):
        active_settings.prepare()
    with _startup_stage("instance-identity"):
        instance_identity = load_or_create_instance_identity(active_settings.data_dir)
    with _startup_stage("pending-restore"):
        BackupManager(active_settings).apply_pending_restore()
    with _startup_stage("configure-database"):
        configure_database(active_settings)
    with _startup_stage("build-services"):
        services = build_services(active_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        with _api_file_logging(active_settings):
            with _startup_stage("database-migrations"):
                upgrade_database(active_settings)
            with SessionLocal() as session:
                with _startup_stage("model-delete-quarantine-recovery"):
                    recover_model_delete_quarantines(
                        session,
                        active_settings.model_dir.resolve(),
                    )
                with _startup_stage("seed-defaults"):
                    seed_defaults(session, active_settings)
                with _startup_stage("session-commit"):
                    session.commit()
            with _startup_stage("orchestrator-recovery"):
                services.orchestrator.recover_interrupted()
            with _startup_stage("download-recovery"):
                services.downloads.recover_interrupted()
            logger.info(
                "LM Atelier %s started on %s:%s",
                __version__,
                active_settings.host,
                active_settings.port,
            )
            services.runtimes.start_restore()
            worker_restore = asyncio.create_task(
                restore_configured_workers(services),
                name="restore-configured-workers",
            )
            backup_maintenance = asyncio.create_task(
                maintain_automatic_recovery_backups(services.backups),
                name="maintain-automatic-recovery-backups",
            )
            # After recovery, never before the yield: recovery removes the
            # preview parts of interrupted jobs, which is what makes those
            # previews eligible, and the port must not wait on any of it.
            retention_sweep = asyncio.create_task(
                sweep_artifact_retention(services.artifacts, active_settings),
                name="artifact-retention-sweep",
            )
            # Exposed so a caller that needs the sweep's result - a test, or a
            # later readiness view - can await it instead of polling the store.
            app.state.retention_sweep = retention_sweep
            background = (worker_restore, backup_maintenance, retention_sweep)
            try:
                yield
            finally:
                for task in background:
                    if not task.done():
                        task.cancel()
                for task in background:
                    with suppress(asyncio.CancelledError):
                        await task
                await shutdown_registry_preparations()
                services.workflow_editor_sessions.clear()
                await services.downloads.close()
                await services.orchestrator.close()
                await services.runtimes.close()
                await services.catalog_sources.close()
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

        def refusal(exc: Exception, fallback_status: int, fallback_detail: str) -> JSONResponse:
            """Answer a refused request the same way the rest of the API does.

            Middleware runs outside the app's exception handlers, so a typed
            error raised here reaches no handler and its code is lost unless
            this carries it. Three refusals live here - untrusted origin, no
            session, failed CSRF - and they are exactly the three a client has
            to tell apart to know whether to stop, authenticate, or refetch a
            token.
            """
            body: dict[str, object] = {
                "detail": getattr(exc, "detail", fallback_detail),
            }
            code = getattr(exc, "code", None)
            if isinstance(code, str):
                body["code"] = code
            return JSONResponse(body, status_code=getattr(exc, "status_code", fallback_status))

        if request.url.path.startswith("/api"):
            try:
                services.security.validate_origin(request.headers.get("origin"))
            except Exception as exc:
                return refusal(exc, 403, "untrusted browser origin")
        if request.url.path.startswith("/api") and request.url.path not in public:
            try:
                services.security.validate_request(request)
            except Exception as exc:
                return refusal(exc, 401, "authentication failed")
        response = await call_next(request)
        if request.url.path == "/api/ready":
            response.headers[INSTANCE_ID_HEADER] = instance_identity
        return response

    # Register this last so it wraps host, session, and body-limit rejections too.
    app.add_middleware(SecurityHeadersMiddleware)
    register_api_error_handler(app)
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
