from __future__ import annotations

import asyncio
import re
import secrets
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sqlalchemy import Engine, Table, create_engine, event, insert, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from .artifacts import ArtifactStore
from .db import Base, SessionLocal
from .events import EventBroker
from .models import (
    AppSetting,
    CustomNodeInstall,
    GenerationPreset,
    Job,
    ModelInstall,
    ModelProfile,
    ModelSource,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
)
from .orchestrator import ConversationOrchestrator
from .scheduler import ResourceScheduler

if TYPE_CHECKING:
    from .config import Settings
    from .engines import EngineRegistry
    from .processes import ProcessSupervisor
    from .security import SessionSecurity

INCOGNITO_HEADER = "X-LM-Atelier-Incognito"
INCOGNITO_SCOPE_ID = re.compile(r"^scope_[0-9a-f]{48}$")
INCOGNITO_SCOPED_PREFIXES = (
    "/api/artifacts",
    "/api/chats",
    "/api/jobs",
    "/api/messages",
    "/api/runs",
    "/api/work-plans",
    "/api/work-steps",
)
INCOGNITO_EXCLUDED_PREFIXES = (
    "/api/backups",
    "/api/diagnostics",
    "/api/projects",
)

_SHARED_TABLES: tuple[Table, ...] = (
    cast(Table, ModelSource.__table__),
    cast(Table, ModelInstall.__table__),
    cast(Table, ModelProfile.__table__),
    cast(Table, GenerationPreset.__table__),
    cast(Table, WorkflowDefinition.__table__),
    cast(Table, WorkflowRevision.__table__),
    cast(Table, CustomNodeInstall.__table__),
    cast(Table, AppSetting.__table__),
)


class IncognitoScopeUnavailable(LookupError):
    """The requested ephemeral scope is absent, ended, or no longer accepting work."""


class IncognitoUnavailableError(RuntimeError):
    """The configured local backends cannot honor the Incognito purge contract."""


@dataclass
class IncognitoConversationServices:
    settings: Settings
    security: SessionSecurity
    engines: EngineRegistry
    processes: ProcessSupervisor
    artifacts: ArtifactStore
    events: EventBroker
    scheduler: ResourceScheduler
    orchestrator: ConversationOrchestrator


@dataclass
class IncognitoScope:
    id: str
    root: Path
    engine: Engine
    session_factory: sessionmaker[Session]
    services: IncognitoConversationServices
    accepting: bool = True
    worker_output_suppressed: bool = True
    purge_rows: list[tuple[str, str, str]] | None = None
    work_settled: bool = False
    backend_purged: bool = False
    events_cleared: bool = False
    database_closed: bool = False
    configuration_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def token(self) -> str:
        return self.id


def is_incognito_scoped_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/") for prefix in INCOGNITO_SCOPED_PREFIXES
    )


def is_incognito_excluded_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/") for prefix in INCOGNITO_EXCLUDED_PREFIXES
    )


class IncognitoLifecycleManager:
    """Own one service-session-only conversation database and artifact root."""

    def __init__(
        self,
        settings: Settings,
        security: SessionSecurity,
        engines: EngineRegistry,
        processes: ProcessSupervisor,
        durable_scheduler: ResourceScheduler,
        *,
        durable_session_factory: sessionmaker[Session] = SessionLocal,
    ) -> None:
        self.settings = settings
        self.security = security
        self.engines = engines
        self.processes = processes
        self.durable_scheduler = durable_scheduler
        self.durable_session_factory = durable_session_factory
        self.root = (settings.state_dir / "incognito").resolve()
        self._scope: IncognitoScope | None = None
        self._lock = asyncio.Lock()

    @property
    def active_scope_id(self) -> str | None:
        scope = self._scope
        return scope.id if scope and scope.accepting else None

    async def start(self) -> IncognitoScope:
        async with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            if self._scope and self._scope.accepting:
                return self._scope
            if self._scope:
                raise IncognitoUnavailableError(
                    "The previous incognito session is still being purged."
                )
            self._require_privacy_capabilities()
            scope_id = f"scope_{secrets.token_hex(24)}"
            scope_root = (self.root / scope_id).resolve()
            if scope_root.parent != self.root:
                raise RuntimeError("incognito scope path escaped its managed root")
            artifact_root = scope_root / "artifacts"
            artifact_root.mkdir(parents=True, exist_ok=False)
            engine = self._create_engine(scope_id)
            Base.metadata.create_all(engine)
            session_factory = sessionmaker(
                bind=engine,
                expire_on_commit=False,
                class_=Session,
            )
            worker_output_suppressed = False
            try:
                self._copy_shared_configuration(engine)
                events = EventBroker(self.settings.event_history_size)
                scheduler = ResourceScheduler(
                    events,
                    session_factory=session_factory,
                    resource_pool=self.durable_scheduler,
                )
                artifacts = ArtifactStore(self.settings, root=artifact_root)
                orchestrator = ConversationOrchestrator(
                    self.engines,
                    artifacts,
                    events,
                    scheduler,
                    self.processes,
                    session_factory=session_factory,
                    persistence_scope="incognito",
                    scope_id=scope_id,
                )
                self.processes.begin_private_session()
                worker_output_suppressed = True
                services = IncognitoConversationServices(
                    settings=self.settings,
                    security=self.security,
                    engines=self.engines,
                    processes=self.processes,
                    artifacts=artifacts,
                    events=events,
                    scheduler=scheduler,
                    orchestrator=orchestrator,
                )
                self._scope = IncognitoScope(
                    id=scope_id,
                    root=scope_root,
                    engine=engine,
                    session_factory=session_factory,
                    services=services,
                )
            except Exception:
                if worker_output_suppressed:
                    self.processes.end_private_session()
                engine.dispose()
                self._remove_scope_root(scope_root)
                raise
            return self._scope

    def require(self, scope_id: str | None) -> IncognitoScope:
        scope = self._scope
        if not scope_id or not scope or scope.id != scope_id or not scope.accepting:
            raise IncognitoScopeUnavailable("incognito session is no longer available")
        return scope

    def get(self, scope_id: str | None) -> IncognitoScope | None:
        try:
            return self.require(scope_id)
        except IncognitoScopeUnavailable:
            return None

    async def end(self, scope_id: str | None) -> None:
        async with self._lock:
            scope = self._scope
            if not scope_id or not scope or scope.id != scope_id:
                raise IncognitoScopeUnavailable("incognito session is no longer available")
            scope.accepting = False
            scope.services.orchestrator.stop_admission()
            if scope.purge_rows is None:
                with scope.session_factory() as session:
                    scope.purge_rows = [
                        (job_id, run_id, operation)
                        for job_id, run_id, operation in session.execute(
                            select(Job.id, Run.id, Run.operation).join(Run, Job.run_id == Run.id)
                        ).all()
                    ]
            if not scope.work_settled:
                for job_id, _run_id, _operation in scope.purge_rows:
                    await scope.services.orchestrator.cancel(job_id)
                await scope.services.orchestrator.close()
                scope.work_settled = True
            if not scope.backend_purged:
                for _job_id, run_id, operation in scope.purge_rows:
                    adapter = self.engines.chat if operation == "text" else self.engines.media
                    try:
                        await adapter.purge_run(run_id)
                    except Exception as exc:
                        raise IncognitoUnavailableError(
                            "The private session could not be completely purged. Retry ending it."
                        ) from exc
                scope.backend_purged = True
            if not scope.events_cleared:
                await scope.services.events.publish("incognito.ended", scope.id)
                await scope.services.events.clear()
                scope.events_cleared = True
            if not scope.database_closed:
                scope.engine.dispose()
                scope.database_closed = True
            try:
                self._remove_scope_root(scope.root)
            except OSError as exc:
                raise IncognitoUnavailableError(
                    "The private session files could not be removed. Retry ending it."
                ) from exc
            if scope.worker_output_suppressed:
                self.processes.end_private_session()
                scope.worker_output_suppressed = False
            self._scope = None

    async def close(self) -> None:
        scope = self._scope
        if scope:
            await self.end(scope.id)

    async def refresh_shared_configuration(self, scope: IncognitoScope) -> None:
        """Make durable model/settings additions available without persisting chat data."""
        async with scope.configuration_lock:
            if scope.accepting and not scope.database_closed:
                self._copy_shared_configuration(scope.engine, update_existing=True)

    async def sweep_stale_roots(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for candidate in self.root.iterdir():
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.parent != self.root or not INCOGNITO_SCOPE_ID.fullmatch(candidate.name):
                continue
            purge_backend_files = getattr(
                self.engines.media,
                "purge_stale_scope_files",
                None,
            )
            if callable(purge_backend_files):
                await purge_backend_files(candidate.name)
            self._remove_scope_root(resolved)

    def _require_privacy_capabilities(self) -> None:
        incompatible = [
            label
            for label, adapter in (
                ("chat", self.engines.chat),
                ("image/video", self.engines.media),
            )
            if adapter.supports_incognito is not True
        ]
        if incompatible:
            joined = ", ".join(incompatible)
            raise IncognitoUnavailableError(
                f"Incognito is unavailable because these local backends cannot "
                f"purge run state: {joined}."
            )

    @staticmethod
    def _create_engine(scope_id: str) -> Engine:
        engine = create_engine(
            (f"sqlite+pysqlite:///file:{scope_id}?mode=memory&cache=shared&uri=true"),
            connect_args={"check_same_thread": False},
            poolclass=QueuePool,
            pool_size=4,
            max_overflow=8,
        )

        @event.listens_for(engine, "connect")
        def configure_sqlite(dbapi_connection: object, _connection_record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=MEMORY")
            cursor.execute("PRAGMA synchronous=OFF")
            cursor.close()

        return engine

    def _copy_shared_configuration(
        self,
        target_engine: Engine,
        *,
        update_existing: bool = False,
    ) -> None:
        with self.durable_session_factory() as source, target_engine.begin() as target:
            for table in _SHARED_TABLES:
                rows = [dict(row) for row in source.execute(select(table)).mappings().all()]
                if rows:
                    if not update_existing:
                        target.execute(insert(table), rows)
                        continue
                    statement = sqlite_insert(table).values(rows)
                    primary_keys = [column.name for column in table.primary_key.columns]
                    updates = {
                        column.name: getattr(statement.excluded, column.name)
                        for column in table.columns
                        if column.name not in primary_keys
                    }
                    if updates:
                        statement = statement.on_conflict_do_update(
                            index_elements=primary_keys,
                            set_=updates,
                        )
                    else:
                        statement = statement.on_conflict_do_nothing(index_elements=primary_keys)
                    target.execute(statement)

    def _remove_scope_root(self, scope_root: Path) -> None:
        resolved = scope_root.resolve()
        if resolved.parent != self.root or not INCOGNITO_SCOPE_ID.fullmatch(resolved.name):
            raise ValueError("refusing to remove an unmanaged incognito path")
        shutil.rmtree(resolved)
