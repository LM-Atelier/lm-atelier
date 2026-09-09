from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from . import artifact_deletion_authority as _artifact_deletion_authority  # noqa: F401
from .config import Settings, get_settings


class Base(DeclarativeBase):
    pass


def create_database_engine(settings: Settings | None = None) -> Engine:
    active_settings = settings or get_settings()
    engine = create_engine(
        active_settings.database_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection: object, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine


engine = create_database_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def configure_database(settings: Settings) -> None:
    global engine
    engine.dispose()
    engine = create_database_engine(settings)
    SessionLocal.configure(bind=engine)


def init_db() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


async def get_session() -> AsyncGenerator[Session, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
