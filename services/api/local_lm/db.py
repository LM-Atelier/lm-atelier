from __future__ import annotations

import logging
import sqlite3
import weakref
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from . import artifact_deletion_authority as _artifact_deletion_authority  # noqa: F401
from .config import Settings, get_settings

logger = logging.getLogger(__name__)

#: How long a write waits for another writer before it fails as busy.
SQLITE_BUSY_TIMEOUT_MS = 5000
#: The cursors a connection has run, while anything still refers to them.
_OPEN_CURSORS = "lm_atelier_open_cursors"
#: The busy timeout a pooled connection had before a session shortened it.
_USUAL_BUSY_TIMEOUT = "lm_atelier_usual_busy_timeout"


def database_is_contended(error: OperationalError) -> bool:
    """Whether this failure is another writer holding the database, not damage.

    A writer that outlasts the busy timeout produces an ordinary
    OperationalError, the same class a missing column produces, so the
    difference has to be read from SQLite's own error code rather than from the
    message. The low byte carries the primary code; the extended codes share it.
    """

    code = getattr(error.orig, "sqlite_errorcode", None)
    if not isinstance(code, int):
        return False
    return code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


def shorten_busy_timeout(session: Session, milliseconds: int) -> None:
    """Let this session's writes wait only `milliseconds` for another writer.

    The shorter wait lasts only while this session holds its connection: the
    pool gives the connection back its own timeout as it takes it back, before
    any other session can draw it.
    """

    pooled = session.connection().connection
    driver = pooled.driver_connection
    if driver is None:
        raise RuntimeError("The database connection closed before its wait could change.")
    if _USUAL_BUSY_TIMEOUT not in pooled.info:
        cursor = driver.execute("PRAGMA busy_timeout")
        try:
            pooled.info[_USUAL_BUSY_TIMEOUT] = int(cursor.fetchone()[0])
        finally:
            cursor.close()
    driver.execute(f"PRAGMA busy_timeout={int(milliseconds)}").close()


class Base(DeclarativeBase):
    pass


def create_database_engine(settings: Settings | None = None) -> Engine:
    active_settings = settings or get_settings()
    engine = create_engine(
        active_settings.database_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
        # A database error's text carries every bound value otherwise, and that
        # text is shown, stored and logged; the statement alone says enough.
        hide_parameters=True,
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection: object, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        cursor.close()

    @event.listens_for(engine, "after_cursor_execute")
    def remember_cursor(
        connection: Any,
        cursor: Any,
        _statement: object,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        connection.info.setdefault(_OPEN_CURSORS, weakref.WeakSet()).add(cursor)

    @event.listens_for(engine, "checkin")
    def close_cursors(dbapi_connection: object, connection_record: Any) -> None:
        """Close what a connection still has open before the next session draws it.

        A result still referred to after its session gave the connection back
        keeps SQLite's read open on that connection. The next session to draw it
        reads an old snapshot, and its first write fails at once with
        SQLITE_BUSY_SNAPSHOT instead of waiting out the busy timeout, however
        briefly the other writer held the database.
        """

        if connection_record is None:
            return
        cursors = connection_record.info.pop(_OPEN_CURSORS, None)
        if dbapi_connection is None:
            # Invalidated: SQLAlchemy closed the connection itself before this
            # checkin, and a closed connection is never handed out again.
            return
        for cursor in list(cursors or ()):
            try:
                cursor.close()
            except Exception:
                logger.warning("Could not close a cursor left open on a pooled connection")

    @event.listens_for(engine, "checkin")
    def restore_busy_timeout(dbapi_connection: Any, connection_record: Any) -> None:
        """Give a connection back the busy timeout a session shortened, before it is drawn again."""

        if connection_record is None:
            return
        usual = connection_record.info.pop(_USUAL_BUSY_TIMEOUT, None)
        if usual is None or dbapi_connection is None:
            return
        try:
            dbapi_connection.execute(f"PRAGMA busy_timeout={int(usual)}").close()
        except Exception:
            # Only a closed connection refuses this, and the pool's ping on the
            # next checkout replaces a closed connection with a new one.
            logger.warning("Could not give a pooled connection back its busy timeout")

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
