from __future__ import annotations

import shutil
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient

from local_lm import db
from local_lm.config import Settings
from local_lm.database_migrations import upgrade_database
from local_lm.db import configure_database
from local_lm.main import create_app


@pytest.fixture(scope="session")
def migrated_database_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("migrated-database-template") / "data"
    settings = Settings(
        data_dir=data_dir,
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    settings.prepare()
    configure_database(settings)
    upgrade_database(settings)
    db.engine.dispose()
    database_path = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    return database_path


@pytest.fixture
def settings(tmp_path: Path, migrated_database_template: Path) -> Settings:
    data_dir = tmp_path / "data"
    state_dir = data_dir / "state"
    state_dir.mkdir(parents=True)
    shutil.copy2(migrated_database_template, state_dir / "local-lm.sqlite3")
    return Settings(
        data_dir=data_dir,
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            session = await test_client.post("/api/session")
            test_client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]
            yield test_client
