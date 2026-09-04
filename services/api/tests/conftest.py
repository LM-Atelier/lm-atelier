from __future__ import annotations

import asyncio
import shutil
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient

import local_lm
from local_lm import db
from local_lm.config import Settings
from local_lm.database_migrations import upgrade_database
from local_lm.db import configure_database
from local_lm.main import create_app

#: Which tree these tests are about to measure.
#:
#: The virtualenv installs `local_lm` as an editable package pinned to whichever
#: checkout created it. Run pytest from a second worktree without PYTHONPATH and
#: the tests collected here are this worktree's while the modules under test are
#: the other one's. Everything passes, every header prints this path, and the
#: commit being verified was never executed.
#:
#: The failure direction is the dangerous one. A NEW module is absent from the
#: other checkout, so its tests error loudly and somebody notices. A MODIFIED
#: module silently resolves to the older copy and the suite goes green against
#: code that is not in the commit.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_EXPECTED_PACKAGE = (_REPOSITORY_ROOT / "services" / "api" / "local_lm").resolve()
_IMPORTED_FROM = Path(local_lm.__file__).resolve().parent

# EXACT identity, not containment. Linked worktrees can live below the main
# checkout, so a nested worktree's package is relative to the main root and a containment test says
# yes to it. Measured: is_relative_to returns True for
# <root>/temp/worktrees/x/services/api/local_lm against <root>, which is
# precisely the wrong answer.
if _IMPORTED_FROM != _EXPECTED_PACKAGE:
    raise RuntimeError(
        "local_lm imports from a different tree than the one these tests "
        "came from, so they would measure the wrong code.\n"
        f"  imported from : {_IMPORTED_FROM}\n"
        f"  expected      : {_EXPECTED_PACKAGE}\n"
        "Set PYTHONPATH to this worktree before running:\n"
        f"  PYTHONPATH={_REPOSITORY_ROOT / 'services' / 'api'}"
    )


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
    test_settings = Settings(
        data_dir=data_dir,
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    # Bind the module-global session to THIS test's copy. Building the
    # template configured the global engine against the template file, and
    # `configure_database` rebinds `SessionLocal` globally - so a test that
    # opens `SessionLocal()` directly, rather than going through the app,
    # wrote into the shared template and every later test copied the result.
    # Tests that create their own app re-bind again through `create_app`.
    configure_database(test_settings)
    return test_settings


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        # The retention sweep runs after startup rather than inside it, so a
        # test that inspects artifacts would otherwise race it. Every test
        # starts from the settled store, exactly as it did when the sweep was
        # a startup stage.
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            session = await test_client.post("/api/session")
            test_client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]
            yield test_client
