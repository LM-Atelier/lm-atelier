from __future__ import annotations

import asyncio
import shutil
import sqlite3
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from application_environment import application_environment_kept
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from local_name_lookups import NameLookupGuard, refused_during_test, refused_unanswered
from verification_phase_timings import register_verification_phase_timings

import local_lm
from local_lm import artifact_reference_index, db
from local_lm import main as application
from local_lm.config import Settings
from local_lm.database_migrations import upgrade_database
from local_lm.db import configure_database
from local_lm.main import create_app
from local_lm.runtime_provisioning import RuntimeProvisioner

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


def pytest_configure(config: pytest.Config) -> None:
    register_verification_phase_timings(config)


@pytest.fixture(autouse=True)
def application_environment_restored() -> Iterator[None]:
    """Every test leaves the application's environment exactly as it found it.

    The launcher and the saved runtime settings write what they configure into
    this process's environment, which every later Settings reads. Clearing an
    absent name with monkeypatch leaves nothing to undo, so a value written
    after it outlived the test, and every later test in the same worker read
    it as configured. Used automatically at function scope, this is set up
    before the test's other function-scoped fixtures and restores only after
    they have finished.
    """

    with application_environment_kept():
        yield


#: Every answer the reference index gives during the tests is also checked
#: against the walk that reads the whole store, which the application skips.
artifact_reference_index.CHECK_AGAINST_WALK = True


@pytest.fixture(autouse=True)
def artifact_reference_index_agrees_with_the_walk() -> Iterator[None]:
    """Every reference answer a test gets from the index agrees with the walk.

    The application asks the index which artifacts are retained instead of
    reading every stored document. Here each answer is also compared with the
    walk, and a disagreement fails the test, as does upkeep that failed
    unexpectedly, so every test that touches references is also a test of the
    index.
    """

    disagreements = artifact_reference_index.shadow_disagreements
    failures = artifact_reference_index.upkeep_failures
    yield
    if artifact_reference_index.shadow_disagreements != disagreements:
        pytest.fail("the artifact reference index disagreed with the reference walk")
    if artifact_reference_index.upkeep_failures != failures:
        pytest.fail("artifact reference index upkeep failed")


@pytest.fixture(scope="session", autouse=True)
def names_resolve_only_locally() -> Iterator[NameLookupGuard]:
    """No name beyond this computer resolves while tests run.

    A client that reaches a service by name is refused at the lookup, before
    anything is sent. Lookups of this computer's own names, such as a test's
    own loopback server, are untouched. A connection made straight to an
    address looks nothing up and is not seen here, as local_name_lookups
    describes. Refusals no test could answer for fail the session at its end.
    """

    guard = NameLookupGuard()
    with pytest.MonkeyPatch.context() as patch:
        guard.install(patch)
        yield guard
    if unanswered := guard.unanswered():
        pytest.fail(refused_unanswered(unanswered), pytrace=False)


@pytest.fixture(autouse=True)
def refused_lookups_fail_the_test(
    request: pytest.FixtureRequest, names_resolve_only_locally: NameLookupGuard
) -> Iterator[None]:
    """A test refused a lookup beyond this computer fails, even where it absorbed the refusal.

    The application treats a failed lookup as a host it could not reach and
    carries on, often successfully, so the refusal alone would pass unnoticed.
    Only lookups this test made count, its threads' included: a thread an
    earlier test left running is that test's, never this one's.
    """

    test = request.node.nodeid
    names_resolve_only_locally.begin(test)
    yield
    if refused := names_resolve_only_locally.end(test):
        pytest.fail(refused_during_test(refused), pytrace=False)


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
    # Certified once here, as startup would, so every copy starts current and
    # a test that never starts the app still has the index checked.
    assert artifact_reference_index.start_reference_index(db.SessionLocal)
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


def _refuse_download(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("A test never downloads a runtime.", request=request)


class _OfflineRuntimes(RuntimeProvisioner):
    """The application's runtime installer, refusing every download it is asked for.

    Activating a llama.cpp model asks for its runtime. An installer with a real
    connection fetches one from the internet in the background and then writes
    where it went into this process's environment, where every later test's
    settings read it as a runtime already configured.
    """

    def __init__(self, settings: Settings, **options: Any) -> None:
        refusing = options.get("client") is None
        if refusing:
            options["client"] = httpx.AsyncClient(transport=httpx.MockTransport(_refuse_download))
        super().__init__(settings, **options)
        # Closed with the installer, as a connection of its own making would be.
        self._owns_client = self._owns_client or refusing


@pytest.fixture(scope="session", autouse=True)
def runtimes_never_download() -> Iterator[None]:
    """Every application a test builds installs runtimes without a connection.

    Patched once for the session, not through each test's `monkeypatch`: a
    fixture that asks for that one sets it up before the test's own fixtures,
    so the test's patches would stay in place until after its application had
    shut down.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(application, "RuntimeProvisioner", _OfflineRuntimes)
        yield


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
