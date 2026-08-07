from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.script.revision import ResolutionError
from alembic.util.exc import CommandError

from .backups import BackupManager
from .config import Settings

logger = logging.getLogger(__name__)


class DatabaseVersionError(RuntimeError):
    """The database revision is not supported by this application build."""


class DatabaseUpgradeError(RuntimeError):
    """The upgrade failed and the previous data is waiting to be restored."""


def alembic_config(settings: Settings) -> Config:
    package_root = Path(__file__).resolve().parent
    config = Config()
    config.set_main_option("script_location", str(package_root / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    config.attributes["settings"] = settings
    return config


def _recorded_revisions(database: Path) -> set[str]:
    """Read the revisions the data claims, without opening the ORM."""

    if not database.is_file():
        return set()
    try:
        with closing(sqlite3.connect(database)) as connection:
            rows = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    except sqlite3.DatabaseError:
        # No version table yet, or unreadable. Either way the upgrade has work
        # to do and deserves the protection below.
        return set()
    return {str(row[0]) for row in rows}


def _upgrade_is_pending(config: Config, settings: Settings) -> bool:
    database = settings.state_dir / "local-lm.sqlite3"
    try:
        heads = set(ScriptDirectory.from_config(config).get_heads())
    except CommandError:
        return True
    return _recorded_revisions(database) != heads


def upgrade_database(settings: Settings) -> None:
    """Bring the data to this build's schema, or give it back unchanged.

    A partly-applied upgrade cannot be undone in place. SQLite's Python driver
    leaves DDL standing when the surrounding transaction fails, while the
    revision marker - ordinary row data - rolls back. The schema then holds half
    of a migration that the data still says was never applied, so every later
    start replays it and fails on what already exists. Guards inside a migration
    cannot close this, because a killed process never reaches them.

    So the upgrade is made recoverable rather than atomic: snapshot first, and on
    failure ask for that snapshot back on the next start. The restore itself
    already runs early enough to matter - `apply_pending_restore` happens before
    the database is opened.
    """

    config = alembic_config(settings)
    manager = BackupManager(settings)
    snapshot: str | None = None
    if _upgrade_is_pending(config, settings):
        try:
            snapshot = manager.create().name
        except Exception:
            # A missing snapshot is worth reporting but not worth refusing to
            # start over: without it the upgrade is exactly as safe as it was
            # before this protection existed.
            logger.warning("Could not snapshot the data before upgrading", exc_info=True)
    try:
        command.upgrade(config, "head")
    except CommandError as error:
        # The data is newer than this build. Nothing was applied, so there is
        # nothing to give back and a restore would be the wrong answer.
        if not isinstance(error.__cause__, ResolutionError):
            raise
        raise DatabaseVersionError(
            "This LM Atelier data uses a database schema revision that this "
            "build does not recognize. Install the latest LM Atelier version "
            "and keep the existing data folder."
        ) from error
    except Exception as error:
        if snapshot is None:
            raise
        manager.request_restore(snapshot)
        raise DatabaseUpgradeError(
            "The data could not be upgraded to this version of LM Atelier. "
            "Restart LM Atelier to return to your data as it was before the "
            "upgrade started."
        ) from error
