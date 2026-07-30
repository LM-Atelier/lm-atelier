from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script.revision import ResolutionError
from alembic.util.exc import CommandError

from .config import Settings


class DatabaseVersionError(RuntimeError):
    """The database revision is not supported by this application build."""


def alembic_config(settings: Settings) -> Config:
    package_root = Path(__file__).resolve().parent
    config = Config()
    config.set_main_option("script_location", str(package_root / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    config.attributes["settings"] = settings
    return config


def upgrade_database(settings: Settings) -> None:
    try:
        command.upgrade(alembic_config(settings), "head")
    except CommandError as error:
        if not isinstance(error.__cause__, ResolutionError):
            raise
        raise DatabaseVersionError(
            "This LM Atelier data uses a database schema revision that this "
            "build does not recognize. Install the latest LM Atelier version "
            "and keep the existing data folder."
        ) from error
