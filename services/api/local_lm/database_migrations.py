from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from .config import Settings


def alembic_config(settings: Settings) -> Config:
    package_root = Path(__file__).resolve().parent
    config = Config()
    config.set_main_option("script_location", str(package_root / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    config.attributes["settings"] = settings
    return config


def upgrade_database(settings: Settings) -> None:
    command.upgrade(alembic_config(settings), "head")
