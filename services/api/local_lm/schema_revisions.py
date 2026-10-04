"""The database schema revisions this build's migrations know."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


@cache
def known_revisions() -> frozenset[str]:
    """Every revision in this build's migration set, read once per process.

    Data recorded at any of these can be opened, upgraded first if it is older.
    Data recorded at any other revision was made by a newer build, or by none,
    and this build cannot open it. Reading the set imports every migration
    module; the modules ship inside the build, so the answer cannot change while
    the process runs.
    """

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    return frozenset(
        script.revision for script in ScriptDirectory.from_config(config).walk_revisions()
    )
