from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import text

from local_lm.config import Settings
from local_lm.db import SessionLocal


def test_direct_sessions_do_not_write_into_the_shared_template(
    settings: Settings, migrated_database_template: Path
) -> None:
    """A test opening SessionLocal() must reach its own copy, not the template.

    The template is copied into every later test, so one stray write here
    comes back as phantom state inside an unrelated file's assertions - the
    cross-file failure that only ever appeared "in company". Table-agnostic
    on purpose: it pins the binding, not any one model.
    """

    with SessionLocal() as session:
        session.execute(text("CREATE TABLE leak_probe (id INTEGER PRIMARY KEY)"))
        session.commit()

    own_database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(own_database) as own:
        own_tables = {row[0] for row in own.execute("SELECT name FROM sqlite_master")}
    assert "leak_probe" in own_tables, "the write did not reach this test's database"

    with sqlite3.connect(migrated_database_template) as template:
        shared_tables = {row[0] for row in template.execute("SELECT name FROM sqlite_master")}
    assert "leak_probe" not in shared_tables, (
        "a direct session wrote into the shared template; every later test "
        "copies it and inherits the row"
    )
