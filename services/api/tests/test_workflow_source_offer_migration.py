"""Source-plan offer constraints retain earlier accepted downloads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.models import Job, WorkflowDefinition, WorkflowInstallOfferDownload, WorkflowRevision

PARENT = "e48d903a7b26"


def _populated_database(tmp_path: Path) -> tuple[Settings, Path]:
    settings = Settings(data_dir=tmp_path / "workflow-source-migration")
    settings.prepare()
    command.upgrade(alembic_config(settings), PARENT)
    database = settings.state_dir / "local-lm.sqlite3"
    engine = create_engine(settings.database_url)
    with Session(engine) as session:
        session.add(
            WorkflowDefinition(id="wf_migration", name="Neutral source", operation="text_to_image")
        )
        session.flush()
        session.add(
            WorkflowRevision(
                id="wfrev_migration", workflow_id="wf_migration", version=1, engine="comfyui"
            )
        )
        session.add(
            Job(
                id="job_migration",
                kind="download",
                status="paused",
                payload_json={"label": "Neutral download"},
            )
        )
        session.flush()
        session.execute(
            text(
                "INSERT INTO workflow_install_offers "
                "(id, workflow_revision_id, workflow_artifact_sha256, dependency_contract_sha256, "
                "binding_plan_sha256, offer_sha256, selections_json, assets_json, plan_count, "
                "total_bytes, status, created_at, updated_at) "
                "VALUES ('offer_migration', 'wfrev_migration', :sha, :sha, :sha, :sha, '[]', '[]', "
                "1, 17, 'queued', '2026-01-01', '2026-01-01')"
            ),
            {"sha": "a" * 64},
        )
        session.add(
            WorkflowInstallOfferDownload(
                id="link_migration",
                offer_id="offer_migration",
                offer_sha256="a" * 64,
                job_id="job_migration",
                request_sha256="b" * 64,
                request_json={"label": "Neutral download"},
            )
        )
        session.commit()
    engine.dispose()
    return settings, database


def test_source_offer_migration_preserves_existing_offers_jobs_and_links(tmp_path: Path) -> None:
    settings, database = _populated_database(tmp_path)
    config = alembic_config(settings)
    with sqlite3.connect(database) as connection:
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(workflow_install_offers)")
        ]
        before_offer = connection.execute("SELECT * FROM workflow_install_offers").fetchall()
        before_links = connection.execute(
            "SELECT * FROM workflow_install_offer_downloads"
        ).fetchall()
        before_jobs = connection.execute("SELECT * FROM jobs").fetchall()
    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT " + ", ".join(columns) + " FROM workflow_install_offers"
            ).fetchall()
            == before_offer
        )
        assert connection.execute(
            "SELECT source_plan_id FROM workflow_install_offers"
        ).fetchall() == [(None,)]
        assert (
            connection.execute("SELECT * FROM workflow_install_offer_downloads").fetchall()
            == before_links
        )
        assert connection.execute("SELECT * FROM jobs").fetchall() == before_jobs
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    command.downgrade(config, PARENT)
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT * FROM workflow_install_offers").fetchall() == before_offer
        )
        assert (
            connection.execute("SELECT * FROM workflow_install_offer_downloads").fetchall()
            == before_links
        )
        with pytest.raises(sqlite3.IntegrityError, match="ck_workflow_install_offer_plan_count"):
            connection.execute("UPDATE workflow_install_offers SET plan_count = 0")


def test_zero_download_offer_requires_a_unique_retained_source_plan(tmp_path: Path) -> None:
    settings, database = _populated_database(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, "head")
    with sqlite3.connect(database, isolation_level=None) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "INSERT INTO workflow_package_install_plans "
            "(id, plan_sha256, request_json, preflight_json, created_at, updated_at) "
            "VALUES ('source_migration', ?, '{}', '{}', '2026-01-01', '2026-01-01')",
            ("c" * 64,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="ck_workflow_install_offer_plan_count"):
            connection.execute("UPDATE workflow_install_offers SET plan_count = 0")
        with pytest.raises(sqlite3.IntegrityError, match="ck_workflow_install_offer_total_bytes"):
            connection.execute("UPDATE workflow_install_offers SET total_bytes = 0")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute("UPDATE workflow_install_offers SET source_plan_id = 'missing'")
        connection.execute("DELETE FROM workflow_install_offer_downloads")
        connection.execute(
            "UPDATE workflow_install_offers SET source_plan_id = 'source_migration', "
            "plan_count = 0, total_bytes = 0"
        )
        with pytest.raises(sqlite3.IntegrityError, match="ck_workflow_install_offer_plan_count"):
            connection.execute("UPDATE workflow_install_offers SET plan_count = -1")
        with pytest.raises(sqlite3.IntegrityError, match="ck_workflow_install_offer_total_bytes"):
            connection.execute("UPDATE workflow_install_offers SET total_bytes = -1")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute("DELETE FROM workflow_package_install_plans")
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            connection.execute(
                "INSERT INTO workflow_install_offers "
                "(id, workflow_revision_id, workflow_artifact_sha256, dependency_contract_sha256, "
                "binding_plan_sha256, offer_sha256, selections_json, assets_json, plan_count, "
                "total_bytes, status, created_at, updated_at, source_plan_id) "
                "SELECT 'duplicate_offer', workflow_revision_id, workflow_artifact_sha256, "
                "dependency_contract_sha256, binding_plan_sha256, offer_sha256, selections_json, "
                "assets_json, plan_count, total_bytes, status, created_at, updated_at, "
                "source_plan_id "
                "FROM workflow_install_offers"
            )
    with pytest.raises(RuntimeError, match="Accepted workflow source plans require"):
        command.downgrade(config, PARENT)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT source_plan_id, plan_count, total_bytes FROM workflow_install_offers"
        ).fetchall() == [("source_migration", 0, 0)]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
