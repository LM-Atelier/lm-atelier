"""Restore a regenerated canvas into an empty database and artifact store."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from test_source_fit_regeneration import completed_source
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.accepted_turn_context import accepted_context, resolve_accepted_workflow
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base, SessionLocal
from local_lm.exports import ProjectExporter
from local_lm.models import Chat, Project, Run, RunContextSnapshot
from local_lm.source_fit_image import replay_source_fit_image


@pytest.mark.parametrize("include_media", [True, False])
async def test_regenerated_canvas_restores_only_with_its_media(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    include_media: bool,
) -> None:
    original = await completed_source(app, client, monkeypatch)
    services = app.state.services
    async with services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/messages/{original['assistant_message']['id']}/regenerate",
            json={"idempotency_key": "regenerate-for-export", "settings": {}},
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            project = Project(name="Neutral canvas archive")
            session.add(project)
            session.flush()
            chat = session.get(Chat, original["run"]["chat_id"])
            assert chat is not None
            chat.project_id = project.id
            session.commit()
            run = session.get(Run, response.json()["run"]["id"])
            assert run is not None
            frozen = accepted_context(session, run)
            assert frozen is not None and frozen.source_fit is not None
            expected = replay_source_fit_image(
                session,
                services.orchestrator.artifacts,
                frozen.source_fit.image,
                selected_source_id=frozen.input_artifact_ids[0],
            ).content
            exporter = ProjectExporter(services.settings, services.orchestrator.artifacts)
            archive = exporter.export(session, project.id, include_media=include_media)
            content = services.orchestrator.artifacts.verified_bytes(
                archive,
                maximum_bytes=32 * 1024 * 1024,
            )

    restored_settings = Settings(data_dir=tmp_path / "restored")
    restored_settings.prepare()
    restored_store = ArtifactStore(restored_settings)
    engine = create_engine(f"sqlite:///{tmp_path / 'restored.sqlite3'}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine, expire_on_commit=False) as session:
            restored_exporter = ProjectExporter(restored_settings, restored_store)
            restored = restored_exporter.import_archive(session, BytesIO(content))
            session.commit()
            runs = list(
                session.scalars(
                    select(Run)
                    .join(Chat, Run.chat_id == Chat.id)
                    .where(Chat.project_id == restored.id)
                )
            )
            assert len(runs) == 2
            rows = [session.get(RunContextSnapshot, run.id) for run in runs]
            assert all(row is not None for row in rows)
            inherited = next(
                row
                for row in rows
                if row is not None and row.payload_json.get("source_run_id") is not None
            )
            source_id = inherited.payload_json["source_run_id"]
            source = session.get(Run, source_id)
            assert source is not None and source.id != original["run"]["id"]
            assert inherited.payload_json["source_message_id"] == source.user_message_id
            imported_run = session.get(Run, inherited.run_id)
            assert imported_run is not None
            if not include_media:
                assert inherited.payload_json["unavailable_reason"] == "imported_media_unavailable"
                assert inherited.payload_json["source_fit"] is None
                with pytest.raises(
                    ValueError, match="Accepted conversation context is unavailable"
                ):
                    accepted_context(session, imported_run)
                return
            snapshot = accepted_context(session, imported_run)
            assert snapshot is not None and snapshot.source_fit is not None
            assert snapshot.workflow is not None and snapshot.workflow.trusted is False
            assert snapshot.workflow.id != frozen.workflow_revision_id
            with pytest.raises(RuntimeError, match="no longer trusted"):
                resolve_accepted_workflow(session, snapshot.workflow)
            assert (snapshot.source_fit.canvas_width, snapshot.source_fit.canvas_height) == (4, 5)
            assert (
                replay_source_fit_image(
                    session,
                    restored_store,
                    snapshot.source_fit.image,
                    selected_source_id=snapshot.input_artifact_ids[0],
                ).content
                == expected
            )
    finally:
        engine.dispose()
