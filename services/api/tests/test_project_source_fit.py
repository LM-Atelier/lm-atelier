"""Source-fit contexts travel with project media, never with missing authority."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_accepted_source_fit import make_run, save
from test_source_fit_image import artifact_session as artifact_session

from local_lm.accepted_turn_context import accepted_context
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.domain import utcnow
from local_lm.exports import ProjectExporter
from local_lm.models import Chat, Message, Project, Run, RunContextSnapshot
from local_lm.source_fit_image import replay_source_fit_image


@pytest.mark.parametrize("disposition", ["retained", "without_media", "removed_context"])
def test_project_round_trip_preserves_only_available_source_fit(
    artifact_session: tuple[ArtifactStore, Session],
    tmp_path: Path,
    disposition: str,
) -> None:
    store, session = artifact_session
    run, recipe = make_run(artifact_session)
    save(session, run, recipe)
    project = Project(name="Source geometry archive")
    session.add(project)
    session.flush()
    chat = session.get(Chat, run.chat_id)
    assert chat is not None
    chat.project_id = project.id
    if disposition == "removed_context":
        message = session.get(Message, run.user_message_id)
        assert message is not None
        message.content_removed_at = utcnow()
    session.commit()

    settings = Settings(data_dir=tmp_path / "exporter")
    settings.prepare()
    exporter = ProjectExporter(settings, store)
    archive = exporter.export(
        session,
        project.id,
        include_media=disposition != "without_media",
    )
    content = store.verified_bytes(archive, maximum_bytes=32 * 1024 * 1024)
    with ZipFile(BytesIO(content)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    payload = manifest["accepted_contexts"][0]["payload_json"]
    if disposition == "removed_context":
        assert payload["unavailable_reason"] == "removed_context"
        assert payload["source_fit"] is None
    else:
        assert (
            payload["source_fit"]["image"]["prepared_artifact_id"]
            == recipe.image.prepared_artifact_id
        )

    imported = exporter.import_archive(session, BytesIO(content))
    session.commit()
    imported_run = session.scalar(
        select(Run).join(Chat, Run.chat_id == Chat.id).where(Chat.project_id == imported.id)
    )
    assert imported_run is not None
    assert imported_run.id != run.id
    snapshot_row = session.get(RunContextSnapshot, imported_run.id)
    assert snapshot_row is not None
    if disposition != "retained":
        assert snapshot_row.payload_json["source_fit"] is None
        assert snapshot_row.payload_json["unavailable_reason"] == (
            "removed_context" if disposition == "removed_context" else "imported_media_unavailable"
        )
        with pytest.raises(ValueError, match="Accepted conversation context is unavailable"):
            accepted_context(session, imported_run)
    else:
        snapshot = accepted_context(session, imported_run)
        assert snapshot is not None
        assert snapshot.source_fit is not None
        assert snapshot.workflow is not None
        assert snapshot.workflow.id == snapshot.workflow_revision_id != run.workflow_revision_id
        assert snapshot.workflow.trusted is False
        prepared = replay_source_fit_image(
            session,
            store,
            snapshot.source_fit.image,
            selected_source_id=snapshot.input_artifact_ids[0],
        )
        assert (prepared.width, prepared.height) == (2, 3)
        assert (snapshot.source_fit.canvas_width, snapshot.source_fit.canvas_height) == (4, 5)
