from __future__ import annotations

import asyncio
import copy
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from httpx2 import ASGITransport, AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import select
from test_project_work_plans import _archive

from local_lm.backups import BackupManager
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.main import create_app
from local_lm.models import Chat, Run


@asynccontextmanager
async def _running(settings: Settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            auth = await client.post("/api/session")
            client.headers["x-local-lm-csrf"] = auth.json()["csrf_token"]
            yield app, client


async def _completed(client: AsyncClient, run_id: str) -> None:
    async def read() -> dict[str, Any]:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        run: dict[str, Any] = response.json()
        return run

    await wait_for_terminal_status(read, what=f"run {run_id}", expected="complete")


@pytest.mark.parametrize("media", [False, True], ids=["text", "retained-image"])
async def test_database_restore_preserves_accepted_edit_and_retained_inputs(
    settings: Settings, tmp_path: Path, media: bool
) -> None:
    async with _running(settings) as (app, client):
        _source, edited, _content = await _archive(
            app, client, complete=True, output_count=3 if media else 1
        )
        if media:
            with SessionLocal() as session:
                first = session.get(Run, edited["run"]["id"])
                assert first is not None
                input_id = first.provenance_json["outputs"][0]["artifact_id"]
            response = await client.post(
                f"/api/messages/{edited['user_message']['id']}/edits",
                json={
                    "text": "Recolor the paper boat orange",
                    "mode": "image",
                    "source_run_id": edited["run"]["id"],
                    "input_artifact_ids": [input_id],
                    "workflow_revision_id": None,
                    "idempotency_key": "backup-retained-input",
                },
            )
            assert response.status_code == 202, response.text
            edited = response.json()
            await _completed(client, edited["run"]["id"])
        run_id = edited["run"]["id"]
        from local_lm.accepted_turn_context import accepted_context
        from local_lm.models import RunContextArtifact, RunContextSnapshot

        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None
            context = accepted_context(session, run)
            assert context is not None
            row = session.get(RunContextSnapshot, run_id)
            assert row is not None
            before_payload = copy.deepcopy(row.payload_json)
            before_digest = run.provenance_json["accepted_context_sha256"]
            retained = set(
                session.scalars(
                    select(RunContextArtifact.artifact_id).where(
                        RunContextArtifact.run_id == run_id
                    )
                )
            )
            if media:
                assert input_id in retained
            chat = session.get(Chat, run.chat_id)
            assert chat is not None
            chat_id, active_head = chat.id, chat.active_head_message_id
            message_id = run.user_message_id
        backup = await asyncio.to_thread(app.state.services.backups.create, include_media=media)
        backup = await asyncio.to_thread(app.state.services.backups.verify, backup.name)
        assert backup.verified

    # No source app, scheduler or maintenance task remains when the global
    # database session is rebound to this independently restored profile.
    restored_settings = Settings(
        data_dir=tmp_path / "restored", dev=True, chat_engine="mock", media_engine="mock"
    )
    restored_settings.prepare()
    shutil.copyfile(settings.backup_dir / backup.name, restored_settings.backup_dir / backup.name)
    if media:
        archive_name = backup.name + ".media.zip"
        shutil.copyfile(
            settings.backup_dir / archive_name, restored_settings.backup_dir / archive_name
        )
    manager = BackupManager(restored_settings)
    assert manager.request_restore(backup.name).restore_pending
    assert manager.apply_pending_restore()

    async with _running(restored_settings) as (app, client):
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None
            restored = accepted_context(session, run)
            assert restored is not None
            row = session.get(RunContextSnapshot, run_id)
            assert row is not None and row.payload_json == before_payload
            assert run.provenance_json["accepted_context_sha256"] == before_digest
            assert (
                set(
                    session.scalars(
                        select(RunContextArtifact.artifact_id).where(
                            RunContextArtifact.run_id == run_id
                        )
                    )
                )
                == retained
            )
            chat = session.get(Chat, chat_id)
            assert chat is not None and chat.active_head_message_id == active_head
        source = await client.get(f"/api/messages/{message_id}/edit-source")
        assert source.status_code == 200, source.text
        if media:
            assert input_id in source.json()["input_artifact_ids"]
            content = await client.get(f"/api/artifacts/{input_id}/content")
            assert content.status_code == 200 and content.content
        queued = await client.post(
            f"/api/messages/{message_id}/edits",
            json={
                "text": "Describe the paper boat again",
                "source_run_id": source.json()["source_run_id"],
                "source_snapshot_sha256": source.json()["source_snapshot_sha256"],
                "idempotency_key": "edit-restored-version",
            },
        )
        assert queued.status_code == 202, queued.text
        await _completed(client, queued.json()["run"]["id"])
        with SessionLocal() as session:
            chat = session.get(Chat, chat_id)
            assert chat is not None and chat.active_head_message_id == active_head
