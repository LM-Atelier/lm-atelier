"""A real restored backup retains recovery identity, replay and original expiry."""

import asyncio
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from test_chat_deletion import _text_exchange
from test_chat_recovery import _command, _history, _impact
from test_media_recovery import CONTENT, _seed
from test_media_recovery_restart import _open
from test_project_recovery_api import _project
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Job,
    Message,
    MessagePart,
    Project,
    RecoveryItem,
)
from local_lm.recovery_maintenance import expire_recovery_batch


@pytest.mark.parametrize("kind", ["chat", "project", "media_library_entry"])
@pytest.mark.parametrize("action", ["restore", "expired-purge"])
async def test_backup_restore_keeps_original_deletion_identity_and_deadline(
    settings: Settings, tmp_path: Path, kind: str, action: str
) -> None:
    async with _open(settings) as (app, client):
        project_id, chat_id = await _project(client)
        exchange = await _text_exchange(client, chat_id, "Keep the garden paths clear")
        leaf_id = exchange["assistant_message"]["id"]
        with SessionLocal() as session:
            artifact, entry, _collection, _tag = _seed(app.state.services.artifacts, session)
            artifact_id, entry_id = artifact.id, entry.id
            session.add(
                MessagePart(
                    message_id=leaf_id,
                    position=len(session.get(Message, leaf_id).parts),
                    type="image",
                    artifact_id=artifact_id,
                )
            )
            session.commit()
            job_count = session.scalar(select(func.count()).select_from(Job))
        before = _history(chat_id)
        path = {
            "chat": f"/api/chats/{chat_id}",
            "project": f"/api/projects/{project_id}",
            "media_library_entry": f"/api/artifact-library/{entry_id}",
        }[kind]
        preview = await _impact(client, f"{path}/deletion-impact")
        trash_command = _command(preview, "trash-before-backup")
        response = await client.post(f"{path}/trash", json=trash_command)
        assert response.status_code == 200, response.text
        item = response.json()
        backup = await asyncio.to_thread(app.state.services.backups.create, include_media=True)
        verified = await asyncio.to_thread(app.state.services.backups.verify, backup.name)
        assert verified.verified and verified.media_included

    restored_settings = Settings(
        data_dir=tmp_path / "restored", dev=True, chat_engine="mock", media_engine="mock"
    )
    restored_settings.prepare()
    for name in (backup.name, backup.name + ".media.zip"):
        shutil.copyfile(settings.backup_dir / name, restored_settings.backup_dir / name)
    from local_lm.backups import BackupManager

    assert BackupManager(restored_settings).request_restore(backup.name).restore_pending
    deletion_id = item["deletion_id"]
    deadline = datetime.fromisoformat(item["purge_after"]).astimezone(UTC)
    deleted_at = datetime.fromisoformat(item["deleted_at"]).astimezone(UTC)

    async with _open(restored_settings) as (app, client):
        assert not (restored_settings.state_dir / "restore-on-next-start.json").exists()
        page = await client.get("/api/recovery-items")
        assert page.status_code == 200 and page.json()["items"] == [item]
        replay = await client.post(f"{path}/trash", json=trash_command)
        assert replay.status_code == 200 and replay.json() == item
        assert _history(chat_id) == before
        with SessionLocal() as session:
            row = session.get(RecoveryItem, deletion_id)
            assert row.deleted_at.replace(tzinfo=UTC) == deleted_at
            assert row.purge_after.replace(tzinfo=UTC) == deadline
            assert session.scalar(select(func.count()).select_from(Job)) == job_count
            assert (
                app.state.services.artifacts.resolve(
                    session.get(Artifact, artifact_id)
                ).read_bytes()
                == CONTENT
            )
        early = await expire_recovery_batch(
            app.state.services, deadline - timedelta(microseconds=1)
        )
        assert early.examined == early.purged == early.deferred == 0
        if action == "restore":
            impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
            restored = await client.post(
                f"/api/recovery-items/{deletion_id}/restore",
                json=_command(impact, "restore-after-backup"),
            )
            assert restored.status_code == 200, restored.text
            assert _history(chat_id) == before
        else:
            expired = await expire_recovery_batch(app.state.services, deadline + timedelta(days=7))
            assert expired.examined == expired.purged == 1 and expired.deferred == 0
            with SessionLocal() as session:
                row = session.get(RecoveryItem, deletion_id)
                assert row.state == "purged"
                assert row.deleted_at.replace(tzinfo=UTC) == deleted_at
                assert row.purge_after.replace(tzinfo=UTC) == deadline
                if kind == "chat":
                    assert session.get(Chat, chat_id) is None
                elif kind == "project":
                    assert session.get(Project, project_id) is None
                    assert session.get(Chat, chat_id).project_id is None
                else:
                    assert session.get(ArtifactLibraryEntry, entry_id) is None
        assert (await client.get("/api/recovery-items")).json()["items"] == []
        assert (await client.get(f"/api/artifacts/{artifact_id}/content")).content == CONTENT

    async with _open(restored_settings) as (app, client):
        assert (await client.get("/api/recovery-items")).json()["items"] == []
        with SessionLocal() as session:
            row = session.get(RecoveryItem, deletion_id)
            if action == "restore":
                assert row is None
                assert _history(chat_id) == before
            else:
                assert row.state == "purged" and row.purge_after.replace(tzinfo=UTC) == deadline
            assert (
                app.state.services.artifacts.resolve(
                    session.get(Artifact, artifact_id)
                ).read_bytes()
                == CONTENT
            )
