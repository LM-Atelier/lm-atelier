"""Materialized selections and committed results survive a real restored backup."""

import asyncio
import shutil
from pathlib import Path

import pytest
from test_media_recovery_api import _media
from test_media_recovery_restart import _open
from test_project_recovery_api import _project
from test_recovery_bulk_api import _apply, _batch, _trash
from test_recovery_expiry import manual_expiry as manual_expiry
from test_workflow_recovery_api import _seed as _workflow

from local_lm.backups import BackupManager
from local_lm.config import Settings


@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_restored_backup_retains_the_exact_bulk_selection_deadlines_and_replay(
    settings: Settings, tmp_path: Path, action: str
) -> None:
    async with _open(settings) as (app, client):
        project_id, chat_id = await _project(client)
        entry_id, _artifact_id = _media(app)
        family_id, _definition_id, _revision_id, _graph = _workflow()
        items = [
            await _trash(client, f"/api/projects/{project_id}", "backup-batch-project"),
            await _trash(client, f"/api/chats/{chat_id}", "backup-batch-chat"),
            await _trash(client, f"/api/artifact-library/{entry_id}", "backup-batch-media"),
            await _trash(client, f"/api/workflow-families/{family_id}", "backup-batch-workflow"),
        ]
        preview = await _batch(client, items, action)
        backup = await asyncio.to_thread(app.state.services.backups.create, include_media=True)
        verified = await asyncio.to_thread(app.state.services.backups.verify, backup.name)
        assert verified.verified and verified.media_included

    restored = Settings(
        data_dir=tmp_path / "restored-batch", dev=True, chat_engine="mock", media_engine="mock"
    )
    restored.prepare()
    for name in (backup.name, backup.name + ".media.zip"):
        shutil.copyfile(settings.backup_dir / name, restored.backup_dir / name)
    assert BackupManager(restored).request_restore(backup.name).restore_pending
    async with _open(restored) as (_app, client):
        page = await client.get("/api/recovery-items", params={"limit": 20})
        assert page.status_code == 200
        actual = {item["deletion_id"]: item for item in page.json()["items"]}
        assert actual == {item["deletion_id"]: item for item in items}
        response = await _apply(client, preview, "backup-materialized-batch")
        assert response.status_code == 200, response.text
        result = response.json()
        assert {member["deletion_id"] for member in result["results"]} == set(actual)
    async with _open(restored) as (_app, client):
        replay = await _apply(client, preview, "backup-materialized-batch")
        assert replay.status_code == 200 and replay.json() == result
