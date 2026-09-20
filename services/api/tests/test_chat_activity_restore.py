from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from test_chat_activity_transitions import activity, completed_turn, finish
from test_prior_turn_backup_restore import _running

from local_lm.backups import BackupManager
from local_lm.config import Settings


async def test_restore_keeps_existing_identity_and_distinguishes_a_reused_future_sequence(
    settings: Settings, tmp_path: Path
) -> None:
    async with _running(settings) as (app, client):
        original = await completed_turn(client)
        chat_id = original["run"]["chat_id"]
        original_activity = (await activity(client, chat_id))["last_output"]
        backup = await asyncio.to_thread(app.state.services.backups.create, include_media=False)
        backup = await asyncio.to_thread(app.state.services.backups.verify, backup.name)
        assert backup.verified
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={"text": "Describe a yellow circle.", "mode": "text"},
        )
        response.raise_for_status()
        await finish(client, response.json()["run"]["id"])
        discarded_future = (await activity(client, chat_id))["last_output"]

    restored_settings = Settings(
        data_dir=tmp_path / "restored", dev=True, chat_engine="mock", media_engine="mock"
    )
    restored_settings.prepare()
    shutil.copyfile(settings.backup_dir / backup.name, restored_settings.backup_dir / backup.name)
    manager = BackupManager(restored_settings)
    assert manager.request_restore(backup.name).restore_pending
    assert manager.apply_pending_restore()
    async with _running(restored_settings) as (_app, client):
        assert (await activity(client, chat_id))["last_output"] == original_activity
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={"text": "Describe a green triangle.", "mode": "text"},
        )
        response.raise_for_status()
        await finish(client, response.json()["run"]["id"])
        new_future = (await activity(client, chat_id))["last_output"]
        assert new_future["sequence"] == discarded_future["sequence"]
        assert new_future["id"] != discarded_future["id"]
        assert new_future["id"] != original_activity["id"]
        assert new_future["message_id"] != discarded_future["message_id"]
