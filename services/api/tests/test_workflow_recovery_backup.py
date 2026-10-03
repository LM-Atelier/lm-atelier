"""Restored backups keep workflow deletion identity and the original expiry clock."""

import asyncio
import copy
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from test_chat_recovery import _command, _impact
from test_media_recovery_restart import _open
from test_recovery_expiry import manual_expiry as manual_expiry
from test_workflow_recovery_api import _seed
from test_workflow_recovery_graph import _consumer
from test_workflow_recovery_history import _history

from local_lm.backups import BackupManager
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    Job,
    RecoveryItem,
    RecoveryOperation,
    WorkflowFamily,
    WorkflowRevision,
)
from local_lm.recovery_maintenance import expire_recovery_batch

pytestmark = pytest.mark.usefixtures("manual_expiry")


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("action", ["restore", "expired-purge"])
async def test_workflow_backup_restore_preserves_the_clock_history_and_disabled_restore(
    settings: Settings, tmp_path: Path, history: bool, action: str
) -> None:
    async with _open(settings) as (app, client):
        family_id, definition_id, revision_id, _graph = _seed()
        with SessionLocal() as session:
            if history:
                _consumer(session, "run", revision_id, "complete")
            session.get(WorkflowRevision, revision_id).artifact_sha256 = "a" * 64
            session.commit()
        path = f"/api/workflow-families/{family_id}"
        preview = await _impact(client, f"{path}/deletion-impact")
        trash_command = _command(preview, "trash-workflow-before-backup")
        response = await client.post(f"{path}/trash", json=trash_command)
        assert response.status_code == 200, response.text
        trash_item = response.json()
        deletion_id = trash_item["deletion_id"]
        deleted_at = datetime.fromisoformat(trash_item["deleted_at"]).astimezone(UTC)
        deadline = datetime.fromisoformat(trash_item["purge_after"]).astimezone(UTC)
        with SessionLocal() as session:
            item = session.get(RecoveryItem, deletion_id)
            original_revision = item.subject_revision
            operation = session.scalar(
                select(RecoveryOperation).where(RecoveryOperation.deletion_id == deletion_id)
            )
            trash_result = copy.deepcopy(operation.response_json)
            graph = copy.deepcopy(session.get(WorkflowRevision, revision_id).api_graph_json)
        before = _history()
        backup = await asyncio.to_thread(app.state.services.backups.create, include_media=True)
        verified = await asyncio.to_thread(app.state.services.backups.verify, backup.name)
        assert verified.verified and verified.media_included

    restored_settings = Settings(
        data_dir=tmp_path / "restored-workflows", dev=True, chat_engine="mock", media_engine="mock"
    )
    restored_settings.prepare()
    for name in (backup.name, backup.name + ".media.zip"):
        shutil.copyfile(settings.backup_dir / name, restored_settings.backup_dir / name)
    assert BackupManager(restored_settings).request_restore(backup.name).restore_pending

    async with _open(restored_settings) as (app, client):
        assert not (restored_settings.state_dir / "restore-on-next-start.json").exists()
        assert (await client.get(f"/api/workflow-families/{family_id}")).status_code == 404
        replay = await client.post(f"{path}/trash", json=trash_command)
        assert replay.status_code == 200 and replay.json() == trash_item
        assert _history() == before
        with SessionLocal() as session:
            item = session.get(RecoveryItem, deletion_id)
            assert item.subject_id == family_id and item.subject_revision == original_revision
            assert item.deleted_at.replace(tzinfo=UTC) == deleted_at
            assert item.purge_after.replace(tzinfo=UTC) == deadline
            assert item.state == "recoverable"
            operation = session.scalar(
                select(RecoveryOperation).where(RecoveryOperation.deletion_id == deletion_id)
            )
            assert operation.response_json == trash_result
            revision = session.get(WorkflowRevision, revision_id)
            assert revision.workflow_id == definition_id and revision.artifact_sha256 == "a" * 64
            assert revision.api_graph_json == graph and not revision.trusted
            assert not session.get(WorkflowFamily, family_id).enabled
            assert session.scalar(select(func.count()).select_from(Job)) == 0
        early = await expire_recovery_batch(
            app.state.services, deadline - timedelta(microseconds=1)
        )
        assert early.examined == early.purged == early.deferred == 0
        if action == "restore":
            impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
            command = _command(impact, "restore-workflow-backup")
            response = await client.post(f"/api/recovery-items/{deletion_id}/restore", json=command)
            repeated = await client.post(f"/api/recovery-items/{deletion_id}/restore", json=command)
            assert response.status_code == repeated.status_code == 200
            assert response.json() == repeated.json()
        else:
            expired = await expire_recovery_batch(app.state.services, deadline + timedelta(days=7))
            assert expired.examined == expired.purged == 1 and expired.deferred == 0
        assert (await client.get("/api/recovery-items", params={"kind": "workflow_family"})).json()[
            "items"
        ] == []

    async with _open(restored_settings) as (_app, _client):
        assert _history() == before
        with SessionLocal() as session:
            item = session.get(RecoveryItem, deletion_id)
            revision = session.get(WorkflowRevision, revision_id)
            family = session.get(WorkflowFamily, family_id)
            if action == "restore":
                assert item is None
                assert family is not None and not family.enabled
                assert revision is not None and revision.api_graph_json == graph
                assert not revision.trusted
            else:
                assert item.state == "purged" and item.subject_id == family_id
                assert item.deleted_at.replace(tzinfo=UTC) == deleted_at
                assert item.purge_after.replace(tzinfo=UTC) == deadline
                if history:
                    assert family is not None and not family.enabled
                    assert revision is not None and revision.api_graph_json == {}
                    assert revision.workflow_id == definition_id
                    assert revision.artifact_sha256 == "a" * 64 and not revision.trusted
                else:
                    assert family is revision is None
            assert session.scalar(select(func.count()).select_from(Job)) == 0
