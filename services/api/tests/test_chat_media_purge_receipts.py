"""Deleting a generated result erases its names from older recovery receipts."""

import json

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_deletion import _image_exchange
from test_chat_recovery import _chat, _command, _history, _impact
from test_chat_recovery_replay import _change

from local_lm.db import SessionLocal
from local_lm.models import ArtifactLibraryEntry, RecoveryBatchRecord, RecoveryOperation


@pytest.mark.parametrize("protected", [False, True])
async def test_chat_purge_erases_old_media_receipt_names_only_when_the_result_is_released(
    client: AsyncClient,
    protected: bool,
) -> None:
    chat_id = await _chat(client)
    await _image_exchange(client, chat_id, "Create a picture of garden beds")
    artifact_id = next(part[-1] for part in _history(chat_id)["parts"] if part[-1])
    with SessionLocal() as session:
        entry = session.scalar(
            select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == artifact_id)
        )
        assert entry is not None
        entry_id = entry.id
        entry.display_name = "Garden output"
        entry.version += 1
        session.commit()
    media = await _change(
        client,
        f"/api/artifact-library/{entry_id}/deletion-impact",
        f"/api/artifact-library/{entry_id}/trash",
        "trash-garden-output",
    )
    batch_response = await client.post(
        "/api/recovery-items/batches",
        json={"deletion_ids": [media["deletion_id"]], "action": "purge"},
    )
    assert batch_response.status_code == 200, batch_response.text
    batch = batch_response.json()
    assert batch["items"][0]["display_label"] == "Garden output"
    await _change(
        client,
        f"/api/recovery-items/{media['deletion_id']}/impact",
        f"/api/recovery-items/{media['deletion_id']}/restore",
        "restore-garden-output",
    )
    if protected:
        favorite = await client.patch(f"/api/artifacts/{artifact_id}", json={"favorite": True})
        assert favorite.status_code == 200, favorite.text
    impact = await _impact(
        client, f"/api/chats/{chat_id}/deletion-impact?delete_generated_media=true"
    )
    chat_response = await client.post(
        f"/api/chats/{chat_id}/trash",
        json={**_command(impact, "trash-garden-chat"), "delete_generated_media": True},
    )
    assert chat_response.status_code == 200, chat_response.text
    deletion_id = chat_response.json()["deletion_id"]
    impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")

    purged = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={
            **_command(impact, "purge-garden-chat"),
            "acknowledgement": "permanently-delete",
        },
    )

    assert purged.status_code == 200, purged.text
    with SessionLocal() as session:
        entry = session.get(ArtifactLibraryEntry, entry_id)
        assert (entry is not None) is protected
        operation = session.get(
            RecoveryOperation, ("media_library_entry", entry_id, "trash-garden-output")
        )
        record = session.get(RecoveryBatchRecord, batch["batch_id"])
        assert operation is not None and record is not None
        label = "Garden output" if protected else "Deleted item"
        assert (
            operation.response_json["display_label"],
            record.preview_json["items"][0]["display_label"],
        ) == (label, label)
        if not protected:
            assert "Garden output" not in json.dumps(operation.response_json)
            assert "Garden output" not in json.dumps(record.preview_json)
        assert operation.operation_key == "trash-garden-output"
        assert operation.deletion_id == media["deletion_id"]
        assert record.id == batch["batch_id"]
