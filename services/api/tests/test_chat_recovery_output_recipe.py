"""A generation record follows its conversation's recovery visibility."""

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_chat_recovery import _command, _history, _impact
from test_output_recipe_api import _generated, _record
from test_output_recipe_bundle import (
    _bundle,
    _engine_png,
    _make_output,
)
from test_output_recipe_bundle import (
    _generated as _generated_bundle,
)

from local_lm.db import SessionLocal
from local_lm.models import Run


@pytest.mark.parametrize("prompts", ["include", "omit"])
@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_a_deleted_conversations_output_record_is_hidden_until_restore(
    client: AsyncClient, prompts: str, action: str
) -> None:
    run, artifact_id = await _generated(client, "Draw a garden with green leaves")
    chat_id = run["chat_id"]
    before = _history(chat_id)
    original = await _record(client, run["id"], artifact_id, prompts)
    assert original.status_code == 200
    preview = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(preview, "trash-record-garden")
    )
    assert trashed.status_code == 200
    assert _history(chat_id) == before
    hidden = await _record(client, run["id"], artifact_id, prompts)
    assert hidden.status_code == 404
    assert hidden.json()["code"] == "output-recipe-run-not-found"
    assert "x-output-recipe-digest" not in hidden.headers
    assert (await client.get(f"/api/artifacts/{artifact_id}/content")).status_code == 200
    deletion_id = trashed.json()["deletion_id"]
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    command = _command(preview, action + "-record-garden")
    if action == "purge":
        command["acknowledgement"] = "permanently-delete"
    changed = await client.post(f"/api/recovery-items/{deletion_id}/{action}", json=command)
    assert changed.status_code == 200
    result = await _record(client, run["id"], artifact_id, prompts)
    if action == "restore":
        assert result.status_code == 200
        assert result.content == original.content
        assert (
            result.headers["x-output-recipe-digest"] == original.headers["x-output-recipe-digest"]
        )
        assert _history(chat_id) == before
    else:
        assert result.status_code == 404
    assert (await client.get(f"/api/artifacts/{artifact_id}/content")).status_code == 200


@pytest.mark.parametrize("prompts", ["include", "omit"])
@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_a_picture_record_bundle_follows_its_deleted_conversation(
    app: FastAPI, client: AsyncClient, prompts: str, action: str
) -> None:
    run_id = await _generated_bundle(client, "A garden with wide paths")
    artifact_id = _make_output(app, run_id, _engine_png())
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        chat_id = run.chat_id
    history = _history(chat_id)
    record = await _record(client, run_id, artifact_id, prompts)
    assert record.status_code == 200
    digest = record.headers["x-output-recipe-digest"]
    original = await _bundle(client, run_id, artifact_id, prompts, digest)
    assert original.status_code == 200
    preview = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(preview, "trash-bundle-garden")
    )
    assert trashed.status_code == 200
    hidden = await _bundle(client, run_id, artifact_id, prompts, digest)
    assert hidden.status_code == 404
    assert hidden.json()["code"] == "output-recipe-run-not-found"
    assert "x-output-recipe-digest" not in hidden.headers
    assert _history(chat_id) == history
    deletion_id = trashed.json()["deletion_id"]
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    command = _command(preview, action + "-bundle-garden")
    if action == "purge":
        command["acknowledgement"] = "permanently-delete"
    changed = await client.post(f"/api/recovery-items/{deletion_id}/{action}", json=command)
    assert changed.status_code == 200
    result = await _bundle(client, run_id, artifact_id, prompts, digest)
    if action == "restore":
        assert result.status_code == 200
        assert result.content == original.content
        assert result.headers["x-output-recipe-digest"] == digest
        assert _history(chat_id) == history
    else:
        assert result.status_code == 404
    assert (await client.get(f"/api/artifacts/{artifact_id}/content")).status_code == 200
