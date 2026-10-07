"""Studio cannot take configuration from a deleted source conversation."""

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_chat_deletion import _text_exchange
from test_chat_recovery import _chat, _command, _history, _impact
from test_studio_uploads import _PNG, _upload

from local_lm.db import SessionLocal
from local_lm.models import Chat, Job


@pytest.mark.parametrize("existing", [False, True])
async def test_studio_refuses_deleted_sources_and_preserves_independent_sessions(
    client: AsyncClient, existing: bool
) -> None:
    chat_id = await _chat(client)
    assert (
        await client.patch(
            f"/api/chats/{chat_id}",
            json={"generation_settings_json": {"image": {"seed": 17}}},
        )
    ).status_code == 200
    await _text_exchange(client, chat_id, "Keep the garden paths clear")
    uploaded = await _upload(client, "garden.png", "image/png", _PNG)
    payload = {"source_artifact_id": uploaded["id"], "source_chat_id": chat_id}
    studio_id = None
    if existing:
        opened = await client.post("/api/studio/sessions", json=payload)
        assert opened.status_code == 200, opened.text
        studio_id = opened.json()["id"]
    before = _history(chat_id)
    with SessionLocal() as session:
        chat_count = session.scalar(select(func.count()).select_from(Chat))
        job_count = session.scalar(select(func.count()).select_from(Job))

    preview = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(preview, "trash-studio-source")
    )
    assert trashed.status_code == 200, trashed.text
    deletion_id = trashed.json()["deletion_id"]
    refused = await client.post("/api/studio/sessions", json=payload)
    assert refused.status_code == 404, refused.text
    assert refused.json()["code"] == "chat-not-found"
    assert _history(chat_id) == before
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Chat)) == chat_count
        assert session.scalar(select(func.count()).select_from(Job)) == job_count
    assert (await client.get(f"/api/artifacts/{uploaded['id']}/content")).content == _PNG
    if studio_id:
        assert (await client.get(f"/api/studio/sessions/{studio_id}")).status_code == 200
        independent = await client.post(
            "/api/studio/sessions", json={"source_artifact_id": uploaded["id"]}
        )
        assert independent.status_code == 200 and independent.json()["id"] == studio_id

    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    assert (
        await client.post(
            f"/api/recovery-items/{deletion_id}/restore",
            json=_command(preview, "restore-studio-source"),
        )
    ).status_code == 200
    reopened = await client.post("/api/studio/sessions", json=payload)
    assert reopened.status_code == 200, reopened.text
    assert reopened.json()["generation_settings_json"]["image"] == {"seed": 17}
    if studio_id:
        assert reopened.json()["id"] == studio_id
    studio_id = reopened.json()["id"]
    assert _history(chat_id) == before

    preview = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(preview, "trash-studio-source-again")
    )
    assert trashed.status_code == 200
    deletion_id = trashed.json()["deletion_id"]
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    assert (
        await client.post(
            f"/api/recovery-items/{deletion_id}/purge",
            json={
                **_command(preview, "purge-studio-source"),
                "acknowledgement": "permanently-delete",
            },
        )
    ).status_code == 200
    assert (await client.post("/api/studio/sessions", json=payload)).status_code == 404
    assert (await client.get(f"/api/studio/sessions/{studio_id}")).status_code == 200
    assert (await client.get(f"/api/artifacts/{uploaded['id']}/content")).content == _PNG
