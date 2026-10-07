"""Unpreviewed Project deletion cannot bypass recoverable deletion."""

from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_deletion import _text_exchange
from test_chat_recovery import _history
from test_project_recovery_api import _project
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm.db import SessionLocal
from local_lm.models import Project, RecoveryItem, RecoveryOperation


async def test_unpreviewed_project_delete_refuses_without_losing_configuration_or_history(
    client: AsyncClient,
) -> None:
    project_id, chat_id = await _project(client)
    await _text_exchange(client, chat_id, "Keep the spacing between garden beds")
    history = _history(chat_id)
    before = (await client.get(f"/api/projects/{project_id}")).json()
    refused = await client.delete(f"/api/projects/{project_id}")
    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "recovery-preview-required"
    assert (await client.get(f"/api/projects/{project_id}")).json() == before
    assert _history(chat_id) == history
    with SessionLocal() as session:
        assert session.get(Project, project_id) is not None
        assert session.scalar(select(RecoveryItem)) is None
        assert session.scalar(select(RecoveryOperation)) is None
    assert (await client.delete("/api/projects/nonexistent-garden")).status_code == 404
