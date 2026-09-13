"""Changes within a configured chat require another deletion preview."""

from datetime import UTC, datetime, timedelta

import pytest
from httpx2 import AsyncClient
from test_empty_chat_deletion_api import EXECUTE, _execute_body, _exists, _preview

from local_lm.db import SessionLocal
from local_lm.models import Chat


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unchanged", "title", "draft"])
async def test_existing_configuration_needs_current_acknowledgement(
    client: AsyncClient, change: str
) -> None:
    with SessionLocal() as session:
        row = Chat(
            title="Landscape study",
            draft_prompt="A blue ceramic bowl",
            created_at=datetime.now(UTC) - timedelta(hours=48),
        )
        session.add(row)
        session.commit()
        chat_id = row.id

    preview = await _preview(client, [chat_id], include_configured=True)
    assert preview["configured_count"] == 1
    if change == "title":
        updated = await client.patch(f"/api/chats/{chat_id}", json={"title": "Mountain study"})
        assert updated.status_code == 200, updated.text
    elif change == "draft":
        with SessionLocal() as session:
            row = session.get(Chat, chat_id)
            assert row is not None
            row.draft_prompt = "A green ceramic bowl beside a window"
            session.commit()

    current = await _preview(client, [chat_id], include_configured=True)
    assert current["configured_count"] == 1
    result = await client.post(
        EXECUTE,
        json=_execute_body(
            preview,
            [chat_id],
            "op-existing-state",
            include_configured=True,
            acknowledged_configured=True,
        ),
    )
    if change == "unchanged":
        assert result.status_code == 200, result.text
        assert not _exists(chat_id)
    else:
        assert result.status_code == 409, {
            "status": result.status_code,
            "same_digest": current["digest"] == preview["digest"],
            "chat_survived": _exists(chat_id),
        }
        assert result.json()["code"] == "empty-chat-selection-drifted"
        assert _exists(chat_id)
