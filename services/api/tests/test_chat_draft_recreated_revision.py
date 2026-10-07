"""A stale window cannot change a later draft after an intervening discard."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from test_chat_composer_drafts import _aged_preview, _chat, _clear_now, _exists, _put

from local_lm.config import Settings


@pytest.mark.parametrize("action", ["write", "discard"])
async def test_a_previous_draft_revision_cannot_change_a_recreated_draft(
    client: AsyncClient,
    settings: Settings,
    action: str,
) -> None:
    chat_id = await _chat(client)
    original = await _put(client, chat_id, 0, text="Original neutral draft")
    assert original.status_code == 200
    old_revision = original.json()["revision"]
    path = f"/api/chats/{chat_id}/composer-draft"
    discarded = await client.delete(path, params={"expected_revision": old_revision})
    assert discarded.status_code == 204
    empty = (await client.get(path)).json()
    picture = _aged_preview(settings, "new draft")
    current = await _put(
        client,
        chat_id,
        empty["revision"],
        text="New neutral draft",
        attachments=[{"artifact_id": picture, "kind": "image", "origin": "generated"}],
    )
    assert current.status_code == 200

    if action == "write":
        stale = await _put(client, chat_id, old_revision, text="Delayed old neutral draft")
    else:
        stale = await client.delete(path, params={"expected_revision": old_revision})
    retained = (await client.get(path)).json()
    await _clear_now(client)
    survived = _exists(picture)
    assert stale.status_code == 409, (
        f"stale {action} returned {stale.status_code}; "
        f"new draft retained={retained['text'] == 'New neutral draft'}; "
        f"new attachment survived cleanup={survived}"
    )
    assert retained["text"] == "New neutral draft" and survived
