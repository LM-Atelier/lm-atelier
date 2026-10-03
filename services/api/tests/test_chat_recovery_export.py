"""Project archives omit deleted chats until their original history is restored."""

from __future__ import annotations

import json
from typing import Any

from httpx2 import AsyncClient
from test_chat_deletion import _text_exchange
from test_chat_recovery import _command, _history, _impact
from test_project_portability import _manifest


async def _export(client: AsyncClient, project_id: str) -> dict[str, Any]:
    exported = await client.post(f"/api/projects/{project_id}/export")
    assert exported.status_code == 201, exported.text
    content = await client.get(exported.json()["url"])
    assert content.status_code == 200
    return _manifest(content.content)


async def test_project_export_omits_deleted_history_and_keeps_archived_history(
    client: AsyncClient,
) -> None:
    created = await client.post("/api/projects", json={"name": "Garden recovery archive"})
    assert created.status_code == 201
    project_id = str(created.json()["id"])
    ids: list[str] = []
    for title in ("Current garden notes", "Archived garden notes", "Removed garden notes"):
        response = await client.post("/api/chats", json={"title": title, "project_id": project_id})
        assert response.status_code == 201
        ids.append(str(response.json()["id"]))
        await _text_exchange(client, ids[-1], title)
    live_id, archived_id, deleted_id = ids
    assert (
        await client.patch(f"/api/chats/{archived_id}", json={"archived": True})
    ).status_code == 200
    original = _history(deleted_id)
    before = await _export(client, project_id)
    assert {chat["id"] for chat in before["chats"]} == set(ids)
    impact = await _impact(client, f"/api/chats/{deleted_id}/deletion-impact")
    trashed = await client.post(
        f"/api/chats/{deleted_id}/trash", json=_command(impact, "trash-exported-garden")
    )
    assert trashed.status_code == 200, trashed.text

    hidden = await _export(client, project_id)

    assert {chat["id"] for chat in hidden["chats"]} == {live_id, archived_id}
    assert all(run["chat_id"] != deleted_id for run in hidden["runs"])
    assert all(plan["chat_id"] != deleted_id for plan in hidden["work_plans"])
    assert "Removed garden notes" not in json.dumps(hidden)
    assert _history(deleted_id) == original
    deletion_id = str(trashed.json()["deletion_id"])
    restore_impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    restored = await client.post(
        f"/api/recovery-items/{deletion_id}/restore",
        json=_command(restore_impact, "restore-exported-garden"),
    )
    assert restored.status_code == 200, restored.text

    after = await _export(client, project_id)

    assert {chat["id"] for chat in after["chats"]} == set(ids)
    original_chat = next(chat for chat in before["chats"] if chat["id"] == deleted_id)
    restored_chat = next(chat for chat in after["chats"] if chat["id"] == deleted_id)
    assert [message["id"] for message in restored_chat["messages"]] == [
        message["id"] for message in original_chat["messages"]
    ]
    assert {run["id"] for run in after["runs"]} == {run["id"] for run in before["runs"]}
    assert {plan["id"] for plan in after["work_plans"]} == {
        plan["id"] for plan in before["work_plans"]
    }
    assert _history(deleted_id) == original
