from __future__ import annotations

from typing import cast

import pytest
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status

pytestmark = pytest.mark.asyncio


async def _wait_for_run(client: AsyncClient, run_id: str) -> dict[str, object]:
    async def read() -> dict[str, object]:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        return cast(dict[str, object], response.json())

    # This one accepts any ending: its callers assert which one themselves.
    return cast(
        dict[str, object],
        await wait_for_terminal_status(read, what=f"run {run_id}", expected=None),
    )


async def test_media_regeneration_replaces_the_source_seed(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def predictable_random(upper: int) -> int:
        calls.append(upper)
        return 42

    monkeypatch.setattr("local_lm.orchestrator.secrets.randbelow", predictable_random)
    chat = (await client.post("/api/chats", json={"title": "Fresh variations"})).json()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create an image of a ceramic cup", "mode": "image"},
    )
    assert original.status_code == 202
    original_run = original.json()["run"]
    assert original_run["settings_json"]["seed"] == 42
    await _wait_for_run(client, original_run["id"])

    regenerated = await client.post(
        f"/api/messages/{original.json()['assistant_message']['id']}/regenerate",
        json={"settings": {"seed": 42}},
    )
    assert regenerated.status_code == 202
    regenerated_run = regenerated.json()["run"]
    assert regenerated_run["settings_json"]["seed"] == 43
    assert regenerated_run["provenance_json"]["resolved_settings"]["seed"] == 43
    assert calls == [2_147_483_648, 2_147_483_647]


async def test_text_regeneration_does_not_gain_a_media_seed(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Text revision"})).json()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Write one sentence", "mode": "text"},
    )
    assert original.status_code == 202
    await _wait_for_run(client, original.json()["run"]["id"])

    regenerated = await client.post(
        f"/api/messages/{original.json()['assistant_message']['id']}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202
    assert regenerated.json()["run"]["settings_json"].get("seed") == original.json()["run"][
        "settings_json"
    ].get("seed")
