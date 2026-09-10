"""Asking for the same picture again asks for it at the same size.

The size of an image is now something someone chooses in the composer rather
than a number they happen to have typed, and a chosen shape that quietly becomes
a different shape on the second attempt is the same class of failure as a
workflow that ignores the description: the answer looks wrong for a reason the
person cannot see.

Regeneration is the path where that could happen. It rebuilds the turn rather
than re-running the old one, and it could perfectly reasonably resolve the
settings afresh from whatever the defaults are now - the workflow may have been
edited, the preset may have moved, the chat may have been given new overrides
since. It does not: it starts from the settings the accepted run froze. Nothing
proved that, so nothing would have caught it changing.

Retry needs no companion test here, and it is worth saying why rather than
leaving the omission to look like an oversight: `retry_job` re-queues the SAME
job row, so its frozen `settings_json` is carried by construction and there is
no second resolution to get wrong.
"""

from __future__ import annotations

from typing import cast

import pytest
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status

pytestmark = pytest.mark.asyncio

# Deliberately not the workflow's defaults, and both off the usual round
# numbers, so a re-resolution from current defaults could not coincide with a
# replay and pass this by accident.
WIDTH = 1216
HEIGHT = 704


async def _wait(client: AsyncClient, run_id: str) -> dict[str, object]:
    async def read() -> dict[str, object]:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        return cast(dict[str, object], response.json())

    return cast(
        dict[str, object],
        await wait_for_terminal_status(read, what=f"run {run_id}", expected=None),
    )


async def test_regeneration_asks_for_the_same_size(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Same shape again"})).json()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of a ceramic cup",
            "mode": "image",
            "settings": {"width": WIDTH, "height": HEIGHT},
        },
    )
    assert original.status_code == 202, original.text
    original_run = original.json()["run"]
    assert original_run["settings_json"]["width"] == WIDTH
    assert original_run["settings_json"]["height"] == HEIGHT
    await _wait(client, original_run["id"])

    regenerated = await client.post(
        f"/api/messages/{original.json()['assistant_message']['id']}/regenerate",
        json={"settings": {}},
    )

    assert regenerated.status_code == 202, regenerated.text
    run = regenerated.json()["run"]
    assert run["settings_json"]["width"] == WIDTH
    assert run["settings_json"]["height"] == HEIGHT
    # What the engine is actually handed, not only what the run recorded.
    resolved = run["provenance_json"]["resolved_settings"]
    assert resolved["width"] == WIDTH
    assert resolved["height"] == HEIGHT
    # And the control that stops this passing for the wrong reason: the seed
    # MUST have moved, or "the settings are identical" would be satisfied by a
    # regeneration that did nothing at all.
    assert run["settings_json"]["seed"] != original_run["settings_json"]["seed"]


async def test_an_explicit_size_on_the_regenerate_still_wins(client: AsyncClient) -> None:
    """Replaying the frozen settings must not make them unchangeable.

    The point is that the previous size is the DEFAULT for the next attempt, not
    that it is fixed - somebody regenerating at a different shape is asking for
    exactly that.
    """
    chat = (await client.post("/api/chats", json={"title": "A different shape"})).json()
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of a ceramic cup",
            "mode": "image",
            "settings": {"width": WIDTH, "height": HEIGHT},
        },
    )
    assert original.status_code == 202, original.text
    await _wait(client, original.json()["run"]["id"])

    regenerated = await client.post(
        f"/api/messages/{original.json()['assistant_message']['id']}/regenerate",
        json={"settings": {"width": 704, "height": 1216}},
    )

    assert regenerated.status_code == 202, regenerated.text
    run = regenerated.json()["run"]
    assert (run["settings_json"]["width"], run["settings_json"]["height"]) == (704, 1216)
