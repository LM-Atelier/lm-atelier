"""An extension names a strength someone chose, which it samples at full strength instead.

An extension paints the canvas past its picture's edge, which only a sample at
full strength fills, so it runs the workflow's mapped strength at full
whatever the turn or a saved layer gave it. The accepted answer names that
change, with the strength given and the one used, and names nothing when the
strength was already full. A saved size the workflow does not take is named
for what the workflow does with it, since the source canvas, not the picture,
decides the size.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_accepted_turns_name_what_they_changed import _changes
from test_source_fit_acceptance import prepared_turn
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

CANVAS = {"mode": "extend", "width": 4, "height": 5}
FULL_FOR_EXTENSION = {
    "code": "strength-full-for-extension",
    "field": "denoise",
    "before": 0.4,
    "after": 1,
}


@pytest.mark.parametrize("saved", [False, True], ids=["sent", "saved for the chat"])
async def test_an_extension_names_the_strength_it_sets_aside(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, saved: bool
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch, runtime_sampler=True)
    if saved:
        stored = await client.patch(
            f"/api/chats/{chat_id}", json={"generation_settings_json": {"image": {"denoise": 0.4}}}
        )
        assert stored.status_code == 200, stored.text
    given = {"steps": 2} if saved else {"steps": 2, "denoise": 0.4}

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={**request, "settings": given, "source_fit": CANVAS},
        )

    assert _changes(response) == [
        {**FULL_FOR_EXTENSION, "source": "chat"} if saved else FULL_FOR_EXTENSION
    ]
    assert response.json()["run"]["settings_json"]["denoise"] == 1


@pytest.mark.parametrize("given", [{"steps": 2}, {"steps": 2, "denoise": 1}], ids=["none", "full"])
async def test_an_extension_given_no_strength_or_full_strength_names_nothing(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    given: dict[str, int],
) -> None:
    """The strength it runs with is the one it was given, or the workflow's own."""

    chat_id, request = await prepared_turn(app, client, monkeypatch, runtime_sampler=True)

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={**request, "settings": given, "source_fit": CANVAS},
        )

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()
    assert response.json()["run"]["settings_json"]["denoise"] == 1


async def test_a_saved_size_an_extension_does_not_take_is_named_for_its_workflow(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    stored = await client.patch(
        f"/api/chats/{chat_id}",
        json={"generation_settings_json": {"image": {"width": 2176, "height": 768}}},
    )
    assert stored.status_code == 200, stored.text

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns", json={**request, "source_fit": CANVAS}
        )

    side = {"code": "set-aside-not-offered", "source": "chat"}
    assert _changes(response) == [
        {**side, "field": "width", "before": 2176},
        {**side, "field": "height", "before": 768},
    ]
