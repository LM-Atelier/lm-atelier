"""A source canvas preview refused for a setting names it beside its own code.

Both previews, of a new turn and of an edit, keep the one sentence and code
they always answered a canvas they cannot use with, and add the same `errors`
items a turn names.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_source_fit_acceptance import prepared_turn
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

#: The fixture workflow takes from one to a hundred steps.
STEPS_PAST = [{"code": "setting-above-maximum", "field": "steps", "limit": 100}]
CANVAS = {"mode": "extend", "width": 6, "height": 7}


def _errors(response: Any) -> list[dict[str, Any]]:
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "source-fit-preview-unavailable", body
    assert body["detail"] == "The selected source and workflow cannot use this canvas."
    items: list[dict[str, Any]] = body["errors"]
    for item in items:
        assert isinstance(item.pop("message"), str), item
    return items


async def test_a_turn_canvas_preview_names_the_setting(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, payload = await prepared_turn(app, client, monkeypatch)

    response = await client.post(
        f"/api/chats/{chat_id}/source-fit/preview",
        json={**payload, "source_fit": CANVAS, "settings": {"steps": 101}},
    )

    assert _errors(response) == STEPS_PAST


async def test_an_edit_canvas_preview_names_the_setting(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, payload = await prepared_turn(app, client, monkeypatch)
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={**payload, "source_fit": {"mode": "extend", "width": 4, "height": 5}},
        )
        assert accepted.status_code == 202, accepted.text

        response = await client.post(
            f"/api/messages/{accepted.json()['user_message']['id']}/edits/source-fit/preview",
            json={
                "text": "Extend the neutral grid with a blue border",
                "mode": "image",
                "idempotency_key": "refused-setting-canvas-edit",
                "source_fit": CANVAS,
                "settings": {"steps": 101},
            },
        )

    assert _errors(response) == STEPS_PAST
