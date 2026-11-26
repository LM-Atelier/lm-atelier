"""A remix names each side of a saved size its workflow cannot make, as sending a turn does.

A remix is resolved in a new chat with no preset, so the only saved size it can
meet is one saved on the model. Its preview still answers with the code and
sentence it always had, and a remix of that preview is still refused with the
status, code and detail it always had. Each refusal also names, in `errors`,
every side the workflow cannot make, with its limit and where it is saved: the
items a turn sent with the same model and workflow is refused with.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from httpx2 import AsyncClient
from test_default_output_shape import _chat
from test_picture_remix import (
    SETTINGS,
    _png,
    _preview,
    _profile,
    _queue,
    _revision,
    _runs,
    _uploaded,
)
from test_saved_sizes_a_workflow_cannot_make import SAVED, TOO_WIDE, _refused_for, _save, _send
from test_saved_sizes_name_both_sides import CAP, TOO_LARGE, _refused, _side

#: The one sentence a preview says of a saved size its workflow cannot make.
SIZE_SENTENCE = "A size saved for this conversation or its model is not one this workflow makes."
#: The detail a remix of a preview that cannot run is refused with.
UNAVAILABLE = "This picture cannot be remixed with these choices."
#: A well-formed digest for a remix whose preview gave none: its refusal answers first.
ANY_DIGEST = "sha256:" + "0" * 64
BOTH_SIDES = [_side("width", "profile"), _side("height", "profile")]


async def _choice(client: AsyncClient) -> tuple[str, str, str]:
    """A picture, a workflow that makes 256 to 2048 px a side in steps of 64, and a model."""

    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    revision_id = _revision(
        "Capped remix workflow",
        input_schema={"type": "object", "properties": {"width": dict(CAP), "height": dict(CAP)}},
    )
    return artifact_id, revision_id, _profile("Neutral remix model")


async def _save_on_the_model(client: AsyncClient, profile_id: str, size: dict[str, Any]) -> None:
    saved = await client.patch(f"/api/profiles/{profile_id}", json={"request_settings": size})
    assert saved.status_code == 200, saved.text


def _size_refusal(response: Any) -> dict[str, Any]:
    """The one refusal of a preview that cannot run, with its code and sentence as they were."""

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ready"] is False, body
    assert body["review_digest"] is None, body
    refusals: list[dict[str, Any]] = body["refusals"]
    [refusal] = refusals
    assert (refusal["code"], refusal["message"]) == ("remix-size-unsupported", SIZE_SENTENCE)
    return refusal


def _named(refusal: dict[str, Any]) -> list[dict[str, Any]]:
    """The sides a refusal names, each without its fixed sentence."""

    assert set(refusal) == {"code", "message", "errors"}, refusal
    items: list[dict[str, Any]] = refusal["errors"]
    for item in items:
        message = item.pop("message")
        assert isinstance(message, str) and message.endswith("."), item
    return items


async def test_a_preview_names_the_one_saved_side_its_workflow_cannot_make(
    client: AsyncClient,
) -> None:
    artifact_id, revision_id, profile_id = await _choice(client)
    await _save_on_the_model(client, profile_id, TOO_WIDE)

    response = await _preview(client, artifact_id, revision_id, profile_id)

    assert _named(_size_refusal(response)) == [_side("width", "profile")]


async def test_a_preview_names_both_saved_sides_its_workflow_cannot_make(
    client: AsyncClient,
) -> None:
    artifact_id, revision_id, profile_id = await _choice(client)
    await _save_on_the_model(client, profile_id, TOO_LARGE)

    response = await _preview(client, artifact_id, revision_id, profile_id)

    assert _named(_size_refusal(response)) == BOTH_SIDES


async def test_each_side_names_the_model_whichever_of_its_settings_holds_it(
    app: FastAPI, client: AsyncClient
) -> None:
    """A model saves a size in its request or its load settings, and either is the model's."""

    artifact_id, revision_id, profile_id = await _choice(client)
    chat_id = await _chat(client)
    await _save(
        client,
        chat_id,
        profile_id,
        {"model profile": {"width": 2176}, "model profile load settings": {"height": 2240}},
    )

    previewed = await _preview(client, artifact_id, revision_id, profile_id)
    sent = await _send(app, client, chat_id, revision_id, profile_id, preset_id=None)

    _refused_for(sent, "width", 2176, SAVED["model profile"])
    assert _refused(sent)[1] == BOTH_SIDES
    refusal = _size_refusal(previewed)
    assert refusal["errors"] == sent.json()["errors"], "the items the turn is refused with"
    assert _named(refusal) == BOTH_SIDES


async def test_a_remix_of_that_preview_names_the_same_sides_beside_its_own_code(
    client: AsyncClient,
) -> None:
    artifact_id, revision_id, profile_id = await _choice(client)
    await _save_on_the_model(client, profile_id, TOO_LARGE)
    previewed = await _preview(client, artifact_id, revision_id, profile_id)
    chat_id = await _chat(client)

    response = await _queue(
        client, chat_id, artifact_id, {**previewed.json(), "review_digest": ANY_DIGEST}, []
    )

    assert response.status_code == 409, response.text
    body = response.json()
    assert (body["code"], body["detail"]) == ("remix-unavailable", UNAVAILABLE)
    assert set(body) == {"code", "detail", "refusals"}, body
    assert body["refusals"] == previewed.json()["refusals"]
    [refusal] = body["refusals"]
    assert (refusal["code"], refusal["message"]) == ("remix-size-unsupported", SIZE_SENTENCE)
    assert _named(refusal) == BOTH_SIDES
    assert _runs(chat_id) == []


async def test_a_refusal_about_no_setting_names_none(client: AsyncClient) -> None:
    artifact_id, revision_id, _profile_id = await _choice(client)
    unusable = {
        "code": "remix-model-unusable",
        "message": "This model cannot run with this workflow.",
    }

    previewed = await _preview(client, artifact_id, revision_id, "profile_missing")
    remixed = await _queue(
        client,
        await _chat(client),
        artifact_id,
        {**previewed.json(), "review_digest": ANY_DIGEST},
        [],
    )

    assert previewed.status_code == 200, previewed.text
    assert previewed.json()["refusals"] == [unusable]
    assert remixed.status_code == 409, remixed.text
    assert remixed.json() == {
        "detail": UNAVAILABLE,
        "code": "remix-unavailable",
        "refusals": [unusable],
    }
