"""A refused setting is named with why, so a client never has to read a sentence.

A render request refused for a setting keeps the `detail` and `code` it always
answered with, and gains `errors`: one item per refusal, naming the setting, a
code from one closed vocabulary, the limit the value passed or the choices it
may take, and, for a value the request did not send, where it came from. An
item repeats nothing the request sent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_generation_experiment_preflight import PREFLIGHT, _arm, _profile, _request, _revision
from test_picture_remix import SETTINGS, _png, _uploaded
from test_saved_sizes_a_workflow_cannot_make import SAVED, _save

#: A picture workflow taking a value of every kind, most of them within limits.
PROBE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {"type": "integer", "default": 12, "minimum": 1, "maximum": 30},
        "width": {
            "type": "integer",
            "default": 1024,
            "minimum": 256,
            "maximum": 2048,
            "multipleOf": 64,
        },
        "height": {
            "type": "integer",
            "default": 1024,
            "minimum": 256,
            "maximum": 2048,
            "multipleOf": 64,
        },
        "sampler": {"type": "string", "default": "euler", "enum": ["euler", "heun"]},
        "tiling": {"type": "boolean", "default": False},
        "detail": {"type": "integer", "default": 1, "minimum": 0, "maximum": 5},
        "regions": {"type": "array", "default": []},
        "layout": {"type": "object", "default": {}},
    },
}
#: Ten lists deep: a list at a depth past the eight levels a setting may nest.
NESTED: list[Any] = [[[[[[[[[[0]]]]]]]]]]
#: The settings each reason is refused for, the detail that has always been
#: said about it, and the one item it now names.
REFUSED: dict[str, tuple[dict[str, Any], str, dict[str, Any]]] = {
    "unknown": (
        {"sparkle": 3},
        "unsupported settings: sparkle",
        {"code": "setting-unknown", "field": "sparkle"},
    ),
    "below its minimum": (
        {"steps": 0},
        "steps must be at least 1.0",
        {"code": "setting-below-minimum", "field": "steps", "limit": 1},
    ),
    "above its maximum": (
        {"steps": 31},
        "steps must be at most 30.0",
        {"code": "setting-above-maximum", "field": "steps", "limit": 30},
    ),
    "off its step": (
        {"width": 1000},
        "width must be a multiple of 64.0",
        {"code": "setting-not-a-multiple", "field": "width", "limit": 64},
    ),
    "not a choice": (
        {"sampler": "lms"},
        "sampler must be one of ['euler', 'heun']",
        {"code": "setting-not-a-choice", "field": "sampler", "allowed": ["euler", "heun"]},
    ),
    "no choice at all": (
        {"sampler": ["euler"]},
        "sampler must be a enum",
        {"code": "setting-not-a-choice", "field": "sampler", "allowed": ["euler", "heun"]},
    ),
    "not a whole number": (
        {"detail": 1.5},
        "detail must be a integer",
        {"code": "setting-not-an-integer", "field": "detail"},
    ),
    "not a number": (
        {"cfg": "high"},
        "cfg must be a number",
        {"code": "setting-not-a-number", "field": "cfg"},
    ),
    "not text": (
        {"negative_prompt": 5},
        "negative_prompt must be a string",
        {"code": "setting-not-a-string", "field": "negative_prompt"},
    ),
    "not true or false": (
        {"tiling": "yes"},
        "tiling must be a boolean",
        {"code": "setting-not-a-boolean", "field": "tiling"},
    ),
    "not a list": (
        {"regions": "all"},
        "regions must be a array",
        {"code": "setting-not-an-array", "field": "regions"},
    ),
    "not named values": (
        {"layout": [1]},
        "layout must be a object",
        {"code": "setting-not-an-object", "field": "layout"},
    ),
    "too large a number": (
        {"cfg": 10**400},
        "cfg numbers must be finite",
        {"code": "setting-not-finite", "field": "cfg"},
    ),
    "too long": (
        {"negative_prompt": "x" * 65_537},
        "negative_prompt strings must be at most 65536 characters",
        {"code": "setting-too-long", "field": "negative_prompt", "limit": 65_536},
    ),
    "too long a name inside it": (
        {"layout": {"k" * 201: 1}},
        "layout object keys must be at most 200 characters",
        {"code": "setting-too-long", "field": "layout", "limit": 200},
    ),
    "too many items": (
        {"regions": list(range(257))},
        "regions arrays must contain at most 256 items",
        {"code": "setting-too-many-items", "field": "regions", "limit": 256},
    ),
    "too deep": (
        {"regions": NESTED},
        "regions is nested too deeply; use at most 8 levels",
        {"code": "setting-too-deep", "field": "regions", "limit": 8},
    ),
    "too many values in all": (
        {"regions": [[0] * 16 for _ in range(256)]},
        "regions is too complex; use at most 4096 nested values",
        {"code": "setting-too-complex", "field": "regions", "limit": 4096},
    ),
    "too many settings": (
        {f"knob_{number}": number for number in range(257)},
        "settings must contain at most 256 fields",
        {"code": "settings-too-many", "field": "*", "limit": 256},
    ),
}
#: Where a saved size is named as coming from, by the layer it is saved on.
SOURCES = {
    "turn preset": "turn_preset",
    "chat": "chat",
    "chat preset": "chat_preset",
    "project": "project",
    "project preset": "project_preset",
    "default preset": "default_preset",
    "model profile": "profile",
    "model profile load settings": "profile",
}


async def _chat(client: AsyncClient) -> str:
    chat = await client.post("/api/chats", json={"title": "Refused settings"})
    assert chat.status_code == 201, chat.text
    chat_id: str = chat.json()["id"]
    return chat_id


async def _turn(
    app: FastAPI,
    client: AsyncClient,
    revision_id: str,
    profile_id: str,
    settings: dict[str, Any],
    *,
    chat_id: str | None = None,
    **turn: Any,
) -> Any:
    chat_id = chat_id if chat_id is not None else await _chat(client)
    async with app.state.services.scheduler.lease("primary"):
        return await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": "A plain gray square on a white table",
                "mode": "image",
                "workflow_revision_id": revision_id,
                "profile_id": profile_id,
                "settings": settings,
                **turn,
            },
        )


def _errors(response: Any, code: str = "turn-invalid", status: int = 422) -> list[dict[str, Any]]:
    """The items a refusal names, each checked to carry a sentence, without the sentences."""

    assert response.status_code == status, response.text
    body = response.json()
    assert body["code"] == code, body
    items: list[dict[str, Any]] = body["errors"]
    for item in items:
        message = item.pop("message")
        assert isinstance(message, str) and message.endswith("."), item
    return items


def _probe() -> tuple[str, str]:
    return _revision("Settings probe", input_schema_json=PROBE), _profile("Settings probe")


@pytest.mark.parametrize("reason", sorted(REFUSED))
async def test_a_refused_setting_is_named_with_why_beside_the_detail_it_always_had(
    app: FastAPI, client: AsyncClient, reason: str
) -> None:
    settings, detail, item = REFUSED[reason]
    revision_id, profile_id = _probe()

    response = await _turn(app, client, revision_id, profile_id, settings)

    assert _errors(response) == [item]
    assert response.json()["detail"] == detail


async def test_a_setting_the_workflow_does_not_offer_is_named(
    app: FastAPI, client: AsyncClient
) -> None:
    """A video workflow that describes its controls and not its length fixes the length itself."""

    revision_id = _revision(
        "Fixed length video",
        operation="text_to_video",
        input_schema_json={
            "type": "object",
            "properties": {"width": {"type": "integer", "default": 512}},
        },
    )
    profile_id = _profile("Fixed length video", role="video")

    response = await _turn(
        app,
        client,
        revision_id,
        profile_id,
        {"frames": 10},
        mode="video",
        text="A slow camera move across a gray table",
        confirm_media=True,
    )

    assert _errors(response) == [{"code": "setting-not-offered", "field": "frames"}]
    assert response.json()["detail"] == "This workflow's length is fixed by its own graph."


async def test_a_workflow_default_it_cannot_take_is_not_named_as_the_requests(
    app: FastAPI, client: AsyncClient
) -> None:
    """The request sent nothing; the workflow's own default is past its own maximum."""

    revision_id = _revision(
        "Inconsistent default",
        input_schema_json={
            "type": "object",
            "properties": {
                "steps": {"type": "integer", "default": 50, "minimum": 1, "maximum": 30}
            },
        },
    )

    response = await _turn(app, client, revision_id, _profile("Inconsistent default"), {})

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "turn-invalid"
    assert response.json()["detail"] == "steps must be at most 30.0"
    assert "errors" not in response.json()


async def test_one_answer_names_every_refused_setting_and_keeps_the_first_detail(
    app: FastAPI, client: AsyncClient
) -> None:
    revision_id, profile_id = _probe()

    response = await _turn(
        app,
        client,
        revision_id,
        profile_id,
        {"steps": 31, "sparkle": 3, "Not A Name": 1, "cfg": "high", "width": 2100},
    )

    assert _errors(response) == [
        {"code": "setting-unknown", "field": "sparkle"},
        {"code": "setting-unknown", "field": "*"},
        {"code": "setting-above-maximum", "field": "steps", "limit": 30},
        {"code": "setting-not-a-number", "field": "cfg"},
        {"code": "setting-above-maximum", "field": "width", "limit": 2048},
        {"code": "setting-not-a-multiple", "field": "width", "limit": 64},
    ]
    assert response.json()["detail"] == "unsupported settings: Not A Name, sparkle"


async def test_a_key_that_is_not_a_setting_name_is_never_said_back(
    app: FastAPI, client: AsyncClient
) -> None:
    revision_id, profile_id = _probe()
    spaced = "Neutral Marker Key"
    long = "k" * 65

    response = await _turn(app, client, revision_id, profile_id, {spaced: 1, long: 2})

    errors = _errors(response)
    assert errors == [
        {"code": "setting-unknown", "field": "*"},
        {"code": "setting-unknown", "field": "*"},
    ]
    assert spaced not in json.dumps(errors) and long not in json.dumps(errors)


async def test_an_item_never_repeats_the_value_it_refuses(
    app: FastAPI, client: AsyncClient
) -> None:
    revision_id, profile_id = _probe()
    value = "neutral-marker-value"

    response = await _turn(
        app, client, revision_id, profile_id, {"sampler": value, "detail": value}
    )

    assert [item["code"] for item in _errors(response)] == [
        "setting-not-a-choice",
        "setting-not-an-integer",
    ]
    assert value not in response.text


async def test_each_reason_is_said_in_one_sentence_of_its_own(
    app: FastAPI, client: AsyncClient
) -> None:
    revision_id, profile_id = _probe()

    response = await _turn(
        app, client, revision_id, profile_id, {"steps": 31, "detail": 9, "sparkle": 1}
    )

    assert response.status_code == 422, response.text
    said: dict[str, set[str]] = {}
    for item in response.json()["errors"]:
        said.setdefault(item["code"], set()).add(item["message"])
    assert sorted(said) == ["setting-above-maximum", "setting-unknown"]
    assert all(len(sentences) == 1 for sentences in said.values())
    assert said["setting-above-maximum"] != said["setting-unknown"]


async def test_a_long_menu_is_named_to_its_first_choices_and_file_names_never(
    app: FastAPI, client: AsyncClient
) -> None:
    menu = [f"style_{number}" for number in range(60)]
    revision_id = _revision(
        "Long menu",
        input_schema_json={
            "type": "object",
            "properties": {
                "style": {"type": "string", "default": "style_0", "enum": menu},
                "checkpoint": {
                    "type": "string",
                    "default": "neutral.safetensors",
                    "enum": ["neutral.safetensors", "other.safetensors"],
                },
            },
        },
    )

    response = await _turn(
        app,
        client,
        revision_id,
        _profile("Long menu"),
        {"style": "style_99", "checkpoint": "missing.safetensors"},
    )

    errors = _errors(response)
    assert errors == [
        {"code": "setting-not-a-choice", "field": "style", "allowed": menu[:50]},
        {"code": "setting-not-a-choice", "field": "checkpoint"},
    ]
    assert "safetensors" not in json.dumps(errors)


@pytest.mark.parametrize("layer", sorted(SAVED))
async def test_a_saved_size_is_named_with_its_limit_and_the_layer_it_came_from(
    app: FastAPI, client: AsyncClient, layer: str
) -> None:
    revision_id, profile_id = _probe()
    chat_id = await _chat(client)
    turn = await _save(client, chat_id, profile_id, {layer: {"width": 2176, "height": 768}})

    response = await _turn(app, client, revision_id, profile_id, {}, chat_id=chat_id, **turn)

    assert _errors(response, "turn-size-unsupported") == [
        {"code": "setting-above-maximum", "field": "width", "limit": 2048, "source": SOURCES[layer]}
    ]
    assert response.json()["detail"].startswith(
        f"The width of 2176 px saved {SAVED[layer]} is not one this workflow makes"
    )


def test_every_layer_a_saved_size_is_refused_from_names_its_own_source() -> None:
    from local_lm.orchestrator import _SIZE_SOURCES
    from local_lm.setting_refusals import saved_setting_source

    named = {origin: saved_setting_source(where) for origin, where in _SIZE_SOURCES if where}

    assert named == {
        "turn_preset": "turn_preset",
        "chat": "chat",
        "chat_preset": "chat_preset",
        "project": "project",
        "project_preset": "project_preset",
        "default_preset": "default_preset",
        "profile_request": "profile",
        "profile_load": "profile",
    }


@pytest.mark.parametrize("sent", [False, True], ids=["carried over", "sent again"])
async def test_a_setting_carried_over_from_the_turn_an_edit_starts_from_says_so(
    app: FastAPI, client: AsyncClient, sent: bool
) -> None:
    """The edit names another workflow, which has no setting the first turn used."""

    detail = PROBE["properties"]["detail"]
    first = _revision(
        "First workflow",
        input_schema_json={"type": "object", "properties": {"detail": detail}},
    )
    other = _revision("Other workflow")
    profile_id = _profile("Edited model")
    chat_id = await _chat(client)
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": "A plain gray square on a white table",
                "mode": "image",
                "workflow_revision_id": first,
                "profile_id": profile_id,
                "settings": {"detail": 3},
            },
        )
        assert source.status_code == 202, source.text
        edited = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/edits",
            json={
                "text": "A plain gray square on a white table",
                "mode": "image",
                "idempotency_key": "carried-over-setting",
                "workflow_revision_id": other,
                "profile_id": profile_id,
                **({"settings": {"detail": 3}} if sent else {}),
            },
        )

    item: dict[str, Any] = {"code": "setting-unknown", "field": "detail"}
    assert _errors(edited) == [item if sent else {**item, "source": "inherited"}]
    assert edited.json()["detail"] == "unsupported settings: detail"


async def test_a_remix_names_each_choice_it_cannot_apply(client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    revision_id = _revision("Remix probe")
    profile_id = _profile("Remix probe")

    async def remix(apply: list[str]) -> Any:
        return await client.post(
            f"/api/artifacts/{artifact_id}/remix-preview",
            json={"workflow_revision_id": revision_id, "profile_id": profile_id, "apply": apply},
        )

    halves = await remix(["sampler", "denoise", "width"])
    shape = await remix(["shape"])

    # The words never take an edit's strength, this workflow knows no sampler
    # names, and a width is never applied without its height.
    assert _errors(halves, "remix-choice-invalid") == [
        {"code": "setting-not-applicable", "field": "denoise"},
        {"code": "setting-not-applicable", "field": "sampler"},
        {"code": "setting-needs-its-pair", "field": "width"},
    ]
    # The picture's own size is one this workflow makes, so no shape is offered.
    assert _errors(shape, "remix-choice-invalid") == [
        {"code": "setting-not-offered", "field": "shape"}
    ]


async def test_a_comparison_names_every_setting_of_the_choice_it_refuses(
    client: AsyncClient,
) -> None:
    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    invalid_first = _arm("Second", _profile("Second model"), _revision("Second workflow"))
    invalid_first["settings"] = {"steps": 999, "sparkle": 1}
    unknown_first = dict(invalid_first, settings={"sparkle": 1, "steps": 999})
    named = [
        {"code": "setting-unknown", "field": "sparkle"},
        {"code": "setting-above-maximum", "field": "steps", "limit": 200},
    ]

    for arm, code, setting in (
        (invalid_first, "arm-setting-invalid", "steps"),
        (unknown_first, "arm-setting-unsupported", "sparkle"),
    ):
        response = await client.post(PREFLIGHT, json=_request(first, arm))
        assert response.status_code == 200, response.text
        [refusal] = response.json()["refusals"]
        assert (refusal["code"], refusal["arm_ordinal"], refusal["setting"]) == (code, 2, setting)
        for item in refusal["errors"]:
            assert isinstance(item.pop("message"), str)
        assert refusal["errors"] == named


async def test_a_comparison_names_a_lora_override_its_workflow_cannot_take(
    client: AsyncClient,
) -> None:
    """The choice is refused for the override first, so its errors must name it too."""

    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    second = _arm("Second", _profile("Second model"), _revision("Second workflow"))
    second["settings"] = {"workflow_lora_overrides": {}, "steps": 999}

    response = await client.post(PREFLIGHT, json=_request(first, second))

    assert response.status_code == 200, response.text
    [refusal] = response.json()["refusals"]
    assert (refusal["code"], refusal["setting"]) == (
        "arm-setting-invalid",
        "workflow_lora_overrides",
    )
    for item in refusal["errors"]:
        assert isinstance(item.pop("message"), str), item
    assert refusal["errors"] == [
        {"code": "setting-unknown", "field": "workflow_lora_overrides"},
        {"code": "setting-above-maximum", "field": "steps", "limit": 200},
    ]


async def test_a_comparison_names_both_sides_of_a_size_it_refuses(client: AsyncClient) -> None:
    """Both sides are past the most the workflows make; the refusal names each."""

    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    second = _arm("Second", _profile("Second model"), _revision("Second workflow"))

    response = await client.post(
        PREFLIGHT,
        json=_request(first, second, geometry={"mode": "size", "width": 8192, "height": 8192}),
    )

    assert response.status_code == 200, response.text
    refusals = response.json()["refusals"]
    assert [(item["code"], item["arm_ordinal"], item["setting"]) for item in refusals] == [
        ("arm-setting-invalid", 1, "width"),
        ("arm-setting-invalid", 2, "width"),
    ]
    for refusal in refusals:
        for item in refusal["errors"]:
            assert isinstance(item.pop("message"), str), item
        assert refusal["errors"] == [
            {"code": "setting-above-maximum", "field": "width", "limit": 4096},
            {"code": "setting-above-maximum", "field": "height", "limit": 4096},
        ]


async def test_a_comparison_refusal_that_names_no_setting_carries_no_errors(
    client: AsyncClient,
) -> None:
    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    second = _arm("Second", "profile_missing", _revision("Second workflow"))

    response = await client.post(PREFLIGHT, json=_request(first, second))

    assert response.status_code == 200, response.text
    [refusal] = response.json()["refusals"]
    assert refusal["code"] == "arm-profile-unavailable"
    assert "errors" not in refusal


def test_every_reason_in_the_vocabulary_is_one_a_request_can_meet() -> None:
    """Each code is reached above through a real route, and none is left unreached."""

    from local_lm.setting_refusals import SETTING_REFUSAL_MESSAGES, SettingRefusalCode

    reached = {item["code"] for _, _, item in REFUSED.values()} | {
        "setting-not-offered",
        "setting-not-applicable",
        "setting-needs-its-pair",
    }
    assert reached == {code.value for code in SettingRefusalCode}
    assert set(SETTING_REFUSAL_MESSAGES) == set(SettingRefusalCode)
    assert len(set(SETTING_REFUSAL_MESSAGES.values())) == len(SettingRefusalCode)
