"""A number too large to work with is refused like any other, never answered with a server error.

A workflow can take a number in fractional steps, as a converted graph does for
its guidance and strength. A value near the largest a number can be cannot be
divided by such a step. It is refused for a bound it passes, beside whatever
else the request is refused for, or else as a number too large to use; one
saved on a layer is set aside, like any other saved value its workflow does
not take. A refusal that names choices names none it cannot write as a number.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_generation_experiment_preflight import PREFLIGHT, _arm, _profile, _request, _revision

from local_lm.db import SessionLocal
from local_lm.models import ModelProfile, Run
from local_lm.schemas import SettingField

#: Guidance in half steps within a range, and a scale in half steps with no bounds.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cfg": {"type": "number", "default": 7, "minimum": 0, "maximum": 30, "multipleOf": 0.5},
        "guidance_scale": {"type": "number", "default": 1.0, "multipleOf": 0.5},
    },
}
#: Requests the turn route has always refused, the detail it always gave, and
#: what each now names.
REFUSED: dict[str, tuple[dict[str, Any], str, list[dict[str, Any]]]] = {
    "far past the maximum": (
        {"cfg": 1e308},
        "cfg must be at most 30.0",
        [{"code": "setting-above-maximum", "field": "cfg", "limit": 30}],
    ),
    "far below the minimum": (
        {"cfg": -1e308},
        "cfg must be at least 0.0",
        [{"code": "setting-below-minimum", "field": "cfg", "limit": 0}],
    ),
    "beside an unknown setting": (
        {"sparkle": 1, "guidance_scale": 1e308},
        "unsupported settings: sparkle",
        [
            {"code": "setting-unknown", "field": "sparkle"},
            {"code": "setting-not-finite", "field": "guidance_scale"},
        ],
    ),
    "after a refused setting": (
        {"steps": 999, "guidance_scale": 1e308},
        "steps must be at most 200.0",
        [
            {"code": "setting-above-maximum", "field": "steps", "limit": 200},
            {"code": "setting-not-finite", "field": "guidance_scale"},
        ],
    ),
}


async def _turn(app: FastAPI, client: AsyncClient, settings: dict[str, Any], **turn: Any) -> Any:
    chat = await client.post("/api/chats", json={"title": "Very large values"})
    assert chat.status_code == 201, chat.text
    async with app.state.services.scheduler.lease("primary"):
        return await client.post(
            f"/api/chats/{chat.json()['id']}/turns",
            json={
                "text": "A plain gray square on a white table",
                "mode": "image",
                "settings": settings,
                **turn,
            },
        )


def _errors(response: Any) -> list[dict[str, Any]]:
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "turn-invalid", response.text
    items: list[dict[str, Any]] = response.json()["errors"]
    for item in items:
        assert isinstance(item.pop("message"), str), item
    return items


def _workflow() -> dict[str, str]:
    return {
        "workflow_revision_id": _revision("Half steps", input_schema_json=SCHEMA),
        "profile_id": _profile("Half steps"),
    }


@pytest.mark.parametrize("case", sorted(REFUSED))
async def test_a_refusal_keeps_its_answer_when_a_value_is_too_large_to_divide(
    app: FastAPI, client: AsyncClient, case: str
) -> None:
    settings, detail, named = REFUSED[case]

    response = await _turn(app, client, settings, **_workflow())

    assert _errors(response) == named
    assert response.json()["detail"] == detail


async def test_a_value_with_no_bound_too_large_for_its_step_is_refused(
    app: FastAPI, client: AsyncClient
) -> None:
    response = await _turn(app, client, {"guidance_scale": 1e308}, **_workflow())

    assert _errors(response) == [{"code": "setting-not-finite", "field": "guidance_scale"}]
    assert response.json()["detail"] == (
        "guidance_scale is too large to use with a multiple of 0.5"
    )


@pytest.mark.parametrize(
    "saved", [{"cfg": 1e308}, {"guidance_scale": -1e308}], ids=["past its maximum", "no bound"]
)
async def test_a_saved_value_too_large_for_its_step_is_set_aside(
    app: FastAPI, client: AsyncClient, saved: dict[str, Any]
) -> None:
    """A saved layer is read value by value, and one it cannot use is left out."""

    workflow = _workflow()
    with SessionLocal() as session:
        profile = session.get(ModelProfile, workflow["profile_id"])
        assert profile is not None
        profile.request_settings_json = saved
        session.commit()

    response = await _turn(app, client, {}, **workflow)

    assert response.status_code == 202, response.text
    with SessionLocal() as session:
        run = session.get(Run, response.json()["run"]["id"])
        assert run is not None
        ran = {key: run.settings_json[key] for key in ("cfg", "guidance_scale")}
    assert ran == {"cfg": 7, "guidance_scale": 1.0}


async def test_a_comparison_names_every_refused_setting_beside_a_value_too_large(
    client: AsyncClient,
) -> None:
    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    second = _arm(
        "Second",
        _profile("Half steps model"),
        _revision("Half steps workflow", input_schema_json=SCHEMA),
        steps=999,
        guidance_scale=-1e308,
    )

    response = await client.post(PREFLIGHT, json=_request(first, second))

    assert response.status_code == 200, response.text
    [refusal] = response.json()["refusals"]
    assert (refusal["code"], refusal["setting"]) == ("arm-setting-invalid", "steps")
    for item in refusal["errors"]:
        assert isinstance(item.pop("message"), str), item
    assert refusal["errors"] == [
        {"code": "setting-above-maximum", "field": "steps", "limit": 200},
        {"code": "setting-not-finite", "field": "guidance_scale"},
    ]


async def test_a_choice_too_large_for_a_number_is_never_named(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine can offer a whole number of any length as a choice; JSON carries it."""

    media = app.state.services.engines.media
    offered = await media.capabilities()
    quality = SettingField(
        key="quality",
        label="Quality",
        type="enum",
        default=1,
        choices=[1, 10**400],
        scope="workflow",
    )
    image = [*offered.settings_by_role["image"], quality]

    async def capabilities() -> Any:
        return offered.model_copy(
            update={
                "settings": [*offered.settings, quality],
                "settings_by_role": {**offered.settings_by_role, "image": image},
            }
        )

    monkeypatch.setattr(media, "capabilities", capabilities)

    response = await _turn(app, client, {"quality": 3})

    assert _errors(response) == [{"code": "setting-not-a-choice", "field": "quality"}]
