"""A selection and a relight request survive every turn rebuilt from an edit's settings.

Neither is a workflow field, so rebuilding a turn's settings against the fields
its workflow offers drops both. A regenerate, a branch and an edited turn,
routed again in Auto mode or not, each start from the source turn's settings on
the same pictures, and each must still change only the part that was selected,
hand an inpainting workflow its selection, and finish a relight against its
source. An edit generated again from its record is one of them. Given new
pictures, the selection drawn on the old ones does not follow.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image, ImageDraw
from run_waits import wait_for_terminal_status, wait_until
from test_output_recipe_replay import _replays_exactly
from test_output_recipe_replay_masks import _masked_edit, _picture
from test_studio_region_edit_turns import BOX, _masked_edit_workflow, _plain_edit_workflow
from test_studio_relight_turns import RELIGHT, _light_map, _lora, _relight_workflow

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import Artifact, Run
from local_lm.studio_relight import RelightSetting, adapter_instruction

GREY = (128, 128, 128)
EDITED = (10, 200, 30)
SIZE = (64, 48)
OUTSIDE = (2, 2)
INSIDE = (BOX[0] + 4, BOX[1] + 4)
ROUTES = ["regenerate", "branch", "edit"]


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _selection() -> bytes:
    alpha = Image.new("L", SIZE, 0)
    ImageDraw.Draw(alpha).rectangle(BOX, fill=255)
    mask = Image.new("RGBA", SIZE, (255, 255, 255, 0))
    mask.putalpha(alpha)
    return _png(mask)


async def _upload(client: AsyncClient, name: str, content: bytes) -> str:
    response = await client.post("/api/artifacts", files={"file": (name, content, "image/png")})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _engine(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[MediaRequest]:
    """Answer every generation with one flat picture, keeping what each was handed."""
    seen: list[MediaRequest] = []

    async def generate(request: MediaRequest) -> Any:
        seen.append(request)
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=_png(Image.new("RGB", SIZE, EDITED)),
                    kind="image",
                    media_type="image/png",
                    name="edited.png",
                )
            ],
        )

    monkeypatch.setattr(app.state.services.orchestrator.engines.media, "generate", generate)
    return seen


async def _settled(client: AsyncClient, accepted: dict[str, Any]) -> Run:
    plan_id = accepted["run"]["work_plan_id"]

    async def read_plan() -> dict[str, Any]:
        plan: dict[str, Any] = (await client.get(f"/api/work-plans/{plan_id}")).json()
        return plan

    plan = await wait_until(
        read_plan,
        lambda plan: plan["status"] in {"complete", "failed", "cancelled", "blocked"},
        what=f"work plan {plan_id}",
    )
    assert plan["status"] == "complete", plan
    run_id = accepted["run"]["id"]

    async def read_run() -> dict[str, Any]:
        run: dict[str, Any] = (await client.get(f"/api/runs/{run_id}")).json()
        return run

    await wait_for_terminal_status(read_run, what=f"run {run_id}")
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        session.expunge(run)
        return run


def _stored(app: FastAPI, run: Run) -> tuple[Image.Image, dict[str, Any]]:
    with SessionLocal() as session:
        (entry,) = run.provenance_json["outputs"]
        artifact = session.get(Artifact, entry["artifact_id"])
        assert artifact is not None
        content = app.state.services.orchestrator.artifacts.resolve(artifact).read_bytes()
    return Image.open(io.BytesIO(content)).convert("RGB"), entry


def _handed(seen: list[MediaRequest], run: Run) -> MediaRequest:
    (request,) = [request for request in seen if request.run_id == run.id]
    return request


async def _edit(client: AsyncClient, kind: str) -> tuple[dict[str, Any], str]:
    """One finished edit of a grey picture: a blended selection, an inpaint or a relight."""
    source_id = await _upload(client, "source.png", _png(Image.new("RGB", SIZE, GREY)))
    if kind == "relight":
        revision_id = _relight_workflow()
        inputs = [source_id, await _upload(client, "light.png", _png(_light_map()))]
        settings: dict[str, Any] = {
            "relight": RELIGHT,
            "loras": [{"asset_id": _lora(), "model_strength": 1}],
        }
        text = "Relight from the left"
    else:
        revision_id = _plain_edit_workflow() if kind == "blend" else _masked_edit_workflow()
        selection: dict[str, Any] = {
            "artifact_id": await _upload(client, "selection.png", _selection()),
            "feather_px": 0,
            "invert": False,
            **({"apply": "blend"} if kind == "blend" else {}),
        }
        inputs = [source_id]
        settings = {"mask": selection}
        text = "Paint the marked part green"
    chat = (await client.post("/api/chats", json={"title": "Selected edit"})).json()
    sent = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": text,
            "mode": "image",
            "input_artifact_ids": inputs,
            "workflow_revision_id": revision_id,
            "settings": settings,
        },
    )
    assert sent.status_code == 202, sent.text
    accepted: dict[str, Any] = sent.json()
    await _settled(client, accepted)
    return accepted, text


async def _follow_up(
    client: AsyncClient,
    route: str,
    user_message_id: str,
    assistant_message_id: str,
    text: str,
    **extra: Any,
) -> Any:
    if route == "regenerate":
        return await client.post(f"/api/messages/{assistant_message_id}/regenerate", json=extra)
    if route == "branch":
        return await client.post(
            f"/api/messages/{user_message_id}/branch", json={"text": f"{text}, once more", **extra}
        )
    return await client.post(
        f"/api/messages/{user_message_id}/edits",
        json={"text": f"{text}, once more", "idempotency_key": f"again-{route}", **extra},
    )


async def _rebuilt(
    client: AsyncClient, route: str, source: dict[str, Any], text: str, **extra: Any
) -> Run:
    follow = await _follow_up(
        client,
        route,
        source["user_message"]["id"],
        source["assistant_message"]["id"],
        text,
        **extra,
    )
    assert follow.status_code == 202, follow.text
    return await _settled(client, follow.json())


def _changed_only_the_selection(app: FastAPI, source: dict[str, Any], run: Run) -> None:
    assert run.settings_json["mask"] == source["run"]["settings_json"]["mask"]
    picture, entry = _stored(app, run)
    assert entry["region_edit"]["mode"] == "blend"
    assert picture.getpixel(OUTSIDE) == GREY
    assert picture.getpixel(INSIDE) == EDITED


def _handed_the_selection(seen: list[MediaRequest], source: dict[str, Any], run: Run) -> None:
    # Without its selection, an inpainter is not chosen for an unpinned turn at all.
    assert run.workflow_revision_id == source["run"]["workflow_revision_id"]
    request = _handed(seen, run)
    selection = source["run"]["settings_json"]["mask"]
    assert request.parameters["mask"]["artifact_id"] == selection["artifact_id"]
    assert request.mask_content is not None


def _finished_against_its_source(
    app: FastAPI, seen: list[MediaRequest], source: dict[str, Any], run: Run
) -> None:
    assert run.settings_json["relight"] == source["run"]["settings_json"]["relight"]
    assert _handed(seen, run).prompt == adapter_instruction(RelightSetting("left", 0.5, None))
    picture, entry = _stored(app, run)
    assert "relight" in entry
    # The stored picture is mixed back toward the source, not the raw result.
    assert picture.getpixel(OUTSIDE) != EDITED


@pytest.mark.parametrize("route", ROUTES)
async def test_a_turn_rebuilt_from_a_blended_selection_changes_only_the_selection(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    _engine(app, monkeypatch)
    source, text = await _edit(client, "blend")

    run = await _rebuilt(client, route, source, text)

    _changed_only_the_selection(app, source, run)


@pytest.mark.parametrize("route", ROUTES)
async def test_a_turn_rebuilt_from_an_inpaint_hands_its_workflow_the_selection(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    seen = _engine(app, monkeypatch)
    source, text = await _edit(client, "inpaint")

    run = await _rebuilt(client, route, source, text)

    _handed_the_selection(seen, source, run)


@pytest.mark.parametrize("route", ROUTES)
async def test_a_turn_rebuilt_from_a_relight_is_finished_against_its_source(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    seen = _engine(app, monkeypatch)
    source, text = await _edit(client, "relight")

    run = await _rebuilt(client, route, source, text)

    _finished_against_its_source(app, seen, source, run)


@pytest.mark.parametrize("kind", ["blend", "inpaint", "relight"])
async def test_an_edit_routed_again_in_auto_mode_keeps_the_selection_and_the_relight(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    seen = _engine(app, monkeypatch)
    source, text = await _edit(client, kind)

    run = await _rebuilt(client, "edit", source, text, mode="auto")

    # Routed afresh, to the same edit of the same pictures.
    assert run.operation == Operation.IMAGE_TO_IMAGE.value
    if kind == "blend":
        _changed_only_the_selection(app, source, run)
    elif kind == "inpaint":
        _handed_the_selection(seen, source, run)
    else:
        _finished_against_its_source(app, seen, source, run)


@pytest.mark.parametrize("route", ["regenerate", "edit"])
async def test_an_edit_generated_again_from_its_record_keeps_its_selection_when_made_again(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """A record generated again lands in a chat that offers Regenerate and Edit on it."""
    seen = _engine(app, monkeypatch)
    mask = await _picture(client, (255, 255, 255))
    _, content = await _masked_edit(client, {"artifact_id": mask})
    again = await _replays_exactly(client, content)
    assert (await client.get(f"/api/chats/{again['chat_id']}")).status_code == 200

    follow = await _follow_up(
        client, route, again["user_message_id"], again["assistant_message_id"], "make the cup blue"
    )

    assert follow.status_code == 202, follow.text
    run = await _settled(client, follow.json())
    assert run.settings_json["mask"] == {"artifact_id": mask}
    assert run.workflow_revision_id == again["workflow_revision_id"]
    assert _handed(seen, run).parameters["mask"]["artifact_id"] == mask


@pytest.mark.parametrize("route", ["branch", "edit"])
async def test_new_pictures_do_not_take_the_old_selection(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    _engine(app, monkeypatch)
    source, text = await _edit(client, "blend")
    other = await _upload(client, "other.png", _png(Image.new("RGB", SIZE, (1, 2, 3))))

    # The settings are still carried over; only the pictures are new.
    run = await _rebuilt(client, route, source, text, input_artifact_ids=[other])

    assert run.settings_json["steps"] == source["run"]["settings_json"]["steps"]
    assert "mask" not in run.settings_json


async def test_a_regenerate_can_leave_the_selection_out(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _engine(app, monkeypatch)
    source, text = await _edit(client, "blend")

    run = await _rebuilt(client, "regenerate", source, text, settings={"mask": None})

    assert run.settings_json.get("mask") is None
    picture, _ = _stored(app, run)
    assert picture.getpixel(OUTSIDE) == EDITED
