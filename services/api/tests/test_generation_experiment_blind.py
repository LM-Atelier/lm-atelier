"""A blind comparison: no answer links a picture to its choice until the preference is said."""

from __future__ import annotations

import hashlib
import io
import json
import secrets
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image, ImageCms, PngImagePlugin
from sqlalchemy import select
from test_generation_experiment_preflight import PREFLIGHT, _request, _two_choices
from test_generation_experiment_records import CREATE, _accept, _records, _tamper
from test_generation_experiment_start import _start, _terminal

from local_lm import generation_experiment_blind as blind
from local_lm import generation_experiment_store as store
from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.artifacts import ArtifactStore
from local_lm.db import SessionLocal
from local_lm.generation_experiments_v1 import (
    GenerationExperimentCreate,
    GenerationExperimentEvaluationCreate,
    GenerationExperimentEvaluationMode,
    canonical_sha256,
)
from local_lm.models import (
    Artifact,
    GenerationExperiment,
    GenerationExperimentArm,
    GenerationExperimentBlindView,
    GenerationExperimentTrial,
)

LABELS = ("Fewer steps", "More steps")


@pytest.fixture(autouse=True)
def raster_pictures(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pictures as a real engine makes them: a PNG, in a colour of each run's own."""

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        shade = hashlib.sha256(request.run_id.encode()).digest()[0]
        made = io.BytesIO()
        Image.new("RGB", (8, 6), (shade, 255 - shade, 90)).save(made, format="PNG")
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=made.getvalue(),
                    kind="image",
                    media_type="image/png",
                    name="picture.png",
                    origin={"node_id": "9", "output_type": "output", "collection": "images"},
                )
            ],
        )

    monkeypatch.setattr(app.state.services.engines.media, "generate", generate)


def _blind_request() -> dict[str, Any]:
    return {**_request(*_two_choices()), "evaluation_mode": "blind"}


def _runs(experiment_id: str) -> dict[int, str]:
    """Each choice's run, read from the database: no answer gives it while blind."""

    with SessionLocal() as session:
        rows = session.execute(
            select(GenerationExperimentArm.ordinal, GenerationExperimentTrial.run_id)
            .join(GenerationExperimentTrial)
            .where(GenerationExperimentArm.experiment_id == experiment_id)
        ).all()
    return {int(ordinal): str(run_id) for ordinal, run_id in rows}


async def _blind_started(client: AsyncClient) -> dict[str, Any]:
    accepted = await _accept(client, _blind_request())
    started = await _start(client, accepted)
    assert started.status_code == 202, started.text
    for run_id in _runs(accepted["id"]).values():
        assert (await _terminal(client, run_id))["status"] == "complete"
    body: dict[str, Any] = started.json()
    return body


def _linked(experiment: dict[str, Any]) -> list[Any]:
    return [
        trial[key]
        for arm in experiment["arms"]
        for trial in arm["trials"]
        for key in ("work_step_id", "run_id", "job_id", "status")
        if trial[key] is not None
    ]


async def _picture(client: AsyncClient, url: str) -> Image.Image:
    response = await client.get(url)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert "content-disposition" not in response.headers
    picture = Image.open(io.BytesIO(response.content))
    picture.load()
    return picture


async def test_no_answer_links_a_picture_to_its_choice_until_the_preference_is_said(
    client: AsyncClient,
) -> None:
    started = await _blind_started(client)
    assert started["evaluation_mode"] == "blind" and started["blind_pending"] is True
    assert _linked(started) == []
    read = (await client.get(f"{CREATE}/{started['id']}")).json()
    assert read["blind_pending"] is True and _linked(read) == []
    # What would lead from a picture to its choice waits for the saying too.
    draft = await client.get(f"{CREATE}/{started['id']}/arms/1/recipe-draft")
    unblinded = await client.post(
        f"{CREATE}/{started['id']}/evaluations", json={"preference": "tied"}
    )
    for refused in (draft, unblinded):
        assert refused.status_code == 409
        assert refused.json()["code"] == "generation-experiment-blind"


async def test_a_viewing_shows_each_picture_plainly_by_where_it_is_shown(
    client: AsyncClient,
) -> None:
    started = await _blind_started(client)
    runs = _runs(started["id"])

    opened = await client.post(f"{CREATE}/{started['id']}/blind-views")

    assert opened.status_code == 201, opened.text
    view = opened.json()
    assert view["positions"] == [
        {"position": 1, "status": "complete", "ready": True},
        {"position": 2, "status": "complete", "ready": True},
    ]
    assert view["evaluation"] is None and view["reveal"] is None
    # Nothing in the viewing names a choice, a run or a picture.
    text = json.dumps(view)
    for secret in (*LABELS, *runs.values(), *[arm["id"] for arm in started["arms"]]):
        assert secret not in text
    pictures = [
        await _picture(client, f"{CREATE}/{started['id']}/blind-views/{view['id']}/pictures/{p}")
        for p in (1, 2)
    ]
    read = (await client.get(f"{CREATE}/{started['id']}/blind-views/{view['id']}")).json()
    assert read == view

    said = await client.post(
        f"{CREATE}/{started['id']}/blind-views/{view['id']}/evaluations",
        json={"preference": "preferred", "position": 1, "note": "Calmer"},
    )

    assert said.status_code == 201, said.text
    revealed = said.json()
    reveal = revealed["reveal"]
    assert [item["position"] for item in reveal] == [1, 2]
    assert {item["label"] for item in reveal} == set(LABELS)
    assert sorted(item["arm_ordinal"] for item in reveal) == [1, 2]
    evaluation = revealed["evaluation"]
    assert evaluation["mode"] == "blind" and evaluation["preference"] == "preferred"
    assert evaluation["arm_ordinal"] == reveal[0]["arm_ordinal"]
    assert evaluation["note"] == "Calmer"
    # Each position showed the picture its revealed choice made.
    with SessionLocal() as session:
        for item, shown in zip(reveal, pictures, strict=True):
            run_id = runs[item["arm_ordinal"]]
            outputs = (await client.get(f"/api/runs/{run_id}")).json()["provenance_json"]["outputs"]
            kept = [
                output["artifact_id"]
                for output in outputs
                if output.get("kind") == "image"
                and (output.get("output_origin") or {}).get("output_type") != "temp"
            ]
            artifact = session.get(Artifact, kept[0])
            assert artifact is not None
            made = Image.open(
                io.BytesIO((await client.get(f"/api/artifacts/{artifact.id}/content")).content)
            )
            assert shown.convert("RGBA").tobytes() == made.convert("RGBA").tobytes()


async def test_nothing_is_said_blind_before_a_picture_is_made(
    app: FastAPI, client: AsyncClient
) -> None:
    accepted = await _accept(client, _blind_request())
    async with app.state.services.scheduler.lease("primary"):
        started = (await _start(client, accepted)).json()
        opened = await client.post(f"{CREATE}/{started['id']}/blind-views")
        assert opened.status_code == 201, opened.text
        view = opened.json()
        assert [position["ready"] for position in view["positions"]] == [False, False]
        url = f"{CREATE}/{started['id']}/blind-views/{view['id']}/evaluations"

        for said in ({"preference": "tied"}, {"preference": "preferred", "position": 1}):
            response = await client.post(url, json=said)
            assert response.status_code == 409, response.text
            assert response.json()["code"] == "generation-experiment-picture-not-ready"

    with SessionLocal() as session:
        experiment = session.get(GenerationExperiment, started["id"])
        assert experiment is not None and experiment.evaluations == []
        assert store.blind_pending(experiment)


async def test_saying_the_preference_names_the_choices_from_then_on(client: AsyncClient) -> None:
    started = await _blind_started(client)
    view = (await client.post(f"{CREATE}/{started['id']}/blind-views")).json()
    url = f"{CREATE}/{started['id']}/blind-views/{view['id']}/evaluations"

    assert (await client.post(url, json={"preference": "tied"})).status_code == 201

    read = (await client.get(f"{CREATE}/{started['id']}")).json()
    assert read["blind_pending"] is False
    assert sorted(_runs(started["id"]).values()) == sorted(
        trial["run_id"] for arm in read["arms"] for trial in arm["trials"]
    )
    assert read["evaluation"]["mode"] == "blind" and read["evaluation"]["preference"] == "tied"
    # One blind saying: the same viewing, or a new one, takes no other.
    again = await client.post(url, json={"preference": "unsuitable"})
    reopened = await client.post(f"{CREATE}/{started['id']}/blind-views")
    for refused in (again, reopened):
        assert refused.status_code == 409
        assert refused.json()["code"] == "generation-experiment-revealed"
    # From then on it is said with the choices named, like any other comparison.
    named = await client.post(
        f"{CREATE}/{started['id']}/evaluations", json={"preference": "preferred", "arm_ordinal": 1}
    )
    assert named.status_code == 201
    assert named.json()["evaluation"]["mode"] == "unblinded"
    assert (await client.get(f"{CREATE}/{started['id']}/arms/1/recipe-draft")).status_code == 200


@pytest.mark.parametrize("drawn", [0, 1])
async def test_each_viewing_draws_its_own_order(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, drawn: int
) -> None:
    started = await _blind_started(client)
    monkeypatch.setattr(secrets, "randbelow", lambda _bound: drawn)
    view = (await client.post(f"{CREATE}/{started['id']}/blind-views")).json()

    said = await client.post(
        f"{CREATE}/{started['id']}/blind-views/{view['id']}/evaluations",
        json={"preference": "unsuitable"},
    )

    assert [item["arm_ordinal"] for item in said.json()["reveal"]] == (
        [1, 2] if not drawn else [2, 1]
    )


async def test_a_viewing_is_refused_where_it_cannot_be_one(client: AsyncClient) -> None:
    named = await _accept(client, _request(*_two_choices()), key="named")
    unstarted = await _accept(client, _blind_request(), key="unstarted")
    started = await _blind_started(client)
    view = (await client.post(f"{CREATE}/{started['id']}/blind-views")).json()

    not_blind = await client.post(f"{CREATE}/{named['id']}/blind-views")
    not_started = await client.post(f"{CREATE}/{unstarted['id']}/blind-views")
    missing = await client.get(f"{CREATE}/{started['id']}/blind-views/gview_missing")
    elsewhere = await client.get(f"{CREATE}/{unstarted['id']}/blind-views/{view['id']}")
    no_picture = await client.get(f"{CREATE}/{started['id']}/blind-views/{view['id']}/pictures/3")
    two_at_once = await client.post(
        f"{CREATE}/{started['id']}/blind-views/{view['id']}/evaluations",
        json={"preference": "tied", "position": 1},
    )

    assert (not_blind.status_code, not_blind.json()["code"]) == (
        409,
        "generation-experiment-not-blind",
    )
    assert (not_started.status_code, not_started.json()["code"]) == (
        409,
        "generation-experiment-not-started",
    )
    for gone in (missing, elsewhere):
        assert (gone.status_code, gone.json()["code"]) == (
            404,
            "generation-experiment-blind-view-not-found",
        )
    assert (no_picture.status_code, no_picture.json()["code"]) == (
        409,
        "generation-experiment-blind-picture-not-ready",
    )
    assert two_at_once.status_code == 422
    _tamper(started["id"], "effective")
    changed = await client.get(f"{CREATE}/{started['id']}/blind-views/{view['id']}")
    assert (changed.status_code, changed.json()["code"]) == (
        409,
        "generation-experiment-record-invalid",
    )


async def test_two_blind_sayings_at_once_do_not_both_count(client: AsyncClient) -> None:
    started = await _blind_started(client)
    first = (await client.post(f"{CREATE}/{started['id']}/blind-views")).json()
    second = (await client.post(f"{CREATE}/{started['id']}/blind-views")).json()
    said = await client.post(
        f"{CREATE}/{started['id']}/blind-views/{first['id']}/evaluations",
        json={"preference": "tied"},
    )
    assert said.status_code == 201

    # The other viewing's saying was already past every check made before the
    # write lock; the one made under it still refuses it.
    with SessionLocal() as session, pytest.raises(ValueError, match="one blind saying"):
        store.evaluate(
            session,
            started["id"],
            GenerationExperimentEvaluationCreate(preference="unsuitable"),
            GenerationExperimentEvaluationMode.BLIND,
            blind_view_id=second["id"],
        )
    with SessionLocal() as session:
        kept = session.scalars(
            select(GenerationExperimentBlindView).where(
                GenerationExperimentBlindView.experiment_id == started["id"]
            )
        ).all()
        assert sorted(view.evaluation_id is not None for view in kept) == [False, True]


def test_a_picture_is_served_without_anything_its_workflow_wrote_into_it() -> None:
    written = PngImagePlugin.PngInfo()
    written.add_text("workflow", '{"model": "the one that made it"}')
    exif = Image.Exif()
    exif[0x0131] = "the program that made it"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    original = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(
        original, format="PNG", pnginfo=written, exif=exif, icc_profile=profile
    )

    class Store:
        def verified_bytes(self, _artifact: Artifact, *, maximum_bytes: int) -> bytes:
            assert maximum_bytes > 0
            return original.getvalue()

    with SessionLocal() as session:
        experiment = GenerationExperiment(id="gexp_plain", common_json={})
        arm = GenerationExperimentArm(id="garm_plain", ordinal=1)
        trial = GenerationExperimentTrial(id="gtrial_plain", ordinal=1, run_id="run_plain")
        arm.trials = [trial]
        experiment.arms = [arm]
        view = GenerationExperimentBlindView(order_json=["garm_plain"])
        picked = Artifact(id="sha256:" + "0" * 64)

        def picture_artifact(_session: Any, chosen: GenerationExperimentTrial) -> Artifact:
            assert chosen is trial
            return picked

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(blind, "_picture_artifact", picture_artifact)
            served = blind.view_picture(session, cast(ArtifactStore, Store()), experiment, view, 1)

    plain = Image.open(io.BytesIO(served))
    plain.load()
    assert "workflow" not in plain.info and not getattr(plain, "text", {})
    # Nor what else a file can carry about where it came from.
    assert "exif" not in plain.info and "icc_profile" not in plain.info
    assert plain.size == (4, 3) and plain.convert("RGB").getpixel((0, 0)) == (10, 20, 30)


async def test_a_named_comparison_accepted_before_blind_comparison_is_still_retried(
    client: AsyncClient,
) -> None:
    """Before evaluation_mode existed, a create hashed without it; its retry must still match."""

    body = _request(*_two_choices())
    accepted = await _accept(client, body, key="accepted-before-blind")
    request = {
        **body,
        "idempotency_key": "accepted-before-blind",
        "preflight_sha256": accepted["preflight_sha256"],
    }
    # The digest that build stored: every field it had, none of them the mode.
    earlier = canonical_sha256(
        GenerationExperimentCreate.model_validate(request).model_dump(
            mode="json", exclude={"idempotency_key", "evaluation_mode"}
        )
    )
    with SessionLocal() as session:
        experiment = session.get(GenerationExperiment, accepted["id"])
        assert experiment is not None
        experiment.request_sha256 = earlier
        session.commit()

    repeated = await client.post(CREATE, json=request)

    assert repeated.status_code == 200, repeated.text
    assert repeated.json() == accepted
    assert _records() == (1, 2, 2)


@pytest.mark.parametrize("first_blind", [False, True], ids=["named first", "blind first"])
async def test_a_blind_and_a_named_create_under_one_key_do_not_match(
    client: AsyncClient, first_blind: bool
) -> None:
    named = _request(*_two_choices())
    blind_body = {**named, "evaluation_mode": "blind"}
    first, second = (blind_body, named) if first_blind else (named, blind_body)
    accepted = await _accept(client, first, key="one-key-two-modes")
    checked = (await client.post(PREFLIGHT, json=second)).json()
    assert checked["outcome"] == "compatible", checked["refusals"]

    response = await client.post(
        CREATE,
        json={
            **second,
            "idempotency_key": "one-key-two-modes",
            "preflight_sha256": checked["preflight_sha256"],
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "generation-experiment-idempotency-conflict"
    assert (await client.get(f"{CREATE}/{accepted['id']}")).json() == accepted
    assert _records() == (1, 2, 2)
