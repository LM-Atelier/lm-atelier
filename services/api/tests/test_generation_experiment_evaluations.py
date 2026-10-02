"""Saying which of a comparison's two pictures is preferred, a tie, or neither suiting."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_generation_experiment_preflight import _request, _two_choices
from test_generation_experiment_records import CREATE, _accept, _tamper
from test_generation_experiment_start import _run_ids, _start, _terminal
from test_generation_retry import choose_retries, failing_media

from local_lm.db import SessionLocal
from local_lm.models import GenerationExperimentEvaluation


def _evaluations() -> list[tuple[int, str, str, str | None]]:
    with SessionLocal() as session:
        return [
            (row.sequence, row.mode, row.preference, row.note)
            for row in session.scalars(
                select(GenerationExperimentEvaluation).order_by(
                    GenerationExperimentEvaluation.sequence
                )
            )
        ]


def _count() -> int:
    with SessionLocal() as session:
        return int(
            session.scalar(select(func.count()).select_from(GenerationExperimentEvaluation)) or 0
        )


async def _started(client: AsyncClient) -> dict[str, Any]:
    """A started comparison once both of its pictures are made."""

    accepted = await _accept(client, _request(*_two_choices()))
    started = await _start(client, accepted)
    assert started.status_code == 202, started.text
    body: dict[str, Any] = started.json()
    finished = [await _terminal(client, run_id) for run_id in _run_ids(body)]
    assert [run["status"] for run in finished] == ["complete", "complete"]
    return body


async def test_each_thing_said_is_kept_and_the_latest_is_the_answer(client: AsyncClient) -> None:
    started = await _started(client)
    assert started["evaluation"] is None
    url = f"{CREATE}/{started['id']}/evaluations"

    preferred = await client.post(
        url, json={"preference": "preferred", "arm_ordinal": 2, "note": "  Calmer water  "}
    )

    assert preferred.status_code == 201, preferred.text
    said = preferred.json()["evaluation"]
    assert {key: said[key] for key in ("preference", "mode", "arm_ordinal", "note")} == {
        "preference": "preferred",
        "mode": "unblinded",
        "arm_ordinal": 2,
        "note": "Calmer water",
    }
    assert said["created_at"].endswith("Z")
    read = (await client.get(f"{CREATE}/{started['id']}")).json()
    assert read["evaluation"] == said

    tied = await client.post(url, json={"preference": "tied"})

    assert tied.status_code == 201, tied.text
    assert tied.json()["evaluation"]["preference"] == "tied"
    assert tied.json()["evaluation"]["arm_ordinal"] is None
    # Nothing said before is lost; the latest is the answer.
    assert _evaluations() == [
        (1, "unblinded", "preferred", "Calmer water"),
        (2, "unblinded", "tied", None),
    ]
    # Saying it changes nothing about what was accepted.
    assert tied.json()["snapshot_sha256"] == started["snapshot_sha256"]
    assert tied.json()["arms"] == read["arms"]


async def test_a_preference_waits_for_the_pictures(client: AsyncClient) -> None:
    accepted = await _accept(client, _request(*_two_choices()))

    response = await client.post(
        f"{CREATE}/{accepted['id']}/evaluations", json={"preference": "unsuitable"}
    )

    assert response.status_code == 409
    assert response.json()["code"] == "generation-experiment-not-started"
    assert _count() == 0


async def test_nothing_is_said_before_a_picture_is_made(app: FastAPI, client: AsyncClient) -> None:
    accepted = await _accept(client, _request(*_two_choices()))
    async with app.state.services.scheduler.lease("primary"):
        started = (await _start(client, accepted)).json()
        statuses = [trial["status"] for arm in started["arms"] for trial in arm["trials"]]
        assert statuses == ["queued", "queued"]
        url = f"{CREATE}/{started['id']}/evaluations"

        for said in (
            {"preference": "tied"},
            {"preference": "unsuitable"},
            {"preference": "preferred", "arm_ordinal": 2},
        ):
            response = await client.post(url, json=said)
            assert response.status_code == 409, response.text
            assert response.json()["code"] == "generation-experiment-picture-not-ready"

    assert _count() == 0


async def test_only_a_choice_whose_picture_was_made_can_be_preferred(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await choose_retries(client, 0)
    attempts = failing_media(app, monkeypatch, 1)
    accepted = await _accept(client, _request(*_two_choices()))
    started = (await _start(client, accepted)).json()
    finished = {run_id: await _terminal(client, run_id) for run_id in _run_ids(started)}
    # Whichever picture was asked for first failed, with no retry; the other was made.
    ordinals = {
        trial["run_id"]: arm["ordinal"] for arm in started["arms"] for trial in arm["trials"]
    }
    failed = ordinals[attempts[0].run_id]
    made = next(ordinal for run_id, ordinal in ordinals.items() if run_id != attempts[0].run_id)
    assert finished[attempts[0].run_id]["status"] == "failed"
    url = f"{CREATE}/{started['id']}/evaluations"

    refused = await client.post(url, json={"preference": "preferred", "arm_ordinal": failed})
    preferred = await client.post(url, json={"preference": "preferred", "arm_ordinal": made})

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "generation-experiment-picture-not-ready"
    assert preferred.status_code == 201, preferred.text
    assert _evaluations() == [(1, "unblinded", "preferred", None)]


@pytest.mark.parametrize(
    "said",
    [
        {"preference": "preferred"},
        {"preference": "tied", "arm_ordinal": 1},
        {"preference": "unsuitable", "arm_ordinal": 2},
        {"preference": "preferred", "arm_ordinal": 3},
        {"preference": "best", "arm_ordinal": 1},
        {"preference": "tied", "note": "x" * 501},
        {"preference": "tied", "score": 9},
    ],
)
async def test_a_preference_names_a_choice_exactly_when_one_is_preferred(
    client: AsyncClient, said: dict[str, Any]
) -> None:
    started = await _started(client)

    response = await client.post(f"{CREATE}/{started['id']}/evaluations", json=said)

    assert response.status_code == 422
    assert _count() == 0


async def test_a_comparison_that_changed_or_is_gone_keeps_nothing(client: AsyncClient) -> None:
    started = await _started(client)
    _tamper(started["id"], "effective")

    changed = await client.post(
        f"{CREATE}/{started['id']}/evaluations", json={"preference": "tied"}
    )
    missing = await client.post(f"{CREATE}/gexp_missing/evaluations", json={"preference": "tied"})

    assert changed.status_code == 409
    assert changed.json()["code"] == "generation-experiment-record-invalid"
    assert missing.status_code == 404
    assert missing.json()["code"] == "generation-experiment-not-found"
    assert _count() == 0
