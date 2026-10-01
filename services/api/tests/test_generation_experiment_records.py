"""An accepted comparison keeps exactly what was checked, and is read back only as accepted."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import func, select
from test_generation_experiment_preflight import (
    PREFLIGHT,
    _arm,
    _counts,
    _family_choice,
    _profile,
    _request,
    _revision,
    _two_choices,
)

from local_lm import generation_experiment_store as store
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.generation_experiment_api import REFUSALS
from local_lm.main import create_app
from local_lm.models import (
    GenerationExperiment,
    GenerationExperimentArm,
    GenerationExperimentTrial,
    GenerationPreset,
    ModelInstall,
    ModelProfile,
    WorkflowDefinition,
    WorkflowRevision,
)
from local_lm.orchestrator import MEDIA_SEED_SPACE

CREATE = "/api/generation-experiments"
RECORDS = (GenerationExperiment, GenerationExperimentArm, GenerationExperimentTrial)


def _records() -> tuple[int, ...]:
    with SessionLocal() as session:
        return tuple(
            int(session.scalar(select(func.count()).select_from(model)) or 0) for model in RECORDS
        )


async def _accept(
    client: AsyncClient, body: dict[str, Any], key: str = "accept-one"
) -> dict[str, Any]:
    checked = (await client.post(PREFLIGHT, json=body)).json()
    assert checked["outcome"] == "compatible", checked["refusals"]
    response = await client.post(
        CREATE,
        json={**body, "idempotency_key": key, "preflight_sha256": checked["preflight_sha256"]},
    )
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    assert created["preflight_sha256"] == checked["preflight_sha256"]
    return created


def _seeds(body: dict[str, Any]) -> list[int]:
    return [arm["trials"][0]["seed"] for arm in body["arms"]]


@pytest.mark.parametrize(
    ("policy", "expected", "draws"),
    [
        ({"kind": "same_recorded_number", "seed": 41}, [41, 41], 0),
        ({"kind": "same_recorded_number"}, [1234, 1234], 1),
        ({"kind": "independent_deterministic", "seed": 4}, [4, 5], 0),
        (
            {"kind": "independent_deterministic", "seed": MEDIA_SEED_SPACE - 1},
            [MEDIA_SEED_SPACE - 1, 0],
            0,
        ),
        ({"kind": "independent_deterministic"}, [1234, 1235], 1),
        ({"kind": "random_per_trial"}, [1234, 5678], 2),
    ],
)
async def test_an_accepted_comparison_gives_each_picture_its_seed(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    policy: dict[str, Any],
    expected: list[int],
    draws: int,
) -> None:
    drawn = iter([1234, 5678])
    asked: list[int] = []

    def randbelow(space: int) -> int:
        asked.append(space)
        return next(drawn)

    monkeypatch.setattr(store, "secrets", SimpleNamespace(randbelow=randbelow))
    first, second = _two_choices()
    before = _counts()
    created = await _accept(client, _request(first, second, seed_policy=policy))
    seeds = _seeds(created)
    # A drawn seed is each picture's own draw from the whole seed space, never a constant.
    assert seeds == expected
    assert asked == [MEDIA_SEED_SPACE] * draws
    assert created["state"] == "ready" and created["seed_equivalence"] == "none"
    assert [trial["state"] for arm in created["arms"] for trial in arm["trials"]] == [
        "planned",
        "planned",
    ]
    # Accepting writes the comparison and nothing that would run it yet.
    assert _counts() == before
    assert _records() == (1, 2, 2)
    read = await client.get(f"{CREATE}/{created['id']}")
    assert read.status_code == 200 and read.json() == created


async def test_one_fixed_seed_for_one_family_is_accepted_with_its_claim(
    client: AsyncClient,
) -> None:
    body = _request(
        _family_choice("Left", "Krea-2"),
        _family_choice("Right", "krea2"),
        seed_policy={"kind": "fixed_numeric", "seed": 9},
    )
    created = await _accept(client, body)
    assert _seeds(created) == [9, 9]
    assert created["seed_equivalence"] == "same_family"


async def test_a_retry_with_one_key_returns_the_comparison_already_accepted(
    client: AsyncClient,
) -> None:
    first, second = _two_choices()
    body = _request(first, second)
    created = await _accept(client, body, key="retry-key")
    replay = await client.post(
        CREATE,
        json={
            **body,
            "idempotency_key": "retry-key",
            "preflight_sha256": created["preflight_sha256"],
        },
    )
    assert replay.status_code == 200 and replay.json() == created
    other = {**body, "name": "Another comparison"}
    conflict = await client.post(
        CREATE,
        json={
            **other,
            "idempotency_key": "retry-key",
            "preflight_sha256": created["preflight_sha256"],
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "generation-experiment-idempotency-conflict"
    assert _records() == (1, 2, 2)


async def test_two_creates_with_one_key_at_once_accept_one_comparison(
    client: AsyncClient,
) -> None:
    first, second = _two_choices()
    body = _request(first, second)
    checked = (await client.post(PREFLIGHT, json=body)).json()
    payload = {
        **body,
        "idempotency_key": "together",
        "preflight_sha256": checked["preflight_sha256"],
    }
    responses = await asyncio.gather(*(client.post(CREATE, json=payload) for _ in range(2)))
    assert sorted(response.status_code for response in responses) == [200, 201]
    assert responses[0].json() == responses[1].json()
    assert _records() == (1, 2, 2)


async def test_a_comparison_with_a_choice_that_cannot_run_accepts_nothing(
    client: AsyncClient,
) -> None:
    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    second = _arm("Second", _profile("Video model", role="video"), _revision("Other"))
    response = await client.post(
        CREATE,
        json={
            **_request(first, second),
            "idempotency_key": "refused",
            "preflight_sha256": "0" * 64,
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "generation-experiment-refused"
    assert [(item["code"], item["arm_ordinal"]) for item in body["refusals"]] == [
        ("arm-profile-unavailable", 2)
    ]
    assert _records() == (0, 0, 0)


async def test_a_choice_that_changed_since_its_check_is_not_accepted(
    client: AsyncClient,
) -> None:
    first, second = _two_choices()
    body = _request(first, second)
    checked = (await client.post(PREFLIGHT, json=body)).json()
    with SessionLocal() as session:
        profile = session.get(ModelProfile, first["profile_id"])
        assert profile is not None
        profile.load_settings_json = {**(profile.load_settings_json or {}), "steps": 30}
        session.commit()
    stale = await client.post(
        CREATE,
        json={**body, "idempotency_key": "stale", "preflight_sha256": checked["preflight_sha256"]},
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "generation-experiment-preflight-changed"
    assert _records() == (0, 0, 0)
    again = (await client.post(PREFLIGHT, json=body)).json()
    assert again["preflight_sha256"] != checked["preflight_sha256"]


def _change_everything_a_choice_came_from(choice: dict[str, Any]) -> None:
    with SessionLocal() as session:
        profile = session.get(ModelProfile, choice["profile_id"])
        assert profile is not None
        profile.name = "Renamed model"
        profile.load_settings_json = {"steps": 50}
        profile.request_settings_json = {"cfg": 9.5}
        session.add(
            GenerationPreset(
                name="Later default", role="image", is_default=True, settings_json={"steps": 2}
            )
        )
        revision = session.get(WorkflowRevision, choice["workflow_revision_id"])
        assert revision is not None
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        newer = WorkflowRevision(
            definition=definition,
            version=2,
            engine="mock",
            api_graph_json={},
            input_schema_json={},
            dependencies_json={},
            trusted=True,
        )
        session.add(newer)
        session.flush()
        definition.current_revision_id = newer.id
        revision.trusted = False
        session.commit()


async def test_later_changes_cannot_retarget_an_accepted_comparison(
    client: AsyncClient,
) -> None:
    first, second = _two_choices()
    created = await _accept(client, _request(first, second))
    _change_everything_a_choice_came_from(first)
    read = await client.get(f"{CREATE}/{created['id']}")
    assert read.status_code == 200 and read.json() == created


@asynccontextmanager
async def _running(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            session = await client.post("/api/session")
            client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]
            yield client


async def test_an_accepted_comparison_reads_the_same_after_a_restart(
    settings: Settings,
) -> None:
    first, second = _two_choices()
    async with _running(create_app(settings)) as client:
        created = await _accept(client, _request(first, second))
    async with _running(create_app(settings)) as client:
        read = await client.get(f"{CREATE}/{created['id']}")
    assert read.status_code == 200 and read.json() == created


def _tamper(experiment_id: str, part: str) -> None:
    with SessionLocal() as session:
        experiment = session.get(GenerationExperiment, experiment_id)
        assert experiment is not None
        arm = experiment.arms[0]
        if part == "snapshot":
            arm.snapshot_json = {**arm.snapshot_json, "model_family": "another"}
        elif part == "effective":
            arm.effective_settings_json = {**arm.effective_settings_json, "steps": 99}
        elif part == "prompt":
            experiment.common_json = {**experiment.common_json, "prompt": "Something else"}
        elif part == "seed":
            arm.trials[0].seed = arm.trials[0].seed + 1
        elif part == "claim":
            experiment.seed_equivalence = "same_family"
        elif part == "name":
            experiment.name = "Another comparison"
        elif part == "seed policy":
            experiment.seed_policy = "random_per_trial"
        elif part in {"profile_id", "workflow_revision_id", "workflow_activation_id"}:
            setattr(arm, part, getattr(experiment.arms[1], part) or "elsewhere")
        elif part == "model_family":
            arm.model_family = "another"
        elif part == "requested_settings":
            arm.requested_settings_json = {"steps": 99}
        elif part == "trial state":
            arm.trials[0].state = "unknown"
        elif part == "comparison state":
            experiment.state = "unknown"
        else:
            raise AssertionError(part)
        session.commit()


@pytest.mark.parametrize(
    "part",
    [
        "snapshot",
        "effective",
        "prompt",
        "seed",
        "claim",
        "name",
        "seed policy",
        "profile_id",
        "workflow_revision_id",
        "workflow_activation_id",
        "model_family",
        "requested_settings",
        "trial state",
        "comparison state",
    ],
)
async def test_a_record_that_changed_after_it_was_accepted_is_refused(
    client: AsyncClient, part: str
) -> None:
    first, second = _two_choices()
    created = await _accept(client, _request(first, second))
    _tamper(created["id"], part)
    read = await client.get(f"{CREATE}/{created['id']}")
    assert read.status_code == 409
    assert read.json()["code"] == "generation-experiment-record-invalid"


async def test_an_accepted_choice_keeps_the_trigger_words_its_picture_will_carry(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        install = ModelInstall(
            name="Worded base",
            role="image",
            engine="mock",
            local_path="C:/managed/worded-record",
            manifest_json={"trigger_words": ["harborlight"]},
            active=True,
        )
        session.add(install)
        session.commit()
        install_id = install.id
    worded = _arm(
        "Worded", _profile("Worded model", model_install_id=install_id), _revision("Worded")
    )
    plain = _arm("Plain", _profile("Plain model"), _revision("Plain"))
    created = await _accept(client, _request(worded, plain))
    assert [arm["trigger_words_applied"] for arm in created["arms"]] == [["harborlight"], []]
    with SessionLocal() as session:
        install_row = session.get(ModelInstall, install_id)
        assert install_row is not None
        install_row.manifest_json = {"trigger_words": ["changed"]}
        session.commit()
    read = await client.get(f"{CREATE}/{created['id']}")
    assert read.status_code == 200 and read.json() == created


async def test_a_missing_comparison_is_not_found(client: AsyncClient) -> None:
    response = await client.get(f"{CREATE}/gexp_missing")
    assert response.status_code == 404
    assert response.json()["code"] == "generation-experiment-not-found"


def test_every_refusal_has_one_kebab_case_code_and_a_status() -> None:
    for code, (status, message) in REFUSALS.items():
        assert re.fullmatch(r"[a-z]+(?:-[a-z]+)+", code), code
        assert status in {404, 409, 422, 503} and message.endswith(".")


@pytest.mark.parametrize("same_request", [True, False])
async def test_a_create_that_loses_the_race_for_its_key_is_settled_by_the_database(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, same_request: bool
) -> None:
    """The early lookup can miss a comparison committed a moment later; the unique key cannot."""

    first, second = _two_choices()
    body = _request(first, second)
    created = await _accept(client, body, key="raced")
    real_find = store.find
    lookups: list[str] = []

    def find_after_the_check(session: Any, key: str) -> Any:
        lookups.append(key)
        return None if len(lookups) == 1 else real_find(session, key)

    monkeypatch.setattr(store, "find", find_after_the_check)
    sent = body if same_request else {**body, "name": "Another comparison"}
    response = await client.post(
        CREATE,
        json={**sent, "idempotency_key": "raced", "preflight_sha256": created["preflight_sha256"]},
    )
    assert lookups == ["raced", "raced"]
    if same_request:
        assert response.status_code == 200 and response.json() == created
    else:
        assert response.status_code == 409
        assert response.json()["code"] == "generation-experiment-idempotency-conflict"
    assert _records() == (1, 2, 2)
