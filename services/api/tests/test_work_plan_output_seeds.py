"""Each output of a multi-output render reports the seed it was made with.

Read through the real routes: a turn asks for several pictures, each output
is made with its own seed, and the plan's outputs route pairs every output
with that seed, whether the workflow read it, and the files it kept.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status

from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.generation_experiments_v1 import EXPERIMENT_CHAT_SCOPE
from local_lm.models import Chat, ModelInstall, ModelProfile, Run, WorkflowRevision
from local_lm.output_recipe_v1 import open_output_recipe

FOUR = "Create four variations of a blue ceramic apple"


def _seeded_engine(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, *, per_call: int | None = None
) -> list[MediaRequest]:
    """Make the mock engine's files depend on the seed and on their place in a call.

    The mock draws a picture from the prompt alone, so every output of one
    turn would be the same file. Marking the seed into the bytes gives each
    output its own file, and a call asking for a batch of N returns N files.
    With `per_call`, every call returns that many whatever it asked for, as a
    workflow whose graph sets its own count does.
    """

    seen: list[MediaRequest] = []
    original = app.state.services.engines.media.generate

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        seen.append(request)
        size = request.parameters.get("batch_size") if per_call is None else per_call
        count = size if isinstance(size, int) and not isinstance(size, bool) and size > 1 else 1
        seed = request.parameters.get("seed")
        async for event in original(request):
            if event.type == "complete":
                event = replace(
                    event,
                    assets=[
                        replace(
                            asset,
                            content=asset.content
                            + f"<!-- seed {seed} place {place} -->".encode("ascii"),
                        )
                        for asset in event.assets
                        for place in range(count)
                    ],
                )
            yield event

    monkeypatch.setattr(app.state.services.engines.media, "generate", generate)
    return seen


async def _turn(client: AsyncClient, text: str, **values: Any) -> dict[str, Any]:
    chat = (await client.post("/api/chats", json={"title": "Seeds"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": text, "mode": "image", **values}
    )
    assert response.status_code == 202, response.text
    return cast(dict[str, Any], response.json())


async def _finished_plan(client: AsyncClient, plan_id: str) -> dict[str, Any]:
    async def read() -> dict[str, Any]:
        return cast(dict[str, Any], (await client.get(f"/api/work-plans/{plan_id}")).json())

    return dict(await wait_for_terminal_status(read, what=f"work plan {plan_id}"))


async def _outputs(client: AsyncClient, plan_id: str) -> list[dict[str, Any]]:
    response = await client.get(f"/api/work-plans/{plan_id}/outputs")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["plan_id"] == plan_id
    return cast(list[dict[str, Any]], body["outputs"])


async def _record_seed(client: AsyncClient, output: dict[str, Any]) -> dict[str, Any]:
    artifact_id = output["artifacts"][0]["artifact_id"]
    response = await client.get(
        f"/api/runs/{output['run_id']}/outputs/{artifact_id}/recipe", params={"prompts": "omit"}
    )
    assert response.status_code == 200, response.text
    return cast(dict[str, Any], open_output_recipe(response.content)["seed"])


async def test_each_picture_of_a_random_seed_batch_reports_its_own_seed(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("local_lm.orchestrator.secrets.randbelow", lambda _upper: 1000)
    _seeded_engine(app, monkeypatch)

    accepted = await _turn(client, FOUR, idempotency_key="four-seeds")
    plan = await _finished_plan(client, accepted["run"]["work_plan_id"])
    outputs = await _outputs(client, plan["id"])

    assert [output["index"] for output in outputs] == [1, 2, 3, 4]
    assert [output["seed"] for output in outputs] == [1000, 1001, 1002, 1003]
    assert [output["run_id"] for output in outputs] == plan["summary_json"]["run_ids"]
    assert [output["message_id"] for output in outputs] == [
        message["id"] for message in accepted["assistant_messages"]
    ]
    assert [output["work_step_id"] for output in outputs] == [step["id"] for step in plan["steps"]]
    assert {output["status"] for output in outputs} == {"complete"}
    assert {output["operation"] for output in outputs} == {"text_to_image"}
    assert {output["batch_size"] for output in outputs} == {1}
    assert [len(output["artifacts"]) for output in outputs] == [1, 1, 1, 1]
    assert {output["artifacts"][0]["batch_index"] for output in outputs} == {0}
    assert {output["artifacts"][0]["kind"] for output in outputs} == {"image"}
    assert len({output["artifacts"][0]["sha256"] for output in outputs}) == 4
    # The route and each output's own record say the same thing about the seed.
    for output in outputs:
        assert await _record_seed(client, output) == {
            "value": output["seed"],
            "binding": output["seed_binding"],
        }


async def test_the_seeds_of_a_batch_wrap_inside_the_seed_space(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("local_lm.orchestrator.secrets.randbelow", lambda _upper: 2_147_483_646)
    _seeded_engine(app, monkeypatch)

    accepted = await _turn(client, FOUR)
    plan = await _finished_plan(client, accepted["run"]["work_plan_id"])

    assert [output["seed"] for output in await _outputs(client, plan["id"])] == [
        2_147_483_646,
        2_147_483_647,
        0,
        1,
    ]


def _profile() -> str:
    with SessionLocal() as session:
        install = ModelInstall(
            name="Seeded model",
            role="image",
            engine="mock",
            local_path="neutral-model-folder",
            compatibility="compatible",
            manifest_json={"expected_sha256": {"model.safetensors": "1" * 64}},
            active=True,
        )
        session.add(install)
        session.flush()
        profile = ModelProfile(
            name="Seeded model", role="image", engine="mock", model_install_id=install.id
        )
        session.add(profile)
        session.commit()
        return profile.id


async def _workflow_reading_the_seed(client: AsyncClient) -> str:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Seed-reading workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"1": {"class_type": "Mock", "inputs": {"seed": "${seed}"}}},
            "input_schema": {"type": "object", "properties": {}},
        },
    )
    assert created.status_code == 201, created.text
    revision_id = cast(str, created.json()["current_revision_id"])
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.trusted = True
        session.commit()
    return revision_id


async def test_a_seed_the_workflow_reads_is_reported_as_the_one_that_ran(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _seeded_engine(app, monkeypatch)
    revision_id = await _workflow_reading_the_seed(client)

    accepted = await _turn(
        client,
        "Create two variations of a blue ceramic apple",
        profile_id=_profile(),
        workflow_revision_id=revision_id,
        settings={"seed": 1234},
    )
    plan = await _finished_plan(client, accepted["run"]["work_plan_id"])
    outputs = await _outputs(client, plan["id"])

    assert [(output["seed"], output["seed_binding"]) for output in outputs] == [
        (1234, "bound"),
        (1235, "bound"),
    ]
    assert sorted(request.parameters["seed"] for request in seen) == [1234, 1235]


async def test_resubmitting_one_outputs_seed_makes_that_picture_again(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("local_lm.orchestrator.secrets.randbelow", lambda _upper: 5000)
    seen = _seeded_engine(app, monkeypatch)
    accepted = await _turn(client, FOUR)
    plan = await _finished_plan(client, accepted["run"]["work_plan_id"])
    third = (await _outputs(client, plan["id"]))[2]
    with SessionLocal() as session:
        run = session.get(Run, third["run_id"])
        assert run is not None
        standalone_prompt = run.standalone_prompt
    made = next(request for request in seen if request.parameters["seed"] == third["seed"])

    again = await _turn(client, standalone_prompt, settings={"seed": third["seed"]})
    repeated = await _finished_plan(client, again["run"]["work_plan_id"])
    [output] = await _outputs(client, repeated["id"])

    assert output["seed"] == third["seed"] == 5002
    assert seen[-1].prompt == made.prompt
    assert output["artifacts"][0]["sha256"] == third["artifacts"][0]["sha256"]


async def test_pictures_one_engine_call_made_share_its_seed_and_say_their_place(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workflow whose graph sets its own count makes several pictures in one call."""

    monkeypatch.setattr("local_lm.orchestrator.secrets.randbelow", lambda _upper: 77)
    seen = _seeded_engine(app, monkeypatch, per_call=3)
    accepted = await _turn(client, "Create an image of a blue ceramic apple")
    plan = await _finished_plan(client, accepted["run"]["work_plan_id"])
    [output] = await _outputs(client, plan["id"])

    assert [request.parameters.get("batch_size", 1) for request in seen] == [1]
    assert [item["batch_index"] for item in output["artifacts"]] == [0, 1, 2]
    assert len({item["sha256"] for item in output["artifacts"]}) == 3
    assert isinstance(output["seed"], int)


async def test_a_regenerate_given_a_batch_size_asks_the_engine_for_one_picture(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regenerate makes one output, so the engine is asked for one picture and makes one."""

    seen = _seeded_engine(app, monkeypatch)
    first = await _turn(client, "Create an image of a blue ceramic apple")
    await _finished_plan(client, first["run"]["work_plan_id"])

    regenerated = await client.post(
        f"/api/messages/{first['run']['assistant_message_id']}/regenerate",
        json={"settings": {"batch_size": 3}},
    )
    assert regenerated.status_code == 202, regenerated.text
    plan = await _finished_plan(client, regenerated.json()["run"]["work_plan_id"])
    [output] = await _outputs(client, plan["id"])

    assert seen[-1].parameters["batch_size"] == 1
    assert output["batch_size"] == 1
    assert [item["batch_index"] for item in output["artifacts"]] == [0]


async def test_a_missing_plan_or_one_in_a_comparison_is_not_found(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = await client.get("/api/work-plans/plan_missing/outputs")
    assert missing.status_code == 404
    assert missing.json()["code"] == "work-plan-not-found"

    _seeded_engine(app, monkeypatch)
    accepted = await _turn(client, "Create an image of a blue ceramic apple")
    plan = await _finished_plan(client, accepted["run"]["work_plan_id"])
    with SessionLocal() as session:
        chat = session.get(Chat, plan["chat_id"])
        assert chat is not None
        chat.scope = EXPERIMENT_CHAT_SCOPE
        session.commit()

    hidden = await client.get(f"/api/work-plans/{plan['id']}/outputs")
    assert hidden.status_code == 404
    assert hidden.json()["code"] == "work-plan-not-found"
