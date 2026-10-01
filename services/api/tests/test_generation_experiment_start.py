"""Starting an accepted comparison queues each choice's picture as independent work."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError
from test_generation_experiment_preflight import (
    PREFLIGHT,
    _arm,
    _bound_revision,
    _profile,
    _request,
    _revision,
)
from test_generation_experiment_records import CREATE
from test_generation_retry import choose_retries, failing_media

from local_lm import generation_experiment_start as start_module
from local_lm.accepted_turn_context import accepted_context
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.auxiliary_assets import checkpoint_lora_extension
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import (
    Chat,
    ChatItemRemovalReceipt,
    ChatWorkflowSelection,
    CustomNodeInstall,
    GenerationExperiment,
    Job,
    Message,
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
    ResponseFeedback,
    Run,
    RunContextSnapshot,
    WorkflowActivation,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)

COUNTED = (
    Chat,
    Message,
    WorkPlan,
    WorkStep,
    Run,
    Job,
    RunContextSnapshot,
    ResponseFeedback,
    ChatWorkflowSelection,
    ChatItemRemovalReceipt,
)


def _counts() -> dict[str, int]:
    with SessionLocal() as session:
        return {
            model.__name__: int(session.scalar(select(func.count()).select_from(model)) or 0)
            for model in COUNTED
        }


def _choices(prompt: str, **values: Any) -> dict[str, Any]:
    first = _arm("Fewer steps", _profile(f"{prompt} one"), _revision(f"{prompt} one"), steps=8)
    second = _arm("More steps", _profile(f"{prompt} two"), _revision(f"{prompt} two"), steps=20)
    return _request(first, second, prompt=prompt, **values)


async def _accepted(
    client: AsyncClient, body: dict[str, Any], key: str = "accept"
) -> dict[str, Any]:
    checked = (await client.post(PREFLIGHT, json=body)).json()
    assert checked["outcome"] == "compatible", checked["refusals"]
    created = await client.post(
        CREATE,
        json={**body, "idempotency_key": key, "preflight_sha256": checked["preflight_sha256"]},
    )
    assert created.status_code == 201, created.text
    accepted: dict[str, Any] = created.json()
    return accepted


async def _start(
    client: AsyncClient, accepted: dict[str, Any], key: str = "start", **values: Any
) -> Any:
    return await client.post(
        f"{CREATE}/{accepted['id']}/start",
        json={"idempotency_key": key, "snapshot_sha256": accepted["snapshot_sha256"], **values},
    )


def _recording(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[MediaRequest]:
    seen: list[MediaRequest] = []
    original = app.state.services.engines.media.generate

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        seen.append(request)
        async for event in original(request):
            yield event

    monkeypatch.setattr(app.state.services.engines.media, "generate", generate)
    return seen


async def _terminal(client: AsyncClient, run_id: str) -> dict[str, Any]:
    async def read() -> dict[str, Any]:
        payload: dict[str, Any] = (await client.get(f"/api/runs/{run_id}")).json()
        return payload

    return cast(
        dict[str, Any], await wait_for_terminal_status(read, what=f"run {run_id}", expected=None)
    )


def _run_ids(started: dict[str, Any]) -> list[str]:
    return [trial["run_id"] for arm in started["arms"] for trial in arm["trials"]]


async def test_starting_queues_one_independent_hidden_picture_per_choice(
    app: FastAPI, client: AsyncClient
) -> None:
    accepted = await _accepted(client, _choices("Quiet harbor at dawn"))
    before = _counts()
    async with app.state.services.scheduler.lease("primary"):
        response = await _start(client, accepted)
        assert response.status_code == 202, response.text
        started = response.json()
        assert started["state"] == "started" and started["work_plan_id"]
        trials = [trial for arm in started["arms"] for trial in arm["trials"]]
        assert [trial["state"] for trial in trials] == ["started", "started"]
        assert all(
            trial["work_step_id"] and trial["run_id"] and trial["job_id"] for trial in trials
        )
        assert [trial["status"] for trial in trials] == ["queued", "queued"]
        after = _counts()
        assert {name: after[name] - before[name] for name in after} == {
            "Chat": 1,
            "Message": 3,
            "WorkPlan": 1,
            "WorkStep": 2,
            "Run": 2,
            "Job": 2,
            "RunContextSnapshot": 2,
            "ResponseFeedback": 0,
            "ChatWorkflowSelection": 0,
            "ChatItemRemovalReceipt": 0,
        }
        with SessionLocal() as session:
            plan = session.get(WorkPlan, started["work_plan_id"])
            assert plan is not None
            assert (plan.source_action, plan.failure_policy) == (
                "generation_experiment",
                "continue_independent",
            )
            chat = session.get(Chat, plan.chat_id)
            assert chat is not None
            assert (chat.scope, chat.archived, chat.project_id) == ("experiment", True, None)
            messages = session.scalars(select(Message).where(Message.chat_id == chat.id)).all()
            assert not any(message.transcript_visible for message in messages)
            steps = session.scalars(select(WorkStep).where(WorkStep.plan_id == plan.id)).all()
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(WorkStepDependency)
                    .where(WorkStepDependency.step_id.in_([step.id for step in steps]))
                )
                == 0
            )
            jobs = session.scalars(select(Job).where(Job.work_plan_id == plan.id)).all()
            assert {(job.kind, job.queue_resource) for job in jobs} == {("image", "media_compute")}
            for run_id in _run_ids(started):
                run = session.get(Run, run_id)
                assert run is not None
                context = accepted_context(session, run)
                assert context is not None and context.messages == []
    read = await client.get(f"{CREATE}/{accepted['id']}")
    assert read.status_code == 200 and read.json()["state"] == "started"


async def test_each_picture_runs_exactly_its_accepted_choice(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recording(app, monkeypatch)
    body = _choices(
        "A lighthouse on a cliff", seed_policy={"kind": "independent_deterministic", "seed": 4}
    )
    accepted = await _accepted(client, body)
    assert [arm["trials"][0]["seed"] for arm in accepted["arms"]] == [4, 5]
    started = (await _start(client, accepted)).json()
    for run_id in _run_ids(started):
        assert (await _terminal(client, run_id))["status"] == "complete"
    by_run = {request.run_id: request for request in seen}
    for arm in accepted["arms"]:
        trial = arm["trials"][0]
        run_id = next(
            item["run_id"]
            for started_arm in started["arms"]
            for item in started_arm["trials"]
            if item["id"] == trial["id"]
        )
        request = by_run[run_id]
        assert request.parameters == {**arm["effective_settings"], "seed": trial["seed"]}
        assert request.prompt == "A lighthouse on a cliff"
    assert sorted(request.parameters["seed"] for request in seen) == [4, 5]
    assert (await client.get(f"{CREATE}/{accepted['id']}")).status_code == 200


async def test_a_picture_that_fails_is_retried_as_a_turns_would_be_and_its_sibling_is_not(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = failing_media(app, monkeypatch, 1)
    accepted = await _accepted(client, _choices("A harbor in fog"))
    started = (await _start(client, accepted)).json()

    finished = [await _terminal(client, run_id) for run_id in _run_ids(started)]

    assert [run["status"] for run in finished] == ["complete", "complete"]
    # Whichever picture was asked for first failed once and was made again,
    # within the allowance frozen when the comparison started.
    retried = {run["id"]: run["provenance_json"]["failure_retries"] for run in finished}
    assert sorted(retry["used"] for retry in retried.values()) == [0, 1]
    assert all(retry["limit"] == 3 and retry["pending"] is False for retry in retried.values())
    failed_first = attempts[0].run_id
    assert retried[failed_first]["used"] == 1
    assert [request.run_id for request in attempts].count(failed_first) == 2
    with SessionLocal() as session:
        jobs = {job.run_id: job.attempt for job in session.scalars(select(Job))}
    assert sorted(jobs[run_id] for run_id in retried) == [1, 2]


async def test_a_picture_runs_as_an_ordinary_turn_with_the_same_choice_would(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recording(app, monkeypatch)
    body = _choices("Three red apples")
    accepted = await _accepted(client, body)
    started = (await _start(client, accepted)).json()
    first_run = _run_ids(started)[0]
    await _terminal(client, first_run)
    arm = accepted["arms"][0]
    chat = (await client.post("/api/chats", json={"title": "Same choice"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Three red apples",
            "mode": "image",
            "profile_id": arm["profile_id"],
            "workflow_revision_id": arm["workflow_revision_id"],
            "preset_id": None,
            "settings": {**arm["effective_settings"], "seed": arm["trials"][0]["seed"]},
        },
    )
    assert turn.status_code == 202, turn.text
    await _terminal(client, turn.json()["run"]["id"])
    by_run = {request.run_id: request for request in seen}
    compared, ordinary = by_run[first_run], by_run[turn.json()["run"]["id"]]
    assert compared.prompt == ordinary.prompt
    assert compared.negative_prompt == ordinary.negative_prompt
    assert compared.parameters == ordinary.parameters
    assert compared.workflow == ordinary.workflow


async def test_one_picture_failing_leaves_the_other_to_finish(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = await _accepted(client, _choices("A paper boat on a pond"))
    failing: set[str] = set()
    original = app.state.services.engines.media.generate

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        if request.run_id in failing:
            raise RuntimeError("The neutral engine refused this picture.")
        async for event in original(request):
            yield event

    monkeypatch.setattr(app.state.services.engines.media, "generate", generate)
    async with app.state.services.scheduler.lease("primary"):
        started = (await _start(client, accepted)).json()
        first, second = _run_ids(started)
        failing.add(first)
    assert (await _terminal(client, first))["status"] == "failed"
    assert (await _terminal(client, second))["status"] == "complete"
    read = (await client.get(f"{CREATE}/{accepted['id']}")).json()
    assert [trial["status"] for arm in read["arms"] for trial in arm["trials"]] == [
        "failed",
        "complete",
    ]


async def test_a_start_retried_with_its_key_returns_the_started_comparison(
    app: FastAPI, client: AsyncClient
) -> None:
    accepted = await _accepted(client, _choices("A small green kettle"))
    async with app.state.services.scheduler.lease("primary"):
        first = await _start(client, accepted)
        assert first.status_code == 202
        before = _counts()
        replay = await _start(client, accepted)
        assert replay.status_code == 200 and replay.json() == first.json()
        other_key = await _start(client, accepted, key="another")
        assert other_key.status_code == 409
        assert other_key.json()["code"] == "generation-experiment-already-started"
        changed = await _start(client, accepted, confirm_expensive=True)
        assert changed.status_code == 409
        assert changed.json()["code"] == "generation-experiment-idempotency-conflict"
        assert _counts() == before


async def test_two_starts_at_once_queue_the_work_once(app: FastAPI, client: AsyncClient) -> None:
    accepted = await _accepted(client, _choices("A wooden chair"))
    before = _counts()
    async with app.state.services.scheduler.lease("primary"):
        responses = await asyncio.gather(*(_start(client, accepted) for _ in range(2)))
        assert sorted(response.status_code for response in responses) == [200, 202]
        after = _counts()
    assert after["WorkPlan"] - before["WorkPlan"] == 1
    assert after["Run"] - before["Run"] == 2


def _change_the_first_choice(accepted: dict[str, Any], change: str) -> None:
    arm = accepted["arms"][0]
    with SessionLocal() as session:
        profile = session.get(ModelProfile, arm["profile_id"])
        assert profile is not None
        if change == "model-settings":
            profile.request_settings_json = {"cfg": 3.5}
        elif change == "model-role":
            profile.role = "video"
        else:
            raise AssertionError(change)
        session.commit()


@pytest.mark.parametrize(
    ("change", "code"),
    [("model-settings", "arm-changed"), ("model-role", "arm-profile-unavailable")],
)
async def test_a_choice_changed_since_acceptance_starts_nothing(
    app: FastAPI, client: AsyncClient, change: str, code: str
) -> None:
    accepted = await _accepted(client, _choices("A blue umbrella"))
    _change_the_first_choice(accepted, change)
    before = _counts()
    response = await _start(client, accepted)
    assert response.status_code == 409, response.text
    body = response.json()
    assert body["code"] == "generation-experiment-preflight-changed"
    assert [(item["code"], item["arm_ordinal"]) for item in body["refusals"]] == [(code, 1)]
    assert _counts() == before
    with SessionLocal() as session:
        experiment = session.get(GenerationExperiment, accepted["id"])
        assert experiment is not None and experiment.state == "ready"


async def test_a_start_aimed_at_a_changed_record_starts_nothing(client: AsyncClient) -> None:
    accepted = await _accepted(client, _choices("A stack of books"))
    before = _counts()
    stale = {**accepted, "snapshot_sha256": "0" * 64}
    response = await _start(client, stale)
    assert response.status_code == 409
    assert response.json()["code"] == "generation-experiment-snapshot-changed"
    assert _counts() == before


async def test_starting_checks_size_and_asks_for_confirmation_when_heavy(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = await _accepted(client, _choices("A glass of water"))
    settings = app.state.services.settings
    monkeypatch.setattr(settings, "video_confirmation_work_units", 1)
    before = _counts()
    asked = await _start(client, accepted)
    assert asked.status_code == 409
    assert asked.json()["code"] == "generation-experiment-confirmation-required"
    assert [item["resource"] for item in asked.json()["estimate"]] == [
        "work_units",
        "output_bytes",
    ]
    assert _counts() == before
    for limit in ("max_media_plan_work_units", "max_media_plan_estimated_bytes"):
        monkeypatch.setattr(settings, limit, 1)
        too_large = await _start(client, accepted, confirm_expensive=True)
        assert too_large.status_code == 422
        assert [item["code"] for item in too_large.json()["refusals"]] == ["experiment-too-large"]
        assert _counts() == before
        monkeypatch.setattr(settings, limit, 10**12)
    async with app.state.services.scheduler.lease("primary"):
        confirmed = await _start(client, accepted, confirm_expensive=True)
        assert confirmed.status_code == 202, confirmed.text


async def test_comparison_work_is_never_listed_as_a_chat(app: FastAPI, client: AsyncClient) -> None:
    accepted = await _accepted(client, _choices("A single yellow leaf"))
    async with app.state.services.scheduler.lease("primary"):
        started = (await _start(client, accepted)).json()
        listed = (await client.get("/api/chats", params={"include_archived": "true"})).json()
        with SessionLocal() as session:
            plan = session.get(WorkPlan, started["work_plan_id"])
            assert plan is not None
            chat_id = plan.chat_id
        assert chat_id not in {chat["id"] for chat in listed}
        assert (await client.get(f"/api/chats/{chat_id}")).status_code == 404


async def test_reading_a_started_comparison_checks_each_picture_link(
    app: FastAPI, client: AsyncClient
) -> None:
    accepted = await _accepted(client, _choices("A white teapot"))
    async with app.state.services.scheduler.lease("primary"):
        started = (await _start(client, accepted)).json()
        with SessionLocal() as session:
            run = session.get(Run, _run_ids(started)[0])
            assert run is not None
            run.settings_json = {**run.settings_json, "seed": run.settings_json["seed"] + 1}
            session.commit()
        read = await client.get(f"{CREATE}/{accepted['id']}")
        assert read.status_code == 409
        assert read.json()["code"] == "generation-experiment-record-invalid"


async def test_the_comparison_chat_takes_no_turns_and_no_chat_wide_stop(
    app: FastAPI, client: AsyncClient
) -> None:
    accepted = await _accepted(client, _choices("A red brick wall"))
    async with app.state.services.scheduler.lease("primary"):
        started = (await _start(client, accepted)).json()
        with SessionLocal() as session:
            plan = session.get(WorkPlan, started["work_plan_id"])
            assert plan is not None
            chat_id = plan.chat_id
            user_message_id = plan.summary_json["user_message_id"]
        before = _counts()
        attempts = [
            await client.post(
                f"/api/chats/{chat_id}/turns", json={"text": "Something else", "mode": "image"}
            ),
            await client.post(f"/api/chats/{chat_id}/cancel"),
            await client.post(
                f"/api/chats/{chat_id}/stop-and-send",
                json={"text": "Something else", "mode": "image"},
            ),
            await client.delete(f"/api/messages/{user_message_id}/exchange"),
        ]
        assert [response.status_code for response in attempts] == [404, 404, 404, 404]
        assert [response.json()["code"] for response in attempts] == [
            "chat-not-found",
            "chat-not-found",
            "chat-not-found",
            "exchange-not-found",
        ]
        assert _counts() == before
        read = (await client.get(f"{CREATE}/{accepted['id']}")).json()
        assert [trial["status"] for arm in read["arms"] for trial in arm["trials"]] == [
            "queued",
            "queued",
        ]


def _step_ids(started: dict[str, Any]) -> list[str]:
    return [trial["work_step_id"] for arm in started["arms"] for trial in arm["trials"]]


async def test_stopping_one_picture_leaves_the_other(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recording(app, monkeypatch)
    accepted = await _accepted(client, _choices("A folded map"))
    async with app.state.services.scheduler.lease("primary"):
        started = (await _start(client, accepted)).json()
        first_step = _step_ids(started)[0]
        stopped = await client.post(f"/api/work-steps/{first_step}/cancel")
        assert stopped.status_code == 200, stopped.text
    first, second = _run_ids(started)
    assert (await _terminal(client, first))["status"] == "cancelled"
    assert (await _terminal(client, second))["status"] == "complete"
    assert [request.run_id for request in seen] == [second]


async def test_retrying_one_picture_reruns_only_it_with_its_seed(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With no automatic retries, the failed picture waits to be retried by hand.
    await choose_retries(client, 0)
    accepted = await _accepted(client, _choices("A row of candles"))
    failing: set[str] = set()
    seen: list[MediaRequest] = []
    original = app.state.services.engines.media.generate

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        seen.append(request)
        if request.run_id in failing:
            failing.discard(request.run_id)
            raise RuntimeError("The neutral engine refused this picture once.")
        async for event in original(request):
            yield event

    monkeypatch.setattr(app.state.services.engines.media, "generate", generate)
    async with app.state.services.scheduler.lease("primary"):
        started = (await _start(client, accepted)).json()
        first, second = _run_ids(started)
        failing.add(first)
    assert (await _terminal(client, first))["status"] == "failed"
    assert (await _terminal(client, second))["status"] == "complete"
    with SessionLocal() as session:
        sibling = session.scalar(select(Job).where(Job.run_id == second))
        assert sibling is not None
        sibling_attempt = sibling.attempt
    retried = await client.post(f"/api/work-steps/{_step_ids(started)[0]}/retry")
    assert retried.status_code == 200, retried.text
    assert (await _terminal(client, first))["status"] == "complete"
    first_requests = [request for request in seen if request.run_id == first]
    assert len(first_requests) == 2
    assert first_requests[0].parameters == first_requests[1].parameters
    assert first_requests[1].parameters["seed"] == accepted["arms"][0]["trials"][0]["seed"]
    with SessionLocal() as session:
        sibling = session.scalar(select(Job).where(Job.run_id == second))
        assert sibling is not None and sibling.attempt == sibling_attempt
    read = await client.get(f"{CREATE}/{accepted['id']}")
    assert read.status_code == 200
    assert [trial["status"] for arm in read.json()["arms"] for trial in arm["trials"]] == [
        "complete",
        "complete",
    ]


async def test_a_choice_whose_ready_dependencies_changed_starts_nothing(
    client: AsyncClient,
) -> None:
    bound = _profile("Bound picture model")
    revision_id = _bound_revision(bound)
    first = _arm("Bound", bound, revision_id, steps=8)
    second = _arm("Plain", _profile("Plain picture model"), _revision("Plain pictures"), steps=20)
    accepted = await _accepted(client, _request(first, second, prompt="A copper bell"))
    with SessionLocal() as session:
        activation = session.scalar(
            select(WorkflowActivation).where(WorkflowActivation.workflow_revision_id == revision_id)
        )
        assert activation is not None
        activation.details_json = {**activation.details_json, "launch_sha256": "c" * 64}
        session.commit()
    before = _counts()
    response = await _start(client, accepted)
    assert response.status_code == 409, response.text
    assert [(item["code"], item["arm_ordinal"]) for item in response.json()["refusals"]] == [
        ("arm-changed", 1)
    ]
    assert _counts() == before


async def test_a_comparison_accepted_for_another_media_engine_starts_nothing(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = await _accepted(client, _choices("A tin of buttons"))
    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    before = _counts()
    response = await _start(client, accepted)
    assert response.status_code == 409, response.text
    assert [(item["code"], item["arm_ordinal"]) for item in response.json()["refusals"]] == [
        ("arm-workflow-unavailable", 1),
        ("arm-workflow-unavailable", 2),
    ]
    assert _counts() == before


def _still_ready(accepted: dict[str, Any]) -> None:
    with SessionLocal() as session:
        experiment = session.get(GenerationExperiment, accepted["id"])
        assert experiment is not None and experiment.state == "ready"


async def test_the_comparison_chat_and_its_messages_are_not_reached_by_id(
    app: FastAPI, client: AsyncClient
) -> None:
    accepted = await _accepted(client, _choices("A green glass bottle"))
    async with app.state.services.scheduler.lease("primary"):
        started = (await _start(client, accepted)).json()
        with SessionLocal() as session:
            run = session.get(Run, _run_ids(started)[0])
            assert run is not None and run.assistant_message_id is not None
            chat_id, request_id, answer_id = (
                run.chat_id,
                run.user_message_id,
                run.assistant_message_id,
            )
            chat = session.get(Chat, chat_id)
            assert chat is not None
            selected_before = chat.active_image_profile_id
        listed_before = (await client.get("/api/chats", params={"include_archived": "true"})).json()
        before = _counts()
        removal = {
            "expected_message_id": request_id,
            "expected_revision_id": "a" * 64,
            "operation_key": "comparison-removal",
        }
        draft = {"text": "Something else", "mode": "image"}
        attempts = {
            "read": await client.get(f"/api/messages/{answer_id}"),
            "feedback": await client.put(
                f"/api/messages/{answer_id}/feedback", json={"rating": "up"}
            ),
            "fork answer": await client.post(f"/api/messages/{answer_id}/fork"),
            "fork request": await client.post(f"/api/messages/{request_id}/fork"),
            "removal impact": await client.get(f"/api/messages/{request_id}/removal-impact"),
            "remove content": await client.post(
                f"/api/messages/{request_id}/remove-content", json=removal
            ),
            "selections": await client.get(f"/api/chats/{chat_id}/workflow-selections"),
            "select": await client.put(
                f"/api/chats/{chat_id}/workflow-selections/image", json={"mode": "default"}
            ),
            "classify": await client.post(f"/api/chats/{chat_id}/classify-draft", json=draft),
            "source fit": await client.post(f"/api/chats/{chat_id}/source-fit/preview", json=draft),
        }
        assert {
            name: (response.status_code, response.json()["code"])
            for name, response in attempts.items()
        } == {
            "read": (404, "message-not-found"),
            "feedback": (404, "message-not-found"),
            "fork answer": (404, "fork-source-not-found"),
            "fork request": (404, "fork-source-not-found"),
            "removal impact": (404, "message-not-found"),
            "remove content": (404, "message-not-found"),
            "selections": (404, "chat-not-found"),
            "select": (404, "chat-not-found"),
            "classify": (404, "chat-not-found"),
            "source fit": (404, "chat-not-found"),
        }
        assert _counts() == before
        listed = (await client.get("/api/chats", params={"include_archived": "true"})).json()
        assert listed == listed_before
        with SessionLocal() as session:
            chat = session.get(Chat, chat_id)
            assert chat is not None and chat.active_image_profile_id == selected_before
            messages = session.scalars(select(Message).where(Message.chat_id == chat_id)).all()
            assert messages and all(message.content_removed_at is None for message in messages)


async def test_a_started_picture_carries_its_trigger_words_as_an_ordinary_turn_would(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recording(app, monkeypatch)
    with SessionLocal() as session:
        install = ModelInstall(
            name="Worded start base",
            role="image",
            engine="mock",
            local_path="C:/managed/worded-start",
            manifest_json={"trigger_words": ["harborlight"]},
            active=True,
        )
        session.add(install)
        session.commit()
        install_id = install.id
    worded = _arm(
        "Worded",
        _profile("Worded start model", model_install_id=install_id),
        _revision("Worded start"),
    )
    plain = _arm("Plain", _profile("Plain start model"), _revision("Plain start"))
    accepted = await _accepted(client, _request(worded, plain, prompt="A quiet street"))
    assert [arm["trigger_words_applied"] for arm in accepted["arms"]] == [["harborlight"], []]
    started = (await _start(client, accepted)).json()
    worded_run, plain_run = _run_ids(started)
    for run_id in (worded_run, plain_run):
        assert (await _terminal(client, run_id))["status"] == "complete"
    arm = accepted["arms"][0]
    chat = (await client.post("/api/chats", json={"title": "Worded turn"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "A quiet street",
            "mode": "image",
            "profile_id": arm["profile_id"],
            "workflow_revision_id": arm["workflow_revision_id"],
            "preset_id": None,
            "settings": {**arm["effective_settings"], "seed": arm["trials"][0]["seed"]},
        },
    )
    assert turn.status_code == 202, turn.text
    ordinary_run = turn.json()["run"]["id"]
    await _terminal(client, ordinary_run)
    by_run = {request.run_id: request for request in seen}
    assert "harborlight" in by_run[worded_run].prompt
    assert by_run[worded_run].prompt == by_run[ordinary_run].prompt
    assert by_run[plain_run].prompt == "A quiet street"
    with SessionLocal() as session:
        run = session.get(Run, worded_run)
        assert run is not None
        assert run.provenance_json["auxiliary_assets"]["trigger_words_applied"] == ["harborlight"]


async def test_starting_without_room_for_the_pictures_starts_nothing(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = await _accepted(client, _choices("A pair of boots"))
    monkeypatch.setattr(
        start_module,
        "shutil",
        SimpleNamespace(disk_usage=lambda _root: SimpleNamespace(free=0)),
    )
    before = _counts()
    response = await _start(client, accepted)
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "generation-experiment-storage-insufficient"
    assert _counts() == before
    _still_ready(accepted)


async def test_a_start_while_the_app_is_closing_starts_nothing(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = await _accepted(client, _choices("A paper lantern"))
    monkeypatch.setattr(app.state.services.orchestrator, "_admission_open", False)
    before = _counts()
    response = await _start(client, accepted)
    assert response.status_code == 503, response.text
    assert response.json()["code"] == "generation-experiment-unavailable"
    assert _counts() == before
    _still_ready(accepted)


async def test_a_choice_whose_node_package_was_removed_starts_nothing(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        package = CustomNodeInstall(
            name="neutral-node-pack",
            source_url="https://example.invalid/neutral-node-pack",
            revision="a" * 40,
            installed_path="C:/managed/nodes/neutral-node-pack",
            tree_hash="b" * 64,
            trusted=True,
            active=True,
        )
        session.add(package)
        session.commit()
        package_id = package.id
    packaged = _revision("Packaged", dependencies_json={"custom_nodes": ["neutral-node-pack"]})
    first = _arm("Packaged", _profile("Packaged model"), packaged, steps=8)
    second = _arm("Plain", _profile("Unpackaged model"), _revision("Unpackaged"), steps=20)
    accepted = await _accepted(client, _request(first, second, prompt="A clay pot"))
    with SessionLocal() as session:
        session.execute(
            update(CustomNodeInstall).where(CustomNodeInstall.id == package_id).values(active=False)
        )
        session.commit()
    before = _counts()
    response = await _start(client, accepted)
    assert response.status_code == 409, response.text
    assert [(item["code"], item["arm_ordinal"]) for item in response.json()["refusals"]] == [
        ("arm-package-missing", 1)
    ]
    assert _counts() == before
    _still_ready(accepted)


def _lora_choice(name: str) -> tuple[dict[str, Any], str]:
    """A choice on a workflow that takes LoRAs, carrying one verified LoRA."""

    graph = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "neutral.safetensors"},
        },
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    extension = checkpoint_lora_extension(graph)
    assert extension is not None
    revision_id = _revision(
        name,
        api_graph_json=graph,
        input_schema_json={
            "type": "object",
            "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
        },
        dependencies_json={"extensions": {"lora": extension}},
    )
    with SessionLocal() as session:
        asset = ModelAssetInstall(
            name=f"{name} ink",
            kind="lora",
            family="sdxl",
            local_path=f"C:/managed/{name.lower().replace(' ', '-')}",
            size_bytes=1024,
            manifest_json={"sha256": "b" * 64, "comfy_name": "neutral-ink.safetensors"},
            active=True,
            verified_at=utcnow(),
        )
        session.add(asset)
        session.commit()
        asset_id = asset.id
    stack = [{"asset_id": asset_id, "model_strength": 0.8, "clip_strength": 0.65, "enabled": True}]
    return _arm(name, _profile(f"{name} model"), revision_id, loras=stack), asset_id


async def test_a_started_picture_runs_the_lora_stack_it_was_accepted_with(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recording(app, monkeypatch)
    inked, asset_id = _lora_choice("Inked")
    plain = _arm("Plain", _profile("Uninked model"), _revision("Uninked"))
    accepted = await _accepted(client, _request(inked, plain, prompt="A harbor sketch"))
    started = (await _start(client, accepted)).json()
    inked_run = _run_ids(started)[0]
    assert (await _terminal(client, inked_run))["status"] == "complete"
    with SessionLocal() as session:
        experiment = session.get(GenerationExperiment, accepted["id"])
        assert experiment is not None
        lora = experiment.arms[0].snapshot_json["lora"]
        assert lora["stack"] and lora["graph_sha256"]
        run = session.get(Run, inked_run)
        assert run is not None
        assets = run.provenance_json["auxiliary_assets"]
        assert assets["lora_stack"] == lora["stack"]
        assert assets["effective_graph_sha256"] == lora["graph_sha256"]
        context = accepted_context(session, run)
        assert context is not None and context.auxiliary_assets == assets
    request = next(request for request in seen if request.run_id == inked_run)
    assert [item["asset_id"] for item in request.parameters["loras"]] == [asset_id]


@pytest.mark.parametrize("change", ["file", "unavailable"])
async def test_a_choice_whose_lora_changed_starts_nothing(client: AsyncClient, change: str) -> None:
    inked, asset_id = _lora_choice(f"Changed {change}")
    plain = _arm("Plain", _profile(f"Plain {change} model"), _revision(f"Plain {change}"))
    accepted = await _accepted(client, _request(inked, plain, prompt="A harbor sketch"))
    with SessionLocal() as session:
        asset = session.get(ModelAssetInstall, asset_id)
        assert asset is not None
        if change == "file":
            asset.manifest_json = {**asset.manifest_json, "sha256": "c" * 64}
        else:
            asset.active = False
        session.commit()
    before = _counts()
    response = await _start(client, accepted)
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "generation-experiment-preflight-changed"
    assert [item["arm_ordinal"] for item in response.json()["refusals"]] == [1]
    assert _counts() == before
    _still_ready(accepted)


async def test_a_start_holds_the_write_lock_while_it_checks_each_choice(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another writer cannot change a choice between the start's checks and its commit."""

    accepted = await _accepted(client, _choices("A brass key"))
    checking, finished = threading.Event(), threading.Event()
    outcome: list[str] = []
    real_check = start_module._check_arm

    def writer() -> None:
        try:
            assert checking.wait(timeout=10)
            with SessionLocal() as session:
                session.connection().exec_driver_sql("PRAGMA busy_timeout=200")
                session.execute(
                    update(ModelProfile)
                    .where(ModelProfile.id == accepted["arms"][0]["profile_id"])
                    .values(request_settings_json={"cfg": 3.5})
                )
                session.commit()
            outcome.append("written")
        except OperationalError:
            outcome.append("locked")
        finally:
            finished.set()

    def check_while_another_writes(*args: Any) -> None:
        if not checking.is_set():
            checking.set()
            assert finished.wait(timeout=10)
        real_check(*args)

    monkeypatch.setattr(start_module, "_check_arm", check_while_another_writes)
    thread = threading.Thread(target=writer)
    thread.start()
    async with app.state.services.scheduler.lease("primary"):
        response = await _start(client, accepted)
    thread.join(timeout=10)
    assert outcome == ["locked"]
    assert response.status_code == 202, response.text
