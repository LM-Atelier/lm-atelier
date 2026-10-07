from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from datetime import timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS, wait_for_terminal_status, wait_until
from sqlalchemy import select

from local_lm.adapters.base import ChatEvent, ChatRequest, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import AppSetting, Chat, Job, Message, Run
from local_lm.scheduler import JobClaim

POLICY_KEY = "generation_failure_retries"


async def read_run(client: AsyncClient, run_id: str) -> dict[str, Any]:
    response = await client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    run = response.json()
    assert isinstance(run, dict)
    return run


async def choose_retries(client: AsyncClient, count: int) -> None:
    policy = await client.get("/api/settings/generation-retries")
    assert policy.status_code == 200
    response = await client.put(
        "/api/settings/generation-retries",
        json={"max_retries": count, "expected_revision": policy.json()["revision"]},
    )
    assert response.status_code == 200


async def start_media(client: AsyncClient, mode: str = "image") -> str:
    chat = await client.post("/api/chats", json={"title": "Retry study"})
    assert chat.status_code == 201
    turn = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={"text": "A blue geometric square", "mode": mode, "settings": {"seed": 17}},
    )
    assert turn.status_code == 202, turn.text
    run_id = turn.json()["run"]["id"]
    assert isinstance(run_id, str)
    return run_id


def failing_media(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, failures: int
) -> list[MediaRequest]:
    adapter = app.state.services.engines.media
    original = adapter.generate
    requests: list[MediaRequest] = []

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        requests.append(deepcopy(request))
        if len(requests) <= failures:
            raise RuntimeError("Temporary media engine failure")
        async for event in original(request):
            yield event

    monkeypatch.setattr(adapter, "generate", generate)
    return requests


async def test_retry_count_starts_at_three_and_refuses_a_stale_settings_write(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/settings/generation-retries")
    assert response.json() == {"max_retries": 3, "revision": 0}
    await choose_retries(client, 0)
    stale = await client.put(
        "/api/settings/generation-retries", json={"max_retries": 7, "expected_revision": 0}
    )
    assert stale.status_code == 409
    assert (await client.get("/api/settings/generation-retries")).json() == {
        "max_retries": 0,
        "revision": 1,
    }


@pytest.mark.parametrize("count", [-1, 11, True, 1.5, "3"])
async def test_retry_settings_accept_only_a_bounded_whole_number(
    client: AsyncClient, count: object
) -> None:
    response = await client.put(
        "/api/settings/generation-retries", json={"max_retries": count, "expected_revision": 0}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("mode", ["image", "video"])
async def test_an_image_or_video_succeeds_with_its_original_settings_after_a_failure(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    requests = failing_media(app, monkeypatch, 1)
    run_id = await start_media(client, mode)
    result = await wait_for_terminal_status(
        lambda: read_run(client, run_id), what="automatically retried generation"
    )
    assert len(requests) == 2
    assert requests[0].run_id == requests[1].run_id == run_id
    assert requests[0].parameters == requests[1].parameters
    assert requests[0].workflow == requests[1].workflow
    assert result["provenance_json"]["failure_retries"] == {"limit": 3, "used": 1, "pending": False}
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        assert job is not None and job.attempt == 2 and job.status == "complete"


@pytest.mark.parametrize(
    ("mode", "operation"), [("image", "image_to_image"), ("video", "image_to_video")]
)
async def test_an_edit_or_animation_retry_keeps_the_accepted_source_picture(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    operation: str,
) -> None:
    source_run_id = await start_media(client)
    source = await wait_for_terminal_status(
        lambda: read_run(client, source_run_id), what="source picture for a retry"
    )
    source_message = (await client.get(f"/api/messages/{source['assistant_message_id']}")).json()
    artifact_id = next(
        part["artifact_id"] for part in source_message["parts"] if part["type"] == "image"
    )
    requests = failing_media(app, monkeypatch, 1)
    response = await client.post(
        f"/api/chats/{source['chat_id']}/turns",
        json={
            "text": "Make the square green" if mode == "image" else "Animate the square gently",
            "mode": mode,
            "input_artifact_ids": [artifact_id],
            "settings": {"seed": 23},
        },
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["run"]["id"]
    result = await wait_for_terminal_status(
        lambda: read_run(client, run_id), what="retried generation from a source picture"
    )
    assert result["operation"] == operation
    assert len(requests) == 2
    first, second = requests
    assert first.run_id == second.run_id == run_id
    assert first.operation == second.operation == operation
    assert first.input_paths and first.input_paths == second.input_paths
    assert first.input_contents == second.input_contents
    assert first.parameters == second.parameters and first.workflow == second.workflow
    assert result["provenance_json"]["input_artifact_ids"] == [artifact_id]
    assert result["provenance_json"]["failure_retries"] == {"limit": 3, "used": 1, "pending": False}


async def test_three_additional_retries_stop_after_four_failed_attempts(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = failing_media(app, monkeypatch, 99)
    run_id = await start_media(client)
    result = await wait_for_terminal_status(
        lambda: read_run(client, run_id), what="exhausted generation", expected="failed"
    )
    assert len(requests) == 4
    assert result["provenance_json"]["failure_retries"] == {"limit": 3, "used": 3, "pending": False}
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        assert job is not None and job.attempt == 4
        assert job.error == "Temporary media engine failure"


async def test_zero_disables_automatic_retries(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await choose_retries(client, 0)
    requests = failing_media(app, monkeypatch, 99)
    run_id = await start_media(client)
    await wait_for_terminal_status(
        lambda: read_run(client, run_id), what="generation with retries disabled", expected="failed"
    )
    assert len(requests) == 1


async def test_chat_failures_never_use_the_media_retry_allowance(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = app.state.services.engines.chat
    original = adapter.stream
    calls: list[str] = []

    async def stream(request: ChatRequest) -> AsyncIterator[ChatEvent]:
        calls.append(request.run_id)
        if calls:
            raise RuntimeError("Temporary chat engine failure")
        async for event in original(request):
            yield event

    monkeypatch.setattr(adapter, "stream", stream)
    chat = await client.post("/api/chats", json={"title": "Chat retry preservation"})
    response = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={"text": "Describe a blue square", "mode": "text"},
    )
    assert response.status_code == 202
    run_id = response.json()["run"]["id"]
    await wait_for_terminal_status(
        lambda: read_run(client, run_id), what="failed chat response", expected="failed"
    )
    assert calls == [run_id]


@pytest.mark.parametrize("mode", ["image", "video"])
async def test_cancelling_a_reserved_retry_prevents_the_next_attempt(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    requests = failing_media(app, monkeypatch, 99)
    run_id = await start_media(client, mode)

    async def reserved() -> bool:
        run = await read_run(client, run_id)
        pending = run["status"] == "queued" and run["provenance_json"]["failure_retries"]["pending"]
        assert isinstance(pending, bool)
        return pending

    await wait_until(reserved, bool, what="reserved automatic retry")
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        assert job is not None
        job_id = job.id
    response = await client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 200
    await wait_for_terminal_status(
        lambda: read_run(client, run_id), what="cancelled retry", expected="cancelled"
    )
    await asyncio.sleep(0)
    assert len(requests) == 1


def test_missing_chat_and_exhausted_budgets_cannot_reserve_another_attempt() -> None:
    from local_lm.generation_retry import reserve_retry, retry_is_pending

    assert reserve_retry({}, "text_to_image") is None
    assert (
        reserve_retry({"failure_retries": {"limit": 3, "used": 0, "pending": False}}, "text")
        is None
    )
    assert (
        reserve_retry(
            {"failure_retries": {"limit": 3, "used": 3, "pending": False}}, "text_to_video"
        )
        is None
    )
    pending = {"failure_retries": {"limit": 3, "used": 2, "pending": True}}
    assert retry_is_pending(pending)
    assert reserve_retry(pending, "text_to_image") is None


async def test_changing_the_setting_does_not_replenish_an_existing_run(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await choose_retries(client, 1)
    requests = failing_media(app, monkeypatch, 99)
    run_id = await start_media(client)
    await choose_retries(client, 10)
    result = await wait_for_terminal_status(
        lambda: read_run(client, run_id),
        what="generation with captured retry allowance",
        expected="failed",
    )
    assert len(requests) == 2
    assert result["provenance_json"]["failure_retries"]["limit"] == 1
    with SessionLocal() as session:
        stored = session.get(Run, run_id)
        assert stored is not None and stored.settings_json["seed"] == 17


@pytest.mark.parametrize(
    "saved",
    [
        {"max_retries": 3, "revision": 0},
        {"max_retries": True, "revision": 1},
        {"max_retries": 11, "revision": 1},
    ],
)
async def test_invalid_saved_retry_policy_is_refused_without_echoing_its_value(
    client: AsyncClient, saved: dict[str, object]
) -> None:
    with SessionLocal() as session:
        session.add(AppSetting(key=POLICY_KEY, value_json=saved))
        session.commit()
    response = await client.get("/api/settings/generation-retries")
    assert response.status_code == 409
    assert response.json()["code"] == "generation-retry-setting-invalid"


@pytest.mark.parametrize("mode", ["image", "video"])
async def test_cancelling_a_running_generation_never_reserves_a_retry(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()
    calls: list[str] = []

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        calls.append(request.run_id)
        started.set()
        await stopped.wait()
        raise RuntimeError("Temporary media engine failure after cancellation")
        yield

    async def cancel(_run_id: str) -> None:
        stopped.set()

    monkeypatch.setattr(app.state.services.engines.media, "generate", generate)
    monkeypatch.setattr(app.state.services.engines.media, "cancel", cancel)
    run_id = await start_media(client, mode)
    await asyncio.wait_for(started.wait(), timeout=PATIENCE_SECONDS)
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        assert job is not None
        job_id = job.id
    assert (await client.post(f"/api/jobs/{job_id}/cancel")).status_code == 200
    result = await wait_for_terminal_status(
        lambda: read_run(client, run_id), what="cancelled running generation", expected="cancelled"
    )
    assert calls == [run_id]
    assert result["provenance_json"]["failure_retries"] == {"limit": 3, "used": 0, "pending": False}


def seed_retry_run(*, queued: bool = False) -> tuple[str, str]:
    with SessionLocal() as session:
        chat = Chat(title="Retry ownership study")
        session.add(chat)
        session.flush()
        user = Message(chat_id=chat.id, role="user")
        assistant = Message(chat_id=chat.id, role="assistant", status="pending")
        session.add_all([user, assistant])
        session.flush()
        run = Run(
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            operation="text_to_image",
            status="queued" if queued else "running",
            provenance_json={
                "failure_retries": {"limit": 3, "used": 1 if queued else 0, "pending": queued}
            },
        )
        session.add(run)
        session.flush()
        job = Job(
            run_id=run.id,
            kind="image",
            status=run.status,
            attempt=2,
            claim_owner=None if queued else "current-attempt",
            claim_expires_at=None if queued else utcnow() + timedelta(minutes=5),
        )
        session.add(job)
        session.commit()
        return job.id, run.id


async def test_a_stale_failure_cannot_spend_the_current_generations_retry_budget(
    app: FastAPI, client: AsyncClient
) -> None:
    job_id, run_id = seed_retry_run()
    await app.state.services.orchestrator._fail(
        job_id,
        run_id,
        "Neutral superseded failure",
        claim=JobClaim(token="expired-attempt", attempt=1),
    )
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        run = session.get(Run, run_id)
        assert job is not None and run is not None
        assert job.status == run.status == "running"
        assert job.claim_owner == "current-attempt" and job.attempt == 2
        assert run.provenance_json["failure_retries"] == {"limit": 3, "used": 0, "pending": False}

    # The current failure can finish truthfully even though this run has lost its work step.
    await app.state.services.orchestrator._fail(
        job_id,
        run_id,
        "Neutral failure without a work step",
        claim=JobClaim(token="current-attempt", attempt=2),
    )
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        run = session.get(Run, run_id)
        assert job is not None and run is not None
        assert run.work_step_id is None and job.work_step_id is None
        assert job.status == run.status == "failed" and job.attempt == 2
        assert job.error == run.error == "Neutral failure without a work step"
        assert job.completed_at is not None and run.completed_at is not None
        assert run.provenance_json["failure_retries"] == {"limit": 3, "used": 0, "pending": False}
        message = session.get(Message, run.assistant_message_id)
        assert message is not None and message.status == "failed"
        assert [(part.type, part.text) for part in message.parts] == [
            ("error", "Neutral failure without a work step")
        ]
        assert len(session.scalars(select(Job).where(Job.run_id == run_id)).all()) == 1


async def test_queued_restart_recovery_preserves_the_reserved_allowance(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, run_id = seed_retry_run(queued=True)
    dispatched: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app.state.services.orchestrator, "start", lambda job, run: dispatched.append((job, run))
    )
    await choose_retries(client, 10)
    app.state.services.orchestrator.recover_interrupted()
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        job = session.get(Job, job_id)
        assert run is not None and job is not None
        assert job.status == run.status == "queued" and job.attempt == 2
        assert run.provenance_json["failure_retries"] == {"limit": 3, "used": 1, "pending": True}
    assert dispatched == [(job_id, run_id)]


async def test_restart_interrupts_running_media_without_spending_another_attempt(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, run_id = seed_retry_run()
    dispatched: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app.state.services.orchestrator, "start", lambda job, run: dispatched.append((job, run))
    )
    app.state.services.orchestrator.recover_interrupted()
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        job = session.get(Job, job_id)
        assert run is not None and job is not None
        assert job.status == "interrupted" and run.status == "failed"
        assert job.attempt == 2
        assert run.provenance_json["failure_retries"] == {"limit": 3, "used": 0, "pending": False}
    assert dispatched == []


async def test_an_ordered_media_retry_preserves_its_predecessor_and_unblocks_its_successor(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = failing_media(app, monkeypatch, 1)
    chat = (await client.post("/api/chats", json={"title": "Retry in an ordered study"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": (
                "Write a short scene, then create an image based on it, then describe the image"
            ),
            "mode": "auto",
            "confirm_media": True,
        },
    )
    assert response.status_code == 202, response.text
    plan_id = response.json()["run"]["work_plan_id"]

    async def read_plan() -> dict[str, Any]:
        plan = (await client.get(f"/api/work-plans/{plan_id}")).json()
        assert isinstance(plan, dict)
        return plan

    result = await wait_for_terminal_status(read_plan, what="ordered media retry")
    assert result["summary_json"]["status_counts"] == {"complete": 3}
    assert len(requests) == 2 and requests[0].run_id == requests[1].run_id
    with SessionLocal() as session:
        runs = [session.get(Run, step["run_id"]) for step in result["steps"]]
        assert all(run is not None for run in runs)
        budgets = [run.provenance_json["failure_retries"] for run in runs if run is not None]
        assert budgets == [
            {"limit": 0, "used": 0, "pending": False},
            {"limit": 3, "used": 1, "pending": False},
            {"limit": 0, "used": 0, "pending": False},
        ]
        jobs = [
            session.scalar(select(Job).where(Job.run_id == step["run_id"]))
            for step in result["steps"]
        ]
        assert [job.attempt for job in jobs if job is not None] == [1, 2, 1]
