"""Search consent is exercised through real chat jobs and API commands."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select, text

from local_lm import orchestrator as orchestrator_module
from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.db import SessionLocal
from local_lm.models import Job, Run, WebSearchProposal
from local_lm.web_search import CrwSearchProvider, SearchResults, search_crw

QUERY = "Compare copper and steel"
SOURCE = "https://materials.example.test/comparison"


@pytest_asyncio.fixture
async def execution(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    orch = app.state.services.orchestrator
    settings = orch.engines.settings
    settings.web_access_enabled = True
    settings.crw_endpoint = "https://search.example.test"
    settings.crw_token = "constructed-runtime-token"
    requests: list[ChatRequest] = []
    posts: list[dict[str, Any]] = []
    resumes: list[str] = []
    mode = {"timeout": False, "block": False}
    proposal = {"query": QUERY, "reason": "Current material data"}
    entered = asyncio.Event()
    release = asyncio.Event()
    interrupted = asyncio.Event()

    async def stream(self: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        requests.append(request)
        if request.tools:
            assert [tool["function"]["name"] for tool in request.tools] == ["search_web"]
            yield ChatEvent(
                type="tool_delta",
                data={
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "name": "search_web",
                                "arguments": json.dumps(proposal),
                            },
                        }
                    ]
                },
            )
        else:
            yield ChatEvent(type="delta", text="Here is the material comparison.")
            yield ChatEvent(type="complete")

    async def send(request: httpx.Request) -> httpx.Response:
        posts.append({"url": str(request.url), "body": json.loads(request.content)})
        # The real HTTP boundary must not retain a database writer.
        with SessionLocal() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            proposal = session.scalar(select(WebSearchProposal))
            assert proposal is not None and proposal.state == "dispatching"
            assert proposal.dispatch_owner is not None
            session.rollback()
        if mode["block"]:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                interrupted.set()
                raise
        if mode["timeout"]:
            raise httpx.ReadTimeout("constructed timeout", request=request)
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "results": [
                        {
                            "url": SOURCE,
                            "title": "Material comparison",
                            "snippet": "A neutral material summary.",
                        }
                    ]
                },
            },
        )

    async def search(provider: CrwSearchProvider, query: str) -> SearchResults:
        return await search_crw(provider, query, transport=httpx.MockTransport(send))

    monkeypatch.setattr(type(orch.engines.chat), "stream", stream)
    monkeypatch.setattr(type(orch), "start", lambda self, *args: None)
    monkeypatch.setattr(type(orch), "resume_search", lambda self, job_id: resumes.append(job_id))
    monkeypatch.setattr(orchestrator_module, "search_crw", search, raising=False)
    created = await client.post("/api/chats", json={"title": "Search execution"})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    updated = await client.patch(
        f"/api/chats/{chat_id}",
        json={"web_settings_json": {"allow_search": True, "allow_url_fetch": True}},
    )
    assert updated.status_code == 200
    return {
        "orch": orch,
        "settings": settings,
        "chat_id": chat_id,
        "client": client,
        "requests": requests,
        "posts": posts,
        "resumes": resumes,
        "mode": mode,
        "proposal": proposal,
        "entered": entered,
        "release": release,
        "interrupted": interrupted,
    }


async def accepted(case: dict[str, Any]) -> tuple[str, str]:
    response = await case["client"].post(
        f"/api/chats/{case['chat_id']}/turns",
        json={"text": QUERY, "mode": "text"},
    )
    assert response.status_code == 202
    run_id = response.json()["run"]["id"]
    with SessionLocal() as session:
        job_id = session.scalar(select(Job.id).where(Job.run_id == run_id))
        assert isinstance(job_id, str)
    return job_id, run_id


async def execute(case: dict[str, Any], job_id: str, run_id: str) -> None:
    async with asyncio.timeout(30):
        await case["orch"]._execute(job_id, run_id)


def pending(job_id: str) -> tuple[int, str]:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        proposal = session.scalar(
            select(WebSearchProposal).where(WebSearchProposal.job_id == job_id)
        )
        assert proposal is not None, "The real chat execution did not retain its proposed search."
        assert job.status == "paused" and job.claim_owner is None
        assert session.get(Run, job.run_id).status == "queued"
        return proposal.revision, proposal.state


async def decide(case: dict[str, Any], job_id: str, revision: int, action: str) -> None:
    result = await case["client"].post(
        f"/api/jobs/{job_id}/search/decision",
        json={"revision": revision, "action": action},
    )
    assert result.status_code == 200


def completed(case: dict[str, Any], job_id: str, state: str) -> None:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        run = session.get(Run, job.run_id)
        assert job.status == "complete" and run.status == "complete"
        assert run.provenance_json["web_search"]["state"] == state
    answers = [request for request in case["requests"] if not request.tools]
    assert len(answers) == 1 and answers[0].tools == []


async def test_real_chat_waits_for_the_exact_edited_query_then_answers_without_tools(
    execution: dict[str, Any],
) -> None:
    case = execution
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, state = pending(job_id)
    assert state == "awaiting_approval" and case["posts"] == []
    assert len(case["requests"]) == 1 and case["requests"][0].tools
    edited_query = "  Compare brass and steel  "
    changed = await case["client"].put(
        f"/api/jobs/{job_id}/search",
        json={"revision": revision, "query": edited_query},
    )
    assert changed.status_code == 200
    revision = changed.json()["revision"]
    await decide(case, job_id, revision, "approve")
    await execute(case, job_id, run_id)
    completed(case, job_id, "complete")
    assert case["posts"] == [
        {
            "url": "https://search.example.test/v1/search",
            "body": {
                "query": edited_query,
                "limit": 5,
                "answer": False,
                "summarizeResults": False,
                "queryExpand": False,
            },
        }
    ]
    assert len([request for request in case["requests"] if request.tools]) == 1
    answer = case["requests"][-1]
    assert SOURCE in json.dumps(answer.messages)
    assert "No linked page has been read." in json.dumps(answer.messages)


async def test_real_chat_decline_answers_without_any_provider_request(
    execution: dict[str, Any],
) -> None:
    case = execution
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, _ = pending(job_id)
    await decide(case, job_id, revision, "decline")
    await execute(case, job_id, run_id)
    completed(case, job_id, "declined")
    assert case["posts"] == []


async def test_real_chat_rechecks_global_permission_after_approval(
    execution: dict[str, Any],
) -> None:
    case = execution
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, _ = pending(job_id)
    await decide(case, job_id, revision, "approve")
    case["settings"].web_access_enabled = False
    await execute(case, job_id, run_id)
    completed(case, job_id, "cancelled")
    assert case["posts"] == []


async def test_real_chat_requires_fresh_provider_approval_before_dispatch(
    execution: dict[str, Any],
) -> None:
    case = execution
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, _ = pending(job_id)
    await decide(case, job_id, revision, "approve")
    case["settings"].crw_endpoint = "https://new-search.example.test"
    await execute(case, job_id, run_id)
    new_revision, state = pending(job_id)
    assert state == "awaiting_approval" and new_revision == revision + 1
    assert case["posts"] == [] and len(case["requests"]) == 1
    await decide(case, job_id, new_revision, "approve")
    await execute(case, job_id, run_id)
    completed(case, job_id, "complete")
    assert [post["url"] for post in case["posts"]] == ["https://new-search.example.test/v1/search"]


async def test_real_chat_automatic_search_enters_the_cancellable_timer(
    execution: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import web_search_consent as consent

    case = execution
    response = await case["client"].patch(
        f"/api/chats/{case['chat_id']}",
        json={"web_settings_json": {"allow_search": True, "allow_search_without_asking": True}},
    )
    assert response.status_code == 200
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    _, state = pending(job_id)
    assert state == "scheduled" and case["posts"] == [] and case["resumes"] == [job_id]
    with SessionLocal() as session:
        proposal = session.scalar(select(WebSearchProposal))
        deadline = proposal.dispatch_after

    async def elapsed(self: Any, moment: Any) -> None:
        monkeypatch.setattr(consent, "utcnow", lambda: moment + timedelta(milliseconds=1))

    monkeypatch.setattr(type(case["orch"]), "_wait_for_search_dispatch", elapsed)
    assert deadline is not None
    async with asyncio.timeout(30):
        await case["orch"]._resume_search_execution(job_id, run_id)
    completed(case, job_id, "complete")
    assert len(case["posts"]) == 1


async def test_real_chat_provider_timeout_preserves_a_local_answer(
    execution: dict[str, Any],
) -> None:
    case = execution
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, _ = pending(job_id)
    await decide(case, job_id, revision, "approve")
    case["mode"]["timeout"] = True
    await execute(case, job_id, run_id)
    completed(case, job_id, "failed")
    assert len(case["posts"]) == 1
    with SessionLocal() as session:
        proposal = session.scalar(select(WebSearchProposal))
        assert proposal.error_code == "search_timeout"


async def test_real_chat_with_search_disabled_does_not_offer_the_search_tool(
    execution: dict[str, Any],
) -> None:
    case = execution
    case["settings"].web_access_enabled = False
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    with SessionLocal() as session:
        assert session.get(Job, job_id).status == "complete"
        assert session.scalar(select(WebSearchProposal)) is None
    assert len(case["requests"]) == 1 and case["requests"][0].tools == []
    assert case["posts"] == []


async def test_real_chat_cancel_interrupts_a_dispatched_request_without_saved_results(
    execution: dict[str, Any],
) -> None:
    case = execution
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, _ = pending(job_id)
    await decide(case, job_id, revision, "approve")
    case["mode"]["block"] = True
    task = asyncio.create_task(execute(case, job_id, run_id))
    case["orch"]._tasks[job_id] = task
    try:
        async with asyncio.timeout(30):
            await case["entered"].wait()
            response = await case["client"].post(f"/api/jobs/{job_id}/cancel")
            assert response.status_code == 200
            with pytest.raises(asyncio.CancelledError):
                await task
        assert case["interrupted"].is_set()
        assert len(case["posts"]) == 1
        assert len(case["requests"]) == 1
        with SessionLocal() as session:
            assert session.get(Job, job_id).status == "cancelled"
            assert session.get(Run, run_id).status == "cancelled"
            proposal = session.scalar(select(WebSearchProposal))
            assert proposal.state == "dispatching"
            assert proposal.result_json == {}
        history = await case["client"].get(f"/api/chats/{case['chat_id']}")
        assert history.status_code == 200
        search = history.json()["web_searches"][0]
        assert search["job_id"] is None and search["results"] == []
        assert search["state"] == "uncertain"
    finally:
        case["release"].set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        case["orch"]._tasks.pop(job_id, None)


async def test_real_chat_lost_claim_cannot_commit_the_http_result_or_answer(
    execution: dict[str, Any],
) -> None:
    case = execution
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, _ = pending(job_id)
    await decide(case, job_id, revision, "approve")
    case["mode"]["block"] = True
    task = asyncio.create_task(execute(case, job_id, run_id))
    try:
        async with asyncio.timeout(30):
            await case["entered"].wait()
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                job.claim_owner = "constructed-successor"
                job.attempt += 1
                session.commit()
            case["release"].set()
            await task
        assert len(case["posts"]) == 1 and len(case["requests"]) == 1
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job.claim_owner == "constructed-successor" and job.status == "running"
            proposal = session.scalar(select(WebSearchProposal))
            assert proposal.state == "dispatching" and proposal.result_json == {}
            assert session.get(Run, run_id).provenance_json["web_search"]["state"] == "dispatching"
    finally:
        case["release"].set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("state", ["awaiting_approval", "approved", "scheduled"])
@pytest.mark.parametrize("route", ["job", "chat", "plan"])
async def test_stopping_a_turn_durably_revokes_pending_search(
    execution: dict[str, Any], state: str, route: str
) -> None:
    case = execution
    if state == "scheduled":
        response = await case["client"].patch(
            f"/api/chats/{case['chat_id']}",
            json={
                "web_settings_json": {
                    "allow_search": True,
                    "allow_search_without_asking": True,
                }
            },
        )
        assert response.status_code == 200
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, _ = pending(job_id)
    if state == "approved":
        await decide(case, job_id, revision, "approve")
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None and run.work_plan_id is not None
        plan_id = run.work_plan_id
        proposal = session.scalar(
            select(WebSearchProposal).where(WebSearchProposal.job_id == job_id)
        )
        assert proposal is not None and proposal.state == state
    path = (
        f"/api/jobs/{job_id}/cancel"
        if route == "job"
        else (
            f"/api/chats/{case['chat_id']}/cancel"
            if route == "chat"
            else f"/api/work-plans/{plan_id}/cancel"
        )
    )
    stopped = await case["client"].post(path)
    assert stopped.status_code == 200
    history = await case["client"].get(f"/api/chats/{case['chat_id']}")
    assert history.status_code == 200
    assert history.json()["web_searches"][0]["state"] == "cancelled"
    assert history.json()["web_searches"][0]["job_id"] is None
    with SessionLocal() as session:
        proposal = session.scalar(
            select(WebSearchProposal).where(WebSearchProposal.job_id == job_id)
        )
        assert proposal is not None and proposal.state == "cancelled"
        assert proposal.dispatch_after is None
        assert not proposal.approved_automatically
        assert session.get(Run, run_id).provenance_json["web_search"]["state"] == "cancelled"
    obsolete = await case["client"].post(
        f"/api/jobs/{job_id}/search/decision",
        json={"revision": revision, "action": "approve"},
    )
    assert obsolete.status_code == 409
    retried = await case["client"].post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200
    await execute(case, job_id, run_id)
    completed(case, job_id, "cancelled")
    assert case["posts"] == []


async def test_retry_never_sends_a_query_that_stopping_reported_cancelled(
    execution: dict[str, Any],
) -> None:
    case = execution
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, _ = pending(job_id)
    await decide(case, job_id, revision, "approve")
    stopped = await case["client"].post(f"/api/jobs/{job_id}/cancel")
    assert stopped.status_code == 200
    history = await case["client"].get(f"/api/chats/{case['chat_id']}")
    assert history.json()["web_searches"][0]["state"] == "cancelled"
    retried = await case["client"].post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200
    await execute(case, job_id, run_id)
    assert case["posts"] == [], "Retry sent the query already shown as cancelled."


@pytest.mark.parametrize(
    "parts, expected",
    [
        (
            [
                {"type": "text", "text": "Current copper question"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
            "Current copper question",
        ),
        (
            [{"type": "text", "text": "First line"}, {"type": "text", "text": "Second line"}],
            "First line\nSecond line",
        ),
        ([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}], None),
    ],
)
async def test_search_decision_uses_latest_multimodal_turn(
    execution: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    parts: list[dict[str, Any]],
    expected: str | None,
) -> None:
    case = execution
    prepare = type(case["orch"])._prepare_chat_context

    async def context(self: Any, *args: Any, **kwargs: Any) -> Any:
        messages, settings, metadata, tools = await prepare(self, *args, **kwargs)
        return (
            [
                {"role": "user", "content": "Previous unrelated question"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": parts},
            ],
            settings,
            metadata,
            tools,
        )

    monkeypatch.setattr(type(case["orch"]), "_prepare_chat_context", context)
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    decisions = [request for request in case["requests"] if request.tools]
    if expected is None:
        assert decisions == []
    else:
        assert len(decisions) == 1
        assert decisions[0].messages[-1]["content"] == expected
    assert case["posts"] == []


async def test_failure_revokes_approved_query_before_retry(execution: dict[str, Any]) -> None:
    case = execution
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, _ = pending(job_id)
    await decide(case, job_id, revision, "approve")
    await case["orch"]._fail(job_id, run_id, "Constructed failure", claim=None)
    with SessionLocal() as session:
        assert session.get(Job, job_id).status == "failed"
        proposal = session.scalar(
            select(WebSearchProposal).where(WebSearchProposal.job_id == job_id)
        )
        assert proposal is not None and proposal.state == "cancelled"
    retried = await case["client"].post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200
    await execute(case, job_id, run_id)
    assert case["posts"] == []


@pytest.mark.parametrize("state", ["awaiting_approval", "approved", "scheduled"])
async def test_retry_revokes_legacy_approval_left_by_an_older_stop(
    execution: dict[str, Any],
    state: str,
) -> None:
    case = execution
    if state == "scheduled":
        updated = await case["client"].patch(
            f"/api/chats/{case['chat_id']}",
            json={
                "web_settings_json": {
                    "allow_search": True,
                    "allow_search_without_asking": True,
                }
            },
        )
        assert updated.status_code == 200
    job_id, run_id = await accepted(case)
    await execute(case, job_id, run_id)
    revision, _ = pending(job_id)
    if state == "approved":
        await decide(case, job_id, revision, "approve")
    with SessionLocal() as session:
        row = session.scalar(select(WebSearchProposal).where(WebSearchProposal.job_id == job_id))
        assert row.state == state
        saved = (row.approved_automatically, row.dispatch_after)
        provenance = dict(session.get(Run, run_id).provenance_json)
    stopped = await case["client"].post(f"/api/jobs/{job_id}/cancel")
    assert stopped.status_code == 200
    # Model a database written by an older stop path: terminal work, but
    # the proposal and its history still carry their earlier approval.
    with SessionLocal() as session:
        row = session.scalar(select(WebSearchProposal).where(WebSearchProposal.job_id == job_id))
        row.state = state
        row.approved_automatically, row.dispatch_after = saved
        row.error_code = None
        session.get(Run, run_id).provenance_json = provenance
        session.commit()
    history = await case["client"].get(f"/api/chats/{case['chat_id']}")
    assert history.json()["web_searches"][0]["state"] == "cancelled"
    retried = await case["client"].post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200
    with SessionLocal() as session:
        row = session.scalar(select(WebSearchProposal).where(WebSearchProposal.job_id == job_id))
        assert row.state == "cancelled"
        assert row.dispatch_after is None and not row.approved_automatically
    await execute(case, job_id, run_id)
    completed(case, job_id, "cancelled")
    assert case["posts"] == []


@pytest.mark.parametrize(
    "reason",
    [
        "Current material data",
        "\u2764\ufe0f",
        "\u0645\u06cc\u200c\u0631\u0648\u0645",
        "\u0915\u094d\u200d\u0937",
        "\U0001f469\u200d\U0001f4bb",
        "\u1820\u180e\u1820",
    ],
)
async def test_unicode_reason_keeps_real_proposal_and_consent_card(
    execution: dict[str, Any], reason: str
) -> None:
    execution["proposal"]["reason"] = reason
    job_id, run_id = await accepted(execution)
    await execute(execution, job_id, run_id)
    _, state = pending(job_id)
    assert state == "awaiting_approval"
    detail = await execution["client"].get(f"/api/chats/{execution['chat_id']}")
    assert detail.status_code == 200
    cards = detail.json()["web_searches"]
    assert len(cards) == 1
    assert cards[0]["query"] == QUERY
    assert cards[0]["state"] == "awaiting_approval"
    assert execution["posts"] == []
