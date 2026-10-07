from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import select

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.adapters.mock import MockChatAdapter
from local_lm.db import SessionLocal
from local_lm.models import Job


async def finish(client: AsyncClient, run_id: str, expected: str = "complete") -> None:
    async def read() -> dict[str, Any]:
        response = await client.get(f"/api/runs/{run_id}")
        response.raise_for_status()
        value: dict[str, Any] = response.json()
        return value

    await wait_for_terminal_status(read, what="chat activity transition", expected=expected)


async def activity(client: AsyncClient, chat_id: str) -> dict[str, Any]:
    response = await client.get("/api/chats/summaries")
    response.raise_for_status()
    value: dict[str, Any] = next(row["activity"] for row in response.json() if row["id"] == chat_id)
    return value


async def completed_turn(client: AsyncClient, mode: str = "text") -> dict[str, Any]:
    chat = (await client.post("/api/chats", json={"title": "Color study"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Describe a blue square.", "mode": mode},
    )
    response.raise_for_status()
    value: dict[str, Any] = response.json()
    await finish(client, value["run"]["id"])
    return value


async def test_selecting_an_older_response_keeps_its_original_activity_identity(
    client: AsyncClient,
) -> None:
    original = await completed_turn(client)
    chat_id = original["run"]["chat_id"]
    message_id = original["assistant_message"]["id"]
    earlier = (await activity(client, chat_id))["last_output"]
    replacement = await client.post(f"/api/messages/{message_id}/regenerate", json={"settings": {}})
    replacement.raise_for_status()
    await finish(client, replacement.json()["run"]["id"])
    latest = (await activity(client, chat_id))["last_output"]
    assert latest["sequence"] > earlier["sequence"]
    assert latest["id"] != earlier["id"]
    assert latest["message_id"] == earlier["message_id"] == message_id
    selected = await client.post(
        f"/api/messages/{message_id}/revisions/{earlier['response_revision_id']}/select"
    )
    selected.raise_for_status()
    message = selected.json()
    revision = next(
        row
        for row in message["response_revisions"]
        if row["id"] == message["active_response_revision_id"]
    )
    assert revision["activity"] == earlier
    assert (await activity(client, chat_id))["last_output"] == latest


async def test_successful_retry_resolves_a_failed_replacement_with_a_new_identity(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = await completed_turn(client)
    chat_id = original["run"]["chat_id"]
    message_id = original["assistant_message"]["id"]
    earlier = (await activity(client, chat_id))["last_output"]

    async def fail(_adapter: MockChatAdapter, _request: ChatRequest) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(type="delta", text="A partial color study.")
        yield ChatEvent(type="error", data={"error": "Neutral generation failure"})

    with monkeypatch.context() as patch:
        patch.setattr(MockChatAdapter, "stream", fail)
        response = await client.post(
            f"/api/messages/{message_id}/regenerate", json={"settings": {}}
        )
        response.raise_for_status()
        run_id = response.json()["run"]["id"]
        await finish(client, run_id, "failed")
    failed = await activity(client, chat_id)
    assert failed["unresolved_failed_count"] == 1
    assert failed["active_work_count"] == 0
    assert failed["last_output"] == earlier
    assert failed["last_failure"]["message_id"] == message_id
    with SessionLocal() as session:
        job_id = session.scalar(select(Job.id).where(Job.run_id == run_id, Job.kind == "chat"))
    assert job_id is not None
    retry = await client.post(f"/api/jobs/{job_id}/retry")
    retry.raise_for_status()
    await finish(client, run_id)
    completed = await activity(client, chat_id)
    assert completed["active_work_count"] == completed["unresolved_failed_count"] == 0
    assert completed["last_failure"] is None
    assert completed["last_output"]["id"] not in {earlier["id"], failed["last_failure"]["id"]}
    assert completed["last_output"]["sequence"] > failed["last_failure"]["sequence"]
    assert completed["last_output"]["message_id"] == message_id


@pytest.mark.parametrize("mode", ["text", "image"])
async def test_removing_completed_content_removes_its_sidebar_activity(
    client: AsyncClient, mode: str
) -> None:
    original = await completed_turn(client, mode)
    chat_id = original["run"]["chat_id"]
    message_id = original["assistant_message"]["id"]
    before = await activity(client, chat_id)
    assert before["last_output"]["message_id"] == message_id
    if mode == "image":
        message = (await client.get(f"/api/messages/{message_id}")).json()
        artifact_id = next(
            part["artifact_id"] for part in message["parts"] if part["type"] == "image"
        )
        refused = await client.delete(f"/api/artifacts/{artifact_id}")
        assert refused.status_code == 409
        assert await activity(client, chat_id) == before
    preview = await client.get(f"/api/messages/{message_id}/removal-impact")
    preview.raise_for_status()
    removed = await client.post(
        f"/api/messages/{message_id}/remove-content",
        json={
            "expected_message_id": message_id,
            "expected_revision_id": preview.json()["message_revision_id"],
            "operation_key": "remove-completed-color-study",
        },
    )
    removed.raise_for_status()
    assert await activity(client, chat_id) == {
        "active_work_count": 0,
        "unresolved_failed_count": 0,
        "last_output": None,
        "last_failure": None,
    }
    message = (await client.get(f"/api/messages/{message_id}")).json()
    assert message["content_removed_at"] is not None
    assert message["parts"] == []


async def test_forked_history_does_not_copy_completion_identities(client: AsyncClient) -> None:
    original = await completed_turn(client)
    source_id = original["run"]["chat_id"]
    before = await activity(client, source_id)
    response = await client.post(f"/api/messages/{original['assistant_message']['id']}/fork")
    response.raise_for_status()
    fork_id = response.json()["id"]
    assert fork_id != source_id
    fork = (await client.get(f"/api/chats/{fork_id}")).json()
    assert any(part["type"] == "text" for message in fork["messages"] for part in message["parts"])
    assert all(
        revision["activity"] is None
        for message in fork["messages"]
        for revision in message["response_revisions"]
    )
    assert (await activity(client, fork_id))["last_output"] is None
    assert await activity(client, source_id) == before


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("partial", [False, True])
async def test_cancelled_responses_only_publish_activity_for_retained_output(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, replacement: bool, partial: bool
) -> None:
    original = await completed_turn(client)
    chat_id = original["run"]["chat_id"]
    message_id = original["assistant_message"]["id"]
    earlier = (await activity(client, chat_id))["last_output"]
    entered = asyncio.Event()

    async def stream(_adapter: MockChatAdapter, _request: ChatRequest) -> AsyncIterator[ChatEvent]:
        if partial:
            yield ChatEvent(type="delta", text="A retained color study.")
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(MockChatAdapter, "stream", stream)
    if replacement:
        response = await client.post(
            f"/api/messages/{message_id}/regenerate", json={"settings": {}}
        )
    else:
        response = await client.post(
            f"/api/chats/{chat_id}/turns", json={"text": "Describe a green circle.", "mode": "text"}
        )
    response.raise_for_status()
    accepted = response.json()
    run_id = accepted["run"]["id"]
    async with asyncio.timeout(30):
        await entered.wait()
    assert (await activity(client, chat_id))["active_work_count"] == 1
    with SessionLocal() as session:
        job_id = session.scalar(select(Job.id).where(Job.run_id == run_id, Job.kind == "chat"))
    assert job_id is not None
    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    cancelled.raise_for_status()
    await finish(client, run_id, "cancelled")
    after = await activity(client, chat_id)
    assert after["active_work_count"] == after["unresolved_failed_count"] == 0
    assert after["last_failure"] is None
    target_id = message_id if replacement else accepted["assistant_message"]["id"]
    target = (await client.get(f"/api/messages/{target_id}")).json()
    revision = next(row for row in target["response_revisions"] if row["run_id"] == run_id)
    assert revision["status"] == "cancelled"
    if partial:
        assert after["last_output"]["id"] != earlier["id"]
        assert after["last_output"]["sequence"] > earlier["sequence"]
        assert after["last_output"]["message_id"] == target_id
        assert revision["activity"] == after["last_output"]
        assert any(part["text"] == "A retained color study." for part in revision["parts"])
    else:
        assert after["last_output"] == earlier
        assert revision["activity"] is None
