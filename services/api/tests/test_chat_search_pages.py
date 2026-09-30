"""Page search history without hiding pending decisions in other branches."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event

from local_lm.db import SessionLocal
from local_lm.models import Chat, Job, Message, Run, WebSearchProposal
from local_lm.prompt_helpers import PROMPT_HELPER_SCOPE


def _seed(count: int = 9) -> list[str]:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    ids = [f"run_search_page_{index:03d}" for index in range(count)]
    with SessionLocal() as session:
        session.add(Chat(id="chat_search_page"))
        session.flush()
        for index, identity in enumerate(ids):
            message_id = f"msg_search_page_{index:03d}"
            session.add(
                Message(
                    id=message_id,
                    chat_id="chat_search_page",
                    parent_id=f"msg_search_page_{index - 1:03d}" if index else None,
                    role="assistant",
                    status="complete",
                    created_at=stamp,
                )
            )
            session.flush()
            session.add(
                Run(
                    id=identity,
                    chat_id="chat_search_page",
                    user_message_id=message_id,
                    assistant_message_id=message_id,
                    status="complete",
                    created_at=stamp,
                    provenance_json={
                        "web_search": {
                            "query": "Compare brass and aluminum",
                            "provider_endpoint": "https://search.example.test",
                            "state": "complete",
                        }
                    },
                )
            )
        session.commit()
    return ids


async def test_search_pages_keep_timestamp_ties_and_bound_hydration(client: AsyncClient) -> None:
    ids = _seed(13)
    loaded: list[str] = []

    def record(instance: object, _context: object) -> None:
        loaded.append(type(instance).__name__)

    for model in (Run, Message):
        event.listen(model, "load", record)
    try:
        full = await client.get("/api/chats/chat_search_page")
        assert full.status_code == 200
        assert loaded.count("Run") == 13 and loaded.count("Message") == 13
        loaded.clear()
        first = await client.get("/api/chats/chat_search_page/searches", params={"limit": 4})
    finally:
        for model in (Run, Message):
            event.remove(model, "load", record)
    assert first.status_code == 200
    assert loaded.count("Run") <= 5 and "Message" not in loaded
    page = first.json()
    assert [item["run_id"] for item in page["searches"]] == ids[-4:]
    seen = [item["run_id"] for item in page["searches"]]
    while page["next_before"] is not None:
        response = await client.get(
            "/api/chats/chat_search_page/searches",
            params={"limit": 4, "before": page["next_before"]},
        )
        assert response.status_code == 200
        page = response.json()
        seen = [item["run_id"] for item in page["searches"]] + seen
    assert seen == ids


async def test_an_empty_malformed_page_still_advances_its_cursor(client: AsyncClient) -> None:
    ids = _seed(5)
    with SessionLocal() as session:
        for identity in ids[-2:]:
            run = session.get(Run, identity)
            assert run is not None
            run.provenance_json = {"web_search": {"state": "complete"}}
        session.commit()
    response = await client.get("/api/chats/chat_search_page/searches", params={"limit": 2})
    assert response.status_code == 200
    assert response.json()["searches"] == []
    assert response.json()["next_before"] == ids[-2]
    older = await client.get(
        "/api/chats/chat_search_page/searches", params={"limit": 2, "before": ids[-2]}
    )
    assert older.status_code == 200
    assert [item["run_id"] for item in older.json()["searches"]] == ids[1:3]


async def test_searches_follow_the_visible_branch_message_range(client: AsyncClient) -> None:
    ids = _seed()
    with SessionLocal() as session:
        sibling = session.get(Message, "msg_search_page_008")
        assert sibling is not None
        sibling.parent_id = "msg_search_page_000"
        hidden = session.get(Message, "msg_search_page_004")
        assert hidden is not None
        hidden.transcript_visible = False
        session.commit()
    response = await client.get(
        "/api/chats/chat_search_page/searches",
        params={"head_id": "msg_search_page_007", "oldest_message_id": "msg_search_page_003"},
    )
    assert response.status_code == 200
    assert [item["run_id"] for item in response.json()["searches"]] == [
        ids[3],
        ids[5],
        ids[6],
        ids[7],
    ]
    assert response.json()["next_before"] is None


async def test_pending_decisions_page_across_branches_and_survive_a_changed_cursor(
    client: AsyncClient,
) -> None:
    ids = _seed(5)
    with SessionLocal() as session:
        for index, identity in enumerate(ids):
            session.add(Job(id=f"job_search_{index}", run_id=identity, status="paused"))
        session.flush()
        for index, identity in enumerate(ids):
            session.add(
                WebSearchProposal(
                    id=f"proposal_search_{index}",
                    job_id=f"job_search_{index}",
                    run_id=identity,
                    state="awaiting_approval",
                    query="Compare brass and aluminum",
                    provider_endpoint="https://search.example.test",
                    provider_revision="test-provider",
                )
            )
        sibling = session.get(Message, "msg_search_page_004")
        assert sibling is not None
        sibling.parent_id = "msg_search_page_000"
        hidden = session.get(Message, "msg_search_page_000")
        assert hidden is not None
        hidden.transcript_visible = False
        session.commit()
    response = await client.get(
        "/api/chats/chat_search_page/searches", params={"pending_only": "true", "limit": 2}
    )
    assert response.status_code == 200
    assert [item["run_id"] for item in response.json()["searches"]] == ids[-2:]
    with SessionLocal() as session:
        proposal = session.get(WebSearchProposal, "proposal_search_3")
        finished = session.get(Job, "job_search_1")
        assert proposal is not None and finished is not None
        proposal.state = "declined"
        finished.status = "complete"
        session.commit()
    older = await client.get(
        "/api/chats/chat_search_page/searches",
        params={"pending_only": "true", "limit": 2, "before": ids[3]},
    )
    assert older.status_code == 200
    assert [item["run_id"] for item in older.json()["searches"]] == [ids[0], ids[2]]
    assert all(item["job_id"] is not None for item in older.json()["searches"])
    assert older.json()["next_before"] is None


@pytest.mark.parametrize("limit", [0, 101])
async def test_search_page_limits_are_enforced(client: AsyncClient, limit: int) -> None:
    _seed()
    response = await client.get("/api/chats/chat_search_page/searches", params={"limit": limit})
    assert response.status_code == 400
    assert response.json()["code"] == "chat-search-window-invalid"


@pytest.mark.parametrize(
    "scope", ["missing", "helper", "foreign-head", "foreign-cursor", "sibling"]
)
async def test_search_scope_refuses_inaccessible_anchors(client: AsyncClient, scope: str) -> None:
    _seed()
    params: dict[str, str] = {}
    with SessionLocal() as session:
        if scope in {"missing", "helper"}:
            chat = session.get(Chat, "chat_search_page")
            assert chat is not None
            chat.scope = PROMPT_HELPER_SCOPE
        else:
            session.add(Chat(id="chat_search_other"))
            session.flush()
            session.add(Message(id="msg_search_other", chat_id="chat_search_other", role="user"))
            session.flush()
            session.add(
                Run(
                    id="run_search_other",
                    chat_id="chat_search_other",
                    user_message_id="msg_search_other",
                    assistant_message_id="msg_search_other",
                )
            )
        session.commit()
    chat_id = "missing" if scope == "missing" else "chat_search_page"
    if scope == "foreign-head":
        params["head_id"] = "msg_search_other"
    elif scope == "foreign-cursor":
        params["before"] = "run_search_other"
    elif scope == "sibling":
        params = {"head_id": "msg_search_page_004", "oldest_message_id": "msg_search_page_008"}
    response = await client.get(f"/api/chats/{chat_id}/searches", params=params)
    assert response.status_code == (400 if scope == "sibling" else 404)
    assert (
        response.json()["code"]
        == {
            "missing": "chat-not-found",
            "helper": "chat-not-found",
            "foreign-head": "message-not-found",
            "foreign-cursor": "search-not-found",
            "sibling": "chat-search-window-invalid",
        }[scope]
    )


async def test_pending_decisions_cannot_be_narrowed_to_a_branch(client: AsyncClient) -> None:
    _seed()
    response = await client.get(
        "/api/chats/chat_search_page/searches",
        params={"pending_only": "true", "head_id": "msg_search_page_008"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "chat-search-window-invalid"
