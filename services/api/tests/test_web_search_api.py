"""Visible exact-query consent uses durable commands, never browser authority."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Chat, Job, Run, WebSearchProposal
from local_lm.scheduler import JobClaim
from local_lm.web_search import CrwSearchProvider
from local_lm.web_search_configuration import search_provider_revision
from local_lm.web_search_consent import pause_for_search

QUERY = "Compare brass and aluminum"
ENDPOINT = "https://search.example.test"


@pytest_asyncio.fixture
async def pending(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    orchestrator = app.state.services.orchestrator
    orchestrator.engines.settings.web_access_enabled = True
    orchestrator.engines.settings.crw_endpoint = ENDPOINT
    orchestrator.engines.settings.crw_token = "constructed-search-token"
    provider = CrwSearchProvider(ENDPOINT, "constructed-search-token")
    paused = asyncio.Event()
    tasks: list[Any] = []
    proposals: list[Any] = []

    async def propose(self: Any, job_id: str, run_id: str, claim: JobClaim) -> None:
        tasks.append(asyncio.current_task())
        with SessionLocal() as session:
            proposals.append(
                pause_for_search(
                    session,
                    job_id,
                    claim,
                    query=QUERY,
                    provider_endpoint=ENDPOINT,
                    provider_revision=search_provider_revision(provider),
                    installation_enabled=True,
                )
            )
        paused.set()

    monkeypatch.setattr(type(orchestrator), "_execute_chat", propose)
    created = await client.post("/api/chats", json={"title": "Search approval controls"})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    updated = await client.patch(
        f"/api/chats/{chat_id}",
        json={"web_settings_json": {"allow_search": True}},
    )
    assert updated.status_code == 200
    accepted = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={"text": QUERY, "mode": "text"},
    )
    assert accepted.status_code == 202
    async with asyncio.timeout(30):
        await paused.wait()
        await tasks[0]
    resumed: list[str] = []
    monkeypatch.setattr(
        type(orchestrator),
        "resume_search",
        lambda self, job_id: resumed.append(job_id),
    )
    return {
        "chat_id": chat_id,
        "run_id": accepted.json()["run"]["id"],
        "message_id": accepted.json()["assistant_message"]["id"],
        "user_message_id": accepted.json()["user_message"]["id"],
        "job_id": proposals[0].job_id,
        "revision": proposals[0].revision,
        "provider_revision": search_provider_revision(provider),
        "resumed": resumed,
    }


async def test_chat_shows_the_exact_query_and_provider_without_credential_identity(
    client: AsyncClient,
    pending: dict[str, Any],
) -> None:
    detail = await client.get(f"/api/chats/{pending['chat_id']}")
    assert detail.status_code == 200
    rows = detail.json()["web_searches"]
    assert len(rows) == 1
    assert rows[0]["query"] == QUERY
    assert rows[0]["provider_endpoint"] == ENDPOINT
    assert rows[0]["assistant_message_id"] == pending["message_id"]
    assert rows[0]["job_id"] == pending["job_id"]
    assert rows[0]["state"] == "awaiting_approval"
    assert "constructed-search-token" not in detail.text
    assert pending["provider_revision"] not in detail.text


@pytest.mark.parametrize(
    "action,state",
    [
        ("approve", "approved"),
        ("decline", "declined"),
        ("cancel", "cancelled"),
    ],
)
async def test_search_commands_are_retryable_and_request_execution_resumption(
    client: AsyncClient,
    pending: dict[str, Any],
    action: str,
    state: str,
) -> None:
    path = f"/api/jobs/{pending['job_id']}/search/decision"
    body = {"revision": pending["revision"], "action": action}
    first = await client.post(path, json=body)
    assert first.status_code == 200
    second = await client.post(path, json=body)
    assert second.status_code == 200 and first.json() == second.json()
    assert first.json()["state"] == state
    assert pending["resumed"] == [pending["job_id"], pending["job_id"]]
    with SessionLocal() as session:
        assert session.get(Job, pending["job_id"]).status == "queued"


async def test_editing_the_query_requires_approval_of_the_replacement(
    client: AsyncClient,
    pending: dict[str, Any],
) -> None:
    path = f"/api/jobs/{pending['job_id']}/search"
    edited = await client.put(path, json={"revision": 1, "query": "Compare copper and steel"})
    assert edited.status_code == 200
    assert edited.json()["revision"] == 2
    assert edited.json()["query"] == "Compare copper and steel"
    assert edited.json()["provider_endpoint"] == ENDPOINT
    stale = await client.post(path + "/decision", json={"revision": 1, "action": "approve"})
    assert stale.status_code == 409
    approved = await client.post(path + "/decision", json={"revision": 2, "action": "approve"})
    assert approved.status_code == 200 and approved.json()["state"] == "approved"


@pytest.mark.parametrize("revision", [True, 0, 1.0, "1"])
async def test_search_command_revision_is_a_strict_positive_integer(
    client: AsyncClient,
    pending: dict[str, Any],
    revision: Any,
) -> None:
    response = await client.post(
        f"/api/jobs/{pending['job_id']}/search/decision",
        json={"revision": revision, "action": "approve"},
    )
    assert response.status_code == 422
    assert not pending["resumed"]


@pytest.mark.parametrize("change", ["cancelled", "hidden"])
async def test_a_terminal_or_nonstandard_owner_cannot_accept_a_search_command(
    client: AsyncClient,
    pending: dict[str, Any],
    change: str,
) -> None:
    with SessionLocal() as session:
        if change == "cancelled":
            session.get(Job, pending["job_id"]).status = "cancelled"
        else:
            session.get(Chat, pending["chat_id"]).scope = "prompt_helper"
        session.commit()
    response = await client.post(
        f"/api/jobs/{pending['job_id']}/search/decision",
        json={"revision": 1, "action": "approve"},
    )
    assert response.status_code == 409
    assert not pending["resumed"]
    if change == "hidden":
        assert (await client.get(f"/api/chats/{pending['chat_id']}")).status_code == 404


async def test_chat_search_history_remains_readable_after_job_cleanup(
    client: AsyncClient,
    pending: dict[str, Any],
) -> None:
    cancelled = await client.post(
        f"/api/jobs/{pending['job_id']}/search/decision",
        json={"revision": 1, "action": "cancel"},
    )
    assert cancelled.status_code == 200
    with SessionLocal() as session:
        session.delete(session.get(Job, pending["job_id"]))
        session.commit()
    detail = await client.get(f"/api/chats/{pending['chat_id']}")
    row = detail.json()["web_searches"][0]
    assert row["state"] == "cancelled" and row["query"] == QUERY
    assert row["job_id"] is None and row["revision"] is None


async def test_imported_history_cannot_supply_an_active_link_or_command(
    client: AsyncClient,
    pending: dict[str, Any],
) -> None:
    with SessionLocal() as session:
        run = session.get(Run, pending["run_id"])
        run.provenance_json = {
            "web_search": {
                "state": "complete",
                "query": QUERY,
                "provider_endpoint": ENDPOINT,
                "job_id": "invented-command-owner",
                "revision": 999,
                "results": {
                    "results": [
                        {"url": "javascript:alert(1)", "title": "Invalid source", "snippet": ""},
                        {
                            "url": "https://example.test/source",
                            "title": "Source",
                            "snippet": "Summary",
                        },
                    ]
                },
            }
        }
        session.delete(session.get(Job, pending["job_id"]))
        session.commit()
        assert session.scalar(select(WebSearchProposal)) is None
    detail = await client.get(f"/api/chats/{pending['chat_id']}")
    row = detail.json()["web_searches"][0]
    assert row["job_id"] is None and row["revision"] is None
    assert row["result_count"] == 1
    assert [item["url"] for item in row["results"]] == ["https://example.test/source"]
    assert "javascript:" not in detail.text


@pytest.mark.parametrize(
    "endpoint,enabled,configured,code",
    [
        (ENDPOINT, True, True, None),
        (None, True, False, "search_not_configured"),
        ("https://account:secret@example.test", True, False, "search_provider_invalid"),
        (ENDPOINT, False, True, None),
    ],
)
async def test_configuration_reports_availability_without_exposing_account_data(
    client: AsyncClient,
    app: FastAPI,
    endpoint: str | None,
    enabled: bool,
    configured: bool,
    code: str | None,
) -> None:
    settings = app.state.services.settings
    settings.crw_endpoint = endpoint
    settings.crw_token = "constructed-search-token"
    settings.web_access_enabled = enabled
    response = await client.get("/api/web-search/configuration")
    assert response.status_code == 200
    result = response.json()
    assert result["installation_enabled"] is enabled
    assert result["configured"] is configured and result["error_code"] == code
    assert result["provider_endpoint"] == (ENDPOINT if configured else None)
    assert "constructed-search-token" not in response.text and "account:secret" not in response.text


async def test_a_new_or_forked_chat_does_not_inherit_search_permission(
    client: AsyncClient,
    pending: dict[str, Any],
) -> None:
    await client.patch(
        f"/api/chats/{pending['chat_id']}",
        json={"web_settings_json": {"allow_search": True, "allow_search_without_asking": True}},
    )
    created = await client.post("/api/chats", json={"title": "Separate search permission"})
    forked = await client.post(f"/api/messages/{pending['user_message_id']}/fork")
    assert created.status_code == 201 and forked.status_code == 201
    for response in (created, forked):
        assert response.json()["web_settings_json"] == {
            "allow_url_fetch": False,
            "allow_search": False,
            "allow_search_without_asking": False,
        }


async def test_null_web_permission_is_rejected_without_changing_the_chat(
    client: AsyncClient,
    pending: dict[str, Any],
) -> None:
    response = await client.patch(
        f"/api/chats/{pending['chat_id']}",
        json={"web_settings_json": None},
    )
    assert response.status_code == 422
    detail = await client.get(f"/api/chats/{pending['chat_id']}")
    assert detail.json()["web_settings_json"]["allow_search"] is True


async def test_portable_project_import_does_not_transfer_search_permission_or_live_approval(
    client: AsyncClient,
    app: FastAPI,
    pending: dict[str, Any],
) -> None:
    import io
    import json
    import zipfile

    from local_lm.models import Project

    services = app.state.services
    with SessionLocal() as session:
        project = Project(name="Portable search permission")
        session.add(project)
        session.flush()
        chat = session.get(Chat, pending["chat_id"])
        chat.project_id = project.id
        chat.web_settings_json = {"allow_search": True, "allow_search_without_asking": True}
        session.commit()
        artifact = services.exports.export(session, project.id, include_media=False)
        session.commit()
        archive = services.artifacts.resolve(artifact).read_bytes()
        with zipfile.ZipFile(io.BytesIO(archive)) as package:
            manifest = json.loads(package.read("manifest.json"))
        exported = json.dumps(manifest)
        assert "constructed-search-token" not in exported
        assert pending["provider_revision"] not in exported
        assert "web_search_proposals" not in exported
        imported = services.exports.import_archive(session, io.BytesIO(archive))
        session.commit()
        imported_chat = session.scalar(select(Chat).where(Chat.project_id == imported.id))
        assert imported_chat is not None and imported_chat.id != pending["chat_id"]
        imported_id = imported_chat.id
        proposals = session.scalars(select(WebSearchProposal)).all()
        assert len(proposals) == 1 and proposals[0].run_id == pending["run_id"]
    detail = await client.get(f"/api/chats/{imported_id}")
    assert detail.status_code == 200
    assert detail.json()["web_settings_json"] == {
        "allow_url_fetch": False,
        "allow_search": False,
        "allow_search_without_asking": False,
    }
    for history in detail.json()["web_searches"]:
        assert history["job_id"] is None and history["revision"] is None


@pytest.mark.parametrize(
    "query",
    [
        "\u0645\u06cc\u200c\u0631\u0648\u0645",
        "\u0915\u094d\u200d\u0937",
        "\u2764\ufe0f",
        "\U0001f469\u200d\U0001f4bb",
        "\u1820\u180e\u1820",
        "caf\xe9",
        "\u6750\u6599",
        "\U0001f600",
    ],
)
async def test_unicode_spelling_survives_edit_approval_and_projection(
    client: AsyncClient,
    pending: dict[str, Any],
    query: str,
) -> None:
    path = f"/api/jobs/{pending['job_id']}/search"
    edited = await client.put(path, json={"revision": pending["revision"], "query": query})
    assert edited.status_code == 200
    assert edited.json()["query"] == query
    approved = await client.post(
        path + "/decision",
        json={"revision": edited.json()["revision"], "action": "approve"},
    )
    assert approved.status_code == 200
    assert approved.json()["query"] == query
    detail = await client.get(f"/api/chats/{pending['chat_id']}")
    assert detail.json()["web_searches"][0]["query"] == query
