"""Deleted projects appear unfiled without rewriting canonical chat links."""

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI, Request
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_chat_recovery import _command, _history, _impact
from test_project_recovery_api import _project, _stored_trash
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm import api
from local_lm.db import SessionLocal
from local_lm.models import Chat, Project


async def test_deleted_project_is_absent_from_normal_reads_but_child_links_restore(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    project_id, chat_id = await _project(client, "Unique garden filing")
    before = _history(chat_id)
    item = _stored_trash(project_id, datetime.now(UTC))
    assert all(row["id"] != project_id for row in (await client.get("/api/projects")).json())
    queries: tuple[dict[str, str | bool], ...] = (
        {"project_id": project_id},
        {"query": "Unique garden", "literal_search": True},
    )
    for params in queries:
        assert (await client.get("/api/projects", params=params)).json() == []
    assert (await client.get(f"/api/projects/{project_id}")).status_code == 404
    for suffix in ("", "/metadata"):
        response = await client.get(f"/api/chats/{chat_id}{suffix}")
        assert response.status_code == 200 and response.json()["project_id"] is None
    for path in ("/api/chats", "/api/chats/summaries"):
        response = await client.get(path)
        assert next(row for row in response.json() if row["id"] == chat_id)["project_id"] is None
        assert (await client.get(path, params={"project_id": project_id})).json() == []
        assert (
            await client.get(path, params={"query": "Unique garden", "search_projects": True})
        ).json() == []
        by_title = await client.get(path, params={"query": "Garden notes", "search_projects": True})
        assert any(row["id"] == chat_id and row["project_id"] is None for row in by_title.json())
    updated = await client.patch(f"/api/chats/{chat_id}", json={"title": "Garden notes"})
    assert updated.status_code == 200 and updated.json()["project_id"] is None
    assert _history(chat_id) == before
    with SessionLocal() as session:
        assert session.get(Project, project_id) is not None
        assert session.scalar(select(Chat.project_id).where(Chat.id == chat_id)) == project_id
    preview = await _impact(client, f"/api/recovery-items/{item.deletion_id}/impact")
    restored = await client.post(
        f"/api/recovery-items/{item.deletion_id}/restore", json=_command(preview, "restore-filing")
    )
    assert restored.status_code == 200
    assert (await client.get(f"/api/chats/{chat_id}/metadata")).json()["project_id"] == project_id
    assert _history(chat_id) == before


@pytest.mark.parametrize("operation", ["update", "delete", "export", "create-chat", "file-chat"])
async def test_deleted_project_refuses_normal_writes_and_export_without_mutation(
    client: AsyncClient,
    operation: str,
) -> None:
    project_id, chat_id = await _project(client)
    _live_id, unfiled_id = await _project(client, "Garden paths")
    item = _stored_trash(project_id, datetime.now(UTC))
    before = _history(chat_id)
    if operation == "update":
        response = await client.patch(
            f"/api/projects/{project_id}", json={"name": "Changed garden"}
        )
    elif operation == "delete":
        response = await client.delete(f"/api/projects/{project_id}")
    elif operation == "export":
        response = await client.post(f"/api/projects/{project_id}/export")
    elif operation == "create-chat":
        response = await client.post(
            "/api/chats", json={"title": "New garden", "project_id": project_id}
        )
    else:
        response = await client.patch(f"/api/chats/{unfiled_id}", json={"project_id": project_id})
    assert response.status_code == 404 and response.json()["code"] == "project-not-found"
    assert _history(chat_id) == before
    assert (await client.get(f"/api/recovery-items/{item.deletion_id}/impact")).status_code == 200


@pytest.mark.parametrize("operation", ["update", "create-chat", "file-chat"])
async def test_project_admission_rechecks_deletion_after_async_settings_validation(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    project_id, chat_id = await _project(client)
    _live_id, moved_id = await _project(client, "Garden paths")
    before = _history(chat_id)
    original = api._validate_generation_defaults

    async def delayed(request: Request, session: Session, values: dict[str, Any]) -> None:
        await original(request, session, values)
        _stored_trash(project_id, datetime.now(UTC))

    monkeypatch.setattr(api, "_validate_generation_defaults", delayed)
    if operation == "update":
        response = await client.patch(
            f"/api/projects/{project_id}", json={"name": "Changed garden"}
        )
    elif operation == "create-chat":
        response = await client.post(
            "/api/chats", json={"title": "New garden", "project_id": project_id}
        )
    else:
        response = await client.patch(f"/api/chats/{moved_id}", json={"project_id": project_id})
    assert response.status_code == 404 and response.json()["code"] == "project-not-found"
    assert _history(chat_id) == before
