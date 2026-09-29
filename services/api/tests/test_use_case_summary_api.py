"""Generating a local suggestion never replaces a saved description."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import anyio
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from starlette.types import Message, Scope

from local_lm import use_case_summary_api as summary_api
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import Chat, Job, ModelAssetInstall, ModelInstall, ModelProfile, Run
from local_lm.use_case_summaries import UseCaseSummaryError


def read_target(
    session: Session, kind: str, record_id: str
) -> ModelProfile | ModelAssetInstall | None:
    if kind == "profile":
        return session.get(ModelProfile, record_id)
    return session.get(ModelAssetInstall, record_id)


def target(kind: str, description: str = "Watercolor landscapes") -> tuple[str, str]:
    with SessionLocal() as session:
        if kind == "profile":
            install = ModelInstall(
                name="Constructed model",
                role="chat",
                engine="mock",
                local_path="neutral",
                manifest_json={"provider_description": description},
                active=True,
            )
            session.add(install)
            session.flush()
            row: ModelProfile | ModelAssetInstall = ModelProfile(
                name="Constructed profile",
                role="chat",
                engine="mock",
                model_install_id=install.id,
                use_case="Saved manual text",
                use_case_derived=False,
            )
            route = "profiles"
        else:
            row = ModelAssetInstall(
                name="Constructed LoRA",
                kind="lora",
                local_path="neutral",
                use_case="Saved manual text",
                use_case_derived=False,
                verified_at=utcnow(),
                manifest_json={"provider_description": description},
            )
            route = "model-assets"
        session.add(row)
        session.commit()
        return f"/api/{route}/{row.id}", row.id


@pytest.mark.parametrize("kind", ["profile", "lora"])
async def test_only_explicit_save_replaces_text_and_keeps_the_chosen_provenance(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    path, record_id = target(kind)
    calls: list[str] = []

    async def suggest(_services: Any, _session: Any, description: str) -> str:
        calls.append(description)
        return "Watercolor scenes."

    monkeypatch.setattr(summary_api, "suggest_managed_use_case_summary", suggest)
    with SessionLocal() as session:
        before = [
            session.scalar(select(func.count()).select_from(model)) for model in (Chat, Run, Job)
        ]
    response = await client.post(
        path + "/use-case-suggestion", json={"expected_use_case": "Saved manual text"}
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"suggestion": "Watercolor scenes."}
    assert calls == ["Watercolor landscapes"]
    with SessionLocal() as session:
        row = read_target(session, kind, record_id)
        assert row and row.use_case == "Saved manual text" and not row.use_case_derived
        assert [
            session.scalar(select(func.count()).select_from(model)) for model in (Chat, Run, Job)
        ] == before
    saved = await client.patch(
        path,
        json={
            "expected_use_case": "Saved manual text",
            "use_case": "Watercolor scenes.",
            "use_case_derived": True,
        },
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["use_case"] == "Watercolor scenes."
    assert saved.json()["use_case_derived"] is True
    edited = await client.patch(path, json={"use_case": "My edited description"})
    assert edited.status_code == 200
    assert edited.json()["use_case_derived"] is False


@pytest.mark.parametrize("kind", ["profile", "lora"])
@pytest.mark.parametrize("missing", ["description", "worker", "stale"])
async def test_unavailable_suggestions_and_stale_requests_leave_text_unchanged(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    missing: str,
) -> None:
    path, record_id = target(kind, "" if missing == "description" else "Watercolor landscapes")

    async def unavailable(_services: Any, _session: Any, description: str) -> str:
        assert missing == "worker", "Refused input reached inference"
        raise UseCaseSummaryError

    monkeypatch.setattr(summary_api, "suggest_managed_use_case_summary", unavailable)
    response = await client.post(
        path + "/use-case-suggestion",
        json={
            "expected_use_case": "Old text" if missing == "stale" else "Saved manual text",
        },
    )
    assert response.status_code == 409, response.text
    assert (
        response.json()["code"]
        == {
            "description": "use-case-description-unavailable",
            "worker": "use-case-suggestion-unavailable",
            "stale": "use-case-changed",
        }[missing]
    )
    assert "Watercolor" not in response.text
    with SessionLocal() as session:
        row = read_target(session, kind, record_id)
        assert row and row.use_case == "Saved manual text" and not row.use_case_derived


@pytest.mark.parametrize("kind", ["profile", "lora"])
async def test_changes_during_generation_and_before_save_refuse_stale_results(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    path, record_id = target(kind)

    async def changed(_services: Any, _session: Any, description: str) -> str:
        with SessionLocal() as session:
            row = read_target(session, kind, record_id)
            assert row is not None
            row.use_case = "A newer manual edit"
            session.commit()
        return "Watercolor scenes."

    monkeypatch.setattr(summary_api, "suggest_managed_use_case_summary", changed)
    response = await client.post(
        path + "/use-case-suggestion", json={"expected_use_case": "Saved manual text"}
    )
    assert response.status_code == 409
    assert response.json()["code"] == "use-case-changed"
    saved = await client.patch(
        path,
        json={
            "expected_use_case": "Saved manual text",
            "use_case": "Stale suggestion",
            "use_case_derived": True,
        },
    )
    assert saved.status_code == 409
    assert saved.json()["code"] == "use-case-changed"
    with SessionLocal() as session:
        row = read_target(session, kind, record_id)
        assert row and row.use_case == "A newer manual edit" and not row.use_case_derived


@pytest.mark.parametrize("kind", ["profile", "lora"])
async def test_provenance_cannot_be_changed_without_an_explicit_text_update(
    client: AsyncClient,
    kind: str,
) -> None:
    path, _ = target(kind)
    response = await client.patch(path, json={"use_case_derived": True})
    assert response.status_code == 422
    assert response.json()["code"] == "use-case-update-missing"


async def test_cancelling_a_suggestion_cancels_inference_without_a_saved_update(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, record_id = target("profile")
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    blocked = asyncio.Event()

    async def suggest(_services: Any, _session: Any, description: str) -> str:
        entered.set()
        try:
            await blocked.wait()
        finally:
            cancelled.set()
        raise AssertionError("Unexpected resume")

    monkeypatch.setattr(summary_api, "suggest_managed_use_case_summary", suggest)
    task = asyncio.create_task(
        client.post(path + "/use-case-suggestion", json={"expected_use_case": "Saved manual text"})
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(cancelled.wait(), timeout=5)
    with SessionLocal() as session:
        row = session.get(ModelProfile, record_id)
        assert row and row.use_case == "Saved manual text"


@pytest.mark.parametrize("kind", ["profile", "lora"])
async def test_confirming_text_holds_off_another_writer_until_commit(
    client: AsyncClient, kind: str
) -> None:
    _path, record_id = target(kind)
    with SessionLocal() as first:
        row = read_target(first, kind, record_id)
        assert row is not None
        summary_api.check_use_case_update(first, row, "Saved manual text")
        with SessionLocal() as competing:
            competing.execute(text("PRAGMA busy_timeout=0"))
            model = type(row)
            with pytest.raises(OperationalError, match="locked"):
                competing.execute(
                    update(model).where(model.id == record_id).values(use_case="Competing edit")
                )
        first.rollback()
    with SessionLocal() as session:
        saved = read_target(session, kind, record_id)
        assert saved is not None and saved.use_case == "Saved manual text"


async def test_connection_disconnect_releases_inference_through_the_application_middleware(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, record_id = target("profile")
    entered = asyncio.Event()
    disconnected = asyncio.Event()
    cancelled = asyncio.Event()

    async def suggest(_services: Any, _session: Any, description: str) -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("Unexpected resume")

    monkeypatch.setattr(summary_api, "suggest_managed_use_case_summary", suggest)
    request = client.build_request(
        "POST", path + "/use-case-suggestion", json={"expected_use_case": "Saved manual text"}
    )
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": request.url.path,
        "raw_path": request.url.raw_path,
        "query_string": b"",
        "root_path": "",
        "headers": [(name.lower(), value) for name, value in request.headers.raw],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        await anyio.lowlevel.checkpoint()
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": request.content, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        pass

    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        disconnected.set()
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        await asyncio.wait_for(task, timeout=2)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    with SessionLocal() as session:
        row = session.get(ModelProfile, record_id)
        assert row and row.use_case == "Saved manual text" and not row.use_case_derived
