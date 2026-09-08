from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message, Project, Run, WorkPlan


async def _source(client: AsyncClient) -> tuple[dict[str, Any], dict[str, Any]]:
    project = (
        await client.post(
            "/api/projects", json={"name": "Edit context", "instructions": "Be brief"}
        )
    ).json()
    chat = (
        await client.post("/api/chats", json={"title": "Bound source", "project_id": project["id"]})
    ).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Original question", "mode": "text"}
    )
    assert response.status_code == 202, response.text
    source = response.json()
    url = f"/api/messages/{source['user_message']['id']}/edit-source"
    loaded = await client.get(url)
    assert loaded.status_code == 200, loaded.text
    return source, loaded.json()


def _change_source(source: dict[str, Any], field: str) -> None:
    with SessionLocal() as session:
        run = session.get(Run, source["run"]["id"])
        message = session.get(Message, source["user_message"]["id"])
        assert run is not None and message is not None
        if field == "text":
            message.parts[0].text = "Changed source question"
        elif field == "settings":
            run.settings_json = {**run.settings_json, "temperature": 0.17}
        else:
            chat = session.get(Chat, run.chat_id)
            assert chat is not None
            project = session.get(Project, chat.project_id)
            assert project is not None
            project.instructions = "Use a numbered list"
        session.commit()


@pytest.mark.parametrize("field", ["text", "settings", "context"])
async def test_edit_refuses_stale_source_and_allows_refreshed_retry(
    app: FastAPI, client: AsyncClient, field: str
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source, editor = await _source(client)
        url = f"/api/messages/{source['user_message']['id']}"
        payload = {
            "text": "Edited question",
            "idempotency_key": "source-bound-edit",
            "source_run_id": editor["source_run_id"],
            "source_snapshot_sha256": editor["source_snapshot_sha256"],
        }
        _change_source(source, field)
        refused = await client.post(url + "/edits", json=payload)
        assert refused.status_code == 409, refused.text
        with SessionLocal() as session:
            assert len(list(session.scalars(select(WorkPlan)))) == 1
        refreshed = (await client.get(url + "/edit-source")).json()
        assert refreshed["source_snapshot_sha256"] != editor["source_snapshot_sha256"]
        accepted = await client.post(
            url + "/edits",
            json={**payload, "source_snapshot_sha256": refreshed["source_snapshot_sha256"]},
        )
        assert accepted.status_code == 202, accepted.text


async def test_edit_records_source_digest_and_replays_after_source_changes(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source, editor = await _source(client)
        url = f"/api/messages/{source['user_message']['id']}/edits"
        payload = {
            "text": "Accepted edit",
            "idempotency_key": "accepted-source",
            "source_snapshot_sha256": editor["source_snapshot_sha256"],
        }
        accepted = await client.post(url, json=payload)
        assert accepted.status_code == 202, accepted.text
        from local_lm.accepted_turn_context import accepted_context

        with SessionLocal() as session:
            run = session.get(Run, accepted.json()["run"]["id"])
            assert run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None
            assert snapshot.source_snapshot_sha256 == editor["source_snapshot_sha256"]
            assert (
                run.provenance_json["edit_source"]["source_snapshot_sha256"]
                == (editor["source_snapshot_sha256"])
            )
        _change_source(source, "text")
        replay = await client.post(url, json=payload)
        assert replay.status_code == 202, replay.text
        assert replay.json()["work_plan_id"] == accepted.json()["work_plan_id"]
        conflict = await client.post(url, json={**payload, "source_snapshot_sha256": "0" * 64})
        assert conflict.status_code == 409, conflict.text


@pytest.mark.parametrize("provide_digest", [False, True], ids=["implicit", "explicit"])
async def test_edit_rechecks_source_after_await_before_acceptance(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, provide_digest: bool
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source, editor = await _source(client)
        original = app.state.services.engines.settings_for_role
        changed = False

        async def change_during_settings(*args: Any, **kwargs: Any) -> Any:
            nonlocal changed
            if not changed:
                changed = True
                _change_source(source, "text")
            return await original(*args, **kwargs)

        monkeypatch.setattr(app.state.services.engines, "settings_for_role", change_during_settings)
        payload = {"text": "Concurrent edit", "idempotency_key": "concurrent-source"}
        if provide_digest:
            payload["source_snapshot_sha256"] = editor["source_snapshot_sha256"]
        refused = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits", json=payload
        )
        assert changed
        assert refused.status_code == 409, refused.text
        with SessionLocal() as session:
            assert len(list(session.scalars(select(WorkPlan)))) == 1
            assert len(list(session.scalars(select(Run)))) == 1


async def test_source_initializer_refuses_a_mixed_view_across_await(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source, _ = await _source(client)
        original = app.state.services.engines.settings_for_role

        async def change_during_settings(*args: Any, **kwargs: Any) -> Any:
            _change_source(source, "settings")
            return await original(*args, **kwargs)

        monkeypatch.setattr(app.state.services.engines, "settings_for_role", change_during_settings)
        refused = await client.get(f"/api/messages/{source['user_message']['id']}/edit-source")
        assert refused.status_code == 409, refused.text


async def test_source_digest_ignores_original_progress_and_later_branch(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        source, editor = await _source(client)
        with SessionLocal() as session:
            run = session.get(Run, source["run"]["id"])
            assert run is not None
            run.status = "running"
            session.commit()
        later = await client.post(
            f"/api/chats/{editor['chat_id']}/turns",
            json={"text": "Keep this later question", "mode": "text"},
        )
        assert later.status_code == 202, later.text
        refreshed = await client.get(f"/api/messages/{source['user_message']['id']}/edit-source")
        assert refreshed.status_code == 200, refreshed.text
        assert refreshed.json()["source_snapshot_sha256"] == editor["source_snapshot_sha256"]


async def test_source_digest_binds_response_revision_even_when_text_is_identical(
    app: FastAPI, client: AsyncClient
) -> None:
    from local_lm.models import ResponseRevision, ResponseRevisionPart

    async with app.state.services.scheduler.lease("primary"):
        first, _ = await _source(client)
        with SessionLocal() as session:
            message = session.get(Message, first["assistant_message"]["id"])
            assert message is not None
            message.status = "complete"
            message.parts[0].text = "The same neutral answer"
            revisions = [
                ResponseRevision(
                    message_id=message.id,
                    sequence=sequence,
                    status="complete",
                    parts=[
                        ResponseRevisionPart(
                            position=0, type="text", text="The same neutral answer"
                        )
                    ],
                )
                for sequence in (10, 11)
            ]
            session.add_all(revisions)
            session.flush()
            revision_ids = [revision.id for revision in revisions]
            message.active_response_revision_id = revision_ids[0]
            session.commit()
        second = await client.post(
            f"/api/chats/{first['run']['chat_id']}/turns",
            json={"text": "Follow-up question", "mode": "text"},
        )
        assert second.status_code == 202, second.text
        source_id = second.json()["user_message"]["id"]
        url = f"/api/messages/{source_id}"
        before = (await client.get(url + "/edit-source")).json()
        with SessionLocal() as session:
            message = session.get(Message, first["assistant_message"]["id"])
            assert message is not None
            message.active_response_revision_id = revision_ids[1]
            session.commit()
        after = (await client.get(url + "/edit-source")).json()
        assert before["context_messages"] == after["context_messages"]
        assert before["source_snapshot_sha256"] != after["source_snapshot_sha256"]
        refused = await client.post(
            url + "/edits",
            json={
                "text": "Edited follow-up",
                "idempotency_key": "revision-bound",
                "source_snapshot_sha256": before["source_snapshot_sha256"],
            },
        )
        assert refused.status_code == 409, refused.text


async def test_source_digest_ignores_input_favorite_decoration(
    app: FastAPI, client: AsyncClient
) -> None:
    from local_lm.models import Artifact

    uploaded = await client.post(
        "/api/artifacts", files={"file": ("note.txt", b"A neutral note", "text/plain")}
    )
    assert uploaded.status_code == 201, uploaded.text
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Stable input identity"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": "Read this note", "mode": "text", "input_artifact_ids": [artifact_id]},
        )
        assert source.status_code == 202, source.text
        url = f"/api/messages/{source.json()['user_message']['id']}/edit-source"
        before = (await client.get(url)).json()
        with SessionLocal() as session:
            artifact = session.get(Artifact, artifact_id)
            assert artifact is not None
            artifact.favorite = not artifact.favorite
            session.commit()
        after = (await client.get(url)).json()
        assert before["input_artifacts"][0]["favorite"] != after["input_artifacts"][0]["favorite"]
        assert before["source_snapshot_sha256"] == after["source_snapshot_sha256"]
