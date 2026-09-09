from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message, Project, ResponseRevision, ResponseRevisionPart, Run


@pytest.mark.parametrize(
    "later_change", ["instructions", "ancestor_text", "response_revision", "vision_settings"]
)
async def test_edit_of_accepted_edit_keeps_the_context_shown_in_its_editor(
    app: FastAPI, client: AsyncClient, later_change: str
) -> None:
    orchestrator = app.state.services.orchestrator
    project_response = await client.post(
        "/api/projects",
        json={"name": "Inherited edit context", "instructions": "Use short paragraphs"},
    )
    assert project_response.status_code == 201
    project_id = project_response.json()["id"]
    chat_response = await client.post(
        "/api/chats", json={"title": "Inherited context", "project_id": project_id}
    )
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    async with app.state.services.scheduler.lease("primary"):
        ancestor_response = await client.post(
            f"/api/chats/{chat_id}/turns", json={"text": "Earlier question", "mode": "text"}
        )
        assert ancestor_response.status_code == 202
        ancestor = ancestor_response.json()
        with SessionLocal() as session:
            message = session.get(Message, ancestor["assistant_message"]["id"])
            assert message is not None
            revision = ResponseRevision(
                message_id=message.id,
                run_id=ancestor["run"]["id"],
                sequence=1,
                status="complete",
                parts=[ResponseRevisionPart(position=0, type="text", text="Earlier answer")],
            )
            session.add(revision)
            session.flush()
            orchestrator.select_response_revision(session, message.id, revision.id)
            message.status = "complete"
            session.commit()
        original_response = await client.post(
            f"/api/chats/{chat_id}/turns", json={"text": "Original question", "mode": "text"}
        )
        assert original_response.status_code == 202
        original = original_response.json()
        edited_response = await client.post(
            f"/api/messages/{original['user_message']['id']}/edits",
            json={"text": "First edited question", "idempotency_key": "first-inherited-edit"},
        )
        assert edited_response.status_code == 202, edited_response.text
        edited = edited_response.json()
        from local_lm.accepted_turn_context import accepted_context

        with SessionLocal() as session:
            source = session.get(Run, edited["run"]["id"])
            assert source is not None
            snapshot = accepted_context(session, source)
            assert snapshot is not None
            expected = [
                entry
                for entry in snapshot.messages
                if entry.source_message_id != source.user_message_id
            ]
            assert any(entry.content == "Earlier answer" for entry in expected)
            if later_change == "instructions":
                project = session.get(Project, project_id)
                assert project is not None
                project.instructions = "Use a numbered list"
            elif later_change == "ancestor_text":
                message = session.get(Message, ancestor["user_message"]["id"])
                assert message is not None
                message.parts[0].text = "A different earlier question"
            elif later_change == "response_revision":
                message = session.get(Message, ancestor["assistant_message"]["id"])
                assert message is not None
                revision = ResponseRevision(
                    message_id=message.id,
                    sequence=2,
                    status="complete",
                    parts=[ResponseRevisionPart(position=0, type="text", text="A later answer")],
                )
                session.add(revision)
                session.flush()
                orchestrator.select_response_revision(session, message.id, revision.id)
            else:
                chat = session.get(Chat, chat_id)
                assert chat is not None
                chat.vision_settings_json = {"include_prior_visual": False, "max_images": 1}
            session.commit()
        loaded = await client.get(f"/api/messages/{edited['user_message']['id']}/edit-source")
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["context_messages"] == [
            {"role": entry.role, "content": entry.content} for entry in expected
        ]
        accepted = await client.post(
            f"/api/messages/{edited['user_message']['id']}/edits",
            json={
                "text": "Second edited question",
                "idempotency_key": "second-inherited-edit",
                "source_snapshot_sha256": loaded.json()["source_snapshot_sha256"],
            },
        )
        assert accepted.status_code == 202, accepted.text
        with SessionLocal() as session:
            run = session.get(Run, accepted.json()["run"]["id"])
            assert run is not None
            current = accepted_context(session, run)
            assert current is not None
            assert [
                entry
                for entry in current.messages
                if entry.source_message_id != run.user_message_id
            ] == expected
            assert current.messages[-1].content == "Second edited question"
            assert current.vision_settings == snapshot.vision_settings
            assert current.vision_sampling == snapshot.vision_sampling
            assert current.vision_bridge_max_tokens == snapshot.vision_bridge_max_tokens
            assert current.source_run_id == edited["run"]["id"]


@pytest.mark.parametrize("placement", ["ancestor", "current"])
async def test_inherited_edit_retains_frozen_context_media_and_honors_cleared_inputs(
    app: FastAPI, client: AsyncClient, placement: str
) -> None:
    import base64

    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    artifacts = []
    for index in range(2):
        upload = await client.post(
            "/api/artifacts",
            files={"file": ("neutral.png", image + bytes([index]), "image/png")},
        )
        assert upload.status_code == 201, upload.text
        artifacts.append(upload.json()["id"])
    chat = (await client.post("/api/chats", json={"title": "Inherited media"})).json()
    async with app.state.services.scheduler.lease("primary"):
        ancestor = None
        if placement == "ancestor":
            response = await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={
                    "text": "Earlier illustration",
                    "mode": "text",
                    "input_artifact_ids": [artifacts[0]],
                },
            )
            assert response.status_code == 202, response.text
            ancestor = response.json()
        original = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Original description",
                "mode": "text",
                "input_artifact_ids": [artifacts[0]] if placement == "current" else [],
            },
        )
        assert original.status_code == 202, original.text
        first = await client.post(
            f"/api/messages/{original.json()['user_message']['id']}/edits",
            json={"text": "First edited description", "idempotency_key": "media-first-edit"},
        )
        assert first.status_code == 202, first.text
        from local_lm.accepted_turn_context import accepted_context

        with SessionLocal() as session:
            source = session.get(Run, first.json()["run"]["id"])
            assert source is not None
            snapshot = accepted_context(session, source)
            assert snapshot is not None
            assert snapshot.visual_artifact_ids == [artifacts[0]]
            if ancestor is not None:
                message = session.get(Message, ancestor["user_message"]["id"])
                assert message is not None
                for part in message.parts:
                    if part.artifact_id == artifacts[0]:
                        part.artifact_id = artifacts[1]
                session.commit()
        source_view = await client.get(
            f"/api/messages/{first.json()['user_message']['id']}/edit-source"
        )
        assert source_view.status_code == 200, source_view.text
        assert [item["id"] for item in source_view.json()["context_visual_artifacts"]] == (
            [artifacts[0]] if placement == "ancestor" else []
        )
        second = await client.post(
            f"/api/messages/{first.json()['user_message']['id']}/edits",
            json={
                "text": "Second edited description",
                "idempotency_key": "media-second-edit",
                "input_artifact_ids": [],
            },
        )
        assert second.status_code == 202, second.text
        with SessionLocal() as session:
            run = session.get(Run, second.json()["run"]["id"])
            assert run is not None
            current = accepted_context(session, run)
            assert current is not None
            assert current.input_artifact_ids == []
            assert current.visual_artifact_ids == (
                [artifacts[0]] if placement == "ancestor" else []
            )
            assert (artifacts[0] in current.artifact_ids) is (placement == "ancestor")
            assert artifacts[1] not in current.artifact_ids
