from __future__ import annotations

import asyncio
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta

from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from local_lm.db import SessionLocal
from local_lm.domain import MessageStatus, RunStatus
from local_lm.models import Artifact, Chat, Message, Run
from local_lm.orchestrator import ConversationOrchestrator


async def _wait_for_run(client: AsyncClient, run_id: str) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            assert run["status"] == "complete", run
            return run
        await asyncio.sleep(0.03)
    raise AssertionError("run did not complete")


def _output_artifact_id(run: dict) -> str:  # type: ignore[type-arg]
    return str(run["provenance_json"]["outputs"][0]["artifact_id"])


async def _image_turn(
    client: AsyncClient,
    chat_id: str,
    text: str,
    **payload: object,
) -> tuple[dict, dict]:  # type: ignore[type-arg]
    response = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={"text": text, "mode": "image", **payload},
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    run = await _wait_for_run(client, accepted["run"]["id"])
    return accepted, run


async def test_turn_inputs_are_durable_message_parts_and_context(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("reference.png", b"reference-image", "image/png")},
    )
    assert uploaded.status_code == 201
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Durable input"})).json()

    accepted, completed = await _image_turn(
        client,
        chat["id"],
        "Restyle this reference",
        input_artifact_ids=[artifact_id],
    )
    input_part = next(
        part
        for part in accepted["user_message"]["parts"]
        if part["metadata_json"].get("input_reference")
    )
    assert input_part["type"] == "image"
    assert input_part["artifact_id"] == artifact_id
    assert input_part["metadata_json"]["input_reference_source"] == "explicit"

    with SessionLocal() as session:
        run = session.get(Run, completed["id"])
        artifact = session.get(Artifact, artifact_id)
        assert run and artifact
        artifact.metadata_json = {
            **artifact.metadata_json,
            "unreferenced_at": (datetime.now(UTC) - timedelta(days=31)).isoformat(),
        }
        context = ConversationOrchestrator._context_messages(session, run)
        session.commit()
    assert context[0]["content"] == "Restyle this reference\n[Attached image: reference.png]"

    cleanup = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert cleanup.status_code == 200
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 200


async def test_legacy_provenance_inputs_remain_live_and_reconstruct_context(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("legacy.png", b"legacy-image", "image/png")},
    )
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Legacy input"})).json()
    _accepted, completed = await _image_turn(
        client,
        chat["id"],
        "Use this old reference",
        input_artifact_ids=[artifact_id],
    )

    with SessionLocal() as session:
        run = session.get(Run, completed["id"])
        assert run
        user_message = session.scalar(
            select(Message)
            .options(selectinload(Message.parts))
            .where(Message.id == run.user_message_id)
        )
        artifact = session.get(Artifact, artifact_id)
        assert user_message and artifact
        for part in list(user_message.parts):
            if part.metadata_json.get("input_reference"):
                session.delete(part)
        artifact.metadata_json = {
            **artifact.metadata_json,
            "unreferenced_at": (datetime.now(UTC) - timedelta(days=31)).isoformat(),
        }
        session.commit()

    with SessionLocal() as session:
        run = session.get(Run, completed["id"])
        assert run
        assert ConversationOrchestrator.input_artifact_ids_for_run(session, run) == [artifact_id]
        assert ConversationOrchestrator._context_messages(session, run)[0]["content"] == (
            "Use this old reference\n[Attached image: legacy.png]"
        )

    cleanup = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert cleanup.status_code == 200
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 200


async def test_prior_image_resolution_ignores_newer_sibling_branch(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Branched images"})).json()
    apple, apple_run = await _image_turn(
        client,
        chat["id"],
        "Make an image of a red apple",
    )
    apple_id = _output_artifact_id(apple_run)

    sibling = await client.post(
        f"/api/messages/{apple['user_message']['id']}/branch",
        json={"text": "Make an image of a blue house", "mode": "image"},
    )
    assert sibling.status_code == 202
    sibling_run = await _wait_for_run(client, sibling.json()["run"]["id"])
    sibling_id = _output_artifact_id(sibling_run)
    assert sibling_id != apple_id

    follow_up = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Make it green",
            "mode": "auto",
            "parent_message_id": apple["assistant_message"]["id"],
        },
    )
    assert follow_up.status_code == 202, follow_up.text
    run = follow_up.json()["run"]
    assert run["operation"] == "image_to_image"
    assert run["provenance_json"]["input_artifact_ids"] == [apple_id]
    assert run["standalone_prompt"] == (
        "Make an image of a red apple. Follow-up instruction: Make it green"
    )
    durable = next(
        part
        for part in follow_up.json()["user_message"]["parts"]
        if part["metadata_json"].get("input_reference")
    )
    assert durable["artifact_id"] == apple_id
    assert durable["metadata_json"]["input_reference_source"] == "ancestor"


async def test_prior_image_resolution_skips_failed_and_cancelled_ancestors(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Failed media ancestry"})).json()
    first, first_run = await _image_turn(client, chat["id"], "Make a red apple")
    first_id = _output_artifact_id(first_run)
    second, second_run = await _image_turn(client, chat["id"], "Make a blue house")
    third, third_run = await _image_turn(client, chat["id"], "Make a yellow boat")

    with SessionLocal() as session:
        failed = session.get(Run, second_run["id"])
        cancelled = session.get(Run, third_run["id"])
        assert failed and cancelled
        failed.status = RunStatus.FAILED.value
        cancelled.status = RunStatus.CANCELLED.value
        failed_message = session.get(Message, failed.assistant_message_id)
        cancelled_message = session.get(Message, cancelled.assistant_message_id)
        assert failed_message and cancelled_message
        failed_message.status = MessageStatus.FAILED.value
        cancelled_message.status = MessageStatus.CANCELLED.value
        session.commit()

    follow_up = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Make it green",
            "mode": "auto",
            "parent_message_id": third["assistant_message"]["id"],
        },
    )
    assert follow_up.status_code == 202, follow_up.text
    run = follow_up.json()["run"]
    assert run["operation"] == "image_to_image"
    assert run["provenance_json"]["input_artifact_ids"] == [first_id]
    assert run["standalone_prompt"] == ("Make a red apple. Follow-up instruction: Make it green")
    assert second["assistant_message"]["id"] != first["assistant_message"]["id"]


async def test_project_round_trip_preserves_durable_and_legacy_turn_inputs(
    client: AsyncClient,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Input references"})).json()
    durable_upload = await client.post(
        "/api/artifacts",
        files={"file": ("durable.png", b"durable-project-input", "image/png")},
    )
    legacy_upload = await client.post(
        "/api/artifacts",
        files={"file": ("legacy.png", b"legacy-project-input", "image/png")},
    )
    durable_id = durable_upload.json()["id"]
    legacy_id = legacy_upload.json()["id"]
    durable_chat = (
        await client.post(
            "/api/chats",
            json={"title": "Durable export input", "project_id": project["id"]},
        )
    ).json()
    legacy_chat = (
        await client.post(
            "/api/chats",
            json={"title": "Legacy export input", "project_id": project["id"]},
        )
    ).json()
    _durable, _durable_run = await _image_turn(
        client,
        durable_chat["id"],
        "Use the durable image",
        input_artifact_ids=[durable_id],
    )
    _legacy, legacy_run = await _image_turn(
        client,
        legacy_chat["id"],
        "Use the legacy image",
        input_artifact_ids=[legacy_id],
    )
    with SessionLocal() as session:
        run = session.get(Run, legacy_run["id"])
        assert run
        message = session.scalar(
            select(Message)
            .options(selectinload(Message.parts))
            .where(Message.id == run.user_message_id)
        )
        assert message
        for part in list(message.parts):
            if part.metadata_json.get("input_reference"):
                session.delete(part)
        session.commit()

    exported = await client.post(f"/api/projects/{project['id']}/export")
    assert exported.status_code == 201, exported.text
    archive_response = await client.get(exported.json()["url"])
    assert archive_response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert {durable_id, legacy_id}.issubset({artifact["id"] for artifact in manifest["artifacts"]})
    durable_record = next(
        chat for chat in manifest["chats"] if chat["title"] == "Durable export input"
    )
    legacy_record = next(
        chat for chat in manifest["chats"] if chat["title"] == "Legacy export input"
    )
    assert any(
        part.get("artifact_id") == durable_id
        and part["metadata_json"].get("input_reference") is True
        for message in durable_record["messages"]
        for part in message["parts"]
    )
    assert not any(
        part.get("artifact_id") == legacy_id
        for message in legacy_record["messages"]
        for part in message["parts"]
    )

    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": (
                "input-references.lm-atelier.zip",
                archive_response.content,
                "application/zip",
            )
        },
    )
    assert imported.status_code == 201, imported.text
    with SessionLocal() as session:
        imported_chats = session.scalars(
            select(Chat).where(Chat.project_id == imported.json()["id"])
        ).all()
        assert len(imported_chats) == 2
        imported_runs = session.scalars(
            select(Run)
            .where(Run.chat_id.in_([chat.id for chat in imported_chats]))
            .order_by(Run.created_at)
        ).all()
        assert len(imported_runs) == 2
        runs_by_prompt = {run.standalone_prompt: run for run in imported_runs}
        imported_durable = runs_by_prompt["Use the durable image"]
        imported_legacy = runs_by_prompt["Use the legacy image"]

        durable_message = session.scalar(
            select(Message)
            .options(selectinload(Message.parts))
            .where(Message.id == imported_durable.user_message_id)
        )
        legacy_message = session.scalar(
            select(Message)
            .options(selectinload(Message.parts))
            .where(Message.id == imported_legacy.user_message_id)
        )
        assert durable_message and legacy_message
        assert any(
            part.artifact_id == durable_id and part.metadata_json.get("input_reference") is True
            for part in durable_message.parts
        )
        assert not any(
            part.metadata_json.get("input_reference") is True for part in legacy_message.parts
        )
        assert ConversationOrchestrator.input_artifact_ids_for_run(session, imported_legacy) == [
            legacy_id
        ]
