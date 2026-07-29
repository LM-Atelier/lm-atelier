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


def test_image_edit_media_prompt_preserves_unrequested_details() -> None:
    run = Run(
        operation="image_to_image",
        standalone_prompt="Replace the jacket with a green coat",
        provenance_json={
            "image_edit": {"policy": "preserve_unrequested_details_v1"},
        },
    )

    assert ConversationOrchestrator._media_prompt(run) == (
        "Apply the requested edit visibly to the supplied image. Preserve areas "
        "that the edit does not affect. Keep each person's facial identity, hair, "
        "skin tone, body proportions, and pose unless the request explicitly changes "
        "them. Do not simply reproduce the source unchanged. Requested edit: "
        "Replace the jacket with a green coat"
    )


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
    assert accepted["run"]["settings_json"]["denoise"] == 0.82
    assert accepted["run"]["provenance_json"]["image_edit"] == {
        "policy": "preserve_unrequested_details_v1",
        "default_change_strength_applied": True,
        "strength": {
            "mode": "auto",
            "parameter": "denoise",
            "value": 0.82,
            "applied_bounds": {"minimum": 0.0, "maximum": 1.0},
            "reason_codes": ["global_transformation"],
            "reused": False,
            "scope": "global",
            "confidence": "high",
            "estimator_version": "prompt-edit-strength-v1",
        },
    }
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


async def test_explicit_image_edit_strength_remains_authoritative(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("reference.png", b"explicit-edit-image", "image/png")},
    )
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Explicit edit"})).json()

    accepted, _completed = await _image_turn(
        client,
        chat["id"],
        "Replace the jacket",
        input_artifact_ids=[artifact_id],
        settings={"denoise": 0.62},
    )

    assert accepted["run"]["settings_json"]["denoise"] == 0.62
    assert accepted["run"]["provenance_json"]["image_edit"] == {
        "policy": "preserve_unrequested_details_v1",
        "default_change_strength_applied": False,
        "strength": {
            "mode": "manual",
            "parameter": "denoise",
            "value": 0.62,
            "applied_bounds": {"minimum": 0.0, "maximum": 1.0},
            "reason_codes": ["explicit_value"],
            "reused": False,
            "source_scope": "turn",
        },
    }


async def test_explicit_image_source_does_not_inherit_prior_media_prompt(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Explicit source isolation"})).json()
    await _image_turn(client, chat["id"], "Add a small blue square badge")
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("original.png", b"original-source-image", "image/png")},
    )
    artifact_id = uploaded.json()["id"]
    prompt = (
        "Make the studio lighting only slightly warmer and subtly brighter. "
        "Keep everything else unchanged."
    )

    accepted, _completed = await _image_turn(
        client,
        chat["id"],
        prompt,
        input_artifact_ids=[artifact_id],
    )

    run = accepted["run"]
    assert run["standalone_prompt"] == prompt
    assert run["provenance_json"]["input_artifact_ids"] == [artifact_id]
    assert "blue square badge" not in run["standalone_prompt"]


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


async def test_prior_image_follow_up_uses_prompt_aware_strength(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Prompt-aware follow-up"})).json()
    first, first_run = await _image_turn(client, chat["id"], "Make an image of a red apple")

    follow_up = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Make it green",
            "mode": "auto",
            "parent_message_id": first["assistant_message"]["id"],
        },
    )

    assert follow_up.status_code == 202, follow_up.text
    run = follow_up.json()["run"]
    assert run["operation"] == "image_to_image"
    assert run["provenance_json"]["input_artifact_ids"] == [_output_artifact_id(first_run)]
    assert run["settings_json"]["denoise"] == 0.5
    assert run["provenance_json"]["image_edit"]["strength"] == {
        "mode": "auto",
        "parameter": "denoise",
        "value": 0.5,
        "applied_bounds": {"minimum": 0.0, "maximum": 1.0},
        "reason_codes": ["localized_change"],
        "reused": False,
        "scope": "localized",
        "confidence": "medium",
        "estimator_version": "prompt-edit-strength-v1",
    }


async def test_common_natural_language_follow_ups_use_latest_image(
    client: AsyncClient,
) -> None:
    cases = [
        ("Make her top red", 0.5, "localized"),
        ("Increase the brightness", 0.38, "minimal"),
    ]
    for prompt, expected_strength, expected_scope in cases:
        chat = (
            await client.post(
                "/api/chats",
                json={"title": "Natural edit follow-up"},
            )
        ).json()
        first, first_run = await _image_turn(
            client,
            chat["id"],
            "Make an image of a person",
        )
        source_id = _output_artifact_id(first_run)

        follow_up = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": prompt,
                "mode": "auto",
                "parent_message_id": first["assistant_message"]["id"],
            },
        )

        assert follow_up.status_code == 202, follow_up.text
        run = follow_up.json()["run"]
        assert run["operation"] == "image_to_image"
        assert run["provenance_json"]["input_artifact_ids"] == [source_id]
        assert run["standalone_prompt"] == (
            f"Make an image of a person. Follow-up instruction: {prompt}"
        )
        assert run["settings_json"]["denoise"] == expected_strength
        assert run["provenance_json"]["image_edit"]["strength"]["scope"] == (expected_scope)


async def test_natural_language_edit_finds_recent_image_after_text_turn(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Recent image ancestry"})).json()
    image, image_run = await _image_turn(client, chat["id"], "Make an image of a person")
    source_id = _output_artifact_id(image_run)

    text = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Give me a short caption",
            "mode": "text",
            "parent_message_id": image["assistant_message"]["id"],
        },
    )
    assert text.status_code == 202, text.text
    await _wait_for_run(client, text.json()["run"]["id"])

    edit = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Make her top red",
            "mode": "auto",
            "parent_message_id": text.json()["assistant_message"]["id"],
        },
    )

    assert edit.status_code == 202, edit.text
    run = edit.json()["run"]
    assert run["operation"] == "image_to_image"
    assert run["provenance_json"]["input_artifact_ids"] == [source_id]
    assert run["settings_json"]["denoise"] == 0.5


async def test_text_to_image_keeps_workflow_default_strength(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Text to image default"})).json()

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Make an image of a red apple", "mode": "image"},
    )

    assert response.status_code == 202, response.text
    run = response.json()["run"]
    assert run["operation"] == "text_to_image"
    assert run["settings_json"]["denoise"] == 1
    assert run["provenance_json"]["image_edit"] is None


async def test_ordered_image_edit_uses_the_shared_strength_resolver(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("reference.png", b"ordered-edit-image", "image/png")},
    )
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Ordered edit"})).json()
    payload = {
        "text": "Make an image with the person in a new suit, then animate it into a video",
        "mode": "auto",
        "input_artifact_ids": [artifact_id],
    }
    preview = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert preview.status_code == 409

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={**payload, "confirm_media": True},
    )

    assert response.status_code == 202, response.text
    run = response.json()["run"]
    assert run["operation"] == "image_to_image"
    assert run["settings_json"]["denoise"] == 0.66
    assert run["provenance_json"]["image_edit"]["strength"]["scope"] == "replacement"


async def test_regeneration_reuses_auto_image_edit_strength(client: AsyncClient) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("reference.png", b"regenerated-edit-image", "image/png")},
    )
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Regenerated edit"})).json()
    accepted, _completed = await _image_turn(
        client,
        chat["id"],
        "Replace the jacket with a green coat",
        input_artifact_ids=[artifact_id],
    )

    response = await client.post(
        f"/api/messages/{accepted['assistant_message']['id']}/regenerate",
        json={"settings": {}},
    )

    assert response.status_code == 202, response.text
    run = response.json()["run"]
    assert run["settings_json"]["denoise"] == 0.66
    strength = run["provenance_json"]["image_edit"]["strength"]
    assert strength["mode"] == "auto"
    assert strength["scope"] == "replacement"
    assert strength["reason_codes"] == ["inherited_auto_value"]
    assert strength["reused"] is True

    await _wait_for_run(client, run["id"])
    overridden = await client.post(
        f"/api/messages/{accepted['assistant_message']['id']}/regenerate",
        json={"settings": {"denoise": 0.61}},
    )
    assert overridden.status_code == 202, overridden.text
    overridden_run = overridden.json()["run"]
    assert overridden_run["settings_json"]["denoise"] == 0.61
    overridden_strength = overridden_run["provenance_json"]["image_edit"]["strength"]
    assert overridden_strength["mode"] == "manual"
    assert overridden_strength["source_scope"] == "turn"
    assert overridden_strength["reused"] is False


async def test_edit_and_branch_reuses_inherited_auto_strength(client: AsyncClient) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("reference.png", b"branched-edit-image", "image/png")},
    )
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Branched edit"})).json()
    accepted, _completed = await _image_turn(
        client,
        chat["id"],
        "Replace the jacket with a green coat",
        input_artifact_ids=[artifact_id],
    )

    branched = await client.post(
        f"/api/messages/{accepted['user_message']['id']}/branch",
        json={"text": "Replace the jacket with a blue coat"},
    )

    assert branched.status_code == 202, branched.text
    run = branched.json()["run"]
    assert run["settings_json"]["denoise"] == 0.66
    strength = run["provenance_json"]["image_edit"]["strength"]
    assert strength["mode"] == "auto"
    assert strength["scope"] == "replacement"
    assert strength["reason_codes"] == ["inherited_auto_value"]
    assert strength["reused"] is True


async def test_chat_auto_mode_overrides_inherited_numeric_edit_strength(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("reference.png", b"auto-mode-edit-image", "image/png")},
    )
    artifact_id = uploaded.json()["id"]
    chat = (await client.post("/api/chats", json={"title": "Auto edit mode"})).json()
    updated = await client.patch(
        f"/api/chats/{chat['id']}",
        json={
            "generation_settings_json": {
                "image": {
                    "denoise": 0.41,
                    "_image_edit_strength_mode": "auto",
                }
            }
        },
    )
    assert updated.status_code == 200

    accepted, _completed = await _image_turn(
        client,
        chat["id"],
        "Replace the jacket with a green coat",
        input_artifact_ids=[artifact_id],
    )

    assert accepted["run"]["settings_json"]["denoise"] == 0.66
    strength = accepted["run"]["provenance_json"]["image_edit"]["strength"]
    assert strength["mode"] == "auto"
    assert strength["reason_codes"] == ["subject_replacement", "explicit_auto_mode"]


async def test_text_to_image_ignores_chat_edit_strength(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Fresh image strength"})).json()
    updated = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"generation_settings_json": {"image": {"denoise": 0.41}}},
    )
    assert updated.status_code == 200

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Make an image of a red apple", "mode": "image"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["run"]["operation"] == "text_to_image"
    assert response.json()["run"]["settings_json"]["denoise"] == 1
