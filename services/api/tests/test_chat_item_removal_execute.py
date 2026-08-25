from __future__ import annotations

import asyncio

from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import (
    Artifact,
    Chat,
    ChatItemRemovalReceipt,
    Job,
    Message,
    MessagePart,
    MessageReference,
    ResponseFeedback,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
)


def _request(message_id: str, revision_id: str, operation_key: str) -> dict[str, str]:
    return {
        "expected_message_id": message_id,
        "expected_revision_id": revision_id,
        "operation_key": operation_key,
    }


async def test_execute_detaches_only_target_payload_and_durably_replays(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        chat = Chat(id="chat_remove_item", title="Retained chat title")
        parent = Message(
            id="msg_remove_parent",
            chat=chat,
            role="user",
            status="complete",
        )
        target = Message(
            id="msg_remove_target",
            chat=chat,
            parent_id=parent.id,
            role="assistant",
            status="complete",
        )
        reply = Message(
            id="msg_remove_reply",
            chat=chat,
            parent_id=target.id,
            role="user",
            status="complete",
        )
        sibling = Message(
            id="msg_remove_sibling",
            chat=chat,
            parent_id=parent.id,
            role="assistant",
            status="complete",
        )
        retained_artifact = Artifact(
            id="artifact_remove_shared",
            sha256="3" * 64,
            kind="image",
            media_type="image/png",
            size_bytes=10,
            relative_path="artifacts/constructed-shared.png",
            metadata_json={},
            favorite=True,
        )
        session.add_all([chat, parent, target, reply, sibling, retained_artifact])
        session.flush()
        run = Run(
            id="run_remove_target",
            chat_id=chat.id,
            user_message_id=parent.id,
            assistant_message_id=target.id,
            status="complete",
            standalone_prompt="retained technical provenance",
            provenance_json={"retained": True},
            settings_json={},
        )
        session.add(run)
        session.flush()
        revision = ResponseRevision(
            id="rev_remove_target",
            message_id=target.id,
            run_id=run.id,
            sequence=1,
            status="complete",
        )
        session.add(revision)
        session.flush()
        target.active_response_revision_id = revision.id
        session.add_all(
            [
                MessagePart(
                    id="part_remove_target",
                    message_id=target.id,
                    position=0,
                    type="text",
                    text="constructed target payload",
                    metadata_json={},
                ),
                MessagePart(
                    id="part_remove_reply",
                    message_id=reply.id,
                    position=0,
                    type="text",
                    text="constructed reply payload",
                    metadata_json={},
                ),
                MessagePart(
                    id="part_remove_target_shared",
                    message_id=target.id,
                    position=1,
                    type="image",
                    artifact_id=retained_artifact.id,
                    metadata_json={},
                ),
                MessagePart(
                    id="part_remove_sibling",
                    message_id=sibling.id,
                    position=0,
                    type="text",
                    text="constructed sibling payload",
                    metadata_json={},
                ),
                MessagePart(
                    id="part_remove_sibling_shared",
                    message_id=sibling.id,
                    position=1,
                    type="image",
                    artifact_id=retained_artifact.id,
                    metadata_json={},
                ),
                ResponseRevisionPart(
                    id="revpart_remove_target",
                    response_revision_id=revision.id,
                    position=0,
                    type="text",
                    text="constructed revision payload",
                    metadata_json={},
                ),
                MessageReference(
                    id="msgref_remove_target",
                    message_id=target.id,
                    position=0,
                    reference_subject_id="subject_remove_target",
                    mention_slug="constructed-subject",
                    subject_name="Constructed subject",
                    subject_kind="person",
                    source="mention",
                    reference_asset_ids_json=[],
                    artifact_ids_json=[],
                ),
                ResponseFeedback(
                    id="fb_remove_target",
                    message_id=target.id,
                    response_revision_id=revision.id,
                    run_id=run.id,
                    rating="up",
                ),
            ]
        )
        session.commit()

    preview = await client.get("/api/messages/msg_remove_target/removal-impact")
    assert preview.status_code == 200
    revision_id = preview.json()["message_revision_id"]
    first = await client.post(
        "/api/messages/msg_remove_target/remove-content",
        json=_request("msg_remove_target", revision_id, "remove-target-once"),
    )

    assert first.status_code == 200
    first_body = first.json()
    assert first_body["replayed"] is False
    assert preview.json()["has_replies"] is True
    assert preview.json()["detached_message_part_count"] == 2
    assert preview.json()["detached_response_revision_part_count"] == 1
    assert preview.json()["detached_reference_count"] == 1
    assert preview.json()["retained_artifact_ids"] == ["artifact_remove_shared"]

    replay = await client.post(
        "/api/messages/msg_remove_target/remove-content",
        json=_request("msg_remove_target", revision_id, "remove-target-once"),
    )
    assert replay.status_code == 200
    assert replay.json() == {**first_body, "replayed": True}

    with SessionLocal() as session:
        stored = session.get(Message, "msg_remove_target")
        assert stored is not None
        assert stored.content_removed_at is not None
        assert stored.transcript_visible is True
        assert stored.active_response_revision_id == "rev_remove_target"
        assert session.get(Message, "msg_remove_reply").parent_id == stored.id  # type: ignore[union-attr]
        assert session.get(MessagePart, "part_remove_target") is None
        assert session.get(MessagePart, "part_remove_target_shared") is None
        assert session.get(ResponseRevisionPart, "revpart_remove_target") is None
        assert session.get(MessageReference, "msgref_remove_target") is None
        assert session.get(MessagePart, "part_remove_reply") is not None
        assert session.get(MessagePart, "part_remove_sibling") is not None
        assert session.get(MessagePart, "part_remove_sibling_shared") is not None
        retained = session.get(Artifact, "artifact_remove_shared")
        assert retained is not None
        assert retained.favorite is True
        assert session.get(ResponseRevision, "rev_remove_target") is not None
        assert session.get(ResponseFeedback, "fb_remove_target") is not None
        stored_run = session.get(Run, "run_remove_target")
        assert stored_run is not None
        assert stored_run.standalone_prompt == "retained technical provenance"
        assert stored_run.provenance_json == {"retained": True}
        receipts = list(session.scalars(select(ChatItemRemovalReceipt)))
        assert len(receipts) == 1
        assert not hasattr(receipts[0], "result_json")
        assert receipts[0].content_removed_at == stored.content_removed_at


async def test_stale_revision_and_reused_operation_key_refuse_without_partial_mutation(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        chat = Chat(id="chat_remove_stale", title="Stale authority")
        target = Message(
            id="msg_remove_stale",
            chat=chat,
            role="user",
            status="complete",
        )
        session.add_all(
            [
                chat,
                target,
                MessagePart(
                    id="part_remove_stale",
                    message=target,
                    position=0,
                    type="text",
                    text="constructed payload",
                    metadata_json={},
                ),
            ]
        )
        session.commit()

    preview = await client.get("/api/messages/msg_remove_stale/removal-impact")
    revision_id = preview.json()["message_revision_id"]
    with SessionLocal() as session:
        target = session.get(Message, "msg_remove_stale")
        assert target is not None
        target.status = "pending"
        session.commit()

    stale = await client.post(
        "/api/messages/msg_remove_stale/remove-content",
        json=_request("msg_remove_stale", revision_id, "remove-stale"),
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "message-revision-conflict"
    with SessionLocal() as session:
        assert session.get(MessagePart, "part_remove_stale") is not None
        assert session.get(Message, "msg_remove_stale").content_removed_at is None  # type: ignore[union-attr]

    fresh = await client.get("/api/messages/msg_remove_stale/removal-impact")
    fresh_revision_id = fresh.json()["message_revision_id"]
    executed = await client.post(
        "/api/messages/msg_remove_stale/remove-content",
        json=_request("msg_remove_stale", fresh_revision_id, "remove-stale"),
    )
    assert executed.status_code == 200
    conflict = await client.post(
        "/api/messages/msg_remove_stale/remove-content",
        json=_request("msg_remove_stale", "0" * 64, "remove-stale"),
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "operation-key-conflict"


async def test_execute_waits_for_chat_guard_then_revalidates_from_fresh_state(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        chat = Chat(id="chat_remove_revalidate", title="Revalidate")
        target = Message(
            id="msg_remove_revalidate",
            chat=chat,
            role="user",
            status="complete",
        )
        part = MessagePart(
            id="part_remove_revalidate",
            message=target,
            position=0,
            type="text",
            text="constructed payload",
            metadata_json={},
        )
        session.add_all([chat, target, part])
        session.commit()

    preview = await client.get("/api/messages/msg_remove_revalidate/removal-impact")
    revision_id = preview.json()["message_revision_id"]
    async with app.state.services.orchestrator.chat_guard("chat_remove_revalidate"):
        execution = asyncio.create_task(
            client.post(
                "/api/messages/msg_remove_revalidate/remove-content",
                json=_request(
                    "msg_remove_revalidate",
                    revision_id,
                    "remove-revalidate",
                ),
            )
        )
        await asyncio.sleep(0.03)
        assert execution.done() is False
        with SessionLocal() as session:
            target = session.get(Message, "msg_remove_revalidate")
            assert target is not None
            target.status = "pending"
            session.commit()

    response = await asyncio.wait_for(execution, timeout=2)
    assert response.status_code == 409
    assert response.json()["code"] == "message-revision-conflict"
    with SessionLocal() as session:
        target = session.get(Message, "msg_remove_revalidate")
        assert target is not None
        assert target.content_removed_at is None
        assert session.get(MessagePart, "part_remove_revalidate") is not None


async def test_active_run_and_edit_verification_jobs_block_removal(client: AsyncClient) -> None:
    with SessionLocal() as session:
        chat = Chat(id="chat_remove_busy", title="Busy chat")
        target = Message(
            id="msg_remove_busy",
            chat=chat,
            role="user",
            status="complete",
        )
        assistant = Message(
            id="msg_remove_busy_assistant",
            chat=chat,
            parent_id=target.id,
            role="assistant",
            status="pending",
        )
        session.add_all([chat, target, assistant])
        session.flush()
        run = Run(
            id="run_remove_busy",
            chat_id=chat.id,
            user_message_id=target.id,
            assistant_message_id=assistant.id,
            status="running",
            standalone_prompt="constructed",
            provenance_json={},
            settings_json={},
        )
        session.add_all(
            [
                run,
                MessagePart(
                    id="part_remove_busy",
                    message=target,
                    position=0,
                    type="text",
                    text="constructed payload",
                    metadata_json={},
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                Job(
                    id="job_remove_busy_run",
                    kind="chat",
                    status="queued",
                    run_id=run.id,
                    payload_json={},
                ),
                Job(
                    id="job_remove_busy_verify",
                    kind="edit_verify",
                    status="paused",
                    run_id=None,
                    payload_json={"chat_id": chat.id, "source_run_id": run.id},
                ),
            ]
        )
        other_chat = Chat(id="chat_remove_busy_other", title="Unrelated work")
        other_user = Message(
            id="msg_remove_busy_other_user",
            chat=other_chat,
            role="user",
            status="complete",
        )
        other_assistant = Message(
            id="msg_remove_busy_other_assistant",
            chat=other_chat,
            parent_id=other_user.id,
            role="assistant",
            status="pending",
        )
        session.add_all([other_chat, other_user, other_assistant])
        session.flush()
        other_run = Run(
            id="run_remove_busy_other",
            chat_id=other_chat.id,
            user_message_id=other_user.id,
            assistant_message_id=other_assistant.id,
            status="running",
            standalone_prompt="constructed unrelated work",
            provenance_json={},
            settings_json={},
        )
        session.add(other_run)
        session.flush()
        session.add(
            Job(
                id="job_remove_busy_other",
                kind="chat",
                status="queued",
                run_id=other_run.id,
                payload_json={},
            )
        )
        session.commit()

    preview = await client.get("/api/messages/msg_remove_busy/removal-impact")
    response = await client.post(
        "/api/messages/msg_remove_busy/remove-content",
        json=_request(
            "msg_remove_busy",
            preview.json()["message_revision_id"],
            "remove-busy",
        ),
    )
    assert response.status_code == 409
    assert response.json()["code"] == "chat-removal-active-work"
    assert response.json()["job_count"] == 2
    with SessionLocal() as session:
        assert session.get(MessagePart, "part_remove_busy") is not None
        assert (
            session.scalar(
                select(ChatItemRemovalReceipt).where(
                    ChatItemRemovalReceipt.operation_key == "remove-busy"
                )
            )
            is None
        )


async def test_identity_and_already_removed_refusals_are_typed(client: AsyncClient) -> None:
    with SessionLocal() as session:
        chat = Chat(id="chat_remove_refusals", title="Refusals")
        target = Message(
            id="msg_remove_refusals",
            chat=chat,
            role="user",
            status="complete",
        )
        chat.active_head_message_id = target.id
        session.add_all([chat, target])
        session.commit()

    preview = await client.get("/api/messages/msg_remove_refusals/removal-impact")
    revision_id = preview.json()["message_revision_id"]
    mismatch = await client.post(
        "/api/messages/msg_remove_refusals/remove-content",
        json=_request("msg_other", revision_id, "remove-mismatch"),
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["code"] == "message-identity-mismatch"

    first = await client.post(
        "/api/messages/msg_remove_refusals/remove-content",
        json=_request("msg_remove_refusals", revision_id, "remove-first"),
    )
    assert first.status_code == 200
    with SessionLocal() as session:
        stored_chat = session.get(Chat, "chat_remove_refusals")
        assert stored_chat is not None
        assert stored_chat.active_head_message_id == "msg_remove_refusals"
    already_removed = await client.post(
        "/api/messages/msg_remove_refusals/remove-content",
        json=_request("msg_remove_refusals", revision_id, "remove-second"),
    )
    assert already_removed.status_code == 409
    assert already_removed.json()["code"] == "message-already-removed"

    invalid_key = await client.post(
        "/api/messages/msg_remove_refusals/remove-content",
        json=_request("msg_remove_refusals", revision_id, "invalid key"),
    )
    assert invalid_key.status_code == 422
