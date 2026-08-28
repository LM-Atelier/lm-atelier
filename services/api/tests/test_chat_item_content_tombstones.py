from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy.exc import IntegrityError as SAIntegrityError

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import (
    ArtifactKind,
    MessageRole,
    MessageStatus,
    Operation,
    PartType,
    RunStatus,
    utcnow,
)
from local_lm.models import (
    Artifact,
    Chat,
    Message,
    MessagePart,
    MessageReference,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
)
from local_lm.orchestrator import ConversationOrchestrator


def _message(chat: Chat, message_id: str, *, removed: bool = False) -> Message:
    return Message(
        id=message_id,
        chat=chat,
        role=MessageRole.USER.value,
        status=MessageStatus.COMPLETE.value,
        content_removed_at=utcnow() if removed else None,
    )


def _reference(message_id: str, reference_id: str) -> MessageReference:
    return MessageReference(
        id=reference_id,
        message_id=message_id,
        position=0,
        reference_subject_id=f"subject_{reference_id}",
        mention_slug=f"mention-{reference_id}",
        subject_name="Reference subject",
        subject_kind="person",
    )


async def test_removed_message_projects_only_its_tombstone(
    client: AsyncClient,
) -> None:
    removed_at = utcnow()
    with SessionLocal() as session:
        message = Message(
            id="msg_removed_projection",
            chat=Chat(id="chat_removed_projection", title="Tombstone projection"),
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
            content_removed_at=removed_at,
        )
        session.add(message)
        session.commit()

    response = await client.get("/api/chats/chat_removed_projection")

    assert response.status_code == 200
    projected = response.json()["messages"][0]
    assert projected["id"] == "msg_removed_projection"
    assert projected["content_removed_at"] == removed_at.isoformat().replace("+00:00", "Z")
    assert projected["parts"] == []


def test_removed_messages_emit_no_payload_or_provenance_fallback(
    settings: Settings,
) -> None:
    del settings  # The fixture binds SessionLocal to this test's migrated copy.
    with SessionLocal() as session:
        chat = Chat(title="Tombstone projection")
        artifact = Artifact(
            id="artifact_removed_projection",
            sha256="a" * 64,
            kind=ArtifactKind.IMAGE.value,
            media_type="image/png",
            size_bytes=1,
            relative_path="aa/removed-projection.png",
            original_name="removed-projection.png",
            metadata_json={"semantic_description": "payload from removed content"},
        )
        removed_at = utcnow()
        inconsistent = Message(
            id="msg_inconsistent_removed",
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
            content_removed_at=removed_at,
            parts=[
                MessagePart(
                    position=0,
                    type=PartType.TEXT.value,
                    text="removed user payload",
                ),
            ],
        )
        assert ConversationOrchestrator._message_context_text(inconsistent, [artifact]) == ""

        user = Message(
            id="msg_removed_user",
            chat=chat,
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
            content_removed_at=removed_at,
        )
        assistant = Message(
            id="msg_removed_assistant",
            chat=chat,
            parent_id=user.id,
            role=MessageRole.ASSISTANT.value,
            status=MessageStatus.COMPLETE.value,
            content_removed_at=removed_at,
        )
        run = Run(
            chat=chat,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            operation=Operation.TEXT_TO_IMAGE.value,
            status=RunStatus.COMPLETE.value,
            standalone_prompt="removed standalone prompt",
            provenance_json={"input_artifact_ids": [artifact.id]},
        )
        session.add_all([artifact, chat, user, assistant, run])
        session.commit()

        assert ConversationOrchestrator._message_input_artifacts(session, user) == []
        assert ConversationOrchestrator.input_artifact_ids_for_run(session, run) == []
        assert ConversationOrchestrator._latest_image_context(
            session,
            chat.id,
            assistant.id,
        ) == (None, None)
        assert (
            ConversationOrchestrator._latest_media_prompt(
                session,
                chat.id,
                assistant.id,
            )
            is None
        )


def test_removed_message_refuses_a_reparented_message_part(settings: Settings) -> None:
    del settings
    with SessionLocal() as session:
        chat = Chat(id="chat_removed_part_reparent", title="Part reparent guard")
        target = _message(chat, "msg_removed_part_target", removed=True)
        source = _message(chat, "msg_removed_part_source")
        part = MessagePart(
            id="part_removed_reparent",
            message=source,
            position=0,
            type=PartType.TEXT.value,
            text="must stay on the live message",
        )
        session.add_all([chat, target, source, part])
        session.commit()

        part.message_id = target.id
        with pytest.raises(
            SAIntegrityError,
            match="removed chat item cannot receive message parts",
        ):
            session.flush()


def test_removed_message_refuses_a_new_revision_part(settings: Settings) -> None:
    del settings
    with SessionLocal() as session:
        chat = Chat(id="chat_removed_revision_part", title="Revision part guard")
        target = _message(chat, "msg_removed_revision_part_target", removed=True)
        revision = ResponseRevision(
            id="revision_removed_part_target",
            message=target,
            sequence=1,
            status=MessageStatus.COMPLETE.value,
        )
        session.add_all([chat, target, revision])
        session.commit()

        session.add(
            ResponseRevisionPart(
                id="revision_part_removed_insert",
                response_revision_id=revision.id,
                position=0,
                type=PartType.TEXT.value,
                text="must not repopulate removed content",
            )
        )
        with pytest.raises(
            SAIntegrityError,
            match="removed chat item cannot receive revision parts",
        ):
            session.flush()


def test_removed_message_refuses_a_reparented_revision_part(settings: Settings) -> None:
    del settings
    with SessionLocal() as session:
        chat = Chat(id="chat_removed_revision_reparent", title="Revision reparent guard")
        target = _message(chat, "msg_removed_revision_target", removed=True)
        source = _message(chat, "msg_removed_revision_source")
        target_revision = ResponseRevision(
            id="revision_removed_reparent_target",
            message=target,
            sequence=1,
            status=MessageStatus.COMPLETE.value,
        )
        source_revision = ResponseRevision(
            id="revision_removed_reparent_source",
            message=source,
            sequence=1,
            status=MessageStatus.COMPLETE.value,
        )
        part = ResponseRevisionPart(
            id="revision_part_removed_reparent",
            revision=source_revision,
            position=0,
            type=PartType.TEXT.value,
            text="must stay on the live revision",
        )
        session.add_all([chat, target, source, target_revision, source_revision, part])
        session.commit()

        part.response_revision_id = target_revision.id
        with pytest.raises(
            SAIntegrityError,
            match="removed chat item cannot receive revision parts",
        ):
            session.flush()


def test_removed_message_refuses_a_new_reference(settings: Settings) -> None:
    del settings
    with SessionLocal() as session:
        chat = Chat(id="chat_removed_reference", title="Reference guard")
        target = _message(chat, "msg_removed_reference_target", removed=True)
        session.add_all([chat, target])
        session.commit()

        session.add(_reference(target.id, "reference_removed_insert"))
        with pytest.raises(
            SAIntegrityError,
            match="removed chat item cannot receive references",
        ):
            session.flush()


def test_removed_message_refuses_a_reparented_reference(settings: Settings) -> None:
    del settings
    with SessionLocal() as session:
        chat = Chat(id="chat_removed_reference_reparent", title="Reference reparent guard")
        target = _message(chat, "msg_removed_reference_reparent_target", removed=True)
        source = _message(chat, "msg_removed_reference_reparent_source")
        reference = _reference(source.id, "reference_removed_reparent")
        session.add_all([chat, target, source])
        session.commit()
        session.add(reference)
        session.commit()

        reference.message_id = target.id
        with pytest.raises(
            SAIntegrityError,
            match="removed chat item cannot receive references",
        ):
            session.flush()
