from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient

from local_lm.artifact_library import REFERENCE_CORRUPT, ArtifactReferenceDataError
from local_lm.chat_item_removal import (
    MAX_PREVIEW_ARTIFACT_ID_BYTES,
    MAX_PREVIEW_ARTIFACT_IDS,
    MAX_PREVIEW_REFERENCES,
    RETAINED_WITNESS_CLASSES,
    preview_chat_item_removal,
)
from local_lm.db import SessionLocal
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


def _artifact(artifact_id: str, digit: str) -> Artifact:
    return Artifact(
        id=artifact_id,
        sha256=digit * 64,
        kind="image",
        media_type="image/png",
        size_bytes=10,
        relative_path=f"artifacts/{artifact_id}.png",
        metadata_json={},
    )


async def test_live_preview_is_bounded_complete_and_read_only(client: AsyncClient) -> None:
    with SessionLocal() as session:
        chat = Chat(title="Retained title")
        target = Message(id="msg_preview_user", chat=chat, role="user", status="complete")
        reply = Message(
            id="msg_preview_assistant",
            chat=chat,
            parent_id=target.id,
            role="assistant",
            status="complete",
        )
        released = _artifact("artifact_released", "1")
        retained = _artifact("artifact_retained", "2")
        reference_only = _artifact("artifact_reference_only", "4")
        session.add_all([chat, target, reply, released, retained, reference_only])
        session.flush()
        session.add_all(
            [
                MessagePart(
                    message_id=target.id,
                    position=0,
                    type="image",
                    artifact_id=released.id,
                    metadata_json={},
                ),
                MessagePart(
                    message_id=target.id,
                    position=1,
                    type="image",
                    artifact_id=retained.id,
                    metadata_json={},
                ),
                Run(
                    chat_id=chat.id,
                    user_message_id=target.id,
                    assistant_message_id=reply.id,
                    status="complete",
                    standalone_prompt="retained technical prompt",
                    provenance_json={"input_artifact_ids": [retained.id]},
                    settings_json={},
                ),
            ]
        )
        for position in range(MAX_PREVIEW_REFERENCES + 8):
            session.add(
                MessageReference(
                    id=f"msgref_{position:02d}",
                    message_id=target.id,
                    position=position,
                    reference_subject_id=f"subject_{position:02d}",
                    mention_slug=f"subject-{position:02d}",
                    subject_name=f"Subject {position:02d}",
                    subject_kind="person",
                    source="mention",
                    reference_asset_ids_json=[],
                    artifact_ids_json=[retained.id, reference_only.id] if position == 0 else [],
                )
            )
        session.commit()
        target_id = target.id

    response = await client.get(f"/api/messages/{target_id}/removal-impact")

    assert response.status_code == 200
    impact = response.json()
    assert impact["message_id"] == target_id
    assert impact["already_removed"] is False
    assert impact["has_replies"] is True
    assert impact["source_backs_regeneration"] is True
    assert impact["detached_message_part_count"] == 2
    assert impact["detached_response_revision_part_count"] == 0
    assert impact["detached_reference_count"] == MAX_PREVIEW_REFERENCES + 8
    assert len(impact["detached_references"]) == MAX_PREVIEW_REFERENCES
    assert impact["detached_references_truncated"] is True
    assert impact["released_artifact_count"] == 2
    assert impact["released_artifact_ids"] == [
        "artifact_reference_only",
        "artifact_released",
    ]
    assert impact["released_artifacts_truncated"] is False
    assert impact["retained_artifact_count"] == 1
    assert impact["retained_artifact_ids"] == ["artifact_retained"]
    assert impact["retained_artifacts_truncated"] is False
    assert impact["retained_witness_classes"] == list(RETAINED_WITNESS_CLASSES)
    assert impact["forensic_erasure"] is False
    assert impact["execute_authorized"] is False

    with SessionLocal() as session:
        stored = session.get(Message, target_id)
        assert stored is not None
        assert stored.content_removed_at is None
        assert len(stored.parts) == 2
        assert len(stored.references) == MAX_PREVIEW_REFERENCES + 8


async def test_preview_refuses_corrupt_target_reference_data(client: AsyncClient) -> None:
    with SessionLocal() as session:
        chat = Chat(title="Corrupt target reference")
        target = Message(
            id="msg_preview_corrupt_reference",
            chat=chat,
            role="user",
            status="complete",
        )
        session.add_all([chat, target])
        session.flush()
        session.add(
            MessageReference(
                id="msgref_preview_corrupt_reference",
                message_id=target.id,
                position=0,
                reference_subject_id="subject_corrupt_reference",
                mention_slug="subject-corrupt-reference",
                subject_name="Corrupt reference fixture",
                subject_kind="person",
                source="mention",
                reference_asset_ids_json=[],
                artifact_ids_json=[],
            )
        )
        session.commit()
        session.connection().exec_driver_sql(
            "DROP TRIGGER message_references_artifact_reference_update_guard"
        )
        session.connection().exec_driver_sql(
            "UPDATE message_references SET artifact_ids_json = ? WHERE id = ?",
            ('"artifact_invalid_shape"', "msgref_preview_corrupt_reference"),
        )
        session.commit()

        with pytest.raises(ArtifactReferenceDataError, match=f"^{REFERENCE_CORRUPT}$"):
            preview_chat_item_removal(session, target.id)


async def test_assistant_preview_counts_every_revision_part(client: AsyncClient) -> None:
    with SessionLocal() as session:
        chat = Chat(title="Assistant target")
        user = Message(id="msg_revision_user", chat=chat, role="user", status="complete")
        target = Message(
            id="msg_revision_assistant",
            chat=chat,
            parent_id=user.id,
            role="assistant",
            status="complete",
        )
        reply = Message(
            id="msg_revision_reply",
            chat=chat,
            parent_id=target.id,
            role="user",
            status="complete",
        )
        revision_artifact = _artifact("artifact_revision_part", "3")
        session.add_all([chat, user, target, reply, revision_artifact])
        session.flush()
        run = Run(
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=target.id,
            status="complete",
            standalone_prompt="retained",
            provenance_json={},
            settings_json={},
        )
        session.add(run)
        session.flush()
        first = ResponseRevision(message_id=target.id, run_id=run.id, sequence=1, status="complete")
        second = ResponseRevision(message_id=target.id, sequence=2, status="complete")
        session.add_all([first, second])
        session.flush()
        session.add_all(
            [
                ResponseRevisionPart(
                    response_revision_id=first.id,
                    position=0,
                    type="image",
                    artifact_id=revision_artifact.id,
                    metadata_json={},
                ),
                ResponseRevisionPart(
                    response_revision_id=second.id,
                    position=0,
                    type="text",
                    text="second",
                    metadata_json={},
                ),
            ]
        )
        session.commit()
        impact = preview_chat_item_removal(session, target.id)

    assert impact.role == "assistant"
    assert impact.has_replies is True
    assert impact.source_backs_regeneration is True
    assert impact.detached_response_revision_part_count == 2
    assert impact.released_artifact_ids == ["artifact_revision_part"]


async def test_preview_reports_an_already_removed_item(client: AsyncClient) -> None:
    with SessionLocal() as session:
        chat = Chat(title="Already removed")
        target = Message(
            id="msg_preview_already_removed",
            chat=chat,
            role="user",
            status="complete",
            content_removed_at=datetime.now(UTC),
        )
        session.add_all([chat, target])
        session.commit()
        target_id = target.id

    response = await client.get(f"/api/messages/{target_id}/removal-impact")

    assert response.status_code == 200
    impact = response.json()
    assert impact["already_removed"] is True
    assert impact["detached_message_part_count"] == 0
    assert impact["detached_response_revision_part_count"] == 0
    assert impact["detached_reference_count"] == 0


async def test_artifact_sample_has_an_independent_byte_bound(client: AsyncClient) -> None:
    artifact_count = MAX_PREVIEW_ARTIFACT_IDS - 36
    with SessionLocal() as session:
        chat = Chat(title="Bounded artifact sample")
        target = Message(
            id="msg_preview_artifact_byte_bound",
            chat=chat,
            role="user",
            status="complete",
        )
        session.add_all([chat, target])
        for position in range(artifact_count):
            prefix = f"artifact_{position:03d}_"
            artifact_id = prefix + ("x" * (80 - len(prefix)))
            artifact = Artifact(
                id=artifact_id,
                sha256=hashlib.sha256(artifact_id.encode("utf-8")).hexdigest(),
                kind="image",
                media_type="image/png",
                size_bytes=10,
                relative_path=f"artifacts/{position:03d}.png",
                metadata_json={},
            )
            session.add(artifact)
            session.add(
                MessagePart(
                    message=target,
                    position=position,
                    type="image",
                    artifact=artifact,
                    metadata_json={},
                )
            )
        session.commit()
        target_id = target.id

    response = await client.get(f"/api/messages/{target_id}/removal-impact")

    assert response.status_code == 200
    impact = response.json()
    sample = impact["released_artifact_ids"]
    assert impact["released_artifact_count"] == artifact_count
    assert len(sample) < artifact_count
    assert len(sample) < MAX_PREVIEW_ARTIFACT_IDS
    assert len(sample) == MAX_PREVIEW_ARTIFACT_ID_BYTES // 80
    sample_bytes = sum(len(artifact_id.encode("utf-8")) for artifact_id in sample)
    assert sample_bytes <= MAX_PREVIEW_ARTIFACT_ID_BYTES
    assert sample_bytes + 80 > MAX_PREVIEW_ARTIFACT_ID_BYTES
    assert impact["released_artifacts_truncated"] is True


async def test_missing_preview_is_typed(client: AsyncClient) -> None:
    response = await client.get("/api/messages/msg_missing/removal-impact")
    assert response.status_code == 404
    assert response.json()["code"] == "message-not-found"
