"""Every delete reconciles, refuses honestly, and spares shared media.

Delete appeared to do nothing. These regressions pin the
five properties every user-facing delete path owes: distinguishable failure,
immediate list reconciliation, shared-artifact survival, actionable refusals,
and honesty about the already-deleted.
"""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient

pytestmark = pytest.mark.asyncio


def _seed_shared_generated_image(chat_ids: list[str]) -> str:
    from local_lm.db import SessionLocal
    from local_lm.domain import ArtifactKind, MessageRole, MessageStatus, PartType
    from local_lm.models import Artifact, Chat, Message, MessagePart

    with SessionLocal() as session:
        artifact = Artifact(
            id=f"sha256:{'d' * 64}",
            sha256="d" * 64,
            kind=ArtifactKind.IMAGE.value,
            media_type="image/png",
            size_bytes=8,
            relative_path=f"{'d' * 2}/{'d' * 2}/{'d' * 64}",
        )
        session.add(artifact)
        for index, chat_id in enumerate(chat_ids):
            chat = session.get(Chat, chat_id)
            assert chat is not None
            message = Message(
                chat_id=chat_id,
                role=MessageRole.ASSISTANT.value,
                status=MessageStatus.COMPLETE.value,
            )
            session.add(message)
            session.flush()
            session.add(
                MessagePart(
                    message_id=message.id,
                    position=0,
                    type=PartType.IMAGE.value,
                    artifact_id=artifact.id,
                )
            )
            del index
        session.commit()
        return artifact.id


async def test_a_deleted_chat_leaves_the_list_and_stays_deleted(client: AsyncClient) -> None:
    created = await client.post("/api/chats", json={"title": "Doomed", "project_id": None})
    chat_id = created.json()["id"]
    assert any(chat["id"] == chat_id for chat in (await client.get("/api/chats")).json())

    deleted = await client.delete(f"/api/chats/{chat_id}")
    assert deleted.status_code == 204
    assert all(chat["id"] != chat_id for chat in (await client.get("/api/chats")).json())

    # Deleting again is an honest 404, never a success that did nothing.
    again = await client.delete(f"/api/chats/{chat_id}")
    assert again.status_code == 404


async def test_chat_media_deletion_spares_artifacts_other_chats_show(
    client: AsyncClient,
) -> None:
    first = (await client.post("/api/chats", json={"title": "A", "project_id": None})).json()["id"]
    second = (await client.post("/api/chats", json={"title": "B", "project_id": None})).json()["id"]
    artifact_id = _seed_shared_generated_image([first, second])

    from local_lm.db import SessionLocal
    from local_lm.models import Artifact

    deleted = await client.delete(f"/api/chats/{first}", params={"delete_generated_media": "true"})
    assert deleted.status_code == 204
    with SessionLocal() as session:
        assert session.get(Artifact, artifact_id) is not None, (
            "an artifact another chat still shows must survive that chat's media deletion"
        )

    deleted = await client.delete(f"/api/chats/{second}", params={"delete_generated_media": "true"})
    assert deleted.status_code == 204
    with SessionLocal() as session:
        assert session.get(Artifact, artifact_id) is None, (
            "the last reference leaving takes the artifact with it"
        )


async def test_exchange_refusals_are_typed_and_actionable(client: AsyncClient) -> None:
    absent = await client.delete("/api/messages/absent/exchange")
    assert absent.status_code == 404
    assert absent.json()["code"] == "exchange-not-found"

    from local_lm.db import SessionLocal
    from local_lm.domain import MessageRole, MessageStatus
    from local_lm.models import Chat, Message

    with SessionLocal() as session:
        chat = Chat(title="Threaded")
        session.add(chat)
        session.flush()
        user = Message(
            chat_id=chat.id, role=MessageRole.USER.value, status=MessageStatus.COMPLETE.value
        )
        session.add(user)
        session.flush()
        answer = Message(
            chat_id=chat.id,
            parent_id=user.id,
            role=MessageRole.ASSISTANT.value,
            status=MessageStatus.COMPLETE.value,
        )
        session.add(answer)
        session.flush()
        reply = Message(
            chat_id=chat.id,
            parent_id=answer.id,
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
        )
        session.add(reply)
        session.commit()
        user_id = user.id

    refused = await client.delete(f"/api/messages/{user_id}/exchange")
    assert refused.status_code == 409
    body = refused.json()
    assert body["code"] == "exchange-has-replies"
    assert body["reply_count"] >= 1


async def test_double_deleting_an_artifact_answers_honestly(client: AsyncClient) -> None:
    from local_lm.db import SessionLocal
    from local_lm.domain import ArtifactKind
    from local_lm.models import Artifact

    with SessionLocal() as session:
        session.add(
            Artifact(
                id=f"sha256:{'e' * 64}",
                sha256="e" * 64,
                kind=ArtifactKind.IMAGE.value,
                media_type="image/png",
                size_bytes=4,
                relative_path=f"{'e' * 2}/{'e' * 2}/{'e' * 64}",
            )
        )
        session.commit()

    first = await client.delete(f"/api/artifacts/sha256:{'e' * 64}")
    assert first.status_code == 200
    second = await client.delete(f"/api/artifacts/sha256:{'e' * 64}")
    assert second.status_code == 404


async def test_small_resource_deletes_reconcile_and_stay_deleted(client: AsyncClient) -> None:
    project = (await client.post("/api/projects", json={"name": "Doomed project"})).json()
    assert (await client.delete(f"/api/projects/{project['id']}")).status_code == 204
    assert (await client.delete(f"/api/projects/{project['id']}")).status_code == 404

    template = (
        await client.post(
            "/api/edit-templates",
            json={"name": "Doomed template", "instruction": "Do the thing."},
        )
    ).json()
    assert (await client.delete(f"/api/edit-templates/{template['id']}")).status_code == 204
    assert (await client.delete(f"/api/edit-templates/{template['id']}")).status_code == 404
