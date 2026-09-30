"""Resolve an edit's input through parent links when timestamps tie."""

from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import Artifact, Chat, Message, MessagePart


async def test_parent_input_does_not_depend_on_lexical_message_order(client: AsyncClient) -> None:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    with SessionLocal() as session:
        session.add(Chat(id="chat_tied_edit"))
        session.add(
            Artifact(
                id="original_image",
                sha256="a" * 64,
                kind="image",
                media_type="image/png",
                size_bytes=1,
                relative_path="original.png",
            )
        )
        session.flush()
        session.add(
            Message(id="msg_z_input", chat_id="chat_tied_edit", role="user", created_at=stamp)
        )
        session.flush()
        session.add(
            Message(
                id="msg_a_result",
                chat_id="chat_tied_edit",
                role="assistant",
                parent_id="msg_z_input",
                created_at=stamp,
            )
        )
        session.add(
            MessagePart(
                message_id="msg_z_input",
                position=0,
                type="image",
                artifact_id="original_image",
                metadata_json={"input_reference": True},
            )
        )
        session.add(
            MessagePart(message_id="msg_z_input", position=1, type="text", text="Add contrast")
        )
        session.commit()

    response = await client.get("/api/chats/chat_tied_edit/messages/msg_a_result/lineage")

    assert response.status_code == 200
    assert response.json()["steps"] == [
        {
            "message_id": "msg_z_input",
            "artifact_id": "original_image",
            "instruction": "Add contrast",
        },
    ]
    assert response.json()["next_before"] is None


def _seed_producer_chain() -> list[str]:
    ids = ["e_old_user", "f_old_result", "c_new_user", "d_new_result"]
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    with SessionLocal() as session:
        session.add(Chat(id="chat_producers"))
        session.add_all(
            Artifact(
                id=f"source_{index}",
                sha256=f"{index:064x}",
                kind="image",
                media_type="image/png",
                size_bytes=1,
                relative_path=f"source_{index}.png",
            )
            for index in range(3)
        )
        session.flush()
        for index, identity in enumerate(ids):
            session.add(
                Message(
                    id=identity,
                    chat_id="chat_producers",
                    parent_id=ids[index - 1] if index else None,
                    role="user" if index % 2 == 0 else "assistant",
                    created_at=stamp,
                )
            )
            session.flush()
            session.add(
                MessagePart(
                    message_id=identity,
                    position=0,
                    type="image",
                    artifact_id=f"source_{(index + 1) // 2}",
                    metadata_json={"input_reference": True} if index % 2 == 0 else {},
                )
            )
        session.commit()
    return ids


@pytest.mark.parametrize("reverse_clock", [False, True])
async def test_producer_pages_follow_parents_regardless_of_ids_or_clock(
    client: AsyncClient, reverse_clock: bool
) -> None:
    ids = _seed_producer_chain()
    if reverse_clock:
        with SessionLocal() as session:
            for index, identity in enumerate(ids):
                message = session.get(Message, identity)
                assert message is not None
                message.created_at = datetime(2026, 1, 4 - index, tzinfo=UTC)
            session.commit()
    path = f"/api/chats/chat_producers/messages/{ids[-1]}/lineage"
    first = await client.get(path, params={"limit": 1})
    assert first.status_code == 200
    assert [step["message_id"] for step in first.json()["steps"]] == [ids[2]]
    assert first.json()["next_before"] == ids[2]
    second = await client.get(path, params={"limit": 1, "before": ids[2]})
    assert second.status_code == 200
    assert [step["message_id"] for step in second.json()["steps"]] == [ids[0]]
    assert second.json()["next_before"] is None


@pytest.mark.parametrize("self_cycle", [False, True])
@pytest.mark.parametrize("continuation", [False, True])
async def test_cyclic_edit_history_refuses_both_initial_and_continuation_reads(
    client: AsyncClient, self_cycle: bool, continuation: bool
) -> None:
    ids = _seed_producer_chain()
    with SessionLocal() as session:
        first = session.get(Message, ids[0])
        assert first is not None
        first.parent_id = ids[0] if self_cycle else ids[-1]
        session.commit()
    params: dict[str, str | int] = {"limit": 1}
    if continuation:
        params["before"] = ids[2]
    response = await client.get(
        f"/api/chats/chat_producers/messages/{ids[-1]}/lineage", params=params
    )
    assert response.status_code == 409
    assert response.json()["code"] == "chat-lineage-cycle"
