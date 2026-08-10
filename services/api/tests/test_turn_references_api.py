from __future__ import annotations

import asyncio

from httpx2 import AsyncClient
from sqlalchemy import func, select

from local_lm.db import SessionLocal
from local_lm.message_references import message_references
from local_lm.models import MessageReference


async def _wait_for_run(client: AsyncClient, run_id: str) -> None:
    for _ in range(200):
        run = (await client.get(f"/api/runs/{run_id}")).json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} did not finish")


async def _chat(client: AsyncClient, title: str) -> dict[str, object]:
    response = await client.post("/api/chats", json={"title": title})
    assert response.status_code == 201
    return response.json()


async def _subject(client: AsyncClient, name: str) -> dict[str, object]:
    response = await client.post("/api/references", json={"name": name, "kind": "person"})
    assert response.status_code == 201
    return response.json()


async def test_a_turn_records_its_typed_reference_snapshot(client: AsyncClient) -> None:
    chat = await _chat(client, "Typed reference")
    subject = await _subject(client, "Ada Lovelace")

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Describe the subject",
            "mode": "text",
            "references": [
                {
                    "reference_subject_id": subject["id"],
                    "role": "subject",
                    "strength": 0.65,
                    "source": "picker",
                }
            ],
        },
    )

    assert accepted.status_code == 202
    with SessionLocal() as session:
        snapshot = message_references(session, accepted.json()["user_message"]["id"])
    assert len(snapshot) == 1
    assert snapshot[0].reference_subject_id == subject["id"]
    assert snapshot[0].subject_name == "Ada Lovelace"
    assert snapshot[0].mention_slug == "ada-lovelace"
    assert snapshot[0].role == "subject"
    assert snapshot[0].strength == 0.65
    assert snapshot[0].source.value == "picker"


async def test_an_unknown_reference_refuses_before_writing_the_turn(client: AsyncClient) -> None:
    chat = await _chat(client, "Missing reference")

    refused = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Use the missing subject",
            "mode": "text",
            "references": [{"reference_subject_id": "refsubject_missing"}],
        },
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "turn-invalid"
    assert "no longer exists" in refused.json()["detail"]
    assert (await client.get(f"/api/chats/{chat['id']}")).json()["messages"] == []


async def test_idempotent_replay_does_not_record_references_twice(client: AsyncClient) -> None:
    chat = await _chat(client, "Reference replay")
    subject = await _subject(client, "Grace Hopper")
    payload = {
        "text": "Describe Grace",
        "mode": "text",
        "idempotency_key": "reference-replay",
        "references": [{"reference_subject_id": subject["id"]}],
    }

    first = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    replay = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json()["user_message"]["id"] == first.json()["user_message"]["id"]
    with SessionLocal() as session:
        count = session.scalar(
            select(func.count(MessageReference.id)).where(
                MessageReference.message_id == first.json()["user_message"]["id"]
            )
        )
    assert count == 1


async def test_an_ordered_turn_records_one_reference_snapshot(client: AsyncClient) -> None:
    chat = await _chat(client, "Ordered reference")
    subject = await _subject(client, "Paper Boat Style")
    payload = {
        "text": (
            "Write a short story about a paper boat, then create an image based on it, "
            "then animate the image into a video, then summarize the video"
        ),
        "mode": "auto",
        "idempotency_key": "ordered-reference",
        "references": [{"reference_subject_id": subject["id"], "role": "style"}],
    }

    preview = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert preview.status_code == 409
    assert (await client.get(f"/api/chats/{chat['id']}")).json()["messages"] == []

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns", json={**payload, "confirm_media": True}
    )
    assert accepted.status_code == 202
    with SessionLocal() as session:
        snapshot = message_references(session, accepted.json()["user_message"]["id"])
    assert [(one.subject_name, one.role) for one in snapshot] == [("Paper Boat Style", "style")]


async def test_regeneration_carries_deleted_subject_snapshot_verbatim(
    client: AsyncClient,
) -> None:
    chat = await _chat(client, "Reference regeneration")
    subject = await _subject(client, "Original Name")
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Describe the subject",
            "mode": "text",
            "references": [{"reference_subject_id": subject["id"]}],
        },
    )
    assert original.status_code == 202
    await _wait_for_run(client, original.json()["run"]["id"])
    source_user_id = original.json()["user_message"]["id"]

    renamed = await client.patch(f"/api/references/{subject['id']}", json={"name": "Renamed Later"})
    assert renamed.status_code == 200
    deleted = await client.delete(
        f"/api/references/{subject['id']}", params={"acknowledged_assets": 0}
    )
    assert deleted.status_code == 204

    regenerated = await client.post(
        f"/api/messages/{original.json()['assistant_message']['id']}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202
    with SessionLocal() as session:
        source = message_references(session, source_user_id)
        repeat = message_references(session, regenerated.json()["user_message"]["id"])
    assert repeat == source
    assert repeat[0].subject_name == "Original Name"


async def test_editing_carries_only_an_omitted_reference_field(client: AsyncClient) -> None:
    chat = await _chat(client, "Reference edit")
    original_subject = await _subject(client, "Original Subject")
    replacement_subject = await _subject(client, "Replacement Subject")
    original = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Describe the original",
            "mode": "text",
            "references": [{"reference_subject_id": original_subject["id"]}],
        },
    )
    assert original.status_code == 202
    await _wait_for_run(client, original.json()["run"]["id"])
    source_user_id = original.json()["user_message"]["id"]

    carried = await client.post(
        f"/api/messages/{source_user_id}/branch",
        json={"text": "Describe the original differently", "mode": "text"},
    )
    assert carried.status_code == 202
    await _wait_for_run(client, carried.json()["run"]["id"])

    cleared = await client.post(
        f"/api/messages/{source_user_id}/branch",
        json={"text": "Describe nobody", "mode": "text", "references": []},
    )
    assert cleared.status_code == 202
    await _wait_for_run(client, cleared.json()["run"]["id"])

    replaced = await client.post(
        f"/api/messages/{source_user_id}/branch",
        json={
            "text": "Describe the replacement",
            "mode": "text",
            "references": [{"reference_subject_id": replacement_subject["id"]}],
        },
    )
    assert replaced.status_code == 202

    with SessionLocal() as session:
        source = message_references(session, source_user_id)
        carried_snapshot = message_references(session, carried.json()["user_message"]["id"])
        cleared_snapshot = message_references(session, cleared.json()["user_message"]["id"])
        replaced_snapshot = message_references(session, replaced.json()["user_message"]["id"])
    assert carried_snapshot == source
    assert cleared_snapshot == ()
    assert [one.subject_name for one in replaced_snapshot] == ["Replacement Subject"]
