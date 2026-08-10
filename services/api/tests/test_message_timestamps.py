from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import Message


async def test_message_timestamps_are_explicit_utc_across_a_dst_boundary(
    client: AsyncClient,
) -> None:
    chat_response = await client.post("/api/chats", json={"title": "Timestamp contract"})
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]

    instants = (
        (datetime(2026, 3, 8, 9, 30), "2026-03-08T09:30:00Z", 1),
        (datetime(2026, 3, 8, 10, 30), "2026-03-08T10:30:00Z", 3),
    )
    message_ids: list[str] = []
    with SessionLocal() as session:
        for created_at, _expected_utc, _expected_local_hour in instants:
            message = Message(
                chat_id=chat_id,
                role="user",
                status="complete",
                created_at=created_at,
                updated_at=created_at,
            )
            session.add(message)
            session.flush()
            message_ids.append(message.id)
        session.commit()

    pacific = ZoneInfo("America/Los_Angeles")
    for message_id, (_created_at, expected_utc, expected_local_hour) in zip(
        message_ids, instants, strict=True
    ):
        response = await client.get(f"/api/messages/{message_id}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["created_at"] == expected_utc
        assert payload["updated_at"] == expected_utc
        parsed = datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00"))
        assert parsed.astimezone(pacific).hour == expected_local_hour
