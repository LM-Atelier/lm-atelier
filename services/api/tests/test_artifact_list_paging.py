from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from local_lm.api import list_artifacts
from local_lm.db import Base
from local_lm.models import Artifact, Chat, Message, MessagePart


@pytest.mark.parametrize(
    ("limit", "offset", "expected"), [(2, 0, [4, 3]), (2, 2, [2, 1]), (2, 5, [])]
)
async def test_paging_precedes_hydration_and_keeps_full_reference_counts(
    limit: int, offset: int, expected: list[int]
) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            session.add(Chat(id="chat"))
            session.add(Message(id="message", chat_id="chat"))
            for index in range(5):
                session.add(
                    Artifact(
                        id=f"image-{index}",
                        sha256=f"{index:064x}",
                        kind="image",
                        media_type="image/png",
                        size_bytes=1,
                        relative_path=f"image-{index}",
                        created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    )
                )
                for repeat in range(2):
                    session.add(
                        MessagePart(
                            message_id="message",
                            position=index * 2 + repeat,
                            type="image",
                            artifact_id=f"image-{index}",
                        )
                    )
            session.commit()
            session.expunge_all()
            loaded: list[str] = []

            def record(_session: Session, instance: object) -> None:
                if isinstance(instance, Artifact):
                    loaded.append(instance.id)

            event.listen(session, "loaded_as_persistent", record)
            rows = await list_artifacts(session, kind="image", query="", limit=limit, offset=offset)
            assert [row.id for row in rows] == [f"image-{index}" for index in expected]
            assert loaded == [row.id for row in rows]
            assert all(row.reference_count == 2 and row.chat_ids == ["chat"] for row in rows)
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "parameters", [{"limit": "0"}, {"limit": "-1"}, {"limit": "201"}, {"offset": "-1"}]
)
async def test_artifact_paging_rejects_invalid_bounds(
    client: AsyncClient, parameters: dict[str, str]
) -> None:
    response = await client.get("/api/artifacts", params=parameters)
    assert response.status_code == 422
