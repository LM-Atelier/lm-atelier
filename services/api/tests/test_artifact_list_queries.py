from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import pytest
from sqlalchemy import Result, create_engine, event
from sqlalchemy.orm import ORMExecuteState, Session

from local_lm.api import list_artifacts
from local_lm.db import Base
from local_lm.models import Artifact, Chat, Message, MessagePart, Project


@pytest.fixture
def library_session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([Project(id="p1", name="First"), Project(id="p2", name="Second")])
        session.add_all([Chat(id="c1", project_id="p1"), Chat(id="c2", project_id="p2")])
        session.add_all([Message(id="m1", chat_id="c1"), Message(id="m2", chat_id="c2")])
        start = datetime(2026, 1, 1, tzinfo=UTC)
        for index in range(43):
            key = f"asset-{index}"
            session.add(
                Artifact(
                    id=key,
                    sha256=f"{index:064x}",
                    kind="image" if index in (0, 42) else "video",
                    media_type="image/png" if index in (0, 42) else "video/mp4",
                    size_bytes=4,
                    relative_path=key,
                    original_name="Selected.png" if index == 0 else None,
                    favorite=index == 0,
                    created_at=start + timedelta(seconds=index),
                )
            )
            if index < 42:
                session.add(
                    MessagePart(message_id="m2", position=index, artifact_id=key, type="image")
                )
        # The selected image occurs twice in one chat and once in another project.
        session.add_all(
            [
                MessagePart(message_id="m1", position=i, artifact_id="asset-0", type="image")
                for i in range(2)
            ]
        )
        session.commit()
        session.expunge_all()
        yield session
    engine.dispose()


@pytest.mark.parametrize(
    ("kind", "chat_id", "project_id", "favorites", "query", "expected"),
    [
        (None, "c1", None, False, "", ["asset-0"]),
        (None, None, "p1", False, "", ["asset-0"]),
        (None, "c1", "p2", False, "", ["asset-0"]),
        ("image", None, None, False, "", ["asset-42", "asset-0"]),
        (None, None, None, True, "", ["asset-0"]),
        (None, None, None, False, " SELECTED ", ["asset-0"]),
        (None, "absent", None, False, "", []),
        (None, None, None, False, "", [f"asset-{i}" for i in range(42, -1, -1)]),
    ],
)
async def test_filters_bound_hydration_and_reference_rows_without_losing_memberships(
    library_session: Session,
    kind: Literal["image", "video"] | None,
    chat_id: str | None,
    project_id: str | None,
    favorites: bool,
    query: str,
    expected: list[str],
) -> None:
    loaded: list[str] = []
    reference_rows: list[tuple[str, str, str | None]] = []

    def on_load(session: Session, instance: object) -> None:
        if isinstance(instance, Artifact):
            loaded.append(instance.id)

    def on_execute(state: ORMExecuteState) -> Result[Any] | None:
        # Observe actual rows returned to the application, then replay them unchanged.
        descriptions = getattr(state.statement, "column_descriptions", [])
        if [item["name"] for item in descriptions] == ["artifact_id", "chat_id", "project_id"]:
            result = state.invoke_statement().freeze()
            reference_rows.extend(tuple(row) for row in result.data)
            return result()
        return None

    event.listen(library_session, "loaded_as_persistent", on_load)
    event.listen(library_session, "do_orm_execute", on_execute)
    try:
        rows = await list_artifacts(
            library_session,
            kind=kind,
            chat_id=chat_id,
            project_id=project_id,
            favorites=favorites,
            query=query,
        )
    finally:
        event.remove(library_session, "loaded_as_persistent", on_load)
        event.remove(library_session, "do_orm_execute", on_execute)
    assert [row.id for row in rows] == expected
    if "asset-0" in expected:
        selected = next(row for row in rows if row.id == "asset-0")
        assert selected.reference_count == 3
        assert selected.chat_ids == ["c1", "c2"]
        assert selected.project_ids == ["p1", "p2"]
        assert selected.url == "/api/artifacts/asset-0/content"
    assert set(loaded) == set(expected), "unselected artifacts were hydrated"
    assert {row[0] for row in reference_rows} == set(expected) - {"asset-42"}
    assert len(reference_rows) == sum(row.reference_count for row in rows)
