from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from local_lm import api
from local_lm.api_errors import register_api_error_handler
from local_lm.db import Base, get_session
from local_lm.models import WorkflowDefinition, WorkflowRevision

STAMP = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value
    engine.dispose()


@pytest_asyncio.fixture
async def client(session: Session) -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    register_api_error_handler(app)
    app.include_router(api.router)

    async def override() -> AsyncIterator[Session]:
        yield session

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as value:
        yield value


def workflow(session: Session, identifier: str, revisions: int = 2) -> None:
    session.add(
        WorkflowDefinition(
            id=identifier,
            name="Example " + identifier,
            operation="text_to_image",
            description="Neutral fixture",
            created_at=STAMP,
            updated_at=STAMP,
            current_revision_id=identifier + "-r" + str(revisions),
        )
    )
    session.flush()
    for version in range(1, revisions + 1):
        session.add(
            WorkflowRevision(
                id=identifier + "-r" + str(version),
                workflow_id=identifier,
                version=version,
                engine="mock",
                trusted=True,
                created_at=STAMP,
                api_graph_json={"neutral_graph": identifier, "version": version},
                ui_graph_json={"neutral_ui_graph": identifier},
                input_schema_json={"neutral_input": True},
                dependencies_json={"neutral_dependency": identifier},
            )
        )


async def test_library_summaries_avoid_all_revision_payloads(
    session: Session, client: AsyncClient
) -> None:
    for index in range(20):
        workflow(session, f"workflow-{index:02}", revisions=10)
    session.commit()
    session.expunge_all()
    loaded: list[object] = []

    def track(_session: Session, instance: object) -> None:
        loaded.append(instance)

    event.listen(session, "loaded_as_persistent", track)
    try:
        legacy = await client.get("/api/workflows")
        assert legacy.status_code == 200
        assert sum(isinstance(row, WorkflowRevision) for row in loaded) == 200
        assert len(legacy.json()) == 20
        loaded.clear()
        session.expunge_all()
        response = await client.get("/api/workflow-summaries")
    finally:
        event.remove(session, "loaded_as_persistent", track)
    assert response.status_code == 200
    result = response.json()
    assert len(result) == 20
    assert [row["id"] for row in result] == [f"workflow-{index:02}" for index in range(20)]
    assert all(row["revision_count"] == 10 and row["family_id"] is None for row in result)
    assert result[0]["current_revision_id"] == "workflow-00-r10"
    assert all("revisions" not in row for row in result)
    assert "neutral_graph" not in response.text and "neutral_input" not in response.text
    assert loaded == []


async def test_detail_loads_only_selected_owned_revisions(
    session: Session, client: AsyncClient
) -> None:
    workflow(session, "selected", 3)
    workflow(session, "other", 30)
    session.commit()
    session.expunge_all()
    loaded: list[str] = []

    def track(_session: Session, instance: object) -> None:
        if isinstance(instance, (WorkflowDefinition, WorkflowRevision)):
            loaded.append(instance.id)

    event.listen(session, "loaded_as_persistent", track)
    try:
        response = await client.get("/api/workflows/selected")
    finally:
        event.remove(session, "loaded_as_persistent", track)
    assert response.status_code == 200
    value = response.json()
    assert value["id"] == "selected" and value["current_revision_id"] == "selected-r3"
    assert {revision["id"] for revision in value["revisions"]} == {
        "selected-r1",
        "selected-r2",
        "selected-r3",
    }
    assert all(
        revision["api_graph_json"]["neutral_graph"] == "selected" for revision in value["revisions"]
    )
    assert set(loaded) == {"selected", "selected-r1", "selected-r2", "selected-r3"}


async def test_missing_detail_is_a_typed_error(client: AsyncClient) -> None:
    response = await client.get("/api/workflows/missing")
    assert response.status_code == 404
    assert response.json()["code"] == "workflow-not-found"


@pytest.mark.parametrize(
    "marker,hidden",
    [
        ({}, True),
        ({"graph_sha256": "a" * 64}, True),
        (True, False),
        (None, False),
        ("neutral", False),
    ],
)
async def test_summary_and_detail_preserve_current_package_draft_visibility(
    session: Session,
    client: AsyncClient,
    marker: object,
    hidden: bool,
) -> None:
    workflow(session, "example", 2)
    session.flush()
    current = session.get(WorkflowRevision, "example-r2")
    old = session.get(WorkflowRevision, "example-r1")
    assert current is not None and old is not None
    current.dependencies_json = {"workflow_package_draft": marker}
    old.dependencies_json = {"workflow_package_draft": {"graph_sha256": "b" * 64}}
    session.commit()
    legacy = await client.get("/api/workflows")
    assert legacy.status_code == 200
    assert bool(legacy.json()) is not hidden
    response = await client.get("/api/workflow-summaries")
    assert response.status_code == 200
    assert bool(response.json()) is not hidden
    detail = await client.get("/api/workflows/example")
    assert detail.status_code == (404 if hidden else 200)
    if not hidden:
        assert response.json()[0]["revision_count"] == 2


async def test_summary_counts_owned_revisions_and_does_not_flush_pending_work(
    session: Session,
    client: AsyncClient,
) -> None:
    workflow(session, "empty", 0)
    workflow(session, "other", 2)
    session.commit()
    empty = session.get(WorkflowDefinition, "empty")
    assert empty is not None
    empty.current_revision_id = "other-r2"
    session.commit()
    session.add(WorkflowDefinition(id="pending", name="Pending", operation="text_to_image"))
    response = await client.get("/api/workflow-summaries")
    assert response.status_code == 200
    value = {row["id"]: row for row in response.json()}
    assert set(value) == {"empty", "other"}
    assert value["empty"]["revision_count"] == 0
    assert value["other"]["revision_count"] == 2
    assert len(session.new) == 1


async def test_revision_choices_keep_history_without_loading_payloads(
    session: Session, client: AsyncClient
) -> None:
    workflow(session, "selected", 3)
    workflow(session, "other", 30)
    session.commit()
    session.expunge_all()
    loaded: list[object] = []
    statements: list[str] = []

    def track(_session: Session, instance: object) -> None:
        loaded.append(instance)

    def sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _many: bool,
    ) -> None:
        statements.append(statement)

    bind = session.get_bind()
    event.listen(session, "loaded_as_persistent", track)
    event.listen(bind, "before_cursor_execute", sql)
    try:
        response = await client.get("/api/workflow-revision-choices")
    finally:
        event.remove(session, "loaded_as_persistent", track)
        event.remove(bind, "before_cursor_execute", sql)
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 33
    assert {row["revision_id"] for row in rows if row["workflow_id"] == "selected"} == {
        "selected-r1",
        "selected-r2",
        "selected-r3",
    }
    assert all(
        set(row) == {"revision_id", "workflow_id", "workflow_name", "operation", "version"}
        for row in rows
    )
    assert rows[0] == {
        "revision_id": "other-r1",
        "workflow_id": "other",
        "workflow_name": "Example other",
        "operation": "text_to_image",
        "version": 1,
    }
    assert not loaded
    assert len(statements) == 1
    select_parts = statements[0].lower().replace("\n", " ").split(" from ")[0]
    assert all(
        name not in select_parts
        for name in ("api_graph_json", "ui_graph_json", "input_schema_json", "dependencies_json")
    )


async def test_selected_schema_reads_one_owned_revision_without_graphs(
    session: Session, client: AsyncClient
) -> None:
    workflow(session, "selected", 3)
    workflow(session, "other", 30)
    session.flush()
    revision = session.get(WorkflowRevision, "selected-r1")
    assert revision is not None
    revision.input_schema_json = {"properties": {"quality": {"type": "number"}}}
    session.commit()
    session.expunge_all()
    loaded: list[object] = []
    statements: list[str] = []

    def track(_session: Session, instance: object) -> None:
        loaded.append(instance)

    def sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _many: bool,
    ) -> None:
        statements.append(statement)

    bind = session.get_bind()
    event.listen(session, "loaded_as_persistent", track)
    event.listen(bind, "before_cursor_execute", sql)
    try:
        response = await client.get("/api/workflow-revisions/selected-r1/settings-schema")
    finally:
        event.remove(session, "loaded_as_persistent", track)
        event.remove(bind, "before_cursor_execute", sql)
    assert response.status_code == 200
    assert response.json() == {
        "revision_id": "selected-r1",
        "workflow_id": "selected",
        "operation": "text_to_image",
        "input_schema_json": {"properties": {"quality": {"type": "number"}}},
    }
    assert not loaded
    assert len(statements) == 1
    columns = statements[0].lower().replace("\n", " ").split(" from ")[0]
    assert "input_schema_json" in columns
    assert all(
        name not in columns for name in ("api_graph_json", "ui_graph_json", "dependencies_json")
    )


@pytest.mark.parametrize(
    "marker,hidden",
    [({}, True), ({"ready": False}, True), (True, False), (None, False), ("neutral", False)],
)
async def test_consumer_reads_preserve_current_draft_visibility(
    session: Session, client: AsyncClient, marker: object, hidden: bool
) -> None:
    workflow(session, "example", 2)
    session.flush()
    current = session.get(WorkflowRevision, "example-r2")
    historical = session.get(WorkflowRevision, "example-r1")
    assert current is not None and historical is not None
    current.dependencies_json = {"workflow_package_draft": marker}
    historical.dependencies_json = {"workflow_package_draft": {}}
    session.commit()
    choices = await client.get("/api/workflow-revision-choices")
    schema = await client.get("/api/workflow-revisions/example-r1/settings-schema")
    assert choices.status_code == 200
    assert bool(choices.json()) is not hidden
    assert schema.status_code == (404 if hidden else 200)
    if hidden:
        assert schema.json()["code"] == "workflow-revision-not-found"
    else:
        assert {row["revision_id"] for row in choices.json()} == {"example-r1", "example-r2"}


@pytest.mark.parametrize("revision_id", ["missing", "orphan-r1"])
async def test_selected_schema_refuses_missing_revision_or_definition(
    session: Session, client: AsyncClient, revision_id: str
) -> None:
    workflow(session, "orphan", 1)
    session.flush()
    revision = session.get(WorkflowRevision, "orphan-r1")
    assert revision is not None
    revision.workflow_id = "missing-definition"
    session.commit()
    response = await client.get(f"/api/workflow-revisions/{revision_id}/settings-schema")
    assert response.status_code == 404
    assert response.json()["code"] == "workflow-revision-not-found"
    choices = await client.get("/api/workflow-revision-choices")
    assert choices.status_code == 200 and choices.json() == []


async def test_consumer_reads_do_not_borrow_another_definitions_draft_visibility(
    session: Session, client: AsyncClient
) -> None:
    workflow(session, "selected", 1)
    workflow(session, "draft", 1)
    session.flush()
    definition = session.get(WorkflowDefinition, "selected")
    revision = session.get(WorkflowRevision, "draft-r1")
    assert definition is not None and revision is not None
    definition.current_revision_id = "draft-r1"
    revision.dependencies_json = {"workflow_package_draft": {}}
    session.commit()
    response = await client.get("/api/workflow-revisions/selected-r1/settings-schema")
    assert response.status_code == 200
    assert response.json()["workflow_id"] == "selected"
    choices = await client.get("/api/workflow-revision-choices")
    assert choices.status_code == 200
    assert [row["revision_id"] for row in choices.json()] == ["selected-r1"]
    summaries = await client.get("/api/workflow-summaries")
    assert summaries.status_code == 200
    assert [row["id"] for row in summaries.json()] == ["selected"]


async def test_consumer_reads_do_not_flush_pending_work(
    session: Session, client: AsyncClient
) -> None:
    workflow(session, "selected", 1)
    session.commit()
    session.add(WorkflowDefinition(id="pending", name="Pending", operation="text_to_image"))
    for endpoint in (
        "/api/workflow-revision-choices",
        "/api/workflow-revisions/selected-r1/settings-schema",
    ):
        response = await client.get(endpoint)
        assert response.status_code == 200
        assert len(session.new) == 1
