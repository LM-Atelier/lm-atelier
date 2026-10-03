"""Deleted projects reject direct writers while retained chats can move out."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm.config import Settings
from local_lm.db import Base, create_database_engine
from local_lm.models import (
    Chat,
    Project,
    ProjectWorkflowSelection,
    ProjectWorkflowUseCaseSelection,
    RecoveryItem,
    WorkflowDefinition,
    WorkflowRevision,
)
from local_lm.project_recovery_schema import PROJECT_COLUMNS


@pytest.fixture
def project_session(tmp_path: Path) -> Iterator[Session]:
    settings = Settings(data_dir=tmp_path / "data", chat_engine="mock", media_engine="mock")
    settings.prepare()
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [Project(id="project-garden", name="Garden"), Project(id="project-live", name="Live")]
        )
        session.flush()
        session.add_all(
            [
                Chat(id="chat-garden", project_id="project-garden", title="Garden notes"),
                Chat(id="chat-live", project_id="project-live", title="Live notes"),
                ProjectWorkflowSelection(
                    id="selection-garden",
                    project_id="project-garden",
                    selector_capability="image",
                    mode="automatic",
                ),
                ProjectWorkflowUseCaseSelection(
                    project_id="project-garden", use_case="image_generate"
                ),
            ]
        )
        session.flush()
        session.add(
            RecoveryItem(
                deletion_id="deleted-garden",
                kind="project",
                subject_id="project-garden",
                display_label="Garden",
                deleted_at=datetime.now(UTC),
                purge_after=datetime.now(UTC) + timedelta(days=30),
                state="recoverable",
                subject_revision="a" * 64,
            )
        )
        session.commit()
        yield session
    engine.dispose()


@pytest.mark.parametrize("action", ["update", "delete"])
def test_a_deleted_project_rejects_bulk_updates_and_deletion(
    project_session: Session, action: str
) -> None:
    statement = (
        update(Project).where(Project.id == "project-garden").values(name="Changed")
        if action == "update"
        else delete(Project).where(Project.id == "project-garden")
    )
    with pytest.raises(IntegrityError, match="project-recovery-write-refused"):
        project_session.execute(statement)
    project_session.rollback()
    assert (
        project_session.scalar(select(Project.name).where(Project.id == "project-garden"))
        == "Garden"
    )
    assert (
        project_session.scalar(select(Chat.project_id).where(Chat.id == "chat-garden"))
        == "project-garden"
    )


@pytest.mark.parametrize("action", ["insert", "update"])
def test_new_filing_cannot_enter_a_deleted_project(project_session: Session, action: str) -> None:
    statement = (
        insert(Chat).values(id="chat-new", title="New notes", project_id="project-garden")
        if action == "insert"
        else update(Chat).where(Chat.id == "chat-live").values(project_id="project-garden")
    )
    with pytest.raises(IntegrityError, match="project-recovery-write-refused"):
        project_session.execute(statement)
    project_session.rollback()
    assert project_session.get(Chat, "chat-new") is None
    assert (
        project_session.scalar(select(Chat.project_id).where(Chat.id == "chat-live"))
        == "project-live"
    )


@pytest.mark.parametrize("selection", [ProjectWorkflowSelection, ProjectWorkflowUseCaseSelection])
@pytest.mark.parametrize("action", ["insert", "update", "delete"])
def test_deleted_project_selections_reject_old_and_new_ownership_writes(
    project_session: Session, selection, action: str
) -> None:
    if action == "insert":
        values = (
            {"id": "selection-new", "selector_capability": "video", "mode": "automatic"}
            if selection is ProjectWorkflowSelection
            else {"use_case": "video_generate"}
        )
        statement = insert(selection).values(project_id="project-garden", **values)
    elif action == "update":
        statement = (
            update(selection)
            .where(selection.project_id == "project-garden")
            .values(project_id="project-live")
        )
    else:
        statement = delete(selection).where(selection.project_id == "project-garden")
    with pytest.raises(IntegrityError, match="project-recovery-write-refused"):
        project_session.execute(statement)
    project_session.rollback()
    assert (
        len(
            project_session.scalars(
                select(selection).where(selection.project_id == "project-garden")
            ).all()
        )
        == 1
    )


def test_retained_chats_can_change_and_move_without_rewriting_the_project(
    project_session: Session,
) -> None:
    project_session.execute(
        update(Chat).where(Chat.id == "chat-garden").values(title="Kept history")
    )
    project_session.commit()
    assert (
        project_session.scalar(select(Chat.project_id).where(Chat.id == "chat-garden"))
        == "project-garden"
    )
    project_session.execute(
        update(Chat).where(Chat.id == "chat-garden").values(project_id="project-live")
    )
    project_session.commit()
    assert (
        project_session.scalar(select(Chat.project_id).where(Chat.id == "chat-garden"))
        == "project-live"
    )
    assert (
        project_session.scalar(select(Project.name).where(Project.id == "project-garden"))
        == "Garden"
    )


def test_the_project_cascade_guard_covers_every_canonical_column() -> None:
    assert set(PROJECT_COLUMNS) == set(Base.metadata.tables["projects"].columns.keys())


def test_only_a_vanished_workflow_can_clear_a_deleted_projects_default(
    project_session: Session,
) -> None:
    session = project_session
    item = session.get(RecoveryItem, "deleted-garden")
    assert item is not None
    session.delete(item)
    session.flush()
    session.add(
        WorkflowDefinition(id="workflow-garden", name="Garden rendering", operation="text_to_image")
    )
    session.flush()
    session.add(
        WorkflowRevision(
            id="revision-garden", workflow_id="workflow-garden", version=1, engine="mock"
        )
    )
    session.flush()
    session.execute(
        update(Project)
        .where(Project.id == "project-garden")
        .values(
            image_workflow_revision_id="revision-garden",
            video_workflow_revision_id="revision-garden",
        )
    )
    session.commit()
    session.add(
        RecoveryItem(
            deletion_id="deleted-garden",
            kind="project",
            subject_id="project-garden",
            display_label="Garden",
            deleted_at=datetime.now(UTC),
            purge_after=datetime.now(UTC) + timedelta(days=30),
            state="recoverable",
            subject_revision="a" * 64,
        )
    )
    session.commit()
    with pytest.raises(IntegrityError, match="project-recovery-write-refused"):
        session.execute(
            update(Project)
            .where(Project.id == "project-garden")
            .values(image_workflow_revision_id=None, name="Changed")
        )
    session.rollback()
    before = dict(
        session.execute(select(Project.__table__).where(Project.id == "project-garden"))
        .mappings()
        .one()
    )
    session.execute(delete(WorkflowRevision).where(WorkflowRevision.id == "revision-garden"))
    session.commit()
    after = dict(
        session.execute(select(Project.__table__).where(Project.id == "project-garden"))
        .mappings()
        .one()
    )
    assert after == before | {
        "image_workflow_revision_id": None,
        "video_workflow_revision_id": None,
    }
    with pytest.raises(IntegrityError, match="project-recovery-write-refused"):
        session.execute(
            update(Project).where(Project.id == "project-garden").values(name="Changed")
        )
    session.rollback()
