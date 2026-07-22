from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .domain import ModelRole, Operation
from .models import ModelProfile, WorkflowDefinition, WorkflowRevision


def seed_defaults(session: Session, settings: Settings) -> None:
    profile_specs = [
        ("Default chat", ModelRole.CHAT.value, settings.chat_engine),
        ("Default image", ModelRole.IMAGE.value, settings.media_engine),
        ("Default video", ModelRole.VIDEO.value, settings.media_engine),
    ]
    for name, role, engine in profile_specs:
        existing_profile = session.scalar(
            select(ModelProfile).where(ModelProfile.role == role, ModelProfile.is_default.is_(True))
        )
        if not existing_profile:
            session.add(ModelProfile(name=name, role=role, engine=engine, is_default=True))

    if settings.media_engine == "mock":
        for operation in (
            Operation.TEXT_TO_IMAGE,
            Operation.IMAGE_TO_IMAGE,
            Operation.TEXT_TO_VIDEO,
            Operation.IMAGE_TO_VIDEO,
        ):
            existing_workflow = session.scalar(
                select(WorkflowDefinition).where(
                    WorkflowDefinition.name == f"Mock {operation.value}"
                )
            )
            if existing_workflow:
                continue
            definition = WorkflowDefinition(
                name=f"Mock {operation.value}",
                operation=operation.value,
                description="Built-in development workflow",
            )
            session.add(definition)
            session.flush()
            revision = WorkflowRevision(
                workflow_id=definition.id,
                version=1,
                engine="mock",
                ui_graph_json={},
                api_graph_json={},
                input_schema_json={},
                dependencies_json={},
                trusted=True,
            )
            session.add(revision)
            session.flush()
            definition.current_revision_id = revision.id
    session.commit()
