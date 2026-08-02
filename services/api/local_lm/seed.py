from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .domain import ModelRole, Operation
from .edit_templates import seed_edit_templates
from .models import ModelInstall, ModelProfile, WorkflowDefinition, WorkflowRevision
from .profile_service import (
    ensure_profile_for_install,
    reconcile_profile_bindings,
    retire_profiles_for_installs,
)


def seed_defaults(session: Session, settings: Settings) -> None:
    seed_edit_templates(session)
    reconcile_profile_bindings(session)
    profile_specs = [
        ("Default", ModelRole.CHAT.value, settings.chat_engine),
        ("Default", ModelRole.IMAGE.value, settings.media_engine),
        ("Default", ModelRole.VIDEO.value, settings.media_engine),
    ]
    for name, role, engine in profile_specs:
        existing_profile = session.scalar(
            select(ModelProfile).where(ModelProfile.role == role, ModelProfile.is_default.is_(True))
        )
        if not existing_profile:
            session.add(ModelProfile(name=name, role=role, engine=engine, is_default=True))

    session.flush()
    _reconcile_media_install_replacements(session)
    for install in session.scalars(select(ModelInstall).where(ModelInstall.active.is_(True))).all():
        ensure_profile_for_install(
            session,
            install,
            default_settings=install.manifest_json.get("default_settings", {}),
        )

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


def _reconcile_media_install_replacements(session: Session) -> None:
    installs = list(
        session.scalars(
            select(ModelInstall)
            .where(ModelInstall.engine == "comfyui", ModelInstall.active.is_(True))
            .order_by(ModelInstall.created_at.desc(), ModelInstall.id.desc())
        ).all()
    )
    claimed_sources: set[tuple[str, str, str]] = set()
    superseded_install_ids: set[str] = set()
    for install in installs:
        template_id = install.manifest_json.get("workflow_template_id")
        source_remote_id = install.manifest_json.get("source_remote_id")
        if not template_id or not isinstance(source_remote_id, str):
            continue
        key = (install.role, install.engine, source_remote_id.casefold())
        if key in claimed_sources:
            install.active = False
            superseded_install_ids.add(install.id)
            continue
        claimed_sources.add(key)
        for candidate in installs:
            if candidate.id == install.id or not candidate.active:
                continue
            identities = {
                str(candidate.manifest_json.get("remote_id") or "").casefold(),
                str(candidate.manifest_json.get("source_remote_id") or "").casefold(),
            }
            if (
                candidate.role == install.role
                and candidate.engine == install.engine
                and source_remote_id.casefold() in identities
            ):
                candidate.active = False
                superseded_install_ids.add(candidate.id)
    retire_profiles_for_installs(session, superseded_install_ids)
