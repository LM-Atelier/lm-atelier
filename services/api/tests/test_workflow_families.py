from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.models import (
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
    engine.dispose()


def test_family_groups_capability_variants_independently_of_operation(session: Session) -> None:
    family = WorkflowFamily(name="Flexible image workflow")
    session.add(family)
    session.flush()
    session.add_all(
        [
            WorkflowDefinition(
                family_id=family.id,
                variant_key="create",
                name="Create",
                operation="text_to_image",
            ),
            WorkflowDefinition(
                family_id=family.id,
                variant_key="edit",
                name="Edit",
                operation="image_to_image",
            ),
            WorkflowDefinition(
                family_id=family.id,
                variant_key="inpaint",
                name="Inpaint",
                operation="image_to_image",
            ),
        ]
    )
    session.commit()

    assert (
        session.scalar(
            select(func.count(WorkflowDefinition.id)).where(
                WorkflowDefinition.family_id == family.id
            )
        )
        == 3
    )

    session.add(
        WorkflowDefinition(
            family_id=family.id,
            variant_key="edit",
            name="Duplicate stable variant",
            operation="image_to_image",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_preferences_enforce_one_default_for_each_selector(session: Session) -> None:
    first = WorkflowFamily(name="First")
    second = WorkflowFamily(name="Second")
    third = WorkflowFamily(name="Third")
    session.add_all([first, second, third])
    session.flush()
    first_image = WorkflowPreference(
        workflow_family_id=first.id,
        selector_capability="image",
        is_default=True,
    )
    second_image = WorkflowPreference(
        workflow_family_id=second.id,
        selector_capability="image",
    )
    session.add_all(
        [
            first_image,
            second_image,
            WorkflowPreference(
                workflow_family_id=second.id,
                selector_capability="chat",
                is_default=True,
            ),
        ]
    )
    session.commit()

    first_image.is_default = False
    session.flush()
    second_image.is_default = True
    session.commit()

    session.add(
        WorkflowPreference(
            workflow_family_id=third.id,
            selector_capability="image",
            is_default=True,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(
        WorkflowPreference(
            workflow_family_id=third.id,
            selector_capability="video",
            enabled=False,
            is_default=True,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(
        WorkflowPreference(
            workflow_family_id=third.id,
            selector_capability="",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(
        WorkflowPreference(
            workflow_family_id=first.id,
            selector_capability="image",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_family_removal_preserves_executable_provenance(session: Session) -> None:
    family = WorkflowFamily(name="Removable selector")
    definition = WorkflowDefinition(
        family=family,
        variant_key="create",
        name="Preserved variant",
        operation="text_to_image",
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        artifact_sha256="a" * 64,
        trusted=True,
    )
    preference = WorkflowPreference(
        family=family,
        selector_capability="image",
        is_default=True,
    )
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    family_id = family.id
    definition_id = definition.id
    revision_id = revision.id
    session.commit()

    session.delete(family)
    session.commit()
    session.expire_all()

    preserved_definition = session.get(WorkflowDefinition, definition_id)
    assert preserved_definition is not None
    assert preserved_definition.family_id is None
    assert preserved_definition.current_revision_id == revision_id
    assert session.get(WorkflowRevision, revision_id) is not None
    assert (
        session.scalar(
            select(func.count(WorkflowPreference.id)).where(
                WorkflowPreference.workflow_family_id == family_id
            )
        )
        == 0
    )


def test_archiving_and_editing_family_metadata_leave_revision_unchanged(
    session: Session,
) -> None:
    family = WorkflowFamily(name="Original", use_case="portraits")
    definition = WorkflowDefinition(
        family=family,
        variant_key="create",
        name="Image variant",
        operation="text_to_image",
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        artifact_sha256="b" * 64,
        capabilities_json=["image", "image_generation"],
        trusted=True,
    )
    session.add_all([family, definition, revision])
    session.flush()
    definition.current_revision_id = revision.id
    revision_id = revision.id
    session.commit()

    family.name = "Renamed"
    family.use_case = "product portraits"
    family.tags_json = ["portrait", "product"]
    family.archived = True
    session.commit()

    stored = session.get(WorkflowRevision, revision_id)
    assert stored is not None
    assert stored.version == 1
    assert stored.artifact_sha256 == "b" * 64
    assert stored.capabilities_json == ["image", "image_generation"]
    assert stored.trusted is True
    assert definition.family_id == family.id
    assert definition.current_revision_id == revision_id
    assert (
        session.scalar(
            select(func.count(WorkflowRevision.id)).where(
                WorkflowRevision.workflow_id == definition.id
            )
        )
        == 1
    )
