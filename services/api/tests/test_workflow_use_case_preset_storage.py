"""Keep recipe choices normalized and bound to the same classified use case."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm import models
from local_lm.config import Settings
from local_lm.database_migrations import upgrade_database
from local_lm.db import Base, create_database_engine
from local_lm.models import Chat, GenerationPreset, Project
from local_lm.workflow_use_case_presets_v1 import WorkflowUseCaseChoice


@pytest.mark.parametrize("schema", ["metadata", "migrated"])
def test_generated_recipe_ids_can_be_selected(tmp_path: Path, schema: str) -> None:
    with _session(tmp_path, schema) as session:
        recipe = models.WorkflowUseCasePreset(name="Example", use_case="image_generation")
        session.add(recipe)
        session.commit()
        choice = WorkflowUseCaseChoice.model_validate({"mode": "preset", "preset_id": recipe.id})
        assert choice.root.mode == "preset" and choice.root.preset_id == recipe.id


@contextmanager
def _session(tmp_path: Path, schema: str) -> Iterator[Session]:
    for name in (
        "WorkflowUseCasePreset",
        "ProjectWorkflowUseCaseSelection",
        "ChatWorkflowUseCaseSelection",
    ):
        assert hasattr(models, name), "Normalized use-case preset storage is absent."
    settings = Settings(data_dir=tmp_path / schema, dev=True)
    settings.prepare()
    engine = create_database_engine(settings)
    try:
        if schema == "migrated":
            upgrade_database(settings)
        else:
            Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([Project(id="project", name="Example"), Chat(id="chat")])
            session.commit()
            yield session
    finally:
        engine.dispose()


@pytest.mark.parametrize("schema", ["metadata", "migrated"])
def test_choices_preserve_inherited_absence_and_explicit_automatic(
    tmp_path: Path, schema: str
) -> None:
    with _session(tmp_path, schema) as session:
        recipe_type = models.WorkflowUseCasePreset
        project_type = models.ProjectWorkflowUseCaseSelection
        chat_type = models.ChatWorkflowUseCaseSelection
        session.add(recipe_type(id="preset", name="Small", use_case="image_generation"))
        session.add(
            GenerationPreset(id="legacy", name="Legacy", role="image", settings_json={"seed": 9})
        )
        session.commit()
        assert session.get(project_type, ("project", "image_generation")) is None
        session.add(
            project_type(project_id="project", use_case="image_generation", preset_id="preset")
        )
        session.add(chat_type(chat_id="chat", use_case="image_generation", preset_id=None))
        session.commit()
        session.expire_all()
        project_choice = session.get(project_type, ("project", "image_generation"))
        chat_choice = session.get(chat_type, ("chat", "image_generation"))
        assert project_choice is not None and project_choice.preset_id == "preset"
        assert chat_choice is not None and chat_choice.preset_id is None
        assert session.get(chat_type, ("chat", "image_edit")) is None
        legacy = session.get(GenerationPreset, "legacy")
        assert legacy is not None and legacy.settings_json == {"seed": 9}
        chat = session.get(Chat, "chat")
        project = session.get(Project, "project")
        assert chat is not None and project is not None
        assert chat.generation_preset_ids_json == project.generation_preset_ids_json == {}


@pytest.mark.parametrize("schema", ["metadata", "migrated"])
def test_workspace_defaults_are_unique_and_enabled(tmp_path: Path, schema: str) -> None:
    with _session(tmp_path, schema) as session:
        recipe_type = models.WorkflowUseCasePreset
        session.add(
            recipe_type(id="first", name="First", use_case="image_generation", is_default=True)
        )
        session.add(recipe_type(id="other", name="Other", use_case="image_edit", is_default=True))
        session.commit()
        session.add(
            recipe_type(
                id="duplicate", name="Duplicate", use_case="image_generation", is_default=True
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        session.add(
            recipe_type(
                id="disabled", name="Disabled", use_case="chat", enabled=False, is_default=True
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        assert {row.id for row in session.scalars(select(recipe_type))} == {"first", "other"}


@pytest.mark.parametrize("schema", ["metadata", "migrated"])
@pytest.mark.parametrize("scope", ["project", "chat"])
@pytest.mark.parametrize("preset_id", ["wrong-case", "missing"])
def test_scoped_choices_require_a_preset_of_the_same_use_case(
    tmp_path: Path, schema: str, scope: str, preset_id: str
) -> None:
    with _session(tmp_path, schema) as session:
        recipe_type = models.WorkflowUseCasePreset
        choice_type = getattr(
            models,
            "ProjectWorkflowUseCaseSelection"
            if scope == "project"
            else "ChatWorkflowUseCaseSelection",
        )
        session.add(recipe_type(id="wrong-case", name="Edit", use_case="image_edit"))
        session.commit()
        session.add(
            choice_type(
                **{f"{scope}_id": scope, "use_case": "image_generation", "preset_id": preset_id}
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        assert list(session.scalars(select(choice_type))) == []


@pytest.mark.parametrize("schema", ["metadata", "migrated"])
@pytest.mark.parametrize("scope", ["project", "chat"])
def test_deleting_a_scope_removes_its_choice_but_preserves_the_recipe(
    tmp_path: Path, schema: str, scope: str
) -> None:
    with _session(tmp_path, schema) as session:
        recipe_type = models.WorkflowUseCasePreset
        choice_type = getattr(
            models,
            "ProjectWorkflowUseCaseSelection"
            if scope == "project"
            else "ChatWorkflowUseCaseSelection",
        )
        session.add(recipe_type(id="preset", name="Small", use_case="image_generation"))
        session.commit()
        session.add(
            choice_type(
                **{f"{scope}_id": scope, "use_case": "image_generation", "preset_id": "preset"}
            )
        )
        session.commit()
        recipe = session.get(recipe_type, "preset")
        assert recipe is not None
        session.delete(recipe)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        parent = session.get(Project if scope == "project" else Chat, scope)
        assert parent is not None
        session.delete(parent)
        session.commit()
        assert list(session.scalars(select(choice_type))) == []
        assert session.get(recipe_type, "preset") is not None
