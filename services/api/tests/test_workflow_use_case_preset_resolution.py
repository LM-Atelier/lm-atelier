"""Resolve scoped recipes without changing workflow or settings selections."""

from collections.abc import Iterator
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

import pytest
from sqlalchemy import event, select, text, update
from sqlalchemy.engine import Connection, Engine, ExecutionContext
from sqlalchemy.orm import Session

from local_lm.config import Settings
from local_lm.db import create_database_engine
from local_lm.models import (
    Chat,
    ChatWorkflowSelection,
    ChatWorkflowUseCaseSelection,
    Project,
    ProjectWorkflowUseCaseSelection,
    WorkflowUseCasePreset,
)
from local_lm.workflow_use_cases_v1 import WorkflowUseCase


@pytest.fixture
def session(settings: Settings) -> Iterator[Session]:
    engine = create_database_engine(settings)
    try:
        with Session(engine) as value:
            yield value
    finally:
        engine.dispose()


def _resolver() -> ModuleType:
    name = "local_lm.workflow_use_case_preset_resolution"
    assert find_spec(name) is not None, "Layered use-case preset resolution is absent."
    return import_module(name)


def _scopes(session: Session) -> None:
    session.add(Project(id="project", name="Example", generation_preset_ids_json={"image": "old"}))
    session.flush()
    session.add(Chat(id="chat", project_id="project"))
    session.flush()
    session.add(
        ChatWorkflowSelection(
            id="workflow", chat_id="chat", selector_capability="image", mode="automatic"
        )
    )
    session.commit()


def _recipe(session: Session, name: str, *, default: bool = False) -> None:
    session.add(
        WorkflowUseCasePreset(
            id=name,
            name=name,
            use_case="image_generation",
            settings_json={"seed": 9, "options": {"sizes": [256, 512]}},
            is_default=default,
        )
    )
    session.commit()


@pytest.mark.parametrize("use_case", list(WorkflowUseCase))
def test_no_choice_returns_unconfigured_for_every_use_case(
    session: Session, use_case: WorkflowUseCase
) -> None:
    result = _resolver().resolve_workflow_use_case_preset(session, use_case)
    assert result.mode == "unconfigured"
    assert result.use_case == use_case and result.scope is None
    assert result.preset_id is None and result.preset_name is None
    assert result.settings_json == {}


@pytest.mark.parametrize("scope", ["workspace", "project", "chat"])
def test_nearest_configured_scope_wins(session: Session, scope: str) -> None:
    resolve = _resolver().resolve_workflow_use_case_preset
    _scopes(session)
    for name in ("workspace", "project", "chat"):
        _recipe(session, name, default=name == "workspace")
    if scope in ("project", "chat"):
        session.add(
            ProjectWorkflowUseCaseSelection(
                project_id="project", use_case="image_generation", preset_id="project"
            )
        )
    if scope == "chat":
        session.add(
            ChatWorkflowUseCaseSelection(
                chat_id="chat", use_case="image_generation", preset_id="chat"
            )
        )
    session.commit()
    result = resolve(session, WorkflowUseCase.IMAGE_GENERATION, chat_id="chat")
    assert result.mode == "preset" and result.scope == scope
    assert result.preset_id == result.preset_name == scope
    assert result.settings_json["seed"] == 9
    workflow = session.get(ChatWorkflowSelection, "workflow")
    project = session.get(Project, "project")
    assert (
        workflow is not None
        and workflow.mode == "automatic"
        and workflow.workflow_family_id is None
    )
    assert project is not None and project.generation_preset_ids_json == {"image": "old"}


@pytest.mark.parametrize("scope", ["project", "chat"])
def test_automatic_stops_inheritance_without_selecting_a_recipe(
    session: Session, scope: str
) -> None:
    resolve = _resolver().resolve_workflow_use_case_preset
    _scopes(session)
    _recipe(session, "workspace", default=True)
    _recipe(session, "project")
    session.add(
        ProjectWorkflowUseCaseSelection(
            project_id="project",
            use_case="image_generation",
            preset_id="project" if scope == "chat" else None,
        )
    )
    if scope == "chat":
        session.add(
            ChatWorkflowUseCaseSelection(
                chat_id="chat", use_case="image_generation", preset_id=None
            )
        )
    session.commit()
    result = resolve(session, WorkflowUseCase.IMAGE_GENERATION, chat_id="chat")
    assert result.mode == "automatic" and result.scope == scope
    assert result.preset_id is None and result.preset_name is None
    assert result.settings_json == {}


def test_project_scope_can_resolve_without_a_chat(session: Session) -> None:
    resolve = _resolver().resolve_workflow_use_case_preset
    _scopes(session)
    _recipe(session, "project")
    session.add(
        ProjectWorkflowUseCaseSelection(
            project_id="project", use_case="image_generation", preset_id="project"
        )
    )
    session.commit()
    result = resolve(session, WorkflowUseCase.IMAGE_GENERATION, project_id="project")
    assert result.preset_id == "project" and result.scope == "project"


@pytest.mark.parametrize("scope", ["project", "chat"])
def test_disabled_selected_recipe_refuses_instead_of_falling_back(
    session: Session, scope: str
) -> None:
    module = _resolver()
    _scopes(session)
    _recipe(session, "workspace", default=True)
    _recipe(session, "disabled")
    recipe = session.get(WorkflowUseCasePreset, "disabled")
    assert recipe is not None
    recipe.enabled = False
    if scope == "chat":
        session.add(
            ChatWorkflowUseCaseSelection(
                chat_id="chat", use_case="image_generation", preset_id="disabled"
            )
        )
    else:
        session.add(
            ProjectWorkflowUseCaseSelection(
                project_id="project", use_case="image_generation", preset_id="disabled"
            )
        )
    session.commit()
    with pytest.raises(module.WorkflowUseCasePresetResolutionError) as caught:
        module.resolve_workflow_use_case_preset(
            session, WorkflowUseCase.IMAGE_GENERATION, chat_id="chat"
        )
    assert caught.value.code == "workflow-use-case-preset-disabled"
    assert str(caught.value) == caught.value.code


def test_snapshot_settings_are_detached_in_both_directions(session: Session) -> None:
    resolve = _resolver().resolve_workflow_use_case_preset
    _recipe(session, "workspace", default=True)
    result = resolve(session, WorkflowUseCase.IMAGE_GENERATION)
    result.settings_json["options"]["sizes"].append(1024)
    recipe = session.get(WorkflowUseCasePreset, "workspace")
    assert recipe is not None
    assert recipe.settings_json["options"]["sizes"] == [256, 512]
    recipe.settings_json["options"]["sizes"].append(128)
    assert result.settings_json["options"]["sizes"] == [256, 512, 1024]


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"chat_id": "missing"}, "workflow-use-case-chat-not-found"),
        ({"project_id": "missing"}, "workflow-use-case-project-not-found"),
        ({"chat_id": "chat", "project_id": "different"}, "workflow-use-case-project-mismatch"),
    ],
)
def test_unknown_or_mismatched_scope_refuses(
    session: Session, kwargs: dict[str, str], code: str
) -> None:
    module = _resolver()
    _scopes(session)
    _recipe(session, "workspace", default=True)
    with pytest.raises(module.WorkflowUseCasePresetResolutionError) as caught:
        module.resolve_workflow_use_case_preset(session, WorkflowUseCase.IMAGE_GENERATION, **kwargs)
    assert caught.value.code == code and str(caught.value) == code


def test_other_use_case_choices_do_not_apply(session: Session) -> None:
    resolve = _resolver().resolve_workflow_use_case_preset
    _scopes(session)
    _recipe(session, "workspace", default=True)
    session.add(ChatWorkflowUseCaseSelection(chat_id="chat", use_case="image_edit", preset_id=None))
    session.commit()
    result = resolve(session, WorkflowUseCase.IMAGE_GENERATION, chat_id="chat")
    assert result.mode == "preset" and result.scope == "workspace"
    assert resolve(session, WorkflowUseCase.VIDEO_GENERATION, chat_id="chat").mode == "unconfigured"


def test_resolution_never_flushes_or_commits_pending_edits(session: Session) -> None:
    resolve = _resolver().resolve_workflow_use_case_preset
    _recipe(session, "workspace", default=True)
    pending = WorkflowUseCasePreset(id="pending", name="Pending", use_case="chat")
    session.add(pending)
    assert resolve(session, WorkflowUseCase.IMAGE_GENERATION).preset_id == "workspace"
    assert pending in session.new
    with session.no_autoflush:
        assert (
            session.scalar(
                select(WorkflowUseCasePreset.id).where(WorkflowUseCasePreset.id == "pending")
            )
            is None
        )


@pytest.mark.parametrize("scope", ["project", "chat"])
@pytest.mark.parametrize(
    "damage,code",
    [
        ("missing", "workflow-use-case-preset-not-found"),
        ("wrong-case", "workflow-use-case-preset-mismatch"),
        ("settings", "workflow-use-case-preset-settings-invalid"),
    ],
)
def test_stale_recipe_state_refuses_without_lower_scope_fallback(
    session: Session, scope: str, damage: str, code: str
) -> None:
    module = _resolver()
    _scopes(session)
    _recipe(session, "workspace", default=True)
    _recipe(session, "selected")
    selection = ChatWorkflowUseCaseSelection if scope == "chat" else ProjectWorkflowUseCaseSelection
    values = {"chat_id" if scope == "chat" else "project_id": scope}
    session.add(selection(**values, use_case="image_generation", preset_id="selected"))
    session.commit()
    # Simulate a database written without its relational constraints.
    session.execute(text("PRAGMA foreign_keys=OFF"))
    try:
        if damage == "missing":
            session.execute(update(selection).values(preset_id="missing"))
        elif damage == "wrong-case":
            session.execute(
                update(WorkflowUseCasePreset)
                .where(WorkflowUseCasePreset.id == "selected")
                .values(use_case="image_edit")
            )
        else:
            session.execute(
                update(WorkflowUseCasePreset)
                .where(WorkflowUseCasePreset.id == "selected")
                .values(settings_json=["neutral-invalid-setting"])
            )
        session.commit()
    finally:
        session.execute(text("PRAGMA foreign_keys=ON"))
    with pytest.raises(module.WorkflowUseCasePresetResolutionError) as caught:
        module.resolve_workflow_use_case_preset(
            session, WorkflowUseCase.IMAGE_GENERATION, chat_id="chat"
        )
    assert caught.value.code == code and str(caught.value) == code


@pytest.mark.parametrize("use_case", list(WorkflowUseCase))
def test_workspace_defaults_apply_only_to_the_exact_use_case(
    session: Session, use_case: WorkflowUseCase
) -> None:
    resolve = _resolver().resolve_workflow_use_case_preset
    for key in WorkflowUseCase:
        session.add(
            WorkflowUseCasePreset(
                id=key.value,
                name=key.value,
                use_case=key.value,
                is_default=True,
                settings_json={"seed": len(key.value)},
            )
        )
    session.commit()
    result = resolve(session, use_case)
    assert result.preset_id == use_case.value
    assert result.settings_json == {"seed": len(use_case.value)}


def test_scope_reads_exclude_conversation_payloads(session: Session) -> None:
    resolve = _resolver().resolve_workflow_use_case_preset
    _scopes(session)
    _recipe(session, "workspace", default=True)
    session.expunge_all()
    statements: list[str] = []
    engine = session.get_bind()
    assert isinstance(engine, Engine)

    def capture(
        connection: Connection,
        cursor: object,
        statement: str,
        parameters: object,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        assert (
            resolve(session, WorkflowUseCase.IMAGE_GENERATION, chat_id="chat").preset_id
            == "workspace"
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert not session.identity_map
    assert all(
        column not in statement
        for statement in statements
        for column in (
            "draft_prompt",
            "generation_settings_json",
            "origin_json",
            "web_settings_json",
        )
    )


def test_concurrent_choice_change_cannot_mix_scope_snapshots(session: Session) -> None:
    resolve = _resolver().resolve_workflow_use_case_preset
    _scopes(session)
    _recipe(session, "workspace", default=True)
    engine = session.get_bind()
    assert isinstance(engine, Engine)
    changed = False

    def change_project(
        connection: Connection,
        cursor: object,
        statement: str,
        parameters: object,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        nonlocal changed
        if changed or "FROM chat_workflow_use_case_selections" not in statement:
            return
        changed = True
        with Session(engine) as writer:
            writer.add(
                ProjectWorkflowUseCaseSelection(
                    project_id="project", use_case="image_generation", preset_id=None
                )
            )
            writer.commit()

    event.listen(engine, "after_cursor_execute", change_project)
    try:
        assert (
            resolve(session, WorkflowUseCase.IMAGE_GENERATION, chat_id="chat").preset_id
            == "workspace"
        )
        assert changed
    finally:
        event.remove(engine, "after_cursor_execute", change_project)
    session.rollback()
    current = resolve(session, WorkflowUseCase.IMAGE_GENERATION, chat_id="chat")
    assert current.mode == "automatic" and current.scope == "project"
