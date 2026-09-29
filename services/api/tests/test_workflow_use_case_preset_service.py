"""Keep recipe management atomic and scoped choices independent."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from importlib.util import find_spec
from threading import Event, local
from types import ModuleType
from typing import Any, cast

import pytest
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session

from local_lm.config import Settings
from local_lm.db import create_database_engine
from local_lm.models import Chat, Project, WorkflowUseCasePreset
from local_lm.workflow_use_case_presets_v1 import WorkflowUseCaseChoice, WorkflowUseCasePresetCreate
from local_lm.workflow_use_cases_v1 import WorkflowUseCase


@pytest.fixture
def session(settings: Settings) -> Iterator[Session]:
    engine = create_database_engine(settings)
    try:
        with Session(engine) as value:
            value.add_all([Chat(id="chat"), Project(id="project", name="Example")])
            value.commit()
            yield value
    finally:
        engine.dispose()


def _service() -> ModuleType:
    name = "local_lm.workflow_use_case_preset_service"
    assert find_spec(name) is not None, "Use-case recipe management is absent."
    return import_module(name)


def _payload(name: str, **values: object) -> WorkflowUseCasePresetCreate:
    return WorkflowUseCasePresetCreate.model_validate(
        {
            "name": name,
            "use_case": "image_generation",
            "settings_json": {"seed": 9},
            **values,
        }
    )


def test_create_and_replace_preserve_detached_recipe_values(session: Session) -> None:
    service = _service()
    payload = _payload("First", settings_json={"options": {"sizes": [256]}})
    first = service.create_preset(session, payload)
    assert first.name == "First" and first.use_case == WorkflowUseCase.IMAGE_GENERATION
    assert first.enabled and not first.builtin and not first.is_default
    cast(dict[str, Any], payload.settings_json)["options"]["sizes"].append(512)
    assert first.settings_json == {"options": {"sizes": [256]}}
    changed = service.replace_preset(session, first.id, _payload("Renamed"))
    assert changed.id == first.id and changed.name == "Renamed"
    assert changed.settings_json == {"seed": 9}
    assert first.name == "First"


def test_workspace_default_transfer_is_atomic_on_name_conflict(session: Session) -> None:
    service = _service()
    first = service.create_preset(session, _payload("First", is_default=True))
    second = service.create_preset(session, _payload("Second"))
    with pytest.raises(service.WorkflowUseCasePresetServiceError) as caught:
        service.replace_preset(session, second.id, _payload("First", is_default=True))
    assert caught.value.code == "workflow-use-case-preset-conflict"
    defaults = service.list_presets(session, use_case=WorkflowUseCase.IMAGE_GENERATION)
    assert [row.id for row in defaults if row.is_default] == [first.id]
    session.rollback()
    switched = service.replace_preset(session, second.id, _payload("Second", is_default=True))
    assert switched.is_default
    assert [row.id for row in service.list_presets(session) if row.is_default] == [second.id]


@pytest.mark.parametrize("scope", ["chat", "project"])
def test_inheritance_automatic_and_explicit_choices_remain_distinct(
    session: Session, scope: str
) -> None:
    service = _service()
    first = service.create_preset(session, _payload("First"))
    assert (
        service.read_choice(session, scope, scope, WorkflowUseCase.IMAGE_GENERATION).root.mode
        == "inherit"
    )
    session.rollback()
    for value in [
        {"mode": "automatic"},
        {"mode": "preset", "preset_id": first.id},
        {"mode": "inherit"},
    ]:
        choice = WorkflowUseCaseChoice.model_validate(value)
        saved = service.write_choice(
            session, scope, scope, WorkflowUseCase.IMAGE_GENERATION, choice
        )
        assert saved.model_dump() == value
        assert (
            service.read_choice(
                session, scope, scope, WorkflowUseCase.IMAGE_GENERATION
            ).model_dump()
            == value
        )
        session.rollback()
    chat, project = session.get(Chat, "chat"), session.get(Project, "project")
    assert chat is not None and project is not None
    assert chat.generation_preset_ids_json == {}
    assert project.generation_preset_ids_json == {}


@pytest.mark.parametrize("scope", ["chat", "project"])
@pytest.mark.parametrize("action", ["disable", "delete"])
def test_selected_recipes_cannot_be_disabled_or_deleted(
    session: Session, scope: str, action: str
) -> None:
    service = _service()
    first = service.create_preset(session, _payload("First"))
    service.write_choice(
        session,
        scope,
        scope,
        WorkflowUseCase.IMAGE_GENERATION,
        WorkflowUseCaseChoice.model_validate({"mode": "preset", "preset_id": first.id}),
    )
    with pytest.raises(service.WorkflowUseCasePresetServiceError) as caught:
        if action == "disable":
            service.replace_preset(session, first.id, _payload("First", enabled=False))
        else:
            service.delete_preset(session, first.id)
    assert caught.value.code == "workflow-use-case-preset-in-use"
    assert (
        service.read_choice(session, scope, scope, WorkflowUseCase.IMAGE_GENERATION).root.preset_id
        == first.id
    )
    session.rollback()
    service.write_choice(
        session,
        scope,
        scope,
        WorkflowUseCase.IMAGE_GENERATION,
        WorkflowUseCaseChoice.model_validate({"mode": "inherit"}),
    )
    service.delete_preset(session, first.id)
    assert service.list_presets(session) == []


@pytest.mark.parametrize("action", ["replace", "delete"])
def test_built_in_recipes_are_not_editable(session: Session, action: str) -> None:
    service = _service()
    session.add(
        WorkflowUseCasePreset(
            id="builtin", name="Example", use_case="image_generation", builtin=True
        )
    )
    session.commit()
    with pytest.raises(service.WorkflowUseCasePresetServiceError) as caught:
        if action == "replace":
            service.replace_preset(session, "builtin", _payload("Changed"))
        else:
            service.delete_preset(session, "builtin")
    assert caught.value.code == "workflow-use-case-preset-builtin"


@pytest.mark.parametrize("scope", ["chat", "project"])
@pytest.mark.parametrize(
    "state,code",
    [
        ("missing", "workflow-use-case-preset-not-found"),
        ("disabled", "workflow-use-case-preset-disabled"),
        ("wrong-case", "workflow-use-case-preset-mismatch"),
    ],
)
def test_invalid_recipe_choices_do_not_replace_automatic(
    session: Session, scope: str, state: str, code: str
) -> None:
    service = _service()
    first = service.create_preset(
        session,
        _payload(
            "First",
            enabled=state != "disabled",
            use_case="image_edit" if state == "wrong-case" else "image_generation",
        ),
    )
    service.write_choice(
        session,
        scope,
        scope,
        WorkflowUseCase.IMAGE_GENERATION,
        WorkflowUseCaseChoice.model_validate({"mode": "automatic"}),
    )
    with pytest.raises(service.WorkflowUseCasePresetServiceError) as caught:
        service.write_choice(
            session,
            scope,
            scope,
            WorkflowUseCase.IMAGE_GENERATION,
            WorkflowUseCaseChoice.model_validate(
                {"mode": "preset", "preset_id": "missing" if state == "missing" else first.id}
            ),
        )
    assert caught.value.code == code and str(caught.value) == code
    assert (
        service.read_choice(session, scope, scope, WorkflowUseCase.IMAGE_GENERATION).root.mode
        == "automatic"
    )


def test_recipe_use_case_is_immutable(session: Session) -> None:
    service = _service()
    first = service.create_preset(session, _payload("First"))
    with pytest.raises(service.WorkflowUseCasePresetServiceError) as caught:
        service.replace_preset(session, first.id, _payload("First", use_case="image_edit"))
    assert caught.value.code == "workflow-use-case-preset-mismatch"


def test_default_recipe_requires_removing_the_default_before_deletion(session: Session) -> None:
    service = _service()
    first = service.create_preset(session, _payload("First", is_default=True))
    with pytest.raises(service.WorkflowUseCasePresetServiceError) as caught:
        service.delete_preset(session, first.id)
    assert caught.value.code == "workflow-use-case-preset-in-use"
    service.replace_preset(session, first.id, _payload("First"))
    service.delete_preset(session, first.id)
    assert service.list_presets(session) == []


def test_workspace_default_can_select_and_clear_a_built_in_recipe(session: Session) -> None:
    service = _service()
    custom = service.create_preset(session, _payload("Custom", is_default=True))
    session.add(
        WorkflowUseCasePreset(
            id="builtin",
            name="Example",
            use_case="image_generation",
            builtin=True,
            settings_json={"seed": 17},
        )
    )
    session.commit()
    service.set_workspace_default(session, WorkflowUseCase.IMAGE_GENERATION, "builtin")
    rows = {row.id: row for row in service.list_presets(session)}
    assert rows["builtin"].is_default and not rows[custom.id].is_default
    assert rows["builtin"].builtin and rows["builtin"].settings_json == {"seed": 17}
    assert rows["builtin"].name == "Example"
    session.rollback()
    service.set_workspace_default(session, WorkflowUseCase.IMAGE_GENERATION, None)
    assert not any(row.is_default for row in service.list_presets(session))


@pytest.mark.parametrize(
    "state,refusal",
    [("missing", "not-found"), ("disabled", "disabled"), ("wrong-case", "mismatch")],
)
def test_invalid_workspace_default_preserves_the_previous_choice(
    session: Session, state: str, refusal: str
) -> None:
    service = _service()
    previous = service.create_preset(session, _payload("Previous", is_default=True))
    target = service.create_preset(
        session,
        _payload(
            "Target",
            enabled=state != "disabled",
            use_case="image_edit" if state == "wrong-case" else "image_generation",
        ),
    )
    with pytest.raises(service.WorkflowUseCasePresetServiceError) as caught:
        service.set_workspace_default(
            session,
            WorkflowUseCase.IMAGE_GENERATION,
            "missing" if state == "missing" else target.id,
        )
    assert caught.value.code == "workflow-use-case-preset-" + refusal
    assert [row.id for row in service.list_presets(session) if row.is_default] == [previous.id]


def test_workspace_default_changes_preserve_other_use_cases_and_scoped_choices(
    session: Session,
) -> None:
    service = _service()
    first = service.create_preset(session, _payload("First", is_default=True))
    second = service.create_preset(session, _payload("Second"))
    edit = service.create_preset(session, _payload("Edit", use_case="image_edit", is_default=True))
    service.write_choice(
        session,
        "chat",
        "chat",
        WorkflowUseCase.IMAGE_GENERATION,
        WorkflowUseCaseChoice.model_validate({"mode": "preset", "preset_id": first.id}),
    )
    service.set_workspace_default(session, WorkflowUseCase.IMAGE_GENERATION, second.id)
    assert {row.id for row in service.list_presets(session) if row.is_default} == {
        second.id,
        edit.id,
    }
    assert (
        service.read_choice(
            session, "chat", "chat", WorkflowUseCase.IMAGE_GENERATION
        ).root.preset_id
        == first.id
    )


def test_mutation_refuses_without_committing_unrelated_pending_work(session: Session) -> None:
    service = _service()
    pending = Chat(id="pending")
    session.add(pending)
    with pytest.raises(service.WorkflowUseCasePresetServiceError) as caught:
        service.create_preset(session, _payload("First"))
    assert caught.value.code == "workflow-use-case-preset-conflict"
    assert pending in session.new
    with session.no_autoflush:
        assert session.scalar(select(WorkflowUseCasePreset.id)) is None


@pytest.mark.parametrize("scope", ["chat", "project"])
@pytest.mark.parametrize("mutation", ["disable", "delete"])
@pytest.mark.parametrize("selection_first", [True, False])
def test_concurrent_selection_and_recipe_changes_remain_consistent(
    session: Session, scope: str, mutation: str, selection_first: bool
) -> None:
    service = _service()
    recipe = service.create_preset(session, _payload("First"))
    engine = session.get_bind()
    assert isinstance(engine, Engine)
    acquired, contending = Event(), Event()
    worker = local()

    def before_execute(
        conn: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        if statement == "BEGIN IMMEDIATE" and getattr(worker, "order", None) == "second":
            contending.set()

    def after_execute(
        conn: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        if statement == "BEGIN IMMEDIATE" and getattr(worker, "order", None) == "first":
            acquired.set()
            assert contending.wait(10), "The competing writer did not reach the database."

    def perform(order: str, action: str) -> str:
        worker.order = order
        with Session(engine) as separate:
            try:
                if action == "select":
                    service.write_choice(
                        separate,
                        scope,
                        scope,
                        WorkflowUseCase.IMAGE_GENERATION,
                        WorkflowUseCaseChoice.model_validate(
                            {"mode": "preset", "preset_id": recipe.id}
                        ),
                    )
                elif action == "disable":
                    service.replace_preset(separate, recipe.id, _payload("First", enabled=False))
                else:
                    service.delete_preset(separate, recipe.id)
            except service.WorkflowUseCasePresetServiceError as exc:
                return cast(str, exc.code)
        return "saved"

    event.listen(engine, "before_cursor_execute", before_execute)
    event.listen(engine, "after_cursor_execute", after_execute)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(perform, "first", "select" if selection_first else mutation)
            assert acquired.wait(10), "The first writer did not acquire the database."
            second = pool.submit(perform, "second", mutation if selection_first else "select")
            assert first.result(timeout=15) == "saved"
            refusal = (
                "in-use"
                if selection_first
                else "disabled"
                if mutation == "disable"
                else "not-found"
            )
            assert second.result(timeout=15) == "workflow-use-case-preset-" + refusal
    finally:
        event.remove(engine, "before_cursor_execute", before_execute)
        event.remove(engine, "after_cursor_execute", after_execute)
    selected = service.read_choice(session, scope, scope, WorkflowUseCase.IMAGE_GENERATION).root
    assert selected.mode == ("preset" if selection_first else "inherit")
    remaining = session.get(WorkflowUseCasePreset, recipe.id)
    if selection_first:
        assert remaining is not None and remaining.enabled
    elif mutation == "disable":
        assert remaining is not None and not remaining.enabled
    else:
        assert remaining is None
