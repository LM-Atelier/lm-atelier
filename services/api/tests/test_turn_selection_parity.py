"""Every chat's turn resolved the workflow way and the legacy way, over a database built here.

The matrix covers each way a chat or its project can choose: nothing at all, a
profile the chat names, Auto named the old way, a workflow family, a profile's
compatibility family, Auto chosen as a workflow choice, a project's pinned
revision, a project's family, and a family whose model is not installed.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from selection_parity import (
    CHAT_PROFILE,
    FALLS_BACK,
    PROJECT_REVISION,
    REFUSED,
    WORKFLOW,
    Choice,
    TurnSelection,
    compare_every_chat,
    compare_turn_selection,
    summarize,
)
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from local_lm.db import Base
from local_lm.domain import Operation, utcnow
from local_lm.models import (
    Chat,
    ChatWorkflowSelection,
    ModelInstall,
    ModelProfile,
    Project,
    ProjectWorkflowSelection,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.workflow_compatibility import AUTO_PROFILE_ID, ensure_legacy_profile_workflow

T2I = Operation.TEXT_TO_IMAGE
I2I = Operation.IMAGE_TO_IMAGE
GRAPH = {"save": {"class_type": "SaveImage", "inputs": {}}}

#: A choice named by the matrix's own labels: (profile, revision).
Named = tuple[str | None, str | None]

#: What each chat's turn does today, each way. Legacy ignores a chat's
#: workflow family, so a family answer and the default profile are not
#: compared; only a compatibility answer can agree or disagree.
EXPECTED: dict[tuple[str, Operation], tuple[str, str, Named | None, Named | None, bool]] = {
    ("nothing", T2I): (WORKFLOW, "", ("painter", "create"), ("painter", "create"), True),
    ("nothing", I2I): (WORKFLOW, "", ("painter", "edit"), ("painter", "edit"), True),
    ("names sketcher", T2I): (FALLS_BACK, CHAT_PROFILE, None, ("sketcher", "create"), False),
    ("names sketcher", I2I): (FALLS_BACK, CHAT_PROFILE, None, ("sketcher", "edit"), False),
    ("old auto", T2I): (FALLS_BACK, CHAT_PROFILE, None, ("painter", "create"), False),
    ("old auto", I2I): (FALLS_BACK, CHAT_PROFILE, None, ("painter", "edit"), False),
    ("chose posters", T2I): (
        WORKFLOW,
        "",
        (None, "posters create"),
        ("painter", "create"),
        False,
    ),
    ("chose posters", I2I): (WORKFLOW, "", (None, "posters edit"), ("painter", "edit"), False),
    ("chose sketcher family", T2I): (
        WORKFLOW,
        "",
        ("sketcher", "create"),
        ("sketcher", "create"),
        True,
    ),
    ("chose sketcher family", I2I): (
        WORKFLOW,
        "",
        ("sketcher", "edit"),
        ("sketcher", "edit"),
        True,
    ),
    # The workflow path refuses a family whose model is not installed; the
    # legacy path it falls back to chooses that same model's profile.
    ("chose unplugged family", T2I): (
        FALLS_BACK,
        "explicit: model_unavailable",
        None,
        ("unplugged", "create"),
        False,
    ),
    ("chose unplugged family", I2I): (
        FALLS_BACK,
        "explicit: model_unavailable",
        None,
        ("unplugged", "edit"),
        False,
    ),
    ("auto", T2I): (WORKFLOW, "", ("painter", "create"), ("painter", "create"), True),
    ("auto", I2I): (WORKFLOW, "", ("painter", "edit"), ("painter", "edit"), True),
    ("in pinned project", T2I): (
        FALLS_BACK,
        PROJECT_REVISION,
        None,
        ("painter", "create"),
        False,
    ),
    # A pinned text-to-image revision cannot serve an edit, and nothing else is asked.
    ("in pinned project", I2I): (
        REFUSED,
        "legacy: ProjectWorkflowPinInvalid: operation_mismatch",
        None,
        None,
        False,
    ),
    # The legacy path refuses a project's family; only the workflow path answers.
    ("in posters project", T2I): (WORKFLOW, "", (None, "posters create"), None, False),
    ("in posters project", I2I): (WORKFLOW, "", (None, "posters edit"), None, False),
}


@pytest.fixture
def engine() -> Generator[Engine]:
    # One shared connection, so every session sees what the matrix committed.
    value = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    with value.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(value)
    yield value
    value.dispose()


def _orchestrator() -> ConversationOrchestrator:
    return ConversationOrchestrator(
        engines=cast(
            Any,
            SimpleNamespace(
                settings=SimpleNamespace(media_engine="comfyui", chat_engine="mock"),
                chat=SimpleNamespace(cancel=AsyncMock()),
                media=SimpleNamespace(cancel=AsyncMock()),
            ),
        ),
        artifacts=Mock(),
        events=cast(Any, SimpleNamespace(publish=AsyncMock())),
        scheduler=cast(Any, SimpleNamespace(publish_job=AsyncMock())),
        processes=cast(Any, SimpleNamespace(statuses=Mock(return_value=[]))),
        session_factory=Mock(),
    )


def _profile(
    session: Session, name: str, *, use_case: str, default: bool, active: bool = True
) -> ModelProfile:
    install = ModelInstall(
        name=name, role="image", engine="comfyui", local_path=f"C:/models/{name}", active=active
    )
    session.add(install)
    session.flush()
    profile = ModelProfile(
        name=name,
        engine="comfyui",
        role="image",
        model_install_id=install.id,
        use_case=use_case,
        is_default=default,
    )
    session.add(profile)
    session.flush()
    return profile


def _generic_revision(session: Session, operation: Operation, *, age: int) -> WorkflowRevision:
    """A trusted workflow outside any family, as the legacy path finds them: newest first."""
    definition = WorkflowDefinition(
        name=f"Generic {operation.value}",
        operation=operation.value,
        created_at=utcnow() - timedelta(days=age),
    )
    session.add(definition)
    session.flush()
    revision = WorkflowRevision(
        workflow_id=definition.id, version=1, engine="comfyui", api_graph_json=GRAPH, trusted=True
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    return revision


def _family(session: Session, name: str, *, use_case: str) -> tuple[WorkflowFamily, list[str]]:
    """A workflow family with a trusted create and edit variant, and their revisions."""
    family = WorkflowFamily(name=name, use_case=use_case)
    session.add(family)
    session.flush()
    revisions = []
    for key, operation in (("create", T2I), ("edit", I2I)):
        definition = WorkflowDefinition(
            family=family,
            variant_key=key,
            name=f"{name} {key}",
            operation=operation.value,
            created_at=utcnow() - timedelta(days=30),
        )
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="comfyui",
            api_graph_json=GRAPH,
            trusted=True,
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        revisions.append(revision.id)
    session.add(WorkflowPreference(family=family, selector_capability="image"))
    session.flush()
    return family, revisions


def _chat(session: Session, title: str, **fields: Any) -> Chat:
    chat = Chat(title=title, **fields)
    session.add(chat)
    session.flush()
    return chat


def _matrix(engine: Engine) -> dict[str, str]:
    """Build the chats, and return the identities the expectations name, by label."""
    with Session(engine, expire_on_commit=False) as session:
        painter = _profile(session, "Painter", use_case="paintings", default=True)
        sketcher = _profile(session, "Sketcher", use_case="sketches", default=False)
        unplugged = _profile(
            session, "Unplugged", use_case="engravings", default=False, active=False
        )
        create = _generic_revision(session, T2I, age=10)
        edit = _generic_revision(session, I2I, age=10)
        compatible: dict[str, WorkflowFamily] = {}
        for profile in (painter, sketcher, unplugged):
            family = ensure_legacy_profile_workflow(session, profile)
            assert family is not None
            compatible[profile.id] = family
        posters, (posters_create, posters_edit) = _family(session, "Posters", use_case="posters")

        ids = {
            "painter": painter.id,
            "sketcher": sketcher.id,
            "unplugged": unplugged.id,
            "create": create.id,
            "edit": edit.id,
            "posters create": posters_create,
            "posters edit": posters_edit,
            "painter family": compatible[painter.id].id,
            "sketcher family": compatible[sketcher.id].id,
        }
        ids["nothing"] = _chat(session, "Nothing chosen").id
        ids["names sketcher"] = _chat(
            session, "Names Sketcher", active_image_profile_id=sketcher.id
        ).id
        ids["old auto"] = _chat(
            session, "Auto the old way", active_image_profile_id=AUTO_PROFILE_ID
        ).id
        for title, mode, family in (
            ("chose posters", "family", posters),
            ("chose sketcher family", "family", compatible[sketcher.id]),
            ("chose unplugged family", "family", compatible[unplugged.id]),
            ("auto", "automatic", None),
        ):
            chat = _chat(session, title)
            session.add(
                ChatWorkflowSelection(
                    chat_id=chat.id,
                    selector_capability="image",
                    mode=mode,
                    workflow_family_id=family.id if family else None,
                )
            )
            ids[title] = chat.id
        pinned = Project(name="Pinned")
        family_project = Project(name="Posters project")
        session.add_all([pinned, family_project])
        session.flush()
        session.add_all(
            [
                ProjectWorkflowSelection(
                    project_id=pinned.id,
                    selector_capability="image",
                    mode="revision",
                    workflow_revision_id=create.id,
                ),
                ProjectWorkflowSelection(
                    project_id=family_project.id,
                    selector_capability="image",
                    mode="family",
                    workflow_family_id=posters.id,
                ),
            ]
        )
        ids["in pinned project"] = _chat(session, "In the pinned project", project_id=pinned.id).id
        ids["in posters project"] = _chat(
            session, "In the posters project", project_id=family_project.id
        ).id
        session.commit()
    return ids


def _named(choice: Choice | None, labels: dict[str, str]) -> Named | None:
    if choice is None:
        return None
    return (
        labels[choice.profile_id] if choice.profile_id else None,
        labels[choice.revision_id] if choice.revision_id else None,
    )


def _observed(
    item: TurnSelection, labels: dict[str, str]
) -> tuple[str, str, Named | None, Named | None, bool]:
    return (
        item.outcome,
        item.cause,
        _named(item.workflow, labels),
        _named(item.legacy, labels),
        item.compatibility,
    )


def test_every_chat_resolves_as_recorded(engine: Engine) -> None:
    ids = _matrix(engine)
    labels = {value: key for key, value in ids.items()}
    factory = sessionmaker(engine, expire_on_commit=False)

    results = compare_every_chat(_orchestrator(), factory, (T2I, I2I))

    observed = {(labels[item.chat_id], item.operation): item for item in results}
    # Every chat was compared for both operations, and nothing else was.
    assert len(results) == len(EXPECTED) == 18
    assert set(observed) == set(EXPECTED)
    for key, expected in EXPECTED.items():
        assert _observed(observed[key], labels) == expected, key
        # Agreement is asked of compatibility answers only, and they all agree.
        assert observed[key].agrees is (True if expected[4] else None), key
    assert summarize(results) == {
        WORKFLOW: 10,
        f"{FALLS_BACK}: {CHAT_PROFILE}": 4,
        f"{FALLS_BACK}: explicit: model_unavailable": 2,
        f"{FALLS_BACK}: {PROJECT_REVISION}": 1,
        f"{REFUSED}: legacy: ProjectWorkflowPinInvalid: operation_mismatch": 1,
    }


def test_a_compatibility_answer_the_legacy_path_would_not_choose_is_a_disagreement(
    engine: Engine,
) -> None:
    """The comparison is not vacuous: with the default workflow preference moved
    to another profile's compatibility family while the default profile stays,
    the two paths choose different models for the same turn, and it says so."""
    ids = _matrix(engine)
    labels = {value: key for key, value in ids.items()}
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        for family, default in (("painter family", False), ("sketcher family", True)):
            preference = session.scalar(
                select(WorkflowPreference).where(
                    WorkflowPreference.workflow_family_id == ids[family],
                    WorkflowPreference.selector_capability == "image",
                )
            )
            assert preference is not None
            preference.is_default = default
            session.flush()
        session.commit()

    item = compare_turn_selection(_orchestrator(), factory, ids["nothing"], T2I)

    assert _observed(item, labels) == (
        WORKFLOW,
        "",
        ("sketcher", "create"),
        ("painter", "create"),
        True,
    )
    assert item.agrees is False
    assert summarize([item])["workflow: disagrees with the legacy path"] == 1
