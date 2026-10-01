from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from sqlalchemy import event
from sqlalchemy.orm import Session

from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import WorkflowDefinition, WorkflowRevision


def _seed(*, bound: bool) -> tuple[str, str]:
    identifiers = []
    with SessionLocal() as session:
        for index in range(151):
            definition = WorkflowDefinition(
                name=f"Neutral workflow {index}",
                operation="text_to_image",
                created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
            )
            revision = WorkflowRevision(
                definition=definition,
                version=1,
                engine="mock",
                trusted=True,
                api_graph_json={"neutral": "x" * 4096},
                dependencies_json={"model_install_ids": ["selected-install"]}
                if bound and index < 2
                else {},
            )
            session.add_all([definition, revision])
            session.flush()
            definition.current_revision_id = revision.id
            identifiers.append(revision.id)
        session.commit()
    return identifiers[1], identifiers[-1]


@pytest.mark.parametrize("bound", [False, True])
def test_fallback_selection_bounds_live_definitions_and_graphs(app: FastAPI, bound: bool) -> None:
    matched, generic = _seed(bound=bound)
    peaks = {"definitions": 0, "revisions": 0}

    def track(session: Session, _instance: object) -> None:
        for key, model in (("definitions", WorkflowDefinition), ("revisions", WorkflowRevision)):
            peaks[key] = max(
                peaks[key], sum(isinstance(row, model) for row in session.identity_map.values())
            )

    event.listen(Session, "loaded_as_persistent", track)
    try:
        with SessionLocal() as session:
            chosen = app.state.services.orchestrator._workflow_for_operation(
                session,
                Operation.TEXT_TO_IMAGE,
                model_install_id="selected-install",
            )
            assert chosen is not None
            assert chosen.id == (matched if bound else generic)
    finally:
        event.remove(Session, "loaded_as_persistent", track)
    assert peaks["definitions"] <= 101, peaks
    assert peaks["revisions"] <= 3, peaks


def test_nonmatching_model_workflows_leave_the_newest_generic_fallback(app: FastAPI) -> None:
    _matched, generic = _seed(bound=True)
    with SessionLocal() as session:
        chosen = app.state.services.orchestrator._workflow_for_operation(
            session,
            Operation.TEXT_TO_IMAGE,
            model_install_id="other-install",
        )
        assert chosen is not None and chosen.id == generic
