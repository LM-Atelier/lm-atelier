from __future__ import annotations

from sqlalchemy import ColumnDefault, ColumnElement, String

from local_lm.db import Base
from local_lm.domain import new_id
from local_lm.models import (
    ComfyRegistryInstall,
    ModelCapabilityEvidence,
    ModelComponentManifest,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)


def _string_type(column: ColumnElement[object]) -> String:
    assert isinstance(column.type, String)
    return column.type


def test_generated_model_ids_fit_their_declared_columns() -> None:
    component_id = new_id("component")
    evidence_id = new_id("evidence")
    workflow_id = new_id("workflow")
    workflow_family_id = new_id("wffamily")
    workflow_preference_id = new_id("wfpref")
    registry_id = new_id("registry")

    assert len(component_id) == 42
    assert len(evidence_id) == 41
    assert len(workflow_id) == 41
    assert len(workflow_family_id) == 41
    assert len(workflow_preference_id) == 39
    assert len(registry_id) == 41
    assert _string_type(ModelComponentManifest.__table__.c.id).length == 64
    assert _string_type(ModelCapabilityEvidence.__table__.c.id).length == 64
    assert _string_type(WorkflowDefinition.__table__.c.id).length == 64
    assert _string_type(WorkflowDefinition.__table__.c.family_id).length == 64
    assert _string_type(WorkflowFamily.__table__.c.id).length == 64
    assert _string_type(WorkflowPreference.__table__.c.id).length == 40
    assert _string_type(WorkflowPreference.__table__.c.workflow_family_id).length == 64
    assert _string_type(WorkflowRevision.__table__.c.workflow_id).length == 64
    assert _string_type(ComfyRegistryInstall.__table__.c.id).length == 64

    for table in Base.metadata.tables.values():
        for column in table.primary_key.columns:
            if column.default is None:
                continue
            assert isinstance(column.default, ColumnDefault)
            if not callable(column.default.arg):
                continue
            generated_id = column.default.arg(None)
            assert isinstance(generated_id, str)
            column_type = _string_type(column)
            assert column_type.length is not None
            assert len(generated_id) <= column_type.length, (
                f"{table.name}.{column.name} stores {len(generated_id)} characters in {column.type}"
            )

    workflow_id_type = _string_type(WorkflowRevision.__table__.c.workflow_id)
    assert workflow_id_type.length is not None
    assert len(workflow_id) <= workflow_id_type.length
