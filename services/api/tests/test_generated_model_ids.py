from __future__ import annotations

from local_lm.domain import new_id
from local_lm.models import (
    Base,
    ComfyRegistryInstall,
    ModelCapabilityEvidence,
    ModelComponentManifest,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)


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
    assert ModelComponentManifest.__table__.c.id.type.length == 64
    assert ModelCapabilityEvidence.__table__.c.id.type.length == 64
    assert WorkflowDefinition.__table__.c.id.type.length == 64
    assert WorkflowDefinition.__table__.c.family_id.type.length == 64
    assert WorkflowFamily.__table__.c.id.type.length == 64
    assert WorkflowPreference.__table__.c.id.type.length == 40
    assert WorkflowPreference.__table__.c.workflow_family_id.type.length == 64
    assert WorkflowRevision.__table__.c.workflow_id.type.length == 64
    assert ComfyRegistryInstall.__table__.c.id.type.length == 64

    for table in Base.metadata.tables.values():
        for column in table.primary_key.columns:
            if column.default is None or not callable(column.default.arg):
                continue
            generated_id = column.default.arg(None)
            assert isinstance(generated_id, str)
            assert column.type.length is not None
            assert len(generated_id) <= column.type.length, (
                f"{table.name}.{column.name} stores {len(generated_id)} characters in {column.type}"
            )

    assert len(workflow_id) <= WorkflowRevision.__table__.c.workflow_id.type.length
