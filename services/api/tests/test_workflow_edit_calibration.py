from __future__ import annotations

from copy import deepcopy

import pytest

from local_lm.project_dependencies import parse_dependency_manifest
from local_lm.workflow_edit_calibration import (
    EDIT_CALIBRATION_SCHEMA_KEY,
    safe_workflow_edit_calibration,
    standard_edit_calibration,
    validate_workflow_edit_calibration,
)


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "strength": {
                "type": "number",
                "default": 0.9,
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "steps": {"type": "integer", "default": 4},
        },
        EDIT_CALIBRATION_SCHEMA_KEY: {
            "version": 1,
            "edit_strength": {
                "parameter": "strength",
                "minimum": 0.0,
                "maximum": 1.0,
                "recommended": {
                    "minimal": 0.3,
                    "localized": 0.45,
                    "replacement": 0.6,
                    "global": 0.8,
                    "fallback": 0.5,
                },
            },
            "schedule": {
                "steps_parameter": "steps",
                "minimum_effective_steps": {
                    "localized": 2,
                    "replacement": 3,
                    "global": 3,
                },
            },
        },
    }


def test_standard_contract_is_valid_and_hashes_deterministically() -> None:
    schema = _schema()
    schema[EDIT_CALIBRATION_SCHEMA_KEY] = standard_edit_calibration(
        parameter="strength",
        minimum=0.0,
        maximum=1.0,
        steps_parameter="steps",
    )

    first = validate_workflow_edit_calibration(schema)
    second = validate_workflow_edit_calibration(deepcopy(schema))

    assert first is not None
    assert first == second
    assert first.parameter == "strength"
    assert first.recommended["replacement"] == 0.66
    assert first.minimum_effective_steps == {
        "localized": 2,
        "replacement": 3,
        "global": 3,
    }
    assert len(first.contract_hash) == 64


def test_absent_contract_remains_backward_compatible() -> None:
    schema = {"type": "object", "properties": {}}

    assert validate_workflow_edit_calibration(schema) is None
    assert safe_workflow_edit_calibration(schema) is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda contract: contract.update({"version": 2}),
            "version must be 1",
        ),
        (
            lambda contract: contract["edit_strength"].update({"parameter": "missing"}),
            "must identify a numeric workflow setting",
        ),
        (
            lambda contract: contract["edit_strength"]["recommended"].update({"replacement": 2.0}),
            "recommendation is outside its bounds",
        ),
        (
            lambda contract: contract["schedule"]["minimum_effective_steps"].update(
                {"replacement": 0}
            ),
            "integers from 1 to 10000",
        ),
        (
            lambda contract: contract.update({"unexpected": True}),
            "contains unsupported fields",
        ),
    ],
)
def test_invalid_contracts_fail_closed(mutation, message: str) -> None:  # type: ignore[no-untyped-def]
    schema = _schema()
    contract = schema[EDIT_CALIBRATION_SCHEMA_KEY]
    assert isinstance(contract, dict)
    mutation(contract)

    with pytest.raises(ValueError, match=message):
        validate_workflow_edit_calibration(schema)
    assert safe_workflow_edit_calibration(schema) is None


def test_project_dependency_manifest_preserves_and_validates_contract() -> None:
    schema = _schema()
    manifest = {
        "profiles": [],
        "presets": [],
        "workflows": [
            {
                "source_id": "workflow-source",
                "name": "Portable calibrated edit",
                "operation": "image_to_image",
                "description": "",
                "current_revision_source_id": "revision-source",
                "revisions": [
                    {
                        "source_id": "revision-source",
                        "source_version": 1,
                        "engine": "mock",
                        "engine_version": None,
                        "ui_graph": {},
                        "api_graph": {"node": {"class_type": "Mock"}},
                        "input_schema": schema,
                        "dependencies": {},
                        "trusted": False,
                    }
                ],
            }
        ],
    }

    parsed = parse_dependency_manifest(manifest)

    assert parsed.workflows[0].revisions[0].input_schema == schema
    invalid = deepcopy(manifest)
    invalid["workflows"][0]["revisions"][0]["input_schema"][EDIT_CALIBRATION_SCHEMA_KEY][
        "version"
    ] = 2
    with pytest.raises(ValueError, match="version must be 1"):
        parse_dependency_manifest(invalid)
