"""Compare the recovery client's selected-item protocol with the served schema."""

from pathlib import Path

import pytest
from test_typescript_contract import (
    _admissible_values,
    _array_item_components,
    _declared_literals,
    _openapi_schemas,
    _typescript_field_type,
    _typescript_fields,
)

CONTRACTS = {
    "RecoveryCounts": "RecoveryCountsV1",
    "RecoveryImpact": "RecoveryImpactV1",
    "RecoveryItem": "RecoveryItemV1",
    "RecoveryPage": "RecoveryPageV1",
    "RecoveryCommand": "RecoveryCommandV1",
    "RecoveryResult": "RecoveryResultV1",
    "RecoveryBatchSelection": "RecoveryBatchSelectionV1",
    "RecoveryBatchMember": "RecoveryBatchMemberV1",
    "RecoveryBatchPreview": "RecoveryBatchPreviewV1",
    "RecoveryBatchCommand": "RecoveryBatchApplyV1",
    "RecoveryBatchResult": "RecoveryBatchResultV1",
}


@pytest.fixture(scope="module")
def recovery_contract():
    source = (Path(__file__).resolve().parents[3] / "apps/web/src/recoveryTypes.ts").read_text(
        encoding="utf-8"
    )
    return source, _openapi_schemas()


@pytest.mark.parametrize(("interface", "component"), sorted(CONTRACTS.items()))
def test_recovery_fields_and_closed_choices_match_the_served_protocol(
    recovery_contract, interface: str, component: str
) -> None:
    source, schemas = recovery_contract
    properties = schemas[component]["properties"]
    assert _typescript_fields(source, interface) == set(properties)
    for field, spec in properties.items():
        declared = _typescript_field_type(source, interface, field)
        values = _admissible_values(spec, schemas)
        if values:
            assert _declared_literals(source, declared) == {
                value for value in values if isinstance(value, str)
            }
        for target in _array_item_components(spec):
            if schemas[target].get("type") == "object":
                assert declared and declared.endswith("[]")
                assert CONTRACTS[declared.removesuffix("[]")] == target
