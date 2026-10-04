"""Install offer responses share the finite codes used by their writer."""

from typing import get_args

import pytest
from pydantic import ValidationError
from test_typescript_contract import _admissible_values

from local_lm.schemas import WorkflowInstallOfferOut
from local_lm.workflow_install_offers import WorkflowInstallOfferInvalidationCode


def _codes(declaration: object) -> set[str]:
    result: set[str] = set()
    for member in get_args(declaration):
        if isinstance(member, str):
            result.add(member)
        else:
            result.update(_codes(member))
    return result


def _offer(code: str | None) -> dict[str, object]:
    return {
        "id": "offer-garden",
        "workflow_revision_id": "revision-garden",
        "workflow_artifact_sha256": "a" * 64,
        "dependency_contract_sha256": "b" * 64,
        "binding_plan_sha256": "c" * 64,
        "offer_sha256": "d" * 64,
        "assets": [],
        "plan_count": 1,
        "total_bytes": 64,
        "status": "invalidated",
        "queued_at": None,
        "completed_at": None,
        "invalidated_at": None,
        "invalidation_code": code,
        "invalidation_reason": "The reviewed installation changed.",
    }


def test_offer_response_codes_match_the_finite_writer_vocabulary() -> None:
    property_schema = WorkflowInstallOfferOut.model_json_schema()["properties"]["invalidation_code"]
    assert set(_admissible_values(property_schema, {}) or []) == _codes(
        WorkflowInstallOfferInvalidationCode
    )


def test_every_declared_offer_code_remains_a_valid_response() -> None:
    for code in _codes(WorkflowInstallOfferInvalidationCode):
        assert WorkflowInstallOfferOut.model_validate(_offer(code)).invalidation_code == code


def test_an_offer_without_an_invalidation_code_remains_a_valid_response() -> None:
    assert WorkflowInstallOfferOut.model_validate(_offer(None)).invalidation_code is None


def test_an_undeclared_offer_code_is_refused_by_the_response_model() -> None:
    with pytest.raises(ValidationError):
        WorkflowInstallOfferOut.model_validate(_offer("unknown-installation-state"))
