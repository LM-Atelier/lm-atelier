"""An omitted source dependency is proven unnecessary, or it is not omitted."""

from __future__ import annotations

import pytest

from local_lm.source_omission_proof import (
    OmissionProofError,
    OmissionRequirement,
    evidence_digest,
    proof_covers,
    prove_omission,
)


def _requirement(**overrides: object) -> OmissionRequirement:
    values: dict[str, object] = {
        "install_id": "install-1",
        "manifest_sha256": "a" * 64,
        "omitted_declarations": ("example @ git+https://github.com/owner/repo",),
        "workflow_revision_id": "revision-1",
        "required_node_types": ("ImpactWildcard", "UpscaleImage"),
    }
    values.update(overrides)
    return OmissionRequirement(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"install_id": ""}, "omission_package_unidentified"),
        ({"manifest_sha256": ""}, "omission_package_unidentified"),
        ({"omitted_declarations": ()}, "omission_declarations_missing"),
        ({"workflow_revision_id": ""}, "omission_workflow_unbound"),
        ({"required_node_types": ()}, "omission_workflow_unbound"),
    ],
)
def test_a_proof_that_cannot_say_what_it_is_about_is_not_stated(
    overrides: dict[str, object], code: str
) -> None:
    with pytest.raises(OmissionProofError) as raised:
        _requirement(**overrides)

    assert raised.value.code == code


def test_every_required_node_present_is_the_proof() -> None:
    evidence = prove_omission(
        _requirement(),
        observed_node_types=frozenset({"ImpactWildcard", "UpscaleImage", "SaveImage"}),
    )

    assert evidence["omitted_declarations"] == ["example @ git+https://github.com/owner/repo"]
    assert evidence["required_node_types"] == ["ImpactWildcard", "UpscaleImage"]


def test_one_missing_node_means_the_dependency_was_required() -> None:
    with pytest.raises(OmissionProofError) as raised:
        prove_omission(_requirement(), observed_node_types=frozenset({"ImpactWildcard"}))

    assert raised.value.code == "omission_required_nodes_missing"
    assert "UpscaleImage" in str(raised.value)


def test_a_runtime_that_reports_nothing_proves_nothing() -> None:
    """A worker that answered without loading anything is not evidence."""
    with pytest.raises(OmissionProofError) as raised:
        prove_omission(_requirement(), observed_node_types=frozenset())

    assert raised.value.code == "omission_inventory_empty"


def test_a_proof_about_one_workflow_is_not_permission_for_another() -> None:
    loaded = frozenset({"ImpactWildcard", "UpscaleImage"})
    evidence = prove_omission(_requirement(), observed_node_types=loaded)

    assert proof_covers(evidence, _requirement()) is True
    assert proof_covers(evidence, _requirement(workflow_revision_id="revision-2")) is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"install_id": "install-2"},
        {"manifest_sha256": "b" * 64},
        {"omitted_declarations": ("other @ git+https://github.com/owner/other",)},
        {"required_node_types": ("ImpactWildcard",)},
    ],
)
def test_the_digest_moves_when_anything_it_is_about_moves(overrides: dict[str, object]) -> None:
    loaded = frozenset({"ImpactWildcard", "UpscaleImage"})
    evidence = prove_omission(_requirement(), observed_node_types=loaded)

    assert proof_covers(evidence, _requirement(**overrides)) is False


def test_the_record_does_not_depend_on_the_order_it_was_given() -> None:
    loaded = frozenset({"ImpactWildcard", "UpscaleImage"})
    one = prove_omission(_requirement(), observed_node_types=loaded)
    other = prove_omission(
        _requirement(required_node_types=("UpscaleImage", "ImpactWildcard")),
        observed_node_types=loaded,
    )

    assert evidence_digest(one) == evidence_digest(other)
