"""An omitted source dependency is proven unnecessary, or it is not omitted."""

from __future__ import annotations

import pytest

from local_lm.source_omission_proof import (
    PENDING_KEY,
    OmissionProofError,
    OmissionRequirement,
    evidence_digest,
    pending_omission_requirement,
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


def test_the_record_says_which_workflow_it_was_about() -> None:
    """So a proof about one workflow can never read as one about another."""
    loaded = frozenset({"ImpactWildcard", "UpscaleImage"})
    evidence = prove_omission(_requirement(), observed_node_types=loaded)

    assert evidence["workflow_revision_id"] == "revision-1"
    assert evidence_digest(evidence) != evidence_digest(
        prove_omission(_requirement(workflow_revision_id="revision-2"), observed_node_types=loaded)
    )


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
    other = prove_omission(_requirement(**overrides), observed_node_types=loaded)

    assert evidence_digest(evidence) != evidence_digest(other)


def test_the_record_does_not_depend_on_the_order_it_was_given() -> None:
    loaded = frozenset({"ImpactWildcard", "UpscaleImage"})
    one = prove_omission(_requirement(), observed_node_types=loaded)
    other = prove_omission(
        _requirement(required_node_types=("UpscaleImage", "ImpactWildcard")),
        observed_node_types=loaded,
    )

    assert evidence_digest(one) == evidence_digest(other)


VALID_CANDIDATE = {
    "manifest_sha256": "a" * 64,
    "omitted_declarations": ["example @ git+https://github.com/owner/repo"],
    "workflow_revision_id": "revision-1",
    "required_node_types": ["ImpactWildcard"],
}


def test_no_candidate_at_all_is_an_ordinary_activation() -> None:
    assert pending_omission_requirement("install-1", {"reviewed_at": "then"}) is None


def test_a_whole_candidate_is_read_verbatim() -> None:
    requirement = pending_omission_requirement("install-1", {PENDING_KEY: VALID_CANDIDATE})

    assert requirement is not None
    assert requirement.omitted_declarations == ("example @ git+https://github.com/owner/repo",)
    assert requirement.required_node_types == ("ImpactWildcard",)


@pytest.mark.parametrize(
    "candidate",
    [
        "not a mapping",
        None,
        [],
        {**VALID_CANDIDATE, "omitted_declarations": [{}]},
        {**VALID_CANDIDATE, "omitted_declarations": [1]},
        {**VALID_CANDIDATE, "omitted_declarations": [""]},
        {**VALID_CANDIDATE, "omitted_declarations": ["  "]},
        {**VALID_CANDIDATE, "omitted_declarations": []},
        {**VALID_CANDIDATE, "omitted_declarations": "one"},
        {**VALID_CANDIDATE, "required_node_types": [{"name": "ImpactWildcard"}]},
        {**VALID_CANDIDATE, "required_node_types": [7]},
        {**VALID_CANDIDATE, "required_node_types": []},
        {**VALID_CANDIDATE, "manifest_sha256": 12345},
        {**VALID_CANDIDATE, "manifest_sha256": ""},
        {**VALID_CANDIDATE, "workflow_revision_id": {"id": "revision-1"}},
        {**VALID_CANDIDATE, "workflow_revision_id": None},
        {k: v for k, v in VALID_CANDIDATE.items() if k != "manifest_sha256"},
        {k: v for k, v in VALID_CANDIDATE.items() if k != "required_node_types"},
    ],
)
def test_a_malformed_candidate_refuses_rather_than_being_normalized(candidate: object) -> None:
    """str() would turn any of these into a plausible-looking identity.

    The proof would then be about whatever that produced, which is the exact
    opposite of what a record like this exists to do.
    """
    with pytest.raises(OmissionProofError) as raised:
        pending_omission_requirement("install-1", {PENDING_KEY: candidate})

    assert raised.value.code == "omission_candidate_unreadable"
