from __future__ import annotations

from local_lm.workflow_attestation import (
    attestation_state,
    build_evidence,
    digest_of_names,
    staleness_reasons,
    unresolved_node_types,
)

ARTIFACT = "a" * 64


def evidence(**overrides: object) -> object:
    base: dict[str, object] = {
        "artifact_sha256": ARTIFACT,
        "node_inventory": ["KSampler", "LoadImage"],
        "whitelist": ["KSampler", "LoadImage"],
        "required_node_types": ["KSampler"],
    }
    base.update(overrides)
    return build_evidence(**base)  # type: ignore[arg-type]


def test_a_relisted_inventory_is_not_a_changed_one() -> None:
    """The inventory comes from a runtime listing and the whitelist from a file,
    and neither promises an order. An order-sensitive digest would report the
    world as changed every time something was merely relisted."""

    assert digest_of_names(["b", "a"]) == digest_of_names(["a", "b"])
    assert digest_of_names(["a", "a", "b"]) == digest_of_names(["a", "b"])
    assert digest_of_names(["a"]) != digest_of_names(["a", "b"])


def test_a_node_must_be_both_installed_and_reviewed() -> None:
    """Present-but-unreviewed is the case that matters: a package can install a
    node type the runtime happily reports, and attesting on presence alone would
    launder an unreviewed package into a passing check."""

    assert (
        unresolved_node_types(["KSampler"], node_inventory=["KSampler"], whitelist=["KSampler"])
        == ()
    )
    # Installed, never reviewed.
    assert unresolved_node_types(
        ["Sketchy"], node_inventory=["Sketchy"], whitelist=["KSampler"]
    ) == ("Sketchy",)
    # Reviewed, not installed.
    assert unresolved_node_types(
        ["Sketchy"], node_inventory=["KSampler"], whitelist=["Sketchy"]
    ) == ("Sketchy",)


def test_an_unchanged_world_leaves_the_attestation_current() -> None:
    stored = evidence()
    assert staleness_reasons(stored, evidence()) == ()  # type: ignore[arg-type]
    assert attestation_state(stored, evidence())["current"] is True  # type: ignore[arg-type]


def test_each_input_that_moves_is_named_when_it_moves() -> None:
    """Every one of these lives outside the revision, in a subsystem with no
    reason to know this table exists, which is why the answer is recomputed
    rather than stored."""

    stored = evidence(
        runtime_contract_sha256="b" * 64, launch_scope_sha256="c" * 64, runtime_managed=True
    )
    cases = {
        "the workflow itself changed since it was checked": {"artifact_sha256": "d" * 64},
        "the installed node types changed": {"node_inventory": ["KSampler"]},
        "the reviewed node list changed": {"whitelist": ["KSampler"]},
        "the runtime it was checked against changed": {"runtime_contract_sha256": "e" * 64},
        "the launch scope changed": {"launch_scope_sha256": "f" * 64},
        "the workflow moved between a managed and an unmanaged runtime": {"runtime_managed": False},
    }
    for expected, mutation in cases.items():
        settings: dict[str, object] = {
            "runtime_contract_sha256": "b" * 64,
            "launch_scope_sha256": "c" * 64,
            "runtime_managed": True,
        }
        settings.update(mutation)
        reasons = staleness_reasons(stored, evidence(**settings))  # type: ignore[arg-type]
        assert expected in reasons, f"{mutation} should report {expected!r}, got {reasons}"


def test_a_field_that_was_never_recorded_cannot_have_changed() -> None:
    """An attestation taken without a managed runtime must not turn stale the
    moment one appears - it never made a claim about it."""

    stored = evidence()
    assert stored.runtime_contract_sha256 is None  # type: ignore[attr-defined]
    later = evidence(runtime_contract_sha256="b" * 64)
    assert staleness_reasons(stored, later) == ()  # type: ignore[arg-type]


def test_never_checked_reads_differently_from_checked_and_stale() -> None:
    """ "Nobody has looked" and "the answer expired" are different answers, and a
    caller that showed them the same way would be hiding which one it is."""

    never = attestation_state(None, evidence())  # type: ignore[arg-type]
    assert never["attested"] is False and never["current"] is False

    stale = attestation_state(evidence(), evidence(artifact_sha256="d" * 64))  # type: ignore[arg-type]
    assert stale["attested"] is True and stale["current"] is False
    assert stale["reasons"] != never["reasons"]
