"""Proving that an omitted source dependency was not needed.

A package can declare a dependency as a Git URL with no exact commit. Impact
Pack declares one and WAS 3.0.1 declares three. Those cannot be installed
without blessing content that may change after review, so the only honest
paths are to refuse the package or to show that the dependency was never
required for the work being asked of it.

This is the second. It is a proof, not an assumption, and it is deliberately
hard to satisfy:

- The package is activated with the declaration omitted, inside the existing
  reversible activation, so a failure restores exactly what was running.
- A service that merely starts proves nothing. The authorized workflow's exact
  required node types must be present in the inventory afterwards - all of
  them, by name.
- The conclusion is bound to what produced it: the package's manifest, the
  declarations omitted, the workflow revision, and the required types. A
  digest over that record is what a later activation is checked against, so
  proving one workflow can never quietly become permission for another.

Nothing here rewrites a requirement, excepts a package by name, or treats a
mutable source as trusted. It records that a specific package, missing
specific declarations, loaded the specific nodes a specific workflow needs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


class OmissionProofError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class OmissionRequirement:
    """What must hold for an omission to be provable, stated before the trial."""

    install_id: str
    manifest_sha256: str
    #: The exact declaration lines left out, verbatim, never normalized.
    omitted_declarations: tuple[str, ...]
    workflow_revision_id: str
    required_node_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.install_id or not self.manifest_sha256:
            raise OmissionProofError(
                "omission_package_unidentified",
                "An omission proof must name the exact package it was made about",
            )
        if not self.omitted_declarations:
            raise OmissionProofError(
                "omission_declarations_missing",
                "An omission proof must state which declarations were left out",
            )
        if not self.workflow_revision_id or not self.required_node_types:
            raise OmissionProofError(
                "omission_workflow_unbound",
                "An omission proof must name the workflow whose nodes it verified",
            )


def prove_omission(
    requirement: OmissionRequirement,
    *,
    observed_node_types: frozenset[str],
) -> dict[str, Any]:
    """Judge one trial activation, or refuse to call it a proof.

    Fail-closed in the direction that matters: a missing node type refuses,
    and so does an empty inventory, which is what a worker that answered
    without loading anything looks like.
    """
    if not observed_node_types:
        raise OmissionProofError(
            "omission_inventory_empty",
            "The runtime reported no node types, so nothing was proven about it",
        )
    missing = sorted(set(requirement.required_node_types) - observed_node_types)
    if missing:
        raise OmissionProofError(
            "omission_required_nodes_missing",
            f"The workflow still needs {missing[0]}, so the omitted dependency was required",
        )
    return _evidence(requirement)


def evidence_digest(evidence: dict[str, Any]) -> str:
    """The digest a later activation is checked against."""
    encoded = json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def proof_covers(evidence: dict[str, Any], requirement: OmissionRequirement) -> bool:
    """Whether a stored proof was made about exactly this situation.

    Compared as a whole rather than field by field: the point of recording the
    workflow and the declarations is that a proof about one of them is not
    permission for another, and a partial match is exactly how that would
    happen.
    """
    return evidence_digest(evidence) == evidence_digest(_evidence(requirement))


def _evidence(requirement: OmissionRequirement) -> dict[str, Any]:
    return {
        "version": 1,
        "install_id": requirement.install_id,
        "manifest_sha256": requirement.manifest_sha256,
        # Sorted for a stable digest; the lines themselves are untouched, so
        # the record says what the package actually declared.
        "omitted_declarations": sorted(requirement.omitted_declarations),
        "workflow_revision_id": requirement.workflow_revision_id,
        "required_node_types": sorted(set(requirement.required_node_types)),
    }
