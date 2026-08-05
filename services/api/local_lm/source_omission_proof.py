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
  declarations omitted, the workflow revision, and the required types, with a
  digest over that record. The candidate is not spent by succeeding, so every
  later activation proves it again against the runtime in front of it. That is
  the conservative reading: a proof recorded once says the nodes loaded then,
  and re-checking costs a comparison against an inventory that has to be read
  at startup anyway.

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


PENDING_KEY = "pending_source_omission"


@dataclass(frozen=True)
class PendingOmission:
    """A candidate on its way into the same commit as its install.

    The manifest hash is deliberately absent: it is the install's, and taking
    it from the row being written is the only way the two cannot disagree.
    """

    omitted_declarations: tuple[str, ...]
    workflow_revision_id: str
    required_node_types: tuple[str, ...]


def record_pending_omission(
    review: dict[str, Any],
    *,
    manifest_sha256: str,
    omitted_declarations: tuple[str, ...],
    workflow_revision_id: str,
    required_node_types: tuple[str, ...],
) -> dict[str, Any]:
    """Write the candidate a later activation will be held to.

    Written at preparation, when the package, the declarations, and the
    workflow that authorized looking at them are all in hand. Activation reads
    it rather than being told: a caller that could name its own manifest hash
    or workflow could prove anything about anything.
    """
    return {
        **review,
        PENDING_KEY: {
            "manifest_sha256": manifest_sha256,
            "omitted_declarations": sorted(omitted_declarations),
            "workflow_revision_id": workflow_revision_id,
            "required_node_types": sorted(set(required_node_types)),
        },
    }


def pending_omission_requirement(
    install_id: str, review: dict[str, Any]
) -> OmissionRequirement | None:
    """The candidate stored against this install, or None when there is none.

    Absent and unreadable are different answers, and treating them alike was a
    fail-open: an activation with no candidate is an ordinary one, so a corrupt
    or half-written record would have skipped the proof entirely and left an
    install active with its dependencies omitted. A record that exists and
    cannot be read is the one case where refusing outright is the only safe
    reading of it.
    """
    if PENDING_KEY not in review:
        return None
    candidate = review.get(PENDING_KEY)
    if not isinstance(candidate, dict):
        raise _unreadable()
    return OmissionRequirement(
        install_id=install_id,
        manifest_sha256=_exact_text(candidate.get("manifest_sha256")),
        omitted_declarations=_exact_texts(candidate.get("omitted_declarations")),
        workflow_revision_id=_exact_text(candidate.get("workflow_revision_id")),
        required_node_types=_exact_texts(candidate.get("required_node_types")),
    )


def _unreadable() -> OmissionProofError:
    return OmissionProofError(
        "omission_candidate_unreadable",
        "This package records an omitted dependency that cannot be read",
    )


def _exact_text(value: object) -> str:
    """A stored identity, refused unless it is already one.

    Never coerced. `str(value)` would turn a number, a mapping, or a blank
    into a plausible-looking identity and the proof would be about whatever
    that produced - which is the opposite of what a record like this is for.
    """
    if not isinstance(value, str) or not value.strip():
        raise _unreadable()
    return value


def _exact_texts(value: object) -> tuple[str, ...]:
    """A stored list of identities, kept verbatim or refused."""
    if not isinstance(value, list) or not value:
        raise _unreadable()
    return tuple(_exact_text(item) for item in value)
