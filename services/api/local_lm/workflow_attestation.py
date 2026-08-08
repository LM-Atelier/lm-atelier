"""What this machine checked about a revision, and whether that check still holds.

Derived trust and attestation answer different questions, and conflating them
would turn import into a way to run an unreviewed package. `workflow_trust`
proves a revision could be rebuilt here from a template this build already
ships. Attestation proves something weaker and more perishable: at one moment,
every node type this graph executes resolved to something installed and
reviewed, and every asset it names was present.

Weaker claims expire. Every input to that check lives outside the revision - a
package is uninstalled, a runtime is replaced, a whitelist is edited - and none
of those events would come back to update a stored verdict. So nothing here
records whether an attestation is still good. It records what was true and what
it was true *of*, and the answer is recomputed against the world each time it is
read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

ATTESTATION_VERSION = 1


def digest_of_names(names: Iterable[str]) -> str:
    """A stable digest of a set of identifiers.

    Sorted and deduplicated first, because these come from a runtime listing and
    a whitelist file, and neither promises an order. An order-sensitive digest
    would report the world as changed every time something was merely relisted.
    """

    unique = sorted({str(name) for name in names})
    payload = json.dumps(unique, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def digest_of_mapping(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AttestationEvidence:
    """The identity of one check: what was verified, and what against."""

    artifact_sha256: str
    node_inventory_sha256: str
    whitelist_sha256: str
    runtime_contract_sha256: str | None = None
    launch_scope_sha256: str | None = None
    runtime_managed: bool = False
    required_node_types: tuple[str, ...] = ()
    declared_dependencies: dict[str, Any] = field(default_factory=dict)
    resolution: dict[str, Any] = field(default_factory=dict)


def build_evidence(
    *,
    artifact_sha256: str,
    node_inventory: Iterable[str],
    whitelist: Iterable[str],
    required_node_types: Iterable[str],
    declared_dependencies: Mapping[str, Any] | None = None,
    runtime_contract_sha256: str | None = None,
    launch_scope_sha256: str | None = None,
    runtime_managed: bool = False,
    resolution: Mapping[str, Any] | None = None,
) -> AttestationEvidence:
    return AttestationEvidence(
        artifact_sha256=artifact_sha256.lower(),
        node_inventory_sha256=digest_of_names(node_inventory),
        whitelist_sha256=digest_of_names(whitelist),
        runtime_contract_sha256=(
            runtime_contract_sha256.lower() if runtime_contract_sha256 else None
        ),
        launch_scope_sha256=(launch_scope_sha256.lower() if launch_scope_sha256 else None),
        runtime_managed=runtime_managed,
        required_node_types=tuple(sorted({str(name) for name in required_node_types})),
        declared_dependencies=dict(declared_dependencies or {}),
        resolution=dict(resolution or {}),
    )


def unresolved_node_types(
    required: Iterable[str], *, node_inventory: Iterable[str], whitelist: Iterable[str]
) -> tuple[str, ...]:
    """Node types that are not both present in the runtime and reviewed.

    Both conditions, not either. Present-but-unreviewed is the case that matters:
    a package can install a node type that the runtime happily reports, and
    attesting on presence alone would launder an unreviewed package into a
    passing check.
    """

    present = {str(name) for name in node_inventory}
    reviewed = {str(name) for name in whitelist}
    return tuple(sorted({str(name) for name in required} - (present & reviewed)))


# Ordered most decisive first, so a caller showing one reason shows the one that
# most explains why the answer cannot be reused.
_STALENESS_CHECKS: tuple[tuple[str, str], ...] = (
    ("artifact_sha256", "the workflow itself changed since it was checked"),
    ("runtime_contract_sha256", "the runtime it was checked against changed"),
    ("node_inventory_sha256", "the installed node types changed"),
    ("whitelist_sha256", "the reviewed node list changed"),
    ("launch_scope_sha256", "the launch scope changed"),
)


def staleness_reasons(stored: AttestationEvidence, current: AttestationEvidence) -> tuple[str, ...]:
    """Why a recorded attestation no longer describes the world.

    An empty result means the check still holds. This is deliberately not a
    boolean on a row: the events that invalidate an attestation happen in other
    subsystems, which have no reason to know this table exists, so a stored
    verdict would decay silently into a false reassurance.
    """

    reasons: list[str] = []
    for attribute, reason in _STALENESS_CHECKS:
        was = getattr(stored, attribute)
        now = getattr(current, attribute)
        # A field that was not recorded cannot have changed. Treating an absent
        # value as a mismatch would make every attestation taken without a
        # managed runtime look stale the moment one appeared.
        if was is None:
            continue
        if was != now:
            reasons.append(reason)
    if stored.runtime_managed != current.runtime_managed:
        reasons.append("the workflow moved between a managed and an unmanaged runtime")
    return tuple(reasons)


def attestation_state(
    stored: AttestationEvidence | None, current: AttestationEvidence
) -> dict[str, Any]:
    """The read model: what was attested, and whether it still applies."""

    if stored is None:
        return {
            "version": ATTESTATION_VERSION,
            "attested": False,
            "current": False,
            "reasons": ["this machine has not checked this workflow"],
        }
    reasons = staleness_reasons(stored, current)
    return {
        "version": ATTESTATION_VERSION,
        "attested": True,
        "current": not reasons,
        "reasons": list(reasons),
    }
