"""Store a revision's declared dependency contract where the install gate reads it.

A workflow revision is only eligible for an install offer when three identities
exist: its artifact digest, trust, and its dependency contract - the ordered
slot rows plus an aggregate digest over them. Creation already wrote the first,
review grants the second, and until this module nothing in production wrote the
third, so no workflow created the ordinary way could ever receive an offer.

WHAT COUNTS AS A DECLARATION. A mapping that carries a ``version`` key is a
versioned declaration and is parsed strictly: a malformed or unsupported one is
REFUSED, never quietly treated as though nothing had been declared. Anything
else - most commonly ``{}`` - is a legacy declaration. It declares nothing,
which is not the same claim as declaring that nothing is needed, so it gets no
slots and no digest, and nothing is inferred from its fields. An explicit
``{"version": 1, "slots": []}`` is a real declaration of no dependencies and
gets the canonical empty digest.

ORDER IS THE PARSER'S, NOT THE AUTHOR'S. ``parse_workflow_dependency_contract``
sorts slots by name and requirements by key, and the install gate rebuilds the
contract from stored rows in ordinal order. Ordinals therefore enumerate the
canonical slots; writing them in declaration order would store rows the gate
reads back as drift.

No trust is granted here, and nothing here decides readiness. A stored digest
says the declaration is recorded, not that anything it names is present.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import WorkflowDependencySlot, WorkflowRevision
from .workflow_dependencies import (
    WorkflowDependencyContract,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_payload,
    workflow_dependency_slot_sha256,
)


def declared_dependency_contract(declared: object) -> WorkflowDependencyContract | None:
    """The versioned contract a revision declares, or None for a legacy declaration.

    Raises ``WorkflowDependencyError`` - a ``ValueError`` - for a versioned
    declaration that does not parse, so a caller's existing validation step
    refuses it before anything is written.
    """

    if not isinstance(declared, dict) or "version" not in declared:
        return None
    return parse_workflow_dependency_contract(declared)


def declared_dependency_contract_sha256(declared: object) -> str | None:
    """The contract digest a declaration produces when persisted, or None for legacy.

    The single answer to "what should this revision's digest be". A check that
    confirms a stored revision is the one it expects compares against this,
    rather than assuming a fresh revision never carries a digest - an assumption
    that stopped being true the moment declarations were stored.
    """

    contract = declared_dependency_contract(declared)
    return None if contract is None else workflow_dependency_contract_sha256(contract)


def persist_dependency_contract(session: Session, revision: WorkflowRevision) -> None:
    """Write the slot rows and aggregate digest for a revision this transaction created.

    Call after the revision is flushed and before the transaction commits, so the
    rows and the digest land together with the revision or not at all.

    Each revision owns its rows. A revision copied, drafted or replayed from
    another gets rows of its own here rather than sharing or moving the
    original's, which is why this refuses to add rows beside ones that already
    exist: a second write would either duplicate identity or silently disagree
    with it.
    """

    contract = declared_dependency_contract(revision.dependencies_json)
    if contract is None:
        revision.dependency_contract_sha256 = None
        return
    existing = session.scalar(
        select(WorkflowDependencySlot.id)
        .where(WorkflowDependencySlot.workflow_revision_id == revision.id)
        .limit(1)
    )
    if existing is not None:
        raise RuntimeError(
            f"workflow revision {revision.id} already has dependency slots; "
            "its contract is written once, by the transaction that creates it"
        )
    for ordinal, slot in enumerate(contract.slots):
        payload: dict[str, Any] = workflow_dependency_slot_payload(slot)
        session.add(
            WorkflowDependencySlot(
                workflow_revision_id=revision.id,
                ordinal=ordinal,
                name=slot.name,
                resource_kind=slot.resource_kind,
                required=slot.required,
                satisfaction=slot.satisfaction,
                requirements_json=payload["requirements"],
                contract_sha256=workflow_dependency_slot_sha256(slot),
            )
        )
    revision.dependency_contract_sha256 = workflow_dependency_contract_sha256(contract)
